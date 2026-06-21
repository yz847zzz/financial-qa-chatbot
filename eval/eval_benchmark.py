"""
eval_benchmark.py — vLLM vs Local inference benchmark for the Financial QA Chatbot.

Modes:
  --mode vllm    Run eval against a live vLLM server (localhost:8001).
  --mode local   Run eval with local HuggingFace weights (Llama-3.2-3B-Instruct).
  --mode compare Print a side-by-side table from two saved result JSON files.

Usage:
  # vLLM mode (server must be running)
  python eval_benchmark.py --mode vllm

  # Local mode (GPU + local weights required)
  python eval_benchmark.py --mode local

  # Compare two saved runs
  python eval_benchmark.py --mode compare \\
      --output eval_results/bench_vllm_20260412_120000.json \\
      --compare eval_results/bench_local_20260412_130000.json

  # Subset of cases, no fluency scoring
  python eval_benchmark.py --mode vllm --cases 1,2,5 --no-fluency

  # Throughput test with 10 concurrent requests
  python eval_benchmark.py --mode vllm --throughput-n 10
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval/ siblings

from eval_system import (
    TEST_CASES,
    value_correct,
    keyword_hit_rate,
    intent_correct,
    fluency_score,
    timed_answer,
)


# ── Local fluency fallback (no LLM needed) ─────────────────────────────────────

def local_fluency(answer: str) -> int:
    """Rule-based fluency score 1-5 (no LLM needed)."""
    if not answer or len(answer) < 10:
        return 1
    words = answer.split()
    n = len(words)
    # Check for sentence completeness
    ends_ok = answer.rstrip().endswith((".", "!", "?", '"'))
    # Check for gross repetition (any 5-gram repeated)
    grams = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
    repeated = len(grams) != len(set(grams))
    # Score
    if repeated:
        return 2
    if n < 15:
        return 3
    if n >= 30 and ends_ok:
        return 5
    if n >= 15 and ends_ok:
        return 4
    return 3


# ── Latency statistics ─────────────────────────────────────────────────────────

def latency_stats(timings_list: list[dict]) -> dict:
    """Compute mean/p50/p95/p99 for total_s and per-module means."""
    total = sorted([t["total_s"] for t in timings_list if "total_s" in t])
    n = len(total)
    if n == 0:
        return {}

    stats = {
        "total_mean": round(sum(total) / n, 3),
        "total_p50":  round(total[n // 2], 3),
        "total_p95":  round(total[int(n * 0.95)], 3),
        "total_p99":  round(total[int(n * 0.99)] if n >= 10 else total[-1], 3),
    }
    for key in ["decompose_s", "intent_s", "sql_s", "retrieve_s", "answer_s"]:
        vals = [t[key] for t in timings_list if key in t]
        if vals:
            stats[f"{key}_mean"] = round(sum(vals) / len(vals), 3)
    return stats


# ── Throughput test ─────────────────────────────────────────────────────────────

def throughput_test(
    n_concurrent: int,
    model,
    tok,
    nl2sql_model,
    nl2sql_tok,
    retriever,
) -> dict:
    """Run n_concurrent identical queries concurrently and report QPS + latencies."""
    question = "What was Apple's total revenue in FY2023?"
    print(f"\n[Throughput] Firing {n_concurrent} concurrent requests...", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        futs = [
            ex.submit(timed_answer, question, model, tok, nl2sql_model, nl2sql_tok, retriever)
            for _ in range(n_concurrent)
        ]
        results = [f.result() for f in futs]
    elapsed = time.time() - t0
    qps = n_concurrent / elapsed
    latencies = [r[1]["total_s"] for r in results]
    sorted_lat = sorted(latencies)

    return {
        "n_concurrent": n_concurrent,
        "total_elapsed_s": round(elapsed, 3),
        "qps": round(qps, 3),
        "latency_p50": round(sorted_lat[len(sorted_lat) // 2], 3),
        "latency_p95": round(sorted_lat[int(len(sorted_lat) * 0.95)], 3),
        "latency_mean": round(sum(latencies) / len(latencies), 3),
    }


# ── Per-case runner ─────────────────────────────────────────────────────────────

def run_case(
    idx: int,
    total: int,
    case: dict,
    model,
    tok,
    nl2sql_model,
    nl2sql_tok,
    retriever,
    oai_client,
    no_fluency: bool,
) -> dict:
    """Run a single test case and return a result record."""
    print(
        f"\n[{idx}/{total}] Case {case['id']} ({case['category']}): {case['question']}",
        flush=True,
    )

    try:
        result, timings = timed_answer(
            case["question"], model, tok, nl2sql_model, nl2sql_tok, retriever
        )
    except Exception as exc:
        print(f"  → ERROR: {exc}", flush=True)
        return {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": "",
            "intent_correct": False,
            "value_correct": False,
            "keyword_hit_rate": 0.0,
            "fluency_score": 1,
            "timings": {"total_s": 0.0},
            "error": str(exc),
        }

    answer_text = result.get("answer") or ""

    # Metrics
    ic = intent_correct(result, case)
    vc = value_correct(answer_text, case.get("expected_value"))
    khr = keyword_hit_rate(answer_text, case.get("expected_keywords") or [])

    # Fluency
    if no_fluency or oai_client is None:
        flu = local_fluency(answer_text)
    else:
        flu = fluency_score(answer_text, oai_client)
        if flu is None:
            flu = local_fluency(answer_text)

    # Console summary line
    sql_snippet = (result.get("sql") or "")[:60]
    print(
        f"  → intent={result.get('intent')}  "
        f"sql={sql_snippet!r}  "
        f"lat={timings.get('total_s', 0):.2f}s  "
        f"kw={khr:.2f}  val={vc}",
        flush=True,
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": answer_text[:300],
        "intent_correct": ic,
        "value_correct": vc,
        "keyword_hit_rate": round(khr, 3),
        "fluency_score": flu,
        "timings": timings,
    }


# ── Aggregate metrics ───────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    """Compute aggregate metrics across all case records."""
    intent_vals = [int(r["intent_correct"]) for r in records if r["intent_correct"] is not None]
    value_vals  = [int(r["value_correct"])  for r in records if r["value_correct"]  is not None]
    kw_vals     = [r["keyword_hit_rate"] for r in records]
    flu_vals    = [r["fluency_score"] for r in records if r["fluency_score"] is not None]
    timings_list = [r["timings"] for r in records]

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    return {
        "intent_accuracy":   avg(intent_vals),
        "value_accuracy":    avg(value_vals),
        "keyword_hit_rate":  avg(kw_vals),
        "fluency_mean":      avg(flu_vals),
        "latency":           latency_stats(timings_list),
    }


# ── Summary table (single run) ─────────────────────────────────────────────────

def print_summary(mode: str, records: list[dict], agg: dict, throughput: dict | None) -> None:
    """Print a nicely formatted summary after a benchmark run."""
    line = "─" * 60
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY — mode={mode.upper()}")
    print(f"{'='*60}")

    # Aggregate
    lat = agg.get("latency", {})
    print(f"\n  Intent accuracy    : {agg['intent_accuracy']}")
    print(f"  Value accuracy     : {agg['value_accuracy']}")
    print(f"  Keyword hit rate   : {agg['keyword_hit_rate']}")
    print(f"  Fluency mean (1-5) : {agg['fluency_mean']}")
    print(f"  Latency mean (s)   : {lat.get('total_mean')}")
    print(f"  Latency p50  (s)   : {lat.get('total_p50')}")
    print(f"  Latency p95  (s)   : {lat.get('total_p95')}")
    print(f"  Latency p99  (s)   : {lat.get('total_p99')}")

    if throughput:
        print(f"\n  Throughput QPS     : {throughput['qps']}")
        print(f"  Throughput p50 (s) : {throughput['latency_p50']}")
        print(f"  Throughput p95 (s) : {throughput['latency_p95']}")

    # Per-category breakdown
    cats: dict[str, list[dict]] = {}
    for r in records:
        cats.setdefault(r["category"], []).append(r)

    print(f"\n{line}")
    print("Per-category:")
    print(f"  {'Category':<18}  {'Cases':>5}  {'kw_mean':>8}  {'val_acc':>8}  {'flu_mean':>9}")
    print(f"  {'─'*18}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}")
    for cat, cases in sorted(cats.items()):
        kw_m   = round(sum(r["keyword_hit_rate"] for r in cases) / len(cases), 3)
        vv     = [int(r["value_correct"]) for r in cases if r["value_correct"] is not None]
        val_m  = round(sum(vv) / len(vv), 3) if vv else "N/A"
        fv     = [r["fluency_score"] for r in cases if r["fluency_score"] is not None]
        flu_m  = round(sum(fv) / len(fv), 1) if fv else "N/A"
        print(f"  {cat:<18}  {len(cases):>5}  {kw_m:>8}  {str(val_m):>8}  {str(flu_m):>9}")

    # Flag failures
    failures = [r for r in records if r.get("intent_correct") is False or r.get("value_correct") is False]
    if failures:
        print(f"\n{line}")
        print("Cases with failures:")
        for r in failures:
            flags = []
            if r.get("intent_correct") is False:
                flags.append("INTENT-FAIL")
            if r.get("value_correct") is False:
                flags.append("VALUE-FAIL")
            print(f"  [{r['id']:2d}] {', '.join(flags)} — {r['question'][:70]}")
            print(f"       answer: {r['answer'][:100]}")
    print()


# ── Compare mode ────────────────────────────────────────────────────────────────

def run_compare(path_a: str, path_b: str) -> None:
    """Load two result JSON files and print a side-by-side comparison table."""
    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    mode_a = data_a.get("mode", "A")
    mode_b = data_b.get("mode", "B")
    agg_a  = data_a.get("aggregate", {})
    agg_b  = data_b.get("aggregate", {})
    lat_a  = agg_a.get("latency", {})
    lat_b  = agg_b.get("latency", {})
    tput_a = data_a.get("throughput") or {}
    tput_b = data_b.get("throughput") or {}

    label_a = mode_a.upper()
    label_b = mode_b.upper()

    def pct(v):
        if v is None:
            return "N/A"
        return f"{v*100:.1f}%"

    def fmt(v, decimals=3):
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"

    print(f"\n{'='*55}")
    print(f"COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*55}")
    print(f"\n  {'Metric':<28} {label_a:>10}  {label_b:>10}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*10}")

    rows = [
        ("Intent accuracy",    pct(agg_a.get("intent_accuracy")),  pct(agg_b.get("intent_accuracy"))),
        ("Value accuracy",     pct(agg_a.get("value_accuracy")),   pct(agg_b.get("value_accuracy"))),
        ("Keyword hit rate",   fmt(agg_a.get("keyword_hit_rate")), fmt(agg_b.get("keyword_hit_rate"))),
        ("Fluency (mean 1-5)", fmt(agg_a.get("fluency_mean"), 1),  fmt(agg_b.get("fluency_mean"), 1)),
        ("Latency mean (s)",   fmt(lat_a.get("total_mean"), 1),    fmt(lat_b.get("total_mean"), 1)),
        ("Latency p50  (s)",   fmt(lat_a.get("total_p50"),  1),    fmt(lat_b.get("total_p50"),  1)),
        ("Latency p95  (s)",   fmt(lat_a.get("total_p95"),  1),    fmt(lat_b.get("total_p95"),  1)),
        ("Throughput QPS",     fmt(tput_a.get("qps"),       2),    fmt(tput_b.get("qps"),       2)),
    ]
    for label, va, vb in rows:
        print(f"  {label:<28} {va:>10}  {vb:>10}")

    # Per-category breakdown
    cases_a = {r["id"]: r for r in data_a.get("cases", [])}
    cases_b = {r["id"]: r for r in data_b.get("cases", [])}
    all_ids  = sorted(set(cases_a) | set(cases_b))

    cats: dict[str, list[int]] = {}
    for cid in all_ids:
        r = cases_a.get(cid) or cases_b.get(cid)
        if r:
            cats.setdefault(r["category"], []).append(cid)

    print(f"\nPer-category breakdown:")
    print(
        f"  {'Category':<18}  "
        f"{label_a+' kw':>10}  {label_b+' kw':>10}  "
        f"{label_a+' val':>10}  {label_b+' val':>10}"
    )
    print(f"  {'─'*18}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for cat, ids in sorted(cats.items()):
        def cat_avg_kw(cases_dict):
            vals = [cases_dict[i]["keyword_hit_rate"] for i in ids if i in cases_dict]
            return round(sum(vals) / len(vals), 3) if vals else None

        def cat_avg_val(cases_dict):
            vals = [
                int(cases_dict[i]["value_correct"])
                for i in ids
                if i in cases_dict and cases_dict[i]["value_correct"] is not None
            ]
            return round(sum(vals) / len(vals), 2) if vals else None

        kw_a  = fmt(cat_avg_kw(cases_a))
        kw_b  = fmt(cat_avg_kw(cases_b))
        val_a = fmt(cat_avg_val(cases_a), 2) if cat_avg_val(cases_a) is not None else "N/A"
        val_b = fmt(cat_avg_val(cases_b), 2) if cat_avg_val(cases_b) is not None else "N/A"
        print(f"  {cat:<18}  {kw_a:>10}  {kw_b:>10}  {val_a:>10}  {val_b:>10}")

    # Per-case disagreements (value_correct differs)
    disagreements = []
    for cid in all_ids:
        ra = cases_a.get(cid)
        rb = cases_b.get(cid)
        if ra and rb:
            if ra.get("value_correct") != rb.get("value_correct"):
                disagreements.append((cid, ra, rb))

    if disagreements:
        print(f"\nPer-case where value_correct disagrees:")
        for cid, ra, rb in disagreements:
            q = ra["question"][:70]
            print(f"  [{cid:2d}] {q}")
            print(f"    {label_a}: {ra['answer'][:80]!r}")
            print(f"    {label_b}: {rb['answer'][:80]!r}")
    print()


# ── vLLM mode setup ─────────────────────────────────────────────────────────────

def setup_vllm() -> tuple:
    """Check vLLM health, load vectordb only. Returns (None, None, None, None, retriever)."""
    from deployment.api.client import VLLMClient
    client = VLLMClient()
    if not client.health():
        print("ERROR: vLLM server not reachable at localhost:8001.", file=sys.stderr)
        print("       Start it with: bash deployment/scripts/start_server.sh", file=sys.stderr)
        sys.exit(1)
    print("vLLM server is healthy.", flush=True)

    from chatbot import load_vectordb
    retriever = load_vectordb()

    # model/tokenizer are None — vLLM routes internally
    return None, None, None, None, retriever


# ── Local mode setup ────────────────────────────────────────────────────────────

def setup_local(no_nl2sql_adapter: bool) -> tuple:
    """Load local weights and patch chatbot to force local inference path."""
    import chatbot
    from chatbot import load_base_model, load_nl2sql_model, load_vectordb

    # Step 1: load weights before patching so closures can capture them
    print("[Local] Loading base model...", flush=True)
    base_model, base_tok = load_base_model()

    if no_nl2sql_adapter:
        nl2sql_model, nl2sql_tok = base_model, base_tok
    else:
        nl2sql_model, nl2sql_tok = load_nl2sql_model(base_model, base_tok)

    retriever = load_vectordb()

    # Step 2: monkey-patch chatbot._vllm so health() returns False AND
    # generate_sql_vllm + generate route through local llm_generate.
    # This forces llm_generate() to use the local path and keeps generate_sql()
    # working (it calls _vllm.generate_sql_vllm directly, bypassing llm_generate).

    _bm = base_model   # capture in closure
    _bt = base_tok

    class _FakeVLLM:
        def health(self) -> bool:
            return False

        def generate_sql_vllm(self, messages: list[dict], max_new_tokens: int = 200) -> str:
            return chatbot.llm_generate(_bm, _bt, messages, max_new_tokens)

        def generate(self, messages: list[dict], role: str = "base",
                     max_new_tokens: int = 256) -> str:
            return chatbot.llm_generate(_bm, _bt, messages, max_new_tokens)

    chatbot._vllm = _FakeVLLM()

    # Step 3: patch query_rewriter._vllm as well (it also calls _vllm.generate)
    try:
        from deployment.rag import query_rewriter as qr
        qr._vllm = _FakeVLLM()
        print("[Local] Patched query_rewriter._vllm.", flush=True)
    except Exception as e:
        print(f"[Local] Could not patch query_rewriter._vllm: {e}", flush=True)

    print("[Local] Monkey-patching complete — using local inference path.", flush=True)
    return base_model, base_tok, nl2sql_model, nl2sql_tok, retriever


# ── Main ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vLLM vs Local benchmark for the Financial QA Chatbot"
    )
    p.add_argument(
        "--mode", required=True, choices=["vllm", "local", "compare"],
        help="Inference mode: vllm | local | compare",
    )
    p.add_argument(
        "--compare", type=str, default=None,
        metavar="PATH",
        help="Path to second results JSON (used with --mode compare)",
    )
    p.add_argument(
        "--cases", type=str, default=None,
        metavar="IDS",
        help="Comma-separated case IDs to run, e.g. '1,2,5' (default: all 20)",
    )
    p.add_argument(
        "--throughput-n", type=int, default=5,
        metavar="N",
        help="Number of concurrent requests for throughput test (default: 5)",
    )
    p.add_argument(
        "--parallel-n", type=int, default=1,
        metavar="N",
        help=(
            "Run N test cases concurrently (vLLM batches them in one GPU pass). "
            "Recommended: 4-8 for vLLM, 1 for local. Default: 1 (sequential)."
        ),
    )
    p.add_argument(
        "--no-fluency", action="store_true",
        help="Skip GPT-4o-mini fluency scoring; use local_fluency() instead",
    )
    p.add_argument(
        "--no-nl2sql-adapter", action="store_true",
        help="(local mode only) Use base model for SQL instead of NL2SQL LoRA adapter",
    )
    p.add_argument(
        "--output", type=str, default=None,
        metavar="PATH",
        help="Path to save results JSON (default: eval_results/bench_{mode}_{ts}.json)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.mode == "compare":
        if not args.output:
            parser.error("--mode compare requires --output PATH (first results file)")
        if not args.compare:
            parser.error("--mode compare requires --compare PATH (second results file)")
        run_compare(args.output, args.compare)
        return

    # ── Select test cases ─────────────────────────────────────────────────────
    cases = TEST_CASES
    if args.cases:
        ids = {int(x.strip()) for x in args.cases.split(",")}
        cases = [c for c in cases if c["id"] in ids]
    print(f"Running {len(cases)} test case(s) in {args.mode.upper()} mode.", flush=True)

    # ── OpenAI client (fluency) ───────────────────────────────────────────────
    oai_client = None
    if not args.no_fluency:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            oai_client = OpenAI(api_key=api_key)
            print("OpenAI client ready (fluency scoring enabled).", flush=True)
        else:
            print(
                "[WARN] OPENAI_API_KEY not set — using local_fluency() rule-based scorer.",
                flush=True,
            )

    # ── Load models / set up inference path ──────────────────────────────────
    if args.mode == "vllm":
        base_model, base_tok, nl2sql_model, nl2sql_tok, retriever = setup_vllm()
    else:  # local
        base_model, base_tok, nl2sql_model, nl2sql_tok, retriever = setup_local(
            no_nl2sql_adapter=args.no_nl2sql_adapter
        )

    # ── Run all test cases ────────────────────────────────────────────────────
    parallel_n = max(1, args.parallel_n)
    if parallel_n > 1:
        print(
            f"[Parallel] Running {len(cases)} cases with concurrency={parallel_n} "
            f"(vLLM will batch simultaneous requests in the same GPU pass).",
            flush=True,
        )

    records: list[dict] = []

    def _run(args_tuple):
        idx, case = args_tuple
        return run_case(
            idx=idx,
            total=len(cases),
            case=case,
            model=base_model,
            tok=base_tok,
            nl2sql_model=nl2sql_model,
            nl2sql_tok=nl2sql_tok,
            retriever=retriever,
            oai_client=oai_client,
            no_fluency=args.no_fluency,
        )

    if parallel_n == 1:
        for i, case in enumerate(cases, 1):
            records.append(_run((i, case)))
    else:
        with ThreadPoolExecutor(max_workers=parallel_n) as ex:
            futures = {ex.submit(_run, (i, case)): case["id"]
                       for i, case in enumerate(cases, 1)}
            # Collect in submission order to keep IDs sorted
            ordered = sorted(futures.keys(), key=lambda f: futures[f])
            records = [f.result() for f in ordered]

    # ── Throughput test ───────────────────────────────────────────────────────
    tput = None
    if args.throughput_n and args.throughput_n > 0:
        try:
            tput = throughput_test(
                n_concurrent=args.throughput_n,
                model=base_model,
                tok=base_tok,
                nl2sql_model=nl2sql_model,
                nl2sql_tok=nl2sql_tok,
                retriever=retriever,
            )
            print(
                f"[Throughput] QPS={tput['qps']}  "
                f"p50={tput['latency_p50']}s  "
                f"p95={tput['latency_p95']}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[Throughput] test failed: {exc}", flush=True)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(records)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(mode=args.mode, records=records, agg=agg, throughput=tput)

    # ── Build result payload ──────────────────────────────────────────────────
    ts = datetime.now().isoformat(timespec="seconds")
    payload = {
        "mode": args.mode,
        "timestamp": ts,
        "n_cases": len(records),
        "cases": records,
        "aggregate": agg,
        "throughput": tput,
    }

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"bench_{args.mode}_{ts_file}.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
