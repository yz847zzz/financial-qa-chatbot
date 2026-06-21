#!/usr/bin/env python3
"""
eval_unified.py -- Unified evaluation across ALL system configurations.

Runs the SAME 52-case test set from eval_system.TEST_CASES on every config,
producing standardized JSON with per-case results, per-type composite scores,
and throughput measurements.

Configs:
  --config local          Direct transformers.generate() (no vLLM)
  --config vllm           vLLM serving (server must be running on :8001)
  --config gpt4o          GPT-4o API (requires OPENAI_API_KEY)

Labels (for vLLM quant runs):
  --label fp16            Saved as unified_vllm_fp16_*.json
  --label int8            Saved as unified_vllm_int8_*.json
  --label awq4            Saved as unified_vllm_awq4_*.json

Workflow:
  # 1. Local (simple deploy)
  python eval_unified.py --config local

  # 2. vLLM fp16
  bash deployment/scripts/start_server_quant.sh fp16
  python eval_unified.py --config vllm --label fp16

  # 3. vLLM int8
  bash deployment/scripts/start_server_quant.sh int8
  python eval_unified.py --config vllm --label int8

  # 4. vLLM AWQ4
  bash deployment/scripts/start_server_quant.sh awq4
  python eval_unified.py --config vllm --label awq4

  # 5. GPT-4o
  OPENAI_API_KEY=sk-... python eval_unified.py --config gpt4o

  # 6. Compare all
  python eval_unified.py --compare

Output: eval_results/unified_{config}_{label}_{timestamp}.json
"""

import argparse
import json
import os
import sys
import time
import statistics
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
    timed_answer,
    ask_gpt,
    GPT_SYSTEM,
)
from eval_benchmark import local_fluency, latency_stats

EVAL_DIR = ROOT / "eval" / "results"
EVAL_DIR.mkdir(exist_ok=True)

# Throughput burst question
THROUGHPUT_Q = "What was Apple's total revenue in FY2023?"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
BURST_REPS = 10

# ---------------------------------------------------------------------------
# Semantic similarity (cosine) using all-MiniLM-L6-v2
# ---------------------------------------------------------------------------
_sim_model = None

# Loaded from --references file: {case_id -> {"answer": [ref1, ref2, ref3], ...}}
_refs_by_id: dict[int, dict] = {}


def _get_sim_model():
    """Lazy-load sentence-transformer for cosine similarity."""
    global _sim_model
    if _sim_model is None:
        from sentence_transformers import SentenceTransformer
        _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Eval] Loaded all-MiniLM-L6-v2 for semantic similarity.", flush=True)
    return _sim_model


def cosine_similarity(answer: str, reference: str) -> float:
    """Compute cosine similarity between answer and a single reference."""
    if not answer or not reference:
        return 0.0
    model = _get_sim_model()
    embs = model.encode([answer, reference], normalize_embeddings=True)
    return max(0.0, float(embs[0] @ embs[1]))


def cosine_similarity_multi_ref(answer: str, references: list[str]) -> float:
    """Max cosine similarity over multiple reference answers (FINDMIND-style).

    Taking max instead of mean rewards any phrasing match, which is appropriate
    when references are deliberately varied phrasings of the same correct answer.
    """
    if not answer or not references:
        return 0.0
    model = _get_sim_model()
    answer_emb = model.encode([answer], normalize_embeddings=True)[0]
    ref_embs = model.encode(references, normalize_embeddings=True)
    sims = [max(0.0, float(answer_emb @ ref_emb)) for ref_emb in ref_embs]
    return max(sims)


def build_reference(case: dict) -> str:
    """Build a reference sentence from question + expected keywords + value.

    Since we don't have hand-written expected_answer fields, we construct
    a plausible reference from what we know: the question phrased as an
    answer with the expected keywords and value embedded.
    """
    q = case.get("question", "")
    kws = case.get("expected_keywords", [])
    val = case.get("expected_value")

    parts = list(kws)
    if val is not None:
        if isinstance(val, float) and val > 1e6:
            parts.append(f"${val/1e9:.1f} billion")
        elif isinstance(val, float) and val < 1.0:
            parts.append(f"{val*100:.1f}%")
        else:
            parts.append(str(val))
    if parts:
        return f"{q} Answer: {', '.join(parts)}"
    return q


# ---------------------------------------------------------------------------
# Intent-aware composite scoring
# ---------------------------------------------------------------------------
# Weights: keyword and value are the primary signals (exact match checks).
# Semantic similarity (cosine) is a softer signal for overall answer quality.
WEIGHTS = {
    "Type1": {"key": 0.15, "value": 0.65, "sem": 0.20},
    "Type2": {"key": 0.40, "value": 0.10, "sem": 0.50},
    "Type3": {"key": 0.50, "value": 0.00, "sem": 0.50},
}


def map_category(cat: str) -> str:
    c = cat.lower()
    if "type3" in c:
        return "Type3"
    elif "type2" in c and "type1" not in c:
        return "Type2"
    else:
        return "Type1"


def case_score(category: str, intent_ok, value_ok, kw_rate: float,
               sem_sim: float = 0.0) -> float:
    t = map_category(category)
    w = WEIGHTS[t]
    key = kw_rate if kw_rate is not None else 0.0
    val = 1.0 if value_ok is True else (0.0 if value_ok is False else 0.5)
    score = w["key"] * key + w["value"] * val + w["sem"] * sem_sim
    if intent_ok is False:
        score = min(score, 0.50)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Run a single case (local / vllm)
# ---------------------------------------------------------------------------
def run_case_pipeline(case: dict, idx: int, total: int,
                      model, tok, nl2sql_model, nl2sql_tok,
                      retriever) -> dict:
    """Run one test case through our pipeline (local or vLLM)."""
    print(f"\n[{idx}/{total}] Case {case['id']} ({case['category']}): "
          f"{case['question'][:60]}", flush=True)

    try:
        result, timings = timed_answer(
            case["question"], model, tok, nl2sql_model, nl2sql_tok, retriever
        )
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        return {
            "id": case["id"], "category": case["category"],
            "question": case["question"], "answer": "",
            "intent_correct": False, "value_correct": False,
            "keyword_hit_rate": 0.0, "semantic_similarity": 0.0,
            "fluency_score": 1,
            "timings": {"total_s": 0.0}, "error": str(exc),
        }

    answer_text = result.get("answer") or ""
    ic = intent_correct(result, case)
    vc = value_correct(answer_text, case.get("expected_value"),
                       acceptable_values=case.get("acceptable_values"))
    khr = keyword_hit_rate(answer_text, case.get("expected_keywords") or [])
    flu = local_fluency(answer_text)
    refs = _refs_by_id.get(case["id"], {}).get("answer")
    if refs:
        sem = cosine_similarity_multi_ref(answer_text, refs)
    else:
        sem = cosine_similarity(answer_text, build_reference(case))

    print(f"  intent={result.get('intent')}  kw={khr:.2f}  val={vc}  sem={sem:.3f}  "
          f"lat={timings.get('total_s', 0):.2f}s", flush=True)

    return {
        "id": case["id"], "category": case["category"],
        "question": case["question"], "answer": answer_text[:300],
        "intent_correct": ic, "value_correct": vc,
        "keyword_hit_rate": round(khr, 3), "semantic_similarity": round(sem, 4),
        "fluency_score": flu,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# Run a single case (GPT-4o)
# ---------------------------------------------------------------------------
def run_case_gpt4o(case: dict, idx: int, total: int, client) -> dict:
    """Run one test case against GPT-4o API."""
    print(f"\n[{idx}/{total}] Case {case['id']} ({case['category']}): "
          f"{case['question'][:60]}", flush=True)

    try:
        answer_text, latency = ask_gpt(case["question"], client)
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        return {
            "id": case["id"], "category": case["category"],
            "question": case["question"], "answer": "",
            "intent_correct": True, "value_correct": False,
            "keyword_hit_rate": 0.0, "semantic_similarity": 0.0,
            "fluency_score": 1,
            "timings": {"total_s": 0.0}, "error": str(exc),
        }

    vc = value_correct(answer_text, case.get("expected_value"),
                       acceptable_values=case.get("acceptable_values"))
    khr = keyword_hit_rate(answer_text, case.get("expected_keywords") or [])
    flu = local_fluency(answer_text)
    refs = _refs_by_id.get(case["id"], {}).get("answer")
    if refs:
        sem = cosine_similarity_multi_ref(answer_text, refs)
    else:
        sem = cosine_similarity(answer_text, build_reference(case))

    print(f"  kw={khr:.2f}  val={vc}  sem={sem:.3f}  lat={latency:.2f}s", flush=True)

    return {
        "id": case["id"], "category": case["category"],
        "question": case["question"], "answer": answer_text[:300],
        "intent_correct": True,  # GPT-4o handles routing internally
        "value_correct": vc,
        "keyword_hit_rate": round(khr, 3), "semantic_similarity": round(sem, 4),
        "fluency_score": flu,
        "timings": {"total_s": round(latency, 3)},
    }


# ---------------------------------------------------------------------------
# Throughput burst test
# ---------------------------------------------------------------------------
def throughput_burst(concurrency: int, model, tok, nl2sql_model, nl2sql_tok,
                     retriever) -> dict:
    """Fire BURST_REPS requests at given concurrency, measure QPS + latency."""
    print(f"  [Burst] c={concurrency}  reps={BURST_REPS}", flush=True)

    args = (THROUGHPUT_Q, model, tok, nl2sql_model, nl2sql_tok, retriever)
    all_latencies = []
    errors = 0
    n_batches = max(1, BURST_REPS // concurrency)
    t0 = time.time()

    for _ in range(n_batches):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_burst_worker, args) for _ in range(concurrency)]
            for f in futs:
                ok, lat = f.result()
                if ok:
                    all_latencies.append(lat)
                else:
                    errors += 1

    wall = time.time() - t0
    total_req = n_batches * concurrency
    sorted_lat = sorted(all_latencies) if all_latencies else [0]
    n = len(sorted_lat)

    return {
        "concurrency": concurrency,
        "total_requests": total_req, "errors": errors,
        "wall_time_s": round(wall, 3),
        "qps": round(total_req / wall, 3),
        "latency_mean": round(sum(sorted_lat) / n, 3),
        "latency_p50": round(sorted_lat[n // 2], 3),
        "latency_p95": round(sorted_lat[int(n * 0.95)], 3),
    }


def _burst_worker(args):
    t0 = time.time()
    try:
        timed_answer(*args)
        return True, time.time() - t0
    except Exception:
        return False, time.time() - t0


def throughput_burst_gpt4o(concurrency: int, client) -> dict:
    """GPT-4o throughput test."""
    print(f"  [Burst GPT-4o] c={concurrency}  reps={BURST_REPS}", flush=True)

    all_latencies = []
    errors = 0
    n_batches = max(1, BURST_REPS // concurrency)
    t0 = time.time()

    for _ in range(n_batches):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(ask_gpt, THROUGHPUT_Q, client) for _ in range(concurrency)]
            for f in futs:
                try:
                    _, lat = f.result()
                    all_latencies.append(lat)
                except Exception:
                    errors += 1

    wall = time.time() - t0
    total_req = n_batches * concurrency
    sorted_lat = sorted(all_latencies) if all_latencies else [0]
    n = len(sorted_lat)

    return {
        "concurrency": concurrency,
        "total_requests": total_req, "errors": errors,
        "wall_time_s": round(wall, 3),
        "qps": round(total_req / wall, 3),
        "latency_mean": round(sum(sorted_lat) / n, 3),
        "latency_p50": round(sorted_lat[n // 2], 3),
        "latency_p95": round(sorted_lat[int(n * 0.95)], 3),
    }


# ---------------------------------------------------------------------------
# Compute aggregate + per-type composite scores
# ---------------------------------------------------------------------------
def compute_aggregate(records: list) -> dict:
    """Compute aggregate metrics and per-type composite scores."""
    # Per-type grouping
    by_type = {"Type1": [], "Type2": [], "Type3": []}
    all_scores = []

    for r in records:
        t = map_category(r["category"])
        sem = r.get("semantic_similarity", 0.0)
        s = case_score(r["category"], r["intent_correct"],
                       r["value_correct"], r["keyword_hit_rate"], sem)
        r["composite_score"] = s  # attach to record
        by_type[t].append(r)
        all_scores.append(s)

    def type_stats(recs):
        if not recs:
            return None
        scores = [r["composite_score"] for r in recs]
        intent_vals = [int(r["intent_correct"]) for r in recs
                       if r["intent_correct"] is not None]
        value_vals = [int(r["value_correct"]) for r in recs
                      if r["value_correct"] is not None]
        kw_vals = [r["keyword_hit_rate"] for r in recs]
        sem_vals = [r.get("semantic_similarity", 0.0) for r in recs]
        return {
            "n": len(recs),
            "composite_mean": round(statistics.mean(scores), 4),
            "intent_accuracy": round(statistics.mean(intent_vals), 4) if intent_vals else None,
            "value_accuracy": round(statistics.mean(value_vals), 4) if value_vals else None,
            "keyword_hit_rate": round(statistics.mean(kw_vals), 4),
            "semantic_similarity": round(statistics.mean(sem_vals), 4),
        }

    # Overall
    intent_all = [int(r["intent_correct"]) for r in records
                  if r["intent_correct"] is not None]
    value_all = [int(r["value_correct"]) for r in records
                 if r["value_correct"] is not None]
    kw_all = [r["keyword_hit_rate"] for r in records]
    sem_all = [r.get("semantic_similarity", 0.0) for r in records]
    flu_all = [r["fluency_score"] for r in records if r["fluency_score"] is not None]
    timings_list = [r["timings"] for r in records]

    return {
        "n_cases": len(records),
        "composite_score": round(statistics.mean(all_scores), 4),
        "intent_accuracy": round(statistics.mean(intent_all), 4) if intent_all else None,
        "value_accuracy": round(statistics.mean(value_all), 4) if value_all else None,
        "keyword_hit_rate": round(statistics.mean(kw_all), 4),
        "semantic_similarity": round(statistics.mean(sem_all), 4),
        "fluency_mean": round(statistics.mean(flu_all), 2) if flu_all else None,
        "latency": latency_stats(timings_list),
        "per_type": {
            "Type1": type_stats(by_type["Type1"]),
            "Type2": type_stats(by_type["Type2"]),
            "Type3": type_stats(by_type["Type3"]),
        },
    }


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------
def setup_local():
    from eval_benchmark import setup_local as _setup
    result = _setup(no_nl2sql_adapter=False)

    # Fix dual module-path patching issue:
    # eval_benchmark patches `deployment.rag.query_rewriter._vllm` but
    # timed_answer imports from `rag.query_rewriter` (different sys.modules entry).
    # We must patch ALL module aliases that reference query_rewriter._vllm.
    import sys as _sys
    import chatbot
    fake_vllm = chatbot._vllm  # already patched by setup_local -> _FakeVLLM

    for mod_name, mod in list(_sys.modules.items()):
        if mod is None:
            continue
        if "query_rewriter" in mod_name and hasattr(mod, "_vllm"):
            mod._vllm = fake_vllm
            print(f"[Patch] {mod_name}._vllm -> _FakeVLLM", flush=True)

    # Also force-import the variant timed_answer will use and patch it
    try:
        from rag import query_rewriter as qr_alt
        qr_alt._vllm = fake_vllm
        print(f"[Patch] rag.query_rewriter._vllm -> _FakeVLLM", flush=True)
    except ImportError:
        pass

    return result


def setup_vllm():
    from eval_benchmark import setup_vllm as _setup
    return _setup()


def setup_gpt4o():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------
def run_accuracy(config: str, label: str, cases: list,
                 model=None, tok=None, nl2sql_model=None, nl2sql_tok=None,
                 retriever=None, oai_client=None) -> list:
    """Run all test cases and return list of per-case records."""
    records = []
    for i, case in enumerate(cases, 1):
        if config == "gpt4o":
            rec = run_case_gpt4o(case, i, len(cases), oai_client)
        else:
            rec = run_case_pipeline(case, i, len(cases),
                                    model, tok, nl2sql_model, nl2sql_tok,
                                    retriever)
        records.append(rec)
    return records


def run_throughput(config: str, label: str,
                   model=None, tok=None, nl2sql_model=None, nl2sql_tok=None,
                   retriever=None, oai_client=None,
                   concurrency_levels=None) -> list:
    """Run throughput burst tests at each concurrency level."""
    levels = concurrency_levels or CONCURRENCY_LEVELS
    results = []

    if config == "local":
        # Local mode: only c=1 (no real concurrency benefit)
        levels = [1]

    print(f"\n{'='*60}")
    print(f"THROUGHPUT TESTS: {config}/{label}")
    print(f"{'='*60}")

    for c in levels:
        try:
            if config == "gpt4o":
                r = throughput_burst_gpt4o(c, oai_client)
            else:
                r = throughput_burst(c, model, tok, nl2sql_model, nl2sql_tok,
                                     retriever)
            print(f"  c={c}: QPS={r['qps']:.3f}  p50={r['latency_p50']:.3f}s  "
                  f"p95={r['latency_p95']:.3f}s", flush=True)
            results.append(r)
        except Exception as exc:
            print(f"  c={c}: ERROR {exc}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------
def find_unified_files() -> list:
    """Find all unified_*.json files in eval_results/."""
    return sorted(EVAL_DIR.glob("unified_*.json"))


def run_compare():
    """Load all unified results and print comparison table."""
    files = find_unified_files()
    if not files:
        print("No unified_*.json files found in eval_results/.")
        print("Run eval_unified.py --config <config> first.")
        return

    print(f"\nFound {len(files)} result file(s):")
    results = []
    for f in files:
        d = json.loads(f.read_text())
        results.append(d)
        print(f"  {f.name}: {d['config']}/{d['label']} ({d['aggregate']['n_cases']} cases)")

    LINE = "=" * 110
    DASH = "-" * 110

    print(f"\n{LINE}")
    print("  UNIFIED BENCHMARK COMPARISON (same 52-case test set)")
    print(LINE)

    # Table A: Composite Accuracy
    print(f"\n  TABLE A: COMPOSITE ACCURACY SCORE (intent-aware weighted)")
    print(DASH)
    hdr = f"  {'Config':<22} {'Overall':>8} {'Type1':>8} {'Type2':>8} {'Type3':>8} {'N':>4}"
    print(hdr)
    print("  " + "-" * 65)
    for d in results:
        agg = d["aggregate"]
        pt = agg.get("per_type", {})
        t1 = pt.get("Type1", {})
        t2 = pt.get("Type2", {})
        t3 = pt.get("Type3", {})
        t1s = f"{t1['composite_mean']:.4f}" if t1 else "  --- "
        t2s = f"{t2['composite_mean']:.4f}" if t2 else "  --- "
        t3s = f"{t3['composite_mean']:.4f}" if t3 else "  --- "
        name = f"{d['config']}/{d['label']}"
        print(f"  {name:<22} {agg['composite_score']:>8.4f} {t1s:>8} {t2s:>8} {t3s:>8} "
              f"{agg['n_cases']:>4}")

    # Table B: Detailed accuracy breakdown
    print(f"\n  TABLE B: ACCURACY DETAIL")
    print(DASH)
    hdr = f"  {'Config':<22} {'Intent%':>8} {'Value%':>8} {'Keyword':>8} {'SemSim':>8} {'Fluency':>8}"
    print(hdr)
    print("  " + "-" * 68)
    for d in results:
        agg = d["aggregate"]
        ia = f"{agg['intent_accuracy']*100:.1f}%" if agg.get("intent_accuracy") is not None else "  N/A"
        va = f"{agg['value_accuracy']*100:.1f}%" if agg.get("value_accuracy") is not None else "  N/A"
        kw = f"{agg['keyword_hit_rate']:.4f}" if agg.get("keyword_hit_rate") is not None else "  N/A"
        ss = f"{agg['semantic_similarity']:.4f}" if agg.get("semantic_similarity") is not None else "  N/A"
        fl = f"{agg['fluency_mean']:.1f}" if agg.get("fluency_mean") is not None else "  N/A"
        name = f"{d['config']}/{d['label']}"
        print(f"  {name:<22} {ia:>8} {va:>8} {kw:>8} {ss:>8} {fl:>8}")

    # Table C: Latency
    print(f"\n  TABLE C: LATENCY (pipeline end-to-end, accuracy run)")
    print(DASH)
    hdr = f"  {'Config':<22} {'Mean(s)':>9} {'p50(s)':>8} {'p95(s)':>8} {'p99(s)':>8}"
    print(hdr)
    print("  " + "-" * 59)
    for d in results:
        lat = d["aggregate"].get("latency", {})
        name = f"{d['config']}/{d['label']}"
        mean = f"{lat['total_mean']:.3f}" if lat.get("total_mean") else "  ---"
        p50 = f"{lat['total_p50']:.3f}" if lat.get("total_p50") else "  ---"
        p95 = f"{lat['total_p95']:.3f}" if lat.get("total_p95") else "  ---"
        p99 = f"{lat['total_p99']:.3f}" if lat.get("total_p99") else "  ---"
        print(f"  {name:<22} {mean:>9} {p50:>8} {p95:>8} {p99:>8}")

    # Table D: Throughput
    print(f"\n  TABLE D: THROUGHPUT (burst test, QPS at each concurrency)")
    print(DASH)
    # Gather all concurrency levels across all configs
    all_c = set()
    for d in results:
        for t in d.get("throughput", []):
            all_c.add(t["concurrency"])
    all_c = sorted(all_c)

    if all_c:
        c_hdrs = "".join(f"  c={c:>2}" for c in all_c)
        print(f"  {'Config':<22} {c_hdrs}")
        print("  " + "-" * (22 + 6 * len(all_c)))
        for d in results:
            name = f"{d['config']}/{d['label']}"
            tput_by_c = {t["concurrency"]: t for t in d.get("throughput", [])}
            vals = []
            for c in all_c:
                if c in tput_by_c:
                    vals.append(f"{tput_by_c[c]['qps']:>5.2f}")
                else:
                    vals.append("  ---")
            print(f"  {name:<22} {'  '.join(vals)}")

        # Also show latency at each concurrency
        print(f"\n  Latency p50 (s) at each concurrency:")
        print(f"  {'Config':<22} {c_hdrs}")
        print("  " + "-" * (22 + 6 * len(all_c)))
        for d in results:
            name = f"{d['config']}/{d['label']}"
            tput_by_c = {t["concurrency"]: t for t in d.get("throughput", [])}
            vals = []
            for c in all_c:
                if c in tput_by_c:
                    vals.append(f"{tput_by_c[c]['latency_p50']:>5.2f}")
                else:
                    vals.append("  ---")
            print(f"  {name:<22} {'  '.join(vals)}")

    # Scoring formula reminder
    print(f"\n  SCORING FORMULA:")
    print(f"  Type1: 0.15*keyword + 0.65*value + 0.20*cosine_sim")
    print(f"  Type2: 0.40*keyword + 0.10*value + 0.50*cosine_sim")
    print(f"  Type3: 0.50*keyword + 0.00*value + 0.50*cosine_sim")
    print(f"  Semantic: cosine similarity (all-MiniLM-L6-v2) between answer and reference.")
    print(f"  Intent miss: capped at 0.50.")

    print(f"\n{LINE}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation across all system configurations")
    parser.add_argument("--config", choices=["local", "vllm", "gpt4o"],
                        help="Inference configuration to evaluate")
    parser.add_argument("--label", type=str, default="default",
                        help="Label for this run (e.g. fp16, int8, awq4)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all unified_*.json results")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs (default: all 52)")
    parser.add_argument("--skip-throughput", action="store_true",
                        help="Skip throughput burst tests")
    parser.add_argument("--concurrency", type=str, default=None,
                        help="Comma-separated concurrency levels (default: 1,2,4,8,16)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path")
    parser.add_argument("--testset", type=str, default=None,
                        help="Path to expanded test-case JSON (default: use built-in 52 cases)")
    parser.add_argument("--references", type=str, default=None,
                        help="Path to eval_references.json (GPT-4o multi-ref answers). "
                             "Enables accurate semantic similarity scoring.")
    args = parser.parse_args()

    if args.compare:
        run_compare()
        return

    if not args.config:
        parser.error("--config is required (or use --compare)")

    # Load multi-reference answers if provided
    if args.references:
        with open(args.references, encoding="utf-8") as f:
            for rec in json.load(f):
                _refs_by_id[rec["id"]] = rec
        print(f"Loaded {len(_refs_by_id)} reference answer sets from {args.references}",
              flush=True)

    # Select cases -- either built-in 52 or expanded JSON
    if args.testset:
        with open(args.testset) as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cases from {args.testset}", flush=True)
    else:
        cases = TEST_CASES
    if args.cases:
        ids = {int(x.strip()) for x in args.cases.split(",")}
        cases = [c for c in cases if c["id"] in ids]
    print(f"Config: {args.config}/{args.label}  |  {len(cases)} test cases", flush=True)

    # Concurrency levels
    c_levels = CONCURRENCY_LEVELS
    if args.concurrency:
        c_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    # Setup
    model = tok = nl2sql_model = nl2sql_tok = retriever = oai_client = None

    if args.config == "local":
        model, tok, nl2sql_model, nl2sql_tok, retriever = setup_local()
        if args.label == "default":
            args.label = "simple"
    elif args.config == "vllm":
        model, tok, nl2sql_model, nl2sql_tok, retriever = setup_vllm()
    elif args.config == "gpt4o":
        oai_client = setup_gpt4o()
        args.label = "gpt4o"

    # -- Phase 1: Accuracy (52-case eval) ------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: ACCURACY EVALUATION ({len(cases)} cases)")
    print(f"{'='*60}")

    records = run_accuracy(
        args.config, args.label, cases,
        model, tok, nl2sql_model, nl2sql_tok, retriever, oai_client,
    )

    agg = compute_aggregate(records)

    # Print summary
    pt = agg.get("per_type", {})
    print(f"\n{'='*60}")
    print(f"ACCURACY SUMMARY: {args.config}/{args.label}")
    print(f"{'='*60}")
    print(f"  Overall composite : {agg['composite_score']:.4f}")
    for t in ["Type1", "Type2", "Type3"]:
        ts = pt.get(t)
        if ts:
            print(f"  {t} composite     : {ts['composite_mean']:.4f}  ({ts['n']} cases)")
    print(f"  Intent accuracy   : {agg.get('intent_accuracy')}")
    print(f"  Value accuracy    : {agg.get('value_accuracy')}")
    print(f"  Keyword hit rate  : {agg.get('keyword_hit_rate')}")
    print(f"  Semantic sim      : {agg.get('semantic_similarity')}")
    lat = agg.get("latency", {})
    print(f"  Latency mean/p50  : {lat.get('total_mean')}s / {lat.get('total_p50')}s")

    # -- Phase 2: Throughput -------------------------------------------------
    throughput = []
    if not args.skip_throughput:
        throughput = run_throughput(
            args.config, args.label,
            model, tok, nl2sql_model, nl2sql_tok, retriever, oai_client,
            c_levels,
        )

    # -- Save results -------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "config": args.config,
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(records),
        "cases": records,
        "aggregate": agg,
        "throughput": throughput,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = EVAL_DIR / f"unified_{args.config}_{args.label}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
