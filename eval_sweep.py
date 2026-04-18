"""
eval_sweep.py — Quantization × Concurrency sweep for the Financial QA Chatbot.

Sweeps two axes:
  Quantization : fp16 (baseline) | int8 (bitsandbytes) | awq4 (AWQ INT4)
  Concurrency  : 1, 2, 4, 8, 16  (parallel pipeline runs hitting vLLM)

For each (quant, concurrency) cell, records:
  Throughput : QPS from a dedicated burst test (N identical simple queries)
  Latency    : p50 / p95 wall-clock seconds per request
  Accuracy   : intent_acc, value_acc, keyword_hit_rate (from 52-case eval)

The accuracy sweep runs once per quantization level (at concurrency=1) since
accuracy is a property of the model+quantization, not of concurrency.

Output
──────
  eval_results/sweep_{quant}_{timestamp}.json   ← one per quantization run
  eval_results/sweep_matrix_{timestamp}.json    ← combined matrix for plotting

Usage
─────
  # Step 1 — start vLLM with fp16 (current default):
  #   bash deployment/scripts/start_server_quant.sh fp16
  python eval_sweep.py --quant fp16

  # Step 2 — restart vLLM with int8:
  #   bash deployment/scripts/start_server_quant.sh int8
  python eval_sweep.py --quant int8

  # Step 3 — download AWQ model (first time only):
  #   huggingface-cli download hugging-quants/Meta-Llama-3.2-3B-Instruct-AWQ-INT4 \\
  #       --local-dir models/llama/llama-3.2-3b-awq4
  # Then restart vLLM:
  #   bash deployment/scripts/start_server_quant.sh awq4
  python eval_sweep.py --quant awq4

  # Combine all three sweeps and plot:
  python eval_plot.py \\
      eval_results/sweep_fp16_*.json \\
      eval_results/sweep_int8_*.json \\
      eval_results/sweep_awq4_*.json

Quantization impact on RTX 3090 Ti (24 GB)
──────────────────────────────────────────
  fp16  : Llama-3B ≈  6 GB VRAM → 18 GB for KV cache
  int8  : Llama-3B ≈  3 GB VRAM → 21 GB for KV cache  → higher max batch size
  awq4  : Llama-3B ≈ 1.5 GB VRAM → 22.5 GB for KV cache → even higher batch size
  Smaller model weight footprint → more VRAM for KV cache → more concurrent
  sequences per forward pass → better SGMV utilization.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

# ── Configuration ──────────────────────────────────────────────────────────────

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]

# Simple Type1 question used for throughput burst test (deterministic, fast)
THROUGHPUT_QUESTION = "What was Apple's total revenue in FY2023?"

# Number of repetitions per concurrency level in the throughput burst
BURST_REPS = 10   # fire BURST_REPS requests at concurrency N, measure wall time

# Accuracy eval: run full 52-case suite at this concurrency (isolates quant effect)
ACCURACY_CONCURRENCY = 1


# ── Per-request worker ─────────────────────────────────────────────────────────

def _single_request(args):
    """Run one timed_answer call. Used inside ThreadPoolExecutor."""
    question, model, tok, nl2sql_model, nl2sql_tok, retriever = args
    from eval_system import timed_answer
    t0 = time.time()
    try:
        result, timings = timed_answer(
            question, model, tok, nl2sql_model, nl2sql_tok, retriever
        )
        latency = time.time() - t0
        return {"ok": True, "latency": latency, "timings": timings}
    except Exception as exc:
        return {"ok": False, "latency": time.time() - t0, "error": str(exc)}


# ── Throughput burst test ──────────────────────────────────────────────────────

def burst_test(concurrency: int, model, tok, nl2sql_model, nl2sql_tok,
               retriever) -> dict:
    """
    Fire BURST_REPS requests at the given concurrency, measure QPS and latency.

    Uses ThreadPoolExecutor: `concurrency` threads each call `_single_request`.
    vLLM's continuous batching scheduler sees all simultaneous requests and
    packs them into forward passes — SGMV batches base + nl2sql in one pass.
    """
    print(f"  [Burst] concurrency={concurrency}  reps={BURST_REPS}", flush=True)

    args = (THROUGHPUT_QUESTION, model, tok, nl2sql_model, nl2sql_tok, retriever)
    all_latencies = []
    errors = 0

    # Fire BURST_REPS requests in batches of `concurrency`
    n_batches = max(1, BURST_REPS // concurrency)
    t_wall_start = time.time()

    for _ in range(n_batches):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_single_request, args) for _ in range(concurrency)]
            for f in futs:
                r = f.result()
                if r["ok"]:
                    all_latencies.append(r["latency"])
                else:
                    errors += 1

    t_wall = time.time() - t_wall_start
    total_requests = n_batches * concurrency
    qps = total_requests / t_wall

    sorted_lat = sorted(all_latencies) if all_latencies else [0]
    n = len(sorted_lat)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "errors": errors,
        "wall_time_s": round(t_wall, 3),
        "qps": round(qps, 3),
        "latency_mean": round(sum(sorted_lat) / n, 3),
        "latency_p50":  round(sorted_lat[n // 2], 3),
        "latency_p95":  round(sorted_lat[int(n * 0.95)], 3),
        "latency_min":  round(sorted_lat[0], 3),
        "latency_max":  round(sorted_lat[-1], 3),
    }


# ── Accuracy eval at fixed concurrency ────────────────────────────────────────

def accuracy_eval(model, tok, nl2sql_model, nl2sql_tok, retriever) -> dict:
    """
    Run full 52-case eval at ACCURACY_CONCURRENCY=1 to isolate quantization effect.
    Returns aggregate accuracy metrics.
    """
    from eval_system import TEST_CASES
    from eval_benchmark import run_case, aggregate

    print(f"\n  [Accuracy] Running {len(TEST_CASES)} cases at concurrency=1...",
          flush=True)

    records = []
    for i, case in enumerate(TEST_CASES, 1):
        rec = run_case(
            idx=i, total=len(TEST_CASES), case=case,
            model=model, tok=tok,
            nl2sql_model=nl2sql_model, nl2sql_tok=nl2sql_tok,
            retriever=retriever,
            oai_client=None, no_fluency=True,
        )
        records.append(rec)

    agg = aggregate(records)
    return {
        "n_cases": len(records),
        "intent_accuracy":  agg["intent_accuracy"],
        "value_accuracy":   agg["value_accuracy"],
        "keyword_hit_rate": agg["keyword_hit_rate"],
        "fluency_mean":     agg["fluency_mean"],
        "latency":          agg["latency"],
    }


# ── vLLM setup ────────────────────────────────────────────────────────────────

def setup_vllm() -> tuple:
    """Check vLLM health, load vectordb only (model/tok = None for vLLM mode)."""
    from deployment.api.client import VLLMClient
    from chatbot import load_vectordb

    client = VLLMClient()
    if not client.health():
        print("ERROR: vLLM server not reachable at localhost:8001", file=sys.stderr)
        print("  Start it: bash deployment/scripts/start_server_quant.sh <quant>",
              file=sys.stderr)
        sys.exit(1)
    print("vLLM server is healthy.", flush=True)
    retriever = load_vectordb()
    return None, None, None, None, retriever


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantization × Concurrency sweep for vLLM serving"
    )
    p.add_argument(
        "--quant", required=True,
        choices=["fp16", "int8", "awq4"],
        help="Quantization level matching the running vLLM server",
    )
    p.add_argument(
        "--concurrency", type=str, default=None,
        metavar="LIST",
        help="Comma-separated concurrency levels to test (default: 1,2,4,8,16)",
    )
    p.add_argument(
        "--burst-reps", type=int, default=BURST_REPS,
        help=f"Total requests per burst test (default: {BURST_REPS})",
    )
    p.add_argument(
        "--skip-accuracy", action="store_true",
        help="Skip the 52-case accuracy eval (throughput only)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: eval_results/sweep_{quant}_{ts}.json)",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    concurrency_levels = CONCURRENCY_LEVELS
    if args.concurrency:
        concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    global BURST_REPS
    BURST_REPS = args.burst_reps

    print(f"\n{'='*60}", flush=True)
    print(f"SWEEP  quant={args.quant}  concurrency={concurrency_levels}", flush=True)
    print(f"{'='*60}\n", flush=True)

    model, tok, nl2sql_model, nl2sql_tok, retriever = setup_vllm()

    # ── Throughput sweep ───────────────────────────────────────────────────────
    print("[Phase 1] Throughput burst tests across concurrency levels", flush=True)
    throughput_results = []
    for c in concurrency_levels:
        result = burst_test(c, model, tok, nl2sql_model, nl2sql_tok, retriever)
        print(f"    concurrency={c:2d}  QPS={result['qps']:.3f}  "
              f"p50={result['latency_p50']:.2f}s  "
              f"p95={result['latency_p95']:.2f}s  "
              f"errors={result['errors']}", flush=True)
        throughput_results.append(result)

    # ── Accuracy eval ──────────────────────────────────────────────────────────
    accuracy_result = None
    if not args.skip_accuracy:
        print(f"\n[Phase 2] Accuracy eval (concurrency=1, 52 cases)", flush=True)
        accuracy_result = accuracy_eval(
            model, tok, nl2sql_model, nl2sql_tok, retriever
        )
        print(f"  intent_acc={accuracy_result['intent_accuracy']}  "
              f"value_acc={accuracy_result['value_accuracy']}  "
              f"kw_hit={accuracy_result['keyword_hit_rate']}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"THROUGHPUT SUMMARY  quant={args.quant}", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"  {'concurr':>8}  {'QPS':>8}  {'p50(s)':>8}  {'p95(s)':>8}  "
          f"{'errors':>6}")
    for r in throughput_results:
        print(f"  {r['concurrency']:>8}  {r['qps']:>8.3f}  "
              f"{r['latency_p50']:>8.2f}  {r['latency_p95']:>8.2f}  "
              f"{r['errors']:>6}")

    if accuracy_result:
        print(f"\nACCURACY SUMMARY  quant={args.quant}")
        print(f"  intent_accuracy  : {accuracy_result['intent_accuracy']}")
        print(f"  value_accuracy   : {accuracy_result['value_accuracy']}")
        print(f"  keyword_hit_rate : {accuracy_result['keyword_hit_rate']}")
        lat = accuracy_result.get("latency", {})
        print(f"  latency mean     : {lat.get('total_mean')}s")
        print(f"  latency p50      : {lat.get('total_p50')}s")

    # ── Save results ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "quant": args.quant,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "concurrency_levels": concurrency_levels,
        "throughput": throughput_results,
        "accuracy": accuracy_result,
    }

    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"sweep_{args.quant}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
