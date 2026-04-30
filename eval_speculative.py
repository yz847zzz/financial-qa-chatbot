"""
eval_speculative.py — Speculative decoding benchmark

Measures the throughput and latency improvement from using Llama-3.2-1B-Instruct
as a draft model for Llama-3.2-3B-Instruct, across different values of K
(speculative tokens per step) and quantization levels.

Requires a running vLLM server started with start_server_spec.sh.
The server must be restarted between K values (K is a server-side parameter).

Usage
─────
# Start server for each K, then run this script with the matching K:
#   (in WSL2)
#   bash deployment/scripts/start_server_spec.sh awq4 1
#   (in Windows, after server is ready)
#   python eval_speculative.py --quant awq4 --spec-tokens 1

# Or run a full automated sweep (you restart the server manually between K values):
#   python eval_speculative.py --quant awq4 --spec-tokens 1 2 3 4 5

# Compare all results and plot:
#   python eval_speculative.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

# ── Configuration ──────────────────────────────────────────────────────────────
VLLM_BASE      = "http://localhost:8001/v1"
METRICS_URL    = "http://localhost:8001/metrics"
RESULTS_DIR    = Path("eval_results")
BURST_REPS     = 20          # total requests per concurrency burst
CONCURRENCY    = 8           # fixed concurrency for spec sweep (from quant sweep sweet-spot)
MAX_TOKENS     = 120
MODEL_NAME     = "base"

# ── Representative financial QA prompts ────────────────────────────────────────
PROMPTS = [
    "What was Apple's total revenue in fiscal year 2023?",
    "How did Microsoft describe their cloud strategy in their annual report?",
    "What is NVIDIA's gross margin trend over the past three years?",
    "Explain the key risk factors mentioned in Amazon's most recent 10-K filing.",
    "What was Google's net income in Q4 2022?",
    "How does Meta plan to monetize its metaverse investments according to their filings?",
    "What is Tesla's capital expenditure for 2022 and 2023?",
    "Describe the competitive landscape section from Netflix's latest annual report.",
    "What was Apple's EPS in fiscal 2023?",
    "How did rising interest rates affect bank stocks according to their 10-K filings?",
]


# ── vLLM helpers ──────────────────────────────────────────────────────────────
def wait_for_server(timeout: int = 120) -> bool:
    """Poll health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:8001/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def chat_once(prompt: str, client: httpx.Client) -> dict:
    """Send one chat completion and return timing + token info."""
    t0 = time.perf_counter()
    resp = client.post(
        f"{VLLM_BASE}/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        timeout=60,
    )
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def burst_test(concurrency: int, n_requests: int, client: httpx.Client) -> dict:
    """Fire n_requests in batches of `concurrency`, measure QPS and latency."""
    prompts_cycle = PROMPTS * ((n_requests // len(PROMPTS)) + 1)
    selected = prompts_cycle[:n_requests]

    latencies: list[float] = []
    errors = 0
    t_start = time.perf_counter()

    # Send in batches of concurrency
    for batch_start in range(0, n_requests, concurrency):
        batch = selected[batch_start: batch_start + concurrency]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [pool.submit(chat_once, p, client) for p in batch]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    latencies.append(r["latency"])
                except Exception as e:
                    errors += 1

    wall = time.perf_counter() - t_start
    latencies.sort()
    n = len(latencies)
    return {
        "concurrency":    concurrency,
        "total_requests": n_requests,
        "errors":         errors,
        "wall_time_s":    round(wall, 3),
        "qps":            round((n_requests - errors) / wall, 4),
        "latency_mean":   round(sum(latencies) / n, 3) if n else 0,
        "latency_p50":    round(latencies[int(n * 0.50)], 3) if n else 0,
        "latency_p95":    round(latencies[int(n * 0.95)], 3) if n else 0,
        "latency_min":    round(latencies[0], 3) if n else 0,
        "latency_max":    round(latencies[-1], 3) if n else 0,
    }


# ── Prometheus metrics scraper ─────────────────────────────────────────────────
def scrape_acceptance_rate() -> float | None:
    """
    Read the speculative decoding acceptance rate from vLLM's /metrics endpoint.
    Returns the acceptance rate (0–1) or None if not available.
    """
    try:
        resp = httpx.get(METRICS_URL, timeout=5)
        resp.raise_for_status()
        text = resp.text
        # Look for: vllm:spec_decode_draft_acceptance_rate{...} <value>
        match = re.search(
            r'vllm:spec_decode_draft_acceptance_rate\{[^}]*\}\s+([\d.]+)',
            text,
        )
        if match:
            return float(match.group(1))
        # Fallback: ngram or older vLLM naming
        match = re.search(
            r'vllm:spec_decode_accepted_tokens_total\{[^}]*\}\s+([\d.]+)',
            text,
        )
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def scrape_avg_accepted_tokens() -> float | None:
    """
    Read mean accepted tokens per step from Prometheus metrics.
    """
    try:
        resp = httpx.get(METRICS_URL, timeout=5)
        resp.raise_for_status()
        text = resp.text
        # vLLM metric: vllm:spec_decode_num_accepted_tokens_per_pos
        match = re.search(
            r'vllm:spec_decode_num_accepted_tokens_per_pos_bucket\{[^}]*le="(\d+)"[^}]*\}\s+([\d.]+)',
            text,
        )
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


# ── Main benchmark function ───────────────────────────────────────────────────
def run_spec_benchmark(
    quant: str,
    spec_tokens: int,
    concurrency: int,
    n_requests: int,
) -> dict:
    """Run one (quant, K) configuration and return results."""
    print(f"\n{'─'*55}")
    print(f"  quant={quant}  K={spec_tokens}  concurrency={concurrency}")
    print(f"{'─'*55}")

    if not wait_for_server(timeout=10):
        raise RuntimeError(
            "vLLM server not responding. Start it first:\n"
            f"  bash deployment/scripts/start_server_spec.sh {quant} {spec_tokens}"
        )

    # Warm-up
    with httpx.Client() as client:
        print("  Warming up...")
        for _ in range(3):
            try:
                chat_once(PROMPTS[0], client)
            except Exception:
                pass

        # Reset metrics baseline
        acceptance_before = scrape_acceptance_rate()

        print(f"  Bursting {n_requests} requests at concurrency={concurrency}...")
        results = burst_test(concurrency, n_requests, client)

        # Read acceptance rate after the burst
        acceptance_after = scrape_acceptance_rate()

    # Acceptance rate: use the post-burst reading (cumulative since server start)
    acceptance_rate = acceptance_after

    print(f"  QPS          : {results['qps']:.3f}")
    print(f"  p50 latency  : {results['latency_p50']:.2f}s")
    print(f"  p95 latency  : {results['latency_p95']:.2f}s")
    if acceptance_rate is not None:
        print(f"  acceptance   : {acceptance_rate:.2%}")
    else:
        print(f"  acceptance   : (not available — check /metrics endpoint)")

    return {
        "quant":           quant,
        "spec_tokens":     spec_tokens,
        "concurrency":     concurrency,
        "n_requests":      n_requests,
        "qps":             results["qps"],
        "latency_p50":     results["latency_p50"],
        "latency_p95":     results["latency_p95"],
        "latency_mean":    results["latency_mean"],
        "acceptance_rate": acceptance_rate,
        "raw":             results,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results_dir: Path) -> None:
    """Load all speculative sweep JSONs and generate comparison plots."""
    import matplotlib.pyplot as plt
    import numpy as np

    spec_files = sorted(results_dir.glob("spec_*.json"))
    if not spec_files:
        print("No speculative sweep results found in eval_results/")
        return

    # Load and group by quant
    from collections import defaultdict
    by_quant: dict[str, list[dict]] = defaultdict(list)
    for f in spec_files:
        d = json.loads(f.read_text())
        by_quant[d["quant"]].append(d)

    # Also load baseline (no spec) from the quant sweep
    baseline: dict[str, dict] = {}  # quant → {qps, p50, p95}
    for f in results_dir.glob("sweep_*.json"):
        d = json.loads(f.read_text())
        q = d["quant"]
        # Find the row for CONCURRENCY match
        for row in d["throughput"]:
            if row["concurrency"] == CONCURRENCY:
                baseline[q] = {
                    "qps": row["qps"],
                    "p50": row["latency_p50"],
                    "p95": row["latency_p95"],
                }
                break

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    colors = {"fp16": "#4C72B0", "int8": "#DD8452", "awq4": "#55A868"}

    # ── Figure 1: QPS vs K ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Speculative Decoding: Llama-3.2-1B draft → 3B target", fontsize=13, y=1.01)

    ax = axes[0]
    for quant, rows in by_quant.items():
        rows.sort(key=lambda x: x["spec_tokens"])
        ks    = [r["spec_tokens"] for r in rows]
        qps   = [r["qps"] for r in rows]
        color = colors.get(quant, "gray")
        ax.plot(ks, qps, "o-", color=color, label=f"{quant} (spec)", linewidth=2, markersize=7)
        if quant in baseline:
            ax.axhline(baseline[quant]["qps"], color=color, linestyle="--",
                       alpha=0.5, label=f"{quant} (baseline, K=0)")
    ax.set_xlabel("Speculative tokens K")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title(f"Throughput vs K  (concurrency={CONCURRENCY})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted({r["spec_tokens"] for rows in by_quant.values() for r in rows}))

    # ── Figure 2: p50 Latency vs K ──────────────────────────────────────────
    ax = axes[1]
    for quant, rows in by_quant.items():
        rows.sort(key=lambda x: x["spec_tokens"])
        ks  = [r["spec_tokens"] for r in rows]
        p50 = [r["latency_p50"] for r in rows]
        p95 = [r["latency_p95"] for r in rows]
        color = colors.get(quant, "gray")
        ax.plot(ks, p50, "o-", color=color, label=f"{quant} p50", linewidth=2, markersize=7)
        ax.plot(ks, p95, "s--", color=color, label=f"{quant} p95", linewidth=1.5,
                markersize=5, alpha=0.7)
        if quant in baseline:
            ax.axhline(baseline[quant]["p50"], color=color, linestyle=":",
                       alpha=0.5, label=f"{quant} baseline p50")
    ax.set_xlabel("Speculative tokens K")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title(f"Latency vs K  (concurrency={CONCURRENCY})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted({r["spec_tokens"] for rows in by_quant.values() for r in rows}))

    plt.tight_layout()
    out = plots_dir / "spec_throughput_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── Figure 3: Acceptance rate vs K ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for quant, rows in by_quant.items():
        rows_with_acc = [r for r in rows if r.get("acceptance_rate") is not None]
        if not rows_with_acc:
            continue
        rows_with_acc.sort(key=lambda x: x["spec_tokens"])
        ks  = [r["spec_tokens"] for r in rows_with_acc]
        acc = [r["acceptance_rate"] for r in rows_with_acc]
        ax.plot(ks, acc, "o-", color=colors.get(quant, "gray"),
                label=quant, linewidth=2, markersize=7)
    ax.set_xlabel("Speculative tokens K")
    ax.set_ylabel("Draft acceptance rate")
    ax.set_title("1B Draft Token Acceptance Rate vs K")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = plots_dir / "spec_acceptance_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── Figure 4: Speedup vs K (relative to no-spec baseline) ───────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for quant, rows in by_quant.items():
        if quant not in baseline:
            continue
        base_qps = baseline[quant]["qps"]
        rows.sort(key=lambda x: x["spec_tokens"])
        ks      = [r["spec_tokens"] for r in rows]
        speedup = [r["qps"] / base_qps for r in rows]
        ax.plot(ks, speedup, "o-", color=colors.get(quant, "gray"),
                label=quant, linewidth=2, markersize=7)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="baseline (1×)")
    ax.set_xlabel("Speculative tokens K")
    ax.set_ylabel("Speedup over baseline (×)")
    ax.set_title(f"Throughput Speedup from Speculative Decoding  (c={CONCURRENCY})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = plots_dir / "spec_speedup.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    # ── Text summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SPECULATIVE DECODING SUMMARY")
    print("═" * 60)
    print(f"  {'quant':<8} {'K':<4} {'QPS':>7} {'speedup':>8} {'p50':>7} {'accept':>8}")
    print(f"  {'─'*8} {'─'*4} {'─'*7} {'─'*8} {'─'*7} {'─'*8}")
    for quant in sorted(by_quant):
        if quant in baseline:
            base_qps = baseline[quant]["qps"]
            base_p50 = baseline[quant]["p50"]
            print(f"  {quant:<8} {'—':<4} {base_qps:>7.3f} {'1.00×':>8} {base_p50:>7.2f} {'(baseline)':>8}")
        for r in sorted(by_quant[quant], key=lambda x: x["spec_tokens"]):
            acc_str = f"{r['acceptance_rate']:.1%}" if r.get("acceptance_rate") else "  n/a"
            speedup = (r["qps"] / baseline[quant]["qps"]) if quant in baseline else 0
            print(f"  {quant:<8} {r['spec_tokens']:<4} {r['qps']:>7.3f} {speedup:>7.2f}× "
                  f"{r['latency_p50']:>7.2f} {acc_str:>8}")
    print("═" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Speculative decoding benchmark for the Financial QA chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark K=4 with awq4 (server must already be running with K=4):
  python eval_speculative.py --quant awq4 --spec-tokens 4

  # Record results for multiple K values (restart server between each):
  python eval_speculative.py --quant awq4 --spec-tokens 1 2 3 4 5

  # Just regenerate plots from previously saved JSONs:
  python eval_speculative.py --plot-only
        """,
    )
    p.add_argument("--quant", choices=["fp16", "int8", "awq4"], default="awq4",
                   help="Quantization level of the running server (default: awq4)")
    p.add_argument("--spec-tokens", type=int, nargs="+", default=[4],
                   metavar="K",
                   help="Speculative token counts to benchmark (default: 4). "
                        "Server must be running with the matching K.")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY,
                   help=f"Request concurrency (default: {CONCURRENCY})")
    p.add_argument("--n-requests", type=int, default=BURST_REPS,
                   help=f"Total requests per K (default: {BURST_REPS})")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip benchmarking — just regenerate plots from saved JSONs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.plot_only:
        print("Generating plots from saved results...")
        plot_results(RESULTS_DIR)
        return

    all_results = []
    for k in args.spec_tokens:
        if len(args.spec_tokens) > 1:
            print(f"\n{'='*55}")
            print(f"  NOTE: For K={k}, the server must be running with:")
            print(f"    bash deployment/scripts/start_server_spec.sh {args.quant} {k}")
            input("  Press ENTER when the server is ready... ")

        result = run_spec_benchmark(
            quant=args.quant,
            spec_tokens=k,
            concurrency=args.concurrency,
            n_requests=args.n_requests,
        )
        all_results.append(result)

        # Save per-K result immediately
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"spec_{args.quant}_K{k}_{ts}.json"
        out.write_text(json.dumps({**result, "timestamp": ts}, indent=2))
        print(f"\n  Saved → {out}")

    if len(all_results) > 1:
        print("\nGenerating plots...")
        plot_results(RESULTS_DIR)

    print("\nDone. To plot all results at any time:")
    print("  python eval_speculative.py --plot-only")


if __name__ == "__main__":
    main()
