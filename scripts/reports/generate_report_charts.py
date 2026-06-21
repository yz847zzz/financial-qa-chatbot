"""Generate comparison charts for the unified benchmark report."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval_results"
OUT_DIR = ROOT / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ─────────────────────────────────────────────────────────────

results = {}
for name, filename in [
    ("Local",     "unified_local_simple_20260502_230756.json"),
    ("GPT-4o",    "unified_gpt4o_gpt4o_20260502_230946.json"),
    ("vLLM FP16", "unified_vllm_fp16_20260503_004347.json"),
    ("vLLM INT8", "unified_vllm_int8_20260503_003311.json"),
    ("vLLM AWQ4", "unified_vllm_awq4_20260503_000512.json"),
]:
    path = EVAL_DIR / filename
    if path.exists():
        with open(path) as f:
            results[name] = json.load(f)

COLORS = {
    "Local":     "#6c757d",
    "GPT-4o":    "#0d6efd",
    "vLLM FP16": "#198754",
    "vLLM INT8": "#fd7e14",
    "vLLM AWQ4": "#dc3545",
}


def _agg(r):
    """Get aggregate dict from result."""
    return r.get("aggregate", {})


def _per_type(r, t):
    """Get per-type dict."""
    return _agg(r).get("per_type", {}).get(t, {})


# ── Chart 1: Composite Accuracy Bar Chart ────────────────────────────────────

def chart_accuracy():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.2

    overall = [_agg(results[c]).get("composite_score", 0) for c in configs]
    type1   = [_per_type(results[c], "Type1").get("composite_mean", 0) for c in configs]
    type2   = [_per_type(results[c], "Type2").get("composite_mean", 0) for c in configs]
    type3   = [_per_type(results[c], "Type3").get("composite_mean", 0) for c in configs]

    bars_overall = ax.bar(x - 1.5*width, overall, width, label="Overall", color="#212529", zorder=3)
    bars_t1      = ax.bar(x - 0.5*width, type1,   width, label="Type1 (fact)", color="#0d6efd", zorder=3)
    bars_t2      = ax.bar(x + 0.5*width, type2,   width, label="Type2 (qualitative)", color="#198754", zorder=3)
    bars_t3      = ax.bar(x + 1.5*width, type3,   width, label="Type3 (chat)", color="#fd7e14", zorder=3)

    for bar, val in zip(bars_overall, overall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Accuracy by Configuration (52 cases)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
    ax.text(len(configs)-0.5, 0.905, "0.9 target", color="red", alpha=0.5, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'accuracy_comparison.png'}")


# ── Chart 2: Throughput (QPS) vs Concurrency ─────────────────────────────────

def chart_throughput():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cfg in results:
        r = results[cfg]
        tp = r.get("throughput", [])
        if not tp:
            continue
        conc = [t["concurrency"] for t in tp]
        qps  = [t["qps"] for t in tp]
        ax.plot(conc, qps, "o-", label=cfg, color=COLORS[cfg], linewidth=2, markersize=7, zorder=3)
        ax.annotate(f"{qps[-1]:.1f}", (conc[-1], qps[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=9, color=COLORS[cfg], fontweight="bold")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title("Throughput vs Concurrency")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "throughput_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'throughput_comparison.png'}")


# ── Chart 3: Latency p50 vs Concurrency ──────────────────────────────────────

def chart_latency():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cfg in results:
        r = results[cfg]
        tp = r.get("throughput", [])
        if not tp:
            continue
        conc = [t["concurrency"] for t in tp]
        p50  = [t["latency_p50"] for t in tp]
        ax.plot(conc, p50, "s-", label=cfg, color=COLORS[cfg], linewidth=2, markersize=7, zorder=3)
        ax.annotate(f"{p50[-1]:.1f}s", (conc[-1], p50[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=9, color=COLORS[cfg], fontweight="bold")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency p50 (seconds)")
    ax.set_title("Median Latency vs Concurrency")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_yscale("log")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "latency_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'latency_comparison.png'}")


# ── Chart 4: Accuracy Detail horizontal bars ─────────────────────────────────

def chart_accuracy_detail():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    metrics = [
        ("Intent Accuracy", "intent_accuracy"),
        ("Value Accuracy", "value_accuracy"),
        ("Keyword Hit Rate", "keyword_hit_rate"),
    ]

    configs = list(results.keys())
    y_pos = np.arange(len(configs))

    for ax, (title, key) in zip(axes, metrics):
        vals = []
        for cfg in configs:
            v = _agg(results[cfg]).get(key, 0)
            if v is None:
                v = 0
            vals.append(v if v <= 1.0 else v / 100.0)

        colors = [COLORS[c] for c in configs]
        bars = ax.barh(y_pos, vals, color=colors, height=0.6, zorder=3)
        ax.set_xlim(0, 1.15)
        ax.set_title(title)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs if ax == axes[0] else [])

        for bar, val in zip(bars, vals):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", fontsize=9, fontweight="bold")

    plt.suptitle("Accuracy Detail by Configuration", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'accuracy_detail.png'}")


# ── Chart 5: Pipeline Latency Bar Chart ──────────────────────────────────────

def chart_pipeline_latency():
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = list(results.keys())
    x = np.arange(len(configs))

    lat_data = [_agg(results[c]).get("latency", {}) for c in configs]
    means = [d.get("total_mean", 0) for d in lat_data]
    p50s  = [d.get("total_p50", 0)  for d in lat_data]
    p95s  = [d.get("total_p95", 0)  for d in lat_data]

    width = 0.25
    ax.bar(x - width, means, width, label="Mean", color="#0d6efd", zorder=3)
    ax.bar(x,         p50s,  width, label="p50",  color="#198754", zorder=3)
    ax.bar(x + width, p95s,  width, label="p95",  color="#dc3545", zorder=3)

    max_val = max(max(means), max(p50s), max(p95s))
    offset = max_val * 0.02

    for i, (m, p5, p9) in enumerate(zip(means, p50s, p95s)):
        ax.text(i - width, m + offset, f"{m:.1f}", ha="center", fontsize=8)
        ax.text(i,         p5 + offset, f"{p5:.1f}", ha="center", fontsize=8)
        ax.text(i + width, p9 + offset, f"{p9:.1f}", ha="center", fontsize=8)

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("End-to-End Pipeline Latency (accuracy run, sequential)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "pipeline_latency.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'pipeline_latency.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating report charts...")
    chart_accuracy()
    chart_throughput()
    chart_latency()
    chart_accuracy_detail()
    chart_pipeline_latency()
    print("Done.")
