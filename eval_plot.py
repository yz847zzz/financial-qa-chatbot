"""
eval_plot.py — Visualise the quantization × concurrency sweep results.

Reads one or more sweep_*.json files (produced by eval_sweep.py) and draws:

  Figure 1 — Throughput (QPS) vs Concurrency
      One line per quantization level.  X = concurrency, Y = QPS.

  Figure 2 — Latency (p50 / p95) vs Concurrency
      Solid lines = p50, dashed lines = p95. One colour per quant level.

  Figure 3 — Accuracy vs Quantization
      Grouped bar chart: intent_acc / value_acc / keyword_hit_rate.
      (Only drawn if at least one sweep file includes accuracy data.)

  Figure 4 (optional) — 3-D surface: QPS over (concurrency, quant_index)
      Enabled with --3d flag.

Usage
─────
  python eval_plot.py eval_results/sweep_fp16_*.json \\
                      eval_results/sweep_int8_*.json \\
                      eval_results/sweep_awq4_*.json

  # Limit to throughput plots (no accuracy bars):
  python eval_plot.py eval_results/sweep_fp16_20260413_*.json --no-accuracy

  # Save figures instead of showing interactively:
  python eval_plot.py eval_results/sweep_*.json --save eval_results/plots/

  # Include 3-D surface:
  python eval_plot.py eval_results/sweep_*.json --3d
"""

import argparse
import json
import sys
from pathlib import Path

# ── Lazy matplotlib import so the module can be imported without a display ─────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe on Windows/WSL2/headless)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Colour scheme ──────────────────────────────────────────────────────────────
QUANT_COLOURS = {
    "fp16":  "#2196F3",   # blue
    "int8":  "#4CAF50",   # green
    "awq4":  "#FF5722",   # deep orange
}
QUANT_ORDER = ["fp16", "int8", "awq4"]
QUANT_LABELS = {
    "fp16": "fp16 (baseline, ~6 GB VRAM)",
    "int8": "int8 bitsandbytes (~3 GB VRAM)",
    "awq4": "AWQ INT4 (~1.5 GB VRAM)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sweep_files(paths: list[Path]) -> dict[str, dict]:
    """
    Load sweep JSON files and merge by quantization level.

    If multiple files share the same quant level (e.g. reruns), the one with
    the highest QPS at concurrency=1 wins (most recent successful run).

    Returns: {quant: {"throughput": [...], "accuracy": {...}|None}}
    """
    best: dict[str, dict] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"WARN: could not load {path}: {e}", file=sys.stderr)
            continue

        quant = data.get("quant", "unknown")
        tp = data.get("throughput", [])
        if not tp:
            print(f"WARN: no throughput data in {path}", file=sys.stderr)
            continue

        # Pick the run with the highest QPS at the lowest concurrency
        c1_qps = next((r["qps"] for r in tp if r["concurrency"] == 1), 0)

        if quant not in best:
            best[quant] = data
        else:
            existing_c1 = next(
                (r["qps"] for r in best[quant]["throughput"] if r["concurrency"] == 1),
                0,
            )
            if c1_qps > existing_c1:
                best[quant] = data

    if not best:
        print("ERROR: no valid sweep files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded quantization levels: {sorted(best.keys())}")
    return best


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _sorted_quants(data: dict[str, dict]) -> list[str]:
    """Return quant keys in canonical order, with any extras appended."""
    ordered = [q for q in QUANT_ORDER if q in data]
    extras  = [q for q in sorted(data) if q not in QUANT_ORDER]
    return ordered + extras


def fig_throughput(data: dict[str, dict], ax=None) -> plt.Figure:
    """Figure 1: QPS vs concurrency, one line per quant."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for quant in _sorted_quants(data):
        info = data[quant]
        rows = sorted(info["throughput"], key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in rows]
        ys = [r["qps"] for r in rows]
        colour = QUANT_COLOURS.get(quant, None)
        label  = QUANT_LABELS.get(quant, quant)
        ax.plot(xs, ys, marker="o", linewidth=2, color=colour, label=label)

    ax.set_xlabel("Concurrency (parallel requests)", fontsize=12)
    ax.set_ylabel("Throughput (QPS)", fontsize=12)
    ax.set_title("Throughput vs Concurrency by Quantization", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
    return fig


def fig_latency(data: dict[str, dict], ax=None) -> plt.Figure:
    """Figure 2: p50 (solid) and p95 (dashed) latency vs concurrency."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for quant in _sorted_quants(data):
        info = data[quant]
        rows = sorted(info["throughput"], key=lambda r: r["concurrency"])
        xs   = [r["concurrency"] for r in rows]
        p50  = [r["latency_p50"] for r in rows]
        p95  = [r["latency_p95"] for r in rows]
        colour = QUANT_COLOURS.get(quant, None)
        label  = QUANT_LABELS.get(quant, quant)
        ax.plot(xs, p50, marker="o", linewidth=2, color=colour, label=f"{label} p50")
        ax.plot(xs, p95, marker="s", linewidth=2, color=colour, linestyle="--",
                label=f"{label} p95", alpha=0.75)

    ax.set_xlabel("Concurrency (parallel requests)", fontsize=12)
    ax.set_ylabel("Latency (s)", fontsize=12)
    ax.set_title("Latency p50 / p95 vs Concurrency", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
    return fig


def fig_accuracy(data: dict[str, dict]) -> plt.Figure | None:
    """Figure 3: grouped bar chart of accuracy metrics vs quantization."""
    quants  = _sorted_quants(data)
    metrics = ["intent_accuracy", "value_accuracy", "keyword_hit_rate"]
    labels  = ["Intent Acc", "Value Acc", "Keyword Hit"]

    # Collect values (None if accuracy not in file)
    vals: dict[str, list[float | None]] = {m: [] for m in metrics}
    has_any = False
    for quant in quants:
        acc = data[quant].get("accuracy") or {}
        for m in metrics:
            v = acc.get(m)
            vals[m].append(v)
            if v is not None:
                has_any = True

    if not has_any:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    n_quants  = len(quants)
    n_metrics = len(metrics)
    bar_w     = 0.22
    x         = np.arange(n_quants)

    offsets = np.linspace(-(n_metrics - 1) * bar_w / 2,
                           (n_metrics - 1) * bar_w / 2, n_metrics)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ys     = [v if v is not None else 0 for v in vals[metric]]
        hatches = ["" if v is not None else "//" for v in vals[metric]]
        bars   = ax.bar(x + offsets[i], ys, bar_w, label=label)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([QUANT_LABELS.get(q, q) for q in quants], fontsize=9)
    ax.set_ylabel("Score (0–1)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy Metrics vs Quantization Level", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_3d_surface(data: dict[str, dict]) -> plt.Figure | None:
    """Figure 4: 3-D QPS surface over (concurrency, quant_index)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    quants = _sorted_quants(data)
    if len(quants) < 2:
        print("WARN: need ≥2 quant levels for 3-D surface; skipping.", file=sys.stderr)
        return None

    # Build aligned concurrency axis (union of all levels)
    all_conc = sorted(
        {r["concurrency"] for info in data.values() for r in info["throughput"]}
    )

    # QPS matrix: rows = quant index, cols = concurrency
    qps_matrix = np.zeros((len(quants), len(all_conc)))
    for qi, quant in enumerate(quants):
        row_map = {r["concurrency"]: r["qps"] for r in data[quant]["throughput"]}
        for ci, c in enumerate(all_conc):
            qps_matrix[qi, ci] = row_map.get(c, np.nan)

    X_raw = np.array(all_conc, dtype=float)
    Y_raw = np.arange(len(quants), dtype=float)
    X, Y  = np.meshgrid(X_raw, Y_raw)

    fig  = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(
        np.log2(X), Y, qps_matrix,
        cmap=cm.viridis, edgecolor="none", alpha=0.85
    )
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label="QPS")

    ax3d.set_xlabel("log₂(Concurrency)", fontsize=10)
    ax3d.set_ylabel("Quantization index", fontsize=10)
    ax3d.set_zlabel("Throughput (QPS)", fontsize=10)
    ax3d.set_yticks(Y_raw)
    ax3d.set_yticklabels([QUANT_LABELS.get(q, q) for q in quants], fontsize=7)
    ax3d.set_title("QPS Surface: Quantization × Concurrency", fontsize=12)
    fig.tight_layout()
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(data: dict[str, dict]) -> None:
    """Print a compact comparison table to stdout."""
    quants = _sorted_quants(data)

    print("\n" + "=" * 72)
    print("THROUGHPUT SUMMARY (QPS)")
    print("-" * 72)
    # header
    header = f"{'conc':>6}" + "".join(f"  {q:>10}" for q in quants)
    print(header)

    all_conc = sorted(
        {r["concurrency"] for info in data.values() for r in info["throughput"]}
    )
    for c in all_conc:
        row = f"{c:>6}"
        for quant in quants:
            row_map = {r["concurrency"]: r["qps"] for r in data[quant]["throughput"]}
            qps = row_map.get(c)
            row += f"  {qps:>10.3f}" if qps is not None else f"  {'—':>10}"
        print(row)

    print("\nACCURACY SUMMARY")
    print("-" * 72)
    metrics = [("intent_accuracy", "intent_acc"),
               ("value_accuracy",  "value_acc"),
               ("keyword_hit_rate", "kw_hit")]
    for key, label in metrics:
        row = f"  {label:<18}"
        for quant in quants:
            acc = data[quant].get("accuracy") or {}
            v = acc.get(key)
            row += f"  {v:>10.4f}" if v is not None else f"  {'n/a':>10}"
        print(row)
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot quantization × concurrency sweep results"
    )
    p.add_argument(
        "files", nargs="+",
        help="sweep_*.json files to load (one or more per quant level)",
    )
    p.add_argument(
        "--save", metavar="DIR", default=None,
        help="Save PNGs to this directory instead of showing interactively",
    )
    p.add_argument(
        "--no-accuracy", action="store_true",
        help="Skip the accuracy bar chart",
    )
    p.add_argument(
        "--3d", dest="three_d", action="store_true",
        help="Include 3-D surface plot (requires ≥2 quant levels)",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for saved figures (default: 150)",
    )
    return p


def main() -> None:
    if not HAS_MPL:
        print("ERROR: matplotlib is not installed.  pip install matplotlib numpy",
              file=sys.stderr)
        sys.exit(1)

    # Import ticker here so the error above fires first
    import matplotlib.ticker  # noqa: F401

    args = build_parser().parse_args()

    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: file not found: {m}", file=sys.stderr)
        sys.exit(1)

    data = load_sweep_files(paths)
    print_summary(data)

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, plt.Figure]] = []

    figures.append(("throughput_vs_concurrency", fig_throughput(data)))
    figures.append(("latency_vs_concurrency",    fig_latency(data)))

    if not args.no_accuracy:
        f = fig_accuracy(data)
        if f:
            figures.append(("accuracy_vs_quant", f))
        else:
            print("NOTE: no accuracy data found — skipping accuracy figure.")

    if args.three_d:
        f = fig_3d_surface(data)
        if f:
            figures.append(("qps_surface_3d", f))

    if save_dir:
        for name, fig in figures:
            out = save_dir / f"{name}.png"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close("all")
    else:
        # Interactive display: switch to a displayable backend if possible
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        plt.show()


if __name__ == "__main__":
    main()
