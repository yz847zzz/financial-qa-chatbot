"""
Metric Frequency Analysis — the FINDMIND-style heatmap pass.

FINDMIND's approach:
  1. Extract ALL key-value pairs from every filing table (first pass — no filtering)
  2. Build a frequency matrix: rows = metric names, columns = tickers
  3. Visualise as a heatmap to see which metrics are "universal" vs one-offs
  4. Pick a threshold (e.g. appears in ≥30% of tickers) → canonical metric list
  5. Only canonical metrics go into SQL; the rest are discarded

This ensures the SQL schema is clean, consistent, and useful for NL2SQL —
not polluted with one-off footnote labels like "Less: accumulated impairment (2019)".

Usage (called by scripts/run_metric_analysis.py):
    from data_pipeline.processing.metric_heatmap import MetricAnalyzer

    analyzer = MetricAnalyzer()
    analyzer.add_rows(rows, ticker="AAPL")
    ...
    canonical = analyzer.compute_canonical(threshold=0.30)
    analyzer.plot_heatmap("output/metric_heatmap.png")
    analyzer.save_canonical("output/canonical_metrics.json")
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

from ..metadata.models import FinancialRow


class MetricAnalyzer:
    """
    Collects FinancialRow objects across multiple tickers and computes
    metric frequency statistics.

    Internal state:
        _ticker_metrics: dict[ticker, set[metric_name]]
            — tracks which unique metrics were seen for each ticker
        _metric_statement: dict[metric_name, Counter[statement_type]]
            — tracks the most common statement classification for each metric
        _metric_value_count: dict[metric_name, int]
            — total number of (ticker, period) pairs with this metric
    """

    def __init__(self) -> None:
        # ticker → set of distinct metric names seen
        self._ticker_metrics: dict[str, set[str]] = defaultdict(set)
        # metric → {statement: count}
        self._metric_statement: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # metric → total value appearances (for secondary ranking)
        self._metric_value_count: dict[str, int] = defaultdict(int)
        self._all_tickers: set[str] = set()

    def add_rows(self, rows: list[FinancialRow], ticker: str) -> None:
        """
        Register a batch of FinancialRow objects for a given ticker.
        Only rows with a non-None value are counted.
        """
        self._all_tickers.add(ticker)
        for row in rows:
            if row.value is None:
                continue
            metric = _normalise_metric_key(row.metric)
            if not metric or len(metric) < 4:
                continue
            self._ticker_metrics[ticker].add(metric)
            self._metric_statement[metric][row.statement] += 1
            self._metric_value_count[metric] += 1

    @property
    def n_tickers(self) -> int:
        return len(self._all_tickers)

    def build_frequency_matrix(self) -> pd.DataFrame:
        """
        Build a binary presence matrix: rows = metrics, columns = tickers.
        Cell = 1 if ticker has that metric, 0 otherwise.
        """
        all_metrics = sorted(
            set(m for metrics in self._ticker_metrics.values() for m in metrics)
        )
        all_tickers = sorted(self._all_tickers)

        ticker_set = {t: self._ticker_metrics[t] for t in all_tickers}
        data = {
            ticker: [1 if m in ticker_set[ticker] else 0 for m in all_metrics]
            for ticker in all_tickers
        }
        df = pd.DataFrame(data, index=all_metrics)
        # Sort rows by total frequency descending
        df["_total"] = df.sum(axis=1)
        df = df.sort_values("_total", ascending=False).drop(columns="_total")
        return df

    def compute_canonical(
        self,
        threshold: float = 0.30,
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Determine the canonical metric list.

        A metric is canonical if it appears in ≥ threshold fraction of all tickers.

        Args:
            threshold: minimum fraction of tickers (0.0–1.0). Default: 0.30
            top_n:     optional hard cap on number of canonical metrics

        Returns:
            list of dicts: {metric, statement, ticker_count, ticker_pct, canonical_name}
            sorted by ticker_pct descending.
        """
        if self.n_tickers == 0:
            return []

        # Collect all unique metric keys across every ticker
        all_metric_keys: set[str] = set()
        for metric_set in self._ticker_metrics.values():
            all_metric_keys.update(metric_set)

        canonical = []
        for metric in all_metric_keys:
            ticker_count = sum(
                1 for t in self._all_tickers if metric in self._ticker_metrics[t]
            )
            pct = ticker_count / self.n_tickers

            if pct < threshold:
                continue

            # Most frequent statement classification for this metric
            stmt_counts = self._metric_statement[metric]
            statement = max(stmt_counts, key=stmt_counts.get) if stmt_counts else "unknown"

            canonical.append({
                "metric":         metric,
                "statement":      statement,
                "ticker_count":   ticker_count,
                "ticker_pct":     round(pct, 4),
                "canonical_name": _to_canonical_name(metric),
            })

        canonical.sort(key=lambda x: (-x["ticker_pct"], x["metric"]))
        if top_n is not None:
            canonical = canonical[:top_n]

        logger.info(
            f"Canonical metrics: {len(canonical)} at threshold={threshold:.0%} "
            f"across {self.n_tickers} tickers"
        )
        return canonical

    def canonical_metric_set(self, threshold: float = 0.30) -> set[str]:
        """Return just the set of canonical metric keys (for filtering in ingest pass)."""
        return {row["metric"] for row in self.compute_canonical(threshold=threshold)}

    def plot_heatmap(
        self,
        output_path: str | Path,
        top_n: int = 60,
        threshold: float = 0.30,
        figsize: tuple[int, int] = (28, 18),
    ) -> None:
        """
        Plot a binary heatmap: metric × ticker, coloured by presence.
        Shows only the top_n most frequent metrics.

        Saves PNG to output_path.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping heatmap")
            return

        df = self.build_frequency_matrix()
        # Keep only top_n rows
        df_top = df.head(top_n)

        # Add ticker_pct column for annotation
        canonical = self.compute_canonical(threshold=0.0)  # all metrics for labelling
        pct_map = {r["metric"]: r["ticker_pct"] for r in canonical}

        row_labels = [
            f"{m}  ({pct_map.get(m, 0):.0%})"
            for m in df_top.index
        ]

        fig, ax = plt.subplots(figsize=figsize)

        # Green = present, light grey = absent
        cmap = mcolors.ListedColormap(["#f0f0f0", "#2ecc71"])

        sns.heatmap(
            df_top,
            ax=ax,
            cmap=cmap,
            vmin=0, vmax=1,
            linewidths=0.3,
            linecolor="#cccccc",
            cbar=False,
            yticklabels=row_labels,
            xticklabels=True,
        )

        # Draw threshold line
        n_above = sum(1 for r in canonical if r["ticker_pct"] >= threshold)
        n_above = min(n_above, top_n)
        ax.axhline(y=n_above, color="red", linewidth=2, linestyle="--",
                   label=f"≥{threshold:.0%} threshold ({n_above} metrics)")

        ax.set_title(
            f"Financial Metric Frequency Across {self.n_tickers} Tickers\n"
            f"(top {top_n} metrics, green = present in filing)",
            fontsize=14, pad=12,
        )
        ax.set_xlabel("Ticker", fontsize=11)
        ax.set_ylabel("Metric  (% of tickers)", fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Heatmap saved → {output_path}")

    def save_canonical(
        self,
        output_path: str | Path,
        threshold: float = 0.30,
    ) -> list[dict]:
        """
        Save canonical metric list to JSON.
        This file is read by run_ingest.py to filter SQL writes,
        and by finetune/data_prep/schema_context.py to build NL2SQL prompts.
        """
        canonical = self.compute_canonical(threshold=threshold)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(canonical, f, indent=2)
        logger.info(f"Canonical metrics saved → {output_path} ({len(canonical)} entries)")
        return canonical

    def summary_table(self, threshold: float = 0.30) -> pd.DataFrame:
        """Return canonical metrics as a DataFrame (for display/analysis)."""
        canonical = self.compute_canonical(threshold=threshold)
        if not canonical:
            return pd.DataFrame()
        df = pd.DataFrame(canonical)
        df["ticker_pct_str"] = df["ticker_pct"].map(lambda x: f"{x:.1%}")
        return df[["canonical_name", "statement", "ticker_count", "ticker_pct_str", "metric"]]


# ── Metric normalisation helpers ───────────────────────────────────────────────

# Words to strip from metric labels before using as dict keys
_STRIP_WORDS = re.compile(
    r"\b(net|total|less|add|plus|gross|consolidated|and|of|the|in|for|from|"
    r"attributable|continuing|operations|per|share|diluted|basic|weighted|average)\b",
    re.IGNORECASE,
)


def _normalise_metric_key(metric: str) -> str:
    """
    Create a normalised key for grouping near-duplicate metric labels.
    "Total net sales" and "Net sales, total" should map to the same key.

    Strategy:
    1. Lowercase
    2. Strip punctuation
    3. Remove high-frequency filler words
    4. Sort remaining words alphabetically (bag-of-words)
    5. Rejoin with spaces
    """
    import re
    m = metric.lower()
    m = re.sub(r"[^a-z\s]", " ", m)
    m = _STRIP_WORDS.sub(" ", m)
    words = sorted(w for w in m.split() if w and len(w) > 1)
    return " ".join(words)


def _to_canonical_name(normalised_key: str) -> str:
    """
    Convert normalised key back to a display-friendly canonical name.
    Title-cases and removes duplicate words.
    """
    words = normalised_key.split()
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w.capitalize())
    return " ".join(unique)


def load_canonical_metrics(path: str | Path) -> list[dict]:
    """Load canonical_metrics.json saved by MetricAnalyzer.save_canonical()."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


