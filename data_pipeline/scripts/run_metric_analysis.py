"""
Pass 1 — Metric Frequency Analysis (the FINDMIND heatmap step).

Run this BEFORE run_ingest.py.

What it does:
  1. Walk all local filings in filings_dir (10-K only by default — annual reports
     have the most complete financial statements)
  2. Extract every table from every filing and parse to FinancialRow objects
     (no SQL filtering yet — we want ALL raw metric names)
  3. Feed into MetricAnalyzer to count: for each metric, how many tickers have it?
  4. Plot a heatmap: rows = top-N metrics, columns = tickers, colour = present/absent
  5. Apply threshold (default 30%): metrics in ≥30% of tickers → canonical list
  6. Save canonical_metrics.json  (read by run_ingest.py)
  7. Save canonical metrics to SQLite canonical_metrics table
  8. Print a summary table to stdout

After this script runs you can inspect metric_heatmap.png to tune the threshold
before running the heavier full ingest.

Usage:
    cd financial-qa-chatbot
    python -m data_pipeline.scripts.run_metric_analysis \
        --filings-dir data/filings \
        --db-path     data/financials.db \
        --output-dir  data/analysis \
        --threshold   0.30 \
        --forms       10-K \
        --tickers     AAPL MSFT GOOGL       # optional: subset for quick test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# Make sure data_pipeline is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.ingestion.sec_loader import walk_filings
from data_pipeline.processing.extractor import make_extractor
from data_pipeline.processing.table_parser import extract_all_rows_from_filing
from data_pipeline.processing.metric_heatmap import MetricAnalyzer
from data_pipeline.storage.sql_store import SQLStore


def run_metric_analysis(
    filings_dir: str | Path,
    db_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.30,
    form_types: list[str] | None = None,
    tickers: list[str] | None = None,
    top_n_heatmap: int = 80,
) -> list[dict]:
    """
    Full metric frequency analysis pass.

    Returns the canonical metric list (also saved to JSON and SQLite).
    """
    filings_dir = Path(filings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    form_types = form_types or ["10-K"]   # annual reports → richest tables
    analyzer = MetricAnalyzer()

    logger.info(
        f"Starting metric analysis | forms={form_types} | threshold={threshold:.0%}"
    )
    t0 = time.time()

    n_processed = 0
    n_skipped = 0
    skip_reasons: list[str] = []

    for filing_meta, file_path in walk_filings(
        filings_dir, tickers=tickers, form_types=form_types
    ):
        ticker = filing_meta.ticker
        date = filing_meta.date
        period = _date_to_period(date, filing_meta.filing_type)

        try:
            extractor = make_extractor(file_path)
            rows = extract_all_rows_from_filing(ticker, period, extractor)
            analyzer.add_rows(rows, ticker=ticker)
            n_processed += 1
            logger.debug(
                f"{ticker} {filing_meta.filing_type} {date}: {len(rows)} raw rows"
            )
        except Exception as e:
            reason = f"{ticker} {filing_meta.filing_type} {date}: {e}"
            logger.warning(f"Skipping — {reason}")
            skip_reasons.append(reason)
            n_skipped += 1

    elapsed = time.time() - t0
    logger.info(
        f"Extraction complete: {n_processed} filings processed, "
        f"{n_skipped} skipped in {elapsed:.1f}s"
    )

    # ── Compute canonical metrics ─────────────────────────────────────────────
    canonical = analyzer.compute_canonical(threshold=threshold)

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    heatmap_path = output_dir / "metric_heatmap.png"
    analyzer.plot_heatmap(heatmap_path, top_n=top_n_heatmap, threshold=threshold)

    # ── Save canonical JSON ───────────────────────────────────────────────────
    json_path = output_dir / "canonical_metrics.json"
    analyzer.save_canonical(json_path, threshold=threshold)

    # ── Save to SQLite ────────────────────────────────────────────────────────
    sql_store = SQLStore(db_path)
    sql_store.upsert_canonical_metrics(canonical)

    # ── Print summary ─────────────────────────────────────────────────────────
    df = analyzer.summary_table(threshold=threshold)
    print("\n" + "=" * 70)
    print(f"METRIC FREQUENCY ANALYSIS — threshold={threshold:.0%}")
    print(f"  Tickers analysed : {analyzer.n_tickers}")
    print(f"  Filings processed: {n_processed}")
    print(f"  Filings skipped  : {n_skipped}")
    print(f"  Canonical metrics: {len(canonical)}")
    print(f"  Heatmap          : {heatmap_path}")
    print(f"  Canonical JSON   : {json_path}")
    print("=" * 70)
    if not df.empty:
        print(df.to_string(index=False))

    if skip_reasons:
        print(f"\nSkipped ({len(skip_reasons)}):")
        for r in skip_reasons[:20]:
            print(f"  {r}")

    return canonical


def _date_to_period(date: str, filing_type: str) -> str:
    """
    Convert filing date string to period key for SQLite.
    10-K annual: "FY{YYYY}" using the fiscal year end
    10-Q quarterly: "YYYY-MM"
    """
    year = date[:4]
    month = date[5:7]
    if filing_type == "10-K":
        return f"FY{year}"
    return f"{year}-{month}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric frequency heatmap analysis (Pass 1)"
    )
    parser.add_argument(
        "--filings-dir",
        default="data/filings",
        help="Root directory of downloaded SEC filings",
    )
    parser.add_argument(
        "--db-path", default="data/financials.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--output-dir", default="data/analysis",
        help="Directory for heatmap PNG and canonical_metrics.json",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Fraction of tickers a metric must appear in to be canonical (default: 0.30)",
    )
    parser.add_argument(
        "--forms", nargs="+", default=["10-K"],
        help="Filing form types to analyse (default: 10-K)",
    )
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Subset of tickers (default: all in filings-dir)",
    )
    parser.add_argument(
        "--top-n", type=int, default=80,
        help="Number of metrics to show in heatmap (default: 80)",
    )
    args = parser.parse_args()

    run_metric_analysis(
        filings_dir=args.filings_dir,
        db_path=args.db_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        form_types=args.forms,
        tickers=args.tickers,
        top_n_heatmap=args.top_n,
    )


if __name__ == "__main__":
    main()
