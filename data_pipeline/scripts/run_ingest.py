"""
Pass 2 — Full Ingest: SEC filings → VectorDB + SQL.

Run AFTER run_metric_analysis.py (which produces canonical_metrics.json).

What it does:
  For each filing in filings_dir:
    A. TEXT PATH → VectorDB
       1. Extract text from .htm or .pdf
       2. Split into ITEM sections (10-K/10-Q) or flat (8-K)
       3. Chunk with overlap
       4. Embed with all-MiniLM-L6-v2 → ChromaDB (deduped by chunk_id)

    B. TABLE PATH → SQL (filtered by canonical metric list)
       1. Extract all HTML <table> elements / PDF tables near FS keywords
       2. Parse to FinancialRow objects
       3. Normalise metric keys
       4. Keep only rows whose metric is in canonical_metrics.json
       5. Upsert into SQLite financials table

    C. PROVENANCE
       6. Upsert into filing_metadata table

Usage:
    cd financial-qa-chatbot

    # Quick test — one ticker
    python -m data_pipeline.scripts.run_ingest \
        --filings-dir data/filings \
        --db-path     data/financials.db \
        --vectordb-dir data/vectordb \
        --canonical   data/analysis/canonical_metrics.json \
        --ticker AAPL \
        --forms 10-K

    # Full SP500 run (96 tickers, all forms)
    python -m data_pipeline.scripts.run_ingest \
        --filings-dir data/filings \
        --db-path     data/financials.db \
        --vectordb-dir data/vectordb \
        --canonical   data/analysis/canonical_metrics.json \
        --forms 10-K 10-Q 8-K

    # Skip canonical filtering (embed everything, no SQL write filter)
    python -m data_pipeline.scripts.run_ingest ... --no-canonical-filter
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.ingestion.sec_loader import walk_filings
from data_pipeline.processing.extractor import make_extractor, HtmlExtractor, PdfExtractor
from data_pipeline.processing.table_parser import (
    extract_all_rows_from_filing,
    detect_scale,
)
from data_pipeline.processing.chunker import chunk_by_section, chunk_text
from data_pipeline.processing.metric_heatmap import (
    _normalise_metric_key,
    load_canonical_metrics,
)
from data_pipeline.metadata.models import FilingMetadata
from data_pipeline.storage.vector_store import VectorStore
from data_pipeline.storage.sql_store import SQLStore


def run_ingest(
    filings_dir: str | Path,
    db_path: str | Path,
    vectordb_dir: str | Path,
    canonical_json: str | Path | None = None,
    form_types: list[str] | None = None,
    tickers: list[str] | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
    use_canonical_filter: bool = True,
) -> dict:
    """
    Full ingest pipeline. Returns summary dict.
    """
    filings_dir = Path(filings_dir)
    form_types = form_types or ["10-K", "10-Q", "8-K"]

    # Load canonical metric filter
    canonical_keys: set[str] | None = None
    if use_canonical_filter and canonical_json:
        canonical_json = Path(canonical_json)
        if canonical_json.exists():
            canonical = load_canonical_metrics(canonical_json)
            canonical_keys = {_normalise_metric_key(r["metric"]) for r in canonical}
            logger.info(
                f"Loaded {len(canonical_keys)} canonical metric keys from {canonical_json}"
            )
        else:
            logger.warning(
                f"canonical_metrics.json not found at {canonical_json}. "
                "Run run_metric_analysis.py first, or use --no-canonical-filter."
            )

    sql_store = SQLStore(db_path)
    vector_store = VectorStore(vectordb_dir)

    total_sql = 0
    total_chunks = 0
    n_processed = 0
    n_skipped = 0
    skip_reasons: list[str] = []

    t0 = time.time()
    logger.info(
        f"Starting ingest | forms={form_types} | "
        f"canonical_filter={use_canonical_filter and canonical_keys is not None}"
    )

    for filing_meta, file_path in walk_filings(
        filings_dir, tickers=tickers, form_types=form_types
    ):
        ticker = filing_meta.ticker
        date = filing_meta.date
        form = filing_meta.filing_type
        period = _date_to_period(date, form)

        try:
            extractor = make_extractor(file_path)
        except Exception as e:
            reason = f"{ticker} {form} {date}: cannot open — {e}"
            logger.warning(reason)
            skip_reasons.append(reason)
            n_skipped += 1
            continue

        # ── A. Table extraction → SQL ─────────────────────────────────────────
        n_sql = 0
        try:
            rows = extract_all_rows_from_filing(ticker, period, extractor)
            if canonical_keys is not None:
                filtered_rows = [
                    r for r in rows
                    if _normalise_metric_key(r.metric) in canonical_keys
                ]
            else:
                filtered_rows = rows

            n_sql = sql_store.upsert(filtered_rows)
            logger.info(
                f"{ticker} {form} {date}: {n_sql}/{len(rows)} SQL rows "
                f"(canonical filter: {len(rows)-n_sql} dropped)"
            )
        except Exception as e:
            logger.warning(f"{ticker} {form} {date}: table extraction failed — {e}")

        # ── B. Text chunking → VectorDB ───────────────────────────────────────
        n_vec = 0
        try:
            if isinstance(extractor, HtmlExtractor):
                sections = extractor.extract_sections()
                chunks = chunk_by_section(
                    sections=sections,
                    ticker=ticker,
                    doc_type=form,
                    filing_date=date,
                    source_path=str(file_path),
                    filing_meta=filing_meta,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                section_count = len(sections)
            else:
                # PDF: no section splitting
                pages = extractor.extract_all_text()
                full_text = "\n\n".join(p.text for p in pages)
                chunks = chunk_text(
                    text=full_text,
                    ticker=ticker,
                    doc_type=form,
                    filing_date=date,
                    source_path=str(file_path),
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata=filing_meta.flat_dict(),
                )
                section_count = 0

            n_vec = vector_store.add(chunks)
            logger.info(f"{ticker} {form} {date}: {n_vec} new vector chunks")
        except Exception as e:
            logger.warning(f"{ticker} {form} {date}: chunking failed — {e}")
            section_count = 0

        # ── C. Filing provenance ──────────────────────────────────────────────
        sql_store.upsert_filing_metadata(
            filing=filing_meta,
            section_count=section_count,
            chunk_count=n_vec,
            sql_row_count=n_sql,
        )

        total_sql += n_sql
        total_chunks += n_vec
        n_processed += 1

    elapsed = time.time() - t0
    summary = {
        "filings_processed":    n_processed,
        "filings_skipped":      n_skipped,
        "sql_rows_written":     total_sql,
        "vector_chunks_added":  total_chunks,
        "vector_store_total":   vector_store.count,
        "elapsed_seconds":      round(elapsed, 1),
        **sql_store.stats(),
    }

    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    for k, v in summary.items():
        if k != "tickers":
            print(f"  {k:<26}: {v}")
    print(f"  {'tickers':<26}: {summary.get('tickers', [])}")
    print("=" * 60)

    if skip_reasons:
        print(f"\nSkipped ({len(skip_reasons)}):")
        for r in skip_reasons[:20]:
            print(f"  {r}")

    return summary


def _date_to_period(date: str, filing_type: str) -> str:
    year = date[:4]
    month = date[5:7]
    if filing_type == "10-K":
        return f"FY{year}"
    return f"{year}-{month}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Full SEC filing ingest (Pass 2)")
    parser.add_argument(
        "--filings-dir",
        default="data/filings",
    )
    parser.add_argument("--db-path",      default="data/financials.db")
    parser.add_argument("--vectordb-dir", default="data/vectordb")
    parser.add_argument(
        "--canonical",
        default="data/analysis/canonical_metrics.json",
        help="Path to canonical_metrics.json from run_metric_analysis.py",
    )
    parser.add_argument("--ticker",  nargs="*", default=None)
    parser.add_argument("--forms",   nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap",    type=int, default=64)
    parser.add_argument(
        "--no-canonical-filter", action="store_true",
        help="Write ALL parsed metrics to SQL (no frequency filtering)",
    )
    args = parser.parse_args()

    run_ingest(
        filings_dir=args.filings_dir,
        db_path=args.db_path,
        vectordb_dir=args.vectordb_dir,
        canonical_json=args.canonical,
        form_types=args.forms,
        tickers=args.ticker,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_canonical_filter=not args.no_canonical_filter,
    )


if __name__ == "__main__":
    main()
