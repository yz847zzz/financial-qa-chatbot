"""
Quick end-to-end smoke test for the data pipeline.
Run: python data_pipeline/scripts/test_pipeline.py
"""
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.processing.extractor import make_extractor, HtmlExtractor
from data_pipeline.processing.table_parser import extract_all_rows_from_filing
from data_pipeline.processing.chunker import chunk_by_section
from data_pipeline.processing.metric_heatmap import MetricAnalyzer, _normalise_metric_key
from data_pipeline.metadata.models import FilingMetadata

FILINGS_DIR = Path("E:/emo/workspace/pintrade/data/filings")
TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


def test_extractor():
    print("\n── Extractor ──────────────────────────────────────────────────")
    path = FILINGS_DIR / "AAPL/10-K/2023-11-03_000032019323000106/aapl-20230930.htm"
    ext = make_extractor(path)
    assert isinstance(ext, HtmlExtractor)

    sections = ext.extract_sections()
    print(f"  Sections ({len(sections)}): {list(sections.keys())[:8]}")

    preamble = sections.get("preamble", "")
    safe = preamble[:120].encode("ascii", errors="replace").decode()
    print(f"  Preamble first 120 chars: {safe!r}")
    assert "UNITED STATES" in preamble or "Apple" in preamble, \
        f"Preamble looks wrong: {safe!r}"

    # Body headings include the full title after the item number, so pick the
    # longest section whose key starts with "Item 7" (but not "Item 7A").
    item7_candidates = {
        k: v for k, v in sections.items()
        if k.upper().startswith("ITEM 7") and "7A" not in k.upper()
    }
    item7 = max(item7_candidates.values(), key=len) if item7_candidates else ""
    item7_key = max(item7_candidates, key=lambda k: len(item7_candidates[k])) if item7_candidates else "?"
    safe_k = item7_key.encode("ascii", errors="replace").decode()
    print(f"  Item 7 key: {safe_k!r}")
    print(f"  Item 7 length: {len(item7)} chars")
    # MD&A is one of the longest sections — should be well over 5 000 chars
    assert len(item7) > 5_000, f"Item 7 too short ({len(item7)} chars)"

    tables = ext.extract_tables_as_rows()
    print(f"  Tables found: {len(tables)}")
    assert len(tables) > 10, "Expected more than 10 tables in a 10-K"

    print("  [PASS] extractor")


def test_table_parser():
    print("\n── Table Parser ───────────────────────────────────────────────")
    path = FILINGS_DIR / "AAPL/10-K/2023-11-03_000032019323000106/aapl-20230930.htm"
    ext = make_extractor(path)
    rows = extract_all_rows_from_filing("AAPL", "FY2023", ext)
    numeric_rows = [r for r in rows if r.value is not None]
    print(f"  Total rows: {len(rows)}, with numeric values: {len(numeric_rows)}")
    assert len(numeric_rows) > 50, "Expected more than 50 parseable financial rows"

    # Check revenue is found
    revenue_rows = [r for r in numeric_rows if "revenue" in r.metric.lower() or "sales" in r.metric.lower()]
    print(f"  Revenue/sales rows: {len(revenue_rows)}")
    for r in revenue_rows[:3]:
        print(f"    {r.statement:<20} {r.metric:<40} {r.value:.0f}")
    assert len(revenue_rows) > 0, "No revenue rows found"

    print("  [PASS] table_parser")


def test_chunker():
    print("\n── Chunker ────────────────────────────────────────────────────")
    path = FILINGS_DIR / "AAPL/10-K/2023-11-03_000032019323000106/aapl-20230930.htm"
    ext = make_extractor(path)
    sections = ext.extract_sections()

    meta = FilingMetadata(
        ticker="AAPL",
        company="Apple Inc.",
        filing_type="10-K",
        date="2023-11-03",
        accession_number="000032019323000106",
        source_path=str(path),
    )
    chunks = chunk_by_section(
        sections=sections,
        ticker="AAPL",
        doc_type="10-K",
        filing_date="2023-11-03",
        source_path=str(path),
        filing_meta=meta,
    )
    print(f"  Total chunks: {len(chunks)}")
    assert len(chunks) > 100, "Expected more than 100 chunks from a full 10-K"

    # Verify chunk_id uniqueness
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found!"
    print(f"  chunk_ids are unique: {len(ids)}")

    # Check a section chunk
    item7_chunks = [c for c in chunks if "7" in c.section and "A" not in c.section]
    print(f"  Item 7 chunks: {len(item7_chunks)}")
    if item7_chunks:
        print(f"  Sample: {item7_chunks[0].text[:100]!r}")

    print("  [PASS] chunker")


def test_metric_heatmap():
    print("\n── Metric Heatmap ─────────────────────────────────────────────")
    analyzer = MetricAnalyzer()

    for ticker in TEST_TICKERS:
        ticker_dir = FILINGS_DIR / ticker / "10-K"
        if not ticker_dir.exists():
            print(f"  Skipping {ticker} — not found")
            continue
        latest = sorted(ticker_dir.iterdir())[-1]
        htm_files = list(latest.glob("*.htm"))
        if not htm_files:
            continue
        ext = make_extractor(htm_files[0])
        rows = extract_all_rows_from_filing(ticker, "FY2023", ext)
        analyzer.add_rows(rows, ticker=ticker)
        numeric = sum(1 for r in rows if r.value is not None)
        print(f"  {ticker}: {numeric} numeric rows")

    print(f"  Tickers in analyzer: {analyzer.n_tickers}")
    assert analyzer.n_tickers >= 3

    # Frequency matrix
    df = analyzer.build_frequency_matrix()
    print(f"  Frequency matrix: {df.shape} (metrics × tickers)")
    assert df.shape[0] > 20 and df.shape[1] >= 3

    # Canonical at 50% threshold
    canon_50 = analyzer.compute_canonical(threshold=0.50)
    print(f"  Canonical metrics at >=50%: {len(canon_50)}")
    assert len(canon_50) > 10, "Expected at least 10 canonical metrics"

    # Top metrics
    print(f"  Top 10 canonical metrics:")
    for r in canon_50[:10]:
        print(f"    {r['ticker_pct']:.0%}  {r['statement']:<18}  {r['canonical_name']}")

    # Normalisation smoke-check
    key1 = _normalise_metric_key("Total Revenue")
    key2 = _normalise_metric_key("Net Revenue")
    key3 = _normalise_metric_key("Total net sales")
    print(f"  'Total Revenue'   → {key1!r}")
    print(f"  'Net Revenue'     → {key2!r}")
    print(f"  'Total net sales' → {key3!r}")

    # Save heatmap
    out = Path("data/analysis")
    out.mkdir(parents=True, exist_ok=True)
    analyzer.plot_heatmap(out / "test_heatmap.png", top_n=50, threshold=0.50)
    analyzer.save_canonical(out / "test_canonical.json", threshold=0.50)
    assert (out / "test_heatmap.png").exists()
    assert (out / "test_canonical.json").exists()
    print(f"  Heatmap saved → {out / 'test_heatmap.png'}")

    print("  [PASS] metric_heatmap")


def test_sec_loader():
    print("\n── SEC Loader ─────────────────────────────────────────────────")
    from data_pipeline.ingestion.sec_loader import walk_filings, count_filings

    counts = count_filings(FILINGS_DIR, form_types=["10-K"])
    print(f"  10-K filings found: {counts.get('10-K', 0)} across all tickers")
    assert counts.get("10-K", 0) > 50, "Expected 50+ 10-K filings"

    # Walk first 3 filings
    n = 0
    for meta, file_path in walk_filings(FILINGS_DIR, form_types=["10-K"]):
        print(f"  {meta.ticker} {meta.filing_type} {meta.date}  →  {file_path.name}")
        assert file_path.exists()
        n += 1
        if n >= 3:
            break
    print("  [PASS] sec_loader")


if __name__ == "__main__":
    print("=" * 60)
    print("Financial QA — Data Pipeline Smoke Test")
    print("=" * 60)

    failures = []
    for fn in [test_sec_loader, test_extractor, test_table_parser,
               test_chunker, test_metric_heatmap]:
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print("=" * 60)
