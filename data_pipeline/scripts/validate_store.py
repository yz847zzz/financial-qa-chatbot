"""
Smoke-test the data stores after ingest.

Run after run_ingest.py to verify:
  1. SQLite has rows, canonical metrics are populated
  2. VectorDB has chunks, a sample query returns sensible results
  3. Pivot table for AAPL works
  4. execute_raw() SELECT works, INSERT is rejected

Usage:
    python -m data_pipeline.scripts.validate_store \
        --db-path     data/financials.db \
        --vectordb-dir data/vectordb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.storage.sql_store import SQLStore
from data_pipeline.storage.vector_store import VectorStore


def validate(db_path: str, vectordb_dir: str) -> bool:
    all_pass = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")
        if not condition:
            all_pass = False

    # ── SQLite checks ─────────────────────────────────────────────────────────
    print("\n── SQLite ──────────────────────────────────────────────────────────")
    sql = SQLStore(db_path)
    stats = sql.stats()

    check("financials table has rows",
          stats["financials_rows"] > 0,
          f"{stats['financials_rows']} rows")

    check("filing_metadata table has rows",
          stats["filing_count"] > 0,
          f"{stats['filing_count']} filings")

    check("canonical_metrics table populated",
          stats["canonical_metrics"] > 0,
          f"{stats['canonical_metrics']} canonical metrics")

    check("multiple tickers stored",
          len(stats["tickers"]) > 1,
          f"tickers={stats['tickers'][:5]}...")

    # Pivot table for first available ticker
    if stats["tickers"]:
        first_ticker = stats["tickers"][0]
        pivot = sql.get_pivot(first_ticker)
        check(f"get_pivot({first_ticker!r}) returns non-empty DataFrame",
              not pivot.empty,
              f"{pivot.shape}")

    # execute_raw SELECT
    try:
        df = sql.execute_raw("SELECT ticker, COUNT(*) as n FROM financials GROUP BY ticker LIMIT 5")
        check("execute_raw SELECT works", len(df) > 0, f"{len(df)} rows")
    except Exception as e:
        check("execute_raw SELECT works", False, str(e))

    # execute_raw INSERT should be rejected
    try:
        sql.execute_raw("INSERT INTO financials VALUES (1,2,3,4,5,6,7,8)")
        check("execute_raw rejects INSERT", False, "should have raised ValueError")
    except ValueError:
        check("execute_raw rejects INSERT", True)

    # Sample revenue query
    try:
        df = sql.execute_raw(
            "SELECT ticker, period, value FROM financials "
            "WHERE metric LIKE '%Revenue%' ORDER BY ticker, period LIMIT 10"
        )
        check("Revenue metric query returns results",
              len(df) > 0, f"{len(df)} rows")
    except Exception as e:
        check("Revenue metric query returns results", False, str(e))

    # ── VectorDB checks ───────────────────────────────────────────────────────
    print("\n── VectorDB ────────────────────────────────────────────────────────")
    try:
        vs = VectorStore(vectordb_dir)
        check("VectorStore loads", True, f"{vs.count} chunks")
        check("VectorStore has chunks", vs.count > 0, f"count={vs.count}")

        # Semantic search
        if vs.count > 0:
            results = vs.search("revenue growth", n=5)
            check("Semantic search returns results",
                  len(results) > 0,
                  f"{len(results)} results, top dist={results[0]['distance']:.3f}")

            # Filtered search
            if stats["tickers"]:
                first_ticker = stats["tickers"][0]
                res_filtered = vs.search(
                    "management discussion and analysis",
                    ticker=first_ticker, n=3,
                )
                check(f"Filtered search ({first_ticker}) returns results",
                      len(res_filtered) > 0,
                      f"{len(res_filtered)} results")

            # multi-query search
            results_mq = vs.search_multi_query(
                ["Apple iPhone revenue", "Services segment growth"],
                n_per_query=3,
            )
            check("Multi-query search works",
                  len(results_mq) > 0, f"{len(results_mq)} deduped results")

    except Exception as e:
        check("VectorStore loads", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Result: {'ALL PASS ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
    print("=" * 55)
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data stores after ingest")
    parser.add_argument("--db-path",      default="data/financials.db")
    parser.add_argument("--vectordb-dir", default="data/vectordb")
    args = parser.parse_args()

    ok = validate(args.db_path, args.vectordb_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
