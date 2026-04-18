"""
SQLite store for structured financial data.

Additions over the base design:
  - filing_metadata table
  - canonical_metrics table (populated by run_metric_analysis.py)
  - upsert_filing_metadata()
  - upsert_canonical_metrics()
  - filter_by_canonical() for the ingest pass
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from ..metadata.models import FilingMetadata, FinancialRow


class SQLStore:
    """
    Persistent SQLite store.

    Tables:
      financials        — one row per (ticker, period, statement, metric)
      filing_metadata   — one row per filing document
      canonical_metrics — filtered metric list from heatmap analysis
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._apply_schema()
        logger.info(f"SQLStore ready: {self.db_path}")

    def _apply_schema(self) -> None:
        import re as _re
        schema_path = Path(__file__).parent / "schema.sql"
        ddl = schema_path.read_text(encoding="utf-8")
        # Strip -- comments before splitting so DDL blocks that begin with
        # a comment line are not silently skipped.
        ddl_no_comments = _re.sub(r"--[^\n]*", "", ddl)
        for stmt in ddl_no_comments.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self._conn.execute(stmt)
                except Exception:
                    pass  # table/index already exists
        self._conn.commit()

    # ── financials ─────────────────────────────────────────────────────────────

    def upsert(self, rows: Sequence[FinancialRow]) -> int:
        """Upsert FinancialRow objects. Returns rows written."""
        if not rows:
            return 0
        data = [
            (r.ticker, r.period, r.statement, r.metric, r.value, r.unit, r.raw_value)
            for r in rows
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO financials
               (ticker, period, statement, metric, value, unit, raw_value)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        self._conn.commit()
        return len(data)

    def upsert_filtered(
        self,
        rows: Sequence[FinancialRow],
        canonical_keys: set[str],
    ) -> int:
        """
        Upsert only rows whose metric key is in canonical_keys.
        canonical_keys comes from MetricAnalyzer.canonical_metric_set().
        """
        from ..processing.metric_heatmap import _normalise_metric_key
        filtered = [r for r in rows if _normalise_metric_key(r.metric) in canonical_keys]
        return self.upsert(filtered)

    def query(
        self,
        ticker: str | None = None,
        period_like: str | None = None,
        statement: str | None = None,
        metric_like: str | None = None,
    ) -> pd.DataFrame:
        conds: list[str] = []
        params: list = []
        if ticker:
            conds.append("ticker = ?"); params.append(ticker)
        if period_like:
            conds.append("period LIKE ?"); params.append(period_like)
        if statement:
            conds.append("statement = ?"); params.append(statement)
        if metric_like:
            conds.append("metric LIKE ?"); params.append(metric_like)
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        sql = (
            f"SELECT ticker, period, statement, metric, value, unit, raw_value "
            f"FROM financials {where} ORDER BY ticker, period, statement, metric"
        )
        return pd.read_sql_query(sql, self._conn, params=params)

    def get_pivot(self, ticker: str, period_like: str = "%") -> pd.DataFrame:
        """Wide-format table: rows=(statement,metric), columns=periods."""
        df = self.query(ticker=ticker, period_like=period_like)
        if df.empty:
            return df
        return df.pivot_table(
            index=["statement", "metric"],
            columns="period",
            values="value",
            aggfunc="first",
        )

    def execute_raw(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute arbitrary SELECT. Hook for NL2SQL in deployment."""
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT statements allowed")
        return pd.read_sql_query(sql, self._conn, params=list(params))

    # ── filing_metadata ────────────────────────────────────────────────────────

    def upsert_filing_metadata(
        self,
        filing: FilingMetadata,
        section_count: int = 0,
        chunk_count: int = 0,
        sql_row_count: int = 0,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO filing_metadata
               (ticker, company, filing_type, date, accession_number,
                section_count, chunk_count, sql_row_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (filing.ticker, filing.company, filing.filing_type, filing.date,
             filing.accession_number, section_count, chunk_count, sql_row_count),
        )
        self._conn.commit()

    def list_tickers(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT ticker FROM financials ORDER BY ticker"
        )
        return [r[0] for r in cur.fetchall()]

    def list_periods(self, ticker: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT period FROM financials WHERE ticker=? ORDER BY period",
            (ticker,),
        )
        return [r[0] for r in cur.fetchall()]

    # ── canonical_metrics ─────────────────────────────────────────────────────

    def upsert_canonical_metrics(self, canonical: list[dict]) -> int:
        """
        Populate canonical_metrics table from MetricAnalyzer.compute_canonical() output.
        Called by run_metric_analysis.py after the heatmap pass.
        """
        data = [
            (
                r["metric"],
                r["statement"],
                r["ticker_count"],
                r["ticker_pct"],
                r["canonical_name"],
            )
            for r in canonical
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO canonical_metrics
               (metric, statement, ticker_count, ticker_pct, canonical_name)
               VALUES (?, ?, ?, ?, ?)""",
            data,
        )
        self._conn.commit()
        logger.info(f"Upserted {len(data)} canonical metrics into SQLite")
        return len(data)

    def get_canonical_metrics(self) -> list[dict]:
        """Return canonical metrics from DB (for schema_context.py in finetune)."""
        cur = self._conn.execute(
            "SELECT metric, statement, ticker_count, ticker_pct, canonical_name "
            "FROM canonical_metrics ORDER BY ticker_pct DESC"
        )
        cols = ["metric", "statement", "ticker_count", "ticker_pct", "canonical_name"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        cur = self._conn.execute("SELECT COUNT(*) FROM financials")
        n_fin = cur.fetchone()[0]
        cur = self._conn.execute("SELECT COUNT(*) FROM filing_metadata")
        n_filings = cur.fetchone()[0]
        cur = self._conn.execute("SELECT COUNT(*) FROM canonical_metrics")
        n_canonical = cur.fetchone()[0]
        return {
            "financials_rows": n_fin,
            "filing_count":    n_filings,
            "canonical_metrics": n_canonical,
            "tickers":         self.list_tickers(),
        }

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SQLStore({self.db_path}, "
            f"rows={s['financials_rows']}, "
            f"tickers={len(s['tickers'])})"
        )


if __name__ == "__main__":
    store = SQLStore("data/test_financials.db")
    test_rows = [
        FinancialRow("AAPL", "FY2023", "income_statement", "Total Revenue", 383_285_000_000.0),
        FinancialRow("AAPL", "FY2023", "income_statement", "Net Income",     96_995_000_000.0),
        FinancialRow("AAPL", "FY2023", "balance_sheet",    "Total Assets",  352_755_000_000.0),
    ]
    store.upsert(test_rows)
    print(store.query(ticker="AAPL"))
    print("\nPivot:")
    print(store.get_pivot("AAPL"))
    print(store.stats())
