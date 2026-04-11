-- Financial QA Chatbot — SQLite Schema
-- This DDL is the canonical source of truth.
-- data_pipeline/storage/sql_store.py and finetune/data_prep/schema_context.py
-- must match this exactly.

-- ── Structured financial metrics ──────────────────────────────────────────────
-- One row per (ticker, period, statement, metric).
-- Populated only for "canonical" metrics that appear in ≥30% of tickers
-- (determined by run_metric_analysis.py heatmap pass).
--
-- period format:
--   "YYYY-MM"  for quarterly filings (e.g. "2023-09")
--   "FYYYY"    for annual filings   (e.g. "FY2023")
--
-- statement values: "income_statement" | "balance_sheet" | "cash_flow" | "unknown"
--
-- value is always in USD. detect_scale() in the extractor converts
-- "in millions" → multiply by 1_000_000 before storing.

CREATE TABLE IF NOT EXISTS financials (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    period      TEXT    NOT NULL,
    statement   TEXT    NOT NULL,
    metric      TEXT    NOT NULL,
    value       REAL,
    unit        TEXT    DEFAULT 'USD',
    raw_value   TEXT    DEFAULT '',
    UNIQUE (ticker, period, statement, metric) ON CONFLICT REPLACE
);

CREATE INDEX IF NOT EXISTS idx_fin_ticker          ON financials (ticker);
CREATE INDEX IF NOT EXISTS idx_fin_ticker_period   ON financials (ticker, period);
CREATE INDEX IF NOT EXISTS idx_fin_metric          ON financials (metric);
CREATE INDEX IF NOT EXISTS idx_fin_statement       ON financials (statement);

-- ── Filing provenance ──────────────────────────────────────────────────────────
-- One row per filing document (accession_number is unique).
-- Tracks how many sections and vector chunks were produced per filing.

CREATE TABLE IF NOT EXISTS filing_metadata (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT    NOT NULL,
    company          TEXT,
    filing_type      TEXT,           -- "10-K" | "10-Q" | "8-K"
    date             TEXT,           -- "YYYY-MM-DD"
    accession_number TEXT    UNIQUE,
    section_count    INTEGER DEFAULT 0,
    chunk_count      INTEGER DEFAULT 0,
    sql_row_count    INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fm_ticker      ON filing_metadata (ticker);
CREATE INDEX IF NOT EXISTS idx_fm_date        ON filing_metadata (date);
CREATE INDEX IF NOT EXISTS idx_fm_filing_type ON filing_metadata (filing_type);

-- ── Canonical metric registry ──────────────────────────────────────────────────
-- Populated by run_metric_analysis.py after the heatmap frequency pass.
-- Only metrics in this table are written to the financials table.
-- used by finetune/data_prep/schema_context.py to build NL2SQL prompts.

CREATE TABLE IF NOT EXISTS canonical_metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    metric         TEXT    NOT NULL UNIQUE,
    statement      TEXT    NOT NULL,   -- most-frequent statement classification
    ticker_count   INTEGER NOT NULL,   -- how many tickers had this metric
    ticker_pct     REAL    NOT NULL,   -- fraction of all tickers (0.0–1.0)
    canonical_name TEXT    NOT NULL    -- normalised display name for NL2SQL prompts
);

CREATE INDEX IF NOT EXISTS idx_cm_statement ON canonical_metrics (statement);
