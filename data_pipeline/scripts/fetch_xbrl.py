"""
fetch_xbrl.py — Replace HTML-parsed financials with authoritative SEC XBRL data.

Replaces the broken table_parser.py extraction for the panel table.
Text chunks in ChromaDB are untouched — only the SQLite panel is rebuilt.

Architecture
────────────
1. Load feature_mapping.json  →  concept fallback lists per column
2. Fetch CIK for each ticker  →  SEC company_tickers.json (one call, cached)
3. Per ticker: GET companyfacts JSON from SEC XBRL API
4. Per column: try xbrl_concepts in order, take first with FY annual data
5. Compute derived ratios (current_ratio, net_margin, roa, debt_to_assets)
6. DROP + recreate panel table with new schema, bulk-insert all rows

Rate limiting
─────────────
SEC EDGAR limits to 10 req/s. We stay at ~8 req/s (0.12s sleep between tickers).

Usage
─────
  python data_pipeline/scripts/fetch_xbrl.py
  python data_pipeline/scripts/fetch_xbrl.py --tickers AAPL MSFT ADBE
  python data_pipeline/scripts/fetch_xbrl.py --years 2021 2022 2023
  python data_pipeline/scripts/fetch_xbrl.py --db-path data/financials.db
  python data_pipeline/scripts/fetch_xbrl.py --dry-run   # print rows, no DB write
"""

import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
MAPPING_PATH = ROOT / "data_pipeline" / "feature_mapping.json"
DEFAULT_DB   = ROOT / "data" / "financials.db"

SEC_HEADERS  = {"User-Agent": "financial-qa-chatbot research@example.com"}
CIK_URL      = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL    = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Fiscal years to fetch
DEFAULT_YEARS = list(range(2019, 2025))   # FY2019 – FY2024


# ── Panel schema (new) ─────────────────────────────────────────────────────────

PANEL_DDL = """
CREATE TABLE IF NOT EXISTS panel (
    ticker              TEXT NOT NULL,
    year                TEXT NOT NULL,
    -- Income statement
    total_revenue       REAL,
    gross_profit        REAL,
    operating_income    REAL,
    net_income          REAL,
    r_and_d             REAL,
    interest_expense    REAL,
    interest_income     REAL,
    da                  REAL,
    -- Cash flow
    cfo                 REAL,
    capex               REAL,
    buybacks            REAL,
    dividends_paid      REAL,
    -- Balance sheet
    cash                REAL,
    total_assets        REAL,
    current_assets      REAL,
    current_liabilities REAL,
    total_liabilities   REAL,
    long_term_debt      REAL,
    goodwill            REAL,
    retained_earnings   REAL,
    inventories         REAL,
    accounts_payable    REAL,
    -- Per share
    eps_diluted         REAL,
    -- Computed ratios
    current_ratio       REAL,
    net_margin          REAL,
    roa                 REAL,
    debt_to_assets      REAL,
    PRIMARY KEY (ticker, year)
);
"""

PANEL_COLUMNS = [
    "total_revenue", "gross_profit", "operating_income", "net_income",
    "r_and_d", "interest_expense", "interest_income", "da",
    "cfo", "capex", "buybacks", "dividends_paid",
    "cash", "total_assets", "current_assets", "current_liabilities",
    "total_liabilities", "long_term_debt", "goodwill", "retained_earnings",
    "inventories", "accounts_payable", "eps_diluted",
    "current_ratio", "net_margin", "roa", "debt_to_assets",
]


# ── SEC API helpers ────────────────────────────────────────────────────────────

def _fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=SEC_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"Rate limited, sleeping 5s (attempt {attempt+1})")
                time.sleep(5)
            elif e.code == 404:
                raise FileNotFoundError(f"Not found: {url}")
            else:
                raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"Retry {attempt+1}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def fetch_cik_map() -> dict[str, str]:
    """
    Returns {ticker: cik_padded_10} e.g. {"AAPL": "0000320193"}.
    One API call, cached in memory for the process lifetime.
    """
    logger.info("Fetching CIK map from SEC...")
    data = _fetch_json(CIK_URL)
    result = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik    = str(entry["cik_str"]).zfill(10)
        result[ticker] = cik
    logger.info(f"CIK map loaded: {len(result):,} companies")
    return result


def fetch_company_facts(cik: str) -> dict:
    """Fetch full XBRL company facts for one CIK. Returns us-gaap dict."""
    url  = FACTS_URL.format(cik=cik)
    data = _fetch_json(url)
    return data.get("facts", {}).get("us-gaap", {})


# ── Value extraction ───────────────────────────────────────────────────────────

def _get_annual_value(
    gaap: dict,
    concepts: list[str],
    fy: int,
    unit: str = "USD",
) -> float | None:
    """
    Try each concept in order; return the first FY-annual value found for the
    given fiscal year. Returns None if no concept has data.

    Filters: fp == 'FY', form in ('10-K', '10-K/A'), fy == requested year.
    Deduplicates by (end date, val) in case the same period appears in
    multiple filings (amended 10-K/A takes precedence — listed last chronologically).
    """
    for concept in concepts:
        if concept not in gaap:
            continue
        units_data = gaap[concept].get("units", {}).get(unit, [])
        if not units_data:
            continue

        # Filter to annual FY filings for the requested year
        hits = [
            x for x in units_data
            if x.get("fp") == "FY"
            and x.get("form") in ("10-K", "10-K/A")
            and x.get("fy") == fy
        ]
        if not hits:
            continue

        # Among hits for this FY, prefer 10-K/A (amended) over 10-K,
        # then take the one with the latest filed date
        hits.sort(key=lambda x: (x.get("form") == "10-K/A", x.get("filed", "")))
        return float(hits[-1]["val"])

    return None


def extract_ticker_data(
    gaap: dict,
    mapping: dict,
    years: list[int],
) -> list[dict]:
    """
    Extract all panel columns for all requested fiscal years from one company's
    XBRL fact dict. Returns list of row dicts, one per year that has at least
    one non-None value.
    """
    rows = []
    for fy in years:
        row: dict[str, float | None] = {}

        for col, spec in mapping["panel_columns"].items():
            if spec.get("computed"):
                continue   # computed after raw fetch

            concepts  = spec["xbrl_concepts"]
            unit      = spec.get("xbrl_unit", "USD")
            row[col]  = _get_annual_value(gaap, concepts, fy, unit=unit)

        # Compute ratios
        ca  = row.get("current_assets")
        cl  = row.get("current_liabilities")
        ni  = row.get("net_income")
        rev = row.get("total_revenue")
        ta  = row.get("total_assets")
        tl  = row.get("total_liabilities")

        row["current_ratio"]  = ca / cl if (ca and cl and cl != 0) else None
        row["net_margin"]     = ni / rev if (ni is not None and rev and rev != 0) else None
        row["roa"]            = ni / ta  if (ni is not None and ta and ta != 0) else None
        row["debt_to_assets"] = tl / ta  if (tl is not None and ta and ta != 0) else None

        # Skip years with no data at all
        has_data = any(v is not None for v in row.values())
        if has_data:
            row["year"] = f"FY{fy}"
            rows.append(row)

    return rows


# ── Database ───────────────────────────────────────────────────────────────────

def rebuild_panel(db_path: Path, all_rows: list[dict]) -> None:
    """Drop and recreate panel table, then bulk-insert all rows."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    logger.info("Dropping old panel table...")
    cur.execute("DROP TABLE IF EXISTS panel")
    cur.execute(PANEL_DDL)

    insert_cols = ["ticker", "year"] + PANEL_COLUMNS
    placeholders = ", ".join("?" * len(insert_cols))
    sql = f"INSERT OR REPLACE INTO panel ({', '.join(insert_cols)}) VALUES ({placeholders})"

    rows_tuples = []
    for row in all_rows:
        rows_tuples.append(tuple(row.get(c) for c in insert_cols))

    cur.executemany(sql, rows_tuples)
    con.commit()
    con.close()
    logger.success(f"Panel rebuilt: {len(rows_tuples):,} rows inserted into {db_path}")


# ── Summary helpers ────────────────────────────────────────────────────────────

def _null_rate(rows: list[dict], col: str) -> float:
    vals = [r.get(col) for r in rows]
    nulls = sum(1 for v in vals if v is None)
    return nulls / len(vals) if vals else 1.0


def print_quality_report(all_rows: list[dict]) -> None:
    """Print null-rate per column after fetch."""
    total = len(all_rows)
    logger.info(f"\nData quality report ({total} rows):")
    for col in PANEL_COLUMNS:
        nr = _null_rate(all_rows, col)
        bar = "█" * int((1 - nr) * 20)
        logger.info(f"  {col:<22} {bar:<20} {100*(1-nr):.0f}% filled")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild panel table from SEC XBRL API")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Subset of tickers (default: all in current panel table)")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS,
                        help=f"Fiscal years to fetch (default: {DEFAULT_YEARS})")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print extracted rows, do not write to DB")
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    args = parser.parse_args()

    # ── Load feature mapping ───────────────────────────────────────────────────
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    logger.info(f"Feature mapping loaded: {len(mapping['panel_columns'])} columns")

    # ── Determine ticker list ─────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        # Read from existing panel table
        con = sqlite3.connect(str(args.db_path))
        tickers = [r[0] for r in con.execute(
            "SELECT DISTINCT ticker FROM panel ORDER BY ticker"
        ).fetchall()]
        con.close()
        if not tickers:
            # Fallback to filing_metadata
            con = sqlite3.connect(str(args.db_path))
            tickers = [r[0] for r in con.execute(
                "SELECT DISTINCT ticker FROM filing_metadata ORDER BY ticker"
            ).fetchall()]
            con.close()
    logger.info(f"Tickers to fetch: {len(tickers)} — {tickers[:10]}...")

    # ── Fetch CIK map ─────────────────────────────────────────────────────────
    cik_map = fetch_cik_map()
    time.sleep(0.12)

    # ── Fetch & extract ───────────────────────────────────────────────────────
    all_rows: list[dict] = []
    skipped: list[str]   = []
    null_revenue: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        cik = cik_map.get(ticker)
        if not cik:
            logger.warning(f"[{i}/{len(tickers)}] {ticker}: no CIK found — skipping")
            skipped.append(ticker)
            continue

        try:
            logger.info(f"[{i}/{len(tickers)}] {ticker} (CIK {cik})...")
            gaap = fetch_company_facts(cik)
            rows = extract_ticker_data(gaap, mapping, args.years)

            for row in rows:
                row["ticker"] = ticker

            all_rows.extend(rows)

            # Quick sanity: check FY2023 revenue
            fy23 = next((r for r in rows if r["year"] == "FY2023"), None)
            if fy23:
                rev = fy23.get("total_revenue")
                ni  = fy23.get("net_income")
                logger.info(f"  FY2023 → revenue={rev/1e9:.2f}B  net_income={ni/1e9:.2f}B"
                            if rev and ni else f"  FY2023 → revenue={rev}  net_income={ni}")
                if rev is None:
                    null_revenue.append(ticker)
            else:
                logger.warning(f"  No FY2023 data found")

        except FileNotFoundError:
            logger.warning(f"[{i}/{len(tickers)}] {ticker}: not found in XBRL API — skipping")
            skipped.append(ticker)
        except Exception as e:
            logger.error(f"[{i}/{len(tickers)}] {ticker}: {e}")
            skipped.append(ticker)

        time.sleep(0.12)   # stay under 10 req/s SEC rate limit

    # ── Output ────────────────────────────────────────────────────────────────
    logger.info(f"\nFetch complete: {len(all_rows)} rows from {len(tickers)-len(skipped)} tickers")
    if skipped:
        logger.warning(f"Skipped ({len(skipped)}): {skipped}")
    if null_revenue:
        logger.warning(f"NULL revenue FY2023 ({len(null_revenue)}): {null_revenue}")

    print_quality_report(all_rows)

    if args.dry_run:
        logger.info("Dry run — printing first 5 rows:")
        for row in all_rows[:5]:
            logger.info(row)
        return

    rebuild_panel(args.db_path, all_rows)

    # ── Final verification ────────────────────────────────────────────────────
    con = sqlite3.connect(str(args.db_path))
    con.row_factory = sqlite3.Row
    total  = con.execute("SELECT COUNT(*) FROM panel").fetchone()[0]
    sample = con.execute(
        "SELECT ticker, year, total_revenue, net_income, net_margin "
        "FROM panel WHERE ticker='AAPL' AND year='FY2023'"
    ).fetchone()
    con.close()

    logger.success(f"Panel table: {total} rows total")
    if sample:
        logger.success(
            f"AAPL FY2023 → revenue={sample['total_revenue']/1e9:.3f}B  "
            f"net_income={sample['net_income']/1e9:.3f}B  "
            f"net_margin={sample['net_margin']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
