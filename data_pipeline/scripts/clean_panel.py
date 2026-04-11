"""
Rebuild the `panel` table from `financials` with proper statement-type filtering.

Fixes vs the original build_panel.py:
  1. Filter each metric by its correct statement type (balance_sheet, income_statement,
     cash_flow) — prevents cash-flow-delta values from polluting balance-sheet columns.
  2. Add `total_revenue` column (was missing; net_margin was wrong because of this).
  3. Fix net_margin = net_income / total_revenue  (was net_income / total_assets).
  4. Handle multiple candidate metric names per column (AAPL uses "Total net sales",
     MSFT uses "Total revenues", etc.) — take whichever has data.
  5. For balance sheet items: require value > 0 (eliminates cash-flow deltas that
     accidentally have the same metric name).

Run:
    python data_pipeline/scripts/clean_panel.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH  = Path(__file__).resolve().parents[2] / "data" / "financials.db"
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "panel.csv"

# ── Metric definitions: (column_name, statement, [candidate metric names], positive_only)
# positive_only=True for balance sheet stock items that must be > 0.
# positive_only=False for flow items that can legitimately be negative.
METRIC_DEFS = [
    # ── Balance sheet ──────────────────────────────────────────────────────────
    ("cash",               "balance_sheet",    ["Cash and cash equivalents", "Cash"],              True),
    ("total_assets",       "balance_sheet",    ["Total assets", "Assets"],                         True),
    ("current_assets",     "balance_sheet",    ["Total current assets", "Current assets"],         True),
    ("current_liabilities","balance_sheet",    ["Total current liabilities", "Current liabilities"],True),
    ("total_liabilities",  "balance_sheet",    ["Total liabilities"],                              True),
    ("goodwill",           "balance_sheet",    ["Goodwill"],                                       True),
    ("deferred_tax",       "balance_sheet",    ["Deferred income taxes", "Deferred tax assets"],   True),
    ("accounts_payable",   "balance_sheet",    ["Accounts payable"],                               True),
    ("retained_earnings",  "balance_sheet",    ["Retained earnings"],                              True),
    ("long_term_debt",     "balance_sheet",    ["Long-term debt", "Long term debt"],               True),
    ("inventories",        "balance_sheet",    ["Inventories", "Inventory"],                       True),
    ("aoci",               "balance_sheet",    ["Accumulated other comprehensive loss",
                                                "Accumulated other comprehensive income/(loss)",
                                                "Accumulated other comprehensive income"],         False),
    ("apic",               "balance_sheet",    ["Additional paid-in capital",
                                                "Additional paid in capital"],                     True),
    ("land",               "balance_sheet",    ["Land"],                                           True),
    ("other_assets",       "balance_sheet",    ["Other assets"],                                   True),
    ("other_current_assets","balance_sheet",   ["Other current assets"],                           True),
    ("other_liabilities",  "balance_sheet",    ["Other liabilities"],                              True),

    # ── Income statement ───────────────────────────────────────────────────────
    # Revenue: try many names; each company uses a different label
    ("total_revenue",      "income_statement", ["Total net sales", "Total revenues",
                                                "Net revenues", "Revenue", "Net sales",
                                                "Total revenue", "Revenues",
                                                "Net Revenue"],                                    True),
    ("net_income",         "income_statement", ["Net income", "Net earnings",
                                                "Net Income", "Net Earnings"],                     False),
    # Operating income: prefer "Total operating income" over segment-level "Operating income"
    ("operating_income",   "income_statement", ["Total operating income",
                                                "Total Operating Income",
                                                "Income from operations",
                                                "Operating income",
                                                "Operating Income"],                               False),
    ("interest_expense",   "income_statement", ["Interest expense", "Interest Expense"],           False),
    ("interest_income",    "income_statement", ["Interest income", "Interest Income"],             False),
    ("da",                 "income_statement", ["Depreciation and amortization",
                                                "Depreciation & amortization",
                                                "Depreciation, depletion and amortization"],       True),
    ("comprehensive_income","income_statement",["Comprehensive income",
                                                "Total comprehensive income"],                     False),

    # ── Cash flow ──────────────────────────────────────────────────────────────
    ("cfo",  "cash_flow", ["Net cash provided by operating activities",
                           "Net cash from operating activities",
                           "Cash provided by operating activities"],                               False),
    ("capex","cash_flow", ["Capital expenditures", "Purchases of property and equipment",
                           "Capital expenditure"],                                                 False),
    ("operating_leases","cash_flow", ["Operating leases", "Operating lease payments"],             False),
]


def build_panel(conn: sqlite3.Connection) -> pd.DataFrame:
    # Load the annual FY* rows from financials (no unknown rows needed)
    df = pd.read_sql(
        "SELECT ticker, period, statement, metric, value "
        "FROM financials "
        "WHERE period LIKE 'FY%' AND value IS NOT NULL AND statement != 'unknown'",
        conn,
    )
    df["period"] = df["period"].str.strip()

    tickers = sorted(df["ticker"].unique())
    years   = sorted(df["period"].unique())
    print(f"Loaded {len(df):,} rows  |  {len(tickers)} tickers  |  years: {years}")

    # Build one column at a time
    col_frames = []
    for col_name, stmt, candidates, positive_only in METRIC_DEFS:
        sub = df[(df["statement"] == stmt) & (df["metric"].isin(candidates))].copy()
        if positive_only:
            sub = sub[sub["value"] > 0]

        if sub.empty:
            print(f"  {col_name:<25}  NO DATA (stmt={stmt})")
            continue

        # If multiple candidates matched, rank by priority (order in list = 0 is highest)
        rank = {m: i for i, m in enumerate(candidates)}
        sub["_rank"] = sub["metric"].map(rank)

        # Deduplicate (ticker, period): prefer lowest-rank (= first candidate in list),
        # then largest value to prefer annual totals over quarterly partials.
        sub = (
            sub.sort_values(["_rank", "value"], ascending=[True, False])
               .drop_duplicates(subset=["ticker", "period"], keep="first")
        )

        col_frames.append(
            sub[["ticker", "period", "value"]].rename(columns={"value": col_name})
        )
        filled = len(sub)
        total  = len(tickers) * len(years)
        print(f"  {col_name:<25}  {filled:>4}/{total}  ({filled/total*100:.0f}% filled)")

    # Start with full ticker × year index so every combination has a row
    idx = pd.MultiIndex.from_product([tickers, years], names=["ticker", "period"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.rename(columns={"period": "year"})

    for frame in col_frames:
        frame = frame.rename(columns={"period": "year"})
        panel = panel.merge(frame, on=["ticker", "year"], how="left")

    panel = panel.sort_values(["ticker", "year"]).reset_index(drop=True)

    # ── Plausibility bounds — null out values that can't be real ──────────────
    # Any single public company exceeding these is an extraction error.
    BOUNDS = {
        # monetary (USD absolute)          (min,         max)
        "total_revenue":      (-1e11,   6e11),   # max ~$600B (Walmart)
        "net_income":         (-2e11,   2e11),   # max ~$100B; losses can be large
        "operating_income":   (-2e11,   2e11),
        "cfo":                (-5e10,   2e11),
        "capex":              (-2e11,   0),      # capex is a cash outflow (negative)
        "cash":               (0,       5e11),
        "total_assets":       (0,       4e12),   # max ~$3.4T (JPMorgan)
        "current_assets":     (0,       2e12),
        "current_liabilities":(0,       2e12),
        "total_liabilities":  (0,       4e12),
        "goodwill":           (0,       5e11),
        "long_term_debt":     (0,       1e12),
        "inventories":        (0,       5e11),
        "accounts_payable":   (0,       5e11),
        "retained_earnings":  (-1e12,   3e12),
        "interest_expense":   (-1e11,   1e11),
        "da":                 (0,       1e11),
    }
    n_nulled = 0
    for col, (lo, hi) in BOUNDS.items():
        if col not in panel.columns:
            continue
        bad = panel[col].notna() & ((panel[col] < lo) | (panel[col] > hi))
        n_bad = bad.sum()
        if n_bad:
            print(f"  [plausibility] nulled {n_bad:3d} bad rows in {col}  "
                  f"(range [{lo:.0e}, {hi:.0e}])")
            panel.loc[bad, col] = None
            n_nulled += n_bad
    print(f"  Total plausibility nulls: {n_nulled}")

    # ── Derived ratios (recomputed after plausibility filter) ─────────────────
    panel["current_ratio"]  = panel["current_assets"]  / panel["current_liabilities"]
    panel["debt_to_assets"] = panel["long_term_debt"]   / panel["total_assets"]
    panel["net_margin"]     = panel["net_income"]       / panel["total_revenue"]
    panel["roa"]            = panel["net_income"]       / panel["total_assets"]

    # Null out impossible ratios (signal that underlying inputs were bad)
    panel.loc[panel["current_ratio"]  > 50,  "current_ratio"]  = None
    panel.loc[panel["current_ratio"]  < 0,   "current_ratio"]  = None
    panel.loc[panel["roa"].abs()      > 1.0, "roa"]            = None
    panel.loc[panel["net_margin"].abs()> 1.5,"net_margin"]     = None
    panel.loc[panel["debt_to_assets"] > 5.0, "debt_to_assets"] = None
    panel.loc[panel["debt_to_assets"] < 0,   "debt_to_assets"] = None

    return panel


def report(panel: pd.DataFrame) -> None:
    features = [c for c in panel.columns if c not in ("ticker", "year")]
    print(f"\nPanel rebuilt: {len(panel)} rows × {len(panel.columns)} columns")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Years:   {sorted(panel['year'].unique())}")
    print(f"\n{'Column':<30} {'Filled':>6}  {'%':>5}")
    print("-" * 45)
    for f in features:
        filled = panel[f].notna().sum()
        pct    = filled / len(panel) * 100
        print(f"  {f:<28} {filled:>5}/{len(panel)}  {pct:>4.0f}%")


def spot_check(panel: pd.DataFrame) -> None:
    print("\n── AAPL FY2023 spot check ──")
    row = panel[(panel["ticker"] == "AAPL") & (panel["year"] == "FY2023")]
    if row.empty:
        print("  NOT FOUND")
        return
    for col, val in row.iloc[0].items():
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            if isinstance(val, float) and abs(val) > 1e6:
                print(f"  {col:<30}  {val:>20,.0f}")
            else:
                print(f"  {col:<30}  {val}")
        else:
            print(f"  {col:<30}  NULL")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    print(f"DB: {DB_PATH}  ({DB_PATH.stat().st_size // 1024} KB)\n")

    print("── Building columns from financials (statement-filtered) ──")
    panel = build_panel(conn)

    report(panel)
    spot_check(panel)

    # Save
    panel.to_sql("panel", conn, if_exists="replace", index=False)
    panel.to_csv(CSV_PATH, index=False)
    conn.close()
    print(f"\nSaved → {DB_PATH}  [table: panel]")
    print(f"Saved → {CSV_PATH}")


if __name__ == "__main__":
    main()
