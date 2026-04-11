"""
Build ticker × year panel table from long-format financials.

Output:
  - SQLite table: `panel`  (in data/financials.db)
  - CSV: data/panel.csv

Run:
    python data_pipeline/scripts/build_panel.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH  = Path(__file__).resolve().parents[2] / "data" / "financials.db"
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "panel.csv"

# ── Metrics to keep (meaningful financials with >40% ticker-year coverage) ────
# Excludes: Item X section headers, stock-option counts (Granted/Vested/Forfeited/Exercised),
#           pension line items (Service cost, Benefits paid, Settlements, Discount rate),
#           geographic splits (United States, Foreign, Federal), and vague catch-alls (Other)
KEEP_METRICS = {
    # Balance sheet
    "Cash and cash equivalents":        "cash",
    "Total assets":                     "total_assets",
    "Total current assets":             "current_assets",
    "Total current liabilities":        "current_liabilities",
    "Total liabilities":                "total_liabilities",
    "Goodwill":                         "goodwill",
    "Deferred income taxes":            "deferred_tax",
    "Accounts payable":                 "accounts_payable",
    "Retained earnings":                "retained_earnings",
    "Other assets":                     "other_assets",
    "Other current assets":             "other_current_assets",
    "Other liabilities":                "other_liabilities",
    "Long-term debt":                   "long_term_debt",
    "Inventories":                      "inventories",
    "Accumulated other comprehensive loss": "aoci",
    "Additional paid-in capital":       "apic",
    "Land":                             "land",

    # Income statement
    "Net income":                       "net_income",
    "Interest expense":                 "interest_expense",
    "Interest income":                  "interest_income",
    "Depreciation and amortization":    "da",
    "Operating income":                 "operating_income",
    "Comprehensive income":             "comprehensive_income",

    # Cash flow
    "Net cash provided by operating activities": "cfo",
    "Capital expenditures":             "capex",
    "Operating leases":                 "operating_leases",
}

# ── Load ──────────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
rows = pd.read_sql(
    "SELECT ticker, period, metric, value FROM financials "
    "WHERE period LIKE 'FY%' AND value IS NOT NULL",
    conn,
)

# Keep only selected metrics and rename to clean column names
rows = rows[rows["metric"].isin(KEEP_METRICS)]
rows["feature"] = rows["metric"].map(KEEP_METRICS)

# Deduplicate: some filings report the same metric twice (e.g. Q4 and annual overlap)
# Take the max value — annual totals are always larger than quarterly partials
rows = rows.sort_values("value").drop_duplicates(
    subset=["ticker", "period", "feature"], keep="last"
)

# ── Pivot to wide format ──────────────────────────────────────────────────────
panel = rows.pivot_table(
    index=["ticker", "period"],
    columns="feature",
    values="value",
    aggfunc="last",
).reset_index()

panel.columns.name = None
panel = panel.rename(columns={"period": "year"})
panel = panel.sort_values(["ticker", "year"]).reset_index(drop=True)

# ── Derived ratios ────────────────────────────────────────────────────────────
# These are the most commonly used in fundamental analysis
panel["current_ratio"]    = panel["current_assets"]  / panel["current_liabilities"]
panel["debt_to_assets"]   = panel["long_term_debt"]   / panel["total_assets"]
panel["net_margin"]       = panel["net_income"]        / panel.get("total_assets")  # proxy; revenue not cleanly available
panel["roa"]              = panel["net_income"]        / panel["total_assets"]

# ── Save ──────────────────────────────────────────────────────────────────────
panel.to_sql("panel", conn, if_exists="replace", index=False)
panel.to_csv(CSV_PATH, index=False)
conn.close()

# ── Report ────────────────────────────────────────────────────────────────────
features = [c for c in panel.columns if c not in ("ticker", "year")]
print(f"\nPanel table built: {len(panel)} rows × {len(panel.columns)} columns")
print(f"Tickers : {panel['ticker'].nunique()}")
print(f"Years   : {sorted(panel['year'].unique())}")
print(f"\nFeatures ({len(features)}):")
for i, f in enumerate(sorted(features), 1):
    filled = panel[f].notna().sum()
    pct    = filled / len(panel) * 100
    print(f"  {i:>2}. {f:<35} {filled:>4}/{len(panel)} rows  ({pct:.0f}% filled)")

print(f"\nSaved → {CSV_PATH}")
print(f"Saved → {DB_PATH}  [table: panel]")

print("\nSample (AAPL):")
print(panel[panel["ticker"] == "AAPL"].to_string(index=False))
