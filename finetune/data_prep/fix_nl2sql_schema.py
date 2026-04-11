#!/usr/bin/env python3
"""
Fix NL2SQL training data: rewrite `panel` table SQL → `financials` long-format SQL.

panel (wide):   SELECT net_income FROM panel WHERE ticker='AAPL' AND year='FY2023'
financials (long): SELECT value FROM financials WHERE ticker='AAPL' AND metric='Net Income' AND period='F2023'

Run:
    python finetune/data_prep/fix_nl2sql_schema.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_IN  = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_IN   = ROOT / "data" / "nl2sql" / "eval.jsonl"
TRAIN_OUT = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_OUT  = ROOT / "data" / "nl2sql" / "eval.jsonl"

# panel column → canonical metric name used in financials.metric
METRIC_MAP: dict[str, str] = {
    "total_assets":       "Total Assets",
    "net_income":         "Net Income",
    "current_ratio":      "Current Ratio",
    "net_margin":         "Net Profit Margin",
    "operating_income":   "Operating Income",
    "cash":               "Cash And Cash Equivalents",
    "debt_to_assets":     "Debt To Assets Ratio",
    "interest_expense":   "Interest Expense",
    "roa":                "Return On Assets",
    "capex":              "Capital Expenditures",
    "current_liabilities":"Total Current Liabilities",
    "accounts_payable":   "Accounts Payable",
    "goodwill":           "Goodwill",
    "current_assets":     "Total Current Assets",
    "total_liabilities":  "Total Liabilities",
    "cfo":                "Operating Cash Flow",
    "retained_earnings":  "Retained Earnings",
    "long_term_debt":     "Long Term Debt",
    "inventories":        "Inventories",
    "da":                 "Depreciation And Amortization",
    "revenue":            "Total Revenue",
    "total_revenue":      "Total Revenue",
    "gross_profit":       "Gross Profit",
    "gross_margin":       "Gross Profit Margin",
    "roe":                "Return On Equity",
    "ebitda":             "EBITDA",
    "eps":                "Earnings Per Share",
    "operating_expenses": "Operating Expenses",
    "total_equity":       "Total Stockholders Equity",
    "working_capital":    "Working Capital",
    "quick_ratio":        "Quick Ratio",
    "free_cash_flow":     "Free Cash Flow",
    "operating_cash_flow":"Operating Cash Flow",
    "total_debt":         "Total Debt",
    "book_value":         "Book Value Per Share",
    "pe_ratio":           "Price Earnings Ratio",
    "dividend_yield":     "Dividend Yield",
}

# Non-metric columns that appear in panel table (never need metric= filter)
NON_METRIC_COLS = {"ticker", "year", "company", "sector", "industry"}


def _fy_to_period(fy: str) -> str:
    """'FY2023' → 'F2023'"""
    return fy.replace("FY", "F")


def _rewrite_year_conditions(sql: str) -> str:
    """Replace year = 'FY{YYYY}' with period = 'F{YYYY}' everywhere."""
    # year = 'FY2023' or year='FY2023'
    sql = re.sub(
        r"\byear\s*=\s*'FY(\d{4})'",
        lambda m: f"period = 'F{m.group(1)}'",
        sql, flags=re.I,
    )
    # year IN ('FY2021', 'FY2022', 'FY2023')
    def _rewrite_in(m: re.Match) -> str:
        years = re.findall(r"FY(\d{4})", m.group(1))
        periods = ", ".join(f"'F{y}'" for y in years)
        return f"period IN ({periods})"
    sql = re.sub(r"\byear\s+IN\s*\(([^)]+)\)", _rewrite_in, sql, flags=re.I)
    # year BETWEEN 'FY2020' AND 'FY2023' → period IN ('F2020','F2021','F2022','F2023')
    def _rewrite_between(m: re.Match) -> str:
        y1, y2 = int(m.group(1)), int(m.group(2))
        periods = ", ".join(f"'F{y}'" for y in range(y1, y2 + 1))
        return f"period IN ({periods})"
    sql = re.sub(
        r"\byear\s+BETWEEN\s+'FY(\d{4})'\s+AND\s+'FY(\d{4})'",
        _rewrite_between, sql, flags=re.I,
    )
    # ORDER BY year → ORDER BY period
    sql = re.sub(r"\bORDER\s+BY\s+year\b", "ORDER BY period", sql, flags=re.I)
    # residual bare 'year' column reference
    sql = re.sub(r"\byear\b(?!\s*=|\s+IN|\s+BETWEEN)", "period", sql, flags=re.I)
    return sql


def _find_metric_col(select_clause: str) -> str | None:
    """
    Return the first non-meta column name from the SELECT clause that maps to a metric.
    Handles: col, ticker, AVG(col), MAX(col), ticker, col, etc.
    """
    # Strip aggregate wrappers
    stripped = re.sub(r"\b(AVG|SUM|MAX|MIN|COUNT)\s*\(([^)]+)\)", r"\2", select_clause, flags=re.I)
    tokens = [t.strip().lower() for t in stripped.split(",")]
    for tok in tokens:
        tok = tok.strip().strip("'\"")
        if tok and tok not in NON_METRIC_COLS and tok in METRIC_MAP:
            return tok
    return None


def transform_panel_sql(sql: str) -> str | None:
    """
    Rewrite a panel-schema SQL to financials-schema SQL.
    Returns None if conversion fails (caller should discard the example).
    """
    if "panel" not in sql.lower():
        return sql  # already correct table

    # ── 1. Extract SELECT clause ─────────────────────────────────────────────
    m = re.match(r"(SELECT\s+)(.+?)\s+FROM\s+panel\b", sql, re.I | re.S)
    if not m:
        return None

    prefix      = m.group(1)   # "SELECT "
    sel_clause  = m.group(2)   # e.g. "ticker, net_income" or "AVG(roa)"
    rest        = sql[m.end():]  # everything after "FROM panel"

    metric_col = _find_metric_col(sel_clause)
    if metric_col is None:
        # No recognisable metric column — discard
        return None

    metric_name = METRIC_MAP[metric_col]
    metric_col_re = re.compile(rf"\b{re.escape(metric_col)}\b", re.I)

    # ── 2. Rewrite SELECT columns ─────────────────────────────────────────────
    # Replace the metric column with 'value' (preserve aggregates)
    new_sel = metric_col_re.sub("value", sel_clause)
    new_sel = re.sub(r"\byear\b", "period", new_sel, flags=re.I)

    # ── 3. Rewrite FROM / WHERE / ORDER BY ───────────────────────────────────
    # Change table name
    rest = re.sub(r"\bpanel\b", "financials", rest, flags=re.I)

    # Replace metric column in WHERE / ORDER BY / HAVING
    rest = metric_col_re.sub("value", rest)

    # Rewrite year conditions → period
    rest = _rewrite_year_conditions(rest)

    # ── 4. Inject metric filter into WHERE ───────────────────────────────────
    metric_filter = f"metric = '{metric_name}'"
    where_match = re.search(r"\bWHERE\b", rest, re.I)
    if where_match:
        insert_pos = where_match.end()
        rest = rest[:insert_pos] + f" {metric_filter} AND" + rest[insert_pos:]
    else:
        # No WHERE clause — insert before ORDER BY / GROUP BY / LIMIT or at end
        order_match = re.search(r"\b(ORDER|GROUP|LIMIT)\b", rest, re.I)
        if order_match:
            rest = rest[:order_match.start()] + f" WHERE {metric_filter} " + rest[order_match.start():]
        else:
            rest = rest.rstrip(";").rstrip() + f" WHERE {metric_filter}"

    # ── 5. Reassemble ────────────────────────────────────────────────────────
    new_sql = f"{prefix}{new_sel} FROM financials{rest}"
    # Tidy up trailing semicolon
    new_sql = new_sql.rstrip()
    if not new_sql.endswith(";"):
        new_sql += ";"
    return new_sql


def fix_file(in_path: Path, out_path: Path) -> tuple[int, int, int]:
    """Returns (total, fixed, discarded)."""
    examples = []
    with open(in_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    fixed = discarded = 0
    for ex in raw:
        sql = ex["messages"][2]["content"]
        if "panel" not in sql.lower():
            examples.append(ex)
            continue
        new_sql = transform_panel_sql(sql)
        if new_sql is None:
            discarded += 1
            continue
        ex["messages"][2]["content"] = new_sql
        examples.append(ex)
        fixed += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(raw), fixed, discarded


if __name__ == "__main__":
    for label, in_p, out_p in [("train", TRAIN_IN, TRAIN_OUT), ("eval", EVAL_IN, EVAL_OUT)]:
        total, fixed, discarded = fix_file(in_p, out_p)
        kept = total - discarded
        print(f"{label}: {total} → {kept} kept ({fixed} rewritten, {discarded} discarded)")

    # Spot-check
    print("\n── spot-check (first 3 rewritten examples) ──")
    with open(TRAIN_OUT, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    shown = 0
    for ex in lines:
        if "financials" in ex["messages"][2]["content"] and shown < 3:
            print("Q:", ex["messages"][1]["content"][:70])
            print("S:", ex["messages"][2]["content"][:110])
            print()
            shown += 1
