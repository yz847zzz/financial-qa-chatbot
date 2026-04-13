"""
Post-processing for NL2SQL model output.

Two steps applied after the model generates SQL:
  1. Synonym repair  — replaces unknown column aliases with canonical panel columns
  2. Schema validation — asserts all referenced columns actually exist in panel

Both steps are deterministic and fast (no model call). They act as a safety net
for the ~10-15% of cases where the fine-tuned model maps a synonym incorrectly
or hallucinates a column name that doesn't exist.
"""

import re
import sqlite3
from pathlib import Path

# ── Canonical panel schema ─────────────────────────────────────────────────────
PANEL_COLUMNS = {
    # keys: actual column names in the panel table
    "ticker", "year",
    # monetary
    "total_revenue", "gross_profit", "operating_income", "net_income",
    "r_and_d", "interest_expense", "interest_income", "da",
    "cfo", "capex", "buybacks", "dividends_paid",
    "cash", "total_assets", "current_assets", "current_liabilities",
    "total_liabilities", "long_term_debt", "goodwill", "retained_earnings",
    "inventories", "accounts_payable", "eps_diluted",
    # ratios (computed)
    "current_ratio", "net_margin", "roa", "debt_to_assets",
}

FILING_META_COLUMNS = {
    "ticker", "company", "filing_type", "date", "accession_number",
}

# ── Synonym → canonical column mapping ────────────────────────────────────────
# These are the aliases the model might generate that don't exist in the schema.
# Ordered from most specific to most general to prevent wrong substitutions.
SYNONYM_MAP: dict[str, str] = {
    # net_income aliases
    "profit":                  "net_income",
    "profits":                 "net_income",
    "net_profit":              "net_income",
    "net_earnings":            "net_income",
    "earnings":                "net_income",
    "bottom_line":             "net_income",
    "net":                     "net_income",      # context-dependent but usually correct
    "income":                  "net_income",      # ambiguous — prefer net_income
    "after_tax_profit":        "net_income",
    "take_home_profit":        "net_income",

    # total_revenue aliases
    "revenue":                 "total_revenue",
    "revenues":                "total_revenue",
    "sales":                   "total_revenue",
    "net_sales":               "total_revenue",
    "total_sales":             "total_revenue",
    "gross_sales":             "total_revenue",
    "turnover":                "total_revenue",
    "top_line":                "total_revenue",
    "net_revenue":             "total_revenue",

    # operating_income aliases
    "ebit":                    "operating_income",
    "op_income":               "operating_income",
    "operating_profit":        "operating_income",
    "operating_earnings":      "operating_income",
    "income_from_operations":  "operating_income",
    "operating_result":        "operating_income",

    # cfo aliases
    "operating_cash_flow":     "cfo",
    "cash_from_operations":    "cfo",
    "cash_flow_from_operations": "cfo",
    "operating_cash":          "cfo",

    # capex aliases
    "capital_expenditures":    "capex",
    "capital_spending":        "capex",
    "ppe_purchases":           "capex",

    # cash aliases
    "cash_balance":            "cash",
    "cash_on_hand":            "cash",
    "cash_and_cash_equivalents": "cash",
    "cash_holdings":           "cash",
    "cash_position":           "cash",
    "liquidity":               "cash",

    # total_assets aliases
    "assets":                  "total_assets",
    "asset_base":              "total_assets",
    "book_assets":             "total_assets",

    # current_assets aliases
    "short_term_assets":       "current_assets",
    "liquid_assets":           "current_assets",

    # current_liabilities aliases
    "short_term_liabilities":  "current_liabilities",
    "short_term_obligations":  "current_liabilities",

    # total_liabilities aliases
    "liabilities":             "total_liabilities",
    "total_obligations":       "total_liabilities",

    # long_term_debt aliases
    "ltd":                     "long_term_debt",
    "long_term_borrowings":    "long_term_debt",
    "debt":                    "long_term_debt",
    "borrowings":              "long_term_debt",

    # inventories aliases
    "inventory":               "inventories",
    "stock":                   "inventories",

    # accounts_payable aliases
    "payables":                "accounts_payable",
    "trade_payables":          "accounts_payable",
    "ap":                      "accounts_payable",

    # retained_earnings aliases
    "accumulated_earnings":    "retained_earnings",
    "retained_profits":        "retained_earnings",

    # interest_expense aliases
    "interest_cost":           "interest_expense",
    "finance_cost":            "interest_expense",
    "borrowing_cost":          "interest_expense",
    "cost_of_debt":            "interest_expense",

    # da aliases
    "depreciation":            "da",
    "amortization":            "da",
    "d_and_a":                 "da",
    "d&a":                     "da",

    # new panel columns — aliases
    "gross_margin_dollars":    "gross_profit",
    "rd":                      "r_and_d",
    "r_and_d_expense":         "r_and_d",
    "research_and_development": "r_and_d",
    "research_expense":        "r_and_d",
    "share_repurchases":       "buybacks",
    "stock_buybacks":          "buybacks",
    "repurchases":             "buybacks",
    "dividends":               "dividends_paid",
    "dividend_payments":       "dividends_paid",
    "eps":                     "eps_diluted",
    "diluted_eps":             "eps_diluted",
    "earnings_per_share":      "eps_diluted",

    # ratio aliases
    "net_profit_margin":       "net_margin",
    "profit_margin":           "net_margin",
    "return_on_sales":         "net_margin",

    "return_on_assets":        "roa",
    "asset_returns":           "roa",
    "asset_productivity":      "roa",

    "liquidity_ratio":         "current_ratio",
    "working_capital_ratio":   "current_ratio",

    "debt_ratio":              "debt_to_assets",
    "leverage_ratio":          "debt_to_assets",
    "debt_to_assets_ratio":    "debt_to_assets",
}

# Pre-compute lowercase key lookup
_SYNONYM_LOWER: dict[str, str] = {k.lower(): v for k, v in SYNONYM_MAP.items()}

# ── Ticker alias repair ────────────────────────────────────────────────────────
# Map common alternate ticker symbols to the canonical ones stored in the DB.
# Applied inside string literals (WHERE ticker = 'GOOGL' → 'GOOG').
TICKER_ALIASES: dict[str, str] = {
    "GOOGL": "GOOG",    # Alphabet class A — DB uses GOOG (class C)
    "META":  "META",    # already correct — placeholder if needed later
    "BRK.B": "BRK-B",  # Berkshire Hathaway class B dash variant
    "BRK/B": "BRK-B",
}


# ── Core repair function ───────────────────────────────────────────────────────

def repair_sql(sql: str) -> tuple[str, list[str]]:
    """
    Scan the SQL for unknown column names and replace with canonical equivalents.

    Returns:
        (repaired_sql, list_of_repairs_made)

    Works by finding bare identifiers (not quoted, not keywords) that aren't in
    PANEL_COLUMNS or FILING_META_COLUMNS, then checking SYNONYM_MAP.
    """
    repairs: list[str] = []

    # Tokenize: find all word-like tokens that could be column names
    # Exclude SQL keywords and aggregate functions
    SQL_KEYWORDS = {
        "select", "from", "where", "and", "or", "not", "in", "is", "null",
        "order", "by", "asc", "desc", "limit", "join", "on", "as", "group",
        "having", "between", "like", "distinct", "count", "sum", "avg", "max",
        "min", "case", "when", "then", "else", "end", "with", "inner", "outer",
        "left", "right", "cross", "union", "all", "insert", "update", "delete",
        "create", "drop", "alter", "table", "index", "view",
        "a", "b",  # common join aliases
    }

    all_columns = PANEL_COLUMNS | FILING_META_COLUMNS

    def replace_token(m: re.Match) -> str:
        token = m.group(0)
        lower = token.lower()

        # Keep: SQL keywords
        if lower in SQL_KEYWORDS:
            return token

        # Keep: known columns (case-insensitive)
        if lower in {c.lower() for c in all_columns}:
            return token

        # Keep: string literals (handled separately — skip inside quotes)
        # Keep: numbers
        if re.fullmatch(r'\d+(\.\d+)?([eE][+-]?\d+)?', token):
            return token

        # Check synonym map
        canonical = _SYNONYM_LOWER.get(lower)
        if canonical:
            repairs.append(f"{token} -> {canonical}")
            return canonical

        # Unknown token — leave as-is (don't break valid SQL we don't understand)
        return token

    # Process tokens outside of string literals
    # Split on quoted strings first, then process unquoted parts
    parts = re.split(r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")", sql)
    result_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside a string literal — don't touch
            result_parts.append(part)
        else:
            # Outside string literal — replace unknown tokens
            repaired = re.sub(r'\b[A-Za-z_][A-Za-z0-9_]*\b', replace_token, part)
            result_parts.append(repaired)

    repaired_sql = "".join(result_parts)

    # Step 2: ticker alias repair inside string literals
    for alias, canonical in TICKER_ALIASES.items():
        if alias == canonical:
            continue
        # Match the alias as a quoted string value (case-sensitive ticker symbols)
        pattern = rf"(?<=['\"]){re.escape(alias)}(?=['\"])"
        if re.search(pattern, repaired_sql):
            repaired_sql = re.sub(pattern, canonical, repaired_sql)
            repairs.append(f"ticker {alias} -> {canonical}")

    return repaired_sql, repairs


def validate_sql(sql: str, db_path: str | None = None) -> tuple[bool, str]:
    """
    Validate that the SQL is safe (SELECT only) and optionally test-execute it.

    Returns (is_valid, error_message).
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return False, "Only SELECT statements allowed"
    for dangerous in ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"):
        if dangerous in stripped:
            return False, f"Dangerous keyword: {dangerous}"

    if db_path:
        try:
            con = sqlite3.connect(db_path)
            con.execute(sql)
            con.close()
        except sqlite3.Error as e:
            return False, str(e)

    return True, ""


def postprocess(sql: str, db_path: str | None = None) -> dict:
    """
    Full post-processing pipeline for a model-generated SQL string.

    Returns a dict with:
        sql_original   : the model output
        sql_repaired   : after synonym repair
        repairs        : list of substitutions made
        is_valid       : True if SQL is safe and (optionally) executes
        error          : error message if not valid
    """
    # Step 1: synonym repair
    sql_repaired, repairs = repair_sql(sql)

    # Step 2: validate
    is_valid, error = validate_sql(sql_repaired, db_path)

    return {
        "sql_original": sql,
        "sql_repaired": sql_repaired,
        "repairs":      repairs,
        "is_valid":     is_valid,
        "error":        error,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # Model uses synonym instead of column name
        "SELECT profit FROM panel WHERE ticker='AAPL' AND year='FY2023'",
        "SELECT revenue, earnings FROM panel WHERE ticker='MSFT' AND year='FY2022'",
        "SELECT ebit FROM panel WHERE ticker='GOOG' AND year='FY2023'",
        "SELECT assets, debt FROM panel WHERE ticker='TSLA' AND year='FY2021'",
        "SELECT turnover FROM panel WHERE ticker='WMT' AND year='FY2023' ORDER BY turnover DESC",
        # Already correct — no repair needed
        "SELECT net_income FROM panel WHERE ticker='AAPL' AND year='FY2023'",
        "SELECT ticker, total_revenue FROM panel WHERE year='FY2023' ORDER BY total_revenue DESC LIMIT 5",
        # Mixed: one synonym, one correct
        "SELECT net_income, liabilities FROM panel WHERE ticker='JPM' AND year='FY2022'",
        # D&A (special character)
        "SELECT d&a FROM panel WHERE ticker='NVDA' AND year='FY2023'",
    ]

    print("=== SQL Post-processing Tests ===\n")
    for sql in test_cases:
        result = postprocess(sql)
        changed = result["sql_original"] != result["sql_repaired"]
        print(f"Input:    {result['sql_original']}")
        if changed:
            print(f"Repaired: {result['sql_repaired']}")
            print(f"Changes:  {result['repairs']}")
        else:
            print(f"(no repair needed)")
        print(f"Valid:    {result['is_valid']}")
        print()
