#!/usr/bin/env python3
"""
Template-based NL2SQL dataset generator — targets the `panel` table (wide format).

Generates ~900 examples covering all query patterns × column × company combinations.
Each (NL, SQL) pair is logically consistent by construction — no API needed.

Output: appends to data/nl2sql/train.jsonl and data/nl2sql/eval.jsonl
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_OUT = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_OUT  = ROOT / "data" / "nl2sql" / "eval.jsonl"
EVAL_RATIO = 0.15

random.seed(99)

# ── Shared with build_nl2sql_dataset.py — must stay in sync ──────────────────
SYSTEM_PROMPT = """You have access to a SQLite database with these tables:

panel (ticker TEXT, year TEXT, total_revenue REAL, gross_profit REAL,
       operating_income REAL, net_income REAL, r_and_d REAL,
       interest_expense REAL, interest_income REAL, da REAL,
       cfo REAL, capex REAL, buybacks REAL, dividends_paid REAL,
       cash REAL, total_assets REAL, current_assets REAL,
       current_liabilities REAL, total_liabilities REAL, long_term_debt REAL,
       goodwill REAL, retained_earnings REAL, inventories REAL,
       accounts_payable REAL, eps_diluted REAL,
       current_ratio REAL, net_margin REAL, roa REAL, debt_to_assets REAL)
  - ticker: stock symbol e.g. 'AAPL', 'MSFT'
  - year: fiscal year string e.g. 'FY2023'
  - all monetary columns are in USD; eps_diluted is USD per share
  - buybacks and dividends_paid are stored as positive values

filing_metadata (ticker TEXT, company TEXT, filing_type TEXT, date TEXT, accession_number TEXT)
  - filing_type: '10-K' or '8-K'
  - date: 'YYYY-MM-DD'

Rules:
- Output ONLY a valid SQL SELECT. No explanation.
- Ticker symbols are uppercase.
- Year format is 'FY{YYYY}' e.g. 'FY2023'.
- Use IS NOT NULL to filter missing values."""

# ── Universe ──────────────────────────────────────────────────────────────────
COMPANIES = [
    ("AAPL", "Apple"),      ("MSFT", "Microsoft"),  ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet"),  ("META", "Meta"),        ("NVDA", "NVIDIA"),
    ("TSLA", "Tesla"),      ("JPM", "JPMorgan"),     ("V", "Visa"),
    ("JNJ", "Johnson & Johnson"), ("WMT", "Walmart"),("PG", "Procter & Gamble"),
    ("UNH", "UnitedHealth"),("HD", "Home Depot"),    ("DIS", "Disney"),
    ("BAC", "Bank of America"),("XOM", "ExxonMobil"),("PFE", "Pfizer"),
    ("CVX", "Chevron"),     ("KO", "Coca-Cola"),     ("ABBV", "AbbVie"),
    ("AVGO", "Broadcom"),   ("COST", "Costco"),      ("MRK", "Merck"),
    ("TMO", "Thermo Fisher"),("CSCO", "Cisco"),      ("NKE", "Nike"),
    ("ORCL", "Oracle"),     ("CRM", "Salesforce"),   ("LLY", "Eli Lilly"),
    ("AMD", "AMD"),         ("INTC", "Intel"),       ("QCOM", "Qualcomm"),
    ("NFLX", "Netflix"),    ("MCD", "McDonald's"),   ("LOW", "Lowe's"),
    ("CAT", "Caterpillar"), ("MMM", "3M"),           ("GILD", "Gilead"),
    ("REGN", "Regeneron"),  ("BIIB", "Biogen"),      ("TXN", "Texas Instruments"),
    ("ACN", "Accenture"),   ("ABT", "Abbott"),       ("PEP", "PepsiCo"),
    ("IBM", "IBM"),         ("BA", "Boeing"),        ("HON", "Honeywell"),
]

YEARS = ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]

# panel column -> NL synonym variants
# Covers colloquial, analyst, and formal phrasings so the model learns
# that "profit" / "bottom line" / "earnings" / "net" all map to net_income, etc.
COLS = {
    "total_revenue": [
        "revenue", "revenues", "total revenue", "net sales", "sales",
        "top line", "turnover", "total sales", "gross sales",
        "net revenue", "total net sales",
    ],
    "gross_profit": [
        "gross profit", "gross income", "gross margin dollars",
        "gross earnings", "profit after COGS",
    ],
    "net_income": [
        "net income", "profit", "profits", "net profit", "net earnings",
        "earnings", "bottom line", "income", "after-tax profit",
        "profit after tax", "net", "take-home profit",
    ],
    "operating_income": [
        "operating income", "operating profit", "EBIT",
        "income from operations", "op income", "operating earnings",
        "operating result", "profit before interest and taxes",
    ],
    "r_and_d": [
        "R&D", "research and development", "research expense",
        "R&D expense", "R&D spending", "research spending",
        "research and development expense",
    ],
    "cfo": [
        "operating cash flow", "cash from operations", "CFO",
        "cash generated from operations", "cash flow from operations",
        "operating cash", "cash from operating activities",
    ],
    "capex": [
        "capital expenditures", "capex", "CapEx",
        "capital spending", "PP&E purchases",
        "investment in property and equipment",
        "infrastructure spending",
    ],
    "buybacks": [
        "buybacks", "share repurchases", "stock buybacks",
        "share buybacks", "repurchase program", "stock repurchases",
    ],
    "dividends_paid": [
        "dividends", "dividends paid", "dividend payments",
        "total dividends", "cash dividends", "dividends to shareholders",
    ],
    "cash": [
        "cash", "cash balance", "cash on hand", "cash and cash equivalents",
        "liquidity", "cash holdings", "available cash", "cash position",
    ],
    "total_assets": [
        "total assets", "assets", "asset base", "total asset base",
        "book assets",
    ],
    "current_assets": [
        "current assets", "short-term assets", "liquid assets",
    ],
    "current_liabilities": [
        "current liabilities", "short-term liabilities",
        "short-term obligations", "current debt",
    ],
    "total_liabilities": [
        "total liabilities", "liabilities", "total obligations",
        "total debt and liabilities",
    ],
    "goodwill": [
        "goodwill", "goodwill intangibles", "acquisition goodwill",
    ],
    "long_term_debt": [
        "long-term debt", "long term debt", "LTD",
        "long-term borrowings", "notes payable", "bonds outstanding",
        "debt", "borrowings",
    ],
    "inventories": [
        "inventories", "inventory", "stock",
        "merchandise inventory", "goods on hand",
    ],
    "accounts_payable": [
        "accounts payable", "AP", "payables",
        "trade payables", "vendor payables", "money owed to suppliers",
    ],
    "retained_earnings": [
        "retained earnings", "accumulated earnings",
        "retained profits", "plowback earnings",
    ],
    "interest_expense": [
        "interest expense", "interest cost", "finance cost",
        "borrowing cost", "interest paid", "cost of debt",
    ],
    "interest_income": [
        "interest income", "interest earned", "investment income",
        "income from cash", "interest received",
    ],
    "da": [
        "depreciation and amortization", "D&A", "depreciation",
        "amortization", "D and A", "non-cash charges",
    ],
    "eps_diluted": [
        "EPS", "earnings per share", "diluted EPS",
        "diluted earnings per share", "EPS diluted", "per-share earnings",
    ],
    "net_margin": [
        "net profit margin", "net margin", "profit margin",
        "margin", "net income margin", "return on sales",
        "how profitable", "profitability",
    ],
    "roa": [
        "return on assets", "ROA", "asset returns",
        "asset productivity", "return on total assets",
        "how efficiently assets are used",
    ],
    "current_ratio": [
        "current ratio", "liquidity ratio", "working capital ratio",
        "short-term liquidity",
    ],
    "debt_to_assets": [
        "debt to assets", "debt-to-assets ratio", "debt ratio",
        "leverage ratio", "financial leverage", "how leveraged",
    ],
}

FLOW_COLS = ["total_revenue", "gross_profit", "net_income", "operating_income",
             "r_and_d", "cfo", "capex", "buybacks", "dividends_paid",
             "interest_expense", "da"]
BALANCE_COLS = ["cash", "total_assets", "current_assets", "current_liabilities",
                "total_liabilities", "goodwill", "long_term_debt", "inventories",
                "accounts_payable", "retained_earnings"]
RATIO_COLS = ["net_margin", "roa", "current_ratio", "debt_to_assets"]
EPS_COLS   = ["eps_diluted"]


def rc() -> tuple[str, str]:
    return random.choice(COMPANIES)


def ry() -> str:
    return random.choice(YEARS)


def rcol(pool=None) -> str:
    return random.choice(pool or list(COLS.keys()))


def rnl(col: str) -> str:
    return random.choice(COLS[col])


def rname(ticker: str, company: str) -> str:
    return random.choice([ticker, company])


def example(nl: str, sql: str) -> dict:
    sql = sql.strip().rstrip(";") + ";"
    return {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": nl},
        {"role": "assistant", "content": sql},
    ]}


# ── Pattern generators ────────────────────────────────────────────────────────

def gen_single_lookup(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        ticker, company = rc()
        col = rcol()
        year = ry()
        ref = rname(ticker, company)
        nl_metric = rnl(col)
        nl = random.choice([
            f"What was {ref}'s {nl_metric} in {year}?",
            f"Show {ref} {nl_metric} for {year}.",
            f"Find {ref}'s {nl_metric} in fiscal year {year[-4:]}.",
            f"Report {ref} {nl_metric} for fiscal {year[-4:]}.",
            f"What is {ref}'s {nl_metric} for {year}?",
        ])
        sql = f"SELECT {col} FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        out.append(example(nl, sql))
    return out


def gen_multi_col(n: int) -> list[dict]:
    """Fetch two columns in one query."""
    out = []
    for _ in range(n):
        ticker, company = rc()
        year = ry()
        c1 = rcol(FLOW_COLS)
        c2 = rcol(BALANCE_COLS)
        nl = random.choice([
            f"Get {company}'s {rnl(c1)} and {rnl(c2)} for {year}.",
            f"What are {company}'s {rnl(c1)} and {rnl(c2)} in {year}?",
            f"Show me {ticker}'s {rnl(c1)} alongside {rnl(c2)} in {year}.",
        ])
        sql = f"SELECT {c1}, {c2} FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        out.append(example(nl, sql))
    return out


def gen_ranking(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        year = ry()
        col = rcol(FLOW_COLS + RATIO_COLS)
        k = random.choice([3, 5, 10])
        order = random.choice(["DESC", "ASC"])
        superlative = "highest" if order == "DESC" else "lowest"
        nl = random.choice([
            f"Which {k} companies had the {superlative} {rnl(col)} in {year}?",
            f"List the top {k} companies by {rnl(col)} in {year}.",
            f"Rank companies by {rnl(col)} for {year} ({superlative} first).",
        ])
        sql = (f"SELECT ticker, {col} FROM panel "
               f"WHERE year='{year}' AND {col} IS NOT NULL "
               f"ORDER BY {col} {order} LIMIT {k}")
        out.append(example(nl, sql))
    return out


def gen_trend(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        ticker, company = rc()
        col = rcol(FLOW_COLS)
        ref = rname(ticker, company)
        nl = random.choice([
            f"Show {ref}'s {rnl(col)} trend from FY2020 to FY2024.",
            f"How has {ref}'s {rnl(col)} changed over the years?",
            f"What is {ref}'s annual {rnl(col)} history?",
            f"Plot {ref}'s {rnl(col)} for all available years.",
        ])
        sql = (f"SELECT year, {col} FROM panel "
               f"WHERE ticker='{ticker}' AND {col} IS NOT NULL ORDER BY year")
        out.append(example(nl, sql))
    return out


def gen_comparison(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        tickers = random.sample(COMPANIES, 2)
        t1, c1 = tickers[0]
        t2, c2 = tickers[1]
        year = ry()
        col = rcol()
        nl = random.choice([
            f"Compare {c1} and {c2} {rnl(col)} in {year}.",
            f"How does {c1}'s {rnl(col)} compare to {c2}'s in {year}?",
            f"Show {t1} vs {t2} {rnl(col)} for {year}.",
        ])
        sql = (f"SELECT ticker, {col} FROM panel "
               f"WHERE ticker IN ('{t1}','{t2}') AND year='{year}' ORDER BY {col} DESC")
        out.append(example(nl, sql))
    return out


def gen_filter(n: int) -> list[dict]:
    out = []
    filters = [
        ("roa",          ">",  0.10, "high return on assets (ROA > 10%)"),
        ("roa",          ">",  0.15, "return on assets above 15%"),
        ("net_margin",   ">",  0.20, "net margin above 20%"),
        ("net_margin",   ">",  0.10, "profitable (net margin > 10%)"),
        ("current_ratio",">",  2.0,  "strong liquidity (current ratio > 2)"),
        ("current_ratio","<",  1.0,  "current ratio below 1"),
        ("debt_to_assets",">", 0.50, "high leverage (debt > 50% of assets)"),
        ("debt_to_assets","<", 0.30, "low leverage"),
        ("net_income",   ">",  1e10, "net income above $10 billion"),
        ("total_revenue",">",  5e10, "revenue above $50 billion"),
    ]
    for _ in range(n):
        year = ry()
        col, op, val, description = random.choice(filters)
        nl = random.choice([
            f"Which companies had {description} in {year}?",
            f"List companies with {rnl(col)} {op} {val} in {year}.",
            f"Find all tickers where {rnl(col)} {op} {val} for {year}.",
        ])
        sql = (f"SELECT ticker, {col} FROM panel "
               f"WHERE year='{year}' AND {col} {op} {val} ORDER BY {col} DESC")
        out.append(example(nl, sql))
    return out


def gen_yoy_change(n: int) -> list[dict]:
    out = []
    year_pairs = [("FY2021", "FY2022"), ("FY2022", "FY2023"), ("FY2020", "FY2023")]
    for _ in range(n):
        ticker, company = rc()
        col = rcol(FLOW_COLS)
        y1, y2 = random.choice(year_pairs)
        ref = rname(ticker, company)
        nl = random.choice([
            f"How did {ref}'s {rnl(col)} change from {y1} to {y2}?",
            f"What was {ref}'s {rnl(col)} growth between {y1} and {y2}?",
            f"Compare {ref} {rnl(col)} in {y1} vs {y2}.",
        ])
        sql = (f"SELECT year, {col} FROM panel "
               f"WHERE ticker='{ticker}' AND year IN ('{y1}','{y2}') ORDER BY year")
        out.append(example(nl, sql))
    return out


def gen_aggregate(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        year = ry()
        col = rcol(FLOW_COLS + BALANCE_COLS)
        agg, agg_nl = random.choice([
            ("AVG", "average"), ("MAX", "maximum"), ("MIN", "minimum"),
            ("SUM", "total"),
        ])
        nl = random.choice([
            f"What is the {agg_nl} {rnl(col)} across all companies in {year}?",
            f"Compute {agg_nl} {rnl(col)} for {year}.",
            f"{agg_nl.capitalize()} {rnl(col)} in the dataset for {year}.",
        ])
        sql = (f"SELECT {agg}({col}) FROM panel "
               f"WHERE year='{year}' AND {col} IS NOT NULL")
        out.append(example(nl, sql))
    return out


def gen_multi_year_all(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        ticker, company = rc()
        col = rcol(FLOW_COLS)
        ref = rname(ticker, company)
        y1, y2 = random.choice([("FY2020","FY2023"), ("FY2021","FY2024"), ("FY2019","FY2024")])
        nl = random.choice([
            f"Show {ref}'s {rnl(col)} for all years between {y1} and {y2}.",
            f"Get annual {rnl(col)} for {ref} from {y1} to {y2}.",
        ])
        sql = (f"SELECT year, {col} FROM panel "
               f"WHERE ticker='{ticker}' AND year BETWEEN '{y1}' AND '{y2}' ORDER BY year")
        out.append(example(nl, sql))
    return out


def gen_filing_metadata(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        ticker, company = rc()
        ref = rname(ticker, company)
        pattern = random.randint(0, 4)
        if pattern == 0:
            nl = f"When did {ref} file their most recent 10-K?"
            sql = (f"SELECT date FROM filing_metadata "
                   f"WHERE ticker='{ticker}' AND filing_type='10-K' ORDER BY date DESC LIMIT 1")
        elif pattern == 1:
            nl = f"How many 10-K filings does {ref} have?"
            sql = (f"SELECT COUNT(*) FROM filing_metadata "
                   f"WHERE ticker='{ticker}' AND filing_type='10-K'")
        elif pattern == 2:
            year = random.choice(["2020","2021","2022","2023"])
            nl = f"List {ref}'s filings from {year}."
            sql = (f"SELECT filing_type, date FROM filing_metadata "
                   f"WHERE ticker='{ticker}' AND date LIKE '{year}%' ORDER BY date")
        elif pattern == 3:
            nl = f"What company does ticker {ticker} represent?"
            sql = f"SELECT company FROM filing_metadata WHERE ticker='{ticker}' LIMIT 1"
        else:
            nl = f"List all 10-K filings after 2022 for {ref}."
            sql = (f"SELECT date, accession_number FROM filing_metadata "
                   f"WHERE ticker='{ticker}' AND filing_type='10-K' AND date > '2022-12-31' ORDER BY date DESC")
        out.append(example(nl, sql))
    return out


def gen_ratio_lookup(n: int) -> list[dict]:
    out = []
    for _ in range(n):
        ticker, company = rc()
        year = ry()
        col = rcol(RATIO_COLS)
        ref = rname(ticker, company)
        nl = random.choice([
            f"What is {ref}'s {rnl(col)} for {year}?",
            f"How does {ref}'s {rnl(col)} look in {year}?",
            f"Report {ref}'s {rnl(col)} in {year}.",
        ])
        sql = f"SELECT {col} FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        out.append(example(nl, sql))
    return out


def gen_best_in_class(n: int) -> list[dict]:
    """Who leads on a ratio metric in a given year?"""
    out = []
    for _ in range(n):
        year = ry()
        col = rcol(RATIO_COLS)
        k = random.choice([1, 3, 5])
        nl = random.choice([
            f"Which company had the best {rnl(col)} in {year}?",
            f"Top {k} companies by {rnl(col)} in {year}.",
            f"Who had the highest {rnl(col)} among all companies in {year}?",
        ])
        sql = (f"SELECT ticker, {col} FROM panel "
               f"WHERE year='{year}' AND {col} IS NOT NULL "
               f"ORDER BY {col} DESC LIMIT {k}")
        out.append(example(nl, sql))
    return out


def gen_eps_lookup(n: int) -> list[dict]:
    """EPS-specific lookups — treated separately because unit is USD/share."""
    out = []
    for _ in range(n):
        ticker, company = rc()
        year = ry()
        ref = rname(ticker, company)
        nl = random.choice([
            f"What was {ref}'s {rnl('eps_diluted')} in {year}?",
            f"Report {ref} {rnl('eps_diluted')} for {year}.",
            f"Show {ref}'s {rnl('eps_diluted')} in fiscal year {year[-4:]}.",
            f"How much did {ref} earn per share (diluted) in {year}?",
        ])
        sql = f"SELECT eps_diluted FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        out.append(example(nl, sql))
    return out


def gen_shareholder_returns(n: int) -> list[dict]:
    """Buybacks + dividends — capital return questions."""
    out = []
    for _ in range(n):
        ticker, company = rc()
        year = ry()
        ref = rname(ticker, company)
        pattern = random.randint(0, 3)
        if pattern == 0:
            col = "buybacks"
            nl = random.choice([
                f"How much did {ref} spend on {rnl(col)} in {year}?",
                f"What were {ref}'s {rnl(col)} in {year}?",
            ])
            sql = f"SELECT buybacks FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        elif pattern == 1:
            col = "dividends_paid"
            nl = random.choice([
                f"How much did {ref} pay in {rnl(col)} in {year}?",
                f"What were {ref}'s {rnl(col)} in {year}?",
            ])
            sql = f"SELECT dividends_paid FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        elif pattern == 2:
            nl = (f"Show {ref}'s total capital return (buybacks + dividends) in {year}.")
            sql = (f"SELECT buybacks, dividends_paid FROM panel "
                   f"WHERE ticker='{ticker}' AND year='{year}'")
        else:
            nl = f"Which companies had the highest {rnl('buybacks')} in {year}?"
            sql = (f"SELECT ticker, buybacks FROM panel "
                   f"WHERE year='{year}' AND buybacks IS NOT NULL "
                   f"ORDER BY buybacks DESC LIMIT 5")
        out.append(example(nl, sql))
    return out


def gen_rd_lookup(n: int) -> list[dict]:
    """R&D expense lookups."""
    out = []
    for _ in range(n):
        ticker, company = rc()
        year = ry()
        ref = rname(ticker, company)
        nl = random.choice([
            f"How much did {ref} spend on {rnl('r_and_d')} in {year}?",
            f"What was {ref}'s {rnl('r_and_d')} in {year}?",
            f"Report {ref}'s {rnl('r_and_d')} for {year}.",
        ])
        sql = f"SELECT r_and_d FROM panel WHERE ticker='{ticker}' AND year='{year}'"
        out.append(example(nl, sql))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_examples: list[dict] = []

    generators = [
        (gen_single_lookup,       250),
        (gen_multi_col,            80),
        (gen_ranking,              80),
        (gen_trend,               100),
        (gen_comparison,          100),
        (gen_filter,               80),
        (gen_yoy_change,           60),
        (gen_aggregate,            60),
        (gen_multi_year_all,       50),
        (gen_filing_metadata,      80),
        (gen_ratio_lookup,         60),
        (gen_best_in_class,        50),
        (gen_eps_lookup,           60),
        (gen_shareholder_returns,  60),
        (gen_rd_lookup,            50),
    ]

    for fn, n in generators:
        batch = fn(n)
        all_examples.extend(batch)
        print(f"  {fn.__name__:<25}  {len(batch):>4} examples")

    random.shuffle(all_examples)
    n_eval  = int(len(all_examples) * EVAL_RATIO)
    n_train = len(all_examples) - n_eval

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(TRAIN_OUT, all_examples[:n_train]), (EVAL_OUT, all_examples[n_train:])]:
        with open(path, "a", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def count(p): return sum(1 for _ in open(p, encoding="utf-8"))
    print(f"\nGenerated: {n_train} train  +  {n_eval} eval")
    print(f"Total now: {count(TRAIN_OUT)} train  +  {count(EVAL_OUT)} eval")


if __name__ == "__main__":
    main()
