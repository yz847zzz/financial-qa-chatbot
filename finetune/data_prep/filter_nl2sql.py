#!/usr/bin/env python3
"""
Post-build quality filter for NL2SQL training data.

Removes examples where the NL question is semantically disconnected from
financial domain (WikiSQL artifacts, generic DB questions, etc.).

Keeps examples where NL contains at least one:
  - financial term (profit, revenue, assets, ...)
  - company reference (ticker or company name)
  - temporal reference (FY2023, fiscal, annual, ...)
  - SQL-action word (show, find, list, compare, ...)

Run after build_nl2sql_dataset.py + generate_nl2sql_templates.py:
    python finetune/data_prep/filter_nl2sql.py
"""

import json
import re
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
TRAIN_OUT = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_OUT  = ROOT / "data" / "nl2sql" / "eval.jsonl"

# ── Financial domain signals ───────────────────────────────────────────────────

FINANCIAL_TERMS = {
    # income statement
    "revenue", "revenues", "sales", "turnover", "top line",
    "profit", "profits", "earnings", "income", "bottom line",
    "operating income", "ebit", "net income", "net profit",
    "gross profit", "gross margin", "margin", "profitability",
    # balance sheet
    "assets", "liabilities", "equity", "goodwill",
    "debt", "borrowings", "long-term debt", "inventory", "inventories",
    "cash", "receivable", "payable", "retained earnings",
    # cash flow
    "cash flow", "cfo", "capex", "capital expenditure",
    "depreciation", "amortization",
    # ratios
    "roa", "roe", "return on assets", "return on equity",
    "current ratio", "debt-to-assets", "leverage", "liquidity",
    "net margin", "profit margin", "return on sales",
    # filing
    "10-k", "10-q", "8-k", "filing", "annual report",
    # temporal
    "fiscal", "fy20", "annual", "quarterly", "quarter",
    # action synonyms
    "revenue growth", "profit growth", "earnings growth",
    "interest expense", "interest income", "dividend",
}

# Tickers in our universe
TICKERS = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "DIS", "BAC", "XOM",
    "PFE", "CVX", "KO", "ABBV", "AVGO", "COST", "MRK", "TMO", "CSCO",
    "NKE", "ORCL", "CRM", "LLY", "AMD", "INTC", "QCOM", "NFLX", "MCD",
    "LOW", "CAT", "MMM", "GILD", "REGN", "BIIB", "TXN", "ACN", "ABT",
    "PEP", "IBM", "BA", "HON", "FDX", "UPS", "GE", "T", "DE",
}

# Company names (partial, lowercase)
COMPANY_NAMES = {
    "apple", "microsoft", "amazon", "alphabet", "google", "meta", "nvidia",
    "tesla", "jpmorgan", "visa", "johnson", "walmart", "procter", "unitedhealth",
    "home depot", "disney", "bank of america", "exxon", "pfizer", "chevron",
    "coca-cola", "abbvie", "broadcom", "costco", "merck", "thermo fisher",
    "cisco", "nike", "oracle", "salesforce", "eli lilly", "intel", "qualcomm",
    "netflix", "mcdonald", "lowe", "caterpillar", "3m", "gilead", "regeneron",
    "biogen", "texas instruments", "accenture", "abbott", "pepsico",
    "ibm", "boeing", "honeywell", "fedex", "ups", "general electric",
}

# Definite non-financial noise patterns
NOISE_PATTERNS = [
    re.compile(r'\b(tournament|season|match|game|player|team|sport|podium)\b', re.I),
    re.compile(r'\b(song|album|artist|band|music|genre)\b', re.I),
    re.compile(r'\b(movie|film|actor|director|award|oscar)\b', re.I),
    re.compile(r'\b(country|capital|population|area|continent)\b', re.I),
    re.compile(r'\b(stamp duty|tax revenue|national insurance)\b', re.I),
    re.compile(r'\b(win.loss|margin of (victory|defeat))\b', re.I),
    re.compile(r'\bfield \d+\b', re.I),           # "Field 103"
    re.compile(r'\bokajima|jeriome\b', re.I),     # baseball player names
]


def is_financial(nl: str) -> bool:
    nl_lower = nl.lower()

    # Hard reject: definite noise keywords
    for pat in NOISE_PATTERNS:
        if pat.search(nl_lower):
            return False

    # Accept: contains a known ticker
    words = set(re.findall(r'\b[A-Z]{2,5}\b', nl))
    if words & TICKERS:
        return True

    # Accept: contains company name
    if any(name in nl_lower for name in COMPANY_NAMES):
        return True

    # Accept: contains financial terminology
    if any(term in nl_lower for term in FINANCIAL_TERMS):
        return True

    # Accept: FY year reference anywhere
    if re.search(r'\bFY20\d{2}\b', nl):
        return True

    return False


def filter_file(path: Path) -> tuple[int, int]:
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]

    clean = [e for e in examples if is_financial(e["messages"][1]["content"])]

    with open(path, "w", encoding="utf-8") as f:
        for e in clean:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return len(examples), len(clean)


def main() -> None:
    for name, path in [("train", TRAIN_OUT), ("eval", EVAL_OUT)]:
        before, after = filter_file(path)
        removed = before - after
        print(f"{name}: {before} -> {after}  (removed {removed} noise examples)")

    print()
    # Spot-check: show 5 random kept examples
    import random; random.seed(3)
    with open(TRAIN_OUT, encoding="utf-8") as f:
        kept = [json.loads(l) for l in f]
    print("Sample kept examples:")
    for e in random.sample(kept, min(8, len(kept))):
        nl  = e["messages"][1]["content"]
        sql = e["messages"][2]["content"]
        print(f"  NL:  {nl[:80]}")
        print(f"  SQL: {sql[:80]}")
        print()


if __name__ == "__main__":
    main()
