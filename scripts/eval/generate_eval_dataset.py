#!/usr/bin/env python3
"""
generate_eval_dataset.py -- Extend the original 52-case eval set to 200+ cases.

Strategy:
  1. Keep the original 52 hand-verified TEST_CASES as-is
  2. For new Type1 cases, run ACTUAL SQL queries against the DB to get
     the exact values the system would return (not scanning metric names)
  3. Add more Type2 (RAG) and Type3 (chat) cases using the same style
  4. Stick to tickers the system already knows well

Output: eval_testcases_expanded.json
"""

import json
import random
import sqlite3
import sys
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "financials.db"
OUT_PATH = ROOT / "eval" / "testdata" / "testcases.json"

sys.path.insert(0, str(ROOT))
from eval_system import TEST_CASES

conn = sqlite3.connect(str(DB_PATH))


def sql_value(query: str) -> float | None:
    """Run a SQL query and return the first numeric result, or None."""
    try:
        row = conn.execute(query).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return None


# ── Start with the original 52 cases ────────────────────────────────────────

cases = list(TEST_CASES)  # deep-ish copy (dicts are mutable but we won't modify originals)
next_id = max(c["id"] for c in cases) + 1

print(f"Starting with {len(cases)} original test cases (ids 1-{next_id - 1})")


def add(category, question, expected_intent, expected_value=None,
        expected_keywords=None, expected_sub_intents=None,
        acceptable_values=None):
    global next_id
    entry = {
        "id": next_id,
        "category": category,
        "question": question,
        "expected_intent": expected_intent,
        "expected_value": expected_value,
        "expected_keywords": expected_keywords or [],
    }
    if expected_sub_intents:
        entry["expected_sub_intents"] = expected_sub_intents
    if acceptable_values:
        entry["acceptable_values"] = acceptable_values
    cases.append(entry)
    next_id += 1


# ── Core tickers the system handles well ────────────────────────────────────
# These are the tickers from the original 52 cases + a few proven extras

CORE_TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOG": "Google",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "ADBE": "Adobe",
    "AMZN": "Amazon",
}

# NL2SQL queries the panel table — use the same source for ground-truth values.
# panel is the cleaned wide table; financials has noise and metric-name ambiguity.
PANEL_COLUMNS = {
    "revenue":          "total_revenue",
    "net_income":       "net_income",
    "operating_income": "operating_income",
    "total_assets":     "total_assets",
    "total_liabilities":"total_liabilities",
    "goodwill":         "goodwill",
    "cash":             "cash",
}


def query_metric(ticker: str, period: str, metric_key: str) -> tuple | None:
    """Query panel table for ground-truth value (same source as NL2SQL).
    Returns (value, [value]) or None if not available."""
    col = PANEL_COLUMNS.get(metric_key)
    if not col:
        return None
    v = sql_value(
        f"SELECT {col} FROM panel WHERE ticker='{ticker}' AND year='{period}'"
    )
    if v is None or abs(v) > 1e13:   # sanity cap at $10T
        return None
    return (v, [v])   # panel has one canonical value; no acceptable_values needed


# ── Get available periods per ticker ────────────────────────────────────────

PERIODS = {}
for ticker in CORE_TICKERS:
    rows = conn.execute(
        "SELECT DISTINCT year FROM panel WHERE ticker=? AND year LIKE 'FY%' ORDER BY year",
        (ticker,),
    ).fetchall()
    PERIODS[ticker] = [r[0] for r in rows]

# ── Existing question ids (avoid duplication) ───────────────────────────────
existing_qs = {c["question"] for c in cases}

# ── Type1: Single metric, various tickers x periods ────────────────────────

TYPE1_TEMPLATES = [
    ("revenue", "What was {name}'s total revenue in {fy}?", ["revenue"]),
    ("revenue", "How much revenue did {name} generate in {fy}?", ["revenue"]),
    ("net_income", "What was {name}'s net income in {fy}?", ["net income"]),
    ("net_income", "How much profit did {name} earn in {fy}?", ["profit", "income"]),
    ("operating_income", "What was {name}'s operating income in {fy}?", ["operating"]),
    ("total_assets", "What were {name}'s total assets in {fy}?", ["total assets"]),
    ("total_liabilities", "What were {name}'s total liabilities in {fy}?", ["liabilities"]),
    ("goodwill", "What was {name}'s goodwill in {fy}?", ["goodwill"]),
    ("cash", "How much cash did {name} have in {fy}?", ["cash"]),
]

n_before = len(cases)
for metric_key, template, extra_kws in TYPE1_TEMPLATES:
    for ticker, name in CORE_TICKERS.items():
        # Last 3 fiscal years for better temporal coverage
        for fy in PERIODS.get(ticker, [])[-3:]:
            question = template.format(name=name, fy=fy)
            if question in existing_qs:
                continue

            result = query_metric(ticker, fy, metric_key)
            if result is None:
                continue

            value, all_vals = result
            # Build keywords
            kws = [ticker.lower(), name.lower()]
            if abs(value) >= 1e9:
                kws.append(str(int(round(value / 1e9))))
            elif abs(value) >= 1e6:
                kws.append(str(int(round(value / 1e6))))
            kws.extend(extra_kws)

            add("Type1", question, "Type1", expected_value=value,
                expected_keywords=kws,
                acceptable_values=all_vals if len(set(all_vals)) > 1 else None)
            existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type1 single-metric cases.")

# ── Type1: Comparisons ─────────────────────────────────────────────────────

COMPARE_TEMPLATES = [
    ("revenue", "Compare {n1} and {n2} revenue in {fy}.", ["revenue"]),
    ("net_income", "Compare {n1} and {n2} net income in {fy}.", ["net income"]),
    ("revenue", "Which had higher revenue in {fy}, {n1} or {n2}?", ["revenue", "higher"]),
    ("total_assets", "Compare {n1} and {n2} total assets in {fy}.", ["total assets"]),
]

n_before = len(cases)
ticker_list = list(CORE_TICKERS.keys())
for metric_key, template, extra_kws in COMPARE_TEMPLATES:
    for i in range(len(ticker_list)):
        for j in range(i + 1, len(ticker_list)):
            t1, t2 = ticker_list[i], ticker_list[j]
            n1, n2 = CORE_TICKERS[t1], CORE_TICKERS[t2]
            # Find a common period
            common = sorted(set(PERIODS.get(t1, [])) & set(PERIODS.get(t2, [])), reverse=True)
            if not common:
                continue
            fy = common[0]
            question = template.format(n1=n1, n2=n2, fy=fy)
            if question in existing_qs:
                continue
            r1 = query_metric(t1, fy, metric_key)
            r2 = query_metric(t2, fy, metric_key)
            if r1 is None or r2 is None:
                continue
            kws = [n1.lower(), n2.lower()] + extra_kws
            add("Type1-compare", question, "Type1", expected_keywords=kws)
            existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type1 comparison cases.")

# ── Type1: YoY change ──────────────────────────────────────────────────────

YOY_TEMPLATES = [
    ("revenue", "How did {name}'s revenue change from {fy1} to {fy2}?", ["revenue", "change"]),
    ("net_income", "How did {name}'s net income change from {fy1} to {fy2}?", ["net income", "change"]),
]

n_before = len(cases)
for metric_key, template, extra_kws in YOY_TEMPLATES:
    for ticker, name in CORE_TICKERS.items():
        periods = sorted(PERIODS.get(ticker, []))
        for i in range(len(periods) - 1):
            fy1, fy2 = periods[i], periods[i + 1]
            question = template.format(name=name, fy1=fy1, fy2=fy2)
            if question in existing_qs:
                continue
            r1 = query_metric(ticker, fy1, metric_key)
            r2 = query_metric(ticker, fy2, metric_key)
            if r1 is None or r2 is None:
                continue
            kws = [name.lower()] + extra_kws
            add("Type1-compare", question, "Type1", expected_keywords=kws)
            existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type1 YoY cases.")

# ── Type1: Compound (two metrics, one company) ─────────────────────────────

COMPOUND_TEMPLATES = [
    (["revenue", "net_income"], "What were {name}'s revenue and net income in {fy}?",
     ["revenue", "net income"]),
    (["total_assets", "total_liabilities"], "What were {name}'s total assets and liabilities in {fy}?",
     ["assets", "liabilities"]),
]

n_before = len(cases)
for metric_pair, template, extra_kws in COMPOUND_TEMPLATES:
    for ticker, name in CORE_TICKERS.items():
        for fy in PERIODS.get(ticker, [])[-2:]:  # last 2 periods
            question = template.format(name=name, fy=fy)
            if question in existing_qs:
                continue
            results = [query_metric(ticker, fy, mk) for mk in metric_pair]
            if any(r is None for r in results):
                continue
            kws = [name.lower()] + extra_kws
            add("Type1-compare", question, "Type1", expected_keywords=kws)
            existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type1 compound cases.")

# ── Type2: Qualitative questions ────────────────────────────────────────────

TYPE2_TEMPLATES = [
    ("How did {name} describe its competitive landscape in its most recent 10-K?",
     ["competition", "competitive", "market"]),
    ("What risks did {name} identify related to its business in recent filings?",
     ["risk", "uncertainty"]),
    ("How did {name} describe its growth strategy in its annual report?",
     ["growth", "strategy"]),
    ("What did {name} say about supply chain risks in its 10-K?",
     ["supply", "chain", "risk"]),
    ("How did {name} discuss regulatory risks in its most recent filing?",
     ["regulatory", "regulation", "compliance"]),
    ("What did {name} say about its capital allocation strategy?",
     ["capital", "allocation"]),
    ("What environmental or sustainability risks did {name} disclose?",
     ["environmental", "sustainability", "climate"]),
    ("How did {name} describe its technology investments in its 10-K?",
     ["technology", "innovation"]),
    ("What did {name} say about its workforce and talent strategy?",
     ["workforce", "talent", "employee"]),
    ("How did {name} discuss macroeconomic risks in its annual report?",
     ["macroeconomic", "economic", "inflation"]),
    ("What cybersecurity risks did {name} identify in its filings?",
     ["cybersecurity", "security", "data"]),
    ("What did {name} say about international operations in its 10-K?",
     ["international", "global"]),
    ("How did {name} discuss its product pipeline in recent filings?",
     ["product", "pipeline"]),
    ("What did {name} disclose about litigation risks?",
     ["litigation", "legal"]),
    ("How did {name} describe its pricing strategy in its 10-K?",
     ["pricing", "strategy"]),
    ("What did {name} say about its R&D investments?",
     ["research", "development", "r&d"]),
    ("How did {name} describe its market position in its annual filing?",
     ["market", "position"]),
    ("What did {name} say about customer concentration risks?",
     ["customer", "concentration", "risk"]),
]

# Tickers with rich filing text in ChromaDB
TYPE2_TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOG": "Google",
    "NVDA": "NVIDIA", "TSLA": "Tesla", "ADBE": "Adobe",
    "AMZN": "Amazon", "NFLX": "Netflix", "CRM": "Salesforce",
    "BA": "Boeing", "JPM": "JPMorgan", "XOM": "Exxon Mobil",
    "DIS": "Disney", "JNJ": "Johnson & Johnson", "PFE": "Pfizer",
    "WMT": "Walmart", "COST": "Costco", "NKE": "Nike",
}

n_before = len(cases)
for template, topic_kws in TYPE2_TEMPLATES:
    # 6 tickers per template (up from 2) for better quantization discrimination
    selected = random.sample(list(TYPE2_TICKERS.items()), min(6, len(TYPE2_TICKERS)))
    for ticker, name in selected:
        question = template.format(name=name)
        if question in existing_qs:
            continue
        kws = [name.lower(), ticker.lower()] + topic_kws
        add("Type2", question, "Type2", expected_keywords=kws)
        existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type2 cases.")

# ── Type3: Chat / meta questions ────────────────────────────────────────────

TYPE3_EXTRAS = [
    ("What's the difference between revenue and net income?",
     ["revenue", "net income", "difference"]),
    ("Do you have quarterly data?",
     ["quarterly", "data"]),
    ("How do you analyze SEC filings?",
     ["analyze", "sec", "filing"]),
    ("Who built this system?",
     ["system", "chatbot"]),
    ("Can you help me with stock trading advice?",
     ["invest", "advice", "not"]),
    ("What is earnings per share?",
     ["earnings", "share"]),
    ("Good morning!",
     ["hello", "good", "help"]),
    ("What SEC filings do you have access to?",
     ["sec", "filing", "10-k"]),
    ("Can you compare two companies for me?",
     ["compare", "companies"]),
    ("How accurate are your financial numbers?",
     ["accurate", "data", "source"]),
    ("What types of questions can I ask?",
     ["type", "question"]),
    ("What time period does your data cover?",
     ["period", "data", "year"]),
    ("What is a 10-K filing?",
     ["10-k", "annual", "sec"]),
    ("How is net income different from operating income?",
     ["net income", "operating", "difference"]),
    ("Can you explain what EBITDA means?",
     ["ebitda", "earnings"]),
    ("What is the difference between assets and liabilities?",
     ["assets", "liabilities", "difference"]),
    ("Thanks for the help, goodbye!",
     []),
    ("Are you an AI?",
     ["ai", "assistant"]),
    # Additional financial concept questions
    ("What is free cash flow and why does it matter?",
     ["free cash flow", "cash"]),
    ("Can you explain gross margin vs operating margin?",
     ["gross margin", "operating margin"]),
    ("What does it mean when a company has negative equity?",
     ["equity", "negative", "liabilities"]),
    ("How do I read a balance sheet?",
     ["balance sheet", "assets", "liabilities"]),
    ("What is goodwill on a balance sheet?",
     ["goodwill", "intangible", "assets"]),
    ("What is the difference between diluted and basic EPS?",
     ["diluted", "basic", "eps"]),
    ("Can you help me understand capital expenditure?",
     ["capital expenditure", "capex", "investment"]),
    ("What does R&D expense tell us about a company?",
     ["research", "development", "r&d", "innovation"]),
    ("What is operating leverage?",
     ["operating", "leverage", "fixed"]),
    ("What does a high debt-to-equity ratio mean?",
     ["debt", "equity", "leverage", "ratio"]),
    ("How do companies use stock buybacks?",
     ["buyback", "repurchase", "shares"]),
    ("What is working capital?",
     ["working capital", "current assets", "current liabilities"]),
    ("Can you compare a company's performance year over year?",
     ["year over year", "growth", "compare"]),
    ("What is the difference between revenue and profit?",
     ["revenue", "profit", "difference"]),
    ("How do you calculate return on equity?",
     ["return on equity", "roe", "net income"]),
    ("What are the main sections of a 10-K filing?",
     ["10-k", "section", "risk", "financial"]),
    ("What is depreciation and amortization?",
     ["depreciation", "amortization", "da"]),
    ("Can you explain what accounts receivable means?",
     ["accounts receivable", "revenue", "customer"]),
    ("What is a debt covenant?",
     ["debt", "covenant", "agreement"]),
    ("What does inventory turnover mean for a retailer?",
     ["inventory", "turnover", "retail"]),
]

n_before = len(cases)
for question, kws in TYPE3_EXTRAS:
    if question in existing_qs:
        continue
    add("Type3", question, "Type3", expected_keywords=kws)
    existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type3 cases.")

# ── Type1+Type2: Mixed ──────────────────────────────────────────────────────

MIXED_TEMPLATES = [
    ("revenue", "What was {name}'s revenue in {fy} and how did they describe their growth strategy?",
     ["revenue", "growth", "strategy"]),
    ("net_income", "What was {name}'s net income in {fy} and what risks did they identify?",
     ["net income", "risk"]),
    ("revenue", "Report {name}'s {fy} revenue and discuss their competitive position.",
     ["revenue", "competitive"]),
]

n_before = len(cases)
for metric_key, template, extra_kws in MIXED_TEMPLATES:
    for ticker, name in list(CORE_TICKERS.items())[:4]:
        fy = PERIODS.get(ticker, [])[-1] if PERIODS.get(ticker) else None
        if not fy:
            continue
        question = template.format(name=name, fy=fy)
        if question in existing_qs:
            continue
        result = query_metric(ticker, fy, metric_key)
        if result is None:
            continue
        value, all_vals = result
        kws = [name.lower()] + extra_kws
        add("Type1+Type2", question, None, expected_value=value,
            expected_keywords=kws, expected_sub_intents=["Type1", "Type2"],
            acceptable_values=all_vals if len(set(all_vals)) > 1 else None)
        existing_qs.add(question)

print(f"Generated {len(cases) - n_before} new Type1+Type2 cases.")

# ── Summary ─────────────────────────────────────────────────────────────────

from collections import Counter
type_counts = Counter(c["category"] for c in cases)

print(f"\n{'='*60}")
print(f"TOTAL CASES: {len(cases)}")
print(f"{'='*60}")
print(f"  Original:  52 (ids 1-52)")
print(f"  New:       {len(cases) - 52} (ids 53+)")
print()
for cat, cnt in sorted(type_counts.items()):
    print(f"  {cat:<20} {cnt:>4}")

# Verify Type1 values
type1_with_val = sum(1 for c in cases if c["category"] == "Type1" and c.get("expected_value") is not None)
type1_total = sum(1 for c in cases if c["category"] == "Type1")
print(f"\n  Type1 with expected_value: {type1_with_val}/{type1_total}")

# Save
with open(OUT_PATH, "w") as f:
    json.dump(cases, f, indent=2)
print(f"\nSaved to: {OUT_PATH}")
