#!/usr/bin/env python3
"""
LLM-distilled NL2SQL dataset generator using OpenAI GPT-4o.

Generates natural-language-varied examples for query patterns that
template generation can't cover naturally:
  - Complex analytical questions
  - Multi-metric comparisons (with UNION)
  - Conditional aggregates
  - Ambiguous phrasing the model must resolve correctly

Usage:
    export OPENAI_API_KEY=sk-...
    python finetune/data_prep/distill_nl2sql_openai.py
    python finetune/data_prep/distill_nl2sql_openai.py --n 500 --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_OUT = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_OUT  = ROOT / "data" / "nl2sql" / "eval.jsonl"
CACHE_DIR = ROOT / "data" / "nl2sql" / "distill_cache"
EVAL_RATIO = 0.15

SCHEMA_CONTEXT = """SQLite schema:

financials (ticker TEXT, period TEXT, statement TEXT, metric TEXT, value REAL, unit TEXT)
  ticker:  e.g. "AAPL", "MSFT"
  period:  "F2023" for annual, "2023-09" for quarterly (YYYY-MM)
  metric:  e.g. "Total Revenue", "Net Income", "Total Assets", "Return On Assets"
  value:   numeric

filing_metadata (ticker TEXT, company TEXT, filing_type TEXT, date TEXT)
  filing_type: "10-K", "10-Q", "8-K"
  date: "YYYY-MM-DD"

RULES:
- Period format: annual = F{year} (e.g. F2023), quarterly = YYYY-MM (e.g. 2023-09)
- Always add: AND value IS NOT NULL   when filtering/aggregating
- metric names are exact strings from the DB — use LIKE '%keyword%' for fuzzy match
- Output ONLY the SQL, nothing else
"""

SYSTEM_PROMPT_GENERATOR = f"""You are a financial SQL expert. Generate diverse, realistic NL→SQL training pairs for a financial QA system.

{SCHEMA_CONTEXT}

For each batch I give you a PATTERN. Generate exactly the number of examples requested.
Output a JSON array of objects, each with:
  "nl": <natural language question, realistic financial analyst phrasing>
  "sql": <valid SQLite SELECT statement>

Requirements:
- NL must sound like something a real financial analyst or investor would ask
- SQL must be valid SQLite, use the correct schema above
- Use a variety of tickers from S&P 500 (AAPL, MSFT, AMZN, GOOGL, NVDA, TSLA, JPM, WMT, etc.)
- Use years F2020–F2024 for annual, 2022-03 to 2023-12 for quarterly
- No explanation, no markdown — pure JSON array
"""

PATTERNS = [
    {
        "name": "analytical_single",
        "count": 60,
        "description": """
Generate examples where the NL question is phrased analytically/conversationally,
not like a template. Examples:
  "Has Apple been growing its free cash flow?"  → SELECT period, value FROM financials WHERE ticker='AAPL' AND metric='Free Cash Flow' AND period LIKE 'F%' ORDER BY period
  "Is Microsoft profitable compared to Google in 2023?" → SELECT ticker, value FROM financials WHERE ticker IN ('MSFT','GOOGL') AND metric='Net Income' AND period='F2023'
Use varied natural phrasing. Mix tickers and years.
""",
    },
    {
        "name": "multi_metric_union",
        "count": 40,
        "description": """
Generate examples where the user asks for TWO different metrics for the same company/period.
Since the schema is long-format, you need UNION ALL:
  "Show Apple's revenue and net income for 2023"
  → SELECT metric, value FROM financials WHERE ticker='AAPL' AND metric IN ('Total Revenue','Net Income') AND period='F2023'
Use IN (...) not UNION when possible. For truly different queries use UNION ALL.
""",
    },
    {
        "name": "conditional_ranking",
        "count": 40,
        "description": """
Generate examples with combined filters and ranking:
  "Which 5 companies with positive net income had the highest ROA in 2022?"
  "Find tech companies with gross margin above 50% in 2023, ranked by net income"
Approximate sector using IN (list of tickers). Use subqueries if needed.
""",
    },
    {
        "name": "growth_rate",
        "count": 40,
        "description": """
Generate examples asking about growth or change between years:
  "What was Amazon's revenue growth from 2021 to 2022?"
  → SELECT period, value FROM financials WHERE ticker='AMZN' AND metric='Total Revenue' AND period IN ('F2021','F2022') ORDER BY period
The SQL should fetch the two years' values so the application can compute the growth rate.
""",
    },
    {
        "name": "filing_complex",
        "count": 30,
        "description": """
Generate varied filing_metadata queries:
  - How many 10-K filings does Apple have?
  - Which companies filed 8-Ks in 2023?
  - List all companies with a 10-Q filed after 2023-06-01
  - What was the most recent filing date for TSLA?
""",
    },
    {
        "name": "ambiguous_resolved",
        "count": 40,
        "description": """
Generate examples where the NL is slightly ambiguous but the SQL correctly resolves it:
  "How much cash does Apple have?" → total cash (latest annual)
  "Is Tesla debt-heavy?" → debt-to-assets ratio, latest year
  "What's Google's bottom line?" → Net Income
The SQL should make a reasonable, specific interpretation.
""",
    },
    {
        "name": "quarterly_deep",
        "count": 40,
        "description": """
Generate varied quarterly queries:
  - "What was Apple's Q3 2023 revenue?"  → period='2023-09'
  - "Show Microsoft's quarterly earnings for all of 2022"  → period LIKE '2022-%'
  - "Which quarter in 2023 had the highest revenue for NVDA?"
Quarter to period mapping: Q1=03, Q2=06, Q3=09, Q4=12
""",
    },
    {
        "name": "negative_positive_filter",
        "count": 30,
        "description": """
Generate examples filtering on positive/negative values or relative size:
  "Which companies had a net loss in 2022?" → value < 0
  "Which companies doubled their revenue between 2020 and 2023?" (approximate with two selects or subquery)
  "Find companies with negative free cash flow in 2023"
""",
    },
]


def validate_sql(sql: str) -> bool:
    """Basic structural validation."""
    s = sql.strip().upper()
    return (s.startswith("SELECT") and "FROM" in s
            and "DROP" not in s and "INSERT" not in s
            and "UPDATE" not in s and "DELETE" not in s)


def call_openai(client, pattern: dict, model: str) -> list[dict]:
    prompt = (
        f"PATTERN: {pattern['name']} — generate {pattern['count']} examples\n\n"
        f"{pattern['description'].strip()}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_GENERATOR},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.9,
                max_tokens=8000,
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON array — handle markdown fences
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                print(f"  [warn] no JSON array found, attempt {attempt+1}")
                continue
            pairs = json.loads(m.group())
            valid = [p for p in pairs
                     if isinstance(p, dict)
                     and "nl" in p and "sql" in p
                     and validate_sql(p["sql"])]
            print(f"  {pattern['name']}: {len(valid)}/{len(pairs)} valid")
            return valid
        except Exception as e:
            print(f"  [error] {e}, retrying in 5s...")
            time.sleep(5)
    return []


def to_message_format(pair: dict) -> dict:
    from generate_nl2sql_templates import SYSTEM_PROMPT
    sql = pair["sql"].strip().rstrip(";") + ";"
    return {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": pair["nl"]},
        {"role": "assistant", "content": sql},
    ]}


def main(model: str, target_n: int) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []

    for pattern in PATTERNS:
        cache_file = CACHE_DIR / f"{pattern['name']}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                pairs = json.load(f)
            print(f"  {pattern['name']}: loaded {len(pairs)} from cache")
        else:
            print(f"  Generating {pattern['name']}...")
            pairs = call_openai(client, pattern, model)
            with open(cache_file, "w") as f:
                json.dump(pairs, f, indent=2)
            time.sleep(1)  # rate limit

        all_pairs.extend(pairs)
        if target_n and len(all_pairs) >= target_n:
            all_pairs = all_pairs[:target_n]
            break

    import random; random.shuffle(all_pairs)
    n_eval  = int(len(all_pairs) * EVAL_RATIO)
    n_train = len(all_pairs) - n_eval

    new_train = [to_message_format(p) for p in all_pairs[:n_train]]
    new_eval  = [to_message_format(p) for p in all_pairs[n_train:]]

    for path, data, label in [(TRAIN_OUT, new_train, "train"), (EVAL_OUT, new_eval, "eval")]:
        with open(path, "a", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def count_lines(p):
        return sum(1 for _ in open(p, encoding="utf-8"))

    print(f"\nDistilled : {n_train} train  +  {n_eval} eval")
    print(f"Total now : {count_lines(TRAIN_OUT)} train  +  {count_lines(EVAL_OUT)} eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (gpt-4o / gpt-4o-mini)")
    parser.add_argument("--n", type=int, default=0, help="Max examples to generate (0=all patterns)")
    args = parser.parse_args()
    main(args.model, args.n)
