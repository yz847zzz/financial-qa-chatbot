"""
Build NL2SQL SFT dataset (900 examples) for the financial panel schema.

Sources:
  - 150 hand-crafted  (template expansion across tickers/years/patterns)
  - 450 GPT-4o distilled  (OpenAI API, set OPENAI_API_KEY)
  - 300 WikiSQL adapted  (finance-adjacent tables → mapped to our schema)

Output: data/nl2sql/
  train.jsonl  (750)
  eval.jsonl   (150)

Run:
    $env:OPENAI_API_KEY = "sk-..."
    python data_pipeline/scripts/build_nl2sql_dataset.py
"""

import json, os, random, re
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "nl2sql"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Schema context (system prompt for all examples) ───────────────────────────
SYSTEM_PROMPT = """You have access to a SQLite database with these tables:

panel (ticker TEXT, year TEXT, cash REAL, total_assets REAL, current_assets REAL,
       current_liabilities REAL, total_liabilities REAL, goodwill REAL, long_term_debt REAL,
       accounts_payable REAL, inventories REAL, deferred_tax REAL, retained_earnings REAL,
       other_assets REAL, total_revenue REAL, net_income REAL, operating_income REAL,
       interest_expense REAL, interest_income REAL, da REAL, cfo REAL, capex REAL,
       current_ratio REAL, debt_to_assets REAL, roa REAL, net_margin REAL)
  - ticker: stock symbol e.g. 'AAPL', 'MSFT'
  - year: fiscal year string e.g. 'FY2023'
  - all monetary columns are in USD

filing_metadata (ticker TEXT, company TEXT, filing_type TEXT, date TEXT, accession_number TEXT)
  - filing_type: '10-K' or '8-K'
  - date: 'YYYY-MM-DD'

Rules:
- Output ONLY a valid SQL SELECT. No explanation.
- Ticker symbols are uppercase.
- Year format is 'FY{YYYY}' e.g. 'FY2023'.
- Use IS NOT NULL to filter missing values."""


# ── Shared constants ──────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "BRK-B",
    "JNJ", "V", "UNH", "XOM", "JPM", "PG", "MA", "HD", "CVX", "ABBV",
    "KO", "PEP", "COST", "TMO", "ABT", "MRK", "AVGO", "ACN", "MCD",
    "NKE", "ADBE", "CRM", "AMD", "INTC", "IBM", "CSCO", "QCOM", "TXN",
    "NFLX", "DIS", "BA", "HON", "GE", "CAT", "MMM", "FDX", "UPS",
    "WMT", "TGT", "SBUX", "LOW", "AMGN", "GILD", "BIIB", "REGN", "VRTX",
]
YEARS = ["FY2020", "FY2021", "FY2022", "FY2023", "FY2024"]
# Each column maps to a list of NL synonyms — rnd() picks one at generation time.
# Having many synonyms means the model sees "profit", "earnings", "bottom line"
# all mapping to net_income, "sales"/"top line"/"turnover" all mapping to total_revenue, etc.
COLS = {
    "total_revenue":      ["revenue", "revenues", "total revenue", "net sales", "sales",
                           "top line", "turnover", "total sales", "gross sales"],
    "net_income":         ["net income", "profit", "profits", "net profit", "net earnings",
                           "earnings", "bottom line", "income", "after-tax profit",
                           "profit after tax", "take-home profit"],
    "operating_income":   ["operating income", "operating profit", "EBIT",
                           "income from operations", "op income", "operating earnings"],
    "cfo":                ["operating cash flow", "cash from operations", "CFO",
                           "cash generated from operations", "cash from operating activities"],
    "capex":              ["capital expenditures", "capex", "CapEx",
                           "capital spending", "PP&E purchases"],
    "cash":               ["cash", "cash balance", "cash on hand", "cash and cash equivalents",
                           "liquidity", "cash holdings", "available cash", "cash position"],
    "total_assets":       ["total assets", "assets", "asset base", "book assets"],
    "current_assets":     ["current assets", "short-term assets", "liquid assets"],
    "current_liabilities":["current liabilities", "short-term liabilities",
                           "short-term obligations"],
    "total_liabilities":  ["total liabilities", "liabilities", "total obligations"],
    "goodwill":           ["goodwill", "acquisition goodwill"],
    "long_term_debt":     ["long-term debt", "long term debt", "LTD",
                           "long-term borrowings", "debt", "borrowings"],
    "accounts_payable":   ["accounts payable", "AP", "payables",
                           "trade payables", "vendor payables"],
    "inventories":        ["inventories", "inventory", "stock", "goods on hand"],
    "retained_earnings":  ["retained earnings", "accumulated earnings", "retained profits"],
    "interest_expense":   ["interest expense", "interest cost", "borrowing cost",
                           "finance cost", "cost of debt"],
    "da":                 ["depreciation and amortization", "D&A", "depreciation",
                           "non-cash charges"],
    "current_ratio":      ["current ratio", "liquidity ratio", "working capital ratio"],
    "debt_to_assets":     ["debt-to-assets ratio", "debt to assets", "debt ratio",
                           "leverage ratio"],
    "roa":                ["return on assets", "ROA", "asset returns", "asset productivity"],
    "net_margin":         ["net profit margin", "net margin", "profit margin",
                           "margin", "return on sales"],
}
COMPANY = {
    "AAPL":"Apple", "MSFT":"Microsoft", "AMZN":"Amazon", "GOOGL":"Alphabet",
    "NVDA":"NVIDIA", "META":"Meta", "TSLA":"Tesla", "JNJ":"Johnson & Johnson",
    "V":"Visa", "UNH":"UnitedHealth", "XOM":"ExxonMobil", "KO":"Coca-Cola",
    "PEP":"PepsiCo", "COST":"Costco", "MCD":"McDonald's", "NKE":"Nike",
    "NFLX":"Netflix", "DIS":"Disney", "BA":"Boeing", "WMT":"Walmart",
    "TGT":"Target", "SBUX":"Starbucks", "IBM":"IBM", "CSCO":"Cisco",
    "INTC":"Intel", "AMD":"AMD", "ADBE":"Adobe", "CRM":"Salesforce",
    "QCOM":"Qualcomm", "TXN":"Texas Instruments", "GE":"GE", "CAT":"Caterpillar",
    "FDX":"FedEx", "UPS":"UPS", "HON":"Honeywell", "MMM":"3M",
    "ABT":"Abbott", "MRK":"Merck", "AMGN":"Amgen", "GILD":"Gilead",
    "BIIB":"Biogen", "REGN":"Regeneron", "VRTX":"Vertex", "TMO":"Thermo Fisher",
    "HD":"Home Depot", "LOW":"Lowe's", "CVX":"Chevron", "XOM":"ExxonMobil",
}

def name(t): return COMPANY.get(t, t)
def rnd(lst): return random.choice(lst)
def pair(tickers): return random.sample(tickers, 2)
def col_nl(col): return rnd(COLS[col])   # random synonym for a column name


def make_example(question: str, sql: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": sql.strip()},
    ]}


# ── 1. Hand-crafted (150) ─────────────────────────────────────────────────────
def generate_handcrafted() -> list[dict]:
    examples = []

    def add(q, sql): examples.append(make_example(q, sql))

    # Pattern 1: Single metric lookup (30)
    for _ in range(30):
        t, y, col = rnd(TICKERS), rnd(YEARS), rnd(list(COLS))
        add(
            f"What was {name(t)}'s {col_nl(col)} in {y}?",
            f"SELECT {col} FROM panel WHERE ticker='{t}' AND year='{y}'",
        )

    # Pattern 2: Multi-ticker compare (20)
    for _ in range(20):
        t1, t2 = pair(TICKERS)
        y, col = rnd(YEARS), rnd(list(COLS))
        add(
            f"Compare {name(t1)} and {name(t2)} {col_nl(col)} in {y}.",
            f"SELECT ticker, {col} FROM panel WHERE ticker IN ('{t1}','{t2}') AND year='{y}' ORDER BY {col} DESC",
        )

    # Pattern 3: Ranking / top N (20)
    for _ in range(20):
        y, col = rnd(YEARS), rnd(["net_income","total_assets","roa","cfo","operating_income"])
        n = rnd([3, 5, 10])
        add(
            f"Which {n} companies had the highest {col_nl(col)} in {y}?",
            f"SELECT ticker, {col} FROM panel WHERE year='{y}' AND {col} IS NOT NULL ORDER BY {col} DESC LIMIT {n}",
        )

    # Pattern 4: Trend over years (20)
    for _ in range(20):
        t, col = rnd(TICKERS), rnd(list(COLS))
        add(
            f"Show {name(t)} {col_nl(col)} from FY2020 to FY2023.",
            f"SELECT year, {col} FROM panel WHERE ticker='{t}' AND year BETWEEN 'FY2020' AND 'FY2023' ORDER BY year",
        )

    # Pattern 5: Filter by threshold (20)
    for _ in range(20):
        y = rnd(YEARS)
        col, op, val = rnd([
            ("current_ratio", ">", 2.0),
            ("debt_to_assets", ">", 0.5),
            ("roa", ">", 0.1),
            ("net_margin", ">", 0.15),
            ("long_term_debt", "<", 5e10),
        ])
        label = col_nl(col)
        add(
            f"Which companies had {label} {op} {val} in {y}?",
            f"SELECT ticker, {col} FROM panel WHERE year='{y}' AND {col} {op} {val} ORDER BY {col} DESC",
        )

    # Pattern 6: Aggregation across tickers (15)
    for _ in range(15):
        y, col = rnd(YEARS), rnd(["net_income","total_assets","roa","net_margin"])
        agg = rnd(["AVG", "MAX", "MIN"])
        add(
            f"What was the {agg.lower()} {col_nl(col)} across all companies in {y}?",
            f"SELECT {agg}({col}) FROM panel WHERE year='{y}' AND {col} IS NOT NULL",
        )

    # Pattern 7: YoY change (15)
    for _ in range(15):
        t, col = rnd(TICKERS), rnd(["net_income","total_assets","cfo","operating_income"])
        y1, y2 = "FY2022", "FY2023"
        add(
            f"How did {name(t)}'s {col_nl(col)} change from {y1} to {y2}?",
            f"""SELECT a.year, a.{col}, b.{col} AS prev_year, (a.{col} - b.{col}) AS change
FROM panel a JOIN panel b ON a.ticker=b.ticker
WHERE a.ticker='{t}' AND a.year='{y2}' AND b.year='{y1}'""",
        )

    # Pattern 8: Filing metadata (10)
    for _ in range(10):
        t = rnd(TICKERS)
        add(
            f"When did {name(t)} file their most recent 10-K?",
            f"SELECT date FROM filing_metadata WHERE ticker='{t}' AND filing_type='10-K' ORDER BY date DESC LIMIT 1",
        )

    # Pattern 9: Multi-metric (10)
    for _ in range(10):
        t, y = rnd(TICKERS), rnd(YEARS)
        c1, c2 = rnd(["net_income","operating_income","cfo"]), rnd(["total_assets","long_term_debt","current_ratio"])
        add(
            f"Get {name(t)} {col_nl(c1)} and {col_nl(c2)} for {y}.",
            f"SELECT {c1}, {c2} FROM panel WHERE ticker='{t}' AND year='{y}'",
        )

    return examples


# ── 2. GPT-4o distilled (450) ─────────────────────────────────────────────────
def generate_distilled(n: int = 450) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  OPENAI_API_KEY not set — skipping distilled examples.")
        return []

    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed — skipping distilled examples.")
        return []

    client = OpenAI(api_key=api_key)
    examples = []
    batch_size = 10
    n_batches = n // batch_size

    patterns = [
        "single metric lookup for one ticker and year",
        "compare two tickers on the same metric",
        "rank top N companies by a metric",
        "trend of a metric for one ticker across multiple years",
        "filter companies by a threshold on a ratio",
        "aggregate (AVG/MAX/MIN) a metric across all companies for a year",
        "year-over-year change using a self-join",
        "filing metadata query (most recent 10-K date)",
        "multi-metric retrieval for one ticker-year",
    ]

    for i in range(n_batches):
        pattern = patterns[i % len(patterns)]
        prompt = f"""Generate {batch_size} diverse NL→SQL pairs for this financial database schema:

{SYSTEM_PROMPT}

Focus on the pattern: {pattern}
Use varied tickers from: {', '.join(random.sample(TICKERS, 10))}
Use varied years from: {', '.join(YEARS)}

Return a JSON array of {batch_size} objects, each with keys "question" and "sql".
Only return the JSON array, no other text."""

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            content = resp.choices[0].message.content
            # extract JSON array from response (may be wrapped in markdown)
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                continue
            items = json.loads(match.group())
            for item in items:
                if "question" in item and "sql" in item:
                    examples.append(make_example(item["question"], item["sql"]))
        except Exception as e:
            print(f"  Batch {i+1}/{n_batches} failed: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Distilled: {len(examples)}/{n} examples")

    print(f"  Distilled total: {len(examples)} examples")
    return examples


# WikiSQL removed: the original NL questions (tournaments, tax records, etc.) were
# semantically disconnected from the rewritten SQL, producing incoherent training
# pairs that teach the model nothing useful about financial language.


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating hand-crafted examples...")
    handcrafted = generate_handcrafted()
    print(f"  Hand-crafted: {len(handcrafted)}")

    print("Generating distilled examples (GPT-4o)...")
    distilled = generate_distilled(450)

    all_examples = handcrafted + distilled
    random.shuffle(all_examples)

    # 85% train / 15% eval
    split = int(len(all_examples) * 0.85)
    train = all_examples[:split]
    eval_ = all_examples[split:]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train, OUT_DIR / "train.jsonl")
    save_jsonl(eval_,  OUT_DIR / "eval.jsonl")

    meta = {"total": len(all_examples), "train": len(train), "eval": len(eval_),
            "handcrafted": len(handcrafted), "distilled": len(distilled)}
    with open(OUT_DIR / "sources.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Saved to {OUT_DIR}")
    print(f"  train.jsonl : {len(train)}")
    print(f"  eval.jsonl  : {len(eval_)}")
    print(f"  Sources     : {meta}")

    # Sanity check
    sample = random.choice(train)
    print("\nSample example:")
    print(f"  Q: {sample['messages'][1]['content']}")
    print(f"  A: {sample['messages'][2]['content']}")


if __name__ == "__main__":
    main()
