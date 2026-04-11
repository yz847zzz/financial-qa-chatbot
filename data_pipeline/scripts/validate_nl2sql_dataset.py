"""
Validate NL2SQL dataset quality by:
  1. SQL syntax check (sqlite3 parse)
  2. Execution against real DB (non-error, non-empty result)
  3. Schema validation (columns/tables referenced exist)
  4. LLM-as-judge on 50 random samples (GPT-4o, optional)
  5. Distribution report (pattern, ticker coverage)

Bad examples are removed and clean files are rewritten.

Run:
    python data_pipeline/scripts/validate_nl2sql_dataset.py
    python data_pipeline/scripts/validate_nl2sql_dataset.py --no-llm-judge
"""

import json, re, random, sqlite3, argparse, os
from pathlib import Path
from collections import Counter

DB_PATH  = Path(__file__).resolve().parents[2] / "data" / "financials.db"
NL2SQL_DIR = Path(__file__).resolve().parents[2] / "data" / "nl2sql"

VALID_TABLES  = {"panel", "filing_metadata", "financials"}
VALID_COLUMNS = {
    "panel": {
        "ticker","year","cash","total_assets","current_assets","current_liabilities",
        "total_liabilities","goodwill","long_term_debt","accounts_payable","inventories",
        "deferred_tax","retained_earnings","other_assets","net_income","interest_expense",
        "interest_income","da","operating_income","cfo","capex","current_ratio",
        "debt_to_assets","roa","net_margin",
    },
    "filing_metadata": {"ticker","company","filing_type","date","accession_number"},
    "financials": {"ticker","period","statement","metric","value","unit"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sql(example: dict) -> str:
    return example["messages"][2]["content"].strip()

def get_question(example: dict) -> str:
    return example["messages"][1]["content"].strip()

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_jsonl(data: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Check: Execution against real DB (covers syntax + schema + results) ───────

def check_execution(sql: str, conn: sqlite3.Connection) -> tuple[bool, str, int]:
    """Returns (ok, error_msg, row_count). Uses real DB — catches syntax + missing tables."""
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return True, "", len(rows)
    except sqlite3.Error as e:
        return False, str(e), 0


# ── Check 3: Schema validation ────────────────────────────────────────────────

def check_schema(sql: str) -> tuple[bool, str]:
    sql_up = sql.upper()

    # Extract table names used
    tables = set(re.findall(r'\bFROM\s+(\w+)', sql_up, re.I))
    tables |= set(re.findall(r'\bJOIN\s+(\w+)', sql_up, re.I))
    tables = {t.lower() for t in tables}

    bad_tables = tables - VALID_TABLES
    if bad_tables:
        return False, f"Unknown tables: {bad_tables}"

    # Extract column references (rough heuristic — catch obvious wrong names)
    for table in tables:
        valid_cols = VALID_COLUMNS.get(table, set())
        # Look for words that appear to be column names (not SQL keywords)
        SQL_KEYWORDS = {
            "select","from","where","and","or","not","in","is","null","order","by",
            "limit","join","on","group","having","as","distinct","between","like",
            "asc","desc","avg","sum","max","min","count","inner","left","right",
            "outer","union","all","case","when","then","else","end","insert","into",
            "values","update","set","delete","create","table","index","view",
        }
        tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', sql.lower())
        candidate_cols = {t for t in tokens if t not in SQL_KEYWORDS and not t.replace(".","").isdigit()}
        # Only flag if clearly a wrong column on a known table
        if valid_cols:
            bad = candidate_cols - valid_cols - VALID_TABLES - SQL_KEYWORDS - {"fy2019","fy2020","fy2021","fy2022","fy2023","fy2024","10","k","8"}
            if bad and len(bad) > 2:  # allow some slack for aliases
                return False, f"Possible unknown columns: {bad}"
    return True, ""


# ── Check 4: LLM-as-judge (GPT-4o, sample 50) ────────────────────────────────

def llm_judge(examples: list[dict], n: int = 50) -> float:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  OPENAI_API_KEY not set — skipping LLM judge.")
        return -1.0
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai not installed — skipping LLM judge.")
        return -1.0

    client = OpenAI(api_key=api_key)
    sample = random.sample(examples, min(n, len(examples)))
    scores = []

    for ex in sample:
        q, sql = get_question(ex), get_sql(ex)
        prompt = f"""Rate how well this SQL answers the natural language question on a scale of 1-5.
1 = completely wrong, 3 = partially correct, 5 = perfectly correct.

Question: {q}
SQL: {sql}

Reply with a single integer 1-5. Nothing else."""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=2,
            )
            score = int(resp.choices[0].message.content.strip())
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return -1.0
    avg = sum(scores) / len(scores)
    dist = Counter(scores)
    print(f"  LLM judge scores: {dict(sorted(dist.items()))}  →  avg={avg:.2f}/5.0")
    return avg


# ── Check 5: Distribution report ─────────────────────────────────────────────

def distribution_report(examples: list[dict]):
    patterns = Counter()
    tickers  = Counter()

    for ex in examples:
        sql = get_sql(ex).upper()
        if "JOIN" in sql:                        patterns["yoy_change"] += 1
        elif "ORDER BY" in sql and "LIMIT" in sql: patterns["ranking"] += 1
        elif any(a in sql for a in ["AVG(","SUM(","MAX(","MIN(","COUNT("]): patterns["aggregation"] += 1
        elif "BETWEEN" in sql or (sql.count("YEAR") > 1): patterns["trend"] += 1
        elif "filing_metadata" in sql.lower():   patterns["filing_metadata"] += 1
        elif "IN (" in sql:                      patterns["multi_ticker"] += 1
        elif ">" in sql or "<" in sql:           patterns["threshold_filter"] += 1
        else:                                    patterns["single_lookup"] += 1

        for t in re.findall(r"ticker\s*=\s*'(\w[\w-]*)'", get_sql(ex), re.I):
            tickers[t] += 1

    print(f"\n  Query pattern distribution:")
    for k, v in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"    {k:<25} {v:>4}  ({v/len(examples)*100:.1f}%)")
    print(f"\n  Ticker coverage: {len(tickers)} unique tickers")
    print(f"  Top 10: {tickers.most_common(10)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def validate_split(name: str, examples: list[dict], conn: sqlite3.Connection, run_llm: bool) -> list[dict]:
    print(f"\n{'='*50}")
    print(f"Validating {name}  ({len(examples)} examples)")
    print(f"{'='*50}")

    fail_exec, fail_empty, fail_schema = [], [], []
    good = []

    for ex in examples:
        sql = get_sql(ex)

        ok, err = check_schema(sql)
        if not ok:
            fail_schema.append((sql, err))
            continue

        ok, err, nrows = check_execution(sql, conn)
        if not ok:
            fail_exec.append((sql, err))
            continue

        if nrows == 0:
            fail_empty.append(sql)
            continue

        good.append(ex)

    total = len(examples)
    print(f"  Schema errors   : {len(fail_schema):>4}  ({len(fail_schema)/total*100:.1f}%)")
    print(f"  Execution errors: {len(fail_exec):>4}  ({len(fail_exec)/total*100:.1f}%)")
    print(f"  Empty results   : {len(fail_empty):>4}  ({len(fail_empty)/total*100:.1f}%)")
    print(f"  Passed          : {len(good):>4}  ({len(good)/total*100:.1f}%)")

    if fail_exec:
        print(f"\n  Sample execution errors:")
        for sql, err in fail_exec[:3]:
            print(f"    SQL: {sql[:80]}")
            print(f"    Err: {err}")

    distribution_report(good)

    if run_llm and good:
        print(f"\n  Running LLM judge on 50 samples...")
        llm_judge(good, n=50)

    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    train = load_jsonl(NL2SQL_DIR / "train.jsonl")
    eval_ = load_jsonl(NL2SQL_DIR / "eval.jsonl")

    clean_train = validate_split("train.jsonl", train, conn, run_llm=not args.no_llm_judge)
    clean_eval  = validate_split("eval.jsonl",  eval_,  conn, run_llm=False)

    conn.close()

    # Overwrite with clean versions
    save_jsonl(clean_train, NL2SQL_DIR / "train.jsonl")
    save_jsonl(clean_eval,  NL2SQL_DIR / "eval.jsonl")

    print(f"\n{'='*50}")
    print(f"Final dataset after filtering:")
    print(f"  train.jsonl : {len(clean_train)}  (was {len(train)})")
    print(f"  eval.jsonl  : {len(clean_eval)}   (was {len(eval_)})")
    print(f"  Removed     : {len(train)-len(clean_train)+len(eval_)-len(clean_eval)} bad examples")


if __name__ == "__main__":
    main()
