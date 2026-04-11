#!/usr/bin/env python3
"""
Execution accuracy evaluation for NL2SQL adapter.

For each eval example:
  1. Apply DB compatibility fixes to both reference SQL and FT-generated SQL
     (period format: F2023 -> FY2023)
  2. Execute reference SQL -> result_ref
  3. Execute FT SQL       -> result_ft
  4. Compare result sets

Reports:
  - exec_ref_ok   : reference SQL executed without error
  - exec_ft_ok    : FT SQL executed without error
  - ref_has_data  : reference SQL returned >= 1 row
  - result_match  : FT result == reference result (only counted when ref has data)
  - empty_both    : both returned 0 rows (metric not in DB - excluded from accuracy)

Usage:
    python finetune/adapters/nl2sql/exec_accuracy.py
    python finetune/adapters/nl2sql/exec_accuracy.py --results models/nl2sql/eval_results.json
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[3]
DB_PATH    = ROOT / "data" / "financials.db"
EVAL_FILE  = ROOT / "data" / "nl2sql" / "eval.jsonl"
RESULTS_FILE = ROOT / "models" / "nl2sql" / "eval_results.json"
OUT_FILE   = ROOT / "models" / "nl2sql" / "exec_accuracy.json"


# ── DB compatibility fixes ─────────────────────────────────────────────────────

def fix_period_format(sql: str) -> str:
    """Model generates F2023, DB stores FY2023."""
    # 'F2023' -> 'FY2023' (only bare F-year, not FY already)
    sql = re.sub(r"'F(20\d{2})'", r"'FY\1'", sql)
    # IN ('F2020','F2021') pattern
    sql = re.sub(r"'F(20\d{2})'", r"'FY\1'", sql)
    # LIKE 'F%' -> LIKE 'FY%'
    sql = re.sub(r"LIKE\s+'F%'", "LIKE 'FY%'", sql, flags=re.I)
    # LIKE 'F20%' -> LIKE 'FY20%'
    sql = re.sub(r"LIKE\s+'F(20[^']*)'", r"LIKE 'FY\1'", sql, flags=re.I)
    return sql


def normalise_result(rows: list) -> list:
    """
    Sort rows and round floats so minor differences don't cause false mismatches.
    Convert each row to a tuple of strings for comparison.
    """
    def fmt_val(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v) if v is not None else "NULL"

    normalised = [tuple(fmt_val(c) for c in row) for row in rows]
    return sorted(normalised)


# ── SQL execution ──────────────────────────────────────────────────────────────

def run_sql(con: sqlite3.Connection, sql: str) -> tuple[bool, list, str]:
    """
    Execute sql against con.
    Returns (success, rows, error_message).
    """
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(results_path: Path) -> None:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return

    # Load predictions from eval_results.json (already has base + ft preds)
    with open(results_path, encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} eval examples from {results_path.name}")
    print(f"DB: {DB_PATH}  ({DB_PATH.stat().st_size // 1024} KB)\n")

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row  # named columns

    records = []

    # Counters
    ref_ok = ref_has_data = ft_ok = ft_match = ft_match_denom = 0
    base_ok = base_has_data = base_match = base_match_denom = 0
    empty_both_ft = empty_both_base = 0

    for pred in predictions:
        question   = pred["question"]
        ref_sql    = fix_period_format(pred["reference"])
        ft_sql     = fix_period_format(pred["ft_pred"])
        base_sql   = fix_period_format(pred["base_pred"])

        # Execute reference
        ref_success, ref_rows, ref_err = run_sql(con, ref_sql)
        if ref_success:
            ref_ok += 1
        ref_data = ref_success and len(ref_rows) > 0
        if ref_data:
            ref_has_data += 1

        # Execute FT prediction
        ft_success, ft_rows, ft_err = run_sql(con, ft_sql)
        if ft_success:
            ft_ok += 1

        # Execute base prediction
        base_success, base_rows, base_err = run_sql(con, base_sql)
        if base_success:
            base_ok += 1

        # Compare results — only meaningful when reference returns data
        if ref_data:
            ft_match_denom += 1
            base_match_denom += 1

            ref_norm  = normalise_result(ref_rows)
            ft_norm   = normalise_result(ft_rows)
            base_norm = normalise_result(base_rows)

            ft_matches   = (ft_norm == ref_norm)
            base_matches = (base_norm == ref_norm)
            if ft_matches:
                ft_match += 1
            if base_matches:
                base_match += 1
        else:
            ft_matches   = None
            base_matches = None
            # Track cases where both ref and FT return nothing (metric not in DB)
            if ref_success and ft_success and len(ref_rows) == 0 and len(ft_rows) == 0:
                empty_both_ft += 1
            if ref_success and base_success and len(ref_rows) == 0 and len(base_rows) == 0:
                empty_both_base += 1

        records.append({
            "question":       question,
            "ref_sql":        ref_sql,
            "ft_sql":         ft_sql,
            "base_sql":       base_sql,
            "ref_rows":       len(ref_rows) if ref_success else None,
            "ft_rows":        len(ft_rows)  if ft_success  else None,
            "base_rows":      len(base_rows) if base_success else None,
            "ref_error":      ref_err  or None,
            "ft_error":       ft_err   or None,
            "base_error":     base_err or None,
            "ft_match":       ft_matches,
            "base_match":     base_matches,
        })

    n = len(predictions)

    print("=" * 65)
    print(f"Execution Accuracy  —  n={n} examples")
    print("=" * 65)
    print(f"\n{'Metric':<40} {'Base':>8}  {'Fine-tuned':>10}")
    print("-" * 62)
    print(f"{'SQL executes without error':<40} {base_ok/n*100:>7.1f}%  {ft_ok/n*100:>9.1f}%")

    if ref_has_data > 0:
        print(f"\n{'--- When reference SQL returns data (n='+str(ref_has_data)+') ---':<40}")
        print(f"{'Result matches reference exactly':<40} {base_match/base_match_denom*100:>7.1f}%  {ft_match/ft_match_denom*100:>9.1f}%")
    else:
        print("\nNo reference queries returned data from the DB.")
        print("Likely cause: metric names in DB don't match canonical training names.")

    print(f"\n{'--- Diagnostics ---':<40}")
    print(f"{'Reference SQL executes OK':<40} {ref_ok/n*100:>7.1f}%")
    print(f"{'Reference SQL returns >= 1 row':<40} {ref_has_data/n*100:>7.1f}%")
    print(f"{'Both ref+FT empty (metric not in DB)':<40} {empty_both_ft:>7}  ({empty_both_ft/n*100:.1f}%)")
    print("=" * 65)

    # Show failures where FT ran but didn't match
    mismatches = [r for r in records if r["ft_match"] is False]
    if mismatches:
        print(f"\n-- FT mismatches ({len(mismatches)}) --")
        for r in mismatches[:5]:
            print(f"\nQ:    {r['question'][:75]}")
            print(f"REF:  {r['ref_sql'][:100]}  [{r['ref_rows']} rows]")
            print(f"FT:   {r['ft_sql'][:100]}  [{r['ft_rows']} rows]")

    # Save full results
    report = {
        "n_total":            n,
        "ref_exec_ok":        ref_ok,
        "ref_has_data":       ref_has_data,
        "ft_exec_ok":         ft_ok,
        "base_exec_ok":       base_ok,
        "ft_result_match":    ft_match,
        "base_result_match":  base_match,
        "match_denominator":  ft_match_denom,
        "empty_both_ft":      empty_both_ft,
        "examples":           records,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull results -> {OUT_FILE}")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_FILE),
                        help="eval_results.json from eval_compare.py")
    args = parser.parse_args()
    main(Path(args.results))
