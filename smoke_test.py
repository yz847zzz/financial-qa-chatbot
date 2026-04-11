"""
End-to-end smoke test for the Financial QA chatbot pipeline.
Loads models once, runs one question per intent type, prints results.

Usage:
    python smoke_test.py
    python smoke_test.py --no-rag   # skip VectorDB (faster if not populated)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Import from chatbot
from chatbot import (
    load_base_model,
    load_nl2sql_model,
    load_vectordb,
    answer,
)

# ── Test cases (one per intent type) ──────────────────────────────────────────
TEST_CASES = [
    {
        "type":     "Type1",
        "question": "What was Apple's revenue in FY2023?",
        "checks": [
            lambda r: r["intent"] == "Type1",
            lambda r: r["sql"] is not None,
            lambda r: "panel" in (r["sql_repaired"] or "").lower(),
            lambda r: "AAPL" in (r["sql_repaired"] or "").upper(),
            lambda r: "FY2023" in (r["sql_repaired"] or ""),
            lambda r: r["error"] is None,
            lambda r: r["answer"] is not None and len(r["answer"]) > 10,
        ],
        "check_names": [
            "intent==Type1",
            "SQL generated",
            "FROM panel",
            "ticker AAPL",
            "year FY2023",
            "no exec error",
            "answer non-empty",
        ],
    },
    {
        "type":     "Type1 (ranking)",
        "question": "Which 3 companies had the highest net income in FY2023?",
        "checks": [
            lambda r: r["intent"] == "Type1",
            lambda r: r["sql"] is not None,
            lambda r: "net_income" in (r["sql_repaired"] or "").lower(),
            lambda r: "limit 3" in (r["sql_repaired"] or "").lower() or "LIMIT 3" in (r["sql_repaired"] or ""),
            lambda r: r["error"] is None,
        ],
        "check_names": [
            "intent==Type1",
            "SQL generated",
            "net_income column",
            "LIMIT 3",
            "no exec error",
        ],
    },
    {
        "type":     "Type3",
        "question": "Hello! What can you help me with?",
        "checks": [
            lambda r: r["intent"] == "Type3",
            lambda r: r["sql"] is None,
            lambda r: r["answer"] is not None and len(r["answer"]) > 10,
        ],
        "check_names": [
            "intent==Type3",
            "no SQL generated",
            "answer non-empty",
        ],
    },
]

TYPE2_CASE = {
    "type":     "Type2 (RAG)",
    "question": "How did Apple describe its AI strategy in recent filings?",
    "checks": [
        lambda r: r["intent"] == "Type2",
        lambda r: r["sql"] is None,
        lambda r: r["answer"] is not None and len(r["answer"]) > 20,
    ],
    "check_names": [
        "intent==Type2",
        "no SQL generated",
        "answer non-empty",
    ],
}

# ── New decomposition test cases ──────────────────────────────────────────────

DECOMPOSE_CASES = [
    {
        "type":     "Single Type1 (no decomposition)",
        "question": "What was Microsoft's net income in FY2022?",
        "checks": [
            lambda r: r.get("sub_results") is None,          # no decomposition
            lambda r: r["intent"] == "Type1",
            lambda r: r["sql"] is not None,
            lambda r: "MSFT" in (r["sql_repaired"] or "").upper(),
            lambda r: "FY2022" in (r["sql_repaired"] or ""),
            lambda r: r["error"] is None,
            lambda r: r["answer"] is not None and len(r["answer"]) > 10,
        ],
        "check_names": [
            "no decomposition (single question)",
            "intent==Type1",
            "SQL generated",
            "ticker MSFT",
            "year FY2022",
            "no exec error",
            "answer non-empty",
        ],
    },
    {
        "type":     "Multi-Type (Type1 + Type2 compound)",
        "question": (
            "What was Apple's revenue in FY2023 and "
            "how did they describe their supply chain risks?"
        ),
        "checks": [
            lambda r: r.get("sub_results") is not None,       # decomposed
            lambda r: len(r["sub_results"]) == 2,             # split into 2
            lambda r: any(s["intent"] == "Type1" for s in r["sub_results"]),
            lambda r: any(s["intent"] == "Type2" for s in r["sub_results"]),
            lambda r: any(s["sql"] is not None for s in r["sub_results"]),
            lambda r: all(s["error"] is None for s in r["sub_results"]),
            lambda r: r["answer"] is not None and len(r["answer"]) > 20,
        ],
        "check_names": [
            "question was decomposed",
            "exactly 2 sub-questions",
            "one Type1 sub-question",
            "one Type2 sub-question",
            "SQL generated for Type1",
            "no exec errors",
            "synthesized answer non-empty",
        ],
    },
    {
        "type":     "Multi-Type1 (two SQL questions)",
        "question": (
            "What was Apple's revenue in FY2023 and "
            "what was Microsoft's net income in FY2022?"
        ),
        "checks": [
            lambda r: r.get("sub_results") is not None,       # decomposed
            lambda r: len(r["sub_results"]) == 2,             # split into 2
            lambda r: all(s["intent"] == "Type1" for s in r["sub_results"]),
            lambda r: all(s["sql"] is not None for s in r["sub_results"]),
            lambda r: any("AAPL" in (s["sql_repaired"] or "").upper() for s in r["sub_results"]),
            lambda r: any("MSFT" in (s["sql_repaired"] or "").upper() for s in r["sub_results"]),
            lambda r: all(s["error"] is None for s in r["sub_results"]),
            lambda r: r["answer"] is not None and len(r["answer"]) > 20,
        ],
        "check_names": [
            "question was decomposed",
            "exactly 2 sub-questions",
            "both sub-questions Type1",
            "both have SQL",
            "AAPL in one SQL",
            "MSFT in other SQL",
            "no exec errors",
            "synthesized answer non-empty",
        ],
    },
]


def run_case(case: dict, base_model, base_tok, nl2sql_model, nl2sql_tok, retriever) -> bool:
    print(f"\n{'─'*60}")
    print(f"[{case['type']}] {case['question']}")
    print(f"{'─'*60}")

    t0 = time.time()
    res = answer(case["question"], base_model, base_tok, nl2sql_model, nl2sql_tok, retriever)
    elapsed = time.time() - t0

    # Print pipeline trace
    sub_results = res.get("sub_results")
    if sub_results:
        print(f"  decomposed : {len(sub_results)} sub-questions")
        for i, s in enumerate(sub_results, 1):
            print(f"    [{i}] intent={s['intent']} | q={s['question']}")
            if s["sql_repaired"]:
                print(f"        sql={s['sql_repaired'][:90]}")
            if s["error"]:
                print(f"        ERROR={s['error']}")
            print(f"        answer={s['answer'][:100] if s['answer'] else 'None'}")
    else:
        print(f"  intent  : {res['intent']}")
        if res["sql"]:
            print(f"  sql     : {res['sql_repaired'][:100]}")
        if res["repairs"]:
            print(f"  repairs : {res['repairs']}")
        if res["error"]:
            print(f"  ERROR   : {res['error']}")
    print(f"  answer  : {(res['answer'] or '')[:200]}")
    print(f"  time    : {elapsed:.1f}s")

    # Run checks
    passed = True
    print(f"\n  Checks:")
    for check, name in zip(case["checks"], case["check_names"]):
        try:
            ok = check(res)
        except Exception as e:
            ok = False
            name = f"{name} (exception: {e})"
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name}")
        if not ok:
            passed = False

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rag", action="store_true", help="Skip VectorDB/RAG test")
    parser.add_argument("--decompose-only", action="store_true",
                        help="Run only the three decomposition test cases")
    args = parser.parse_args()

    print("=" * 60)
    print("Financial QA Chatbot — End-to-End Smoke Test")
    print("=" * 60)

    # ── Load models ────────────────────────────────────────────────
    print("\n[Loading models...]")
    t0 = time.time()
    base_model, base_tok = load_base_model()
    print(f"  Base model: {time.time()-t0:.1f}s")

    t1 = time.time()
    nl2sql_model, nl2sql_tok = load_nl2sql_model(base_model, base_tok)
    print(f"  NL2SQL adapter: {time.time()-t1:.1f}s")

    retriever = None
    if not args.no_rag:
        t2 = time.time()
        retriever = load_vectordb()
        if retriever:
            print(f"  VectorDB+retriever: {time.time()-t2:.1f}s")
        else:
            print("  VectorDB: not available — skipping Type2 test")

    # ── Run test cases ─────────────────────────────────────────────
    if args.decompose_only:
        # Multi-Type2 decompose case needs RAG; others don't
        cases = DECOMPOSE_CASES
    else:
        cases = list(TEST_CASES)
        if retriever and not args.no_rag:
            cases.append(TYPE2_CASE)
        cases.extend(DECOMPOSE_CASES)

    results = []
    for case in cases:
        ok = run_case(case, base_model, base_tok, nl2sql_model, nl2sql_tok, retriever)
        results.append((case["type"], ok))

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
