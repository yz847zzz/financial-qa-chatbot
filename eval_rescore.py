#!/usr/bin/env python3
"""
eval_rescore.py -- Offline re-scoring of existing eval results.

Recomputes composite scores using the current scoring formula (with cosine
similarity) from stored answer text, without re-running any inference.

Usage:
  # Re-score all unified result files
  python eval_rescore.py

  # Re-score a specific file
  python eval_rescore.py --file eval_results/unified_vllm_awq4_20260503_000512.json

  # Re-score and overwrite the original files (default: writes *_rescored.json)
  python eval_rescore.py --overwrite
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

from eval_system import TEST_CASES, value_correct as eval_value_correct
from eval_benchmark import latency_stats
from eval_unified import (
    WEIGHTS,
    map_category,
    case_score,
    cosine_similarity,
    cosine_similarity_multi_ref,
    build_reference,
)

EVAL_DIR = ROOT / "eval_results"

# Build lookup: test case id -> test case dict (default: built-in 52)
CASES_BY_ID = {c["id"]: c for c in TEST_CASES}

# Optional: GPT-4o multi-reference answers {id -> {answer: [ref1, ref2, ref3]}}
REFS_BY_ID: dict[int, dict] = {}


def load_testset(path: str) -> None:
    """Override CASES_BY_ID with an expanded test-case JSON."""
    global CASES_BY_ID
    with open(path) as f:
        data = json.load(f)
    CASES_BY_ID = {c["id"]: c for c in data}
    print(f"  Loaded {len(CASES_BY_ID)} test cases from {path}")


def load_references(path: str) -> None:
    """Load GPT-4o multi-reference answers for accurate semantic scoring."""
    global REFS_BY_ID
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    REFS_BY_ID = {rec["id"]: rec for rec in data}
    print(f"  Loaded {len(REFS_BY_ID)} multi-reference answer sets from {path}")


def rescore_file(path: Path, overwrite: bool = False) -> dict:
    """Re-score a single eval result file."""
    with open(path) as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if not cases:
        print(f"  SKIP {path.name}: no cases found")
        return data

    print(f"\n  Re-scoring {path.name} ({len(cases)} cases)...")

    all_scores = []
    by_type = {"Type1": [], "Type2": [], "Type3": []}

    for r in cases:
        # Look up the original test case for expected_value / expected_keywords
        tc = CASES_BY_ID.get(r["id"], {})
        answer = r.get("answer", "")

        # Re-evaluate value_correct using updated test case (acceptable_values)
        if tc and tc.get("expected_value") is not None:
            vc = eval_value_correct(
                answer,
                tc["expected_value"],
                acceptable_values=tc.get("acceptable_values"),
            )
            r["value_correct"] = vc

        # Compute cosine similarity (multi-ref if available, else single pseudo-ref)
        refs = REFS_BY_ID.get(r["id"], {}).get("answer")
        if refs:
            sem = cosine_similarity_multi_ref(answer, refs)
        else:
            ref = build_reference(tc) if tc else r.get("question", "")
            sem = cosine_similarity(answer, ref)
        r["semantic_similarity"] = round(sem, 4)

        # Recompute composite score with new weights
        kw = r.get("keyword_hit_rate", 0.0)
        s = case_score(
            r["category"],
            r.get("intent_correct"),
            r.get("value_correct"),
            kw,
            sem,
        )
        r["composite_score"] = s

        t = map_category(r["category"])
        by_type[t].append(r)
        all_scores.append(s)

    # Recompute aggregate
    def type_stats(recs):
        if not recs:
            return None
        scores = [r["composite_score"] for r in recs]
        intent_vals = [int(r["intent_correct"]) for r in recs
                       if r.get("intent_correct") is not None]
        value_vals = [int(r["value_correct"]) for r in recs
                      if r.get("value_correct") is not None]
        kw_vals = [r.get("keyword_hit_rate", 0.0) for r in recs]
        sem_vals = [r.get("semantic_similarity", 0.0) for r in recs]
        return {
            "n": len(recs),
            "composite_mean": round(statistics.mean(scores), 4),
            "intent_accuracy": round(statistics.mean(intent_vals), 4) if intent_vals else None,
            "value_accuracy": round(statistics.mean(value_vals), 4) if value_vals else None,
            "keyword_hit_rate": round(statistics.mean(kw_vals), 4),
            "semantic_similarity": round(statistics.mean(sem_vals), 4),
        }

    intent_all = [int(r["intent_correct"]) for r in cases
                  if r.get("intent_correct") is not None]
    value_all = [int(r["value_correct"]) for r in cases
                 if r.get("value_correct") is not None]
    kw_all = [r.get("keyword_hit_rate", 0.0) for r in cases]
    sem_all = [r.get("semantic_similarity", 0.0) for r in cases]
    flu_all = [r["fluency_score"] for r in cases
               if r.get("fluency_score") is not None]
    timings_list = [r["timings"] for r in cases if r.get("timings")]

    agg = {
        "n_cases": len(cases),
        "composite_score": round(statistics.mean(all_scores), 4),
        "intent_accuracy": round(statistics.mean(intent_all), 4) if intent_all else None,
        "value_accuracy": round(statistics.mean(value_all), 4) if value_all else None,
        "keyword_hit_rate": round(statistics.mean(kw_all), 4),
        "semantic_similarity": round(statistics.mean(sem_all), 4),
        "fluency_mean": round(statistics.mean(flu_all), 2) if flu_all else None,
        "latency": latency_stats(timings_list) if timings_list else {},
        "per_type": {
            "Type1": type_stats(by_type["Type1"]),
            "Type2": type_stats(by_type["Type2"]),
            "Type3": type_stats(by_type["Type3"]),
        },
    }
    data["aggregate"] = agg
    data["scoring"] = {
        "method": "cosine_similarity",
        "model": "all-MiniLM-L6-v2",
        "weights": WEIGHTS,
    }

    # Print summary
    pt = agg["per_type"]
    name = f"{data.get('config', '?')}/{data.get('label', '?')}"
    print(f"    {name}:")
    print(f"      Overall composite : {agg['composite_score']:.4f}")
    for t in ["Type1", "Type2", "Type3"]:
        ts = pt.get(t)
        if ts:
            print(f"      {t}: {ts['composite_mean']:.4f}  ({ts['n']} cases, sem={ts['semantic_similarity']:.4f})")
    print(f"      Semantic sim mean : {agg['semantic_similarity']:.4f}")

    # Save
    if overwrite:
        out_path = path
    else:
        out_path = path.with_name(path.stem + "_rescored.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    -> {out_path.name}")

    return data


def main():
    parser = argparse.ArgumentParser(description="Offline re-scoring of eval results")
    parser.add_argument("--file", type=str, help="Re-score a specific file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite original files (default: write *_rescored.json)")
    parser.add_argument("--testset", type=str, default=None,
                        help="Path to expanded test-case JSON for reference lookup")
    parser.add_argument("--references", type=str, default=None,
                        help="Path to eval_references.json for multi-ref cosine scoring")
    args = parser.parse_args()

    if args.testset:
        load_testset(args.testset)
    if args.references:
        load_references(args.references)

    print("=" * 60)
    print("EVAL RE-SCORER (offline)")
    print(f"  Weights: {WEIGHTS}")
    sem_mode = "multi-ref max cosine (GPT-4o)" if REFS_BY_ID else "single pseudo-ref cosine"
    print(f"  Semantic: {sem_mode} (all-MiniLM-L6-v2)")
    print("=" * 60)

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(EVAL_DIR.glob("unified_*.json"))
        # Skip already-rescored files if not overwriting
        if not args.overwrite:
            files = [f for f in files if "_rescored" not in f.name]

    if not files:
        print("No files found to re-score.")
        return

    print(f"\nFound {len(files)} file(s) to re-score.")

    results = []
    for f in files:
        data = rescore_file(f, overwrite=args.overwrite)
        results.append(data)

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("  RESCORED COMPARISON")
    print(f"{'=' * 70}")
    hdr = f"  {'Config':<22} {'Overall':>8} {'Type1':>8} {'Type2':>8} {'Type3':>8} {'SemSim':>8}"
    print(hdr)
    print("  " + "-" * 64)
    for d in results:
        agg = d.get("aggregate", {})
        pt = agg.get("per_type", {})
        name = f"{d.get('config', '?')}/{d.get('label', '?')}"
        t1 = f"{pt['Type1']['composite_mean']:.4f}" if pt.get("Type1") else "  N/A"
        t2 = f"{pt['Type2']['composite_mean']:.4f}" if pt.get("Type2") else "  N/A"
        t3 = f"{pt['Type3']['composite_mean']:.4f}" if pt.get("Type3") else "  N/A"
        ss = f"{agg.get('semantic_similarity', 0):.4f}"
        print(f"  {name:<22} {agg.get('composite_score', 0):>8.4f} {t1:>8} {t2:>8} {t3:>8} {ss:>8}")

    print(f"\n  Scoring: {WEIGHTS}")
    print(f"  Semantic: cosine similarity (all-MiniLM-L6-v2)")
    print()


if __name__ == "__main__":
    main()
