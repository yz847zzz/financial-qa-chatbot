#!/usr/bin/env python3
"""
eval_crossppl.py — Cross-perplexity scoring using fp16 Llama as reference.

For each stored answer in int8/awq4 result files, compute the perplexity of
that answer text under the fp16 model.  Lower PPL = fp16 considers the text
more natural/fluent.  Higher PPL for int8/awq4 vs fp16-on-fp16 = quantization
degraded language quality.

This measures language generation quality, NOT factual accuracy
(that is covered by value_correct + keyword_hit_rate).

How it works
────────────
  vLLM completions API with echo=True + logprobs=1 returns the log-probability
  of every token in the prompt.  We condition on the question and compute PPL
  only over the answer tokens:

      prompt = "Question: {q}\\nAnswer: {a}"
      feed to fp16 vLLM with echo=True
      PPL(answer | question) = exp( -mean(logprob of answer tokens) )

Usage
─────
  # 1. Start fp16 vLLM server in WSL
  bash deployment/scripts/start_server_quant.sh fp16

  # 2. Run (on Windows, vLLM must be up on :8001)
  python eval/eval_crossppl.py

  # Compare specific files
  python eval/eval_crossppl.py \\
      --fp16  eval/results/unified_vllm_fp16_20260620_211129.json \\
      --int8  eval/results/unified_vllm_int8_20260621_212656.json \\
      --awq4  eval/results/unified_vllm_awq4_20260620_230506.json

Output
──────
  eval/results/crossppl_results.json
  Console: per-type mean PPL table
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # eval/ siblings

RESULTS_DIR = ROOT / "eval" / "results"

# Default result files (latest 556-case runs)
DEFAULT_FILES = {
    "fp16": RESULTS_DIR / "unified_vllm_fp16_20260620_211129.json",
    "int8": RESULTS_DIR / "unified_vllm_int8_20260621_212656.json",
    "awq4": RESULTS_DIR / "unified_vllm_awq4_20260620_230506.json",
}


# ── vLLM cross-PPL ───────────────────────────────────────────────────────────

def get_openai_client(port: int = 8001):
    from openai import OpenAI
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none", timeout=60)


def compute_cross_ppl(
    question: str,
    answer: str,
    client,
    model: str = "base",
) -> float | None:
    """
    Compute PPL of `answer` conditioned on `question` under the fp16 model.

    Uses the completions endpoint with echo=True to get token log-probs for
    the full prompt, then isolates the answer portion.

    Returns None if answer is too short to score reliably.
    """
    answer = answer.strip()
    if not answer or len(answer.split()) < 3:
        return None

    # Prompt format: condition the model on the question for context
    prefix = f"Question: {question}\nAnswer: "
    full_prompt = prefix + answer

    try:
        resp = client.completions.create(
            model=model,
            prompt=full_prompt,
            max_tokens=1,       # generate 1 token to satisfy API; echo covers the rest
            logprobs=1,
            echo=True,          # return logprobs for prompt tokens
            temperature=0,
        )
    except Exception as e:
        print(f"  [PPL] API error: {e}", flush=True)
        return None

    choice = resp.choices[0]
    if not choice.logprobs:
        return None

    tokens   = choice.logprobs.tokens        # list of token strings
    offsets  = choice.logprobs.text_offset   # character offset of each token in full_prompt
    lp_list  = choice.logprobs.token_logprobs  # log-prob of each token

    if not tokens or not offsets or not lp_list:
        return None

    prefix_len = len(prefix)

    # Collect log-probs only for answer tokens (offset >= prefix_len)
    answer_lps = []
    for tok, off, lp in zip(tokens, offsets, lp_list):
        if off >= prefix_len and lp is not None and not math.isinf(lp):
            answer_lps.append(lp)

    if len(answer_lps) < 2:
        return None

    ppl = math.exp(-sum(answer_lps) / len(answer_lps))
    return round(ppl, 3)


# ── Per-file scoring ─────────────────────────────────────────────────────────

def score_file(
    label: str,
    result_path: Path,
    client,
    model: str = "base",
    skip_empty: bool = True,
) -> list[dict]:
    """Score all answers in a result file. Returns per-case records."""
    data = json.loads(result_path.read_text())
    cases = data.get("cases", [])
    scored = []
    n = len(cases)

    print(f"\n{'='*60}")
    print(f"Scoring {label}: {result_path.name}  ({n} cases)")
    print(f"{'='*60}")

    for i, case in enumerate(cases, 1):
        answer = case.get("answer", "")
        question = case.get("question", "")
        cat = case.get("category", "")

        if skip_empty and not answer:
            ppl = None
        else:
            if i % 50 == 1:
                print(f"  [{i}/{n}] scoring...", flush=True)
            ppl = compute_cross_ppl(question, answer, client, model)

        scored.append({
            "id":       case["id"],
            "category": cat,
            "question": question[:60],
            "answer_len": len(answer.split()),
            "cross_ppl":  ppl,
        })

    return scored


# ── Aggregate stats ──────────────────────────────────────────────────────────

def aggregate_ppl(records: list[dict]) -> dict:
    def map_cat(c: str) -> str:
        c = c.lower()
        if "type3" in c: return "Type3"
        if "type2" in c and "type1" not in c: return "Type2"
        return "Type1"

    by_type: dict[str, list[float]] = {"Type1": [], "Type2": [], "Type3": []}
    all_ppls: list[float] = []

    for r in records:
        ppl = r.get("cross_ppl")
        if ppl is None or math.isnan(ppl) or math.isinf(ppl) or ppl > 1e6:
            continue
        t = map_cat(r["category"])
        by_type[t].append(ppl)
        all_ppls.append(ppl)

    def stats(vals):
        if not vals: return None
        vals_s = sorted(vals)
        n = len(vals_s)
        return {
            "n":      n,
            "mean":   round(sum(vals_s) / n, 2),
            "median": round(vals_s[n // 2], 2),
            "p90":    round(vals_s[int(n * 0.90)], 2),
        }

    return {
        "overall": stats(all_ppls),
        "Type1":   stats(by_type["Type1"]),
        "Type2":   stats(by_type["Type2"]),
        "Type3":   stats(by_type["Type3"]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-PPL eval using fp16 Llama")
    parser.add_argument("--fp16", default=None)
    parser.add_argument("--int8", default=None)
    parser.add_argument("--awq4", default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="base")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    files = {
        "fp16": Path(args.fp16) if args.fp16 else DEFAULT_FILES["fp16"],
        "int8": Path(args.int8) if args.int8 else DEFAULT_FILES["int8"],
        "awq4": Path(args.awq4) if args.awq4 else DEFAULT_FILES["awq4"],
    }

    # Check files exist
    for label, path in files.items():
        if not path.exists():
            print(f"ERROR: {label} result not found: {path}", file=sys.stderr)
            print("  Specify with --fp16 / --int8 / --awq4", file=sys.stderr)
            sys.exit(1)

    # Check vLLM server
    print(f"Connecting to fp16 vLLM at localhost:{args.port}...")
    client = get_openai_client(args.port)
    try:
        client.models.list()
        print("vLLM server ready.\n")
    except Exception:
        print("ERROR: vLLM server not reachable.", file=sys.stderr)
        print("  Start it with: bash deployment/scripts/start_server_quant.sh fp16", file=sys.stderr)
        sys.exit(1)

    # Score each config
    all_results = {}
    for label, path in files.items():
        records = score_file(label, path, client, model=args.model)
        agg = aggregate_ppl(records)
        all_results[label] = {"records": records, "aggregate": agg}
        print(f"\n  {label} PPL summary:")
        print(f"    Overall: mean={agg['overall']['mean'] if agg['overall'] else 'N/A'}  "
              f"median={agg['overall']['median'] if agg['overall'] else 'N/A'}")

    # Print comparison table
    LINE = "=" * 70
    print(f"\n\n{LINE}")
    print("  CROSS-PERPLEXITY COMPARISON (fp16 Llama as reference model)")
    print("  Lower PPL = fp16 considers the generated text more natural/fluent")
    print(LINE)

    hdr = f"  {'Config':<10} {'Overall mean':>14} {'median':>8} {'Type1':>8} {'Type2':>10} {'Type3':>8}"
    print(hdr)
    print("  " + "-" * 62)

    for label in ["fp16", "int8", "awq4"]:
        agg = all_results[label]["aggregate"]
        ov  = agg["overall"]
        t1  = agg["Type1"]
        t2  = agg["Type2"]
        t3  = agg["Type3"]

        def fmt(s, key): return f"{s[key]:.1f}" if s else " N/A"

        print(f"  {label:<10} {fmt(ov,'mean'):>14} {fmt(ov,'median'):>8} "
              f"{fmt(t1,'mean'):>8} {fmt(t2,'mean'):>10} {fmt(t3,'mean'):>8}")

    print(f"\n  NOTE: fp16 scored on its OWN outputs (baseline).")
    print(f"  int8/awq4 PPL > fp16 PPL → quantization made text less natural to fp16.")
    print(LINE)

    # Save results
    out = Path(args.output) if args.output else RESULTS_DIR / "crossppl_results.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_model": "fp16 Llama-3.2-3B-Instruct",
        "configs": {
            label: {
                "source_file": str(files[label].name),
                "aggregate":   all_results[label]["aggregate"],
                "cases":       all_results[label]["records"],
            }
            for label in ["fp16", "int8", "awq4"]
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
