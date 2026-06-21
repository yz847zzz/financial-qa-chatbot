#!/usr/bin/env python3
"""
scripts/generate_eval_references.py

Generate 3 human-quality reference answers per test case using GPT-4o.
Mirrors FINDMIND's C-list-answer.json: each entry has a structured `prompt`
(ground-truth values / keywords) and an `answer` list of 3 varied phrasings.

Type1 (exact fact)  : GPT-4o given the exact DB value → 3 phrasings
Type2 (qualitative) : ChromaDB retrieval → GPT-4o grounded in real filing text
Type3 (chat/meta)   : GPT-4o free-form

Usage:
    OPENAI_API_KEY=sk-... python scripts/generate_eval_references.py
    OPENAI_API_KEY=sk-... python scripts/generate_eval_references.py --testset eval_testcases_expanded.json
    OPENAI_API_KEY=sk-... python scripts/generate_eval_references.py --ids 13,14,15,16,44,45,46,47,48
    OPENAI_API_KEY=sk-... python scripts/generate_eval_references.py --no-chromadb  # skip retrieval

Output: eval_references.json
Resume: re-running skips already-completed cases automatically.
Cost:   ~$0.50-1.00 for 52 cases (gpt-4o at $5/1M tokens)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

# Fix Windows GBK stdout — SEC filing text has ™, ®, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_system import TEST_CASES

OUT_PATH = ROOT / "eval_references.json"


# ── GPT-4o helpers ────────────────────────────────────────────────────────────

def get_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    return OpenAI(api_key=key)


SYSTEM = (
    "You are a precise financial analyst generating benchmark reference answers. "
    "CRITICAL: Your entire response must be a single valid JSON array containing exactly "
    "3 strings, like this: [\"answer1\", \"answer2\", \"answer3\"]. "
    "No markdown. No explanation. No prefix. Just the JSON array."
)


def gpt4o(client, prompt: str, temperature: float = 0.4, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        t = temperature if attempt == 0 else 0.0  # temperature=0 on retry for determinism
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user",   "content": prompt}],
            temperature=t,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        # Quick check: if it already contains a parseable 3-string array, accept it
        try:
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                lst = json.loads(m.group())
                if isinstance(lst, list) and len(lst) >= 3 and all(isinstance(x, str) for x in lst):
                    return raw
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.5)
    return raw


def parse_json_list(raw: str, fallback: str) -> list[str]:
    """Extract a JSON array of strings from GPT output. Handles multiple formats."""
    # Try direct JSON array
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            lst = json.loads(m.group())
            if isinstance(lst, list) and all(isinstance(x, str) for x in lst) and lst:
                while len(lst) < 3:
                    lst.append(fallback)
                return lst[:3]
        except Exception:
            pass

    # Try JSON object with "answers" / "references" / "array" key
    try:
        obj = json.loads(raw)
        for key in ("answers", "references", "array", "items", "results"):
            if key in obj and isinstance(obj[key], list):
                lst = [str(x) for x in obj[key] if x]
                if lst:
                    while len(lst) < 3:
                        lst.append(fallback)
                    return lst[:3]
    except Exception:
        pass

    # Line-split fallback: grab non-empty lines longer than 20 chars
    lines = [ln.strip().lstrip('0123456789.-) "\'').strip()
              for ln in raw.splitlines() if len(ln.strip()) > 20]
    lines = [ln for ln in lines if ln and ln != fallback][:3]
    while len(lines) < 3:
        lines.append(fallback)
    return lines


# ── ChromaDB retrieval (lazy-loaded) ─────────────────────────────────────────

_embed = None
_col = None


def _init_chroma():
    global _embed, _col
    if _col is not None:
        return
    from sentence_transformers import SentenceTransformer
    import chromadb
    print("[Init] Loading all-MiniLM-L6-v2 ...", flush=True)
    _embed = SentenceTransformer("all-MiniLM-L6-v2")
    print("[Init] Connecting to ChromaDB ...", flush=True)
    client = chromadb.PersistentClient(path=str(ROOT / "data/vectordb"))
    _col = client.get_collection("financial_docs")
    print(f"[Init] Collection: {_col.count()} chunks", flush=True)


def retrieve_context(question: str, n: int = 5) -> str:
    _init_chroma()
    emb = _embed.encode([question]).tolist()
    res = _col.query(query_embeddings=emb, n_results=n,
                     include=["documents", "metadatas"])
    parts = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        src = f"{meta.get('company','')} {meta.get('filing_type','')} {meta.get('date','')}"
        parts.append(f"[{src.strip()}]\n{doc[:600]}")
    return "\n\n---\n\n".join(parts)


# ── Value formatting ──────────────────────────────────────────────────────────

def fmt(val: float) -> str:
    if val is None:
        return ""
    if val < 0:
        mag = abs(val)
        if mag >= 1e9:
            return f"-${mag/1e9:.3f}B (net loss)"
        return f"-${mag:,.0f}"
    if val >= 1e12:
        return f"${val/1e12:.3f}T"
    if val >= 1e9:
        return f"${val/1e9:.3f}B"
    if val >= 1e6:
        return f"${val/1e6:.1f}M"
    if val < 1:
        return f"{val*100:.2f}%"
    return f"${val:,.2f}"


def extract_year(text: str) -> str:
    m = re.search(r"FY\d{4}", text)
    return m.group() if m else ""


# ── Per-type reference generators ────────────────────────────────────────────

def gen_type1(client, case: dict) -> tuple[dict, list[str]]:
    """Type1 / Type1-ranking / Type1-compare: exact value or keyword hints."""
    q = case["question"]
    val = case.get("expected_value")
    kws = case.get("expected_keywords", [])

    if val is not None:
        val_str = fmt(val)
        prompt = (
            f"Question: {q}\n"
            f"Exact answer: {val_str}  (raw numeric: {val})\n"
            f"These terms must appear: {', '.join(kws[:3])}\n\n"
            "Generate 3 reference answers. Each must contain the exact numeric value "
            "and be a complete, natural-sounding sentence."
        )
        prom = {
            "year": extract_year(q),
            "key_word": kws[0] if kws else "",
            "prom_answer": val_str,
        }
    else:
        prompt = (
            f"Question: {q}\n"
            f"Expected answer must mention: {', '.join(kws)}\n\n"
            "Generate 3 reference answers. Each should be a complete sentence "
            "that mentions the relevant companies, values, or rankings indicated."
        )
        prom = {"year": extract_year(q), "key_word": ", ".join(kws[:3])}

    raw = gpt4o(client, prompt)
    return prom, parse_json_list(raw, fallback=q)


def gen_type2(client, case: dict, use_chroma: bool = True) -> tuple[dict, list[str]]:
    """Type2 (qualitative RAG): retrieve real context, generate grounded refs."""
    q = case["question"]
    kws = case.get("expected_keywords", [])

    if use_chroma:
        ctx = retrieve_context(q, n=5)
        prompt = (
            f"Question: {q}\n\n"
            f"Context from SEC filings (use ONLY this — do not invent facts):\n{ctx}\n\n"
            f"Key terms that should appear: {', '.join(kws)}\n\n"
            "Generate 3 reference answers based strictly on the provided context. "
            "If context is limited, acknowledge that. Each answer: 2-4 sentences."
        )
    else:
        prompt = (
            f"Question: {q}\n"
            f"Key terms to include: {', '.join(kws)}\n\n"
            "Generate 3 reference answers for a qualitative financial QA benchmark. "
            "Each answer: 2-4 sentences, professional tone."
        )

    prom = {"year": extract_year(q), "key_word": ", ".join(kws[:4])}
    raw = gpt4o(client, prompt, temperature=0.5)
    return prom, parse_json_list(raw, fallback=q)


def gen_type3(client, case: dict) -> tuple[dict, list[str]]:
    """Type3 (chat/meta): free-form financial assistant responses."""
    q = case["question"]
    kws = case.get("expected_keywords", [])

    prompt = (
        f"Question: {q}\n"
        f"Key terms to include (if relevant): {', '.join(kws)}\n\n"
        "Generate 3 different natural, helpful responses for a financial assistant chatbot. "
        "Each response: 1-3 sentences."
    )
    prom = {"key_word": ", ".join(kws)}
    raw = gpt4o(client, prompt, temperature=0.6)
    return prom, parse_json_list(raw, fallback=q)


# ── Category mapper ───────────────────────────────────────────────────────────

def map_cat(cat: str) -> str:
    c = cat.lower()
    if "type3" in c:
        return "Type3"
    if "type2" in c and "type1" not in c:
        return "Type2"
    return "Type1"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o reference answers for eval cases")
    parser.add_argument("--testset", type=str, default=None,
                        help="Path to expanded test-case JSON (default: built-in 52)")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated case IDs to run (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output path (default: {OUT_PATH})")
    parser.add_argument("--no-chromadb", action="store_true",
                        help="Skip ChromaDB retrieval for Type2 (faster, less accurate refs)")
    args = parser.parse_args()

    client = get_client()

    # Load cases
    if args.testset:
        with open(args.testset, encoding="utf-8") as f:
            cases = json.load(f)
        print(f"Loaded {len(cases)} cases from {args.testset}")
    else:
        cases = TEST_CASES
        print(f"Using built-in {len(cases)} test cases")

    if args.ids:
        ids = {int(x.strip()) for x in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in ids]

    out = Path(args.output) if args.output else OUT_PATH

    # Resume: load existing
    existing: dict[int, dict] = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for rec in json.load(f):
                existing[rec["id"]] = rec
        print(f"Resuming: {len(existing)} already done, will skip")

    results = dict(existing)
    n_total = len(cases)
    n_done = 0
    n_skip = 0

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        if cid in results:
            n_skip += 1
            print(f"[{i}/{n_total}] Case {cid} — skip", flush=True)
            continue

        cat = map_cat(case["category"])
        print(f"\n[{i}/{n_total}] Case {cid} ({case['category']}) [{cat}]", flush=True)
        print(f"  Q: {case['question'][:80]}", flush=True)

        try:
            if cat == "Type1":
                prom, refs = gen_type1(client, case)
            elif cat == "Type2":
                prom, refs = gen_type2(client, case, use_chroma=not args.no_chromadb)
            else:
                prom, refs = gen_type3(client, case)

            rec = {
                "id": cid,
                "category": case["category"],
                "question": case["question"],
                "prompt": prom,
                "answer": refs,
            }
            results[cid] = rec
            n_done += 1

            for j, ref in enumerate(refs, 1):
                print(f"  [{j}] {ref[:90]}", flush=True)

            # Save incrementally (safe to interrupt)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(sorted(results.values(), key=lambda x: x["id"]),
                          f, indent=2, ensure_ascii=False)

            time.sleep(0.3)   # avoid rate-limit bursts

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    print(f"\n{'='*50}")
    print(f"Done: {n_done} generated, {n_skip} skipped, {len(results)} total")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
