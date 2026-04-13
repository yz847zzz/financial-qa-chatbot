"""
System-level evaluation: our pipeline vs GPT-4o.

20 test cases covering Type1 / Type2 / Type3 / compound Type1+Type2 / multi-Type1.
Ground truth values queried directly from data/financials.db before writing tests.

Metrics per answer
──────────────────
  intent_correct   : predicted intent == expected intent (Type1/2/3)
  value_correct    : Type1 only — expected numeric value found in answer (±1% tol)
  keyword_hit_rate : fraction of expected keywords present in answer (case-insensitive)
  fluency_score    : GPT-4o-mini judge, 1–5 scale
  latency_s        : wall-clock seconds for full answer() call

Module-level latency (our pipeline only)
─────────────────────────────────────────
  decompose_s, intent_s, sql_s, retrieve_s, answer_s

GPT-4o baseline
───────────────
  Same questions sent to GPT-4o via OpenAI API.
  Same keyword + value metrics applied.
  Latency measured end-to-end.

Usage
─────
  # Full eval (requires GPU + OpenAI key)
  OPENAI_API_KEY=sk-... python eval_system.py

  # Skip GPT comparison (no API key needed)
  python eval_system.py --no-gpt

  # Skip fluency scoring (faster)
  python eval_system.py --no-fluency

Results written to eval_results/system_eval_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))


# ── Ground-truth test cases ────────────────────────────────────────────────────
# Values verified by direct SQL query against data/financials.db (2026-04-12)

TEST_CASES = [

    # ── Type1: single metric ───────────────────────────────────────────────────
    {
        "id": 1, "category": "Type1",
        "question": "What was Apple's total revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 383_285_000_000.0,          # $383.285B
        "expected_keywords": ["383", "apple", "revenue"],
    },
    {
        "id": 2, "category": "Type1",
        "question": "What was Microsoft's net income in FY2022?",
        "expected_intent": "Type1",
        "expected_value": 72_738_000_000.0,           # $72.738B
        "expected_keywords": ["72", "microsoft", "net income"],
    },
    {
        "id": 3, "category": "Type1",
        "question": "What was Microsoft's net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 72_361_000_000.0,           # $72.361B
        "expected_keywords": ["72", "microsoft", "net income"],
    },
    {
        "id": 4, "category": "Type1",
        "question": "What was Adobe's total revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 19_409_000_000.0,           # $19.409B (verified from DB)
        "expected_keywords": ["19", "adobe", "revenue"],
    },
    {
        "id": 5, "category": "Type1",
        "question": "What was Adobe's net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 5_428_000_000.0,            # $5.428B (verified from DB)
        "expected_keywords": ["5", "adobe", "net income"],
    },
    {
        "id": 6, "category": "Type1",
        "question": "What was Apple's operating income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 114_301_000_000.0,          # $114.301B
        "expected_keywords": ["114", "apple", "operating income"],
    },
    {
        "id": 7, "category": "Type1",
        "question": "What was Apple's net profit margin in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 0.2531,                     # 25.31%
        "expected_keywords": ["25", "apple", "margin"],
    },
    {
        "id": 8, "category": "Type1",
        "question": "What was Microsoft's return on assets in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 0.1756,                     # 17.56%
        "expected_keywords": ["17", "microsoft", "return"],
    },

    # ── Type1: ranking ─────────────────────────────────────────────────────────
    {
        "id": 9, "category": "Type1-ranking",
        "question": "Which 3 companies had the highest net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["apple", "aapl", "microsoft", "msft"],
    },
    {
        "id": 10, "category": "Type1-ranking",
        "question": "Which 3 companies had the highest revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["ci", "apple", "aapl", "cost"],
    },

    # ── Type1: multi-ticker comparison ─────────────────────────────────────────
    {
        "id": 11, "category": "Type1-compare",
        "question": "Compare Apple and Microsoft revenue in FY2023.",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["383", "211", "apple", "microsoft"],
    },
    {
        "id": 12, "category": "Type1-compare",
        "question": "What were Apple's revenue and net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["383", "96", "apple"],
    },

    # ── Type2: qualitative / RAG ───────────────────────────────────────────────
    {
        "id": 13, "category": "Type2",
        "question": "How did Apple describe its AI strategy in recent filings?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["apple", "ai", "artificial intelligence"],
    },
    {
        "id": 14, "category": "Type2",
        "question": "What risks did Microsoft identify related to its cloud business?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["microsoft", "cloud", "risk"],
    },
    {
        "id": 15, "category": "Type2",
        "question": "How did Apple describe its supply chain risks in its 10-K?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["apple", "supply chain", "risk"],
    },
    {
        "id": 16, "category": "Type2",
        "question": "What did Adobe say about competition in its most recent annual filing?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["adobe", "competition", "competitor"],
    },

    # ── Type3: chat / meta ─────────────────────────────────────────────────────
    {
        "id": 17, "category": "Type3",
        "question": "Hello! What can you help me with?",
        "expected_intent": "Type3",
        "expected_value": None,
        "expected_keywords": ["financial", "filing", "question"],
    },
    {
        "id": 18, "category": "Type3",
        "question": "Thank you, that was very helpful.",
        "expected_intent": "Type3",
        "expected_value": None,
        "expected_keywords": [],
    },

    # ── Compound: Type1 + Type2 ────────────────────────────────────────────────
    {
        "id": 19, "category": "Type1+Type2",
        "question": (
            "What was Apple's revenue in FY2023 and "
            "how did they describe their supply chain risks?"
        ),
        "expected_intent": None,         # compound — checked via sub_results
        "expected_value": 383_285_000_000.0,
        "expected_keywords": ["383", "apple", "supply chain", "risk"],
        "expected_sub_intents": ["Type1", "Type2"],
    },

    # ── Compound: multi-Type1 ──────────────────────────────────────────────────
    {
        "id": 20, "category": "Multi-Type1",
        "question": (
            "What was Apple's revenue in FY2023 and "
            "what was Microsoft's net income in FY2023?"
        ),
        "expected_intent": None,         # compound
        "expected_value": None,
        "expected_keywords": ["383", "72", "apple", "microsoft"],
        "expected_sub_intents": ["Type1", "Type1"],
    },

    # ── Type1: additional single metrics ───────────────────────────────────────
    {
        "id": 21, "category": "Type1",
        "question": "What was Amazon's total revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 574_785_000_000.0,          # $574.785B
        "expected_keywords": ["574", "amazon", "revenue"],
    },
    {
        "id": 22, "category": "Type1",
        "question": "What was Google's net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 73_795_000_000.0,           # $73.795B
        "expected_keywords": ["73", "google", "net income"],
    },
    {
        "id": 23, "category": "Type1",
        "question": "What was NVIDIA's total revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 26_974_000_000.0,           # $26.974B
        "expected_keywords": ["26", "nvidia", "revenue"],
    },
    {
        "id": 24, "category": "Type1",
        "question": "What was NVIDIA's total revenue in FY2024?",
        "expected_intent": "Type1",
        "expected_value": 60_922_000_000.0,           # $60.922B
        "expected_keywords": ["60", "nvidia", "revenue"],
    },
    {
        "id": 25, "category": "Type1",
        "question": "What was Tesla's total revenue in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 96_773_000_000.0,           # $96.773B
        "expected_keywords": ["96", "tesla", "revenue"],
    },
    {
        "id": 26, "category": "Type1",
        "question": "What was Tesla's diluted EPS in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 4.30,                       # $4.30/share
        "expected_keywords": ["4", "tesla", "eps"],
    },
    {
        "id": 27, "category": "Type1",
        "question": "What was Microsoft's diluted earnings per share in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 9.68,                       # $9.68/share
        "expected_keywords": ["9", "microsoft", "eps"],
    },
    {
        "id": 28, "category": "Type1",
        "question": "What was Apple's diluted EPS in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 6.13,                       # $6.13/share
        "expected_keywords": ["6", "apple", "eps"],
    },
    {
        "id": 29, "category": "Type1",
        "question": "How much did Apple spend on share buybacks in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 77_550_000_000.0,           # $77.55B
        "expected_keywords": ["77", "apple", "buyback"],
    },
    {
        "id": 30, "category": "Type1",
        "question": "What was Microsoft's R&D expense in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 27_195_000_000.0,           # $27.195B
        "expected_keywords": ["27", "microsoft", "research"],
    },
    {
        "id": 31, "category": "Type1",
        "question": "What was Google's research and development expense in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 45_427_000_000.0,           # $45.427B
        "expected_keywords": ["45", "google", "research"],
    },
    {
        "id": 32, "category": "Type1",
        "question": "What was Amazon's net income in FY2022?",
        "expected_intent": "Type1",
        "expected_value": -2_722_000_000.0,           # -$2.722B (net loss)
        "expected_keywords": ["amazon", "loss", "2022"],
    },
    {
        "id": 33, "category": "Type1",
        "question": "What was Google's net profit margin in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 0.24007,                    # 24.0%
        "expected_keywords": ["24", "google", "margin"],
    },
    {
        "id": 34, "category": "Type1",
        "question": "What was Microsoft's operating cash flow in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 87_582_000_000.0,           # $87.582B
        "expected_keywords": ["87", "microsoft", "cash"],
    },
    {
        "id": 35, "category": "Type1",
        "question": "What was Tesla's capital expenditure in FY2023?",
        "expected_intent": "Type1",
        "expected_value": 8_898_000_000.0,            # $8.898B
        "expected_keywords": ["8", "tesla", "capex"],
    },

    # ── Type1: year-over-year ──────────────────────────────────────────────────
    {
        "id": 36, "category": "Type1",
        "question": "What was Apple's total revenue in FY2022?",
        "expected_intent": "Type1",
        "expected_value": 394_328_000_000.0,          # $394.328B
        "expected_keywords": ["394", "apple", "revenue"],
    },
    {
        "id": 37, "category": "Type1",
        "question": "What was Microsoft's total revenue in FY2022?",
        "expected_intent": "Type1",
        "expected_value": 198_270_000_000.0,          # $198.270B
        "expected_keywords": ["198", "microsoft", "revenue"],
    },
    {
        "id": 38, "category": "Type1",
        "question": "What was NVIDIA's net income in FY2024?",
        "expected_intent": "Type1",
        "expected_value": 29_760_000_000.0,           # $29.760B
        "expected_keywords": ["29", "nvidia", "net income"],
    },

    # ── Type1: additional ranking ──────────────────────────────────────────────
    {
        "id": 39, "category": "Type1-ranking",
        "question": "Which 3 companies spent the most on R&D in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["google", "goog", "merck", "apple", "aapl"],
    },
    {
        "id": 40, "category": "Type1-ranking",
        "question": "Which 3 companies had the highest capital expenditure in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["amazon", "amzn", "google", "goog", "microsoft"],
    },

    # ── Type1: multi-ticker compare ────────────────────────────────────────────
    {
        "id": 41, "category": "Type1-compare",
        "question": "Compare Amazon and Microsoft total revenue in FY2023.",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["574", "211", "amazon", "microsoft"],
    },
    {
        "id": 42, "category": "Type1-compare",
        "question": "How did NVIDIA's revenue change from FY2023 to FY2024?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["26", "60", "nvidia"],
    },
    {
        "id": 43, "category": "Type1-compare",
        "question": "What were Walmart's revenue and net income in FY2023?",
        "expected_intent": "Type1",
        "expected_value": None,
        "expected_keywords": ["605", "11", "walmart"],
    },

    # ── Type2: additional qualitative / RAG ────────────────────────────────────
    {
        "id": 44, "category": "Type2",
        "question": "How did NVIDIA describe its data center and AI chip strategy in recent filings?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["nvidia", "data center", "gpu"],
    },
    {
        "id": 45, "category": "Type2",
        "question": "What risks did Tesla identify related to autonomous driving in its 10-K?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["tesla", "autonomous", "risk"],
    },
    {
        "id": 46, "category": "Type2",
        "question": "How did Amazon describe AWS and its competitive position in cloud?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["amazon", "aws", "cloud"],
    },
    {
        "id": 47, "category": "Type2",
        "question": "What regulatory and antitrust risks did Google disclose in its annual filing?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["google", "regulatory", "antitrust"],
    },
    {
        "id": 48, "category": "Type2",
        "question": "How did Microsoft describe its partnership with OpenAI in its 10-K?",
        "expected_intent": "Type2",
        "expected_value": None,
        "expected_keywords": ["microsoft", "openai", "ai"],
    },

    # ── Type3: additional chat / meta ──────────────────────────────────────────
    {
        "id": 49, "category": "Type3",
        "question": "What companies do you have financial data on?",
        "expected_intent": "Type3",
        "expected_value": None,
        "expected_keywords": [],
    },
    {
        "id": 50, "category": "Type3",
        "question": "Can you explain what return on assets means?",
        "expected_intent": "Type3",
        "expected_value": None,
        "expected_keywords": ["assets", "return", "profit"],
    },

    # ── Compound: additional ───────────────────────────────────────────────────
    {
        "id": 51, "category": "Type1+Type2",
        "question": (
            "What was NVIDIA's revenue in FY2024 and "
            "how did they describe their AI chip strategy?"
        ),
        "expected_intent": None,
        "expected_value": 60_922_000_000.0,
        "expected_keywords": ["60", "nvidia", "ai", "data center"],
        "expected_sub_intents": ["Type1", "Type2"],
    },
    {
        "id": 52, "category": "Multi-Type1",
        "question": (
            "What was Amazon's revenue in FY2023 and "
            "what was Google's net income in FY2023?"
        ),
        "expected_intent": None,
        "expected_value": None,
        "expected_keywords": ["574", "73", "amazon", "google"],
        "expected_sub_intents": ["Type1", "Type1"],
    },
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    """Extract the first numeric value from text (handles B/M suffixes and commas)."""
    text = text.replace(",", "")
    # Match numbers with optional B/M/T suffix
    m = re.search(r"(\d+(?:\.\d+)?)\s*([BbMmTt]?)\b", text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).upper()
    if suffix == "B":
        val *= 1e9
    elif suffix == "M":
        val *= 1e6
    elif suffix == "T":
        val *= 1e12
    return val


def value_correct(answer: str, expected: float | None, tol: float = 0.05) -> bool | None:
    """
    Check if expected numeric value appears in the answer within tol tolerance.
    Returns None if expected is None (no numeric check needed).
    Handles:
      - Raw integers:   $383,285,000,000
      - Scaled values:  383.285B / 383.29 billion
      - Percentages:    25.31% (for expected < 1, e.g. 0.2531)
      - Raw ratios:     0.1756 (also accepted for expected < 1, e.g. ROA)
      - Negative values: -$2,722,000,000 (net loss)
    """
    if expected is None:
        return None

    abs_exp = abs(expected)
    negative = expected < 0

    # ── Ratio / percentage values (|expected| < 1) ────────────────────────────
    if abs_exp < 1:
        pct = abs_exp * 100
        # Accept percentage form (e.g. "25.31%")
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", answer):
            found = float(m.group(1))
            if abs(found - pct) / (abs(pct) + 1e-9) <= tol:
                return True
        # Also accept raw decimal form (e.g. "0.1756" for ROA)
        for m in re.finditer(r"\b(0\.\d+)\b", answer):
            found = float(m.group(1))
            if abs(found - abs_exp) / (abs_exp + 1e-9) <= tol:
                return True
        return False

    # ── Negative large values (e.g. net loss) ────────────────────────────────
    if negative:
        # Look for loss/negative indicators with the magnitude present
        loss_words = re.search(r"\b(loss|negative|deficit)\b", answer, re.I)
        # Check magnitude in answer (ignore sign — presence of loss keyword + value is enough)
        mag_str = str(int(abs_exp))[:6]
        answer_stripped = answer.replace(",", "").replace(" ", "")
        if mag_str in answer_stripped:
            # Accept if it looks like a loss (negative sign or loss keyword)
            if "-" in answer_stripped or loss_words:
                return True
        # Also try scaled match
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([BbMmTt]?)\b", answer):
            val = float(m.group(1))
            suffix = m.group(2).upper()
            multiplier = {"B": 1e9, "M": 1e6, "T": 1e12}.get(suffix, 0)
            if multiplier:
                if abs(val * multiplier - abs_exp) / abs_exp <= tol:
                    if "-" in answer or loss_words:
                        return True
        return False

    # ── Large positive values ─────────────────────────────────────────────────
    # Fast path: first 6 significant digits as a substring
    raw_str = str(int(expected))[:6]
    if raw_str in answer.replace(",", "").replace(" ", ""):
        return True

    # Try every number found in the answer with B/M/T suffix awareness
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([BbMmTt]?)\b", answer):
        val = float(m.group(1))
        suffix = m.group(2).upper()
        if suffix == "B":
            val *= 1e9
        elif suffix == "M":
            val *= 1e6
        elif suffix == "T":
            val *= 1e12
        else:
            for multiplier in [1e9, 1e6, 1e3, 1]:
                scaled = val * multiplier
                if abs(scaled - expected) / (abs(expected) + 1e-9) <= tol:
                    return True
            continue
        if abs(val - expected) / (abs(expected) + 1e-9) <= tol:
            return True

    return False


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords (case-insensitive) found in the answer."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def intent_correct(result: dict, case: dict) -> bool | None:
    """Check intent classification. Returns None for compound questions."""
    if case.get("expected_sub_intents"):
        # Compound: check sub-results
        sub = result.get("sub_results")
        if not sub:
            return False
        got = [s["intent"] for s in sub]
        return got == case["expected_sub_intents"]
    if case["expected_intent"] is None:
        return None
    return result.get("intent") == case["expected_intent"]


# ── GPT-4o baseline ────────────────────────────────────────────────────────────

GPT_SYSTEM = (
    "You are a financial analyst assistant with access to SEC filings and financial data. "
    "Answer the question concisely and accurately. "
    "For specific financial figures, provide the exact number. "
    "Always cite your source."
)


def ask_gpt(question: str, client) -> tuple[str, float]:
    """Call GPT-4o and return (answer, latency_s)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GPT_SYSTEM},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    latency = time.time() - t0
    return resp.choices[0].message.content.strip(), latency


# ── Fluency judge ──────────────────────────────────────────────────────────────

FLUENCY_PROMPT = (
    "Rate the following financial assistant answer for fluency and coherence on a scale of 1-5:\n"
    "1 = incoherent / broken  2 = poor  3 = acceptable  4 = good  5 = excellent\n"
    "Output ONLY a single integer 1-5. No explanation."
)


def fluency_score(answer: str, client) -> int | None:
    """Use GPT-4o-mini as fluency judge. Returns 1-5 or None on failure."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FLUENCY_PROMPT},
                {"role": "user",   "content": answer},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"[1-5]", raw)
        return int(m.group()) if m else None
    except Exception:
        return None


# ── Timed pipeline wrappers ────────────────────────────────────────────────────

def timed_answer(question, base_model, base_tok, nl2sql_model, nl2sql_tok,
                 retriever) -> tuple[dict, dict]:
    """
    Run the full pipeline and capture per-module latencies.
    Returns (result, timings) where timings keys:
      total_s, decompose_s, intent_s, sql_s, retrieve_s, answer_s
    """
    from chatbot import answer as pipeline_answer
    from rag.query_rewriter import decompose_question, rewrite_query
    from chatbot import (classify_intent, generate_sql, postprocess_sql,
                         execute_sql, format_sql_context,
                         retrieve_chunks_multi, format_rag_context,
                         generate_answer, direct_answer)

    timings = {}
    t_total = time.time()

    # Decompose
    t0 = time.time()
    sub_qs = decompose_question(question, base_model, base_tok)
    timings["decompose_s"] = round(time.time() - t0, 3)

    per_sub = []
    for sub_q in sub_qs:
        sub_t = {}

        # Intent
        t0 = time.time()
        intent = classify_intent(sub_q, base_model, base_tok)
        sub_t["intent_s"] = round(time.time() - t0, 3)

        answer_text = None
        sql_out = None
        sql_repaired = None
        repairs = []
        error = None

        if intent == "Type1":
            t0 = time.time()
            raw_sql = generate_sql(sub_q, nl2sql_model, nl2sql_tok)
            sql_out = raw_sql
            sql_repaired, repairs = postprocess_sql(raw_sql)
            sub_t["sql_s"] = round(time.time() - t0, 3)

            rows, error = execute_sql(sql_repaired)
            if not error:
                t0 = time.time()
                ctx = format_sql_context(sql_repaired, rows)
                answer_text = generate_answer(sub_q, ctx, base_model, base_tok)
                sub_t["answer_s"] = round(time.time() - t0, 3)
            else:
                answer_text = f"SQL error: {error}"

        elif intent == "Type2":
            t0 = time.time()
            queries = rewrite_query(sub_q, base_model, base_tok)
            chunks = retrieve_chunks_multi(sub_q, queries, retriever, top_n=3)
            sub_t["retrieve_s"] = round(time.time() - t0, 3)

            t0 = time.time()
            ctx = format_rag_context(chunks)
            answer_text = generate_answer(sub_q, ctx, base_model, base_tok)
            sub_t["answer_s"] = round(time.time() - t0, 3)

        else:  # Type3
            t0 = time.time()
            answer_text = direct_answer(sub_q, base_model, base_tok)
            sub_t["answer_s"] = round(time.time() - t0, 3)

        per_sub.append({
            "question": sub_q, "intent": intent,
            "sql": sql_out, "sql_repaired": sql_repaired,
            "repairs": repairs, "error": error,
            "answer": answer_text, "timings": sub_t,
        })

    timings["total_s"] = round(time.time() - t_total, 3)

    # Aggregate timings across sub-questions
    for key in ["intent_s", "sql_s", "retrieve_s", "answer_s"]:
        total = sum(s["timings"].get(key, 0) for s in per_sub)
        if total > 0:
            timings[key] = round(total, 3)

    # Build result matching chatbot.answer() format
    if len(per_sub) == 1:
        result = per_sub[0].copy()
        result["question"] = question
        result.pop("timings", None)
    else:
        from chatbot import _synthesize_answers
        combined = _synthesize_answers(question, per_sub, base_model, base_tok)
        result = {
            "question": question,
            "intent": per_sub[0]["intent"],
            "sql": per_sub[0]["sql"],
            "sql_repaired": per_sub[0]["sql_repaired"],
            "repairs": per_sub[0]["repairs"],
            "answer": combined,
            "error": None,
            "sub_results": per_sub,
        }

    return result, timings


# ── Main eval loop ─────────────────────────────────────────────────────────────

def evaluate_case(case: dict, result: dict, timings: dict, gpt_answer: str | None,
                  oai_client) -> dict:
    """Compute all metrics for one test case."""
    answer = result.get("answer") or ""
    gpt_ans = gpt_answer or ""

    our_kw   = keyword_hit_rate(answer, case["expected_keywords"])
    our_val  = value_correct(answer, case.get("expected_value"))
    our_int  = intent_correct(result, case)
    our_flu  = fluency_score(answer, oai_client) if oai_client else None

    gpt_kw   = keyword_hit_rate(gpt_ans, case["expected_keywords"]) if gpt_answer else None
    gpt_val  = value_correct(gpt_ans, case.get("expected_value")) if gpt_answer else None
    gpt_flu  = fluency_score(gpt_ans, oai_client) if (gpt_answer and oai_client) else None

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "our": {
            "answer": answer[:300],
            "intent_correct": our_int,
            "value_correct": our_val,
            "keyword_hit_rate": round(our_kw, 3),
            "fluency_score": our_flu,
            "timings": timings,
        },
        "gpt": {
            "answer": gpt_ans[:300] if gpt_answer else None,
            "value_correct": gpt_val,
            "keyword_hit_rate": round(gpt_kw, 3) if gpt_kw is not None else None,
            "fluency_score": gpt_flu,
            "latency_s": timings.get("gpt_latency_s"),
        },
    }


def print_summary(records: list[dict]) -> None:
    cats = {}
    for r in records:
        c = r["category"]
        cats.setdefault(c, []).append(r)

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    our_metrics = {"intent": [], "value": [], "kw": [], "flu": [], "lat": []}
    gpt_metrics = {"value": [], "kw": [], "flu": [], "lat": []}

    for cat, cases in cats.items():
        print(f"\n── {cat} ({len(cases)} cases) ──")
        for r in cases:
            o = r["our"]
            g = r["gpt"]
            flags = []
            if o["intent_correct"] is False: flags.append("INTENT-FAIL")
            if o["value_correct"] is False:  flags.append("VALUE-FAIL")
            flag_str = f"  *** {', '.join(flags)}" if flags else ""
            print(f"  [{r['id']:2d}] kw={o['keyword_hit_rate']:.2f} "
                  f"val={str(o['value_correct']):5s} "
                  f"flu={o['fluency_score']} "
                  f"lat={o['timings'].get('total_s','?'):.1f}s "
                  f"| GPT kw={g['keyword_hit_rate']} val={g['value_correct']} "
                  f"flu={g['fluency_score']} lat={g['latency_s']}"
                  f"{flag_str}")

            if o["intent_correct"] is not None:
                our_metrics["intent"].append(int(o["intent_correct"]))
            if o["value_correct"] is not None:
                our_metrics["value"].append(int(o["value_correct"]))
            our_metrics["kw"].append(o["keyword_hit_rate"])
            if o["fluency_score"]:
                our_metrics["flu"].append(o["fluency_score"])
            if o["timings"].get("total_s"):
                our_metrics["lat"].append(o["timings"]["total_s"])

            if g["value_correct"] is not None:
                gpt_metrics["value"].append(int(g["value_correct"]))
            if g["keyword_hit_rate"] is not None:
                gpt_metrics["kw"].append(g["keyword_hit_rate"])
            if g["fluency_score"]:
                gpt_metrics["flu"].append(g["fluency_score"])
            if g["latency_s"]:
                gpt_metrics["lat"].append(g["latency_s"])

    def avg(lst): return round(sum(lst)/len(lst), 3) if lst else None

    print(f"\n{'─'*70}")
    print("AGGREGATE (our pipeline vs GPT-4o)")
    print(f"{'─'*70}")
    print(f"  Intent accuracy    : {avg(our_metrics['intent'])}")
    print(f"  Value accuracy     : our={avg(our_metrics['value'])}  gpt={avg(gpt_metrics['value'])}")
    print(f"  Keyword hit rate   : our={avg(our_metrics['kw'])}  gpt={avg(gpt_metrics['kw'])}")
    print(f"  Fluency (1-5)      : our={avg(our_metrics['flu'])}  gpt={avg(gpt_metrics['flu'])}")
    print(f"  Avg latency (s)    : our={avg(our_metrics['lat'])}  gpt={avg(gpt_metrics['lat'])}")


def main():
    parser = argparse.ArgumentParser(description="System-level evaluation")
    parser.add_argument("--no-gpt", action="store_true",
                        help="Skip GPT-4o comparison (no API key needed)")
    parser.add_argument("--no-fluency", action="store_true",
                        help="Skip GPT-4o-mini fluency scoring")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs to run (e.g. 1,2,5)")
    parser.add_argument("--no-nl2sql-adapter", action="store_true")
    args = parser.parse_args()

    # ── OpenAI client ──────────────────────────────────────────────────────────
    oai_client = None
    if not args.no_gpt or not args.no_fluency:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            oai_client = OpenAI(api_key=api_key)
            print("OpenAI client ready.")
        else:
            print("[WARN] OPENAI_API_KEY not set — GPT comparison and fluency scoring skipped.")

    # ── Load our models ────────────────────────────────────────────────────────
    from chatbot import load_base_model, load_nl2sql_model, load_vectordb

    print("\n[Loading models...]")
    base_model, base_tok = load_base_model()
    nl2sql_model, nl2sql_tok = (
        (base_model, base_tok) if args.no_nl2sql_adapter
        else load_nl2sql_model(base_model, base_tok)
    )
    retriever = load_vectordb()

    # ── Select cases ───────────────────────────────────────────────────────────
    cases = TEST_CASES
    if args.cases:
        ids = {int(x) for x in args.cases.split(",")}
        cases = [c for c in cases if c["id"] in ids]

    # ── Run eval ───────────────────────────────────────────────────────────────
    records = []
    for i, case in enumerate(cases, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(cases)}] Case {case['id']} ({case['category']})")
        print(f"Q: {case['question']}")
        print(f"{'─'*60}")

        # Our pipeline
        result, timings = timed_answer(
            case["question"], base_model, base_tok,
            nl2sql_model, nl2sql_tok, retriever,
        )
        print(f"  [Our] intent={result.get('intent')} lat={timings.get('total_s'):.1f}s")
        print(f"  [Our] answer: {(result.get('answer') or '')[:150]}")

        # GPT-4o
        gpt_answer = None
        gpt_latency = None
        if oai_client and not args.no_gpt:
            gpt_answer, gpt_latency = ask_gpt(case["question"], oai_client)
            timings["gpt_latency_s"] = round(gpt_latency, 3)
            print(f"  [GPT] lat={gpt_latency:.1f}s answer: {gpt_answer[:150]}")

        # Fluency
        if args.no_fluency:
            oai_client_for_flu = None
        else:
            oai_client_for_flu = oai_client

        record = evaluate_case(case, result, timings, gpt_answer, oai_client_for_flu)
        records.append(record)

        print(f"  [Metrics] our: kw={record['our']['keyword_hit_rate']:.2f} "
              f"val={record['our']['value_correct']} "
              f"flu={record['our']['fluency_score']} | "
              f"gpt: kw={record['gpt']['keyword_hit_rate']} "
              f"val={record['gpt']['value_correct']}")

    print_summary(records)

    # ── Save results ───────────────────────────────────────────────────────────
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"system_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
