"""
Pipeline demo — one question per intent type, full trace of every stage.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from chatbot import (
    load_base_model, load_nl2sql_model, load_vectordb,
    classify_intent, generate_sql, postprocess_sql,
    execute_sql, format_sql_result, answer_from_sql,
    retrieve_chunks, format_context, answer_from_context,
    direct_answer,
)

SEP  = "=" * 68
SEP2 = "-" * 68

QUESTIONS = [
    ("Type1", "What was Apple's revenue in FY2023?"),
    ("Type2", "How did Apple describe its AI strategy in recent filings?"),
    ("Type3", "Hello, what can you help me with?"),
]

def hr(title=""):
    if title:
        print(f"\n  ┌─ {title} {'─'*(60-len(title))}")
    else:
        print(f"  └{'─'*63}")

def show(label, value, indent=4):
    prefix = " " * indent
    lines = str(value).split("\n")
    print(f"{prefix}{label}: {lines[0]}")
    for l in lines[1:]:
        print(f"{prefix}{'':>{len(label)+2}}{l}")

# ── load ──────────────────────────────────────────────────────────────────────
print(SEP)
print("  Financial QA — Pipeline Demo")
print(SEP)
print("\n[Loading models...]")
t0 = time.time()
base_model, base_tok     = load_base_model()
nl2sql_model, nl2sql_tok = load_nl2sql_model(base_model, base_tok)
retriever                = load_vectordb()
print(f"Ready in {time.time()-t0:.1f}s\n")

# ── run each question ─────────────────────────────────────────────────────────
for expected_type, question in QUESTIONS:
    print(SEP)
    print(f"  QUESTION  : {question}")
    print(SEP)

    t_start = time.time()

    # ── Step 1: Intent ────────────────────────────────────────────
    hr("Step 1 · Intent Classification")
    intent = classify_intent(question, base_model, base_tok)
    show("Intent", intent)
    hr()

    # ── Step 2: Route ─────────────────────────────────────────────
    if intent == "Type1":
        # NL2SQL
        hr("Step 2 · NL2SQL Generation  (fine-tuned adapter)")
        raw_sql = generate_sql(question, nl2sql_model, nl2sql_tok)
        show("Raw SQL", raw_sql)

        repaired_sql, repairs = postprocess_sql(raw_sql)
        if repairs:
            show("Repairs", repairs)
            show("Fixed SQL", repaired_sql)
        else:
            show("Postprocess", "no changes needed")
        hr()

        hr("Step 3 · SQL Execution  (financials.db → panel table)")
        rows, error = execute_sql(repaired_sql)
        if error:
            show("ERROR", error)
        else:
            show("Rows returned", len(rows))
            show("SQL result", format_sql_result(rows))
        hr()

        hr("Step 4 · Answer Generation  (base Llama)")
        sql_result_str = format_sql_result(rows) if not error else "execution failed"
        final = answer_from_sql(question, repaired_sql, sql_result_str, base_model, base_tok)
        show("Final answer", final)
        hr()

    elif intent == "Type2":
        if retriever is None:
            print("  [VectorDB not available — skipping RAG retrieval]")
            final = "VectorDB not loaded."
        else:
            hr("Step 2 · Hybrid Retrieval  (BM25 + Dense → RRF → Cross-encoder)")
            chunks = retrieve_chunks(question, retriever, top_n=3)
            for i, c in enumerate(chunks, 1):
                m = c.metadata
                src = f"{m.get('ticker','?')} {m.get('filing_type','?')} {m.get('date','?')}"
                show(f"Chunk {i}", f"[{src}]  rerank={c.rerank_score:.3f}  rrf={c.rrf_score:.4f}")
                show(f"  text", c.text[:120] + "...")
            hr()

            hr("Step 3 · Answer Generation  (base Llama + context)")
            context = format_context(chunks)
            final = answer_from_context(question, context, base_model, base_tok)
            show("Final answer", final)
            hr()

    else:  # Type3
        hr("Step 2 · Direct Answer  (base Llama, no retrieval)")
        final = direct_answer(question, base_model, base_tok)
        show("Final answer", final)
        hr()

    print(f"\n  Total time: {time.time()-t_start:.1f}s\n")

print(SEP)
print("  Demo complete.")
print(SEP)
