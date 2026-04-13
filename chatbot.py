"""
Financial QA Chatbot — simple interactive pipeline.

Architecture:
  user question
    → decompose_question()           → sub-questions (or [question] if not compound)
    → per sub-question:
        → classify_intent()          → Type1 / Type2 / Type3
        → Type1: generate_sql() → execute_sql() → generate_answer(context=sql+rows)
        → Type2: rewrite_query() → retrieve_chunks_multi() → generate_answer(context=chunks)
        → Type3: direct_answer()
    → if multiple sub-questions: synthesize answers

  generate_answer() is a single unified module for both Type1 and Type2.
  The source is always cited — baked into the context string by the caller,
  not toggled by the prompt.

All LLM calls are isolated in functions marked [VLLM_SWAP].
To migrate to vLLM (Linux/WSL2 only):
    1. Start the server:  bash deployment/scripts/start_server.sh
    2. Add 3 lines near the top of this file:
           from deployment.api.client import VLLMClient as _C
           _vllm = _C()
           llm_generate = _vllm.llm_generate
    3. In generate_sql(), change:
           return llm_generate(model, tokenizer, messages, max_new_tokens=200)
       to:
           return _vllm.generate_sql_vllm(messages)
    That's it. All other call sites use llm_generate() unchanged.
    The server handles adapter routing (SGMV batching) transparently.

Usage:
    python chatbot.py                        # interactive REPL
    python chatbot.py --question "..."       # single question
    python chatbot.py --no-nl2sql-adapter    # skip loading adapter (faster cold start)

Environment:
    HF_HUB_CACHE — path to cached HF models (default: models/llama)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH      = ROOT / "data" / "financials.db"
VECTORDB_DIR = ROOT / "data" / "vectordb"
MODEL_DIR    = ROOT / "models" / "llama" / "models--meta-llama--Llama-3.2-3B-Instruct"
ADAPTER_DIR  = ROOT / "models" / "nl2sql"

os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "llama"))


# ── Model loading (done once at startup) ──────────────────────────────────────

def load_base_model():
    """Load Llama-3.2-3B-Instruct in 4-bit NF4."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    # Find the snapshot directory inside the HF cache
    snapshots = list(MODEL_DIR.glob("snapshots/*/"))
    if not snapshots:
        raise FileNotFoundError(
            f"No model snapshot found in {MODEL_DIR}.\n"
            "Run: huggingface-cli download meta-llama/Llama-3.2-3B-Instruct"
        )
    model_path = str(sorted(snapshots)[-1])  # latest snapshot

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base model from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Base model loaded.", flush=True)
    return model, tokenizer


def load_nl2sql_model(base_model, tokenizer):
    """Overlay the NL2SQL LoRA adapter on the base model."""
    from peft import PeftModel

    if not ADAPTER_DIR.exists():
        print(f"[WARN] NL2SQL adapter not found at {ADAPTER_DIR} — using base model for SQL generation.")
        return base_model, tokenizer

    print(f"Loading NL2SQL adapter from {ADAPTER_DIR} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    print("NL2SQL adapter loaded.", flush=True)
    return model, tokenizer


def load_vectordb():
    """
    Load ChromaDB collection and build the HybridRetriever (BM25 + dense + cross-encoder).
    Returns a HybridRetriever instance, or None if VectorDB is unavailable.
    """
    try:
        import chromadb

        sys.path.insert(0, str(ROOT / "deployment"))
        from rag.retriever import HybridRetriever  # noqa: F401 (also used in retrieve_chunks_multi)

        client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
        # Open without embedding_function to avoid ChromaDB 1.x name-conflict error.
        # HybridRetriever embeds queries internally via SentenceTransformer.
        collection = client.get_collection(name="financial_docs")
        print(f"VectorDB loaded: {collection.count():,} chunks.", flush=True)

        # recall_k=20 per side → up to 40 candidates → reranked by cross-encoder → MMR
        retriever = HybridRetriever(collection, recall_k=20)
        return retriever
    except Exception as e:
        print(f"[WARN] VectorDB not available: {e}")
        return None


# ── [VLLM_SWAP] LLM generation helper ─────────────────────────────────────────
# Replace this function body with:
#   import openai
#   client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="none")
#   resp = client.chat.completions.create(model="llama3", messages=messages,
#                                         max_tokens=max_new_tokens, temperature=0.0)
#   return resp.choices[0].message.content.strip()

def llm_generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    """
    [VLLM_SWAP] Run a chat-format prompt through the local model.
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}

    Two-step tokenisation (apply_chat_template → string, then tokenizer → tensor)
    is more robust across transformers versions than passing return_tensors="pt"
    directly to apply_chat_template, which returns different types in different versions.
    """
    import torch

    # Step 1: render messages to a prompt string
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Step 2: tokenise to tensor
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids      = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_len     = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Step 1: Intent Classification ─────────────────────────────────────────────

INTENT_SYSTEM = (
    "You are a financial query classifier. "
    "Classify the user query into exactly one of: Type1, Type2, Type3.\n"
    "Type1 = exact financial fact (specific number: revenue, EPS, ratio, balance, date-specific)\n"
    "Type2 = vague, qualitative, or analytical question about filings or strategy\n"
    "Type3 = casual chat, greeting, or meta question\n"
    "Output only the label. Do not explain."
)

# Fast keyword-based pre-filter to avoid an LLM call for obvious cases
_TYPE3_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|bye|goodbye|what can you|how are you|"
    r"who are you|what do you do|help me understand what)\b",
    re.I,
)
_TYPE1_PATTERNS = re.compile(
    r"\b(revenue|income|profit|earnings|assets|liabilities|debt|cash|capex|cfo|"
    r"ebit|margin|roa|current ratio|fy20\d{2}|fiscal 20\d{2}|in 20\d{2}|"
    r"how much|how many|what was|what is the|total|balance|ratio)\b",
    re.I,
)


def classify_intent(question: str, model, tokenizer) -> str:
    """
    Returns 'Type1', 'Type2', or 'Type3'.
    Uses keyword shortcuts first, then LLM for ambiguous cases.
    """
    # Fast path: obvious Type3 (chat)
    if _TYPE3_PATTERNS.match(question):
        return "Type3"

    # [VLLM_SWAP] call classify via LLM
    messages = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user",   "content": question},
    ]
    raw = llm_generate(model, tokenizer, messages, max_new_tokens=5)

    # Extract first occurrence of Type1/Type2/Type3
    match = re.search(r"Type[123]", raw)
    if match:
        return match.group()

    # Fallback: keyword heuristic
    if _TYPE1_PATTERNS.search(question):
        return "Type1"
    return "Type2"


# ── Step 2a: Type1 — NL2SQL path ──────────────────────────────────────────────

NL2SQL_SYSTEM = """You have access to a SQLite database with these tables:

panel (ticker TEXT, year TEXT, total_revenue REAL, gross_profit REAL,
       operating_income REAL, net_income REAL, r_and_d REAL,
       interest_expense REAL, interest_income REAL, da REAL,
       cfo REAL, capex REAL, buybacks REAL, dividends_paid REAL,
       cash REAL, total_assets REAL, current_assets REAL,
       current_liabilities REAL, total_liabilities REAL, long_term_debt REAL,
       goodwill REAL, retained_earnings REAL, inventories REAL,
       accounts_payable REAL, eps_diluted REAL,
       current_ratio REAL, net_margin REAL, roa REAL, debt_to_assets REAL)
  - ticker: stock symbol e.g. 'AAPL', 'MSFT'
  - year: fiscal year string e.g. 'FY2023'
  - all monetary columns are in USD; eps_diluted is USD per share
  - buybacks and dividends_paid are stored as positive values

filing_metadata (ticker TEXT, company TEXT, filing_type TEXT, date TEXT, accession_number TEXT)
  - filing_type: '10-K' or '8-K'
  - date: 'YYYY-MM-DD'

Rules:
- Output ONLY a valid SQL SELECT. No explanation.
- Ticker symbols are uppercase.
- Year format is 'FY{YYYY}' e.g. 'FY2023'.
- Use IS NOT NULL to filter missing values."""


def generate_sql(question: str, model, tokenizer) -> str:
    """
    [VLLM_SWAP] Generate SQL from NL question using the NL2SQL adapter.
    In vLLM mode, specify model="nl2sql" (LoRA adapter name in the server config).
    """
    messages = [
        {"role": "system", "content": NL2SQL_SYSTEM},
        {"role": "user",   "content": question},
    ]
    return llm_generate(model, tokenizer, messages, max_new_tokens=200)


def postprocess_sql(sql: str) -> tuple[str, list[str]]:
    """Apply deterministic synonym repair (sql_postprocess.py)."""
    try:
        sys.path.insert(0, str(ROOT / "finetune" / "adapters" / "nl2sql"))
        from sql_postprocess import repair_sql
        repaired, repairs = repair_sql(sql)
        return repaired, repairs
    except ImportError:
        return sql, []


def execute_sql(sql: str) -> tuple[list[dict] | None, str]:
    """
    Execute a SELECT against financials.db.
    Returns (rows, error_message). rows is None on error.
    """
    if not DB_PATH.exists():
        return None, f"Database not found: {DB_PATH}"

    # Safety: only allow SELECT
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return None, "Only SELECT statements are allowed."
    for kw in ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"):
        if kw in stripped:
            return None, f"Disallowed keyword: {kw}"

    try:
        con = sqlite3.connect(str(DB_PATH))
        con.row_factory = sqlite3.Row
        cur = con.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows, ""
    except sqlite3.Error as e:
        return None, str(e)


def format_sql_result(rows: list[dict]) -> str:
    """Format SQL result rows as a readable string for the answer LLM."""
    if not rows:
        return "No data found."
    if len(rows) == 1:
        return ", ".join(f"{k}: {v}" for k, v in rows[0].items() if v is not None)
    # Multiple rows: simple table
    headers = list(rows[0].keys())
    lines = [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
    return "\n".join(lines)


# ── Step 2b: Type2 — RAG path ─────────────────────────────────────────────────

def retrieve_chunks_multi(
    original_question: str,
    queries: list[str],
    retriever,
    top_n: int = 3,
) -> list:
    """
    Multi-query hybrid retrieval.

    Each query in `queries` runs a full BM25 + dense recall pass.
    All candidate pools are merged (dedup by chunk_id, best RRF score wins),
    then reranked by cross-encoder using `original_question` (not sub-queries),
    then MMR-selected.

    Using the original question for reranking keeps the final relevance judgement
    anchored to what the user actually asked, not which sub-query happened to match.
    """
    return retriever.retrieve_multi(queries, rerank_query=original_question, top_n=top_n)


def format_sql_context(sql: str, rows: list[dict]) -> str:
    """Format SQL result as a sourced context block for generate_answer()."""
    return (
        f"Source: financial database (panel table)\n"
        f"SQL: {sql}\n"
        f"Result: {format_sql_result(rows)}"
    )


def format_rag_context(chunks: list) -> str:
    """Format RetrievedChunk list as a numbered, sourced context block for generate_answer()."""
    parts = []
    for i, c in enumerate(chunks, 1):
        meta   = c.metadata
        source = f"{meta.get('ticker','?')} {meta.get('filing_type','?')} {meta.get('date','?')}"
        parts.append(f"[{i}] Source: {source}\n{c.text}")
    return "\n\n".join(parts)


ANSWER_SYSTEM = (
    "You are a financial analyst assistant. "
    "Answer the question based ONLY on the provided context. "
    "Always cite the source at the end of your answer. "
    "For filing excerpts, use the source label e.g. [1], [2]. "
    "For database results, cite 'financial database'. "
    "If the context does not contain enough information, say so clearly."
)


def generate_answer(question: str, context: str, model, tokenizer) -> str:
    """
    [VLLM_SWAP] Single unified answer module for Type1 and Type2.

    The caller is responsible for formatting context with the source included:
      Type1 → format_sql_context(sql, rows)   e.g. "Source: financial database\\nSQL: ...\\nResult: ..."
      Type2 → format_rag_context(chunks)       e.g. "[1] Source: AAPL 10-K 2023-09-30\\n..."
    """
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user",   "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    return llm_generate(model, tokenizer, messages, max_new_tokens=500)


# ── Step 2c: Type3 — Direct answer ────────────────────────────────────────────

DIRECT_SYSTEM = (
    "You are a helpful financial assistant chatbot. "
    "Answer the user's question naturally and concisely. "
    "You specialize in SEC filings, financial metrics, and company analysis."
)


def direct_answer(question: str, model, tokenizer) -> str:
    """[VLLM_SWAP] Direct LLM response for chat/meta questions."""
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user",   "content": question},
    ]
    return llm_generate(model, tokenizer, messages, max_new_tokens=300)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def _answer_single(sub_q: str, base_model, base_tok, nl2sql_model, nl2sql_tok,
                   retriever) -> dict:
    """
    Route a single, focused sub-question through the full pipeline.
    Returns dict with intent, sql, answer, and debug info.
    """
    result = {"question": sub_q, "intent": None, "sql": None,
              "sql_repaired": None, "repairs": [], "answer": None, "error": None}

    intent = classify_intent(sub_q, base_model, base_tok)
    result["intent"] = intent
    print(f"[Intent: {intent}]", flush=True)

    if intent == "Type1":
        raw_sql = generate_sql(sub_q, nl2sql_model, nl2sql_tok)
        result["sql"] = raw_sql
        print(f"[SQL generated]: {raw_sql}", flush=True)

        repaired_sql, repairs = postprocess_sql(raw_sql)
        result["sql_repaired"] = repaired_sql
        result["repairs"] = repairs
        if repairs:
            print(f"[SQL repairs]: {repairs}", flush=True)

        rows, error = execute_sql(repaired_sql)
        if error:
            result["error"] = error
            result["answer"] = f"I generated a SQL query but couldn't execute it: {error}"
            return result

        context = format_sql_context(repaired_sql, rows)
        result["answer"] = generate_answer(sub_q, context, base_model, base_tok)

    elif intent == "Type2":
        if retriever is None:
            result["answer"] = (
                "I'd need to search through financial filings to answer that, "
                "but the document database isn't available right now."
            )
            return result

        # Rewrite → multi-query retrieval → answer
        sys.path.insert(0, str(ROOT / "deployment"))
        from rag.query_rewriter import rewrite_query
        queries = rewrite_query(sub_q, base_model, base_tok)
        chunks  = retrieve_chunks_multi(sub_q, queries, retriever, top_n=3)
        context = format_rag_context(chunks)
        result["answer"] = generate_answer(sub_q, context, base_model, base_tok)

    else:  # Type3
        result["answer"] = direct_answer(sub_q, base_model, base_tok)

    return result


def answer(question: str, base_model, base_tok, nl2sql_model, nl2sql_tok,
           retriever) -> dict:
    """
    Full pipeline: decompose → per-sub-question route → synthesize.

    Step 0: decompose_question() — splits compound questions into sub-questions.
            "What was Apple revenue in 2023 and how did they discuss AI risks?"
            → ["What was Apple revenue in FY2023?",
               "How did Apple discuss AI risks in their 10-K?"]
            Single questions pass through unchanged as [question].

    Step 1 (per sub-question): classify_intent → Type1/Type2/Type3 routing.

    Step 2 (Type2 only): rewrite_query() → 2-3 retrieval queries → retrieve_multi()
            improves recall by merging candidate pools from different phrasings.

    Step 3: if multiple sub-questions, synthesize partial answers into one reply.
    """
    sys.path.insert(0, str(ROOT / "deployment"))
    from rag.query_rewriter import decompose_question

    # Step 0: decompose
    sub_questions = decompose_question(question, base_model, base_tok)

    if len(sub_questions) == 1:
        # Common case — single question, no synthesis needed
        res = _answer_single(sub_questions[0], base_model, base_tok,
                             nl2sql_model, nl2sql_tok, retriever)
        res["question"] = question  # restore original phrasing
        return res

    # Compound question — route each sub-question independently
    print(f"[Decomposed into {len(sub_questions)} sub-questions]", flush=True)
    partial_results = []
    for i, sub_q in enumerate(sub_questions, 1):
        print(f"\n--- Sub-question {i}/{len(sub_questions)}: {sub_q} ---", flush=True)
        partial_results.append(_answer_single(sub_q, base_model, base_tok,
                                              nl2sql_model, nl2sql_tok, retriever))

    # Synthesize partial answers
    combined_answer = _synthesize_answers(question, partial_results, base_model, base_tok)

    # Return merged result (first sub-question's metadata + combined answer)
    merged = partial_results[0].copy()
    merged["question"] = question
    merged["answer"]   = combined_answer
    merged["sub_results"] = partial_results
    return merged


SYNTHESIZE_SYSTEM = (
    "You are a financial analyst assistant. "
    "You have answers to several sub-questions that together address the user's original question. "
    "Combine them into a single, coherent, well-structured response. "
    "Do not repeat information. Be concise."
)


def _synthesize_answers(original_q: str, partial_results: list[dict],
                         model, tokenizer) -> str:
    """Combine partial answers from multiple sub-questions into one reply."""
    parts = []
    for i, r in enumerate(partial_results, 1):
        parts.append(f"Sub-question {i}: {r['question']}\nAnswer: {r['answer']}")
    combined = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
        {"role": "user",   "content":
            f"Original question: {original_q}\n\n{combined}"},
    ]
    return llm_generate(model, tokenizer, messages, max_new_tokens=600)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Financial QA Chatbot")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question (non-interactive mode)")
    parser.add_argument("--no-nl2sql-adapter", action="store_true",
                        help="Skip loading NL2SQL adapter (use base model for SQL)")
    parser.add_argument("--debug", action="store_true",
                        help="Print SQL and intent info even in interactive mode")
    args = parser.parse_args()

    # Load models
    base_model, base_tok = load_base_model()
    if args.no_nl2sql_adapter:
        nl2sql_model, nl2sql_tok = base_model, base_tok
    else:
        nl2sql_model, nl2sql_tok = load_nl2sql_model(base_model, base_tok)
    retriever = load_vectordb()

    print("\n=== Financial QA Chatbot ready ===")
    print("Type your question, or 'quit' to exit.\n")

    if args.question:
        # Single-question mode
        res = answer(args.question, base_model, base_tok, nl2sql_model, nl2sql_tok, retriever)
        print(f"\n{res['answer']}")
        if args.debug:
            print(f"\n[debug] intent={res['intent']} sql={res['sql']} repairs={res['repairs']}")
        return

    # Interactive REPL
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        res = answer(question, base_model, base_tok, nl2sql_model, nl2sql_tok, retriever)
        print(f"\nBot: {res['answer']}\n")

        if args.debug and res["sql"]:
            print(f"     [SQL] {res['sql_repaired']}")
            if res["repairs"]:
                print(f"     [Repairs] {res['repairs']}")
            print()


if __name__ == "__main__":
    main()
