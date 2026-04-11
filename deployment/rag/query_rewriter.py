"""
Query rewriting for Type2 retrieval.

Two functions:

  rewrite_query(question, model, tokenizer) → list[str]
    Rewrites a single-intent question into 2-3 retrieval-focused phrasings.
    Example:
      "Tell me about Apple's AI strategy"
      → ["Apple artificial intelligence strategy 10-K",
         "Apple AI initiatives machine learning",
         "Apple technology investments AI"]

  decompose_question(question, model, tokenizer) → list[str]
    Splits a compound question into independent sub-questions (one per intent).
    Returns [question] unchanged if not compound.
    Example:
      "What was Apple's revenue in 2023 and how did they describe AI risks?"
      → ["What was Apple revenue in FY2023?",
         "How did Apple describe AI risks in their 10-K?"]

Both use few-shot prompting on the already-loaded base Llama model — no SFT needed.
The cross-encoder reranker always uses the original user question (not sub-queries)
so relevance judgement stays anchored to what the user actually asked.
"""

import json
import re


# ── System prompts ─────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You are a search query optimizer for financial SEC filings (10-K, 10-Q, 8-K).

Given a user question, output 2-3 retrieval-focused queries as a JSON array.
Each query should be 5-15 words, specific, and match the language used in SEC filings.

Rules:
- Remove conversational filler ("can you tell me", "I want to know", "please")
- Replace pronouns with the actual company name
- Use financial/SEC terminology where appropriate
- Generate 2-3 different phrasings that cover the same topic from different angles
- Output ONLY a JSON array of strings, nothing else

Output format: ["query 1", "query 2", "query 3"]"""

_REWRITE_EXAMPLES = [
    (
        "How did Apple describe the risks to their supply chain?",
        '["Apple supply chain risk factors", "Apple manufacturing concentration risks", "Apple logistics disruption 10-K"]',
    ),
    (
        "Tell me about Microsoft's cloud strategy and Azure growth",
        '["Microsoft Azure cloud strategy", "Microsoft cloud revenue growth", "Microsoft intelligent cloud segment"]',
    ),
    (
        "What did Tesla say about competition in the EV market?",
        '["Tesla competitive landscape electric vehicles", "Tesla competition risks 10-K", "Tesla EV market share risks"]',
    ),
]

_DECOMPOSE_SYSTEM = """\
You are a question decomposer for a financial QA system.

If the user question contains multiple independent questions (connected by "and", "also", \
"as well as", "additionally", etc.), split them into separate questions.
If it is a single focused question, return it unchanged in a JSON array.

Rules:
- Replace pronouns (their, its, they) with the actual company name
- Each sub-question should be self-contained and answerable independently
- Output ONLY a JSON array of strings, nothing else

Output format: ["question 1"] or ["question 1", "question 2"]"""

_DECOMPOSE_EXAMPLES = [
    (
        "What was Apple's revenue in FY2023 and how did they describe their AI strategy?",
        '["What was Apple revenue in FY2023?", "How did Apple describe their AI strategy in their 10-K?"]',
    ),
    (
        "How did Microsoft discuss cloud risks and also their dividend policy?",
        '["How did Microsoft discuss cloud risks?", "What is Microsoft dividend policy?"]',
    ),
    (
        "What was Amazon's operating income in FY2022?",
        '["What was Amazon operating income in FY2022?"]',
    ),
    (
        "Explain Google's revenue growth and also their capital expenditure plans",
        '["What drove Google revenue growth?", "What are Google capital expenditure plans?"]',
    ),
]


# ── LLM helper ─────────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], model, tokenizer, max_new_tokens: int = 128) -> str:
    """Two-step tokenisation matching chatbot.py to avoid apply_chat_template tensor issues."""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_json_list(raw: str) -> list[str] | None:
    """Extract a JSON array of strings from LLM output. Returns None on failure."""
    try:
        result = json.loads(raw)
        if isinstance(result, list) and all(isinstance(q, str) for q in result):
            return [q.strip() for q in result if q.strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find [...] substring in case the model adds preamble
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and all(isinstance(q, str) for q in result):
                return [q.strip() for q in result if q.strip()]
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def rewrite_query(question: str, model, tokenizer) -> list[str]:
    """
    Expand a Type2 question into 2-3 retrieval-optimized queries.

    The expanded queries are all passed to HybridRetriever.retrieve_multi(),
    which merges the candidate pools before reranking with the original question.
    This improves recall: different phrasings surface different relevant chunks.

    Falls back to [question] on LLM parse failure — retrieval still works.
    """
    messages = [{"role": "system", "content": _REWRITE_SYSTEM}]
    for user_q, assistant_a in _REWRITE_EXAMPLES:
        messages.append({"role": "user",      "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a})
    messages.append({"role": "user", "content": question})

    raw = _call_llm(messages, model, tokenizer, max_new_tokens=128)
    result = _parse_json_list(raw)
    if result:
        print(f"[QueryRewrite] {question!r} → {result}", flush=True)
        return result

    print(f"[QueryRewrite] parse failed, using original: {raw!r}", flush=True)
    return [question]


def decompose_question(question: str, model, tokenizer) -> list[str]:
    """
    Split a compound question into independent sub-questions.

    Returns [question] if not compound. Used before intent classification
    so each sub-question can be routed independently — one might go to
    Type1 (SQL) and another to Type2 (RAG).

    Example:
      "What was Apple's revenue in 2023 and how did they discuss AI risks?"
      → ["What was Apple revenue in FY2023?",
         "How did Apple discuss AI risks in their 10-K?"]
    """
    messages = [{"role": "system", "content": _DECOMPOSE_SYSTEM}]
    for user_q, assistant_a in _DECOMPOSE_EXAMPLES:
        messages.append({"role": "user",      "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a})
    messages.append({"role": "user", "content": question})

    raw = _call_llm(messages, model, tokenizer, max_new_tokens=128)
    result = _parse_json_list(raw)
    if result and len(result) >= 1:
        if len(result) > 1:
            print(f"[Decompose] Split into {len(result)} sub-questions: {result}", flush=True)
        return result

    print(f"[Decompose] parse failed, treating as single question: {raw!r}", flush=True)
    return [question]
