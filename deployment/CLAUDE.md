# Deployment — Part 3 CLAUDE.md

## Purpose

Serve three LoRA adapters on a single **Llama-3-8B-Instruct** base using
**vLLM + Punica multi-LoRA**. Expose a **FastAPI** chatbot endpoint with
multi-turn conversation and a full RAG orchestration pipeline.

## Prerequisites

Before starting Part 3:
- Part 1 must be complete: `../data/vectordb/` (ChromaDB) and `../data/financials.db` (SQLite) must exist
- Part 2 must be complete: `../models/intent_classifier/`, `../models/query_rewriter/`, `../models/nl2sql/` must exist

## Commands

```bash
pip install -r requirements.txt

# Start vLLM (port 8001) + FastAPI (port 8000):
bash scripts/start_server.sh

# Or start individually:
# Terminal 1 — vLLM with all adapters:
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-lora \
  --lora-modules intent_classifier=../models/intent_classifier/ \
                 query_rewriter=../models/query_rewriter/ \
                 nl2sql=../models/nl2sql/ \
  --max-lora-rank 16 \
  --gpu-memory-utilization 0.85 \
  --port 8001

# Terminal 2 — FastAPI chatbot:
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# Smoke test:
python scripts/smoke_test.py

# Unit tests (no GPU, no vLLM required):
pytest tests/ -v
```

## Architecture — Full Request Flow

```
POST /chat {"session_id": "s1", "message": "What was Apple revenue in 2023?"}
    │
    ▼
orchestrator/pipeline.py: handle_message(session_id, message)
    │
    ├─ Step 1: intent_router.py
    │          call vLLM with lora_request="intent_classifier"
    │          → "Type1" | "Type2" | "Type3"
    │
    ├─ Step 2: [Type2 only] query_rewriter.py
    │          call vLLM with lora_request="query_rewriter"
    │          → ["sub-query 1", "sub-query 2", ...]
    │
    ├─ Step 3: retriever.py
    │          Type1 → sql_executor.py:
    │              call vLLM with lora_request="nl2sql" → SQL string
    │              → execute_raw(sql) → DataFrame rows as list[dict]
    │          Type2 → VectorStore.search_multi_query(sub_queries) → list[dict]
    │          Type3 → [] (no retrieval)
    │
    ├─ Step 4: context_builder.py
    │          Format retrieved chunks or SQL rows → context_str
    │          Add citations: [Source: AAPL 10-K 2023-11-03 ITEM 7]
    │          Truncate to fit context budget (see below)
    │
    ├─ Step 5: session_store.py
    │          get_history(session_id) → last N messages
    │
    └─ Step 6: generator.py
               call vLLM base model (NO lora_request)
               System prompt: "Answer from the provided context only. Cite sources."
               → answer string
               → save to session history
               → return to user
```

## vLLM + Punica Multi-LoRA (`server/vllm_server.py`)

```python
import subprocess, time, httpx, sys
from loguru import logger

VLLM_PORT = 8001
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATHS = {
    "intent_classifier": "../models/intent_classifier/",
    "query_rewriter":    "../models/query_rewriter/",
    "nl2sql":            "../models/nl2sql/",
}

def build_vllm_command() -> list[str]:
    lora_modules = " ".join(f"{k}={v}" for k, v in ADAPTER_PATHS.items())
    return [
        "vllm", "serve", BASE_MODEL,
        "--enable-lora",
        "--lora-modules", *[f"{k}={v}" for k, v in ADAPTER_PATHS.items()],
        "--max-lora-rank", "16",
        "--gpu-memory-utilization", "0.85",
        "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
    ]

def wait_for_vllm(timeout: int = 120) -> bool:
    """Poll vLLM health endpoint until ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False
```

## LoRA Client (`server/lora_client.py`)

```python
import httpx
from loguru import logger

VLLM_BASE = "http://localhost:8001/v1"

def generate(
    prompt: str,
    lora_name: str | None = None,   # None = use base model
    max_tokens: int = 256,
    temperature: float = 0.1,
    stop: list[str] | None = None,
) -> str:
    """Call vLLM completions API with optional LoRA adapter."""
    payload = {
        "model": lora_name or "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop or [],
    }
    resp = httpx.post(f"{VLLM_BASE}/completions", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()
```

## RAG Modules

### `rag/intent_router.py`

```python
from server.lora_client import generate
from finetune.adapters.intent_classifier.prompt_template import SYSTEM_PROMPT, format_prompt

VALID_TYPES = {"Type1", "Type2", "Type3"}

def classify_intent(message: str) -> str:
    """Returns "Type1", "Type2", or "Type3". Defaults to "Type2" on parse failure."""
    prompt = format_prompt(SYSTEM_PROMPT, message)  # applies Llama-3 chat template
    output = generate(prompt, lora_name="intent_classifier",
                      max_tokens=5, temperature=0.0)
    for t in VALID_TYPES:
        if t in output:
            return t
    logger.warning(f"Intent parse failed for: {message!r} → output: {output!r}")
    return "Type2"  # safe default: VectorDB search
```

### `rag/query_rewriter.py`

```python
import json
from server.lora_client import generate

def rewrite(message: str) -> list[str]:
    """Returns list of sub-queries. Falls back to [message] on parse failure."""
    prompt = ...  # apply query_rewriter chat template
    output = generate(prompt, lora_name="query_rewriter",
                      max_tokens=256, temperature=0.1)
    try:
        result = json.loads(output)
        if isinstance(result, list) and all(isinstance(q, str) for q in result):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return [message]  # fallback: treat original query as single sub-query
```

### `rag/sql_executor.py`

```python
import sqlite3, re
from server.lora_client import generate

ALLOWED_STMT = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

def nl_to_sql(question: str, schema_context: str) -> str:
    """Call nl2sql adapter to convert question → SQL."""
    prompt = ...  # apply nl2sql chat template with schema context
    return generate(prompt, lora_name="nl2sql", max_tokens=256, temperature=0.0)

def execute_sql(db_path: str, sql: str, max_rows: int = 50,
                timeout_ms: int = 5000) -> list[dict] | dict:
    """Execute SQL safely. Returns list of row dicts or {"error": ..., "sql": ...}."""
    if not ALLOWED_STMT.match(sql):
        return {"error": "Only SELECT statements allowed", "sql": sql}
    try:
        conn = sqlite3.connect(db_path, timeout=timeout_ms / 1000)
        conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchmany(max_rows)]
        conn.close()
        return rows
    except Exception as e:
        return {"error": str(e), "sql": sql}
```

### `rag/retriever.py`

```python
def retrieve(
    queries: list[str],
    intent_type: str,
    vector_store,
    sql_store,
    nl2sql_question: str,
    schema_context: str,
) -> list[dict]:
    """Unified retrieval. Returns list of result dicts with a 'text' key."""
    if intent_type == "Type1":
        sql = nl_to_sql(nl2sql_question, schema_context)
        rows = execute_sql(sql_store.db_path, sql)
        if isinstance(rows, list):
            return [{"text": str(row), "source": "SQL", "sql": sql} for row in rows]
        # SQL error → fall back to Type2
        return vector_store.search_multi_query(queries, n_results=5)
    elif intent_type == "Type2":
        return vector_store.search_multi_query(queries, n_results=8)
    else:  # Type3
        return []
```

### `rag/context_builder.py`

Context budget (must stay within 4096 total tokens):

| Slot | Budget |
|---|---|
| System prompt | ~200 tokens |
| Conversation history (last 6 turns) | ~800 tokens |
| Retrieved context | ~2,000 tokens |
| User message | ~200 tokens |
| Response budget | ~800 tokens |

```python
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
CONTEXT_TOKEN_BUDGET = 2000

def build_context(results: list[dict], budget_tokens: int = CONTEXT_TOKEN_BUDGET) -> str:
    """Format retrieved results into numbered context string with citations.
    Truncate to fit token budget."""
    lines = []
    total = 0
    for i, r in enumerate(results, 1):
        src = r.get("metadata", {})
        citation = f"[Source: {src.get('ticker','')} {src.get('filing_type','')} {src.get('date','')} {src.get('section','')}]"
        chunk = f"[{i}] {r['text']}\n{citation}"
        tokens = len(TOKENIZER.encode(chunk))
        if total + tokens > budget_tokens:
            break
        lines.append(chunk)
        total += tokens
    return "\n\n".join(lines) if lines else "No relevant information found in the loaded filings."
```

## FastAPI Application (`api/main.py`)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize shared resources
    app.state.vector_store = VectorStore(persist_dir="../data/vectordb")
    app.state.sql_store    = SQLStore(db_path="../data/financials.db")
    app.state.session_store = SessionStore()
    logger.info("Services initialized")
    yield
    # Shutdown: clean up if needed

app = FastAPI(title="Financial QA Chatbot", lifespan=lifespan)
app.include_router(chat_router,    prefix="/chat")
app.include_router(history_router, prefix="/history")
app.include_router(health_router)
```

### API Endpoints

```
POST /chat
  Request:  {"session_id": str, "message": str}
  Response: {"reply": str, "intent": str, "sources": list[str], "session_id": str}

GET  /history/{session_id}
  Response: {"messages": [{"role": str, "content": str, "timestamp": str}]}

DELETE /history/{session_id}
  Response: {"deleted": true}

GET  /health
  Response: {"status": "ok", "vllm": bool, "vectordb": bool, "sql": bool}
  vllm check: GET http://localhost:8001/health
  vectordb check: vector_store.collection.count() > 0
  sql check: sql_store.execute_raw("SELECT 1") returns a row
```

### Pydantic Models (`api/models.py`)

```python
from pydantic import BaseModel
from datetime import datetime

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    intent: str           # "Type1" | "Type2" | "Type3"
    sources: list[str]    # citation strings
    session_id: str

class Message(BaseModel):
    role: str             # "user" | "assistant"
    content: str
    timestamp: str        # ISO 8601
```

## Session Store (`api/session_store.py`)

```python
from collections import deque, defaultdict
from datetime import datetime, timedelta
from api.models import Message

MAX_TURNS = 50
SESSION_TTL_MINUTES = 60
HISTORY_WINDOW = 6  # turns to include in LLM context

class SessionStore:
    def __init__(self):
        self._store: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=MAX_TURNS))
        self._last_active: dict[str, datetime] = {}

    def add(self, session_id: str, role: str, content: str) -> None: ...
    def get_history(self, session_id: str, n: int = HISTORY_WINDOW) -> list[Message]: ...
    def delete(self, session_id: str) -> None: ...
    def purge_expired(self) -> int:
        """Remove sessions inactive > SESSION_TTL_MINUTES. Call periodically."""
```

## Orchestrator (`orchestrator/pipeline.py`)

The single function that wires all steps:

```python
from loguru import logger
from rag import intent_router, query_rewriter, retriever, context_builder
from server import lora_client
from api.models import ChatResponse

def handle_message(
    session_id: str,
    message: str,
    vector_store,
    sql_store,
    session_store,
    schema_context: str,
) -> ChatResponse:
    # Step 1: intent
    intent = intent_router.classify_intent(message)
    logger.info(f"[{session_id}] intent={intent} | query={message!r}")

    # Step 2: query rewrite (Type2 only)
    queries = query_rewriter.rewrite(message) if intent == "Type2" else [message]

    # Step 3: retrieve
    results = retriever.retrieve(queries, intent, vector_store, sql_store,
                                 nl2sql_question=message, schema_context=schema_context)

    # Step 4: build context
    context = context_builder.build_context(results)
    sources = [r.get("metadata", {}).get("source_path", "") for r in results
               if "metadata" in r]

    # Step 5: conversation history
    history = session_store.get_history(session_id)

    # Step 6: generate answer
    reply = _generate_answer(message, context, history)

    # Persist turn
    session_store.add(session_id, "user", message)
    session_store.add(session_id, "assistant", reply)

    return ChatResponse(reply=reply, intent=intent,
                        sources=sources, session_id=session_id)
```

## `scripts/start_server.sh`

```bash
#!/bin/bash
set -e

# Check prerequisites
[ -d "../data/vectordb" ] || { echo "ERROR: Run data_pipeline first"; exit 1; }
[ -f "../data/financials.db" ] || { echo "ERROR: Run data_pipeline first"; exit 1; }
[ -d "../models/intent_classifier" ] || { echo "ERROR: Run finetune first"; exit 1; }

echo "=== Starting vLLM on port 8001 ==="
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-lora \
  --lora-modules intent_classifier=../models/intent_classifier/ \
                 query_rewriter=../models/query_rewriter/ \
                 nl2sql=../models/nl2sql/ \
  --max-lora-rank 16 \
  --gpu-memory-utilization 0.85 \
  --port 8001 \
  --max-model-len 4096 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
python -c "
import time, httpx, sys
for _ in range(40):
    try:
        if httpx.get('http://localhost:8001/health', timeout=2).status_code == 200:
            print('vLLM ready'); sys.exit(0)
    except: pass
    time.sleep(3)
print('ERROR: vLLM did not start'); sys.exit(1)
"

echo "=== Starting FastAPI on port 8000 ==="
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## `scripts/smoke_test.py`

```python
"""End-to-end smoke test — requires running server."""
import httpx

BASE = "http://localhost:8000"
SESSION = "smoke_test_001"

tests = [
    # (message, expected_intent)
    ("What was Apple's total revenue in fiscal 2023?",         "Type1"),
    ("How did Microsoft describe its AI strategy in its 10-K?","Type2"),
    ("Hello, what can you help me with?",                      "Type3"),
]

for msg, expected_intent in tests:
    r = httpx.post(f"{BASE}/chat",
                   json={"session_id": SESSION, "message": msg}, timeout=60)
    r.raise_for_status()
    data = r.json()
    status = "PASS" if data["intent"] == expected_intent else "FAIL"
    print(f"[{status}] intent={data['intent']} (expected {expected_intent})")
    print(f"  Reply: {data['reply'][:120]}...")

# Test history
r = httpx.get(f"{BASE}/history/{SESSION}")
assert len(r.json()["messages"]) == len(tests) * 2  # user + assistant per turn
print("[PASS] History endpoint")

# Test health
r = httpx.get(f"{BASE}/health")
assert r.json()["status"] == "ok"
print("[PASS] Health endpoint")
```

## Dependencies (`requirements.txt`)

```
vllm>=0.4.0
fastapi>=0.111
uvicorn[standard]>=0.30
httpx>=0.27
pydantic>=2.7
chromadb>=0.5
sentence-transformers>=3.0
transformers>=4.41
peft>=0.11
loguru>=0.7
pandas>=2.0
pytest>=8.0
pytest-asyncio>=0.23
```

**Pin vLLM to `>=0.4.0`** — this is the minimum version where Punica multi-LoRA
is stable.

## Testing Requirements

Tests must pass **without a GPU and without a running vLLM server**.
Mock `server.lora_client.generate` using `pytest` fixtures.

| Test file | What to verify |
|---|---|
| `test_intent_router.py` | 9 fixture queries (3 per type); mock vLLM; assert correct Type; assert Type2 fallback on bad output |
| `test_retriever.py` | Mock VectorStore + SQLStore; assert returned dict has 'text' key; assert SQL error falls back to VectorDB |
| `test_sql_executor.py` | SELECT allowed; INSERT rejected; timeout enforced; max_rows respected |
| `test_api.py` | FastAPI TestClient; POST /chat; GET /history; DELETE /history; GET /health |

## Error Handling

| Failure | Behavior |
|---|---|
| Intent classification fails / parse error | Default to `"Type2"` (VectorDB search) |
| VectorDB returns 0 results | Reply: "I don't have information about that in the loaded filings." |
| SQL execution fails | Log SQL + error; fall back to Type2 VectorDB retrieval |
| vLLM unreachable | HTTP 503 with `Retry-After: 10` header |
| JSON parse error in query_rewriter | Return `[original_message]` and continue |

## Docker (`docker/`)

```dockerfile
# docker/Dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
RUN pip install vllm>=0.4.0 fastapi uvicorn httpx peft transformers loguru
COPY . /app
WORKDIR /app
EXPOSE 8000 8001
CMD ["bash", "scripts/start_server.sh"]
```

```yaml
# docker/docker-compose.yml
services:
  chatbot:
    build: ..
    ports:
      - "8000:8000"
      - "8001:8001"
    volumes:
      - ../data:/app/data
      - ../models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
