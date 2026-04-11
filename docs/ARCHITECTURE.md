# System Architecture

## Query Flow (End-to-End)

```
User Query
    │
    ▼
POST /chat  (FastAPI :8000)
    │
    ▼
orchestrator/pipeline.py
    │
    ├─[1]─ intent_router ──────► vLLM :8001 (lora: intent_classifier)
    │           │
    │           ▼
    │       "Type1" / "Type2" / "Type3"
    │
    ├─[2]─ query_rewriter ─────► vLLM :8001 (lora: query_rewriter)  [Type2 only]
    │           │
    │           ▼
    │       ["sub-query 1", "sub-query 2", ...]
    │
    ├─[3]─ retriever
    │       ├─ Type1 ─► nl2sql ─► vLLM :8001 (lora: nl2sql) ─► SQLite
    │       ├─ Type2 ─► ChromaDB semantic search
    │       └─ Type3 ─► (skip)
    │
    ├─[4]─ context_builder  →  formatted context string + citations
    │
    ├─[5]─ session_store    →  last 6 conversation turns
    │
    └─[6]─ generator ──────────► vLLM :8001 (base model, no LoRA)
                │
                ▼
            Final answer string
                │
                ▼
        ChatResponse {"reply", "intent", "sources", "session_id"}
```

## Data Flow (Offline)

```
SEC EDGAR .htm/.pdf files (local disk)
    │
    ▼
data_pipeline/scripts/run_ingest.py
    ├─ sec_loader      → FilingMetadata
    ├─ extractor       → raw text
    ├─ section_splitter→ {ITEM N: text}
    ├─ chunker         → list[TextChunk]
    ├─ table_parser    → list[FinancialRow]
    ├─► vector_store   → ChromaDB (data/vectordb/)
    └─► sql_store      → SQLite   (data/financials.db)
```

## Fine-tuning Flow (Offline)

```
SFT datasets (JSONL, chat format)
    │
    ▼
finetune/scripts/train_all.sh
    ├─► intent_classifier adapter → models/intent_classifier/
    ├─► query_rewriter adapter    → models/query_rewriter/
    └─► nl2sql adapter            → models/nl2sql/

Base model: meta-llama/Meta-Llama-3-8B-Instruct (4-bit BnB NF4)
LoRA: rank=16, alpha=32, target=q/k/v/o_proj
```

## Port Assignments

| Service | Port | Protocol |
|---|---|---|
| FastAPI chatbot | 8000 | HTTP |
| vLLM + Punica | 8001 | HTTP (OpenAI-compatible) |

## GPU Memory Budget (single 4090, 24GB)

| Component | VRAM |
|---|---|
| Llama-3-8B base (4-bit) | ~5 GB |
| KV cache (PagedAttention) | ~14 GB |
| Active LoRA adapter (rank=16) | ~300 MB |
| OS + CUDA overhead | ~2 GB |
| **Total** | **~22 GB** |
