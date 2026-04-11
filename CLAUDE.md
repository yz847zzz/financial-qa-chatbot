# Financial QA Chatbot — CLAUDE.md

## Project Overview

A local-first Financial QA chatbot with RAG over SEC EDGAR filings.
Three independent parts that integrate via shared data stores.

**Implement in order: data_pipeline → finetune → deployment**

## Project Structure

```
financial-qa-chatbot/
├── data_pipeline/    # Part 1: SEC filings → VectorDB + SQL
├── finetune/         # Part 2: LoRA fine-tuning of 3 adapters on Llama-3-8B
├── deployment/       # Part 3: vLLM + Punica serving + FastAPI chatbot
└── docs/             # Architecture and schema references
```

Each part has its own `CLAUDE.md` with specific implementation instructions.
Read the part-specific `CLAUDE.md` before implementing anything in that part.

## Shared Conventions

- **Python 3.11+**, all parts installable as packages (`pip install -e .`)
- **loguru** for all logging — no `print()` in library code
- **dataclasses** for all data contracts between modules
- **Type hints** on every public function signature
- **`__main__` blocks** on every module for quick manual verification
- **pytest** for all tests; no test should require network access or GPU (use fixtures/mocks)
- No `.env` files — use environment variables set before running

## Shared Data Stores

Produced by Part 1, consumed by Parts 2 and 3:

| Store | Path | Purpose |
|---|---|---|
| ChromaDB | `data/vectordb/` | Semantic search over filing text chunks |
| SQLite | `data/financials.db` | Structured financial metrics + filing metadata |

### SQLite Tables (canonical — must match `data_pipeline/storage/schema.sql`)

```sql
-- Structured financial metrics
financials (ticker TEXT, period TEXT, statement TEXT, metric TEXT,
            value REAL, unit TEXT, raw_value TEXT)
UNIQUE (ticker, period, statement, metric)

-- Filing provenance
filing_metadata (id INTEGER PRIMARY KEY, ticker TEXT, company TEXT,
                 filing_type TEXT, date TEXT, accession_number TEXT,
                 section_count INTEGER, chunk_count INTEGER)
```

### ChromaDB Collection

```
collection_name = "financial_docs"
embedding_model = "all-MiniLM-L6-v2"   # local, no API key required
```

Metadata fields stored per chunk (flat dict):
`company`, `ticker`, `filing_type`, `date`, `section`, `accession_number`, `source_path`, `chunk_index`

## Shared Model Paths

Produced by Part 2, consumed by Part 3:

```
models/intent_classifier/   # LoRA adapter (PEFT format)
models/query_rewriter/      # LoRA adapter (PEFT format)
models/nl2sql/              # LoRA adapter (PEFT format)
```

Base model: `meta-llama/Meta-Llama-3-8B-Instruct`

## Intent Types (canonical — used by Parts 2 and 3)

| Label | Description | Retrieval Path |
|---|---|---|
| `Type1` | Exact financial fact (revenue, EPS, ratio) | NL2SQL → SQLite |
| `Type2` | Vague / analytical / qualitative question | VectorDB semantic search |
| `Type3` | Casual chat, greeting, meta question | No retrieval |

## Running the Full System

```bash
# Part 1: ingest filings
cd data_pipeline && python scripts/run_ingest.py --ticker AAPL MSFT --filings-dir ../data/filings

# Part 2: train adapters
cd finetune && bash scripts/train_all.sh

# Part 3: serve
cd deployment && bash scripts/start_server.sh
# Then in another terminal:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "message": "What was Apple revenue in 2023?"}'
```

## Port from pintrade

The following files in `E:/emo/workspace/pintrade/` are direct port templates:

| pintrade file | New file |
|---|---|
| (pipeline VectorStore) | `data_pipeline/storage/vector_store.py` |
| (pipeline SQLStore) | `data_pipeline/storage/sql_store.py` |
| (pipeline chunker) | `data_pipeline/processing/chunker.py` |

The `chunk_id` scheme and metadata dict structure **must remain identical** between
Part 1 (where chunks are created) and Part 3 (where they are retrieved).
