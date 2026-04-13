# Financial QA Chatbot

A local-first question-answering system over SEC EDGAR filings. Ask plain-English questions about public company financials — the system routes them through SQL lookups, document retrieval, or direct LLM response, all running on your own GPU with no cloud fees.

---

## What It Does

You type a question. The system figures out what kind of question it is and answers it from the right source:

| Question type | Example | How it's answered |
|---|---|---|
| **Exact financial fact** | "What was Apple's revenue in FY2023?" | Converts to SQL → queries SQLite panel table → generates answer |
| **Qualitative / analytical** | "How did Microsoft describe their AI strategy?" | Rewrites query → searches ChromaDB (10-K chunks) → generates answer |
| **Chat / meta** | "What can you help me with?" | Direct LLM response |

Compound questions ("What was Apple revenue in 2023 and how did they discuss AI risks?") are automatically split into sub-questions, each routed independently, then synthesized into one reply.

---

## System Architecture

```
User question
  │
  ▼
decompose_question()          ← splits compound questions (Llama prompt)
  │
  ▼ (per sub-question)
classify_intent()             ← Type1 / Type2 / Type3  (Llama prompt)
  │
  ├─ Type1 (exact fact) ──→ generate_sql() [NL2SQL LoRA] → SQLite → generate_answer()
  │
  ├─ Type2 (qualitative) ─→ rewrite_query() → ChromaDB hybrid search → generate_answer()
  │
  └─ Type3 (chat) ────────→ direct_answer()
  │
  ▼
synthesize (if compound)      ← merges partial answers into one reply
```

**All LLM calls** use `meta-llama/Llama-3.2-3B-Instruct` in 4-bit NF4 quantization (~2.5 GB VRAM).
The NL2SQL path additionally loads a LoRA adapter fine-tuned on financial SQL patterns.

---

## Project Structure

```
financial-qa-chatbot/
│
├── chatbot.py                  ← main pipeline (interactive REPL + single-question mode)
├── smoke_test.py               ← quick sanity checks (decompose / intent / SQL / RAG)
├── eval_system.py              ← 20-case system evaluation vs GPT-4o baseline
│
├── data_pipeline/              ── PART 1: build the data stores ──────────────────────
│   ├── scripts/
│   │   ├── fetch_xbrl.py       ← downloads structured financials via SEC EDGAR XBRL API
│   │   ├── run_ingest.py       ← downloads 10-K/8-K filings → chunks → ChromaDB + SQLite
│   │   └── build_nl2sql_dataset.py  ← generates NL2SQL SFT dataset (hand-crafted + GPT-4o)
│   ├── ingestion/              ← SEC EDGAR downloader + loader
│   ├── processing/             ← PDF/HTML chunker
│   ├── storage/                ← ChromaDB vector store + SQLite sql store
│   └── feature_mapping.json    ← canonical XBRL concept → panel column mapping + NL synonyms
│
├── finetune/                   ── PART 2: fine-tune adapters ─────────────────────────
│   ├── scripts/
│   │   └── setup_base_model.py ← downloads Llama-3.2-3B-Instruct to models/llama/
│   ├── data_prep/
│   │   ├── generate_nl2sql_templates.py  ← template-based examples (~1100 pairs)
│   │   ├── distill_nl2sql_openai.py      ← GPT-4o distillation (optional, needs API key)
│   │   └── filter_nl2sql.py              ← removes noise from generated dataset
│   └── adapters/nl2sql/
│       ├── NL2SQL_SFT.py       ← QLoRA fine-tuning script (run this to train)
│       ├── sql_postprocess.py  ← deterministic synonym repair (column name fixes)
│       └── exec_accuracy.py    ← evaluation: execution accuracy on eval set
│
├── deployment/                 ── PART 3: serving ────────────────────────────────────
│   └── rag/
│       ├── retriever.py        ← BM25 + dense + cross-encoder reranker + MMR
│       └── query_rewriter.py   ← query rewriting + question decomposition (Llama prompts)
│
├── data/
│   ├── financials.db           ← SQLite: panel table (financial metrics) + filing_metadata
│   ├── vectordb/               ← ChromaDB: 10-K/8-K text chunks with metadata
│   └── nl2sql/                 ← SFT dataset: train.jsonl + eval.jsonl
│
└── models/
    ├── llama/                  ← cached Llama-3.2-3B-Instruct weights (HF format)
    └── nl2sql/                 ← trained LoRA adapter (adapter_model.safetensors)
```

---

## Data Stores

### SQLite (`data/financials.db`)

**`panel` table** — structured annual financial metrics, one row per (ticker, fiscal year):

```
ticker, year,
total_revenue, gross_profit, operating_income, net_income,
r_and_d, interest_expense, interest_income, da,
cfo, capex, buybacks, dividends_paid,
cash, total_assets, current_assets, current_liabilities,
total_liabilities, long_term_debt, goodwill, retained_earnings,
inventories, accounts_payable, eps_diluted,
current_ratio, net_margin, roa, debt_to_assets
```

Built from the **SEC EDGAR XBRL API** (free, no auth) — authoritative values straight from company filings.

**`filing_metadata` table** — provenance for each filing (ticker, company, type, date, accession number).

### ChromaDB (`data/vectordb/`)

Collection `financial_docs` — 10-K/8-K filings chunked into ~500-token passages.
Embedding model: `all-MiniLM-L6-v2` (runs locally, no API key).
Metadata per chunk: `ticker`, `company`, `filing_type`, `date`, `section`, `accession_number`.

---

## How the Data Pipeline Works

### Step 1 — Structured financials (XBRL API)

```bash
python data_pipeline/scripts/fetch_xbrl.py
# Options:
#   --tickers AAPL MSFT NVDA    (subset; default: all 92 tickers)
#   --years 2021 2022 2023      (fiscal years; default: 2019-2024)
#   --dry-run                   (print without writing to DB)
```

Calls `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` for each ticker.
Extracts annual values using fallback XBRL concept lists (defined in `feature_mapping.json`).
Computes derived ratios: `current_ratio`, `net_margin`, `roa`, `debt_to_assets`.
Writes to the `panel` table in `data/financials.db`. Rate-limited to 10 req/s (SEC policy).

### Step 2 — Filing text (ChromaDB + filing_metadata)

```bash
python data_pipeline/scripts/run_ingest.py --ticker AAPL MSFT --filings-dir data/filings
```

Downloads 10-K/8-K filings from SEC EDGAR, extracts text, chunks them, embeds with `all-MiniLM-L6-v2`, and stores in ChromaDB. Also writes filing provenance to `filing_metadata`.

---

## How the NL2SQL Fine-Tuning Works

### 1. Download the base model

> ⚠️ **Llama is a gated model.** Before downloading, you must:
> 1. Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
> 2. Accept Meta's license terms (takes ~1 minute)
> 3. Generate a HuggingFace token at https://huggingface.co/settings/tokens

```bash
# Set your token first
export HF_TOKEN="hf_your_token_here"          # Linux/Mac
$env:HF_TOKEN = "hf_your_token_here"          # Windows PowerShell

python finetune/scripts/setup_base_model.py
# Downloads ~6 GB to models/llama/ on E: drive
# Runs a quick prompt test to confirm the model works
```

### 2. Generate the SFT dataset

```bash
# Template-based examples (~1,100 pairs — no API key needed)
python finetune/data_prep/generate_nl2sql_templates.py

# Optional: GPT-4o distilled examples (~450 more, requires OpenAI key)
export OPENAI_API_KEY="sk-..."
python data_pipeline/scripts/build_nl2sql_dataset.py

# Filter noise
python finetune/data_prep/filter_nl2sql.py

# Result: data/nl2sql/train.jsonl (~1,500 examples) + eval.jsonl (~270 examples)
```

Dataset format — each example is a 3-turn chat:
```json
{"messages": [
  {"role": "system",    "content": "<schema + rules>"},
  {"role": "user",      "content": "What was Apple's revenue in FY2023?"},
  {"role": "assistant", "content": "SELECT total_revenue FROM panel WHERE ticker='AAPL' AND year='FY2023'"}
]}
```

### 3. Fine-tune the NL2SQL adapter

```bash
python finetune/adapters/nl2sql/NL2SQL_SFT.py
# Options: --epochs 5 --lr 2e-4 --rank 16 --batch 4
# Output: models/nl2sql/adapter_model.safetensors  (~47 MB)
# Trains in ~30-60 min on a single GPU (tested on RTX 3090)
```

Uses **QLoRA** (4-bit base + bf16 LoRA adapters) with `paged_adamw_8bit` optimizer.
Loss computed only on SQL completion tokens — prompt is masked to -100.
Target modules: all attention + MLP projections (q/k/v/o/gate/up/down).

---

## How the Full System Works

```
chatbot.py
  │
  ├─ load_base_model()       loads Llama-3.2-3B-Instruct (4-bit, ~2.5 GB VRAM)
  ├─ load_nl2sql_model()     overlays NL2SQL LoRA adapter on base model
  └─ load_vectordb()         opens ChromaDB collection + builds HybridRetriever
  │
  ▼
answer(question)
  │
  1. decompose_question()    Llama few-shot prompt → splits compound questions
  │                          "revenue AND AI risks?" → 2 sub-questions
  │
  2. per sub-question:
  │   classify_intent()      Llama prompt → Type1 / Type2 / Type3
  │
  │   Type1 path:
  │     generate_sql()       NL2SQL LoRA → SQL string
  │     postprocess_sql()    deterministic synonym repair
  │     execute_sql()        runs SELECT on data/financials.db
  │     generate_answer()    Llama formats result as natural language + cites source
  │
  │   Type2 path:
  │     rewrite_query()      Llama few-shot → 2-3 retrieval phrasings
  │     retrieve_multi()     BM25 + dense recall per query → RRF merge →
  │                          cross-encoder rerank → MMR diversity select
  │     generate_answer()    Llama synthesizes answer from chunks + cites [1],[2]
  │
  │   Type3 path:
  │     direct_answer()      Llama responds directly (no retrieval)
  │
  3. synthesize (if compound) Llama merges partial answers into one reply
```

---

## Quick Start Guide

### Step 1 — Clone

```bash
git clone https://github.com/your-username/financial-qa-chatbot.git
cd financial-qa-chatbot
```

### Step 2 — Install dependencies

```bash
# Option A: conda (recommended — handles PyTorch + CUDA automatically)
conda env create -f environment.yml
conda activate finqa

# Option B: pip
pip install -r requirements.txt
```

> Requires Python 3.11+, CUDA-capable GPU (8 GB+ VRAM recommended), CUDA 12.x drivers.

### Step 3 — Build the data stores

```bash
# 3a. Fetch structured financials from SEC EDGAR (free, no auth)
python data_pipeline/scripts/fetch_xbrl.py
# ~5 min for 92 tickers — writes to data/financials.db

# 3b. Ingest filing text into ChromaDB (needs disk space for filings ~10-20 GB)
python data_pipeline/scripts/run_ingest.py --ticker AAPL MSFT NVDA GOOGL META \
    --filings-dir data/filings
# Downloads 10-K/8-K PDFs, chunks, embeds, stores in data/vectordb/
```

> To use a custom ticker list, edit the `TICKERS` list at the top of `fetch_xbrl.py`.

### Step 4 — Download the base model

> ⚠️ **Accept Llama license first:** https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

```bash
export HF_TOKEN="hf_your_token_here"
python finetune/scripts/setup_base_model.py
# Downloads ~6 GB to models/llama/
```

### Step 5 — Run and test

```bash
# Interactive REPL
python chatbot.py

# Single question
python chatbot.py --question "What was Apple's net income in FY2023?"

# Skip loading the NL2SQL adapter (faster cold start, weaker SQL generation)
python chatbot.py --no-nl2sql-adapter

# Run smoke tests (no GPU needed for import checks)
python smoke_test.py --decompose-only

# Run full system evaluation (20 test cases vs GPT-4o)
python eval_system.py
```

Example session:
```
You: What was Apple's revenue and net margin in FY2023?
Bot: Apple's total revenue in FY2023 was $383.3 billion, with a net margin of 25.3%.
     [Source: financial database]

You: How did Apple discuss AI in their most recent 10-K?
Bot: In their FY2023 10-K, Apple highlighted machine learning capabilities across
     their product lines... [1] Source: AAPL 10-K 2023-11-03
```

### Step 6 — Fine-tune NL2SQL (optional, but recommended)

The pre-trained adapter in `models/nl2sql/` already works. Re-run this if you:
- Add new tickers and want the model to know their company names
- Change the panel schema
- Want to improve SQL accuracy on your specific query patterns

```bash
# Generate fresh SFT dataset
python finetune/data_prep/generate_nl2sql_templates.py
python finetune/data_prep/filter_nl2sql.py

# Train (30-60 min on a single GPU)
python finetune/adapters/nl2sql/NL2SQL_SFT.py

# Evaluate execution accuracy on eval set
python finetune/adapters/nl2sql/exec_accuracy.py
```

> **Windows users:** Before training, move the Windows paging file to a non-system drive
> (Control Panel → System → Advanced → Performance → Virtual Memory).
> Training loads a large model into memory and Windows will page aggressively to C: otherwise.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 12 GB+ |
| RAM | 16 GB | 32 GB |
| Disk (E: or data drive) | 50 GB | 100 GB |
| CUDA | 11.8 | 12.4 |

Tested on: Windows 11 + RTX 3090 (24 GB VRAM), CUDA 12.4.

---

## Evaluation Results (2026-04-12)

20-case system evaluation across all intent types:

| Metric | Our Pipeline | GPT-4o |
|---|---|---|
| Intent accuracy | 85% | — |
| Value correct (±5%) | 80% | 75% |
| Keyword hit rate | 78% | 82% |
| Fluency (1-5) | 3.9 | 4.3 |

GPT-4o has higher fluency but sometimes refuses to answer due to training-cutoff confusion on recent fiscal years. Our pipeline answers from the actual DB and never hallucinates a number.

Full results in `eval_results/`.
