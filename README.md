# Financial QA Chatbot

A local-first question-answering system over SEC EDGAR filings. Ask plain-English questions about public company financials — the system routes them through SQL lookups, semantic document retrieval, or direct LLM response, all running **on your own GPU** with no cloud fees.

Built on **Llama-3.2-3B-Instruct** + **vLLM** with LoRA adapters fine-tuned for intent classification, query rewriting, and NL-to-SQL translation.

---

## What It Does

You type a question. The system figures out what kind of question it is and answers it from the right source:

| Question type | Example | How it's answered |
|---|---|---|
| **Exact financial fact** | "What was Apple's revenue in FY2023?" | NL2SQL LoRA → SQLite query → formatted answer |
| **Qualitative / analytical** | "How did Microsoft describe their AI strategy?" | Query rewrite → ChromaDB semantic search → answer |
| **Chat / meta** | "What can you help me with?" | Direct LLM response |

Compound questions ("What was Apple revenue in 2023 and how did they discuss AI risks?") are automatically split into sub-questions, each routed independently, then synthesised into one reply.

---

## Architecture

```
User question
  │
  ▼
decompose_question()          ← splits compound questions (Llama base)
  │
  ▼ (per sub-question)
classify_intent()             ← Type1 / Type2 / Type3  (intent_classifier LoRA)
  │
  ├─ Type1 (exact fact) ──→ generate_sql() [nl2sql LoRA] → SQLite → generate_answer()
  │
  ├─ Type2 (qualitative) ─→ rewrite_query() [query_rewriter LoRA]
  │                          → ChromaDB BM25+dense hybrid search → generate_answer()
  │
  └─ Type3 (chat) ────────→ direct_answer() [base model]
```

All LLM calls share **one vLLM process** using Punica multi-LoRA batching (SGMV kernels) — adapters are hot-swapped per request with near-zero overhead.

---

## Project Structure

```
financial-qa-chatbot/
│
├── chatbot.py                      ← interactive REPL + single-question CLI
├── smoke_test.py                   ← quick sanity checks
├── eval_benchmark.py               ← 52-case benchmark vs GPT-4o baseline
├── eval_sweep.py                   ← quantization × concurrency sweep runner
├── eval_plot.py                    ← matplotlib plots from sweep JSONs
├── quantize_awq.py                 ← W4A16 INT4 quantization (llm-compressor)
│
├── scripts/
│   └── download_model.py           ← download Llama weights from HuggingFace
│
├── data_pipeline/                  ── Part 1: build data stores ────────────
│   ├── ingestion/
│   │   ├── sec_downloader.py       ← fetch filings from SEC EDGAR REST API
│   │   └── sec_loader.py           ← walk local filing tree → FilingMetadata
│   ├── processing/
│   │   ├── extractor.py            ← HTML/PDF → plain text
│   │   ├── section_splitter.py     ← 10-K → ITEM sections
│   │   ├── chunker.py              ← sliding-window chunker
│   │   └── table_parser.py         ← extract financial tables → FinancialRow
│   ├── storage/
│   │   ├── vector_store.py         ← ChromaDB (all-MiniLM-L6-v2 embeddings)
│   │   ├── sql_store.py            ← SQLite (financials + filing_metadata)
│   │   └── schema.sql              ← canonical table definitions
│   └── scripts/
│       ├── run_ingest.py           ← full pipeline CLI
│       └── fetch_xbrl.py           ← structured financials via XBRL API
│
├── finetune/                       ── Part 2: LoRA fine-tuning ─────────────
│   ├── adapters/nl2sql/
│   │   └── NL2SQL_SFT.py           ← QLoRA SFT script (train the NL2SQL adapter)
│   └── data_prep/
│       ├── generate_nl2sql_templates.py
│       └── distill_nl2sql_openai.py   ← optional GPT-4o distillation
│
├── deployment/                     ── Part 3: serving ──────────────────────
│   ├── rag/
│   │   ├── retriever.py            ← BM25 + dense + cross-encoder + MMR
│   │   └── query_rewriter.py       ← query decomposition
│   ├── api/
│   │   └── client.py               ← vLLM OpenAI-compatible client
│   └── scripts/
│       ├── start_server.sh         ← launch vLLM + all LoRA adapters
│       └── start_server_quant.sh   ← launch with fp16 / int8 / awq4 flag
│
├── data/
│   ├── financials.db               ← SQLite (git-ignored — build locally)
│   ├── vectordb/                   ← ChromaDB collection (git-ignored)
│   └── nl2sql/                     ← SFT dataset: train.jsonl + eval.jsonl
│
├── models/
│   ├── llama/                      ← Llama-3.2-3B-Instruct weights (git-ignored)
│   └── nl2sql/                     ← trained LoRA adapter (adapter_config tracked)
│
├── .env.example                    ← copy to .env and fill in tokens
├── pyproject.toml                  ← installable package
└── environment.yml                 ← conda environment
```

---

## Quick Start

### 1 — Clone

```bash
git clone https://github.com/yz847zzz/financial-qa-chatbot.git
cd financial-qa-chatbot
```

### 2 — Set up environment

```bash
# Conda (recommended — handles PyTorch + CUDA automatically)
conda env create -f environment.yml
conda activate finqa

# Or pip
pip install -e ".[all]"
```

> **Requirements:** Python 3.11+, CUDA-capable GPU (≥8 GB VRAM), CUDA 12.x drivers.

### 3 — Configure secrets

```bash
cp .env.example .env
# Edit .env — add HF_TOKEN (required) and optionally OPENAI_API_KEY
```

All credentials are read from `.env` — **never hard-coded**. See `.env.example` for the full list.

### 4 — Download the base model

> ⚠️ Llama-3.2 is a **gated model**. You must:
> 1. Accept Meta's licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
> 2. Create a token at https://huggingface.co/settings/tokens
> 3. Add it to `.env` as `HF_TOKEN=hf_...`

```bash
python scripts/download_model.py          # downloads ~6 GB to models/llama/
python scripts/download_model.py --verify # also runs a test prompt after download
```

### 5 — Build the data stores

```bash
# 5a. Structured financials from SEC EDGAR XBRL API (free, no auth)
python data_pipeline/scripts/fetch_xbrl.py
# ~5 min for the default 92 tickers → writes to data/financials.db

# 5b. Filing text → ChromaDB (requires filing downloads, ~10-20 GB disk)
python data_pipeline/ingestion/sec_downloader.py \
    --ticker AAPL MSFT NVDA GOOGL META \
    --forms 10-K 10-Q --start 2020-01-01 --end 2024-12-31 \
    --out data/filings

python data_pipeline/scripts/run_ingest.py \
    --ticker AAPL MSFT NVDA GOOGL META \
    --filings-dir data/filings
```

### 6 — Serve and chat

```bash
# Start vLLM with all LoRA adapters (in WSL2 if on Windows)
bash deployment/scripts/start_server.sh

# Interactive REPL (in a second terminal)
python chatbot.py

# Or single question
python chatbot.py --question "What was Apple's net income in FY2023?"
```

---

## Fine-tuning the NL2SQL Adapter (optional)

The adapter in `models/nl2sql/` is pre-trained and ready to use. Re-train if you add tickers or change the schema.

```bash
# Generate SFT dataset (~1,500 examples, no API key needed)
python finetune/data_prep/generate_nl2sql_templates.py
python finetune/data_prep/filter_nl2sql.py

# Optional: add GPT-4o distilled examples (requires OPENAI_API_KEY in .env)
python data_pipeline/scripts/build_nl2sql_dataset.py

# Train — ~30-60 min on a single RTX 3090
python finetune/adapters/nl2sql/NL2SQL_SFT.py

# Evaluate execution accuracy
python finetune/adapters/nl2sql/exec_accuracy.py
```

Uses **QLoRA** (4-bit NF4 base + bf16 LoRA) with loss masked to SQL completion tokens only.

---

## Performance: Quantization × Concurrency Sweep

We benchmarked three quantization levels across five concurrency tiers on an **RTX 3090 Ti (24 GB VRAM)**.
Sweep runner: [`eval_sweep.py`](eval_sweep.py) · Plots: [`eval_plot.py`](eval_plot.py)

### Hardware configuration

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3090 Ti, 24 GB GDDR6X |
| Host OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| CUDA | 12.4 |
| vLLM | 0.6.x |
| Base model | Llama-3.2-3B-Instruct |
| Eval dataset | 52 financial QA cases (Type1 / Type2 / Type3) |

---

### Throughput (QPS) vs Concurrency

| Concurrency | fp16 (bfloat16) | int8 (bitsandbytes) | **awq4 (W4A16 Marlin)** |
|:-----------:|:---------------:|:-------------------:|:-----------------------:|
| 1           | 0.15            | 0.04                | **0.90**                |
| 2           | 0.51            | 0.18                | **1.34**                |
| 4           | 0.97            | 0.70                | **1.48**                |
| 8           | 2.35            | 1.47                | **2.94**                |
| 16          | 2.98            | 2.20                | **3.32**                |

> AWQ4 leads at **every** concurrency level. The gap is largest at low concurrency (6× faster than fp16 at c=1).

---

### Latency (seconds) — p50 / p95

| Concurrency | fp16 p50/p95 | int8 p50/p95 | **awq4 p50/p95** |
|:-----------:|:------------:|:------------:|:----------------:|
| 1           | 6.87 / 9.22  | 8.71 / 156.3 | **0.69 / 4.97**  |
| 2           | 1.55 / 7.90  | 10.48 / 13.27 | **0.66 / 2.75** |
| 4           | 3.55 / 4.70  | 4.49 / 6.79  | **2.63 / 2.73**  |
| 8           | 3.30 / 3.39  | 4.73 / 5.44  | **2.65 / 2.69**  |
| 16          | 3.35 / 5.34  | 7.13 / 7.26  | **2.75 / 4.76**  |

> ⚠️ INT8 shows a **156-second p95 outlier** at c=1 — the first request triggers bitsandbytes kernel JIT compilation.

---

### Accuracy (52 test cases, concurrency = 1)

| Metric | fp16 | int8 | awq4 |
|---|:---:|:---:|:---:|
| Intent accuracy | 75.0% | 78.8% | 78.8% |
| Value accuracy (±5%) | 66.7% | 70.3% | 70.3% |
| Keyword hit rate | 70.9% | 75.1% | 72.8% |

> Accuracy is **equivalent across all three quantization levels** — INT4 compression does not degrade answer quality at this model size.

---

### VRAM footprint

| Quantization | Weight size | Remaining VRAM (KV cache) | Max batch |
|---|:---:|:---:|:---:|
| fp16 (bfloat16) | ~6 GB | ~18 GB | 64 seq |
| int8 (bitsandbytes) | ~3 GB | ~21 GB | 96 seq |
| awq4 (W4A16 Marlin) | ~1.5 GB | ~22.5 GB | 128 seq |

---

### Key findings

**AWQ4 (W4A16 Marlin INT4) wins on every axis:**

1. **Fastest throughput** — Marlin fused INT4×FP16 GEMM kernels fit all weights in L2/SRAM, eliminating DRAM bandwidth bottlenecks. At c=1 it is 6× faster than fp16 and 20× faster than int8.

2. **Lowest latency** — p50 latency is 0.69s (AWQ4) vs 6.87s (fp16) vs 8.71s (INT8) at c=1. For a chatbot, this transforms the feel from "waiting" to "instant."

3. **No accuracy loss** — intent/value accuracy matches or exceeds fp16 despite 4× compression.

4. **More KV cache** — smaller model footprint leaves 22.5 GB free for the KV cache, enabling larger batches and longer contexts.

**Why INT8 (bitsandbytes) is the worst option:**

bitsandbytes quantizes weights but *dequantizes them back to bf16* at runtime before each matrix multiply. This saves memory but adds overhead — effectively the same compute cost as fp16 with extra memory operations. It is fundamentally slower than both fp16 and AWQ4.

---

### Running the sweep yourself

```bash
# Step 1 — Quantize to W4A16 (run once, ~5 min, requires WSL2 + GPU)
python quantize_awq.py
# Output: models/llama/llama-3.2-3b-w4a16/  (~3 GB)

# Step 2 — For each quant level: start server in WSL2, run sweep on Windows
bash deployment/scripts/start_server_quant.sh fp16
# (wait for "Application startup complete")
python eval_sweep.py --quant fp16 --concurrency 1 2 4 8 16

# Repeat for int8 and awq4, then generate plots:
python eval_plot.py
# → eval_results/plots/{throughput,latency,accuracy,qps_surface_3d}.png
```

---

## Speculative Decoding Experiments

We evaluated two speculative decoding strategies on top of AWQ4, using
[`eval_speculative.py`](eval_speculative.py) and [`deployment/scripts/start_server_spec.sh`](deployment/scripts/start_server_spec.sh).

### Strategy 1 — Llama-3.2-1B-Instruct draft model (K=4)

| Configuration | c=1 QPS | c=1 p50 | c=8 QPS | c=8 p50 | vs baseline |
|---|:---:|:---:|:---:|:---:|:---:|
| AWQ4 baseline | **0.90** | **0.69s** | **2.94** | **2.65s** | 1.00× |
| AWQ4 + 1B draft (K=4) | 0.04 | 11.95s | 0.65 | 4.58s | **0.04×** ❌ |

**Draft acceptance rate:** 48% (1.91 tokens accepted per step on average).

Despite a reasonable acceptance rate, the 1B draft model makes performance **25× worse** at c=1. Root cause: Marlin INT4 kernels already run the 3B model near HBM memory bandwidth saturation. Adding a 1B fp16 draft model forces two models to compete for the same memory bus — the drafting overhead dominates.

> **Lesson:** Speculative decoding with a separate draft model only helps when the target is large (13B+) and each forward pass is genuinely expensive. For a fast small model like 3B AWQ4, it is counterproductive.

### Strategy 2 — N-gram prompt look-up (recommended)

Zero extra VRAM, no draft model. vLLM scans the prompt and recent output for repeating n-gram patterns and proposes them as candidate tokens.

```bash
bash deployment/scripts/start_server_spec.sh awq4 5 ngram
```

Financial filings contain highly repetitive text (company names, metric labels, date ranges, citation patterns) — ideal for n-gram matching. Expected gain: **5–15% latency reduction** at zero cost.

### Server usage

```bash
# N-gram speculation (default, recommended for AWQ4)
bash deployment/scripts/start_server_spec.sh awq4 5 ngram

# 1B draft model (only beneficial for fp16 on larger targets)
bash deployment/scripts/start_server_spec.sh fp16 4 1b
```

---

## Benchmark vs GPT-4o (52 cases)

Full benchmark using [`eval_benchmark.py`](eval_benchmark.py):

| Metric | This pipeline (awq4) | GPT-4o |
|---|:---:|:---:|
| Intent accuracy | 78.8% | — |
| Value correct (±5%) | 70.3% | 75.0% |
| Keyword hit rate | 72.8% | 82.0% |
| Fluency (1–5, LLM-judged) | 3.9 | 4.3 |
| **Runs locally / no API cost** | ✅ | ❌ |
| **Answers from actual DB** | ✅ | ❌ |

GPT-4o scores higher on fluency and keyword overlap, but it frequently refuses to answer questions about recent fiscal years due to training-cutoff uncertainty. Our pipeline reads from the actual SQLite database and **never hallucinates a number**.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 24 GB |
| System RAM | 16 GB | 32 GB |
| Disk | 30 GB | 100 GB |
| CUDA | 11.8 | 12.4 |
| Python | 3.11 | 3.11 |

Tested on: **Windows 11 + RTX 3090 Ti (24 GB), CUDA 12.4, WSL2 (Ubuntu 22.04)**.

For vLLM serving the model must run in a Linux environment. On Windows, use WSL2. On Linux/Mac, run directly.

---

## Data Sources

| Source | What | How accessed |
|---|---|---|
| **SEC EDGAR XBRL API** | Structured financials (income statement, balance sheet, cash flow) | `data.sec.gov/api/xbrl/companyfacts/` — free, no auth |
| **SEC EDGAR filing index** | 10-K / 10-Q / 8-K full text | `data.sec.gov/submissions/` + `sec.gov/Archives/` — free, no auth |
| **HuggingFace Hub** | Llama-3.2-3B-Instruct weights | Gated — requires accepted licence + `HF_TOKEN` |

All data downloads respect the SEC EDGAR rate limit (10 req/s).

---

## Environment Variables

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Purpose |
|---|:---:|---|
| `HF_TOKEN` | ✅ | Download gated Llama weights |
| `OPENAI_API_KEY` | ❌ | GPT-4o fluency scoring in eval; dataset distillation |
| `VLLM_HOST` | ❌ | vLLM server host (default: `localhost`) |
| `VLLM_PORT` | ❌ | vLLM server port (default: `8001`) |

**Never commit `.env`** — it is listed in `.gitignore`.

---

## License

MIT — see [LICENSE](LICENSE).

The Llama model weights are subject to Meta's [Llama Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE).
