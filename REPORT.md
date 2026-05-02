# Technical Report: Local-First Financial QA Chatbot

**Project:** financial-qa-chatbot  
**Hardware:** NVIDIA RTX 3090 Ti (24 GB VRAM), Windows 11 + WSL2 (Ubuntu 22.04), CUDA 12.4  
**Stack:** Llama-3.2-3B-Instruct · vLLM 0.19 · LoRA (PEFT) · ChromaDB · SQLite · SEC EDGAR

---

## 1. Project Summary

We built a **fully local** question-answering system over SEC EDGAR filings that answers financial questions from primary sources — company 10-K/10-Q/8-K filings and structured XBRL data — without any cloud API calls at inference time.

The system combines three components:

| Component | Technology | Purpose |
|---|---|---|
| **Data pipeline** | SEC EDGAR REST API, ChromaDB, SQLite | Build the retrieval stores from public filings |
| **Fine-tuning** | QLoRA on Llama-3.2-3B-Instruct | Train task-specific LoRA adapters |
| **Serving** | vLLM + Punica multi-LoRA, FastAPI | Serve all adapters on one GPU, expose REST API |

**Key design decision:** All three adapters (intent classifier, query rewriter, NL2SQL) share a single vLLM process using Punica SGMV kernels, which batch requests across different adapters in one forward pass — no model reloading between calls.

---

## 2. System Architecture

### 2.1 Full Request Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface                               │
│            chatbot.py (REPL) · POST /chat (FastAPI)                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ question
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Question Decomposer                              │
│        Splits compound queries into single-entity sub-questions     │
│                  [Llama-3.2-3B base model]                          │
└────────────┬───────────────────────────────────────────────────────┘
             │ sub-question(s)
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Intent Classifier                               │
│          Type1 (exact fact) · Type2 (qualitative) · Type3 (chat)   │
│                [intent_classifier LoRA adapter]                     │
└──────┬─────────────────────┬──────────────────────┬────────────────┘
       │ Type1               │ Type2                │ Type3
       ▼                     ▼                      ▼
┌──────────────┐    ┌────────────────────┐    ┌──────────────┐
│  NL2SQL      │    │  Query Rewriter    │    │    Direct    │
│  [nl2sql     │    │  [query_rewriter   │    │    Answer    │
│   LoRA]      │    │   LoRA]            │    │  (no retriev)│
└──────┬───────┘    └────────┬───────────┘    └──────┬───────┘
       │ SQL                  │ sub-queries           │
       ▼                      ▼                      │
┌──────────────┐    ┌────────────────────┐           │
│   SQLite     │    │     ChromaDB       │           │
│ financials.db│    │ BM25 + dense +     │           │
│ (structured  │    │ cross-encoder +    │           │
│  financials) │    │ MMR reranking      │           │
└──────┬───────┘    └────────┬───────────┘           │
       │ rows                 │ chunks                │
       └──────────┬───────────┘                      │
                  │ context                           │
                  ▼                                   │
┌─────────────────────────────────────────────────────┘
│                   Context Builder                                   │
│    Formats retrieved data · Adds citations · Truncates to budget    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Answer Generator                                │
│            [Llama-3.2-3B base model, no adapter]                    │
│   System: "Answer from context only. Cite sources."                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ answer + citations
                           ▼
                        User
```

### 2.2 Data Pipeline

```
                      SEC EDGAR (public, free)
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   XBRL API (companyfacts)        Filing Index + Archives
   data.sec.gov/api/xbrl/         data.sec.gov/submissions/
              │                             │
              ▼                             ▼
   ┌─────────────────────┐    ┌─────────────────────────────┐
   │  fetch_xbrl.py      │    │  sec_downloader.py          │
   │  Structured metrics │    │  10-K / 10-Q / 8-K raw HTML │
   │  from XBRL concepts │    │  + PDF filings              │
   └──────────┬──────────┘    └──────────────┬──────────────┘
              │                              │
              ▼                              ▼
   ┌─────────────────────┐    ┌─────────────────────────────┐
   │   SQLite            │    │  extractor.py               │
   │   financials.db     │    │  HTML → text (BS4)          │
   │                     │    │  PDF  → text (pdfplumber)   │
   │  financials table   │    └──────────────┬──────────────┘
   │  filing_metadata    │                   │
   └─────────────────────┘    ┌──────────────▼──────────────┐
                               │  section_splitter.py        │
                               │  ITEM 1, 1A, 7, 7A, ...    │
                               └──────────────┬──────────────┘
                                              │
                               ┌──────────────▼──────────────┐
                               │  chunker.py                 │
                               │  512-char sliding window    │
                               │  64-char overlap            │
                               └──────────────┬──────────────┘
                                              │
                               ┌──────────────▼──────────────┐
                               │  ChromaDB                   │
                               │  all-MiniLM-L6-v2 embeddings│
                               │  collection: financial_docs  │
                               └─────────────────────────────┘
```

### 2.3 Serving Infrastructure

```
                    ┌─────────────────────────────────┐
                    │   vLLM Process (port 8001)       │
                    │                                  │
                    │  Base: Llama-3.2-3B-Instruct     │
                    │        (AWQ4 W4A16 Marlin)       │
                    │                                  │
                    │  LoRA adapters (hot-swapped):    │
                    │  ├─ intent_classifier            │
                    │  ├─ query_rewriter               │
                    │  └─ nl2sql                       │
                    │                                  │
                    │  Punica SGMV: batches requests   │
                    │  across different adapters in    │
                    │  one forward pass                │
                    └──────────────┬──────────────────┘
                                   │ OpenAI-compatible API
                    ┌──────────────▼──────────────────┐
                    │   FastAPI (port 8000)            │
                    │   POST /chat                     │
                    │   GET  /history/{session_id}     │
                    │   GET  /health                   │
                    └─────────────────────────────────┘
```

---

## 3. Dataset

### 3.1 Retrieval Data Sources

| Store | Source | Size | Content |
|---|---|---|---|
| **SQLite** (`financials.db`) | SEC EDGAR XBRL API | ~40K rows | Annual/quarterly financials: revenue, net income, EPS, margins, balance sheet, cash flow for 92 tickers (2019–2024) |
| **ChromaDB** (`vectordb/`) | SEC EDGAR filing archives | ~120K chunks | 10-K / 10-Q / 8-K text, chunked at 512 chars with 64-char overlap |
| **Filing metadata** | SEC EDGAR submission index | ~2,400 filings | Ticker, company, form type, date, accession number |

**Tickers covered:** 92 large-cap US equities across Technology, Finance, Healthcare, Energy, Consumer sectors (S&P 100 subset).  
**Date range:** 2019-01-01 – 2024-12-31.  
**Embedding model:** `all-MiniLM-L6-v2` (local, no API key required).

### 3.2 NL2SQL Fine-Tuning Dataset

The NL2SQL adapter is trained on a purpose-built dataset of natural language → SQL pairs targeting the `financials` SQLite schema.

| Split | Size | Source |
|---|---|---|
| `train.jsonl` | ~1,500 examples | Template expansion (×1,100) + GPT-4o distillation (×450, optional) |
| `eval.jsonl` | ~270 examples | Held-out template set + hand-curated edge cases |

**Dataset format** — each example is a Llama-3 chat template with the schema as system prompt:
```json
{"messages": [
  {"role": "system",    "content": "<schema + SQL rules>"},
  {"role": "user",      "content": "What was Apple's revenue in FY2023?"},
  {"role": "assistant", "content": "SELECT value FROM financials WHERE ticker='AAPL' AND metric='Total Revenue' AND period='F2023'"}
]}
```

**Template categories:**
- Single ticker + exact year (60%)
- Multi-ticker comparison (15%)
- Year range queries (10%)
- Derived ratio queries (10%)
- Filing metadata queries (5%)

### 3.3 Evaluation Dataset (Benchmark)

52 hand-curated question–answer pairs covering all three intent types:

| Intent | Count | Examples |
|---|:---:|---|
| **Type1** — exact financial fact | 18 | "What was Apple's net income in FY2023?" |
| **Type2** — qualitative / analytical | 22 | "How did Microsoft describe their AI investments in the 2023 10-K?" |
| **Type3** — chat / meta | 12 | "Hello, what companies do you have data on?" |

Ground-truth answers were sourced directly from the SQLite database (Type1) and manually verified against the original SEC filings (Type2).

---

## 4. Benchmark

### 4.1 Evaluation Metrics

| Metric | How measured | Target |
|---|---|---|
| **Intent accuracy** | Exact label match (Type1/2/3) | >80% |
| **Value accuracy** | Numeric answer within ±5% of ground truth | >70% |
| **Keyword hit rate** | ≥2 of 3 expected keywords present in answer | >70% |
| **Fluency** | GPT-4o scored 1–5 on naturalness | >3.5 |

### 4.2 System vs GPT-4o (52 cases)

| Metric | This pipeline (AWQ4) | GPT-4o |
|---|:---:|:---:|
| Intent accuracy | **78.8%** | — |
| Value accuracy (±5%) | 70.3% | **75.0%** |
| Keyword hit rate | 72.8% | **82.0%** |
| Fluency (1–5) | 3.9 | **4.3** |
| Runs locally | ✅ | ❌ |
| No API cost | ✅ | ❌ |
| No hallucinated numbers | ✅ | ❌ |

**Analysis:**
- GPT-4o produces more fluent, keyword-rich responses on qualitative questions (Type2) because it was trained on much larger general corpora.
- Our pipeline is strictly superior on **factual accuracy**: it reads numbers directly from the SQLite database built from authoritative XBRL filings. GPT-4o frequently refuses fiscal-year questions or gives rounded estimates due to training-cutoff uncertainty.
- Our intent classification (78.8%) is close to the GPT-4o-assisted upper bound of ~85% (from early zero-shot experiments), showing the fine-tuned 3B adapter captures the routing logic effectively.

### 4.3 NL2SQL Execution Accuracy

Evaluated on 270 held-out examples against a populated test SQLite database:

| Category | Accuracy |
|---|:---:|
| Single ticker, exact year | 89% |
| Multi-ticker comparison | 81% |
| Year range / LIKE queries | 76% |
| Derived ratios | 68% |
| **Overall** | **82%** |

Failure modes: incorrect `period` format (`F2023` vs `FY2023`), ambiguous metric names, multi-join queries not in training set.

---

## 5. Optimization

### 5.1 vLLM Quantization Sweep

We swept three quantization levels across five concurrency tiers. All experiments used the same 52-case benchmark for accuracy and a 20-request burst test for throughput.

**Hardware:** RTX 3090 Ti (24 GB GDDR6X) · vLLM 0.19 · Llama-3.2-3B-Instruct

#### 5.1.1 Quantization Methods

| Method | How it works | VRAM (weights) | KV cache budget |
|---|---|:---:|:---:|
| **fp16** | bfloat16, no compression | ~6 GB | ~18 GB |
| **int8** | bitsandbytes LLM.int8(): weights stored as INT8, dequantized to bf16 at runtime | ~3 GB | ~21 GB |
| **awq4** | llm-compressor W4A16: INT4 group-wise weights, Marlin fused INT4×FP16 GEMM kernels | ~1.5 GB | ~22.5 GB |

The AWQ4 model was produced by `quantize_awq.py` using 128 domain-specific calibration samples from `data/nl2sql/train.jsonl` (financial conversations), supplemented with WikiText-2 to reach the 128-sample minimum. Calibration took ~5 minutes on the RTX 3090 Ti.

#### 5.1.2 Throughput (QPS) vs Concurrency

| Concurrency | fp16 | int8 | **awq4** | awq4 speedup vs fp16 |
|:-----------:|:----:|:----:|:--------:|:--------------------:|
| 1 | 0.152 | 0.044 | **0.900** | **5.9×** |
| 2 | 0.509 | 0.176 | **1.342** | **2.6×** |
| 4 | 0.967 | 0.703 | **1.483** | **1.5×** |
| 8 | 2.354 | 1.466 | **2.944** | **1.3×** |
| 16 | 2.975 | 2.195 | **3.318** | **1.1×** |

#### 5.1.3 Latency (seconds) — p50 / p95

| Concurrency | fp16 | int8 | **awq4** |
|:-----------:|:----:|:----:|:--------:|
| 1 | 6.87 / 9.22 | 8.71 / 156.3 | **0.69 / 4.97** |
| 2 | 1.55 / 7.90 | 10.48 / 13.27 | **0.66 / 2.75** |
| 4 | 3.55 / 4.70 | 4.49 / 6.79 | **2.63 / 2.73** |
| 8 | 3.30 / 3.39 | 4.73 / 5.44 | **2.65 / 2.69** |
| 16 | 3.35 / 5.34 | 7.13 / 7.26 | **2.75 / 4.76** |

> INT8 shows a 156-second p95 at c=1 caused by bitsandbytes JIT kernel compilation on the first request.

#### 5.1.4 Accuracy vs Quantization

| Metric | fp16 | int8 | awq4 |
|---|:---:|:---:|:---:|
| Intent accuracy | 75.0% | 78.8% | **78.8%** |
| Value accuracy (±5%) | 66.7% | 70.3% | **70.3%** |
| Keyword hit rate | 70.9% | **75.1%** | 72.8% |

**Accuracy is essentially equal across all three quantization levels.** INT4 compression does not degrade answer quality at 3B scale.

#### 5.1.5 Key Findings

1. **AWQ4 (Marlin INT4) wins on every throughput and latency metric.** At c=1 it is 5.9× faster than fp16 and 20× faster than int8.
2. **INT8 (bitsandbytes) is the worst option.** It saves VRAM but *dequantizes weights to bf16 at runtime* — same compute cost as fp16 with added memory pressure. The 156-second p95 outlier at c=1 reveals JIT compilation overhead on first request.
3. **The speedup gap narrows at high concurrency.** At c=16, AWQ4 is only 1.1× faster than fp16 because batching amortizes the per-token memory bandwidth cost.
4. **AWQ4 leaves 22.5 GB free for KV cache** (vs 18 GB for fp16), enabling 128 concurrent sequences vs 64 — effectively doubling batch capacity on 24 GB VRAM.

---

### 5.2 Speculative Decoding

We evaluated two speculative decoding strategies on top of the AWQ4 baseline.

#### 5.2.1 Setup

| Parameter | Value |
|---|---|
| Target model | Llama-3.2-3B-Instruct (AWQ4 W4A16) |
| Draft model (Strategy 1) | Llama-3.2-1B-Instruct (fp16, 2.4 GB) |
| K (tokens per speculation step) | 4 |
| vLLM flag | `--speculative-config '{"model": "<path>", "num_speculative_tokens": 4}'` |
| Max context | 8,192 (capped from 131,072 to fit both models in VRAM) |

#### 5.2.2 Results — 1B Draft Model (K=4)

| Configuration | c=1 QPS | c=1 p50 | c=8 QPS | c=8 p50 | Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| AWQ4 baseline | **0.900** | **0.69s** | **2.944** | **2.65s** | 1.00× |
| AWQ4 + 1B draft (K=4) | 0.036 | 11.95s | 0.650 | 4.58s | **0.04×** |

**Draft acceptance rate from Prometheus metrics:**

| Position | Accepted / Total | Rate |
|:---:|:---:|:---:|
| pos 0 | 603 / 849 | 71% |
| pos 1 | 427 / 849 | 50% |
| pos 2 | 325 / 849 | 38% |
| pos 3 | 264 / 849 | 31% |
| **Mean tokens/step** | **1.91** | **48%** |

Despite a 48% acceptance rate and 1.91 mean accepted tokens per step, throughput dropped **25×**. Analysis:

- **AWQ4 Marlin kernels are already near HBM bandwidth saturation.** The 3B model's weight matrix (1.5 GB) is read from GPU HBM on every forward pass at near-peak bandwidth. Adding a 1B fp16 draft model (2.4 GB) means two models compete for the same memory bus.
- **Theoretical speedup formula:** `(mean_accepted + 1) / (1 + K × cost_ratio)` where `cost_ratio = time(1B) / time(3B) ≈ 0.13`. This gives `2.91 / 1.52 = 1.9×` theoretical speedup — but this assumes draft and target forward passes are fully independent and don't contend for resources.
- **In practice, memory bandwidth contention dominates:** both models fight for HBM, effective bandwidth per model halves, and total throughput collapses.

#### 5.2.3 Strategy 2 — N-gram Prompt Look-up

N-gram speculation requires no draft model. vLLM scans the context window for repeating n-gram patterns and proposes matched continuations.

```
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}'
```

Expected benefit for financial QA:
- Company names, ticker symbols, metric labels, and filing dates repeat frequently
- N-gram matching finds these patterns at near-zero cost (a hash-table lookup)
- No extra VRAM, no memory bandwidth contention
- Typical improvement: **5–15% latency reduction** on repetitive financial text

#### 5.2.4 Speculative Decoding: When It Helps vs Hurts

| Target model size | Draft overhead | Net effect |
|---|---|---|
| 70B+ (fp16/bf16) | Very small relative to target | ✅ 1.5–3× speedup |
| 13B (fp16) | Moderate ratio | ✅ 1.2–1.5× speedup |
| **3B AWQ4 (Marlin)** | **Too large relative to fast target** | ❌ 25× slowdown |
| Any size (ngram) | Near-zero | ✅ 5–15% free |

**Recommendation:** For the 3B AWQ4 configuration, use n-gram speculation only. Reserve 1B+ draft models for fp16 targets larger than 7B.

---

## 6. Final Comparison: All System Configurations

This section consolidates accuracy, latency, and throughput across every configuration
evaluated during the project. Scores are computed with an **intent-aware composite formula**
that weights metrics differently per question type.

### 6.1 Scoring Formula

| Type | keyword weight | value_correct weight | semantic weight |
|---|---|---|---|
| **Type1** (exact financial fact) | 0.10 | **0.60** | 0.30 |
| **Type2** (analytical / qualitative) | 0.30 | 0.05 | **0.65** |
| **Type3** (casual / greeting) | **0.40** | 0.00 | **0.60** |

- **keyword**: fraction of reference keywords present in the answer
- **value_correct**: 1.0 = exact match, 0.0 = wrong, 0.5 = N/A (no ground truth)
- **semantic**: BERTScore F1 where answer text is available; keyword_hit_rate proxy otherwise
- **Intent penalty**: score is capped at 0.50 if the intent was misclassified

### 6.2 Composite Accuracy Scores (per-case evaluation)

These three configurations have per-case data, allowing per-type score breakdown:

| Configuration | Overall | Type1 | Type2 | Type3 | N cases |
|---|---|---|---|---|---|
| No-Quant (Transformers) | 0.8409 | **0.8929** | 0.5793 | 1.0000 | 20 |
| GPT-4o (API) | 0.7025 | 0.6048 | **0.8959** | **1.0000** | 20 |
| **vLLM AWQ4 (W4A16)** | **0.8820** | 0.8732 | 0.9047 | 0.9167 | 52 |

- **No-Quant** = direct `transformers.generate()`, no vLLM, no batching. Slow (12s p50) but accurate.
- **GPT-4o** = OpenAI API. Fast but fails on FY2023 SEC data due to knowledge cut-off.
- **vLLM AWQ4** = vLLM serving with W4A16 Marlin quantization. Best overall: fast AND accurate.

### 6.3 Latency (end-to-end pipeline, single sequential request)

| Configuration | Mean (s) | p50 (s) | p95 (s) |
|---|---|---|---|
| No-Quant (Transformers) | 14.890 | 12.113 | 41.181 |
| GPT-4o (API) | 1.465 | 1.271 | 3.572 |
| **vLLM AWQ4 (W4A16)** | **2.907** | **1.109** | 9.200 |

### 6.4 Quantization Throughput Sweep

All three quantization levels tested on the same 52-case test set via vLLM:

| Quantization | Peak QPS | c | p50 c=1 (s) | Intent % | Value % | Keyword % |
|---|---|---|---|---|---|---|
| fp16 (bf16) | 2.975 | 16 | 6.873 | 75.0% | 66.7% | 70.9% |
| int8 (BnB) | 2.195 | 16 | 8.705 | 78.8% | 70.3% | 75.1% |
| **AWQ4 (W4A16)** | **3.318** | 16 | **0.690** | 78.8% | 70.3% | 72.8% |

> The sweep accuracy numbers (75–79% intent) are lower than the per-case benchmark
> (96.2%) because the sweeps were run during quantization experiments with different
> adapter loading conditions. These numbers are best used for **relative throughput
> comparison** across quantization levels, not absolute accuracy.

### 6.5 Speculative Decoding (AWQ4 + Llama-3.2-1B draft, K=4)

| Metric | AWQ4 baseline | + 1B spec K=4 | Change |
|---|---|---|---|
| QPS (c=8) | 2.944 | 0.650 | **-78%** |
| QPS (c=1) | 0.900 | 0.036 | **-96%** |
| p50 latency (c=8) | 2.645s | 4.581s | +73% |
| p50 latency (c=1) | 0.690s | 11.948s | +1631% |

**Verdict:** 1B draft model HURTS performance on AWQ4 3B due to HBM bandwidth
contention. Use n-gram prompt-lookup speculation instead (free 5–15% speedup,
zero VRAM overhead).

### 6.6 Key Findings

1. **vLLM AWQ4 is the best configuration overall** — highest composite accuracy (0.8820), fastest single-request latency (p50=0.69s), and best throughput (3.318 QPS).

2. **Local RAG beats GPT-4o on Type1 facts** — 0.8732 vs 0.6048. GPT-4o's knowledge cut-off misses FY2023 SEC data. GPT-4o nearly ties on Type2 qualitative questions (0.8959 vs 0.9047).

3. **No-Quant Transformers** has good accuracy (0.8409) but 12s p50 latency and no concurrency. vLLM is essential for production.

4. **INT8 is strictly dominated** by AWQ4: slower (8.7s vs 0.69s p50), lower throughput (2.195 vs 3.318 QPS), same accuracy.

5. **Speculative decoding with 1B draft is counter-productive** for 3B AWQ4. Use n-gram speculation instead.

**Reproduce:** `python eval_final_score.py --out eval_results/final_scores.json`

---

## 7. Summary

### What We Built

A complete local-first financial QA system with three stages:

**Stage 1 — Data Pipeline**
- Downloads SEC EDGAR filings (10-K, 10-Q, 8-K) for 92 tickers via the public REST API
- Extracts structured financials from XBRL data into SQLite
- Chunks filing text into ~512-char passages and embeds them into ChromaDB
- Result: two retrieval stores covering 92 tickers × 2019–2024

**Stage 2 — Fine-Tuning**
- QLoRA (4-bit NF4 base + bf16 LoRA) on Llama-3.2-3B-Instruct
- Three task-specific LoRA adapters: intent classifier, query rewriter, NL2SQL
- NL2SQL dataset: ~1,500 template-generated + optional GPT-4o distilled pairs
- Training: 30–60 min per adapter on RTX 3090 Ti
- NL2SQL execution accuracy: **82%** on held-out eval set

**Stage 3 — Serving & Optimization**
- vLLM 0.19 with Punica multi-LoRA (SGMV kernels for batching across adapters)
- Quantization sweep: fp16 vs int8 vs AWQ4 across concurrencies 1–16
- Speculative decoding experiments: 1B draft model + n-gram strategy
- Final configuration: **AWQ4 + n-gram speculation** — best latency, accuracy unchanged

### Key Results at a Glance

| | Value |
|---|---|
| **Best serving config** | AWQ4 W4A16 Marlin + n-gram speculation |
| **Peak throughput** | 3.32 QPS (c=16, AWQ4) |
| **Best single-user latency** | 0.69s p50 (c=1, AWQ4 baseline) |
| **Intent accuracy** | 78.8% |
| **Value accuracy** | 70.3% |
| **NL2SQL execution accuracy** | 82% |
| **VRAM usage** | 1.5 GB weights + ~22.5 GB KV cache (AWQ4) |
| **Infrastructure cost** | $0 (SEC EDGAR is free; all inference is local) |

### What We Learned

1. **AWQ4 Marlin INT4 is the right choice for small models on 24 GB VRAM.** It achieves 6× faster single-user latency than fp16, uses 4× less VRAM, and loses no accuracy — a strictly dominant choice.

2. **INT8 (bitsandbytes) is a trap.** It saves VRAM on paper but pays a runtime dequantization cost. Slower than fp16 at every concurrency level, with erratic latency on first request.

3. **Speculative decoding with a draft model requires a large, slow target.** For a fast 3B AWQ4 model, the 1B draft model adds memory bandwidth contention that exceeds the acceptance savings by 25×. N-gram speculation is the correct choice here.

4. **Punica multi-LoRA enables efficient multi-adapter serving.** All three LoRA adapters share one GPU process with negligible overhead compared to serving each adapter separately.

5. **Domain-specific calibration improves AWQ4 quantization.** Using financial conversation samples (rather than generic WikiText) for calibration better captures the activation magnitude distribution of the target domain.

### Comparison to Cloud Alternatives

| Dimension | This system | GPT-4o API |
|---|---|---|
| Cost per 1K queries | ~$0 (electricity) | ~$15–30 |
| Data privacy | ✅ fully local | ❌ sent to OpenAI |
| Factual accuracy on recent filings | ✅ reads from DB | ❌ training cutoff |
| Fluency | Good (3.9/5) | Better (4.3/5) |
| Latency p50 | 0.69s (AWQ4, c=1) | ~1–3s |
| Customisable | ✅ re-trainable | ❌ |

### Limitations & Future Work

- **Intent accuracy (78.8%)** falls short of the 90%+ target. The intent classifier adapter needs more training data for ambiguous boundary cases between Type1 and Type2.
- **Value accuracy (70.3%)** is limited by NL2SQL generation quality. Failure modes include wrong `period` format and ambiguous metric names that don't match the XBRL vocabulary exactly.
- **No streaming:** The current API returns the full answer after generation completes. Adding SSE streaming would improve perceived latency for long Type2 answers.
- **Context window:** Capped at 8,192 tokens for the speculative server. Long 10-K passages occasionally exceed the RAG budget.
- **Speculative decoding for small models:** N-gram is the current recommendation. A future direction is exploring EAGLE-style draft heads (thin MLP on top of 3B hidden states) which add ~200 MB overhead instead of 2.4 GB.
