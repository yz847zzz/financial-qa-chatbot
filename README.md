# LLM Inference Acceleration — Financial QA on SEC Filings

A production-oriented study of **LLM inference acceleration** applied to a financial question-answering system. The application answers plain-English questions over 11,000+ SEC EDGAR filings using a RAG + NL2SQL pipeline; the primary research focus is throughput, latency, and accuracy across three quantization strategies running on consumer GPU hardware.

---

## Highlights

- **AWQ4 is the clear winner** — 6× faster than fp16 at c=1, 20% faster p50 latency overall, with only 0.46% composite accuracy loss
- **bitsandbytes INT8 is 52% slower than fp16** on RTX 3090 Ti due to runtime dequantization overhead — not recommended for production
- **All-layer INT4 quantization (W4A16)** — every Linear layer (attention + FFN) quantized with group-size 128; only `lm_head` kept in fp16 as it maps directly to token probabilities and is accuracy-sensitive
- **vLLM single-base multi-LoRA** — one 6 GB base model serves NL2SQL, intent classification, and query-rewriting adapters via SGMV batching; ~3× storage reduction vs. separate fine-tuned models
- **vLLM PagedAttention + continuous batching** — prefix cache hit rate 89.9% in practice; KV cache managed as virtual memory pages, enabling 3.32 QPS peak at c=16
- **99.5% value accuracy on 556-case benchmark** vs 42.9% for GPT-4o (knowledge cutoff causes factual hallucination)
- **Speculative decoding with 1B draft model hurts AWQ4** (25× slower at c=1); n-gram lookup is the correct strategy for small, fast models

---

## Throughput (QPS) vs Concurrency

| Concurrency | fp16 (bfloat16) | int8 (bitsandbytes) | **awq4 (Marlin)** |
|:-----------:|:---------------:|:-------------------:|:-----------------------:|
| 1           | 0.15            | 0.04                | **0.90**                |
| 2           | 0.51            | 0.18                | **1.34**                |
| 4           | 0.97            | 0.70                | **1.48**                |
| 8           | 2.35            | 1.47                | **2.94**                |
| 16          | 2.98            | 2.20                | **3.32**                |

> AWQ4 leads at **every** concurrency level. The gap is largest at low concurrency (6× faster than fp16 at c=1).

---

## Latency — p50 / p95 (seconds)

| Concurrency | fp16 p50/p95 | int8 p50/p95 | **awq4 p50/p95** |
|:-----------:|:------------:|:------------:|:----------------:|
| 1           | 6.87 / 9.22  | 8.71 / 156.3 | **0.69 / 4.97**  |
| 2           | 1.55 / 7.90  | 10.48 / 13.27 | **0.66 / 2.75** |
| 4           | 3.55 / 4.70  | 4.49 / 6.79  | **2.63 / 2.73**  |
| 8           | 3.30 / 3.39  | 4.73 / 5.44  | **2.65 / 2.69**  |
| 16          | 3.35 / 5.34  | 7.13 / 7.26  | **2.75 / 4.76**  |

> ⚠️ INT8 p95 at c=1 is **156 s** — bitsandbytes triggers CUDA kernel JIT on first request.

---

## Accuracy (556 test cases, c=1)

Scoring: intent-aware composite of value accuracy (65% weight for Type1), keyword hit rate, and cosine similarity vs 3 GPT-4o reference answers per question.

| Metric | fp16 | int8 | awq4 |
|---|:---:|:---:|:---:|
| **Overall composite** | **0.7177** | 0.7153 | 0.7131 |
| Type1 — exact facts (397 cases) | **0.7101** | 0.7122 | 0.7065 |
| Type2 — qualitative RAG (117 cases) | **0.7160** | 0.7009 | 0.6848 |
| Type3 — chat / meta (42 cases) | 0.7943 | 0.7844 | **0.8548** |
| Value accuracy (±5%) | 99.0% | **99.5%** | **99.5%** |
| Intent accuracy | **99.1%** | 98.6% | 97.8% |
| Latency p50 | 1.47s | 2.24s | **1.18s** |

> Type2 (qualitative) is the most quantization-sensitive dimension: awq4 trails fp16 by 3.1%.
> Type1 (SQL-grounded) is unaffected — all three reach ≥99% value accuracy.

---

## vs GPT-4o

| Metric | This pipeline (awq4) | GPT-4o |
|---|:---:|:---:|
| Value accuracy (±5%) | **99.5%** | 42.9% |
| Overall composite | **0.7131** | 0.6172 |
| Keyword hit rate | **78.0%** | 75.4% |
| Semantic similarity | 0.548 | **0.857** |
| Runs locally, no API cost | ✅ | ❌ |
| Answers from live DB | ✅ | ❌ |

GPT-4o generates more fluent prose (higher semantic sim) but **fails on 57% of exact-value questions** due to knowledge-cutoff hallucination. This pipeline reads from a live SQLite database and never fabricates a number.

---

## VRAM Footprint

| Quantization | Model weights | Free for KV cache | Max batch |
|---|:---:|:---:|:---:|
| fp16 (bfloat16) | ~6 GB | ~18 GB | 64 seq |
| int8 (bitsandbytes) | ~3 GB | ~21 GB | 96 seq |
| awq4 (Marlin) | ~1.5 GB | ~22.5 GB | 128 seq |

---

## Quantization Strategy: All Linear Layers (lm_head excluded)

W4A16 INT4 quantization is applied to **all Linear layers** with group-size 128 via llm-compressor oneshot GPTQ. The single exception is `lm_head`.

```python
recipe = QuantizationModifier(
    config_groups={"group_0": QuantizationScheme(
        targets=["Linear"],          # all attention + FFN layers
        weights=QuantizationArgs(num_bits=4, group_size=128, ...),
    )},
    ignore=["lm_head"],              # kept in fp16
)
```

**Why exclude only `lm_head`:**
`lm_head` maps the last hidden state directly to logits over the full 32k-token vocabulary — errors here propagate without correction and directly change which token is sampled. It is also tiny (~50 MB) so keeping it in fp16 costs almost nothing in VRAM.

**Why quantize all other layers equally:**
Both attention and FFN layers are memory-bandwidth-bound at low batch sizes. Quantizing all of them to INT4 reduces the weight footprint from ~6 GB to ~1.5 GB (4× reduction), which is the primary driver of the throughput gains. Calibration-based GPTQ on domain-specific financial QA data compensates for reduced precision by choosing per-group scale factors that minimise layer-wise reconstruction error.

**Result:** <0.5% overall accuracy degradation (99.5% value accuracy vs 99.0% for fp16) across 556 financial QA cases.

**Why not bitsandbytes INT8?** bitsandbytes dequantizes weights back to bf16 at runtime before each matrix multiply, adding coordination overhead with no compute savings. Result: 52% slower p50 latency than fp16 on RTX 3090 Ti in current tests; warrants further investigation before production use.

---

## vLLM Serving Design

### Single Base + Multi-LoRA

The system deploys **one Llama-3.2-3B-Instruct base model** registered with task-specific LoRA adapters:

| Role | Adapter | Storage |
|---|---|---|
| NL-to-SQL translation | `nl2sql` LoRA (r=16) | ~16 MB |
| Intent classification | base + few-shot prompt | 0 MB |
| Query rewriting | base + few-shot prompt | 0 MB |
| Answer generation | base model | 0 MB |
| **Total** | 1 base + 1 trained adapter | **~6 GB** |

Traditional approach (3 fine-tuned 3B models) would require ~18 GB. vLLM's **SGMV (Segmented Gather Matrix-Vector) kernel** enables multiple LoRA adapters to be active in the same GPU batch simultaneously — adapter weights are hot-swapped per request with near-zero overhead.

### Continuous Batching + PagedAttention

```
Request A ──┐                    ┌── Response A
Request B ──┤  vLLM Engine       ├── Response B
Request C ──┘  (continuous       └── Response C
               batching)
               ↕
           PagedAttention
           KV cache pool
           (virtual pages)
```

- **Continuous batching** — new requests join the batch mid-generation; GPU is never idle waiting for one long request to finish
- **PagedAttention** — KV cache is managed in fixed-size pages like OS virtual memory; no fragmentation, enables longer contexts and larger effective batch sizes
- **Prefix caching** — system prompt + few-shot examples are cached as KV pages; in practice 89.9% of requests hit the prefix cache, giving near-instant time-to-first-token for common query patterns

---

## System Architecture

```
User question
  │
  ├─ decompose()          splits compound questions (base model)
  │
  └─ per sub-question:
      ├─ classify_intent()  → Type1 / Type2 / Type3
      │
      ├─ Type1 (exact fact)
      │     nl2sql LoRA → SQL → SQLite panel table → answer
      │
      ├─ Type2 (qualitative)
      │     rewrite_query() → BM25s + dense hybrid retrieval
      │     → cross-encoder rerank → MMR dedup → answer from chunks
      │
      └─ Type3 (chat / meta)
            direct base model answer
```

**RAG pipeline** (Type2): bm25s sparse inverted index + ChromaDB HNSW dense → Reciprocal Rank Fusion → `ms-marco-MiniLM-L-6-v2` cross-encoder rerank → MMR diversity selection.

---

## Speculative Decoding Experiments

### Strategy 1 — Llama-3.2-1B draft model (K=4) ❌

| Configuration | c=1 QPS | c=1 p50 | vs baseline |
|---|:---:|:---:|:---:|
| AWQ4 baseline | **0.90** | **0.69s** | 1.0× |
| AWQ4 + 1B draft | 0.04 | 11.95s | **0.04×** |

Draft acceptance rate was 48%, but performance was **25× worse**. Root cause: Marlin INT4 kernels already saturate HBM bandwidth; a second fp16 model competing for the same memory bus causes the drafting overhead to dominate.

> **Lesson:** speculative decoding with a separate draft model only helps when the target model is large (≥13B) and each forward pass is the bottleneck. For a fast small model like 3B AWQ4, it is counterproductive.

### Strategy 2 — N-gram prompt look-up ✅

Zero VRAM cost; vLLM scans prompt + recent output for repeating n-gram patterns. Financial filings are highly repetitive (company names, metric labels, date ranges) — expected **5–15% latency reduction** at zero cost.

```bash
bash deployment/scripts/start_server_spec.sh awq4 5 ngram
```

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3090 Ti, 24 GB GDDR6X |
| Host OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| CUDA | 12.4 · vLLM 0.6.x |
| Base model | Llama-3.2-3B-Instruct |
| Eval | 556 financial QA cases · GPT-4o reference answers |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yz847zzz/financial-qa-chatbot.git
cd financial-qa-chatbot

# 2. Environment (requires CUDA 12.x)
conda env create -f environment.yml && conda activate finqa

# 3. Secrets
cp .env.example .env   # add HF_TOKEN (required) and OPENAI_API_KEY (optional)

# 4. Download model (~6 GB)
python scripts/data/download_model.py

# 5. Build data stores
python data_pipeline/scripts/fetch_xbrl.py        # structured financials → SQLite
python data_pipeline/scripts/run_ingest.py \       # filings text → ChromaDB
    --ticker AAPL MSFT NVDA GOOGL META --filings-dir data/filings

# 6. Serve (WSL2) + chat (Windows)
bash deployment/scripts/start_server_quant.sh awq4   # recommended
python chatbot.py
```

### Quantization sweep

```bash
# Quantize to W4A16 (one-time, ~5 min)
python scripts/model/quantize_awq.py

# For each quant level: start server → run eval → Ctrl-C → next
bash deployment/scripts/start_server_quant.sh fp16
python eval/eval_unified.py --config vllm --label fp16 \
    --testset eval/testdata/testcases.json \
    --references eval/testdata/references.json \
    --skip-throughput
```

---

## Fine-tuning the NL2SQL Adapter

```bash
python finetune/data_prep/generate_nl2sql_templates.py
python finetune/adapters/nl2sql/NL2SQL_SFT.py   # ~45 min on RTX 3090 Ti
```

Uses QLoRA (NF4 4-bit base + bf16 adapters, r=16) with completion-only loss. Trained on 1,381 examples; exec accuracy 100% on held-out set.

---

## Data Sources

| Source | Content | Access |
|---|---|---|
| SEC EDGAR XBRL API | Structured financials (IS, BS, CF) | `data.sec.gov/api/xbrl/` — free |
| SEC EDGAR filing index | 10-K / 10-Q / 8-K full text | `sec.gov/Archives/` — free |
| HuggingFace Hub | Llama-3.2-3B-Instruct weights | Gated — requires `HF_TOKEN` |

---

## License

MIT. Llama weights subject to [Meta's Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE).
