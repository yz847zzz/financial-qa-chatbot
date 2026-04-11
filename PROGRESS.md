# Financial QA Chatbot — Daily Progress Log

---

## 2026-04-09

**Part 1 (Data Pipeline) — Completed**
- Ingested SEC 10-K filings for ~96 S&P tickers (FY2019–FY2024)
- ChromaDB: 516,955 chunks at `data/vectordb/`
- SQLite: `data/financials.db` — `financials` (62k rows), `filing_metadata` (2,624 rows), `panel` (490 rows, 30 features)

**Part 2 (Finetune) — Started**
- Downloaded `meta-llama/Llama-3.2-3B-Instruct` to `models/llama/`; tested on RTX 3090 Ti (2.24GB VRAM in NF4, ~20 tok/s)
- Built NL2SQL SFT dataset: 662 examples (560 train / 102 eval) at `data/nl2sql/`
  - 160 hand-crafted + 450 GPT-4o distilled + 300 WikiSQL adapted
  - Execution-tested against real DB; LLM-judge avg 3.80/5.0

---

## 2026-04-10

**Part 2 (Finetune) — NL2SQL LoRA design**
- Decided training strategy for NL2SQL LoRA adapter:
  - Loss: completion-only cross-entropy (mask prompt tokens, loss on SQL output only)
  - QLoRA: NF4 4-bit base + bf16 adapters; TRL `SFTTrainer`
  - 5 epochs, lr=2e-4 cosine, effective batch=16 (2×grad_accum 8), max_seq=512
  - LoRA: r=16, alpha=32, dropout=0.05; targets all attention + MLP projections
- Implemented `finetune/adapters/nl2sql/NL2SQL_SFT.py` — QLoRA SFT with completion-only loss, loss curve plot, run metadata JSON

**Part 1 (Data Pipeline) — DB audit & rebuild**
- Discovered two critical bugs in `panel` table: 52.6% rows had `statement='unknown'`; balance sheet columns contaminated with cash-flow delta values (negative inventories, negative accounts_payable)
- Rewrote `data_pipeline/scripts/clean_panel.py`:
  - Statement-type filtering: balance sheet metrics only from `statement='balance_sheet'` rows
  - Priority sort bug fixed: `sort_values(["_rank","value"], ascending=[True,False]).drop_duplicates(keep="first")` — AAPL revenue corrected from 29.6B → 383.3B
  - Added `total_revenue` column (was missing entirely)
  - Plausibility bounds: nulled 1,611 extraction errors (e.g. ACN $14 quadrillion revenue, TSLA ROA=260%)
  - Fixed `net_margin` formula: was `net_income/total_assets` (= ROA), corrected to `net_income/total_revenue`
- Panel result: 92 tickers × 6 FY years, 33 columns, all values verified

**Part 2 (Finetune) — NL2SQL dataset rebuild & training**
- Rebuilt NL2SQL dataset from scratch targeting `panel` schema (previous 1,099 examples all used wrong `financials` schema):
  - `build_nl2sql_dataset.py`: 610 base examples (160 handcrafted + 450 GPT-4o distilled); WikiSQL removed (NL/SQL semantically disconnected)
  - `generate_nl2sql_templates.py`: +820 template-generated examples (12 generator types: single lookup, ranking, trend, comparison, YoY, aggregate, etc.)
  - `filter_nl2sql.py`: removed 39 noise examples → final **1,381 train / 240 eval**
  - Synonym coverage: `COLS` dict with 10+ synonyms per column; `sql_postprocess.py` for deterministic repair at inference
- Training run: 870 steps, 10 epochs, **64.8 min** on RTX 3090 Ti
  - Final train loss: **0.0012** · Best eval loss: **0.0223** @ step 500 (epoch 5.75)
  - Mild overfit from epoch 6 → promoted step-500 checkpoint as production adapter
  - Eval token accuracy: **99.3%** stable throughout

**Part 2 (Finetune) — NL2SQL evaluation (panel schema)**
- Rewrote `eval_compare.py` with correct panel-schema metrics (replaced old `financials`-schema checks)
- New metrics: valid_sql, correct_table (`FROM panel`), correct_column, period_format (`FY20XX`), keyword_match, exact_match, exec_ok
- Results (n=60):

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| Valid SQL | 98.3% | 100.0% | +1.7% |
| Correct Table | 98.3% | 100.0% | +1.7% |
| Correct Column | 93.3% | 96.7% | +3.3% |
| Period Format | 100.0% | 100.0% | +0.0% |
| Keyword Match | 96.7% | 96.7% | +0.0% |
| Exact Match | 3.3% | **78.3%** | **+75.0%** |
| **Exec OK ★** | 83.3% | **100.0%** | **+16.7%** |

**Part 3 (Deployment) — RAG retrieval pipeline**
- Built `deployment/rag/retriever.py` — production hybrid retrieval:
  - **Stage 1**: BM25 (sparse, rank-bm25) + ChromaDB dense (all-MiniLM-L6-v2) — both pre-filtered by auto-extracted metadata (ticker/filing_type/year)
  - **Stage 1b**: Reciprocal Rank Fusion (RRF k=60) — merges two ranked lists, up to 40 candidates
  - **Stage 2**: Cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) — full (query, passage) cross-attention
  - **Stage 3**: MMR dedup (λ=0.7, Jaccard trigram similarity) — prevents adjacent-chunk overlap in top-N
  - Metadata filter parsing: auto-extracts ticker, filing_type, year from NL question via regex + company-name dict

**Part 3 (Deployment) — Chatbot pipeline**
- Built `chatbot.py` — full interactive pipeline:
  - Intent → Type1: fine-tuned NL2SQL → `sql_postprocess.py` → SQLite execute → base Llama answer
  - Intent → Type2: `HybridRetriever` (BM25+dense+RRF+rerank+MMR) → base Llama answer from context
  - Intent → Type3: base Llama direct answer
  - All LLM calls marked `[VLLM_SWAP]` for future vLLM migration
- Fixed `apply_chat_template` tensor type bug (transformers version mismatch → two-step tokenisation)
- Fixed ChromaDB 1.5.x API changes: open collection without `embedding_function`, embed manually; removed `"ids"` from include list

**Smoke test & demo — all paths verified**
- `smoke_test.py`: all 3 intent types PASS
- `demo.py`: full pipeline trace for Type1/Type2/Type3
  - Type1: `"What was Apple's revenue in FY2023?"` → SQL → 383.29B → answer ✓ (4.7s)
  - Type2: `"How did Apple describe its AI strategy?"` → RAG retrieval → honest "not in excerpts" ✓ (9.7s)
  - Type3: `"Hello, what can you help me with?"` → direct answer ✓ (12.2s)
- `docs/report_2026-04-10.html`: self-contained HTML report with embedded training curve + eval bar chart

---

## Remaining

- [ ] Train intent classifier adapter (currently keyword + base Llama placeholder)
- [ ] Train query rewriter adapter
- [ ] vLLM deployment (requires Linux/WSL2; `[VLLM_SWAP]` markers already in chatbot.py)
- [ ] Improve Type2 RAG quality (AI strategy chunks score low — consider larger chunk windows)
