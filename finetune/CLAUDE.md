# Fine-tuning — Part 2 CLAUDE.md

## Purpose

Fine-tune three independent LoRA adapters on top of **Llama-3-8B-Instruct**
for the three specialized tasks in the RAG pipeline.

| Adapter | Task | Output |
|---|---|---|
| `intent_classifier` | Classify user query → Type1 / Type2 / Type3 | Single label token |
| `query_rewriter` | Rewrite multi-entity query → list of single-entity queries | JSON array |
| `nl2sql` | Natural language → SQL (for the financials SQLite schema) | SQL string |

Each adapter is trained independently and saved to `models/{adapter_name}/`.

**This part has NO dependency on Part 3 (deployment).**
It optionally benefits from Part 1 (uses the SQLite metric vocabulary for nl2sql prompts).

## Commands

```bash
pip install -r requirements.txt

# Step 1: generate training datasets
python data_prep/intent_dataset.py      # → data/intent_train.jsonl, data/intent_eval.jsonl
python data_prep/rewriter_dataset.py    # → data/rewriter_train.jsonl, data/rewriter_eval.jsonl
python data_prep/nl2sql_dataset.py      # → data/nl2sql_train.jsonl, data/nl2sql_eval.jsonl

# Step 2: train all adapters (sequentially)
bash scripts/train_all.sh
# Or train individually:
python adapters/intent_classifier/train.py --config adapters/intent_classifier/config.yaml
python adapters/query_rewriter/train.py   --config adapters/query_rewriter/config.yaml
python adapters/nl2sql/train.py           --config adapters/nl2sql/config.yaml

# Step 3: evaluate
python adapters/intent_classifier/evaluate.py  # target: >90% accuracy
python adapters/query_rewriter/evaluate.py     # target: valid JSON list output
python adapters/nl2sql/evaluate.py             # target: >80% execution accuracy

pytest tests/ -v
```

## Hardware Assumptions

- GPU: NVIDIA 4090 (24GB) or 3090 (24GB) — tested at 4-bit quantization
- Base model in 4-bit BnB NF4 quantization: ~5-6GB VRAM
- LoRA adds ~300MB per adapter at rank=16
- Effective batch: 4 per GPU × 4 grad accumulation steps = 16
- Training time: ~1-2 hours per adapter on a single 4090

## Base Model Setup (`common/base_model.py`)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_base_model(model_id: str = MODEL_ID):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer
```

All three training scripts import `load_base_model()` — never duplicate this.

## LoRA Configuration (`common/lora_utils.py`)

```python
from peft import LoraConfig, get_peft_model, TaskType

LORA_DEFAULT_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

def get_lora_config(**overrides) -> LoraConfig:
    return LoraConfig(**{**LORA_DEFAULT_CONFIG, **overrides})

def save_adapter(model, output_dir: str) -> None:
    model.save_pretrained(output_dir)

def load_adapter(base_model, adapter_dir: str):
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, adapter_dir)
```

## SFT Data Format (all adapters)

Every training file is JSONL. Each line is a `messages` list in the
**Llama-3 chat template** format:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Use `trl.SFTTrainer` with `dataset_text_field` set to the formatted chat string
(via `tokenizer.apply_chat_template`).

---

## Adapter 1: Intent Classifier

### Task Definition

```
Input:  a user query string
Output: exactly one of "Type1", "Type2", "Type3"
```

| Type | Description | Examples |
|---|---|---|
| Type1 | Exact financial fact that can be fetched from a structured DB | "What was Apple's revenue in Q3 2023?", "Show MSFT net income FY2022" |
| Type2 | Vague, qualitative, or analytical — needs document search | "How did Apple describe supply chain risks?", "What did mgmt say about AI in 10-K?" |
| Type3 | Casual chat, greeting, meta questions | "Hello", "What can you help me with?", "Thanks" |

### Prompt Template (`adapters/intent_classifier/prompt_template.py`)

```python
SYSTEM_PROMPT = (
    "You are a financial query classifier. "
    "Classify the user query into exactly one of: Type1, Type2, Type3.\n"
    "Type1 = exact financial fact (revenue, EPS, ratio, date-specific number)\n"
    "Type2 = vague, qualitative, or analytical question about filings\n"
    "Type3 = casual chat, greeting, or meta question\n"
    "Output only the label. Do not explain."
)

def format_example(query: str, label: str) -> dict:
    return {"messages": [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": query},
        {"role": "assistant","content": label},
    ]}
```

### Dataset (`data_prep/intent_dataset.py`)

Target: **3,000 train / 500 eval**, balanced across three classes.

Generation strategy:
1. Start from a curated seed set of ~300 examples (100 per class) — hardcode these
2. Augment Type1 with templates:
   - `"What was {company}'s {metric} in {period}?"`
   - `"Show me {ticker} {metric} for fiscal {year}"`
   - metrics = ["revenue", "net income", "EPS", "gross margin", "EBITDA", "total assets"]
   - companies/tickers = S&P 100 list
3. Augment Type2 with templates:
   - `"How did {company} describe {topic} in their {year} annual report?"`
   - `"What are {ticker}'s main risk factors regarding {topic}?"`
   - topics = ["supply chain", "AI investments", "competition", "regulatory risks", "cybersecurity"]
4. Augment Type3 from a fixed list of casual phrases (greetings, thanks, meta)

Output: `data/intent_train.jsonl`, `data/intent_eval.jsonl`

### Training Config (`adapters/intent_classifier/config.yaml`)

```yaml
output_dir: ../../models/intent_classifier
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.05
lr_scheduler_type: cosine
max_seq_length: 256       # queries are short
save_strategy: epoch
evaluation_strategy: epoch
load_best_model_at_end: true
metric_for_best_model: eval_loss
```

### Evaluation (`adapters/intent_classifier/evaluate.py`)

Load saved adapter → run on `data/intent_eval.jsonl` → compute accuracy.
Parse the first token of the generated output as the label.
**Target: >90% accuracy**. Print per-class precision/recall/F1.

---

## Adapter 2: Query Rewriter

### Task Definition

```
Input:  a user query (possibly multi-entity or vague)
Output: a JSON array of strings — precise, single-entity sub-queries
```

### Examples

```
Input:  "Compare Apple and Microsoft revenue growth and margin trends"
Output: ["What was Apple's annual revenue growth rate?",
         "What was Apple's gross margin trend?",
         "What was Microsoft's annual revenue growth rate?",
         "What was Microsoft's gross margin trend?"]

Input:  "How are tech companies handling AI investments?"
Output: ["How is Apple investing in artificial intelligence according to their filings?",
         "How is Microsoft investing in artificial intelligence according to their filings?",
         "How is Alphabet investing in artificial intelligence according to their filings?"]

Input:  "What was Apple's Q3 2023 revenue?"   # single-entity → return as-is
Output: ["What was Apple's Q3 2023 revenue?"]
```

### Prompt Template (`adapters/query_rewriter/prompt_template.py`)

```python
SYSTEM_PROMPT = (
    "You are a financial query rewriter. "
    "Decompose the user query into a JSON array of precise, single-entity sub-queries "
    "suitable for database or document search. "
    "If the query is already single-entity and specific, return it unchanged in an array. "
    "Output ONLY the JSON array. Do not explain."
)
```

### Dataset (`data_prep/rewriter_dataset.py`)

Target: **1,500 train / 300 eval**.

Generation strategy:
1. Multi-company comparison templates: `"Compare {A} and {B} {metric}"` → decompose
2. Sector-level questions: `"How are tech/bank/energy companies doing with {topic}?"` → per-company
3. Pass-through examples: single-entity queries → `[query]` (prevents over-decomposition)

Output must be parseable by `json.loads()` — enforce this in data generation and evaluation.

### Training Config (`adapters/query_rewriter/config.yaml`)

```yaml
output_dir: ../../models/query_rewriter
num_train_epochs: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
max_seq_length: 512
```

### Evaluation (`adapters/query_rewriter/evaluate.py`)

1. Assert `json.loads(output)` succeeds for every generated response
2. For multi-entity inputs: assert `len(output) >= 2`
3. For single-entity inputs: assert `len(output) == 1` and content is preserved
4. Compute ROUGE-L of generated sub-queries vs reference sub-queries

---

## Adapter 3: NL2SQL

### Task Definition

```
Input:  natural language financial question + SQLite schema context
Output: a valid SQL SELECT statement against the financials schema
```

### Schema Context (`data_prep/schema_context.py`)

This file MUST match `data_pipeline/storage/schema.sql` exactly.
If Part 1 has been run, load the actual metric vocabulary from SQLite.

```python
SCHEMA_CONTEXT = """
You have access to a SQLite database with these tables:

financials (ticker TEXT, period TEXT, statement TEXT, metric TEXT, value REAL, unit TEXT)
  - ticker:    stock ticker, e.g. "AAPL", "MSFT"
  - period:    "YYYY-MM" for quarterly, "FYYYY" for annual, e.g. "2023-09", "F2023"
  - statement: one of "income_statement", "balance_sheet", "cash_flow"
  - metric:    financial metric name, e.g. "Total Revenue", "Net Income", "Total Assets"
  - value:     numeric value in USD (millions unless unit says otherwise)

filing_metadata (ticker TEXT, company TEXT, filing_type TEXT, date TEXT)
  - filing_type: "10-K", "10-Q", "8-K"
  - date: "YYYY-MM-DD"

Rules:
- Output ONLY a valid SQL SELECT statement. No explanation.
- Use LIKE for partial metric name matches.
- Always filter by ticker if one is mentioned.
- Period: annual questions → use "F{year}" or LIKE "F%"; quarterly → LIKE "YYYY-%"
"""

def load_schema_context(db_path: str | None = None) -> str:
    """If db_path provided, extend SCHEMA_CONTEXT with the actual metric vocabulary
    from the database (SELECT DISTINCT metric FROM financials LIMIT 100)."""
```

### Example Pairs

```
NL:  "What was Apple's total revenue in fiscal 2023?"
SQL: SELECT value FROM financials WHERE ticker='AAPL' AND metric='Total Revenue' AND period='F2023'

NL:  "Show me Microsoft's net income for Q3 2023"
SQL: SELECT value FROM financials WHERE ticker='MSFT' AND metric='Net Income' AND period LIKE '2023-%' ORDER BY period

NL:  "Which companies had gross margin above 50% in 2022?"
SQL: SELECT ticker, value FROM financials WHERE metric LIKE '%Gross Margin%' AND period LIKE 'F2022' AND value > 50

NL:  "Compare Apple and Google revenue over the last 3 years"
SQL: SELECT ticker, period, value FROM financials WHERE ticker IN ('AAPL','GOOGL') AND metric='Total Revenue' AND period LIKE 'F20%' ORDER BY ticker, period
```

### Dataset (`data_prep/nl2sql_dataset.py`)

Target: **2,000 train / 400 eval**.

Sources:
1. Curated handwritten pairs using the financials schema (~200 pairs)
2. Template expansion: NL template × ticker × period × metric → SQL
3. Optional: filter Spider/WikiSQL for finance-adjacent tables

Each example uses `SCHEMA_CONTEXT` as the system prompt.

Output: `data/nl2sql_train.jsonl`, `data/nl2sql_eval.jsonl`

### Training Config (`adapters/nl2sql/config.yaml`)

```yaml
output_dir: ../../models/nl2sql
num_train_epochs: 5
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-4
max_seq_length: 768
```

### Evaluation (`adapters/nl2sql/evaluate.py`)

**Execution accuracy** — the only metric that matters:
1. Generate SQL from each eval question
2. Run predicted SQL against a test SQLite DB (pre-populated with fixture data)
3. Compare result DataFrame to expected result (exact match or numeric closeness)
4. Report execution accuracy = (correct / total)

**Target: >80% execution accuracy**

Also report: syntax error rate, empty result rate.

---

## `scripts/train_all.sh`

```bash
#!/bin/bash
set -e

echo "=== Generating datasets ==="
python data_prep/intent_dataset.py
python data_prep/rewriter_dataset.py
python data_prep/nl2sql_dataset.py

echo "=== Training intent_classifier ==="
python adapters/intent_classifier/train.py --config adapters/intent_classifier/config.yaml

echo "=== Training query_rewriter ==="
python adapters/query_rewriter/train.py --config adapters/query_rewriter/config.yaml

echo "=== Training nl2sql ==="
python adapters/nl2sql/train.py --config adapters/nl2sql/config.yaml

echo "=== Evaluating all adapters ==="
python adapters/intent_classifier/evaluate.py
python adapters/query_rewriter/evaluate.py
python adapters/nl2sql/evaluate.py

echo "=== Done. Adapters saved to models/ ==="
```

## Output Adapter Paths

```
models/
├── intent_classifier/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── query_rewriter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── nl2sql/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Dependencies (`requirements.txt`)

```
torch>=2.3
transformers>=4.41
peft>=0.11
trl>=0.9
bitsandbytes>=0.43
accelerate>=0.30
datasets>=2.19
loguru>=0.7
pyyaml>=6.0
pandas>=2.0
pytest>=8.0
rouge-score>=0.1
```

## Testing Requirements

Tests must pass **without GPU** — use tiny fixture datasets and mock model calls.

| Test file | What to verify |
|---|---|
| `test_datasets.py` | JSONL format valid; token length ≤ max_seq_length; label distribution balanced |
| `test_intent_eval.py` | Load fixture predictions, assert accuracy calculation logic |
| `test_nl2sql_eval.py` | SQL execution against fixture SQLite DB; assert exec accuracy math |
| `test_rewriter_eval.py` | Assert `json.loads()` succeeds; assert multi-entity count |

## Important Notes

1. **nl2sql metric vocabulary**: `data_prep/schema_context.py` should load actual
   metric names from the SQLite DB if `../data/financials.db` exists, rather than
   hardcoding. This prevents training on metric names that don't exist in the DB.

2. **Intent classifier label format**: The assistant output must be exactly one of
   `"Type1"`, `"Type2"`, `"Type3"` — no other text. Use a max_new_tokens=5 at
   inference time and take the first non-whitespace token.

3. **Query rewriter JSON enforcement**: At inference time, if `json.loads()` fails,
   fall back to wrapping the output in a list: `[output_text]`.

4. **vLLM compatibility**: Adapters must be saved in standard PEFT format
   (`model.save_pretrained()`). Do NOT use merged weights — vLLM/Punica requires
   the adapter weights separate from the base model.
