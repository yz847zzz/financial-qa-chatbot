#!/usr/bin/env python3
"""
NL2SQL evaluation: fine-tuned adapter vs base Llama-3.2-3B-Instruct.
Target schema: panel (wide format) + filing_metadata.

Metrics:
  valid_sql      — output is a syntactically valid SELECT … FROM …
  correct_table  — references `panel` or `filing_metadata` (not `financials`)
  correct_column — SELECT or WHERE references a known panel column name
  period_format  — year written as 'FY20XX' (panel convention)
  keyword_match  — ticker and year from question appear in the SQL
  exact_match    — normalised SQL equals reference
  exec_ok        — SQL executes against financials.db without error (--exec flag)

Usage:
    python finetune/adapters/nl2sql/eval_compare.py
    python finetune/adapters/nl2sql/eval_compare.py --n 60 --exec
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "llama"))
os.environ.setdefault("HF_HOME",      str(ROOT / "models" / "hf_home"))
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EVAL_FILE   = ROOT / "data" / "nl2sql" / "eval.jsonl"
ADAPTER_DIR = ROOT / "models" / "nl2sql"
DB_PATH     = ROOT / "data" / "financials.db"
MODEL_ID    = "meta-llama/Llama-3.2-3B-Instruct"
OUT_DIR     = ROOT / "models" / "nl2sql"
MAX_NEW_TOKENS = 150

# All column names in the panel table
PANEL_COLUMNS = {
    "ticker", "year", "cash", "total_assets", "current_assets",
    "current_liabilities", "total_liabilities", "goodwill",
    "long_term_debt", "accounts_payable", "inventories", "deferred_tax",
    "retained_earnings", "other_assets", "total_revenue", "net_income",
    "operating_income", "interest_expense", "interest_income", "da",
    "cfo", "capex", "current_ratio", "debt_to_assets", "roa", "net_margin",
}

VALID_TABLES = {"panel", "filing_metadata"}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _norm(sql: str) -> str:
    """Normalise SQL for exact-match comparison."""
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.rstrip(";").lower()


def score(pred: str, ref: str, question: str, run_exec: bool = False) -> dict[str, int]:
    p = pred.strip()

    # valid_sql: starts with SELECT and has FROM
    valid_sql = bool(
        re.match(r"^\s*SELECT\b", p, re.I) and re.search(r"\bFROM\b", p, re.I)
    )

    # correct_table: references panel or filing_metadata; must NOT use financials
    tables_found = set(re.findall(r"\bFROM\s+(\w+)", p, re.I))
    tables_found |= set(re.findall(r"\bJOIN\s+(\w+)", p, re.I))
    correct_table = bool(
        tables_found & VALID_TABLES
        and not re.search(r"\bFROM\s+financials\b", p, re.I)
    )

    # correct_column: at least one known panel column (excluding ticker/year)
    # strip string literals first to avoid false matches inside quoted values
    p_no_strings = re.sub(r"'[^']*'", " ", p)
    tokens = set(re.findall(r"\b\w+\b", p_no_strings.lower()))
    correct_column = bool(tokens & (PANEL_COLUMNS - {"ticker", "year"}))

    # period_format: if question mentions a year, SQL must use 'FY20XX' format
    q_years = re.findall(r"\b20\d{2}\b", question)
    if q_years:
        period_ok = any(f"'FY{y}'" in p or f"FY{y}" in p.upper() for y in q_years)
    else:
        period_ok = True  # no year in question → not applicable

    # keyword_match: tickers and years from the question appear in SQL
    tickers  = re.findall(r"\b[A-Z]{2,5}\b", question)
    ticker_ok = all(t in p.upper() for t in tickers) if tickers else True
    year_ok   = all(y in p for y in q_years) if q_years else True
    keyword_ok = ticker_ok and year_ok

    # exact_match
    exact = int(_norm(p) == _norm(ref))

    # exec_ok: execute against the real DB (only with --exec flag)
    exec_ok = 0
    if run_exec and valid_sql and DB_PATH.exists():
        try:
            con = sqlite3.connect(str(DB_PATH))
            con.execute(p)
            con.close()
            exec_ok = 1
        except Exception:
            exec_ok = 0

    return {
        "valid_sql":      int(valid_sql),
        "correct_table":  int(correct_table),
        "correct_column": int(correct_column),
        "period_format":  int(period_ok),
        "keyword_match":  int(keyword_ok),
        "exact_match":    exact,
        "exec_ok":        exec_ok,
    }


# ── Model helpers ──────────────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(model_id: str):
    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=_bnb_config(),
        device_map="auto", dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def load_finetuned(adapter_dir: Path, model_id: str):
    logger.info(f"Loading adapter: {adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=_bnb_config(),
        device_map="auto", dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def generate(model, tokenizer, messages: list[dict], max_new: int = MAX_NEW_TOKENS) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Report / plots ─────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "valid_sql":      "Valid SQL",
    "correct_table":  "Correct Table",
    "correct_column": "Correct Column",
    "period_format":  "Period Format (FY)",
    "keyword_match":  "Keyword Match",
    "exact_match":    "Exact Match",
    "exec_ok":        "Exec OK",
}


def plot_comparison(base_scores: list[dict], ft_scores: list[dict],
                    out_dir: Path, run_exec: bool) -> None:
    keys   = [k for k in METRIC_LABELS if k != "exec_ok" or run_exec]
    labels = [METRIC_LABELS[k] for k in keys]
    n      = len(base_scores)

    base_means = [sum(s[k] for s in base_scores) / n * 100 for k in keys]
    ft_means   = [sum(s[k] for s in ft_scores)   / n * 100 for k in keys]

    x, width = np.arange(len(keys)), 0.35
    fig, ax  = plt.subplots(figsize=(12, 5))
    bars_b = ax.bar(x - width/2, base_means, width, label="Base Llama-3.2-3B",
                    color="steelblue", alpha=0.85)
    bars_f = ax.bar(x + width/2, ft_means,   width, label="Fine-tuned NL2SQL (QLoRA r=16)",
                    color="tomato",    alpha=0.85)
    for bar in list(bars_b) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "NL2SQL Eval: panel schema — Fine-tuned vs Base Llama-3.2-3B-Instruct\n"
        f"(n={n} · QLoRA r=16 α=32 · best checkpoint epoch 5.75)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "eval_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot → {path}")


def print_report(base_scores: list[dict], ft_scores: list[dict],
                 run_exec: bool) -> None:
    keys = [k for k in METRIC_LABELS if k != "exec_ok" or run_exec]
    n    = len(base_scores)
    print(f"\n{'='*65}")
    print(f"NL2SQL Eval  (panel schema)  —  n={n} examples")
    print(f"{'='*65}")
    print(f"{'Metric':<24} {'Base':>8} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 56)
    for k in keys:
        bm = sum(s[k] for s in base_scores) / n * 100
        fm = sum(s[k] for s in ft_scores)   / n * 100
        d  = fm - bm
        sign = "+" if d >= 0 else ""
        print(f"{METRIC_LABELS[k]:<24} {bm:>7.1f}%  {fm:>10.1f}%  {sign}{d:>6.1f}%")
    print("=" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(adapter_dir: Path, n_samples: int, run_exec: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    with open(EVAL_FILE, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    if n_samples and n_samples < len(examples):
        import random; random.seed(42)
        examples = random.sample(examples, n_samples)
    logger.info(f"Evaluating on {len(examples)} examples")

    def run_model(model, tok, label):
        preds, scores_list = [], []
        for i, ex in enumerate(examples, 1):
            pred = generate(model, tok, ex["messages"][:-1])
            ref  = ex["messages"][-1]["content"]
            q    = ex["messages"][1]["content"]
            preds.append(pred)
            scores_list.append(score(pred, ref, q, run_exec=run_exec))
            if i % 20 == 0:
                logger.info(f"  {label}: {i}/{len(examples)}")
        return preds, scores_list

    # Base model
    base_model, base_tok = load_base(MODEL_ID)
    base_preds, base_scores = run_model(base_model, base_tok, "Base")
    del base_model; torch.cuda.empty_cache()

    # Fine-tuned model (best checkpoint at epoch 5.75)
    ft_model, ft_tok = load_finetuned(adapter_dir, MODEL_ID)
    ft_preds, ft_scores = run_model(ft_model, ft_tok, "FT")
    del ft_model; torch.cuda.empty_cache()

    # Report
    print_report(base_scores, ft_scores, run_exec)

    # Save detailed results
    results = []
    for ex, bp, fp, bs, fs in zip(examples, base_preds, ft_preds, base_scores, ft_scores):
        results.append({
            "question":    ex["messages"][1]["content"],
            "reference":   ex["messages"][-1]["content"],
            "base_pred":   bp,
            "ft_pred":     fp,
            "base_scores": bs,
            "ft_scores":   fs,
        })
    out_path = OUT_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Detailed results → {out_path}")

    plot_comparison(base_scores, ft_scores, OUT_DIR, run_exec)

    # Sample predictions
    import random; random.seed(0)
    print(f"\n── Sample predictions (5 examples) {'─'*28}\n")
    for r in random.sample(results, min(5, len(results))):
        ft_ok = "OK" if r["ft_scores"]["correct_table"] and r["ft_scores"]["correct_column"] else "FAIL"
        print(f"Q:   {r['question'][:80]}")
        print(f"REF: {r['reference'][:100]}")
        print(f"BASE:{r['base_pred'][:100]}")
        print(f"FT:  {r['ft_pred'][:100]}")
        print(f"     [FT] table+col={ft_ok}  period={r['ft_scores']['period_format']}  exact={r['ft_scores']['exact_match']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(ADAPTER_DIR))
    parser.add_argument("--n",    type=int, default=0,
                        help="Number of eval samples (0 = all 240)")
    parser.add_argument("--exec", action="store_true",
                        help="Also execute SQL against financials.db")
    args = parser.parse_args()
    main(Path(args.adapter), args.n, args.exec)
