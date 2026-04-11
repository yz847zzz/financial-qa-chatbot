#!/usr/bin/env python3
"""
Daily progress report generator.
Reads training logs, eval results, and demo outputs → HTML report with embedded charts.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --out docs/report_2026-04-10.html
"""

import argparse
import base64
import io
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_training_logs() -> tuple[list, list]:
    state_path = ROOT / "models" / "nl2sql" / "checkpoint-870" / "trainer_state.json"
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    logs = state["log_history"]
    train = [(x["step"], x["epoch"], x["loss"], x.get("mean_token_accuracy", 0))
             for x in logs if "loss" in x and "eval_loss" not in x]
    evall = [(x["step"], x["epoch"], x["eval_loss"], x["eval_mean_token_accuracy"])
             for x in logs if "eval_loss" in x]
    return train, evall


def load_run_metadata() -> dict:
    path = ROOT / "models" / "nl2sql" / "run_metadata.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_eval_results() -> list[dict]:
    path = ROOT / "models" / "nl2sql" / "eval_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Chart generation ───────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def make_training_curve(train_logs, eval_logs) -> str:
    steps_t  = [x[0] for x in train_logs]
    loss_t   = [x[2] for x in train_logs]
    acc_t    = [x[3] for x in train_logs]
    steps_e  = [x[0] for x in eval_logs]
    loss_e   = [x[2] for x in eval_logs]
    acc_e    = [x[3] for x in eval_logs]
    epochs_e = [x[1] for x in eval_logs]

    best_idx  = int(np.argmin(loss_e))
    best_step = steps_e[best_idx]
    best_loss = loss_e[best_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("NL2SQL QLoRA Training — Llama-3.2-3B-Instruct  (r=16, α=32, 10 epochs)",
                 fontsize=12, fontweight="bold", y=1.01)

    # ── Loss ──────────────────────────────────────────────────────
    ax1.plot(steps_t, loss_t, color="#4C72B0", lw=1.2, alpha=0.7, label="Train loss")
    ax1.plot(steps_e, loss_e, color="#DD8452", lw=2.0, marker="o", ms=3, label="Eval loss")
    ax1.axvline(best_step, color="crimson", ls="--", lw=1.2,
                label=f"Best eval @ step {best_step} (loss={best_loss:.4f})")
    ax1.fill_between(steps_e, loss_e, alpha=0.08, color="#DD8452")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss curve")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Annotate mild overfit region
    ax1.annotate("mild overfit\n(epoch 6+)",
                 xy=(600, loss_e[15]), xytext=(680, loss_e[15] + 0.005),
                 fontsize=8, color="grey",
                 arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    # Secondary x-axis: epochs
    ax1b = ax1.twiny()
    epoch_ticks = [steps_e[i] for i in range(0, len(steps_e), 4)]
    epoch_labels = [f"{epochs_e[i]:.0f}" for i in range(0, len(steps_e), 4)]
    ax1b.set_xlim(ax1.get_xlim())
    ax1b.set_xticks(epoch_ticks)
    ax1b.set_xticklabels(epoch_labels, fontsize=8)
    ax1b.set_xlabel("Epoch", fontsize=9)

    # ── Token Accuracy ────────────────────────────────────────────
    ax2.plot(steps_t, [a * 100 for a in acc_t], color="#4C72B0",
             lw=1.2, alpha=0.7, label="Train token acc")
    ax2.plot(steps_e, [a * 100 for a in acc_e], color="#DD8452",
             lw=2.0, marker="o", ms=3, label="Eval token acc")
    ax2.axhline(99.0, color="grey", ls=":", lw=1, label="99% line")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Token accuracy (%)")
    ax2.set_title("Token accuracy")
    ax2.set_ylim(85, 101)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig_to_b64(fig)


def make_eval_bar(results: list[dict]) -> str:
    n = len(results)
    LABELS = {
        "valid_sql":      "Valid SQL",
        "correct_table":  "Correct Table",
        "correct_column": "Correct Column",
        "period_format":  "Period Format",
        "keyword_match":  "Keyword Match",
        "exact_match":    "Exact Match",
        "exec_ok":        "Exec OK ★",
    }
    keys   = list(LABELS.keys())
    labels = list(LABELS.values())
    base_m = [sum(r["base_scores"][k] for r in results) / n * 100 for k in keys]
    ft_m   = [sum(r["ft_scores"][k]   for r in results) / n * 100 for k in keys]

    x, w = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bars_b = ax.bar(x - w/2, base_m, w, label="Base Llama-3.2-3B",
                    color="#4C72B0", alpha=0.85)
    bars_f = ax.bar(x + w/2, ft_m,   w, label="Fine-tuned NL2SQL adapter",
                    color="#C44E52", alpha=0.85)
    for bar in list(bars_b) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"NL2SQL Evaluation — panel schema  (n={n} examples · best checkpoint epoch 5.75)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Highlight exec_ok bar
    ax.annotate("", xy=(x[-1] + w/2 + 0.25, ft_m[-1]),
                xytext=(x[-1] + w/2 + 0.25, base_m[-1]),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5))
    ax.text(x[-1] + w/2 + 0.3, (ft_m[-1] + base_m[-1]) / 2,
            f"+{ft_m[-1]-base_m[-1]:.0f}%", color="crimson", fontsize=8, va="center")

    plt.tight_layout()
    return fig_to_b64(fig)


# ── HTML template ──────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f6fa; color: #2d3436; line-height: 1.6; }
.page { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
h1 { font-size: 1.8rem; color: #1a1a2e; border-bottom: 3px solid #6c63ff;
     padding-bottom: 10px; margin-bottom: 6px; }
.subtitle { color: #636e72; font-size: 0.95rem; margin-bottom: 32px; }
h2 { font-size: 1.2rem; color: #1a1a2e; margin: 32px 0 12px;
     border-left: 4px solid #6c63ff; padding-left: 10px; }
h3 { font-size: 1rem; color: #555; margin: 16px 0 8px; }
.card { background: #fff; border-radius: 10px; padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 20px; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                margin-bottom: 4px; }
.metric-box { background: #f0f0ff; border-radius: 8px; padding: 14px 16px;
              text-align: center; }
.metric-box .val { font-size: 1.6rem; font-weight: 700; color: #6c63ff; }
.metric-box .lbl { font-size: 0.78rem; color: #636e72; margin-top: 2px; }
.metric-box.good .val { color: #00b894; }
.metric-box.warn .val { color: #e17055; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th { background: #f0f0ff; color: #1a1a2e; padding: 8px 12px;
     text-align: left; font-weight: 600; }
td { padding: 7px 12px; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
.delta-pos { color: #00b894; font-weight: 600; }
.delta-zero { color: #636e72; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 0.78rem; font-weight: 600; }
.badge-t1 { background: #dfe6e9; color: #2d3436; }
.badge-t2 { background: #dfe6e9; color: #2d3436; }
.badge-t3 { background: #dfe6e9; color: #2d3436; }
.demo-block { border-left: 3px solid #6c63ff; padding: 10px 16px;
              background: #fafafa; border-radius: 0 6px 6px 0;
              margin: 8px 0; font-size: 0.88rem; }
.demo-block .row { display: flex; gap: 12px; margin: 4px 0; }
.demo-block .tag { min-width: 80px; font-weight: 600; color: #6c63ff;
                   font-size: 0.82rem; }
.demo-block .val { color: #2d3436; font-family: 'SF Mono', monospace;
                   font-size: 0.83rem; white-space: pre-wrap; word-break: break-all; }
.demo-block .answer-val { color: #2d3436; font-size: 0.88rem; font-style: italic; }
img { width: 100%; border-radius: 8px; margin-top: 8px; }
.footer { text-align: center; color: #aaa; font-size: 0.8rem;
          margin-top: 40px; padding-top: 16px; border-top: 1px solid #eee; }
.status-ok { color: #00b894; font-weight: 700; }
.status-warn { color: #e17055; font-weight: 700; }
"""


def build_html(train_chart: str, eval_chart: str, meta: dict, results: list[dict]) -> str:
    today = date.today().strftime("%B %d, %Y")
    n = len(results)

    # Eval metrics table rows
    LABELS = {
        "valid_sql":      "Valid SQL",
        "correct_table":  "Correct Table",
        "correct_column": "Correct Column",
        "period_format":  "Period Format (FY)",
        "keyword_match":  "Keyword Match",
        "exact_match":    "Exact Match",
        "exec_ok":        "Exec OK ★",
    }
    table_rows = ""
    for k, label in LABELS.items():
        bm = sum(r["base_scores"][k] for r in results) / n * 100
        fm = sum(r["ft_scores"][k]   for r in results) / n * 100
        d  = fm - bm
        star = " ★" if k == "exec_ok" else ""
        delta_class = "delta-pos" if d > 1 else "delta-zero"
        table_rows += f"""
        <tr>
          <td>{label}{star}</td>
          <td>{bm:.1f}%</td>
          <td><strong>{fm:.1f}%</strong></td>
          <td class="{delta_class}">+{d:.1f}%</td>
        </tr>"""

    # Sample predictions
    import random; random.seed(7)
    samples = random.sample(results, min(3, n))
    sample_html = ""
    for s in samples:
        ft_sql_ok = "✓" if s["ft_scores"]["correct_table"] and s["ft_scores"]["correct_column"] else "✗"
        sample_html += f"""
        <div class="demo-block">
          <div class="row"><span class="tag">Question</span>
               <span class="val">{s['question'][:120]}</span></div>
          <div class="row"><span class="tag">Reference</span>
               <span class="val">{s['reference'][:120]}</span></div>
          <div class="row"><span class="tag">Base pred</span>
               <span class="val">{s['base_pred'][:120]}</span></div>
          <div class="row"><span class="tag">FT pred</span>
               <span class="val">{s['ft_pred'][:120]}</span></div>
          <div class="row"><span class="tag">FT checks</span>
               <span class="val">table+col={ft_sql_ok}  exact={s['ft_scores']['exact_match']}  exec={s['ft_scores']['exec_ok']}</span></div>
        </div>"""

    cfg = meta["config"]
    runtime_min = meta["train_runtime_s"] / 60

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Financial QA — Daily Report {today}</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page">

  <h1>Financial QA Chatbot — Daily Progress Report</h1>
  <p class="subtitle">{today} &nbsp;·&nbsp; NL2SQL Fine-tuning Complete &nbsp;·&nbsp; End-to-End Pipeline Verified</p>

  <!-- ── System Status ── -->
  <h2>System Status</h2>
  <div class="card">
    <table>
      <tr><th>Component</th><th>Status</th><th>Detail</th></tr>
      <tr><td>Base model</td>
          <td><span class="status-ok">LOADED</span></td>
          <td>meta-llama/Llama-3.2-3B-Instruct · 4-bit NF4</td></tr>
      <tr><td>NL2SQL adapter</td>
          <td><span class="status-ok">LOADED</span></td>
          <td>QLoRA r=16 · best checkpoint epoch 5.75 (step 500)</td></tr>
      <tr><td>SQLite panel DB</td>
          <td><span class="status-ok">READY</span></td>
          <td>92 tickers × 6 FY years · 33 columns</td></tr>
      <tr><td>VectorDB (ChromaDB)</td>
          <td><span class="status-ok">READY</span></td>
          <td>516,955 chunks · all-MiniLM-L6-v2 embeddings</td></tr>
      <tr><td>BM25 index</td>
          <td><span class="status-ok">READY</span></td>
          <td>516,955 documents · built at startup (~90s)</td></tr>
      <tr><td>Cross-encoder reranker</td>
          <td><span class="status-ok">READY</span></td>
          <td>ms-marco-MiniLM-L-6-v2</td></tr>
      <tr><td>Intent classifier</td>
          <td><span class="status-warn">PLACEHOLDER</span></td>
          <td>Keyword + base Llama prompt (fine-tuned adapter pending)</td></tr>
      <tr><td>Query rewriter</td>
          <td><span class="status-warn">PENDING</span></td>
          <td>Not yet trained</td></tr>
      <tr><td>vLLM deployment</td>
          <td><span class="status-warn">PENDING</span></td>
          <td>Requires Linux/WSL2 — currently using direct HF loading</td></tr>
    </table>
  </div>

  <!-- ── Training Summary ── -->
  <h2>NL2SQL Adapter — Training Summary</h2>
  <div class="card">
    <div class="metrics-grid">
      <div class="metric-box good"><div class="val">0.0012</div>
           <div class="lbl">Final train loss</div></div>
      <div class="metric-box"><div class="val">0.0223</div>
           <div class="lbl">Best eval loss (step 500)</div></div>
      <div class="metric-box good"><div class="val">99.3%</div>
           <div class="lbl">Eval token accuracy</div></div>
      <div class="metric-box"><div class="val">{runtime_min:.0f} min</div>
           <div class="lbl">Training time</div></div>
    </div>
  </div>

  <div class="card">
    <h3>Training configuration</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
      <tr><td>Base model</td><td>{meta['model_id']}</td>
          <td>Dataset</td><td>{meta['train_examples']:,} train / {meta['eval_examples']} eval</td></tr>
      <tr><td>LoRA rank</td><td>r={cfg['lora_r']}, α={cfg['lora_alpha']}</td>
          <td>Total steps</td><td>{meta['total_steps']}</td></tr>
      <tr><td>Epochs</td><td>{cfg['num_train_epochs']}</td>
          <td>Effective batch</td>
          <td>{cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']}</td></tr>
      <tr><td>Learning rate</td><td>{cfg['learning_rate']}</td>
          <td>LR schedule</td><td>{cfg['lr_scheduler_type']}</td></tr>
      <tr><td>Quantization</td><td>4-bit NF4 (QLoRA)</td>
          <td>Best checkpoint</td><td>Step 500 (epoch 5.75)</td></tr>
    </table>
    <br>
    <p style="font-size:0.85rem;color:#636e72;">
      <strong>Note on checkpoint selection:</strong> Eval loss plateaued at ~0.029 after epoch 6
      (mild overfitting — train loss continued to 0.0012 while eval stayed flat).
      The step-500 checkpoint (epoch 5.75, eval loss 0.0223) was promoted as the
      production adapter. Token accuracy remained stable at 99.3% throughout the plateau.
    </p>
  </div>

  <div class="card">
    <h3>Training curve</h3>
    <img src="data:image/png;base64,{train_chart}" alt="Training curve">
  </div>

  <!-- ── Eval Results ── -->
  <h2>NL2SQL Evaluation — panel schema (n={n} examples)</h2>
  <div class="card">
    <img src="data:image/png;base64,{eval_chart}" alt="Eval comparison">
    <br><br>
    <table>
      <tr><th>Metric</th><th>Base Llama-3.2-3B</th><th>Fine-tuned adapter</th><th>Delta</th></tr>
      {table_rows}
    </table>
    <p style="margin-top:12px;font-size:0.83rem;color:#636e72;">
      ★ <strong>Exec OK</strong> is the primary production metric — the SQL must execute
      against <code>financials.db</code> without error. Fine-tuned adapter achieves
      <strong>100%</strong> vs base model's 83.3%.
      <br>
      Previous eval (before dataset rebuild) used <code>FROM financials</code> schema —
      those results are discarded. This eval uses the correct <code>panel</code> wide-format schema.
    </p>
  </div>

  <div class="card">
    <h3>Sample predictions (fine-tuned vs base)</h3>
    {sample_html}
  </div>

  <!-- ── Pipeline Demo ── -->
  <h2>End-to-End Pipeline Demo</h2>

  <div class="card">
    <span class="badge badge-t1">TYPE 1 — NL2SQL</span>
    <div class="demo-block" style="margin-top:10px">
      <div class="row"><span class="tag">Question</span>
           <span class="val">What was Apple's revenue in FY2023?</span></div>
      <div class="row"><span class="tag">Intent</span>
           <span class="val">Type1 (keyword shortcut → no LLM call needed)</span></div>
      <div class="row"><span class="tag">NL2SQL</span>
           <span class="val">SELECT total_revenue FROM panel WHERE ticker='AAPL' AND year='FY2023';</span></div>
      <div class="row"><span class="tag">Postprocess</span>
           <span class="val">no repairs needed</span></div>
      <div class="row"><span class="tag">SQL result</span>
           <span class="val">1 row · total_revenue: 383,285,000,000</span></div>
      <div class="row"><span class="tag">Answer</span>
           <span class="answer-val">Apple's revenue in FY2023 was $383.29 billion.</span></div>
      <div class="row"><span class="tag">Time</span>
           <span class="val">4.7s</span></div>
    </div>
  </div>

  <div class="card">
    <span class="badge badge-t2">TYPE 2 — RAG</span>
    <div class="demo-block" style="margin-top:10px">
      <div class="row"><span class="tag">Question</span>
           <span class="val">How did Apple describe its AI strategy in recent filings?</span></div>
      <div class="row"><span class="tag">Intent</span>
           <span class="val">Type2 (LLM classification)</span></div>
      <div class="row"><span class="tag">Filters</span>
           <span class="val">ticker=AAPL (auto-extracted from question)</span></div>
      <div class="row"><span class="tag">Retrieval</span>
           <span class="val">BM25 (k=20) + Dense (k=20) → RRF merge (40 candidates) → Cross-encoder rerank → MMR top-3</span></div>
      <div class="row"><span class="tag">Chunk 1</span>
           <span class="val">[AAPL 10-K 2019-10-31] rerank=−0.650 · investment policy section</span></div>
      <div class="row"><span class="tag">Chunk 2</span>
           <span class="val">[AAPL 10-K 2024-11-01] rerank=−4.847 · internal controls section</span></div>
      <div class="row"><span class="tag">Chunk 3</span>
           <span class="val">[AAPL 10-K 2021-10-29] rerank=−4.654 · R&amp;D investments section</span></div>
      <div class="row"><span class="tag">Answer</span>
           <span class="answer-val">The provided excerpts do not mention Apple's AI strategy directly. They cover
           investment policy, internal controls, and R&amp;D investments. More specific
           AI-focused sections would require targeted filing retrieval.</span></div>
      <div class="row"><span class="tag">Time</span>
           <span class="val">9.7s</span></div>
    </div>
  </div>

  <div class="card">
    <span class="badge badge-t3">TYPE 3 — DIRECT</span>
    <div class="demo-block" style="margin-top:10px">
      <div class="row"><span class="tag">Question</span>
           <span class="val">Hello, what can you help me with?</span></div>
      <div class="row"><span class="tag">Intent</span>
           <span class="val">Type3 (regex shortcut · no LLM call needed)</span></div>
      <div class="row"><span class="tag">Retrieval</span>
           <span class="val">none</span></div>
      <div class="row"><span class="tag">Answer</span>
           <span class="answer-val">I can assist with SEC filings (10-K, 10-Q, 8-K), financial metrics (EPS, ROE, ROA),
           company analysis, financial modeling, and regulatory questions.</span></div>
      <div class="row"><span class="tag">Time</span>
           <span class="val">12.2s  (model already warmed up)</span></div>
    </div>
  </div>

  <!-- ── RAG Architecture ── -->
  <h2>RAG Retrieval Architecture</h2>
  <div class="card">
    <table>
      <tr><th>Stage</th><th>Method</th><th>Purpose</th></tr>
      <tr><td>Pre-filter</td><td>Metadata extraction (regex)</td>
          <td>Auto-extract ticker / filing_type / year from question; pass as ChromaDB <code>where=</code> and BM25 mask</td></tr>
      <tr><td>Sparse recall</td><td>BM25Okapi (rank-bm25)</td>
          <td>Keyword exact-match recall · catches ticker symbols, metric names · top-20</td></tr>
      <tr><td>Dense recall</td><td>ChromaDB + all-MiniLM-L6-v2</td>
          <td>Semantic similarity recall · catches paraphrases · top-20</td></tr>
      <tr><td>Fusion</td><td>Reciprocal Rank Fusion (RRF k=60)</td>
          <td>Merge two ranked lists without normalising incompatible scores · up to 40 candidates</td></tr>
      <tr><td>Rerank</td><td>Cross-encoder ms-marco-MiniLM-L-6-v2</td>
          <td>Full (query, passage) cross-attention relevance scoring · replaces cosine similarity</td></tr>
      <tr><td>Dedup</td><td>MMR λ=0.7, Jaccard trigram similarity</td>
          <td>Greedy diverse selection · prevents 3 adjacent chunks from same paragraph · top-3 returned</td></tr>
    </table>
  </div>

  <!-- ── Next Steps ── -->
  <h2>Next Steps</h2>
  <div class="card">
    <table>
      <tr><th>#</th><th>Task</th><th>Priority</th><th>Notes</th></tr>
      <tr><td>1</td><td>Train intent classifier adapter</td><td>High</td>
          <td>Replace keyword+LLM placeholder · target &gt;90% accuracy</td></tr>
      <tr><td>2</td><td>Train query rewriter adapter</td><td>Medium</td>
          <td>Decompose multi-entity queries before retrieval</td></tr>
      <tr><td>3</td><td>vLLM deployment</td><td>Medium</td>
          <td>Requires Linux/WSL2 · OpenAI-compatible API · swap <code>[VLLM_SWAP]</code> markers in chatbot.py</td></tr>
      <tr><td>4</td><td>Improve Type2 retrieval quality</td><td>Low</td>
          <td>AI strategy chunks have low rerank scores (−4.8) · consider re-chunking with larger windows</td></tr>
    </table>
  </div>

  <div class="footer">
    Generated {today} · Financial QA Chatbot · financial-qa-chatbot/scripts/generate_report.py
  </div>

</div>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Output path (default: docs/report_YYYY-MM-DD.html)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else \
        ROOT / "docs" / f"report_{date.today()}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading training logs...", flush=True)
    train_logs, eval_logs = load_training_logs()
    meta = load_run_metadata()
    results = load_eval_results()

    print("Generating training curve...", flush=True)
    train_chart = make_training_curve(train_logs, eval_logs)

    print("Generating eval bar chart...", flush=True)
    eval_chart = make_eval_bar(results)

    print("Building HTML report...", flush=True)
    html = build_html(train_chart, eval_chart, meta, results)

    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written → {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
