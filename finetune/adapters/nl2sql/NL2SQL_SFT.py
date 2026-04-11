#!/usr/bin/env python3
"""
NL2SQL LoRA fine-tuning — QLoRA + completion-only SFT.

Loss   : cross-entropy on SQL completion tokens only (prompt masked to -100)
Optim  : paged_adamw_8bit   (AdamW with decoupled weight decay, 8-bit states)
Quant  : NF4 4-bit base + bf16 LoRA adapters
Target : all attention + MLP projections (q/k/v/o/gate/up/down)

Usage:
    python NL2SQL_SFT.py
    python NL2SQL_SFT.py --epochs 10 --lr 2e-4 --rank 16

Outputs (all under models/nl2sql/):
    adapter_config.json + adapter_model.safetensors  ← PEFT adapter
    tokenizer files
    loss_curves.png
    run_metadata.json
"""

# ── stdlib first — must set env vars before any HF import ─────────────────────
import argparse
import json
import os
import sys
from pathlib import Path

# ROOT = financial-qa-chatbot/  (3 levels up from finetune/adapters/nl2sql/)
ROOT = Path(__file__).resolve().parents[3]

# Instructs safetensors to load weights directly to GPU, bypassing the CPU
# memory-map that exhausts the Windows paging file on large models.
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

# Point HF hub cache to our local E-drive model store BEFORE any HF import.
# The model lives at:
#   models/llama/models--meta-llama--Llama-3.2-3B-Instruct/
# so the hub cache root is models/llama/ → set HF_HUB_CACHE (not HF_HOME,
# which would resolve to models/llama/hub/ and miss the stored weights).
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "llama"))

# Keep all HF metadata (tokens, configs) on E-drive too.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_home"))

# ── HF / ML imports (AFTER env vars are set) ──────────────────────────────────
from dataclasses import asdict, dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer  # TRL 1.0: SFTConfig replaces TrainingArguments+collator

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "models" / "nl2sql"
TRAIN_FILE = ROOT / "data" / "nl2sql" / "train.jsonl"
EVAL_FILE  = ROOT / "data" / "nl2sql" / "eval.jsonl"


# ── Hyperparameters ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Model ID — resolved from HF_HUB_CACHE (models/llama/) set above
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"

    # LoRA — all attention + MLP so the model learns SQL structure, not just style
    lora_r: int = 16
    lora_alpha: int = 32          # alpha/r = 2  (standard scaling)
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ])

    # Training
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8           # effective batch = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 35                         # ~10% of 350 total steps
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"

    # Eval + checkpointing
    logging_steps: int = 5
    eval_steps: int = 25                           # ~3 eval points per epoch
    save_steps: int = 50
    save_total_limit: int = 2
    max_length: int = 512

    # Output
    output_dir: str = str(OUTPUT_DIR)
    seed: int = 42
    bf16: bool = True


# ── Loss recorder ──────────────────────────────────────────────────────────────
class LossRecorder(TrainerCallback):
    """Records (step, loss) pairs from trainer logs for plotting."""

    def __init__(self) -> None:
        self.train_log: list[tuple[int, float]] = []
        self.eval_log:  list[tuple[int, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_log.append((step, float(logs["loss"])))
        if "eval_loss" in logs:
            self.eval_log.append((step, float(logs["eval_loss"])))


def _ema(values: list[float], alpha: float = 0.6) -> list[float]:
    """Exponential moving average for smoothing noisy training loss."""
    smoothed, v = [], values[0]
    for x in values:
        v = alpha * x + (1 - alpha) * v
        smoothed.append(v)
    return smoothed


def save_loss_plot(
    recorder: LossRecorder,
    out_dir: Path,
    cfg: TrainConfig,
    total_steps: int,
) -> None:
    """
    Paper-ready loss curve figure:
      - Raw training loss (faint) + EMA-smoothed (solid)
      - Eval loss with markers
      - Vertical epoch boundaries
      - 300 DPI, tight layout
    """
    steps_per_epoch = total_steps / cfg.num_train_epochs if cfg.num_train_epochs > 0 else 0
    epoch_boundaries = [round(steps_per_epoch * i) for i in range(1, cfg.num_train_epochs)]

    # ── Derive y-axis limits from data ────────────────────────────────────────
    all_vals: list[float] = []
    if recorder.train_log:
        all_vals += [v for _, v in recorder.train_log]
    if recorder.eval_log:
        all_vals += [v for _, v in recorder.eval_log]
    y_max = max(all_vals) * 1.08 if all_vals else 1.0
    y_min = max(0.0, min(all_vals) * 0.92) if all_vals else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))

    # ── Training loss: raw (faint) + EMA (bold) ───────────────────────────────
    if recorder.train_log:
        t_steps, t_vals = zip(*recorder.train_log)
        t_steps, t_vals = list(t_steps), list(t_vals)
        ax.plot(t_steps, t_vals,
                color="steelblue", alpha=0.20, linewidth=0.8)
        ax.plot(t_steps, _ema(t_vals, alpha=0.6),
                color="steelblue", linewidth=2.0, label="Train loss (EMA-smoothed)")

    # ── Eval loss: markers + line ─────────────────────────────────────────────
    if recorder.eval_log:
        e_steps, e_vals = zip(*recorder.eval_log)
        ax.plot(e_steps, e_vals,
                color="tomato", marker="o", markersize=5,
                linewidth=2.0, label="Eval loss")

    # ── Epoch boundary lines ──────────────────────────────────────────────────
    for ep_num, ep_step in enumerate(epoch_boundaries, start=1):
        ax.axvline(ep_step, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(ep_step, y_min + (y_max - y_min) * 0.97,
                f"E{ep_num}", fontsize=7, ha="center", va="top",
                color="gray", alpha=0.7)

    ax.set_xlim(left=0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Optimizer step", fontsize=12)
    ax.set_ylabel("Cross-entropy loss (completion tokens only)", fontsize=12)
    ax.set_title(
        f"NL2SQL QLoRA SFT — Loss Curves\n"
        f"(Llama-3.2-3B-Instruct · LoRA r={cfg.lora_r} · "
        f"lr={cfg.learning_rate:.0e} · eff. batch={cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = out_dir / "loss_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss plot → {path}")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(examples: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    """
    Prompt-completion format for TRL 1.0 completion-only loss.
      prompt     : system + user turns rendered with add_generation_prompt=True
                   (ends right before the SQL tokens)
      completion : SQL text + <|eot_id|>   (the tokens we want loss on)
    TRL 1.0 auto-detects prompt-completion format → completion_only_loss=True.
    """
    prompts, completions = [], []
    for ex in examples:
        messages = ex["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],              # system + user only
            tokenize=False,
            add_generation_prompt=True, # appends <|start_header_id|>assistant<|end_header_id|>\n\n
        )
        completion = messages[-1]["content"] + "<|eot_id|>"  # SQL + end-of-turn
        prompts.append(prompt)
        completions.append(completion)
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


# ── Model + LoRA ───────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg: TrainConfig):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # double quant saves ~0.4 GB extra
    )
    logger.info(
        f"Loading base model : {cfg.model_id}\n"
        f"  HF_HUB_CACHE     : {os.environ.get('HF_HUB_CACHE')}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,         # ← torch_dtype (not dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"        # required for completion-only masking
    return model, tokenizer


def apply_lora(model, cfg: TrainConfig):
    # Gradient checkpointing via prepare_model_for_kbit_training (saves ~300 MB activations)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main(cfg: TrainConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir : {out_dir}")
    logger.info(f"Root       : {ROOT}")

    # Data
    train_examples = load_jsonl(TRAIN_FILE)
    eval_examples  = load_jsonl(EVAL_FILE)
    logger.info(f"Dataset    : {len(train_examples)} train / {len(eval_examples)} eval")

    # Model
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # Prompt-completion format — TRL 1.0 auto-enables completion_only_loss for this format
    train_ds = build_dataset(train_examples, tokenizer)
    eval_ds  = build_dataset(eval_examples,  tokenizer)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to="none",
        dataloader_pin_memory=False,      # avoid Windows CUDA pin_memory issues
        max_length=cfg.max_length,
        packing=False,                    # no packing — keeps completion boundaries clean
        # completion_only_loss auto-enabled: TRL 1.0 detects prompt-completion format
        gradient_checkpointing=False,     # handled by prepare_model_for_kbit_training above
    )

    loss_recorder = LossRecorder()

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,       # TRL 1.0: processing_class replaces tokenizer=
        callbacks=[loss_recorder],
    )

    logger.info("Training ...")
    t0 = datetime.now()
    train_result = trainer.train()
    elapsed = (datetime.now() - t0).total_seconds()
    total_steps = train_result.global_step

    # Save adapter (PEFT format — separate from base weights, required for vLLM/Punica)
    logger.info(f"Saving adapter → {out_dir}")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Loss plot
    save_loss_plot(loss_recorder, out_dir, cfg, total_steps)

    # Run metadata
    final_train = loss_recorder.train_log[-1][1] if loss_recorder.train_log else None
    final_eval  = loss_recorder.eval_log[-1][1]  if loss_recorder.eval_log  else None

    metadata = {
        "timestamp":        datetime.now().isoformat(),
        "model_id":         cfg.model_id,
        "hf_hub_cache":     os.environ.get("HF_HUB_CACHE"),
        "train_file":       str(TRAIN_FILE),
        "eval_file":        str(EVAL_FILE),
        "output_dir":       str(out_dir),
        "train_examples":   len(train_examples),
        "eval_examples":    len(eval_examples),
        "total_steps":      total_steps,
        "train_runtime_s":  round(elapsed, 1),
        "final_train_loss": round(final_train, 4) if final_train else None,
        "final_eval_loss":  round(final_eval,  4) if final_eval  else None,
        "config":           asdict(cfg),
    }
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata   → {meta_path}")

    logger.info("=" * 50)
    logger.info(f"Done in {elapsed / 60:.1f} min  |  steps: {total_steps}")
    if final_train:
        logger.info(f"Final train loss : {final_train:.4f}")
    if final_eval:
        logger.info(f"Final eval  loss : {final_eval:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL QLoRA SFT")
    parser.add_argument("--epochs",   type=int,   default=10,    help="Training epochs")
    parser.add_argument("--lr",       type=float, default=2e-4,  help="Learning rate")
    parser.add_argument("--rank",     type=int,   default=16,    help="LoRA rank r")
    parser.add_argument("--batch",    type=int,   default=2,     help="Per-device batch size")
    parser.add_argument("--grad_acc", type=int,   default=8,     help="Gradient accumulation steps")
    parser.add_argument("--output",   type=str,   default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    cfg = TrainConfig(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.rank,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        output_dir=args.output,
    )
    main(cfg)
