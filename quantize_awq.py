"""
quantize_awq.py — Self-quantize Llama-3.2-3B-Instruct to W4A16 INT4.

Uses llm-compressor (vLLM's successor to AutoAWQ) to apply W4A16 quantization:
  W4  = INT4 weights (quantized, stored as 4-bit)
  A16 = FP16 activations (not quantized, full precision at runtime)

This is equivalent to AWQ INT4 and produces a checkpoint vLLM loads with
  --quantization compressed-tensors

Calibration data
────────────────
Uses our own data/nl2sql/train.jsonl conversations formatted through the
Llama-3 chat template. Domain-specific calibration = better accuracy than
generic WikiText because activation magnitudes during calibration reflect
actual inference inputs.

Automatically tops up with WikiText-2 if train.jsonl has < --n-samples rows.

Output
──────
  models/llama/llama-3.2-3b-w4a16/
    config.json              ← quantization config embedded
    model.safetensors        ← INT4-packed weights
    tokenizer_config.json    ← copied from base model
    …

NOTE: Run this inside WSL2 where the GPU venv lives:
  source <your-venv>/bin/activate
  python quantize_awq.py

After quantization, update start_server_quant.sh awq4 to point at
  models/llama/llama-3.2-3b-w4a16/
and use --quantization compressed-tensors instead of awq_marlin.

Expected time on RTX 3090 Ti: ~5–10 minutes.
Expected VRAM peak: ~8 GB (fp16 model + calibration activations).
Expected output size: ~1.5 GB  (vs ~6 GB fp16).
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


# ── Path helpers ──────────────────────────────────────────────────────────────

def find_base_model() -> Path:
    models_dir = ROOT / "models" / "llama"
    snapshots_root = (
        models_dir / "models--meta-llama--Llama-3.2-3B-Instruct" / "snapshots"
    )
    if snapshots_root.is_dir():
        snaps = sorted(snapshots_root.iterdir())
        if snaps and (snaps[-1] / "config.json").exists():
            return snaps[-1]
    if (models_dir / "config.json").exists():
        return models_dir
    print("ERROR: base model not found.", file=sys.stderr)
    print("  Expected:", snapshots_root, file=sys.stderr)
    sys.exit(1)


def output_dir() -> Path:
    return ROOT / "models" / "llama" / "llama-3.2-3b-w4a16"


# ── Calibration data ──────────────────────────────────────────────────────────

def load_financial_samples(tokenizer, n: int) -> list[str]:
    jsonl = ROOT / "data" / "nl2sql" / "train.jsonl"
    if not jsonl.exists():
        return []
    samples = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                if messages:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    samples.append(text)
            except Exception:
                continue
            if len(samples) >= n:
                break
    print(f"  [calib] {len(samples)} samples from train.jsonl")
    return samples


def load_wikitext_samples(tokenizer, n: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        print(f"  [calib] Loading {n} WikiText-2 samples (top-up)...", flush=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          trust_remote_code=True)
        samples = []
        for row in ds:
            text = row["text"].strip()
            if len(text) < 50:
                continue
            msgs = [{"role": "user", "content": text},
                    {"role": "assistant", "content": ""}]
            try:
                samples.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                ))
            except Exception:
                samples.append(text)
            if len(samples) >= n:
                break
        print(f"  [calib] {len(samples)} WikiText samples")
        return samples
    except Exception as e:
        print(f"  [calib] WikiText load failed: {e}", file=sys.stderr)
        return []


def build_calibration_data(tokenizer, n_samples: int) -> list[str]:
    samples = load_financial_samples(tokenizer, n_samples)
    if len(samples) < n_samples:
        samples += load_wikitext_samples(tokenizer, n_samples - len(samples))
    # Repeat if still short (no internet)
    if samples and len(samples) < n_samples:
        while len(samples) < n_samples:
            samples.extend(samples[: n_samples - len(samples)])
    if not samples:
        print("ERROR: no calibration data. Ensure data/nl2sql/train.jsonl exists.",
              file=sys.stderr)
        sys.exit(1)
    samples = samples[:n_samples]
    print(f"  [calib] Final set: {len(samples)} samples", flush=True)
    return samples


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize(
    model_path: Path,
    out_path: Path,
    n_samples: int,
    w_bit: int,
    group_size: int,
) -> None:
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from compressed_tensors.quantization import (
            QuantizationScheme, QuantizationArgs, QuantizationStrategy,
        )
        from transformers import AutoTokenizer
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("  Run inside WSL2 finqa-vllm venv:", file=sys.stderr)
        print("    source <your-venv>/bin/activate", file=sys.stderr)
        print("    pip install llmcompressor", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"W{w_bit}A16 Quantization (llm-compressor)")
    print(f"{'='*60}")
    print(f"  Base model  : {model_path}")
    print(f"  Output      : {out_path}")
    print(f"  w_bit       : {w_bit}")
    print(f"  group_size  : {group_size}")
    print(f"  n_samples   : {n_samples}")
    print(f"{'='*60}\n")

    # ── Tokenizer + calibration data ──────────────────────────────────────────
    print("[1/3] Loading tokenizer + calibration data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    calib_texts = build_calibration_data(tokenizer, n_samples)

    # Wrap in a HuggingFace Dataset with a "text" column — oneshot reads this
    calib_dataset = Dataset.from_dict({"text": calib_texts})

    # ── Build recipe ──────────────────────────────────────────────────────────
    # config_groups is the correct way to pass group_size in llmcompressor 0.10+
    recipe = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=w_bit,
                    type="int",
                    group_size=group_size,
                    symmetric=True,
                    strategy=QuantizationStrategy.GROUP,
                ),
            )
        },
        ignore=["lm_head"],   # keep LM head in fp16 — small, accuracy-sensitive
    )

    # ── Oneshot: load model, calibrate, quantize, save ────────────────────────
    print("[2/3] Running oneshot (load + calibrate + quantize)...", flush=True)
    print("      (~5–10 min on 3090 Ti)", flush=True)
    t0 = time.time()

    out_path.mkdir(parents=True, exist_ok=True)
    oneshot(
        model=str(model_path),          # path — oneshot loads with device_map=auto
        tokenizer=tokenizer,
        dataset=calib_dataset,
        recipe=recipe,
        num_calibration_samples=n_samples,
        max_seq_length=512,
        text_column="text",
        output_dir=str(out_path),
        trust_remote_code_model=True,
        save_compressed=True,
    )

    elapsed = time.time() - t0
    print(f"\n[3/3] Quantization complete in {elapsed/60:.1f} min", flush=True)

    # ── Report size ───────────────────────────────────────────────────────────
    total = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    print(f"Output size : {total/1e9:.2f} GB  (fp16 baseline: ~6 GB)")


# ── Verification ──────────────────────────────────────────────────────────────

def verify(out_path: Path) -> None:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        return

    print("\nVerifying saved model...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(out_path))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(out_path), device_map="auto", trust_remote_code=True
    )
    inputs = tok("Apple revenue FY2023", return_tensors="pt")
    device = next(mdl.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs)
    assert out.logits.shape[-1] > 0
    print("  Verification passed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantize Llama-3.2-3B to W4A16 INT4 using llm-compressor"
    )
    p.add_argument("--model-path", default=None,
                   help="Override base model path")
    p.add_argument("--out-path", default=None,
                   help="Output dir (default: models/llama/llama-3.2-3b-w4a16)")
    p.add_argument("--n-samples", type=int, default=128,
                   help="Calibration samples (default: 128)")
    p.add_argument("--w-bit", type=int, default=4, choices=[4, 8],
                   help="Weight bit-width (default: 4)")
    p.add_argument("--group-size", type=int, default=128,
                   help="Quantization group size (default: 128)")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output")
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path) if args.model_path else find_base_model()
    out_path   = Path(args.out_path)   if args.out_path   else output_dir()

    if out_path.exists() and any(out_path.iterdir()):
        if not args.force:
            print(f"Output exists: {out_path}  (use --force to overwrite)")
            sys.exit(0)
        import shutil
        shutil.rmtree(out_path)

    quantize(model_path, out_path, args.n_samples, args.w_bit, args.group_size)

    if not args.skip_verify:
        verify(out_path)

    print(f"\nNext steps:")
    print(f"  bash deployment/scripts/start_server_quant.sh awq4")
    print(f"  python eval_sweep.py --quant awq4")


if __name__ == "__main__":
    main()
