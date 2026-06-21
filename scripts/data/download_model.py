"""
scripts/download_model.py — Download Llama-3.2-3B-Instruct to models/llama/

Downloads the gated Llama model from HuggingFace into the project-local
models/llama/ directory (excluded from git via .gitignore).

Prerequisites
─────────────
1. Accept the Llama licence:
   https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

2. Set your HuggingFace token in .env (or as an environment variable):
   HF_TOKEN=hf_your_token_here

3. Install dependencies:
   pip install transformers huggingface_hub python-dotenv

Usage
─────
  python scripts/download_model.py
  python scripts/download_model.py --model-id meta-llama/Llama-3.2-3B-Instruct
  python scripts/download_model.py --verify   # load model + run a test prompt
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_CACHE_DIR = ROOT / "models" / "llama"


def download(model_id: str, cache_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} → {cache_dir}")
    if not token:
        print(
            "WARNING: HF_TOKEN not set. Download will fail for gated models.\n"
            "  Set it in .env or export HF_TOKEN=hf_xxx"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=model_id,
        local_dir=str(cache_dir / model_id.split("/")[-1].lower()),
        token=token,
        ignore_patterns=["*.pt", "original/*"],  # skip PyTorch bin, keep safetensors
    )
    print(f"Saved to: {local_dir}")
    return Path(local_dir)


def verify(model_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nVerifying model at {model_path} ...")
    if not torch.cuda.is_available():
        print("No GPU found — loading on CPU (slow, just checks files)")
        device_map = "cpu"
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device_map = "auto"

    tok = AutoTokenizer.from_pretrained(str(model_path))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map=device_map, torch_dtype="auto"
    )

    prompt = "What was Apple's revenue in 2023?"
    inputs = tok(prompt, return_tensors="pt")
    if device_map != "cpu":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=30, do_sample=False)
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Test prompt : {prompt}")
    print(f"Response    : {response}")
    print("Verification passed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Llama-3.2-3B-Instruct")
    p.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"Local directory to cache weights (default: {DEFAULT_CACHE_DIR})",
    )
    p.add_argument(
        "--token", default=None,
        help="HuggingFace token (default: reads HF_TOKEN from env / .env)",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="After download, load the model and run a test prompt",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    model_path = download(args.model_id, args.cache_dir, token)
    if args.verify:
        verify(model_path)
    print(
        f"\nNext steps:\n"
        f"  # Train NL2SQL adapter:\n"
        f"  python finetune/adapters/nl2sql/NL2SQL_SFT.py\n\n"
        f"  # Or start the vLLM server directly:\n"
        f"  bash deployment/scripts/start_server.sh\n"
    )


if __name__ == "__main__":
    main()
