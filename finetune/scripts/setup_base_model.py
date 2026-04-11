"""
Download meta-llama/Llama-3.2-3B-Instruct, cache to models/llama/, load on GPU, run a prompt test.

Before running:
    $env:HF_TOKEN = "hf_xxx"   # Llama is gated — accept terms at hf.co/meta-llama/Llama-3.2-3B-Instruct
    python finetune/scripts/setup_base_model.py
"""

import os, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "llama"

# ── GPU check ─────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "No CUDA GPU found"
print(f"GPU : {torch.cuda.get_device_name(0)}")

# ── Load ──────────────────────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, token=token,
    quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
)
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ── Prompt test ───────────────────────────────────────────────────────────────
# Llama-3 instruct requires the chat template format
messages = [{"role": "user", "content": "What is the difference between revenue and net income?"}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

t0 = time.time()
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=150, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
elapsed = time.time() - t0

new_tokens = out[0][input_ids.shape[-1]:]
print(f"\nPrompt  : What is the difference between revenue and net income?")
print(f"Response: {tokenizer.decode(new_tokens, skip_special_tokens=True)}")
print(f"\n{len(new_tokens)} tokens in {elapsed:.1f}s ({len(new_tokens)/elapsed:.1f} tok/s)")
