#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_server_quant.sh — Launch vLLM with a chosen quantization level.
#
# Extends start_server.sh with --quantization support for the sweep study.
#
# Quantization options (RTX 3090 Ti, 24 GB VRAM):
#   fp16   Llama-3.2-3B in bfloat16 — ~6 GB weights → ~18 GB KV cache
#   int8   bitsandbytes INT8          — ~3 GB weights → ~21 GB KV cache
#   awq4   AWQ INT4 (separate model)  — ~1.5 GB weights → ~22.5 GB KV cache
#
# Usage (from project root inside WSL2):
#   bash deployment/scripts/start_server_quant.sh fp16
#   bash deployment/scripts/start_server_quant.sh int8
#   bash deployment/scripts/start_server_quant.sh awq4
#
# Before running awq4, quantize the model yourself (5–8 min, uses local data):
#   pip install autoawq
#   python quantize_awq.py
#
# After each run:
#   Ctrl-C to stop, then start with the next quant level, then run:
#   python eval_sweep.py --quant <level>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
QUANT="${1:-fp16}"
if [[ ! "$QUANT" =~ ^(fp16|int8|awq4)$ ]]; then
    echo "Usage: $0 <fp16|int8|awq4>"
    echo "  fp16   bfloat16 baseline (default)"
    echo "  int8   bitsandbytes INT8 quantization"
    echo "  awq4   W4A16 INT4 (run python quantize_awq.py first)"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODELS_DIR="$ROOT/models"

ADAPTER_NL2SQL="$MODELS_DIR/nl2sql"

# ── Model path selection ───────────────────────────────────────────────────────
if [[ "$QUANT" == "awq4" ]]; then
    # W4A16 INT4 model produced by python quantize_awq.py (llm-compressor)
    AWQ_PATH="$MODELS_DIR/llama/llama-3.2-3b-w4a16"
    if [[ ! -f "$AWQ_PATH/config.json" ]]; then
        echo "ERROR: W4A16 model not found at $AWQ_PATH"
        echo ""
        echo "Quantize it first (activate your venv, then from project root):"
        echo "  source <your-venv>/bin/activate"
        echo "  python quantize_awq.py"
        echo ""
        echo "Takes ~5–10 min on 3090 Ti."
        exit 1
    fi
    MODEL_PATH="$AWQ_PATH"
else
    # fp16 / int8: use the standard HF-cache Llama snapshot
    LLAMA_SNAPSHOTS="$MODELS_DIR/llama/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
    if [[ -d "$LLAMA_SNAPSHOTS" ]]; then
        SNAP=$(ls -1 "$LLAMA_SNAPSHOTS" | sort | tail -1)
        MODEL_PATH="$LLAMA_SNAPSHOTS/$SNAP"
    else
        MODEL_PATH="$MODELS_DIR/llama"
    fi
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "ERROR: base model config.json not found at $MODEL_PATH"
        echo "  Run: python finetune/scripts/setup_base_model.py"
        exit 1
    fi
fi

# ── Quantization flags ────────────────────────────────────────────────────────
QUANT_FLAGS=""
case "$QUANT" in
    fp16)
        DTYPE="bfloat16"
        # No extra flags needed — vLLM defaults to fp16/bf16
        ;;
    int8)
        DTYPE="bfloat16"
        QUANT_FLAGS="--quantization bitsandbytes --load-format bitsandbytes"
        # bitsandbytes quantizes weights on the fly at load time;
        # activations remain in bf16, so --dtype stays bfloat16.
        ;;
    awq4)
        DTYPE="half"
        QUANT_FLAGS="--quantization compressed-tensors"
        # compressed-tensors is vLLM's native format for llm-compressor output.
        # W4A16 weights use Marlin kernels automatically when available.
        ;;
esac

# ── Server tuning by quant level ──────────────────────────────────────────────
# Smaller model footprint → more VRAM for KV cache → larger max batch.
case "$QUANT" in
    fp16) MAX_NUM_SEQS=64  ;;   # ~18 GB KV cache budget
    int8) MAX_NUM_SEQS=96  ;;   # ~21 GB KV cache budget
    awq4) MAX_NUM_SEQS=128 ;;   # ~22.5 GB KV cache budget
esac

GPU_UTIL="0.90"          # leave ~10% headroom for OS + CUDA overhead
PORT="${VLLM_PORT:-8001}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_LORA_RANK=16
MAX_CPU_LORAS=4

# ── LoRA adapters (same for all quant levels) ─────────────────────────────────
LORA_MODULES=""
if [[ -f "$ADAPTER_NL2SQL/adapter_config.json" ]]; then
    LORA_MODULES="nl2sql=$ADAPTER_NL2SQL"
    echo "[adapter] nl2sql : $ADAPTER_NL2SQL"
else
    echo "[adapter] nl2sql : NOT FOUND — SQL queries will fail"
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  vLLM Quantization Sweep Server                             │"
echo "├─────────────────────────────────────────────────────────────┤"
printf "│  %-20s %-39s │\n" "quantization:"  "$QUANT"
printf "│  %-20s %-39s │\n" "model:"         "$MODEL_PATH"
printf "│  %-20s %-39s │\n" "dtype:"         "$DTYPE"
printf "│  %-20s %-39s │\n" "quant_flags:"   "${QUANT_FLAGS:-none}"
printf "│  %-20s %-39s │\n" "host:port:"     "$HOST:$PORT"
printf "│  %-20s %-39s │\n" "gpu_util:"      "$GPU_UTIL"
printf "│  %-20s %-39s │\n" "max_num_seqs:"  "$MAX_NUM_SEQS"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""
echo "After server is ready, run the sweep from Windows:"
echo "  python eval_sweep.py --quant $QUANT"
echo ""

# ── Launch vLLM ───────────────────────────────────────────────────────────────
CMD=(
    vllm serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --enable-lora
    --max-lora-rank "$MAX_LORA_RANK"
    --max-cpu-loras "$MAX_CPU_LORAS"
    --max-num-seqs  "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_UTIL"
    --served-model-name "base"
    --trust-remote-code
)

# Append LoRA modules if any
if [[ -n "$LORA_MODULES" ]]; then
    CMD+=(--lora-modules $LORA_MODULES)
fi

# Append quantization flags (split on spaces, so each flag is a separate element)
if [[ -n "$QUANT_FLAGS" ]]; then
    read -ra QFLAGS <<< "$QUANT_FLAGS"
    CMD+=("${QFLAGS[@]}")
fi

exec "${CMD[@]}"
