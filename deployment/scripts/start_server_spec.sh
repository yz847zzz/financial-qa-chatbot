#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_server_spec.sh — vLLM with speculative decoding
#
# Combines quantization selection with a Llama-3.2-1B-Instruct draft model.
# The draft model generates K candidate tokens; the 3B target verifies them
# all in one forward pass — same output quality, lower latency.
#
# Usage (from project root, inside WSL2):
#   bash deployment/scripts/start_server_spec.sh [quant] [K]
#
# Arguments:
#   quant   fp16 | int8 | awq4  (default: awq4)
#   K       speculative tokens per step: 1–8  (default: 4)
#
# Examples:
#   bash deployment/scripts/start_server_spec.sh awq4 4   # recommended
#   bash deployment/scripts/start_server_spec.sh fp16 3
#
# Prerequisites:
#   - Base model:  python scripts/download_model.py
#   - Draft model: python scripts/download_model.py --model-id meta-llama/Llama-3.2-1B-Instruct
#   - For awq4:    python quantize_awq.py  (run once, ~5 min)
#
# After server is ready, run the sweep:
#   python eval_speculative.py --quant awq4 --spec-tokens 1 2 3 4 5
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

QUANT="${1:-awq4}"
K="${2:-4}"

# ── Validate args ─────────────────────────────────────────────────────────────
if [[ ! "$QUANT" =~ ^(fp16|int8|awq4)$ ]]; then
    echo "Usage: $0 <fp16|int8|awq4> [K]"
    exit 1
fi
if ! [[ "$K" =~ ^[1-9]$ ]]; then
    echo "K must be 1–9 (got: $K)"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$ROOT/models"

ADAPTER_NL2SQL="$MODELS_DIR/nl2sql"

# ── Locate base (target) model ────────────────────────────────────────────────
if [[ "$QUANT" == "awq4" ]]; then
    TARGET_PATH="$MODELS_DIR/llama/llama-3.2-3b-w4a16"
    if [[ ! -f "$TARGET_PATH/config.json" ]]; then
        echo "ERROR: W4A16 model not found at $TARGET_PATH"
        echo "  Run: python quantize_awq.py"
        exit 1
    fi
else
    LLAMA_SNAPSHOTS="$MODELS_DIR/llama/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
    if [[ -d "$LLAMA_SNAPSHOTS" ]]; then
        SNAP=$(ls -1 "$LLAMA_SNAPSHOTS" | sort | tail -1)
        TARGET_PATH="$LLAMA_SNAPSHOTS/$SNAP"
    else
        TARGET_PATH="$MODELS_DIR/llama/llama-3.2-3b-instruct"
    fi
    if [[ ! -f "$TARGET_PATH/config.json" ]]; then
        echo "ERROR: Base model not found at $TARGET_PATH"
        echo "  Run: python scripts/download_model.py"
        exit 1
    fi
fi

# ── Locate draft (1B) model ───────────────────────────────────────────────────
LLAMA1B_SNAPSHOTS="$MODELS_DIR/llama/models--meta-llama--Llama-3.2-1B-Instruct/snapshots"
if [[ -d "$LLAMA1B_SNAPSHOTS" ]]; then
    SNAP1B=$(ls -1 "$LLAMA1B_SNAPSHOTS" | sort | tail -1)
    DRAFT_PATH="$LLAMA1B_SNAPSHOTS/$SNAP1B"
else
    DRAFT_PATH="$MODELS_DIR/llama/llama-3.2-1b-instruct"
fi
if [[ ! -f "$DRAFT_PATH/config.json" ]]; then
    echo "ERROR: Draft model (1B) not found at $DRAFT_PATH"
    echo "  Run: python scripts/download_model.py --model-id meta-llama/Llama-3.2-1B-Instruct"
    exit 1
fi

# ── Quantization flags ────────────────────────────────────────────────────────
case "$QUANT" in
    fp16)
        DTYPE="bfloat16"
        QUANT_FLAGS=""
        MAX_NUM_SEQS=64
        ;;
    int8)
        DTYPE="bfloat16"
        QUANT_FLAGS="--quantization bitsandbytes --load-format bitsandbytes"
        MAX_NUM_SEQS=96
        ;;
    awq4)
        DTYPE="half"
        QUANT_FLAGS="--quantization compressed-tensors"
        MAX_NUM_SEQS=128
        ;;
esac

# ── Server settings ───────────────────────────────────────────────────────────
GPU_UTIL="0.90"
PORT="${VLLM_PORT:-8001}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_LORA_RANK=16
MAX_CPU_LORAS=4

# ── LoRA adapters ─────────────────────────────────────────────────────────────
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
echo "│  vLLM Speculative Decoding Server                           │"
echo "├─────────────────────────────────────────────────────────────┤"
printf "│  %-20s %-39s │\n" "target model:"   "$TARGET_PATH"
printf "│  %-20s %-39s │\n" "draft model:"    "$DRAFT_PATH"
printf "│  %-20s %-39s │\n" "spec tokens K:"  "$K"
printf "│  %-20s %-39s │\n" "quantization:"   "$QUANT"
printf "│  %-20s %-39s │\n" "dtype:"          "$DTYPE"
printf "│  %-20s %-39s │\n" "host:port:"      "$HOST:$PORT"
printf "│  %-20s %-39s │\n" "gpu_util:"       "$GPU_UTIL"
printf "│  %-20s %-39s │\n" "max_num_seqs:"   "$MAX_NUM_SEQS"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""
echo "After server is ready, benchmark speculative decoding:"
echo "  python eval_speculative.py --quant $QUANT --spec-tokens 1 2 3 4 5"
echo ""

# ── Speculative config JSON (vLLM 0.6+ uses --speculative-config) ─────────────
SPEC_CONFIG="{\"model\": \"$DRAFT_PATH\", \"num_speculative_tokens\": $K}"

# ── Launch vLLM ───────────────────────────────────────────────────────────────
CMD=(
    vllm serve "$TARGET_PATH"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --speculative-config "$SPEC_CONFIG"
    --enable-lora
    --max-lora-rank "$MAX_LORA_RANK"
    --max-cpu-loras "$MAX_CPU_LORAS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_UTIL"
    --served-model-name "base"
    --trust-remote-code
)

if [[ -n "$LORA_MODULES" ]]; then
    CMD+=(--lora-modules $LORA_MODULES)
fi

if [[ -n "$QUANT_FLAGS" ]]; then
    read -ra QFLAGS <<< "$QUANT_FLAGS"
    CMD+=("${QFLAGS[@]}")
fi

exec "${CMD[@]}"
