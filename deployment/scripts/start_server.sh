#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_server.sh — Launch the vLLM multi-adapter server.
#
# Serves Llama-3.2-3B-Instruct as the base model with all LoRA adapters
# registered at startup. Adapter switching per-request is handled by vLLM
# using SGMV (Segmented Gather Matrix-Vector) kernels — multiple adapters
# can be active in the same GPU batch with zero overhead switching cost.
#
# Adapter routing:
#   model="base"      → plain Llama (intent, decompose, answer, rewrite)
#   model="nl2sql"    → NL2SQL LoRA adapter
#   model="intent"    → intent classifier LoRA adapter (when trained)
#   model="rewriter"  → query rewriter LoRA adapter (when trained)
#
# Usage (inside WSL2, from project root):
#   bash deployment/scripts/start_server.sh
#   bash deployment/scripts/start_server.sh --port 8002 --gpu-util 0.85
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve project root (works whether called from root or scripts/) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Windows → WSL2 path translation ──────────────────────────────────────────
# If models/ is on E: drive, map it to /mnt/e/ inside WSL2
if [[ -d "/mnt/e/emo/workspace/financial-qa-chatbot/models" ]]; then
    MODELS_DIR="/mnt/e/emo/workspace/financial-qa-chatbot/models"
else
    MODELS_DIR="$ROOT/models"
fi

MODEL_BASE="$MODELS_DIR/llama"
ADAPTER_NL2SQL="$MODELS_DIR/nl2sql"
ADAPTER_INTENT="$MODELS_DIR/intent_classifier"
ADAPTER_REWRITER="$MODELS_DIR/query_rewriter"

# ── Defaults (overridable via env or CLI args) ────────────────────────────────
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.88}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-16}"
MAX_CPU_LORAS="${VLLM_MAX_CPU_LORAS:-4}"   # adapters kept warm in CPU RAM
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"    # max concurrent sequences in a batch
DTYPE="${VLLM_DTYPE:-bfloat16}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)       PORT="$2";     shift 2 ;;
        --host)       HOST="$2";     shift 2 ;;
        --gpu-util)   GPU_UTIL="$2"; shift 2 ;;
        --max-seqs)   MAX_NUM_SEQS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Find Llama snapshot (HF cache layout) ────────────────────────────────────
SNAPSHOT=$(find "$MODEL_BASE" -type d -name "snapshots" 2>/dev/null | head -1)
if [[ -n "$SNAPSHOT" ]]; then
    MODEL_PATH=$(find "$SNAPSHOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
else
    MODEL_PATH="$MODEL_BASE"
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: model config.json not found at $MODEL_PATH"
    echo "  Run: python finetune/scripts/setup_base_model.py"
    exit 1
fi

# ── Build --lora-modules argument ─────────────────────────────────────────────
# Format: name=path  (register each adapter that exists on disk)
LORA_MODULES=""

if [[ -f "$ADAPTER_NL2SQL/adapter_config.json" ]]; then
    LORA_MODULES="$LORA_MODULES nl2sql=$ADAPTER_NL2SQL"
    echo "[adapter] nl2sql    : $ADAPTER_NL2SQL"
else
    echo "[adapter] nl2sql    : NOT FOUND (run NL2SQL_SFT.py to train)"
fi

if [[ -f "$ADAPTER_INTENT/adapter_config.json" ]]; then
    LORA_MODULES="$LORA_MODULES intent=$ADAPTER_INTENT"
    echo "[adapter] intent    : $ADAPTER_INTENT"
else
    echo "[adapter] intent    : not trained yet (base model used)"
fi

if [[ -f "$ADAPTER_REWRITER/adapter_config.json" ]]; then
    LORA_MODULES="$LORA_MODULES rewriter=$ADAPTER_REWRITER"
    echo "[adapter] rewriter  : $ADAPTER_REWRITER"
else
    echo "[adapter] rewriter  : not trained yet (base model used)"
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo ""
echo "Starting vLLM server"
echo "  model      : $MODEL_PATH"
echo "  host:port  : $HOST:$PORT"
echo "  dtype      : $DTYPE"
echo "  gpu_util   : $GPU_UTIL"
echo "  max_seqs   : $MAX_NUM_SEQS"
echo "  lora_rank  : $MAX_LORA_RANK"
echo "  cpu_loras  : $MAX_CPU_LORAS  (adapters pre-loaded in CPU RAM)"
echo "  SGMV       : auto-enabled for batched multi-adapter inference"
echo ""

# ── Launch vLLM ───────────────────────────────────────────────────────────────
# --enable-lora           : activates LoRA runtime + SGMV kernel selection
# --lora-modules          : registers adapters at startup (hot-swap via request)
# --max-lora-rank         : must match rank used in training (16)
# --max-cpu-loras         : how many adapters to keep in CPU RAM (paged to GPU on demand)
# --max-num-seqs          : batch size ceiling; higher = better GPU utilization
# --gpu-memory-utilization: fraction of VRAM for KV cache (leave ~10% headroom)
# --disable-log-requests  : suppress per-request access log (keep terminal clean)

exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --enable-lora \
    ${LORA_MODULES:+--lora-modules $LORA_MODULES} \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-cpu-loras "$MAX_CPU_LORAS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --served-model-name "base" \
    --trust-remote-code \
    --disable-log-requests
