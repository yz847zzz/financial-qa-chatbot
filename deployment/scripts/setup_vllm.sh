#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_vllm.sh — Install vLLM inside WSL2 and verify GPU access.
#
# vLLM is Linux-only. On Windows you must run this inside WSL2.
#
# Pre-requisites (do these BEFORE running this script):
#   1. Install WSL2:
#        wsl --install          (run in Windows PowerShell as Admin)
#        wsl --set-default-version 2
#   2. Install Ubuntu 22.04 from the Microsoft Store.
#   3. Install NVIDIA CUDA drivers for WSL:
#        https://developer.nvidia.com/cuda/wsl  (install the Windows-side driver,
#        it automatically exposes the GPU inside WSL2 — do NOT install a separate
#        Linux driver inside WSL)
#   4. Verify GPU is visible inside WSL:
#        nvidia-smi
#
# Usage (run inside WSL2 terminal):
#   bash deployment/scripts/setup_vllm.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Guard: must be Linux ──────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: vLLM requires Linux. Run this inside WSL2."
    echo "  Open WSL2: press Win+R → type 'wsl' → Enter"
    exit 1
fi

# ── Guard: GPU must be visible ────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found or GPU not visible."
    echo "  Install NVIDIA WSL2 drivers from: https://developer.nvidia.com/cuda/wsl"
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Python check ─────────────────────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Python3 not found. Installing..."
    sudo apt-get update -q && sudo apt-get install -y python3 python3-pip python3-venv
fi

PYTHON_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VER"

# ── Create / activate virtualenv ─────────────────────────────────────────────
VENV_DIR="${HOME}/venvs/finqa-vllm"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Activated: $VENV_DIR"

# ── Install vLLM ─────────────────────────────────────────────────────────────
# vLLM ships pre-built wheels for CUDA 12.1 and 12.4.
# The wheel bundles its own CUDA runtime so you don't need a full CUDA toolkit.
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "12.4")
echo "Detected CUDA: $CUDA_VER"

pip install --upgrade pip wheel setuptools

echo "Installing vLLM (this downloads ~2 GB wheel, takes a few minutes)..."
pip install vllm

# ── Install other runtime deps ────────────────────────────────────────────────
pip install \
    openai>=1.30 \
    peft>=0.11 \
    transformers>=4.41 \
    huggingface_hub>=0.23

# ── Verify install ────────────────────────────────────────────────────────────
echo ""
echo "Verifying vLLM install..."
python -c "
import vllm, torch
print(f'vLLM version : {vllm.__version__}')
print(f'PyTorch      : {torch.__version__}')
print(f'CUDA avail   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU          : {torch.cuda.get_device_name(0)}')
    print(f'VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Model path note ───────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "vLLM installed successfully."
echo ""
echo "Your Windows model files are visible inside WSL2 at:"
echo "  /mnt/e/emo/workspace/financial-qa-chatbot/models/"
echo ""
echo "Next step — start the server:"
echo "  source $VENV_DIR/bin/activate"
echo "  bash deployment/scripts/start_server.sh"
echo "──────────────────────────────────────────────────────────────"
