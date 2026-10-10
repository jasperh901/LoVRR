#!/bin/bash
# Start DeepThinker Service
# Usage: ./start_deepthinker_service.sh [--config config.yaml]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default config
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config config.yaml]"
            echo "  --config    Path to configuration file (default: config.yaml)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Starting DeepThinker Service"
echo "============================================"
echo "Config: $CONFIG_FILE"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found, using defaults"
fi

# Activate conda environment
echo "Activating conda environment: deepthinker_env"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || {
    echo "Error: Could not find conda. Please ensure conda is installed."
    exit 1
}

conda activate deepthinker_env || {
    echo "Error: Could not activate 'deepthinker_env' conda environment."
    echo "Please create it first: conda create -n deepthinker_env python=3.11"
    exit 1
}

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check for required packages
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "Error: PyTorch not installed"
    exit 1
}
python - "$CONFIG_FILE" <<'PY' || {
import sys
from importlib.metadata import version
from pathlib import Path

import yaml
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
config = config.get("deepthinker_service", config)
expected = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "transformers": "4.45.2",
    "qwen-vl-utils": "0.0.4",
}
if config.get("use_vllm", True):
    import vllm
    expected["vllm"] = "0.6.3.post1"
for package, required in expected.items():
    installed = version(package)
    if installed.split("+", 1)[0] != required:
        raise RuntimeError(f"{package}=={required} is required; found {installed}")
    print(f"{package}: {installed}")
PY
    echo "Error: Could not load required dependencies."
    echo "Run: pip install -r requirements-deepthinker.txt"
    exit 1
}
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || {
    echo "Error: FastAPI not installed"
    echo "Run: pip install -r requirements-deepthinker.txt"
    exit 1
}

echo ""
echo "Starting service..."
echo "============================================"

# GPU configuration: DeepThinker uses GPU 0
export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Start the service
python deepthinker_service.py --config "$CONFIG_FILE"
