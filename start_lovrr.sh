#!/bin/bash
# Start LoVRR with DeepThinker Integration
# Usage: ./start_lovrr.sh [--original]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOVRR_DIR="$(dirname "$SCRIPT_DIR")/LoVRR-main"

# Default: use DeepThinker integrated version
USE_ORIGINAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original)
            USE_ORIGINAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--original]"
            echo "  --original  Use original LoVRR without DeepThinker integration"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Starting LoVRR"
echo "============================================"

# Activate conda environment
echo "Activating conda environment: lovrr_env"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || {
    echo "Error: Could not find conda. Please ensure conda is installed."
    exit 1
}

conda activate lovrr_env || {
    echo "Error: Could not activate 'lovrr_env' conda environment."
    echo "Please create it first: conda create -n lovrr_env python=3.11"
    exit 1
}

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd "$LOVRR_DIR"

# Add LoVRR to Python path
export PYTHONPATH="$LOVRR_DIR:$PYTHONPATH"

# GPU configuration: LoVRR uses GPU 1
export CUDA_VISIBLE_DEVICES=1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

if [ "$USE_ORIGINAL" = true ]; then
    echo "Running ORIGINAL LoVRR (without DeepThinker)..."
    echo "============================================"
    python demo/app.py
else
    echo "Running LoVRR with DeepThinker Integration..."
    echo "============================================"
    echo ""
    echo "NOTE: Make sure DeepThinker service is running!"
    echo "      Run './start_deepthinker_service.sh' in another terminal first."
    echo ""
    python demo/app_with_deepthinker.py
fi
