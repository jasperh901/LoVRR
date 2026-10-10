#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--help" ]]; then
    python -m lovrr.serve --help
    exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "lovrr_env" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Activate lovrr_env before starting LoVRR."
        exit 1
    fi
    conda activate lovrr_env
fi

exec python -m lovrr.serve "$@"
