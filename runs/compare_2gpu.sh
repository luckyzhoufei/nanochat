#!/bin/bash
set -euo pipefail

# Evaluate a distilled checkpoint and an RL checkpoint with the same settings.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Setup (skip with SKIP_SETUP=1).
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || {
        echo "uv is required. Install it from https://docs.astral.sh/uv/"
        exit 1
    }
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu --extra distill
    source .venv/bin/activate
else
    if [ ! -d ".venv" ]; then
        echo "SKIP_SETUP=1 requires an existing .venv"
        exit 1
    fi
    source .venv/bin/activate
fi

MODEL_A_SOURCE="${MODEL_A_SOURCE:-distill}"
MODEL_B_SOURCE="${MODEL_B_SOURCE:-rl}"
TASKS="${TASKS:-ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval}"

torchrun --standalone --nproc_per_node=2 -m scripts.chat_compare -- \
    --model-a-source="$MODEL_A_SOURCE" \
    --model-b-source="$MODEL_B_SOURCE" \
    --tasks="$TASKS" \
    "$@"
