#!/bin/bash
set -euo pipefail

# Fine-tune the nanochat student on two GPUs. Each torchrun rank binds to one
# GPU, samples its own prompts, and keeps a local 4-bit teacher replica.

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

WANDB_RUN="${WANDB_RUN:-distill_2gpu}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.8-27B}"

# One process per GPU keeps the existing nanochat optimizer's gradient
# synchronization and ZeRO-2 sharding active across both devices.
torchrun --standalone --nproc_per_node=2 -m scripts.chat_distill -- \
    --run="$WANDB_RUN" \
    --teacher-model="$TEACHER_MODEL" \
    --teacher-load-in-4bit \
    "$@"
