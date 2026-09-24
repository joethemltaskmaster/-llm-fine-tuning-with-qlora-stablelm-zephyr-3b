#!/usr/bin/env bash
#
# run_train.sh — one-command training entry point.
#
# Sets the CUDA memory allocator config (to reduce fragmentation-related
# OutOfMemoryError crashes) and invokes train.py. Any extra arguments
# are forwarded to train.py's argparse CLI, e.g.:
#
#   ./scripts/run_train.sh
#   ./scripts/run_train.sh --lora_r 16 --max_seq_length 256
#   ./scripts/run_train.sh --resume_from_checkpoint auto
#
# Run from the project root, or the script will cd there automatically.

set -euo pipefail

# Resolve project root (parent of this script's directory) so the
# script works regardless of the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Reduce CUDA memory fragmentation (see utils.set_cuda_alloc_conf,
# which also sets this at runtime — set here too so it takes effect
# before the Python process even starts).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Load .env if present (e.g. HF_TOKEN, CUDA_VISIBLE_DEVICES).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "Starting training..."
echo "Project root: ${PROJECT_ROOT}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

python -m llm_finetune.train "$@"
