#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"

# Check arguments
if [ -z "${1:-}" ]; then
    echo "Usage: bash scripts/rl/merge_grpo_checkpoint.sh <checkpoint_dir> [output_dir]"
    echo ""
    echo "Available checkpoints:"
    echo "---------------------"
    find "${GRPO_RUNS_DIR:-$ROOT_DIR/runs/grpo}" -name "adapter_config.json" -exec dirname {} \; 2>/dev/null | sort
    exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-${CHECKPOINT_DIR}_merged}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"

echo "=============================================="
echo "LaP-Forensic GRPO Checkpoint Merger"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "=============================================="

cmd=(
  python scripts/rl/merge_grpo_checkpoint.py
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --vision_pretrained "${SAM_CKPT}"
)

if [ -n "${BASE_MODEL_PATH}" ]; then
  cmd+=(--base_model_path "${BASE_MODEL_PATH}")
fi

"${cmd[@]}"

echo ""
echo "Done! Merged model saved to: ${OUTPUT_DIR}"
