#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/scripts/loc_exp:$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-./checkpoints/base_model}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./runs/sft/lap_forensic_sft/ckpt_model_ce_1.0_dice_1.0_bce_1.0}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-global_step0000}"
CHECKPOINT_BIN="${CHECKPOINT_BIN:-$CHECKPOINT_ROOT/$CHECKPOINT_TAG/pytorch_model.bin}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-./checkpoints/lap_forensic_merged}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"

mkdir -p "$ROOT_DIR/runs/merge_logs"
LOG_FILE="$ROOT_DIR/runs/merge_logs/merge_${CHECKPOINT_TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

python scripts/merge_weights/merge_lora_weights.py \
  --version "$BASE_MODEL_PATH" \
  --weight "$CHECKPOINT_BIN" \
  --save_path "$MERGED_MODEL_PATH" \
  --vision_pretrained "$SAM_CKPT" \
  --use_consistency_stream \
  2>&1 | tee "$LOG_FILE"

echo "Log saved to: $LOG_FILE"
