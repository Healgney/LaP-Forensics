#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./runs/sft/lap_forensic_sft/ckpt_model_ce_1.0_dice_1.0_bce_1.0}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-global_step0000}"
CHECKPOINT_BIN="${CHECKPOINT_BIN:-$CHECKPOINT_ROOT/$CHECKPOINT_TAG/pytorch_model.bin}"

python scripts/merge_weights/convert_non_zero.py \
  "$CHECKPOINT_ROOT" \
  "$CHECKPOINT_BIN" \
  --tag "$CHECKPOINT_TAG"
 
