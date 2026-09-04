#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/lap_cls_train}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"
TEST_JSON="${TEST_JSON:-./data/detection/test.json}"
TEST_IMAGE_ROOT="${TEST_IMAGE_ROOT:-./data/detection/test/images}"
EXP_NAME="${EXP_NAME:-lap_forensic_cls_eval}"

python scripts/cls/eval.py \
  --version "$MODEL_PATH" \
  --vision_pretrained "$SAM_CKPT" \
  --exp_name "$EXP_NAME" \
  --lr "${LR:-1e-3}" \
  --pretrained \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --test_json_file "$TEST_JSON" \
  --data_base_test "$TEST_IMAGE_ROOT"

