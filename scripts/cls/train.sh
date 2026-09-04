#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/lap_forensic_merged}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/lap_cls_train}"
TRAIN_JSON="${TRAIN_JSON:-./data/detection/train.json}"
TEST_JSON="${TEST_JSON:-./data/detection/test.json}"
TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT:-./data/detection/train/images}"
TEST_IMAGE_ROOT="${TEST_IMAGE_ROOT:-./data/detection/test/images}"
EXP_NAME="${EXP_NAME:-lap_forensic_cls}"

python scripts/cls/train.py \
  --version "$MODEL_PATH" \
  --vision_pretrained "$SAM_CKPT" \
  --exp_name "$EXP_NAME" \
  --lr "${LR:-1e-3}" \
  --pretrained \
  --epochs "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --epoch_samples "${EPOCH_SAMPLES:-720119}" \
  --steps_per_epoch "${STEPS_PER_EPOCH:-5626}" \
  --save_path "$SAVE_PATH" \
  --train_json_file "$TRAIN_JSON" \
  --test_json_file "$TEST_JSON" \
  --data_base_train "$TRAIN_IMAGE_ROOT" \
  --data_base_test "$TEST_IMAGE_ROOT"
