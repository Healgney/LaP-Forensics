#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MASTER_PORT="${MASTER_PORT:-12347}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./checkpoints/base_model}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./runs/sft/lap_forensic_sft/ckpt_model_ce_1.0_dice_1.0_bce_1.0}"
DATASET_DIR="${DATASET_DIR:-./data/SynthScars}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"
LOG_DIR="${LOG_DIR:-./runs/eval}"
EXP_NAME="${EXP_NAME:-lap_forensic_eval}"

echo "=========================================="
echo "Starting LaP-Forensic evaluation"
echo "Base Model: $BASE_MODEL_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_DIR"
echo "=========================================="

deepspeed --num_gpus="$NUM_GPUS" --master_port "$MASTER_PORT" scripts/loc_exp/train.py \
  --version "$BASE_MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --vision_pretrained "$SAM_CKPT" \
  --exp_name "$EXP_NAME" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --pretrained \
  --use_segm_data \
  --seg_dataset "LaP" \
  --segm_sample_rates "1" \
  --val_dataset "LaP" \
  --mask_validation \
  --eval_only \
  --log_base_dir "$LOG_DIR" \
  --resume "$CHECKPOINT_PATH" \
  --use_consistency_stream \
  --vae_path "${VAE_PATH:-stabilityai/sd-vae-ft-mse}" \
  --consistency_encoder_type "${CONSISTENCY_ENCODER_TYPE:-clip}"

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
