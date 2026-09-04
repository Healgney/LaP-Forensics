#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_PORT="${MASTER_PORT:-12346}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./checkpoints/base_model}"
DATASET_DIR="${DATASET_DIR:-./data/SynthScars_CoT}"
SAM_CKPT="${SAM_CKPT:-./checkpoints/sam_vit_h_4b8939.pth}"
LOG_BASE_DIR="${LOG_BASE_DIR:-./runs/sft}"
EXP_NAME="${EXP_NAME:-lap_forensic_sft}"

deepspeed --num_gpus="$NUM_GPUS" --master_port "$MASTER_PORT" scripts/loc_exp/train.py \
  --version "$BASE_MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --vision_pretrained "$SAM_CKPT" \
  --exp_name "$EXP_NAME" \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lr "${LR:-2e-5}" \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 1.0 \
  --bce_loss_weight 1.0 \
  --pretrained \
  --use_segm_data \
  --seg_dataset "LaP" \
  --segm_sample_rates "1" \
  --val_dataset "LaP" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-7}" \
  --log_base_dir "$LOG_BASE_DIR" \
  --use_consistency_stream \
  --vae_path "${VAE_PATH:-stabilityai/sd-vae-ft-mse}" \
  --consistency_encoder_type "${CONSISTENCY_ENCODER_TYPE:-clip}" \
  --mask_validation \
  --steps_per_epoch "${STEPS_PER_EPOCH:-165}" \
  --zero_stage "${ZERO_STAGE:-0}"
