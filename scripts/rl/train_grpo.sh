#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"

if command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_NUM_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  DEFAULT_NUM_GPUS=1
fi

MODEL_PATH="${1:-${MODEL_PATH:-./checkpoints/lap_forensic_merged}}"
MAX_STEPS="${2:-${MAX_STEPS:--1}}"
NUM_GPUS="${3:-${NUM_GPUS:-$DEFAULT_NUM_GPUS}}"
RESUME_CHECKPOINT="${4:-${RESUME_CHECKPOINT:-}}"
CONFIG_PATH="${CONFIG_PATH:-configs/rl_config.yaml}"
DS_CONFIG="${DS_CONFIG:-configs/ds_config_zero2.json}"
VISION_PRETRAINED="${VISION_PRETRAINED:-./checkpoints/sam_vit_h_4b8939.pth}"

# Build resume argument if checkpoint provided
RESUME_ARG=""
if [ -n "${RESUME_CHECKPOINT}" ] && [ -d "${RESUME_CHECKPOINT}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

echo "=============================================="
echo "LaP-Forensic GRPO Training"
echo "=============================================="
echo "Model Path: ${MODEL_PATH}"
echo "Max Steps: ${MAX_STEPS}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Resume From: ${RESUME_CHECKPOINT:-none}"
echo "Config: ${CONFIG_PATH}"
echo "DeepSpeed: ${DS_CONFIG}"
echo "=============================================="

accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ${DS_CONFIG} \
    scripts/rl/train_grpo.py \
    --config ${CONFIG_PATH} \
    --model_path ${MODEL_PATH} \
    --max_steps ${MAX_STEPS} \
    --vision_pretrained ${VISION_PRETRAINED} \
    ${RESUME_ARG}

echo "Training complete!"
