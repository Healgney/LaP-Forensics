#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

MODEL_PATH="${MODEL_PATH:-./checkpoints/lap_forensic_merged}"
IMAGE_DIR="${IMAGE_DIR:-./inference_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/inference}"

python scripts/loc_exp/infer.py \
  --hf_model_path "$MODEL_PATH" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --use_consistency_stream \
  --vae_path "${VAE_PATH:-stabilityai/sd-vae-ft-mse}" \
  --consistency_encoder_type "${CONSISTENCY_ENCODER_TYPE:-clip}"
