#!/bin/bash
set -e

CONFIG=${1:-data/vi_speaker_test/config.json}
MODEL_DIR=${2:-data/vi_speaker_test/models}
GPU_ID=${3:-0}

MASTER_ADDR=localhost \
MASTER_PORT=10086 \
LOCAL_RANK=0 \
RANK=0 \
WORLD_SIZE=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
/content/melotts-vi/bin/python train.py \
  -c "$CONFIG" \
  --model "$MODEL_DIR"
