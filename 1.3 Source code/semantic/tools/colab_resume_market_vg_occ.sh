#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash tools/colab_resume_market_vg_occ.sh <start_epoch> <new_max_epoch> [extra opts...]"
  echo "Example: bash tools/colab_resume_market_vg_occ.sh 120 150"
  exit 1
fi

START_EPOCH="$1"
NEW_MAX_EPOCH="$2"
shift 2

python train.py \
  --config_file configs/Market/vit_transreid_stride.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_CHOICE self \
  MODEL.PRETRAIN_PATH ../logs/market_vit_transreid_stride_vg_occ/transformer_"${START_EPOCH}".pth \
  DATASETS.ROOT_DIR ../data \
  OUTPUT_DIR ../logs/market_vit_transreid_stride_vg_occ \
  SOLVER.START_EPOCH "${START_EPOCH}" \
  SOLVER.MAX_EPOCHS "${NEW_MAX_EPOCH}" \
  "$@"
