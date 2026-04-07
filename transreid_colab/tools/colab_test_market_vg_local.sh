#!/usr/bin/env bash
set -euo pipefail

python test.py \
  --config_file configs/Market/vit_transreid_stride.yml \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  TEST.WEIGHT ../logs/market_vit_transreid_stride_vg_local/transformer_120.pth \
  "$@"
