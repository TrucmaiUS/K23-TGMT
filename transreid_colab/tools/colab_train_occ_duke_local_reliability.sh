#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PATH="${1:-/content/pretrained/jx_vit_base_p16_224-80ecf9dd.pth}"
OUTPUT_DIR="${2:-runs/occ_duke_local_reliability}"

python train.py \
  --config_file configs/OCC_Duke/vit_transreid_stride_local_reliability.yml \
  MODEL.DEVICE_ID "'0'" \
  MODEL.PRETRAIN_CHOICE "imagenet" \
  MODEL.PRETRAIN_PATH "${PRETRAIN_PATH}" \
  DATASETS.ROOT_DIR "/content/data/" \
  OUTPUT_DIR "${OUTPUT_DIR}"
