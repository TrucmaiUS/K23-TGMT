#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_PATH="${1:-/content/pretrained/jx_vit_base_p16_224-80ecf9dd.pth}"
OUTPUT_DIR="${2:-runs/occ_duke_sem_decoder_reliability}"
DATASET_ROOT="${3:-/content/data/Occluded_Duke}"

if [ ! -d "${DATASET_ROOT}/semantic_groups" ]; then
  python tools/prepare_semantic_maps.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-root "${DATASET_ROOT}/semantic_groups" \
    --raw-output-root "${DATASET_ROOT}/raw_parsing" \
    --preset lip6 \
    --subdirs bounding_box_train query bounding_box_test \
    --batch-size 64 \
    --device cuda
fi

python train.py \
  --config_file configs/OCC_Duke/vit_transreid_stride_sem_decoder_reliability.yml \
  MODEL.DEVICE_ID "'0'" \
  MODEL.PRETRAIN_CHOICE "imagenet" \
  MODEL.PRETRAIN_PATH "${PRETRAIN_PATH}" \
  DATASETS.ROOT_DIR "/content/data/" \
  OUTPUT_DIR "${OUTPUT_DIR}"
