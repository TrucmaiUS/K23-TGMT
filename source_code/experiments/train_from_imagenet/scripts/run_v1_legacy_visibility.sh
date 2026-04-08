#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/../../../core/transreid_modified" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/market/v1_legacy_visibility.yml"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../../../data}"
PRETRAIN_PATH="${PRETRAIN_PATH:-$SCRIPT_DIR/../../../references/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../../../runtime_logs/train_from_imagenet/v1_legacy_visibility}"
DEVICE_ID="${DEVICE_ID:-0}"

cd "$SRC_ROOT"
python train.py \
  --config_file "$CONFIG" \
  MODEL.DEVICE_ID "('$DEVICE_ID')" \
  MODEL.PRETRAIN_CHOICE "imagenet" \
  MODEL.PRETRAIN_PATH "('$PRETRAIN_PATH')" \
  DATASETS.ROOT_DIR "('$DATA_ROOT')" \
  OUTPUT_DIR "('$OUTPUT_DIR')"
