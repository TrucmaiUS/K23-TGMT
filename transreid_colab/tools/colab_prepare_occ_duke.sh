#!/usr/bin/env bash
set -euo pipefail

DUKE_ZIP_ARG=()
if [[ "${1:-}" == *.zip ]]; then
  DUKE_ZIP_ARG=(--duke-zip "$1")
  OUTPUT_ROOT="${2:-/content/data/Occluded_Duke}"
  WORK_DIR="${3:-/content/data/.cache/occluded_duke_prepare}"
else
  OUTPUT_ROOT="${1:-/content/data/Occluded_Duke}"
  WORK_DIR="${2:-/content/data/.cache/occluded_duke_prepare}"
fi

if ! command -v aria2c >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  apt-get update && apt-get install -y aria2 || true
fi

python tools/prepare_occluded_duke.py \
  "${DUKE_ZIP_ARG[@]}" \
  --output-root "${OUTPUT_ROOT}" \
  --work-dir "${WORK_DIR}" \
  --duke-source auto
