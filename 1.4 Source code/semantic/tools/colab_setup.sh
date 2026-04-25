#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p ../pretrained
if [ ! -f ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth ]; then
  wget -O ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
fi

mkdir -p ../data
mkdir -p ../logs
