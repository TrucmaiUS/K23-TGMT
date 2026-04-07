# TransReID + Visibility Weighting + Local Grouping

This directory contains an official TransReID baseline with two toggleable additions:

- visibility-guided patch weighting
- local token grouping supervision

The default full-model config is `configs/Market/vit_transreid_stride.yml`.

## Colab quick start

Clone your repo, enter `transreid_colab`, then run:

```bash
bash tools/colab_setup.sh
```

Place Market-1501 under:

```text
../data/market1501/
```

Expected folders:

```text
../data/market1501/bounding_box_train
../data/market1501/query
../data/market1501/bounding_box_test
```

Train the full model:

```bash
bash tools/colab_train_market_vg_local.sh
```

Or run directly:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride.yml \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  OUTPUT_DIR ../logs/market_vit_transreid_stride_vg_local
```

## Ablations

Baseline TransReID:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_baseline MODEL.VIS_WEIGHT.ENABLED False MODEL.LOCAL_GROUP.ENABLED False
```

Patch weighting only:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_patch_only MODEL.VIS_WEIGHT.ENABLED True MODEL.LOCAL_GROUP.ENABLED False
```

Local grouping only:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_local_only MODEL.VIS_WEIGHT.ENABLED False MODEL.LOCAL_GROUP.ENABLED True
```

Full model:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_vg_local MODEL.VIS_WEIGHT.ENABLED True MODEL.LOCAL_GROUP.ENABLED True
```

## Notes

- Inference uses the global retrieval feature only.
- Local branches are training-only supervision in this v1 implementation.
- For `configs/Market/vit_transreid_stride.yml`, the original TransReID stride setup is preserved: `STRIDE_SIZE [12,12]`, `JPM True`, `SIE_CAMERA True`.
- With `STRIDE_SIZE [12,12]` and input `256x128`, the patch grid is `21x10`, so the default local grouping rows are `[0,7)`, `[7,14)`, `[14,21)`.
- The train/test scripts in `tools/` override only environment-specific paths for Colab and keep the rest of the original config intact.
