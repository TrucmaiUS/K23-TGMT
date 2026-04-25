# TransReID + Visibility Weighting + Token Enrichment + Occlusion Pair Training

This directory contains an official TransReID baseline with two toggleable additions:

- visibility-guided patch weighting
- dynamic top-k token enrichment
- synthetic occlusion pair training

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
bash tools/colab_train_market_vg_occ.sh
```

Resume from an existing checkpoint and continue to a later epoch:

```bash
bash tools/colab_resume_market_vg_occ.sh 120 150
```

Or run directly:

```bash
python train.py \
  --config_file configs/Market/vit_transreid_stride.yml \
  MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR ../data \
  OUTPUT_DIR ../logs/market_vit_transreid_stride_vg_occ
```

## Ablations

Baseline TransReID:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_baseline MODEL.VIS_WEIGHT.ENABLED False MODEL.TOKEN_ENRICH.ENABLED False MODEL.OCC_AUG.ENABLED False MODEL.LOCAL_GROUP.ENABLED False
```

Patch weighting only:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_patch_only MODEL.VIS_WEIGHT.ENABLED True MODEL.TOKEN_ENRICH.ENABLED False MODEL.OCC_AUG.ENABLED False MODEL.LOCAL_GROUP.ENABLED False
```

Patch weighting + token enrichment:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_token_enrich MODEL.VIS_WEIGHT.ENABLED True MODEL.TOKEN_ENRICH.ENABLED True MODEL.OCC_AUG.ENABLED False MODEL.LOCAL_GROUP.ENABLED False
```

Full model:

```bash
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.PRETRAIN_PATH ../pretrained/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR ../data OUTPUT_DIR ../logs/market_vit_transreid_stride_vg_occ MODEL.VIS_WEIGHT.ENABLED True MODEL.TOKEN_ENRICH.ENABLED True MODEL.OCC_AUG.ENABLED True MODEL.LOCAL_GROUP.ENABLED False
```

## Notes

- Inference uses the global retrieval feature only.
- For `configs/Market/vit_transreid_stride.yml`, the original TransReID stride setup is preserved: `STRIDE_SIZE [12,12]`, `JPM True`, `SIE_CAMERA True`.
- With `STRIDE_SIZE [12,12]` and input `256x128`, the patch grid is `21x10`.
- Validation frequency is controlled by `SOLVER.EVAL_PERIOD`. Current config runs validation every 10 epochs.
- You can override validation frequency from Colab, for example: `bash tools/colab_train_market_vg_occ.sh SOLVER.EVAL_PERIOD 5`
- You can also override it while resuming, for example: `bash tools/colab_resume_market_vg_occ.sh 120 150 SOLVER.EVAL_PERIOD 5 SOLVER.CHECKPOINT_PERIOD 5`
- Best validation checkpoint is saved automatically as `transformer_best.pth`.
- The train/test scripts in `tools/` override only environment-specific paths for Colab and keep the rest of the original config intact.
