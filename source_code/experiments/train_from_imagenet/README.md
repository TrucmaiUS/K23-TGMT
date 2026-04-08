# Train From ImageNet

These experiments initialize from the ImageNet ViT checkpoint
`jx_vit_base_p16_224-80ecf9dd.pth`.

## Included runs

- `baseline_transreid.yml`: baseline TransReID training setup
- `v1_legacy_visibility.yml`: first visibility-guided patch weighting run
- `residual_fusion_imagenet.yml`: residual visibility fusion proposal

## Scripts

- `scripts/run_baseline_transreid.sh`
- `scripts/run_v1_legacy_visibility.sh`
- `scripts/run_residual_fusion_imagenet.sh`

## Results

- `results/v1/`: extracted logs copied from the original run.
- `results/residual_fusion/`: no extracted logs were found in the source
  folder, but the config and proposal note are preserved.

## Important note

The `v1` run was recorded before later fusion refactors. The config is kept
here for traceability, but exact replay of the historical result may require
restoring the older source behavior described in
`docs/RESULTS_1_EVALUATION_REPORT.md`.
