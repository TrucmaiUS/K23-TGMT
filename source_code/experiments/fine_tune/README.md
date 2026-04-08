# Fine-Tune Experiments

These experiments fine-tune from the Market-1501 TransReID checkpoint
`vit_transreid_market.pth`.

## Included runs

- `v2_global_fusion.yml`: weighted global fusion with `VIS_FUSION_ALPHA = 0.8`
- `v3_near_baseline_alpha1.yml`: near-baseline control with `VIS_FUSION_ALPHA = 1.0`
- `v4_local_jpm.yml`: local JPM weighting
- `v5_foreground_filter.yml`: foreground-aware patch filtering
- `v6_foreground_filter_occ.yml`: foreground filtering plus synthetic occlusion

## Scripts

- `scripts/run_v2_global_fusion.sh`
- `scripts/run_v3_near_baseline_alpha1.sh`
- `scripts/run_v4_local_jpm.sh`
- `scripts/run_v5_foreground_filter.sh`
- `scripts/run_v6_foreground_filter_occ.sh`

All scripts override the config to use:

- `MODEL.PRETRAIN_CHOICE = self`
- `MODEL.PRETRAIN_PATH = <path to vit_transreid_market.pth>`

## Results

- `results/v2/`: extracted train logs were available and copied here.
- `results/v3/` to `results/v6/`: the folders are kept as placeholders for the
  corresponding runs, but the large archived artifacts remain outside this repo
  in the original `Requirement 2 & 3/RESULTS` directory.
