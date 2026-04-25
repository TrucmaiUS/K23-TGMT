# Source Code Layout

This folder reorganizes the materials from `Requirement 2 & 3` into a cleaner
layout for source-code evidence and experiment reproduction.

## Structure

- `core/transreid_modified/`: the modified TransReID codebase copied from the
  working Colab version under `Colab/req23/TransReID-official`.
- `experiments/fine_tune/`: configs, scripts, docs, and lightweight result
  artifacts for experiments fine-tuned from a strong ReID checkpoint.
- `experiments/train_from_imagenet/`: configs, scripts, docs, and lightweight
  result artifacts for experiments initialized from ImageNet.
- `references/checkpoints/`: checkpoint manifest only. The actual `.pth` files
  are intentionally not copied into the repo.

## Notes

- Dataset files, zipped result archives, and model checkpoints were not copied here to keep the repository lightweight.
- The run scripts assume you execute them from a Linux/WSL/Colab-style shell.
- Some old runs, especially the first ImageNet experiment, came from earlier code states. For those cases, the config and reports are preserved here, but exact numerical reproduction may require reverting code to the corresponding historical implementation described in the docs.
