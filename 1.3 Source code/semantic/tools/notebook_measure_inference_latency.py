# Paste this whole file into one notebook cell and run it.

from pathlib import Path
import sys
import time
import statistics
import random

import numpy as np
import torch


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

BASELINE_CONFIG = "configs/Market/vit_transreid_stride.yml"
SEMANTIC_CONFIG = "configs/Market/vit_transreid_stride_sem_align.yml"

# Optional: set to real checkpoints if you want to benchmark trained weights.
# You can keep them as None because inference latency is mostly architecture-dependent.
BASELINE_WEIGHT = None
SEMANTIC_WEIGHT = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_WARMUP = 50
NUM_ITERS = 200
USE_AMP = False

# Safe defaults for Market-1501. Classifier weights are ignored if checkpoint shapes mismatch.
NUM_CLASSES = 751
CAMERA_NUM = 6
VIEW_NUM = 1
SEED = 1234


# -----------------------------------------------------------------------------
# Repo import setup
# -----------------------------------------------------------------------------

cwd = Path.cwd().resolve()
if (cwd / "config").exists() and (cwd / "model").exists():
    repo_root = cwd
elif (cwd / "transreid_colab" / "config").exists():
    repo_root = cwd / "transreid_colab"
else:
    raise RuntimeError(
        "Cannot find transreid_colab root. Run this cell from repo root or from inside transreid_colab."
    )

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import cfg  # noqa: E402
from model import make_model  # noqa: E402


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    loaded_cfg = cfg.clone()
    loaded_cfg.merge_from_file(str(config_path))
    loaded_cfg.freeze()
    return loaded_cfg, config_path


def load_checkpoint_partial(model, checkpoint_path):
    if checkpoint_path is None:
        return

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "")
        if clean_key not in model_state:
            skipped.append(clean_key)
            continue
        if model_state[clean_key].shape != value.shape:
            skipped.append(clean_key)
            continue
        filtered_state[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded tensors: {len(filtered_state)}")
    print(f"Missing tensors after partial load: {len(missing)}")
    print(f"Unexpected tensors after partial load: {len(unexpected)}")
    print(f"Skipped tensors due to missing key or shape mismatch: {len(skipped)}")


def build_model(config_path, checkpoint_path=None):
    loaded_cfg, resolved_config_path = load_cfg(config_path)
    model = make_model(
        loaded_cfg,
        num_class=NUM_CLASSES,
        camera_num=CAMERA_NUM,
        view_num=VIEW_NUM,
    )
    load_checkpoint_partial(model, checkpoint_path)
    model.eval()
    model.to(DEVICE)
    return loaded_cfg, resolved_config_path, model


def make_inputs(loaded_cfg):
    h, w = loaded_cfg.INPUT.SIZE_TEST
    images = torch.randn(BATCH_SIZE, 3, h, w, device=DEVICE)
    camids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    viewids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    return images, camids, viewids


def benchmark_model(model, loaded_cfg, label):
    images, camids, viewids = make_inputs(loaded_cfg)
    autocast_device = "cuda" if DEVICE.startswith("cuda") else "cpu"

    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            with torch.autocast(device_type=autocast_device, enabled=USE_AMP):
                _ = model(images, cam_label=camids, view_label=viewids)

        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
            timings_ms = []
            for _ in range(NUM_ITERS):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.autocast(device_type=autocast_device, enabled=USE_AMP):
                    _ = model(images, cam_label=camids, view_label=viewids)
                end_event.record()
                torch.cuda.synchronize()
                timings_ms.append(start_event.elapsed_time(end_event))
        else:
            timings_ms = []
            for _ in range(NUM_ITERS):
                start_time = time.perf_counter()
                with torch.autocast(device_type=autocast_device, enabled=USE_AMP):
                    _ = model(images, cam_label=camids, view_label=viewids)
                end_time = time.perf_counter()
                timings_ms.append((end_time - start_time) * 1000.0)

    timings_ms = np.array(timings_ms, dtype=np.float64)
    result = {
        "label": label,
        "mean_ms": float(timings_ms.mean()),
        "std_ms": float(timings_ms.std()),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
        "min_ms": float(timings_ms.min()),
        "max_ms": float(timings_ms.max()),
        "fps": float(1000.0 / timings_ms.mean() * BATCH_SIZE),
    }
    return result


def print_result(result):
    print(f"[{result['label']}]")
    print(f"  mean: {result['mean_ms']:.3f} ms")
    print(f"  std : {result['std_ms']:.3f} ms")
    print(f"  p50 : {result['p50_ms']:.3f} ms")
    print(f"  p95 : {result['p95_ms']:.3f} ms")
    print(f"  p99 : {result['p99_ms']:.3f} ms")
    print(f"  min : {result['min_ms']:.3f} ms")
    print(f"  max : {result['max_ms']:.3f} ms")
    print(f"  fps : {result['fps']:.2f}")


def compare_results(baseline_result, semantic_result):
    delta_ms = semantic_result["mean_ms"] - baseline_result["mean_ms"]
    delta_pct = (delta_ms / baseline_result["mean_ms"]) * 100.0
    direction = "slower" if delta_ms > 0 else "faster"
    print("\nComparison")
    print(f"  semantic mean latency is {abs(delta_ms):.3f} ms {direction}")
    print(f"  relative difference: {delta_pct:+.2f}%")


# -----------------------------------------------------------------------------
# Run benchmark
# -----------------------------------------------------------------------------

set_seed(SEED)
torch.backends.cudnn.benchmark = True

print(f"repo_root   : {repo_root}")
print(f"device      : {DEVICE}")
print(f"batch_size  : {BATCH_SIZE}")
print(f"warmup      : {NUM_WARMUP}")
print(f"iterations  : {NUM_ITERS}")
print(f"use_amp     : {USE_AMP}")
print(f"camera_num  : {CAMERA_NUM}")
print(f"view_num    : {VIEW_NUM}")
print(f"num_classes : {NUM_CLASSES}")
print("")

baseline_cfg, baseline_cfg_path, baseline_model = build_model(BASELINE_CONFIG, BASELINE_WEIGHT)
semantic_cfg, semantic_cfg_path, semantic_model = build_model(SEMANTIC_CONFIG, SEMANTIC_WEIGHT)

print(f"baseline config: {baseline_cfg_path}")
print(f"semantic config: {semantic_cfg_path}")
print("")

baseline_result = benchmark_model(baseline_model, baseline_cfg, "baseline")
semantic_result = benchmark_model(semantic_model, semantic_cfg, "semantic")

print_result(baseline_result)
print("")
print_result(semantic_result)
compare_results(baseline_result, semantic_result)
