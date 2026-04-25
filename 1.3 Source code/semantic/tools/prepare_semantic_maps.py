import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from build_semantic_group_masks import PRESET_MAPPINGS, convert_mask


DEFAULT_MODEL_ID = "fashn-ai/fashn-human-parser"
DEFAULT_SUBDIRS = ("bounding_box_train", "query", "bounding_box_test")
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def create_progress_bar(total):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="Preparing semantic maps", unit="img")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run human parsing on a ReID dataset and save grouped semantic masks."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. /content/data/market1501")
    parser.add_argument("--output-root", default=None, help="Output root for grouped masks. Defaults to <dataset-root>/semantic_groups")
    parser.add_argument("--raw-output-root", default=None, help="Optional output root for raw parsing label maps")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id for human parsing")
    parser.add_argument("--preset", choices=sorted(PRESET_MAPPINGS.keys()), default="fashn6")
    parser.add_argument("--subdirs", nargs="+", default=list(DEFAULT_SUBDIRS), help="Image subdirectories to process")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda, cpu, or cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id, trust_remote_code, device):
    try:
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForSemanticSegmentation,
            SegformerImageProcessor,
            SegformerForSemanticSegmentation,
        )
    except ImportError as exc:
        raise ImportError(
            "transformers is required for prepare_semantic_maps.py. Install it with "
            "`pip install transformers accelerate`."
        ) from exc

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if getattr(config, "model_type", None) == "segformer":
        processor = SegformerImageProcessor.from_pretrained(model_id, use_fast=False)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    else:
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
    model.to(device)
    model.eval()
    return processor, model


def collect_image_paths(dataset_root, subdirs):
    image_paths = []
    for subdir in subdirs:
        subdir_path = dataset_root / subdir
        if not subdir_path.exists():
            continue
        for ext in DEFAULT_IMAGE_EXTS:
            image_paths.extend(sorted(subdir_path.rglob(f"*{ext}")))
            image_paths.extend(sorted(subdir_path.rglob(f"*{ext.upper()}")))
    # Deduplicate in case of mixed-case suffix matches.
    return sorted(set(image_paths))


def select_logits(outputs):
    if hasattr(outputs, "parsing_logits") and outputs.parsing_logits is not None:
        return outputs.parsing_logits
    return outputs.logits


def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


def infer_batch(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return select_logits(outputs)


def save_mask(mask_array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array, mode="L").save(path)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_root = Path(args.output_root) if args.output_root else dataset_root / "semantic_groups"
    raw_output_root = Path(args.raw_output_root) if args.raw_output_root else None
    mapping = PRESET_MAPPINGS[args.preset]
    device = resolve_device(args.device)

    image_paths = collect_image_paths(dataset_root, args.subdirs)
    if not image_paths:
        raise FileNotFoundError(
            f"No images were found under {dataset_root} for subdirs {args.subdirs}."
        )

    processor, model = load_model(args.model_id, args.trust_remote_code, device)

    num_written = 0
    num_skipped = 0
    progress_bar = create_progress_bar(len(image_paths))
    for image_batch_paths in batched(image_paths, args.batch_size):
        images = [Image.open(path).convert("RGB") for path in image_batch_paths]
        logits = infer_batch(images, processor, model, device)

        for batch_index, image_path in enumerate(image_batch_paths):
            relative_image_path = image_path.relative_to(dataset_root)
            mask_rel_path = relative_image_path.with_suffix(".png")
            grouped_output_path = output_root / mask_rel_path
            raw_output_path = raw_output_root / mask_rel_path if raw_output_root is not None else None

            if grouped_output_path.exists() and not args.overwrite:
                num_skipped += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                continue

            height, width = images[batch_index].size[1], images[batch_index].size[0]
            sample_logits = logits[batch_index:batch_index + 1]
            sample_logits = F.interpolate(sample_logits, size=(height, width), mode="bilinear", align_corners=False)
            raw_mask = sample_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            grouped_mask = convert_mask(raw_mask, mapping)

            if raw_output_path is not None:
                save_mask(raw_mask, raw_output_path)
            save_mask(grouped_mask, grouped_output_path)
            num_written += 1
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.set_postfix(written=num_written, skipped=num_skipped)

        for image in images:
            image.close()

    if progress_bar is not None:
        progress_bar.close()

    print(f"Model: {args.model_id}")
    print(f"Preset: {args.preset}")
    print(f"Device: {device}")
    print(f"Images found: {len(image_paths)}")
    print(f"Grouped masks written: {num_written}")
    print(f"Skipped existing: {num_skipped}")
    print(f"Grouped output root: {output_root}")
    if raw_output_root is not None:
        print(f"Raw parsing output root: {raw_output_root}")


if __name__ == "__main__":
    main()
