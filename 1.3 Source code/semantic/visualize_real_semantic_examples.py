import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_semantic_group_masks import PRESET_MAPPINGS, convert_mask
from prepare_semantic_maps import load_model, resolve_device, select_logits
from visualize_semantic_no_checkpoint import (
    DEFAULT_DATASET_ROOT,
    colorize_mask,
    collect_images,
    draw_card,
    draw_legend,
    draw_patch_grid,
    fit_image,
    fit_mask,
    get_font,
    overlay_mask,
    patch_majority_mask,
    resize_model_input_mask,
)


def select_example_images(dataset_root, split, count):
    images, _ = collect_images(dataset_root, split)
    return images[:count]


def run_human_parser(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return select_logits(outputs)


def save_label_mask(mask_array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array.astype(np.uint8)).save(path)


def create_real_mask_examples(args):
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_masks"
    grouped_dir = output_dir / "semantic_groups"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    samples = select_example_images(args.dataset_root, args.split, args.num_images)
    pil_images = [Image.open(item["path"]).convert("RGB") for item in samples]

    print(f"Loading human parser: {args.model_id}")
    processor, model = load_model(args.model_id, args.trust_remote_code, device)
    logits = run_human_parser(pil_images, processor, model, device)
    mapping = PRESET_MAPPINGS[args.preset]

    rows = []
    for idx, (item, image) in enumerate(zip(samples, pil_images)):
        height, width = image.size[1], image.size[0]
        sample_logits = logits[idx:idx + 1]
        sample_logits = F.interpolate(sample_logits, size=(height, width), mode="bilinear", align_corners=False)
        raw_mask = sample_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        grouped_mask = convert_mask(raw_mask, mapping)

        raw_path = raw_dir / item["split"] / (item["path"].stem + ".png")
        grouped_path = grouped_dir / item["split"] / (item["path"].stem + ".png")
        save_label_mask(raw_mask, raw_path)
        save_label_mask(grouped_mask, grouped_path)

        grouped_pil = Image.fromarray(grouped_mask, mode="L")
        rows.append({
            "item": item,
            "image": fit_image(item["path"], (128, 256)),
            "raw": Image.fromarray(raw_mask, mode="L"),
            "grouped": grouped_pil,
            "raw_path": raw_path,
            "grouped_path": grouped_path,
        })

    for image in pil_images:
        image.close()

    visual_path = save_visual(rows, output_dir, args.model_id, args.preset)
    return visual_path, raw_dir, grouped_dir


def raw_palette(raw_mask):
    arr = np.asarray(raw_mask, dtype=np.uint8)
    palette = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for label in np.unique(arr):
        if label == 0:
            color = (245, 247, 251)
        else:
            color = (
                int((37 * int(label) + 80) % 255),
                int((83 * int(label) + 120) % 255),
                int((149 * int(label) + 40) % 255),
            )
        palette[arr == label] = color
    return Image.fromarray(palette, mode="RGB")


def save_visual(rows, output_dir, model_id, preset):
    card_h = 360
    width = 1240
    height = 118 + len(rows) * (card_h + 26) + 26
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Real semantic masks from a pretrained human parser", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 58), f"Model: {model_id} | grouped preset: {preset} | no Re-ID checkpoint required", font=get_font(16), fill="#526071")

    y = 108
    for row in rows:
        item = row["item"]
        grouped_display = fit_mask(row["grouped"], (128, 256))
        raw_display = fit_mask(row["raw"], (128, 256))
        grouped_color = colorize_mask(grouped_display)
        overlay = overlay_mask(item["path"], row["grouped"])
        patch_grid = draw_patch_grid(patch_majority_mask(resize_model_input_mask(row["grouped"], (128, 256))))
        raw_color = raw_palette(raw_display)

        draw_card(canvas, 28, y, "Input", row["image"], [f"pid={item['pid']} c{item['cam']}", item["split"]], "#2563eb")
        draw_card(canvas, 228, y, "Raw parser", raw_color, ["parser labels"], "#475569")
        draw_card(canvas, 428, y, "Grouped mask", grouped_color, ["6 body groups"], "#047857")
        draw_card(canvas, 628, y, "Overlay", overlay, ["image + grouped"], "#b45309")
        draw_card(canvas, 828, y, "Patch target", patch_grid, ["21x10 labels"], "#6d28d9")
        y += card_h + 26

    draw_legend(canvas, 1030, 126, "real parser")
    path = output_dir / "real_semantic_examples.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate real semantic masks for a few Market1501 images and visualize them.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "results" / "semantic_real_examples"))
    parser.add_argument("--split", default="query", choices=["query", "bounding_box_train", "bounding_box_test"])
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--model-id", default="fashn-ai/fashn-human-parser")
    parser.add_argument("--preset", choices=sorted(PRESET_MAPPINGS.keys()), default="fashn6")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    args = parser.parse_args()

    visual_path, raw_dir, grouped_dir = create_real_mask_examples(args)
    print("Saved real semantic example:")
    print(visual_path)
    print("Raw masks:")
    print(raw_dir)
    print("Grouped masks:")
    print(grouped_dir)


if __name__ == "__main__":
    main()
