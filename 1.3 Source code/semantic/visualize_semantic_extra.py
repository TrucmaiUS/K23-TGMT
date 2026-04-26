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
    PART_COLORS,
    PART_NAMES,
    colorize_mask,
    collect_images,
    draw_card,
    draw_patch_grid,
    fit_image,
    get_font,
    heuristic_semantic_mask,
    overlay_mask,
    patch_majority_mask,
    fit_mask,
    resize_model_input_mask,
)


def select_image_sets(dataset_root):
    query_items, _ = collect_images(dataset_root, "query")
    gallery_items, _ = collect_images(dataset_root, "bounding_box_test")

    gallery_by_pid = {}
    for item in gallery_items:
        gallery_by_pid.setdefault(item["pid"], []).append(item)

    selected_query = None
    same_identity = None
    for query in query_items:
        positives = [item for item in gallery_by_pid.get(query["pid"], []) if item["cam"] != query["cam"]]
        if len(positives) >= 2:
            selected_query = query
            same_identity = [query] + positives[:3]
            break

    if selected_query is None:
        selected_query = query_items[0]
        same_identity = query_items[:3]
    return selected_query, same_identity


def infer_grouped_masks(items, args):
    device = resolve_device(args.device)
    print(f"Loading human parser: {args.model_id}")
    processor, model = load_model(args.model_id, args.trust_remote_code, device)
    images = [Image.open(item["path"]).convert("RGB") for item in items]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = select_logits(outputs)
    mapping = PRESET_MAPPINGS[args.preset]

    grouped_masks = {}
    raw_masks = {}
    for idx, (item, image) in enumerate(zip(items, images)):
        height, width = image.size[1], image.size[0]
        sample_logits = logits[idx:idx + 1]
        sample_logits = F.interpolate(sample_logits, size=(height, width), mode="bilinear", align_corners=False)
        raw_mask = sample_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        grouped_mask = convert_mask(raw_mask, mapping)
        key = str(item["path"])
        raw_masks[key] = Image.fromarray(raw_mask, mode="L")
        grouped_masks[key] = Image.fromarray(grouped_mask, mode="L")
    for image in images:
        image.close()
    return raw_masks, grouped_masks


def save_real_vs_heuristic(query, grouped_masks, output_dir):
    real_mask = grouped_masks[str(query["path"])]
    image = fit_image(query["path"], (128, 256))
    heuristic = heuristic_semantic_mask(Image.open(query["path"]).convert("RGB"), (128, 256))
    real_overlay = overlay_mask(query["path"], real_mask)
    heuristic_overlay = overlay_mask(image, heuristic)

    canvas = Image.new("RGB", (1030, 470), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Real human-parser mask vs heuristic demo mask", font=get_font(27, True), fill="#1f2937")
    draw.text((28, 58), "This shows why semantic_groups from a parser are better than simple vertical body splits.", font=get_font(16), fill="#526071")

    cards = [
        ("Input", image, [f"pid={query['pid']} c{query['cam']}"], "#2563eb"),
        ("Heuristic mask", colorize_mask(heuristic), ["vertical split"], "#b45309"),
        ("Heuristic overlay", heuristic_overlay, ["approximate"], "#b45309"),
        ("Real mask", colorize_mask(fit_mask(real_mask, (128, 256))), ["human parser"], "#047857"),
        ("Real overlay", real_overlay, ["parser output"], "#047857"),
    ]
    x = 28
    for title, img, lines, border in cards:
        draw_card(canvas, x, 104, title, img, lines, border, card_size=(180, 340))
        x += 196

    path = output_dir / "05_real_vs_heuristic_mask.png"
    canvas.save(path)
    return path


def save_same_identity(same_identity, grouped_masks, output_dir):
    width = 260 + len(same_identity) * 210
    height = 500
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    pid = same_identity[0]["pid"]
    draw.text((28, 22), "Same identity across cameras with semantic masks", font=get_font(27, True), fill="#1f2937")
    draw.text((28, 58), f"pid={pid}. Semantic masks expose body-region consistency under different cameras/poses.", font=get_font(16), fill="#526071")

    x = 28
    for item in same_identity:
        image = fit_image(item["path"], (128, 256))
        mask = grouped_masks[str(item["path"])]
        overlay = overlay_mask(item["path"], mask)
        draw_card(
            canvas,
            x,
            104,
            f"cam c{item['cam']}",
            overlay,
            [item["path"].name[:18], "image + real mask"],
            "#047857",
            card_size=(190, 350),
        )
        x += 210

    path = output_dir / "06_same_identity_semantic_consistency.png"
    canvas.save(path)
    return path


def count_patch_parts(mask):
    patch_labels = patch_majority_mask(mask)
    counts = {label: int((patch_labels == label).sum()) for label in range(1, 7)}
    return patch_labels, counts


def save_patch_distribution(query, grouped_masks, output_dir):
    mask = grouped_masks[str(query["path"])]
    model_input_mask = resize_model_input_mask(mask, (128, 256))
    patch_labels, counts = count_patch_parts(model_input_mask)
    patch_img = draw_patch_grid(patch_labels)
    total = max(1, sum(counts.values()))

    canvas = Image.new("RGB", (980, 560), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Semantic patch distribution", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 58), "Counts 21x10 ViT patches by body-part label. This is what patch-level semantic supervision sees.", font=get_font(16), fill="#526071")

    image = fit_image(query["path"], (128, 256))
    overlay = overlay_mask(query["path"], mask)
    draw_card(canvas, 40, 120, "Overlay", overlay, [f"pid={query['pid']} c{query['cam']}"], "#2563eb", card_size=(180, 350))
    draw_card(canvas, 250, 120, "Patch target", patch_img, ["21x10 labels"], "#6d28d9", card_size=(180, 350))

    chart_x, chart_y = 500, 130
    max_count = max(counts.values()) or 1
    for idx, label in enumerate(range(1, 7)):
        y = chart_y + idx * 56
        color = PART_COLORS[label]
        bar_w = int(330 * counts[label] / max_count)
        if bar_w > 0:
            draw.rounded_rectangle([chart_x, y, chart_x + bar_w, y + 30], radius=6, fill=color)
            text_x = chart_x + 10
            text_fill = "white"
            if bar_w < 145:
                text_x = chart_x + bar_w + 10
                text_fill = "#1f2937"
        else:
            draw.rounded_rectangle([chart_x, y, chart_x + 330, y + 30], radius=6, fill="#ffffff", outline="#cbd5e1", width=1)
            text_x = chart_x + 10
            text_fill = "#1f2937"
        draw.text((text_x, y + 6), f"{PART_NAMES[label]}: {counts[label]} patches", font=get_font(15, True), fill=text_fill)
        pct = 100.0 * counts[label] / total
        draw.text((chart_x + 350, y + 6), f"{pct:.1f}%", font=get_font(15), fill="#1f2937")

    path = output_dir / "07_semantic_patch_distribution.png"
    canvas.save(path)
    return path


def save_summary_panel(output_dir):
    canvas = Image.new("RGB", (1060, 420), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 26), "What these semantic visualizations show", font=get_font(28, True), fill="#1f2937")
    bullets = [
        ("Real vs heuristic mask", "Human parser captures actual silhouette and clothing/body regions; heuristic only approximates vertical bands."),
        ("Same ID across cameras", "Semantic masks help compare body parts even when pose, crop, and camera viewpoint change."),
        ("Patch distribution", "SEM_ALIGN can supervise ViT patch tokens with part labels before the Re-ID ranking head is trained."),
    ]
    y = 94
    for title, body in bullets:
        draw.rounded_rectangle([42, y, 1010, y + 78], radius=12, fill="white", outline="#cbd5e1", width=2)
        draw.text((62, y + 15), title, font=get_font(18, True), fill="#2563eb")
        draw.text((300, y + 17), body, font=get_font(16), fill="#1f2937")
        y += 96
    path = output_dir / "08_semantic_visualization_summary.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate extra semantic Re-ID visualizations from a few real human-parser masks.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "results" / "semantic_extra_visuals"))
    parser.add_argument("--model-id", default="fashn-ai/fashn-human-parser")
    parser.add_argument("--preset", choices=sorted(PRESET_MAPPINGS.keys()), default="fashn6")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query, same_identity = select_image_sets(args.dataset_root)
    unique = []
    seen = set()
    for item in [query] + same_identity:
        key = str(item["path"])
        if key not in seen:
            unique.append(item)
            seen.add(key)

    _, grouped_masks = infer_grouped_masks(unique, args)
    paths = [
        save_real_vs_heuristic(query, grouped_masks, output_dir),
        save_same_identity(same_identity, grouped_masks, output_dir),
        save_patch_distribution(query, grouped_masks, output_dir),
        save_summary_panel(output_dir),
    ]

    print("Saved extra semantic visualizations:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
