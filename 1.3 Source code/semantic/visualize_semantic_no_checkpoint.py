import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MARKET_PATTERN = re.compile(r"([-\d]+)_c(\d)")
DEFAULT_DATASET_ROOT = r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\Requirement 2 & 3\Colab\data\market1501"

PART_NAMES = {
    0: "background",
    1: "head",
    2: "torso",
    3: "arms",
    4: "hands",
    5: "legs",
    6: "feet",
}

PART_COLORS = {
    0: (245, 247, 251),
    1: (37, 99, 235),
    2: (4, 120, 87),
    3: (180, 83, 9),
    4: (109, 40, 217),
    5: (190, 18, 60),
    6: (8, 145, 178),
}


def get_font(size, bold=False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    win_font = Path("C:/Windows/Fonts") / font_name
    if win_font.exists():
        return ImageFont.truetype(str(win_font), size)
    return ImageFont.load_default()


def parse_market(path):
    match = MARKET_PATTERN.search(path.name)
    if not match:
        return None
    pid, cam = map(int, match.groups())
    if pid <= 0:
        return None
    return pid, cam


def find_split_dir(dataset_root, split):
    root = Path(dataset_root)
    candidates = [root / split, root / "market1501" / split]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find split '{split}' under {dataset_root}")


def collect_images(dataset_root, split):
    split_dir = find_split_dir(dataset_root, split)
    images = []
    for path in sorted(split_dir.glob("*.jpg")):
        parsed = parse_market(path)
        if parsed is None:
            continue
        pid, cam = parsed
        images.append({"path": path, "pid": pid, "cam": cam, "split": split})
    if not images:
        raise RuntimeError(f"No valid Market1501 images in {split_dir}")
    return images, split_dir


def select_samples(dataset_root, count, seed):
    query, _ = collect_images(dataset_root, "query")
    gallery, _ = collect_images(dataset_root, "bounding_box_test")
    by_pid_gallery = {}
    for item in gallery:
        by_pid_gallery.setdefault(item["pid"], []).append(item)
    candidates = [
        item for item in query
        if any(g["cam"] != item["cam"] for g in by_pid_gallery.get(item["pid"], []))
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def _resize_to_fit(image, size, resample, fill, mode):
    image = image.convert(mode)
    fitted = image.copy()
    fitted.thumbnail(size, resample)
    canvas = Image.new(mode, size, fill)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def fit_image(path_or_image, size, fill="white"):
    image = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image).convert("RGB")
    return _resize_to_fit(image, size, Image.Resampling.LANCZOS, fill, "RGB")


def fit_mask(mask, size, fill=0):
    return _resize_to_fit(mask, size, Image.Resampling.NEAREST, fill, "L")


def resize_model_input_mask(mask, size=(128, 256)):
    return mask.convert("L").resize(size, Image.Resampling.NEAREST)


def semantic_mask_path(dataset_root, image_item, mask_dir="semantic_groups", mask_ext=".png"):
    root = Path(dataset_root)
    candidates = [
        root / mask_dir / image_item["split"] / (image_item["path"].stem + mask_ext),
        root / "market1501" / mask_dir / image_item["split"] / (image_item["path"].stem + mask_ext),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def heuristic_semantic_mask(image, target_size=(128, 256)):
    resized = fit_image(image, target_size)
    arr = np.asarray(resized).astype(np.int16)
    non_white = np.any(arr < 245, axis=2)
    ys, xs = np.where(non_white)
    mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(mask, mode="L")

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    h = max(1, bottom - top + 1)
    w = max(1, right - left + 1)

    def fill_region(y0, y1, label, x_margin=0.0):
        x0 = int(left + x_margin * w)
        x1 = int(right - x_margin * w)
        mask[max(0, y0):min(target_size[1], y1), max(0, x0):min(target_size[0], x1)] = label

    fill_region(top, top + int(0.18 * h), 1, 0.28)
    fill_region(top + int(0.18 * h), top + int(0.52 * h), 2, 0.18)
    fill_region(top + int(0.24 * h), top + int(0.55 * h), 3, -0.08)
    fill_region(top + int(0.43 * h), top + int(0.56 * h), 4, -0.04)
    fill_region(top + int(0.52 * h), top + int(0.90 * h), 5, 0.15)
    fill_region(top + int(0.88 * h), bottom + 1, 6, 0.10)
    mask[~non_white] = 0
    return Image.fromarray(mask, mode="L")


def load_or_build_mask(dataset_root, image_item, target_size=(128, 256)):
    mask_path = semantic_mask_path(dataset_root, image_item)
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L").resize(target_size, Image.Resampling.NEAREST)
        return mask, "semantic_groups"
    image = Image.open(image_item["path"]).convert("RGB")
    return heuristic_semantic_mask(image, target_size), "heuristic vertical parts"


def colorize_mask(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    color = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for label, rgb in PART_COLORS.items():
        color[arr == label] = rgb
    return Image.fromarray(color, mode="RGB")


def overlay_mask(image, mask, alpha=0.42, size=(128, 256)):
    base = fit_image(image, size)
    aligned_mask = fit_mask(mask, size)
    color = colorize_mask(aligned_mask)
    mask_arr = np.asarray(aligned_mask, dtype=np.uint8)
    alpha_arr = np.where(mask_arr > 0, int(alpha * 255), 0).astype(np.uint8)
    color_rgba = color.convert("RGBA")
    color_rgba.putalpha(Image.fromarray(alpha_arr, mode="L"))
    output = base.convert("RGBA")
    output.alpha_composite(color_rgba)
    return output.convert("RGB")


def patch_majority_mask(mask, patch_grid=(21, 10)):
    arr = np.asarray(mask, dtype=np.uint8)
    h, w = arr.shape
    gh, gw = patch_grid
    patch = np.zeros((gh, gw), dtype=np.uint8)
    for yy in range(gh):
        y0 = int(round(yy * h / gh))
        y1 = int(round((yy + 1) * h / gh))
        for xx in range(gw):
            x0 = int(round(xx * w / gw))
            x1 = int(round((xx + 1) * w / gw))
            crop = arr[y0:y1, x0:x1].reshape(-1)
            if crop.size == 0:
                continue
            values, counts = np.unique(crop, return_counts=True)
            patch[yy, xx] = int(values[np.argmax(counts)])
    return patch


def draw_patch_grid(patch_labels, size=(128, 256)):
    gh, gw = patch_labels.shape
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    cell_w = size[0] / gw
    cell_h = size[1] / gh
    for yy in range(gh):
        for xx in range(gw):
            label = int(patch_labels[yy, xx])
            color = PART_COLORS.get(label, (200, 200, 200))
            x0 = xx * cell_w
            y0 = yy * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=color, outline=(255, 255, 255))
    return image


def draw_card(canvas, x, y, title, image, lines, border="#475569", card_size=(178, 360)):
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([x, y, x + card_size[0], y + card_size[1]], radius=12, fill="white", outline=border, width=3)
    draw.text((x + 12, y + 12), title, font=get_font(18, True), fill=border)
    canvas.paste(image, (x + (card_size[0] - image.width) // 2, y + 48))
    text_y = y + 48 + image.height + 12
    for line in lines:
        draw.text((x + 12, text_y), line, font=get_font(14), fill="#1f2937")
        text_y += 20


def save_semantic_overlay_grid(samples, dataset_root, output_dir):
    width = 1160
    card_h = 354
    height = 96 + len(samples) * (card_h + 24) + 20
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Semantic masks before training", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 58), "No best_model.pt required. Uses semantic_groups if present, otherwise heuristic body parts.", font=get_font(16), fill="#526071")

    y = 96
    mask_source = None
    for item in samples:
        image = fit_image(item["path"], (128, 256))
        mask, source = load_or_build_mask(dataset_root, item)
        mask_source = source
        colored = colorize_mask(mask)
        overlay = overlay_mask(image, mask)
        patch = draw_patch_grid(patch_majority_mask(mask))
        draw_card(canvas, 28, y, "Image", image, [f"pid={item['pid']} c{item['cam']}", item["split"]], "#2563eb")
        draw_card(canvas, 250, y, "Part mask", colored, [source], "#047857")
        draw_card(canvas, 472, y, "Overlay", overlay, ["image + mask"], "#b45309")
        draw_card(canvas, 694, y, "Patch labels", patch, ["21x10 target"], "#6d28d9")
        y += card_h + 24

    draw_legend(canvas, 920, 110, mask_source or "mask")
    path = output_dir / "01_semantic_overlay_grid.png"
    canvas.save(path)
    return path


def draw_legend(canvas, x, y, source):
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), "Legend", font=get_font(18, True), fill="#1f2937")
    draw.text((x, y + 24), f"source: {source}", font=get_font(13), fill="#526071")
    y += 56
    for label in range(1, 7):
        draw.rounded_rectangle([x, y, x + 18, y + 18], radius=4, fill=PART_COLORS[label])
        draw.text((x + 28, y - 1), f"{label}: {PART_NAMES[label]}", font=get_font(14), fill="#1f2937")
        y += 26


def erase_region(image, mask, seed=2026):
    rng = random.Random(seed)
    erased = image.copy()
    erased_mask = mask.copy()
    draw_img = ImageDraw.Draw(erased)
    draw_mask = ImageDraw.Draw(erased_mask)
    w, h = erased.size
    ew = int(w * rng.uniform(0.28, 0.42))
    eh = int(h * rng.uniform(0.20, 0.34))
    left = rng.randint(max(0, int(w * 0.18)), max(1, w - ew))
    top = rng.randint(max(0, int(h * 0.18)), max(1, h - eh))
    draw_img.rectangle([left, top, left + ew, top + eh], fill=(0, 0, 0))
    draw_mask.rectangle([left, top, left + ew, top + eh], fill=0)
    return erased, erased_mask


def save_paired_transform_visual(sample, dataset_root, output_dir):
    image = fit_image(sample["path"], (128, 256))
    mask, source = load_or_build_mask(dataset_root, sample)
    flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flipped_mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    erased_image, erased_mask = erase_region(image, mask)

    width, height = 1100, 520
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Paired image + semantic-mask transforms", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 58), "The semantic dataloader applies resize/flip/crop/erasing consistently to image and mask.", font=get_font(16), fill="#526071")

    cards = [
        ("Original image", image, "#2563eb"),
        ("Original mask", colorize_mask(mask), "#047857"),
        ("Flipped image", flipped_image, "#2563eb"),
        ("Flipped mask", colorize_mask(flipped_mask), "#047857"),
        ("Erased image", erased_image, "#be123c"),
        ("Erased mask", colorize_mask(erased_mask), "#be123c"),
    ]
    x = 28
    for title, img, border in cards:
        draw_card(canvas, x, 110, title, img, [source], border, card_size=(160, 360))
        x += 174

    path = output_dir / "02_paired_transform_visual.png"
    canvas.save(path)
    return path


def save_local_group_visual(sample, dataset_root, output_dir, row_bounds=(0, 5, 10, 16), patch_grid=(21, 10)):
    image = fit_image(sample["path"], (128, 256))
    mask, source = load_or_build_mask(dataset_root, sample)
    patch_labels = patch_majority_mask(mask, patch_grid)
    patch_img = draw_patch_grid(patch_labels)
    draw_patch = ImageDraw.Draw(patch_img)
    gh, gw = patch_grid
    cell_h = 256 / gh
    for bound in row_bounds:
        y = int(round(bound * cell_h))
        draw_patch.line([0, y, 128, y], fill=(0, 0, 0), width=3)

    width, height = 1000, 540
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Local group + semantic patch supervision", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 58), "ROW_BOUNDS split patch rows into upper/middle/lower regions; semantic labels supervise patch classes.", font=get_font(15), fill="#526071")

    draw_card(canvas, 50, 110, "Image", image, [f"pid={sample['pid']} c{sample['cam']}"], "#2563eb")
    draw_card(canvas, 270, 110, "Semantic patches", patch_img, ["21x10 grid", source], "#6d28d9")

    x0, y0 = 530, 130
    rows = [
        ("upper", row_bounds[0], row_bounds[1], "#2563eb"),
        ("middle", row_bounds[1], row_bounds[2], "#047857"),
        ("lower", row_bounds[2], row_bounds[3], "#be123c"),
    ]
    for i, (name, start, end, color) in enumerate(rows):
        y = y0 + i * 72
        draw.rounded_rectangle([x0, y, x0 + 245, y + 48], radius=10, fill="white", outline=color, width=3)
        draw.text((x0 + 16, y + 12), f"{name}: rows {start}..{end}", font=get_font(17, True), fill=color)
    draw.text((x0, y0 + 235), "Used by LOCAL_GROUP and SEM_ALIGN configs.", font=get_font(15), fill="#526071")

    path = output_dir / "03_local_group_patch_visual.png"
    canvas.save(path)
    return path


def save_pipeline_diagram(output_dir):
    width, height = 1280, 560
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 24), "Semantic Re-ID pipeline visualizable before training", font=get_font(30, True), fill="#1f2937")
    draw.text((36, 62), "These outputs explain the semantic branch even when best_model.pt is not available.", font=get_font(17), fill="#526071")

    boxes = [
        (40, 130, 190, "Market image", ["query/gallery/train", "person crop"], "#2563eb"),
        (270, 130, 220, "Human parsing", ["prepare_semantic_maps.py", "or heuristic demo mask"], "#047857"),
        (530, 130, 220, "Grouped parts", ["head, torso, arms", "hands, legs, feet"], "#b45309"),
        (790, 130, 190, "Patch targets", ["resize mask to", "21x10 ViT grid"], "#6d28d9"),
        (1020, 130, 210, "Training losses", ["patch / pixel / align", "then retrieval loss"], "#be123c"),
    ]
    for x, y, w, title, lines, color in boxes:
        draw.rounded_rectangle([x, y, x + w, y + 190], radius=14, fill="white", outline=color, width=4)
        draw.text((x + 18, y + 20), title, font=get_font(20, True), fill=color)
        ty = y + 70
        for line in lines:
            draw.text((x + 18, ty), "- " + line, font=get_font(16), fill="#1f2937")
            ty += 28
    for x in [230, 490, 750, 980]:
        draw.line([x, 225, x + 40, 225], fill="#94a3b8", width=5)
        draw.polygon([(x + 40, 225), (x + 24, 214), (x + 24, 236)], fill="#94a3b8")

    draw.rounded_rectangle([190, 390, 1090, 500], radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((220, 412), "Can run now:", font=get_font(19, True), fill="#1f2937")
    draw.text((350, 412), "visualize_semantic_no_checkpoint.py", font=get_font(19, True), fill="#2563eb")
    draw.text((220, 450), "Optional later:", font=get_font(19, True), fill="#1f2937")
    draw.text((350, 450), "prepare_semantic_maps.py -> train semantic model -> test with best checkpoint", font=get_font(17), fill="#526071")

    path = output_dir / "04_semantic_pipeline_diagram.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize semantic Re-ID inputs without a trained checkpoint.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "results" / "semantic_visual_demo"))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_samples(args.dataset_root, args.num_samples, args.seed)
    if not samples:
        raise RuntimeError("No samples selected.")

    outputs = [
        save_semantic_overlay_grid(samples, args.dataset_root, output_dir),
        save_paired_transform_visual(samples[0], args.dataset_root, output_dir),
        save_local_group_visual(samples[0], args.dataset_root, output_dir),
        save_pipeline_diagram(output_dir),
    ]

    print("Saved semantic visualizations:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
