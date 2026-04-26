import argparse
import csv
import io
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from yacs.config import CfgNode as CN

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from model import make_model


MARKET_PATTERN = re.compile(r"([-\d]+)_c(\d)")


@dataclass(frozen=True)
class ReidImage:
    path: Path
    pid: int
    camid: int


def parse_market_name(path):
    match = MARKET_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse Market1501 pid/camera from: {path}")
    pid, cam = map(int, match.groups())
    return pid, cam - 1


def find_market_split_dir(dataset_root, split):
    root = Path(dataset_root)
    candidates = [
        root / "market1501" / split,
        root / split,
    ]
    split_dir = next((p for p in candidates if p.exists()), None)
    if split_dir is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Cannot find {split}. Tried:\n{tried}")
    return split_dir


def load_market_split(dataset_root, split):
    split_dir = find_market_split_dir(dataset_root, split)
    items = []
    for img_path in sorted(split_dir.glob("*.jpg")):
        pid, camid = parse_market_name(img_path)
        if pid <= 0:
            continue
        items.append(ReidImage(img_path, pid, camid))
    if not items:
        raise RuntimeError(f"No valid Market1501 images found in {split_dir}")
    return items, split_dir


def load_market_gallery(dataset_root):
    return load_market_split(dataset_root, "bounding_box_test")


def choose_query(items, query_image=None, seed=1234):
    if query_image:
        path = Path(query_image)
        pid, camid = parse_market_name(path)
        return ReidImage(path, pid, camid)

    grouped = {}
    for item in items:
        grouped.setdefault(item.pid, []).append(item)

    valid_groups = []
    for pid, group in grouped.items():
        cams = {item.camid for item in group}
        if len(group) >= 3 and len(cams) >= 2:
            valid_groups.append((pid, group))
    if not valid_groups:
        raise RuntimeError("Need at least one pid that appears in two cameras.")

    rng = random.Random(seed)
    pid, group = rng.choice(valid_groups)
    candidates = [
        item for item in group
        if any(other.camid != item.camid for other in group)
    ]
    return rng.choice(candidates)


def build_demo_gallery(items, query, num_distractors, seed):
    positives = [
        item for item in items
        if item.path != query.path and item.pid == query.pid and item.camid != query.camid
    ]
    if not positives:
        positives = [
            item for item in items
            if item.path != query.path and item.pid == query.pid
        ]

    distractors = [item for item in items if item.pid != query.pid]
    rng = random.Random(seed)
    rng.shuffle(distractors)
    selected = positives + distractors[:num_distractors]
    rng.shuffle(selected)
    return selected, positives


def apply_checkpoint_config(checkpoint):
    if "config" in checkpoint and checkpoint["config"]:
        config_buffer = io.StringIO(checkpoint["config"])
        config_buffer.name = "checkpoint_config.yaml"
        loaded_cfg = CN.load_cfg(config_buffer)
        if "MODEL" in loaded_cfg and "DEVICE_ID" in loaded_cfg.MODEL:
            del loaded_cfg.MODEL["DEVICE_ID"]
        if "MODEL" in loaded_cfg and isinstance(loaded_cfg.MODEL.get("IF_LABELSMOOTH"), bool):
            loaded_cfg.MODEL.IF_LABELSMOOTH = "on" if loaded_cfg.MODEL.IF_LABELSMOOTH else "off"
        if "MODEL" in loaded_cfg and isinstance(loaded_cfg.MODEL.get("IF_WITH_CENTER"), bool):
            loaded_cfg.MODEL.IF_WITH_CENTER = "yes" if loaded_cfg.MODEL.IF_WITH_CENTER else "no"
        if "TEST" in loaded_cfg and isinstance(loaded_cfg.TEST.get("FEAT_NORM"), bool):
            loaded_cfg.TEST.FEAT_NORM = "yes" if loaded_cfg.TEST.FEAT_NORM else "no"
        cfg.defrost()
        cfg.merge_from_other_cfg(loaded_cfg)

    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE_ID = "0"
    cfg.freeze()


def infer_num_classes(state_dict):
    classifier = state_dict.get("classifier.weight")
    if classifier is None:
        raise RuntimeError("Cannot infer num_classes: classifier.weight is missing.")
    return classifier.shape[0]


def build_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    apply_checkpoint_config(checkpoint if isinstance(checkpoint, dict) else {})
    num_classes = infer_num_classes(state_dict)
    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Unexpected keys: {unexpected[:10]}")
    if missing:
        print(f"Missing keys: {missing[:10]}")
    model.to(device)
    model.eval()
    return model, checkpoint


def make_transform():
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def extract_features(model, items, device, batch_size):
    transform = make_transform()
    all_feats = []
    all_paths = []

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        images = []
        camids = []
        viewids = []
        for item in batch:
            image = Image.open(item.path).convert("RGB")
            images.append(transform(image))
            camids.append(item.camid)
            viewids.append(1)

        img_tensor = torch.stack(images, dim=0).to(device)
        cam_tensor = torch.tensor(camids, dtype=torch.long, device=device)
        view_tensor = torch.tensor(viewids, dtype=torch.long, device=device)
        with torch.no_grad():
            feats = model(img_tensor, cam_label=cam_tensor, view_label=view_tensor)
            feats = F.normalize(feats, dim=1)
        all_feats.append(feats.cpu())
        all_paths.extend([item.path for item in batch])

    return torch.cat(all_feats, dim=0), all_paths


def rank_gallery(model, query, gallery_items, device, batch_size):
    features, _ = extract_features(model, [query] + gallery_items, device, batch_size)
    qf = features[0:1]
    gf = features[1:]
    distances = torch.cdist(qf, gf, p=2).squeeze(0)
    order = torch.argsort(distances).tolist()
    ranked = []
    for rank_index, gallery_index in enumerate(order, start=1):
        item = gallery_items[gallery_index]
        ranked.append({
            "rank": rank_index,
            "path": item.path,
            "pid": item.pid,
            "camid": item.camid,
            "distance": float(distances[gallery_index].item()),
            "correct": item.pid == query.pid and item.camid != query.camid,
        })
    return ranked


def get_font(size, bold=False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    win_font = Path("C:/Windows/Fonts") / font_name
    if win_font.exists():
        return ImageFont.truetype(str(win_font), size)
    return ImageFont.load_default()


def fit_image(path, size):
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_labeled_person(draw, image, x, y, title, lines, border, card_size, image_size):
    font_title = get_font(22, True)
    font_body = get_font(17)
    draw.rounded_rectangle(
        [x, y, x + card_size[0], y + card_size[1]],
        radius=12,
        fill="#ffffff",
        outline=border,
        width=4,
    )
    draw.text((x + 14, y + 12), title, font=font_title, fill=border)
    image_x = x + (card_size[0] - image_size[0]) // 2
    image_y = y + 48
    draw.bitmap((0, 0), Image.new("1", (1, 1)))
    return image_x, image_y


def paste_card(canvas, item_path, x, y, title, lines, border, card_size=(190, 360), image_size=(128, 256)):
    draw = ImageDraw.Draw(canvas)
    font_title = get_font(21, True)
    font_body = get_font(16)
    draw.rounded_rectangle(
        [x, y, x + card_size[0], y + card_size[1]],
        radius=12,
        fill="#ffffff",
        outline=border,
        width=4,
    )
    draw.text((x + 14, y + 12), title, font=font_title, fill=border)
    image = fit_image(item_path, image_size)
    image_x = x + (card_size[0] - image_size[0]) // 2
    image_y = y + 48
    canvas.paste(image, (image_x, image_y))
    text_y = image_y + image_size[1] + 12
    for line in lines:
        draw.text((x + 14, text_y), line, font=font_body, fill="#1f2937")
        text_y += 22


def save_query_image(query, output_dir):
    canvas = Image.new("RGB", (420, 420), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), "Input query image", font=get_font(28, True), fill="#1f2937")
    draw.text(
        (24, 58),
        f"pid={query.pid} | camera=c{query.camid + 1}",
        font=get_font(18),
        fill="#526071",
    )
    person = fit_image(query.path, (170, 300))
    canvas.paste(person, (125, 95))
    draw.rounded_rectangle([118, 88, 302, 402], radius=12, outline="#2563eb", width=5)
    path = output_dir / "01_input_query.png"
    canvas.save(path)
    return path


def save_topk_visual(query, ranked, output_dir, topk):
    top = ranked[:topk]
    card_w, card_h = 190, 360
    gap = 24
    cols = min(5, topk)
    rows = math.ceil(topk / cols)
    left_w = 260
    width = 40 + left_w + gap + cols * card_w + (cols - 1) * gap + 40
    height = 120 + rows * card_h + (rows - 1) * gap + 44
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 26), "Person Re-ID output: ranked gallery list", font=get_font(30, True), fill="#1f2937")
    draw.text(
        (34, 64),
        "Green border = same person in another camera. Red border = different person.",
        font=get_font(18),
        fill="#526071",
    )

    paste_card(
        canvas,
        query.path,
        34,
        112,
        "QUERY",
        [f"pid={query.pid}", f"cam=c{query.camid + 1}", "input person"],
        "#2563eb",
        card_size=(220, card_h),
        image_size=(128, 256),
    )

    start_x = 34 + left_w + gap
    for i, result in enumerate(top):
        row = i // cols
        col = i % cols
        x = start_x + col * (card_w + gap)
        y = 112 + row * (card_h + gap)
        border = "#047857" if result["correct"] else "#be123c"
        verdict = "MATCH" if result["correct"] else "NO MATCH"
        paste_card(
            canvas,
            result["path"],
            x,
            y,
            f"Rank {result['rank']}",
            [
                f"{verdict}",
                f"pid={result['pid']} c{result['camid'] + 1}",
                f"dist={result['distance']:.3f}",
            ],
            border,
            card_size=(card_w, card_h),
            image_size=(128, 256),
        )

    path = output_dir / "02_ranked_topk_results.png"
    canvas.save(path)
    return path


def save_top1_comparison(query, ranked, output_dir):
    top1 = ranked[0]
    canvas = Image.new("RGB", (780, 520), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 24), "Top-1 retrieval example", font=get_font(30, True), fill="#1f2937")
    border = "#047857" if top1["correct"] else "#be123c"
    verdict = "CORRECT MATCH" if top1["correct"] else "WRONG MATCH"
    draw.text((28, 62), f"Output decision for Top-1: {verdict}", font=get_font(20, True), fill=border)
    paste_card(canvas, query.path, 70, 106, "QUERY", [f"pid={query.pid}", f"cam=c{query.camid + 1}"], "#2563eb")
    paste_card(
        canvas,
        top1["path"],
        420,
        106,
        "TOP-1",
        [verdict, f"pid={top1['pid']} c{top1['camid'] + 1}", f"dist={top1['distance']:.3f}"],
        border,
    )
    path = output_dir / "03_top1_comparison.png"
    canvas.save(path)
    return path


def write_csv(query, ranked, output_dir):
    path = output_dir / "ranked_gallery.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_path", "query_pid", "query_cam", "rank", "gallery_path", "gallery_pid", "gallery_cam", "distance", "correct_match"])
        for row in ranked:
            writer.writerow([
                str(query.path),
                query.pid,
                query.camid + 1,
                row["rank"],
                str(row["path"]),
                row["pid"],
                row["camid"] + 1,
                f"{row['distance']:.6f}",
                row["correct"],
            ])
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize a Person Re-ID retrieval example from a trained TransReID checkpoint.")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT.parent / "results" / "best_checkpoint.pth"))
    parser.add_argument("--dataset-root", default=r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\Requirement 2 & 3\Colab\data\market1501")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "results" / "visual_reid_demo"))
    parser.add_argument("--query-image", default="", help="Optional path to one Market1501 image used as the query.")
    parser.add_argument("--num-distractors", type=int, default=80)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items, gallery_dir = load_market_gallery(args.dataset_root)
    query = choose_query(items, args.query_image or None, args.seed)
    gallery_items, positives = build_demo_gallery(items, query, args.num_distractors, args.seed)
    if not positives:
        raise RuntimeError(f"Query pid={query.pid} has no positive gallery image.")

    device = torch.device(args.device)
    print(f"Gallery folder: {gallery_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Query: {query.path.name} | pid={query.pid} | cam=c{query.camid + 1}")
    print(f"Candidate gallery: {len(gallery_items)} images ({len(positives)} positives + distractors)")

    model, checkpoint = build_model_from_checkpoint(args.checkpoint, device)
    ranked = rank_gallery(model, query, gallery_items, device, args.batch_size)

    query_path = save_query_image(query, output_dir)
    topk_path = save_topk_visual(query, ranked, output_dir, args.topk)
    top1_path = save_top1_comparison(query, ranked, output_dir)
    csv_path = write_csv(query, ranked, output_dir)

    top1 = ranked[0]
    correct_in_topk = sum(1 for row in ranked[:args.topk] if row["correct"])
    best_map = checkpoint.get("best_mAP", None) if isinstance(checkpoint, dict) else None
    print("\nDemo result")
    if best_map is not None:
        print(f"Checkpoint best mAP: {best_map:.4f}")
    print(f"Top-1: rank={top1['rank']} pid={top1['pid']} cam=c{top1['camid'] + 1} dist={top1['distance']:.4f} correct={top1['correct']}")
    print(f"Correct matches in Top-{args.topk}: {correct_in_topk}")
    print("\nSaved files:")
    print(query_path)
    print(topk_path)
    print(top1_path)
    print(csv_path)


if __name__ == "__main__":
    main()
