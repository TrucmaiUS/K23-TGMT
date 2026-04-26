from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tools" else Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "semantic"
if not SOURCE_ROOT.exists():
    SOURCE_ROOT = ROOT.parent.parent / "semantic" / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config import cfg as base_cfg  # noqa: E402
from model import make_model  # noqa: E402


MARKET_PATTERN = re.compile(r"^(-?\d+)_c(\d+)")

MODELS = [
    {
        "name": "market_baseline",
        "label": "Baseline",
        "config": ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_fair.yml",
        "checkpoint": ROOT / "runs" / "fair_market_baseline" / "transformer_best.pth",
        "color": "#334155",
    },
    {
        "name": "market_sem_align",
        "label": "Semantic",
        "config": ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_sem_align.yml",
        "checkpoint": ROOT / "runs" / "fair_market_sem_align" / "transformer_best.pth",
        "color": "#7c3aed",
    },
    {
        "name": "market_local_reliability",
        "label": "Local Rel.",
        "config": ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_local_reliability.yml",
        "checkpoint": ROOT / "runs" / "fair_market_local_reliability" / "transformer_best.pth",
        "color": "#047857",
    },
    {
        "name": "market_sem_align_reliability",
        "label": "Semantic+Rel.",
        "config": ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_sem_align_reliability.yml",
        "checkpoint": ROOT / "runs" / "fair_market_sem_align_reliability" / "transformer_best.pth",
        "color": "#b45309",
    },
]


@dataclass(frozen=True)
class ReidImage:
    path: Path
    pid: int
    camid: int


def get_font(size: int, bold: bool = False):
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    win_font = Path("C:/Windows/Fonts") / font_name
    if win_font.exists():
        return ImageFont.truetype(str(win_font), size)
    return ImageFont.load_default()


def parse_market_name(path: Path) -> tuple[int, int]:
    match = MARKET_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse Market1501 filename: {path.name}")
    pid, cam = map(int, match.groups())
    return pid, cam - 1


def find_split_dir(dataset_root: Path, split: str) -> Path:
    candidates = [dataset_root / "market1501" / split, dataset_root / split]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find split {split}. Tried:\n{tried}")


def load_split(dataset_root: Path, split: str) -> list[ReidImage]:
    split_dir = find_split_dir(dataset_root, split)
    items: list[ReidImage] = []
    for path in sorted(split_dir.glob("*.jpg")):
        pid, camid = parse_market_name(path)
        if pid > 0:
            items.append(ReidImage(path, pid, camid))
    if not items:
        raise RuntimeError(f"No valid images found in {split_dir}")
    return items


def choose_queries(query_items: list[ReidImage], gallery_items: list[ReidImage], count: int, seed: int) -> list[ReidImage]:
    gallery_by_pid: dict[int, list[ReidImage]] = {}
    for item in gallery_items:
        gallery_by_pid.setdefault(item.pid, []).append(item)

    candidates = []
    for query in query_items:
        positives = [item for item in gallery_by_pid.get(query.pid, []) if item.camid != query.camid]
        if positives:
            candidates.append(query)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if not candidates:
        raise RuntimeError("No query has a cross-camera positive match in gallery.")
    return candidates[:count]


def build_gallery(query: ReidImage, gallery_items: list[ReidImage], distractors: int, seed: int) -> list[ReidImage]:
    positives = [item for item in gallery_items if item.pid == query.pid and item.camid != query.camid]
    negatives = [item for item in gallery_items if item.pid != query.pid]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    selected = positives + negatives[:distractors]
    rng.shuffle(selected)
    return selected


def infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("classifier.weight", "module.classifier.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise RuntimeError("Cannot infer num_classes from classifier.weight")


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return {key.replace("module.", ""): value for key, value in state.items()}


def build_cfg(config_path: Path):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE_ID = "0"
    cfg.TEST.FEAT_NORM = "yes"
    cfg.freeze()
    return cfg


def build_model(spec: dict[str, object], device: torch.device):
    state = load_state_dict(Path(spec["checkpoint"]))
    cfg = build_cfg(Path(spec["config"]))
    model = make_model(cfg, num_class=infer_num_classes(state), camera_num=6, view_num=1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"{spec['label']}: missing keys {missing[:5]}")
    if unexpected:
        print(f"{spec['label']}: unexpected keys {unexpected[:5]}")
    model.to(device)
    model.eval()
    return model, cfg


def make_transform(cfg):
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )


def extract_features(model, cfg, items: list[ReidImage], device: torch.device, batch_size: int) -> torch.Tensor:
    transform = make_transform(cfg)
    feats = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        images = []
        camids = []
        viewids = []
        for item in batch:
            image = Image.open(item.path).convert("RGB")
            images.append(transform(image))
            camids.append(item.camid)
            viewids.append(1)
        image_tensor = torch.stack(images).to(device)
        cam_tensor = torch.tensor(camids, dtype=torch.long, device=device)
        view_tensor = torch.tensor(viewids, dtype=torch.long, device=device)
        with torch.no_grad():
            output = model(image_tensor, cam_label=cam_tensor, view_label=view_tensor)
            if isinstance(output, dict):
                output = output["retrieval_feat"]
            output = F.normalize(output, dim=1)
        feats.append(output.cpu())
    return torch.cat(feats, dim=0)


def rank_for_model(model, cfg, query: ReidImage, gallery: list[ReidImage], device: torch.device, batch_size: int):
    features = extract_features(model, cfg, [query] + gallery, device, batch_size)
    qf = features[0:1]
    gf = features[1:]
    distances = torch.cdist(qf, gf, p=2).squeeze(0)
    order = torch.argsort(distances).tolist()
    ranked = []
    for rank_index, gallery_index in enumerate(order, start=1):
        item = gallery[gallery_index]
        ranked.append(
            {
                "rank": rank_index,
                "path": item.path,
                "pid": item.pid,
                "camid": item.camid,
                "distance": float(distances[gallery_index]),
                "correct": item.pid == query.pid and item.camid != query.camid,
            }
        )
    return ranked


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def paste_card(
    canvas: Image.Image,
    path: Path,
    x: int,
    y: int,
    title: str,
    lines: list[str],
    border: str,
    card_size: tuple[int, int] = (152, 300),
    image_size: tuple[int, int] = (108, 216),
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([x, y, x + card_size[0], y + card_size[1]], radius=8, fill="#ffffff", outline=border, width=4)
    draw.text((x + 10, y + 10), title, font=get_font(16, True), fill=border)
    image = fit_image(path, image_size)
    canvas.paste(image, (x + (card_size[0] - image_size[0]) // 2, y + 42))
    ty = y + 42 + image_size[1] + 8
    for line in lines:
        draw.text((x + 10, ty), line, font=get_font(13), fill="#1f2937")
        ty += 18


def save_query_comparison(query: ReidImage, results: dict[str, list[dict]], output_dir: Path, topk: int) -> Path:
    card_w, card_h = 152, 300
    query_w = 170
    label_w = 155
    gap = 14
    header_h = 86
    row_h = card_h + 26
    width = 32 + label_w + query_w + gap + topk * card_w + (topk - 1) * gap + 32
    height = header_h + len(MODELS) * row_h + 30
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "Same-query Top-K comparison across models", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 54), "Green = correct identity, red = wrong identity. Lower distance means more similar.", font=get_font(17), fill="#526071")

    y = header_h
    for spec in MODELS:
        model_name = str(spec["name"])
        label = str(spec["label"])
        color = str(spec["color"])
        draw.text((34, y + 18), label, font=get_font(20, True), fill=color)
        top1 = results[model_name][0]
        first_true = next((row for row in results[model_name] if row["correct"]), None)
        true_rank = first_true["rank"] if first_true else "NA"
        draw.text((34, y + 48), f"Top-1: {'OK' if top1['correct'] else 'Fail'}", font=get_font(14), fill="#1f2937")
        draw.text((34, y + 68), f"True rank: {true_rank}", font=get_font(14), fill="#1f2937")

        paste_card(canvas, query.path, 32 + label_w, y, "QUERY", [f"pid={query.pid}", f"cam=c{query.camid + 1}"], "#2563eb", (query_w, card_h), (108, 216))
        x = 32 + label_w + query_w + gap
        for row in results[model_name][:topk]:
            border = "#047857" if row["correct"] else "#be123c"
            paste_card(
                canvas,
                row["path"],
                x,
                y,
                f"R{row['rank']}",
                [("MATCH" if row["correct"] else "NO MATCH"), f"pid={row['pid']} c{row['camid'] + 1}", f"d={row['distance']:.3f}"],
                border,
                (card_w, card_h),
                (108, 216),
            )
            x += card_w + gap
        y += row_h

    path = output_dir / "01_same_query_topk_comparison.png"
    canvas.save(path)
    return path


def save_rank_matrix(queries: list[ReidImage], all_results: dict[str, dict[Path, list[dict]]], output_dir: Path) -> Path:
    cell_w, cell_h = 165, 72
    left_w = 240
    header_h = 90
    width = left_w + len(MODELS) * cell_w + 40
    height = header_h + len(queries) * cell_h + 50
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "True-match rank matrix", font=get_font(28, True), fill="#1f2937")
    draw.text((28, 54), "Each cell shows where the first correct gallery image appears for a query.", font=get_font(16), fill="#526071")

    for j, spec in enumerate(MODELS):
        x = left_w + j * cell_w
        draw.text((x + 10, header_h - 30), str(spec["label"]), font=get_font(15, True), fill=str(spec["color"]))

    for i, query in enumerate(queries):
        y = header_h + i * cell_h
        draw.text((28, y + 12), f"{query.path.name}", font=get_font(13, True), fill="#1f2937")
        draw.text((28, y + 34), f"pid={query.pid}, cam=c{query.camid + 1}", font=get_font(13), fill="#526071")
        for j, spec in enumerate(MODELS):
            name = str(spec["name"])
            rows = all_results[name][query.path]
            top1 = rows[0]
            first_true = next((row for row in rows if row["correct"]), None)
            true_rank = first_true["rank"] if first_true else None
            x = left_w + j * cell_w
            fill = "#dcfce7" if top1["correct"] else "#fee2e2"
            draw.rounded_rectangle([x + 8, y + 8, x + cell_w - 8, y + cell_h - 8], radius=8, fill=fill, outline="#cbd5e1")
            text = "Top-1 OK" if top1["correct"] else f"True R{true_rank}"
            draw.text((x + 20, y + 20), text, font=get_font(16, True), fill="#047857" if top1["correct"] else "#be123c")
            draw.text((x + 20, y + 43), f"top1 pid={top1['pid']}", font=get_font(12), fill="#526071")

    path = output_dir / "02_true_match_rank_matrix.png"
    canvas.save(path)
    return path


def write_csv(output_dir: Path, queries: list[ReidImage], all_results: dict[str, dict[Path, list[dict]]], topk: int) -> Path:
    path = output_dir / "multimodel_topk_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "query_pid", "query_cam", "model", "rank", "gallery", "gallery_pid", "gallery_cam", "distance", "correct"])
        for query in queries:
            for spec in MODELS:
                name = str(spec["name"])
                for row in all_results[name][query.path][:topk]:
                    writer.writerow(
                        [
                            query.path.name,
                            query.pid,
                            query.camid + 1,
                            spec["label"],
                            row["rank"],
                            Path(row["path"]).name,
                            row["pid"],
                            row["camid"] + 1,
                            f"{row['distance']:.6f}",
                            row["correct"],
                        ]
                    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval outputs from four TransReID checkpoints.")
    parser.add_argument("--dataset-root", default=r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\Requirement 2 & 3\Colab\data\market1501")
    parser.add_argument("--output-dir", default=str(ROOT / "visualizations" / "qualitative"))
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--num-distractors", type=int, default=60)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset_root = Path(args.dataset_root)
    query_items = load_split(dataset_root, "query")
    gallery_items = load_split(dataset_root, "bounding_box_test")
    queries = choose_queries(query_items, gallery_items, args.num_queries, args.seed)
    galleries = {query.path: build_gallery(query, gallery_items, args.num_distractors, args.seed + idx) for idx, query in enumerate(queries)}

    all_results: dict[str, dict[Path, list[dict]]] = {str(spec["name"]): {} for spec in MODELS}
    for spec in MODELS:
        print(f"Loading {spec['label']} from {spec['checkpoint']}")
        model, cfg = build_model(spec, device)
        for query in queries:
            ranked = rank_for_model(model, cfg, query, galleries[query.path], device, args.batch_size)
            all_results[str(spec["name"])][query.path] = ranked
            top1 = ranked[0]
            first_true = next((row for row in ranked if row["correct"]), None)
            true_rank = first_true["rank"] if first_true else "NA"
            print(f"  {query.path.name}: top1 pid={top1['pid']} correct={top1['correct']} true_rank={true_rank}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    first_query = queries[0]
    one_query_results = {name: per_query[first_query.path] for name, per_query in all_results.items()}
    out1 = save_query_comparison(first_query, one_query_results, output_dir, args.topk)
    out2 = save_rank_matrix(queries, all_results, output_dir)
    out3 = write_csv(output_dir, queries, all_results, args.topk)

    print("Saved:")
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
