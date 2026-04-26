import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg
from visualize_person_reid_demo import (
    ReidImage,
    build_demo_gallery,
    build_model_from_checkpoint,
    choose_query,
    extract_features,
    fit_image,
    get_font,
    load_market_gallery,
    load_market_split,
    make_transform,
    paste_card,
    rank_gallery,
)


COLORS = [
    "#2563eb", "#047857", "#b45309", "#6d28d9", "#be123c",
    "#0891b2", "#854d0e", "#4338ca", "#0f766e", "#c2410c",
]


def group_valid_queries(items):
    by_pid = {}
    for item in items:
        by_pid.setdefault(item.pid, []).append(item)
    groups = []
    for pid, group in by_pid.items():
        cams = {item.camid for item in group}
        if len(group) >= 3 and len(cams) >= 2:
            groups.append((pid, sorted(group, key=lambda x: x.path.name)))
    return sorted(groups, key=lambda pair: pair[0])


def select_queries(items, count, seed):
    rng = random.Random(seed)
    groups = group_valid_queries(items)
    rng.shuffle(groups)
    queries = []
    for _, group in groups:
        candidates = [
            item for item in group
            if any(other.camid != item.camid for other in group)
        ]
        if candidates:
            queries.append(rng.choice(candidates))
        if len(queries) >= count:
            break
    if not queries:
        raise RuntimeError("No valid query candidates found.")
    return queries


def select_queries_with_gallery(query_items, gallery_items, count, seed):
    gallery_by_pid = {}
    for item in gallery_items:
        gallery_by_pid.setdefault(item.pid, []).append(item)

    candidates = []
    for query in query_items:
        positives = [
            item for item in gallery_by_pid.get(query.pid, [])
            if item.camid != query.camid
        ]
        if positives:
            candidates.append(query)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if not candidates:
        raise RuntimeError("No query images have positive matches in gallery.")
    return candidates[:count]


def save_multi_query_grid(query_results, output_dir, topk):
    card_w, card_h = 170, 348
    query_w = 190
    gap = 18
    header_h = 92
    row_h = card_h + 28
    width = 34 + query_w + gap + topk * card_w + (topk - 1) * gap + 34
    height = header_h + len(query_results) * row_h + 34
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Multi-query retrieval overview", font=get_font(30, True), fill="#1f2937")
    draw.text((28, 58), "Each row is one input person. Green = correct ID, red = wrong ID.", font=get_font(18), fill="#526071")

    y = header_h
    for query, ranked in query_results:
        paste_card(
            canvas,
            query.path,
            34,
            y,
            "QUERY",
            [f"pid={query.pid}", f"cam=c{query.camid + 1}"],
            "#2563eb",
            card_size=(query_w, card_h),
            image_size=(112, 224),
        )
        x = 34 + query_w + gap
        for result in ranked[:topk]:
            border = "#047857" if result["correct"] else "#be123c"
            verdict = "MATCH" if result["correct"] else "NO MATCH"
            paste_card(
                canvas,
                result["path"],
                x,
                y,
                f"R{result['rank']}",
                [verdict, f"pid={result['pid']} c{result['camid'] + 1}", f"d={result['distance']:.3f}"],
                border,
                card_size=(card_w, card_h),
                image_size=(112, 224),
            )
            x += card_w + gap
        y += row_h

    path = output_dir / "04_multi_query_retrieval_grid.png"
    canvas.save(path)
    return path


def save_distance_bar_chart(query, ranked, output_dir, topk):
    top = ranked[:topk]
    width, height = 1080, 560
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), "Distance ranking for one query", font=get_font(30, True), fill="#1f2937")
    draw.text((32, 62), f"Query pid={query.pid}, cam=c{query.camid + 1}. Lower distance means more similar.", font=get_font(18), fill="#526071")

    q_img = fit_image(query.path, (120, 240))
    canvas.paste(q_img, (38, 132))
    draw.rounded_rectangle([30, 122, 170, 382], radius=10, outline="#2563eb", width=4)
    draw.text((48, 392), "QUERY", font=get_font(18, True), fill="#2563eb")

    chart_x, chart_y = 230, 130
    chart_w, chart_h = 780, 330
    max_d = max(row["distance"] for row in top) or 1.0
    bar_gap = 10
    bar_h = (chart_h - (len(top) - 1) * bar_gap) / len(top)
    for i, row in enumerate(top):
        y = chart_y + i * (bar_h + bar_gap)
        color = "#047857" if row["correct"] else "#be123c"
        bar_w = int((row["distance"] / max_d) * chart_w)
        draw.rounded_rectangle([chart_x, y, chart_x + bar_w, y + bar_h], radius=5, fill=color)
        label = f"Rank {row['rank']:02d} | pid={row['pid']} c{row['camid'] + 1} | dist={row['distance']:.3f}"
        draw.text((chart_x + 10, y + 6), label, font=get_font(16, True), fill="white")

    draw.text((230, 486), "Interpretation: model converts images to embeddings, then ranks gallery images by distance.", font=get_font(18), fill="#1f2937")
    path = output_dir / "05_distance_bar_chart.png"
    canvas.save(path)
    return path


def make_no_failure_image(output_dir):
    canvas = Image.new("RGB", (1000, 360), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 34), "Failure-case search", font=get_font(30, True), fill="#1f2937")
    draw.text((34, 86), "No Top-1 failure was found within the configured search budget.", font=get_font(22, True), fill="#047857")
    draw.text((34, 126), "Increase --failure-search or --failure-distractors to search harder.", font=get_font(18), fill="#526071")
    path = output_dir / "06_failure_cases.png"
    canvas.save(path)
    return path


def save_failure_cases(failures, output_dir):
    if not failures:
        return make_no_failure_image(output_dir)

    card_w, card_h = 190, 360
    gap = 28
    width = 920
    height = 100 + len(failures) * (card_h + 42) + 20
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 24), "Failure cases: wrong Top-1, correct match ranked lower", font=get_font(26, True), fill="#1f2937")
    draw.text((34, 58), "Useful for explaining model limitations.", font=get_font(18), fill="#526071")
    y = 100
    for query, wrong, best_correct in failures:
        paste_card(canvas, query.path, 40, y, "QUERY", [f"pid={query.pid}", f"cam=c{query.camid + 1}"], "#2563eb", card_size=(card_w, card_h))
        paste_card(
            canvas,
            wrong["path"],
            40 + card_w + gap,
            y,
            "WRONG TOP-1",
            [f"pid={wrong['pid']} c{wrong['camid'] + 1}", f"dist={wrong['distance']:.3f}"],
            "#be123c",
            card_size=(card_w, card_h),
        )
        paste_card(
            canvas,
            best_correct["path"],
            40 + 2 * (card_w + gap),
            y,
            f"TRUE AT R{best_correct['rank']}",
            [f"pid={best_correct['pid']} c{best_correct['camid'] + 1}", f"dist={best_correct['distance']:.3f}"],
            "#047857",
            card_size=(card_w, card_h),
        )
        y += card_h + 42
    path = output_dir / "06_failure_cases.png"
    canvas.save(path)
    return path


def find_failures(model, query_items, gallery_items, device, batch_size, search_count, distractors, seed, max_failures):
    queries = select_queries_with_gallery(query_items, gallery_items, search_count, seed + 999)
    query_pids = {query.pid for query in queries}

    positives = [
        item for item in gallery_items
        if item.pid in query_pids and any(query.pid == item.pid and query.camid != item.camid for query in queries)
    ]
    distractor_pool = [item for item in gallery_items if item.pid not in query_pids]
    rng = random.Random(seed + 12345)
    rng.shuffle(distractor_pool)
    candidate_gallery = positives + distractor_pool[:distractors]

    # De-duplicate while preserving order.
    seen = set()
    unique_gallery = []
    for item in candidate_gallery:
        if item.path in seen:
            continue
        seen.add(item.path)
        unique_gallery.append(item)

    query_features, _ = extract_features(model, queries, device, batch_size)
    gallery_features, _ = extract_features(model, unique_gallery, device, batch_size)
    distances = torch.cdist(query_features, gallery_features, p=2)

    failures = []
    for q_idx, query in enumerate(queries):
        order = torch.argsort(distances[q_idx]).tolist()
        ranked = []
        rank = 1
        for gallery_index in order:
            item = unique_gallery[gallery_index]
            if item.pid == query.pid and item.camid == query.camid:
                continue
            ranked.append({
                "rank": rank,
                "path": item.path,
                "pid": item.pid,
                "camid": item.camid,
                "distance": float(distances[q_idx, gallery_index].item()),
                "correct": item.pid == query.pid and item.camid != query.camid,
            })
            rank += 1
        if ranked and not ranked[0]["correct"]:
            correct_rows = [row for row in ranked if row["correct"]]
            if correct_rows:
                failures.append((query, ranked[0], correct_rows[0]))
        if len(failures) >= max_failures:
            break
    return failures


def pca_2d(features):
    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def save_embedding_pca(model, items, output_dir, device, batch_size, pid_count, images_per_pid, seed):
    rng = random.Random(seed)
    groups = group_valid_queries(items)
    rng.shuffle(groups)
    selected_groups = groups[:pid_count]
    selected = []
    labels = []
    for _, group in selected_groups:
        chosen = group[:]
        rng.shuffle(chosen)
        for item in chosen[:images_per_pid]:
            selected.append(item)
            labels.append(item.pid)

    features, _ = extract_features(model, selected, device, batch_size)
    points = pca_2d(features)
    min_xy = points.min(dim=0).values
    max_xy = points.max(dim=0).values
    denom = torch.clamp(max_xy - min_xy, min=1e-6)
    norm = (points - min_xy) / denom

    width, height = 980, 720
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "Feature embedding PCA", font=get_font(30, True), fill="#1f2937")
    draw.text((30, 62), "Each point is one image embedding. Same color = same person ID.", font=get_font(18), fill="#526071")
    plot = [70, 110, 760, 650]
    draw.rectangle(plot, outline="#cbd5e1", width=2)

    pid_to_color = {pid: COLORS[i % len(COLORS)] for i, (pid, _) in enumerate(selected_groups)}
    for point, item in zip(norm, selected):
        x = plot[0] + float(point[0]) * (plot[2] - plot[0])
        y = plot[3] - float(point[1]) * (plot[3] - plot[1])
        color = pid_to_color[item.pid]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color, outline="white", width=2)

    legend_x, legend_y = 790, 120
    draw.text((legend_x, legend_y - 38), "Legend", font=get_font(20, True), fill="#1f2937")
    for i, (pid, _) in enumerate(selected_groups):
        y = legend_y + i * 30
        color = pid_to_color[pid]
        draw.ellipse([legend_x, y, legend_x + 16, y + 16], fill=color)
        draw.text((legend_x + 26, y - 2), f"pid={pid}", font=get_font(16), fill="#1f2937")

    path = output_dir / "07_embedding_pca.png"
    canvas.save(path)
    return path


def reliability_for_image(model, item, device):
    if not hasattr(model, "patch_reliability_modeling") or not hasattr(model, "b1"):
        return None, None
    transform = make_transform()
    image = Image.open(item.path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    cam = torch.tensor([item.camid], dtype=torch.long, device=device)
    view = torch.tensor([1], dtype=torch.long, device=device)
    with torch.no_grad():
        features = model.base(tensor, cam_label=cam, view_label=view)
        b1_feat = model.b1(features)
        base_global_feat = b1_feat[:, 0]
        patch_tokens = b1_feat[:, 1:]
        _, aux = model.patch_reliability_modeling(base_global_feat, patch_tokens)
        scores = aux.get("patch_weights", aux["patch_reliability"])[0].detach().cpu()
    grid = getattr(model, "patch_grid", None)
    return scores, grid


def heat_color(value):
    value = max(0.0, min(1.0, float(value)))
    r = int(255 * (1.0 - value))
    g = int(180 * value + 40 * (1.0 - value))
    b = int(40 * (1.0 - value))
    return r, g, b


def resize_to_model_input(path):
    height, width = cfg.INPUT.SIZE_TEST
    image = Image.open(path).convert("RGB")
    return image.resize((int(width), int(height)), Image.Resampling.BILINEAR)


def save_reliability_heatmap(model, query, output_dir, device):
    scores, grid = reliability_for_image(model, query, device)
    if scores is None or grid is None:
        canvas = Image.new("RGB", (760, 330), "#f7f8fb")
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 30), "Reliability heatmap unavailable", font=get_font(24, True), fill="#be123c")
        draw.text((30, 70), "This checkpoint/model does not expose patch reliability scores.", font=get_font(17), fill="#526071")
        path = output_dir / "08_reliability_heatmap.png"
        canvas.save(path)
        return path

    h, w = grid
    grid_scores = scores.reshape(h, w)
    score_span = grid_scores.max() - grid_scores.min()
    is_uniform = float(score_span) < 1e-8
    if is_uniform:
        grid_scores = torch.full_like(grid_scores, 0.5)
    else:
        grid_scores = (grid_scores - grid_scores.min()) / (score_span + 1e-6)
    input_h, input_w = cfg.INPUT.SIZE_TEST
    display_size = (int(input_w), int(input_h))
    cell = 18
    heatmap = Image.new("RGB", (w * cell, h * cell), "white")
    draw_h = ImageDraw.Draw(heatmap)
    for yy in range(h):
        for xx in range(w):
            color = heat_color(grid_scores[yy, xx].item())
            draw_h.rectangle(
                [xx * cell, yy * cell, (xx + 1) * cell, (yy + 1) * cell],
                fill=color,
                outline="#ffffff",
            )
    heatmap = heatmap.resize(display_size, Image.Resampling.BILINEAR)

    original = resize_to_model_input(query.path)
    alpha = 0.22 if is_uniform else 0.45
    overlay = Image.blend(original, heatmap, alpha=alpha)
    width, height = 760, 330
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    x_positions = [72, 316, 560]
    labels = ["Original", "Reliability map", "Aligned overlay"]
    images = [original, heatmap, overlay]
    for x, label, image in zip(x_positions, labels, images):
        draw.rounded_rectangle([x - 14, 18, x + display_size[0] + 14, 292], radius=10, fill="#ffffff", outline="#cbd5e1", width=2)
        canvas.paste(image, (x, 30))
        draw.text((x, 296), label, font=get_font(17, True), fill="#1f2937")
    if is_uniform:
        draw.text((318, 52), "uniform", font=get_font(15, True), fill="#1f2937")
    path = output_dir / "08_reliability_heatmap.png"
    canvas.save(path)
    return path


def build_synthetic_occlusion(image, seed=2026):
    rng = random.Random(seed)
    occ = image.copy()
    width, height = occ.size
    area = width * height
    target_area = rng.uniform(float(cfg.INPUT.REL_OCC_MIN_AREA), float(cfg.INPUT.REL_OCC_MAX_AREA)) * area
    aspect = rng.uniform(float(cfg.INPUT.REL_OCC_MIN_ASPECT), 1.0 / float(cfg.INPUT.REL_OCC_MIN_ASPECT))
    occ_h = max(24, min(height - 1, int(round(math.sqrt(target_area * aspect)))))
    occ_w = max(18, min(width - 1, int(round(math.sqrt(target_area / aspect)))))
    left = rng.randint(0, max(0, width - occ_w))
    top = rng.randint(0, max(0, height - occ_h))
    draw = ImageDraw.Draw(occ)
    draw.rectangle([left, top, left + occ_w, top + occ_h], fill=(0, 0, 0))
    return occ, (left, top, occ_w, occ_h)


def save_occlusion_target_visual(query, output_dir, seed=2026):
    original = fit_image(query.path, (128, 256))
    occluded, rect = build_synthetic_occlusion(original, seed)
    left, top, occ_w, occ_h = rect
    grid_h, grid_w = 21, 10
    cell_w, cell_h = 128 / grid_w, 256 / grid_h
    mask = Image.new("RGB", (128, 256), "#16a34a")
    draw_m = ImageDraw.Draw(mask)
    for yy in range(grid_h):
        for xx in range(grid_w):
            x1, y1 = xx * cell_w, yy * cell_h
            x2, y2 = (xx + 1) * cell_w, (yy + 1) * cell_h
            intersects = not (x2 < left or x1 > left + occ_w or y2 < top or y1 > top + occ_h)
            color = "#dc2626" if intersects else "#16a34a"
            draw_m.rectangle([x1, y1, x2, y2], fill=color, outline="#ffffff")
    overlay = Image.blend(occluded, mask, alpha=0.35)

    width, height = 930, 455
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "Synthetic occlusion target for reliability training", font=get_font(28, True), fill="#1f2937")
    draw.text((30, 60), "Red patches are supervised as occluded; green patches remain visible.", font=get_font(18), fill="#526071")
    x_positions = [70, 285, 500, 715]
    labels = ["Original", "Synthetic occlusion", "Patch target", "Overlay"]
    images = [original, occluded, mask, overlay]
    for x, label, image in zip(x_positions, labels, images):
        canvas.paste(image, (x, 122))
        draw.rounded_rectangle([x - 8, 114, x + 136, 386], radius=10, outline="#cbd5e1", width=2)
        draw.text((x, 398), label, font=get_font(17, True), fill="#1f2937")
    path = output_dir / "08_synthetic_occlusion_target.png"
    canvas.save(path)
    return path


def write_suite_summary(output_dir, query_results, failures, paths):
    path = output_dir / "visualization_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "detail", "path"])
        for output_path in paths:
            writer.writerow(["image", output_path.name, str(output_path)])
        for query, ranked in query_results:
            top1 = ranked[0]
            writer.writerow([
                "query_top1",
                f"query={query.path.name}; top1={Path(top1['path']).name}; correct={top1['correct']}; distance={top1['distance']:.6f}",
                "",
            ])
        writer.writerow(["failure_count", str(len(failures)), ""])
    return path


def main():
    parser = argparse.ArgumentParser(description="Create additional Person Re-ID visualizations.")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT.parent / "results" / "best_checkpoint.pth"))
    parser.add_argument("--config-file", default=None, help="Optional TransReID config file used when the checkpoint is a raw state_dict.")
    parser.add_argument("--dataset-root", default=r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\Requirement 2 & 3\Colab\data\market1501")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "results" / "visual_reid_suite"))
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-distractors", type=int, default=40)
    parser.add_argument("--failure-search", type=int, default=20)
    parser.add_argument("--failure-distractors", type=int, default=250)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--pca-pids", type=int, default=6)
    parser.add_argument("--pca-images-per-pid", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    items, gallery_dir = load_market_gallery(args.dataset_root)
    try:
        query_items, query_dir = load_market_split(args.dataset_root, "query")
    except FileNotFoundError:
        query_items, query_dir = items, gallery_dir
    print(f"Gallery folder: {gallery_dir}")
    print(f"Query folder: {query_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    if args.config_file:
        cfg.defrost()
        cfg.merge_from_file(args.config_file)
        cfg.MODEL.PRETRAIN_CHOICE = "none"
        cfg.MODEL.PRETRAIN_PATH = ""
        cfg.MODEL.DEVICE_ID = "0"
        cfg.freeze()
        print(f"Config: {args.config_file}")
    print(f"Device: {device}")

    model, _ = build_model_from_checkpoint(args.checkpoint, device)

    queries = select_queries_with_gallery(query_items, items, args.num_queries, args.seed)
    query_results = []
    for idx, query in enumerate(queries):
        gallery_items, positives = build_demo_gallery(items, query, args.num_distractors, args.seed + idx)
        if not positives:
            continue
        ranked = rank_gallery(model, query, gallery_items, device, args.batch_size)
        query_results.append((query, ranked))
        top1 = ranked[0]
        print(f"Query {idx + 1}: pid={query.pid} c{query.camid + 1} -> Top1 pid={top1['pid']} c{top1['camid'] + 1} correct={top1['correct']}")

    if not query_results:
        raise RuntimeError("No query result could be generated.")

    paths = []
    paths.append(save_multi_query_grid(query_results, output_dir, args.topk))
    paths.append(save_distance_bar_chart(query_results[0][0], query_results[0][1], output_dir, max(10, args.topk)))

    failures = find_failures(
        model,
        query_items,
        items,
        device,
        args.batch_size,
        args.failure_search,
        args.failure_distractors,
        args.seed,
        args.max_failures,
    )
    paths.append(save_failure_cases(failures, output_dir))
    paths.append(save_embedding_pca(model, items, output_dir, device, args.batch_size, args.pca_pids, args.pca_images_per_pid, args.seed))
    paths.append(save_reliability_heatmap(model, query_results[0][0], output_dir, device))
    paths.append(save_occlusion_target_visual(query_results[0][0], output_dir, args.seed))
    summary_path = write_suite_summary(output_dir, query_results, failures, paths)

    print("\nSaved visualizations:")
    for path in paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
