from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tools" else Path(__file__).resolve().parent
OUT = ROOT / "visualizations" / "quantitative"

MODEL_ORDER = [
    "market_baseline",
    "market_sem_align",
    "market_local_reliability",
    "market_sem_align_reliability",
]

LABELS = {
    "market_baseline": "Baseline",
    "market_sem_align": "Semantic",
    "market_local_reliability": "Local Rel.",
    "market_sem_align_reliability": "Semantic+Rel.",
}

TRAIN_LOGS = {
    "market_baseline": ROOT / "runs" / "fair_market_baseline" / "train_log.txt",
    "market_sem_align": ROOT / "runs" / "fair_market_sem_align" / "train_log.txt",
    "market_local_reliability": ROOT / "runs" / "fair_market_local_reliability" / "train_log.txt",
    "market_sem_align_reliability": ROOT / "runs" / "fair_market_sem_align_reliability" / "train_log.txt",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def parse_train_log(path: Path) -> dict[str, dict[int, float]]:
    iter_re = re.compile(
        r"Epoch\[(?P<epoch>\d+)\] Iteration\[\d+/\d+\] "
        r"Loss: (?P<loss>[0-9.]+), Acc: (?P<acc>[0-9.]+), Base Lr: (?P<lr>[0-9.eE+-]+)"
    )
    done_re = re.compile(
        r"Epoch (?P<epoch>\d+) done\. Time per batch: (?P<time>[0-9.]+)\[s\] "
        r"Speed: (?P<speed>[0-9.]+)\[samples/s\]"
    )
    val_epoch_re = re.compile(r"Validation Results - Epoch: (?P<epoch>\d+)")
    map_re = re.compile(r"mAP: (?P<value>[0-9.]+)%")
    rank_re = re.compile(r"CMC curve, Rank-(?P<rank>\d+)\s+:(?P<value>[0-9.]+)%")

    loss_by_epoch: dict[int, list[float]] = defaultdict(list)
    acc_by_epoch: dict[int, list[float]] = defaultdict(list)
    lr_by_epoch: dict[int, list[float]] = defaultdict(list)
    speed: dict[int, float] = {}
    time_per_batch: dict[int, float] = {}
    val_map: dict[int, float] = {}
    val_rank1: dict[int, float] = {}
    val_rank5: dict[int, float] = {}
    val_rank10: dict[int, float] = {}

    current_val_epoch: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = iter_re.search(line)
            if m:
                epoch = int(m.group("epoch"))
                loss_by_epoch[epoch].append(float(m.group("loss")))
                acc_by_epoch[epoch].append(float(m.group("acc")) * 100.0)
                lr_by_epoch[epoch].append(float(m.group("lr")))
                continue

            m = done_re.search(line)
            if m:
                epoch = int(m.group("epoch"))
                time_per_batch[epoch] = float(m.group("time"))
                speed[epoch] = float(m.group("speed"))
                continue

            m = val_epoch_re.search(line)
            if m:
                current_val_epoch = int(m.group("epoch"))
                continue

            if current_val_epoch is not None:
                m = map_re.search(line)
                if m:
                    val_map[current_val_epoch] = float(m.group("value"))
                    continue
                m = rank_re.search(line)
                if m:
                    value = float(m.group("value"))
                    rank = int(m.group("rank"))
                    if rank == 1:
                        val_rank1[current_val_epoch] = value
                    elif rank == 5:
                        val_rank5[current_val_epoch] = value
                    elif rank == 10:
                        val_rank10[current_val_epoch] = value

    def mean_dict(values: dict[int, list[float]]) -> dict[int, float]:
        return {k: float(np.mean(v)) for k, v in values.items()}

    return {
        "loss": mean_dict(loss_by_epoch),
        "acc": mean_dict(acc_by_epoch),
        "lr": mean_dict(lr_by_epoch),
        "speed": speed,
        "time_per_batch": time_per_batch,
        "val_map": val_map,
        "val_rank1": val_rank1,
        "val_rank5": val_rank5,
        "val_rank10": val_rank10,
    }


def plot_metric_bars(metrics: list[dict[str, str]]) -> None:
    data = {row["name"]: row for row in metrics}
    metric_names = ["mAP", "rank1", "rank5", "rank10"]
    x = np.arange(len(metric_names))
    width = 0.19

    plt.figure(figsize=(11, 5.6))
    for i, name in enumerate(MODEL_ORDER):
        values = [as_float(data[name], m) for m in metric_names]
        bars = plt.bar(x + (i - 1.5) * width, values, width, label=LABELS[name])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.1f}", ha="center", fontsize=8)

    plt.xticks(x, ["mAP", "Rank-1", "Rank-5", "Rank-10"])
    plt.ylabel("Score (%)")
    plt.ylim(88, 100)
    plt.title("Market1501 evaluation: baseline vs proposed branches")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig(OUT / "01_metrics_grouped_bar.png", dpi=200)
    plt.close()


def plot_delta(metrics: list[dict[str, str]]) -> None:
    data = {row["name"]: row for row in metrics}
    baseline = data["market_baseline"]
    metric_names = ["mAP", "rank1", "rank5", "rank10"]
    x = np.arange(len(metric_names))
    width = 0.25
    names = MODEL_ORDER[1:]

    plt.figure(figsize=(10.5, 5.2))
    for i, name in enumerate(names):
        values = [as_float(data[name], m) - as_float(baseline, m) for m in metric_names]
        bars = plt.bar(x + (i - 1) * width, values, width, label=LABELS[name])
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.025 if value >= 0 else -0.08),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, ["mAP", "Rank-1", "Rank-5", "Rank-10"])
    plt.ylabel("Delta vs baseline (percentage point)")
    plt.title("Improvement over baseline")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig(OUT / "02_delta_vs_baseline.png", dpi=200)
    plt.close()


def plot_latency_tradeoff(metrics: list[dict[str, str]], latency: list[dict[str, str]]) -> None:
    metric_data = {row["name"]: row for row in metrics}
    latency_data = {row["name"]: row for row in latency}

    plt.figure(figsize=(8.5, 5.8))
    for name in MODEL_ORDER:
        mean_ms = as_float(latency_data[name], "mean_ms")
        map_score = as_float(metric_data[name], "mAP")
        rank1 = as_float(metric_data[name], "rank1")
        plt.scatter(mean_ms, map_score, s=180 + (rank1 - 95.0) * 450, label=LABELS[name])
        plt.text(mean_ms + 0.006, map_score + 0.02, f"{LABELS[name]}\nR1 {rank1:.1f}", fontsize=9)

    plt.xlabel("Mean inference latency per image (ms, batch=1)")
    plt.ylabel("mAP (%)")
    plt.title("Accuracy-latency trade-off")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT / "03_accuracy_latency_tradeoff.png", dpi=200)
    plt.close()


def plot_train_curves(parsed_logs: dict[str, dict[str, dict[int, float]]]) -> None:
    plt.figure(figsize=(11, 5.8))
    for name in MODEL_ORDER:
        curve = parsed_logs[name]["loss"]
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        plt.plot(xs, ys, label=LABELS[name], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean training loss")
    plt.title("Training loss curves")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout()
    plt.savefig(OUT / "04_training_loss_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    for name in MODEL_ORDER:
        curve = parsed_logs[name]["acc"]
        xs = sorted(curve)
        ys = [curve[x] for x in xs]
        plt.plot(xs, ys, label=LABELS[name], linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Mean training accuracy (%)")
    plt.title("Training accuracy curves")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    plt.tight_layout()
    plt.savefig(OUT / "05_training_accuracy_curves.png", dpi=200)
    plt.close()


def plot_validation_curves(parsed_logs: dict[str, dict[str, dict[int, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for name in MODEL_ORDER:
        map_curve = parsed_logs[name]["val_map"]
        r1_curve = parsed_logs[name]["val_rank1"]
        xs = sorted(map_curve)
        axes[0].plot(xs, [map_curve[x] for x in xs], marker="o", linewidth=2, label=LABELS[name])
        xs_r1 = sorted(r1_curve)
        axes[1].plot(xs_r1, [r1_curve[x] for x in xs_r1], marker="o", linewidth=2, label=LABELS[name])

    axes[0].set_title("Validation mAP")
    axes[0].set_ylabel("mAP (%)")
    axes[1].set_title("Validation Rank-1")
    axes[1].set_ylabel("Rank-1 (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()
    plt.savefig(OUT / "06_validation_map_rank1_curves.png", dpi=200)
    plt.close()


def plot_latency_distribution(latency_rows: list[dict[str, str]]) -> None:
    rows = {row["name"]: row for row in latency_rows}
    raw_values = []
    labels = []
    for name in MODEL_ORDER:
        raw_path = rows[name].get("raw_path", "")
        local_raw = ROOT / "latency" / Path(raw_path).name
        if local_raw.exists():
            raw_values.append(np.load(local_raw))
            labels.append(LABELS[name])

    if raw_values:
        plt.figure(figsize=(10, 5.4))
        plt.boxplot(raw_values, tick_labels=labels, showfliers=False)
        plt.ylabel("Latency (ms)")
        plt.title("Inference latency distribution")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(OUT / "07_latency_distribution_boxplot.png", dpi=200)
        plt.close()

    x = np.arange(len(MODEL_ORDER))
    mean = [as_float(rows[name], "mean_ms") for name in MODEL_ORDER]
    p95 = [as_float(rows[name], "p95_ms") for name in MODEL_ORDER]
    plt.figure(figsize=(9.5, 5.2))
    plt.bar(x - 0.18, mean, width=0.36, label="Mean")
    plt.bar(x + 0.18, p95, width=0.36, label="P95")
    plt.xticks(x, [LABELS[name] for name in MODEL_ORDER])
    plt.ylabel("Latency (ms)")
    plt.title("Mean and P95 inference latency")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "08_latency_mean_p95_bar.png", dpi=200)
    plt.close()


def plot_summary_table(metrics: list[dict[str, str]], latency: list[dict[str, str]]) -> list[dict[str, float | str]]:
    metric_data = {row["name"]: row for row in metrics}
    latency_data = {row["name"]: row for row in latency}
    baseline = metric_data["market_baseline"]
    baseline_latency = latency_data["market_baseline"]

    summary: list[dict[str, float | str]] = []
    table_rows = []
    for name in MODEL_ORDER:
        row = metric_data[name]
        lat = latency_data[name]
        item = {
            "model": LABELS[name],
            "mAP": as_float(row, "mAP"),
            "rank1": as_float(row, "rank1"),
            "rank5": as_float(row, "rank5"),
            "rank10": as_float(row, "rank10"),
            "mean_ms": as_float(lat, "mean_ms"),
            "delta_mAP": as_float(row, "mAP") - as_float(baseline, "mAP"),
            "delta_rank1": as_float(row, "rank1") - as_float(baseline, "rank1"),
            "delta_latency_ms": as_float(lat, "mean_ms") - as_float(baseline_latency, "mean_ms"),
        }
        summary.append(item)
        table_rows.append(
            [
                item["model"],
                f"{item['mAP']:.1f}",
                f"{item['rank1']:.1f}",
                f"{item['rank5']:.1f}",
                f"{item['rank10']:.1f}",
                f"{item['mean_ms']:.3f}",
                f"{item['delta_mAP']:+.1f}",
                f"{item['delta_latency_ms']:+.3f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Model", "mAP", "R1", "R5", "R10", "ms/img", "Delta mAP", "Delta ms"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1E5A8A")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F4F7")
    plt.title("Market1501 summary", pad=18)
    plt.tight_layout()
    plt.savefig(OUT / "09_summary_table.png", dpi=200)
    plt.close()

    return summary


def write_summary(summary: list[dict[str, float | str]]) -> None:
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# TransReID artifact visualization summary",
        "",
        "| Model | mAP | Rank-1 | Rank-5 | Rank-10 | Mean ms/img | Delta mAP | Delta Rank-1 | Delta ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {mAP:.1f} | {rank1:.1f} | {rank5:.1f} | {rank10:.1f} | "
            "{mean_ms:.3f} | {delta_mAP:+.1f} | {delta_rank1:+.1f} | {delta_latency_ms:+.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Suggested interpretation:",
            "- Semantic+Rel. has the best mAP and Rank-1 in this artifact set.",
            "- Local Rel. improves mAP more strongly than Rank-1, which is useful to discuss ranking quality.",
            "- The added branches introduce small latency overhead, so the report should mention the accuracy-latency trade-off.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(ROOT / "metrics_summary.csv")
    latency = read_csv(ROOT / "latency_summary.csv")
    parsed_logs = {name: parse_train_log(path) for name, path in TRAIN_LOGS.items()}

    plot_metric_bars(metrics)
    plot_delta(metrics)
    plot_latency_tradeoff(metrics, latency)
    plot_train_curves(parsed_logs)
    plot_validation_curves(parsed_logs)
    plot_latency_distribution(latency)
    summary = plot_summary_table(metrics, latency)
    write_summary(summary)

    print(f"Wrote report visuals to: {OUT}")
    for path in sorted(OUT.glob("*")):
        print(path.name)


if __name__ == "__main__":
    main()
