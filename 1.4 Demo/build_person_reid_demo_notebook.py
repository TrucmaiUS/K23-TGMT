from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\K23-TGMT")
DEMO_DIR = PROJECT_ROOT / "1.4 Demo"
NOTEBOOKS = [
    DEMO_DIR / "person_reid_3_branches_full_demo.ipynb",
    DEMO_DIR / "dual_approach_demo_full.ipynb",
]


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

cells = []

cells.append(md("""
# Demo Person Re-Identification

**TransReID, Local Reliability, Semantic Alignment và hướng kết hợp**

Notebook này dùng để demo **quy trình chạy thực nghiệm**, không chỉ để xem ảnh có sẵn.

Mạch chạy chính:

1. Thiết lập đường dẫn, dataset, checkpoint và cấu hình.
2. Kiểm tra môi trường và dữ liệu Market-1501.
3. Chạy đánh giá từ checkpoint đã huấn luyện.
4. Chạy visualization cho retrieval, reliability, semantic mask và so sánh nhiều mô hình.
5. Xem kết quả định lượng và định tính ngay trong notebook.

Các cell train được cung cấp đầy đủ nhưng mặc định không chạy, vì train đầy đủ 120 epoch tốn nhiều thời gian. Khi thuyết trình/demo, nên dùng checkpoint đã có để chạy evaluation và visualization.
"""))

cells.append(md("""
## 0. Cấu hình điều khiển demo

Các biến `RUN_*` quyết định notebook có thực sự chạy lệnh nặng hay chỉ in lệnh và hiển thị artifact đã có.

- `RUN_EVAL = True`: chạy `test.py` cho các checkpoint.
- `RUN_VISUALIZE = True`: chạy lại các script tạo ảnh visualization.
- `RUN_TRAIN = True`: chạy train thử. Mặc định tắt vì rất tốn thời gian.
- `FAST_TRAIN_DEMO = True`: nếu bật train, chỉ chạy 1 epoch để kiểm tra pipeline, không dùng làm kết quả chính thức.
"""))

cells.append(code(r"""
from pathlib import Path
import os
import sys
import csv
import json
import shlex
import subprocess
from datetime import datetime

from IPython.display import display, Markdown, Image as IPyImage

PROJECT_ROOT = Path(r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\K23-TGMT")
SOURCE_ROOT = PROJECT_ROOT / "1.3 Source code"
DEMO_ROOT = PROJECT_ROOT / "1.4 Demo"
ARTIFACT_ROOT = SOURCE_ROOT / "transreid_artifacts"

LOCAL_SRC = SOURCE_ROOT / "local-reliability"
SEMANTIC_SRC = SOURCE_ROOT / "semantic"
TOOLS_DIR = ARTIFACT_ROOT / "tools"

DATASET_ROOT = Path(r"D:\DAI_HOC\NAM_3\SEM 2\TGMT\PROJECT\Requirement 2 & 3\Colab\data\market1501")
DATA_PARENT = DATASET_ROOT.parent

DEMO_OUTPUT = DEMO_ROOT / "generated_outputs"
DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

RUN_INSTALL = False
RUN_TRAIN = False
FAST_TRAIN_DEMO = True
RUN_EVAL = False
RUN_VISUALIZE = False

DEVICE_ID = "0"
PYTHON = sys.executable

print("Project:", PROJECT_ROOT)
print("Dataset exists:", DATASET_ROOT.exists(), DATASET_ROOT)
print("Artifacts:", ARTIFACT_ROOT)
"""))

cells.append(md("""
## 1. Khai báo 4 mô hình trong thí nghiệm

Ba hướng mở rộng được demo là:

1. **Local Reliability**: điều chỉnh đóng góp của patch cục bộ theo độ tin cậy.
2. **Semantic Alignment**: gán semantic body-part cho patch token.
3. **Semantic + Reliability**: kết hợp hai tín hiệu trên.

Baseline TransReID được giữ để so sánh.
"""))

cells.append(code(r"""
BRANCHES = {
    "baseline": {
        "title": "Baseline TransReID",
        "source": SEMANTIC_SRC,
        "config": ARTIFACT_ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_fair.yml",
        "checkpoint": ARTIFACT_ROOT / "runs" / "fair_market_baseline" / "transformer_best.pth",
        "output": DEMO_OUTPUT / "eval_baseline",
        "note": "Mô hình nền TransReID.",
    },
    "local": {
        "title": "Local Reliability",
        "source": LOCAL_SRC,
        "config": ARTIFACT_ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_local_reliability.yml",
        "checkpoint": ARTIFACT_ROOT / "runs" / "fair_market_local_reliability" / "transformer_best.pth",
        "output": DEMO_OUTPUT / "eval_local_reliability",
        "note": "Ước lượng reliability score cho patch/local token.",
    },
    "semantic": {
        "title": "Semantic Alignment",
        "source": SEMANTIC_SRC,
        "config": ARTIFACT_ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_sem_align.yml",
        "checkpoint": ARTIFACT_ROOT / "runs" / "fair_market_sem_align" / "transformer_best.pth",
        "output": DEMO_OUTPUT / "eval_semantic_alignment",
        "note": "Dùng semantic mask để giám sát patch token.",
    },
    "combine": {
        "title": "Semantic + Reliability",
        "source": SEMANTIC_SRC,
        "config": ARTIFACT_ROOT / "configs_snapshot" / "Market" / "vit_transreid_stride_sem_align_reliability.yml",
        "checkpoint": ARTIFACT_ROOT / "runs" / "fair_market_sem_align_reliability" / "transformer_best.pth",
        "output": DEMO_OUTPUT / "eval_semantic_reliability",
        "note": "Kết hợp semantic supervision và reliability-aware local fusion.",
    },
}

rows = []
for key, branch in BRANCHES.items():
    ckpt = branch["checkpoint"]
    cfg = branch["config"]
    rows.append({
        "key": key,
        "model": branch["title"],
        "source": str(branch["source"]),
        "config_exists": cfg.exists(),
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_size_MB": round(ckpt.stat().st_size / 1024**2, 2) if ckpt.exists() else None,
        "note": branch["note"],
    })

try:
    import pandas as pd
    display(pd.DataFrame(rows))
except Exception:
    for row in rows:
        print(row)
"""))

cells.append(md("""
## 2. Kiểm tra môi trường và dataset

Cell này kiểm tra phiên bản Python/PyTorch, CUDA và số ảnh trong các split Market-1501. Nếu dataset không đúng cấu trúc, các bước train/test sẽ lỗi.
"""))

cells.append(code(r"""
def count_images(split_name):
    split_dir = DATASET_ROOT / split_name
    if not split_dir.exists():
        return 0
    return len(list(split_dir.glob("*.jpg")))

print("Python:", sys.version)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch check failed:", exc)

for split in ["bounding_box_train", "query", "bounding_box_test"]:
    print(f"{split:20s}", count_images(split))
"""))

cells.append(md("""
## 3. Hàm tiện ích để chạy lệnh và hiển thị kết quả

Các lệnh train/test/visualize được chạy bằng `subprocess`. Notebook giữ nguyên working directory theo từng nhánh để import module nội bộ đúng.
"""))

cells.append(code(r"""
def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)

def run_command(cmd, cwd, run=False, env=None):
    print("CWD:", cwd)
    print("CMD:", quote_cmd(cmd))
    if not run:
        print("Skip: đặt cờ RUN_* = True để chạy lệnh này.")
        return None
    env_all = os.environ.copy()
    if env:
        env_all.update(env)
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env_all,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout[-6000:])
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}")
    return proc

def show_image(path, width=1000):
    path = Path(path)
    if not path.exists():
        display(Markdown(f"**Chưa có ảnh:** `{path}`"))
        return
    display(Markdown(f"`{path}`"))
    display(IPyImage(filename=str(path), width=width))

def show_csv(path):
    path = Path(path)
    if not path.exists():
        print("Missing:", path)
        return
    try:
        import pandas as pd
        display(pd.read_csv(path))
    except Exception:
        print(path.read_text(encoding="utf-8-sig"))
"""))

cells.append(md("""
## 4. Setup môi trường

Thông thường chỉ cần chạy phần kiểm tra môi trường ở trên. Nếu thiếu package, bật `RUN_INSTALL = True` để cài requirements của hai nhánh.
"""))

cells.append(code(r"""
install_commands = [
    [PYTHON, "-m", "pip", "install", "-r", str(LOCAL_SRC / "requirements.txt")],
    [PYTHON, "-m", "pip", "install", "-r", str(SEMANTIC_SRC / "requirements.txt")],
]

for cmd in install_commands:
    run_command(cmd, cwd=PROJECT_ROOT, run=RUN_INSTALL)
"""))

cells.append(md("""
## 5. Pipeline train

Đây là các lệnh train tương ứng với từng nhánh. Mặc định notebook **không chạy train** vì train đầy đủ 120 epoch rất lâu.

Khi muốn kiểm tra pipeline train nhanh, bật:

```python
RUN_TRAIN = True
FAST_TRAIN_DEMO = True
```

Chế độ fast demo chỉ chạy 1 epoch, batch nhỏ, dùng để kiểm tra code chạy được; không dùng làm kết quả báo cáo.
"""))

cells.append(code(r"""
def build_train_command(key, fast_demo=True):
    branch = BRANCHES[key]
    output_dir = DEMO_OUTPUT / f"train_{key}"
    cmd = [
        PYTHON, "train.py",
        "--config_file", branch["config"],
        "DATASETS.ROOT_DIR", str(DATA_PARENT),
        "MODEL.DEVICE_ID", DEVICE_ID,
        "OUTPUT_DIR", str(output_dir),
    ]
    if fast_demo:
        cmd += [
            "MODEL.PRETRAIN_CHOICE", "none",
            "SOLVER.MAX_EPOCHS", "1",
            "SOLVER.EVAL_PERIOD", "1",
            "SOLVER.CHECKPOINT_PERIOD", "1",
            "SOLVER.IMS_PER_BATCH", "8",
            "TEST.IMS_PER_BATCH", "64",
            "DATALOADER.NUM_WORKERS", "0",
        ]
    return cmd

for key in ["baseline", "local", "semantic", "combine"]:
    display(Markdown(f"### Train: {BRANCHES[key]['title']}"))
    cmd = build_train_command(key, fast_demo=FAST_TRAIN_DEMO)
    run_command(cmd, cwd=BRANCHES[key]["source"], run=RUN_TRAIN)
"""))

cells.append(md("""
## 6. Pipeline validation/test từ checkpoint

Đây là bước nên chạy khi demo. Notebook dùng checkpoint đã huấn luyện và gọi `test.py` để tính mAP, Rank-1, Rank-5, Rank-10.

Bật `RUN_EVAL = True` để chạy lại đánh giá. Kết quả sẽ được ghi vào thư mục `1.4 Demo/generated_outputs/eval_*`.
"""))

cells.append(code(r"""
def build_eval_command(key):
    branch = BRANCHES[key]
    cmd = [
        PYTHON, "test.py",
        "--config_file", branch["config"],
        "DATASETS.ROOT_DIR", str(DATA_PARENT),
        "MODEL.DEVICE_ID", DEVICE_ID,
        "TEST.WEIGHT", str(branch["checkpoint"]),
        "TEST.IMS_PER_BATCH", "128",
        "DATALOADER.NUM_WORKERS", "0",
        "OUTPUT_DIR", str(branch["output"]),
    ]
    return cmd

for key in ["baseline", "local", "semantic", "combine"]:
    display(Markdown(f"### Evaluate: {BRANCHES[key]['title']}"))
    cmd = build_eval_command(key)
    run_command(cmd, cwd=BRANCHES[key]["source"], run=RUN_EVAL)
"""))

cells.append(md("""
## 7. Xem nhanh metric đã tổng hợp

Nếu chưa chạy lại evaluation trong notebook, cell này vẫn đọc bảng metric đã tổng hợp từ artifact hiện có.
"""))

cells.append(code(r"""
show_csv(ARTIFACT_ROOT / "metrics_summary.csv")
show_csv(ARTIFACT_ROOT / "latency_summary.csv")
"""))

cells.append(md("""
## 8. Pipeline visualization định lượng

Script `generate_report_visuals.py` đọc log/CSV đã có và sinh biểu đồ:

- mAP và CMC Rank-k.
- Delta so với baseline.
- Training/validation curves.
- Accuracy-latency trade-off.

Bật `RUN_VISUALIZE = True` để chạy lại.
"""))

cells.append(code(r"""
cmd = [PYTHON, str(TOOLS_DIR / "generate_report_visuals.py")]
run_command(cmd, cwd=ARTIFACT_ROOT, run=RUN_VISUALIZE)

quant_dir = ARTIFACT_ROOT / "visualizations" / "quantitative"
show_image(quant_dir / "09_summary_table.png", width=1000)
show_image(quant_dir / "01_metrics_grouped_bar.png", width=1000)
show_image(quant_dir / "02_delta_vs_baseline.png", width=900)
show_image(quant_dir / "06_validation_map_rank1_curves.png", width=1000)
show_image(quant_dir / "03_accuracy_latency_tradeoff.png", width=900)
"""))

cells.append(md("""
## 9. Visualization so sánh Top-K giữa 4 mô hình

Script này load cả 4 checkpoint, chọn cùng query/gallery và so sánh ranking Top-K của từng mô hình.

Đây là phần quan trọng nhất để demo trực quan bài toán Re-ID: query ở bên trái, gallery được sắp xếp theo khoảng cách embedding.
"""))

cells.append(code(r"""
cmd = [
    PYTHON, str(TOOLS_DIR / "visualize_multimodel_retrieval.py"),
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", str(ARTIFACT_ROOT / "visualizations" / "qualitative"),
    "--topk", "5",
    "--num-queries", "3",
    "--num-distractors", "60",
    "--batch-size", "32",
]
run_command(cmd, cwd=ARTIFACT_ROOT, run=RUN_VISUALIZE)

qual_dir = ARTIFACT_ROOT / "visualizations" / "qualitative"
show_image(qual_dir / "01_same_query_topk_comparison.png", width=950)
show_image(qual_dir / "02_true_match_rank_matrix.png", width=1000)
"""))

cells.append(md("""
## 10. Visualization riêng cho Local Reliability

Phần này chạy checkpoint Local Reliability để tạo:

- Top-K retrieval.
- Multi-query retrieval grid.
- Distance chart.
- Failure cases.
- PCA embedding.
- Reliability heatmap.
- Synthetic occlusion target.

Script đã được truyền đúng `--config-file` để checkpoint `.pth` được dựng bằng kiến trúc Local Reliability, không bị fallback sang backbone khác.
"""))

cells.append(code(r"""
local_out = ARTIFACT_ROOT / "visualizations" / "local_reliability"
cmd = [
    PYTHON, str(LOCAL_SRC / "visualize_person_reid_suite.py"),
    "--checkpoint", str(BRANCHES["local"]["checkpoint"]),
    "--config-file", str(BRANCHES["local"]["config"]),
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", str(local_out),
    "--num-queries", "3",
    "--topk", "5",
    "--batch-size", "16",
]
run_command(cmd, cwd=LOCAL_SRC, run=RUN_VISUALIZE)

show_image(local_out / "01_ranked_topk_results.png", width=1000)
show_image(local_out / "03_multi_query_retrieval_grid.png", width=1000)
show_image(local_out / "05_failure_cases.png", width=850)
show_image(local_out / "07_reliability_heatmap.png", width=900)
show_image(local_out / "08_synthetic_occlusion_target.png", width=950)
show_image(local_out / "06_embedding_pca.png", width=800)
"""))

cells.append(md("""
## 11. Visualization cho Semantic Alignment

Phần này minh họa pipeline semantic:

- Sinh/đọc human parsing mask.
- Gom nhãn parser thành nhóm semantic cấp cao.
- Tạo semantic patch target.
- So sánh human parsing mask với mask heuristic.
- Quan sát tính nhất quán semantic giữa các camera.

Một số script human parser có thể cần model cache hoặc internet nếu chạy lại từ đầu.
"""))

cells.append(code(r"""
sem_out = ARTIFACT_ROOT / "visualizations" / "semantic"

cmd_basic = [
    PYTHON, str(SEMANTIC_SRC / "visualize_semantic_no_checkpoint.py"),
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", str(sem_out),
    "--num-samples", "3",
]
run_command(cmd_basic, cwd=SEMANTIC_SRC, run=RUN_VISUALIZE)

# Hai script dưới đây dùng human parser thật. Nếu máy chưa cache model, lần đầu có thể cần tải model.
cmd_real = [
    PYTHON, str(SEMANTIC_SRC / "visualize_real_semantic_examples.py"),
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", str(sem_out),
    "--num-images", "2",
]
cmd_extra = [
    PYTHON, str(SEMANTIC_SRC / "visualize_semantic_extra.py"),
    "--dataset-root", str(DATASET_ROOT),
    "--output-dir", str(sem_out),
]

run_command(cmd_real, cwd=SEMANTIC_SRC, run=RUN_VISUALIZE)
run_command(cmd_extra, cwd=SEMANTIC_SRC, run=RUN_VISUALIZE)

show_image(sem_out / "01_real_semantic_examples.png", width=1000)
show_image(sem_out / "02_real_vs_heuristic_mask.png", width=1000)
show_image(sem_out / "03_local_group_patch_targets.png", width=900)
show_image(sem_out / "04_semantic_patch_distribution.png", width=1000)
show_image(sem_out / "05_same_identity_semantic_consistency.png", width=980)
show_image(sem_out / "06_semantic_pipeline_diagram.png", width=1000)
show_image(sem_out / "07_semantic_overlay_grid.png", width=1000)
"""))

cells.append(md("""
## 12. Diễn giải kết quả trong lúc demo

Khi chạy xong notebook, nên trình bày theo thứ tự:

1. **Baseline TransReID** tạo embedding và ranking gallery.
2. **Local Reliability** xử lý câu hỏi patch nào đáng tin để đóng góp vào embedding.
3. **Semantic Alignment** xử lý câu hỏi patch thuộc vùng cơ thể nào.
4. **Semantic + Reliability** kết hợp hai tín hiệu cục bộ có cấu trúc.
5. **Kết quả định lượng** cho thấy mAP tăng rõ hơn Rank-1.
6. **Failure cases** nhắc rằng Re-ID là bài toán ranking, không phải chỉ kiểm tra đúng/sai Top-1.

Không nên nói mô hình đã giải quyết triệt để bài toán. Kết luận đúng hơn là: các tín hiệu patch-level cho thấy xu hướng cải thiện trên Market-1501 và cần thêm ablation/dataset khó hơn để khẳng định tính khái quát.
"""))

nb["cells"] = cells

for notebook_path in NOTEBOOKS:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with notebook_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(notebook_path)
