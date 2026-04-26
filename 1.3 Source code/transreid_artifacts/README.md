# TransReID report artifacts

Folder này gom các checkpoint, log đánh giá và hình minh họa cho báo cáo Person Re-ID.

## Cấu trúc chính

- `runs/`: checkpoint `.pth` và `train_log.txt` gốc của từng run.
- `configs_snapshot/`: snapshot config dùng khi train/evaluate.
- `logs/train/`: bản copy train log theo tên model.
- `logs/eval/`: bản copy eval log theo tên model.
- `logs/eval_runs/`: output test run chi tiết.
- `latency/`: latency raw `.npy` cho từng model.
- `visualizations/quantitative/`: biểu đồ định lượng, training curve, latency, summary table.
- `visualizations/qualitative/`: hình so sánh Top-K retrieval giữa các model.
- `visualizations/local_reliability/`: hình visualize cũ của nhánh Local Reliability, đã gom từ `local-reliability/results`.
- `visualizations/semantic/`: hình visualize của nhánh Semantic đang dùng trong slide. `semantic/results` hiện rỗng nên bộ này được gom lại từ `1.2 Slide/figures`.
- `tools/`: script sinh lại hình từ artifact hiện có.

## File summary nên dùng trong báo cáo

- `metrics_summary.csv`: mAP, Rank-1, Rank-5, Rank-10.
- `latency_summary.csv`: latency trung bình, P95, FPS.
- `visualizations/quantitative/summary.md`: bảng kết quả đã tính delta so với baseline.

## Script

Sinh lại hình định lượng:

```powershell
python tools\generate_report_visuals.py
```

Sinh lại hình so sánh retrieval cho 4 checkpoint:

```powershell
python tools\visualize_multimodel_retrieval.py --num-queries 3 --num-distractors 60 --topk 5
```

## Đã dọn

- Đã xoá `plots/` cũ vì chỉ chứa các bar chart rời, bị trùng với `visualizations/quantitative/01_metrics_grouped_bar.png` và các hình phân tích mới.
- Đã gom `plots_extra/` vào `visualizations/quantitative/`.
- Đã gom `qualitative_multimodel/` vào `visualizations/qualitative/`.
- Đã gom hình trong `local-reliability/results/visual_reid_demo` và `local-reliability/results/visual_reid_suite` vào `visualizations/local_reliability/`.
- Đã xoá `01_input_query.png` vì ảnh query đã xuất hiện trong các grid Top-K.
- Đã chuyển script sinh hình vào `tools/`.
