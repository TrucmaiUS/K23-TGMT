# TransReID artifact visualization summary

| Model | mAP | Rank-1 | Rank-5 | Rank-10 | Mean ms/img | Delta mAP | Delta Rank-1 | Delta ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 89.2 | 95.1 | 98.3 | 98.9 | 4.626 | +0.0 | +0.0 | +0.000 |
| Semantic | 89.5 | 95.3 | 98.5 | 99.0 | 4.637 | +0.3 | +0.2 | +0.011 |
| Local Rel. | 89.9 | 95.2 | 98.6 | 99.1 | 4.786 | +0.7 | +0.1 | +0.160 |
| Semantic+Rel. | 90.1 | 95.4 | 98.6 | 99.1 | 4.783 | +0.9 | +0.3 | +0.157 |

Suggested interpretation:
- Semantic+Rel. has the best mAP and Rank-1 in this artifact set.
- Local Rel. improves mAP more strongly than Rank-1, which is useful to discuss ranking quality.
- The added branches introduce small latency overhead, so the report should mention the accuracy-latency trade-off.