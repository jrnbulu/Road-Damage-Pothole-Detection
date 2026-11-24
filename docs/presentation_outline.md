# Final Presentation Outline (8 slides, following brief §15)

`python scripts/build_presentation.py` turns this outline into `docs/presentation.pptx`. It inserts the plots and tables from `reports/` once they exist.

| # | Slide | Content | Visual / source |
|---|---|---|---|
| 1 | Title and problem | Team, the pothole problem in India (MoRTH figure), who suffers, and the cost of manual surveys | One striking pothole photo |
| 2 | Dataset | Roboflow and RDD2022 India: sources, licences, how you got them, image and box counts | Table, sample grid (notebook 01) |
| 3 | EDA and preprocessing | Box sizes, potholes per image, quality issues fixed, background images, splits | `reports/eda/*/eda_overview.png` |
| 4 | Models | Baseline YOLOv8n vs advanced YOLOv8s: what changed and why | Two-column comparison |
| 5 | Experiments | Hyperparameters, training curves, experiment log | `runs/*/results.png`, `reports/experiments.csv` |
| 6 | Results and errors | mAP/P/R/IoU table on both test sets, recall by size, 3 error patterns | `comparison.md`, `errors/*.jpg` |
| 7 | Demo | Live Streamlit: upload → boxes coloured by severity → priority list → JSON API; latency, ONNX | Screen recording as a backup |
| 8 | Limits, ethics, future | Failure cases, human checks, privacy (faces and plates), roadmap: dashcam + GPS → city dashboard → pilot | Roadmap graphic |

## Demo checklist (brief §10)
- [ ] The model loads without retraining (`models/best.pt`)
- [ ] At least 3 sample predictions (put them in `samples/`)
- [ ] 1 known failure case shown and explained
- [ ] The API `/docs` page opens, and `/predict` returns JSON
