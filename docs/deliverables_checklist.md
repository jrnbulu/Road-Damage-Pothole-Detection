# Deliverables Checklist

This maps every requirement in the capstone brief (Problem Statement #2, Road Damage and Pothole Detection using YOLOv8) to the file that meets it. Tick each box once its results are filled in.

## §12 Expected deliverables

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | Problem statement and scope document | `docs/scope.md` | [ ] |
| 2 | Data access and preprocessing notebook/script | `pothole/data/*.py`, `notebooks/01_data_access_and_eda.ipynb` | [ ] |
| 3 | EDA notebook with key observations and data-quality checks | `notebooks/01_data_access_and_eda.ipynb`, `reports/eda/` | [ ] |
| 4 | Baseline model notebook | `notebooks/02_baseline_model.ipynb`, `configs/baseline.yaml` | [ ] |
| 5 | Advanced model notebook | `notebooks/03_advanced_model.ipynb`, `configs/advanced.yaml` | [ ] |
| 6 | Model evaluation report with metrics, plots and sample errors | `notebooks/04_evaluation_and_error_analysis.ipynb`, `docs/evaluation_report.md`, `reports/eval/` | [ ] |
| 7 | Deployment prototype (Streamlit / FastAPI) | `app/streamlit_app.py`, `api/main.py`, `Dockerfile`, `docker-compose.yml` | [ ] |
| 8 | Final presentation deck | `docs/presentation.pptx` (built by `scripts/build_presentation.py`), `docs/presentation_outline.md` | [ ] |
| 9 | Project folder with README and reproducible instructions | `README.md`, `requirements.txt` | [ ] |

## §7 Scope of work (in scope)

| Requirement | Where |
|---|---|
| Labelled pothole dataset with bounding boxes | `pothole/data/download.py`, `pothole/data/rdd2022_india.py` |
| Fine-tune YOLOv8 on pothole images | `pothole/train.py` |
| mAP, IoU, precision, recall and qualitative error analysis | `pothole/evaluate.py` |
| Demo app that takes road images and returns annotated images | `app/streamlit_app.py`, `api/main.py` (`/predict/image`) |
| Inference pipeline packaged for reproducible runs | `pothole/inference.py`, `Dockerfile` |
| Severity scoring as a rule-based extension | `pothole/severity.py` |

## §9 Evaluation plan

| Required | Where |
|---|---|
| mAP@0.5, mAP@0.5:0.95 | `evaluate.py` → `metrics.json` |
| Precision, recall, IoU | `evaluate.py` (`precision_at_conf`, `recall_at_conf`, `mean_iou_matched`) |
| False positive / false negative examples | `reports/eval/<tag>/<model>/errors/` |
| Inference time per image | `latency_ms_mean`, `latency_ms_p95` |
| More than one headline metric, segment-wise results | Recall by pothole size; separate Roboflow and Indian test sets |
| Is the model usable for the user? | Notebook 04, final section; `docs/evaluation_report.md` §6 |

## §10 Deployment expectations

| Required | Where |
|---|---|
| Loads the trained model without retraining | `models/best.pt` via `MODEL_PATH` |
| Accepts input in the expected format and returns clear output | `/predict` (JSON), `/predict/image` (JPEG); Streamlit uploads |
| At least 3 sample predictions and 1 known failure case | `samples/` (add images), notebook 04 error gallery |
| README with setup, dependencies and run instructions | `README.md` |

## §13 Success criteria

| Criterion | Evidence |
|---|---|
| Dataset loaded reproducibly from a documented public source | Download scripts + README "Data sources" |
| Meaningful baseline and an improved advanced model | `reports/experiments.csv`, `reports/eval/*/comparison.md` |
| Metrics suit the task and are interpreted correctly | Notebook 04, evaluation report |
| Prototype works on new inputs | `tests/test_api.py`, `scripts/smoke_test.py`, live demo |
| Technical results linked to user and business value | `docs/scope.md` §4, presentation slides 1 and 8 |
| Limitations, risks and responsible use stated honestly | `docs/evaluation_report.md` §7, README note |

## §14 Risks

| Risk | Mitigation in this project |
|---|---|
| Small dataset → overfitting | RDD2022 India added; augmentation; early stopping |
| Lighting, angle and texture variation | HSV/scale augmentation; a separate Indian test set |
| Bounding box quality | `validate.py` checks; review of the worst error cases |
| Responsible use | Disclaimer in the app, API and README; human verification step |

## Quality checks

| Check | Command |
|---|---|
| Unit and API tests | `pytest -q` |
| End-to-end pipeline on synthetic data | `python scripts/smoke_test.py` |
