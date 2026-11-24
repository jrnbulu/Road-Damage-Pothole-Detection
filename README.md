# Road Damage & Pothole Detection (YOLOv8)

PGDS AI/ML capstone project: Problem Statement #2, Smart Cities / Object Detection.

This project detects potholes in road images, rates each one **Low / Medium / High** severity, and ranks images by repair priority. You can use it through a Streamlit app or a FastAPI endpoint. On top of the brief, it adds **Indian road images (RDD2022 India)** and a **rule-based severity score**.

> **Responsible use:** the system helps prioritise inspections. It does not replace checking a road in person. If the model finds no pothole, that does not prove the road has none.

## Pipeline

```
Roboflow potholes ─┐
                   ├─► merge ─► validate/EDA ─► train (baseline YOLOv8n → advanced YOLOv8s)
RDD2022 India D40 ─┘                               │
                                                   ▼
                     evaluate (mAP, P/R, IoU, size-wise recall, latency, error cases)
                                                   │
                                                   ▼
                     export (ONNX / TFLite) ─► Streamlit app · FastAPI · Docker
```

## Project layout

| Path | What it holds |
|---|---|
| `pothole/data/download.py` | Roboflow dataset, via the API or a ZIP export |
| `pothole/data/rdd2022_india.py` | Downloads RDD2022 India and converts the `D40` (pothole) labels to YOLO format |
| `pothole/data/remote_zip.py` | Takes only `India.zip` out of the 13 GB Figshare archive (HTTP range requests, resumable, CRC-checked) |
| `pothole/data/merge.py` | Combines datasets, keeping each one's train/val/test split |
| `pothole/data/validate.py` | Data-quality checks and EDA plots → `reports/eda/` |
| `pothole/train.py` | Trains from `configs/*.yaml` and logs every run to `reports/experiments.csv` |
| `pothole/evaluate.py` | Metrics, error overlays and model comparison → `reports/eval/` |
| `pothole/severity.py` | Rule-based severity and repair priority |
| `pothole/inference.py` | Shared prediction code, also usable from the command line |
| `pothole/export.py` | ONNX/TFLite export and speed/size benchmark |
| `app/streamlit_app.py` | Demo web app |
| `api/main.py` | REST API |
| `notebooks/01-04` | Deliverable notebooks: EDA, baseline, advanced, evaluation |
| `docs/` | Scope document, deliverables checklist, report, presentation outline and deck |
| `scripts/` | Notebook and presentation builders, smoke test |
| `tests/` | Unit, API and pipeline tests (`pytest -q`) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**You need a GPU for full training.** On a CPU-only laptop, do the data preparation, evaluation and demo locally, and run training on **Google Colab** (free T4 GPU):

1. Upload this folder to Google Drive, at `MyDrive/Road-Damage-Pothole-Detection`.
2. Open a notebook from `notebooks/` in Colab.
3. Set *Runtime → Change runtime type → T4 GPU*.

The first cell of each notebook mounts Drive automatically.

## Run it step by step

```bash
# 1. Data
set ROBOFLOW_API_KEY=your_key                    # free key: roboflow.com → Settings (macOS/Linux: export ...)
python -m pothole.data.download                  # or: --zip path\to\export.zip
python -m pothole.data.rdd2022_india             # ~527 MB download
python -m pothole.data.merge
python -m pothole.data.validate --dataset roboflow
python -m pothole.data.validate --dataset merged

# 2. Train (on a GPU; add --epochs 5 --device cpu for a quick local check)
python -m pothole.train --config configs/baseline.yaml
python -m pothole.train --config configs/advanced.yaml

# 3. Evaluate both models on both test sets
python -m pothole.evaluate --models models/baseline_yolov8n.pt models/advanced_yolov8s_india.pt --data data/processed/roboflow/data.yaml
python -m pothole.evaluate --models models/baseline_yolov8n.pt models/advanced_yolov8s_india.pt --data data/processed/merged/data.yaml

# 4. Optimise
python -m pothole.export --model models/advanced_yolov8s_india.pt --formats onnx

# 5. Pick the demo model and run the demo
copy models\advanced_yolov8s_india.pt models\best.pt
streamlit run app/streamlit_app.py               # http://localhost:8501
uvicorn api.main:app --reload                    # http://127.0.0.1:8000/docs

# 6. Build the presentation deck (picks up plots and tables from reports/)
python scripts/build_presentation.py
```

Single image from the command line:

```bash
python -m pothole.inference samples/road1.jpg
```

## Tests

```bash
pytest -q                          # fast tests, no dataset or GPU needed
python scripts/smoke_test.py       # end-to-end run on a tiny synthetic dataset (~2 min on CPU)
```

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Model status |
| POST | `/predict?conf=0.25` | JSON with detections (box, confidence, severity) and an image summary |
| POST | `/predict/image?conf=0.25` | Annotated JPEG, with the summary in `X-Pothole-*` headers |

```bash
curl -F "file=@samples/road1.jpg" http://127.0.0.1:8000/predict
```

## Docker

```bash
docker compose up --build        # API → :8000, app → :8501
```

The image is CPU-only. It mounts `models/` at runtime, so you don't need to rebuild it after retraining.

## Severity scoring

Severity is a heuristic, not a learned model. It estimates pothole size from the bounding box and corrects for perspective: a pothole near the bottom of the frame is close to the camera and looks bigger. Potholes in the driving lane get a priority bonus. The thresholds are in `pothole/severity.py`. Calibrate them on a few images where you know the real pothole size, and report the values you used.

## Data sources and licences

- **Roboflow Pothole Dataset**: https://public.roboflow.com/object-detection/pothole (`brad-dwyer/pothole-voxrl`). Check the licence on the dataset page and cite it.
- **RDD2022 India**: Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*. See https://github.com/sekilab/RoadDamageDetector. The images are licensed CC BY-SA 4.0. Only class `D40` (pothole) is used. The original AWS download link returns 403 (Nov 2026), so the script takes `RDD2022/India.zip` (527 MB) out of the official 13 GB Figshare archive (doi:10.6084/m9.figshare.21431547) with HTTP range requests. It doesn't download the rest of the archive.

## Deliverables

See [docs/deliverables_checklist.md](docs/deliverables_checklist.md). It maps every item in the brief (§12 deliverables, §13 success criteria, §15 presentation) to the file that covers it.
