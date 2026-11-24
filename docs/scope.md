# Problem Statement and Scope

## 1. Context

Potholes are a major road safety and maintenance problem in India. Government road-accident data (MoRTH) has recorded roughly 2,000 deaths a year linked to potholes in recent years. *(Cite the exact figure and year from the MoRTH "Road Accidents in India" report you use.)* City and highway teams usually find potholes through manual surveys or citizen complaints. Both are slow and inconsistent, and they rarely produce a list that ranks repairs by urgency.

## 2. Problem statement

City maintenance teams need a scalable way to find potholes in road inspection images. This project trains an object detection model that locates potholes in road images and returns a bounding box and confidence score for each one. It adds a severity rating so teams can decide which repairs come first.

## 3. ML task

| Item | Definition |
|---|---|
| Task | Single-class object detection (`pothole`) |
| Input | An RGB road image (phone, dashcam or survey camera) |
| Output | Boxes, confidence, severity (Low/Medium/High), repair priority (0-100), and a road condition for the image |
| Primary metrics | mAP@0.5 and mAP@0.5:0.95 |
| Supporting metrics | Precision, recall, F1 at the working threshold, IoU, recall by pothole size, latency per image |
| Target | *(set after the baseline, e.g. mAP@0.5 ≥ 0.70 on the Indian test set, recall ≥ 0.80)* |

## 4. Users and business value

| User | What they get |
|---|---|
| Municipal road maintenance engineers | A ranked repair list instead of complaint-driven repairs |
| Smart-city command centres | Road condition data from ordinary photos or dashcams |
| Highway operators and contractors | Evidence for maintenance obligations (e.g. under HAM/BOT contracts) |
| Road safety analysts | Pothole counts and severity by stretch |

## 5. Data

| Dataset | Source | Use |
|---|---|---|
| Roboflow Pothole | https://public.roboflow.com/object-detection/pothole | The brief's primary dataset (baseline) |
| RDD2022 India, class D40 | https://github.com/sekilab/RoadDamageDetector (Figshare archive) | Indian roads for the advanced model; extra Roboflow-style sources are allowed by brief §6 if documented |

## 6. In scope

- A YOLOv8n baseline and a YOLOv8s advanced model, with an optional YOLO11 comparison
- Evaluation with mAP, IoU, precision and recall, plus error analysis
- Rule-based severity scoring (an extension the brief allows)
- A Streamlit demo, a FastAPI endpoint, Docker packaging, and ONNX export

## 7. Out of scope (future work)

- Real-time dashcam video on moving vehicles
- GIS integration and live maps
- Learned depth or volume estimation
- Other road damage types (cracks D00/D10/D20)

## 8. Assumptions and risks

| Risk | Mitigation |
|---|---|
| Small dataset → overfitting | Add RDD2022 India; augmentation; early stopping (`patience`) |
| Lighting, camera angle and texture vary | Strong HSV augmentation; test on Indian images separately |
| Inconsistent box labels | `validate.py` checks; review the worst error cases |
| Water-filled potholes and shadows confuse the model | Background images in training; document them as a known failure |
| Severity is a heuristic | Document the formula; calibrate the thresholds; present it as a ranking aid |

## 9. Team roles (5 members, brief §11)

| Role | Owns |
|---|---|
| Project lead and business analyst | Scope, stakeholder story, final narrative, deck |
| Data engineer | `pothole/data/*`, data quality, splits, notebook 01 |
| Modelling lead | `train.py`, configs, notebooks 02-03, tuning |
| Evaluation and research lead | `evaluate.py`, notebook 04, error analysis, ethics |
| Deployment and documentation lead | App, API, Docker, export, README |
