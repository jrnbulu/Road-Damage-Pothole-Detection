# Model Evaluation Report

*Fill in from `reports/eval/<tag>/comparison.md`, `reports/eval/<tag>/<model>/metrics.json`, `reports/experiments.csv` and `reports/export_benchmark.json`.*

## 1. Data

| Dataset | Train | Val | Test | Boxes | Notes |
|---|---|---|---|---|---|
| Roboflow | | | | | |
| RDD2022 India (D40 + background) | | | | | |
| Merged | | | | | |

Data-quality issues found and how you fixed them:

## 2. Models

| | Baseline | Advanced |
|---|---|---|
| Architecture | YOLOv8n | YOLOv8s |
| Training data | Roboflow | Roboflow + RDD India |
| Epochs / optimiser / LR | 50 / auto / default | 120 / AdamW / 0.001, cosine |
| Augmentation | default | road-tuned (see `configs/advanced.yaml`) |
| Training time | | |

## 3. Results

### Roboflow test set
| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 @ conf | Mean IoU | ms/img (CPU) |
|---|---|---|---|---|---|---|---|
| Baseline | | | | | | | |
| Advanced | | | | | | | |

### Indian roads (merged test set)
| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 @ conf | Mean IoU | ms/img (CPU) |
|---|---|---|---|---|---|---|---|
| Baseline | | | | | | | |
| Advanced | | | | | | | |

### Recall by pothole size
| Model | Small (<1%) | Medium (1-5%) | Large (>5%) |
|---|---|---|---|
| Baseline | | | |
| Advanced | | | |

### Choosing the threshold
Chosen confidence: ___. Reason (missing a pothole costs more than a false alarm):

## 4. Error analysis

Group the saved error images (`errors/`) into patterns:

| Failure pattern | Example image | Count | Likely cause | Fix |
|---|---|---|---|---|
| Water-filled pothole missed | | | | |
| Shadow / patch repair flagged as pothole | | | | |
| Small far-away pothole missed | | | | |

## 5. Optimisation

| Format | Size MB | ms/img CPU |
|---|---|---|
| PyTorch | | |
| ONNX | | |
| TFLite INT8 | | |

## 6. Is it usable?

## 7. Limitations, ethics and responsible use

- This is a decision aid for inspection, and every detection is checked by a person.
- Training data can be biased toward some road types, weather, daytime and camera heights.
- Severity is a heuristic and has not been calibrated against real depth.
- Privacy: dashcam images can contain faces and number plates. Blur them before storage in production.

## 8. Future scope

Dashcam video with GPS tagging, an Android app using TFLite, a ward-level hotspot map, more damage classes (cracks), and pilots with a municipal corporation or highway operator.
