"""Load a trained model once and turn images into detections + severity.

Shared by the Streamlit app, the FastAPI service and the CLI:
    python -m pothole.inference path/to/image.jpg [--model models/best.pt]
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pothole import severity as sev
from pothole.config import CONF_THRESHOLD, IMG_SIZE, IOU_THRESHOLD, MODEL_PATH

COLORS = {"High": (0, 0, 230), "Medium": (0, 140, 255), "Low": (0, 200, 0)}  # BGR


@lru_cache(maxsize=4)
def load_model(model_path: str = str(MODEL_PATH)):
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train one (python -m pothole.train) and copy "
            f"best.pt to models/, or set MODEL_PATH.")
    model = YOLO(model_path)
    # warm-up so the first real request doesn't report a misleading multi-second latency
    model.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8), imgsz=IMG_SIZE, verbose=False)
    return model


def predict(image_bgr: np.ndarray, model_path: str = str(MODEL_PATH),
            conf: float = CONF_THRESHOLD, iou: float = IOU_THRESHOLD) -> dict:
    """Run detection on one BGR image. Returns a JSON-safe dict."""
    model = load_model(model_path)
    h, w = image_bgr.shape[:2]
    t0 = time.perf_counter()
    result = model.predict(image_bgr, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    return build_prediction(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                            [result.names[int(c)] for c in result.boxes.cls.tolist()],
                            w, h, latency_ms, conf, iou)


def build_prediction(boxes, scores, labels, w, h, latency_ms, conf, iou) -> dict:
    """Turn raw boxes into the response format (kept separate so it can be tested without a model)."""
    detections, severities = [], []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        s = sev.score_box(x1, y1, x2, y2, w, h)
        severities.append(s)
        detections.append({
            "label": label,
            "confidence": round(float(score), 4),
            "box_xyxy": [round(float(v), 1) for v in box],
            "severity": sev.to_dict(s),
        })
    detections.sort(key=lambda d: -d["severity"]["priority"])
    return {
        "image_size": {"width": w, "height": h},
        "detections": detections,
        "summary": sev.summarise(severities),
        "latency_ms": round(latency_ms, 1),
        "thresholds": {"confidence": conf, "iou": iou},
    }


def draw(image_bgr: np.ndarray, prediction: dict) -> np.ndarray:
    """Draw boxes coloured by severity. Returns a new BGR image."""
    out = image_bgr.copy()
    thick = max(2, round(min(out.shape[:2]) / 300))
    for d in prediction["detections"]:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        color = COLORS[d["severity"]["level"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f"{d['severity']['level']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick / 2, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick / 2,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image. Use JPG, PNG, BMP or WEBP.")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--out", type=Path, help="Where to save the annotated image")
    args = ap.parse_args()

    img = decode_image(args.image.read_bytes())
    pred = predict(img, args.model, args.conf)
    print(json.dumps(pred, indent=2))
    out = args.out or args.image.with_name(args.image.stem + "_pred.jpg")
    cv2.imwrite(str(out), draw(img, pred))
    print(f"Annotated image: {out}")


if __name__ == "__main__":
    main()
