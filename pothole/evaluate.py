"""Evaluate one or more trained models on a held-out test split.

    python -m pothole.evaluate --models models/baseline_yolov8n.pt models/advanced_yolov8s_india.pt \
                               --data data/processed/roboflow/data.yaml

For each model it reports:
  * mAP@0.5, mAP@0.5:0.95, precision, recall (ultralytics val, plus PR curve and confusion matrix)
  * at the app's working confidence: TP / FP / FN, F1, mean IoU of matched boxes
  * recall by pothole size (small / medium / large), the "segment-wise" view the brief asks for
  * latency per image (mean and p95) on this machine
  * the worst false-negative and false-positive images, saved as overlays for error analysis
Results go to reports/eval/<tag>/<model>/ and a table to reports/eval/<tag>/comparison.md,
where <tag> defaults to the dataset folder name (e.g. roboflow, merged).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from pothole.config import CONF_THRESHOLD, IMG_SIZE, REPORTS_DIR, ROOT
from pothole.data.yolo_utils import label_path_for, list_images, read_labels

MATCH_IOU = 0.5
SIZE_BUCKETS = [("small (<1% of image)", 0, 0.01), ("medium (1-5%)", 0.01, 0.05), ("large (>5%)", 0.05, 1.01)]


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts: list, preds: list) -> tuple[list, list, list]:
    """Greedy match by confidence. Returns (matched (gt_i, pred_i, iou), unmatched gt idx, unmatched pred idx)."""
    pairs, used_gt = [], set()
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i][4]):
        best, best_iou = None, MATCH_IOU
        for gi, g in enumerate(gts):
            if gi not in used_gt:
                v = iou(g, preds[pi])
                if v >= best_iou:
                    best, best_iou = gi, v
        if best is not None:
            used_gt.add(best)
            pairs.append((best, pi, best_iou))
    matched_p = {p for _, p, _ in pairs}
    return pairs, [g for g in range(len(gts)) if g not in used_gt], \
        [p for p in range(len(preds)) if p not in matched_p]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def overlay(img, gts, preds, fn_idx, fp_idx):
    out = img.copy()
    for i, g in enumerate(gts):  # ground truth: green (thick if missed)
        cv2.rectangle(out, tuple(map(int, g[:2])), tuple(map(int, g[2:4])), (0, 200, 0), 4 if i in fn_idx else 2)
    for i, p in enumerate(preds):  # predictions: red if false positive, blue if correct
        color = (0, 0, 230) if i in fp_idx else (230, 120, 0)
        cv2.rectangle(out, tuple(map(int, p[:2])), tuple(map(int, p[2:4])), color, 2)
        cv2.putText(out, f"{p[4]:.2f}", (int(p[0]), max(12, int(p[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(out, "green=truth  blue=correct  red=false alarm  thick green=missed",
                (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def evaluate_model(model_path: Path, data_yaml: Path, split: str, conf: float, device, n_errors: int,
                   tag: str) -> dict:
    from ultralytics import YOLO

    name = model_path.stem
    out = REPORTS_DIR / "eval" / tag / name
    (out / "errors").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    # 1) Standard detection metrics (low conf so the full PR curve is used)
    val = model.val(data=str(data_yaml), split=split, imgsz=IMG_SIZE, device=device, plots=True,
                    project=str(out), name="ultralytics_val", exist_ok=True, verbose=False)
    metrics = {"mAP50": float(val.box.map50), "mAP50_95": float(val.box.map),
               "precision": float(val.box.mp), "recall": float(val.box.mr)}

    # 2) Operating-point analysis on the test images
    cfg = yaml.safe_load(data_yaml.read_text())
    img_dir = Path(cfg[split]) if Path(cfg[split]).is_absolute() else Path(cfg["path"]) / cfg[split]
    images = list_images(img_dir)
    tp = fp = fn = 0
    ious, latencies, per_image = [], [], []
    bucket_hits = {b[0]: [0, 0] for b in SIZE_BUCKETS}

    model.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8), verbose=False, device=device)  # warm-up
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = [((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
               for _, xc, yc, bw, bh in read_labels(label_path_for(img_path))]
        t0 = time.perf_counter()
        r = model.predict(img, conf=conf, imgsz=IMG_SIZE, verbose=False, device=device)[0]
        latencies.append((time.perf_counter() - t0) * 1000)
        preds = [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

        pairs, fn_idx, fp_idx = match(gts, preds)
        tp, fp, fn = tp + len(pairs), fp + len(fp_idx), fn + len(fn_idx)
        ious += [v for *_, v in pairs]
        hit = {g for g, _, _ in pairs}
        for gi, g in enumerate(gts):
            area = (g[2] - g[0]) * (g[3] - g[1]) / (w * h)
            for bname, lo, hi in SIZE_BUCKETS:
                if lo <= area < hi:
                    bucket_hits[bname][1] += 1
                    bucket_hits[bname][0] += gi in hit
        per_image.append((img_path, gts, preds, fn_idx, fp_idx))  # no pixels kept, to save memory

    # 3) Save the worst cases for the error-analysis slide
    for kind, key in (("missed", 3), ("false_alarm", 4)):
        worst = sorted((x for x in per_image if x[key]), key=lambda x: -len(x[key]))[:n_errors]
        for img_path, gts, preds, fn_idx, fp_idx in worst:
            cv2.imwrite(str(out / "errors" / f"{kind}__{img_path.stem}.jpg"),
                        overlay(cv2.imread(str(img_path)), gts, preds, set(fn_idx), set(fp_idx)))

    p, rc, f1 = prf(tp, fp, fn)
    metrics.update({
        "operating_conf": conf, "test_images": len(per_image), "TP": tp, "FP": fp, "FN": fn,
        "precision_at_conf": round(p, 4), "recall_at_conf": round(rc, 4), "f1_at_conf": round(f1, 4),
        "mean_iou_matched": round(statistics.mean(ious), 4) if ious else None,
        "recall_by_size": {k: (round(h / n, 3) if n else None, n) for k, (h, n) in bucket_hits.items()},
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else None,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "device": str(device or "auto"),
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n{name}: {json.dumps(metrics, indent=2)}\nSaved to {out}")
    return {"model": name, **metrics}


def write_comparison(rows: list[dict], data_yaml: Path, tag: str) -> Path:
    cols = ["model", "mAP50", "mAP50_95", "precision", "recall", "f1_at_conf", "mean_iou_matched",
            "FP", "FN", "latency_ms_mean"]
    lines = [f"Test data: `{data_yaml}`\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    lines.append("\nRecall by pothole size (recall, count):\n")
    for r in rows:
        lines.append(f"- **{r['model']}**: " + "; ".join(f"{k}: {v[0]} (n={v[1]})" for k, v in r["recall_by_size"].items()))
    path = REPORTS_DIR / "eval" / tag / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nComparison table: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/roboflow/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--device", default=None, help="cpu, 0 (GPU) ... default: auto")
    ap.add_argument("--n-errors", type=int, default=8, help="Worst images to save per error type")
    ap.add_argument("--tag", help="Name for this evaluation (default: dataset folder name)")
    args = ap.parse_args()
    tag = args.tag or args.data.resolve().parent.name

    rows = [evaluate_model(m, args.data.resolve(), args.split, args.conf, args.device, args.n_errors, tag)
            for m in args.models]
    write_comparison(rows, args.data, tag)


if __name__ == "__main__":
    main()
