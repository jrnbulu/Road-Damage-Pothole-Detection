"""Model optimisation: export to ONNX / TFLite and compare speed and size.

    python -m pothole.export --model models/advanced_yolov8s_india.pt --formats onnx tflite

ONNX runs fast on CPU servers. TFLite (FP16 / INT8) is the format for the
Android dashcam app planned in the product roadmap.
Results are written to reports/export_benchmark.json.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pothole.config import IMG_SIZE, REPORTS_DIR


def bench(model, n: int = 30) -> float:
    img = (np.random.rand(IMG_SIZE, IMG_SIZE, 3) * 255).astype(np.uint8)
    for _ in range(3):
        model.predict(img, imgsz=IMG_SIZE, verbose=False)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img, imgsz=IMG_SIZE, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--formats", nargs="+", default=["onnx"], help="onnx, tflite, openvino, ...")
    ap.add_argument("--int8", action="store_true", help="INT8 quantisation (needs --data for calibration)")
    ap.add_argument("--data", type=Path, help="data.yaml used for INT8 calibration")
    args = ap.parse_args()

    from ultralytics import YOLO

    results = [{"format": "pytorch", "file": str(args.model),
                "size_mb": round(args.model.stat().st_size / 1e6, 1),
                "median_ms_cpu": bench(YOLO(str(args.model)))}]
    for fmt in args.formats:
        extra = {"int8": True, "data": str(args.data)} if args.int8 else {}
        path = Path(YOLO(str(args.model)).export(format=fmt, imgsz=IMG_SIZE, **extra))
        size = sum(f.stat().st_size for f in path.rglob("*")) if path.is_dir() else path.stat().st_size
        entry = {"format": fmt + ("-int8" if args.int8 else ""), "file": str(path), "size_mb": round(size / 1e6, 1)}
        try:
            entry["median_ms_cpu"] = bench(YOLO(str(path), task="detect"))
        except Exception as e:  # some formats need extra runtimes to run locally
            entry["median_ms_cpu"] = f"not benchmarked: {e.__class__.__name__}"
        results.append(entry)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "export_benchmark.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"{r['format']:14s} {r['size_mb']:7.1f} MB   {r['median_ms_cpu']} ms")


if __name__ == "__main__":
    main()
