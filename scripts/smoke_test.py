"""End-to-end check of the whole pipeline on a tiny synthetic dataset (CPU, ~2 min).

    python scripts/smoke_test.py            # runs and then deletes everything it created
    python scripts/smoke_test.py --keep     # keep outputs to inspect them

Steps: make fake road images with dark "potholes" -> validate -> train 2 epochs
-> evaluate -> single-image inference -> API call with the trained model.
The model is meaningless; this only proves every stage runs and connects.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pothole.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, RUNS_DIR  # noqa: E402
from pothole.data.yolo_utils import write_data_yaml  # noqa: E402

NAME = "_smoke"


def make_dataset(root: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    for split, n in (("train", 24), ("valid", 6), ("test", 6)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = (np.full((320, 480, 3), 120) + nrng.integers(0, 30, (320, 480, 3))).astype(np.uint8)
            lines = []
            for _ in range(rng.randint(0, 2)):
                cx, cy, a, b = rng.randint(60, 420), rng.randint(150, 280), rng.randint(15, 50), rng.randint(8, 25)
                cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, (30, 30, 30), -1)
                lines.append(f"0 {cx / 480:.4f} {cy / 320:.4f} {2 * a / 480:.4f} {2 * b / 320:.4f}")
            cv2.imwrite(str(root / split / "images" / f"{i}.jpg"), img)
            (root / split / "labels" / f"{i}.txt").write_text("\n".join(lines))
    return write_data_yaml(root)


def run(*args, env=None):
    print("\n$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env=env)


def cleanup():
    for p in (PROCESSED_DIR / NAME, RUNS_DIR / NAME, REPORTS_DIR / "eda" / NAME, REPORTS_DIR / "eval" / NAME):
        shutil.rmtree(p, ignore_errors=True)
    for p in (MODELS_DIR / f"{NAME}.pt", ROOT / "yolov8n.pt"):
        p.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    log = REPORTS_DIR / "experiments.csv"
    log_before = log.read_text() if log.exists() else None
    try:
        data_yaml = make_dataset(PROCESSED_DIR / NAME)
        run("-m", "pothole.data.validate", "--dataset", NAME)
        run("-m", "pothole.train", "--config", "configs/baseline.yaml", "--name", NAME,
            "--data", str(data_yaml), "--epochs", "2", "--imgsz", "320", "--batch", "8",
            "--device", "cpu", "--workers", "0")
        model = MODELS_DIR / f"{NAME}.pt"
        assert model.exists(), "training did not produce weights"
        run("-m", "pothole.evaluate", "--models", str(model), "--data", str(data_yaml),
            "--conf", "0.01", "--device", "cpu", "--n-errors", "2", "--tag", NAME)
        assert (REPORTS_DIR / "eval" / NAME / "comparison.md").exists()
        test_img = PROCESSED_DIR / NAME / "test" / "images" / "0.jpg"
        out_img = RUNS_DIR / NAME / "pred.jpg"
        run("-m", "pothole.inference", str(test_img), "--model", str(model), "--conf", "0.01",
            "--out", str(out_img))
        assert out_img.exists()

        from fastapi.testclient import TestClient
        os.environ["MODEL_PATH"] = str(model)
        import importlib

        import pothole.config
        importlib.reload(pothole.config)
        import api.main as api_main
        importlib.reload(api_main)
        r = TestClient(api_main.app).post("/predict", files={"file": ("0.jpg", test_img.read_bytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        print("\nAPI /predict:", r.json()["summary"])
        print("\nSMOKE TEST PASSED")
    finally:
        if not args.keep:
            cleanup()
            if log_before is None:
                log.unlink(missing_ok=True)
            else:
                log.write_text(log_before)


if __name__ == "__main__":
    main()
