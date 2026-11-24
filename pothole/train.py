"""Train a YOLOv8 pothole detector from a YAML config.

    python -m pothole.train --config configs/baseline.yaml
    python -m pothole.train --config configs/advanced.yaml --epochs 5 --device cpu   # quick check

Any extra --key value pair overrides the config. Every run is appended to
reports/experiments.csv (config + best validation metrics) for experiment tracking,
and its best weights are copied to models/<run name>.pt.
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from pothole.config import MODELS_DIR, REPORTS_DIR, ROOT, RUNS_DIR

TRACKED_KEYS = ["model", "data", "epochs", "imgsz", "batch", "optimizer", "lr0", "cos_lr", "mosaic", "mixup"]


def parse_overrides(extra: list[str]) -> dict:
    """['--epochs', '5', '--device', 'cpu'] -> {'epochs': 5, 'device': 'cpu'}"""
    out, key = {}, None
    for tok in extra:
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_")
            out[key] = True
        elif key:
            out[key] = yaml.safe_load(tok)
            key = None
    return out


def resolve(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else (ROOT / p))


def best_epoch_metrics(results_csv: Path) -> dict:
    """Metrics from the epoch with the highest mAP50-95 in ultralytics' results.csv."""
    if not results_csv.exists():
        return {}
    with results_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    key = next(k for k in rows[0] if "mAP50-95" in k)
    best = max(rows, key=lambda r: float(r[key]))
    return {k.strip(): float(v) for k, v in best.items() if v.strip()}


def log_experiment(cfg: dict, metrics: dict, weights: Path, elapsed_min: float,
                   log: Path = REPORTS_DIR / "experiments.csv"):
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "name": cfg["name"],
           **{k: cfg.get(k, "") for k in TRACKED_KEYS},
           "precision": metrics.get("metrics/precision(B)"), "recall": metrics.get("metrics/recall(B)"),
           "mAP50": metrics.get("metrics/mAP50(B)"), "mAP50_95": metrics.get("metrics/mAP50-95(B)"),
           "train_minutes": round(elapsed_min, 1), "weights": str(weights)}
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"Logged to {log}")


TRAINING_ARTIFACTS = ["results.csv", "results.png", "train_config.json", "args.yaml", "labels.jpg",
                      "BoxPR_curve.png", "PR_curve.png", "BoxF1_curve.png", "F1_curve.png",
                      "confusion_matrix.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]


def copy_training_artifacts(run_dir: Path, dest: Path) -> list[str]:
    """Copy the small, report-worthy files from a run (runs/ itself is not committed)."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in TRAINING_ARTIFACTS:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, dest / name)
            copied.append(name)
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    args, extra = ap.parse_known_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg.update(parse_overrides(extra))
    cfg["data"] = resolve(cfg["data"])
    if not Path(cfg["data"]).exists():
        raise SystemExit(f"{cfg['data']} not found. Prepare the data first (see README).")

    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    train_args = {k: v for k, v in cfg.items() if k != "model"}
    train_args.update(project=str(RUNS_DIR), exist_ok=True, plots=True)

    start = datetime.now()
    model.train(**train_args)
    elapsed = (datetime.now() - start).total_seconds() / 60

    run_dir = RUNS_DIR / cfg["name"]
    MODELS_DIR.mkdir(exist_ok=True)
    final = MODELS_DIR / f"{cfg['name']}.pt"
    shutil.copy2(run_dir / "weights" / "best.pt", final)

    log_experiment(cfg, best_epoch_metrics(run_dir / "results.csv"), final, elapsed)
    (run_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    copy_training_artifacts(run_dir, REPORTS_DIR / "training" / cfg["name"])
    print(f"Best weights: {final}\nTo use in the app: copy to models/best.pt or set MODEL_PATH={final}")


if __name__ == "__main__":
    main()
