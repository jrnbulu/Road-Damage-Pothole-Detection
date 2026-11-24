"""Training helpers: CLI overrides, metric extraction, experiment log, configs."""
import csv

import yaml

from pothole import train
from pothole.config import ROOT


def test_parse_overrides_types():
    out = train.parse_overrides(["--epochs", "5", "--device", "cpu", "--cos-lr", "true", "--lr0", "0.01", "--flag"])
    assert out == {"epochs": 5, "device": "cpu", "cos_lr": True, "lr0": 0.01, "flag": True}


def test_resolve_relative_to_project_root():
    assert train.resolve("configs/x.yaml").startswith(str(ROOT))


def test_best_epoch_metrics_picks_highest_map(tmp_path):
    f = tmp_path / "results.csv"
    f.write_text("epoch,  metrics/mAP50(B),  metrics/mAP50-95(B)\n1,0.4,0.2\n2,0.6,0.35\n3,0.55,0.3\n")
    m = train.best_epoch_metrics(f)
    assert m["epoch"] == 2 and m["metrics/mAP50-95(B)"] == 0.35
    assert train.best_epoch_metrics(tmp_path / "missing.csv") == {}


def test_log_experiment_appends_rows(tmp_path):
    log = tmp_path / "experiments.csv"
    cfg = {"name": "run1", "model": "yolov8n.pt", "epochs": 1}
    train.log_experiment(cfg, {"metrics/mAP50(B)": 0.5}, tmp_path / "w.pt", 1.234, log=log)
    train.log_experiment({**cfg, "name": "run2"}, {}, tmp_path / "w.pt", 2, log=log)
    rows = list(csv.DictReader(log.open()))
    assert [r["name"] for r in rows] == ["run1", "run2"]
    assert rows[0]["mAP50"] == "0.5" and rows[0]["train_minutes"] == "1.2"


def test_configs_are_valid_and_differ():
    base = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
    adv = yaml.safe_load((ROOT / "configs" / "advanced.yaml").read_text())
    for cfg in (base, adv):
        assert {"name", "model", "data", "epochs", "imgsz"} <= cfg.keys()
        assert cfg["model"].startswith("yolov8")  # brief: YOLOv8
    assert base["name"] != adv["name"] and adv["epochs"] > base["epochs"]
    assert adv["flipud"] == 0.0  # roads are never upside down
