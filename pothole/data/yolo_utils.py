"""Small helpers for YOLO-format datasets (images/ + labels/ per split)."""
from pathlib import Path

import yaml

from pothole.config import CLASS_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Roboflow exports call the validation split "valid"; ultralytics calls it "val".
SPLIT_ALIASES = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def find_split_dirs(dataset_dir: Path) -> dict[str, Path]:
    """Return {"train"|"val"|"test": <split>/images} for splits that exist."""
    found = {}
    for split, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            img_dir = dataset_dir / alias / "images"
            if img_dir.is_dir():
                found[split] = img_dir
                break
    return found


def list_images(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for(img_path: Path) -> Path:
    """<split>/images/x.jpg -> <split>/labels/x.txt"""
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def read_labels(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO boxes (cls, xc, yc, w, h), all normalised 0-1. Missing file = no boxes."""
    if not label_file.exists():
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
    return boxes


def write_data_yaml(dataset_dir: Path, names: list[str] = CLASS_NAMES) -> Path:
    """Write a data.yaml with absolute paths (Roboflow's relative ones often break)."""
    dataset_dir = dataset_dir.resolve()
    splits = find_split_dirs(dataset_dir)
    if "train" not in splits or "val" not in splits:
        raise SystemExit(f"{dataset_dir} needs at least train/ and valid/ (or val/) folders with images/")
    cfg = {"path": str(dataset_dir), **{k: str(v) for k, v in splits.items()},
           "names": dict(enumerate(names))}
    out = dataset_dir / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
