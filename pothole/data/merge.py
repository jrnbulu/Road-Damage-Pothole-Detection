"""Merge several YOLO datasets (same single 'pothole' class) into one.

    python -m pothole.data.merge                       # roboflow + rdd_india
    python -m pothole.data.merge --sources roboflow rdd_india other_universe_set

Files are prefixed with their source name so nothing collides, and each
source keeps its own train/val/test split (so test images never leak into train).
Result: data/processed/merged + data.yaml
"""
import argparse
import shutil
from pathlib import Path

from pothole.config import PROCESSED_DIR
from pothole.data.yolo_utils import find_split_dirs, label_path_for, list_images, write_data_yaml

OUT_SPLIT_NAME = {"train": "train", "val": "valid", "test": "test"}


def merge(sources: list[str], out: Path, base: Path = PROCESSED_DIR) -> dict:
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for source in sources:
        src_dir = base / source
        if not src_dir.is_dir():
            raise SystemExit(f"Missing {src_dir}. Build it first (see README).")
        for split, img_dir in find_split_dirs(src_dir).items():
            dst = out / OUT_SPLIT_NAME[split]
            (dst / "images").mkdir(parents=True, exist_ok=True)
            (dst / "labels").mkdir(parents=True, exist_ok=True)
            for img in list_images(img_dir):
                new_stem = f"{source}__{img.stem}"
                shutil.copy2(img, dst / "images" / f"{new_stem}{img.suffix}")
                lbl = label_path_for(img)
                (dst / "labels" / f"{new_stem}.txt").write_text(lbl.read_text() if lbl.exists() else "")
                counts[(source, split)] = counts.get((source, split), 0) + 1
    write_data_yaml(out)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["roboflow", "rdd_india"])
    ap.add_argument("--out", default="merged")
    args = ap.parse_args()

    out = PROCESSED_DIR / args.out
    counts = merge(args.sources, out)
    for (source, split), c in sorted(counts.items()):
        print(f"  {source:12s} {split:5s}: {c}")
    print(f"Merged dataset: {out}")


if __name__ == "__main__":
    main()
