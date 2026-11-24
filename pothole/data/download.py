"""Get the Roboflow public pothole dataset in YOLOv8 format.

Two ways (pick one):
  1. API:  set ROBOFLOW_API_KEY (free account -> Settings -> API key), then
           python -m pothole.data.download
  2. ZIP:  export "YOLOv8" from the Roboflow page in a browser, then
           python -m pothole.data.download --zip path/to/Pothole.v1.yolov8.zip

Result: data/processed/roboflow/{train,valid,test}/{images,labels} + data.yaml
"""
import argparse
import os
import shutil
import zipfile
from pathlib import Path

from pothole.config import (PROCESSED_DIR, RAW_DIR, ROBOFLOW_PROJECT,
                            ROBOFLOW_VERSION, ROBOFLOW_WORKSPACE)
from pothole.data.yolo_utils import find_split_dirs, list_images, write_data_yaml


def from_api(out_dir: Path, workspace: str, project: str, version: int) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first, or pass --zip with a manual export.")
    from roboflow import Roboflow  # imported here so the ZIP path needs no roboflow package

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=str(out_dir), overwrite=True)
    return Path(ds.location)


def from_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def locate_dataset_root(extracted: Path) -> Path:
    """The export may be nested one folder deep; find the folder holding train/."""
    for cand in [extracted, *[p for p in extracted.rglob("*") if p.is_dir()]]:
        if "train" in find_split_dirs(cand):
            return cand
    raise SystemExit(f"No train/images folder found under {extracted}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Path to a Roboflow YOLOv8 export ZIP")
    ap.add_argument("--workspace", default=ROBOFLOW_WORKSPACE)
    ap.add_argument("--project", default=ROBOFLOW_PROJECT)
    ap.add_argument("--version", type=int, default=ROBOFLOW_VERSION)
    args = ap.parse_args()

    raw_dir = RAW_DIR / "roboflow"
    extracted = from_zip(args.zip, raw_dir) if args.zip else from_api(
        raw_dir, args.workspace, args.project, args.version)
    src = locate_dataset_root(extracted)

    dst = PROCESSED_DIR / "roboflow"
    if dst.exists():
        shutil.rmtree(dst)
    for split_dir in find_split_dirs(src).values():
        shutil.copytree(split_dir.parent, dst / split_dir.parent.name)
    yaml_path = write_data_yaml(dst)

    print(f"Dataset ready: {dst}")
    for split, img_dir in find_split_dirs(dst).items():
        print(f"  {split:5s}: {len(list_images(img_dir))} images")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
