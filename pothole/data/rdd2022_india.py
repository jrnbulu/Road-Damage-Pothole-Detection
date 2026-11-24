"""Download RDD2022 India (~527 MB) and convert its potholes (class D40) to YOLO format.

Source: https://github.com/sekilab/RoadDamageDetector (images CC BY-SA 4.0).
The original AWS link now returns 403, so by default we take RDD2022/India.zip
out of the official Figshare archive (doi:10.6084/m9.figshare.21431547)
without downloading the other 12.7 GB.
Only the annotated train set is used; we split it 80/10/10 ourselves because
the official test set has no public labels.

    python -m pothole.data.rdd2022_india                            # download + convert
    python -m pothole.data.rdd2022_india --zip RDD2022_India.zip    # use a local copy

Result: data/processed/rdd_india/{train,valid,test}/{images,labels} + data.yaml
"""
import argparse
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pothole.config import PROCESSED_DIR, RAW_DIR, RDD2022_FIGSHARE_URL, RDD2022_INDIA_URL
from pothole.data.remote_zip import download_member
from pothole.data.yolo_utils import IMG_EXTS, write_data_yaml

POTHOLE_CODE = "D40"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    def progress(blocks, block_size, total):
        if total > 0:
            print(f"\r  {min(blocks * block_size, total) / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    tmp.rename(dest)
    return dest


def voc_to_yolo(xml_file: Path) -> tuple[list[str], int]:
    """Return (YOLO label lines for potholes, number of non-pothole objects)."""
    root = ET.parse(xml_file).getroot()
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    lines, others = [], 0
    for obj in root.iter("object"):
        if obj.findtext("name", "").strip() != POTHOLE_CODE:
            others += 1
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # drop degenerate boxes
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, others


def split_samples(samples: list, seed: int) -> dict[str, list]:
    """Shuffle and split 80/10/10 into train/valid/test."""
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_val = n_test = max(1, n // 10)
    return {"test": samples[:n_test], "valid": samples[n_test:n_test + n_val],
            "train": samples[n_test + n_val:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="Local RDD2022_India.zip (skips download)")
    ap.add_argument("--neg-ratio", type=float, default=0.25,
                    help="Road images with no pothole to keep, as a fraction of pothole images "
                         "(teaches the model what cracks/normal road look like)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zip_path = args.zip or RAW_DIR / "RDD2022_India.zip"
    if not zip_path.exists():
        try:
            download_member(RDD2022_FIGSHARE_URL, "RDD2022/India.zip", zip_path)
        except Exception as e:
            print(f"Figshare failed ({e}); trying the original AWS link")
            download(RDD2022_INDIA_URL, zip_path)

    extract_dir = RAW_DIR / "rdd2022_india"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        # some releases nest per-split zips inside the country zip
        for inner in list(extract_dir.rglob("*.zip")):
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent)
            inner.unlink()

    images = {p.stem: p for p in extract_dir.rglob("*") if p.suffix.lower() in IMG_EXTS}
    positives, negatives = [], []
    for xml_file in extract_dir.rglob("*.xml"):
        img = images.get(xml_file.stem)
        if img is None:
            continue
        lines, _ = voc_to_yolo(xml_file)
        (positives if lines else negatives).append((img, lines))

    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    samples = positives + negatives[: int(len(positives) * args.neg_ratio)]
    splits = split_samples(samples, args.seed)

    out = PROCESSED_DIR / "rdd_india"
    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for img, lines in items:
            shutil.copy2(img, out / split / "images" / img.name)
            (out / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
    write_data_yaml(out)

    print(f"RDD2022 India: {len(positives)} images with potholes, "
          f"{len(samples) - len(positives)} background images kept")
    for split, items in splits.items():
        print(f"  {split:5s}: {len(items)} images")
    print(f"Ready: {out}")


if __name__ == "__main__":
    main()
