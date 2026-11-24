"""Data-quality checks and EDA for a YOLO dataset.

    python -m pothole.data.validate --dataset roboflow

Checks: unreadable images, missing/empty labels, out-of-range or tiny boxes,
duplicate files across splits. Saves stats + plots to reports/eda/<dataset>/.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from pothole.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from pothole.data.yolo_utils import find_split_dirs, label_path_for, list_images, read_labels  # noqa: E402

TINY_BOX = 0.0005  # box area < 0.05% of the image is likely a labelling mistake


def analyse(dataset: str, base: Path = PROCESSED_DIR, reports: Path = REPORTS_DIR) -> dict:
    ds_dir = base / dataset
    splits = find_split_dirs(ds_dir)
    if not splits:
        raise SystemExit(f"No splits found in {ds_dir}")

    stats = {"dataset": dataset, "splits": {}, "issues": defaultdict(list)}
    widths, heights, box_areas, box_aspects, boxes_per_img, centers = [], [], [], [], [], []
    hashes = defaultdict(list)

    for split, img_dir in splits.items():
        imgs = list_images(img_dir)
        n_boxes = n_empty = 0
        for img in imgs:
            try:
                with Image.open(img) as im:
                    w, h = im.size
            except Exception as e:  # corrupt file
                stats["issues"]["unreadable"].append(f"{img.name}: {e}")
                continue
            widths.append(w)
            heights.append(h)
            hashes[hashlib.md5(img.read_bytes()).hexdigest()].append(f"{split}/{img.name}")

            lbl = label_path_for(img)
            if not lbl.exists():
                stats["issues"]["missing_label_file"].append(f"{split}/{img.name}")
            boxes = read_labels(lbl)
            if not boxes:
                n_empty += 1
            boxes_per_img.append(len(boxes))
            for cls, xc, yc, bw, bh in boxes:
                n_boxes += 1
                if cls != 0:
                    stats["issues"]["unexpected_class"].append(f"{split}/{img.name}: class {cls}")
                if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    stats["issues"]["out_of_range_box"].append(f"{split}/{img.name}")
                if bw * bh < TINY_BOX:
                    stats["issues"]["tiny_box"].append(f"{split}/{img.name}")
                box_areas.append(bw * bh)
                box_aspects.append(bw * w / max(bh * h, 1e-6))
                centers.append((xc, yc))
        stats["splits"][split] = {"images": len(imgs), "boxes": n_boxes,
                                  "images_without_potholes": n_empty}

    for files in hashes.values():
        if len({f.split("/")[0] for f in files}) > 1:
            stats["issues"]["duplicate_across_splits"].append(files)

    out = reports / "eda" / dataset
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(boxes_per_img, bins=range(0, max(boxes_per_img, default=1) + 2))
    ax[0, 0].set(title="Potholes per image", xlabel="count", ylabel="images")
    ax[0, 1].hist(box_areas, bins=50)
    ax[0, 1].set(title="Box area (fraction of image)", xlabel="area", ylabel="boxes")
    ax[1, 0].hist(box_aspects, bins=50, range=(0, 5))
    ax[1, 0].set(title="Box aspect ratio (w/h)", xlabel="ratio")
    if centers:
        xs, ys = zip(*centers)
        ax[1, 1].hist2d(xs, ys, bins=20)
    ax[1, 1].set(title="Box centre heatmap", xlabel="x", ylabel="y")
    ax[1, 1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=120)
    plt.close(fig)

    sizes = Counter(zip(widths, heights)).most_common(5)
    stats["image_sizes_top5"] = [f"{w}x{h} ({n})" for (w, h), n in sizes]
    stats["box_area_median"] = sorted(box_areas)[len(box_areas) // 2] if box_areas else None
    stats["small_boxes_pct"] = round(100 * sum(a < 0.01 for a in box_areas) / max(len(box_areas), 1), 1)
    stats["issues"] = {k: {"count": len(v), "examples": v[:5]} for k, v in stats["issues"].items()}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="roboflow", help="Folder name under data/processed/")
    args = ap.parse_args()
    stats = analyse(args.dataset)
    print(json.dumps({k: stats[k] for k in ("splits", "issues", "image_sizes_top5", "small_boxes_pct")}, indent=2))
    print(f"Plots and stats saved to {REPORTS_DIR / 'eda' / args.dataset}")


if __name__ == "__main__":
    main()
