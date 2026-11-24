"""Shared test helpers. Tests need no dataset, GPU or trained model."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_split(root: Path, split: str, n: int, boxes: str = "0 0.5 0.5 0.2 0.2\n", start: int = 0):
    """Create <root>/<split>/{images,labels} with n small images, each labelled with `boxes`."""
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        img = np.full((32, 48, 3), 100 + i, np.uint8)  # unique pixels -> no false duplicates
        cv2.imwrite(str(root / split / "images" / f"{i}.png"), img)
        (root / split / "labels" / f"{i}.txt").write_text(boxes)


@pytest.fixture
def yolo_dataset(tmp_path):
    """A tiny valid YOLO dataset with train/valid/test."""
    root = tmp_path / "ds"
    make_split(root, "train", 4, start=0)
    make_split(root, "valid", 2, start=10)
    make_split(root, "test", 2, start=20)
    return root


@pytest.fixture
def jpg_bytes():
    return cv2.imencode(".jpg", np.full((64, 96, 3), 128, np.uint8))[1].tobytes()
