"""Inference helpers that don't need a trained model."""
import numpy as np
import pytest

from pothole import inference


def test_build_prediction_sorts_by_priority_and_summarises():
    boxes = [(0, 0, 10, 10), (400, 500, 900, 720)]
    pred = inference.build_prediction(boxes, [0.5, 0.9], ["pothole", "pothole"], 1280, 720, 12.34, 0.25, 0.45)
    assert pred["detections"][0]["severity"]["level"] == "High"  # big box first
    assert pred["summary"]["pothole_count"] == 2
    assert pred["latency_ms"] == 12.3
    assert pred["image_size"] == {"width": 1280, "height": 720}


def test_build_prediction_empty():
    pred = inference.build_prediction([], [], [], 100, 100, 1.0, 0.25, 0.45)
    assert pred["detections"] == [] and pred["summary"]["condition"] == "Good"


def test_draw_returns_new_image_with_boxes():
    img = np.zeros((200, 300, 3), np.uint8)
    pred = inference.build_prediction([(50, 50, 150, 150)], [0.8], ["pothole"], 300, 200, 1, 0.25, 0.45)
    out = inference.draw(img, pred)
    assert out.shape == img.shape and out.any() and not img.any()


def test_decode_image(jpg_bytes):
    assert inference.decode_image(jpg_bytes).shape == (64, 96, 3)
    with pytest.raises(ValueError):
        inference.decode_image(b"not an image")


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.load_model(str(tmp_path / "none.pt"))
