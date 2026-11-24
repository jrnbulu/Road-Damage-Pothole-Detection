"""API tests with a stubbed model, so they run without trained weights."""
from fastapi.testclient import TestClient

import api.main as api

FAKE = {"image_size": {"width": 96, "height": 64},
        "detections": [{"label": "pothole", "confidence": 0.9, "box_xyxy": [10, 10, 40, 40],
                        "severity": {"level": "High", "priority": 80, "area_fraction": 0.2,
                                     "adjusted_size": 0.3, "in_lane": True}}],
        "summary": {"condition": "Poor - repair urgently", "pothole_count": 1, "max_severity": "High",
                    "max_priority": 80, "counts": {"Low": 0, "Medium": 0, "High": 1}},
        "latency_ms": 5.0, "thresholds": {"confidence": 0.25, "iou": 0.45}}


def client(monkeypatch):
    monkeypatch.setattr(api, "get_model_or_503", lambda: None)
    monkeypatch.setattr(api, "predict", lambda img, path, conf: FAKE)
    return TestClient(api.app)


def test_health():
    r = TestClient(api.app).get("/health").json()
    assert r["status"] in {"ok", "model_missing"} and "version" in r


def test_predict_json(monkeypatch, jpg_bytes):
    r = client(monkeypatch).post("/predict", files={"file": ("road.jpg", jpg_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "road.jpg" and body["summary"]["pothole_count"] == 1


def test_predict_image(monkeypatch, jpg_bytes):
    r = client(monkeypatch).post("/predict/image", files={"file": ("road.jpg", jpg_bytes, "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-pothole-max-severity"] == "High" and r.headers["x-pothole-count"] == "1"
    assert r.content[:2] == b"\xff\xd8"  # JPEG magic number


def test_rejects_non_image(monkeypatch):
    r = client(monkeypatch).post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_empty_file(monkeypatch):
    r = client(monkeypatch).post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_rejects_bad_conf(monkeypatch, jpg_bytes):
    r = client(monkeypatch).post("/predict?conf=2", files={"file": ("r.jpg", jpg_bytes, "image/jpeg")})
    assert r.status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path, jpg_bytes):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "none.pt")
    r = TestClient(api.app).post("/predict", files={"file": ("r.jpg", jpg_bytes, "image/jpeg")})
    assert r.status_code == 503
