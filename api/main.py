"""FastAPI service returning JSON detections and annotated images.

    uvicorn api.main:app --reload        # docs at http://127.0.0.1:8000/docs

Endpoints
  GET  /health                 model status
  POST /predict                image file -> JSON detections + severity summary
  POST /predict/image          image file -> annotated JPEG (summary in X-Pothole-* headers)
"""
import sys
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pothole import __version__  # noqa: E402
from pothole.config import CONF_THRESHOLD, MODEL_PATH  # noqa: E402
from pothole.inference import decode_image, draw, load_model, predict  # noqa: E402

MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Pothole Detection API", version=__version__,
              description="YOLOv8 pothole detection with rule-based severity. "
                          "Use to prioritise inspections, not to replace them.")


async def read_image(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    try:
        return decode_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e


def get_model_or_503():
    try:
        load_model(str(MODEL_PATH))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok" if ok else "model_missing", "model": str(MODEL_PATH), "version": __version__}


@app.post("/predict")
async def predict_json(file: UploadFile = File(...), conf: float = Query(CONF_THRESHOLD, ge=0.01, le=0.99)):
    get_model_or_503()
    img = await read_image(file)
    return {"filename": file.filename, **predict(img, str(MODEL_PATH), conf)}


@app.post("/predict/image", response_class=Response,
          responses={200: {"content": {"image/jpeg": {}}}})
async def predict_image(file: UploadFile = File(...), conf: float = Query(CONF_THRESHOLD, ge=0.01, le=0.99)):
    get_model_or_503()
    img = await read_image(file)
    pred = predict(img, str(MODEL_PATH), conf)
    ok, buf = cv2.imencode(".jpg", draw(img, pred), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Could not encode the output image")
    s = pred["summary"]
    return Response(buf.tobytes(), media_type="image/jpeg", headers={
        "X-Pothole-Count": str(s["pothole_count"]),
        "X-Pothole-Max-Severity": str(s["max_severity"]),
        "X-Pothole-Condition": s["condition"],
        "X-Latency-Ms": str(pred["latency_ms"]),
    })
