"""Streamlit demo: upload road photos, see potholes with severity and a repair priority list.

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pothole.config import CONF_THRESHOLD, MODEL_PATH, MODELS_DIR, SAMPLES_DIR  # noqa: E402
from pothole.inference import decode_image, draw, load_model, predict  # noqa: E402

st.set_page_config(page_title="Pothole Detector", page_icon="🛣️", layout="wide")
st.title("Road Damage & Pothole Detection")
st.caption("Upload road images to find potholes, rate their severity and rank them for repair. "
           "This tool helps prioritise inspections. It does not replace checking the road in person.")

with st.sidebar:
    st.header("Settings")
    models = sorted(MODELS_DIR.glob("*.pt"))
    default = MODEL_PATH if MODEL_PATH.exists() else (models[0] if models else MODEL_PATH)
    options = sorted({str(default), *map(str, models)})
    model_path = st.selectbox("Model", options, index=options.index(str(default)),
                              format_func=lambda p: Path(p).name)
    conf = st.slider("Confidence threshold", 0.05, 0.95, CONF_THRESHOLD, 0.05,
                     help="Lower finds more potholes but gives more false alarms.")
    st.markdown("**Severity colours**  \n🟥 High  🟧 Medium  🟩 Low")
    st.markdown("Severity is a rule-based estimate from box size and position (see `pothole/severity.py`).")

try:
    load_model(model_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

uploads = st.file_uploader("Road images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
samples = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
use_samples = bool(samples) and st.checkbox(f"Use the {len(samples)} sample images in samples/", value=not uploads)

inputs = [(u.name, u.getvalue()) for u in uploads or []]
if use_samples:
    inputs += [(p.name, p.read_bytes()) for p in samples]
if not inputs:
    st.info("Upload one or more road images to start.")
    st.stop()

rows = []
for name, data in inputs:
    try:
        img = decode_image(data)
    except ValueError as e:
        st.warning(f"{name}: {e}")
        continue
    pred = predict(img, model_path, conf)
    s = pred["summary"]
    rows.append({"image": name, "potholes": s["pothole_count"], "condition": s["condition"],
                 "max severity": s["max_severity"] or "-", "priority": s["max_priority"],
                 "latency ms": pred["latency_ms"]})

    with st.container(border=True):
        st.subheader(name)
        c1, c2 = st.columns([3, 2])
        c1.image(cv2.cvtColor(draw(img, pred), cv2.COLOR_BGR2RGB), width="stretch")
        with c2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Potholes", s["pothole_count"])
            m2.metric("Worst", s["max_severity"] or "None")
            m3.metric("Priority", s["max_priority"])
            st.write(f"**Road condition:** {s['condition']}")
            if pred["detections"]:
                st.dataframe(pd.DataFrame([{
                    "#": i + 1, "severity": d["severity"]["level"], "priority": d["severity"]["priority"],
                    "confidence": d["confidence"], "in lane": d["severity"]["in_lane"],
                    "area %": round(100 * d["severity"]["area_fraction"], 2)}
                    for i, d in enumerate(pred["detections"])]), hide_index=True, width="stretch")
            st.download_button("Download JSON", json.dumps(pred, indent=2), f"{Path(name).stem}.json",
                               "application/json", key=f"dl_{name}")

if len(rows) > 1:
    st.header("Repair priority list")
    df = pd.DataFrame(rows).sort_values(["priority", "potholes"], ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "pothole_priority.csv", "text/csv")
