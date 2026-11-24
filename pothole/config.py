"""Central paths and settings, so every script agrees on where things live."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"
SAMPLES_DIR = ROOT / "samples"

CLASS_NAMES = ["pothole"]

# Roboflow public pothole dataset (public.roboflow.com/object-detection/pothole)
ROBOFLOW_WORKSPACE = "brad-dwyer"
ROBOFLOW_PROJECT = "pothole-voxrl"
ROBOFLOW_VERSION = 1

# RDD2022 India subset (sekilab/RoadDamageDetector), class D40 = pothole.
# Figshare holds the official full archive; we pull only RDD2022/India.zip out of it.
RDD2022_FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"
# Original AWS link (returned 403 Forbidden in Nov 2026), kept as a fallback
RDD2022_INDIA_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/"
    "Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
)

# Model used by the app and API. Override with the MODEL_PATH env var.
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best.pt"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
