"""Generate the four deliverable notebooks in notebooks/.

    python scripts/build_notebooks.py

Keeping the notebooks as code here makes them easy to review and regenerate.
Run them afterwards (locally or on Colab) so they contain real outputs.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''# --- Setup: works locally and on Google Colab (free T4 GPU) ---
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Upload the project folder to Google Drive first (MyDrive/Road-Damage-Pothole-Detection)
    from google.colab import drive
    drive.mount("/content/drive")
    os.chdir("/content/drive/MyDrive/Road-Damage-Pothole-Detection")
    !pip -q install -r requirements.txt
else:
    # run from the notebooks/ folder -> go to the project root
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
sys.path.insert(0, os.getcwd())
print("Project root:", os.getcwd())'''

NOTEBOOKS = {
    "01_data_access_and_eda.ipynb": [
        ("md", "# 01 · Data access, quality checks and EDA\n\n**Goal:** load the Roboflow pothole dataset reproducibly, add Indian road images from RDD2022, check label quality and understand the data before modelling.\n\nSources:\n- Roboflow public pothole dataset: https://public.roboflow.com/object-detection/pothole (`brad-dwyer/pothole-voxrl`)\n- RDD2022 India (class `D40` = pothole): https://github.com/sekilab/RoadDamageDetector (images CC BY-SA 4.0)"),
        ("code", SETUP),
        ("md", "## 1. Download the Roboflow dataset\nGet a free API key at roboflow.com (Settings → API key). Or export a *YOLOv8* ZIP from the website and use `--zip`."),
        ("code", 'import os\n# os.environ["ROBOFLOW_API_KEY"] = "paste-your-key"   # or set it in your shell, never commit it\n!python -m pothole.data.download\n# !python -m pothole.data.download --zip "path/to/Pothole.v1.yolov8.zip"'),
        ("md", "## 2. Add Indian road images (RDD2022 India, ~527 MB)\nOnly potholes (`D40`) are kept as labels. Some road images without potholes (cracks, normal road) are kept as background examples to reduce false alarms."),
        ("code", "!python -m pothole.data.rdd2022_india\n!python -m pothole.data.merge --sources roboflow rdd_india"),
        ("md", "## 3. Data-quality checks"),
        ("code", 'import json\nfrom pothole.data.validate import analyse\nfor name in ["roboflow", "rdd_india", "merged"]:\n    try:\n        s = analyse(name)\n        print(name, json.dumps({k: s[k] for k in ("splits", "issues", "image_sizes_top5", "small_boxes_pct")}, indent=1))\n    except SystemExit as e:\n        print(name, "skipped:", e)'),
        ("md", "## 4. EDA plots"),
        ("code", 'from IPython.display import Image, display\nfor name in ["roboflow", "rdd_india"]:\n    p = f"reports/eda/{name}/eda_overview.png"\n    if os.path.exists(p):\n        print(name); display(Image(p))'),
        ("md", "## 5. Look at sample images with their labels"),
        ("code", '''import random, cv2, matplotlib.pyplot as plt
from pothole.config import PROCESSED_DIR
from pothole.data.yolo_utils import find_split_dirs, list_images, read_labels, label_path_for

def show_samples(dataset, n=6, seed=0):
    imgs = list_images(find_split_dirs(PROCESSED_DIR / dataset)["train"])
    random.Random(seed).shuffle(imgs)
    fig, axes = plt.subplots(2, n // 2, figsize=(16, 7))
    for ax, p in zip(axes.flat, imgs[:n]):
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]
        for _, xc, yc, bw, bh in read_labels(label_path_for(p)):
            cv2.rectangle(im, (int((xc-bw/2)*w), int((yc-bh/2)*h)), (int((xc+bw/2)*w), int((yc+bh/2)*h)), (255, 0, 0), 3)
        ax.imshow(im); ax.set_title(p.name[:30], fontsize=8); ax.axis("off")
    fig.suptitle(dataset); plt.tight_layout(); plt.show()

show_samples("roboflow")
try: show_samples("rdd_india")
except Exception as e: print("rdd_india not prepared:", e)'''),
        ("md", "## 6. Key observations (fill in)\n\n| Observation | Evidence | Decision |\n|---|---|---|\n| e.g. many small potholes (<1% of image) | `small_boxes_pct` | keep imgsz 640, check small-object recall |\n| e.g. Roboflow images are mostly non-Indian roads | sample grid | add RDD2022 India in the advanced model |\n| e.g. duplicate images across splits | `duplicate_across_splits` | remove before training |\n| | | |"),
    ],
    "02_baseline_model.ipynb": [
        ("md", "# 02 · Baseline model: YOLOv8n\n\nYOLOv8 nano, mostly default settings, Roboflow data only. This sets the bar the advanced model must beat.\n\nConfig: `configs/baseline.yaml`. On CPU, use `--epochs 5` for a quick check; run the full 50 epochs on a Colab GPU (~15-20 min)."),
        ("code", SETUP),
        ("code", "import torch; print('GPU:', torch.cuda.is_available())\n!python -m pothole.train --config configs/baseline.yaml\n# quick CPU check: !python -m pothole.train --config configs/baseline.yaml --epochs 5 --device cpu"),
        ("md", "## Training curves"),
        ("code", 'from IPython.display import Image, display\nfor f in ["results.png", "PR_curve.png", "confusion_matrix.png", "val_batch0_pred.jpg"]:\n    p = f"runs/baseline_yolov8n/{f}"\n    if os.path.exists(p): print(f); display(Image(p, width=900))'),
        ("code", "import pandas as pd\npd.read_csv('reports/experiments.csv')"),
    ],
    "03_advanced_model.ipynb": [
        ("md", "# 03 · Advanced model: YOLOv8s + Indian roads + tuned training\n\nChanges vs baseline (one reason each, for the slides):\n\n| Change | Why |\n|---|---|\n| YOLOv8**s** instead of **n** | more capacity for varied road textures |\n| + RDD2022 India potholes and background images | Indian roads, lighting and cameras; fewer false alarms on cracks/shadows |\n| AdamW, lr0 0.001, cosine LR, 120 epochs | smoother convergence on a bigger dataset |\n| Road-specific augmentation (no vertical flip, strong brightness/saturation) | dashcam lighting varies; roads are never upside down |\n\nConfig: `configs/advanced.yaml`. Run on a GPU (Colab T4: ~1-1.5 h)."),
        ("code", SETUP),
        ("code", "!python -m pothole.train --config configs/advanced.yaml"),
        ("md", "## Optional: hyperparameter experiments\nEach run is logged to `reports/experiments.csv`. Change one thing at a time."),
        ("code", "# !python -m pothole.train --config configs/advanced.yaml --name adv_img800 --imgsz 800\n# !python -m pothole.train --config configs/advanced.yaml --name adv_yolo11s --model yolo11s.pt\n# !python -m pothole.train --config configs/advanced.yaml --name adv_no_mixup --mixup 0.0"),
        ("code", "import pandas as pd\npd.read_csv('reports/experiments.csv')[['name','model','epochs','imgsz','optimizer','precision','recall','mAP50','mAP50_95','train_minutes']]"),
        ("md", "## Optimisation: export for deployment"),
        ("code", "!python -m pothole.export --model models/advanced_yolov8s_india.pt --formats onnx\n# Android / edge: !python -m pothole.export --model models/advanced_yolov8s_india.pt --formats tflite --int8 --data data/processed/merged/data.yaml"),
    ],
    "04_evaluation_and_error_analysis.ipynb": [
        ("md", "# 04 · Evaluation and error analysis\n\nThe brief asks for more than one headline number: overall metrics, results by group, failure cases, latency and whether the model is usable.\n\nBoth models are tested on **both** test sets, so the comparison is fair:\n- Roboflow test: the brief's original benchmark\n- Merged test: includes Indian roads, closer to real use in India"),
        ("code", SETUP),
        ("code", "MODELS = 'models/baseline_yolov8n.pt models/advanced_yolov8s_india.pt'\n!python -m pothole.evaluate --models {MODELS} --data data/processed/roboflow/data.yaml\n!python -m pothole.evaluate --models {MODELS} --data data/processed/merged/data.yaml"),
        ("code", "from IPython.display import Markdown, display\nfor tag in ['roboflow', 'merged']:\n    p = f'reports/eval/{tag}/comparison.md'\n    if os.path.exists(p): display(Markdown(f'### Test set: {tag}\\n' + open(p).read()))"),
        ("md", "## Failure cases\nGreen = ground truth (thick green = missed), blue = correct detection, red = false alarm."),
        ("code", 'import glob\nfrom IPython.display import Image, display\nfor p in sorted(glob.glob("reports/eval/merged/advanced_yolov8s_india/errors/*.jpg"))[:8]:\n    print(os.path.basename(p)); display(Image(p, width=700))'),
        ("md", "## Choosing the confidence threshold\nFor inspection, missing a pothole costs more than a false alarm (a person checks every alarm anyway), so pick a threshold that favours **recall**."),
        ("code", '''import json, subprocess
rows = []
for c in [0.1, 0.15, 0.25, 0.35, 0.5]:
    subprocess.run([sys.executable, "-m", "pothole.evaluate", "--models", "models/advanced_yolov8s_india.pt",
                    "--data", "data/processed/merged/data.yaml", "--conf", str(c), "--n-errors", "0",
                    "--tag", f"conf_{c}"], check=True, capture_output=True)
    m = json.load(open(f"reports/eval/conf_{c}/advanced_yolov8s_india/metrics.json"))
    rows.append({"conf": c, "precision": m["precision_at_conf"], "recall": m["recall_at_conf"],
                 "f1": m["f1_at_conf"], "FP": m["FP"], "FN": m["FN"]})
import pandas as pd; pd.DataFrame(rows)'''),
        ("md", "## Is it usable? (fill in)\n\n- **Overall:** mAP@0.5 = … on Indian roads; recall at the chosen threshold = …\n- **Weak spots:** small/far potholes (recall by size), water-filled potholes, night/shadow images …\n- **Speed:** … ms/image on CPU → about … images/s, enough for a phone dashcam at 1-2 frames/s\n- **Verdict:** suitable to *rank roads for inspection*, not to certify a road as safe.\n- **Responsible use:** every flagged pothole is checked by a person before repair orders; \"no detection\" does not mean \"no pothole\"."),
    ],
}


def build(out_dir: Path = OUT) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, cells in NOTEBOOKS.items():
        book = nbf.v4.new_notebook()
        book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "accelerator": "GPU", "colab": {"provenance": []}}
        book.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                      for kind, src in cells]
        path = out_dir / name
        nbf.write(book, path)
        written.append(path)
        print("wrote", path.name)
    return written


if __name__ == "__main__":
    build()
