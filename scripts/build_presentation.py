"""Build the 8-slide final presentation (brief §15) as docs/presentation.pptx.

    python scripts/build_presentation.py [--team "Name 1, Name 2, ..."]

Slides follow docs/presentation_outline.md. Plots and tables from reports/ and
runs/ are inserted automatically when they exist; otherwise a placeholder
note says which command produces them. Re-run after training and evaluation.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pothole.config import REPORTS_DIR, RUNS_DIR  # noqa: E402

ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
W, H = Inches(13.333), Inches(7.5)


def add_slide(prs, title: str, bullets: list[str], image: Path | None = None, note: str = ""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), W - Inches(1), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text, p.font.size, p.font.bold, p.font.color.rgb = title, Pt(32), True, ACCENT

    text_w = Inches(6.2) if image else W - Inches(1)
    body = slide.shapes.add_textbox(Inches(0.5), Inches(1.4), text_w, H - Inches(2)).text_frame
    body.word_wrap = True
    for i, b in enumerate(bullets):
        para = body.paragraphs[0] if i == 0 else body.add_paragraph()
        para.text, para.font.size, para.space_after = f"• {b}", Pt(18), Pt(8)

    if image and image.exists():
        pic = slide.shapes.add_picture(str(image), Inches(6.9), Inches(1.4), width=Inches(6))
        if pic.height > H - Inches(1.8):  # keep tall images on the slide
            ratio = (H - Inches(1.8)) / pic.height
            pic.height, pic.width = Emu(int(pic.height * ratio)), Emu(int(pic.width * ratio))
    elif image:
        ph = slide.shapes.add_textbox(Inches(6.9), Inches(3), Inches(6), Inches(1)).text_frame
        ph.text = f"[Image appears here after running the pipeline: {image.relative_to(ROOT)}]"
        ph.paragraphs[0].font.size, ph.paragraphs[0].font.color.rgb = Pt(12), GREY
    if note:
        slide.notes_slide.notes_text_frame.text = note
    return slide


def first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.exists()), paths[0])


def read_split_counts(dataset: str) -> str:
    stats = REPORTS_DIR / "eda" / dataset / "stats.json"
    if not stats.exists():
        return f"{dataset}: run `python -m pothole.data.validate --dataset {dataset}`"
    s = json.loads(stats.read_text())["splits"]
    parts = ", ".join(f"{k} {v['images']} imgs / {v['boxes']} boxes" for k, v in s.items())
    return f"{dataset}: {parts}"


def results_bullets() -> list[str]:
    out = []
    for tag in ("roboflow", "merged"):
        table = REPORTS_DIR / "eval" / tag
        found = False
        for m in sorted(table.glob("*/metrics.json")) if table.exists() else []:
            d = json.loads(m.read_text())
            out.append(f"[{tag}] {m.parent.name}: mAP50 {d['mAP50']:.3f}, mAP50-95 {d['mAP50_95']:.3f}, "
                       f"P {d['precision']:.2f}, R {d['recall']:.2f}, {d['latency_ms_mean']} ms/img")
            found = True
        if not found:
            out.append(f"[{tag}] run `python -m pothole.evaluate ... --data data/processed/{tag}/data.yaml`")
    return out


def experiment_bullets() -> list[str]:
    log = REPORTS_DIR / "experiments.csv"
    if not log.exists():
        return ["Run training to fill this slide (reports/experiments.csv)"]
    with log.open() as f:
        rows = list(csv.DictReader(f))
    return [f"{r['name']}: {r['model']}, {r['epochs']} ep, imgsz {r['imgsz']}, opt {r['optimizer']} "
            f"→ mAP50 {float(r['mAP50'] or 0):.3f} ({r['train_minutes']} min)" for r in rows[-6:]]


def build(team: str, out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    errors = sorted((REPORTS_DIR / "eval" / "merged").glob("*/errors/*.jpg"))

    add_slide(prs, "Road Damage & Pothole Detection with YOLOv8", [
        f"Team: {team}",
        "Problem: potholes cause accidents and vehicle damage; manual road surveys are slow and inconsistent",
        "Goal: detect potholes in road images, rate severity, and rank repairs for city maintenance teams",
        "Users: municipal engineers, smart-city centres, highway operators, road safety analysts",
    ], note="Open with one MoRTH statistic on pothole-related deaths (cite year).")
    add_slide(prs, "Dataset", [
        "Roboflow public pothole dataset (brief's primary source), YOLOv8 export",
        "RDD2022 India, class D40 = pothole (CC BY-SA 4.0), background images kept to cut false alarms",
        read_split_counts("roboflow"), read_split_counts("rdd_india"),
        "Each source keeps its own train/val/test split, so no test leakage",
    ])
    add_slide(prs, "EDA and preprocessing", [
        "Checks: unreadable images, missing labels, out-of-range and tiny boxes, cross-split duplicates",
        "Box size distribution shows many small potholes, so recall on small objects matters",
        "Potholes cluster in the lower-middle of the frame (driving lane)",
        "Labels converted to YOLO format; data.yaml written with absolute paths",
    ], image=REPORTS_DIR / "eda" / "rdd_india" / "eda_overview.png")
    add_slide(prs, "Models: baseline vs advanced", [
        "Baseline: YOLOv8n, default settings, Roboflow only, 50 epochs",
        "Advanced: YOLOv8s, Roboflow + India, AdamW + cosine LR, 120 epochs",
        "Road-specific augmentation: strong brightness/saturation, no vertical flip",
        "Severity: rule-based, perspective-corrected box size + lane bonus",
    ])
    add_slide(prs, "Experiments and tuning", experiment_bullets(),
              image=first_existing(RUNS_DIR / "advanced_yolov8s_india" / "results.png",
                                   RUNS_DIR / "baseline_yolov8n" / "results.png"))
    add_slide(prs, "Results and error analysis", results_bullets() + [
        "See reports/eval/<tag>/comparison.md for recall by pothole size",
    ], image=errors[0] if errors else REPORTS_DIR / "eval" / "merged" / "<model>" / "errors" / "missed__x.jpg")
    add_slide(prs, "Deployment demo", [
        "Streamlit: upload → boxes coloured by severity → repair priority list → CSV/JSON download",
        "FastAPI: /predict (JSON) and /predict/image (annotated JPEG), Docker packaging",
        "Model optimisation: ONNX export for CPU servers, TFLite for phones",
        "Demo set: 3 good predictions + 1 known failure case (brief §10)",
    ], note="Keep a screen recording as backup in case the live demo fails.")
    add_slide(prs, "Limitations, ethics and future scope", [
        "Decision aid only: every detection is verified by a person before repair",
        "Known failures: water-filled potholes, shadows, small far-away potholes, night images",
        "Severity is a heuristic, not a depth measurement",
        "Privacy: blur faces and number plates before storing dashcam images",
        "Next: dashcam video + GPS → ward-level hotspot map → pilot with a city or highway operator",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"Saved {out} ({len(prs.slides)} slides)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="Team member names")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "presentation.pptx")
    args = ap.parse_args()
    build(args.team, args.out)
