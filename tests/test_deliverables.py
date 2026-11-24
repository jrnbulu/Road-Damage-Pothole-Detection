"""Checks that every deliverable required by the capstone brief (§12) is present and well-formed."""
import json
import re

import pytest

from pothole.config import ROOT

REQUIRED_FILES = [
    "README.md", "requirements.txt", "Dockerfile", "docker-compose.yml",
    "docs/scope.md", "docs/deliverables_checklist.md", "docs/evaluation_report.md",
    "docs/presentation_outline.md",
    "pothole/data/download.py", "pothole/data/rdd2022_india.py", "pothole/data/validate.py",
    "pothole/train.py", "pothole/evaluate.py", "pothole/severity.py", "pothole/inference.py",
    "pothole/export.py", "app/streamlit_app.py", "api/main.py",
    "configs/baseline.yaml", "configs/advanced.yaml",
    "scripts/build_presentation.py", "scripts/smoke_test.py",
]
NOTEBOOKS = ["01_data_access_and_eda", "02_baseline_model", "03_advanced_model",
             "04_evaluation_and_error_analysis"]


@pytest.mark.parametrize("rel", REQUIRED_FILES)
def test_required_file_exists(rel):
    assert (ROOT / rel).is_file(), f"missing deliverable: {rel}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 3
    assert nb["cells"][0]["cell_type"] == "markdown"


def test_readme_has_setup_and_run_instructions():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for section in ("## Setup", "## Run it step by step", "## Data sources", "## Tests"):
        assert section in text


def test_checklist_covers_every_brief_deliverable():
    text = (ROOT / "docs" / "deliverables_checklist.md").read_text(encoding="utf-8").lower()
    for item in ("scope document", "eda notebook", "baseline model", "advanced model",
                 "evaluation report", "deployment prototype", "presentation deck", "readme"):
        assert item in text


def test_presentation_builds(tmp_path):
    pytest.importorskip("pptx")
    from scripts.build_presentation import build
    out = build("Test Team", tmp_path / "deck.pptx")
    from pptx import Presentation
    assert len(Presentation(str(out)).slides) == 8  # brief §15: 8 slides


def test_no_third_party_tool_references():
    """Submission files must not mention the tools used to write them."""
    pattern = re.compile(r"claude|anthropic", re.IGNORECASE)
    skip = {".venv", "data", "runs", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or skip & set(p.relative_to(ROOT).parts) or p.name == "test_deliverables.py":
            continue
        if p.suffix.lower() in {".py", ".md", ".txt", ".yaml", ".yml", ".ipynb", ".json", ".toml", ""} \
                or p.name in {"Dockerfile", ".gitignore", ".dockerignore"}:
            if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(p.relative_to(ROOT)))
    assert not hits, f"tool references found in: {hits}"
