"""Data preparation: YOLO utils, RDD2022 conversion, merge, validation, remote ZIP download."""
import io
import zipfile

import pytest
import yaml

from conftest import make_split
from pothole.data import merge as merge_mod
from pothole.data import remote_zip, yolo_utils
from pothole.data.download import locate_dataset_root
from pothole.data.rdd2022_india import split_samples, voc_to_yolo
from pothole.data.validate import analyse


# ---- yolo_utils ----
def test_find_split_dirs_maps_valid_to_val(yolo_dataset):
    assert set(yolo_utils.find_split_dirs(yolo_dataset)) == {"train", "val", "test"}


def test_write_data_yaml(yolo_dataset):
    cfg = yaml.safe_load(yolo_utils.write_data_yaml(yolo_dataset).read_text())
    assert cfg["names"] == {0: "pothole"}
    assert cfg["val"].replace("\\", "/").endswith("valid/images")


def test_write_data_yaml_needs_val(tmp_path):
    make_split(tmp_path, "train", 1)
    with pytest.raises(SystemExit):
        yolo_utils.write_data_yaml(tmp_path)


def test_read_labels(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.1 0.2\nbad line\n\n0 0.1 0.1 0.05 0.05\n")
    assert yolo_utils.read_labels(f) == [(0, 0.5, 0.5, 0.1, 0.2), (0, 0.1, 0.1, 0.05, 0.05)]
    assert yolo_utils.read_labels(tmp_path / "missing.txt") == []


def test_label_path_for(tmp_path):
    p = tmp_path / "train" / "images" / "x.jpg"
    assert yolo_utils.label_path_for(p) == tmp_path / "train" / "labels" / "x.txt"


def test_locate_dataset_root_finds_nested_export(tmp_path):
    make_split(tmp_path / "Pothole.v1", "train", 1)
    assert locate_dataset_root(tmp_path) == tmp_path / "Pothole.v1"


# ---- RDD2022 conversion ----
VOC = """<annotation><size><width>100</width><height>50</height></size>
  <object><name>D40</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>20</ymax></bndbox></object>
  <object><name>D00</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
  <object><name>D40</name><bndbox><xmin>50</xmin><ymin>5</ymin><xmax>51</xmax><ymax>6</ymax></bndbox></object>
  <object><name>D40</name><bndbox><xmin>120</xmin><ymin>40</ymin><xmax>90</xmax><ymax>60</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo_keeps_potholes_drops_tiny_and_clips(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text(VOC)
    lines, others = voc_to_yolo(xml)
    assert others == 1
    assert lines[0] == "0 0.200000 0.300000 0.200000 0.200000"
    assert len(lines) == 2  # 1x1 box dropped; swapped/out-of-image box clipped to (90,40)-(100,50)
    assert lines[1] == "0 0.950000 0.900000 0.100000 0.200000"


def test_split_samples_is_80_10_10_and_deterministic():
    items = list(range(100))
    a, b = split_samples(items, 42), split_samples(items, 42)
    assert a == b
    assert (len(a["train"]), len(a["valid"]), len(a["test"])) == (80, 10, 10)
    assert sorted(a["train"] + a["valid"] + a["test"]) == items


# ---- merge ----
def test_merge_prefixes_and_keeps_splits(tmp_path):
    base = tmp_path / "processed"
    make_split(base / "a", "train", 2)
    make_split(base / "a", "valid", 1, start=5)
    make_split(base / "b", "train", 3, start=10)
    make_split(base / "b", "valid", 1, start=20)
    make_split(base / "b", "test", 1, start=30)
    counts = merge_mod.merge(["a", "b"], base / "merged", base=base)
    assert counts[("a", "train")] == 2 and counts[("b", "test")] == 1
    train_imgs = sorted(p.name for p in (base / "merged" / "train" / "images").iterdir())
    assert train_imgs[0].startswith("a__") and len(train_imgs) == 5
    assert (base / "merged" / "train" / "labels" / "b__10.txt").exists()
    assert (base / "merged" / "data.yaml").exists()


def test_merge_missing_source(tmp_path):
    with pytest.raises(SystemExit):
        merge_mod.merge(["nope"], tmp_path / "out", base=tmp_path)


# ---- validate / EDA ----
def test_analyse_counts_and_flags_issues(tmp_path):
    base, reports = tmp_path / "processed", tmp_path / "reports"
    make_split(base / "d", "train", 3, start=0)
    make_split(base / "d", "valid", 1, start=10, boxes="0 0.5 0.5 0.001 0.001\n1 1.5 0.5 0.2 0.2\n")
    # duplicate image across splits
    (base / "d" / "valid" / "images" / "dup.png").write_bytes((base / "d" / "train" / "images" / "0.png").read_bytes())
    stats = analyse("d", base=base, reports=reports)
    assert stats["splits"]["train"] == {"images": 3, "boxes": 3, "images_without_potholes": 0}
    issues = stats["issues"]
    assert issues["tiny_box"]["count"] == 1
    assert issues["unexpected_class"]["count"] == 1
    assert issues["out_of_range_box"]["count"] == 1
    assert issues["missing_label_file"]["count"] == 1  # dup.png has no label
    assert issues["duplicate_across_splits"]["count"] == 1
    assert (reports / "eda" / "d" / "eda_overview.png").exists()
    assert (reports / "eda" / "d" / "stats.json").exists()


# ---- remote ZIP (no network: Range requests served from memory) ----
@pytest.fixture
def fake_remote(monkeypatch):
    inner = bytes(range(256)) * 5000  # ~1.3 MB payload standing in for India.zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("RDD2022/Czech.zip", b"x" * 1000)
        zf.writestr("RDD2022/India.zip", inner)
        zf.writestr("RDD2022/Japan.zip", b"y" * 1000)
    blob = buf.getvalue()
    calls = []

    def get_range(url, start, end, retries=6):
        calls.append((start, end))
        return blob[start:end + 1]

    monkeypatch.setattr(remote_zip, "_get_range", get_range)
    monkeypatch.setattr(remote_zip, "_remote_size", lambda url: len(blob))
    monkeypatch.setattr(remote_zip, "CHUNK", 200_000)
    return inner, blob, calls


def test_http_range_file_reads_like_a_file(fake_remote):
    _, blob, _ = fake_remote
    f = remote_zip.HttpRangeFile("u", block=1000)
    f.seek(-10, 2)
    assert f.read(10) == blob[-10:]
    f.seek(5)
    assert f.read(3) == blob[5:8] and f.tell() == 8


def test_download_member_extracts_only_that_member(fake_remote, tmp_path):
    inner, blob, calls = fake_remote
    out = remote_zip.download_member("u", "RDD2022/India.zip", tmp_path / "India.zip")
    assert out.read_bytes() == inner
    assert not (tmp_path / "India.zip.part").exists()


def test_download_member_resumes_partial_file(fake_remote, tmp_path):
    inner, _, calls = fake_remote
    part = tmp_path / "India.zip.part"
    part.write_bytes(inner[:500_000])
    remote_zip.download_member("u", "RDD2022/India.zip", tmp_path / "India.zip")
    assert (tmp_path / "India.zip").read_bytes() == inner


def test_download_member_rejects_corrupt_resume(fake_remote, tmp_path):
    inner, _, _ = fake_remote
    (tmp_path / "India.zip.part").write_bytes(b"\0" * 1000)  # wrong bytes -> CRC fails
    with pytest.raises(IOError, match="CRC"):
        remote_zip.download_member("u", "RDD2022/India.zip", tmp_path / "India.zip")
    assert not (tmp_path / "India.zip.part").exists()
