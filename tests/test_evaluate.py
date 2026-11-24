"""Evaluation maths: IoU, matching, precision/recall/F1 and the comparison table."""
import pytest

import pothole.evaluate as ev


def test_iou_cases():
    assert ev.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert ev.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert ev.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert ev.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0  # zero area, no division error


def test_match_counts_tp_fp_fn():
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
    preds = [(1, 1, 10, 10, 0.9), (100, 100, 110, 110, 0.8)]
    pairs, fn, fp = ev.match(gts, preds)
    assert len(pairs) == 1 and fn == [1] and fp == [1]


def test_match_one_gt_two_preds_highest_conf_wins():
    gts = [(0, 0, 10, 10)]
    preds = [(0, 0, 10, 10, 0.3), (0, 0, 10, 10, 0.9)]
    pairs, fn, fp = ev.match(gts, preds)
    assert pairs[0][1] == 1 and fp == [0] and fn == []


def test_match_below_iou_threshold_is_not_a_hit():
    pairs, fn, fp = ev.match([(0, 0, 10, 10)], [(6, 0, 16, 10, 0.9)])  # IoU 0.25
    assert pairs == [] and fn == [0] and fp == [0]


def test_match_empty_inputs():
    assert ev.match([], []) == ([], [], [])


def test_prf():
    p, r, f1 = ev.prf(8, 2, 2)
    assert (p, r) == (0.8, 0.8) and f1 == pytest.approx(0.8)
    assert ev.prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_write_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_DIR", tmp_path)
    row = {"model": "m", "mAP50": 0.5, "mAP50_95": 0.3, "precision": 0.6, "recall": 0.7,
           "f1_at_conf": 0.65, "mean_iou_matched": 0.8, "FP": 3, "FN": 2, "latency_ms_mean": 12.5,
           "recall_by_size": {"small": (0.5, 4)}}
    path = ev.write_comparison([row], tmp_path / "data.yaml", "t")
    text = path.read_text()
    assert path == tmp_path / "eval" / "t" / "comparison.md"
    assert "| m | 0.500 | 0.300 |" in text and "small: 0.5 (n=4)" in text
