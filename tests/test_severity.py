"""Rule-based severity scoring."""
import pytest

from pothole import severity as sev

W, H = 1280, 720


def test_big_close_centred_pothole_is_high_priority():
    s = sev.score_box(400, 500, 900, 720, W, H)
    assert s.level == "High" and s.in_lane and s.priority == 100


def test_small_far_edge_pothole_is_low():
    s = sev.score_box(100, 300, 120, 310, W, H)
    assert s.level == "Low" and not s.in_lane and s.priority < 20


def test_medium_band():
    # 200x100 box with bottom at 720 -> area 0.0217 -> Medium
    s = sev.score_box(540, 620, 740, 720, W, H)
    assert s.level == "Medium"


def test_perspective_makes_far_box_count_bigger():
    near = sev.score_box(600, 600, 700, 700, W, H)
    far = sev.score_box(600, 200, 700, 300, W, H)
    assert far.adjusted_size > near.adjusted_size


def test_lane_bonus_raises_priority():
    side = sev.score_box(10, 600, 110, 680, W, H)
    lane = sev.score_box(590, 600, 690, 680, W, H)
    assert lane.adjusted_size == pytest.approx(side.adjusted_size)
    assert lane.priority > side.priority


def test_degenerate_box_is_safe():
    s = sev.score_box(50, 50, 40, 40, W, H)  # inverted coordinates
    assert s.area_fraction == 0 and s.level == "Low"


def test_priority_stays_in_range():
    for box in [(0, 0, W, H), (0, 0, 1, 1), (600, 700, 700, 720)]:
        assert 0 <= sev.score_box(*box, W, H).priority <= 100


def test_summary_empty():
    s = sev.summarise([])
    assert s["condition"] == "Good" and s["pothole_count"] == 0 and s["max_severity"] is None


def test_summary_counts_and_worst():
    items = [sev.score_box(400, 500, 900, 720, W, H), sev.score_box(0, 0, 5, 5, W, H)]
    s = sev.summarise(items)
    assert s["pothole_count"] == 2 and s["max_severity"] == "High"
    assert s["counts"] == {"Low": 1, "Medium": 0, "High": 1}
    assert s["condition"].startswith("Poor")


def test_summary_many_small_potholes_is_poor():
    items = [sev.score_box(0, 0, 5, 5, W, H)] * 5
    assert sev.summarise(items)["condition"].startswith("Poor")


def test_summary_single_low_is_minor():
    assert sev.summarise([sev.score_box(0, 0, 5, 5, W, H)])["condition"].startswith("Minor")


def test_to_dict_is_json_ready():
    d = sev.to_dict(sev.score_box(0, 0, 100, 100, W, H))
    assert set(d) == {"level", "priority", "area_fraction", "adjusted_size", "in_lane"}
