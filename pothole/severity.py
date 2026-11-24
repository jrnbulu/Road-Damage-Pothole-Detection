"""Rule-based pothole severity (the brief's allowed extension).

This is a heuristic, not a learned model. It estimates how big a pothole is on
the road using its bounding box, corrected for perspective:
a pothole near the bottom of a dashcam frame is close to the camera and looks
bigger than the same pothole far up the road.

  adjusted_size = box_area_fraction / max(box_bottom_y, MIN_DEPTH) ** 2

Potholes in the middle of the frame (the driving lane) get a priority bonus.
Thresholds are starting points: calibrate them on a few images with known
real-world pothole sizes and record the values in the report.
"""
from dataclasses import asdict, dataclass

MIN_DEPTH = 0.35          # stop tiny far-away boxes from being blown up
HIGH_THRESHOLD = 0.04     # adjusted_size at or above -> High
MEDIUM_THRESHOLD = 0.01   # adjusted_size at or above -> Medium
LANE_X_RANGE = (0.25, 0.75)
LANE_BONUS = 1.25
LEVELS = ["Low", "Medium", "High"]


@dataclass
class Severity:
    level: str             # Low / Medium / High
    priority: int          # 0-100, for sorting a repair list
    area_fraction: float   # raw box area / image area
    adjusted_size: float   # perspective-corrected size
    in_lane: bool


def score_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Severity:
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    area_fraction = (bw * bh) / float(img_w * img_h)
    bottom = min(1.0, y2 / img_h)
    adjusted = area_fraction / max(bottom, MIN_DEPTH) ** 2

    cx = (x1 + x2) / 2 / img_w
    in_lane = LANE_X_RANGE[0] <= cx <= LANE_X_RANGE[1]

    if adjusted >= HIGH_THRESHOLD:
        level = "High"
    elif adjusted >= MEDIUM_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    # sqrt keeps priority roughly proportional to pothole width rather than area
    priority = min(1.0, (adjusted / (2 * HIGH_THRESHOLD)) ** 0.5) * 100
    if in_lane:
        priority = min(100.0, priority * LANE_BONUS)
    return Severity(level, round(priority), round(area_fraction, 5), round(adjusted, 5), in_lane)


def summarise(severities: list[Severity]) -> dict:
    """Image-level road condition from all detected potholes."""
    if not severities:
        return {"condition": "Good", "pothole_count": 0, "max_severity": None, "max_priority": 0,
                "counts": {lvl: 0 for lvl in LEVELS}}
    counts = {lvl: sum(s.level == lvl for s in severities) for lvl in LEVELS}
    worst = max(severities, key=lambda s: (LEVELS.index(s.level), s.priority))
    if counts["High"] or len(severities) >= 5:
        condition = "Poor - repair urgently"
    elif counts["Medium"] or len(severities) >= 2:
        condition = "Fair - schedule repair"
    else:
        condition = "Minor damage - monitor"
    return {"condition": condition, "pothole_count": len(severities), "max_severity": worst.level,
            "max_priority": worst.priority, "counts": counts}


def to_dict(s: Severity) -> dict:
    return asdict(s)
