import math

from apex_fusion_research.core.clearance import FootprintClearance
from apex_fusion_research.core.map_metrics import se2_compose
from apex_fusion_research.core.pose_correction import CorrectionFilter
from apex_fusion_research.core.race_config import CorrectionConfig


def test_step_absorbed_at_bounded_rate():
    f = CorrectionFilter(CorrectionConfig())
    odom = (5.0, 0.0, 0.0)
    f.update(0.0, (0.0, 0.0, 0.0), odom)
    positions = []
    for k in range(1, 101):
        app = f.update(0.02 * k, (0.2, 0.0, 0.0), odom)
        positions.append(se2_compose(app, odom)[0])
    steps = [b - a for a, b in zip([5.0] + positions, positions)]
    assert max(steps) <= 0.4 * 0.02 + 1e-9
    assert abs(positions[-1] - 5.2) < 1e-3


def test_yaw_correction_far_from_origin_does_not_jump():
    f = CorrectionFilter(CorrectionConfig())
    odom = (20.0, 0.0, 0.0)
    f.update(0.0, (0.0, 0.0, 0.0), odom)
    raw = (0.0, 0.0, math.radians(2.0))  # rotates the car pose by 0.7 m laterally
    app = f.update(0.02, raw, odom)
    p = se2_compose(app, odom)
    assert math.hypot(p[0] - 20.0, p[1]) <= 0.4 * 0.02 + 1e-9


def test_footprint_clearance_sign():
    pts = [(1.0, 0.0), (0.0, 0.5)]
    fc = FootprintClearance(pts, 0.46, 0.32)
    assert abs(fc(0.0, 0.0, 0.0) - min(1.0 - 0.23, 0.5 - 0.16)) < 1e-9
    assert fc(0.9, 0.0, 0.0) < 0.0  # the point is inside the footprint
