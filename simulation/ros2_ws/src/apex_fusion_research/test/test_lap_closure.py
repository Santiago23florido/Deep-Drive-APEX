import math

import numpy as np

from apex_fusion_research.core.lap_closure import LapCounter, cut_loop
from apex_fusion_research.core.race_config import ClosureConfig


def circle_track(r=8.0, laps=3.05, ds=0.1, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    s = np.arange(0.0, laps * 2 * math.pi * r, ds)
    a = s / r - math.pi / 2  # start at (0, -r) heading +x
    xy = np.column_stack((r * np.cos(a), r * np.sin(a) + r))
    yaw = a + math.pi / 2
    return s, xy + rng.normal(0, noise, xy.shape), yaw


def test_counts_each_lap_once():
    s, xy, yaw = circle_track()
    lc = LapCounter((0.0, 0.0, 0.0), ClosureConfig(arm_time_s=0.0))
    crossings = [c for k in range(len(s)) if (c := lc.update(k * 0.05, (xy[k, 0], xy[k, 1], yaw[k]), s[k])) is not None]
    assert len(crossings) == 3
    L = 2 * math.pi * 8.0
    for i, c in enumerate(crossings, start=1):
        assert abs(c.travel - i * L) < 0.5 and c.kind == "line"


def test_not_armed_before_minimum_travel():
    s, xy, yaw = circle_track(r=2.0, laps=2.5)  # 12.6 m per lap < arm distance 20 m
    lc = LapCounter((0.0, 0.0, 0.0), ClosureConfig(arm_time_s=0.0))
    crossings = [c for k in range(len(s)) if (c := lc.update(0.0, (xy[k, 0], xy[k, 1], yaw[k]), s[k])) is not None]
    assert all(c.travel >= 20.0 for c in crossings)


def test_reverse_pass_rejected():
    s, xy, yaw = circle_track(laps=1.2)
    lc = LapCounter((0.0, 0.0, 0.0), ClosureConfig(arm_time_s=0.0))
    # Drive the circle backwards (opposite heading): no crossing counts.
    xy, yaw = xy[::-1], yaw[::-1] + math.pi
    assert all(lc.update(0.0, (xy[k, 0], xy[k, 1], yaw[k]), s[k]) is None for k in range(len(s)))


def test_radius_fallback_when_the_line_is_missed():
    s, xy, yaw = circle_track(laps=1.3, noise=0.0)
    xy = xy.copy()
    lc = LapCounter((0.0, -1.7, 0.0), ClosureConfig(arm_time_s=0.0, lateral_tol_m=0.5))  # start line 1.7 m off the loop
    out = [c for k in range(len(s)) if (c := lc.update(0.0, (xy[k, 0], xy[k, 1], yaw[k]), s[k])) is not None]
    assert len(out) == 1 and out[0].kind == "radius"


def test_cut_loop_length_and_seam():
    s, xy, yaw = circle_track(laps=1.4, noise=0.0)
    loop, seam = cut_loop(xy, (0.0, 0.0, 0.0), ClosureConfig())
    length = np.sum(np.hypot(np.diff(loop[:, 0]), np.diff(loop[:, 1])))
    assert abs(length - 2 * math.pi * 8.0) < 0.2
    assert abs(seam["lateral"]) < 0.05 and abs(seam["heading_deg"]) < 3.0
