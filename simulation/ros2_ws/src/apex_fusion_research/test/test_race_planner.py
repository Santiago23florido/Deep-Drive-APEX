import math

import numpy as np

from apex_fusion_research.core.occupancy import GridInfo, read_pgm_yaml, write_pgm_yaml
from apex_fusion_research.core.race_config import PlannerConfig, RaceConfig, VehicleLimits
from apex_fusion_research.core.race_planner import Grid, closed_reference, corridor, layers, plan_race, smoothed_curvature

VEH = VehicleLimits()
RES = 0.05


def stadium(straight=8.0, radius=3.0, ds=0.05):
    """Closed stadium centreline, counter-clockwise, starting mid bottom straight heading +x."""
    pts = []
    for x in np.arange(0.0, straight / 2, ds):
        pts.append((x, -radius))
    for a in np.arange(-math.pi / 2, math.pi / 2, ds / radius):
        pts.append((straight / 2 + radius * math.cos(a), radius * math.sin(a)))
    for x in np.arange(straight / 2, -straight / 2, -ds):
        pts.append((x, radius))
    for a in np.arange(math.pi / 2, 3 * math.pi / 2, ds / radius):
        pts.append((-straight / 2 + radius * math.cos(a), radius * math.sin(a)))
    for x in np.arange(-straight / 2, 0.0, ds):
        pts.append((x, -radius))
    return np.array(pts)


def offset(cl, d):
    cl = stadium(ds=0.01) if len(cl) == len(stadium()) else cl  # dense walls: no gaps between cells
    t = np.gradient(cl, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack((-t[:, 1], t[:, 0]))
    return cl + d * n


def make_grid(points_occ, free_poly_pts, lim=(-10, 10, -8, 8)):
    x0, x1, y0, y1 = lim
    w, h = int((x1 - x0) / RES), int((y1 - y0) / RES)
    data = np.full((h, w), 0, dtype=np.int16)  # everything observed free
    for p in points_occ:
        c, r = int((p[0] - x0) / RES), int((p[1] - y0) / RES)
        if 0 <= r < h and 0 <= c < w:
            data[r, c] = 100
    return Grid(data, RES, x0, y0)


def test_min_curvature_within_bounds_and_smoother():
    cl = stadium()
    walls = np.vstack((offset(cl, 1.0), offset(cl, -1.0)))  # 2 m lane, walls both sides
    grid = make_grid(walls, None)
    log = {"xy": cl, "left": np.ones(len(cl), bool), "right": np.ones(len(cl), bool)}
    plan = plan_race(grid, cl, (0.0, -3.0, 0.0), log, VEH, PlannerConfig(), RaceConfig(), 1.5)
    assert np.all(plan.alpha >= plan.cor.lo - 1e-6) and np.all(plan.alpha <= plan.cor.hi + 1e-6)
    assert plan.diag["max_kappa"] < plan.diag["max_kappa_ref"]
    assert plan.diag["min_map_clearance_m"] >= PlannerConfig().verify_clearance_m - 1e-6
    assert plan.path.length < plan.ref.length + 1e-6 or plan.diag["race_length_m"] <= plan.diag["lap_ref_length_m"]
    # Speed profile: within the lateral acceleration limit and positive.
    k = smoothed_curvature(plan.path)
    assert np.all(plan.v <= np.sqrt(RaceConfig().a_lat_mps2 / np.maximum(k, 1e-6)) + 0.05)
    assert plan.v.min() > 0.5 and plan.t_lap_pred > 0


def test_blind_side_bounded_by_w_min():
    cl = stadium()
    walls = offset(cl, 1.0)  # only the left (inner) wall: the right side is a curb the LiDAR does not see
    grid = make_grid(walls, None)
    log = {"xy": cl, "left": np.ones(len(cl), bool), "right": np.zeros(len(cl), bool)}
    pcfg = PlannerConfig()
    ref = closed_reference(cl, (0.0, -3.0, 0.0), pcfg)
    cor = corridor(ref, layers(grid, pcfg), log, VEH, pcfg, 1.5)
    assert cor.kind_l.mean() > 0.9 and not cor.kind_r.any()
    # right bound <= w_min - d_wall_left - W/2 - blind margin (the true curb is at least w_min from the wall)
    lim = 1.5 - cor.d_wall_l - VEH.width_m / 2 - pcfg.blind_margin_m
    assert np.all(-cor.lo <= np.maximum(lim, 0.0) + 1e-6)


def test_pillar_kept_away_and_lap_path_contained():
    cl = stadium()
    walls = np.vstack((offset(cl, 1.2), offset(cl, -1.2)))
    ang = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    pillar = np.column_stack((2.0 + 0.16 * np.cos(ang), -3.0 + 0.55 + 0.16 * np.sin(ang)))
    grid = make_grid(np.vstack((walls, pillar)), None)
    lap = cl.copy()
    lap[:, 1] += np.where(np.abs(lap[:, 0] - 2.0) < 2.0, -0.35 * (1 + np.cos(np.pi * (lap[:, 0] - 2.0) / 2.0)) / 2, 0.0) * (lap[:, 1] < 0)
    log = {"xy": lap, "left": np.ones(len(lap), bool), "right": np.ones(len(lap), bool)}
    plan = plan_race(grid, lap, (0.0, -3.0, 0.0), log, VEH, PlannerConfig(), RaceConfig(), 1.5)
    assert np.all(plan.cor.lo <= 1e-9) and np.all(plan.cor.hi >= -1e-9)
    d = np.hypot(plan.path.xy[:, 0] - 2.0, plan.path.xy[:, 1] - (-3.0 + 0.55)) - 0.16
    assert d.min() > VEH.width_m / 2


def test_centre_mode_is_near_the_middle():
    cl = stadium()
    walls = np.vstack((offset(cl, 1.0), offset(cl, -1.0)))
    grid = make_grid(walls, None)
    lap = offset(cl, 0.3)  # lap 1 drove off-centre
    log = {"xy": lap, "left": np.ones(len(lap), bool), "right": np.ones(len(lap), bool)}
    plan = plan_race(grid, lap, tuple(lap[0]) + (0.0,), log, VEH, PlannerConfig(mode="centre"), RaceConfig(), 1.5)
    mid = 0.5 * (plan.cor.lo + plan.cor.hi)
    assert np.median(np.abs(plan.alpha - mid)) < 0.1


def test_seam_is_blended():
    cl = stadium()
    lap = cl + np.linspace(0, 1, len(cl))[:, None] * np.array([0.0, 0.25])  # drift: the end is 25 cm off
    ref = closed_reference(lap, (0.0, -3.0, 0.0), PlannerConfig())
    assert smoothed_curvature(ref).max() < 0.6


def test_pgm_round_trip(tmp_path):
    data = np.array([[-1, 0, 100], [0, 100, -1]], dtype=np.int16)
    info = GridInfo(3, 2, 0.05, -1.0, 2.0)
    write_pgm_yaml(data.reshape(-1), info, tmp_path / "m")
    back, info2 = read_pgm_yaml(tmp_path / "m")
    assert np.array_equal(back, data)
    assert info2.origin_x == -1.0 and info2.resolution == 0.05


def test_blind_side_bounded_by_w_min_when_the_wall_is_not_recognised():
    # The inner wall is mapped in short pieces (not a wall component) and the lap-1 log saw no wall: the nearest
    # mapped boundary still bounds the blind (curb) side through w_min, instead of the +-0.25 m fallback.
    cl = stadium()
    inner = offset(cl, 1.1)
    pieces = inner[(np.arange(len(inner)) // 60) % 2 == 0]  # 0.6 m pieces with 0.6 m gaps
    grid = make_grid(pieces, None)
    log = {"xy": cl, "left": np.zeros(len(cl), bool), "right": np.zeros(len(cl), bool)}
    pcfg = PlannerConfig()
    ref = closed_reference(cl, (0.0, -3.0, 0.0), pcfg)
    cor = corridor(ref, layers(grid, pcfg), log, VEH, pcfg, 1.5)
    assert not cor.kind_l.any() and not cor.kind_r.any()
    near = cor.d_obs_l <= pcfg.wall_detect_max_m
    lim = 1.5 - cor.d_obs_l - VEH.width_m / 2 - pcfg.blind_margin_m
    assert near.mean() > 0.4
    assert np.all(-cor.lo[near] <= np.maximum(lim[near], 0.0) + 1e-6)
