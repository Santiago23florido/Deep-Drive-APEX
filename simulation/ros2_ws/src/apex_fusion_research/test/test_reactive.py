import math

import numpy as np
import pytest

from apex_fusion_research.core.local_cloud import ScanAccumulator, scan_points, supported, voxel_downsample
from apex_fusion_research.core.race_config import ReactiveConfig, VehicleLimits
from apex_fusion_research.core.reactive import (LEFT, RIGHT, LocalGrid, ReactiveDriver, ReactivePlanner, WallTracker, classify, known_length, lane_known,
                                                lane_speed, lane_walls, pursue, two_arc_paths)

VEH, CFG = VehicleLimits(), ReactiveConfig()


def line(x0, y0, x1, y1, step=0.02):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    u = np.linspace(0.0, 1.0, n)
    return np.column_stack((x0 + u * (x1 - x0), y0 + u * (y1 - y0)))


def circle(cx, cy, r, step=0.02):
    n = max(8, int(2 * math.pi * r / step))
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack((cx + r * np.cos(a), cy + r * np.sin(a)))


def drive(cloud, v=1.2, steps=1, kappa_prev=0.0):
    d = ReactiveDriver(VEH, CFG)
    d.kappa_prev = kappa_prev
    out = None
    for _ in range(steps):
        out = d.step(cloud, (0.0, 0.0, 0.0), 0.0, v)
    return out


def test_arcs_closed_form():
    x, y, psi = two_arc_paths(np.array([0.0, 1.0]), np.array([0.0]), 10.0, 1.5, 0.01, 0.0)
    s = 1.5
    assert abs(x[0, 0, -1] - s) < 1e-9 and abs(y[0, 0, -1]) < 1e-9
    assert abs(x[1, 0, -1] - math.sin(s)) < 1e-9 and abs(y[1, 0, -1] - (1 - math.cos(s))) < 1e-9  # unit circle
    assert abs(psi[1, 0, -1] - s) < 1e-9
    # second arc continues from the end of the first
    x, y, psi = two_arc_paths(np.array([1.0]), np.array([-1.0]), 1.0, 2.0, 0.5, 0.0)
    assert abs(psi[0, 0, -1]) < 1e-9


def test_clearance_matches_point_distances_across_cell_boundaries():
    points = np.array([[-0.153, -0.218], [0.153, -0.207], [0.3, -0.199]])
    # The old raster's current/future queries could differ by a whole 5 cm
    # cell. A 0.2 mm displacement cannot change a distance by more than 0.2 mm.
    queries = np.array([[0.1999, 0.0], [0.2001, 0.0], [0.0, 0.0]])
    expected = np.maximum(np.linalg.norm(queries[:, None] - points, axis=-1).min(axis=1) - 0.025, 0.0)
    actual = LocalGrid(points).query(queries)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert abs(actual[1] - actual[0]) <= 0.0002 + 1e-12


def test_squeezed_start_can_turn_away_from_a_slanted_wall():
    veh = VehicleLimits(steer_max_deg=12.5, steer_plan_deg=12.0)
    # A wall 21 cm right of base_link, slanting into the straight path.
    # The footprint fits; the existing escape rule must allow turning left.
    cloud = line(-0.23, -0.2222, 3.0, -0.093, step=0.01)
    cmd, _ = ReactiveDriver(veh, ReactiveConfig()).step(cloud, (0, 0, 0), 0, 0)
    assert cmd.feasible and cmd.v > 0.0, cmd.diag
    assert cmd.kappa > 0.0


def test_clearance_keeps_frontal_obstacle_blocked():
    cloud = line(0.35, -1.0, 0.35, 1.0, step=0.01)
    cmd, _ = drive(cloud, v=0.0)
    assert not cmd.feasible and cmd.v == 0.0


def test_short_checked_arc_keeps_its_curvature_with_long_lookahead():
    # Exact bicycle geometry: following the end of a radius-2 m arc requires
    # atan(wheelbase / radius), even when the requested lookahead is longer.
    radius, wheelbase = 2.0, 0.3
    theta = np.linspace(0.05, 0.15, 5)
    path = np.column_stack((radius * np.sin(theta), radius * (1 - np.cos(theta))))
    assert pursue(path, (0, 0, 0), 0.85, wheelbase) == pytest.approx(math.atan(wheelbase / radius))


def test_arc_command_excludes_the_blocked_continuation():
    cloud = line(1.0, -3.0, 1.0, 3.0, step=0.01)
    cmd, _ = drive(cloud, v=0.0)
    assert cmd.feasible and 0.3 <= cmd.diag['free'] < 1.0
    # Arc samples are at 0.1 m stations from the rear axle. None of the
    # returned targets may be past the certified free prefix.
    assert len(cmd.path) > 1
    assert len(cmd.path) * 0.1 <= cmd.diag['free'] + 1e-9
    assert cmd.path[:, 0].max() < 1.0 - VEH.circle_radius


def test_follow_single_wall_turns_towards_target():
    cloud = line(-2, 1.4, 6, 1.4)  # left wall at 1.4 m, farther than d_follow = 0.75
    cmd, st = drive(cloud)
    assert st.mode == "L" and abs(st.d_target - 0.75) < 1e-9
    assert cmd.kappa > 0.0 and cmd.feasible and cmd.v > 0.3


def test_single_wall_at_target_goes_straight():
    cloud = line(-2, -0.75, 6, -0.75)
    cmd, st = drive(cloud, kappa_prev=0.0)
    assert st.mode == "R" and st.ref.side == RIGHT
    assert abs(cmd.kappa) < 0.15


def test_two_walls_centre_clamped():
    cloud = np.vstack((line(-2, 1.0, 6, 1.0), line(-2, -1.0, 6, -1.0)))
    _, st = drive(cloud)
    assert st.mode == "BOTH" and abs(st.d_target - 1.0) < 0.05
    cloud = np.vstack((line(-2, 1.35, 6, 1.35), line(-2, -1.35, 6, -1.35)))
    tr = WallTracker(VEH, CFG)
    tr.update(classify(cloud, CFG), (0.0, 0.0, 0.0), 0.0)
    st = tr.update(classify(cloud, CFG), (0.0, 0.0, 0.0), 10.0)  # after the acquisition relaxation: d_hi applies
    assert st.d_target <= tr.d_hi + 1e-9


def test_curved_wall_curvature():
    # Inner wall: arc of radius 2 centred at (0, 2.75), the car 0.75 m from it heading +x.
    a = np.linspace(-math.pi / 2 - 0.9, -math.pi / 2 + 1.4, 400)
    cloud = np.column_stack((2.0 * np.cos(a), 2.75 + 2.0 * np.sin(a)))
    d = ReactiveDriver(VEH, CFG)
    d.kappa_prev = 1 / 2.75
    cmd, _ = d.step(cloud, (0.0, 0.0, 0.0), 0.0, 1.0)
    assert 0.15 < cmd.kappa < 0.6


def simulate(cloud, steps, v=1.0, step=0.12, x0=(0.0, 0.0, 0.0)):
    """Kinematic replay: replan every ``step`` metres and follow the local path by pure pursuit."""
    d = ReactiveDriver(VEH, CFG)
    x = np.array(x0, dtype=float)
    trace = []
    for _ in range(steps):
        c, s = math.cos(x[2]), math.sin(x[2])
        local = (cloud - x[:2]) @ np.array([[c, -s], [s, c]])
        cmd, st = d.step(local, tuple(x), float(np.hypot(*x[:2])), v)
        trace.append((x.copy(), cmd, st))
        if cmd.v <= 0.0:
            break
        delta = pursue(cmd.path, (VEH.rear_offset_m, 0.0, 0.0), 0.7, VEH.wheelbase_m)
        k = math.tan(delta) / VEH.wheelbase_m
        x[0] += step * math.cos(x[2] + 0.5 * step * k)
        x[1] += step * math.sin(x[2] + 0.5 * step * k)
        x[2] += step * k
    return trace


def test_pillar_passed_on_blind_side():
    # Left wall at 0.75 m (on target); a pillar 0.57 m from the wall, 2 m ahead: the car must go right, within w_min.
    pillar = (2.0, 0.75 - 0.57)
    cloud = np.vstack((line(-2, 0.75, 9, 0.75), circle(*pillar, 0.16)))
    trace = simulate(cloud, 45)
    assert len(trace) == 45, "the car stopped"
    tr = WallTracker(VEH, CFG)
    for x, cmd, st in trace:
        assert 0.75 - x[1] <= tr.d_hi + 0.06  # never farther from the wall than the lane rule allows
        assert math.hypot(x[0] - pillar[0], x[1] - pillar[1]) - 0.16 > VEH.width_m / 2
    assert trace[-1][0][0] > pillar[0] + 1.5  # passed it
    assert abs(0.75 - trace[-1][0][1] - 0.75) < 0.2  # back near the wall offset


def test_follows_wall_through_a_curve():
    # Inner wall: arc of radius 1.5 (hairpin-like), the car starts 0.75 m from it.
    a = np.linspace(-math.pi / 2, math.pi / 2, 240)
    wall = np.vstack((line(-3, 0.75, 0, 0.75), np.column_stack((1.5 * np.cos(a), 2.25 + 1.5 * np.sin(a)))))
    trace = simulate(wall, 40)
    assert len(trace) == 40
    d = [np.min(np.hypot(wall[:, 0] - x[0], wall[:, 1] - x[1])) for x, _, _ in trace]
    assert min(d) > 0.45 and max(d[5:]) < 1.1


def test_blocked_ahead_stops():
    cloud = np.vstack((line(-2, 0.75, 3, 0.75), line(0.45, -2.0, 0.45, 2.0)))
    cmd, _ = drive(cloud, v=1.0)
    assert cmd.v == 0.0


def test_decoration_box_is_not_a_wall():
    box = np.vstack((line(0, 1, 1.4, 1), line(1.4, 1, 1.4, 1.9), line(1.4, 1.9, 0, 1.9), line(0, 1.9, 0, 1)))
    assert not any(c.is_wall for c in classify(box, CFG))
    assert any(c.is_wall for c in classify(line(-2, 1, 3, 1), CFG))


def test_wall_switch_after_the_followed_wall_ends():
    d = ReactiveDriver(VEH, CFG)
    left = line(-3, 0.75, 1.0, 0.75)
    right = line(-3, -0.8, 6, -0.8)
    _, st = d.step(np.vstack((left, right)), (0.0, 0.0, 0.0), 0.0, 1.0)
    assert st.ref.side == LEFT
    # 3 m later the left wall is behind the car: switch to the right wall.
    shift = np.array([3.0, 0.0])
    sides = []
    for k, tr in enumerate(np.linspace(0.5, 3.0, 6)):
        cloud = np.vstack((left, right)) - np.array([tr, 0.0])
        _, st = d.step(cloud, (tr, 0.0, 0.0), tr, 1.0)
        sides.append(st.ref.side if st.ref is not None else 0)
    assert sides[-1] == RIGHT
    assert shift[0] > 0


def test_steering_limits():
    cloud = line(-2, 1.4, 6, 1.4)
    cmd, _ = drive(cloud)
    assert abs(cmd.kappa) <= VEH.kappa_max + 1e-9


def test_accumulator_window_and_voxel():
    acc = ScanAccumulator(window_s=1.0, window_m=1.0, voxel=0.05)
    r = np.full(360, 2.0)
    for k in range(30):
        acc.add_scan(int(k * 0.077e9), r, -math.pi, 2 * math.pi / 360, (0.1 * k, 0.0, 0.0))
    ages = [s[0] for s in acc.scans]
    assert (ages[-1] - ages[0]) * 1e-9 <= 1.0 + 0.08 or acc.travel - acc.scans[0][1] <= 1.0 + 0.11
    pts = voxel_downsample(np.random.default_rng(0).uniform(0, 1, (1000, 2)), 0.1)
    assert len(pts) <= 100
    local = acc.cloud_in((acc.travel, 0.0, 0.0))
    assert len(local) > 0


def test_pursue_straight_and_left():
    path = np.column_stack((np.linspace(0, 5, 51), np.zeros(51)))
    assert abs(pursue(path, (0.0, 0.0, 0.0), 0.8, 0.3)) < 1e-9
    assert pursue(path, (0.0, -0.3, 0.0), 0.8, 0.3) > 0.0


def test_isolated_spurious_returns_are_rejected():
    r = np.full(360, 1.0)
    r[100] = 0.2  # dust / cross-talk: a single beam
    r[200] = np.inf  # a dropout inside a surface keeps its neighbours
    pts = scan_points(r, -np.pi, np.radians(1.0), 0.15, 6.0, reject_isolated=True)
    assert len(pts) == 358
    assert np.all(np.hypot(pts[:, 0], pts[:, 1]) > 0.9)
    assert supported(r)[[199, 201]].all()


def test_beyond_the_bound_is_not_a_deadlock():
    # The reference wall (left, acquired alone) bends away to 1.6 m (> d_hi once the acquisition relaxation is over)
    # while the right wall shows up: there is no reverse, the car must keep driving.
    d = ReactiveDriver(VEH, CFG)
    d.step(line(-2, 1.4, 6, 1.4), (0.0, 0.0, 0.0), 0.0, 0.5)
    cloud = np.vstack((line(-2, 1.6, 6, 1.6), line(-2, -1.0, 6, -1.0)))
    guide = line(0, 0, 4, 0, 0.1)
    for travel in (5.0, 10.0):
        cmd, st = d.step(cloud, (0.0, 0.0, 0.0), travel, 0.5)
        assert st.ref.side == LEFT and st.ref.distance > st.d_hi + 0.1
        assert cmd.v > 0.2, (st.mode, cmd.diag)
        cmd, _ = d.step(cloud, (0.0, 0.0, 0.0), travel, 0.5, guide)
        assert cmd.diag["n_feasible"] > 0 and cmd.v > 0.2


def _ranges_to(points_laser, n=360):
    """Ranges of a full revolution (angle_min -pi, 1 deg) that hit the given points (nearest per beam)."""
    r = np.full(n, np.inf)
    a = np.arctan2(points_laser[:, 1], points_laser[:, 0])
    b = np.rint((a + np.pi) / np.radians(1.0)).astype(int) % n
    np.minimum.at(r, b, np.hypot(points_laser[:, 0], points_laser[:, 1]))
    return r


def test_carving_drops_floor_hits_and_keeps_static_walls():
    walls = np.vstack((line(-3, 1.0, 8, 1.0, 0.01), line(-3, -1.0, 8, -1.0, 0.01)))
    floor = line(2.3, -0.8, 2.3, 0.8, 0.01)  # the laser plane on the floor while braking
    acc = ScanAccumulator(window_s=5.0, window_m=10.0)
    inc = np.radians(1.0)
    acc.add_scan(0, _ranges_to(np.vstack((walls, floor))), -np.pi, inc, (0.0, 0.0, 0.0))
    assert np.any((np.abs(acc.cloud_odom()[:, 0] - 2.3) < 0.05) & (np.abs(acc.cloud_odom()[:, 1]) < 0.5))
    # Level again 0.2 m further: the beams pass over the old floor points; the rear sector is blind (no return).
    rel = walls - [0.2, 0.0]
    r = _ranges_to(rel)
    r[np.abs(np.degrees(-np.pi + inc * np.arange(360))) > 150] = np.inf
    acc.add_scan(1, r, -np.pi, inc, (0.2, 0.0, 0.0))
    cloud = acc.cloud_odom()
    assert not np.any((np.abs(cloud[:, 0] - 2.3) < 0.1) & (np.abs(cloud[:, 1]) < 0.7))
    behind = cloud[cloud[:, 0] < -2.0]
    assert len(behind) > 0 and np.all(np.abs(np.abs(behind[:, 1]) - 1.0) < 0.05)  # blind sector: memory kept
    near_wall = np.abs(np.abs(cloud[:, 1]) - 1.0) < 0.05
    assert near_wall.sum() >= 0.95 * len(cloud)


def test_receding_wall_relaxes_the_bound_instead_of_stopping():
    # Left wall veers away at ~60 deg, faster than the car can turn (a sharp turn of the lane): no path keeps
    # the w_min bound. Instead of stopping (and then taking the longest free path), follow the wall as closely as possible, slowly.
    left = np.vstack((line(-3, 0.8, -0.84, 0.8), line(-0.84, 0.8, -0.53, 1.07), line(-0.53, 1.07, -0.21, 1.7), line(-0.21, 1.7, 0.41, 2.93)))
    cloud = np.vstack((left, line(-4.5, -0.64, 0.4, -0.64)))  # (the wall shape of a harness run that got stuck)
    tr = WallTracker(VEH, CFG)
    tr.update(classify(left, CFG), (0.0, 0.0, 0.0), 0.0)  # followed since before (hysteresis)
    st = tr.update(classify(cloud, CFG), (0.0, 0.0, 0.0), 10.0)
    assert st.ref is not None and st.ref.side == LEFT
    cmd = ReactivePlanner(VEH, CFG).plan(cloud, 0.6, 0.0, st)
    assert cmd.diag["n_feasible"] > 0 and cmd.diag["bound_relaxed"]
    assert cmd.diag["k1"] > 0.3 and 0.0 < cmd.v <= CFG.v_blind_mps + 1e-9


def test_grazing_wall_is_supported():
    # Wall y = 0.5 seen from the origin at 1 deg steps: ranges grow fast near grazing incidence.
    th = np.radians(np.arange(3.0, 20.0, 1.0))
    r = np.full(360, np.inf)
    idx = np.rint((th + np.pi) / np.radians(1.0)).astype(int)
    r[idx] = 0.5 / np.sin(th)
    ok = supported(r)
    assert ok[idx[1:-1]].all()
    r[idx[8]] = 0.3  # a spurious return in the middle of it
    assert not supported(r)[idx[8]]
    r2 = np.full(360, np.inf)
    r2[50], r2[51] = 0.4, 3.0  # a spurious return next to a dropout / the blind sector
    assert not supported(r2)[50]



def test_deskewed_self_hit_ahead_of_the_body_is_dropped():
    from apex_fusion_research.core.race_driver import EXPLORE, RaceDriverCore
    from apex_fusion_research.core.race_config import RaceDriverConfig

    core = RaceDriverCore(RaceDriverConfig(), 3)
    core.phase = EXPLORE
    r = np.full(360, 2.0)
    # The wheel hit (laser frame (-0.02, 0.155)) moved 0.11 m ahead by the de-skewing at 2 m/s: bearing ~60 deg, 0.18 m.
    th = -np.pi + np.radians(1.0) * np.arange(360)
    b = int(np.argmin(np.abs(th - np.arctan2(0.155, 0.09))))
    r[b - 1:b + 2] = np.hypot(0.09, 0.155)
    core.on_scan(0.0, r, -np.pi, np.radians(1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2.0)
    cloud = core.acc.cloud_odom()
    assert np.all(np.hypot(cloud[:, 0] - 0.18, cloud[:, 1]) > 1.5)


def test_ended_wall_is_left_for_the_wall_that_goes_on_without_hysteresis():
    # The followed left wall ends at the car (its edge goes on as a curb); the lane turns right along the right wall.
    tr = WallTracker(VEH, CFG)
    left = line(-4.0, 1.0, 1.5, 1.0)
    tr.update(classify(left, CFG), (0.0, 0.0, 0.0), 0.0)
    ended = line(-4.0, 1.0, 0.05, 1.0)
    right = line(-0.5, -0.5, 2.5, -2.6)
    st = tr.update(classify(np.vstack((ended, right)), CFG), (0.0, 0.0, 0.0), 0.5)  # 0.5 m after acquiring: within the hold
    assert st.ref is not None and st.ref.side == RIGHT


def test_lane_known_beside_between_and_beyond_the_end():
    walls = lane_walls(classify(np.vstack((line(-2, 1.2, 3, 1.2), line(-2, -1.2, 3, -1.2))), CFG), CFG)
    pts = np.array([[0.0, 0.0], [2.5, 0.5], [4.5, 0.0], [0.0, 2.4]])
    known = lane_known(pts, walls, 1.0, CFG.both_max_width_m)
    # (0, 0) is 1.2 m from each wall (more than d_max) but between them; beyond the ends nothing is known
    assert known.tolist() == [True, True, False, False]
    single = lane_walls(classify(line(-2, 1.2, 3, 1.2), CFG), CFG)
    assert lane_known(pts, single, 1.0, CFG.both_max_width_m).tolist() == [False, True, False, False]


def test_known_length_is_the_proven_prefix():
    known = np.array([[True, True, True, False, True], [True] * 5])
    s = np.arange(1, 6) * 0.5
    kl = known_length(known, s, 2.5)
    assert abs(kl[0] - 1.9) < 1e-9 and kl[1] == 2.5  # the last proven station before the first unproven one
    assert lane_speed(2.5, 2.5, CFG) == math.inf and lane_speed(CFG.lane_margin_m, 3.0, CFG) == 0.0


def test_pillar_next_to_the_wall_does_not_hide_it():
    cloud = np.vstack((line(-2, 1.0, 5, 1.0), circle(1.0, 0.55, 0.15)))
    walls = lane_walls(classify(cloud, CFG), CFG)
    assert len(walls) == 2  # the wall and the pillar joined to it, kept apart
    assert lane_known(np.array([[1.0, -0.1], [1.0, 0.1]]), walls, 1.24, CFG.both_max_width_m).all()


def test_sparse_trail_extends_the_wall_around_a_corner():
    a = np.radians(np.arange(0, 61, 12))  # a few returns on the far side of a 60 deg bend, 0.45 m apart
    trail = np.column_stack((2.0 + 1.9 * np.sin(a), 1.2 + 1.9 * (1 - np.cos(a))))
    cloud = np.vstack((line(-2, 1.2, 2.0, 1.2), trail[1:]))
    walls = lane_walls(classify(cloud, CFG), CFG)
    assert sum(len(w.points) for w in walls) == len(cloud)
    p = np.array([[3.55, 0.95]])  # 0.8 m from the trail, on the lane side of the bend
    assert lane_known(p, walls, 1.24, CFG.both_max_width_m).all()


def test_speed_drops_before_the_end_of_the_proven_lane():
    # Two walls that end 1.2 m ahead (beyond: invisible curbs): the car must be able to stop within the proven lane.
    short = np.vstack((line(-3, 0.9, 0.8, 0.9), line(-3, -0.9, 0.8, -0.9)))
    cmd, _ = drive(short, v=1.5)
    assert cmd.diag["lane_known"] < 1.5 and cmd.v <= max(cmd.diag["v_lane"], CFG.v_min_mps) + 1e-9 and cmd.v < 1.25
    long = np.vstack((line(-3, 0.9, 6, 0.9), line(-3, -0.9, 6, -0.9)))
    cmd, _ = drive(long, v=1.5)
    assert cmd.diag["v_lane"] == math.inf and cmd.v > 1.2



def test_wheel_shadow_does_not_erase_the_wall_beside_the_car():
    # A front wheel hides the beams from -110 to -85 deg in every revolution. The right wall the car passes was seen
    # ahead before; no return in the shadow must not carve it (it used to split the wall, and the driver lost it).
    wall = line(-3, -1.0, 10, -1.0, 0.01)
    acc = ScanAccumulator(window_s=5.0, window_m=10.0)
    inc = np.radians(1.0)
    shadow = (np.degrees(-np.pi + inc * np.arange(360)) > -110) & (np.degrees(-np.pi + inc * np.arange(360)) < -85)
    for k in range(20):
        x = 0.1 * k
        r = _ranges_to(wall - [x, 0.0])
        r[shadow] = np.inf
        acc.add_scan(k, r, -np.pi, inc, (x, 0.0, 0.0))
    cloud = acc.cloud_odom()
    beside = cloud[(np.abs(cloud[:, 0] - 1.9) < 0.3) & (np.abs(cloud[:, 1] + 1.0) < 0.05)]
    assert len(beside) >= 12  # the wall under the shadow is still there (18 points; 7 when no return carved it)
    assert acc.return_rate[shadow].max() < 0.5 and np.median(acc.return_rate[~shadow & (acc.return_rate > 0)]) > 0.9


# --------------------------------------------------------------- recovery
def test_the_rear_offset_reaches_every_planner():
    """The rear axle is where the car measures it to be, and the candidate
    paths, the wall distances and the footprint are all built around it."""
    from apex_fusion_research.core.race_config import RaceDriverConfig
    from apex_fusion_research.core.reactive import ReactiveDriver

    cfg = RaceDriverConfig()
    driver = ReactiveDriver(cfg.vehicle, cfg.reactive)
    driver.set_rear_offset(-0.085)
    for holder in (driver, driver.tracker, driver.planner, driver.wall_planner):
        assert holder.veh.rear_offset_m == pytest.approx(-0.085)
    driver.set_rear_offset(-0.0855)  # below the millimetre: not worth rebuilding
    assert driver.planner.veh.rear_offset_m == pytest.approx(-0.085)


def test_the_car_backs_out_of_a_place_with_no_feasible_path():
    """Today a car with no feasible candidate brakes and stays there for the
    rest of the run. It must get itself out, with a bounded manoeuvre."""
    from apex_fusion_research.core.race_config import RaceDriverConfig
    from apex_fusion_research.core.race_driver import EXPLORE, RaceDriverCore
    from apex_fusion_research.core.reactive import ReactiveCommand

    cfg = RaceDriverConfig()
    core = RaceDriverCore(cfg, 3)
    core.phase = EXPLORE
    core.cmd = ReactiveCommand(0.0, 0.0, np.zeros((0, 2)), False, {"free": 0.1})
    core.rear_clear = 1.0  # the car drove in through there
    t = 0.0
    for _ in range(60):  # a second of standing still against the block
        t += 0.02
        v, _steer, _row = core.control(t, (0.0, 0.0, 0.0), 0.0, None, True, 0.0)
    assert core.recover == 2 and v < 0.0, "the car should be backing out"
    travelled = 0.0
    while core.recover and t < 20.0:
        t += 0.02
        travelled += abs(v) * 0.02
        core.travel = travelled
        v, _steer, _row = core.control(t, (travelled, 0.0, 0.0), -0.4, None, True, 0.0)
    assert travelled <= cfg.adaptive.recover_distance_m + 0.2
    assert core.recover == 0 and core.events["recovery"][-1]["kind"] == 2


def test_it_creeps_instead_of_reversing_when_there_is_room_ahead():
    from apex_fusion_research.core.race_config import RaceDriverConfig
    from apex_fusion_research.core.race_driver import EXPLORE, RaceDriverCore
    from apex_fusion_research.core.reactive import ReactiveCommand

    core = RaceDriverCore(RaceDriverConfig(), 3)
    core.phase = EXPLORE
    core.cmd = ReactiveCommand(0.0, 0.0, np.zeros((0, 2)), False, {"free": 1.2})
    t = 0.0
    for _ in range(60):
        t += 0.02
        v, _s, _r = core.control(t, (0.0, 0.0, 0.0), 0.0, None, True, 0.0)
    assert core.recover == 1 and v > 0.0


def test_it_does_not_reverse_into_something_it_cannot_see():
    """The LiDAR is blind behind the car. Without proven room back there the
    car must stay where it is, exactly as it does today."""
    from apex_fusion_research.core.race_config import RaceDriverConfig
    from apex_fusion_research.core.race_driver import EXPLORE, RaceDriverCore
    from apex_fusion_research.core.reactive import ReactiveCommand

    core = RaceDriverCore(RaceDriverConfig(), 3)
    core.phase = EXPLORE
    core.cmd = ReactiveCommand(0.0, 0.0, np.zeros((0, 2)), False, {"free": 0.1})
    core.rear_clear = 0.05
    t = 0.0
    for _ in range(120):
        t += 0.02
        v, _s, _r = core.control(t, (0.0, 0.0, 0.0), 0.0, None, True, 0.0)
    assert core.recover == 0 and v == 0.0


def test_the_margins_widen_while_the_car_is_unsure_and_return_after():
    from apex_fusion_research.core.race_config import RaceDriverConfig
    from apex_fusion_research.core.reactive import ReactiveDriver

    cfg = RaceDriverConfig()
    driver = ReactiveDriver(cfg.vehicle, cfg.reactive)
    base_wall, base_soft = cfg.reactive.wall_margin_m, cfg.reactive.c_soft_m
    driver.set_margins(0.05, 0.08)
    for holder in (driver, driver.tracker, driver.planner, driver.wall_planner):
        assert holder.cfg.wall_margin_m == pytest.approx(base_wall + 0.05)
        assert holder.cfg.c_soft_m == pytest.approx(base_soft + 0.08)
    driver.set_margins(0.0, 0.0)  # converged: back to the nominal car exactly
    assert driver.planner.cfg.wall_margin_m == pytest.approx(base_wall)
    assert driver.planner.cfg.c_soft_m == pytest.approx(base_soft)
