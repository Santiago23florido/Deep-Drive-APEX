"""Reactive lap-1 driving from the local LiDAR cloud (ROS-independent).

On a track it has never seen and without a map, the car cannot follow an
ideal trajectory: it only sees the walls and the obstacles around it. The 2D
LiDAR sees walls (0.30 m) but not the curbs (0.07 m, below its scan plane),
so one side of the lane is often invisible. The driver therefore

* follows the wall it sees at ``d_follow`` (the centre of the lane when it
  sees a wall on each side), never farther from that wall than the generic
  lane rule ``w_min`` allows (the invisible edge is at least ``w_min`` away
  from the visible one);
* treats everything it sees (walls, pillars, decoration) as a hard obstacle
  for the footprint;
* chooses, at every revolution, the best of a set of forward two-arc paths
  (there is no reverse gear) and a speed that can stop within the free length.

Frames: ``base`` = base_link at the current (latency-compensated) pose, x
forward, y left. Paths are those of the rear axle (bicycle model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .map_metrics import se2_apply, se2_inverse
from .race_config import ReactiveConfig, VehicleLimits

LEFT, RIGHT = 1, -1


# ------------------------------------------------------------------ perception
@dataclass
class Cluster:
    points: np.ndarray  # (K, 2) base frame
    length: float  # extent along the principal axes [m]
    is_wall: bool


def cluster_points(points: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Euclidean clustering: labels (N,) and the number of clusters."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int), 0
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    count, labels = connected_components(adj, directed=False)
    return labels, count


def _extent(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    c = points - points.mean(axis=0)
    _, vecs = np.linalg.eigh(c.T @ c)
    proj = c @ vecs
    ext = proj.max(axis=0) - proj.min(axis=0)
    return float(math.hypot(ext[0], ext[1]))


def classify(points: np.ndarray, cfg: ReactiveConfig) -> list[Cluster]:
    labels, count = cluster_points(points, cfg.cluster_eps_m)
    out = []
    for k in range(count):
        pts = points[labels == k]
        length = _extent(pts)
        out.append(Cluster(pts, length, length >= cfg.wall_min_length_m and len(pts) >= cfg.wall_min_points))
    return out


def point_tangents(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Unit tangent (principal direction of the k nearest neighbours) per point."""
    n = len(points)
    if n < 3:
        return np.tile([1.0, 0.0], (n, 1))
    _, idx = cKDTree(points).query(points, k=min(k, n))
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cxx = np.sum(nb[..., 0] ** 2, axis=1)
    cyy = np.sum(nb[..., 1] ** 2, axis=1)
    cxy = np.sum(nb[..., 0] * nb[..., 1], axis=1)
    ang = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
    return np.column_stack((np.cos(ang), np.sin(ang)))


@dataclass
class Wall:
    points: np.ndarray  # base frame
    tangents: np.ndarray
    side: int  # LEFT / RIGHT of the car
    distance: float  # from base_link to the part of the wall beside the car
    length: float


def _beside(points: np.ndarray, cfg: ReactiveConfig) -> tuple[float, int] | None:
    """(distance, side) of the part of a wall beside the car, None if not beside."""
    win = points[(points[:, 0] >= cfg.side_x_min_m) & (points[:, 0] <= cfg.side_x_max_m)]
    if len(win) == 0:
        return None
    d = np.hypot(win[:, 0], win[:, 1])
    k = int(np.argmin(d))
    x, y = win[k]
    if abs(y) < 0.6 * abs(x) or abs(y) > cfg.wall_max_lateral_m:  # ahead of the car, not beside it
        return None
    return float(d[k]), (LEFT if y > 0 else RIGHT)


def _beside_or_ahead(points: np.ndarray, cfg: ReactiveConfig) -> tuple[float, int] | None:
    """Like :func:`_beside`, with the window extended 2 m ahead (a wall that starts ahead of the car)."""
    win = points[(points[:, 0] >= cfg.side_x_min_m) & (points[:, 0] <= 2.0)]
    if len(win) == 0:
        return None
    d = np.hypot(win[:, 0], win[:, 1])
    k = int(np.argmin(d))
    x, y = win[k]
    if abs(y) < 0.6 * abs(x) or d[k] > cfg.wall_max_lateral_m:
        return None
    return float(d[k]), (LEFT if y > 0 else RIGHT)


# --------------------------------------------------------------- wall tracker
@dataclass
class TrackerState:
    mode: str  # "L" | "R" | "BOTH" | "NONE"
    ref: Wall | None
    other: Wall | None
    d_target: float
    d_hi: float  # bound on the distance to the reference wall (w_min rule, relaxed after an acquisition)
    switched: bool = False
    left_wall: bool = False  # a wall beside the car on each side (log for the planner)
    right_wall: bool = False


class WallTracker:
    """Which wall the car follows, kept across revolutions (odometry frame)."""

    def __init__(self, veh: VehicleLimits, cfg: ReactiveConfig) -> None:
        self.veh, self.cfg = veh, cfg
        self.ref_odom: np.ndarray | None = None  # points of the followed wall in the odometry frame
        self.side = 0
        self.absent_from: float | None = None
        self.none_from: float | None = None
        self.relax_from = 0.0
        self.relax_d = 0.0
        self.last_switch = -1e9

    @property
    def d_lo(self) -> float:
        return 0.5 * self.veh.width_m + self.cfg.wall_margin_m

    @property
    def d_hi(self) -> float:
        return self.cfg.w_min_m - 0.5 * self.veh.width_m - self.cfg.blind_margin_m

    def _acquire(self, wall: Wall, travel: float) -> None:
        self.last_switch = travel
        self.side = wall.side
        self.relax_from, self.relax_d = travel, max(self.d_hi, wall.distance + 0.05)

    def d_hi_eff(self, travel: float) -> float:
        u = min(1.0, max(0.0, (travel - self.relax_from) / max(self.cfg.start_relax_m, 1e-6)))
        return max(self.d_hi, (1.0 - u) * self.relax_d + u * self.d_hi)

    def update(self, clusters: list[Cluster], pose_odom: tuple[float, float, float], travel: float) -> TrackerState:
        cfg = self.cfg
        walls = [(c, _beside(c.points, cfg)) for c in clusters if c.is_wall]
        ref_wall: Wall | None = None
        switched = False
        ref_base = se2_apply(se2_inverse(pose_odom), self.ref_odom) if self.ref_odom is not None else None
        if ref_base is not None and walls:
            tree = cKDTree(ref_base)
            best, best_d = None, cfg.assoc_m
            for c, b in walls:
                d = float(np.min(tree.query(c.points)[0]))
                if d < best_d:
                    best, best_d = (c, b), d
            if best is not None and best[1] is not None:
                c, b = best
                ref_wall = Wall(c.points, point_tangents(c.points), b[1], b[0], c.length)
        if ref_wall is not None:
            self.absent_from = None
        elif ref_base is not None:
            if self.absent_from is None:
                self.absent_from = travel
            if travel - self.absent_from < cfg.switch_absent_m:
                # Grace period: keep following the last points seen of the wall.
                b = _beside(ref_base, cfg) if len(ref_base) >= 3 else None
                if b is not None:
                    ref_wall = Wall(ref_base, point_tangents(ref_base), b[1], b[0], _extent(ref_base))
        # A followed wall that no longer reaches ahead of the car has ended: it cannot be followed, so the switch
        # below does not wait for the hysteresis.
        ended = ref_wall is not None and float(ref_wall.points[:, 0].max()) < cfg.wall_ended_ahead_m
        if ref_wall is not None and (ended or travel - self.last_switch >= cfg.switch_hold_m):
            # The followed wall ends just ahead (e.g. it becomes a curb) while a wall on the other side continues:
            # switch now, before the straight extrapolation of the wall leads the car into the invisible edge.
            ref_len = wall_polyline(ref_wall.points, np.zeros(2), 3.0)[1] * 0.1
            if ref_len < cfg.wall_end_switch_m or ended:
                best = None
                for c, _ in walls:
                    if c.points is ref_wall.points:
                        continue
                    b = _beside_or_ahead(c.points, cfg)
                    if b is None or b[1] == ref_wall.side:
                        continue
                    goes_on = float(c.points[:, 0].max()) > cfg.wall_ended_ahead_m + 1.0
                    if wall_polyline(c.points, np.zeros(2), 3.0)[1] * 0.1 < ref_len + 1.0 and not (ended and goes_on):
                        continue
                    if best is None or b[0] < best[1][0]:
                        best = (c, b)
                if best is not None:
                    c, b = best
                    ref_wall = Wall(c.points, point_tangents(c.points), b[1], b[0], c.length)
                    switched = True
                    self._acquire(ref_wall, travel)
                    self.absent_from = None
        if ref_wall is None:
            # (Re)acquisition: the nearest long wall beside the car.
            beside = [(c, b) for c, b in walls if b is not None]
            if beside:
                d_near = min(b[0] for _, b in beside)
                pool = [(c, b) for c, b in beside if b[0] <= d_near + 0.5]
                c, b = max(pool, key=lambda cb: min(cb[0].length, 4.0) - 0.5 * cb[1][0])
                ref_wall = Wall(c.points, point_tangents(c.points), b[1], b[0], c.length)
                switched = self.ref_odom is not None and b[1] != self.side
                self._acquire(ref_wall, travel)
                self.absent_from = None
        if ref_wall is not None:
            if self.absent_from is None:  # a fresh observation (not the grace-period memory)
                near = ref_wall.points[np.hypot(ref_wall.points[:, 0], ref_wall.points[:, 1]) <= 2.5]
                self.ref_odom = se2_apply(pose_odom, near if len(near) >= 3 else ref_wall.points)
            self.side = ref_wall.side
            self.none_from = None
        elif self.none_from is None:
            self.none_from = travel
        other = None
        if ref_wall is not None:
            opp = [(c, b) for c, b in walls if b is not None and b[1] == -ref_wall.side]
            if opp:
                c, b = min(opp, key=lambda cb: cb[1][0])
                if ref_wall.distance + b[0] <= cfg.both_max_width_m:
                    other = Wall(c.points, point_tangents(c.points), b[1], b[0], c.length)
        d_hi = self.d_hi_eff(travel)
        sides = {b[1] for _, b in walls if b is not None}
        if ref_wall is not None:
            sides.add(ref_wall.side)
        lw, rw = LEFT in sides, RIGHT in sides
        if ref_wall is None:
            return TrackerState("NONE", None, None, 0.0, d_hi, switched, lw, rw)
        if other is not None:
            d_t = min(max(0.5 * (ref_wall.distance + other.distance), self.d_lo), d_hi)
            return TrackerState("BOTH", ref_wall, other, d_t, d_hi, switched, lw, rw)
        d_t = min(max(cfg.d_follow_m, self.d_lo), self.d_hi)
        return TrackerState("L" if ref_wall.side == LEFT else "R", ref_wall, None, d_t, d_hi, switched, lw, rw)

    def blind_travel(self, travel: float) -> float:
        return 0.0 if self.none_from is None else travel - self.none_from


# ------------------------------------------------------------------- planning
@dataclass
class ReactiveCommand:
    kappa: float  # curvature of the first segment [1/m]
    v: float  # speed command [m/s]
    path: np.ndarray  # (S, 2) rear-axle path in the base frame
    feasible: bool
    diag: dict = field(default_factory=dict)


class LocalGrid:
    """Distance to the nearest LiDAR point on a small grid around the car."""

    def __init__(self, points: np.ndarray, res: float = 0.05, x_lim: tuple[float, float] = (-2.0, 6.0), y_half: float = 4.5) -> None:
        self.res, self.x0, self.y0 = res, x_lim[0], -y_half
        nx, ny = int(round((x_lim[1] - x_lim[0]) / res)), int(round(2 * y_half / res))
        occ = np.zeros((nx, ny), dtype=bool)
        if len(points):
            ix = np.floor((points[:, 0] - self.x0) / res).astype(int)
            iy = np.floor((points[:, 1] - self.y0) / res).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            occ[ix[ok], iy[ok]] = True
        self.dist = distance_transform_edt(~occ) * res if occ.any() else np.full((nx, ny), 99.0)
        self.shape = (nx, ny)

    def query(self, xy: np.ndarray) -> np.ndarray:
        ix = np.floor((xy[..., 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[..., 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])
        out = np.full(ix.shape, 99.0)
        out[ok] = self.dist[ix[ok], iy[ok]]
        return np.maximum(out - 0.5 * self.res, 0.0)  # the point may lie anywhere in its cell


def two_arc_paths(k1: np.ndarray, k2: np.ndarray, l1: float, horizon: float, ds: float, x0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rear-axle poses (n1, n2, S) of the paths "k1 over l1, then k2 up to horizon"."""
    s = np.arange(ds, horizon + 1e-9, ds)
    K1 = k1[:, None, None]
    K2 = k2[None, :, None]
    S = s[None, None, :]
    s1 = np.minimum(S, l1)
    s2 = np.maximum(S - l1, 0.0)
    # Chord form of an arc: length s * sinc(k s / 2) along heading psi0 + k s / 2.
    h1 = K1 * s1
    c1 = s1 * np.sinc(h1 / (2 * np.pi))
    x = x0 + c1 * np.cos(0.5 * h1)
    y = c1 * np.sin(0.5 * h1)
    psi1 = K1 * l1
    h2 = K2 * s2
    c2 = s2 * np.sinc(h2 / (2 * np.pi))
    x = x + c2 * np.cos(psi1 + 0.5 * h2)
    y = y + c2 * np.sin(psi1 + 0.5 * h2)
    psi = K1 * s1 + K2 * s2
    shape = (len(k1), len(k2), len(s))
    return np.broadcast_to(x, shape), np.broadcast_to(y, shape), np.broadcast_to(psi, shape)


class ReactivePlanner:
    def __init__(self, veh: VehicleLimits, cfg: ReactiveConfig) -> None:
        self.veh, self.cfg = veh, cfg
        k = veh.kappa_max
        self.k1 = np.linspace(-k, k, cfg.n_k1)
        self.k2 = np.linspace(-k, k, cfg.n_k2)
        self.r_circle = veh.circle_radius

    def stop_distance(self, v: float) -> float:
        return v * v / (2.0 * self.cfg.a_dec_mps2) + 0.2 * v + 0.3

    def plan(self, cloud: np.ndarray, v_now: float, kappa_prev: float, state: TrackerState, blind_travel: float = 0.0,
             guide: np.ndarray | None = None, v_cap: float | None = None) -> ReactiveCommand:
        """Choose a path and a speed. ``guide``: optional (M, 2) base-frame
        points of a path to follow instead of the wall offset (race line)."""
        cfg, veh = self.cfg, self.veh
        v = max(0.0, v_now)
        ds = 0.1
        l1 = min(max(0.4 + 0.5 * v, 0.6), 1.2)
        horizon = min(max(1.8 + 1.2 * v, 2.2), 4.0)
        x, y, psi = two_arc_paths(self.k1, self.k2, l1, horizon, ds, veh.rear_offset_m)
        s = np.arange(ds, horizon + 1e-9, ds)
        ns = len(s)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        bx, by = x - veh.rear_offset_m * cpsi, y - veh.rear_offset_m * spsi  # base_link along the path
        grid = LocalGrid(cloud)
        clr = np.full(x.shape, np.inf)
        for off in veh.circle_offsets:
            pts = np.stack((bx + off * cpsi, by + off * spsi), axis=-1)
            clr = np.minimum(clr, grid.query(pts) - self.r_circle)
        clr_now = float(min(grid.query(np.array([[o, 0.0]]))[0] for o in veh.circle_offsets)) - self.r_circle
        c_hard = min(cfg.c_hard_m, clr_now - 0.02)  # squeezed already: allow paths that do not get worse
        blocked = clr < c_hard
        first = np.where(blocked.any(axis=2), blocked.argmax(axis=2), ns)
        free = np.where(first < ns, s[np.minimum(first, ns - 1)] - ds, horizon)
        feasible = free >= min(self.stop_distance(v), horizon - 1e-6)

        cost = cfg.w_block * (1.0 - free / horizon)
        cost += cfg.w_smooth * (((self.k1[:, None] - kappa_prev) / 0.3) ** 2 + 0.5 * ((self.k2[None, :] - self.k1[:, None]) / 0.5) ** 2)
        upto = np.arange(ns)[None, None, :] < first[..., None]
        soft = np.maximum(0.0, cfg.c_soft_m - np.where(upto, clr, cfg.c_soft_m)) ** 2 / cfg.c_soft_m**2
        cost += cfg.w_clear * soft.sum(axis=2) / np.maximum(upto.sum(axis=2), 1)
        diag: dict = {"mode": state.mode, "d_target": state.d_target, "d_hi": state.d_hi, "clr_now": clr_now}
        bound_relaxed = False
        base_pts = np.stack((bx, by), axis=-1)
        if state.ref is not None:
            wall = state.ref
            tree = cKDTree(wall.points)
            d_w, iq = tree.query(base_pts.reshape(-1, 2))
            d_w, iq = d_w.reshape(x.shape), iq.reshape(x.shape)
            rel = base_pts - wall.points[iq]
            tq = wall.tangents[iq]
            along = np.abs(rel[..., 0] * tq[..., 0] + rel[..., 1] * tq[..., 1]) / np.maximum(d_w, 1e-6)
            valid = (along < 0.5) & (d_w < cfg.wall_max_lateral_m + 0.5)
            excess = np.maximum(0.0, d_w - state.d_hi) * valid
            cost += cfg.w_bound * np.mean(excess**2, axis=2) / 0.05**2
            # Already beyond the bound (a new reference wall): no reverse, so only paths that do not get worse.
            excess_now = max(0.0, wall.distance - state.d_hi)
            viol = np.max(np.where(s[None, None, :] <= 1.0, excess, 0.0), axis=2) - excess_now
            within = viol <= 0.05
            # The wall recedes faster than the car can turn (the lane turns sharply): the bound cannot be kept.
            # Follow the wall as closely as possible, slowly: the paths that break the bound least.
            bound_relaxed = bool(feasible.any() and not (feasible & within).any())
            if bound_relaxed:
                within = viol <= float(np.min(viol[feasible])) + 0.05
            feasible &= within
            if guide is None:
                late = valid & (s[None, None, :] >= 0.4 * horizon)
                n_late = late.sum(axis=2)
                off = np.where(late, ((d_w - state.d_target) / cfg.off_scale_m) ** 2, 0.0).sum(axis=2) / np.maximum(n_late, 1)
                cost += cfg.w_off * off
                # Heading at the end of the path against the wall direction there.
                last = np.where(late.any(axis=2), ns - 1 - np.argmax(late[..., ::-1], axis=2), -1)
                ii, jj = np.indices(last.shape)
                lk = np.maximum(last, 0)
                t_end = tq[ii, jj, lk]
                p_end = psi[ii, jj, lk]
                head = 1.0 - np.abs(np.cos(p_end) * t_end[..., 0] + np.sin(p_end) * t_end[..., 1])
                cost += cfg.w_head * np.where(last >= 0, head, 0.0)
            diag["d_wall"] = wall.distance
            diag["side"] = wall.side
        else:
            cost += 0.2 * (self.k1[:, None] / 0.5) ** 2 * np.ones_like(cost)
        if guide is not None and len(guide) >= 2:
            dg = cKDTree(guide).query(np.stack((x, y), axis=-1).reshape(-1, 2))[0].reshape(x.shape)
            cost += cfg.w_guide * np.mean((dg / cfg.off_scale_m) ** 2, axis=2)

        ok = feasible.any()
        pick = np.where(feasible, cost, np.inf) if ok else -free  # nothing feasible: steer to the longest free path and brake
        i, j = np.unravel_index(int(np.argmin(pick)), pick.shape)
        k1, k2 = float(self.k1[i]), float(self.k2[j])
        f = float(free[i, j])
        near = s <= 1.5
        clr_min = float(np.min(clr[i, j][near])) if near.any() else float(np.min(clr[i, j]))
        v_top = cfg.v_explore_mps if state.ref is not None and not bound_relaxed else cfg.v_blind_mps
        if v_cap is not None:
            v_top = min(v_top, v_cap)
        v_curv = math.sqrt(cfg.a_lat_mps2 / max(abs(k1), abs(k2), 1e-3))
        v_free = math.sqrt(max(0.0, 2.0 * cfg.a_dec_mps2 * (f - 0.3)))
        v_clr = v_top * min(1.0, max(0.35, (clr_min - cfg.c_hard_m) / max(cfg.c_soft_m - cfg.c_hard_m, 1e-6)))
        v_cmd = min(v_top, v_curv, v_free, v_clr)
        if state.ref is None and blind_travel >= cfg.blind_hold_m:
            v_cmd = 0.0  # no wall for too long: outside the lane rule, stop
        elif ok and v_free >= cfg.v_min_mps:
            v_cmd = max(v_cmd, cfg.v_min_mps)
        if not ok:
            v_cmd = 0.0
        diag.update(k1=k1, k2=k2, free=f, clr_min=clr_min, n_feasible=int(feasible.sum()), bound_relaxed=bound_relaxed, cost=float(cost[i, j]) if ok else float("nan"))
        path = np.column_stack((x[i, j], y[i, j]))
        return ReactiveCommand(k1, v_cmd, path, bool(ok), diag)


def wall_polyline(points: np.ndarray, start_xy: np.ndarray, length: float, step: float = 0.1) -> tuple[np.ndarray, int]:
    """Ordered points along a wall, forward (+x of the car) from the point
    nearest to ``start_xy``, extrapolated straight beyond its end. Returns the
    polyline (M, 2) and how many of its points lie on the observed wall."""
    tree = cKDTree(points)
    p = points[tree.query(start_xy)[1]]
    idx = tree.query_ball_point(p, 0.3)
    t = point_tangents(points[idx], k=min(12, len(idx)))[0] if len(idx) >= 3 else np.array([1.0, 0.0])
    if t[0] < 0.0:
        t = -t
    out = [p]
    n_obs = 1
    n_steps = int(math.ceil(length / step))
    for _ in range(n_steps):
        q = p + step * t
        near = tree.query_ball_point(q, 0.15)
        if not near:
            break
        p_new = points[near].mean(axis=0)
        d = p_new - p
        nd = float(np.hypot(d[0], d[1]))
        if nd < 0.3 * step or float(d @ t) / nd < 0.5:
            break
        t = 0.5 * t + 0.5 * d / nd
        t /= np.hypot(t[0], t[1])
        p = p_new
        out.append(p)
        n_obs += 1
    while len(out) <= n_steps:  # beyond the observed wall: straight on
        out.append(out[-1] + step * t)
    return np.asarray(out), n_obs


class WallPlanner:
    """Paths at a lateral distance from the followed wall (Frenet frame of the wall).

    Candidate ``(d_c, L_t)``: move from the current distance ``d0`` to ``d_c``
    over ``L_t`` (smoothstep), then keep ``d_c``. ``d_c`` never exceeds the lane
    rule bound, so the car can not drift towards an invisible curb; obstacles
    (pillars) are passed by choosing a larger or smaller ``d_c``.
    """

    def __init__(self, veh: VehicleLimits, cfg: ReactiveConfig) -> None:
        self.veh, self.cfg = veh, cfg
        self.d_prev: float | None = None

    def plan(self, grid: "LocalGrid", wall: Wall, state: TrackerState, v_now: float, d_lo: float) -> ReactiveCommand | None:
        cfg, veh = self.cfg, self.veh
        v = max(0.0, v_now)
        ds = 0.1
        horizon = min(max(2.0 + 1.2 * v, 2.5), 4.5)
        rear = np.array([veh.rear_offset_m, 0.0])
        poly, n_obs = wall_polyline(wall.points, np.array([0.0, 0.0]), horizon + 0.5, ds)
        if n_obs < 3:
            return None
        tang = np.gradient(poly, axis=0)
        tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
        nrm = np.column_stack((-tang[:, 1], tang[:, 0]))
        if float((rear - poly[0]) @ nrm[0]) < 0.0:
            nrm = -nrm  # normals towards the car's side of the wall
        d0 = float((rear - poly[0]) @ nrm[0])
        # skip the part of the wall behind the rear axle
        s_w = np.arange(len(poly)) * ds
        d_hi = state.d_hi
        d_c = np.arange(max(d_lo - 0.1, 0.5 * veh.width_m + 0.05), d_hi + 1e-9, 0.05)
        l_t = np.array([0.8, 1.5, 2.5]) * max(1.0, 0.7 + 0.3 * v)
        dc, lt = np.meshgrid(d_c, l_t, indexing="ij")
        u = np.clip(s_w[None, None, :] / lt[..., None], 0.0, 1.0)
        prof = d0 + (dc[..., None] - d0) * (3 * u**2 - 2 * u**3)  # (Nd, Nl, S)
        xy = poly[None, None, :, :] + prof[..., None] * nrm[None, None, :, :]
        # headings / curvature of the candidate paths
        dxy = np.gradient(xy, axis=2)
        psi = np.arctan2(dxy[..., 1], dxy[..., 0])
        seg = np.maximum(np.hypot(dxy[..., 0], dxy[..., 1]), 1e-6)
        kap = np.gradient(np.unwrap(psi, axis=2), axis=2) / seg
        acc = np.cumsum(seg, axis=2) - seg[..., :1]
        cpsi, spsi = np.cos(psi), np.sin(psi)
        bx, by = xy[..., 0] - veh.rear_offset_m * cpsi, xy[..., 1] - veh.rear_offset_m * spsi
        clr = np.full(bx.shape, np.inf)
        for off in veh.circle_offsets:
            clr = np.minimum(clr, grid.query(np.stack((bx + off * cpsi, by + off * spsi), axis=-1)) - veh.circle_radius)
        clr_now = float(min(grid.query(np.array([[o, 0.0]]))[0] for o in veh.circle_offsets)) - veh.circle_radius
        c_hard = min(cfg.c_hard_m, clr_now - 0.02)
        ahead = acc <= horizon
        blocked = (clr < c_hard) & ahead
        first = np.where(blocked.any(axis=2), blocked.argmax(axis=2), -1)
        free = np.where(first >= 0, np.take_along_axis(acc, np.maximum(first, 0)[..., None], axis=2)[..., 0] - ds, horizon)
        stop = v * v / (2.0 * cfg.a_dec_mps2) + 0.2 * v + 0.3
        kmax = np.max(np.abs(kap) * ahead, axis=2)
        feasible = (free >= min(stop, horizon - 1e-6)) & (kmax <= veh.kappa_max * 1.05)
        cost = cfg.w_off * ((dc - state.d_target) / cfg.off_scale_m) ** 2
        cost += cfg.w_block * (1.0 - free / horizon)
        soft = np.where(ahead, np.maximum(0.0, cfg.c_soft_m - clr), 0.0) ** 2 / cfg.c_soft_m**2
        cost += cfg.w_clear * soft.sum(axis=2) / np.maximum(ahead.sum(axis=2), 1)
        cost += 0.3 * (1.0 / lt)  # prefer gentle transitions
        cost += 0.2 * (kmax / veh.kappa_max) ** 2
        if self.d_prev is not None:
            cost += cfg.w_smooth * ((dc - self.d_prev) / cfg.off_scale_m) ** 2
        if not feasible.any():
            return None
        i, j = np.unravel_index(int(np.argmin(np.where(feasible, cost, np.inf))), cost.shape)
        self.d_prev = float(dc[i, j])
        path = xy[i, j][ahead[i, j]]
        near = acc[i, j] <= 1.5
        k_near = float(np.max(np.abs(kap[i, j][near]))) if near.any() else 0.0
        f = float(free[i, j])
        clr_min = float(np.min(clr[i, j][near])) if near.any() else clr_now
        v_top = cfg.v_explore_mps
        v_curv = math.sqrt(cfg.a_lat_mps2 / max(k_near, 1e-3))
        v_free = math.sqrt(max(0.0, 2.0 * cfg.a_dec_mps2 * (f - 0.3)))
        v_clr = v_top * min(1.0, max(0.35, (clr_min - cfg.c_hard_m) / max(cfg.c_soft_m - cfg.c_hard_m, 1e-6)))
        v_cmd = min(v_top, v_curv, v_free, v_clr)
        if v_free >= cfg.v_min_mps:
            v_cmd = max(v_cmd, cfg.v_min_mps)
        diag = {"mode": state.mode, "d_target": state.d_target, "d_hi": d_hi, "d_now": d0, "d_c": float(dc[i, j]), "l_t": float(lt[i, j]),
                "free": f, "clr_now": clr_now, "clr_min": clr_min, "n_feasible": int(feasible.sum()), "k_near": k_near,
                "d_wall": wall.distance, "side": wall.side, "planner": "wall", "k1": float(kap[i, j][min(3, len(kap[i, j]) - 1)])}
        return ReactiveCommand(diag["k1"], v_cmd, path, True, diag)


class ReactiveDriver:
    """Perception + wall tracking + planning for one revolution."""

    def __init__(self, veh: VehicleLimits, cfg: ReactiveConfig) -> None:
        self.veh, self.cfg = veh, cfg
        self.tracker = WallTracker(veh, cfg)
        self.planner = ReactivePlanner(veh, cfg)
        self.wall_planner = WallPlanner(veh, cfg)
        self.kappa_prev = 0.0

    def step(self, cloud_base: np.ndarray, pose_odom: tuple[float, float, float], travel: float, v_now: float,
             guide: np.ndarray | None = None, v_cap: float | None = None) -> tuple[ReactiveCommand, TrackerState]:
        """One revolution: follow the wall (wall-relative paths), or, without a
        wall or with a guide path (race line), the two-arc planner."""
        clusters = classify(cloud_base, self.cfg)
        state = self.tracker.update(clusters, pose_odom, travel)
        cmd = None
        if state.ref is not None and guide is None:
            cmd = self.wall_planner.plan(LocalGrid(cloud_base), state.ref, state, v_now, self.tracker.d_lo)
            if cmd is not None and v_cap is not None:
                cmd.v = min(cmd.v, v_cap)
        if cmd is None:
            self.wall_planner.d_prev = None
            cmd = self.planner.plan(cloud_base, v_now, self.kappa_prev, state, self.tracker.blind_travel(travel), guide, v_cap)
            cmd.diag["planner"] = "arcs"
        self.kappa_prev = cmd.kappa
        cmd.diag["n_walls"] = sum(c.is_wall for c in clusters)
        cmd.diag["n_obstacles"] = sum(not c.is_wall for c in clusters)
        return cmd, state


def pursue(path: np.ndarray, pose: tuple[float, float, float], lookahead: float, wheelbase: float) -> float:
    """Pure-pursuit steering [rad] on a polyline (same frame as ``pose``, rear axle)."""
    if len(path) < 2:
        return 0.0
    d = np.hypot(path[:, 0] - pose[0], path[:, 1] - pose[1])
    k0 = int(np.argmin(d))
    seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    acc = np.concatenate(([0.0], np.cumsum(seg)))
    target_s = acc[k0] + lookahead
    if target_s >= acc[-1]:  # extend beyond the end along its last direction
        u = (path[-1] - path[-2]) / max(seg[-1], 1e-9)
        tx, ty = path[-1] + u * (target_s - acc[-1])
    else:
        tx, ty = np.interp(target_s, acc, path[:, 0]), np.interp(target_s, acc, path[:, 1])
    alpha = math.atan2(ty - pose[1], tx - pose[0]) - pose[2]
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    ld = max(math.hypot(tx - pose[0], ty - pose[1]), 1e-3)
    return math.atan2(2.0 * wheelbase * math.sin(alpha), ld)
