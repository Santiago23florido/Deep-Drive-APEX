"""Race line on the car's own map, planned after lap 1 (ROS-independent).

Inputs are only what the car has built itself: the occupancy grid of
slam_toolbox at the end of lap 1, the lap-1 path in the map frame (the
optimised SLAM graph, or the logged poses) and the log of which side it saw a
wall on. Steps:

1. **Reference**: the lap-1 path, closed at the start line, resampled and
   smoothed. It is known to be drivable: the car just drove it.
2. **Corridor**: at every station, how far the car centre may move to each
   side. A side with a mapped wall is bounded by the wall minus a margin; a
   blind side (curb under the LiDAR plane) only by the generic lane rule
   ``w_min`` measured from the wall on the other side; everything mapped
   (walls, pillars, decoration) bounds the free interval. The lap-1 path is
   always inside.
3. **Line**: minimum curvature (``min_curvature``) or centre of the corridor
   (``centre``), as bounded linear least squares on the lateral offsets.
4. **Check**: the footprint along the line keeps a clearance to every mapped
   cell, and its curvature stays within the car's steering.
5. **Speed**: lateral acceleration and corridor-width caps, then the ESC
   acceleration / braking limits (periodic lap profile).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, gaussian_filter1d, label, maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.sparse import coo_matrix, identity, vstack
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .occupancy import GridInfo, write_pgm_yaml
from .race_config import PlannerConfig, RaceConfig, VehicleLimits
from .sim_paths import ensure_tools_path

ensure_tools_path()
from pose_dataset.controller import _forward_backward  # noqa: E402
from pose_dataset.tracks import Path2D, path_from_points  # noqa: E402


# ----------------------------------------------------------------------- map
@dataclass
class Grid:
    """Occupancy grid, ``data[row, col]`` with row = y index from the origin."""

    data: np.ndarray  # int16 (H, W): -1 unknown, 0..100
    res: float
    ox: float
    oy: float

    @classmethod
    def from_flat(cls, flat, width: int, height: int, res: float, ox: float, oy: float) -> "Grid":  # noqa: ANN001
        return cls(np.asarray(flat, dtype=np.int16).reshape(height, width), float(res), float(ox), float(oy))

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.floor((xy[..., 0] - self.ox) / self.res).astype(int)
        r = np.floor((xy[..., 1] - self.oy) / self.res).astype(int)
        inside = (r >= 0) & (r < self.data.shape[0]) & (c >= 0) & (c < self.data.shape[1])
        return np.clip(r, 0, self.data.shape[0] - 1), np.clip(c, 0, self.data.shape[1] - 1), inside

    def info(self) -> GridInfo:
        return GridInfo(self.data.shape[1], self.data.shape[0], self.res, self.ox, self.oy, 0.0)


@dataclass
class MapLayers:
    grid: Grid
    blocked: np.ndarray  # occupied (and unknown, if configured)
    wall: np.ndarray  # occupied cells of long connected parts (walls, not pillars / decoration)
    edt: np.ndarray  # distance to the nearest blocked cell [m]
    unknown_is_obstacle: bool

    def sample(self, layer: np.ndarray, xy: np.ndarray, outside) -> np.ndarray:  # noqa: ANN001
        r, c, inside = self.grid.cells(xy)
        return np.where(inside, layer[r, c], outside)

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        """Distance from ``xy`` to the nearest blocked cell (cell-size conservative)."""
        return np.maximum(self.sample(self.edt, xy, 0.0 if self.unknown_is_obstacle else 99.0) - 0.5 * self.grid.res, 0.0)


def swept_mask(grid: Grid, path_xy: np.ndarray, radius: float) -> np.ndarray:
    """Cells within ``radius`` of a driven path (where the car body has been)."""
    mask = np.zeros(grid.data.shape, dtype=bool)
    if len(path_xy) == 0:
        return mask
    k = int(math.ceil(radius / grid.res))
    offs = np.array([(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if math.hypot(i, j) * grid.res <= radius])
    dense = resample_open(np.asarray(path_xy, dtype=float), 0.5 * grid.res)
    r, c, _ = grid.cells(dense)
    rr = np.clip(r[:, None] + offs[None, :, 0], 0, grid.data.shape[0] - 1)
    cc = np.clip(c[:, None] + offs[None, :, 1], 0, grid.data.shape[1] - 1)
    mask[rr.ravel(), cc.ravel()] = True
    return mask


def layers(grid: Grid, cfg: PlannerConfig, driven_xy: np.ndarray | None = None, body_radius: float = 0.0) -> MapLayers:
    """Map layers for planning. Two cleanings the car can justify itself:
    isolated occupied specks (a scan plane tilted by the suspension grazing the
    floor) are noise, and nothing can stand where the car body drove on lap 1."""
    occ = grid.data >= cfg.occupied_threshold
    if cfg.speck_max_cells > 0:
        lab_o, _ = label(occ, structure=np.ones((3, 3), dtype=bool))
        occ &= (np.bincount(lab_o.ravel()) > cfg.speck_max_cells)[lab_o]
    if driven_xy is not None and body_radius > 0.0:
        occ &= ~swept_mask(grid, driven_xy, body_radius)
    blocked = occ.copy()
    if cfg.unknown_is_obstacle:
        # Unexplored areas block; the specks between rays inside explored space do not.
        unk_lab, _ = label(grid.data < 0)
        sizes = np.bincount(unk_lab.ravel())
        big = sizes * grid.res**2 >= cfg.unknown_min_area_m2
        big[0] = False
        blocked |= big[unk_lab]
    edt = distance_transform_edt(~blocked) * grid.res
    lab, _ = label(binary_dilation(occ, iterations=1), structure=np.ones((3, 3), dtype=bool))
    wall_ids = [i for i, sl in enumerate(find_objects(lab), start=1)
                if sl is not None and math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * grid.res >= cfg.wall_component_min_m]
    wall = np.isin(lab, wall_ids) & occ
    return MapLayers(grid, blocked, wall, edt, cfg.unknown_is_obstacle)


# ----------------------------------------------------------------- reference
def _dedupe(xy: np.ndarray, eps: float = 0.01) -> np.ndarray:
    keep = [0]
    for k in range(1, len(xy)):
        if math.hypot(*(xy[k] - xy[keep[-1]])) >= eps:
            keep.append(k)
    return xy[keep]


def resample_open(xy: np.ndarray, ds: float) -> np.ndarray:
    if len(xy) < 2:
        return xy
    acc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))))
    s = np.arange(0.0, acc[-1] + 1e-9, ds)
    return np.column_stack((np.interp(s, acc, xy[:, 0]), np.interp(s, acc, xy[:, 1])))


def resample_closed(xy: np.ndarray, ds: float) -> np.ndarray:
    closed = np.vstack((xy, xy[:1]))
    acc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1])))))
    n = max(int(round(acc[-1] / ds)), 16)
    s = np.linspace(0.0, acc[-1], n, endpoint=False)
    return np.column_stack((np.interp(s, acc, closed[:, 0]), np.interp(s, acc, closed[:, 1])))


def _start_at(path: Path2D, p0: np.ndarray) -> Path2D:
    k0 = int(np.argmin(np.hypot(path.xy[:, 0] - p0[0], path.xy[:, 1] - p0[1])))
    return path_from_points(np.roll(path.xy, -k0, axis=0))


def closed_reference(loop_xy: np.ndarray, start: tuple[float, float, float], cfg: PlannerConfig) -> Path2D:
    """The lap-1 path (start -> crossing of the start line) as a closed, smooth path."""
    xy = _dedupe(np.asarray(loop_xy, dtype=float))
    p0 = np.array(start[:2], dtype=float)
    acc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))))
    blend = min(cfg.seam_blend_m, 0.3 * acc[-1])
    u = np.clip((acc - (acc[-1] - blend)) / max(blend, 1e-6), 0.0, 1.0)
    xy = xy + (0.5 - 0.5 * np.cos(np.pi * u))[:, None] * (p0 - xy[-1])  # the end meets the start
    xy = _dedupe(xy[:-1]) if len(xy) > 3 else xy
    pts = resample_closed(xy, cfg.ds_m)
    step = math.hypot(*(pts[1] - pts[0]))
    sig = cfg.smooth_sigma_m / max(step, 1e-6)
    pts = np.column_stack((gaussian_filter1d(pts[:, 0], sig, mode="wrap"), gaussian_filter1d(pts[:, 1], sig, mode="wrap")))
    return _start_at(path_from_points(pts), p0)


# ------------------------------------------------------------------ corridor
@dataclass
class Corridor:
    lo: np.ndarray  # right bound of the lateral offset of the car centre (<= 0)
    hi: np.ndarray  # left bound (>= 0)
    kind_l: np.ndarray  # True: a mapped wall bounds the left side
    kind_r: np.ndarray
    d_wall_l: np.ndarray
    d_wall_r: np.ndarray
    d_obs_l: np.ndarray
    d_obs_r: np.ndarray
    n_free_l: np.ndarray
    n_free_r: np.ndarray
    forced: np.ndarray  # stations where the lap-1 path had to be kept although the margins exclude it


def _runs_at_least(mask: np.ndarray, n_min: int) -> np.ndarray:
    """Keep only the (cyclic) runs of True at least ``n_min`` long."""
    if mask.all() or not mask.any():
        return mask.copy()
    shift = int(np.argmin(mask))  # start on a False
    m = np.roll(mask, -shift)
    out = np.zeros_like(m)
    k = 0
    while k < len(m):
        if m[k]:
            j = k
            while j < len(m) and m[j]:
                j += 1
            if j - k >= n_min:
                out[k:j] = True
            k = j
        else:
            k += 1
    return np.roll(out, shift)


def corridor(ref: Path2D, lay: MapLayers, side_log: dict | None, veh: VehicleLimits, cfg: PlannerConfig, w_min: float) -> Corridor:
    n = ref.normal
    step = 0.5 * lay.grid.res
    tau = np.arange(0.0, cfg.ray_max_m + 1e-9, step)
    r_need = veh.circle_radius + cfg.obstacle_margin_m
    out: dict[str, np.ndarray] = {}
    for sign, name in ((1.0, "l"), (-1.0, "r")):
        pts = ref.xy[:, None, :] + sign * tau[None, :, None] * n[:, None, :]
        blk = lay.sample(lay.blocked, pts, lay.unknown_is_obstacle)
        wal = lay.sample(lay.wall, pts, False)
        clr = lay.clearance(pts)
        out[f"d_obs_{name}"] = np.where(blk.any(axis=1), tau[blk.argmax(axis=1)], cfg.ray_max_m)
        out[f"d_wall_{name}"] = np.where(wal.any(axis=1), tau[wal.argmax(axis=1)], np.inf)
        # Moving away from an obstacle is always allowed: the side is closed where the
        # clearance is too small *and* still decreasing (getting closer to something).
        closer = np.concatenate((np.zeros((len(clr), 1), dtype=bool), np.diff(clr, axis=1) < 0.0), axis=1)
        bad = (clr < r_need) & closer
        k10 = min(len(tau) - 1, int(round(0.1 / step)))  # the EDT is quantised: compare 10 cm apart
        bad[:, 0] = (clr[:, 0] < r_need) & (clr[:, k10] <= clr[:, 0] + 0.01)
        j = bad.argmax(axis=1)
        out[f"n_free_{name}"] = np.where(bad.any(axis=1), np.where(j > 0, tau[np.maximum(j - 1, 0)], 0.0), cfg.ray_max_m)
    geo_l = out["d_wall_l"] <= cfg.wall_detect_max_m
    geo_r = out["d_wall_r"] <= cfg.wall_detect_max_m
    kind_l, kind_r = geo_l.copy(), geo_r.copy()
    if side_log is not None and len(side_log.get("xy", [])):
        d, k = cKDTree(np.asarray(side_log["xy"], dtype=float)).query(ref.xy)
        near = d <= 1.0
        kind_l = np.where(near, geo_l & np.asarray(side_log["left"], dtype=bool)[k], geo_l)
        kind_r = np.where(near, geo_r & np.asarray(side_log["right"], dtype=bool)[k], geo_r)
    n_min = max(1, int(round(0.5 / cfg.ds_m)))
    kind_l, kind_r = _runs_at_least(kind_l, n_min), _runs_at_least(kind_r, n_min)
    half = 0.5 * veh.width_m
    e_l = np.where(kind_l, out["d_wall_l"], np.where(kind_r, np.minimum(out["d_obs_l"], w_min - out["d_wall_r"]), np.nan))
    e_r = np.where(kind_r, out["d_wall_r"], np.where(kind_l, np.minimum(out["d_obs_r"], w_min - out["d_wall_l"]), np.nan))
    hi = e_l - half - np.where(kind_l, cfg.wall_margin_m, cfg.blind_margin_m)
    lo = -(e_r - half - np.where(kind_r, cfg.wall_margin_m, cfg.blind_margin_m))
    none = ~kind_l & ~kind_r
    # No wall recognised on either side (a wall mapped in short pieces, a lap-1 view that missed it): the nearest
    # mapped boundary still bounds the opposite, blind side through the lane-width prior.
    near_l = out["d_obs_l"] <= cfg.wall_detect_max_m
    near_r = out["d_obs_r"] <= cfg.wall_detect_max_m
    hi_none = np.where(near_r, np.minimum(out["d_obs_l"], w_min - out["d_obs_r"]) - half - cfg.blind_margin_m, np.inf)
    lo_none = np.where(near_l, -(np.minimum(out["d_obs_r"], w_min - out["d_obs_l"]) - half - cfg.blind_margin_m), -np.inf)
    hi = np.where(none, np.minimum(cfg.none_half_width_m, hi_none), hi)
    lo = np.where(none, np.maximum(-cfg.none_half_width_m, lo_none), lo)
    hi = np.minimum(hi, out["n_free_l"])
    lo = np.maximum(lo, -out["n_free_r"])
    forced = (hi < 0.0) | (lo > 0.0)
    hi, lo = np.maximum(hi, 0.0), np.minimum(lo, 0.0)
    size = 2 * max(0, int(round(cfg.bound_erosion_m / cfg.ds_m))) + 1
    hi = minimum_filter1d(hi, size, mode="wrap")
    lo = maximum_filter1d(lo, size, mode="wrap")
    return Corridor(lo, hi, kind_l, kind_r, out["d_wall_l"], out["d_wall_r"], out["d_obs_l"], out["d_obs_r"], out["n_free_l"], out["n_free_r"], forced)


# ---------------------------------------------------------------------- line
def _cyclic(n_pts: int, stencil: dict[int, float]):  # noqa: ANN202
    idx = np.arange(n_pts)
    rows = np.concatenate([idx for _ in stencil])
    cols = np.concatenate([(idx + o) % n_pts for o in stencil])
    vals = np.concatenate([np.full(n_pts, c) for c in stencil.values()])
    return coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()


def solve_offsets(ref: Path2D, lo: np.ndarray, hi: np.ndarray, cfg: PlannerConfig, lambda_centre: float | None = None) -> np.ndarray:
    """Lateral offsets ``alpha`` (left +) of the line from the reference.

    The curvature of the offset curve is, to first order,
    ``kappa + kappa^2 alpha + alpha''`` (linear in alpha), so both modes are
    bounded linear least squares:

    * ``min_curvature``: sum kappa^2 + lambda_jerk sum (d kappa / ds)^2 + lambda_centre sum alpha^2;
    * ``centre``: the middle of the corridor, with a little curvature smoothing.
    """
    n_pts = len(ref.xy)
    ds = ref.length / n_pts
    k_ref = uniform_filter1d(ref.curvature, max(1, int(round(0.3 / ds))), mode="wrap")
    d2 = _cyclic(n_pts, {-1: 1.0, 0: -2.0, 1: 1.0}) / ds**2
    d1 = _cyclic(n_pts, {-1: -0.5, 1: 0.5}) / ds
    a_k = (d2 + coo_matrix((k_ref**2, (np.arange(n_pts), np.arange(n_pts))), shape=(n_pts, n_pts))).tocsr()
    b_k = -k_ref
    hi_s = np.maximum(hi, lo + 1e-4)
    if cfg.mode == "centre":
        mid = 0.5 * (lo + hi_s)
        w = math.sqrt(0.02)
        a = vstack([w * a_k, identity(n_pts) / 0.1]).tocsr()
        b = np.concatenate([w * b_k, mid / 0.1])
    else:
        lam_c = cfg.lambda_centre if lambda_centre is None else lambda_centre
        lam_j = math.sqrt(cfg.lambda_jerk)
        a = vstack([a_k, lam_j * (d1 @ a_k), math.sqrt(lam_c) * identity(n_pts)]).tocsr()
        b = np.concatenate([b_k, lam_j * (d1 @ b_k), np.zeros(n_pts)])
    return box_least_squares(a, b, lo, hi_s)


def box_least_squares(a, b: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_iter: int = 20000, tol: float = 1e-6) -> np.ndarray:  # noqa: ANN001
    """min ||a x - b||^2 subject to lo <= x <= hi, by ADMM (over-relaxed) with one
    sparse factorisation: the curvature rows make the problem too ill-conditioned
    for iterative least-squares solvers, while this converges in < 1 s."""
    h = (a.T @ a).tocsc()
    g = -(a.T @ b)
    n = h.shape[0]
    rho = 1e-4 * float(np.mean(h.diagonal()))
    lu = splu((h + rho * identity(n, format="csc")).tocsc())
    z = np.clip(np.zeros(n), lo, hi)
    u = np.zeros(n)
    for it in range(max_iter):
        x = lu.solve(rho * (z - u) - g)
        xh = 1.6 * x - 0.6 * z
        z_old = z
        z = np.clip(xh + u, lo, hi)
        u += xh - z
        if it % 100 == 99 and np.max(np.abs(x - z)) < tol and np.max(np.abs(z - z_old)) < 0.1 * tol:
            break
    return z


def footprint_clearance(path: Path2D, lay: MapLayers, veh: VehicleLimits) -> np.ndarray:
    """Clearance of the 3 footprint circles to the mapped cells along a path."""
    c, s = np.cos(path.heading), np.sin(path.heading)
    clr = np.full(len(path.xy), np.inf)
    for off in veh.circle_offsets:
        pts = path.xy + off * np.column_stack((c, s))
        clr = np.minimum(clr, lay.clearance(pts) - veh.circle_radius)
    return clr


def smoothed_curvature(path: Path2D, window_m: float = 0.5) -> np.ndarray:
    ds = path.length / len(path.xy)
    return uniform_filter1d(np.abs(path.curvature), max(1, int(round(window_m / ds))), mode="wrap")


# --------------------------------------------------------------------- speed
def lap_speed(path: Path2D, width: np.ndarray, rcfg: RaceConfig) -> tuple[np.ndarray, float]:
    """Periodic speed profile of one lap [m/s] and the predicted lap time [s]."""
    n_pts = len(path.xy)
    ds = path.length / n_pts
    k = smoothed_curvature(path)
    k = maximum_filter1d(k, 2 * max(1, int(round(0.3 / ds))) + 1, mode="wrap")
    cap = np.minimum(rcfg.v_max_mps, np.sqrt(rcfg.a_lat_mps2 / np.maximum(k, 1e-6)))
    cap = np.minimum(cap, rcfg.v_tight_base_mps + rcfg.v_tight_gain * np.maximum(width, 0.0))
    tiled = np.tile(cap, 3)
    v = _forward_backward(tiled, ds, rcfg.a_acc_mps2, rcfg.a_dec_mps2, v_start=float(cap[0]), v_end=float(cap[0]))[n_pts : 2 * n_pts]
    return v, float(np.sum(ds / np.maximum(v, 0.05)))


# ---------------------------------------------------------------------- plan
@dataclass
class RacePlan:
    path: Path2D  # race line (car centre), s = 0 at the start line
    v: np.ndarray  # lap speed profile on path stations
    ref: Path2D  # reference the corridor is expressed on (same stations)
    alpha: np.ndarray
    cor: Corridor
    t_lap_pred: float
    diag: dict = field(default_factory=dict)
    clearance: np.ndarray = field(default_factory=lambda: np.zeros(0))  # footprint clearance to the map per station [m]

    def speed_at(self, s: float) -> float:
        return float(self.path.interp_scalar(self.v, s))


def plan_race(grid: Grid, loop_xy: np.ndarray, start: tuple[float, float, float], side_log: dict | None, veh: VehicleLimits,
              pcfg: PlannerConfig, rcfg: RaceConfig, w_min: float) -> RacePlan:
    t0 = time.perf_counter()
    lay = layers(grid, pcfg, loop_xy, 0.5 * veh.width_m)
    ref = closed_reference(loop_xy, start, pcfg)
    lap_ref = ref
    cor = corridor(ref, lay, side_log, veh, pcfg, w_min)
    forced0 = int(cor.forced.sum())
    alpha = np.zeros(len(ref.xy))
    for it in range(max(1, pcfg.outer_iters)):
        alpha = solve_offsets(ref, cor.lo, cor.hi, pcfg)
        if it == pcfg.outer_iters - 1 or np.max(np.abs(alpha)) < 0.01:
            break
        new = resample_closed(ref.xy + alpha[:, None] * ref.normal, pcfg.ds_m)
        ref = _start_at(path_from_points(new), np.array(start[:2]))
        cor = corridor(ref, lay, side_log, veh, pcfg, w_min)
    # Check the footprint on the map and the curvature; tighten towards the reference if needed.
    tightened = 0
    lam_c = pcfg.lambda_centre
    for _ in range(pcfg.max_tighten_iters):
        path = path_from_points(ref.xy + alpha[:, None] * ref.normal)
        clr = footprint_clearance(path, lay, veh)
        bad = clr < pcfg.verify_clearance_m
        k_bad = smoothed_curvature(path) > veh.kappa_max
        if not bad.any() and not k_bad.any():
            break
        if bad.any():
            # Push the line, where it is too close, in the lateral direction that increases the clearance.
            win = 2 * max(1, int(round(0.3 / pcfg.ds_m))) + 1
            grow = maximum_filter1d(bad.astype(np.uint8), win, mode="wrap").astype(bool)
            viol = np.where(grow, np.maximum(pcfg.verify_clearance_m - np.minimum(clr, pcfg.verify_clearance_m), 0.0) + 0.03, 0.0)
            viol = maximum_filter1d(viol, win, mode="wrap")
            c_left = footprint_clearance(path_from_points(ref.xy + (alpha + 0.05)[:, None] * ref.normal), lay, veh)
            c_right = footprint_clearance(path_from_points(ref.xy + (alpha - 0.05)[:, None] * ref.normal), lay, veh)
            go_left = grow & (c_left > c_right)
            go_right = grow & ~(c_left > c_right)
            cor.lo = np.where(go_left, np.minimum(alpha + viol, cor.hi), cor.lo)
            cor.hi = np.where(go_right, np.maximum(alpha - viol, cor.lo), cor.hi)
            tightened += int(grow.sum())
        if k_bad.any():
            lam_c *= 10.0
        alpha = solve_offsets(ref, cor.lo, cor.hi, pcfg, lam_c)
    path = path_from_points(ref.xy + alpha[:, None] * ref.normal)
    clr = footprint_clearance(path, lay, veh)
    v, t_lap = lap_speed(path, cor.hi - cor.lo, rcfg)
    kappa = smoothed_curvature(path)
    diag = {
        "mode": pcfg.mode, "lap_ref_length_m": lap_ref.length, "race_length_m": path.length, "stations": len(path.xy),
        "max_kappa": float(kappa.max()), "kappa_limit": veh.kappa_max, "max_kappa_ref": float(smoothed_curvature(lap_ref).max()),
        "min_map_clearance_m": float(clr.min()), "p5_map_clearance_m": float(np.percentile(clr, 5)),
        "forced_stations": forced0, "tightened_stations": tightened, "max_offset_m": float(np.max(np.abs(alpha))),
        "blind_left_frac": float(np.mean(~cor.kind_l)), "blind_right_frac": float(np.mean(~cor.kind_r)),
        "mean_width_m": float(np.mean(cor.hi - cor.lo)), "min_width_m": float(np.min(cor.hi - cor.lo)),
        "t_lap_pred_s": t_lap, "v_mean_mps": float(path.length / t_lap), "compute_s": time.perf_counter() - t0,
        "w_min_m": w_min,
    }
    return RacePlan(path, v, ref, alpha, cor, t_lap, diag, clr)


def save_plan(plan: RacePlan, grid: Grid, loop_xy: np.ndarray, side_log: dict | None, out_dir: str | Path, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_pgm_yaml(grid.data.reshape(-1), grid.info(), out / "map_snapshot")
    with open(out / "lap1_path.csv", "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in np.asarray(loop_xy):
            f.write(f"{x:.4f},{y:.4f}\n")
    if side_log is not None and len(side_log.get("xy", [])):
        with open(out / "wall_log.csv", "w", encoding="utf-8") as f:
            f.write("x,y,left_wall,right_wall\n")
            for (x, y), l_, r_ in zip(side_log["xy"], side_log["left"], side_log["right"]):
                f.write(f"{x:.4f},{y:.4f},{int(bool(l_))},{int(bool(r_))}\n")
    c = plan.cor
    with open(out / "corridor.csv", "w", encoding="utf-8") as f:
        f.write("s,ref_x,ref_y,n_x,n_y,lo,hi,left_wall,right_wall,d_wall_l,d_wall_r,d_obs_l,d_obs_r,alpha\n")
        n = plan.ref.normal
        for i in range(len(plan.ref.xy)):
            f.write(f"{plan.ref.s[i]:.3f},{plan.ref.xy[i, 0]:.4f},{plan.ref.xy[i, 1]:.4f},{n[i, 0]:.5f},{n[i, 1]:.5f},{c.lo[i]:.4f},{c.hi[i]:.4f},"
                    f"{int(c.kind_l[i])},{int(c.kind_r[i])},{min(c.d_wall_l[i], 99):.3f},{min(c.d_wall_r[i], 99):.3f},{c.d_obs_l[i]:.3f},{c.d_obs_r[i]:.3f},{plan.alpha[i]:.4f}\n")
    with open(out / "raceline.csv", "w", encoding="utf-8") as f:
        f.write("s,x,y,heading,curvature,v,map_clearance\n")
        p = plan.path
        clr = plan.clearance if len(plan.clearance) == len(p.xy) else np.full(len(p.xy), np.nan)
        for i in range(len(p.xy)):
            f.write(f"{p.s[i]:.3f},{p.xy[i, 0]:.4f},{p.xy[i, 1]:.4f},{p.heading[i]:.5f},{p.curvature[i]:.5f},{plan.v[i]:.3f},{clr[i]:.3f}\n")
    (out / "plan.json").write_text(json.dumps({**plan.diag, **(extra or {})}, indent=1, default=float), encoding="utf-8")
    return out


def config_dict(*configs) -> dict:  # noqa: ANN002
    return {type(c).__name__: asdict(c) for c in configs}
