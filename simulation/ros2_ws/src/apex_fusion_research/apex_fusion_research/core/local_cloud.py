"""Local LiDAR point cloud around the car (ROS-independent).

The reactive driver sees the track through the last revolutions of the
LiDAR, de-skewed by the learned odometry. They are accumulated in the
odometry frame (smooth, unlike the SLAM correction that jumps a few
centimetres) and re-expressed at the car's current pose, which also fills the
rear blind sector of the A2M8 and compensates the ~150 ms estimation latency.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from .map_metrics import se2_apply, se2_compose, se2_inverse


def supported(r: np.ndarray, abs_tol: float = 0.05, rel_tol: float = 0.03, reach: int = 2) -> np.ndarray:
    """Beams whose range agrees with a neighbour up to ``reach`` beams away.

    A surface returns on consecutive beams; a spurious return (dust, cross-talk,
    a random reading) is an isolated beam. Looking two beams away keeps a
    surface point whose direct neighbour dropped out. A wall seen at grazing
    incidence changes range quickly from beam to beam (~0.3 m per degree at 3 m
    and 10 deg), but its returns lie between their neighbours: that also
    counts as support. The scan wraps around."""
    r = np.asarray(r, dtype=float)
    tol = abs_tol + rel_tol * np.where(np.isfinite(r), r, 0.0)
    ok = np.zeros(len(r), dtype=bool)
    with np.errstate(invalid="ignore"):  # inf - inf
        for k in range(1, reach + 1):
            for sh in (k, -k):
                ok |= np.abs(np.roll(r, sh) - r) <= tol
        prev, nxt = np.roll(r, 1), np.roll(r, -1)
        between = np.abs(r - 0.5 * (prev + nxt)) <= tol + 0.5 * np.abs(nxt - prev)
        ok |= between & np.isfinite(prev) & np.isfinite(nxt)
    return ok


def scan_points(ranges: np.ndarray, angle_min: float, angle_inc: float, range_min: float, range_max: float,
                reject_isolated: bool = False) -> np.ndarray:
    """(M, 2) points of a LaserScan in its own frame (beam i at angle_min + i * inc)."""
    r = np.asarray(ranges, dtype=float)
    th = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    if reject_isolated:
        ok &= supported(r)
    return np.column_stack((r[ok] * np.cos(th[ok]), r[ok] * np.sin(th[ok])))


def voxel_downsample(xy: np.ndarray, voxel: float) -> np.ndarray:
    """One point (the mean) per ``voxel`` x ``voxel`` cell."""
    if len(xy) == 0 or voxel <= 0.0:
        return xy
    keys = np.floor(xy / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(counts), 2))
    np.add.at(out, inv, xy)
    return out / counts[:, None]


class ScanAccumulator:
    """Scans in the odometry frame over a time / travel window."""

    def __init__(self, window_s: float = 2.0, window_m: float = 3.0, range_min: float = 0.15, range_max: float = 6.0, voxel: float = 0.05,
                 reject_isolated: bool = True, carve: bool = True, carve_margin_m: float = 0.10, carve_no_return_m: float = 4.0,
                 carve_no_return_fov_deg: float = 120.0) -> None:
        self.window_ns = int(window_s * 1e9)
        self.window_m = float(window_m)
        self.range_min, self.range_max, self.voxel = float(range_min), float(range_max), float(voxel)
        self.reject_isolated = bool(reject_isolated)
        self.carve, self.carve_margin, self.carve_no_return = bool(carve), float(carve_margin_m), float(carve_no_return_m)
        self.carve_fov = math.radians(carve_no_return_fov_deg)
        self.scans: deque[tuple[int, float, np.ndarray]] = deque()  # (stamp, travel at the stamp, points in odom)
        self.travel = 0.0
        self._last_xy: tuple[float, float] | None = None
        self._cache: tuple[int, np.ndarray] | None = None

    def add_scan(self, stamp_ns: int, ranges: np.ndarray, angle_min: float, angle_inc: float, pose_odom_base: tuple[float, float, float],
                 base_to_laser: tuple[float, float, float] = (0.0, 0.0, 0.0), body: tuple[float, float, float, float] | None = None) -> None:
        """Add a revolution de-skewed to ``stamp_ns``, taken with the car at ``pose_odom_base``.

        ``body`` = (x_min, x_max, y_min, y_max) of the car in base_link: returns
        inside it are the car itself (wheels, chassis) and are dropped."""
        x, y = pose_odom_base[0], pose_odom_base[1]
        if self._last_xy is not None:
            self.travel += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        pts = se2_apply(base_to_laser, scan_points(ranges, angle_min, angle_inc, self.range_min, self.range_max, self.reject_isolated))
        if body is not None and len(pts):
            pts = pts[~((pts[:, 0] >= body[0]) & (pts[:, 0] <= body[1]) & (pts[:, 1] >= body[2]) & (pts[:, 1] <= body[3]))]
        if self.carve and self.scans:
            laser_odom = se2_compose(pose_odom_base, base_to_laser)
            self.scans = deque((st, tr, p[~self._seen_through(p, laser_odom, ranges, angle_min, angle_inc)]) for st, tr, p in self.scans)
        self.scans.append((int(stamp_ns), self.travel, se2_apply(pose_odom_base, pts)))
        while len(self.scans) > 1 and (stamp_ns - self.scans[0][0] > self.window_ns and self.travel - self.scans[0][1] > self.window_m):
            self.scans.popleft()
        while len(self.scans) > 1 and (self.travel - self.scans[0][1] > 3.0 * self.window_m or stamp_ns - self.scans[0][0] > 3 * self.window_ns):
            self.scans.popleft()  # hard cap: long stops or fast driving
        self._cache = None

    def _seen_through(self, pts_odom: np.ndarray, laser_odom: tuple[float, float, float], ranges: np.ndarray, angle_min: float,
                      angle_inc: float) -> np.ndarray:
        """Stored points the new revolution sees through: its beam at the point and both neighbours return farther.

        A static obstacle is never seen through. What is: the floor hit by the
        laser plane while the body pitches (braking, bumps), and anything that
        moved. No return on the three beams counts as free only for points closer
        than ``carve_no_return_m`` and within ``carve_no_return_fov_deg`` of the
        front: a close obstacle always answers, far away a missing return may be
        a dropout, and behind the car the chassis hides the beams."""
        if len(pts_odom) == 0:
            return np.zeros(0, dtype=bool)
        r = np.asarray(ranges, dtype=float)
        n = len(r)
        rel = se2_apply(se2_inverse(laser_odom), pts_odom)
        rng = np.hypot(rel[:, 0], rel[:, 1])
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        raw = np.rint(np.mod(bearing - angle_min, 2 * math.pi) / angle_inc).astype(int)
        b = raw % n
        full = n * abs(angle_inc) >= 2 * math.pi - 1.5 * abs(angle_inc)
        beyond = np.ones(len(rel), dtype=bool) if full else (raw >= 0) & (raw < n)
        limit = rng + self.carve_margin + 0.03 * rng
        no_return_free = (rng < self.carve_no_return) & (np.abs(bearing) < self.carve_fov)
        for k in (-1, 0, 1):
            rk = r[(b + k) % n]
            beyond &= np.where(np.isfinite(rk), rk > limit, no_return_free)
        return beyond & (rng > self.range_min) & (rng < self.range_max)

    @property
    def empty(self) -> bool:
        return not self.scans

    def cloud_odom(self) -> np.ndarray:
        if self._cache is None or self._cache[0] != len(self.scans):
            pts = np.concatenate([s[2] for s in self.scans]) if self.scans else np.zeros((0, 2))
            self._cache = (len(self.scans), voxel_downsample(pts, self.voxel))
        return self._cache[1]

    def cloud_in(self, pose_odom_base: tuple[float, float, float]) -> np.ndarray:
        """The accumulated cloud expressed in the car frame at ``pose_odom_base``."""
        return se2_apply(se2_inverse(pose_odom_base), self.cloud_odom())
