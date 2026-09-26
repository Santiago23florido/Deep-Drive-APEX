"""Lap detection on the car's own map pose (ROS-independent).

The car does not know the track: a lap is closed when its SLAM pose crosses
again the line through its start pose, perpendicular to its start heading,
after a minimum travel (a lap is longer than ``arm_distance_m``), close to
the start and in the same direction. If the SLAM pose misses the line
segment (drift larger than the lateral tolerance), the closest approach to
the start within ``return_radius_m`` counts instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .race_config import ClosureConfig


@dataclass
class Crossing:
    t: float
    x: float
    y: float
    travel: float
    lap: int  # laps completed with this crossing
    lateral: float  # offset along the start line from the start point [m]
    heading_err_deg: float
    kind: str  # "line" | "radius"


class LapCounter:
    def __init__(self, start: tuple[float, float, float], cfg: ClosureConfig, t0: float = 0.0, travel0: float = 0.0) -> None:
        self.p0 = np.array(start[:2], dtype=float)
        self.yaw0 = float(start[2])
        self.t0 = np.array([math.cos(self.yaw0), math.sin(self.yaw0)])
        self.n0 = np.array([-self.t0[1], self.t0[0]])
        self.cfg = cfg
        self.laps = 0
        self.last_t, self.last_travel = t0, travel0
        self.travel0 = travel0
        self.prev: tuple[float, float, float, float, float] | None = None  # (t, u, w, travel, heading err)
        self.closest: tuple[float, float, float, float, float] | None = None  # (d, t, x, y, travel)
        self.crossings: list[Crossing] = []

    def _armed(self, t: float, travel: float) -> bool:
        if self.crossings:  # the lap length is known now: race laps may be short and fast
            first = self.crossings[0].travel - self.travel0
            return travel - self.last_travel >= 0.6 * first and t - self.last_t >= 1.0
        return travel - self.last_travel >= self.cfg.arm_distance_m and t - self.last_t >= self.cfg.arm_time_s

    def _count(self, t: float, xy: np.ndarray, travel: float, lateral: float, head_deg: float, kind: str) -> Crossing:
        self.laps += 1
        self.last_t, self.last_travel = t, travel
        self.closest = None
        c = Crossing(t, float(xy[0]), float(xy[1]), travel, self.laps, lateral, head_deg, kind)
        self.crossings.append(c)
        return c

    def update(self, t: float, pose: tuple[float, float, float], travel: float) -> Crossing | None:
        rel = np.array(pose[:2]) - self.p0
        u, w = float(rel @ self.t0), float(rel @ self.n0)
        head = math.degrees(abs(math.atan2(math.sin(pose[2] - self.yaw0), math.cos(pose[2] - self.yaw0))))
        prev, self.prev = self.prev, (t, u, w, travel, head)
        if not self._armed(t, travel):
            return None
        if prev is not None and prev[1] < 0.0 <= u:
            a = -prev[1] / max(u - prev[1], 1e-9)  # interpolate the exact crossing
            w_c = prev[2] + a * (w - prev[2])
            if abs(w_c) <= self.cfg.lateral_tol_m and head <= self.cfg.heading_tol_deg:
                xy = self.p0 + w_c * self.n0
                return self._count(prev[0] + a * (t - prev[0]), xy, prev[3] + a * (travel - prev[3]), w_c, head, "line")
        d = math.hypot(rel[0], rel[1])
        if d <= self.cfg.return_radius_m and head <= self.cfg.heading_tol_deg:
            if self.closest is None or d < self.closest[0]:
                self.closest = (d, t, pose[0], pose[1], travel)
            return None
        if self.closest is not None:  # left the start region without crossing the line: closest approach
            _, tc, xc, yc, trc = self.closest
            lat = float((np.array([xc, yc]) - self.p0) @ self.n0)
            return self._count(tc, np.array([xc, yc]), trc, lat, head, "radius")
        return None


def cut_loop(xy: np.ndarray, start: tuple[float, float, float], cfg: ClosureConfig) -> tuple[np.ndarray, dict] | None:
    """First lap of an ordered pose sequence (offline, e.g. the SLAM graph).

    Returns ``(loop_xy, seam)``: the points from the start up to the first
    crossing (the crossing point appended) and the residual of the crossing in
    the start frame (``longitudinal`` ~ 0, ``lateral``, ``heading_deg``,
    ``index``). ``None`` if the sequence does not close.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return None
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    travel = np.concatenate(([0.0], np.cumsum(seg)))
    heading = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    counter = LapCounter(start, ClosureConfig(**{**cfg.__dict__, "arm_time_s": 0.0}))
    for k in range(len(xy)):
        c = counter.update(float(k), (xy[k, 0], xy[k, 1], float(heading[k])), float(travel[k]))
        if c is not None:
            end = int(math.floor(c.t)) if c.kind == "line" else int(c.t)
            loop = np.vstack((xy[: end + 1], [[c.x, c.y]]))
            yaw_end = float(heading[min(end + 1, len(xy) - 1)])
            dh = math.degrees(math.atan2(math.sin(yaw_end - start[2]), math.cos(yaw_end - start[2])))
            return loop, {"index": end, "lateral": c.lateral, "longitudinal": 0.0, "heading_deg": dh, "kind": c.kind, "length": c.travel}
    return None
