"""Evaluation of the race driver against the truth (evaluation only).

Everything here uses the true track (walls, curbs, pillars, lane widths),
which the car never sees: it measures how safe what the car planned or drove
really was. Poses given in the car's ``map`` frame are placed in the world
with ``T_world_map`` (the true start pose, as for the SLAM map figures).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .clearance import FootprintClearance, outline_points
from .map_metrics import se2_apply
from .race_config import VehicleLimits
from .sim_paths import ensure_tools_path


def load_truth_track(name: str):  # noqa: ANN201
    ensure_tools_path()
    from pose_dataset.tracks import load_track  # noqa: PLC0415

    return load_track(name)


class TrueTrack:
    def __init__(self, track, veh: VehicleLimits) -> None:  # noqa: ANN001
        self.track = track
        self.veh = veh
        pts, kinds = outline_points(track.geometry)
        self.clear = FootprintClearance(pts, veh.length_m, veh.width_m)
        self.cl = track.centerline.xy
        self.half = 0.5 * np.asarray(track.width, dtype=float)
        self.cl_tree = cKDTree(self.cl)

    def corners(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        hl, hw = 0.5 * self.veh.length_m, 0.5 * self.veh.width_m
        loc = np.array([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        return np.stack((x[:, None] + c * loc[None, :, 0] - s * loc[None, :, 1], y[:, None] + s * loc[None, :, 0] + c * loc[None, :, 1]), axis=-1)

    def lane_excess(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        """How far the footprint sticks out of the lane [m] (<= 0: inside)."""
        cor = self.corners(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(yaw)).reshape(-1, 2)
        d, k = self.cl_tree.query(cor)
        return (d - self.half[k]).reshape(-1, 4).max(axis=1)

    def clearance(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        return np.array([self.clear(float(a), float(b), float(c)) for a, b, c in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(yaw))])


def to_world(t_world_map, xy: np.ndarray, yaw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:  # noqa: ANN001
    w = se2_apply(t_world_map, xy)
    return w, (None if yaw is None else np.asarray(yaw) + t_world_map[2])


def plan_truth_metrics(plan_xy: np.ndarray, plan_heading: np.ndarray, ref_xy: np.ndarray, ref_normal: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       t_world_map, truth: TrueTrack, kappa: np.ndarray | None = None) -> dict:  # noqa: ANN001
    """Safety of a planned line and of its corridor, on the true track."""
    xy, yaw = to_world(t_world_map, plan_xy, plan_heading)
    clr = truth.clearance(xy[:, 0], xy[:, 1], yaw)
    exc = truth.lane_excess(xy[:, 0], xy[:, 1], yaw)
    ref_head = np.arctan2(np.gradient(ref_xy[:, 1]), np.gradient(ref_xy[:, 0]))
    sound = np.ones(len(ref_xy), dtype=bool)
    worst = 0.0
    for bound in (lo, hi):
        pts = ref_xy + bound[:, None] * ref_normal
        bw, byaw = to_world(t_world_map, pts, ref_head)
        c = truth.clearance(bw[:, 0], bw[:, 1], byaw)
        e = truth.lane_excess(bw[:, 0], bw[:, 1], byaw)
        viol = np.maximum(-c, 0.0) + np.maximum(e, 0.0)
        sound &= viol <= 0.0
        worst = max(worst, float(viol.max()))
    out = {
        "line_min_clearance_m": float(clr.min()), "line_p5_clearance_m": float(np.percentile(clr, 5)),
        "line_inside_lane_frac": float(np.mean(exc <= 0.0)), "line_max_lane_excess_m": float(exc.max()),
        "corridor_sound_frac": float(np.mean(sound)), "corridor_worst_violation_m": worst,
    }
    if kappa is not None:
        out["line_max_kappa"] = float(np.max(np.abs(kappa)))
    return out


def lap_progress(track, xy: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Arc length of the nearest centreline point (for per-lap statistics)."""
    k = cKDTree(track.centerline.xy).query(xy)[1]
    return track.centerline.s[k]


def heading_of(xy: np.ndarray) -> np.ndarray:
    return np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))


def wrap(a):  # noqa: ANN001, ANN201
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi
