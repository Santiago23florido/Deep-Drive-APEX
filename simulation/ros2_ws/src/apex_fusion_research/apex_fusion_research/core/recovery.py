"""Geometry-based backing maneuvers, independent of track shape and ROS.

Check the swept footprint of each reverse arc, then the forward space opened
by its end pose. The local planner supplies the desired curvature; no side or
sequence of turns is prescribed here. Unobserved space is not proven clear.
"""
from dataclasses import dataclass
import math

import numpy as np

from .race_config import ReactiveConfig, VehicleLimits
from .reactive import LocalGrid


@dataclass(frozen=True)
class RecoveryArc:
    kappa: float
    distance_m: float
    forward_free_m: float


def arc_poses(curvatures: np.ndarray, stations: np.ndarray, veh: VehicleLimits) -> np.ndarray:
    """Base-link poses for signed rear-axle travel, including reverse."""
    angle = curvatures[:, None] * stations[None, :]
    chord = stations[None, :] * np.sinc(angle / (2 * np.pi))
    return np.stack((veh.rear_offset_m + chord * np.cos(angle / 2) - veh.rear_offset_m * np.cos(angle),
                     chord * np.sin(angle / 2) - veh.rear_offset_m * np.sin(angle), angle), axis=-1)


def _clearance(grid: LocalGrid, poses: np.ndarray, veh: VehicleLimits, offsets: np.ndarray | None = None) -> np.ndarray:
    angle = poses[..., 2]
    forward = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    return np.minimum.reduce([grid.query(poses[..., :2] + off * forward) - veh.circle_radius
                              for off in (veh.circle_offsets if offsets is None else offsets)])


class _RecoverySpace:
    """Check each nearby obstacle against its own initial clearance.

    A side return behind the bumper is not a wall across the reverse path.
    Add the rear stopping reserve along the car's longitudinal axis, rather
    than projecting every nearby side return onto that axis. Obstacles outside
    the hard margin must stay outside it. Only an already squeezed obstacle
    gets the existing 2 cm escape tolerance, independently of other obstacles.
    """

    def __init__(self, cloud: np.ndarray, veh: VehicleLimits, cfg: ReactiveConfig, rear_buffer: float):
        self.veh = veh
        self.margin = cfg.c_hard_m
        self.offsets = np.asarray(veh.circle_offsets)
        if rear_buffer > 0:
            self.offsets = np.append(self.offsets, min(self.offsets) - rear_buffer)
        centres = np.column_stack((self.offsets, np.zeros(len(self.offsets))))
        initial = np.linalg.norm(cloud[:, None, :] - centres[None, :, :], axis=-1).min(axis=1) - veh.circle_radius - .025
        near = initial < self.margin
        self.grid = LocalGrid(cloud[~near])
        self.near = cloud[near]
        initial = initial[near]
        self.threshold = initial - .02

    def blocked(self, poses: np.ndarray) -> np.ndarray:
        blocked = _clearance(self.grid, poses, self.veh, self.offsets) < self.margin
        if len(self.near):
            forward = np.stack((np.cos(poses[..., 2]), np.sin(poses[..., 2])), axis=-1)
            clearance = np.minimum.reduce([
                np.linalg.norm((poses[..., :2] + off * forward)[..., None, :] - self.near, axis=-1)
                - self.veh.circle_radius - .025 for off in self.offsets])
            blocked |= (clearance < self.threshold - 1e-9).any(axis=-1)
        return blocked


def reverse_path_clear(cloud: np.ndarray, veh: VehicleLimits, cfg: ReactiveConfig,
                       kappa: float, distance_m: float, rear_buffer_m: float = 0.0) -> bool:
    if not len(cloud) or not math.isfinite(distance_m):
        return False
    space = _RecoverySpace(cloud, veh, cfg, rear_buffer_m)
    stations = -np.linspace(0, max(0., distance_m), max(2, int(math.ceil(distance_m / .025)) + 1))
    return not bool(space.blocked(arc_poses(np.array([kappa]), stations, veh)).any())


def choose_recovery_arc(cloud: np.ndarray, veh: VehicleLimits, cfg: ReactiveConfig,
                        distance_limit_m: float, desired_kappa: float,
                        rear_buffer_m: float = 0.0) -> RecoveryArc | None:
    """Choose a checked reverse arc which opens a useful forward continuation.

    Distance/time/power authorization remains the caller's responsibility.
    The forward probes only score the maneuver; they are never sent as an
    unchecked driving path. Normal LiDAR planning must run again afterwards.
    """
    if not len(cloud) or not math.isfinite(distance_limit_m) or distance_limit_m < .1:
        return None
    left, right = veh.curvature_limit(True, planning=True), veh.curvature_limit(False, planning=True)
    curvatures = np.array([-right, -.5 * right, 0., .5 * left, left])
    stations = -np.linspace(0., distance_limit_m, max(3, int(math.ceil(distance_limit_m / .025)) + 1))
    reverse = arc_poses(curvatures, stations, veh)
    space = _RecoverySpace(cloud, veh, cfg, rear_buffer_m)
    blocked = space.blocked(reverse)
    first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(stations))
    end = np.maximum(0, first - 1)
    distance = -stations[end]
    valid = distance >= .1
    if not valid.any():
        return None
    endpoints = reverse[np.arange(len(curvatures)), end]
    desired_kappa = float(np.clip(desired_kappa, -right, left))
    forward_k = np.unique(np.append(curvatures, desired_kappa))
    forward_s = np.arange(.05, 2.201, .05)
    forward = arc_poses(forward_k, forward_s, veh)
    c, s = np.cos(endpoints[:, 2]), np.sin(endpoints[:, 2])
    probes = np.empty((len(curvatures), len(forward_k), len(forward_s), 3))
    probes[..., 0] = endpoints[:, None, None, 0] + c[:, None, None] * forward[None, ..., 0] - s[:, None, None] * forward[None, ..., 1]
    probes[..., 1] = endpoints[:, None, None, 1] + s[:, None, None] * forward[None, ..., 0] + c[:, None, None] * forward[None, ..., 1]
    probes[..., 2] = endpoints[:, None, None, 2] + forward[None, ..., 2]
    blocked = space.blocked(probes)
    first = np.where(blocked.any(axis=2), blocked.argmax(axis=2), len(forward_s))
    free = np.where(first < len(forward_s), forward_s[np.minimum(first, len(forward_s) - 1)] - .05, forward_s[-1])
    # Trade usable space against the desired local direction. The desired
    # direction comes from perception, equally for left and right curves.
    target_heading = desired_kappa * min(1., float(forward_s[-1]))
    heading = endpoints[:, None, 2] + forward_k[None, :] * np.minimum(free, 1.)
    error = np.arctan2(np.sin(heading - target_heading), np.cos(heading - target_heading))
    score = free - .3 * error**2 - .05 * (abs(curvatures) / max(left, right))[:, None]
    score[~valid] = -np.inf
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return RecoveryArc(float(curvatures[i]), float(distance[i]), float(free[i, j]))
