#!/usr/bin/env python3
"""ROS 2 node for PC: subscribe LaserScan, inspect scans, and build LiDAR-only maps."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def _safe_value(ranges: np.ndarray, idx: int) -> float:
    value = float(ranges[idx % ranges.size])
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return value


def _safe_value_by_angle(msg: LaserScan, ranges: np.ndarray, target_deg: float) -> float:
    if ranges.size == 0 or msg.angle_increment == 0.0:
        return 0.0

    target_rad = np.deg2rad(target_deg)
    idx = int(round((target_rad - msg.angle_min) / msg.angle_increment))
    return _safe_value(ranges, idx)


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _display_angle_to_laser_angle_deg(angle_deg: float, right_positive: bool) -> float:
    if right_positive:
        return _normalize_angle_deg(-float(angle_deg))
    return _normalize_angle_deg(float(angle_deg))


def _laser_angle_to_display_angle_deg(angle_deg: float, right_positive: bool) -> float:
    if right_positive:
        return _normalize_angle_deg(-float(angle_deg))
    return _normalize_angle_deg(float(angle_deg))


def _parse_probe_angles(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        values.append(_normalize_angle_deg(float(token)))
    return values


def _window_values_by_angle(
    msg: LaserScan,
    ranges: np.ndarray,
    valid_mask: np.ndarray,
    target_deg: float,
    window_deg: float,
) -> np.ndarray:
    if ranges.size == 0 or msg.angle_increment == 0.0:
        return np.empty((0,), dtype=np.float32)

    angles_deg = np.degrees(
        msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
    )
    delta_deg = np.abs(((angles_deg - float(target_deg) + 180.0) % 360.0) - 180.0)
    mask = valid_mask & (delta_deg <= max(0.1, float(window_deg)))
    return ranges[mask]


def _probe_distance_by_angle(
    msg: LaserScan,
    ranges: np.ndarray,
    valid_mask: np.ndarray,
    target_deg: float,
    window_deg: float,
) -> float:
    values = _window_values_by_angle(msg, ranges, valid_mask, target_deg, window_deg)
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    stride = max(1, int(math.ceil(points.shape[0] / float(max_points))))
    return points[::stride]


def _rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    if points.size == 0 or abs(angle_rad) < 1e-9:
        return points.copy()

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rot.T


def _display_xy_from_polar(
    angles_rad: np.ndarray,
    ranges_m: np.ndarray,
    *,
    right_positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = -ranges_m * np.cos(angles_rad)
    y = ranges_m * np.sin(angles_rad)
    if right_positive:
        y = -y
    return x, y


class PointCloudPlotter:
    """Live XY point-cloud view from LaserScan ranges."""

    def __init__(self, max_range_m: float, draw_every_s: float, right_positive: bool) -> None:
        self._draw_every_s = max(0.02, float(draw_every_s))
        self._last_draw = 0.0
        self._base_limit_m = max(0.5, float(max_range_m))
        self._right_positive = bool(right_positive)

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for --plot. Install it in your PC venv with "
                "`python -m pip install matplotlib`."
            ) from exc

        self._plt = plt
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=(7, 7))
        self._scatter = self._ax.scatter([], [], s=7, c="tab:cyan", alpha=0.85)
        self._ax.scatter([0.0], [0.0], s=35, c="tab:red", label="LiDAR")
        y_axis_label = "y [m] (derecha+)" if self._right_positive else "y [m] (izquierda+)"
        self._ax.set_title("Live LiDAR Point Cloud (XY)")
        self._ax.set_xlabel("x [m] (frente+)")
        self._ax.set_ylabel(y_axis_label)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.grid(True, alpha=0.25)
        self._ax.legend(loc="upper right")
        self._set_axes_limit(self._base_limit_m)
        self._fig.tight_layout()
        self._draw()

    def _set_axes_limit(self, limit_m: float) -> None:
        self._ax.set_xlim(-limit_m, limit_m)
        self._ax.set_ylim(-limit_m, limit_m)
        if self._right_positive:
            self._ax.invert_yaxis()

    def _draw(self) -> None:
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def update(self, angles: np.ndarray, ranges: np.ndarray) -> None:
        now = time.time()
        if now - self._last_draw < self._draw_every_s:
            return
        self._last_draw = now

        if ranges.size == 0:
            self._scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            self._draw()
            return

        x, y = _display_xy_from_polar(
            angles,
            ranges,
            right_positive=self._right_positive,
        )
        points = np.column_stack((x, y))
        self._scatter.set_offsets(points)

        dynamic_limit = float(np.percentile(ranges, 95)) * 1.2
        if np.isfinite(dynamic_limit):
            self._set_axes_limit(max(self._base_limit_m, max(1.0, dynamic_limit)))

        self._draw()

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)


class OccupancyGridMap:
    """2D occupancy grid in log-odds form."""

    def __init__(
        self,
        resolution_m: float,
        size_m: float,
        log_occ: float = 0.85,
        log_free: float = -0.40,
    ) -> None:
        self.resolution_m = max(0.01, float(resolution_m))
        self.size_m = max(2.0, float(size_m))
        self.width = int(math.ceil(self.size_m / self.resolution_m))
        self.height = self.width
        self.origin_x = -0.5 * self.width * self.resolution_m
        self.origin_y = -0.5 * self.height * self.resolution_m

        self._log_occ = float(log_occ)
        self._log_free = float(log_free)
        self._log_min = -4.0
        self._log_max = 4.0
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        cx = int(math.floor((x_m - self.origin_x) / self.resolution_m))
        cy = int(math.floor((y_m - self.origin_y) / self.resolution_m))
        return cx, cy

    def _inside(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def _raytrace_and_update(self, x0: int, y0: int, x1: int, y1: int) -> None:
        # Bresenham ray update: free along the ray, occupied at hit endpoint.
        x = x0
        y = y0
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            if x == x1 and y == y1:
                self.log_odds[y, x] += self._log_occ
                break

            self.log_odds[y, x] += self._log_free
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def update_scan(self, pose_xy_yaw: np.ndarray, points_local_xy: np.ndarray) -> None:
        if points_local_xy.size == 0:
            return

        px, py, yaw = float(pose_xy_yaw[0]), float(pose_xy_yaw[1]), float(pose_xy_yaw[2])
        start_cx, start_cy = self.world_to_cell(px, py)
        if not self._inside(start_cx, start_cy):
            return

        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        points_world = points_local_xy @ rot.T + np.array([px, py], dtype=np.float32)

        for end in points_world:
            end_cx, end_cy = self.world_to_cell(float(end[0]), float(end[1]))
            if not self._inside(end_cx, end_cy):
                continue
            self._raytrace_and_update(start_cx, start_cy, end_cx, end_cy)

        np.clip(self.log_odds, self._log_min, self._log_max, out=self.log_odds)

    def to_probability(self) -> np.ndarray:
        return 1.0 - (1.0 / (1.0 + np.exp(self.log_odds)))


class OccupancyMapPlotter:
    """Live occupancy map view."""

    def __init__(self, grid_map: OccupancyGridMap, draw_every_s: float, title: str) -> None:
        self._draw_every_s = max(0.05, float(draw_every_s))
        self._last_draw = 0.0

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for --map-plot. Install it in your PC venv with "
                "`python -m pip install matplotlib`."
            ) from exc

        self._plt = plt
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=(8, 8))
        extent = (
            grid_map.origin_x,
            grid_map.origin_x + grid_map.width * grid_map.resolution_m,
            grid_map.origin_y,
            grid_map.origin_y + grid_map.height * grid_map.resolution_m,
        )
        self._img = self._ax.imshow(
            grid_map.to_probability(),
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
        )
        (self._path_line,) = self._ax.plot([], [], color="tab:orange", linewidth=1.2, label="Estimated path")
        self._pose_dot = self._ax.scatter([0.0], [0.0], s=30, c="tab:red", label="Current pose")
        self._ax.set_title(title)
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.grid(True, alpha=0.20)
        self._ax.legend(loc="upper right")
        self._fig.tight_layout()
        self._draw()

    def _draw(self) -> None:
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def update(self, grid_map: OccupancyGridMap, path_xy: list[tuple[float, float]], pose: np.ndarray) -> None:
        now = time.time()
        if now - self._last_draw < self._draw_every_s:
            return
        self._last_draw = now

        self._img.set_data(grid_map.to_probability())
        if path_xy:
            path_arr = np.asarray(path_xy, dtype=np.float32)
            self._path_line.set_data(path_arr[:, 0], path_arr[:, 1])
        self._pose_dot.set_offsets(np.array([[float(pose[0]), float(pose[1])]], dtype=np.float32))
        self._draw()

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)


@dataclass
class MappingState:
    x_m: float
    y_m: float
    yaw_deg: float
    rot_score: float
    trans_score: float
    pose_source: str


class LidarOnlyMapper:
    """LiDAR mapper with selectable pose source (scan matching or external odom)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._grid = OccupancyGridMap(
            resolution_m=args.map_resolution,
            size_m=args.map_size_m,
        )
        self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._path_xy: list[tuple[float, float]] = [(0.0, 0.0)]
        self._max_path_size = 5000

        self._prev_scan: Optional[np.ndarray] = None
        self._prev_points_for_match: Optional[np.ndarray] = None
        self._pose_source = str(args.map_pose_source).strip().lower()
        if self._pose_source not in {"odom", "scanmatch"}:
            self._pose_source = "odom"
        self._odom_pose: Optional[np.ndarray] = None

        self._scanmatch_max_rot_deg = max(0.5, float(args.scanmatch_max_rot_deg))
        self._scanmatch_max_trans_m = max(0.02, float(args.scanmatch_max_trans_m))
        self._scanmatch_trans_step_m = max(0.01, float(args.scanmatch_trans_step_m))
        self._scanmatch_max_points = max(30, int(args.scanmatch_max_points))
        self._map_update_max_points = max(30, int(args.map_update_max_points))
        self._scanmatch_min_overlap = 20

        self._map_min_range = max(0.01, float(args.map_min_range))
        self._map_max_range = max(self._map_min_range + 0.1, float(args.map_max_range))
        self._save_path = str(args.map_save_path).strip()
        self._plotter: Optional[OccupancyMapPlotter] = None
        if args.map_plot:
            title = (
                "LiDAR Occupancy Grid (Pose from /odom)"
                if self._pose_source == "odom"
                else "LiDAR Occupancy Grid (Scan Matching)"
            )
            self._plotter = OccupancyMapPlotter(self._grid, draw_every_s=args.map_plot_every_s, title=title)

    @property
    def pose_source(self) -> str:
        return self._pose_source

    def set_odom_pose(self, x_m: float, y_m: float, yaw_rad: float) -> None:
        self._odom_pose = np.array([float(x_m), float(y_m), float(yaw_rad)], dtype=np.float32)

    def _range_valid_mask(self, msg: LaserScan, ranges: np.ndarray) -> np.ndarray:
        valid = np.isfinite(ranges)
        min_r = self._map_min_range
        max_r = self._map_max_range
        if msg.range_min > 0.0 and np.isfinite(msg.range_min):
            min_r = max(min_r, float(msg.range_min))
        if msg.range_max > 0.0 and np.isfinite(msg.range_max):
            max_r = min(max_r, float(msg.range_max))
        valid = valid & (ranges >= min_r) & (ranges <= max_r)
        return valid

    @staticmethod
    def _scan_to_points(angles: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        if ranges.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.column_stack((x, y)).astype(np.float32)

    def _estimate_rotation(self, prev_scan: np.ndarray, curr_scan: np.ndarray, angle_increment: float) -> tuple[float, float]:
        if prev_scan.size == 0 or curr_scan.size == 0 or prev_scan.size != curr_scan.size:
            return 0.0, 0.0
        if abs(angle_increment) < 1e-6:
            return 0.0, 0.0

        max_shift = int(round(math.radians(self._scanmatch_max_rot_deg) / abs(angle_increment)))
        if max_shift <= 0:
            return 0.0, 0.0

        prev_valid = prev_scan > 0.0
        curr_valid = curr_scan > 0.0
        if np.count_nonzero(prev_valid) < self._scanmatch_min_overlap:
            return 0.0, 0.0

        best_shift = 0
        best_error = float("inf")
        best_overlap = 0

        for shift in range(-max_shift, max_shift + 1):
            rolled_curr = np.roll(curr_scan, shift)
            rolled_valid = np.roll(curr_valid, shift)
            overlap_mask = prev_valid & rolled_valid
            overlap = int(np.count_nonzero(overlap_mask))
            if overlap < self._scanmatch_min_overlap:
                continue

            err = float(np.mean(np.abs(prev_scan[overlap_mask] - rolled_curr[overlap_mask])))
            if err < best_error or (math.isclose(err, best_error) and overlap > best_overlap):
                best_error = err
                best_overlap = overlap
                best_shift = shift

        if not np.isfinite(best_error):
            return 0.0, 0.0

        rot_conf = max(0.0, min(1.0, best_overlap / float(max(1, np.count_nonzero(prev_valid)))))
        rot_conf *= 1.0 / (1.0 + best_error)
        return best_shift * angle_increment, rot_conf

    def _estimate_translation_icp(self, prev_points: np.ndarray, curr_points_rot: np.ndarray) -> tuple[float, float, float]:
        """Robust translation estimate from nearest-neighbor correspondences."""
        if prev_points.size == 0 or curr_points_rot.size == 0:
            return 0.0, 0.0, 0.0

        prev = prev_points.astype(np.float32, copy=False)
        curr = curr_points_rot.astype(np.float32, copy=False)
        max_corr_dist = max(0.08, self._scanmatch_max_trans_m * 1.8)
        max_corr_d2 = max_corr_dist * max_corr_dist

        diffs = prev[None, :, :] - curr[:, None, :]
        d2 = np.sum(diffs * diffs, axis=2)
        nn_idx = np.argmin(d2, axis=1)
        nn_d2 = d2[np.arange(curr.shape[0]), nn_idx]
        keep = nn_d2 <= max_corr_d2

        keep_count = int(np.count_nonzero(keep))
        if keep_count < max(8, self._scanmatch_min_overlap // 2):
            return 0.0, 0.0, 0.0

        base_delta = prev[nn_idx[keep]] - curr[keep]
        tx, ty = np.median(base_delta, axis=0).astype(np.float32)

        shifted_curr = curr + np.array([tx, ty], dtype=np.float32)
        diffs_2 = prev[None, :, :] - shifted_curr[:, None, :]
        d2_2 = np.sum(diffs_2 * diffs_2, axis=2)
        nn_idx_2 = np.argmin(d2_2, axis=1)
        nn_d2_2 = d2_2[np.arange(shifted_curr.shape[0]), nn_idx_2]
        keep_2 = nn_d2_2 <= max_corr_d2

        if int(np.count_nonzero(keep_2)) >= 6:
            refine_delta = prev[nn_idx_2[keep_2]] - shifted_curr[keep_2]
            dtx, dty = np.median(refine_delta, axis=0).astype(np.float32)
            tx += dtx
            ty += dty

        norm = float(math.hypot(float(tx), float(ty)))
        if norm > self._scanmatch_max_trans_m > 0.0:
            scale = self._scanmatch_max_trans_m / norm
            tx *= scale
            ty *= scale

        mean_err = float(math.sqrt(max(0.0, float(np.mean(nn_d2[keep])))))
        conf_count = keep_count / float(max(1, curr.shape[0]))
        conf_err = 1.0 / (1.0 + mean_err)
        conf = max(0.0, min(1.0, conf_count * conf_err))
        return float(tx), float(ty), conf

    def _estimate_translation_grid(self, prev_points: np.ndarray, curr_points_rot: np.ndarray) -> tuple[float, float, float]:
        if prev_points.size == 0 or curr_points_rot.size == 0:
            return 0.0, 0.0, 0.0

        step = self._scanmatch_trans_step_m
        search = self._scanmatch_max_trans_m
        padding = search + step

        min_xy = np.min(prev_points, axis=0) - padding
        max_xy = np.max(prev_points, axis=0) + padding
        grid_size = np.ceil((max_xy - min_xy) / step).astype(np.int32) + 1
        width = int(grid_size[0])
        height = int(grid_size[1])

        if width <= 1 or height <= 1 or (width * height) > 2_500_000:
            return 0.0, 0.0, 0.0

        occ_grid = np.zeros((height, width), dtype=np.uint8)
        prev_idx = np.floor((prev_points - min_xy) / step).astype(np.int32)
        valid_prev = (
            (prev_idx[:, 0] >= 0)
            & (prev_idx[:, 0] < width)
            & (prev_idx[:, 1] >= 0)
            & (prev_idx[:, 1] < height)
        )
        occ_grid[prev_idx[valid_prev, 1], prev_idx[valid_prev, 0]] = 1

        candidates = np.arange(-search, search + 0.5 * step, step, dtype=np.float32)
        best_score = -1
        best_tx = 0.0
        best_ty = 0.0
        best_norm = float("inf")

        for tx in candidates:
            shifted_x = curr_points_rot[:, 0] + tx
            idx_x = np.floor((shifted_x - min_xy[0]) / step).astype(np.int32)

            for ty in candidates:
                shifted_y = curr_points_rot[:, 1] + ty
                idx_y = np.floor((shifted_y - min_xy[1]) / step).astype(np.int32)

                in_bounds = (
                    (idx_x >= 0)
                    & (idx_x < width)
                    & (idx_y >= 0)
                    & (idx_y < height)
                )
                if not np.any(in_bounds):
                    continue

                score = int(np.count_nonzero(occ_grid[idx_y[in_bounds], idx_x[in_bounds]]))
                norm = float(tx * tx + ty * ty)
                if score > best_score or (score == best_score and norm < best_norm):
                    best_score = score
                    best_tx = float(tx)
                    best_ty = float(ty)
                    best_norm = norm

        if best_score < 0:
            return 0.0, 0.0, 0.0

        conf = max(0.0, min(1.0, best_score / float(max(1, curr_points_rot.shape[0]))))
        return best_tx, best_ty, conf

    def update_from_scan(self, msg: LaserScan, ranges: np.ndarray) -> Optional[MappingState]:
        if ranges.size == 0:
            return None

        angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
        valid_mask = self._range_valid_mask(msg, ranges)
        if not np.any(valid_mask):
            return None

        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        points = self._scan_to_points(valid_angles, valid_ranges)

        current_scan = np.zeros_like(ranges, dtype=np.float32)
        current_scan[valid_mask] = ranges[valid_mask]
        points_for_match = _downsample_points(points, self._scanmatch_max_points)

        delta_theta = 0.0
        delta_tx = 0.0
        delta_ty = 0.0
        rot_score = 0.0
        trans_score = 0.0

        if self._pose_source == "odom":
            if self._odom_pose is None:
                return None
            self._pose[:] = self._odom_pose
            rot_score = 1.0
            trans_score = 1.0
        elif self._prev_scan is not None and self._prev_points_for_match is not None:
            delta_theta, rot_score = self._estimate_rotation(
                self._prev_scan,
                current_scan,
                float(msg.angle_increment),
            )
            curr_rot = _rotate_points(points_for_match, delta_theta)
            delta_tx, delta_ty, trans_score = self._estimate_translation_icp(
                self._prev_points_for_match,
                curr_rot,
            )
            if trans_score < 0.08:
                grid_tx, grid_ty, grid_score = self._estimate_translation_grid(
                    self._prev_points_for_match,
                    curr_rot,
                )
                if grid_score > trans_score:
                    delta_tx, delta_ty, trans_score = grid_tx, grid_ty, grid_score

            yaw_prev = float(self._pose[2])
            dx_world = math.cos(yaw_prev) * delta_tx - math.sin(yaw_prev) * delta_ty
            dy_world = math.sin(yaw_prev) * delta_tx + math.cos(yaw_prev) * delta_ty
            self._pose[0] += dx_world
            self._pose[1] += dy_world
            self._pose[2] = _wrap_angle_rad(yaw_prev + delta_theta)

        points_for_map = _downsample_points(points, self._map_update_max_points)
        self._grid.update_scan(self._pose, points_for_map)

        self._path_xy.append((float(self._pose[0]), float(self._pose[1])))
        if len(self._path_xy) > self._max_path_size:
            self._path_xy = self._path_xy[-self._max_path_size :]

        if self._plotter is not None:
            self._plotter.update(self._grid, self._path_xy, self._pose)

        self._prev_scan = current_scan
        self._prev_points_for_match = points_for_match

        return MappingState(
            x_m=float(self._pose[0]),
            y_m=float(self._pose[1]),
            yaw_deg=math.degrees(float(self._pose[2])),
            rot_score=rot_score,
            trans_score=trans_score,
            pose_source=self._pose_source,
        )

    def shutdown(self) -> None:
        if self._plotter is not None:
            self._plotter.close()

        if not self._save_path:
            return

        out_path = Path(self._save_path).expanduser()
        if out_path.suffix != ".npy":
            out_path = out_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        probability = self._grid.to_probability().astype(np.float32)
        np.save(out_path, probability)

        meta_path = out_path.with_suffix(".npy.meta.txt")
        with meta_path.open("w", encoding="utf-8") as meta_file:
            meta_file.write(f"resolution_m={self._grid.resolution_m}\n")
            meta_file.write(f"origin_x={self._grid.origin_x}\n")
            meta_file.write(f"origin_y={self._grid.origin_y}\n")
            meta_file.write(f"width={self._grid.width}\n")
            meta_file.write(f"height={self._grid.height}\n")
            meta_file.write(f"pose_x={float(self._pose[0])}\n")
            meta_file.write(f"pose_y={float(self._pose[1])}\n")
            meta_file.write(f"pose_yaw_rad={float(self._pose[2])}\n")


class LidarSubscriberNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pc_lidar_subscriber")

        self._topic = args.topic
        self._full = args.full
        self._right_positive = bool(args.right_positive)
        self._print_every_s = max(0.1, float(args.print_every_s))
        self._last_print = 0.0
        self._plotter: Optional[PointCloudPlotter] = None
        self._probe_angles_deg = _parse_probe_angles(args.probe_angles_deg)
        self._probe_window_deg = max(0.25, float(args.probe_window_deg))
        self._snapshot_scans_target = max(0, int(args.snapshot_scans))
        self._snapshot_output = str(args.snapshot_output).strip()
        self._snapshot_exit = bool(args.snapshot_exit)
        self._snapshot_sum: Optional[np.ndarray] = None
        self._snapshot_counts: Optional[np.ndarray] = None
        self._snapshot_scans_collected = 0
        self._snapshot_saved = False

        if args.plot:
            self._plotter = PointCloudPlotter(
                max_range_m=args.plot_max_range,
                draw_every_s=args.plot_every_s,
                right_positive=self._right_positive,
            )
            self.get_logger().info("Live point-cloud plotting enabled.")

        enable_mapping = bool(args.map or args.map_plot or str(args.map_save_path).strip())
        self._mapper: Optional[LidarOnlyMapper] = None
        if enable_mapping:
            self._mapper = LidarOnlyMapper(args)
            self.get_logger().info("LiDAR-only mapper enabled.")
            if self._mapper.pose_source == "odom":
                self.create_subscription(Odometry, args.odom_topic, self._odom_cb, 20)
                self.get_logger().info(f"Mapping pose source: /odom ({args.odom_topic})")
            else:
                self.get_logger().info("Mapping pose source: scan matching")

        self.create_subscription(LaserScan, self._topic, self._scan_cb, qos_profile_sensor_data)
        self.get_logger().info(f"Suscrito a {self._topic}")
        convention = "x positiva hacia delante; derecha positiva / izquierda negativa" if self._right_positive else "x positiva hacia delante; izquierda positiva / derecha negativa"
        self.get_logger().info(f"Convencion visual: {convention}")
        if self._probe_angles_deg:
            probe_tokens = ", ".join(f"{angle:+.0f}" for angle in self._probe_angles_deg)
            self.get_logger().info(
                f"Sondas angulares activas: [{probe_tokens}] con ventana +/-{self._probe_window_deg:.1f} deg"
            )
        if self._snapshot_scans_target > 0:
            self.get_logger().info(
                f"Snapshot estatico activado: {self._snapshot_scans_target} scans"
            )

    def _probe_distance_display_angle(
        self,
        msg: LaserScan,
        ranges: np.ndarray,
        valid_mask: np.ndarray,
        display_angle_deg: float,
    ) -> float:
        return _probe_distance_by_angle(
            msg,
            ranges,
            valid_mask,
            _display_angle_to_laser_angle_deg(display_angle_deg, self._right_positive),
            self._probe_window_deg,
        )

    def _format_probe_summary(self, msg: LaserScan, ranges: np.ndarray, valid_mask: np.ndarray) -> str:
        if not self._probe_angles_deg:
            return ""

        parts: list[str] = []
        for angle_deg in self._probe_angles_deg:
            distance_m = self._probe_distance_display_angle(
                msg,
                ranges,
                valid_mask,
                angle_deg,
            )
            if distance_m > 0.0:
                parts.append(f"{angle_deg:+.0f}={distance_m:.3f}m")
            else:
                parts.append(f"{angle_deg:+.0f}=nan")
        return " | probes[" + " ".join(parts) + "]"

    def _snapshot_base_path(self) -> Path:
        if self._snapshot_output:
            return Path(self._snapshot_output).expanduser().resolve()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return (Path.cwd() / f"lidar_static_snapshot_{timestamp}").resolve()

    def _save_snapshot_plot(
        self,
        *,
        msg: LaserScan,
        png_path: Path,
        averaged_ranges: np.ndarray,
        valid_mask: np.ndarray,
        probe_summary: dict[str, float],
        closest_distance_m: float,
        closest_angle_deg: float,
    ) -> bool:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.get_logger().warning(
                "matplotlib no esta disponible; no se pudo guardar la imagen del snapshot"
            )
            return False

        valid_ranges = averaged_ranges[valid_mask]
        if valid_ranges.size == 0:
            self.get_logger().warning(
                "Snapshot sin puntos validos; no se pudo generar la imagen"
            )
            return False

        angles_deg = np.degrees(
            msg.angle_min + np.arange(averaged_ranges.size, dtype=np.float32) * msg.angle_increment
        )
        valid_angles_rad = np.radians(angles_deg[valid_mask])
        x, y = _display_xy_from_polar(
            valid_angles_rad,
            valid_ranges,
            right_positive=self._right_positive,
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, s=8, c="tab:cyan", alpha=0.80, label="LiDAR promedio")
        ax.scatter([0.0], [0.0], s=40, c="tab:red", label="LiDAR")

        max_range_m = float(np.nanpercentile(valid_ranges, 95))
        axis_limit_m = max(1.0, max_range_m * 1.20)
        ax.set_xlim(-axis_limit_m, axis_limit_m)
        ax.set_ylim(-axis_limit_m, axis_limit_m)
        if self._right_positive:
            ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x [m] (frente+)")
        ax.set_ylabel("y [m] (derecha+)" if self._right_positive else "y [m] (izquierda+)")
        ax.set_title(
            "Snapshot LiDAR estatico\n"
            f"topic={self._topic} scans={self._snapshot_scans_collected}"
        )

        for probe_key, distance_m in probe_summary.items():
            if distance_m <= 0.0:
                continue
            angle_deg = float(probe_key.replace("deg", ""))
            angle_rad = math.radians(angle_deg)
            px_arr, py_arr = _display_xy_from_polar(
                np.asarray([angle_rad], dtype=np.float32),
                np.asarray([distance_m], dtype=np.float32),
                right_positive=self._right_positive,
            )
            px = float(px_arr[0])
            py = float(py_arr[0])
            ax.plot([0.0, px], [0.0, py], linestyle="--", linewidth=1.0, alpha=0.6, color="tab:orange")
            ax.scatter([px], [py], s=28, c="tab:orange")
            label_dx_arr, label_dy_arr = _display_xy_from_polar(
                np.asarray([angle_rad], dtype=np.float32),
                np.asarray([0.05 * axis_limit_m], dtype=np.float32),
                right_positive=self._right_positive,
            )
            label_dx = float(label_dx_arr[0])
            label_dy = float(label_dy_arr[0])
            ax.text(
                px + label_dx,
                py + label_dy,
                f"{angle_deg:+.0f} deg\n{distance_m:.3f} m",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )

        if closest_distance_m > 0.0:
            closest_angle_rad = math.radians(closest_angle_deg)
            cx_arr, cy_arr = _display_xy_from_polar(
                np.asarray([closest_angle_rad], dtype=np.float32),
                np.asarray([closest_distance_m], dtype=np.float32),
                right_positive=self._right_positive,
            )
            cx = float(cx_arr[0])
            cy = float(cy_arr[0])
            ax.scatter([cx], [cy], s=44, c="tab:purple", label="Punto mas cercano")
            ax.text(
                cx,
                cy,
                f"min\n{closest_distance_m:.3f} m\n{closest_angle_deg:+.0f} deg",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f5e6ff", "alpha": 0.90, "edgecolor": "#9b59b6"},
            )

        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        return True

    def _save_snapshot(self, msg: LaserScan) -> None:
        if (
            self._snapshot_saved
            or self._snapshot_scans_target <= 0
            or self._snapshot_sum is None
            or self._snapshot_counts is None
        ):
            return

        averaged_ranges = np.divide(
            self._snapshot_sum,
            np.maximum(self._snapshot_counts, 1),
            dtype=np.float32,
        )
        averaged_ranges[self._snapshot_counts <= 0] = np.nan
        valid_mask = np.isfinite(averaged_ranges) & (averaged_ranges > 0.0)
        valid_ranges = averaged_ranges[valid_mask]

        base_path = self._snapshot_base_path()
        base_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = base_path if base_path.suffix == ".json" else base_path.with_suffix(".json")
        csv_path = base_path.with_suffix(".csv")
        png_path = base_path.with_suffix(".png")

        closest_distance_m = 0.0
        closest_angle_deg = 0.0
        if valid_ranges.size > 0:
            valid_indices = np.flatnonzero(valid_mask)
            closest_idx = int(valid_indices[int(np.argmin(valid_ranges))])
            closest_distance_m = float(averaged_ranges[closest_idx])
            closest_angle_deg = _laser_angle_to_display_angle_deg(
                math.degrees(msg.angle_min + closest_idx * msg.angle_increment),
                self._right_positive,
            )

        probe_summary: dict[str, float] = {}
        for angle_deg in self._probe_angles_deg:
            probe_summary[f"{angle_deg:+.0f}deg"] = self._probe_distance_display_angle(
                msg,
                averaged_ranges,
                valid_mask,
                angle_deg,
            )

        angle_values_deg = np.degrees(
            msg.angle_min + np.arange(averaged_ranges.size, dtype=np.float32) * msg.angle_increment
        )
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("angle_deg,range_m,count\n")
            for angle_deg, range_m, count in zip(
                angle_values_deg.tolist(),
                averaged_ranges.tolist(),
                self._snapshot_counts.tolist(),
            ):
                range_str = "nan" if not np.isfinite(range_m) else f"{float(range_m):.6f}"
                display_angle_deg = _laser_angle_to_display_angle_deg(
                    angle_deg,
                    self._right_positive,
                )
                handle.write(f"{display_angle_deg:.3f},{range_str},{int(count)}\n")

        payload = {
            "topic": self._topic,
            "scans_averaged": int(self._snapshot_scans_collected),
            "range_min_m": float(msg.range_min),
            "range_max_m": float(msg.range_max),
            "closest_point": {
                "distance_m": closest_distance_m,
                "angle_deg": closest_angle_deg,
            },
            "front_m": self._probe_distance_display_angle(
                msg, averaged_ranges, valid_mask, 0.0
            ),
            "front_left_45_m": self._probe_distance_display_angle(
                msg, averaged_ranges, valid_mask, -45.0
            ),
            "front_right_45_m": self._probe_distance_display_angle(
                msg, averaged_ranges, valid_mask, 45.0
            ),
            "left_90_m": self._probe_distance_display_angle(
                msg, averaged_ranges, valid_mask, -90.0
            ),
            "right_90_m": self._probe_distance_display_angle(
                msg, averaged_ranges, valid_mask, 90.0
            ),
            "probes_m": probe_summary,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        plot_saved = self._save_snapshot_plot(
            msg=msg,
            png_path=png_path,
            averaged_ranges=averaged_ranges,
            valid_mask=valid_mask,
            probe_summary=probe_summary,
            closest_distance_m=closest_distance_m,
            closest_angle_deg=closest_angle_deg,
        )

        self._snapshot_saved = True
        snapshot_targets = [str(json_path), str(csv_path)]
        if plot_saved:
            snapshot_targets.append(str(png_path))
        self.get_logger().info("Snapshot guardado: %s" % " | ".join(snapshot_targets))
        if self._snapshot_exit:
            self.get_logger().info("Snapshot completado; cerrando subscriber.")
            rclpy.shutdown()

    def _update_snapshot(self, msg: LaserScan, ranges: np.ndarray, valid_mask: np.ndarray) -> None:
        if self._snapshot_scans_target <= 0 or self._snapshot_saved:
            return
        if self._snapshot_sum is None or self._snapshot_counts is None:
            self._snapshot_sum = np.zeros_like(ranges, dtype=np.float64)
            self._snapshot_counts = np.zeros_like(ranges, dtype=np.int32)

        self._snapshot_sum[valid_mask] += ranges[valid_mask]
        self._snapshot_counts[valid_mask] += 1
        self._snapshot_scans_collected += 1
        if self._snapshot_scans_collected >= self._snapshot_scans_target:
            self._save_snapshot(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        if self._mapper is None:
            return
        pose = msg.pose.pose
        yaw = _yaw_from_quaternion(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        self._mapper.set_odom_pose(float(pose.position.x), float(pose.position.y), yaw)

    def _scan_cb(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            now = time.time()
            if now - self._last_print >= self._print_every_s:
                self._last_print = now
                self.get_logger().warning("Scan vacio")
            return

        valid_mask = np.isfinite(ranges) & (ranges > 0.0)
        if msg.range_min > 0.0 and np.isfinite(msg.range_min):
            valid_mask = valid_mask & (ranges >= float(msg.range_min))
        if msg.range_max > 0.0 and np.isfinite(msg.range_max):
            valid_mask = valid_mask & (ranges <= float(msg.range_max))
        valid = ranges[valid_mask]
        self._update_snapshot(msg, ranges, valid_mask)

        if self._plotter is not None and valid.size > 0:
            angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            self._plotter.update(angles[valid_mask], valid)

        map_state: Optional[MappingState] = None
        if self._mapper is not None:
            map_state = self._mapper.update_from_scan(msg, ranges)

        now = time.time()
        if now - self._last_print < self._print_every_s:
            return
        self._last_print = now

        if valid.size == 0:
            self.get_logger().warning("Sin puntos validos en este scan")
            return

        front = self._probe_distance_display_angle(msg, ranges, valid_mask, 0.0)
        left = self._probe_distance_display_angle(msg, ranges, valid_mask, -90.0)
        right = self._probe_distance_display_angle(msg, ranges, valid_mask, 90.0)

        summary = (
            "Puntos=%d | min=%.3fm avg=%.3fm max=%.3fm | front=%.3fm left=%.3fm right=%.3fm"
            % (
                valid.size,
                float(np.min(valid)),
                float(np.mean(valid)),
                float(np.max(valid)),
                front,
                left,
                right,
            )
        )
        summary += self._format_probe_summary(msg, ranges, valid_mask)

        if map_state is not None:
            summary += (
                " | pose=(%.2f, %.2f, %.1fdeg) src=%s | match(rot=%.2f trans=%.2f)"
                % (
                    map_state.x_m,
                    map_state.y_m,
                    map_state.yaw_deg,
                    map_state.pose_source,
                    map_state.rot_score,
                    map_state.trans_score,
                )
            )
        self.get_logger().info(summary)

        if self._full:
            rounded = np.round(ranges, 3)
            self.get_logger().info(f"ranges[{ranges.size}]: {rounded.tolist()}")

    def shutdown(self) -> None:
        if self._plotter is not None:
            self._plotter.close()
        if self._mapper is not None:
            self._mapper.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 LaserScan subscriber for PC/WSL")
    parser.add_argument("--topic", default="/lidar/scan", help="Topico LaserScan de la Raspberry")
    parser.add_argument(
        "--print-every-s",
        type=float,
        default=0.5,
        help="Periodo minimo entre impresiones en consola",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Imprime el arreglo completo de 360 mediciones",
    )
    parser.add_argument(
        "--right-positive",
        action="store_true",
        help="Usa convencion visual con derecha positiva e izquierda negativa",
    )
    parser.add_argument(
        "--probe-angles-deg",
        default="0,45,90,-45,-90,180",
        help="Lista separada por comas de angulos a medir en consola",
    )
    parser.add_argument(
        "--probe-window-deg",
        type=float,
        default=2.0,
        help="Semiancho angular usado para promediar cada sonda",
    )
    parser.add_argument(
        "--snapshot-scans",
        type=int,
        default=0,
        help="Si es >0, promedia este numero de scans y guarda una captura estatica",
    )
    parser.add_argument(
        "--snapshot-output",
        default="",
        help="Ruta base de salida para snapshot (.json y .csv). Si se omite, usa el directorio actual",
    )
    parser.add_argument(
        "--snapshot-exit",
        action="store_true",
        help="Salir automaticamente al terminar el snapshot",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a live 2D XY point-cloud window",
    )
    parser.add_argument(
        "--plot-max-range",
        type=float,
        default=3.0,
        help="Minimum axis range (meters) for the point-cloud plot",
    )
    parser.add_argument(
        "--plot-every-s",
        type=float,
        default=0.1,
        help="Minimum plotting period (seconds)",
    )

    parser.add_argument(
        "--map",
        action="store_true",
        help="Enable occupancy-grid mapping on the PC",
    )
    parser.add_argument(
        "--map-plot",
        action="store_true",
        help="Show a live occupancy-grid map window",
    )
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=0.05,
        help="Occupancy map resolution in meters/cell",
    )
    parser.add_argument(
        "--map-size-m",
        type=float,
        default=30.0,
        help="Square map size in meters (size x size)",
    )
    parser.add_argument(
        "--map-min-range",
        type=float,
        default=0.08,
        help="Min LiDAR range used for mapping (meters)",
    )
    parser.add_argument(
        "--map-max-range",
        type=float,
        default=8.0,
        help="Max LiDAR range used for mapping (meters)",
    )
    parser.add_argument(
        "--scanmatch-max-rot-deg",
        type=float,
        default=12.0,
        help="Max absolute rotation search per scan step (degrees)",
    )
    parser.add_argument(
        "--scanmatch-max-trans-m",
        type=float,
        default=0.40,
        help="Max absolute translation search per scan step (meters)",
    )
    parser.add_argument(
        "--scanmatch-trans-step-m",
        type=float,
        default=0.02,
        help="Translation search step (meters)",
    )
    parser.add_argument(
        "--scanmatch-max-points",
        type=int,
        default=180,
        help="Max points used for scan-to-scan translation matching",
    )
    parser.add_argument(
        "--map-update-max-points",
        type=int,
        default=180,
        help="Max points integrated into occupancy map per scan",
    )
    parser.add_argument(
        "--map-plot-every-s",
        type=float,
        default=0.25,
        help="Minimum occupancy-plot update period (seconds)",
    )
    parser.add_argument(
        "--map-save-path",
        default="",
        help="Optional output path for map probability grid as .npy",
    )
    parser.add_argument(
        "--map-pose-source",
        choices=["odom", "scanmatch"],
        default="odom",
        help="Pose source for mapping: odom (Ackermann kinematics) or scanmatch",
    )
    parser.add_argument(
        "--odom-topic",
        default="/odom",
        help="Odometry topic used when --map-pose-source=odom",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)

    node = LidarSubscriberNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
