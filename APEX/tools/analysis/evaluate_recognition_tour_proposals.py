#!/usr/bin/env python3
"""Generate offline approval images and metrics for recognition_tour runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "apex_telemetry"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from apex_telemetry.perception.curve_window_detection import (  # noqa: E402
    CurveWindowDetectionConfig,
    detect_curve_window_points,
)


@dataclass
class PathHistoryItem:
    stamp_s: float
    path_xy: np.ndarray
    path_yaw: np.ndarray
    path_s: np.ndarray
    path_curvature_m_inv: np.ndarray


@dataclass
class ScanFrame:
    stamp_s: float
    scan_index: int
    local_xy: np.ndarray
    world_xy: np.ndarray


@dataclass
class TrajectorySeries:
    times_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    yaw_rad: np.ndarray
    speed_mps: np.ndarray
    tracker_state: list[str]
    planner_state: list[str]
    path_age_s: np.ndarray
    path_deviation_m: np.ndarray

    def latest_before(self, t_s: float) -> tuple[np.ndarray, float, float] | None:
        if self.times_s.size == 0 or t_s < float(self.times_s[0]):
            return None
        index = int(np.searchsorted(self.times_s, t_s, side="right")) - 1
        index = max(0, min(index, self.times_s.size - 1))
        pose_xy_yaw = np.asarray(
            [self.x_m[index], self.y_m[index], self.yaw_rad[index]],
            dtype=np.float64,
        )
        return pose_xy_yaw, float(self.speed_mps[index]), float(self.times_s[index])

    def interpolate(self, t_s: float) -> tuple[np.ndarray, float] | None:
        if self.times_s.size == 0:
            return None
        if t_s <= float(self.times_s[0]):
            pose_xy_yaw = np.asarray(
                [self.x_m[0], self.y_m[0], self.yaw_rad[0]],
                dtype=np.float64,
            )
            return pose_xy_yaw, float(self.speed_mps[0])
        if t_s >= float(self.times_s[-1]):
            pose_xy_yaw = np.asarray(
                [self.x_m[-1], self.y_m[-1], self.yaw_rad[-1]],
                dtype=np.float64,
            )
            return pose_xy_yaw, float(self.speed_mps[-1])
        right = int(np.searchsorted(self.times_s, t_s, side="right"))
        left = max(0, right - 1)
        t0 = float(self.times_s[left])
        t1 = float(self.times_s[right])
        if (t1 - t0) <= 1.0e-9:
            alpha = 0.0
        else:
            alpha = (float(t_s) - t0) / (t1 - t0)
        x_m = ((1.0 - alpha) * self.x_m[left]) + (alpha * self.x_m[right])
        y_m = ((1.0 - alpha) * self.y_m[left]) + (alpha * self.y_m[right])
        yaw_rad = ((1.0 - alpha) * self.yaw_rad[left]) + (alpha * self.yaw_rad[right])
        speed_mps = ((1.0 - alpha) * self.speed_mps[left]) + (alpha * self.speed_mps[right])
        pose_xy_yaw = np.asarray([x_m, y_m, yaw_rad], dtype=np.float64)
        return pose_xy_yaw, float(speed_mps)


@dataclass
class Centerline:
    x_m: np.ndarray
    y_m: np.ndarray
    width_m: float
    valid_bin_count: int


def _rotation(theta_rad: float) -> np.ndarray:
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )


def _transform_local_to_world(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) @ _rotation(yaw_rad).T) + origin_xy.reshape(1, 2)


def _transform_world_to_local(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) - origin_xy.reshape(1, 2)) @ _rotation(yaw_rad)


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _compute_path_s(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])


def _polyline_yaw(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return np.zeros((path_xy.shape[0],), dtype=np.float64)
    yaw = np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    yaw[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw[-1] = yaw[-2]
    return yaw


def _estimate_path_curvature(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    point_count = path_xy.shape[0]
    curvature = np.zeros((point_count,), dtype=np.float64)
    if point_count < 3:
        return curvature
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    for index in range(1, point_count - 1):
        ds = max(1.0e-6, 0.5 * (seg_lengths[index - 1] + seg_lengths[index]))
        dtheta = math.atan2(
            math.sin(float(headings[index] - headings[index - 1])),
            math.cos(float(headings[index] - headings[index - 1])),
        )
        curvature[index] = dtheta / ds
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def _polyline_length_m(path_xy: np.ndarray) -> float:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return 0.0
    diffs = np.diff(path_xy, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def _resample_polyline_xy(path_xy: np.ndarray, step_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length_m = float(cumulative_s[-1])
    if total_length_m <= 1.0e-9:
        return path_xy[[0, -1]].copy()
    step_m = max(1.0e-3, float(step_m))
    sample_s = np.arange(0.0, total_length_m, step_m, dtype=np.float64)
    if sample_s.size == 0 or sample_s[0] > 1.0e-9:
        sample_s = np.concatenate([[0.0], sample_s])
    if (total_length_m - sample_s[-1]) > 1.0e-9:
        sample_s = np.concatenate([sample_s, [total_length_m]])
    xs = np.interp(sample_s, cumulative_s, path_xy[:, 0])
    ys = np.interp(sample_s, cumulative_s, path_xy[:, 1])
    return np.column_stack([xs, ys])


def _resample_polyline_xy_to_count(path_xy: np.ndarray, point_count: int) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    point_count = max(2, int(point_count))
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length_m = float(cumulative_s[-1])
    if total_length_m <= 1.0e-9:
        return np.repeat(path_xy[:1], point_count, axis=0)
    sample_s = np.linspace(0.0, total_length_m, point_count, dtype=np.float64)
    xs = np.interp(sample_s, cumulative_s, path_xy[:, 0])
    ys = np.interp(sample_s, cumulative_s, path_xy[:, 1])
    return np.column_stack([xs, ys])


def _smooth_path_to_curvature_limit(
    *,
    path_xy: np.ndarray,
    max_curvature_m_inv: float,
    resample_step_m: float,
    smoothing_alpha: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    if path_xy.shape[0] < 3 or max_curvature_m_inv <= 1.0e-9:
        curvature = np.abs(_estimate_path_curvature(path_xy))
        return path_xy, float(np.max(curvature)) if curvature.size else 0.0

    smoothed_xy = _resample_polyline_xy(path_xy, resample_step_m)
    smoothing_alpha = max(0.0, min(1.0, float(smoothing_alpha)))
    max_iterations = max(0, int(max_iterations))

    for _ in range(max_iterations):
        curvature = np.abs(_estimate_path_curvature(smoothed_xy))
        max_curvature = float(np.max(curvature)) if curvature.size else 0.0
        if max_curvature <= max_curvature_m_inv:
            return smoothed_xy, max_curvature
        next_xy = smoothed_xy.copy()
        next_xy[1:-1] = (
            ((1.0 - smoothing_alpha) * smoothed_xy[1:-1])
            + (0.5 * smoothing_alpha * (smoothed_xy[:-2] + smoothed_xy[2:]))
        )
        next_xy[0] = smoothed_xy[0]
        next_xy[-1] = smoothed_xy[-1]
        smoothed_xy = _resample_polyline_xy(next_xy, resample_step_m)

    curvature = np.abs(_estimate_path_curvature(smoothed_xy))
    return smoothed_xy, float(np.max(curvature)) if curvature.size else 0.0


def _cubic_bezier_xy(
    *,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    p2_xy: np.ndarray,
    p3_xy: np.ndarray,
    point_count: int,
) -> np.ndarray:
    if point_count <= 1:
        return p0_xy.reshape(1, 2)
    ts = np.linspace(0.0, 1.0, point_count, endpoint=False, dtype=np.float64)
    one_minus_t = 1.0 - ts
    return (
        ((one_minus_t ** 3).reshape(-1, 1) * p0_xy.reshape(1, 2))
        + (3.0 * (one_minus_t ** 2) * ts).reshape(-1, 1) * p1_xy.reshape(1, 2)
        + (3.0 * one_minus_t * (ts ** 2)).reshape(-1, 1) * p2_xy.reshape(1, 2)
        + ((ts ** 3).reshape(-1, 1) * p3_xy.reshape(1, 2))
    )


def _truncate_polyline_length(path_xy: np.ndarray, max_length_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if float(cumulative_s[-1]) <= max_length_m:
        return path_xy.copy()
    cutoff_index = int(np.searchsorted(cumulative_s, max_length_m, side="right"))
    cutoff_index = max(1, min(cutoff_index, path_xy.shape[0] - 1))
    truncated_xy = path_xy[:cutoff_index].copy()
    last_s = float(cumulative_s[cutoff_index - 1])
    if last_s < max_length_m and cutoff_index < path_xy.shape[0]:
        prev_xy = path_xy[cutoff_index - 1]
        next_xy = path_xy[cutoff_index]
        seg_length = max(1.0e-9, float(seg_lengths[cutoff_index - 1]))
        ratio = (max_length_m - last_s) / seg_length
        truncated_xy = np.vstack([truncated_xy, prev_xy + (ratio * (next_xy - prev_xy))])
    return truncated_xy


def _extend_path_forward(path_xy: np.ndarray, target_forward_x_m: float, step_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    target_forward_x_m = max(float(target_forward_x_m), float(path_xy[-1, 0]))
    if float(path_xy[-1, 0]) >= (target_forward_x_m - 1.0e-6):
        return path_xy.copy()
    tail_window_xy = path_xy[-min(6, path_xy.shape[0]) :]
    delta_xy = tail_window_xy[-1] - tail_window_xy[0]
    dx_m = float(delta_xy[0])
    slope = 0.0 if abs(dx_m) <= 1.0e-6 else float(np.clip(delta_xy[1] / dx_m, -1.1, 1.1))
    xs = np.arange(
        float(path_xy[-1, 0]) + max(1.0e-3, float(step_m)),
        target_forward_x_m + (0.5 * max(1.0e-3, float(step_m))),
        max(1.0e-3, float(step_m)),
        dtype=np.float64,
    )
    if xs.size == 0:
        return path_xy.copy()
    ys = float(path_xy[-1, 1]) + slope * (xs - float(path_xy[-1, 0]))
    return np.vstack([path_xy, np.column_stack([xs, ys])])


def _enforce_monotonic_forward_x(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64).copy()
    if path_xy.shape[0] == 0:
        return path_xy
    path_xy[:, 0] = np.maximum.accumulate(path_xy[:, 0])
    return path_xy


def _apply_straight_entry_hold(path_xy: np.ndarray, hold_length_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64).copy()
    if path_xy.shape[0] <= 1 or hold_length_m <= 1.0e-6:
        return path_xy
    path_s = _compute_path_s(path_xy)
    hold_length_m = max(1.0e-3, float(hold_length_m))
    mask = path_s < hold_length_m
    if not np.any(mask):
        return path_xy
    blend = np.clip(path_s[mask] / hold_length_m, 0.0, 1.0)
    path_xy[mask, 1] *= blend
    return path_xy


def _deduplicate_polyline_xy(path_xy: np.ndarray, min_segment_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    kept_points = [path_xy[0]]
    last_xy = path_xy[0]
    for point_xy in path_xy[1:]:
        if float(np.linalg.norm(point_xy - last_xy)) < min_segment_m:
            continue
        kept_points.append(point_xy)
        last_xy = point_xy
    if len(kept_points) == 1:
        kept_points.append(path_xy[-1])
    elif not np.allclose(kept_points[-1], path_xy[-1]):
        kept_points.append(path_xy[-1])
    return np.asarray(kept_points, dtype=np.float64)


def _blend_paths_by_arclength(
    new_path_xy: np.ndarray,
    previous_path_xy: np.ndarray,
    *,
    new_path_weight: float,
) -> np.ndarray:
    new_path_xy = np.asarray(new_path_xy, dtype=np.float64)
    previous_path_xy = np.asarray(previous_path_xy, dtype=np.float64)
    if new_path_xy.shape[0] < 2 or previous_path_xy.shape[0] < 2:
        return new_path_xy.copy()
    sample_count = max(new_path_xy.shape[0], previous_path_xy.shape[0], 36)
    new_eval_xy = _resample_polyline_xy_to_count(new_path_xy, sample_count)
    previous_eval_xy = _resample_polyline_xy_to_count(previous_path_xy, sample_count)
    alpha = max(0.0, min(1.0, float(new_path_weight)))
    blended_xy = (alpha * new_eval_xy) + ((1.0 - alpha) * previous_eval_xy)
    blended_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
    return blended_xy


def _path_terminal_heading(path_xy: np.ndarray) -> float:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] < 2:
        return 0.0
    tail_xy = path_xy[-min(4, path_xy.shape[0]) :]
    delta_xy = tail_xy[-1] - tail_xy[0]
    if float(np.linalg.norm(delta_xy)) <= 1.0e-9:
        delta_xy = path_xy[-1] - path_xy[-2]
    return math.atan2(float(delta_xy[1]), float(delta_xy[0]))


def _graft_previous_tail(
    path_xy: np.ndarray,
    previous_path_xy: np.ndarray,
    *,
    step_m: float,
    min_start_forward_delta_m: float,
) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    previous_path_xy = np.asarray(previous_path_xy, dtype=np.float64)
    if path_xy.shape[0] < 2 or previous_path_xy.shape[0] < 4:
        return path_xy.copy()
    tail_start_index = int(
        np.searchsorted(
            previous_path_xy[:, 0],
            float(path_xy[-1, 0]) + max(0.02, float(min_start_forward_delta_m)),
            side="left",
        )
    )
    if tail_start_index >= (previous_path_xy.shape[0] - 3):
        return path_xy.copy()
    tail_xy = previous_path_xy[tail_start_index:].copy()
    if tail_xy.shape[0] < 4:
        return path_xy.copy()
    gap_m = float(np.linalg.norm(tail_xy[0] - path_xy[-1]))
    if gap_m < max(0.03, 0.75 * step_m):
        return np.vstack([path_xy[:-1], tail_xy])
    entry_heading_rad = _path_terminal_heading(path_xy)
    tail_heading_rad = _path_terminal_heading(tail_xy[: min(5, tail_xy.shape[0])])
    entry_tangent_xy = np.asarray([math.cos(entry_heading_rad), math.sin(entry_heading_rad)], dtype=np.float64)
    tail_tangent_xy = np.asarray([math.cos(tail_heading_rad), math.sin(tail_heading_rad)], dtype=np.float64)
    control_length_m = min(0.30, max(0.08, 0.45 * gap_m))
    connector_xy = _cubic_bezier_xy(
        p0_xy=path_xy[-1],
        p1_xy=path_xy[-1] + (control_length_m * entry_tangent_xy),
        p2_xy=tail_xy[0] - (control_length_m * tail_tangent_xy),
        p3_xy=tail_xy[0],
        point_count=max(4, int(math.ceil(gap_m / max(1.0e-3, step_m))) + 2),
    )
    return np.vstack([path_xy[:-1], connector_xy, tail_xy[1:]])


def _fill_small_gaps(values: np.ndarray, max_gap_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    valid_mask = np.isfinite(values)
    index = 0
    while index < values.shape[0]:
        if valid_mask[index]:
            index += 1
            continue
        start = index
        while index < values.shape[0] and not valid_mask[index]:
            index += 1
        end = index
        gap_size = end - start
        if (
            gap_size <= max_gap_bins
            and start > 0
            and end < values.shape[0]
            and valid_mask[start - 1]
            and valid_mask[end]
        ):
            left = float(values[start - 1])
            right = float(values[end])
            for local_idx in range(gap_size):
                alpha = float(local_idx + 1) / float(gap_size + 1)
                values[start + local_idx] = ((1.0 - alpha) * left) + (alpha * right)
    return values


def _extract_centerline(
    *,
    points_xy: np.ndarray,
    horizon_m: float,
    bin_m: float,
    lower_quantile: float,
    upper_quantile: float,
    min_bin_points: int,
    min_width_m: float,
    max_width_m: float,
    max_gap_bins: int,
) -> Centerline | None:
    if points_xy.shape[0] < max(40, 3 * min_bin_points):
        return None
    bin_edges = np.arange(0.0, max(bin_m, float(horizon_m)) + bin_m, bin_m, dtype=np.float64)
    if bin_edges.size < 4:
        return None
    lower_y = np.full((bin_edges.size - 1,), np.nan, dtype=np.float64)
    upper_y = np.full((bin_edges.size - 1,), np.nan, dtype=np.float64)
    width = np.full((bin_edges.size - 1,), np.nan, dtype=np.float64)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for index, (x0_m, x1_m) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
        mask = (
            np.isfinite(points_xy[:, 0])
            & np.isfinite(points_xy[:, 1])
            & (points_xy[:, 0] >= x0_m)
            & (points_xy[:, 0] < x1_m)
        )
        if int(np.count_nonzero(mask)) < min_bin_points:
            continue
        ys = points_xy[mask, 1]
        q_low = float(np.quantile(ys, lower_quantile))
        q_high = float(np.quantile(ys, upper_quantile))
        width_m = q_high - q_low
        if width_m < min_width_m or width_m > max_width_m:
            continue
        lower_y[index] = q_low
        upper_y[index] = q_high
        width[index] = width_m
    lower_y = _fill_small_gaps(lower_y, max_gap_bins)
    upper_y = _fill_small_gaps(upper_y, max_gap_bins)
    width = upper_y - lower_y
    valid_mask = np.isfinite(lower_y) & np.isfinite(upper_y) & np.isfinite(width)
    valid_mask &= (width >= min_width_m) & (width <= max_width_m)
    if int(np.count_nonzero(valid_mask)) < 4:
        return None
    x_valid = x_centers[valid_mask]
    center_valid = 0.5 * (lower_y[valid_mask] + upper_y[valid_mask])
    if x_valid.shape[0] < 4:
        return None
    if float(x_valid[-1] - x_valid[0]) < max(0.5, 3.0 * bin_m):
        return None
    return Centerline(
        x_m=x_valid,
        y_m=center_valid,
        width_m=float(np.median(width[valid_mask])),
        valid_bin_count=int(np.count_nonzero(valid_mask)),
    )


def _count_segments(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    return int(np.count_nonzero(mask & ~np.concatenate([[False], mask[:-1]])))


def _segment_start_points(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    start_mask = mask & ~np.concatenate([[False], mask[:-1]])
    return points_xy[start_mask]


def _unwrap_angles(yaw_rad: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(yaw_rad, dtype=np.float64))


def _load_params(config_path: Path) -> tuple[dict, dict]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return (
        payload["recognition_tour_planner_node"]["ros__parameters"],
        payload["recognition_tour_tracker_node"]["ros__parameters"],
    )


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_route_json(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    return points[:, :3]


def _load_local_path_history(path: Path) -> list[PathHistoryItem]:
    if not path.exists():
        return []
    items: list[PathHistoryItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        path_xy_yaw = np.asarray(payload.get("path_xy_yaw", []), dtype=np.float64)
        if path_xy_yaw.ndim != 2 or path_xy_yaw.shape[0] < 2 or path_xy_yaw.shape[1] < 3:
            continue
        path_xy = path_xy_yaw[:, :2]
        path_yaw = path_xy_yaw[:, 2]
        stamp_s = float(payload.get("stamp_sec", 0)) + (1.0e-9 * float(payload.get("stamp_nanosec", 0)))
        items.append(
            PathHistoryItem(
                stamp_s=stamp_s,
                path_xy=path_xy,
                path_yaw=path_yaw,
                path_s=_compute_path_s(path_xy),
                path_curvature_m_inv=_estimate_path_curvature(path_xy),
            )
        )
    items.sort(key=lambda item: item.stamp_s)
    return items


def _load_trajectory(path: Path) -> TrajectorySeries:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    times_s = []
    x_m = []
    y_m = []
    yaw_rad = []
    tracker_state = []
    planner_state = []
    path_age_s = []
    path_deviation_m = []
    for row in rows:
        try:
            stamp_s = float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"]))
            x_val = float(row["x_m"])
            y_val = float(row["y_m"])
            yaw_val = float(row["yaw_rad"])
        except (KeyError, ValueError):
            continue
        times_s.append(stamp_s)
        x_m.append(x_val)
        y_m.append(y_val)
        yaw_rad.append(yaw_val)
        tracker_state.append(str(row.get("tracker_state", "")))
        planner_state.append(str(row.get("planner_state", "")))
        path_age_s.append(float(row.get("path_age_s") or 0.0))
        path_deviation_m.append(float(row.get("path_deviation_m") or 0.0))
    times = np.asarray(times_s, dtype=np.float64)
    xs = np.asarray(x_m, dtype=np.float64)
    ys = np.asarray(y_m, dtype=np.float64)
    yaws = _unwrap_angles(np.asarray(yaw_rad, dtype=np.float64))
    if times.size >= 2:
        dx_dt = np.gradient(xs, times, edge_order=1)
        dy_dt = np.gradient(ys, times, edge_order=1)
        speeds = np.hypot(dx_dt, dy_dt)
    else:
        speeds = np.zeros_like(times)
    return TrajectorySeries(
        times_s=times,
        x_m=xs,
        y_m=ys,
        yaw_rad=yaws,
        speed_mps=speeds,
        tracker_state=tracker_state,
        planner_state=planner_state,
        path_age_s=np.asarray(path_age_s, dtype=np.float64),
        path_deviation_m=np.asarray(path_deviation_m, dtype=np.float64),
    )


def _load_lidar_points(path: Path, *, max_points: int | None = None) -> tuple[np.ndarray, list[ScanFrame]]:
    groups: dict[tuple[int, int, int], dict[str, object]] = {}
    lidar_world_points: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                stamp_sec = int(row["stamp_sec"])
                stamp_nanosec = int(row["stamp_nanosec"])
                scan_index = int(row["scan_index"])
                x_forward_m = float(row["x_forward_m"])
                y_left_m = float(row["y_left_m"])
                x_world_m = float(row["x_world_m"])
                y_world_m = float(row["y_world_m"])
            except (KeyError, ValueError):
                continue
            key = (stamp_sec, stamp_nanosec, scan_index)
            entry = groups.setdefault(
                key,
                {
                    "stamp_s": float(stamp_sec) + (1.0e-9 * float(stamp_nanosec)),
                    "scan_index": scan_index,
                    "local": [],
                    "world": [],
                },
            )
            entry["local"].append([x_forward_m, y_left_m])
            entry["world"].append([x_world_m, y_world_m])
            lidar_world_points.append([x_world_m, y_world_m])
    scans = [
        ScanFrame(
            stamp_s=float(entry["stamp_s"]),
            scan_index=int(entry["scan_index"]),
            local_xy=np.asarray(entry["local"], dtype=np.float64),
            world_xy=np.asarray(entry["world"], dtype=np.float64),
        )
        for entry in groups.values()
    ]
    scans.sort(key=lambda scan: (scan.stamp_s, scan.scan_index))
    world_points = np.asarray(lidar_world_points, dtype=np.float64)
    if max_points is not None and world_points.shape[0] > max_points:
        stride = max(1, int(math.ceil(world_points.shape[0] / max_points)))
        world_points = world_points[::stride]
    return world_points, scans


class OfflineRecognitionPlanner:
    def __init__(self, planner_params: dict) -> None:
        self._planning_horizon_m = float(planner_params["planning_horizon_m"])
        self._rear_axle_offset = np.asarray(
            [
                float(planner_params["rear_axle_offset_x_m"]),
                float(planner_params["rear_axle_offset_y_m"]),
            ],
            dtype=np.float64,
        )
        self._lidar_offset = np.asarray(
            [
                float(planner_params["lidar_offset_x_m"]),
                float(planner_params["lidar_offset_y_m"]),
            ],
            dtype=np.float64,
        )
        self._rear_to_lidar_local = self._lidar_offset - self._rear_axle_offset
        self._rolling_window_s = float(planner_params["rolling_window_s"])
        self._path_resample_step_m = float(planner_params["path_resample_step_m"])
        self._path_smoothing_alpha = float(planner_params["path_smoothing_alpha"])
        self._path_smoothing_max_iterations = int(planner_params["path_smoothing_max_iterations"])
        self._path_min_forward_progress_m = float(planner_params["path_min_forward_progress_m"])
        self._straight_entry_hold_length_m = float(planner_params["straight_entry_hold_length_m"])
        self._replan_path_blend_alpha = float(planner_params["replan_path_blend_alpha"])
        self._previous_path_tail_extension_m = float(planner_params["previous_path_tail_extension_m"])
        self._previous_path_tail_graft_min_span_m = float(
            planner_params["previous_path_tail_graft_min_span_m"]
        )
        self._origin_bridge_point_count = int(planner_params["origin_bridge_point_count"])
        self._corridor_bin_m = float(planner_params["corridor_bin_m"])
        corridor_quantile = float(planner_params["corridor_quantile"])
        self._corridor_lower_quantile = corridor_quantile
        self._corridor_upper_quantile = 1.0 - corridor_quantile
        self._corridor_min_bin_points = int(planner_params["corridor_min_bin_points"])
        self._corridor_min_width_m = float(planner_params["corridor_min_width_m"])
        self._corridor_max_width_m = float(planner_params["corridor_max_width_m"])
        self._corridor_gap_fill_bins = int(planner_params["corridor_gap_fill_bins"])
        self._curve_window_config = CurveWindowDetectionConfig(
            second_corridor_target_depth_m=float(planner_params["second_corridor_target_depth_m"]),
            second_corridor_target_depth_min_m=float(planner_params["second_corridor_target_depth_min_m"]),
            second_corridor_target_depth_max_m=float(planner_params["second_corridor_target_depth_max_m"]),
            inner_vertex_clearance_m=float(planner_params["inner_vertex_clearance_m"]),
            curve_apex_width_fraction=float(planner_params["curve_apex_width_fraction"]),
        )
        wheelbase_m = float(planner_params["planning_wheelbase_m"])
        max_steering_deg = float(planner_params["planning_max_steering_deg"])
        curvature_scale = float(planner_params["path_curvature_limit_scale"])
        self._max_path_curvature_m_inv = curvature_scale * math.tan(math.radians(max_steering_deg)) / max(
            1.0e-3, wheelbase_m
        )
        self._last_local_path_world_xy: np.ndarray | None = None
        self._last_local_path_source = "none"
        self._low_confidence_hold_path_s = float(planner_params["low_confidence_hold_path_s"])

    def _previous_local_path_xy(self, rear_xy: np.ndarray, yaw_rad: float) -> np.ndarray | None:
        if self._last_local_path_world_xy is None or self._last_local_path_world_xy.shape[0] < 2:
            return None
        previous_local_xy = _transform_world_to_local(self._last_local_path_world_xy, rear_xy, yaw_rad)
        mask = (
            np.isfinite(previous_local_xy[:, 0])
            & np.isfinite(previous_local_xy[:, 1])
            & (previous_local_xy[:, 0] >= -0.20)
            & (previous_local_xy[:, 0] <= (self._planning_horizon_m + 0.50))
        )
        previous_local_xy = previous_local_xy[mask]
        if previous_local_xy.shape[0] < 2:
            return None
        return _enforce_monotonic_forward_x(previous_local_xy)

    def _turn_severity(self, path_xy: np.ndarray) -> float:
        if path_xy.shape[0] < 3:
            return 0.0
        path_yaw = _polyline_yaw(path_xy)
        tail_count = max(1, min(6, path_yaw.shape[0]))
        tail_heading_rad = float(np.median(path_yaw[-tail_count:]))
        heading_score = min(1.0, abs(_normalize_angle(tail_heading_rad)) / math.radians(55.0))
        lateral_excursion_m = max(
            abs(float(path_xy[-1, 1])),
            float(np.percentile(np.abs(path_xy[:, 1]), 92)),
        )
        lateral_score = min(1.0, lateral_excursion_m / 0.50)
        curvature = np.abs(_estimate_path_curvature(path_xy))
        max_curvature = float(np.max(curvature)) if curvature.size else 0.0
        curvature_score = min(1.0, max_curvature / max(1.0e-3, 0.85 * self._max_path_curvature_m_inv))
        return max(heading_score, lateral_score, curvature_score)

    def _stabilize_local_path(self, local_path_xy: np.ndarray, *, rear_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
        local_path_xy = np.asarray(local_path_xy, dtype=np.float64).copy()
        if local_path_xy.shape[0] == 0:
            return local_path_xy
        turn_severity = self._turn_severity(local_path_xy)
        base_forward_progress_m = min(self._planning_horizon_m, self._path_min_forward_progress_m)
        effective_forward_progress_m = (
            ((1.0 - turn_severity) * base_forward_progress_m)
            + (turn_severity * max(0.95, 0.58 * self._planning_horizon_m))
        )
        effective_hold_length_m = (
            ((1.0 - turn_severity) * self._straight_entry_hold_length_m)
            + (turn_severity * min(self._straight_entry_hold_length_m, 0.14))
        )
        effective_blend_alpha = min(0.90, self._replan_path_blend_alpha + (0.25 * turn_severity))
        local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
        local_path_xy = _extend_path_forward(
            local_path_xy,
            target_forward_x_m=effective_forward_progress_m,
            step_m=self._path_resample_step_m,
        )
        local_path_xy = _apply_straight_entry_hold(local_path_xy, effective_hold_length_m)
        previous_local_xy = self._previous_local_path_xy(rear_xy, yaw_rad)
        if previous_local_xy is not None:
            current_forward_span_m = float(local_path_xy[-1, 0] - local_path_xy[0, 0])
            if current_forward_span_m < self._previous_path_tail_graft_min_span_m:
                local_path_xy = _graft_previous_tail(
                    local_path_xy,
                    previous_local_xy,
                    step_m=self._path_resample_step_m,
                    min_start_forward_delta_m=self._previous_path_tail_extension_m,
                )
            previous_local_xy = _extend_path_forward(
                previous_local_xy,
                target_forward_x_m=effective_forward_progress_m,
                step_m=self._path_resample_step_m,
            )
            local_path_xy = _blend_paths_by_arclength(
                local_path_xy,
                previous_local_xy,
                new_path_weight=effective_blend_alpha,
            )
        local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
        local_path_xy = _truncate_polyline_length(local_path_xy, self._planning_horizon_m)
        local_path_xy, path_max_curvature_m_inv = _smooth_path_to_curvature_limit(
            path_xy=local_path_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=max(self._path_smoothing_alpha, 0.24),
            max_iterations=max(self._path_smoothing_max_iterations, 260),
        )
        if (
            path_max_curvature_m_inv > (1.35 * self._max_path_curvature_m_inv)
            and previous_local_xy is not None
        ):
            local_path_xy = _blend_paths_by_arclength(
                local_path_xy,
                previous_local_xy,
                new_path_weight=0.45,
            )
            local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
            local_path_xy, _ = _smooth_path_to_curvature_limit(
                path_xy=local_path_xy,
                max_curvature_m_inv=self._max_path_curvature_m_inv,
                resample_step_m=self._path_resample_step_m,
                smoothing_alpha=max(self._path_smoothing_alpha, 0.28),
                max_iterations=max(self._path_smoothing_max_iterations, 320),
            )
        local_path_xy = _extend_path_forward(
            local_path_xy,
            target_forward_x_m=effective_forward_progress_m,
            step_m=self._path_resample_step_m,
        )
        local_path_xy = _truncate_polyline_length(local_path_xy, self._planning_horizon_m)
        local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
        local_path_xy = _deduplicate_polyline_xy(local_path_xy, min_segment_m=0.35 * self._path_resample_step_m)
        local_path_xy = _resample_polyline_xy(local_path_xy, self._path_resample_step_m)
        local_path_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=local_path_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=max(self._path_smoothing_alpha, 0.22),
            max_iterations=max(self._path_smoothing_max_iterations, 220),
        )
        local_path_xy = _truncate_polyline_length(local_path_xy, self._planning_horizon_m)
        local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
        local_path_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return local_path_xy

    def _build_local_path_from_centerline(self, centerline: Centerline, *, rear_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
        centerline_xy = np.column_stack([centerline.x_m, centerline.y_m]).astype(np.float64)
        if centerline_xy.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        if float(centerline_xy[0, 0]) > 1.0e-6:
            tangent_length_m = max(0.10, min(0.45, 0.5 * float(centerline_xy[0, 0])))
            connector_xy = _cubic_bezier_xy(
                p0_xy=np.asarray([0.0, 0.0], dtype=np.float64),
                p1_xy=np.asarray([tangent_length_m, 0.0], dtype=np.float64),
                p2_xy=np.asarray(
                    [
                        max(0.04, float(centerline_xy[0, 0]) - tangent_length_m),
                        float(centerline_xy[0, 1]),
                    ],
                    dtype=np.float64,
                ),
                p3_xy=centerline_xy[0],
                point_count=self._origin_bridge_point_count,
            )
            local_path_xy = np.vstack([connector_xy, centerline_xy])
        else:
            local_path_xy = centerline_xy.copy()
            local_path_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return self._stabilize_local_path(local_path_xy, rear_xy=rear_xy, yaw_rad=yaw_rad)

    def _build_fallback_curve_window_path(
        self,
        *,
        latest_scan_local_xy: np.ndarray,
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> tuple[np.ndarray, float] | None:
        if latest_scan_local_xy.shape[0] < 40:
            return None
        detection = detect_curve_window_points(
            latest_scan_local_xy[:, 0],
            latest_scan_local_xy[:, 1],
            config=self._curve_window_config,
        )
        if not detection.valid or detection.trajectory is None:
            return None
        local_path_lidar_xy = np.column_stack(
            [detection.trajectory.x_m, detection.trajectory.y_m]
        ).astype(np.float64)
        local_path_rear_xy = local_path_lidar_xy + self._rear_to_lidar_local.reshape(1, 2)
        local_path_rear_xy = self._stabilize_local_path(local_path_rear_xy, rear_xy=rear_xy, yaw_rad=yaw_rad)
        if local_path_rear_xy.shape[0] == 0:
            return None
        candidate = detection.candidate
        candidate_width_m = 0.0
        if candidate is not None:
            candidate_width_m = max(
                float(candidate.window_width_m),
                float(candidate.entry_width_m),
                float(candidate.straight_width_m),
                float(candidate.curve_width_m),
            )
        corridor_width_m = max(
            self._corridor_min_width_m,
            min(self._corridor_max_width_m, candidate_width_m),
        )
        return local_path_rear_xy, corridor_width_m

    def build_candidate(
        self,
        *,
        eval_time_s: float,
        rear_xy: np.ndarray,
        rear_yaw_rad: float,
        scan_frames: list[ScanFrame],
    ) -> tuple[np.ndarray | None, dict[str, float | str | None]]:
        recent_frames = [
            frame
            for frame in scan_frames
            if (frame.stamp_s <= eval_time_s) and (frame.stamp_s >= (eval_time_s - self._rolling_window_s))
        ]
        rolling_parts: list[np.ndarray] = []
        for frame in recent_frames:
            current_local_xy = _transform_world_to_local(frame.world_xy, rear_xy, rear_yaw_rad)
            mask = (
                np.isfinite(current_local_xy[:, 0])
                & np.isfinite(current_local_xy[:, 1])
                & (current_local_xy[:, 0] >= -0.15)
                & (current_local_xy[:, 0] <= (self._planning_horizon_m + 0.60))
                & (np.abs(current_local_xy[:, 1]) <= (self._corridor_max_width_m + 1.0))
            )
            if int(np.count_nonzero(mask)) > 0:
                rolling_parts.append(current_local_xy[mask])
        rolling_points_xy = (
            np.vstack(rolling_parts) if rolling_parts else np.empty((0, 2), dtype=np.float64)
        )
        centerline = _extract_centerline(
            points_xy=rolling_points_xy,
            horizon_m=self._planning_horizon_m,
            bin_m=self._corridor_bin_m,
            lower_quantile=self._corridor_lower_quantile,
            upper_quantile=self._corridor_upper_quantile,
            min_bin_points=self._corridor_min_bin_points,
            min_width_m=self._corridor_min_width_m,
            max_width_m=self._corridor_max_width_m,
            max_gap_bins=self._corridor_gap_fill_bins,
        )
        local_path_xy = None
        corridor_width_m: float | None = None
        source = "waiting_local_path"
        if centerline is not None:
            local_path_xy = self._build_local_path_from_centerline(
                centerline,
                rear_xy=rear_xy,
                yaw_rad=rear_yaw_rad,
            )
            corridor_width_m = centerline.width_m
            source = "tracking"
        elif recent_frames:
            fallback = self._build_fallback_curve_window_path(
                latest_scan_local_xy=recent_frames[-1].local_xy,
                rear_xy=rear_xy,
                yaw_rad=rear_yaw_rad,
            )
            if fallback is not None:
                local_path_xy, corridor_width_m = fallback
                source = "fallback_curve_window"

        if local_path_xy is not None and local_path_xy.shape[0] >= 2:
            world_path_xy = _transform_local_to_world(local_path_xy, rear_xy, rear_yaw_rad)
            self._last_local_path_world_xy = world_path_xy
            self._last_local_path_source = source
            metrics = {
                "source": source,
                "corridor_width_m": float(corridor_width_m or 0.0),
                "path_forward_span_m": float(local_path_xy[-1, 0] - local_path_xy[0, 0]),
                "path_length_m": float(_polyline_length_m(local_path_xy)),
                "path_max_curvature_m_inv": float(
                    np.max(np.abs(_estimate_path_curvature(local_path_xy)))
                ),
                "rolling_scan_count": float(len(recent_frames)),
                "rolling_point_count": float(rolling_points_xy.shape[0]),
            }
            return world_path_xy, metrics

        if self._last_local_path_world_xy is not None:
            metrics = {
                "source": "holding_last_path",
                "corridor_width_m": float(corridor_width_m or 0.0),
                "path_forward_span_m": math.nan,
                "path_length_m": math.nan,
                "path_max_curvature_m_inv": math.nan,
                "rolling_scan_count": float(len(recent_frames)),
                "rolling_point_count": float(rolling_points_xy.shape[0]),
            }
            return self._last_local_path_world_xy, metrics
        return None, {
            "source": "waiting_local_path",
            "corridor_width_m": float(corridor_width_m or 0.0),
            "path_forward_span_m": math.nan,
            "path_length_m": math.nan,
            "path_max_curvature_m_inv": math.nan,
            "rolling_scan_count": float(len(recent_frames)),
            "rolling_point_count": float(rolling_points_xy.shape[0]),
        }


class OfflineTrackerProxy:
    def __init__(self, tracker_params: dict) -> None:
        self._min_lookahead_m = float(tracker_params["min_lookahead_m"])
        self._max_lookahead_m = float(tracker_params["max_lookahead_m"])
        self._lookahead_speed_gain = float(tracker_params["lookahead_speed_gain"])
        self._sharp_turn_lookahead_min_m = float(tracker_params["sharp_turn_lookahead_min_m"])
        self._lookahead_curvature_gain = float(tracker_params["lookahead_curvature_gain"])
        self._lookahead_curvature_window_points = int(tracker_params["lookahead_curvature_window_points"])
        self._path_stale_max_age_s = float(tracker_params["path_stale_max_age_s"])
        self._max_path_deviation_m = float(tracker_params["max_path_deviation_m"])
        self._path_end_goal_tolerance_m = float(tracker_params["path_end_goal_tolerance_m"])
        self._path_end_yaw_tolerance_rad = float(tracker_params["path_end_yaw_tolerance_rad"])
        self._path_end_line_stop_margin_m = float(tracker_params["path_end_line_stop_margin_m"])
        self._path_end_line_activation_tail_points = int(
            tracker_params["path_end_line_activation_tail_points"]
        )

    def _latest_path_at(self, path_items: list[PathHistoryItem], t_s: float) -> PathHistoryItem | None:
        if not path_items:
            return None
        times = [item.stamp_s for item in path_items]
        index = int(np.searchsorted(times, t_s, side="right")) - 1
        if index < 0:
            return None
        return path_items[index]

    def _step(self, *, pose_xy_yaw: np.ndarray, speed_mps: float, path_item: PathHistoryItem, t_s: float) -> dict[str, float | bool]:
        rear_xy = pose_xy_yaw[:2]
        rear_yaw = float(pose_xy_yaw[2])
        path_points_xy = path_item.path_xy
        deltas = path_points_xy - rear_xy.reshape(1, 2)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        nearest_index = int(np.argmin(distances))
        path_deviation_m = float(distances[nearest_index])
        path_age_s = max(0.0, float(t_s - path_item.stamp_s))
        path_is_stale = path_age_s > self._path_stale_max_age_s
        goal_point = path_points_xy[-1]
        goal_yaw_rad = float(path_item.path_yaw[-1])
        goal_tangent = np.asarray([math.cos(goal_yaw_rad), math.sin(goal_yaw_rad)], dtype=np.float64)
        goal_normal_left = np.asarray([-math.sin(goal_yaw_rad), math.cos(goal_yaw_rad)], dtype=np.float64)
        goal_delta_xy = rear_xy - goal_point
        goal_line_projection_m = float(np.dot(goal_delta_xy, goal_tangent))
        goal_line_crossed = bool(
            nearest_index >= max(0, path_points_xy.shape[0] - self._path_end_line_activation_tail_points)
            and goal_line_projection_m >= self._path_end_line_stop_margin_m
        )
        goal_distance_m = float(np.linalg.norm(goal_point - rear_xy))
        goal_yaw_error_rad = abs(_normalize_angle(goal_yaw_rad - rear_yaw))
        goal_line_alignment_ready = goal_yaw_error_rad <= self._path_end_yaw_tolerance_rad
        local_path_goal_ready = bool(
            (goal_line_crossed and goal_line_alignment_ready)
            or (
                goal_distance_m <= self._path_end_goal_tolerance_m
                and goal_yaw_error_rad <= self._path_end_yaw_tolerance_rad
            )
        )
        base_lookahead_m = min(
            self._max_lookahead_m,
            max(
                self._min_lookahead_m,
                self._min_lookahead_m + (self._lookahead_speed_gain * float(speed_mps)),
            ),
        )
        ahead_path_curvature_m_inv = 0.0
        curvature_window_end = min(
            path_item.path_curvature_m_inv.shape[0],
            nearest_index + self._lookahead_curvature_window_points,
        )
        curvature_window = np.abs(path_item.path_curvature_m_inv[nearest_index:curvature_window_end])
        if curvature_window.size > 0:
            ahead_path_curvature_m_inv = float(np.max(curvature_window))
        lookahead_scale = 1.0 / (1.0 + (self._lookahead_curvature_gain * ahead_path_curvature_m_inv))
        lookahead_m = min(
            self._max_lookahead_m,
            max(self._sharp_turn_lookahead_min_m, base_lookahead_m * lookahead_scale),
        )
        target_s = float(path_item.path_s[nearest_index] + lookahead_m)
        target_index = int(np.searchsorted(path_item.path_s, target_s, side="left"))
        target_index = min(target_index, path_points_xy.shape[0] - 1)
        target_xy = path_points_xy[target_index]
        cos_yaw = math.cos(rear_yaw)
        sin_yaw = math.sin(rear_yaw)
        dx_world = float(target_xy[0] - rear_xy[0])
        dy_world = float(target_xy[1] - rear_xy[1])
        dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
        dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)
        while dx_local <= 1.0e-3 and target_index < (path_points_xy.shape[0] - 1):
            target_index += 1
            target_xy = path_points_xy[target_index]
            dx_world = float(target_xy[0] - rear_xy[0])
            dy_world = float(target_xy[1] - rear_xy[1])
            dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
            dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)
        no_forward_target = bool((dx_local <= 1.0e-3) and not local_path_goal_ready)
        return {
            "path_age_s": path_age_s,
            "path_is_stale": path_is_stale,
            "path_deviation_m": path_deviation_m,
            "path_loss": path_deviation_m > self._max_path_deviation_m,
            "goal_ready": local_path_goal_ready,
            "no_forward_target": no_forward_target,
        }

    def evaluate(
        self,
        *,
        trajectory: TrajectorySeries,
        path_items: list[PathHistoryItem],
    ) -> dict[str, dict[str, np.ndarray | float | int]]:
        if trajectory.times_s.size == 0:
            raise ValueError("empty trajectory")
        start_s = float(trajectory.times_s[0])
        end_s = float(trajectory.times_s[-1])
        dense_times_s = np.arange(start_s, end_s + (0.5 / 30.0), 1.0 / 30.0, dtype=np.float64)

        sparse_odom_age = []
        dense_odom_age = []
        path_age_series = []
        sparse_no_forward = []
        dense_no_forward = []
        sparse_path_loss = []
        dense_path_loss = []
        sparse_positions = []
        dense_positions = []

        for t_s in dense_times_s:
            path_item = self._latest_path_at(path_items, float(t_s))
            if path_item is None:
                sparse_odom_age.append(math.nan)
                dense_odom_age.append(math.nan)
                path_age_series.append(math.nan)
                sparse_no_forward.append(False)
                dense_no_forward.append(False)
                sparse_path_loss.append(False)
                dense_path_loss.append(False)
                sparse_positions.append([math.nan, math.nan])
                dense_positions.append([math.nan, math.nan])
                continue

            sparse_pose = trajectory.latest_before(float(t_s))
            dense_pose = trajectory.interpolate(float(t_s))
            if sparse_pose is None or dense_pose is None:
                sparse_odom_age.append(math.nan)
                dense_odom_age.append(math.nan)
                path_age_series.append(math.nan)
                sparse_no_forward.append(False)
                dense_no_forward.append(False)
                sparse_path_loss.append(False)
                dense_path_loss.append(False)
                sparse_positions.append([math.nan, math.nan])
                dense_positions.append([math.nan, math.nan])
                continue

            sparse_pose_xy_yaw, sparse_speed_mps, sparse_stamp_s = sparse_pose
            dense_pose_xy_yaw, dense_speed_mps = dense_pose
            sparse_state = self._step(
                pose_xy_yaw=sparse_pose_xy_yaw,
                speed_mps=sparse_speed_mps,
                path_item=path_item,
                t_s=float(t_s),
            )
            dense_state = self._step(
                pose_xy_yaw=dense_pose_xy_yaw,
                speed_mps=dense_speed_mps,
                path_item=path_item,
                t_s=float(t_s),
            )

            sparse_odom_age.append(max(0.0, float(t_s - sparse_stamp_s)))
            dense_odom_age.append(0.0)
            path_age_series.append(float(sparse_state["path_age_s"]))
            sparse_no_forward.append(bool(sparse_state["no_forward_target"]))
            dense_no_forward.append(bool(dense_state["no_forward_target"]))
            sparse_path_loss.append(bool(sparse_state["path_loss"]))
            dense_path_loss.append(bool(dense_state["path_loss"]))
            sparse_positions.append(sparse_pose_xy_yaw[:2].tolist())
            dense_positions.append(dense_pose_xy_yaw[:2].tolist())

        return {
            "timeline": {
                "times_s": dense_times_s,
                "sparse_odom_age_s": np.asarray(sparse_odom_age, dtype=np.float64),
                "dense_odom_age_s": np.asarray(dense_odom_age, dtype=np.float64),
                "path_age_s": np.asarray(path_age_series, dtype=np.float64),
                "sparse_no_forward_mask": np.asarray(sparse_no_forward, dtype=bool),
                "dense_no_forward_mask": np.asarray(dense_no_forward, dtype=bool),
                "sparse_path_loss_mask": np.asarray(sparse_path_loss, dtype=bool),
                "dense_path_loss_mask": np.asarray(dense_path_loss, dtype=bool),
                "sparse_positions_xy": np.asarray(sparse_positions, dtype=np.float64),
                "dense_positions_xy": np.asarray(dense_positions, dtype=np.float64),
            },
            "metrics": {
                "baseline_sparse_mean_odom_age_s": float(np.nanmean(sparse_odom_age)),
                "baseline_sparse_p95_odom_age_s": float(np.nanpercentile(sparse_odom_age, 95)),
                "baseline_sparse_max_odom_age_s": float(np.nanmax(sparse_odom_age)),
                "proxy_dense_mean_odom_age_s": float(np.nanmean(dense_odom_age)),
                "proxy_dense_p95_odom_age_s": float(np.nanpercentile(dense_odom_age, 95)),
                "proxy_dense_max_odom_age_s": float(np.nanmax(dense_odom_age)),
                "baseline_sparse_no_forward_segments": _count_segments(np.asarray(sparse_no_forward, dtype=bool)),
                "proxy_dense_no_forward_segments": _count_segments(np.asarray(dense_no_forward, dtype=bool)),
                "baseline_sparse_path_loss_segments": _count_segments(np.asarray(sparse_path_loss, dtype=bool)),
                "proxy_dense_path_loss_segments": _count_segments(np.asarray(dense_path_loss, dtype=bool)),
            },
        }


def _load_run_data(run_dir: Path) -> dict[str, object]:
    analysis_dir = run_dir / "analysis_recognition_tour"
    return {
        "run_dir": run_dir,
        "analysis_dir": analysis_dir,
        "summary": _load_summary(analysis_dir / "recognition_tour_summary.json"),
        "route_xy_yaw": _load_route_json(analysis_dir / "recognition_tour_route.json"),
        "path_items": _load_local_path_history(analysis_dir / "recognition_tour_local_path_history.jsonl"),
        "trajectory": _load_trajectory(analysis_dir / "recognition_tour_trajectory.csv"),
        "lidar_world_xy": _load_lidar_points(run_dir / "lidar_points.csv", max_points=30000)[0],
        "scan_frames": _load_lidar_points(run_dir / "lidar_points.csv")[1],
    }


def _actual_tracker_event_masks(trajectory: TrajectorySeries) -> dict[str, np.ndarray]:
    tracker_states = np.asarray(trajectory.tracker_state, dtype=object)
    return {
        "waiting_forward": tracker_states == "waiting_forward_path",
        "holding_last_path": tracker_states == "holding_last_path",
        "aborted_path_loss": tracker_states == "aborted_path_loss",
        "aborted_odom_timeout": tracker_states == "aborted_odom_timeout",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_proposals_dir(run_dir: Path) -> Path:
    proposals_dir = run_dir / "analysis_recognition_tour" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    return proposals_dir


def _plot_timing_comparison(
    *,
    run_name: str,
    summary: dict,
    route_xy_yaw: np.ndarray,
    baseline_paths: list[PathHistoryItem],
    trajectory: TrajectorySeries,
    lidar_world_xy: np.ndarray,
    timing_result: dict[str, object],
    output_path: Path,
) -> None:
    timeline = timing_result["timeline"]
    metrics = timing_result["metrics"]
    actual_events = _actual_tracker_event_masks(trajectory)
    sparse_positions = np.asarray(timeline["sparse_positions_xy"], dtype=np.float64)
    dense_positions = np.asarray(timeline["dense_positions_xy"], dtype=np.float64)
    sparse_no_forward_points = _segment_start_points(
        np.asarray(timeline["sparse_no_forward_mask"], dtype=bool),
        sparse_positions,
    )
    dense_no_forward_points = _segment_start_points(
        np.asarray(timeline["dense_no_forward_mask"], dtype=bool),
        dense_positions,
    )

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.22)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[1, 0])

    if lidar_world_xy.shape[0] > 0:
        ax_map.scatter(
            lidar_world_xy[:, 0],
            lidar_world_xy[:, 1],
            s=4,
            c="#8a8a8a",
            alpha=0.14,
            linewidths=0.0,
            label=f"LiDAR aggregated ({lidar_world_xy.shape[0]} pts)",
            zorder=1,
        )
    if route_xy_yaw.shape[0] > 0:
        ax_map.plot(
            route_xy_yaw[:, 0],
            route_xy_yaw[:, 1],
            color="#1f77b4",
            linewidth=2.4,
            alpha=0.75,
            label="Saved route",
            zorder=3,
        )
    if baseline_paths:
        label_used = False
        for item in baseline_paths:
            ax_map.plot(
                item.path_xy[:, 0],
                item.path_xy[:, 1],
                color="#4ea8de",
                linewidth=1.0,
                alpha=0.10,
                label="Baseline local paths" if not label_used else None,
                zorder=2,
            )
            label_used = True
    if trajectory.times_s.size > 0:
        driven_xy = np.column_stack([trajectory.x_m, trajectory.y_m])
        ax_map.plot(
            driven_xy[:, 0],
            driven_xy[:, 1],
            color="#111111",
            linewidth=2.0,
            label="Driven trajectory",
            zorder=4,
        )
    actual_waiting_points = _segment_start_points(
        actual_events["waiting_forward"],
        np.column_stack([trajectory.x_m, trajectory.y_m]),
    )
    actual_holding_points = _segment_start_points(
        actual_events["holding_last_path"],
        np.column_stack([trajectory.x_m, trajectory.y_m]),
    )
    actual_abort_points = _segment_start_points(
        actual_events["aborted_path_loss"],
        np.column_stack([trajectory.x_m, trajectory.y_m]),
    )
    if actual_waiting_points.shape[0] > 0:
        ax_map.scatter(
            actual_waiting_points[:, 0],
            actual_waiting_points[:, 1],
            c="#f4a261",
            s=42,
            marker="s",
            label="Actual waiting_forward_path",
            zorder=6,
        )
    if actual_holding_points.shape[0] > 0:
        ax_map.scatter(
            actual_holding_points[:, 0],
            actual_holding_points[:, 1],
            c="#9c6644",
            s=42,
            marker="^",
            label="Actual holding_last_path",
            zorder=6,
        )
    if actual_abort_points.shape[0] > 0:
        ax_map.scatter(
            actual_abort_points[:, 0],
            actual_abort_points[:, 1],
            c="#6a040f",
            s=50,
            marker="X",
            label="Actual aborted_path_loss",
            zorder=7,
        )
    if sparse_no_forward_points.shape[0] > 0:
        ax_map.scatter(
            sparse_no_forward_points[:, 0],
            sparse_no_forward_points[:, 1],
            facecolors="none",
            edgecolors="#d62828",
            s=70,
            marker="o",
            linewidths=1.5,
            label="Proxy sparse no-forward",
            zorder=7,
        )
    if dense_no_forward_points.shape[0] > 0:
        ax_map.scatter(
            dense_no_forward_points[:, 0],
            dense_no_forward_points[:, 1],
            c="#2a9d8f",
            s=45,
            marker="+",
            linewidths=1.5,
            label="Proxy dense no-forward",
            zorder=7,
        )

    times_rel = np.asarray(timeline["times_s"], dtype=np.float64) - float(timeline["times_s"][0])
    ax_time.plot(
        times_rel,
        np.asarray(timeline["sparse_odom_age_s"], dtype=np.float64),
        color="#d62828",
        linewidth=2.0,
        label="Sparse odom age",
    )
    ax_time.plot(
        times_rel,
        np.asarray(timeline["dense_odom_age_s"], dtype=np.float64),
        color="#2a9d8f",
        linewidth=2.0,
        label="Dense proxy odom age",
    )
    ax_time.plot(
        times_rel,
        np.asarray(timeline["path_age_s"], dtype=np.float64),
        color="#264653",
        linewidth=1.6,
        alpha=0.75,
        label="Path age",
    )
    sparse_mask = np.asarray(timeline["sparse_no_forward_mask"], dtype=bool)
    dense_mask = np.asarray(timeline["dense_no_forward_mask"], dtype=bool)
    ax_time.scatter(
        times_rel[sparse_mask],
        np.full(int(np.count_nonzero(sparse_mask)), 0.01, dtype=np.float64),
        c="#d62828",
        s=11,
        marker="o",
        alpha=0.55,
        label="Sparse no-forward",
    )
    ax_time.scatter(
        times_rel[dense_mask],
        np.full(int(np.count_nonzero(dense_mask)), 0.02, dtype=np.float64),
        c="#2a9d8f",
        s=12,
        marker="+",
        alpha=0.75,
        label="Dense no-forward",
    )

    info_lines = [
        f"run: {run_name}",
        f"end: {summary.get('end_cause')}",
        f"sparse mean odom age: {float(metrics['baseline_sparse_mean_odom_age_s']):.3f} s",
        f"dense mean odom age: {float(metrics['proxy_dense_mean_odom_age_s']):.3f} s",
        f"sparse p95 odom age: {float(metrics['baseline_sparse_p95_odom_age_s']):.3f} s",
        f"dense p95 odom age: {float(metrics['proxy_dense_p95_odom_age_s']):.3f} s",
        f"actual problem segments: {int(metrics['actual_problem_segments'])}",
        f"sparse no-forward segments: {int(metrics['baseline_sparse_no_forward_segments'])}",
        f"dense no-forward segments: {int(metrics['proxy_dense_no_forward_segments'])}",
    ]
    ax_map.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.94,
        },
    )

    ax_map.set_title("Recognition Tour Timing: sparse vs dense odometry proxy")
    ax_map.set_xlabel("x [m] (forward+)")
    ax_map.set_ylabel("y [m] (left+)")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right", framealpha=0.92)

    ax_time.set_title("Odom/path age and proxy tracker events")
    ax_time.set_xlabel("time from first trajectory sample [s]")
    ax_time.set_ylabel("age [s]")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(loc="upper right", ncol=2, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_planner_comparison(
    *,
    run_name: str,
    summary: dict,
    route_xy_yaw: np.ndarray,
    lidar_world_xy: np.ndarray,
    trajectory: TrajectorySeries,
    baseline_paths: list[PathHistoryItem],
    candidate_world_paths: list[np.ndarray],
    baseline_metrics: dict[str, np.ndarray],
    candidate_metrics: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15.5, 8.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    ax_map = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, hspace=0.28)
    ax_span = fig.add_subplot(right[0, 0])
    ax_length = fig.add_subplot(right[1, 0])
    ax_curv = fig.add_subplot(right[2, 0])

    if lidar_world_xy.shape[0] > 0:
        ax_map.scatter(
            lidar_world_xy[:, 0],
            lidar_world_xy[:, 1],
            s=4,
            c="#8a8a8a",
            alpha=0.14,
            linewidths=0.0,
            label=f"LiDAR aggregated ({lidar_world_xy.shape[0]} pts)",
            zorder=1,
        )
    if route_xy_yaw.shape[0] > 0:
        ax_map.plot(
            route_xy_yaw[:, 0],
            route_xy_yaw[:, 1],
            color="#1f77b4",
            linewidth=2.4,
            alpha=0.70,
            label="Saved route",
            zorder=3,
        )
    if baseline_paths:
        label_used = False
        for item in baseline_paths:
            ax_map.plot(
                item.path_xy[:, 0],
                item.path_xy[:, 1],
                color="#4ea8de",
                linewidth=1.0,
                alpha=0.10,
                label="Baseline local paths" if not label_used else None,
                zorder=2,
            )
            label_used = True
    if candidate_world_paths:
        label_used = False
        for path_xy in candidate_world_paths:
            ax_map.plot(
                path_xy[:, 0],
                path_xy[:, 1],
                color="#f77f00",
                linewidth=1.15,
                alpha=0.12,
                label="Candidate local paths" if not label_used else None,
                zorder=4,
            )
            label_used = True
        ax_map.plot(
            candidate_world_paths[-1][:, 0],
            candidate_world_paths[-1][:, 1],
            color="#d62828",
            linewidth=2.4,
            alpha=0.92,
            label="Latest candidate path",
            zorder=6,
        )
    if baseline_paths:
        ax_map.plot(
            baseline_paths[-1].path_xy[:, 0],
            baseline_paths[-1].path_xy[:, 1],
            color="#0b6aa2",
            linewidth=2.1,
            alpha=0.92,
            label="Latest baseline path",
            zorder=5,
        )
    if trajectory.times_s.size > 0:
        ax_map.plot(
            trajectory.x_m,
            trajectory.y_m,
            color="#111111",
            linewidth=2.0,
            label="Driven trajectory",
            zorder=6,
        )
        ax_map.scatter(
            [trajectory.x_m[-1]],
            [trajectory.y_m[-1]],
            c="#ff7f0e",
            s=55,
            marker="D",
            label="Tracking end",
            zorder=7,
        )

    indexes = np.arange(max(len(baseline_metrics["forward_span_m"]), len(candidate_metrics["forward_span_m"])))
    ax_span.plot(baseline_metrics["forward_span_m"], color="#0b6aa2", linewidth=1.8, label="Baseline")
    ax_span.plot(candidate_metrics["forward_span_m"], color="#d62828", linewidth=1.8, label="Candidate")
    ax_span.axhline(0.95, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_span.set_title("Forward span per replan")
    ax_span.set_ylabel("span [m]")
    ax_span.grid(True, alpha=0.25)
    ax_span.legend(loc="upper right", framealpha=0.92)

    ax_length.plot(baseline_metrics["path_length_m"], color="#0b6aa2", linewidth=1.8)
    ax_length.plot(candidate_metrics["path_length_m"], color="#d62828", linewidth=1.8)
    ax_length.set_title("Path length per replan")
    ax_length.set_ylabel("length [m]")
    ax_length.grid(True, alpha=0.25)

    ax_curv.plot(baseline_metrics["path_max_curvature_m_inv"], color="#0b6aa2", linewidth=1.8)
    ax_curv.plot(candidate_metrics["path_max_curvature_m_inv"], color="#d62828", linewidth=1.8)
    ax_curv.set_title("Max curvature per replan")
    ax_curv.set_ylabel("curvature [1/m]")
    ax_curv.set_xlabel("replan index")
    ax_curv.grid(True, alpha=0.25)

    info_lines = [
        f"run: {run_name}",
        f"end: {summary.get('end_cause')}",
        f"baseline short paths: {int(np.count_nonzero(np.asarray(baseline_metrics['forward_span_m']) < 0.95))}",
        f"candidate short paths: {int(np.count_nonzero(np.asarray(candidate_metrics['forward_span_m']) < 0.95))}",
        f"baseline mean span: {float(np.nanmean(baseline_metrics['forward_span_m'])):.2f} m",
        f"candidate mean span: {float(np.nanmean(candidate_metrics['forward_span_m'])):.2f} m",
        f"baseline p95 curvature: {float(np.nanpercentile(baseline_metrics['path_max_curvature_m_inv'], 95)):.2f} 1/m",
        f"candidate p95 curvature: {float(np.nanpercentile(candidate_metrics['path_max_curvature_m_inv'], 95)):.2f} 1/m",
    ]
    ax_map.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.94,
        },
    )

    ax_map.set_title("Recognition Tour Planner: baseline vs candidate local paths")
    ax_map.set_xlabel("x [m] (forward+)")
    ax_map.set_ylabel("y [m] (left+)")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _evaluate_timing(
    *,
    run_data: dict[str, object],
    tracker_params: dict,
    output_dir: Path,
) -> dict:
    tracker_proxy = OfflineTrackerProxy(tracker_params)
    timing_result = tracker_proxy.evaluate(
        trajectory=run_data["trajectory"],
        path_items=run_data["path_items"],
    )
    actual_events = _actual_tracker_event_masks(run_data["trajectory"])
    metrics = {
        "run": run_data["run_dir"].name,
        "subsystem": "timing",
        "actual_waiting_forward_segments": _count_segments(actual_events["waiting_forward"]),
        "actual_holding_last_path_segments": _count_segments(actual_events["holding_last_path"]),
        "actual_aborted_path_loss_segments": _count_segments(actual_events["aborted_path_loss"]),
        "actual_aborted_odom_timeout_segments": _count_segments(actual_events["aborted_odom_timeout"]),
        "actual_problem_segments": (
            _count_segments(actual_events["waiting_forward"])
            + _count_segments(actual_events["holding_last_path"])
            + _count_segments(actual_events["aborted_path_loss"])
            + _count_segments(actual_events["aborted_odom_timeout"])
        ),
        **timing_result["metrics"],
    }
    timing_result["metrics"] = metrics
    _write_json(output_dir / "timing_metrics.json", metrics)
    _plot_timing_comparison(
        run_name=run_data["run_dir"].name,
        summary=run_data["summary"],
        route_xy_yaw=run_data["route_xy_yaw"],
        baseline_paths=run_data["path_items"],
        trajectory=run_data["trajectory"],
        lidar_world_xy=run_data["lidar_world_xy"],
        timing_result=timing_result,
        output_path=output_dir / "timing_comparison.png",
    )
    return metrics


def _evaluate_planner(
    *,
    run_data: dict[str, object],
    planner_params: dict,
    output_dir: Path,
) -> dict:
    planner = OfflineRecognitionPlanner(planner_params)
    trajectory: TrajectorySeries = run_data["trajectory"]
    scan_frames: list[ScanFrame] = run_data["scan_frames"]
    baseline_paths: list[PathHistoryItem] = run_data["path_items"]

    candidate_world_paths: list[np.ndarray] = []
    baseline_forward_span = []
    baseline_path_length = []
    baseline_curvature = []
    candidate_forward_span = []
    candidate_path_length = []
    candidate_curvature = []
    missing_candidate_count = 0

    for item in baseline_paths:
        pose = trajectory.interpolate(item.stamp_s)
        if pose is None:
            continue
        pose_xy_yaw, _speed_mps = pose
        yaw_rad = float(pose_xy_yaw[2])
        rear_xy = np.asarray(
            [
                float(pose_xy_yaw[0]) + (_rotation(yaw_rad) @ planner._rear_axle_offset)[0],
                float(pose_xy_yaw[1]) + (_rotation(yaw_rad) @ planner._rear_axle_offset)[1],
            ],
            dtype=np.float64,
        )
        baseline_local_xy = _transform_world_to_local(item.path_xy, rear_xy, yaw_rad)
        baseline_forward_span.append(
            float(baseline_local_xy[-1, 0] - baseline_local_xy[0, 0]) if baseline_local_xy.shape[0] >= 2 else math.nan
        )
        baseline_path_length.append(float(_polyline_length_m(baseline_local_xy)))
        baseline_curvature.append(float(np.max(np.abs(_estimate_path_curvature(baseline_local_xy)))))

        candidate_world_xy, candidate_metrics = planner.build_candidate(
            eval_time_s=item.stamp_s,
            rear_xy=rear_xy,
            rear_yaw_rad=yaw_rad,
            scan_frames=scan_frames,
        )
        if candidate_world_xy is None:
            missing_candidate_count += 1
            candidate_forward_span.append(math.nan)
            candidate_path_length.append(math.nan)
            candidate_curvature.append(math.nan)
            continue
        candidate_world_paths.append(candidate_world_xy)
        candidate_local_xy = _transform_world_to_local(candidate_world_xy, rear_xy, yaw_rad)
        candidate_forward_span.append(float(candidate_metrics["path_forward_span_m"]))
        candidate_path_length.append(float(candidate_metrics["path_length_m"]))
        candidate_curvature.append(float(candidate_metrics["path_max_curvature_m_inv"]))

    baseline_metrics = {
        "forward_span_m": np.asarray(baseline_forward_span, dtype=np.float64),
        "path_length_m": np.asarray(baseline_path_length, dtype=np.float64),
        "path_max_curvature_m_inv": np.asarray(baseline_curvature, dtype=np.float64),
    }
    candidate_metrics = {
        "forward_span_m": np.asarray(candidate_forward_span, dtype=np.float64),
        "path_length_m": np.asarray(candidate_path_length, dtype=np.float64),
        "path_max_curvature_m_inv": np.asarray(candidate_curvature, dtype=np.float64),
    }

    metrics = {
        "run": run_data["run_dir"].name,
        "subsystem": "planner",
        "baseline_path_count": int(len(baseline_paths)),
        "candidate_path_count": int(len(candidate_world_paths)),
        "missing_candidate_count": int(missing_candidate_count),
        "baseline_mean_forward_span_m": float(np.nanmean(baseline_metrics["forward_span_m"])),
        "candidate_mean_forward_span_m": float(np.nanmean(candidate_metrics["forward_span_m"])),
        "baseline_short_path_count": int(np.count_nonzero(baseline_metrics["forward_span_m"] < 0.95)),
        "candidate_short_path_count": int(np.count_nonzero(candidate_metrics["forward_span_m"] < 0.95)),
        "baseline_mean_path_length_m": float(np.nanmean(baseline_metrics["path_length_m"])),
        "candidate_mean_path_length_m": float(np.nanmean(candidate_metrics["path_length_m"])),
        "baseline_p95_path_max_curvature_m_inv": float(np.nanpercentile(baseline_metrics["path_max_curvature_m_inv"], 95)),
        "candidate_p95_path_max_curvature_m_inv": float(np.nanpercentile(candidate_metrics["path_max_curvature_m_inv"], 95)),
    }
    _write_json(output_dir / "planner_metrics.json", metrics)
    _plot_planner_comparison(
        run_name=run_data["run_dir"].name,
        summary=run_data["summary"],
        route_xy_yaw=run_data["route_xy_yaw"],
        lidar_world_xy=run_data["lidar_world_xy"],
        trajectory=trajectory,
        baseline_paths=baseline_paths,
        candidate_world_paths=candidate_world_paths,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        output_path=output_dir / "planner_comparison.png",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Recognition tour run directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--subsystem",
        choices=["all", "timing", "planner"],
        default="all",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "ros2_ws" / "src" / "apex_telemetry" / "config" / "apex_params.yaml"),
    )
    args = parser.parse_args()

    planner_params, tracker_params = _load_params(Path(args.config).expanduser().resolve())
    for raw_run_dir in args.run_dir:
        run_dir = Path(raw_run_dir).expanduser().resolve()
        run_data = _load_run_data(run_dir)
        output_dir = _make_proposals_dir(run_dir)
        if args.subsystem in {"all", "timing"}:
            _evaluate_timing(
                run_data=run_data,
                tracker_params=tracker_params,
                output_dir=output_dir,
            )
        if args.subsystem in {"all", "planner"}:
            _evaluate_planner(
                run_data=run_data,
                planner_params=planner_params,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
