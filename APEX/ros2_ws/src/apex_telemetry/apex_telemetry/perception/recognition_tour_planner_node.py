#!/usr/bin/env python3
"""Continuously plan a short corridor-following path for full-lap recognition."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from .curve_entry_path_planner_node import (
    _cubic_bezier_xy,
    _estimate_path_curvature,
    _polyline_length_m,
    _resample_polyline_xy,
    _rotation,
    _sanitize_ranges,
    _smooth_path_to_curvature_limit,
    _yaw_to_quat,
)
from .curve_window_detection import (
    CurveWindowDetectionConfig,
    detect_curve_window_points,
    scan_ranges_to_forward_left_xy,
)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _transform_local_to_world(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) @ _rotation(yaw_rad).T) + origin_xy.reshape(1, 2)


def _transform_world_to_local(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) - origin_xy.reshape(1, 2)) @ _rotation(yaw_rad)


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


def _truncate_polyline_length(path_xy: np.ndarray, max_length_m: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return path_xy.copy()
    max_length_m = max(1.0e-3, float(max_length_m))
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
        remaining_s = max_length_m - last_s
        seg_length = max(1.0e-9, float(seg_lengths[cutoff_index - 1]))
        ratio = remaining_s / seg_length
        interpolated_xy = prev_xy + (ratio * (next_xy - prev_xy))
        truncated_xy = np.vstack([truncated_xy, interpolated_xy])
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
    dy_m = float(delta_xy[1])
    if abs(dx_m) <= 1.0e-6:
        slope = 0.0
    else:
        slope = dy_m / dx_m
    slope = float(np.clip(slope, -1.1, 1.1))

    xs = np.arange(
        float(path_xy[-1, 0]) + max(1.0e-3, float(step_m)),
        target_forward_x_m + (0.5 * max(1.0e-3, float(step_m))),
        max(1.0e-3, float(step_m)),
        dtype=np.float64,
    )
    if xs.size == 0:
        return path_xy.copy()
    ys = float(path_xy[-1, 1]) + slope * (xs - float(path_xy[-1, 0]))
    extension_xy = np.column_stack([xs, ys])
    return np.vstack([path_xy, extension_xy])


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
    min_segment_m = max(1.0e-6, float(min_segment_m))
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
    new_path_weight = max(0.0, min(1.0, float(new_path_weight)))
    blended_xy = (
        (new_path_weight * new_eval_xy)
        + ((1.0 - new_path_weight) * previous_eval_xy)
    )
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


def _path_forward_span_m(path_xy: np.ndarray) -> float:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] < 2:
        return 0.0
    return max(0.0, float(path_xy[-1, 0] - path_xy[0, 0]))


def _bridge_path_from_origin(path_xy: np.ndarray, bridge_point_count: int) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] == 0:
        return path_xy.copy()
    if float(path_xy[0, 0]) <= 1.0e-6:
        bridged_xy = path_xy.copy()
        bridged_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return bridged_xy
    tangent_length_m = max(0.10, min(0.45, 0.5 * float(path_xy[0, 0])))
    connector_xy = _cubic_bezier_xy(
        p0_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        p1_xy=np.asarray([tangent_length_m, 0.0], dtype=np.float64),
        p2_xy=np.asarray(
            [
                max(0.04, float(path_xy[0, 0]) - tangent_length_m),
                float(path_xy[0, 1]),
            ],
            dtype=np.float64,
        ),
        p3_xy=path_xy[0],
        point_count=max(2, int(bridge_point_count)),
    )
    return np.vstack([connector_xy, path_xy])


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
    entry_tangent_xy = np.asarray(
        [math.cos(entry_heading_rad), math.sin(entry_heading_rad)],
        dtype=np.float64,
    )
    tail_tangent_xy = np.asarray(
        [math.cos(tail_heading_rad), math.sin(tail_heading_rad)],
        dtype=np.float64,
    )
    control_length_m = min(0.30, max(0.08, 0.45 * gap_m))
    connector_xy = _cubic_bezier_xy(
        p0_xy=path_xy[-1],
        p1_xy=path_xy[-1] + (control_length_m * entry_tangent_xy),
        p2_xy=tail_xy[0] - (control_length_m * tail_tangent_xy),
        p3_xy=tail_xy[0],
        point_count=max(4, int(math.ceil(gap_m / max(1.0e-3, step_m))) + 2),
    )
    return np.vstack([path_xy[:-1], connector_xy, tail_xy[1:]])


def _compute_path_s(path_xy: np.ndarray) -> np.ndarray:
    if path_xy.shape[0] <= 1:
        return np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])


def _extend_path_along_terminal_heading(
    path_xy: np.ndarray,
    *,
    target_forward_span_m: float,
    step_m: float,
) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] < 2:
        return path_xy.copy()

    current_forward_span_m = _path_forward_span_m(path_xy)
    needed_forward_m = float(target_forward_span_m) - current_forward_span_m
    if needed_forward_m <= max(0.01, 0.5 * step_m):
        return path_xy.copy()

    terminal_heading_rad = _path_terminal_heading(path_xy)
    terminal_tangent_xy = np.asarray(
        [math.cos(terminal_heading_rad), math.sin(terminal_heading_rad)],
        dtype=np.float64,
    )
    if terminal_tangent_xy[0] <= 1.0e-3:
        terminal_tangent_xy = np.asarray([1.0, 0.0], dtype=np.float64)

    extension_point_count = max(2, int(math.ceil(needed_forward_m / max(step_m, 1.0e-3))) + 1)
    extension_points = [path_xy[-1]]
    for idx in range(1, extension_point_count + 1):
        extension_points.append(
            path_xy[-1] + (idx * step_m * terminal_tangent_xy)
        )
    extended_xy = np.vstack([path_xy[:-1], np.asarray(extension_points, dtype=np.float64)])
    return _enforce_monotonic_forward_x(extended_xy)


def _polyline_yaw(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return np.zeros((path_xy.shape[0],), dtype=np.float64)
    yaw = np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    yaw[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
    yaw[-1] = yaw[-2]
    return yaw


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


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


@dataclass
class _CorridorCenterline:
    x_m: np.ndarray
    y_m: np.ndarray
    width_m: float
    valid_bin_count: int


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
) -> _CorridorCenterline | None:
    if points_xy.shape[0] < max(40, 3 * min_bin_points):
        return None

    horizon_m = max(bin_m, float(horizon_m))
    bin_edges = np.arange(0.0, horizon_m + bin_m, bin_m, dtype=np.float64)
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

    median_width_m = float(np.median(width[valid_mask]))
    return _CorridorCenterline(
        x_m=x_valid,
        y_m=center_valid,
        width_m=median_width_m,
        valid_bin_count=int(np.count_nonzero(valid_mask)),
    )


@dataclass
class _ScanSnapshot:
    stamp_s: float
    lidar_points_local_xy: np.ndarray
    lidar_world_xy: np.ndarray


class RecognitionTourPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("recognition_tour_planner_node")

        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("fusion_status_topic", "/apex/estimation/status")
        self.declare_parameter("local_path_topic", "/apex/planning/recognition_tour_local_path")
        self.declare_parameter("route_topic", "/apex/planning/recognition_tour_route")
        self.declare_parameter("status_topic", "/apex/planning/recognition_tour_status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("plan_rate_hz", 12.0)
        self.declare_parameter("status_publish_rate_hz", 5.0)
        self.declare_parameter("rolling_window_s", 0.5)
        self.declare_parameter("planning_horizon_m", 1.8)
        self.declare_parameter("lidar_offset_x_m", 0.18)
        self.declare_parameter("lidar_offset_y_m", 0.0)
        self.declare_parameter("rear_axle_offset_x_m", -0.15)
        self.declare_parameter("rear_axle_offset_y_m", 0.0)
        self.declare_parameter("origin_bridge_point_count", 10)
        self.declare_parameter("planning_wheelbase_m", 0.30)
        self.declare_parameter("planning_max_steering_deg", 18.0)
        self.declare_parameter("path_curvature_limit_scale", 0.90)
        self.declare_parameter("path_resample_step_m", 0.04)
        self.declare_parameter("path_smoothing_alpha", 0.20)
        self.declare_parameter("path_smoothing_max_iterations", 140)
        self.declare_parameter("path_min_forward_progress_m", 1.45)
        self.declare_parameter("min_publish_forward_span_m", 0.55)
        self.declare_parameter("straight_entry_hold_length_m", 0.45)
        self.declare_parameter("replan_path_blend_alpha", 0.68)
        self.declare_parameter("previous_path_tail_extension_m", 0.55)
        self.declare_parameter("previous_path_tail_graft_min_span_m", 0.85)
        self.declare_parameter("corridor_bin_m", 0.10)
        self.declare_parameter("corridor_quantile", 0.18)
        self.declare_parameter("corridor_min_bin_points", 8)
        self.declare_parameter("corridor_min_width_m", 0.45)
        self.declare_parameter("corridor_max_width_m", 2.40)
        self.declare_parameter("corridor_gap_fill_bins", 2)
        self.declare_parameter("route_point_spacing_m", 0.05)
        self.declare_parameter("accepted_path_prefix_length_m", 0.80)
        self.declare_parameter("route_resample_step_m", 0.05)
        self.declare_parameter("route_smoothing_alpha", 0.20)
        self.declare_parameter("route_smoothing_max_iterations", 180)
        self.declare_parameter("start_axis_arm_distance_m", 4.0)
        self.declare_parameter("start_axis_arm_time_s", 6.0)
        self.declare_parameter("start_axis_crossing_neg_margin_m", -0.05)
        self.declare_parameter("start_axis_crossing_pos_margin_m", 0.05)
        self.declare_parameter("start_axis_lateral_tolerance_m", 0.60)
        self.declare_parameter("global_timeout_s", 60.0)
        self.declare_parameter("low_confidence_hold_path_s", 0.60)
        self.declare_parameter("second_corridor_target_depth_m", 0.15)
        self.declare_parameter("second_corridor_target_depth_min_m", 0.15)
        self.declare_parameter("second_corridor_target_depth_max_m", 0.15)
        self.declare_parameter("inner_vertex_clearance_m", 0.12)
        self.declare_parameter("curve_apex_width_fraction", 0.40)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._fusion_status_topic = str(self.get_parameter("fusion_status_topic").value)
        self._local_path_topic = str(self.get_parameter("local_path_topic").value)
        self._route_topic = str(self.get_parameter("route_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._plan_rate_hz = max(1.0, float(self.get_parameter("plan_rate_hz").value))
        self._status_publish_rate_hz = max(
            0.5, float(self.get_parameter("status_publish_rate_hz").value)
        )
        self._rolling_window_s = max(0.1, float(self.get_parameter("rolling_window_s").value))
        self._planning_horizon_m = max(0.6, float(self.get_parameter("planning_horizon_m").value))
        self._lidar_offset = np.asarray(
            [
                float(self.get_parameter("lidar_offset_x_m").value),
                float(self.get_parameter("lidar_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._rear_axle_offset = np.asarray(
            [
                float(self.get_parameter("rear_axle_offset_x_m").value),
                float(self.get_parameter("rear_axle_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._origin_bridge_point_count = max(
            2, int(self.get_parameter("origin_bridge_point_count").value)
        )
        planning_wheelbase_m = max(
            1.0e-3, float(self.get_parameter("planning_wheelbase_m").value)
        )
        planning_max_steering_deg = max(
            1.0, float(self.get_parameter("planning_max_steering_deg").value)
        )
        self._path_curvature_limit_scale = max(
            0.1, min(1.0, float(self.get_parameter("path_curvature_limit_scale").value))
        )
        self._path_resample_step_m = max(
            1.0e-3, float(self.get_parameter("path_resample_step_m").value)
        )
        self._path_smoothing_alpha = max(
            0.0, min(1.0, float(self.get_parameter("path_smoothing_alpha").value))
        )
        self._path_smoothing_max_iterations = max(
            0, int(self.get_parameter("path_smoothing_max_iterations").value)
        )
        self._path_min_forward_progress_m = max(
            0.4, float(self.get_parameter("path_min_forward_progress_m").value)
        )
        self._min_publish_forward_span_m = max(
            0.15, float(self.get_parameter("min_publish_forward_span_m").value)
        )
        self._straight_entry_hold_length_m = max(
            0.0, float(self.get_parameter("straight_entry_hold_length_m").value)
        )
        self._replan_path_blend_alpha = max(
            0.0, min(1.0, float(self.get_parameter("replan_path_blend_alpha").value))
        )
        self._previous_path_tail_extension_m = max(
            0.0, float(self.get_parameter("previous_path_tail_extension_m").value)
        )
        self._previous_path_tail_graft_min_span_m = max(
            0.2, float(self.get_parameter("previous_path_tail_graft_min_span_m").value)
        )
        self._corridor_bin_m = max(0.02, float(self.get_parameter("corridor_bin_m").value))
        corridor_quantile = max(0.01, min(0.49, float(self.get_parameter("corridor_quantile").value)))
        self._corridor_lower_quantile = corridor_quantile
        self._corridor_upper_quantile = 1.0 - corridor_quantile
        self._corridor_min_bin_points = max(
            4, int(self.get_parameter("corridor_min_bin_points").value)
        )
        self._corridor_min_width_m = max(
            0.10, float(self.get_parameter("corridor_min_width_m").value)
        )
        self._corridor_max_width_m = max(
            self._corridor_min_width_m,
            float(self.get_parameter("corridor_max_width_m").value),
        )
        self._corridor_gap_fill_bins = max(
            0, int(self.get_parameter("corridor_gap_fill_bins").value)
        )
        self._route_point_spacing_m = max(
            0.01, float(self.get_parameter("route_point_spacing_m").value)
        )
        self._accepted_path_prefix_length_m = max(
            self._route_point_spacing_m,
            float(self.get_parameter("accepted_path_prefix_length_m").value),
        )
        self._route_resample_step_m = max(
            0.01, float(self.get_parameter("route_resample_step_m").value)
        )
        self._route_smoothing_alpha = max(
            0.0, min(1.0, float(self.get_parameter("route_smoothing_alpha").value))
        )
        self._route_smoothing_max_iterations = max(
            0, int(self.get_parameter("route_smoothing_max_iterations").value)
        )
        self._start_axis_arm_distance_m = max(
            0.5, float(self.get_parameter("start_axis_arm_distance_m").value)
        )
        self._start_axis_arm_time_s = max(
            0.5, float(self.get_parameter("start_axis_arm_time_s").value)
        )
        self._start_axis_crossing_neg_margin_m = min(
            -1.0e-3, float(self.get_parameter("start_axis_crossing_neg_margin_m").value)
        )
        self._start_axis_crossing_pos_margin_m = max(
            1.0e-3, float(self.get_parameter("start_axis_crossing_pos_margin_m").value)
        )
        self._start_axis_lateral_tolerance_m = max(
            0.05, float(self.get_parameter("start_axis_lateral_tolerance_m").value)
        )
        self._global_timeout_s = max(2.0, float(self.get_parameter("global_timeout_s").value))
        self._low_confidence_hold_path_s = max(
            0.1, float(self.get_parameter("low_confidence_hold_path_s").value)
        )
        self._curve_window_config = CurveWindowDetectionConfig(
            second_corridor_target_depth_m=float(
                self.get_parameter("second_corridor_target_depth_m").value
            ),
            second_corridor_target_depth_min_m=float(
                self.get_parameter("second_corridor_target_depth_min_m").value
            ),
            second_corridor_target_depth_max_m=float(
                self.get_parameter("second_corridor_target_depth_max_m").value
            ),
            inner_vertex_clearance_m=float(
                self.get_parameter("inner_vertex_clearance_m").value
            ),
            curve_apex_width_fraction=float(
                self.get_parameter("curve_apex_width_fraction").value
            ),
        )
        self._max_path_curvature_m_inv = (
            self._path_curvature_limit_scale
            * math.tan(math.radians(planning_max_steering_deg))
            / planning_wheelbase_m
        )
        self._rear_to_lidar_local = self._lidar_offset - self._rear_axle_offset

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)
        self.create_subscription(String, self._fusion_status_topic, self._fusion_status_cb, 20)

        self._local_path_pub = self.create_publisher(Path, self._local_path_topic, latched_qos)
        self._route_pub = self.create_publisher(Path, self._route_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)
        self.create_timer(1.0 / self._plan_rate_hz, self._plan_step)
        self.create_timer(1.0 / self._status_publish_rate_hz, self._publish_outputs)

        self._latest_odom: dict[str, float] | None = None
        self._latest_fusion_status: dict[str, object] = {}
        self._scan_buffer: deque[_ScanSnapshot] = deque()
        self._mission_started = False
        self._mission_start_monotonic: float | None = None
        self._mission_terminal = False
        self._terminal_cause: str | None = None
        self._start_pose: dict[str, float] | None = None
        self._start_axis_normal_xy: np.ndarray | None = None
        self._start_axis_tangent_xy: np.ndarray | None = None
        self._travel_distance_m = 0.0
        self._last_rear_xy: np.ndarray | None = None
        self._loop_closure_armed = False
        self._loop_closed = False
        self._previous_axis_signed_distance_m: float | None = None
        self._last_local_path_msg: Path | None = None
        self._last_route_msg: Path | None = None
        self._last_local_path_world_xy: np.ndarray | None = None
        self._last_local_path_source = "none"
        self._last_local_path_planned_monotonic: float | None = None
        self._latest_corridor_width_m = 0.0
        self._last_candidate_path_forward_span_m = 0.0
        self._last_candidate_path_length_m = 0.0
        self._last_candidate_path_max_curvature_m_inv = 0.0
        self._last_path_rejected = False
        self._last_path_rejection_reason: str | None = None
        self._path_rejection_count = 0
        self._path_rescue_count = 0
        self._route_points_world: list[np.ndarray] = []
        self._status_payload: dict[str, object] = {
            "state": "waiting_fusion",
            "ready": False,
            "loop_closure_armed": False,
            "loop_closed": False,
            "travel_distance_m": 0.0,
            "elapsed_s": 0.0,
            "local_path_age_s": None,
            "fusion_confidence": "unknown",
            "corridor_width_m": None,
            "route_point_count": 0,
        }

        self.get_logger().info(
            "RecognitionTourPlannerNode started (scan=%s odom=%s status=%s path=%s)"
            % (
                self._scan_topic,
                self._odom_topic,
                self._status_topic,
                self._local_path_topic,
            )
        )

    def _fusion_ready(self) -> bool:
        return bool(
            self._latest_fusion_status.get("alignment_ready", False)
            and str(self._latest_fusion_status.get("state", "")) == "tracking"
        )

    def _fusion_confidence(self) -> str:
        latest_pose = self._latest_fusion_status.get("latest_pose") or {}
        return str(latest_pose.get("confidence", "unknown"))

    def _rear_axle_pose(self) -> tuple[np.ndarray, float] | None:
        if self._latest_odom is None:
            return None
        base_xy = np.asarray(
            [self._latest_odom["x_m"], self._latest_odom["y_m"]],
            dtype=np.float64,
        )
        yaw_rad = float(self._latest_odom["yaw_rad"])
        rear_xy = base_xy + (_rotation(yaw_rad) @ self._rear_axle_offset)
        return rear_xy, yaw_rad

    def _lidar_pose(self) -> tuple[np.ndarray, float] | None:
        if self._latest_odom is None:
            return None
        base_xy = np.asarray(
            [self._latest_odom["x_m"], self._latest_odom["y_m"]],
            dtype=np.float64,
        )
        yaw_rad = float(self._latest_odom["yaw_rad"])
        lidar_xy = base_xy + (_rotation(yaw_rad) @ self._lidar_offset)
        return lidar_xy, yaw_rad

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = {
            "stamp_s": float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
            "x_m": float(msg.pose.pose.position.x),
            "y_m": float(msg.pose.pose.position.y),
            "yaw_rad": _quat_to_yaw(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ),
            "vx_mps": float(msg.twist.twist.linear.x),
            "vy_mps": float(msg.twist.twist.linear.y),
            "yaw_rate_rps": float(msg.twist.twist.angular.z),
        }
        rear_pose = self._rear_axle_pose()
        if rear_pose is None:
            return
        rear_xy, rear_yaw_rad = rear_pose
        if self._last_rear_xy is None:
            self._last_rear_xy = rear_xy.copy()
            return
        step_distance_m = float(np.linalg.norm(rear_xy - self._last_rear_xy))
        if self._mission_started and step_distance_m > 1.0e-6:
            self._travel_distance_m += step_distance_m
        self._last_rear_xy = rear_xy.copy()
        if self._mission_started and step_distance_m >= self._route_point_spacing_m:
            self._append_route_point(rear_xy)

    def _fusion_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._latest_fusion_status = payload

    def _scan_cb(self, msg: LaserScan) -> None:
        if self._mission_terminal:
            return
        lidar_pose = self._lidar_pose()
        if lidar_pose is None:
            return
        lidar_xy, yaw_rad = lidar_pose
        ranges = _sanitize_ranges(list(msg.ranges))
        points_x_m, points_y_m, _ = scan_ranges_to_forward_left_xy(ranges)
        local_points_xy = np.column_stack([points_x_m, points_y_m]).astype(np.float64)
        finite_mask = np.isfinite(local_points_xy[:, 0]) & np.isfinite(local_points_xy[:, 1])
        local_points_xy = local_points_xy[finite_mask]
        if local_points_xy.shape[0] == 0:
            return
        world_points_xy = _transform_local_to_world(local_points_xy, lidar_xy, yaw_rad)
        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        self._scan_buffer.append(
            _ScanSnapshot(
                stamp_s=stamp_s,
                lidar_points_local_xy=local_points_xy,
                lidar_world_xy=world_points_xy,
            )
        )

    def _append_route_point(self, point_xy: np.ndarray) -> None:
        point_xy = np.asarray(point_xy, dtype=np.float64)
        if not self._route_points_world:
            self._route_points_world.append(point_xy.copy())
            return
        if float(np.linalg.norm(point_xy - self._route_points_world[-1])) < self._route_point_spacing_m:
            return
        self._route_points_world.append(point_xy.copy())

    def _append_local_path_prefix(self, world_path_xy: np.ndarray) -> None:
        if world_path_xy.shape[0] <= 1:
            return
        path_s = _compute_path_s(world_path_xy)
        mask = (path_s >= 0.20) & (path_s <= self._accepted_path_prefix_length_m)
        prefix_xy = world_path_xy[mask]
        for point_xy in prefix_xy:
            self._append_route_point(point_xy)

    def _ensure_mission_started(self) -> None:
        if self._mission_started or not self._fusion_ready() or self._latest_odom is None:
            return
        rear_pose = self._rear_axle_pose()
        if rear_pose is None:
            return
        rear_xy, yaw_rad = rear_pose
        heading_xy = np.asarray([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
        axis_tangent_xy = np.asarray([-heading_xy[1], heading_xy[0]], dtype=np.float64)
        self._mission_started = True
        self._mission_start_monotonic = time.monotonic()
        self._start_pose = {
            "x_m": float(rear_xy[0]),
            "y_m": float(rear_xy[1]),
            "yaw_rad": float(yaw_rad),
        }
        self._start_axis_normal_xy = heading_xy
        self._start_axis_tangent_xy = axis_tangent_xy
        self._travel_distance_m = 0.0
        self._previous_axis_signed_distance_m = 0.0
        self._append_route_point(rear_xy)
        self.get_logger().info(
            "Recognition tour start captured at (%.3f, %.3f, %.2f deg)"
            % (float(rear_xy[0]), float(rear_xy[1]), math.degrees(float(yaw_rad)))
        )

    def _prune_scan_buffer(self, now_stamp_s: float) -> None:
        while self._scan_buffer and (now_stamp_s - self._scan_buffer[0].stamp_s) > self._rolling_window_s:
            self._scan_buffer.popleft()

    def _rolling_points_in_current_frame(self) -> np.ndarray:
        rear_pose = self._rear_axle_pose()
        if rear_pose is None or not self._scan_buffer:
            return np.empty((0, 2), dtype=np.float64)
        rear_xy, yaw_rad = rear_pose
        local_parts: list[np.ndarray] = []
        for snapshot in self._scan_buffer:
            current_local_xy = _transform_world_to_local(snapshot.lidar_world_xy, rear_xy, yaw_rad)
            mask = (
                np.isfinite(current_local_xy[:, 0])
                & np.isfinite(current_local_xy[:, 1])
                & (current_local_xy[:, 0] >= -0.15)
                & (current_local_xy[:, 0] <= (self._planning_horizon_m + 0.60))
                & (np.abs(current_local_xy[:, 1]) <= (self._corridor_max_width_m + 1.0))
            )
            if int(np.count_nonzero(mask)) == 0:
                continue
            local_parts.append(current_local_xy[mask])
        if not local_parts:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(local_parts)

    def _previous_local_path_xy(self, rear_xy: np.ndarray, yaw_rad: float) -> np.ndarray | None:
        if self._last_local_path_world_xy is None or self._last_local_path_world_xy.shape[0] < 2:
            return None
        previous_local_xy = _transform_world_to_local(
            self._last_local_path_world_xy,
            rear_xy,
            yaw_rad,
        )
        mask = (
            np.isfinite(previous_local_xy[:, 0])
            & np.isfinite(previous_local_xy[:, 1])
            & (previous_local_xy[:, 0] >= -0.20)
            & (previous_local_xy[:, 0] <= (self._planning_horizon_m + 0.50))
        )
        previous_local_xy = previous_local_xy[mask]
        if previous_local_xy.shape[0] < 2:
            return None
        previous_local_xy = _enforce_monotonic_forward_x(previous_local_xy)
        return previous_local_xy

    def _turn_severity(self, path_xy: np.ndarray) -> float:
        path_xy = np.asarray(path_xy, dtype=np.float64)
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
        curvature_score = min(
            1.0,
            max_curvature / max(1.0e-3, 0.85 * self._max_path_curvature_m_inv),
        )
        return max(heading_score, lateral_score, curvature_score)

    def _stabilize_local_path(
        self,
        local_path_xy: np.ndarray,
        *,
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> np.ndarray:
        local_path_xy = np.asarray(local_path_xy, dtype=np.float64).copy()
        if local_path_xy.shape[0] == 0:
            return local_path_xy

        turn_severity = self._turn_severity(local_path_xy)
        base_forward_progress_m = min(self._planning_horizon_m, self._path_min_forward_progress_m)
        effective_forward_progress_m = (
            ((1.0 - turn_severity) * base_forward_progress_m)
            + (turn_severity * max(0.95, 0.58 * self._planning_horizon_m))
        )
        effective_straight_hold_length_m = (
            ((1.0 - turn_severity) * self._straight_entry_hold_length_m)
            + (turn_severity * min(self._straight_entry_hold_length_m, 0.14))
        )
        effective_replan_path_blend_alpha = min(
            0.90,
            self._replan_path_blend_alpha + (0.25 * turn_severity),
        )

        local_path_xy = _enforce_monotonic_forward_x(local_path_xy)
        local_path_xy = _extend_path_forward(
            local_path_xy,
            target_forward_x_m=effective_forward_progress_m,
            step_m=self._path_resample_step_m,
        )
        local_path_xy = _apply_straight_entry_hold(
            local_path_xy,
            effective_straight_hold_length_m,
        )

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
                new_path_weight=effective_replan_path_blend_alpha,
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
        local_path_xy = _deduplicate_polyline_xy(
            local_path_xy,
            min_segment_m=0.35 * self._path_resample_step_m,
        )
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

    def _rescue_previous_local_path(
        self,
        *,
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> np.ndarray | None:
        previous_local_xy = self._previous_local_path_xy(rear_xy, yaw_rad)
        if previous_local_xy is None or previous_local_xy.shape[0] < 2:
            return None

        rescue_target_forward_x_m = max(
            self._min_publish_forward_span_m + 0.25,
            min(self._planning_horizon_m, self._path_min_forward_progress_m),
        )
        rescued_xy = _bridge_path_from_origin(
            previous_local_xy,
            self._origin_bridge_point_count,
        )
        rescued_xy = _enforce_monotonic_forward_x(rescued_xy)
        rescued_xy = _extend_path_forward(
            rescued_xy,
            target_forward_x_m=rescue_target_forward_x_m,
            step_m=self._path_resample_step_m,
        )
        rescued_xy = _truncate_polyline_length(rescued_xy, self._planning_horizon_m)
        rescued_xy = _deduplicate_polyline_xy(
            rescued_xy,
            min_segment_m=0.35 * self._path_resample_step_m,
        )
        rescued_xy = _resample_polyline_xy(rescued_xy, self._path_resample_step_m)
        rescued_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=rescued_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=max(self._path_smoothing_alpha, 0.22),
            max_iterations=max(self._path_smoothing_max_iterations, 220),
        )
        rescued_xy = _enforce_monotonic_forward_x(rescued_xy)
        if rescued_xy.shape[0] == 0:
            return None
        rescued_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return rescued_xy

    def _rescue_candidate_local_path(
        self,
        candidate_local_path_xy: np.ndarray,
    ) -> np.ndarray | None:
        candidate_local_path_xy = np.asarray(candidate_local_path_xy, dtype=np.float64)
        if candidate_local_path_xy.shape[0] < 2:
            return None

        rescue_target_forward_x_m = max(
            self._min_publish_forward_span_m + 0.25,
            min(self._planning_horizon_m, self._path_min_forward_progress_m),
        )
        rescued_xy = _bridge_path_from_origin(
            candidate_local_path_xy,
            self._origin_bridge_point_count,
        )
        rescued_xy = _enforce_monotonic_forward_x(rescued_xy)
        rescued_xy = _extend_path_along_terminal_heading(
            rescued_xy,
            target_forward_span_m=rescue_target_forward_x_m,
            step_m=self._path_resample_step_m,
        )
        rescued_xy = _extend_path_forward(
            rescued_xy,
            target_forward_x_m=rescue_target_forward_x_m,
            step_m=self._path_resample_step_m,
        )
        rescued_xy = _truncate_polyline_length(rescued_xy, self._planning_horizon_m)
        rescued_xy = _deduplicate_polyline_xy(
            rescued_xy,
            min_segment_m=0.35 * self._path_resample_step_m,
        )
        rescued_xy = _resample_polyline_xy(rescued_xy, self._path_resample_step_m)
        rescued_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=rescued_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=max(self._path_smoothing_alpha, 0.22),
            max_iterations=max(self._path_smoothing_max_iterations, 220),
        )
        rescued_xy = _enforce_monotonic_forward_x(rescued_xy)
        if rescued_xy.shape[0] == 0:
            return None
        rescued_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return rescued_xy

    def _candidate_path_metrics(self, path_xy: np.ndarray) -> tuple[float, float, float]:
        path_xy = np.asarray(path_xy, dtype=np.float64)
        if path_xy.shape[0] < 2:
            return 0.0, 0.0, 0.0
        path_length_m = float(_polyline_length_m(path_xy))
        path_forward_span_m = _path_forward_span_m(path_xy)
        path_max_curvature_m_inv = (
            float(np.max(np.abs(_estimate_path_curvature(path_xy))))
            if path_xy.shape[0] >= 3
            else 0.0
        )
        return path_forward_span_m, path_length_m, path_max_curvature_m_inv

    def _select_publishable_local_path(
        self,
        *,
        candidate_local_path_xy: np.ndarray | None,
        candidate_source: str,
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> tuple[np.ndarray | None, str, str | None]:
        self._last_path_rejected = False
        self._last_path_rejection_reason = None

        if candidate_local_path_xy is None or candidate_local_path_xy.shape[0] < 2:
            self._last_candidate_path_forward_span_m = 0.0
            self._last_candidate_path_length_m = 0.0
            self._last_candidate_path_max_curvature_m_inv = 0.0
            rescue_xy = self._rescue_previous_local_path(rear_xy=rear_xy, yaw_rad=yaw_rad)
            if rescue_xy is None:
                return None, candidate_source, "empty_candidate"
            self._path_rescue_count += 1
            self._last_path_rejected = True
            self._last_path_rejection_reason = "empty_candidate"
            return rescue_xy, "rescue_previous_path", "empty_candidate"

        candidate_forward_span_m, candidate_length_m, candidate_max_curvature_m_inv = (
            self._candidate_path_metrics(candidate_local_path_xy)
        )
        self._last_candidate_path_forward_span_m = candidate_forward_span_m
        self._last_candidate_path_length_m = candidate_length_m
        self._last_candidate_path_max_curvature_m_inv = candidate_max_curvature_m_inv

        previous_local_xy = self._previous_local_path_xy(rear_xy, yaw_rad)
        previous_forward_span_m = (
            _path_forward_span_m(previous_local_xy)
            if previous_local_xy is not None and previous_local_xy.shape[0] >= 2
            else 0.0
        )
        forward_span_floor_m = max(
            self._min_publish_forward_span_m,
            min(self._path_min_forward_progress_m, 0.65 * previous_forward_span_m),
        )

        rejection_reason: str | None = None
        if candidate_forward_span_m < forward_span_floor_m:
            rejection_reason = "short_forward_span"

        if rejection_reason is None:
            return candidate_local_path_xy, candidate_source, None

        candidate_rescue_xy = self._rescue_candidate_local_path(candidate_local_path_xy)
        if candidate_rescue_xy is not None:
            candidate_rescue_forward_span_m, _, _ = self._candidate_path_metrics(candidate_rescue_xy)
            if candidate_rescue_forward_span_m >= self._min_publish_forward_span_m:
                self._path_rejection_count += 1
                self._path_rescue_count += 1
                self._last_path_rejected = True
                self._last_path_rejection_reason = rejection_reason
                return candidate_rescue_xy, "rescue_candidate_extension", rejection_reason

        rescue_xy = self._rescue_previous_local_path(rear_xy=rear_xy, yaw_rad=yaw_rad)
        if rescue_xy is not None:
            rescue_forward_span_m, _, _ = self._candidate_path_metrics(rescue_xy)
            if rescue_forward_span_m >= self._min_publish_forward_span_m:
                self._path_rejection_count += 1
                self._path_rescue_count += 1
                self._last_path_rejected = True
                self._last_path_rejection_reason = rejection_reason
                return rescue_xy, "rescue_previous_path", rejection_reason

        self._path_rejection_count += 1
        self._last_path_rejected = True
        self._last_path_rejection_reason = rejection_reason
        return None, candidate_source, rejection_reason

    def _build_local_path_from_centerline(
        self,
        centerline: _CorridorCenterline,
        *,
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> np.ndarray:
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
        rear_xy: np.ndarray,
        yaw_rad: float,
    ) -> tuple[np.ndarray, float] | None:
        if not self._scan_buffer:
            return None
        latest_scan = self._scan_buffer[-1].lidar_points_local_xy
        if latest_scan.shape[0] < 40:
            return None
        detection = detect_curve_window_points(
            latest_scan[:, 0],
            latest_scan[:, 1],
            config=self._curve_window_config,
        )
        if not detection.valid or detection.trajectory is None:
            return None
        local_path_lidar_xy = np.column_stack(
            [detection.trajectory.x_m, detection.trajectory.y_m]
        ).astype(np.float64)
        local_path_rear_xy = local_path_lidar_xy + self._rear_to_lidar_local.reshape(1, 2)
        local_path_rear_xy = self._stabilize_local_path(
            local_path_rear_xy,
            rear_xy=rear_xy,
            yaw_rad=yaw_rad,
        )
        if local_path_rear_xy.shape[0] == 0:
            return None
        local_path_rear_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
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

    def _build_path_message(
        self,
        *,
        world_path_xy: np.ndarray,
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> Path:
        path_msg = Path()
        path_msg.header.frame_id = self._odom_frame
        path_msg.header.stamp.sec = int(stamp_sec)
        path_msg.header.stamp.nanosec = int(stamp_nanosec)
        yaw_samples = _polyline_yaw(world_path_xy)
        from geometry_msgs.msg import PoseStamped  # local import keeps module import light

        for (x_m, y_m), yaw_rad in zip(world_path_xy, yaw_samples, strict=False):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x_m)
            pose.pose.position.y = float(y_m)
            pose.pose.position.z = 0.0
            qx, qy, qz, qw = _yaw_to_quat(float(yaw_rad))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)
        return path_msg

    def _build_route_message(self) -> Path | None:
        if len(self._route_points_world) < 2:
            return None
        route_xy = np.vstack(self._route_points_world).astype(np.float64)
        if self._loop_closed and self._start_pose is not None:
            start_xy = np.asarray(
                [self._start_pose["x_m"], self._start_pose["y_m"]],
                dtype=np.float64,
            )
            if float(np.linalg.norm(route_xy[-1] - start_xy)) >= self._route_point_spacing_m:
                route_xy = np.vstack([route_xy, start_xy.reshape(1, 2)])
        route_xy = _resample_polyline_xy(route_xy, self._route_resample_step_m)
        route_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=route_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._route_resample_step_m,
            smoothing_alpha=self._route_smoothing_alpha,
            max_iterations=self._route_smoothing_max_iterations,
        )
        route_xy = _resample_polyline_xy(route_xy, self._route_resample_step_m)
        if route_xy.shape[0] < 2:
            return None
        stamp_sec = int(self._latest_odom["stamp_s"]) if self._latest_odom is not None else 0
        stamp_nanosec = int(
            (float(self._latest_odom["stamp_s"]) - stamp_sec) * 1.0e9
        ) if self._latest_odom is not None else 0
        return self._build_path_message(
            world_path_xy=route_xy,
            stamp_sec=stamp_sec,
            stamp_nanosec=stamp_nanosec,
        )

    def _start_axis_metrics(self, rear_xy: np.ndarray) -> tuple[float, float]:
        if self._start_pose is None or self._start_axis_normal_xy is None or self._start_axis_tangent_xy is None:
            return 0.0, 0.0
        delta_xy = rear_xy - np.asarray(
            [self._start_pose["x_m"], self._start_pose["y_m"]],
            dtype=np.float64,
        )
        signed_distance_m = float(np.dot(delta_xy, self._start_axis_normal_xy))
        lateral_offset_m = float(np.dot(delta_xy, self._start_axis_tangent_xy))
        return signed_distance_m, lateral_offset_m

    def _set_terminal(self, cause: str) -> None:
        if self._mission_terminal:
            return
        self._mission_terminal = True
        self._terminal_cause = str(cause)
        self._loop_closed = cause == "loop_closed"
        self._last_route_msg = self._build_route_message()
        self._publish_outputs()

    def _plan_step(self) -> None:
        self._ensure_mission_started()
        now_monotonic = time.monotonic()
        if not self._mission_started:
            self._status_payload = {
                "state": "waiting_fusion" if not self._fusion_ready() else "waiting_odom",
                "ready": False,
                "loop_closure_armed": False,
                "loop_closed": False,
                "travel_distance_m": 0.0,
                "elapsed_s": 0.0,
                "local_path_age_s": None,
                "fusion_confidence": self._fusion_confidence(),
                "corridor_width_m": None,
                "route_point_count": len(self._route_points_world),
            }
            return

        rear_pose = self._rear_axle_pose()
        if rear_pose is None:
            self._status_payload = {
                "state": "waiting_odom",
                "ready": False,
                "loop_closure_armed": self._loop_closure_armed,
                "loop_closed": False,
                "travel_distance_m": self._travel_distance_m,
                "elapsed_s": max(0.0, now_monotonic - float(self._mission_start_monotonic or now_monotonic)),
                "local_path_age_s": None,
                "fusion_confidence": self._fusion_confidence(),
                "corridor_width_m": self._latest_corridor_width_m or None,
                "route_point_count": len(self._route_points_world),
            }
            return

        rear_xy, rear_yaw_rad = rear_pose
        elapsed_s = max(0.0, now_monotonic - float(self._mission_start_monotonic or now_monotonic))
        odom_age_s = (
            max(0.0, time.time() - float(self._latest_odom["stamp_s"]))
            if self._latest_odom is not None
            else None
        )
        current_axis_distance_m, current_axis_lateral_m = self._start_axis_metrics(rear_xy)
        self._loop_closure_armed = (
            self._travel_distance_m >= self._start_axis_arm_distance_m
            and elapsed_s >= self._start_axis_arm_time_s
        )
        if (
            not self._mission_terminal
            and self._loop_closure_armed
            and self._previous_axis_signed_distance_m is not None
            and self._previous_axis_signed_distance_m <= self._start_axis_crossing_neg_margin_m
            and current_axis_distance_m >= self._start_axis_crossing_pos_margin_m
            and abs(current_axis_lateral_m) <= self._start_axis_lateral_tolerance_m
        ):
            self._set_terminal("loop_closed")
        self._previous_axis_signed_distance_m = current_axis_distance_m

        if not self._mission_terminal and elapsed_s >= self._global_timeout_s:
            self._set_terminal("timeout")

        if self._latest_odom is None:
            now_stamp_s = 0.0
        else:
            now_stamp_s = float(self._latest_odom["stamp_s"])
        self._prune_scan_buffer(now_stamp_s)

        if self._mission_terminal:
            local_path_age_s = (
                max(0.0, now_monotonic - self._last_local_path_planned_monotonic)
                if self._last_local_path_planned_monotonic is not None
                else None
            )
            self._status_payload = {
                "state": self._terminal_cause,
                "ready": False,
                "loop_closure_armed": self._loop_closure_armed,
                "loop_closed": self._loop_closed,
                "travel_distance_m": self._travel_distance_m,
                "elapsed_s": elapsed_s,
                "start_pose": self._start_pose,
                "start_axis": {
                    "normal_xy": (
                        self._start_axis_normal_xy.tolist() if self._start_axis_normal_xy is not None else None
                    ),
                    "tangent_xy": (
                        self._start_axis_tangent_xy.tolist() if self._start_axis_tangent_xy is not None else None
                    ),
                    "signed_distance_m": current_axis_distance_m,
                    "lateral_offset_m": current_axis_lateral_m,
                },
                "local_path_age_s": local_path_age_s,
                "odom_age_s": odom_age_s,
                "fusion_confidence": self._fusion_confidence(),
                "corridor_width_m": self._latest_corridor_width_m or None,
                "route_point_count": len(self._route_points_world),
                "local_path_source": self._last_local_path_source,
                "terminal_cause": self._terminal_cause,
            }
            return

        rolling_points_xy = self._rolling_points_in_current_frame()
        planner_state = "tracking"
        ready = False
        local_path_xy: np.ndarray | None = None
        corridor_width_m: float | None = None

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
        if centerline is not None:
            local_path_xy = self._build_local_path_from_centerline(
                centerline,
                rear_xy=rear_xy,
                yaw_rad=rear_yaw_rad,
            )
            corridor_width_m = centerline.width_m
            planner_state = "tracking"
        else:
            fallback = self._build_fallback_curve_window_path(
                rear_xy=rear_xy,
                yaw_rad=rear_yaw_rad,
            )
            if fallback is not None:
                local_path_xy, corridor_width_m = fallback
                planner_state = "fallback_curve_window"
        candidate_source = planner_state

        publish_local_path_xy, publish_source, rejection_reason = self._select_publishable_local_path(
            candidate_local_path_xy=local_path_xy,
            candidate_source=candidate_source,
            rear_xy=rear_xy,
            yaw_rad=rear_yaw_rad,
        )

        if publish_local_path_xy is not None and publish_local_path_xy.shape[0] >= 2:
            world_path_xy = _transform_local_to_world(publish_local_path_xy, rear_xy, rear_yaw_rad)
            stamp_sec = int(self._latest_odom["stamp_s"])
            stamp_nanosec = int((float(self._latest_odom["stamp_s"]) - stamp_sec) * 1.0e9)
            self._last_local_path_msg = self._build_path_message(
                world_path_xy=world_path_xy,
                stamp_sec=stamp_sec,
                stamp_nanosec=stamp_nanosec,
            )
            self._last_local_path_world_xy = world_path_xy
            self._last_local_path_planned_monotonic = now_monotonic
            self._last_local_path_source = publish_source
            self._latest_corridor_width_m = float(corridor_width_m or 0.0)
            self._append_local_path_prefix(world_path_xy)
            self._local_path_pub.publish(self._last_local_path_msg)
            ready = True
            planner_state = publish_source
        else:
            local_path_age_s = (
                max(0.0, now_monotonic - self._last_local_path_planned_monotonic)
                if self._last_local_path_planned_monotonic is not None
                else None
            )
            if self._last_local_path_msg is not None and local_path_age_s is not None and local_path_age_s <= self._low_confidence_hold_path_s:
                planner_state = "holding_last_path"
                ready = True
            else:
                planner_state = (
                    "holding_last_path" if rejection_reason is not None else "waiting_local_path"
                )
                ready = False

        if len(self._route_points_world) >= 2:
            self._last_route_msg = self._build_route_message()
            if self._last_route_msg is not None:
                self._route_pub.publish(self._last_route_msg)

        local_path_age_s = (
            max(0.0, now_monotonic - self._last_local_path_planned_monotonic)
            if self._last_local_path_planned_monotonic is not None
            else None
        )
        path_forward_span_m = (
            _path_forward_span_m(publish_local_path_xy)
            if publish_local_path_xy is not None and publish_local_path_xy.shape[0] >= 2
            else 0.0
        )
        curvature = (
            float(np.max(np.abs(_estimate_path_curvature(self._last_local_path_world_xy))))
            if self._last_local_path_world_xy is not None and self._last_local_path_world_xy.shape[0] >= 3
            else 0.0
        )
        self._status_payload = {
            "state": planner_state,
            "ready": ready,
            "loop_closure_armed": self._loop_closure_armed,
            "loop_closed": False,
            "travel_distance_m": self._travel_distance_m,
            "elapsed_s": elapsed_s,
            "start_pose": self._start_pose,
            "start_axis": {
                "normal_xy": self._start_axis_normal_xy.tolist() if self._start_axis_normal_xy is not None else None,
                "tangent_xy": self._start_axis_tangent_xy.tolist() if self._start_axis_tangent_xy is not None else None,
                "signed_distance_m": current_axis_distance_m,
                "lateral_offset_m": current_axis_lateral_m,
            },
            "local_path_age_s": local_path_age_s,
            "odom_age_s": odom_age_s,
            "fusion_confidence": self._fusion_confidence(),
            "corridor_width_m": corridor_width_m if corridor_width_m is not None else (self._latest_corridor_width_m or None),
            "route_point_count": len(self._route_points_world),
            "local_path_source": self._last_local_path_source,
            "rolling_scan_count": len(self._scan_buffer),
            "rolling_point_count": int(rolling_points_xy.shape[0]),
            "candidate_path_source": candidate_source,
            "candidate_path_rejected": self._last_path_rejected,
            "candidate_path_rejection_reason": self._last_path_rejection_reason,
            "path_rejection_count": self._path_rejection_count,
            "path_rescue_count": self._path_rescue_count,
            "path_point_count": (
                int(self._last_local_path_world_xy.shape[0]) if self._last_local_path_world_xy is not None else 0
            ),
            "path_length_m": (
                float(_polyline_length_m(self._last_local_path_world_xy))
                if self._last_local_path_world_xy is not None
                else 0.0
            ),
            "path_forward_span_m": path_forward_span_m,
            "path_max_curvature_m_inv": curvature,
            "candidate_path_forward_span_m": self._last_candidate_path_forward_span_m,
            "candidate_path_length_m": self._last_candidate_path_length_m,
            "candidate_path_max_curvature_m_inv": self._last_candidate_path_max_curvature_m_inv,
        }

    def _publish_outputs(self) -> None:
        if self._last_local_path_msg is not None:
            self._local_path_pub.publish(self._last_local_path_msg)
        if self._last_route_msg is not None:
            self._route_pub.publish(self._last_route_msg)
        msg = String()
        msg.data = json.dumps(self._status_payload, separators=(",", ":"))
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = RecognitionTourPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
