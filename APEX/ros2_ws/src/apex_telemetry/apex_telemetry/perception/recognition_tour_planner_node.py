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


def _compute_path_s(path_xy: np.ndarray) -> np.ndarray:
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
        rear_xy, _ = rear_pose
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

    def _build_local_path_from_centerline(self, centerline: _CorridorCenterline) -> np.ndarray:
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
        local_path_xy = _truncate_polyline_length(local_path_xy, self._planning_horizon_m)
        local_path_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=local_path_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=self._path_smoothing_alpha,
            max_iterations=self._path_smoothing_max_iterations,
        )
        local_path_xy = _truncate_polyline_length(local_path_xy, self._planning_horizon_m)
        if local_path_xy.shape[0] > 0:
            local_path_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        return local_path_xy

    def _build_fallback_curve_window_path(self) -> tuple[np.ndarray, float] | None:
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
        local_path_rear_xy = _truncate_polyline_length(local_path_rear_xy, self._planning_horizon_m)
        local_path_rear_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=local_path_rear_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=self._path_smoothing_alpha,
            max_iterations=self._path_smoothing_max_iterations,
        )
        if local_path_rear_xy.shape[0] == 0:
            return None
        local_path_rear_xy[0] = np.asarray([0.0, 0.0], dtype=np.float64)
        corridor_width_m = max(
            self._corridor_min_width_m,
            min(self._corridor_max_width_m, float(detection.summary.get("window_width_m", 0.0) or 0.0)),
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

        rear_xy, _ = rear_pose
        elapsed_s = max(0.0, now_monotonic - float(self._mission_start_monotonic or now_monotonic))
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
            local_path_xy = self._build_local_path_from_centerline(centerline)
            corridor_width_m = centerline.width_m
            planner_state = "tracking"
        else:
            fallback = self._build_fallback_curve_window_path()
            if fallback is not None:
                local_path_xy, corridor_width_m = fallback
                planner_state = "fallback_curve_window"

        if local_path_xy is not None and local_path_xy.shape[0] >= 2:
            world_path_xy = _transform_local_to_world(local_path_xy, rear_xy, float(self._latest_odom["yaw_rad"]))
            stamp_sec = int(self._latest_odom["stamp_s"])
            stamp_nanosec = int((float(self._latest_odom["stamp_s"]) - stamp_sec) * 1.0e9)
            self._last_local_path_msg = self._build_path_message(
                world_path_xy=world_path_xy,
                stamp_sec=stamp_sec,
                stamp_nanosec=stamp_nanosec,
            )
            self._last_local_path_world_xy = world_path_xy
            self._last_local_path_planned_monotonic = now_monotonic
            self._last_local_path_source = planner_state
            self._latest_corridor_width_m = float(corridor_width_m or 0.0)
            self._append_local_path_prefix(world_path_xy)
            ready = True
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
                planner_state = "waiting_local_path"
                ready = False

        if len(self._route_points_world) >= 2:
            self._last_route_msg = self._build_route_message()

        local_path_age_s = (
            max(0.0, now_monotonic - self._last_local_path_planned_monotonic)
            if self._last_local_path_planned_monotonic is not None
            else None
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
            "fusion_confidence": self._fusion_confidence(),
            "corridor_width_m": corridor_width_m if corridor_width_m is not None else (self._latest_corridor_width_m or None),
            "route_point_count": len(self._route_points_world),
            "local_path_source": self._last_local_path_source,
            "rolling_scan_count": len(self._scan_buffer),
            "rolling_point_count": int(rolling_points_xy.shape[0]),
            "path_point_count": (
                int(self._last_local_path_world_xy.shape[0]) if self._last_local_path_world_xy is not None else 0
            ),
            "path_length_m": (
                float(_polyline_length_m(self._last_local_path_world_xy))
                if self._last_local_path_world_xy is not None
                else 0.0
            ),
            "path_max_curvature_m_inv": curvature,
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
