#!/usr/bin/env python3
"""Plan one entry path to a visible curve from an initial static LiDAR snapshot."""

from __future__ import annotations

import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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

from .curve_window_detection import (
    CurveWindowDetectionConfig,
    curve_window_result_summary,
    detect_curve_window_points,
    scan_ranges_to_forward_left_xy,
)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


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


def _sanitize_ranges(ranges: list[float]) -> np.ndarray:
    arr = np.asarray(ranges, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    return arr


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


def _path_pointwise_deviation_m(
    reference_path_xy: np.ndarray,
    candidate_path_xy: np.ndarray,
    *,
    point_count: int,
) -> float:
    reference_eval_xy = _resample_polyline_xy_to_count(reference_path_xy, point_count)
    candidate_eval_xy = _resample_polyline_xy_to_count(candidate_path_xy, point_count)
    return float(np.max(np.linalg.norm(candidate_eval_xy - reference_eval_xy, axis=1)))


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


def _enforce_terminal_heading(
    *,
    path_xy: np.ndarray,
    terminal_heading_rad: float,
    tail_length_m: float,
) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64).copy()
    if path_xy.shape[0] < 3 or tail_length_m <= 1.0e-6:
        return path_xy

    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length_m = float(cumulative_s[-1])
    if total_length_m <= 1.0e-9:
        return path_xy

    tail_length_m = min(float(tail_length_m), total_length_m)
    tail_start_s = total_length_m - tail_length_m
    start_index = int(np.searchsorted(cumulative_s, tail_start_s, side="left"))
    start_index = min(max(1, start_index), path_xy.shape[0] - 2)

    tail_point_count = path_xy.shape[0] - start_index
    if tail_point_count <= 1:
        return path_xy

    terminal_dir_xy = np.asarray(
        [math.cos(terminal_heading_rad), math.sin(terminal_heading_rad)],
        dtype=np.float64,
    )
    start_point_xy = path_xy[start_index].copy()
    end_point_xy = path_xy[-1].copy()
    if start_index <= 0:
        start_tangent_xy = path_xy[1] - path_xy[0]
    elif start_index >= (path_xy.shape[0] - 1):
        start_tangent_xy = path_xy[-1] - path_xy[-2]
    else:
        start_tangent_xy = path_xy[start_index + 1] - path_xy[start_index - 1]
    start_tangent_norm = float(np.linalg.norm(start_tangent_xy))
    if start_tangent_norm <= 1.0e-9:
        start_tangent_xy = end_point_xy - start_point_xy
        start_tangent_norm = float(np.linalg.norm(start_tangent_xy))
    if start_tangent_norm <= 1.0e-9:
        return path_xy

    start_dir_xy = start_tangent_xy / start_tangent_norm
    tail_chord_m = float(np.linalg.norm(end_point_xy - start_point_xy))
    control_length_m = max(
        0.10,
        min(
            0.45 * tail_length_m,
            max(0.12, 0.55 * max(tail_chord_m, 0.10)),
        ),
    )
    p0_xy = start_point_xy
    p1_xy = start_point_xy + (control_length_m * start_dir_xy)
    p2_xy = end_point_xy - (control_length_m * terminal_dir_xy)
    p3_xy = end_point_xy
    ts = np.linspace(0.0, 1.0, tail_point_count, dtype=np.float64)
    one_minus_t = 1.0 - ts
    bezier_tail_xy = (
        ((one_minus_t ** 3).reshape(-1, 1) * p0_xy.reshape(1, 2))
        + (3.0 * (one_minus_t ** 2) * ts).reshape(-1, 1) * p1_xy.reshape(1, 2)
        + (3.0 * one_minus_t * (ts ** 2)).reshape(-1, 1) * p2_xy.reshape(1, 2)
        + ((ts ** 3).reshape(-1, 1) * p3_xy.reshape(1, 2))
    )
    if tail_point_count >= 2:
        final_seg_length_m = max(
            1.0e-3,
            float(np.linalg.norm(bezier_tail_xy[-1] - bezier_tail_xy[-2])),
        )
        bezier_tail_xy[-2] = end_point_xy - (final_seg_length_m * terminal_dir_xy)
    path_xy[start_index:] = bezier_tail_xy
    path_xy[-1] = end_point_xy
    return path_xy


def _enforce_terminal_heading_with_constraints(
    *,
    path_xy: np.ndarray,
    terminal_heading_rad: float,
    min_tail_length_m: float,
    max_curvature_m_inv: float,
    resample_step_m: float,
    smoothing_alpha: float,
    smoothing_max_iterations: int,
    max_tail_fraction: float,
    max_path_deviation_m: float,
    max_attempts: int = 10,
) -> tuple[np.ndarray, float, float, bool]:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] < 3:
        curvature = np.abs(_estimate_path_curvature(path_xy))
        max_curvature = float(np.max(curvature)) if curvature.size else 0.0
        return path_xy.copy(), max_curvature, 0.0, True

    total_length_m = _polyline_length_m(path_xy)
    if total_length_m <= 1.0e-9:
        curvature = np.abs(_estimate_path_curvature(path_xy))
        max_curvature = float(np.max(curvature)) if curvature.size else 0.0
        return path_xy.copy(), max_curvature, 0.0, True

    max_tail_length_m = min(
        total_length_m,
        max(4.0 * float(resample_step_m), float(max_tail_fraction) * total_length_m),
    )
    tail_length_m = min(
        max_tail_length_m,
        max(float(min_tail_length_m), 4.0 * float(resample_step_m)),
    )
    eval_point_count = max(
        path_xy.shape[0],
        int(math.ceil(total_length_m / max(1.0e-3, float(resample_step_m)))) + 1,
    )
    reference_path_xy = path_xy.copy()
    rescue_smoothing_alpha = max(0.30, float(smoothing_alpha))
    rescue_iterations = max(180, 2 * int(smoothing_max_iterations))

    best_path_xy = reference_path_xy.copy()
    best_tail_length_m = 0.0
    best_path_deviation_m = 0.0
    base_curvature = np.abs(_estimate_path_curvature(reference_path_xy))
    best_max_curvature = float(np.max(base_curvature)) if base_curvature.size else 0.0
    best_score = (
        max(0.0, best_max_curvature - float(max_curvature_m_inv)),
        0.0,
        best_max_curvature,
    )
    current_path_xy = reference_path_xy.copy()

    for _ in range(max(1, int(max_attempts))):
        candidate_path_xy = _enforce_terminal_heading(
            path_xy=current_path_xy,
            terminal_heading_rad=terminal_heading_rad,
            tail_length_m=tail_length_m,
        )
        candidate_curvature = np.abs(_estimate_path_curvature(candidate_path_xy))
        candidate_max_curvature = (
            float(np.max(candidate_curvature)) if candidate_curvature.size else 0.0
        )
        candidate_path_deviation_m = _path_pointwise_deviation_m(
            reference_path_xy,
            candidate_path_xy,
            point_count=eval_point_count,
        )
        candidate_score = (
            max(0.0, candidate_path_deviation_m - float(max_path_deviation_m)),
            max(0.0, candidate_max_curvature - float(max_curvature_m_inv)),
            candidate_path_deviation_m,
        )
        if candidate_score < best_score:
            best_path_xy = candidate_path_xy.copy()
            best_tail_length_m = tail_length_m
            best_path_deviation_m = candidate_path_deviation_m
            best_max_curvature = candidate_max_curvature
            best_score = candidate_score
        if (
            candidate_path_deviation_m <= float(max_path_deviation_m)
            and candidate_max_curvature <= float(max_curvature_m_inv)
        ):
            return candidate_path_xy, candidate_max_curvature, tail_length_m, True

        current_path_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=candidate_path_xy,
            max_curvature_m_inv=max_curvature_m_inv,
            resample_step_m=resample_step_m,
            smoothing_alpha=rescue_smoothing_alpha,
            max_iterations=rescue_iterations,
        )
        if tail_length_m >= (max_tail_length_m - 1.0e-9):
            continue
        tail_length_m = min(
            max_tail_length_m,
            tail_length_m + max(0.04, 0.35 * (max_tail_length_m - tail_length_m)),
        )

    return (
        best_path_xy,
        best_max_curvature,
        best_tail_length_m,
        (
            best_path_deviation_m <= float(max_path_deviation_m)
            and best_max_curvature <= float(max_curvature_m_inv)
        ),
    )


class CurveEntryPathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("curve_entry_path_planner_node")

        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("fusion_status_topic", "/apex/estimation/status")
        self.declare_parameter("path_topic", "/apex/planning/curve_entry_path")
        self.declare_parameter("target_topic", "/apex/planning/curve_entry_target")
        self.declare_parameter("status_topic", "/apex/planning/curve_entry_status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("snapshot_scan_count", 8)
        self.declare_parameter("stationary_speed_threshold_mps", 0.05)
        self.declare_parameter("stationary_yaw_rate_threshold_rps", 0.05)
        self.declare_parameter("stationary_hold_s", 0.6)
        self.declare_parameter("lidar_offset_x_m", 0.18)
        self.declare_parameter("lidar_offset_y_m", 0.0)
        self.declare_parameter("tracking_origin_offset_x_m", -0.15)
        self.declare_parameter("tracking_origin_offset_y_m", 0.0)
        self.declare_parameter("origin_bridge_point_count", 12)
        self.declare_parameter("planning_wheelbase_m", 0.30)
        self.declare_parameter("planning_max_steering_deg", 18.0)
        self.declare_parameter("path_curvature_limit_scale", 0.75)
        self.declare_parameter("path_resample_step_m", 0.04)
        self.declare_parameter("path_smoothing_alpha", 0.22)
        self.declare_parameter("path_smoothing_max_iterations", 120)
        self.declare_parameter("terminal_heading_tail_length_m", 0.22)
        self.declare_parameter("terminal_heading_max_tail_fraction", 0.45)
        self.declare_parameter("terminal_heading_max_path_deviation_m", 0.28)
        self.declare_parameter("second_corridor_target_depth_m", 0.12)
        self.declare_parameter("second_corridor_target_depth_min_m", 0.10)
        self.declare_parameter("second_corridor_target_depth_max_m", 0.15)
        self.declare_parameter("inner_vertex_clearance_m", 0.12)
        self.declare_parameter("curve_apex_width_fraction", 0.40)
        self.declare_parameter("status_publish_rate_hz", 2.0)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._fusion_status_topic = str(self.get_parameter("fusion_status_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._target_topic = str(self.get_parameter("target_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._snapshot_scan_count = max(3, int(self.get_parameter("snapshot_scan_count").value))
        self._stationary_speed_threshold = max(
            0.0, float(self.get_parameter("stationary_speed_threshold_mps").value)
        )
        self._stationary_yaw_rate_threshold = max(
            0.0, float(self.get_parameter("stationary_yaw_rate_threshold_rps").value)
        )
        self._stationary_hold_s = max(0.1, float(self.get_parameter("stationary_hold_s").value))
        self._lidar_offset = np.asarray(
            [
                float(self.get_parameter("lidar_offset_x_m").value),
                float(self.get_parameter("lidar_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._tracking_origin_offset = np.asarray(
            [
                float(self.get_parameter("tracking_origin_offset_x_m").value),
                float(self.get_parameter("tracking_origin_offset_y_m").value),
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
        self._terminal_heading_tail_length_m = max(
            0.05, float(self.get_parameter("terminal_heading_tail_length_m").value)
        )
        self._terminal_heading_max_tail_fraction = max(
            0.15,
            min(0.95, float(self.get_parameter("terminal_heading_max_tail_fraction").value)),
        )
        self._terminal_heading_max_path_deviation_m = max(
            0.05, float(self.get_parameter("terminal_heading_max_path_deviation_m").value)
        )
        self._detection_config = CurveWindowDetectionConfig(
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
        status_rate_hz = max(0.5, float(self.get_parameter("status_publish_rate_hz").value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)
        self.create_subscription(String, self._fusion_status_topic, self._fusion_status_cb, 20)

        self._path_pub = self.create_publisher(Path, self._path_topic, latched_qos)
        self._target_pub = self.create_publisher(PoseStamped, self._target_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)
        self.create_timer(1.0 / status_rate_hz, self._publish_status)

        self._latest_odom: dict[str, float] | None = None
        self._latest_fusion_status: dict[str, object] = {}
        self._static_since_monotonic: float | None = None
        self._scan_window: deque[np.ndarray] = deque(maxlen=self._snapshot_scan_count)
        self._path_msg: Path | None = None
        self._target_msg: PoseStamped | None = None
        self._status_payload: dict[str, object] = {
            "state": "waiting_fusion",
            "ready": False,
            "reason": "fusion_not_ready",
        }
        self._planned = False
        self._terminal = False

        self.get_logger().info(
            "CurveEntryPathPlannerNode started (scan=%s odom=%s fusion=%s path=%s)"
            % (
                self._scan_topic,
                self._odom_topic,
                self._fusion_status_topic,
                self._path_topic,
            )
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = {
            "stamp_sec": float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
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

    def _fusion_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._latest_fusion_status = payload

    def _fusion_ready(self) -> bool:
        return bool(
            self._latest_fusion_status.get("alignment_ready", False)
            and str(self._latest_fusion_status.get("state", "")) == "tracking"
        )

    def _stationary_now(self) -> bool:
        latest_pose = self._latest_fusion_status.get("latest_pose") or {}
        if latest_pose:
            vx_mps = float(latest_pose.get("vx_mps") or 0.0)
            vy_mps = float(latest_pose.get("vy_mps") or 0.0)
            yaw_rate_rps = float(latest_pose.get("yaw_rate_rps") or 0.0)
        elif self._latest_odom is not None:
            vx_mps = float(self._latest_odom["vx_mps"])
            vy_mps = float(self._latest_odom["vy_mps"])
            yaw_rate_rps = float(self._latest_odom["yaw_rate_rps"])
        else:
            return False
        speed_mps = math.hypot(vx_mps, vy_mps)
        return (
            speed_mps <= self._stationary_speed_threshold
            and abs(yaw_rate_rps) <= self._stationary_yaw_rate_threshold
        )

    def _publish_status(self) -> None:
        if self._path_msg is not None:
            self._path_pub.publish(self._path_msg)
        if self._target_msg is not None:
            self._target_pub.publish(self._target_msg)
        msg = String()
        msg.data = json.dumps(self._status_payload, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _fail(self, reason: str, extra: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "state": reason,
            "ready": False,
            "reason": reason,
        }
        if extra:
            payload.update(extra)
        self._status_payload = payload
        self._terminal = True
        self.get_logger().warn(f"Curve entry planning failed: {reason}")
        self._publish_status()

    def _build_world_path(
        self,
        *,
        stamp_sec: int,
        stamp_nanosec: int,
        local_path_xy: np.ndarray,
        base_pose_xy: np.ndarray,
        base_yaw_rad: float,
        terminal_heading_rad: float | None = None,
        terminal_heading_tail_length_m: float = 0.0,
    ) -> tuple[Path, PoseStamped, np.ndarray, dict[str, float]]:
        lidar_origin_xy = base_pose_xy + (_rotation(base_yaw_rad) @ self._lidar_offset)
        world_path_xy = (local_path_xy @ _rotation(base_yaw_rad).T) + lidar_origin_xy
        tracking_origin_xy = base_pose_xy + (_rotation(base_yaw_rad) @ self._tracking_origin_offset)

        if world_path_xy.shape[0] >= 1:
            path_start_xy = world_path_xy[0]
            if world_path_xy.shape[0] >= 2:
                first_path_yaw = math.atan2(
                    float(world_path_xy[1, 1] - world_path_xy[0, 1]),
                    float(world_path_xy[1, 0] - world_path_xy[0, 0]),
                )
            else:
                first_path_yaw = base_yaw_rad
            start_to_path_distance_m = float(np.linalg.norm(path_start_xy - tracking_origin_xy))
            if start_to_path_distance_m > 1.0e-6:
                tangent_length_m = max(0.08, min(0.40, 0.35 * start_to_path_distance_m))
                start_dir = np.asarray(
                    [math.cos(base_yaw_rad), math.sin(base_yaw_rad)],
                    dtype=np.float64,
                )
                end_dir = np.asarray(
                    [math.cos(first_path_yaw), math.sin(first_path_yaw)],
                    dtype=np.float64,
                )
                connector_xy = _cubic_bezier_xy(
                    p0_xy=tracking_origin_xy,
                    p1_xy=tracking_origin_xy + (tangent_length_m * start_dir),
                    p2_xy=path_start_xy - (tangent_length_m * end_dir),
                    p3_xy=path_start_xy,
                    point_count=self._origin_bridge_point_count,
                )
                world_path_xy = np.vstack([connector_xy, world_path_xy])
            else:
                world_path_xy[0] = tracking_origin_xy
        else:
            world_path_xy = tracking_origin_xy.reshape(1, 2)

        world_path_xy, path_max_curvature_m_inv = _smooth_path_to_curvature_limit(
            path_xy=world_path_xy,
            max_curvature_m_inv=self._max_path_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=self._path_smoothing_alpha,
            max_iterations=self._path_smoothing_max_iterations,
        )
        terminal_heading_tail_length_m_applied = 0.0
        path_curvature_limit_respected = True
        if terminal_heading_rad is not None:
            (
                world_path_xy,
                path_max_curvature_m_inv,
                terminal_heading_tail_length_m_applied,
                path_curvature_limit_respected,
            ) = _enforce_terminal_heading_with_constraints(
                path_xy=world_path_xy,
                terminal_heading_rad=float(terminal_heading_rad),
                min_tail_length_m=float(terminal_heading_tail_length_m),
                max_curvature_m_inv=self._max_path_curvature_m_inv,
                resample_step_m=self._path_resample_step_m,
                smoothing_alpha=self._path_smoothing_alpha,
                smoothing_max_iterations=self._path_smoothing_max_iterations,
                max_tail_fraction=self._terminal_heading_max_tail_fraction,
                max_path_deviation_m=self._terminal_heading_max_path_deviation_m,
            )
        else:
            actual_curvature = np.abs(_estimate_path_curvature(world_path_xy))
            path_max_curvature_m_inv = (
                float(np.max(actual_curvature)) if actual_curvature.size else 0.0
            )

        path_msg = Path()
        path_msg.header.frame_id = self._odom_frame
        path_msg.header.stamp.sec = int(stamp_sec)
        path_msg.header.stamp.nanosec = int(stamp_nanosec)

        yaw_samples: list[float] = []
        if world_path_xy.shape[0] >= 2:
            for index in range(world_path_xy.shape[0]):
                if index < (world_path_xy.shape[0] - 1):
                    dx = float(world_path_xy[index + 1, 0] - world_path_xy[index, 0])
                    dy = float(world_path_xy[index + 1, 1] - world_path_xy[index, 1])
                else:
                    dx = float(world_path_xy[index, 0] - world_path_xy[index - 1, 0])
                    dy = float(world_path_xy[index, 1] - world_path_xy[index - 1, 1])
                yaw_samples.append(math.atan2(dy, dx))
        else:
            yaw_samples = [base_yaw_rad]

        for index, (x_m, y_m) in enumerate(world_path_xy):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x_m)
            pose.pose.position.y = float(y_m)
            pose.pose.position.z = 0.0
            qx, qy, qz, qw = _yaw_to_quat(yaw_samples[min(index, len(yaw_samples) - 1)])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)

        target_msg = PoseStamped()
        target_msg.header = path_msg.header
        target_msg.pose = path_msg.poses[-1].pose
        path_metrics = {
            "path_max_curvature_m_inv": float(path_max_curvature_m_inv),
            "path_min_turn_radius_m": (
                float(1.0 / path_max_curvature_m_inv)
                if path_max_curvature_m_inv > 1.0e-9
                else float("inf")
            ),
            "path_curvature_limit_m_inv": float(self._max_path_curvature_m_inv),
            "path_curvature_limit_respected": bool(path_curvature_limit_respected),
            "terminal_heading_tail_length_m_applied": float(
                terminal_heading_tail_length_m_applied
            ),
        }
        return path_msg, target_msg, tracking_origin_xy, path_metrics

    def _plan_from_snapshot(self, msg: LaserScan) -> None:
        if self._latest_odom is None:
            self._fail("missing_odom")
            return

        scan_stack = np.asarray(list(self._scan_window), dtype=np.float64)
        snapshot_ranges = np.nanmedian(scan_stack, axis=0)
        snapshot_ranges[~np.isfinite(snapshot_ranges)] = 0.0

        points_x_m, points_y_m, _ = scan_ranges_to_forward_left_xy(snapshot_ranges)
        detection = detect_curve_window_points(
            points_x_m,
            points_y_m,
            config=self._detection_config,
        )
        if not detection.valid or detection.trajectory is None:
            self._fail(
                "no_curve_detected",
                {
                    "snapshot_scan_count": len(self._scan_window),
                },
            )
            return

        local_path_xy = np.column_stack([detection.trajectory.x_m, detection.trajectory.y_m])
        planner_pose_xy = np.asarray(
            [self._latest_odom["x_m"], self._latest_odom["y_m"]],
            dtype=np.float64,
        )
        planner_yaw_rad = float(self._latest_odom["yaw_rad"])
        world_rot = _rotation(planner_yaw_rad)
        lidar_origin_xy = planner_pose_xy + (world_rot @ self._lidar_offset)
        entry_line_center_local_xy = np.asarray(
            [
                detection.trajectory.entry_line_center_x_m,
                detection.trajectory.entry_line_center_y_m,
            ],
            dtype=np.float64,
        )
        entry_line_center_world_xy = (world_rot @ entry_line_center_local_xy) + lidar_origin_xy
        second_corridor_axis_world_xy = world_rot @ np.asarray(
            [
                detection.trajectory.second_corridor_axis_x,
                detection.trajectory.second_corridor_axis_y,
            ],
            dtype=np.float64,
        )
        path_msg, target_msg, tracking_origin_xy, path_metrics = self._build_world_path(
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
            local_path_xy=local_path_xy,
            base_pose_xy=planner_pose_xy,
            base_yaw_rad=planner_yaw_rad,
            terminal_heading_rad=float(
                math.atan2(
                    second_corridor_axis_world_xy[1],
                    second_corridor_axis_world_xy[0],
                )
            ),
            terminal_heading_tail_length_m=max(
                self._terminal_heading_tail_length_m,
                detection.trajectory.target_depth_into_second_corridor_m + 0.04,
            ),
        )

        summary = curve_window_result_summary(detection)
        summary.pop("path_xy_m", None)
        summary.pop("anchor_points_xy_m", None)

        self._path_msg = path_msg
        self._target_msg = target_msg
        self._planned = True
        self._terminal = True
        target_yaw_rad = _quat_to_yaw(
            target_msg.pose.orientation.x,
            target_msg.pose.orientation.y,
            target_msg.pose.orientation.z,
            target_msg.pose.orientation.w,
        )
        self._status_payload = {
            "state": "ready",
            "ready": True,
            "reason": "curve_path_planned",
            "plan_frame_id": self._odom_frame,
            "snapshot_scan_count": len(self._scan_window),
            "path_point_count": len(path_msg.poses),
            "planner_pose": {
                "x_m": float(self._latest_odom["x_m"]),
                "y_m": float(self._latest_odom["y_m"]),
                "yaw_rad": planner_yaw_rad,
            },
            "tracking_origin_pose": {
                "x_m": float(tracking_origin_xy[0]),
                "y_m": float(tracking_origin_xy[1]),
                "yaw_rad": planner_yaw_rad,
            },
            "path_start_pose": {
                "x_m": float(path_msg.poses[0].pose.position.x),
                "y_m": float(path_msg.poses[0].pose.position.y),
                "yaw_rad": _quat_to_yaw(
                    path_msg.poses[0].pose.orientation.x,
                    path_msg.poses[0].pose.orientation.y,
                    path_msg.poses[0].pose.orientation.z,
                    path_msg.poses[0].pose.orientation.w,
                ),
            },
            "target_pose": {
                "x_m": float(target_msg.pose.position.x),
                "y_m": float(target_msg.pose.position.y),
                "yaw_rad": float(target_yaw_rad),
            },
            "entry_line_center_pose": {
                "x_m": float(entry_line_center_world_xy[0]),
                "y_m": float(entry_line_center_world_xy[1]),
                "yaw_rad": float(
                    math.atan2(
                        second_corridor_axis_world_xy[1],
                        second_corridor_axis_world_xy[0],
                    )
                ),
            },
            "second_corridor_axis_xy": {
                "x": float(second_corridor_axis_world_xy[0]),
                "y": float(second_corridor_axis_world_xy[1]),
            },
            "target_depth_into_second_corridor_m": float(
                detection.trajectory.target_depth_into_second_corridor_m
            ),
            "target_clearance_inner_m": float(
                detection.trajectory.target_clearance_inner_m
            ),
            "target_clearance_outer_m": float(
                detection.trajectory.target_clearance_outer_m
            ),
            "curve_summary": summary,
            "path_metrics": path_metrics,
            "local_lidar_target": {
                "x_m": float(detection.trajectory.target_x_m),
                "y_m": float(detection.trajectory.target_y_m),
            },
        }
        self.get_logger().info(
            "Curve entry path planned (points=%d target=(%.3f, %.3f))"
            % (
                len(path_msg.poses),
                float(target_msg.pose.position.x),
                float(target_msg.pose.position.y),
            )
        )
        self._publish_status()

    def _scan_cb(self, msg: LaserScan) -> None:
        if self._terminal:
            return

        if not self._fusion_ready():
            self._scan_window.clear()
            self._static_since_monotonic = None
            self._status_payload = {
                "state": "waiting_fusion",
                "ready": False,
                "reason": "fusion_not_ready",
            }
            return

        if self._latest_odom is None:
            self._status_payload = {
                "state": "waiting_odom",
                "ready": False,
                "reason": "odom_not_available",
            }
            return

        now_monotonic = time.monotonic()
        if not self._stationary_now():
            self._scan_window.clear()
            self._static_since_monotonic = None
            self._status_payload = {
                "state": "waiting_static",
                "ready": False,
                "reason": "vehicle_not_static",
            }
            return

        if self._static_since_monotonic is None:
            self._static_since_monotonic = now_monotonic

        held_s = now_monotonic - self._static_since_monotonic
        if held_s < self._stationary_hold_s:
            self._scan_window.clear()
            self._status_payload = {
                "state": "waiting_static",
                "ready": False,
                "reason": "static_hold_in_progress",
                "static_hold_elapsed_s": held_s,
                "static_hold_required_s": self._stationary_hold_s,
            }
            return

        self._scan_window.append(_sanitize_ranges(list(msg.ranges)))
        if len(self._scan_window) < self._snapshot_scan_count:
            self._status_payload = {
                "state": "collecting_snapshot",
                "ready": False,
                "reason": "snapshot_in_progress",
                "snapshot_scan_count": len(self._scan_window),
                "snapshot_scan_count_required": self._snapshot_scan_count,
            }
            return

        self._plan_from_snapshot(msg)


def main() -> None:
    rclpy.init()
    node = CurveEntryPathPlannerNode()
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
