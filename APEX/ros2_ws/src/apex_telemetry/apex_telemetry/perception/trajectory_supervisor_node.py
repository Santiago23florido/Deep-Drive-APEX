#!/usr/bin/env python3
"""Local obstacle-avoidance supervisor for a fixed global route."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
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
    _estimate_path_curvature,
    _resample_polyline_xy,
    _rotation,
    _sanitize_ranges,
    _smooth_path_to_curvature_limit,
    _yaw_to_quat,
)
from .curve_window_detection import scan_ranges_to_forward_left_xy


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _polyline_s(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] <= 1:
        return np.zeros((path_xy.shape[0],), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])


def _polyline_length_m(path_xy: np.ndarray) -> float:
    path_s = _polyline_s(path_xy)
    return float(path_s[-1]) if path_s.size else 0.0


def _path_yaw(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if path_xy.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if path_xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    diffs = np.diff(path_xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.concatenate([headings, [headings[-1]]])


def _interp_path_xy(path_xy: np.ndarray, path_s: np.ndarray, sample_s: np.ndarray) -> np.ndarray:
    xs = np.interp(sample_s, path_s, path_xy[:, 0])
    ys = np.interp(sample_s, path_s, path_xy[:, 1])
    return np.column_stack([xs, ys])


def _interp_path_yaw(path_yaw: np.ndarray, path_s: np.ndarray, sample_s: np.ndarray) -> np.ndarray:
    if path_yaw.size == 0:
        return np.zeros_like(sample_s, dtype=np.float64)
    return np.interp(sample_s, path_s, np.unwrap(path_yaw))


def _transform_local_to_world(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) @ _rotation(yaw_rad).T) + origin_xy.reshape(1, 2)


def _transform_world_to_local(points_xy: np.ndarray, origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    return (np.asarray(points_xy, dtype=np.float64) - origin_xy.reshape(1, 2)) @ _rotation(yaw_rad)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return values * values * (3.0 - (2.0 * values))


@dataclass
class PathProjection:
    segment_index: int
    projection_xy: np.ndarray
    projection_s_m: float
    distance_m: float
    heading_rad: float
    normal_left_xy: np.ndarray
    signed_lateral_error_m: float


@dataclass
class CandidatePath:
    offset_m: float
    path_xy: np.ndarray
    score: float
    feasible: bool
    reason: str
    min_clearance_m: float
    max_curvature_m_inv: float


def _project_onto_polyline(
    point_xy: np.ndarray,
    path_xy: np.ndarray,
    path_s: np.ndarray,
) -> PathProjection | None:
    if path_xy.shape[0] < 2 or path_s.shape[0] != path_xy.shape[0]:
        return None

    seg_start_xy = path_xy[:-1]
    seg_end_xy = path_xy[1:]
    seg_vec_xy = seg_end_xy - seg_start_xy
    seg_len_sq = np.sum(seg_vec_xy * seg_vec_xy, axis=1)
    valid_mask = seg_len_sq > 1.0e-9
    if not np.any(valid_mask):
        return None

    rel_xy = point_xy.reshape(1, 2) - seg_start_xy
    alpha = np.zeros_like(seg_len_sq, dtype=np.float64)
    alpha[valid_mask] = np.clip(
        np.sum(rel_xy[valid_mask] * seg_vec_xy[valid_mask], axis=1) / seg_len_sq[valid_mask],
        0.0,
        1.0,
    )
    projection_xy = seg_start_xy + (alpha.reshape(-1, 1) * seg_vec_xy)
    distances = np.linalg.norm(projection_xy - point_xy.reshape(1, 2), axis=1)
    index = int(np.argmin(distances))
    heading_rad = math.atan2(float(seg_vec_xy[index, 1]), float(seg_vec_xy[index, 0]))
    segment_length_m = math.sqrt(max(1.0e-12, float(seg_len_sq[index])))
    projection_s_m = float(path_s[index]) + (float(alpha[index]) * segment_length_m)
    normal_left_xy = np.asarray(
        [-math.sin(heading_rad), math.cos(heading_rad)],
        dtype=np.float64,
    )
    signed_lateral_error_m = float(np.dot(point_xy - projection_xy[index], normal_left_xy))
    return PathProjection(
        segment_index=index,
        projection_xy=projection_xy[index].copy(),
        projection_s_m=projection_s_m,
        distance_m=float(distances[index]),
        heading_rad=heading_rad,
        normal_left_xy=normal_left_xy,
        signed_lateral_error_m=signed_lateral_error_m,
    )


def _parse_offsets(raw_value: object, fallback_count: int, max_width_m: float) -> list[float]:
    values: list[float] = []
    if isinstance(raw_value, str):
        for token in raw_value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    else:
        try:
            values = [float(value) for value in raw_value]  # type: ignore[arg-type]
        except TypeError:
            values = []
        except ValueError:
            values = []

    if not values:
        count = max(3, int(fallback_count))
        half_width = max(0.10, 0.5 * float(max_width_m))
        values = np.linspace(-half_width, half_width, count, dtype=np.float64).tolist()

    if not any(abs(value) <= 1.0e-9 for value in values):
        values.append(0.0)

    values = sorted(set(round(float(value), 4) for value in values))
    return [float(value) for value in values]


class TrajectorySupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_supervisor_node")

        self.declare_parameter("global_path_topic", "/apex/planning/fixed_map_path")
        self.declare_parameter("global_status_topic", "/apex/planning/fixed_map_status")
        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/fixed_map_localized")
        self.declare_parameter("local_path_topic", "/apex/planning/trajectory_supervisor/local_path")
        self.declare_parameter("status_topic", "/apex/planning/trajectory_supervisor/status")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("scan_projection_mode", "apex_forward_left")
        self.declare_parameter("plan_rate_hz", 10.0)
        self.declare_parameter("scan_timeout_s", 0.45)
        self.declare_parameter("odom_timeout_s", 0.75)
        self.declare_parameter("lookahead_distance", 1.8)
        self.declare_parameter("local_window_length", 2.0)
        self.declare_parameter("local_window_width", 1.8)
        self.declare_parameter("path_resample_step_m", 0.04)
        self.declare_parameter("obstacle_inflation_radius", 0.14)
        self.declare_parameter("collision_distance_threshold", 0.14)
        self.declare_parameter("path_corridor_width_m", 0.14)
        self.declare_parameter("rejoin_distance_threshold", 0.16)
        self.declare_parameter("max_avoid_curvature", 0.0)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("max_steering_deg", 18.0)
        self.declare_parameter("emergency_stop_distance", 0.28)
        self.declare_parameter("num_candidate_paths", 7)
        self.declare_parameter(
            "candidate_offset_values",
            [-0.42, -0.28, -0.14, 0.0, 0.14, 0.28, 0.42],
        )
        self.declare_parameter("candidate_clearance_weight", 0.70)
        self.declare_parameter("candidate_deviation_weight", 1.20)
        self.declare_parameter("candidate_curvature_weight", 0.20)
        self.declare_parameter("candidate_previous_offset_weight", 0.35)
        self.declare_parameter("path_smoothing_alpha", 0.18)
        self.declare_parameter("path_smoothing_max_iterations", 80)
        self.declare_parameter("min_path_forward_span_m", 0.55)
        self.declare_parameter("lidar_offset_x_m", 0.18)
        self.declare_parameter("lidar_offset_y_m", 0.0)
        self.declare_parameter("tracking_origin_offset_x_m", 0.0)
        self.declare_parameter("tracking_origin_offset_y_m", 0.0)
        self.declare_parameter("publish_pass_through_when_scan_missing", True)

        self._global_path_topic = str(self.get_parameter("global_path_topic").value)
        self._global_status_topic = str(self.get_parameter("global_status_topic").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._local_path_topic = str(self.get_parameter("local_path_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._configured_frame_id = str(self.get_parameter("frame_id").value).strip()
        self._scan_projection_mode = str(self.get_parameter("scan_projection_mode").value).strip()
        self._scan_projection_mode = self._scan_projection_mode or "apex_forward_left"
        self._plan_rate_hz = max(1.0, float(self.get_parameter("plan_rate_hz").value))
        self._scan_timeout_s = max(0.05, float(self.get_parameter("scan_timeout_s").value))
        self._odom_timeout_s = max(0.05, float(self.get_parameter("odom_timeout_s").value))
        self._lookahead_distance_m = max(0.20, float(self.get_parameter("lookahead_distance").value))
        self._local_window_length_m = max(0.25, float(self.get_parameter("local_window_length").value))
        self._local_window_width_m = max(0.25, float(self.get_parameter("local_window_width").value))
        self._path_resample_step_m = max(0.01, float(self.get_parameter("path_resample_step_m").value))
        self._obstacle_inflation_radius_m = max(
            0.0, float(self.get_parameter("obstacle_inflation_radius").value)
        )
        self._collision_distance_threshold_m = max(
            0.0, float(self.get_parameter("collision_distance_threshold").value)
        )
        self._required_clearance_m = max(
            self._obstacle_inflation_radius_m,
            self._collision_distance_threshold_m,
        )
        self._path_corridor_width_m = max(
            self._required_clearance_m,
            float(self.get_parameter("path_corridor_width_m").value),
        )
        self._rejoin_distance_threshold_m = max(
            0.0, float(self.get_parameter("rejoin_distance_threshold").value)
        )
        self._wheelbase_m = max(1.0e-3, float(self.get_parameter("wheelbase_m").value))
        self._max_steering_deg = max(1.0, float(self.get_parameter("max_steering_deg").value))
        max_curvature_param = float(self.get_parameter("max_avoid_curvature").value)
        self._max_avoid_curvature_m_inv = (
            max_curvature_param
            if max_curvature_param > 1.0e-9
            else math.tan(math.radians(self._max_steering_deg)) / self._wheelbase_m
        )
        self._emergency_stop_distance_m = max(
            0.0, float(self.get_parameter("emergency_stop_distance").value)
        )
        self._candidate_offsets_m = _parse_offsets(
            self.get_parameter("candidate_offset_values").value,
            int(self.get_parameter("num_candidate_paths").value),
            self._local_window_width_m,
        )
        self._candidate_clearance_weight = max(
            0.0, float(self.get_parameter("candidate_clearance_weight").value)
        )
        self._candidate_deviation_weight = max(
            0.0, float(self.get_parameter("candidate_deviation_weight").value)
        )
        self._candidate_curvature_weight = max(
            0.0, float(self.get_parameter("candidate_curvature_weight").value)
        )
        self._candidate_previous_offset_weight = max(
            0.0, float(self.get_parameter("candidate_previous_offset_weight").value)
        )
        self._path_smoothing_alpha = max(
            0.0, min(1.0, float(self.get_parameter("path_smoothing_alpha").value))
        )
        self._path_smoothing_max_iterations = max(
            0, int(self.get_parameter("path_smoothing_max_iterations").value)
        )
        self._min_path_forward_span_m = max(
            0.05, float(self.get_parameter("min_path_forward_span_m").value)
        )
        self._lidar_offset_xy = np.asarray(
            [
                float(self.get_parameter("lidar_offset_x_m").value),
                float(self.get_parameter("lidar_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._tracking_origin_offset_xy = np.asarray(
            [
                float(self.get_parameter("tracking_origin_offset_x_m").value),
                float(self.get_parameter("tracking_origin_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._publish_pass_through_when_scan_missing = bool(
            self.get_parameter("publish_pass_through_when_scan_missing").value
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Path, self._global_path_topic, self._global_path_cb, latched_qos)
        self.create_subscription(
            String, self._global_status_topic, self._global_status_cb, latched_qos
        )
        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)

        self._local_path_pub = self.create_publisher(Path, self._local_path_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)
        self.create_timer(1.0 / self._plan_rate_hz, self._plan_step)

        self._global_path_xy: np.ndarray | None = None
        self._global_path_s: np.ndarray | None = None
        self._global_path_yaw: np.ndarray | None = None
        self._global_frame_id: str | None = None
        self._global_status: dict[str, object] = {}
        self._latest_scan: LaserScan | None = None
        self._latest_scan_monotonic: float | None = None
        self._latest_odom: Odometry | None = None
        self._latest_odom_monotonic: float | None = None
        self._state = "FOLLOW"
        self._last_selected_offset_m = 0.0
        self._status_payload: dict[str, object] = {"state": "initializing", "ready": False}

        self.get_logger().info(
            "TrajectorySupervisorNode started (global_path=%s scan=%s odom=%s local_path=%s status=%s)"
            % (
                self._global_path_topic,
                self._scan_topic,
                self._odom_topic,
                self._local_path_topic,
                self._status_topic,
            )
        )

    def _global_path_cb(self, msg: Path) -> None:
        points: list[list[float]] = []
        yaws: list[float] = []
        for pose in msg.poses:
            points.append([float(pose.pose.position.x), float(pose.pose.position.y)])
            yaws.append(
                _quat_to_yaw(
                    float(pose.pose.orientation.x),
                    float(pose.pose.orientation.y),
                    float(pose.pose.orientation.z),
                    float(pose.pose.orientation.w),
                )
            )
        if len(points) < 2:
            return
        path_xy = np.asarray(points, dtype=np.float64)
        path_s = _polyline_s(path_xy)
        if path_s.size < 2 or float(path_s[-1]) <= 1.0e-6:
            return
        self._global_path_xy = path_xy
        self._global_path_s = path_s
        self._global_path_yaw = np.asarray(yaws, dtype=np.float64)
        self._global_frame_id = msg.header.frame_id or self._configured_frame_id

    def _global_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._global_status = payload

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._latest_scan_monotonic = time.monotonic()

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._latest_odom_monotonic = time.monotonic()

    def _pose_from_odom(self) -> tuple[np.ndarray, float, np.ndarray]:
        if self._latest_odom is None:
            raise RuntimeError("missing odom")
        pose = self._latest_odom.pose.pose
        yaw_rad = _quat_to_yaw(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        base_xy = np.asarray([float(pose.position.x), float(pose.position.y)], dtype=np.float64)
        rotation = _rotation(yaw_rad)
        tracking_origin_xy = base_xy + (self._tracking_origin_offset_xy @ rotation.T)
        lidar_xy = base_xy + (self._lidar_offset_xy @ rotation.T)
        return tracking_origin_xy, yaw_rad, lidar_xy

    def _scan_points_local(self, msg: LaserScan) -> np.ndarray:
        ranges = _sanitize_ranges(list(msg.ranges))
        if ranges.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        if self._scan_projection_mode == "laser_scan_angles":
            angles = float(msg.angle_min) + (np.arange(ranges.size, dtype=np.float64) * float(msg.angle_increment))
            valid_mask = np.isfinite(ranges) & (ranges > 0.0)
            if not np.any(valid_mask):
                return np.empty((0, 2), dtype=np.float64)
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            return np.column_stack(
                [valid_ranges * np.cos(valid_angles), valid_ranges * np.sin(valid_angles)]
            )

        points_x_m, points_y_m, _ = scan_ranges_to_forward_left_xy(ranges)
        if points_x_m.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([points_x_m, points_y_m])

    def _current_obstacles(
        self,
        *,
        tracking_origin_xy: np.ndarray,
        yaw_rad: float,
        lidar_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, float | None]:
        if self._latest_scan is None or self._latest_scan_monotonic is None:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                False,
                None,
            )
        scan_age_s = max(0.0, time.monotonic() - self._latest_scan_monotonic)
        scan_is_fresh = scan_age_s <= self._scan_timeout_s
        if not scan_is_fresh:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                False,
                scan_age_s,
            )

        lidar_local_xy = self._scan_points_local(self._latest_scan)
        if lidar_local_xy.shape[0] == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                True,
                scan_age_s,
            )

        world_xy = _transform_local_to_world(lidar_local_xy, lidar_xy, yaw_rad)
        origin_local_xy = _transform_world_to_local(world_xy, tracking_origin_xy, yaw_rad)
        lateral_window_m = 0.5 * self._local_window_width_m
        lateral_window_m += max(abs(value) for value in self._candidate_offsets_m)
        lateral_window_m += self._required_clearance_m
        mask = origin_local_xy[:, 0] >= -0.15
        mask &= origin_local_xy[:, 0] <= (self._local_window_length_m + 0.50)
        mask &= np.abs(origin_local_xy[:, 1]) <= lateral_window_m
        return world_xy[mask], origin_local_xy[mask], True, scan_age_s

    def _build_candidate_path(
        self,
        *,
        offset_m: float,
        projection: PathProjection,
        tracking_origin_xy: np.ndarray,
        forward_span_m: float,
    ) -> np.ndarray | None:
        if self._global_path_xy is None or self._global_path_s is None or self._global_path_yaw is None:
            return None
        end_s_m = min(
            float(self._global_path_s[-1]),
            float(projection.projection_s_m) + max(self._min_path_forward_span_m, forward_span_m),
        )
        if end_s_m <= projection.projection_s_m + 1.0e-3:
            return None

        point_count = max(3, int(math.ceil((end_s_m - projection.projection_s_m) / self._path_resample_step_m)) + 1)
        sample_s = np.linspace(projection.projection_s_m, end_s_m, point_count, dtype=np.float64)
        base_xy = _interp_path_xy(self._global_path_xy, self._global_path_s, sample_s)
        sample_yaw = _interp_path_yaw(self._global_path_yaw, self._global_path_s, sample_s)
        normals_xy = np.column_stack([-np.sin(sample_yaw), np.cos(sample_yaw)])
        t = np.linspace(0.0, 1.0, point_count, dtype=np.float64)

        rejoin_profile = projection.signed_lateral_error_m * (1.0 - _smoothstep(t))
        avoid_profile = float(offset_m) * (np.sin(math.pi * t) ** 2)
        lateral_profile = rejoin_profile + avoid_profile
        path_xy = base_xy + (lateral_profile.reshape(-1, 1) * normals_xy)
        path_xy[0] = tracking_origin_xy
        path_xy[-1] = base_xy[-1]

        smoothed_xy, _ = _smooth_path_to_curvature_limit(
            path_xy=path_xy,
            max_curvature_m_inv=self._max_avoid_curvature_m_inv,
            resample_step_m=self._path_resample_step_m,
            smoothing_alpha=self._path_smoothing_alpha,
            max_iterations=self._path_smoothing_max_iterations,
        )
        if smoothed_xy.shape[0] >= 2:
            smoothed_xy[0] = tracking_origin_xy
            smoothed_xy[-1] = base_xy[-1]
        return smoothed_xy

    def _min_path_clearance(self, path_xy: np.ndarray, obstacles_world_xy: np.ndarray) -> float:
        if obstacles_world_xy.shape[0] == 0 or path_xy.shape[0] == 0:
            return math.inf
        sampled_path_xy = _resample_polyline_xy(path_xy, self._path_resample_step_m)
        if sampled_path_xy.shape[0] == 0:
            return math.inf
        deltas = obstacles_world_xy[:, None, :] - sampled_path_xy[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        return float(np.min(distances))

    def _evaluate_candidate(
        self,
        *,
        offset_m: float,
        path_xy: np.ndarray | None,
        obstacles_world_xy: np.ndarray,
    ) -> CandidatePath:
        if path_xy is None or path_xy.shape[0] < 2:
            return CandidatePath(
                offset_m=float(offset_m),
                path_xy=np.empty((0, 2), dtype=np.float64),
                score=float("inf"),
                feasible=False,
                reason="empty_path",
                min_clearance_m=0.0,
                max_curvature_m_inv=0.0,
            )

        curvature = np.abs(_estimate_path_curvature(path_xy))
        max_curvature_m_inv = float(np.max(curvature)) if curvature.size else 0.0
        min_clearance_m = self._min_path_clearance(path_xy, obstacles_world_xy)
        if min_clearance_m < self._required_clearance_m:
            return CandidatePath(
                offset_m=float(offset_m),
                path_xy=path_xy,
                score=float("inf"),
                feasible=False,
                reason="collision",
                min_clearance_m=float(min_clearance_m),
                max_curvature_m_inv=max_curvature_m_inv,
            )
        if max_curvature_m_inv > (self._max_avoid_curvature_m_inv + 1.0e-6):
            return CandidatePath(
                offset_m=float(offset_m),
                path_xy=path_xy,
                score=float("inf"),
                feasible=False,
                reason="curvature_limit",
                min_clearance_m=float(min_clearance_m),
                max_curvature_m_inv=max_curvature_m_inv,
            )

        clearance_reward = 0.0 if math.isinf(min_clearance_m) else min(1.0, min_clearance_m)
        score = (
            (self._candidate_deviation_weight * abs(float(offset_m)))
            + (self._candidate_curvature_weight * max_curvature_m_inv)
            + (self._candidate_previous_offset_weight * abs(float(offset_m) - self._last_selected_offset_m))
            - (self._candidate_clearance_weight * clearance_reward)
        )
        return CandidatePath(
            offset_m=float(offset_m),
            path_xy=path_xy,
            score=float(score),
            feasible=True,
            reason="ok",
            min_clearance_m=float(min_clearance_m),
            max_curvature_m_inv=max_curvature_m_inv,
        )

    def _select_candidate(
        self,
        *,
        projection: PathProjection,
        tracking_origin_xy: np.ndarray,
        forward_span_m: float,
        obstacles_world_xy: np.ndarray,
        offsets_m: list[float],
    ) -> tuple[CandidatePath | None, list[CandidatePath]]:
        candidates: list[CandidatePath] = []
        for offset_m in offsets_m:
            path_xy = self._build_candidate_path(
                offset_m=offset_m,
                projection=projection,
                tracking_origin_xy=tracking_origin_xy,
                forward_span_m=forward_span_m,
            )
            candidates.append(
                self._evaluate_candidate(
                    offset_m=offset_m,
                    path_xy=path_xy,
                    obstacles_world_xy=obstacles_world_xy,
                )
            )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            return None, candidates
        return min(feasible, key=lambda candidate: candidate.score), candidates

    def _blocked_by_obstacles(self, path_xy: np.ndarray, obstacles_world_xy: np.ndarray) -> tuple[bool, float]:
        min_clearance_m = self._min_path_clearance(path_xy, obstacles_world_xy)
        return bool(min_clearance_m < self._path_corridor_width_m), float(min_clearance_m)

    def _emergency_obstacle(self, obstacles_local_xy: np.ndarray) -> bool:
        if obstacles_local_xy.shape[0] == 0 or self._emergency_stop_distance_m <= 1.0e-9:
            return False
        mask = obstacles_local_xy[:, 0] >= 0.02
        mask &= obstacles_local_xy[:, 0] <= self._emergency_stop_distance_m
        mask &= np.abs(obstacles_local_xy[:, 1]) <= self._required_clearance_m
        return bool(np.any(mask))

    def _build_hold_path(self, tracking_origin_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
        local_xy = np.asarray([[0.0, 0.0], [0.03, 0.0]], dtype=np.float64)
        return _transform_local_to_world(local_xy, tracking_origin_xy, yaw_rad)

    def _build_path_msg(self, path_xy: np.ndarray, *, frame_id: str, stamp_msg) -> Path:
        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp_msg
        yaw_samples = _path_yaw(path_xy)
        for index, point_xy in enumerate(path_xy):
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = frame_id
            pose_msg.header.stamp = stamp_msg
            pose_msg.pose.position.x = float(point_xy[0])
            pose_msg.pose.position.y = float(point_xy[1])
            pose_msg.pose.position.z = 0.0
            yaw_rad = float(yaw_samples[min(index, yaw_samples.size - 1)]) if yaw_samples.size else 0.0
            qx, qy, qz, qw = _yaw_to_quat(yaw_rad)
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            msg.poses.append(pose_msg)
        return msg

    def _publish_status(self, payload: dict[str, object]) -> None:
        self._status_payload = payload
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _publish_wait_status(self, *, state: str, reason: str) -> None:
        payload = {
            "state": state,
            "ready": False,
            "reason": reason,
            "local_path_source": "trajectory_supervisor",
            "global_path_topic": self._global_path_topic,
            "scan_topic": self._scan_topic,
            "odom_topic": self._odom_topic,
            "local_path_topic": self._local_path_topic,
            "stamp_s": 1.0e-9 * float(self.get_clock().now().nanoseconds),
        }
        self._publish_status(payload)

    def _plan_step(self) -> None:
        if self._global_path_xy is None or self._global_path_s is None or self._global_path_yaw is None:
            self._publish_wait_status(state="waiting_global_path", reason="missing_global_path")
            return
        if self._latest_odom is None or self._latest_odom_monotonic is None:
            self._publish_wait_status(state="waiting_odom", reason="missing_odom")
            return

        odom_age_s = max(0.0, time.monotonic() - self._latest_odom_monotonic)
        if odom_age_s > self._odom_timeout_s:
            self._publish_wait_status(state="waiting_odom", reason="stale_odom")
            return

        tracking_origin_xy, yaw_rad, lidar_xy = self._pose_from_odom()
        projection = _project_onto_polyline(
            tracking_origin_xy,
            self._global_path_xy,
            self._global_path_s,
        )
        if projection is None:
            self._publish_wait_status(state="waiting_projection", reason="invalid_global_path_projection")
            return

        frame_id = self._configured_frame_id or self._global_frame_id or self._latest_odom.header.frame_id
        obstacles_world_xy, obstacles_local_xy, scan_ready, scan_age_s = self._current_obstacles(
            tracking_origin_xy=tracking_origin_xy,
            yaw_rad=yaw_rad,
            lidar_xy=lidar_xy,
        )

        remaining_global_m = max(0.0, float(self._global_path_s[-1]) - float(projection.projection_s_m))
        forward_span_m = min(
            self._lookahead_distance_m,
            self._local_window_length_m,
            max(self._min_path_forward_span_m, remaining_global_m),
        )
        zero_path_xy = self._build_candidate_path(
            offset_m=0.0,
            projection=projection,
            tracking_origin_xy=tracking_origin_xy,
            forward_span_m=forward_span_m,
        )
        if zero_path_xy is None:
            hold_xy = self._build_hold_path(tracking_origin_xy, yaw_rad)
            now_msg = self.get_clock().now().to_msg()
            self._local_path_pub.publish(self._build_path_msg(hold_xy, frame_id=frame_id, stamp_msg=now_msg))
            self._publish_status(
                {
                    "state": "stop_recovery",
                    "ready": True,
                    "reason": "failed_to_build_path",
                    "local_path_source": "emergency_hold",
                    "continuation_source": "emergency_hold",
                    "local_path_age_s": 0.0,
                    "path_forward_span_m": _polyline_length_m(hold_xy),
                    "stamp_s": 1.0e-9 * float(self.get_clock().now().nanoseconds),
                }
            )
            return

        scan_missing = not scan_ready
        if scan_missing and not self._publish_pass_through_when_scan_missing:
            self._publish_wait_status(state="waiting_scan", reason="missing_or_stale_scan")
            return

        global_blocked, global_min_clearance_m = self._blocked_by_obstacles(
            zero_path_xy,
            obstacles_world_xy,
        )
        emergency_blocked = self._emergency_obstacle(obstacles_local_xy)

        if scan_missing:
            desired_state = "FOLLOW"
            selected = self._evaluate_candidate(
                offset_m=0.0,
                path_xy=zero_path_xy,
                obstacles_world_xy=np.empty((0, 2), dtype=np.float64),
            )
            candidates = [selected]
            reason = "scan_unavailable_pass_through"
        elif global_blocked or emergency_blocked:
            selected, candidates = self._select_candidate(
                projection=projection,
                tracking_origin_xy=tracking_origin_xy,
                forward_span_m=forward_span_m,
                obstacles_world_xy=obstacles_world_xy,
                offsets_m=self._candidate_offsets_m,
            )
            if selected is None:
                desired_state = "STOP_RECOVERY"
                reason = "no_safe_candidate"
            else:
                desired_state = "AVOID"
                reason = "obstacle_in_corridor"
        elif self._state in {"AVOID", "REJOIN", "STOP_RECOVERY"} and (
            projection.distance_m > self._rejoin_distance_threshold_m
        ):
            selected, candidates = self._select_candidate(
                projection=projection,
                tracking_origin_xy=tracking_origin_xy,
                forward_span_m=forward_span_m,
                obstacles_world_xy=obstacles_world_xy,
                offsets_m=[0.0],
            )
            if selected is None:
                selected, candidates = self._select_candidate(
                    projection=projection,
                    tracking_origin_xy=tracking_origin_xy,
                    forward_span_m=forward_span_m,
                    obstacles_world_xy=obstacles_world_xy,
                    offsets_m=self._candidate_offsets_m,
                )
            if selected is None:
                desired_state = "STOP_RECOVERY"
                reason = "rejoin_path_blocked"
            else:
                desired_state = "REJOIN"
                reason = "returning_to_global_path"
        else:
            selected = self._evaluate_candidate(
                offset_m=0.0,
                path_xy=zero_path_xy,
                obstacles_world_xy=obstacles_world_xy,
            )
            candidates = [selected]
            desired_state = "FOLLOW"
            reason = "global_corridor_clear"

        if desired_state == "STOP_RECOVERY" or selected is None:
            publish_path_xy = self._build_hold_path(tracking_origin_xy, yaw_rad)
            selected_offset_m = 0.0
            local_path_source = "emergency_hold"
            continuation_source = "emergency_hold"
            selected_min_clearance_m = 0.0
            selected_curvature_m_inv = 0.0
        else:
            publish_path_xy = selected.path_xy
            selected_offset_m = float(selected.offset_m)
            local_path_source = "trajectory_supervisor_%s" % desired_state.lower()
            continuation_source = "tracking" if desired_state == "FOLLOW" else local_path_source
            selected_min_clearance_m = float(selected.min_clearance_m)
            selected_curvature_m_inv = float(selected.max_curvature_m_inv)

        now_msg = self.get_clock().now().to_msg()
        self._local_path_pub.publish(
            self._build_path_msg(publish_path_xy, frame_id=frame_id, stamp_msg=now_msg)
        )
        self._state = desired_state
        self._last_selected_offset_m = selected_offset_m

        candidate_status = [
            {
                "offset_m": float(candidate.offset_m),
                "feasible": bool(candidate.feasible),
                "reason": candidate.reason,
                "min_clearance_m": float(candidate.min_clearance_m),
                "max_curvature_m_inv": float(candidate.max_curvature_m_inv),
                "score": float(candidate.score) if math.isfinite(candidate.score) else None,
            }
            for candidate in candidates
        ]
        path_heading_alignment_deg = abs(
            math.degrees(_normalize_angle(_path_yaw(publish_path_xy)[0] - projection.heading_rad))
        )
        payload = {
            "state": desired_state.lower(),
            "ready": True,
            "reason": reason,
            "local_path_source": local_path_source,
            "continuation_source": continuation_source,
            "local_path_age_s": 0.0,
            "path_forward_span_m": _polyline_length_m(publish_path_xy),
            "path_heading_alignment_deg": path_heading_alignment_deg,
            "global_path_topic": self._global_path_topic,
            "local_path_topic": self._local_path_topic,
            "status_topic": self._status_topic,
            "scan_topic": self._scan_topic,
            "odom_topic": self._odom_topic,
            "scan_ready": bool(scan_ready),
            "scan_age_s": scan_age_s,
            "odom_age_s": odom_age_s,
            "obstruction_detected": bool(global_blocked),
            "emergency_obstacle": bool(emergency_blocked),
            "global_min_clearance_m": float(global_min_clearance_m),
            "selected_offset_m": selected_offset_m,
            "selected_min_clearance_m": selected_min_clearance_m,
            "selected_max_curvature_m_inv": selected_curvature_m_inv,
            "candidate_count": len(candidates),
            "candidate_offsets_m": self._candidate_offsets_m,
            "candidates": candidate_status,
            "path_deviation_m": float(projection.distance_m),
            "projection_s_m": float(projection.projection_s_m),
            "required_clearance_m": float(self._required_clearance_m),
            "path_corridor_width_m": float(self._path_corridor_width_m),
            "global_planner_state": self._global_status.get("state"),
            "stamp_s": 1.0e-9 * float(self.get_clock().now().nanoseconds),
        }
        self._publish_status(payload)


def main() -> None:
    rclpy.init()
    node = TrajectorySupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
