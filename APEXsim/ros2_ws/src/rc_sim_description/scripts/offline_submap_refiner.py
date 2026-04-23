#!/usr/bin/env python3
"""Offline-style submap refinement over live simulation buffers.

This node consumes simulated scans plus IMU, optionally seeded by an odometry
topic, and refines short overlapping scan windows. After each local refinement
it republishes an accumulated map/path so RViz updates incrementally while the
vehicle is still driving.
"""

from __future__ import annotations

import json
import math
import threading
import time
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * ((w * z) + (x * y)),
        1.0 - (2.0 * ((y * y) + (z * z))),
    )


def _rotation_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
        dtype=np.float64,
    )


def _voxel_downsample(points_xy: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xy.size == 0 or voxel_size_m <= 1.0e-6:
        return points_xy
    grid = np.floor(points_xy / voxel_size_m).astype(np.int64)
    _, unique_indexes = np.unique(grid, axis=0, return_index=True)
    unique_indexes.sort()
    return points_xy[unique_indexes]


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    rotation = _rotation_matrix(float(pose_xyyaw[2]))
    return (points_xy @ rotation.T) + pose_xyyaw[:2]


def _compose_poses(lhs_pose_xyyaw: np.ndarray, rhs_pose_xyyaw: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs_pose_xyyaw, dtype=np.float64)
    rhs = np.asarray(rhs_pose_xyyaw, dtype=np.float64)
    xy = lhs[:2] + (_rotation_matrix(float(lhs[2])) @ rhs[:2])
    return np.asarray(
        [float(xy[0]), float(xy[1]), _wrap_angle(float(lhs[2] + rhs[2]))],
        dtype=np.float64,
    )


def _best_fit_rigid_transform_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    if source_xy.shape != target_xy.shape or source_xy.shape[0] < 3:
        return np.zeros(3, dtype=np.float64)
    source_center = np.mean(source_xy, axis=0)
    target_center = np.mean(target_xy, axis=0)
    source_zero = source_xy - source_center
    target_zero = target_xy - target_center
    covariance = source_zero.T @ target_zero
    u_mat, _, v_t = np.linalg.svd(covariance)
    rotation = v_t.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        v_t[-1, :] *= -1.0
        rotation = v_t.T @ u_mat.T
    translation = target_center - (rotation @ source_center)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.asarray(
        [float(translation[0]), float(translation[1]), _wrap_angle(yaw)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ScanRecord:
    scan_index: int
    stamp_s: float
    frame_id: str
    points_xy: np.ndarray


@dataclass(frozen=True)
class ImuRecord:
    stamp_s: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class OdomRecord:
    stamp_s: float
    frame_id: str
    pose_xyyaw: np.ndarray


@dataclass(frozen=True)
class ScanQuality:
    scan_index: int
    median_residual_m: float
    valid_correspondence_count: int
    confidence: str


@dataclass(frozen=True)
class SeedStatusRecord:
    received_wall_s: float
    latest_pose_t_s: float
    state: str
    latest_confidence: str
    median_submap_residual_m: float


@dataclass(frozen=True)
class RelativeMotionEstimate:
    delta_pose_xyyaw: np.ndarray
    valid_match_count: int
    median_residual_m: float


@dataclass(frozen=True)
class InterWindowAlignmentResult:
    enabled: bool
    applied: bool
    reason: str
    overlap_scan_count: int
    paired_point_count: int
    raw_delta_xyyaw: np.ndarray
    applied_delta_xyyaw: np.ndarray


class OfflineSubmapRefiner(Node):
    def __init__(self) -> None:
        super().__init__("offline_submap_refiner")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.declare_parameter("replay_mode", "live_buffer")
        self.declare_parameter("scan_topic", "/apex/sim/scan")
        self.declare_parameter("imu_topic", "/apex/sim/imu")
        self.declare_parameter("seed_odom_topic", "")
        self.declare_parameter("seed_status_topic", "")
        self.declare_parameter("input_dir", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "offline_refined_base_link")
        self.declare_parameter("map_topic", "/apex/sim/offline_refined_map")
        self.declare_parameter("grid_topic", "/apex/sim/offline_refined_grid")
        self.declare_parameter("path_topic", "/apex/sim/offline_refined_path")
        self.declare_parameter("submap_topic", "/apex/sim/offline_current_submap")
        self.declare_parameter("status_topic", "/apex/sim/offline_refined_status")
        self.declare_parameter("odom_topic", "/apex/sim/offline_refined_odom")
        self.declare_parameter("window_scan_count", 48)
        self.declare_parameter("window_overlap_count", 16)
        self.declare_parameter("initial_scan_count", 24)
        self.declare_parameter("submap_window_scans", 8)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("max_correspondence_m", 0.35)
        self.declare_parameter("offline_update_period_sec", 0.5)
        self.declare_parameter("seed_status_timeout_sec", 2.0)
        self.declare_parameter("seed_status_max_median_submap_residual_m", 0.12)
        self.declare_parameter("publish_global_correction", True)
        self.declare_parameter("global_correction_topic", "/apex/sim/offline_global_correction")
        self.declare_parameter("anchor_pose_topic", "/apex/sim/offline_anchor_pose")
        self.declare_parameter("seed_odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("enable_inter_window_alignment", False)
        self.declare_parameter("inter_window_alignment_gain", 0.85)
        self.declare_parameter("inter_window_min_overlap_scans", 4)
        self.declare_parameter("inter_window_min_points", 80)
        self.declare_parameter("inter_window_max_points", 2500)
        self.declare_parameter("inter_window_max_translation_m", 0.25)
        self.declare_parameter("inter_window_max_yaw_deg", 10.0)
        self.declare_parameter("enable_manual_idle_finalize", False)
        self.declare_parameter("manual_status_topic", "/apex/manual_control/status")
        self.declare_parameter("manual_idle_timeout_s", 4.0)
        self.declare_parameter("manual_motion_linear_deadband_mps", 0.02)
        self.declare_parameter("final_min_scan_count", 8)
        self.declare_parameter("save_on_finalize", False)
        self.declare_parameter("save_output_dir", "")
        self.declare_parameter("use_track_geometry_prior", False)
        self.declare_parameter("track_geometry_file", "")
        self.declare_parameter("track_geometry_weight", 0.08)
        self.declare_parameter("map_persistence_filter", True)
        self.declare_parameter("map_persistence_voxel_size_m", 0.08)
        self.declare_parameter("map_persistence_min_observations", 3)
        self.declare_parameter("map_persistence_decay_per_missed_scan", 0.35)
        self.declare_parameter("map_persistence_max_decay_gap_scans", 8)

        self._replay_mode = str(self.get_parameter("replay_mode").value).strip().lower()
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._seed_odom_topic = str(self.get_parameter("seed_odom_topic").value).strip()
        self._seed_status_topic = str(self.get_parameter("seed_status_topic").value).strip()
        self._input_dir = str(self.get_parameter("input_dir").value).strip()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._child_frame_id = str(self.get_parameter("child_frame_id").value)
        self._window_scan_count = max(8, int(self.get_parameter("window_scan_count").value))
        self._window_overlap_count = max(
            0, min(int(self.get_parameter("window_overlap_count").value), self._window_scan_count - 1)
        )
        self._window_step = max(1, self._window_scan_count - self._window_overlap_count)
        self._initial_scan_count = max(
            4,
            min(int(self.get_parameter("initial_scan_count").value), self._window_scan_count),
        )
        self._submap_window_scans = max(
            2,
            min(int(self.get_parameter("submap_window_scans").value), self._window_scan_count),
        )
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._max_correspondence_m = max(
            0.05, float(self.get_parameter("max_correspondence_m").value)
        )
        self._offline_update_period_s = max(
            0.1, float(self.get_parameter("offline_update_period_sec").value)
        )
        self._seed_status_timeout_s = max(
            0.5, float(self.get_parameter("seed_status_timeout_sec").value)
        )
        self._seed_max_median_submap_residual_m = max(
            0.01, float(self.get_parameter("seed_status_max_median_submap_residual_m").value)
        )
        self._publish_global_correction = bool(
            self.get_parameter("publish_global_correction").value
        )
        self._global_correction_topic = str(
            self.get_parameter("global_correction_topic").value
        )
        self._anchor_pose_topic = str(self.get_parameter("anchor_pose_topic").value)
        self._seed_odom_frame_id = str(self.get_parameter("seed_odom_frame_id").value).strip() or "odom_imu_lidar_fused"
        self._enable_inter_window_alignment = bool(
            self.get_parameter("enable_inter_window_alignment").value
        )
        self._inter_window_alignment_gain = min(
            1.0,
            max(0.0, float(self.get_parameter("inter_window_alignment_gain").value)),
        )
        self._inter_window_min_overlap_scans = max(
            1,
            int(self.get_parameter("inter_window_min_overlap_scans").value),
        )
        self._inter_window_min_points = max(
            3,
            int(self.get_parameter("inter_window_min_points").value),
        )
        self._inter_window_max_points = max(
            self._inter_window_min_points,
            int(self.get_parameter("inter_window_max_points").value),
        )
        self._inter_window_max_translation_m = max(
            0.0,
            float(self.get_parameter("inter_window_max_translation_m").value),
        )
        self._inter_window_max_yaw_rad = math.radians(
            max(0.0, float(self.get_parameter("inter_window_max_yaw_deg").value))
        )
        self._enable_manual_idle_finalize = bool(
            self.get_parameter("enable_manual_idle_finalize").value
        )
        self._manual_status_topic = str(self.get_parameter("manual_status_topic").value).strip()
        self._manual_idle_timeout_s = max(
            0.5, float(self.get_parameter("manual_idle_timeout_s").value)
        )
        self._manual_motion_linear_deadband_mps = max(
            0.0, float(self.get_parameter("manual_motion_linear_deadband_mps").value)
        )
        self._final_min_scan_count = max(
            4, int(self.get_parameter("final_min_scan_count").value)
        )
        self._save_on_finalize = bool(self.get_parameter("save_on_finalize").value)
        self._save_output_dir = str(self.get_parameter("save_output_dir").value).strip()
        self._use_track_geometry_prior = bool(
            self.get_parameter("use_track_geometry_prior").value
        )
        self._track_geometry_file = str(self.get_parameter("track_geometry_file").value).strip()
        self._track_geometry_weight = max(
            0.0, float(self.get_parameter("track_geometry_weight").value)
        )
        self._map_persistence_filter = bool(
            self.get_parameter("map_persistence_filter").value
        )
        self._map_persistence_voxel_size_m = max(
            0.0,
            float(self.get_parameter("map_persistence_voxel_size_m").value),
        )
        self._map_persistence_min_observations = max(
            1,
            int(self.get_parameter("map_persistence_min_observations").value),
        )
        self._map_persistence_decay_per_missed_scan = max(
            0.0,
            float(self.get_parameter("map_persistence_decay_per_missed_scan").value),
        )
        self._map_persistence_max_decay_gap_scans = max(
            0,
            int(self.get_parameter("map_persistence_max_decay_gap_scans").value),
        )
        self._global_voxel_size_m = 0.04
        self._submap_voxel_size_m = 0.025
        self._grid_resolution_m = 0.05
        self._grid_padding_m = 0.60
        self._grid_line_radius_cells = 1
        self._max_scan_buffer = max(512, 6 * self._window_scan_count)
        self._max_imu_buffer = 10000
        self._max_odom_buffer = 6000

        self._latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("map_topic").value),
            self._latched_qos,
        )
        self._grid_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("grid_topic").value),
            self._latched_qos,
        )
        self._path_pub = self.create_publisher(
            NavPath,
            str(self.get_parameter("path_topic").value),
            self._latched_qos,
        )
        self._submap_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("submap_topic").value),
            self._latched_qos,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            self._latched_qos,
        )
        self._odom_pub = self.create_publisher(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._latched_qos,
        )
        self._correction_pub = self.create_publisher(
            TransformStamped,
            self._global_correction_topic,
            self._latched_qos,
        )
        self._anchor_pose_pub = self.create_publisher(
            PoseStamped,
            self._anchor_pose_topic,
            self._latched_qos,
        )

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        self._lock = threading.Lock()
        self._scan_records: deque[ScanRecord] = deque(maxlen=self._max_scan_buffer)
        self._imu_records: deque[ImuRecord] = deque(maxlen=self._max_imu_buffer)
        self._odom_records: deque[OdomRecord] = deque(maxlen=self._max_odom_buffer)
        self._seed_status_records: deque[SeedStatusRecord] = deque(maxlen=256)
        self._next_scan_index = 0
        self._next_window_start_index = 0
        self._processing_thread: threading.Thread | None = None
        self._processed_window_count = 0
        self._last_gate_reason = ""
        self._analysis_closed = False
        self._analysis_closing = False
        self._closed_reason = ""
        self._saved_output_dir = ""
        self._manual_motion_seen = False
        self._manual_motion_active = False
        self._last_manual_status_monotonic = 0.0
        self._last_manual_motion_monotonic = 0.0
        self._last_manual_motion_scan_index = -1
        self._manual_idle_started_monotonic: float | None = None
        self._manual_idle_start_scan_index: int | None = None

        self._refined_records: dict[int, dict[str, object]] = {}
        self._latest_map_points = np.empty((0, 2), dtype=np.float64)
        self._latest_submap_points = np.empty((0, 2), dtype=np.float64)
        self._latest_path_poses = np.empty((0, 3), dtype=np.float64)
        self._latest_pose = np.zeros(3, dtype=np.float64)
        self._latest_map_persistence_stats: dict[str, object] = {
            "enabled": bool(self._map_persistence_filter),
        }
        self._latest_global_correction = np.zeros(3, dtype=np.float64)
        self._latest_global_correction_child_frame_id = self._seed_odom_frame_id
        self._latest_status = String()
        self._latest_status.data = json.dumps(
            {
                "state": "initializing",
                "replay_mode": self._replay_mode,
            },
            separators=(",", ":"),
        )

        if self._replay_mode == "live_buffer":
            self.create_subscription(
                LaserScan,
                self._scan_topic,
                self._scan_cb,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Imu,
                self._imu_topic,
                self._imu_cb,
                qos_profile_sensor_data,
            )
            if self._seed_odom_topic:
                self.create_subscription(Odometry, self._seed_odom_topic, self._odom_cb, 40)
            if self._seed_status_topic:
                self.create_subscription(String, self._seed_status_topic, self._seed_status_cb, 20)
            if self._enable_manual_idle_finalize and self._manual_status_topic:
                self.create_subscription(String, self._manual_status_topic, self._manual_status_cb, 20)
        else:
            self.get_logger().info(
                "offline_replay_mode=%s is accepted but not implemented in v1"
                % self._replay_mode
            )
            self._latest_status.data = json.dumps(
                {
                    "state": "unsupported_mode",
                    "replay_mode": self._replay_mode,
                    "input_dir": self._input_dir,
                },
                separators=(",", ":"),
            )
        self._track_geometry_points = self._load_track_geometry_points()
        self._track_geometry_tree = (
            cKDTree(self._track_geometry_points)
            if self._track_geometry_points.shape[0] >= 2
            else None
        )
        if self._use_track_geometry_prior and self._track_geometry_points.shape[0] < 2:
            self.get_logger().info(
                "Track geometry prior enabled but no valid geometry points were loaded"
            )
        elif self._track_geometry_points.shape[0] >= 2:
            self.get_logger().info(
                "Loaded %d track-geometry prior points from %s"
                % (self._track_geometry_points.shape[0], self._track_geometry_file or "<embedded>")
            )

        self.create_timer(self._offline_update_period_s, self._tick)
        self.get_logger().info(
            "OfflineSubmapRefiner started (scan=%s imu=%s seed_odom=%s seed_status=%s mode=%s window=%d overlap=%d inter_window=%s manual_idle_finalize=%s)"
            % (
                self._scan_topic,
                self._imu_topic,
                self._seed_odom_topic or "<none>",
                self._seed_status_topic or "<none>",
                self._replay_mode,
                self._window_scan_count,
                self._window_overlap_count,
                "on" if self._enable_inter_window_alignment else "off",
                "on" if self._enable_manual_idle_finalize else "off",
            )
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        points_xy = self._scan_points_in_base_frame(msg)
        if points_xy.size == 0:
            return
        record = ScanRecord(
            scan_index=self._next_scan_index,
            stamp_s=float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
            frame_id=str(msg.header.frame_id),
            points_xy=points_xy[:: self._point_stride].copy(),
        )
        self._next_scan_index += 1
        with self._lock:
            self._scan_records.append(record)

    def _imu_cb(self, msg: Imu) -> None:
        record = ImuRecord(
            stamp_s=float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
            yaw_rate_rps=float(msg.angular_velocity.z),
        )
        with self._lock:
            self._imu_records.append(record)

    def _odom_cb(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        record = OdomRecord(
            stamp_s=float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
            frame_id=str(msg.header.frame_id).strip() or self._seed_odom_frame_id,
            pose_xyyaw=np.asarray(
                [
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    _quat_to_yaw(
                        float(orientation.x),
                        float(orientation.y),
                        float(orientation.z),
                        float(orientation.w),
                    ),
                ],
                dtype=np.float64,
            ),
        )
        with self._lock:
            self._odom_records.append(record)

    def _load_track_geometry_points(self) -> np.ndarray:
        if not self._use_track_geometry_prior:
            return np.empty((0, 2), dtype=np.float64)
        if not self._track_geometry_file:
            return np.empty((0, 2), dtype=np.float64)
        try:
            payload = json.loads(Path(self._track_geometry_file).read_text(encoding="utf-8"))
        except Exception:
            try:
                import yaml  # local import to keep the hot path small

                payload = yaml.safe_load(Path(self._track_geometry_file).read_text(encoding="utf-8"))
            except Exception:
                return np.empty((0, 2), dtype=np.float64)
        raw_points = payload
        if isinstance(payload, dict):
            for key in ("centerline", "points", "polyline"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    raw_points = candidate
                    break
        if not isinstance(raw_points, list):
            return np.empty((0, 2), dtype=np.float64)
        parsed_points: list[tuple[float, float]] = []
        for entry in raw_points:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                parsed_points.append((float(entry[0]), float(entry[1])))
                continue
            if isinstance(entry, dict):
                if {"x_m", "y_m"}.issubset(entry):
                    parsed_points.append((float(entry["x_m"]), float(entry["y_m"])))
                    continue
                if {"x", "y"}.issubset(entry):
                    parsed_points.append((float(entry["x"]), float(entry["y"])))
                    continue
        if len(parsed_points) < 2:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(parsed_points, dtype=np.float64)

    def _seed_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        latest_pose = payload.get("latest_pose", {})
        quality = payload.get("quality", {})
        if not isinstance(latest_pose, dict):
            latest_pose = {}
        if not isinstance(quality, dict):
            quality = {}
        latest_pose_t_s = float(latest_pose.get("t_s", float("nan")))
        if not math.isfinite(latest_pose_t_s):
            latest_pose_t_s = float(payload.get("t_s", float("nan")))
        record = SeedStatusRecord(
            received_wall_s=time.monotonic(),
            latest_pose_t_s=latest_pose_t_s,
            state=str(payload.get("state", "")),
            latest_confidence=str(latest_pose.get("confidence", "")),
            median_submap_residual_m=float(
                quality.get("median_submap_residual_m", float("nan"))
            ),
        )
        with self._lock:
            self._seed_status_records.append(record)

    def _manual_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        motion_active = self._manual_status_has_motion_command(payload)
        now_monotonic = time.monotonic()
        with self._lock:
            latest_scan_index = (
                int(self._scan_records[-1].scan_index)
                if self._scan_records
                else max(-1, int(self._next_scan_index) - 1)
            )
            self._last_manual_status_monotonic = now_monotonic
            if motion_active:
                self._manual_motion_seen = True
                self._manual_motion_active = True
                self._last_manual_motion_monotonic = now_monotonic
                self._last_manual_motion_scan_index = latest_scan_index
                self._manual_idle_started_monotonic = None
                self._manual_idle_start_scan_index = None
                return
            if self._manual_motion_seen and self._manual_idle_started_monotonic is None:
                self._manual_idle_started_monotonic = now_monotonic
                self._manual_idle_start_scan_index = latest_scan_index
            self._manual_motion_active = False

    def _manual_status_has_motion_command(self, payload: dict[str, Any]) -> bool:
        if not bool(payload.get("bridge_connected", False)):
            return False
        if not bool(payload.get("controller_connected", False)):
            return False
        if not bool(payload.get("enabled", False)):
            return False
        try:
            linear_x_mps = float(payload.get("linear_x_mps", 0.0))
        except Exception:
            return False
        return abs(linear_x_mps) > self._manual_motion_linear_deadband_mps

    def _scan_points_in_base_frame(self, msg: LaserScan) -> np.ndarray:
        points_xy: list[tuple[float, float]] = []
        angle_rad = float(msg.angle_min)
        for raw_range in msg.ranges:
            range_m = float(raw_range)
            if math.isfinite(range_m) and float(msg.range_min) <= range_m <= float(msg.range_max):
                points_xy.append(
                    (
                        range_m * math.cos(angle_rad),
                        range_m * math.sin(angle_rad),
                    )
                )
            angle_rad += float(msg.angle_increment)
        if not points_xy:
            return np.empty((0, 2), dtype=np.float64)
        points = np.asarray(points_xy, dtype=np.float64)
        if not msg.header.frame_id or msg.header.frame_id == "base_link":
            return points
        try:
            transform = self._tf_buffer.lookup_transform(
                "base_link",
                str(msg.header.frame_id),
                rclpy.time.Time(),
            )
        except TransformException:
            return points
        translation_x = float(transform.transform.translation.x)
        translation_y = float(transform.transform.translation.y)
        rotation = transform.transform.rotation
        yaw_rad = _quat_to_yaw(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        transformed = (points @ _rotation_matrix(yaw_rad).T) + np.asarray(
            [translation_x, translation_y],
            dtype=np.float64,
        )
        return transformed

    def _tick(self) -> None:
        self._republish_latest_outputs()
        if self._replay_mode != "live_buffer":
            return
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return
        close_reason = self._manual_idle_close_reason()
        if close_reason:
            self._last_gate_reason = ""
            self._processing_thread = threading.Thread(
                target=self._process_final_window,
                args=(close_reason,),
                daemon=True,
            )
            self._processing_thread.start()
            return
        with self._lock:
            if self._analysis_closed or self._analysis_closing:
                return
        pending_window, gate_reason = self._has_pending_window()
        if gate_reason and gate_reason != self._last_gate_reason:
            self.get_logger().info(f"Deferring offline window: {gate_reason}")
            self._last_gate_reason = gate_reason
        if not pending_window:
            return
        self._last_gate_reason = ""
        self._processing_thread = threading.Thread(target=self._process_next_window, daemon=True)
        self._processing_thread.start()

    def _manual_idle_close_reason(self) -> str:
        if not self._enable_manual_idle_finalize:
            return ""
        now_monotonic = time.monotonic()
        with self._lock:
            if self._analysis_closed or self._analysis_closing:
                return ""
            if not self._manual_motion_seen:
                return ""
            if self._manual_motion_active:
                status_age_s = (
                    now_monotonic - self._last_manual_status_monotonic
                    if self._last_manual_status_monotonic > 0.0
                    else 0.0
                )
                if status_age_s <= self._manual_idle_timeout_s:
                    return ""
                self._manual_motion_active = False
                self._manual_idle_started_monotonic = self._last_manual_status_monotonic
                if self._manual_idle_start_scan_index is None:
                    self._manual_idle_start_scan_index = (
                        self._last_manual_motion_scan_index
                        if self._last_manual_motion_scan_index >= 0
                        else (
                            int(self._scan_records[-1].scan_index)
                            if self._scan_records
                            else max(-1, int(self._next_scan_index) - 1)
                        )
                    )
            idle_started = self._manual_idle_started_monotonic
            if idle_started is None:
                idle_started = self._last_manual_motion_monotonic
                self._manual_idle_started_monotonic = idle_started
                if self._manual_idle_start_scan_index is None:
                    self._manual_idle_start_scan_index = (
                        self._last_manual_motion_scan_index
                        if self._last_manual_motion_scan_index >= 0
                        else (
                            int(self._scan_records[-1].scan_index)
                            if self._scan_records
                            else max(-1, int(self._next_scan_index) - 1)
                        )
                    )
            if idle_started <= 0.0:
                return ""
            idle_s = now_monotonic - idle_started
            if idle_s < self._manual_idle_timeout_s:
                return ""
            self._analysis_closing = True
            return "manual_idle_timeout"

    def _has_pending_window(self) -> tuple[bool, str]:
        with self._lock:
            scan_records = list(self._scan_records)
            next_window_start = self._next_window_start_index
            seed_status_records = list(self._seed_status_records)
        if len(scan_records) < self._initial_scan_count:
            return False, ""
        oldest_index = int(scan_records[0].scan_index)
        newest_index = int(scan_records[-1].scan_index)
        effective_window_start = max(next_window_start, oldest_index)
        available_count = newest_index - effective_window_start + 1
        if effective_window_start == 0:
            if available_count < self._initial_scan_count:
                return False, ""
            window_end = min(newest_index, effective_window_start + self._window_scan_count - 1)
        else:
            if available_count < self._window_scan_count:
                return False, ""
            window_end = effective_window_start + self._window_scan_count - 1
        window_records = [
            record
            for record in scan_records
            if effective_window_start <= int(record.scan_index) <= window_end
        ]
        if len(window_records) < self._initial_scan_count:
            return False, ""
        seed_gate_ok, seed_gate_reason = self._seed_status_allows_window(
            window_records,
            seed_status_records,
        )
        return seed_gate_ok, seed_gate_reason

    def _process_next_window(self) -> None:
        try:
            snapshot = self._window_snapshot()
            if snapshot is None:
                return
            self._process_window_snapshot(snapshot, final_window=False, close_reason="")
            self._republish_latest_outputs()
        except Exception as exc:
            self.get_logger().warn(f"Offline submap refinement failed: {repr(exc)}")

    def _process_final_window(self, close_reason: str) -> None:
        try:
            self.get_logger().info(f"Closing offline analysis (reason={close_reason})")
            snapshot, skip_reason = self._final_window_snapshot()
            if snapshot is None:
                self._finalize_without_new_window(
                    close_reason=close_reason,
                    skip_reason=skip_reason,
                )
                self._republish_latest_outputs()
                return
            self._process_window_snapshot(
                snapshot,
                final_window=True,
                close_reason=close_reason,
            )
            self._republish_latest_outputs()
        except Exception as exc:
            with self._lock:
                self._analysis_closing = False
            self.get_logger().warn(f"Final offline submap refinement failed: {repr(exc)}")

    def _process_window_snapshot(
        self,
        snapshot: tuple[list[ScanRecord], list[ImuRecord], list[OdomRecord], int, int],
        *,
        final_window: bool,
        close_reason: str,
    ) -> None:
        (
            window_records,
            imu_records,
            seed_records,
            window_start_scan_index,
            window_end_scan_index,
        ) = snapshot
        start_perf = time.perf_counter()
        initial_poses, seed_available = self._build_initial_pose_chain(
            window_records,
            imu_records,
            seed_records,
        )
        overlap_anchor_points = self._build_overlap_anchor_points(window_start_scan_index)
        reference_deltas = self._build_reference_deltas(
            window_records,
            initial_poses,
            imu_records,
            seed_available,
        )
        refined_poses = initial_poses.copy()
        for _ in range(3):
            correspondences, qualities = self._build_correspondences(
                window_records,
                refined_poses,
                overlap_anchor_points,
            )
            updated_poses = self._refine_window_poses(
                window_records,
                refined_poses,
                initial_poses,
                imu_records,
                seed_available,
                correspondences,
                qualities,
                reference_deltas,
            )
            translation_update_m = float(
                np.max(np.linalg.norm(updated_poses[:, :2] - refined_poses[:, :2], axis=1))
            )
            yaw_update_rad = float(
                np.max(
                    np.abs(
                        [
                            _wrap_angle(float(updated_poses[index, 2] - refined_poses[index, 2]))
                            for index in range(updated_poses.shape[0])
                        ]
                    )
                )
            )
            refined_poses = updated_poses
            if translation_update_m < 0.015 and yaw_update_rad < 0.02:
                break
        correspondences, qualities = self._build_correspondences(
            window_records,
            refined_poses,
            overlap_anchor_points,
        )
        inter_window_alignment = self._estimate_inter_window_alignment(
            window_records,
            refined_poses,
            qualities,
        )
        if inter_window_alignment.applied:
            refined_poses = self._apply_world_delta_to_poses(
                refined_poses,
                inter_window_alignment.applied_delta_xyyaw,
            )
            correspondences, qualities = self._build_correspondences(
                window_records,
                refined_poses,
                overlap_anchor_points,
            )
        current_submap_points = self._build_window_submap_points(window_records, refined_poses)
        self._merge_window_results(window_records, refined_poses, qualities)
        accumulated_map_points, accumulated_path_poses = self._build_accumulated_outputs()
        latest_pose = (
            accumulated_path_poses[-1]
            if accumulated_path_poses.shape[0]
            else refined_poses[-1]
        )
        global_correction, global_correction_child_frame_id = self._compute_global_correction(
            window_records,
            refined_poses,
            seed_records,
        )
        duration_ms = 1000.0 * (time.perf_counter() - start_perf)
        status = self._status_payload(
            window_start_scan_index=window_start_scan_index,
            window_end_scan_index=window_end_scan_index,
            qualities=qualities,
            current_submap_points=current_submap_points,
            accumulated_map_points=accumulated_map_points,
            accumulated_path_poses=accumulated_path_poses,
            latest_pose=latest_pose,
            duration_ms=duration_ms,
            seed_available=seed_available,
            global_correction=global_correction,
            global_correction_child_frame_id=global_correction_child_frame_id,
            inter_window_alignment=inter_window_alignment,
            final_window=final_window,
            analysis_closed=final_window,
            close_reason=close_reason,
        )
        saved_output_dir = ""
        if final_window:
            saved_output_dir = self._save_final_outputs(
                map_points=accumulated_map_points,
                path_poses=accumulated_path_poses,
                status=status,
                close_reason=close_reason,
            )
            if saved_output_dir:
                status["saved_output_dir"] = saved_output_dir
        with self._lock:
            self._latest_submap_points = current_submap_points
            self._latest_map_points = accumulated_map_points
            self._latest_path_poses = accumulated_path_poses
            self._latest_pose = latest_pose
            self._latest_global_correction = global_correction
            self._latest_global_correction_child_frame_id = global_correction_child_frame_id
            self._latest_status.data = json.dumps(
                self._json_ready(status),
                separators=(",", ":"),
            )
            self._processed_window_count += 1
            self._next_window_start_index = max(
                0,
                int(window_end_scan_index) - self._window_overlap_count + 1,
            )
            if final_window:
                self._analysis_closed = True
                self._analysis_closing = False
                self._closed_reason = close_reason
                self._saved_output_dir = saved_output_dir
        if final_window:
            self.get_logger().info(
                "Offline analysis closed with final window "
                f"{window_start_scan_index}-{window_end_scan_index}"
            )

    def _window_snapshot(
        self,
    ) -> tuple[
        list[ScanRecord],
        list[ImuRecord],
        list[OdomRecord],
        int,
        int,
    ] | None:
        with self._lock:
            scan_records = list(self._scan_records)
            if len(scan_records) < self._initial_scan_count:
                return None
            oldest_index = int(scan_records[0].scan_index)
            newest_index = int(scan_records[-1].scan_index)
            window_start = max(self._next_window_start_index, oldest_index)
            if window_start == 0:
                window_end = min(newest_index, window_start + self._window_scan_count - 1)
            else:
                if newest_index < (window_start + self._window_scan_count - 1):
                    return None
                window_end = window_start + self._window_scan_count - 1
            window_records = [
                record
                for record in scan_records
                if window_start <= int(record.scan_index) <= window_end
            ]
            if len(window_records) < self._initial_scan_count:
                return None
            imu_records = list(self._imu_records)
            seed_records = list(self._odom_records)
            seed_status_records = list(self._seed_status_records)
        seed_gate_ok, _ = self._seed_status_allows_window(window_records, seed_status_records)
        if not seed_gate_ok:
            return None
        t_start = float(window_records[0].stamp_s)
        t_end = float(window_records[-1].stamp_s)
        imu_window = [record for record in imu_records if (t_start - 0.2) <= record.stamp_s <= (t_end + 0.2)]
        odom_window = [record for record in seed_records if (t_start - 0.5) <= record.stamp_s <= (t_end + 0.5)]
        return (
            window_records,
            imu_window,
            odom_window,
            int(window_records[0].scan_index),
            int(window_records[-1].scan_index),
        )

    def _final_window_snapshot(
        self,
    ) -> tuple[
        tuple[list[ScanRecord], list[ImuRecord], list[OdomRecord], int, int] | None,
        str,
    ]:
        with self._lock:
            scan_records = list(self._scan_records)
            if not scan_records:
                return None, "no_scan_records"
            oldest_index = int(scan_records[0].scan_index)
            newest_index = int(scan_records[-1].scan_index)
            latest_refined_index = max(self._refined_records) if self._refined_records else -1
            requested_end_index = (
                self._manual_idle_start_scan_index
                if self._manual_idle_start_scan_index is not None
                else self._last_manual_motion_scan_index
            )
            if requested_end_index is None or requested_end_index < oldest_index:
                requested_end_index = newest_index
            final_end = min(newest_index, max(oldest_index, int(requested_end_index)))
            if latest_refined_index >= final_end:
                return None, "no_unclosed_window"

            window_start = max(self._next_window_start_index, oldest_index)
            if window_start > final_end:
                window_start = max(oldest_index, final_end - self._final_min_scan_count + 1)
            if (final_end - window_start + 1) > self._window_scan_count:
                window_start = max(oldest_index, final_end - self._window_scan_count + 1)

            window_records = [
                record
                for record in scan_records
                if window_start <= int(record.scan_index) <= final_end
            ]
            if len(window_records) < self._final_min_scan_count:
                fallback_start = max(oldest_index, final_end - self._final_min_scan_count + 1)
                window_records = [
                    record
                    for record in scan_records
                    if fallback_start <= int(record.scan_index) <= final_end
                ]
            if len(window_records) < self._final_min_scan_count:
                return (
                    None,
                    "insufficient_final_scans="
                    f"{len(window_records)}/{self._final_min_scan_count}",
                )

            imu_records = list(self._imu_records)
            seed_records = list(self._odom_records)

        t_start = float(window_records[0].stamp_s)
        t_end = float(window_records[-1].stamp_s)
        imu_window = [
            record
            for record in imu_records
            if (t_start - 0.2) <= float(record.stamp_s) <= (t_end + 0.2)
        ]
        odom_window = [
            record
            for record in seed_records
            if (t_start - 0.5) <= float(record.stamp_s) <= (t_end + 0.5)
        ]
        return (
            (
                window_records,
                imu_window,
                odom_window,
                int(window_records[0].scan_index),
                int(window_records[-1].scan_index),
            ),
            "",
        )

    def _finalize_without_new_window(self, *, close_reason: str, skip_reason: str) -> None:
        with self._lock:
            map_points = self._latest_map_points.copy()
            submap_point_count = int(self._latest_submap_points.shape[0])
            path_poses = self._latest_path_poses.copy()
            latest_pose = self._latest_pose.copy()
            processed_window_count = int(self._processed_window_count)
        initial_pose = (
            path_poses[0].copy()
            if path_poses.shape[0]
            else latest_pose.copy()
        )
        final_pose = (
            path_poses[-1].copy()
            if path_poses.shape[0]
            else latest_pose.copy()
        )
        status: dict[str, object] = {
            "state": "analysis_closed",
            "replay_mode": self._replay_mode,
            "seed_odom_topic": self._seed_odom_topic,
            "processed_window_count": processed_window_count,
            "final_window": False,
            "analysis_closed": True,
            "close_reason": close_reason,
            "skip_final_window_reason": skip_reason,
            "manual_idle": {
                "enabled": bool(self._enable_manual_idle_finalize),
                "timeout_s": float(self._manual_idle_timeout_s),
                "motion_linear_deadband_mps": float(self._manual_motion_linear_deadband_mps),
                "status_topic": self._manual_status_topic,
            },
            "outputs": {
                "current_submap_point_count": submap_point_count,
                "accumulated_map_point_count": int(map_points.shape[0]),
                "accumulated_path_pose_count": int(path_poses.shape[0]),
            },
            "map_persistence": self._latest_map_persistence_stats,
            "initial_pose": self._pose_payload(initial_pose),
            "latest_pose": self._pose_payload(final_pose),
            "final_pose": self._pose_payload(final_pose),
            "save_on_finalize": bool(self._save_on_finalize),
        }
        saved_output_dir = self._save_final_outputs(
            map_points=map_points,
            path_poses=path_poses,
            status=status,
            close_reason=close_reason,
        )
        if saved_output_dir:
            status["saved_output_dir"] = saved_output_dir
        with self._lock:
            self._latest_status.data = json.dumps(
                self._json_ready(status),
                separators=(",", ":"),
            )
            self._analysis_closed = True
            self._analysis_closing = False
            self._closed_reason = close_reason
            self._saved_output_dir = saved_output_dir
        self.get_logger().info(
            f"Offline analysis closed without final window ({skip_reason})"
        )

    def _seed_status_allows_window(
        self,
        window_records: list[ScanRecord],
        seed_status_records: list[SeedStatusRecord],
    ) -> tuple[bool, str]:
        if not self._seed_status_topic:
            return True, ""
        if not seed_status_records:
            return False, "waiting for /apex/estimation/status"
        window_end_t_s = float(window_records[-1].stamp_s)
        now_wall_s = time.monotonic()
        relevant_records = [
            record
            for record in seed_status_records
            if math.isfinite(record.latest_pose_t_s)
            and record.latest_pose_t_s >= (window_end_t_s - 1.0)
            and (now_wall_s - record.received_wall_s) <= self._seed_status_timeout_s
        ]
        if not relevant_records:
            return False, "recent online seed status not available"
        candidate = max(relevant_records, key=lambda record: record.latest_pose_t_s)
        if candidate.state != "tracking":
            return False, f"online seed state={candidate.state or 'unknown'}"
        if candidate.latest_confidence == "low":
            return False, "online seed latest_pose.confidence=low"
        if (
            math.isfinite(candidate.median_submap_residual_m)
            and candidate.median_submap_residual_m > self._seed_max_median_submap_residual_m
        ):
            return (
                False,
                "online seed median_submap_residual_m="
                f"{candidate.median_submap_residual_m:.3f} exceeds "
                f"{self._seed_max_median_submap_residual_m:.3f}",
            )
        return True, ""

    def _build_initial_pose_chain(
        self,
        window_records: list[ScanRecord],
        imu_records: list[ImuRecord],
        seed_records: list[OdomRecord],
    ) -> tuple[np.ndarray, bool]:
        scan_times_s = np.asarray([record.stamp_s for record in window_records], dtype=np.float64)
        poses = np.zeros((len(window_records), 3), dtype=np.float64)
        seed_available = bool(seed_records)
        if seed_available:
            for index, stamp_s in enumerate(scan_times_s):
                interpolated_pose = self._interpolate_odom(seed_records, float(stamp_s))
                if interpolated_pose is None:
                    seed_available = False
                    break
                poses[index] = interpolated_pose
        if seed_available:
            return poses, True

        if self._refined_records:
            last_index = max(self._refined_records)
            poses[0] = np.asarray(self._refined_records[last_index]["pose_xyyaw"], dtype=np.float64)
        elif seed_records:
            poses[0] = seed_records[-1].pose_xyyaw.copy()
        else:
            poses[0] = np.zeros(3, dtype=np.float64)

        for index in range(1, len(window_records)):
            interval_start_s = float(scan_times_s[index - 1])
            interval_end_s = float(scan_times_s[index])
            yaw_delta = self._integrated_imu_yaw_delta(
                imu_records,
                interval_start_s,
                interval_end_s,
            )
            relative_motion = self._estimate_relative_scan_motion(
                window_records[index - 1].points_xy,
                window_records[index].points_xy,
                yaw_delta,
            )
            relative_delta = relative_motion.delta_pose_xyyaw.copy()
            relative_delta[2] = _wrap_angle(float((0.70 * yaw_delta) + (0.30 * relative_delta[2])))
            poses[index] = _compose_poses(poses[index - 1], relative_delta)
        return poses, False

    def _estimate_relative_scan_motion(
        self,
        previous_points_xy: np.ndarray,
        current_points_xy: np.ndarray,
        yaw_delta_guess_rad: float,
    ) -> RelativeMotionEstimate:
        if previous_points_xy.shape[0] < 12 or current_points_xy.shape[0] < 12:
            return RelativeMotionEstimate(
                delta_pose_xyyaw=np.asarray(
                    [0.0, 0.0, _wrap_angle(float(yaw_delta_guess_rad))],
                    dtype=np.float64,
                ),
                valid_match_count=0,
                median_residual_m=float("nan"),
            )

        target_points = _voxel_downsample(previous_points_xy, self._submap_voxel_size_m)
        source_points = _voxel_downsample(current_points_xy, self._submap_voxel_size_m)
        if target_points.shape[0] < 12 or source_points.shape[0] < 12:
            return RelativeMotionEstimate(
                delta_pose_xyyaw=np.asarray(
                    [0.0, 0.0, _wrap_angle(float(yaw_delta_guess_rad))],
                    dtype=np.float64,
                ),
                valid_match_count=0,
                median_residual_m=float("nan"),
            )

        relative_pose = np.asarray(
            [0.0, 0.0, _wrap_angle(float(yaw_delta_guess_rad))],
            dtype=np.float64,
        )
        tree = cKDTree(target_points)
        min_required_matches = max(10, min(40, source_points.shape[0] // 6))
        last_valid_match_count = 0
        last_median_residual_m = float("nan")
        for _ in range(6):
            transformed_source = _transform_points(source_points, relative_pose)
            distances_m, nearest_indexes = tree.query(
                transformed_source,
                distance_upper_bound=max(self._max_correspondence_m * 1.25, 0.40),
            )
            valid_mask = np.isfinite(distances_m) & (
                distances_m < max(self._max_correspondence_m * 1.25, 0.40)
            )
            last_valid_match_count = int(np.count_nonzero(valid_mask))
            if last_valid_match_count > 0:
                last_median_residual_m = float(np.median(distances_m[valid_mask]))
            if last_valid_match_count < min_required_matches:
                break
            incremental_pose = _best_fit_rigid_transform_2d(
                transformed_source[valid_mask],
                target_points[nearest_indexes[valid_mask]],
            )
            relative_pose = _compose_poses(incremental_pose, relative_pose)
            relative_pose[2] = _wrap_angle(float((0.35 * yaw_delta_guess_rad) + (0.65 * relative_pose[2])))

        max_translation_m = 0.75
        relative_pose[0] = float(np.clip(relative_pose[0], -max_translation_m, max_translation_m))
        relative_pose[1] = float(np.clip(relative_pose[1], -max_translation_m, max_translation_m))
        relative_pose[2] = _wrap_angle(float(relative_pose[2]))
        return RelativeMotionEstimate(
            delta_pose_xyyaw=relative_pose,
            valid_match_count=last_valid_match_count,
            median_residual_m=last_median_residual_m,
        )

    def _interpolate_odom(
        self,
        seed_records: list[OdomRecord],
        stamp_s: float,
    ) -> np.ndarray | None:
        stamps = [record.stamp_s for record in seed_records]
        if not stamps:
            return None
        position = bisect_left(stamps, stamp_s)
        if position <= 0:
            if abs(stamps[0] - stamp_s) <= 0.5:
                return seed_records[0].pose_xyyaw.copy()
            return None
        if position >= len(stamps):
            if abs(stamps[-1] - stamp_s) <= 0.5:
                return seed_records[-1].pose_xyyaw.copy()
            return None
        previous_record = seed_records[position - 1]
        next_record = seed_records[position]
        dt_s = max(1.0e-6, float(next_record.stamp_s - previous_record.stamp_s))
        alpha = float((stamp_s - previous_record.stamp_s) / dt_s)
        interp_xy = ((1.0 - alpha) * previous_record.pose_xyyaw[:2]) + (
            alpha * next_record.pose_xyyaw[:2]
        )
        yaw_delta = _wrap_angle(float(next_record.pose_xyyaw[2] - previous_record.pose_xyyaw[2]))
        interp_yaw = _wrap_angle(float(previous_record.pose_xyyaw[2] + (alpha * yaw_delta)))
        return np.asarray([float(interp_xy[0]), float(interp_xy[1]), interp_yaw], dtype=np.float64)

    def _track_geometry_residuals(
        self,
        poses_xyyaw: np.ndarray,
        qualities: list[ScanQuality],
    ) -> list[np.ndarray]:
        if (
            not self._use_track_geometry_prior
            or self._track_geometry_tree is None
            or self._track_geometry_points.shape[0] < 2
            or self._track_geometry_weight <= 0.0
        ):
            return []
        pose_points = poses_xyyaw[:, :2]
        distances_m, indexes = self._track_geometry_tree.query(pose_points, k=1)
        residuals: list[np.ndarray] = []
        for index, nearest_index in enumerate(indexes):
            confidence = qualities[index].confidence if index < len(qualities) else "low"
            if confidence == "high":
                quality_scale = 0.05
            elif confidence == "medium":
                quality_scale = 0.25
            else:
                quality_scale = 0.55
            weight = self._track_geometry_weight * quality_scale
            if weight <= 0.0:
                continue
            nearest_point = self._track_geometry_points[int(nearest_index)]
            delta_xy = pose_points[index] - nearest_point
            residuals.append((weight * delta_xy).astype(np.float64))
            if math.isfinite(float(distances_m[index])) and float(distances_m[index]) <= 0.35:
                continue
            residuals.append(np.asarray([0.20 * weight * float(np.linalg.norm(delta_xy))], dtype=np.float64))
        return residuals

    def _build_reference_deltas(
        self,
        window_records: list[ScanRecord],
        initial_poses: np.ndarray,
        imu_records: list[ImuRecord],
        seed_available: bool,
    ) -> np.ndarray:
        if len(window_records) < 2:
            return np.empty((0, 3), dtype=np.float64)
        reference_deltas = np.zeros((len(window_records) - 1, 3), dtype=np.float64)
        for index in range(1, len(window_records)):
            seed_delta = np.asarray(
                [
                    float(initial_poses[index, 0] - initial_poses[index - 1, 0]),
                    float(initial_poses[index, 1] - initial_poses[index - 1, 1]),
                    _wrap_angle(float(initial_poses[index, 2] - initial_poses[index - 1, 2])),
                ],
                dtype=np.float64,
            )
            yaw_delta = self._integrated_imu_yaw_delta(
                imu_records,
                float(window_records[index - 1].stamp_s),
                float(window_records[index].stamp_s),
            )
            relative_motion = self._estimate_relative_scan_motion(
                window_records[index - 1].points_xy,
                window_records[index].points_xy,
                yaw_delta,
            )
            scan_delta = relative_motion.delta_pose_xyyaw.copy()
            if seed_available:
                if (
                    relative_motion.valid_match_count >= 18
                    and math.isfinite(relative_motion.median_residual_m)
                    and relative_motion.median_residual_m <= 0.10
                ):
                    blend_weight = 0.65
                elif (
                    relative_motion.valid_match_count >= 12
                    and math.isfinite(relative_motion.median_residual_m)
                    and relative_motion.median_residual_m <= 0.16
                ):
                    blend_weight = 0.45
                else:
                    blend_weight = 0.0
                blended_delta = seed_delta.copy()
                blended_delta[:2] = (
                    ((1.0 - blend_weight) * seed_delta[:2]) + (blend_weight * scan_delta[:2])
                )
                blended_delta[2] = _wrap_angle(
                    float(((1.0 - blend_weight) * seed_delta[2]) + (blend_weight * scan_delta[2]))
                )
                reference_deltas[index - 1] = blended_delta
            else:
                scan_delta[2] = _wrap_angle(float((0.70 * yaw_delta) + (0.30 * scan_delta[2])))
                reference_deltas[index - 1] = scan_delta
        return reference_deltas

    def _integrated_imu_yaw_delta(
        self,
        imu_records: list[ImuRecord],
        start_s: float,
        end_s: float,
    ) -> float:
        if end_s <= start_s:
            return 0.0
        relevant = [record for record in imu_records if start_s <= record.stamp_s <= end_s]
        if not relevant:
            return 0.0
        if len(relevant) == 1:
            return float(relevant[0].yaw_rate_rps * (end_s - start_s))
        delta = 0.0
        previous = relevant[0]
        for current in relevant[1:]:
            dt_s = max(0.0, float(current.stamp_s - previous.stamp_s))
            delta += 0.5 * float(previous.yaw_rate_rps + current.yaw_rate_rps) * dt_s
            previous = current
        tail_dt_s = max(0.0, float(end_s - relevant[-1].stamp_s))
        delta += float(relevant[-1].yaw_rate_rps) * tail_dt_s
        return delta

    def _build_overlap_anchor_points(self, window_start_scan_index: int) -> np.ndarray:
        if self._window_overlap_count <= 0 or not self._refined_records:
            return np.empty((0, 2), dtype=np.float64)
        overlap_indexes = range(
            max(0, window_start_scan_index - self._window_overlap_count),
            window_start_scan_index,
        )
        clouds: list[np.ndarray] = []
        for scan_index in overlap_indexes:
            record = self._refined_records.get(scan_index)
            if record is None:
                continue
            confidence = str(record.get("confidence", "low"))
            if confidence == "low":
                continue
            points_xy = np.asarray(record["points_xy"], dtype=np.float64)
            pose_xyyaw = np.asarray(record["pose_xyyaw"], dtype=np.float64)
            clouds.append(_transform_points(points_xy, pose_xyyaw))
        if not clouds:
            return np.empty((0, 2), dtype=np.float64)
        return _voxel_downsample(np.vstack(clouds), self._submap_voxel_size_m)

    def _build_correspondences(
        self,
        window_records: list[ScanRecord],
        initial_poses: np.ndarray,
        overlap_anchor_points: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[ScanQuality]]:
        correspondences: list[tuple[np.ndarray, np.ndarray]] = []
        qualities: list[ScanQuality] = []
        transformed_points = [
            _transform_points(record.points_xy, initial_poses[index])
            for index, record in enumerate(window_records)
        ]
        for index, record in enumerate(window_records):
            target_clouds: list[np.ndarray] = []
            if overlap_anchor_points.size:
                target_clouds.append(overlap_anchor_points)
            prior_start = max(0, index - self._submap_window_scans)
            for candidate_index in range(prior_start, index):
                target_clouds.append(transformed_points[candidate_index])
            if not target_clouds:
                correspondences.append(
                    (
                        np.empty((0, 2), dtype=np.float64),
                        np.empty((0, 2), dtype=np.float64),
                    )
                )
                qualities.append(
                    ScanQuality(
                        scan_index=int(record.scan_index),
                        median_residual_m=float("nan"),
                        valid_correspondence_count=0,
                        confidence="low",
                    )
                )
                continue
            target_points = _voxel_downsample(np.vstack(target_clouds), self._submap_voxel_size_m)
            if target_points.shape[0] < 6 or transformed_points[index].shape[0] == 0:
                correspondences.append(
                    (
                        np.empty((0, 2), dtype=np.float64),
                        np.empty((0, 2), dtype=np.float64),
                    )
                )
                qualities.append(
                    ScanQuality(
                        scan_index=int(record.scan_index),
                        median_residual_m=float("nan"),
                        valid_correspondence_count=0,
                        confidence="low",
                    )
                )
                continue
            tree = cKDTree(target_points)
            distances_m, nearest_indexes = tree.query(
                transformed_points[index],
                distance_upper_bound=self._max_correspondence_m,
            )
            valid_mask = np.isfinite(distances_m) & (distances_m < self._max_correspondence_m)
            source_points = record.points_xy[valid_mask]
            matched_points = target_points[nearest_indexes[valid_mask]]
            valid_count = int(source_points.shape[0])
            median_residual_m = (
                float(np.median(distances_m[valid_mask]))
                if valid_count
                else float("nan")
            )
            if valid_count >= 24 and median_residual_m < 0.08:
                confidence = "high"
            elif valid_count >= 12 and median_residual_m < 0.16:
                confidence = "medium"
            else:
                confidence = "low"
            correspondences.append((source_points.astype(np.float64), matched_points.astype(np.float64)))
            qualities.append(
                ScanQuality(
                    scan_index=int(record.scan_index),
                    median_residual_m=median_residual_m,
                    valid_correspondence_count=valid_count,
                    confidence=confidence,
                )
            )
        return correspondences, qualities

    def _refine_window_poses(
        self,
        window_records: list[ScanRecord],
        current_poses: np.ndarray,
        initial_poses: np.ndarray,
        imu_records: list[ImuRecord],
        seed_available: bool,
        correspondences: list[tuple[np.ndarray, np.ndarray]],
        qualities: list[ScanQuality],
        reference_deltas: np.ndarray,
    ) -> np.ndarray:
        scan_times_s = np.asarray([record.stamp_s for record in window_records], dtype=np.float64)
        initial_flat = current_poses.reshape(-1)
        sequential_deltas = reference_deltas.copy()

        def _quality_weight(label: str) -> float:
            if label == "high":
                return 1.0
            if label == "medium":
                return 0.45
            return 0.0

        def _residual_fn(flat_poses: np.ndarray) -> np.ndarray:
            poses = flat_poses.reshape((-1, 3))
            residuals: list[np.ndarray] = []
            for index, (source_points, matched_points) in enumerate(correspondences):
                weight = _quality_weight(qualities[index].confidence)
                if weight <= 0.0 or source_points.size == 0:
                    continue
                transformed = _transform_points(source_points, poses[index])
                delta = transformed - matched_points
                residuals.append((weight * 1.6) * delta.reshape(-1))
            for index, pose in enumerate(poses):
                pose_delta = pose - initial_poses[index]
                pose_delta[2] = _wrap_angle(float(pose_delta[2]))
                if seed_available:
                    if index == 0:
                        residuals.append(
                            np.asarray(
                                [
                                    0.60 * float(pose_delta[0]),
                                    0.60 * float(pose_delta[1]),
                                    1.00 * float(pose_delta[2]),
                                ],
                                dtype=np.float64,
                            )
                        )
                else:
                    residuals.append(
                        np.asarray([0.08 * float(pose_delta[2])], dtype=np.float64)
                    )
            for index in range(1, len(window_records)):
                delta_pose = poses[index] - poses[index - 1]
                delta_pose[2] = _wrap_angle(float(delta_pose[2]))
                yaw_from_imu = self._integrated_imu_yaw_delta(
                    imu_records,
                    float(scan_times_s[index - 1]),
                    float(scan_times_s[index]),
                )
                if seed_available:
                    reference_delta = sequential_deltas[index - 1].copy()
                    residuals.append(
                        np.asarray(
                            [
                                0.14 * float(delta_pose[0] - reference_delta[0]),
                                0.14 * float(delta_pose[1] - reference_delta[1]),
                                0.35 * _wrap_angle(float(delta_pose[2] - reference_delta[2])),
                            ],
                            dtype=np.float64,
                        )
                    )
                    residuals.append(
                        np.asarray(
                            [0.55 * _wrap_angle(float(delta_pose[2] - yaw_from_imu))],
                            dtype=np.float64,
                        )
                    )
                else:
                    reference_delta = sequential_deltas[index - 1].copy()
                    residuals.append(
                        np.asarray(
                            [
                                0.18 * float(delta_pose[0] - reference_delta[0]),
                                0.18 * float(delta_pose[1] - reference_delta[1]),
                                0.70 * _wrap_angle(float(delta_pose[2] - yaw_from_imu)),
                            ],
                            dtype=np.float64,
                        )
                    )
            residuals.extend(self._track_geometry_residuals(poses, qualities))
            if not residuals:
                return np.zeros(1, dtype=np.float64)
            return np.concatenate(residuals)

        result = least_squares(
            _residual_fn,
            initial_flat,
            method="trf",
            max_nfev=40,
            loss="soft_l1",
            f_scale=0.15,
        )
        refined = result.x.reshape((-1, 3))
        refined[:, 2] = np.asarray([_wrap_angle(value) for value in refined[:, 2]], dtype=np.float64)
        return refined

    def _inter_window_result(
        self,
        *,
        applied: bool,
        reason: str,
        overlap_scan_count: int = 0,
        paired_point_count: int = 0,
        raw_delta_xyyaw: np.ndarray | None = None,
        applied_delta_xyyaw: np.ndarray | None = None,
    ) -> InterWindowAlignmentResult:
        return InterWindowAlignmentResult(
            enabled=bool(self._enable_inter_window_alignment),
            applied=bool(applied),
            reason=reason,
            overlap_scan_count=int(overlap_scan_count),
            paired_point_count=int(paired_point_count),
            raw_delta_xyyaw=(
                np.asarray(raw_delta_xyyaw, dtype=np.float64).copy()
                if raw_delta_xyyaw is not None
                else np.zeros(3, dtype=np.float64)
            ),
            applied_delta_xyyaw=(
                np.asarray(applied_delta_xyyaw, dtype=np.float64).copy()
                if applied_delta_xyyaw is not None
                else np.zeros(3, dtype=np.float64)
            ),
        )

    def _estimate_inter_window_alignment(
        self,
        window_records: list[ScanRecord],
        refined_poses: np.ndarray,
        qualities: list[ScanQuality],
    ) -> InterWindowAlignmentResult:
        if not self._enable_inter_window_alignment:
            return self._inter_window_result(applied=False, reason="disabled")
        if not self._refined_records:
            return self._inter_window_result(applied=False, reason="no_prior_window")

        source_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        overlap_scan_count = 0
        for index, record in enumerate(window_records):
            prior_record = self._refined_records.get(int(record.scan_index))
            if prior_record is None:
                continue
            if str(prior_record.get("confidence", "low")) == "low":
                continue
            if index < len(qualities) and qualities[index].confidence == "low":
                continue
            points_xy = np.asarray(record.points_xy, dtype=np.float64)
            if points_xy.shape[0] < 3:
                continue
            per_scan_limit = 120
            stride = max(1, int(math.ceil(points_xy.shape[0] / per_scan_limit)))
            sampled_points = points_xy[::stride]
            previous_pose = np.asarray(prior_record["pose_xyyaw"], dtype=np.float64)
            source_chunks.append(_transform_points(sampled_points, refined_poses[index]))
            target_chunks.append(_transform_points(sampled_points, previous_pose))
            overlap_scan_count += 1

        if overlap_scan_count < self._inter_window_min_overlap_scans:
            return self._inter_window_result(
                applied=False,
                reason="insufficient_overlap_scans",
                overlap_scan_count=overlap_scan_count,
            )
        if not source_chunks or not target_chunks:
            return self._inter_window_result(
                applied=False,
                reason="no_overlap_points",
                overlap_scan_count=overlap_scan_count,
            )

        source_xy = np.vstack(source_chunks)
        target_xy = np.vstack(target_chunks)
        if source_xy.shape[0] > self._inter_window_max_points:
            stride = max(1, int(math.ceil(source_xy.shape[0] / self._inter_window_max_points)))
            source_xy = source_xy[::stride]
            target_xy = target_xy[::stride]
        paired_point_count = int(source_xy.shape[0])
        if paired_point_count < self._inter_window_min_points:
            return self._inter_window_result(
                applied=False,
                reason="insufficient_paired_points",
                overlap_scan_count=overlap_scan_count,
                paired_point_count=paired_point_count,
            )

        raw_delta_xyyaw = _best_fit_rigid_transform_2d(source_xy, target_xy)
        if not np.all(np.isfinite(raw_delta_xyyaw)):
            return self._inter_window_result(
                applied=False,
                reason="non_finite_delta",
                overlap_scan_count=overlap_scan_count,
                paired_point_count=paired_point_count,
                raw_delta_xyyaw=raw_delta_xyyaw,
            )

        raw_translation_m = float(np.linalg.norm(raw_delta_xyyaw[:2]))
        raw_yaw_rad = abs(float(raw_delta_xyyaw[2]))
        if raw_translation_m > self._inter_window_max_translation_m:
            return self._inter_window_result(
                applied=False,
                reason="translation_limit",
                overlap_scan_count=overlap_scan_count,
                paired_point_count=paired_point_count,
                raw_delta_xyyaw=raw_delta_xyyaw,
            )
        if raw_yaw_rad > self._inter_window_max_yaw_rad:
            return self._inter_window_result(
                applied=False,
                reason="yaw_limit",
                overlap_scan_count=overlap_scan_count,
                paired_point_count=paired_point_count,
                raw_delta_xyyaw=raw_delta_xyyaw,
            )

        applied_delta_xyyaw = raw_delta_xyyaw.copy()
        applied_delta_xyyaw[:2] *= self._inter_window_alignment_gain
        applied_delta_xyyaw[2] = _wrap_angle(
            float(applied_delta_xyyaw[2]) * self._inter_window_alignment_gain
        )
        return self._inter_window_result(
            applied=True,
            reason="applied",
            overlap_scan_count=overlap_scan_count,
            paired_point_count=paired_point_count,
            raw_delta_xyyaw=raw_delta_xyyaw,
            applied_delta_xyyaw=applied_delta_xyyaw,
        )

    def _apply_world_delta_to_poses(
        self,
        poses_xyyaw: np.ndarray,
        delta_xyyaw: np.ndarray,
    ) -> np.ndarray:
        poses = np.asarray(poses_xyyaw, dtype=np.float64)
        delta = np.asarray(delta_xyyaw, dtype=np.float64)
        corrected = poses.copy()
        rotation = _rotation_matrix(float(delta[2]))
        corrected[:, :2] = (poses[:, :2] @ rotation.T) + delta[:2]
        corrected[:, 2] = np.asarray(
            [_wrap_angle(float(yaw) + float(delta[2])) for yaw in poses[:, 2]],
            dtype=np.float64,
        )
        return corrected

    def _build_window_submap_points(
        self,
        window_records: list[ScanRecord],
        refined_poses: np.ndarray,
    ) -> np.ndarray:
        clouds = [
            _transform_points(record.points_xy, refined_poses[index])
            for index, record in enumerate(window_records)
            if record.points_xy.size
        ]
        if not clouds:
            return np.empty((0, 2), dtype=np.float64)
        return _voxel_downsample(np.vstack(clouds), self._submap_voxel_size_m)

    def _merge_window_results(
        self,
        window_records: list[ScanRecord],
        refined_poses: np.ndarray,
        qualities: list[ScanQuality],
    ) -> None:
        for index, record in enumerate(window_records):
            self._refined_records[int(record.scan_index)] = {
                "stamp_s": float(record.stamp_s),
                "points_xy": record.points_xy.copy(),
                "pose_xyyaw": refined_poses[index].copy(),
                "confidence": qualities[index].confidence,
                "median_residual_m": float(qualities[index].median_residual_m),
                "valid_correspondence_count": int(qualities[index].valid_correspondence_count),
            }

    def _filter_map_clouds_by_persistence(
        self,
        map_clouds: list[tuple[int, np.ndarray]],
    ) -> tuple[np.ndarray, dict[str, object]]:
        raw_parts = [points_xy for _, points_xy in map_clouds if points_xy.size]
        raw_points = np.vstack(raw_parts) if raw_parts else np.empty((0, 2), dtype=np.float64)
        raw_point_count = int(raw_points.shape[0])
        if (
            not self._map_persistence_filter
            or self._map_persistence_min_observations <= 1
            or self._map_persistence_voxel_size_m <= 1.0e-6
            or raw_point_count == 0
        ):
            return raw_points, {
                "enabled": bool(self._map_persistence_filter),
                "voxel_size_m": float(self._map_persistence_voxel_size_m),
                "min_observations": int(self._map_persistence_min_observations),
                "confidence_threshold": float(
                    max(1.0, float(self._map_persistence_min_observations) - 0.5)
                ),
                "decay_per_missed_scan": float(self._map_persistence_decay_per_missed_scan),
                "max_decay_gap_scans": int(self._map_persistence_max_decay_gap_scans),
                "raw_point_count": raw_point_count,
                "kept_point_count": raw_point_count,
                "raw_voxel_count": 0,
                "kept_voxel_count": 0,
                "dropped_voxel_count": 0,
            }

        voxel_size_m = float(self._map_persistence_voxel_size_m)
        min_observations = int(self._map_persistence_min_observations)
        confidence_threshold = max(1.0, float(min_observations) - 0.5)
        decay_per_missed = float(self._map_persistence_decay_per_missed_scan)
        max_decay_gap = int(self._map_persistence_max_decay_gap_scans)
        states: dict[tuple[int, int], dict[str, object]] = {}
        points_by_voxel: dict[tuple[int, int], list[np.ndarray]] = {}

        for scan_index, points_xy in map_clouds:
            if points_xy.size == 0:
                continue
            voxel_xy = np.floor(points_xy / voxel_size_m).astype(np.int64)
            unique_voxels, inverse = np.unique(voxel_xy, axis=0, return_inverse=True)
            for unique_index, voxel in enumerate(unique_voxels):
                voxel_key = (int(voxel[0]), int(voxel[1]))
                state = states.get(voxel_key)
                if state is None:
                    confidence = 0.0
                    observations = 0
                    max_confidence = 0.0
                    last_scan_index = int(scan_index)
                else:
                    confidence = float(state["confidence"])
                    observations = int(state["observations"])
                    max_confidence = float(state["max_confidence"])
                    last_scan_index = int(state["last_scan_index"])
                    missed_scans = max(0, int(scan_index) - last_scan_index - 1)
                    if missed_scans > 0 and decay_per_missed > 0.0:
                        decay_gap = missed_scans if max_decay_gap <= 0 else min(missed_scans, max_decay_gap)
                        confidence = max(0.0, confidence - (decay_per_missed * float(decay_gap)))

                confidence += 1.0
                observations += 1
                max_confidence = max(max_confidence, confidence)
                states[voxel_key] = {
                    "confidence": confidence,
                    "observations": observations,
                    "max_confidence": max_confidence,
                    "last_scan_index": int(scan_index),
                }
                voxel_points = points_xy[inverse == unique_index]
                if voxel_points.size:
                    points_by_voxel.setdefault(voxel_key, []).append(voxel_points.copy())

        kept_voxels = {
            voxel_key
            for voxel_key, state in states.items()
            if int(state["observations"]) >= min_observations
            and float(state["max_confidence"]) >= confidence_threshold
        }
        kept_parts: list[np.ndarray] = []
        for voxel_key in sorted(kept_voxels):
            voxel_chunks = points_by_voxel.get(voxel_key, [])
            if voxel_chunks:
                kept_parts.append(np.vstack(voxel_chunks))
        kept_points = (
            np.vstack(kept_parts)
            if kept_parts
            else np.empty((0, 2), dtype=np.float64)
        )
        return kept_points, {
            "enabled": True,
            "voxel_size_m": voxel_size_m,
            "min_observations": min_observations,
            "confidence_threshold": confidence_threshold,
            "decay_per_missed_scan": decay_per_missed,
            "max_decay_gap_scans": max_decay_gap,
            "raw_point_count": raw_point_count,
            "kept_point_count": int(kept_points.shape[0]),
            "raw_voxel_count": int(len(states)),
            "kept_voxel_count": int(len(kept_voxels)),
            "dropped_voxel_count": int(max(0, len(states) - len(kept_voxels))),
        }

    def _build_accumulated_outputs(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._refined_records:
            self._latest_map_persistence_stats = {
                "enabled": bool(self._map_persistence_filter),
                "raw_point_count": 0,
                "kept_point_count": 0,
                "raw_voxel_count": 0,
                "kept_voxel_count": 0,
                "dropped_voxel_count": 0,
            }
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        ordered_indexes = sorted(self._refined_records)
        map_clouds: list[tuple[int, np.ndarray]] = []
        path_poses: list[np.ndarray] = []
        for scan_index in ordered_indexes:
            record = self._refined_records[scan_index]
            pose_xyyaw = np.asarray(record["pose_xyyaw"], dtype=np.float64)
            path_poses.append(pose_xyyaw)
            if str(record["confidence"]) == "low":
                continue
            points_xy = np.asarray(record["points_xy"], dtype=np.float64)
            if points_xy.size == 0:
                continue
            map_clouds.append((int(scan_index), _transform_points(points_xy, pose_xyyaw)))
        persistent_points, persistence_stats = self._filter_map_clouds_by_persistence(map_clouds)
        accumulated_map = (
            _voxel_downsample(persistent_points, self._global_voxel_size_m)
            if persistent_points.size
            else np.empty((0, 2), dtype=np.float64)
        )
        persistence_stats["final_downsampled_point_count"] = int(accumulated_map.shape[0])
        self._latest_map_persistence_stats = persistence_stats
        accumulated_path = (
            np.vstack(path_poses).astype(np.float64)
            if path_poses
            else np.empty((0, 3), dtype=np.float64)
        )
        return accumulated_map, accumulated_path

    def _status_payload(
        self,
        *,
        window_start_scan_index: int,
        window_end_scan_index: int,
        qualities: list[ScanQuality],
        current_submap_points: np.ndarray,
        accumulated_map_points: np.ndarray,
        accumulated_path_poses: np.ndarray,
        latest_pose: np.ndarray,
        duration_ms: float,
        seed_available: bool,
        global_correction: np.ndarray,
        global_correction_child_frame_id: str,
        inter_window_alignment: InterWindowAlignmentResult,
        final_window: bool = False,
        analysis_closed: bool = False,
        close_reason: str = "",
    ) -> dict[str, object]:
        high_count = sum(1 for quality in qualities if quality.confidence == "high")
        medium_count = sum(1 for quality in qualities if quality.confidence == "medium")
        low_count = sum(1 for quality in qualities if quality.confidence == "low")
        valid_counts = [quality.valid_correspondence_count for quality in qualities]
        residuals = [
            quality.median_residual_m
            for quality in qualities
            if math.isfinite(quality.median_residual_m)
        ]
        return {
            "state": "analysis_closed" if analysis_closed else "window_refined",
            "replay_mode": self._replay_mode,
            "seed_odom_topic": self._seed_odom_topic,
            "seed_available": seed_available,
            "processed_window_count": int(self._processed_window_count + 1),
            "final_window": bool(final_window),
            "analysis_closed": bool(analysis_closed),
            "close_reason": str(close_reason),
            "manual_idle": {
                "enabled": bool(self._enable_manual_idle_finalize),
                "timeout_s": float(self._manual_idle_timeout_s),
                "motion_linear_deadband_mps": float(self._manual_motion_linear_deadband_mps),
                "status_topic": self._manual_status_topic,
            },
            "window": {
                "start_scan_index": int(window_start_scan_index),
                "end_scan_index": int(window_end_scan_index),
                "scan_count": len(qualities),
                "overlap_count": int(self._window_overlap_count),
            },
            "quality": {
                "high_count": int(high_count),
                "medium_count": int(medium_count),
                "low_count": int(low_count),
                "median_residual_m": float(np.median(np.asarray(residuals, dtype=np.float64)))
                if residuals
                else float("nan"),
                "median_valid_correspondence_count": float(
                    np.median(np.asarray(valid_counts, dtype=np.float64))
                )
                if valid_counts
                else 0.0,
            },
            "outputs": {
                "current_submap_point_count": int(current_submap_points.shape[0]),
                "accumulated_map_point_count": int(accumulated_map_points.shape[0]),
                "accumulated_path_pose_count": int(accumulated_path_poses.shape[0]),
            },
            "map_persistence": self._latest_map_persistence_stats,
            "latest_pose": {
                "x_m": float(latest_pose[0]),
                "y_m": float(latest_pose[1]),
                "yaw_rad": float(latest_pose[2]),
            },
            "initial_pose": self._pose_payload(
                accumulated_path_poses[0] if accumulated_path_poses.shape[0] else latest_pose
            ),
            "final_pose": self._pose_payload(latest_pose),
            "global_correction": {
                "enabled": bool(self._publish_global_correction),
                "child_frame_id": str(global_correction_child_frame_id),
                "x_m": float(global_correction[0]),
                "y_m": float(global_correction[1]),
                "yaw_rad": float(global_correction[2]),
                "translation_norm_m": float(np.linalg.norm(global_correction[:2])),
            },
            "inter_window_alignment": {
                "enabled": bool(inter_window_alignment.enabled),
                "applied": bool(inter_window_alignment.applied),
                "reason": str(inter_window_alignment.reason),
                "overlap_scan_count": int(inter_window_alignment.overlap_scan_count),
                "paired_point_count": int(inter_window_alignment.paired_point_count),
                "gain": float(self._inter_window_alignment_gain),
                "raw": {
                    "x_m": float(inter_window_alignment.raw_delta_xyyaw[0]),
                    "y_m": float(inter_window_alignment.raw_delta_xyyaw[1]),
                    "yaw_rad": float(inter_window_alignment.raw_delta_xyyaw[2]),
                    "translation_norm_m": float(
                        np.linalg.norm(inter_window_alignment.raw_delta_xyyaw[:2])
                    ),
                },
                "applied_delta": {
                    "x_m": float(inter_window_alignment.applied_delta_xyyaw[0]),
                    "y_m": float(inter_window_alignment.applied_delta_xyyaw[1]),
                    "yaw_rad": float(inter_window_alignment.applied_delta_xyyaw[2]),
                    "translation_norm_m": float(
                        np.linalg.norm(inter_window_alignment.applied_delta_xyyaw[:2])
                    ),
                },
            },
            "track_geometry_prior": {
                "enabled": bool(self._use_track_geometry_prior and self._track_geometry_tree is not None),
                "point_count": int(self._track_geometry_points.shape[0]),
                "weight": float(self._track_geometry_weight),
                "file": self._track_geometry_file,
            },
            "save_on_finalize": bool(self._save_on_finalize),
            "processing_ms": float(duration_ms),
        }

    def _pose_payload(self, pose_xyyaw: np.ndarray) -> dict[str, object]:
        pose = np.asarray(pose_xyyaw, dtype=np.float64)
        qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
        return {
            "frame_id": self._frame_id,
            "child_frame_id": self._child_frame_id,
            "x_m": float(pose[0]),
            "y_m": float(pose[1]),
            "yaw_rad": float(pose[2]),
            "orientation": {
                "x": qx,
                "y": qy,
                "z": qz,
                "w": qw,
            },
        }

    def _save_final_outputs(
        self,
        *,
        map_points: np.ndarray,
        path_poses: np.ndarray,
        status: dict[str, object],
        close_reason: str,
    ) -> str:
        if not self._save_on_finalize:
            return ""
        output_root = Path(
            self._save_output_dir or "/tmp/apex_offline_refined_maps"
        ).expanduser()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            base_name = time.strftime("offline_refined_%Y%m%d_%H%M%S")
            output_dir = output_root / base_name
            for suffix in range(1000):
                candidate = (
                    output_dir
                    if suffix == 0
                    else output_root / f"{base_name}_{suffix:03d}"
                )
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                output_dir = candidate
                break
            else:
                output_dir = output_root / f"{base_name}_{time.monotonic_ns()}"
                output_dir.mkdir(parents=True, exist_ok=False)

            map_xy = np.asarray(map_points, dtype=np.float64)
            if map_xy.size == 0:
                map_xy = np.empty((0, 2), dtype=np.float64)
            else:
                map_xy = map_xy.reshape((-1, 2))
            path_xyyaw = np.asarray(path_poses, dtype=np.float64)
            if path_xyyaw.size == 0:
                path_xyyaw = np.empty((0, 3), dtype=np.float64)
            else:
                path_xyyaw = path_xyyaw.reshape((-1, 3))

            np.savetxt(
                output_dir / "map_points_xy.csv",
                map_xy,
                delimiter=",",
                header="x_m,y_m",
                comments="",
            )
            indexed_path = (
                np.column_stack((np.arange(path_xyyaw.shape[0], dtype=np.int64), path_xyyaw))
                if path_xyyaw.shape[0]
                else np.empty((0, 4), dtype=np.float64)
            )
            np.savetxt(
                output_dir / "path_poses_xyyaw.csv",
                indexed_path,
                delimiter=",",
                header="index,x_m,y_m,yaw_rad",
                comments="",
            )
            initial_pose = (
                path_xyyaw[0] if path_xyyaw.shape[0] else np.zeros(3, dtype=np.float64)
            )
            final_pose = (
                path_xyyaw[-1] if path_xyyaw.shape[0] else np.zeros(3, dtype=np.float64)
            )
            poses_payload = {
                "frame_id": self._frame_id,
                "child_frame_id": self._child_frame_id,
                "close_reason": close_reason,
                "initial_pose": status.get(
                    "initial_pose",
                    self._pose_payload(initial_pose),
                ),
                "final_pose": status.get(
                    "final_pose",
                    self._pose_payload(final_pose),
                ),
                "path_pose_count": int(path_xyyaw.shape[0]),
                "map_point_count": int(map_xy.shape[0]),
            }
            (output_dir / "poses.json").write_text(
                json.dumps(self._json_ready(poses_payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            status_payload = dict(status)
            status_payload["saved_output_dir"] = str(output_dir)
            status_payload["saved_files"] = [
                "map_points_xy.csv",
                "path_poses_xyyaw.csv",
                "poses.json",
                "status.json",
            ]
            (output_dir / "status.json").write_text(
                json.dumps(self._json_ready(status_payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.get_logger().info(f"Offline refined map saved to {output_dir}")
            return str(output_dir)
        except Exception as exc:
            self.get_logger().warn(f"Could not save offline refined map: {repr(exc)}")
            return ""

    def _json_ready(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._json_ready(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    def _republish_latest_outputs(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        with self._lock:
            map_points = self._latest_map_points.copy()
            submap_points = self._latest_submap_points.copy()
            path_poses = self._latest_path_poses.copy()
            latest_pose = self._latest_pose.copy()
            latest_global_correction = self._latest_global_correction.copy()
            latest_global_correction_child_frame_id = str(
                self._latest_global_correction_child_frame_id
            )
            status_msg = String()
            status_msg.data = str(self._latest_status.data)
        self._map_pub.publish(self._pointcloud_message_from_xy(map_points, now_msg))
        self._grid_pub.publish(self._occupancy_grid_message_from_xy(map_points, now_msg))
        self._submap_pub.publish(self._pointcloud_message_from_xy(submap_points, now_msg))
        self._path_pub.publish(self._path_message_from_poses(path_poses, now_msg))
        self._odom_pub.publish(self._odom_message_from_pose(latest_pose, now_msg))
        if self._publish_global_correction:
            self._correction_pub.publish(
                self._transform_message_from_pose(
                    latest_global_correction,
                    child_frame_id=latest_global_correction_child_frame_id,
                    stamp_msg=now_msg,
                )
            )
            self._anchor_pose_pub.publish(self._pose_message_from_pose(latest_pose, now_msg))
        self._status_pub.publish(status_msg)

    def _pointcloud_message_from_xy(self, points_xy: np.ndarray, stamp_msg) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        message.height = 1
        message.width = int(points_xy.shape[0])
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = message.point_step * message.width
        message.is_dense = True
        if points_xy.size == 0:
            message.data = b""
            return message
        points_xyz = np.column_stack(
            (
                points_xy.astype(np.float32, copy=False),
                np.zeros((points_xy.shape[0], 1), dtype=np.float32),
            )
        )
        message.data = points_xyz.tobytes()
        return message

    def _path_message_from_poses(self, poses: np.ndarray, stamp_msg) -> NavPath:
        message = NavPath()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        for pose in poses:
            message.poses.append(self._pose_message_from_pose(pose, stamp_msg))
        return message

    def _pose_message_from_pose(self, pose_xyyaw: np.ndarray, stamp_msg) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp_msg
        pose_msg.header.frame_id = self._frame_id
        pose_msg.pose.position.x = float(pose_xyyaw[0])
        pose_msg.pose.position.y = float(pose_xyyaw[1])
        qx, qy, qz, qw = _yaw_to_quat(float(pose_xyyaw[2]))
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        return pose_msg

    def _occupancy_grid_message_from_xy(self, points_xy: np.ndarray, stamp_msg) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        message.info.resolution = float(self._grid_resolution_m)
        if points_xy.size == 0:
            message.info.width = 1
            message.info.height = 1
            message.info.origin.orientation.w = 1.0
            message.data = [-1]
            return message

        min_xy = np.min(points_xy, axis=0) - self._grid_padding_m
        max_xy = np.max(points_xy, axis=0) + self._grid_padding_m
        width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / self._grid_resolution_m)))
        height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / self._grid_resolution_m)))
        grid = np.full((height, width), -1, dtype=np.int8)
        cell_xy = np.floor((points_xy - min_xy) / self._grid_resolution_m).astype(np.int32)
        cell_xy[:, 0] = np.clip(cell_xy[:, 0], 0, width - 1)
        cell_xy[:, 1] = np.clip(cell_xy[:, 1], 0, height - 1)
        radius = int(self._grid_line_radius_cells)
        for cell_x, cell_y in cell_xy:
            x_min = max(0, cell_x - radius)
            x_max = min(width - 1, cell_x + radius)
            y_min = max(0, cell_y - radius)
            y_max = min(height - 1, cell_y + radius)
            grid[y_min : y_max + 1, x_min : x_max + 1] = 100

        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = float(min_xy[0])
        message.info.origin.position.y = float(min_xy[1])
        message.info.origin.orientation.w = 1.0
        message.data = grid.reshape(-1).astype(np.int8).tolist()
        return message

    def _odom_message_from_pose(self, pose_xyyaw: np.ndarray, stamp_msg) -> Odometry:
        message = Odometry()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        message.child_frame_id = self._child_frame_id
        message.pose.pose.position.x = float(pose_xyyaw[0])
        message.pose.pose.position.y = float(pose_xyyaw[1])
        qx, qy, qz, qw = _yaw_to_quat(float(pose_xyyaw[2]))
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        return message

    def _transform_message_from_pose(
        self,
        pose_xyyaw: np.ndarray,
        *,
        child_frame_id: str,
        stamp_msg,
    ) -> TransformStamped:
        message = TransformStamped()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        message.child_frame_id = child_frame_id or self._seed_odom_frame_id
        message.transform.translation.x = float(pose_xyyaw[0])
        message.transform.translation.y = float(pose_xyyaw[1])
        qx, qy, qz, qw = _yaw_to_quat(float(pose_xyyaw[2]))
        message.transform.rotation.x = qx
        message.transform.rotation.y = qy
        message.transform.rotation.z = qz
        message.transform.rotation.w = qw
        return message

    def _compute_global_correction(
        self,
        window_records: list[ScanRecord],
        refined_poses: np.ndarray,
        seed_records: list[OdomRecord],
    ) -> tuple[np.ndarray, str]:
        child_frame_id = self._seed_odom_frame_id
        if not self._publish_global_correction or not seed_records:
            return self._latest_global_correction.copy(), child_frame_id
        corrections: list[np.ndarray] = []
        start_index = max(0, len(window_records) - 8)
        for window_index in range(start_index, len(window_records)):
            record = window_records[window_index]
            seed_pose = self._interpolate_odom(seed_records, float(record.stamp_s))
            if seed_pose is None:
                continue
            seed_position = next(
                (
                    odom_record
                    for odom_record in reversed(seed_records)
                    if abs(float(odom_record.stamp_s - record.stamp_s)) <= 0.5
                ),
                None,
            )
            if seed_position is not None and seed_position.frame_id:
                child_frame_id = seed_position.frame_id
            refined_pose = refined_poses[window_index]
            yaw = _wrap_angle(float(refined_pose[2] - seed_pose[2]))
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            tx = float(refined_pose[0]) - (
                (cos_yaw * float(seed_pose[0])) - (sin_yaw * float(seed_pose[1]))
            )
            ty = float(refined_pose[1]) - (
                (sin_yaw * float(seed_pose[0])) + (cos_yaw * float(seed_pose[1]))
            )
            corrections.append(np.asarray([tx, ty, yaw], dtype=np.float64))
        if not corrections:
            return self._latest_global_correction.copy(), child_frame_id
        correction_stack = np.vstack(corrections)
        mean_xy = np.mean(correction_stack[:, :2], axis=0)
        mean_yaw = math.atan2(
            float(np.mean(np.sin(correction_stack[:, 2]))),
            float(np.mean(np.cos(correction_stack[:, 2]))),
        )
        return np.asarray([float(mean_xy[0]), float(mean_xy[1]), mean_yaw], dtype=np.float64), child_frame_id


def main() -> None:
    rclpy.init()
    node = OfflineSubmapRefiner()
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
