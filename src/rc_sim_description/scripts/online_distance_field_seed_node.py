#!/usr/bin/env python3
"""Causal LiDAR + IMU seed odometry using local occupancy distance fields."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.optimize import least_squares
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _rotation_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(pose_xyyaw[2])).T) + pose_xyyaw[:2]


def _voxel_downsample(points_xy: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xy.size == 0 or voxel_size_m <= 1.0e-6:
        return points_xy
    grid = np.floor(points_xy / voxel_size_m).astype(np.int64)
    _, unique_indexes = np.unique(grid, axis=0, return_index=True)
    unique_indexes.sort()
    return points_xy[unique_indexes]


@dataclass(frozen=True)
class ScanRecord:
    stamp_s: float
    points_xy: np.ndarray
    pose_xyyaw: np.ndarray
    confidence: str
    median_residual_m: float


@dataclass(frozen=True)
class ImuRecord:
    stamp_s: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class DistanceFieldLevel:
    resolution_m: float
    origin_xy: np.ndarray
    distance_field_m: np.ndarray
    max_distance_m: float


def _build_hit_count_grid(
    points_xy: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    min_extent_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points_xy.size == 0:
        hit_counts = np.zeros((min_extent_cells, min_extent_cells), dtype=np.int32)
        origin_xy = np.asarray(
            [
                -0.5 * min_extent_cells * resolution_m,
                -0.5 * min_extent_cells * resolution_m,
            ],
            dtype=np.float64,
        )
        return hit_counts, origin_xy
    min_xy = np.min(points_xy, axis=0) - margin_m
    max_xy = np.max(points_xy, axis=0) + margin_m
    width = max(min_extent_cells, int(math.ceil((max_xy[0] - min_xy[0]) / resolution_m)) + 1)
    height = max(min_extent_cells, int(math.ceil((max_xy[1] - min_xy[1]) / resolution_m)) + 1)
    hit_counts = np.zeros((height, width), dtype=np.int32)
    gx = np.floor((points_xy[:, 0] - min_xy[0]) / resolution_m).astype(np.int32)
    gy = np.floor((points_xy[:, 1] - min_xy[1]) / resolution_m).astype(np.int32)
    valid_mask = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    np.add.at(hit_counts, (gy[valid_mask], gx[valid_mask]), 1)
    return hit_counts, min_xy.astype(np.float64)


def _build_distance_field_level(
    submap_points_xy: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
) -> DistanceFieldLevel:
    hit_counts, origin_xy = _build_hit_count_grid(
        submap_points_xy,
        resolution_m=resolution_m,
        margin_m=margin_m,
        min_extent_cells=24,
    )
    occupancy = hit_counts > 0
    occupancy = binary_dilation(occupancy, iterations=1)
    distance_field = distance_transform_edt(~occupancy) * resolution_m
    return DistanceFieldLevel(
        resolution_m=float(resolution_m),
        origin_xy=origin_xy,
        distance_field_m=np.asarray(distance_field, dtype=np.float32),
        max_distance_m=max(0.7, 5.0 * resolution_m),
    )


def _sample_distance_field_bilinear(
    level: DistanceFieldLevel,
    world_points_xy: np.ndarray,
    *,
    outside_distance_m: float,
) -> np.ndarray:
    if world_points_xy.size == 0:
        return np.empty((0,), dtype=np.float64)
    grid_x = ((world_points_xy[:, 0] - level.origin_xy[0]) / level.resolution_m) - 0.5
    grid_y = ((world_points_xy[:, 1] - level.origin_xy[1]) / level.resolution_m) - 0.5
    x0 = np.floor(grid_x).astype(np.int32)
    y0 = np.floor(grid_y).astype(np.int32)
    tx = grid_x - x0
    ty = grid_y - y0
    width = level.distance_field_m.shape[1]
    height = level.distance_field_m.shape[0]
    valid = (x0 >= 0) & ((x0 + 1) < width) & (y0 >= 0) & ((y0 + 1) < height)
    sampled = np.full((world_points_xy.shape[0],), float(outside_distance_m), dtype=np.float64)
    if not np.any(valid):
        return sampled
    x0v = x0[valid]
    y0v = y0[valid]
    txv = tx[valid]
    tyv = ty[valid]
    field = level.distance_field_m
    v00 = field[y0v, x0v].astype(np.float64)
    v10 = field[y0v, x0v + 1].astype(np.float64)
    v01 = field[y0v + 1, x0v].astype(np.float64)
    v11 = field[y0v + 1, x0v + 1].astype(np.float64)
    sampled_valid = (
        ((1.0 - txv) * (1.0 - tyv) * v00)
        + (txv * (1.0 - tyv) * v10)
        + ((1.0 - txv) * tyv * v01)
        + (txv * tyv * v11)
    )
    sampled[valid] = sampled_valid
    return sampled


def _build_multires_levels(submap_points_xy: np.ndarray) -> list[DistanceFieldLevel]:
    return [
        _build_distance_field_level(submap_points_xy, resolution_m=resolution_m, margin_m=0.8)
        for resolution_m in (0.20, 0.10, 0.05)
    ]


def _optimize_pose_against_levels(
    points_local: np.ndarray,
    initial_pose: np.ndarray,
    prior_pose: np.ndarray,
    levels: list[DistanceFieldLevel],
    *,
    max_correspondence_m: float,
    prior_weight_xy: float,
    prior_weight_yaw: float,
    max_nfev: int,
) -> tuple[np.ndarray, float]:
    if points_local.shape[0] < 8 or not levels:
        return initial_pose.copy(), float("inf")
    state = initial_pose.astype(np.float64).copy()
    for level in levels:
        result = least_squares(
            lambda state_vec: _distance_field_residuals(
                state_vec,
                points_local=points_local,
                level=level,
                max_correspondence_m=max_correspondence_m,
                prior_pose=prior_pose,
                prior_weight_xy=prior_weight_xy,
                prior_weight_yaw=prior_weight_yaw,
            ),
            x0=state,
            loss="soft_l1",
            f_scale=max(0.03, 0.5 * max_correspondence_m),
            max_nfev=max_nfev,
        )
        state = result.x.astype(np.float64)
        state[2] = _wrap_angle(float(state[2]))
    final_level = levels[-1]
    final_distances = _sample_distance_field_bilinear(
        final_level,
        _transform_points(points_local, state),
        outside_distance_m=max(final_level.max_distance_m, max_correspondence_m * 2.0),
    )
    final_distances = np.clip(
        final_distances,
        0.0,
        max(final_level.max_distance_m, max_correspondence_m * 2.0),
    )
    return state, float(np.median(final_distances))


def _distance_field_residuals(
    candidate_pose: np.ndarray,
    *,
    points_local: np.ndarray,
    level: DistanceFieldLevel,
    max_correspondence_m: float,
    prior_pose: np.ndarray,
    prior_weight_xy: float,
    prior_weight_yaw: float,
) -> np.ndarray:
    pose = candidate_pose.astype(np.float64).copy()
    pose[2] = _wrap_angle(float(pose[2]))
    world_points = _transform_points(points_local, pose)
    distances_m = _sample_distance_field_bilinear(
        level,
        world_points,
        outside_distance_m=max(level.max_distance_m, max_correspondence_m * 2.0),
    )
    distances_m = np.clip(
        distances_m,
        0.0,
        max(level.max_distance_m, max_correspondence_m * 2.0),
    )
    prior_residual = np.asarray(
        [
            prior_weight_xy * float(pose[0] - prior_pose[0]),
            prior_weight_xy * float(pose[1] - prior_pose[1]),
            prior_weight_yaw * _wrap_angle(float(pose[2] - prior_pose[2])),
        ],
        dtype=np.float64,
    )
    return np.concatenate((0.35 * distances_m, prior_residual))


def _match_statistics(
    points_local: np.ndarray,
    pose_xyyaw: np.ndarray,
    level: DistanceFieldLevel,
    *,
    max_correspondence_m: float,
) -> dict[str, float]:
    if points_local.shape[0] == 0:
        return {
            "median_distance_m": float("inf"),
            "inlier_ratio": 0.0,
            "support_ratio": 0.0,
        }
    distances = _sample_distance_field_bilinear(
        level,
        _transform_points(points_local, pose_xyyaw),
        outside_distance_m=max(level.max_distance_m, max_correspondence_m * 2.0),
    )
    distances = np.clip(
        distances,
        0.0,
        max(level.max_distance_m, max_correspondence_m * 2.0),
    )
    return {
        "median_distance_m": float(np.median(distances)),
        "inlier_ratio": float(np.mean((distances <= max_correspondence_m).astype(np.float64))),
        "support_ratio": float(np.mean((distances <= min(0.12, 0.5 * max_correspondence_m)).astype(np.float64))),
    }


class OnlineDistanceFieldSeedNode(Node):
    def __init__(self) -> None:
        super().__init__("online_distance_field_seed_node")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.declare_parameter("scan_topic", "/apex/sim/scan")
        self.declare_parameter("imu_topic", "/apex/sim/imu")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("path_topic", "/apex/estimation/path")
        self.declare_parameter("pose_topic", "/apex/estimation/current_pose")
        self.declare_parameter("live_map_topic", "/apex/estimation/live_map_points")
        self.declare_parameter("status_topic", "/apex/estimation/status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("submap_window_scans", 24)
        self.declare_parameter("point_stride", 1)
        self.declare_parameter("max_correspondence_m", 0.30)
        self.declare_parameter("max_scan_optimization_evals", 60)
        self.declare_parameter("yaw_bias_init_duration_s", 0.8)
        self.declare_parameter("velocity_decay_tau_s", 0.9)
        self.declare_parameter("live_map_max_points", 8000)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._live_map_topic = str(self.get_parameter("live_map_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._child_frame = str(self.get_parameter("child_frame_id").value)
        self._submap_window_scans = max(6, int(self.get_parameter("submap_window_scans").value))
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._max_correspondence_m = max(0.05, float(self.get_parameter("max_correspondence_m").value))
        self._max_scan_optimization_evals = max(
            20, int(self.get_parameter("max_scan_optimization_evals").value)
        )
        self._yaw_bias_init_duration_s = max(
            0.2, float(self.get_parameter("yaw_bias_init_duration_s").value)
        )
        self._velocity_decay_tau_s = max(
            0.2, float(self.get_parameter("velocity_decay_tau_s").value)
        )
        self._live_map_max_points = max(128, int(self.get_parameter("live_map_max_points").value))

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._path_pub = self.create_publisher(Path, self._path_topic, 20)
        self._pose_pub = self.create_publisher(PoseStamped, self._pose_topic, 20)
        self._live_map_pub = self.create_publisher(PointCloud2, self._live_map_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)

        self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)

        self._imu_records: deque[ImuRecord] = deque(maxlen=12000)
        self._raw_yaw_rates: deque[tuple[float, float]] = deque(maxlen=12000)
        self._yaw_bias_rps = 0.0
        self._imu_initialized = False
        self._imu_init_start_s: float | None = None

        self._scan_records: list[ScanRecord] = []
        self._latest_status = ""
        self._path_msg = Path()
        self._path_msg.header.frame_id = self._odom_frame

        self.get_logger().info(
            "OnlineDistanceFieldSeedNode started (scan=%s imu=%s odom=%s path=%s map=%s status=%s)"
            % (
                self._scan_topic,
                self._imu_topic,
                self._odom_topic,
                self._path_topic,
                self._live_map_topic,
                self._status_topic,
            )
        )

    def _imu_cb(self, msg: Imu) -> None:
        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        yaw_rate_rps = float(msg.angular_velocity.z)
        self._imu_records.append(ImuRecord(stamp_s=stamp_s, yaw_rate_rps=yaw_rate_rps))
        self._raw_yaw_rates.append((stamp_s, yaw_rate_rps))
        if self._imu_init_start_s is None:
            self._imu_init_start_s = stamp_s
        if not self._imu_initialized and (stamp_s - self._imu_init_start_s) >= self._yaw_bias_init_duration_s:
            yaw_rates = np.asarray([sample[1] for sample in self._raw_yaw_rates], dtype=np.float64)
            self._yaw_bias_rps = float(np.mean(yaw_rates)) if yaw_rates.size else 0.0
            self._imu_initialized = True

    def _scan_cb(self, msg: LaserScan) -> None:
        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        points_xy = self._scan_points_in_base_frame(msg)
        if points_xy.shape[0] < 12:
            self._publish_status("waiting_points", None, float("nan"), 0.0, 0.0)
            return
        if not self._imu_initialized:
            self._publish_status("waiting_static_initialization", None, float("nan"), 0.0, 0.0)
            return

        sampled_points_xy = points_xy[:: self._point_stride].copy()
        if sampled_points_xy.shape[0] < 12:
            sampled_points_xy = points_xy.copy()

        if not self._scan_records:
            pose_xyyaw = np.zeros(3, dtype=np.float64)
            confidence = "high"
            median_residual_m = 0.0
        else:
            prediction = self._predict_pose(stamp_s)
            submap_points = self._build_submap_points()
            if submap_points.shape[0] < 20:
                pose_xyyaw = prediction
                confidence = "medium"
                median_residual_m = float("nan")
            else:
                levels = _build_multires_levels(submap_points)
                heading = np.asarray(
                    [math.cos(float(prediction[2])), math.sin(float(prediction[2]))],
                    dtype=np.float64,
                )
                candidate_offsets_m = (-0.60, -0.30, 0.0, 0.30, 0.60)
                best_pose = prediction.copy()
                best_median_residual_m = float("inf")
                for offset_m in candidate_offsets_m:
                    candidate_initial = prediction.copy()
                    candidate_initial[:2] = candidate_initial[:2] + (offset_m * heading)
                    optimized_pose, candidate_median = _optimize_pose_against_levels(
                        sampled_points_xy,
                        candidate_initial,
                        prediction,
                        levels,
                        max_correspondence_m=self._max_correspondence_m,
                        prior_weight_xy=0.18,
                        prior_weight_yaw=0.95,
                        max_nfev=self._max_scan_optimization_evals,
                    )
                    if candidate_median < best_median_residual_m:
                        best_median_residual_m = candidate_median
                        best_pose = optimized_pose
                stats = _match_statistics(
                    sampled_points_xy,
                    best_pose,
                    levels[-1],
                    max_correspondence_m=self._max_correspondence_m,
                )
                median_residual_m = float(stats["median_distance_m"])
                if median_residual_m <= 0.06 and stats["inlier_ratio"] >= 0.55:
                    confidence = "high"
                elif median_residual_m <= 0.14 and stats["inlier_ratio"] >= 0.24:
                    confidence = "medium"
                else:
                    confidence = "low"
                pose_xyyaw = best_pose

        record = ScanRecord(
            stamp_s=stamp_s,
            points_xy=sampled_points_xy,
            pose_xyyaw=pose_xyyaw,
            confidence=confidence,
            median_residual_m=median_residual_m,
        )
        self._scan_records.append(record)
        vx_mps, vy_mps = self._latest_velocity()
        self._publish_outputs(record, vx_mps=vx_mps, vy_mps=vy_mps)

    def _scan_points_in_base_frame(self, msg: LaserScan) -> np.ndarray:
        points_xy: list[tuple[float, float]] = []
        angle_rad = float(msg.angle_min)
        for range_m in msg.ranges:
            value_m = float(range_m)
            if math.isfinite(value_m) and msg.range_min <= value_m <= msg.range_max:
                points_xy.append(
                    (
                        value_m * math.cos(angle_rad),
                        value_m * math.sin(angle_rad),
                    )
                )
            angle_rad += float(msg.angle_increment)
        if not points_xy:
            return np.empty((0, 2), dtype=np.float64)
        points_laser_xy = np.asarray(points_xy, dtype=np.float64)
        if msg.header.frame_id == "base_link":
            return points_laser_xy
        try:
            transform = self._tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                rclpy.time.Time(),
            )
        except TransformException:
            return points_laser_xy
        qx = float(transform.transform.rotation.x)
        qy = float(transform.transform.rotation.y)
        qz = float(transform.transform.rotation.z)
        qw = float(transform.transform.rotation.w)
        yaw = math.atan2(2.0 * ((qw * qz) + (qx * qy)), 1.0 - (2.0 * ((qy * qy) + (qz * qz))))
        tx = float(transform.transform.translation.x)
        ty = float(transform.transform.translation.y)
        transformed = (points_laser_xy @ _rotation_matrix(yaw).T) + np.asarray([tx, ty], dtype=np.float64)
        return transformed

    def _integrated_yaw_delta(self, start_s: float, end_s: float) -> float:
        if end_s <= start_s or not self._imu_records:
            return 0.0
        relevant = [record for record in self._imu_records if start_s <= record.stamp_s <= end_s]
        if not relevant:
            return 0.0
        if len(relevant) == 1:
            return float((relevant[0].yaw_rate_rps - self._yaw_bias_rps) * (end_s - start_s))
        delta = 0.0
        previous = relevant[0]
        for current in relevant[1:]:
            dt_s = max(0.0, float(current.stamp_s - previous.stamp_s))
            yaw_prev = float(previous.yaw_rate_rps - self._yaw_bias_rps)
            yaw_curr = float(current.yaw_rate_rps - self._yaw_bias_rps)
            delta += 0.5 * (yaw_prev + yaw_curr) * dt_s
            previous = current
        tail_dt_s = max(0.0, float(end_s - relevant[-1].stamp_s))
        delta += float(relevant[-1].yaw_rate_rps - self._yaw_bias_rps) * tail_dt_s
        return delta

    def _predict_pose(self, stamp_s: float) -> np.ndarray:
        last = self._scan_records[-1]
        yaw_delta = self._integrated_yaw_delta(last.stamp_s, stamp_s)
        dt_s = max(1.0e-3, float(stamp_s - last.stamp_s))
        if len(self._scan_records) >= 2:
            previous = self._scan_records[-2]
            prev_dt_s = max(1.0e-3, float(last.stamp_s - previous.stamp_s))
            velocity_xy = (last.pose_xyyaw[:2] - previous.pose_xyyaw[:2]) / prev_dt_s
            decay = math.exp(-dt_s / self._velocity_decay_tau_s)
            velocity_xy = np.clip(decay * velocity_xy, -2.0, 2.0)
        else:
            velocity_xy = np.zeros(2, dtype=np.float64)
        return np.asarray(
            [
                float(last.pose_xyyaw[0] + (velocity_xy[0] * dt_s)),
                float(last.pose_xyyaw[1] + (velocity_xy[1] * dt_s)),
                _wrap_angle(float(last.pose_xyyaw[2] + yaw_delta)),
            ],
            dtype=np.float64,
        )

    def _build_submap_points(self) -> np.ndarray:
        if not self._scan_records:
            return np.empty((0, 2), dtype=np.float64)
        start_index = max(0, len(self._scan_records) - self._submap_window_scans)
        clouds: list[np.ndarray] = []
        for record in self._scan_records[start_index:]:
            if record.confidence == "low" and len(self._scan_records) > 6:
                continue
            clouds.append(_transform_points(record.points_xy, record.pose_xyyaw))
        if not clouds:
            clouds = [
                _transform_points(record.points_xy, record.pose_xyyaw)
                for record in self._scan_records[start_index:]
            ]
        if not clouds:
            return np.empty((0, 2), dtype=np.float64)
        return _voxel_downsample(np.vstack(clouds), 0.04)

    def _latest_velocity(self) -> tuple[float, float]:
        if len(self._scan_records) < 2:
            return 0.0, 0.0
        latest = self._scan_records[-1]
        previous = self._scan_records[-2]
        dt_s = max(1.0e-3, float(latest.stamp_s - previous.stamp_s))
        velocity_xy = (latest.pose_xyyaw[:2] - previous.pose_xyyaw[:2]) / dt_s
        return float(velocity_xy[0]), float(velocity_xy[1])

    def _publish_outputs(self, record: ScanRecord, *, vx_mps: float, vy_mps: float) -> None:
        stamp_msg = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child_frame
        odom.pose.pose.position.x = float(record.pose_xyyaw[0])
        odom.pose.pose.position.y = float(record.pose_xyyaw[1])
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(float(record.pose_xyyaw[2]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(vx_mps)
        odom.twist.twist.linear.y = float(vy_mps)
        odom.twist.twist.angular.z = float(
            (self._imu_records[-1].yaw_rate_rps - self._yaw_bias_rps) if self._imu_records else 0.0
        )
        self._odom_pub.publish(odom)

        pose_msg = PoseStamped()
        pose_msg.header = odom.header
        pose_msg.pose = odom.pose.pose
        self._pose_pub.publish(pose_msg)

        self._path_msg.header.stamp = stamp_msg
        self._path_msg.header.frame_id = self._odom_frame
        self._path_msg.poses.append(pose_msg)
        if len(self._path_msg.poses) > 4000:
            del self._path_msg.poses[: len(self._path_msg.poses) - 4000]
        self._path_pub.publish(self._path_msg)

        map_points = self._build_submap_points()
        self._live_map_pub.publish(self._pointcloud_message_from_xy(map_points, stamp_msg))
        median_recent_residual = float(
            np.median(
                np.asarray(
                    [
                        entry.median_residual_m
                        for entry in self._scan_records[-32:]
                        if math.isfinite(entry.median_residual_m)
                    ],
                    dtype=np.float64,
                )
            )
        ) if any(math.isfinite(entry.median_residual_m) for entry in self._scan_records[-32:]) else float("nan")
        self._publish_status(
            "tracking",
            record,
            median_recent_residual,
            vx_mps,
            vy_mps,
        )

    def _publish_status(
        self,
        state: str,
        latest_record: ScanRecord | None,
        median_recent_residual_m: float,
        vx_mps: float,
        vy_mps: float,
    ) -> None:
        latest_pose = None
        if latest_record is not None:
            latest_pose = {
                "t_s": float(latest_record.stamp_s),
                "x_m": float(latest_record.pose_xyyaw[0]),
                "y_m": float(latest_record.pose_xyyaw[1]),
                "yaw_rad": float(latest_record.pose_xyyaw[2]),
                "vx_mps": float(vx_mps),
                "vy_mps": float(vy_mps),
                "confidence": str(latest_record.confidence),
                "median_submap_residual_m": float(latest_record.median_residual_m),
            }
        payload = {
            "estimation_backend": "distance_field_online_seed",
            "state": state,
            "imu_initialized": bool(self._imu_initialized),
            "alignment_ready": bool(self._imu_initialized),
            "best_effort_init": False,
            "pending_scan_count": 0,
            "processed_scan_count": len(self._scan_records),
            "quality": {
                "median_submap_residual_m": float(median_recent_residual_m),
                "low_confidence_scan_count": int(
                    sum(1 for record in self._scan_records[-64:] if record.confidence == "low")
                ),
            },
            "latest_pose": latest_pose,
            "parameters": {
                "submap_window_scans": self._submap_window_scans,
                "point_stride": self._point_stride,
                "max_correspondence_m": self._max_correspondence_m,
                "backend": "distance_field_online_seed",
            },
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._latest_status = message.data
        self._status_pub.publish(message)

    def _pointcloud_message_from_xy(self, map_points_xy: np.ndarray, stamp_msg) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._odom_frame
        message.height = 1
        if map_points_xy.size == 0:
            message.width = 0
            message.is_bigendian = False
            message.is_dense = True
            message.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            message.point_step = 12
            message.row_step = 0
            message.data = b""
            return message
        cloud_xy = _voxel_downsample(map_points_xy, 0.04)
        if cloud_xy.shape[0] > self._live_map_max_points:
            indexes = np.linspace(0, cloud_xy.shape[0] - 1, num=self._live_map_max_points, dtype=np.int32)
            cloud_xy = cloud_xy[indexes]
        message.width = int(cloud_xy.shape[0])
        message.is_bigendian = False
        message.is_dense = True
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.point_step = 12
        message.row_step = message.point_step * message.width
        cloud = np.zeros(
            (cloud_xy.shape[0],),
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
        cloud["x"] = cloud_xy[:, 0].astype(np.float32)
        cloud["y"] = cloud_xy[:, 1].astype(np.float32)
        cloud["z"] = 0.0
        message.data = cloud.tobytes()
        return message


def main() -> None:
    rclpy.init()
    node = OnlineDistanceFieldSeedNode()
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
