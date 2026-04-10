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
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
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


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * ((w * z) + (x * y)),
        1.0 - (2.0 * ((y * y) + (z * z))),
    )


def _rotation_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(pose_xyyaw[2])).T) + pose_xyyaw[:2]


def _compose_poses(lhs_pose_xyyaw: np.ndarray, rhs_pose_xyyaw: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs_pose_xyyaw, dtype=np.float64)
    rhs = np.asarray(rhs_pose_xyyaw, dtype=np.float64)
    xy = lhs[:2] + (_rotation_matrix(float(lhs[2])) @ rhs[:2])
    return np.asarray(
        [float(xy[0]), float(xy[1]), _wrap_angle(float(lhs[2] + rhs[2]))],
        dtype=np.float64,
    )


def _inverse_pose(pose_xyyaw: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyyaw, dtype=np.float64)
    rotation_inv = _rotation_matrix(float(-pose[2]))
    xy_inv = -(rotation_inv @ pose[:2])
    return np.asarray([float(xy_inv[0]), float(xy_inv[1]), _wrap_angle(float(-pose[2]))], dtype=np.float64)


def _blend_pose(
    current_pose_xyyaw: np.ndarray,
    target_pose_xyyaw: np.ndarray,
    *,
    blend: float,
    max_jump_xy: float,
    max_jump_yaw: float,
) -> np.ndarray:
    current = np.asarray(current_pose_xyyaw, dtype=np.float64)
    target = np.asarray(target_pose_xyyaw, dtype=np.float64)
    delta_xy = target[:2] - current[:2]
    delta_norm = float(np.linalg.norm(delta_xy))
    if delta_norm > max_jump_xy > 0.0:
        delta_xy = (max_jump_xy / delta_norm) * delta_xy
    delta_yaw = _wrap_angle(float(target[2] - current[2]))
    if max_jump_yaw > 0.0:
        delta_yaw = float(np.clip(delta_yaw, -max_jump_yaw, max_jump_yaw))
    alpha = float(np.clip(blend, 0.0, 1.0))
    blended_yaw = _wrap_angle(float(current[2] + (alpha * delta_yaw)))
    return np.asarray(
        [
            float(current[0] + (alpha * delta_xy[0])),
            float(current[1] + (alpha * delta_xy[1])),
            blended_yaw,
        ],
        dtype=np.float64,
    )


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
    raw_pose_xyyaw: np.ndarray
    confidence: str
    median_residual_m: float
    inlier_ratio: float
    inserted_into_map: bool


@dataclass(frozen=True)
class ImuRecord:
    stamp_s: float
    ax_body_mps2: float
    ay_body_mps2: float
    az_body_mps2: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class OdomRecord:
    stamp_s: float
    frame_id: str
    pose_xyyaw: np.ndarray


@dataclass(frozen=True)
class OfflineCorrectionRecord:
    stamp_s: float
    pose_xyyaw: np.ndarray
    child_frame_id: str


@dataclass(frozen=True)
class DistanceFieldLevel:
    resolution_m: float
    origin_xy: np.ndarray
    distance_field_m: np.ndarray
    max_distance_m: float


@dataclass(frozen=True)
class CandidateScore:
    pose_xyyaw: np.ndarray
    support_ratio: float
    inlier_ratio: float
    median_distance_m: float
    prior_translation_delta_m: float
    prior_yaw_delta_rad: float


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


def _offset_samples(extent: float, step: float) -> np.ndarray:
    extent = max(0.0, float(extent))
    step = max(1.0e-3, float(step))
    if extent <= 1.0e-6:
        return np.asarray([0.0], dtype=np.float64)
    sample_count = int(math.floor(extent / step))
    samples = step * np.arange(-sample_count, sample_count + 1, dtype=np.float64)
    samples = np.concatenate((samples, np.asarray([-extent, 0.0, extent], dtype=np.float64)))
    samples = np.unique(np.round(samples, decimals=6))
    samples.sort()
    return samples.astype(np.float64)


def _optimize_pose_against_levels(
    points_local: np.ndarray,
    initial_pose: np.ndarray,
    prior_pose: np.ndarray,
    levels: list[DistanceFieldLevel],
    *,
    max_correspondence_m: float,
    prior_weight_xy: float,
    prior_weight_yaw: float,
    lidar_residual_weight: float,
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
                lidar_residual_weight=lidar_residual_weight,
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
    lidar_residual_weight: float,
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
    return np.concatenate((float(lidar_residual_weight) * distances_m, prior_residual))


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


def _candidate_score(
    points_local: np.ndarray,
    candidate_pose_xyyaw: np.ndarray,
    prior_pose_xyyaw: np.ndarray,
    level: DistanceFieldLevel,
    *,
    max_correspondence_m: float,
) -> CandidateScore:
    stats = _match_statistics(
        points_local,
        candidate_pose_xyyaw,
        level,
        max_correspondence_m=max_correspondence_m,
    )
    prior_delta_xy = np.asarray(candidate_pose_xyyaw[:2], dtype=np.float64) - np.asarray(
        prior_pose_xyyaw[:2], dtype=np.float64
    )
    return CandidateScore(
        pose_xyyaw=np.asarray(candidate_pose_xyyaw, dtype=np.float64).copy(),
        support_ratio=float(stats["support_ratio"]),
        inlier_ratio=float(stats["inlier_ratio"]),
        median_distance_m=float(stats["median_distance_m"]),
        prior_translation_delta_m=float(np.linalg.norm(prior_delta_xy)),
        prior_yaw_delta_rad=abs(_wrap_angle(float(candidate_pose_xyyaw[2] - prior_pose_xyyaw[2]))),
    )


def _candidate_sort_key(score: CandidateScore) -> tuple[float, float, float, float, float]:
    return (
        float(score.support_ratio),
        float(score.inlier_ratio),
        -float(score.median_distance_m),
        -float(score.prior_translation_delta_m),
        -float(score.prior_yaw_delta_rad),
    )


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
        self.declare_parameter("corrected_odom_topic", "/apex/estimation/odom_corrected")
        self.declare_parameter("corrected_path_topic", "/apex/estimation/path_corrected")
        self.declare_parameter("corrected_pose_topic", "/apex/estimation/current_pose_corrected")
        self.declare_parameter("corrected_live_map_topic", "/apex/estimation/live_map_points_corrected")
        self.declare_parameter("status_topic", "/apex/estimation/status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("corrected_frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("prior_odom_topic", "")
        self.declare_parameter("use_external_prior", False)
        self.declare_parameter("external_prior_weight_xy", 0.45)
        self.declare_parameter("external_prior_weight_yaw", 0.85)
        self.declare_parameter("freeze_scan_insertion_on_low_confidence", True)
        self.declare_parameter("low_confidence_residual_threshold_m", 0.14)
        self.declare_parameter("low_confidence_inlier_ratio_threshold", 0.24)
        self.declare_parameter("use_offline_correction", False)
        self.declare_parameter("offline_correction_topic", "/apex/sim/offline_global_correction")
        self.declare_parameter("offline_correction_use_yaw", False)
        self.declare_parameter("offline_correction_blend", 0.25)
        self.declare_parameter("offline_correction_max_jump_xy", 0.30)
        self.declare_parameter("offline_correction_max_jump_yaw", 0.18)
        self.declare_parameter("use_offline_submap_as_reference", False)
        self.declare_parameter("use_offline_grid_as_reference", False)
        self.declare_parameter("offline_submap_topic", "/apex/sim/offline_current_submap")
        self.declare_parameter("offline_grid_topic", "/apex/sim/offline_refined_grid")
        self.declare_parameter("submap_window_scans", 24)
        self.declare_parameter("point_stride", 1)
        self.declare_parameter("max_correspondence_m", 0.30)
        self.declare_parameter("local_prior_weight_xy", 0.12)
        self.declare_parameter("local_prior_weight_yaw", 0.45)
        self.declare_parameter("lidar_residual_weight", 0.55)
        self.declare_parameter("max_scan_optimization_evals", 60)
        self.declare_parameter("correlative_search_forward_extent_m", 1.20)
        self.declare_parameter("correlative_search_lateral_extent_m", 0.35)
        self.declare_parameter("correlative_search_step_m", 0.20)
        self.declare_parameter("correlative_search_yaw_extent_rad", 0.14)
        self.declare_parameter("correlative_search_yaw_step_rad", 0.05)
        self.declare_parameter("correlative_search_top_k", 5)
        self.declare_parameter("low_confidence_pose_blend", 0.35)
        self.declare_parameter("low_confidence_pose_max_jump_xy", 0.35)
        self.declare_parameter("low_confidence_pose_max_jump_yaw", 0.12)
        self.declare_parameter("yaw_bias_init_duration_s", 0.8)
        self.declare_parameter("imu_filter_alpha", 0.25)
        self.declare_parameter("velocity_decay_tau_s", 0.9)
        self.declare_parameter("live_map_max_points", 8000)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._live_map_topic = str(self.get_parameter("live_map_topic").value)
        self._corrected_odom_topic = str(self.get_parameter("corrected_odom_topic").value)
        self._corrected_path_topic = str(self.get_parameter("corrected_path_topic").value)
        self._corrected_pose_topic = str(self.get_parameter("corrected_pose_topic").value)
        self._corrected_live_map_topic = str(self.get_parameter("corrected_live_map_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._corrected_frame = str(self.get_parameter("corrected_frame_id").value)
        self._child_frame = str(self.get_parameter("child_frame_id").value)
        self._prior_odom_topic = str(self.get_parameter("prior_odom_topic").value).strip()
        self._use_external_prior = bool(self.get_parameter("use_external_prior").value)
        self._external_prior_weight_xy = max(
            0.0, float(self.get_parameter("external_prior_weight_xy").value)
        )
        self._external_prior_weight_yaw = max(
            0.0, float(self.get_parameter("external_prior_weight_yaw").value)
        )
        self._freeze_scan_insertion_on_low_confidence = bool(
            self.get_parameter("freeze_scan_insertion_on_low_confidence").value
        )
        self._low_confidence_residual_threshold_m = max(
            0.02, float(self.get_parameter("low_confidence_residual_threshold_m").value)
        )
        self._low_confidence_inlier_ratio_threshold = float(
            np.clip(float(self.get_parameter("low_confidence_inlier_ratio_threshold").value), 0.0, 1.0)
        )
        self._use_offline_correction = bool(self.get_parameter("use_offline_correction").value)
        self._offline_correction_topic = str(
            self.get_parameter("offline_correction_topic").value
        )
        self._offline_correction_use_yaw = bool(
            self.get_parameter("offline_correction_use_yaw").value
        )
        self._offline_correction_blend = float(
            np.clip(float(self.get_parameter("offline_correction_blend").value), 0.0, 1.0)
        )
        self._offline_correction_max_jump_xy = max(
            0.0, float(self.get_parameter("offline_correction_max_jump_xy").value)
        )
        self._offline_correction_max_jump_yaw = max(
            0.0, float(self.get_parameter("offline_correction_max_jump_yaw").value)
        )
        self._use_offline_submap_as_reference = bool(
            self.get_parameter("use_offline_submap_as_reference").value
        )
        self._use_offline_grid_as_reference = bool(
            self.get_parameter("use_offline_grid_as_reference").value
        )
        self._offline_submap_topic = str(self.get_parameter("offline_submap_topic").value)
        self._offline_grid_topic = str(self.get_parameter("offline_grid_topic").value)
        self._submap_window_scans = max(6, int(self.get_parameter("submap_window_scans").value))
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._max_correspondence_m = max(0.05, float(self.get_parameter("max_correspondence_m").value))
        self._local_prior_weight_xy = max(
            0.0, float(self.get_parameter("local_prior_weight_xy").value)
        )
        self._local_prior_weight_yaw = max(
            0.0, float(self.get_parameter("local_prior_weight_yaw").value)
        )
        self._lidar_residual_weight = max(
            0.05, float(self.get_parameter("lidar_residual_weight").value)
        )
        self._max_scan_optimization_evals = max(
            20, int(self.get_parameter("max_scan_optimization_evals").value)
        )
        self._correlative_search_forward_extent_m = max(
            0.20, float(self.get_parameter("correlative_search_forward_extent_m").value)
        )
        self._correlative_search_lateral_extent_m = max(
            0.05, float(self.get_parameter("correlative_search_lateral_extent_m").value)
        )
        self._correlative_search_step_m = max(
            0.05, float(self.get_parameter("correlative_search_step_m").value)
        )
        self._correlative_search_yaw_extent_rad = max(
            0.02, float(self.get_parameter("correlative_search_yaw_extent_rad").value)
        )
        self._correlative_search_yaw_step_rad = max(
            0.01, float(self.get_parameter("correlative_search_yaw_step_rad").value)
        )
        self._correlative_search_top_k = max(
            1, int(self.get_parameter("correlative_search_top_k").value)
        )
        self._low_confidence_pose_blend = float(
            np.clip(float(self.get_parameter("low_confidence_pose_blend").value), 0.0, 1.0)
        )
        self._low_confidence_pose_max_jump_xy = max(
            0.0, float(self.get_parameter("low_confidence_pose_max_jump_xy").value)
        )
        self._low_confidence_pose_max_jump_yaw = max(
            0.0, float(self.get_parameter("low_confidence_pose_max_jump_yaw").value)
        )
        self._yaw_bias_init_duration_s = max(
            0.2, float(self.get_parameter("yaw_bias_init_duration_s").value)
        )
        self._imu_filter_alpha = float(
            np.clip(float(self.get_parameter("imu_filter_alpha").value), 0.0, 1.0)
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
        self._corrected_odom_pub = self.create_publisher(Odometry, self._corrected_odom_topic, 20)
        self._corrected_path_pub = self.create_publisher(Path, self._corrected_path_topic, 20)
        self._corrected_pose_pub = self.create_publisher(PoseStamped, self._corrected_pose_topic, 20)
        self._corrected_live_map_pub = self.create_publisher(
            PointCloud2,
            self._corrected_live_map_topic,
            latched_qos,
        )
        self._status_pub = self.create_publisher(String, self._status_topic, 20)

        self._imu_records: deque[ImuRecord] = deque(maxlen=12000)
        self._raw_imu_samples: deque[tuple[float, float, float, float, float]] = deque(maxlen=12000)
        self._prior_odom_records: deque[OdomRecord] = deque(maxlen=4000)
        self._yaw_bias_rps = 0.0
        self._accel_bias_x_mps2 = 0.0
        self._accel_bias_y_mps2 = 0.0
        self._accel_bias_z_mps2 = 0.0
        self._filtered_ax_mps2 = 0.0
        self._filtered_ay_mps2 = 0.0
        self._filtered_az_mps2 = 0.0
        self._filtered_yaw_rate_rps = 0.0
        self._imu_initialized = False
        self._imu_init_start_s: float | None = None

        self._scan_records: list[ScanRecord] = []
        self._latest_status = ""
        self._latest_offline_correction_target = np.zeros(3, dtype=np.float64)
        self._latest_offline_correction_applied = np.zeros(3, dtype=np.float64)
        self._latest_offline_correction_child_frame_id = self._odom_frame
        self._offline_submap_points = np.empty((0, 2), dtype=np.float64)
        self._offline_submap_levels: list[DistanceFieldLevel] = []
        self._offline_grid_points = np.empty((0, 2), dtype=np.float64)
        self._offline_grid_levels: list[DistanceFieldLevel] = []
        self._last_map_insertion_state: bool | None = None
        self._last_reference_mode = "local"

        self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        if self._use_external_prior and self._prior_odom_topic:
            self.create_subscription(Odometry, self._prior_odom_topic, self._prior_odom_cb, 40)
        if self._use_offline_correction:
            self.create_subscription(
                TransformStamped,
                self._offline_correction_topic,
                self._offline_correction_cb,
                self._latched_qos_from_depth(),
            )
        if self._use_offline_submap_as_reference:
            self.create_subscription(
                PointCloud2,
                self._offline_submap_topic,
                self._offline_submap_cb,
                self._latched_qos_from_depth(),
            )
        if self._use_offline_grid_as_reference:
            self.create_subscription(
                OccupancyGrid,
                self._offline_grid_topic,
                self._offline_grid_cb,
                self._latched_qos_from_depth(),
            )

        self.get_logger().info(
            "OnlineDistanceFieldSeedNode started (scan=%s imu=%s odom=%s corrected_odom=%s path=%s corrected_path=%s map=%s corrected_map=%s status=%s prior_odom=%s offline_correction=%s offline_reference=%s)"
            % (
                self._scan_topic,
                self._imu_topic,
                self._odom_topic,
                self._corrected_odom_topic,
                self._path_topic,
                self._corrected_path_topic,
                self._live_map_topic,
                self._corrected_live_map_topic,
                self._status_topic,
                self._prior_odom_topic or "<none>",
                str(self._use_offline_correction).lower(),
                str(self._use_offline_submap_as_reference or self._use_offline_grid_as_reference).lower(),
            )
        )

    def _latched_qos_from_depth(self, depth: int = 1) -> QoSProfile:
        return QoSProfile(
            depth=max(1, int(depth)),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

    def _imu_cb(self, msg: Imu) -> None:
        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        ax_mps2 = float(msg.linear_acceleration.x)
        ay_mps2 = float(msg.linear_acceleration.y)
        az_mps2 = float(msg.linear_acceleration.z)
        yaw_rate_rps = float(msg.angular_velocity.z)
        self._raw_imu_samples.append((stamp_s, ax_mps2, ay_mps2, az_mps2, yaw_rate_rps))
        if self._imu_init_start_s is None:
            self._imu_init_start_s = stamp_s
        if not self._imu_initialized and (stamp_s - self._imu_init_start_s) >= self._yaw_bias_init_duration_s:
            raw_samples = np.asarray(self._raw_imu_samples, dtype=np.float64)
            if raw_samples.size:
                self._accel_bias_x_mps2 = float(np.mean(raw_samples[:, 1]))
                self._accel_bias_y_mps2 = float(np.mean(raw_samples[:, 2]))
                self._accel_bias_z_mps2 = float(np.mean(raw_samples[:, 3]))
                self._yaw_bias_rps = float(np.mean(raw_samples[:, 4]))
            self._imu_initialized = True
        if not self._imu_initialized:
            return
        corrected_ax = ax_mps2 - self._accel_bias_x_mps2
        corrected_ay = ay_mps2 - self._accel_bias_y_mps2
        corrected_az = az_mps2 - self._accel_bias_z_mps2
        corrected_yaw_rate = yaw_rate_rps - self._yaw_bias_rps
        alpha = self._imu_filter_alpha
        if not self._imu_records or alpha >= 1.0:
            self._filtered_ax_mps2 = corrected_ax
            self._filtered_ay_mps2 = corrected_ay
            self._filtered_az_mps2 = corrected_az
            self._filtered_yaw_rate_rps = corrected_yaw_rate
        else:
            self._filtered_ax_mps2 = (alpha * corrected_ax) + ((1.0 - alpha) * self._filtered_ax_mps2)
            self._filtered_ay_mps2 = (alpha * corrected_ay) + ((1.0 - alpha) * self._filtered_ay_mps2)
            self._filtered_az_mps2 = (alpha * corrected_az) + ((1.0 - alpha) * self._filtered_az_mps2)
            self._filtered_yaw_rate_rps = (alpha * corrected_yaw_rate) + (
                (1.0 - alpha) * self._filtered_yaw_rate_rps
            )
        self._imu_records.append(
            ImuRecord(
                stamp_s=stamp_s,
                ax_body_mps2=float(self._filtered_ax_mps2),
                ay_body_mps2=float(self._filtered_ay_mps2),
                az_body_mps2=float(self._filtered_az_mps2),
                yaw_rate_rps=float(self._filtered_yaw_rate_rps),
            )
        )

    def _prior_odom_cb(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        self._prior_odom_records.append(
            OdomRecord(
                stamp_s=float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
                frame_id=str(msg.header.frame_id).strip() or self._odom_frame,
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
        )

    def _offline_correction_cb(self, msg: TransformStamped) -> None:
        rotation = msg.transform.rotation
        self._latest_offline_correction_target = np.asarray(
            [
                float(msg.transform.translation.x),
                float(msg.transform.translation.y),
                _quat_to_yaw(
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ),
            ],
            dtype=np.float64,
        )
        self._latest_offline_correction_child_frame_id = str(msg.child_frame_id).strip() or self._odom_frame

    def _offline_submap_cb(self, msg: PointCloud2) -> None:
        self._offline_submap_points = self._pointcloud_xy_from_message(msg)
        self._offline_submap_levels = (
            _build_multires_levels(self._offline_submap_points)
            if self._offline_submap_points.shape[0] >= 8
            else []
        )

    def _offline_grid_cb(self, msg: OccupancyGrid) -> None:
        self._offline_grid_points = self._occupied_xy_from_grid(msg)
        self._offline_grid_levels = (
            _build_multires_levels(self._offline_grid_points)
            if self._offline_grid_points.shape[0] >= 8
            else []
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        points_xy = self._scan_points_in_base_frame(msg)
        if points_xy.shape[0] < 12:
            self._publish_status("waiting_points", None, float("nan"), 0.0, 0.0, reference_mode="local")
            return
        if not self._imu_initialized:
            self._publish_status(
                "waiting_static_initialization",
                None,
                float("nan"),
                0.0,
                0.0,
                reference_mode="local",
            )
            return

        sampled_points_xy = points_xy[:: self._point_stride].copy()
        if sampled_points_xy.shape[0] < 12:
            sampled_points_xy = points_xy.copy()

        self._update_applied_offline_correction()
        reference_mode, reference_levels = self._select_reference_levels()
        if not self._scan_records:
            raw_pose_xyyaw = np.zeros(3, dtype=np.float64)
            confidence = "high"
            median_residual_m = 0.0
            inlier_ratio = 1.0
            inserted_into_map = True
        else:
            prediction_raw = self._predict_pose(stamp_s)
            external_prior_raw = self._interpolate_prior_odom(stamp_s)
            if reference_levels:
                prediction_reference = self._raw_pose_to_reference_pose(prediction_raw, reference_mode)
                last_reference_pose = self._raw_pose_to_reference_pose(
                    self._scan_records[-1].raw_pose_xyyaw,
                    reference_mode,
                )
                if external_prior_raw is not None:
                    external_prior_reference = self._raw_pose_to_reference_pose(
                        external_prior_raw,
                        reference_mode,
                    )
                    prior_pose = self._blend_with_external_prior(
                        prediction_reference,
                        external_prior_reference,
                    )
                    prior_weight_xy = self._external_prior_weight_xy
                    prior_weight_yaw = self._external_prior_weight_yaw
                else:
                    prior_pose = prediction_reference.copy()
                    prior_weight_xy = self._local_prior_weight_xy
                    prior_weight_yaw = self._local_prior_weight_yaw
                best_pose, _ = self._search_initial_reference_pose(
                    points_local=sampled_points_xy,
                    prior_pose=prior_pose,
                    last_reference_pose=last_reference_pose,
                    reference_levels=reference_levels,
                    prior_weight_xy=prior_weight_xy,
                    prior_weight_yaw=prior_weight_yaw,
                )
                stats = _match_statistics(
                    sampled_points_xy,
                    best_pose,
                    reference_levels[-1],
                    max_correspondence_m=self._max_correspondence_m,
                )
                median_residual_m = float(stats["median_distance_m"])
                inlier_ratio = float(stats["inlier_ratio"])
                if median_residual_m <= 0.06 and inlier_ratio >= 0.55:
                    confidence = "high"
                elif median_residual_m <= 0.14 and inlier_ratio >= 0.24:
                    confidence = "medium"
                else:
                    confidence = "low"
                inserted_into_map = self._should_insert_scan(
                    confidence=confidence,
                    median_residual_m=median_residual_m,
                    inlier_ratio=inlier_ratio,
                )
                if not inserted_into_map and self._freeze_scan_insertion_on_low_confidence:
                    # Match the real offline sensor_fusion behavior: if the
                    # current scan is low-confidence, do not let the bad local
                    # registration pull the pose backward. Keep publishing the
                    # predicted/prior pose, but freeze map insertion so the
                    # contaminated scan does not degrade the local submap.
                    final_reference_pose = prior_pose.copy()
                else:
                    final_reference_pose = best_pose
                raw_pose_xyyaw = self._reference_pose_to_raw_pose(final_reference_pose, reference_mode)
            else:
                raw_pose_xyyaw = prediction_raw
                confidence = "medium"
                median_residual_m = float("nan")
                inlier_ratio = 0.0
                inserted_into_map = True

        record = ScanRecord(
            stamp_s=stamp_s,
            points_xy=sampled_points_xy,
            raw_pose_xyyaw=raw_pose_xyyaw,
            confidence=confidence,
            median_residual_m=median_residual_m,
            inlier_ratio=inlier_ratio,
            inserted_into_map=inserted_into_map,
        )
        self._scan_records.append(record)
        vx_mps, vy_mps = self._latest_velocity()
        if self._last_reference_mode != reference_mode:
            self.get_logger().info(
                "Online reference mode switched to %s" % reference_mode
            )
            self._last_reference_mode = reference_mode
        if self._last_map_insertion_state is None or self._last_map_insertion_state != inserted_into_map:
            self.get_logger().info(
                "Online scan insertion %s (confidence=%s residual=%.3f inlier_ratio=%.3f)"
                % (
                    "enabled" if inserted_into_map else "frozen",
                    confidence,
                    median_residual_m if math.isfinite(median_residual_m) else float("nan"),
                    inlier_ratio,
                )
            )
            self._last_map_insertion_state = inserted_into_map
        self._publish_outputs(record, vx_mps=vx_mps, vy_mps=vy_mps, reference_mode=reference_mode)

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
            return float(relevant[0].yaw_rate_rps * (end_s - start_s))
        delta = 0.0
        previous = relevant[0]
        for current in relevant[1:]:
            dt_s = max(0.0, float(current.stamp_s - previous.stamp_s))
            yaw_prev = float(previous.yaw_rate_rps)
            yaw_curr = float(current.yaw_rate_rps)
            delta += 0.5 * (yaw_prev + yaw_curr) * dt_s
            previous = current
        tail_dt_s = max(0.0, float(end_s - relevant[-1].stamp_s))
        delta += float(relevant[-1].yaw_rate_rps) * tail_dt_s
        return delta

    def _average_world_accel(
        self,
        start_s: float,
        end_s: float,
        start_yaw_rad: float,
        yaw_delta_rad: float,
    ) -> np.ndarray:
        if end_s <= start_s or not self._imu_records:
            return np.zeros(2, dtype=np.float64)
        relevant = [record for record in self._imu_records if start_s <= record.stamp_s <= end_s]
        if not relevant:
            return np.zeros(2, dtype=np.float64)
        mean_ax_body = float(np.mean([record.ax_body_mps2 for record in relevant]))
        mean_ay_body = float(np.mean([record.ay_body_mps2 for record in relevant]))
        mid_yaw = _wrap_angle(float(start_yaw_rad + (0.5 * yaw_delta_rad)))
        accel_world = _rotation_matrix(mid_yaw) @ np.asarray(
            [mean_ax_body, mean_ay_body],
            dtype=np.float64,
        )
        return accel_world.astype(np.float64)

    def _update_applied_offline_correction(self) -> None:
        if not self._use_offline_correction:
            self._latest_offline_correction_applied = np.zeros(3, dtype=np.float64)
            return
        target_pose = self._latest_offline_correction_target.copy()
        current_pose = self._latest_offline_correction_applied.copy()
        if not self._offline_correction_use_yaw:
            target_pose[2] = 0.0
            current_pose[2] = 0.0
        self._latest_offline_correction_applied = _blend_pose(
            current_pose,
            target_pose,
            blend=self._offline_correction_blend,
            max_jump_xy=self._offline_correction_max_jump_xy,
            max_jump_yaw=(
                self._offline_correction_max_jump_yaw
                if self._offline_correction_use_yaw
                else 0.0
            ),
        )

    def _interpolate_prior_odom(self, stamp_s: float) -> np.ndarray | None:
        if not self._use_external_prior or not self._prior_odom_records:
            return None
        stamps = [record.stamp_s for record in self._prior_odom_records]
        position = bisect_left(stamps, stamp_s)
        if position <= 0:
            if abs(stamps[0] - stamp_s) <= 0.5:
                return self._prior_odom_records[0].pose_xyyaw.copy()
            return None
        if position >= len(stamps):
            if abs(stamps[-1] - stamp_s) <= 0.5:
                return self._prior_odom_records[-1].pose_xyyaw.copy()
            return None
        previous_record = self._prior_odom_records[position - 1]
        next_record = self._prior_odom_records[position]
        dt_s = max(1.0e-6, float(next_record.stamp_s - previous_record.stamp_s))
        alpha = float((stamp_s - previous_record.stamp_s) / dt_s)
        interp_xy = ((1.0 - alpha) * previous_record.pose_xyyaw[:2]) + (
            alpha * next_record.pose_xyyaw[:2]
        )
        yaw_delta = _wrap_angle(float(next_record.pose_xyyaw[2] - previous_record.pose_xyyaw[2]))
        interp_yaw = _wrap_angle(float(previous_record.pose_xyyaw[2] + (alpha * yaw_delta)))
        return np.asarray([float(interp_xy[0]), float(interp_xy[1]), interp_yaw], dtype=np.float64)

    def _predict_pose(self, stamp_s: float) -> np.ndarray:
        last = self._scan_records[-1]
        yaw_delta = self._integrated_yaw_delta(last.stamp_s, stamp_s)
        dt_s = max(1.0e-3, float(stamp_s - last.stamp_s))
        if len(self._scan_records) >= 2:
            previous = self._scan_records[-2]
            prev_dt_s = max(1.0e-3, float(last.stamp_s - previous.stamp_s))
            velocity_xy = (last.raw_pose_xyyaw[:2] - previous.raw_pose_xyyaw[:2]) / prev_dt_s
            decay = math.exp(-dt_s / self._velocity_decay_tau_s)
            velocity_xy = np.clip(decay * velocity_xy, -3.0, 3.0)
        else:
            velocity_xy = np.zeros(2, dtype=np.float64)
        accel_world = self._average_world_accel(
            last.stamp_s,
            stamp_s,
            float(last.raw_pose_xyyaw[2]),
            yaw_delta,
        )
        delta_xy = (velocity_xy * dt_s) + (0.5 * accel_world * dt_s * dt_s)
        prediction = np.asarray(
            [
                float(last.raw_pose_xyyaw[0] + delta_xy[0]),
                float(last.raw_pose_xyyaw[1] + delta_xy[1]),
                _wrap_angle(float(last.raw_pose_xyyaw[2] + yaw_delta)),
            ],
            dtype=np.float64,
        )
        external_prior = self._interpolate_prior_odom(stamp_s)
        if external_prior is None:
            return prediction
        return self._blend_with_external_prior(prediction, external_prior)

    def _build_submap_points(self) -> np.ndarray:
        if not self._scan_records:
            return np.empty((0, 2), dtype=np.float64)
        start_index = max(0, len(self._scan_records) - self._submap_window_scans)
        clouds: list[np.ndarray] = []
        for record in self._scan_records[start_index:]:
            if not record.inserted_into_map:
                continue
            clouds.append(_transform_points(record.points_xy, record.raw_pose_xyyaw))
        if not clouds:
            clouds = [
                _transform_points(record.points_xy, record.raw_pose_xyyaw)
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
        velocity_xy = (latest.raw_pose_xyyaw[:2] - previous.raw_pose_xyyaw[:2]) / dt_s
        return float(velocity_xy[0]), float(velocity_xy[1])

    def _select_reference_levels(self) -> tuple[str, list[DistanceFieldLevel]]:
        if self._use_offline_submap_as_reference and self._offline_submap_levels:
            return "offline_submap", list(self._offline_submap_levels)
        if self._use_offline_grid_as_reference and self._offline_grid_levels:
            return "offline_grid", list(self._offline_grid_levels)
        submap_points = self._build_submap_points()
        if submap_points.shape[0] < 20:
            return "local", []
        return "local", _build_multires_levels(submap_points)

    def _raw_pose_to_reference_pose(self, raw_pose_xyyaw: np.ndarray, reference_mode: str) -> np.ndarray:
        if reference_mode == "local":
            return np.asarray(raw_pose_xyyaw, dtype=np.float64).copy()
        reference_pose = _compose_poses(self._latest_offline_correction_applied, raw_pose_xyyaw)
        if not self._offline_correction_use_yaw:
            reference_pose[2] = float(raw_pose_xyyaw[2])
        return reference_pose

    def _reference_pose_to_raw_pose(
        self,
        reference_pose_xyyaw: np.ndarray,
        reference_mode: str,
    ) -> np.ndarray:
        if reference_mode == "local":
            return np.asarray(reference_pose_xyyaw, dtype=np.float64).copy()
        raw_pose = _compose_poses(_inverse_pose(self._latest_offline_correction_applied), reference_pose_xyyaw)
        if not self._offline_correction_use_yaw:
            raw_pose[2] = float(reference_pose_xyyaw[2])
        return raw_pose

    def _blend_with_external_prior(
        self,
        prediction_pose_xyyaw: np.ndarray,
        external_prior_pose_xyyaw: np.ndarray,
    ) -> np.ndarray:
        blended_pose = np.asarray(prediction_pose_xyyaw, dtype=np.float64).copy()
        blended_pose[:2] = (
            ((1.0 - self._external_prior_weight_xy) * blended_pose[:2])
            + (self._external_prior_weight_xy * np.asarray(external_prior_pose_xyyaw[:2], dtype=np.float64))
        )
        blended_pose[2] = _wrap_angle(
            float(
                ((1.0 - self._external_prior_weight_yaw) * blended_pose[2])
                + (self._external_prior_weight_yaw * float(external_prior_pose_xyyaw[2]))
            )
        )
        return blended_pose

    def _search_initial_reference_pose(
        self,
        *,
        points_local: np.ndarray,
        prior_pose: np.ndarray,
        last_reference_pose: np.ndarray,
        reference_levels: list[DistanceFieldLevel],
        prior_weight_xy: float,
        prior_weight_yaw: float,
    ) -> tuple[np.ndarray, float]:
        if not reference_levels:
            return np.asarray(prior_pose, dtype=np.float64).copy(), float("inf")

        coarse_level = reference_levels[0]
        top_candidates: list[CandidateScore] = []
        prior_pose = np.asarray(prior_pose, dtype=np.float64).copy()
        last_reference_pose = np.asarray(last_reference_pose, dtype=np.float64).copy()
        base_delta_xy = prior_pose[:2] - last_reference_pose[:2]
        heading = np.asarray(
            [math.cos(float(prior_pose[2])), math.sin(float(prior_pose[2]))],
            dtype=np.float64,
        )
        lateral = np.asarray([-heading[1], heading[0]], dtype=np.float64)
        base_forward_m = float(np.dot(base_delta_xy, heading))
        base_lateral_m = float(np.dot(base_delta_xy, lateral))

        forward_offsets_m = base_forward_m + _offset_samples(
            self._correlative_search_forward_extent_m,
            self._correlative_search_step_m,
        )
        lateral_offsets_m = base_lateral_m + _offset_samples(
            self._correlative_search_lateral_extent_m,
            self._correlative_search_step_m,
        )
        yaw_offsets_rad = _offset_samples(
            self._correlative_search_yaw_extent_rad,
            self._correlative_search_yaw_step_rad,
        )

        for yaw_offset_rad in yaw_offsets_rad:
            yaw_rad = _wrap_angle(float(prior_pose[2] + yaw_offset_rad))
            heading_candidate = np.asarray(
                [math.cos(float(yaw_rad)), math.sin(float(yaw_rad))],
                dtype=np.float64,
            )
            lateral_candidate = np.asarray(
                [-heading_candidate[1], heading_candidate[0]],
                dtype=np.float64,
            )
            for forward_offset_m in forward_offsets_m:
                for lateral_offset_m in lateral_offsets_m:
                    candidate_pose = np.asarray(
                        [
                            float(
                                last_reference_pose[0]
                                + (forward_offset_m * heading_candidate[0])
                                + (lateral_offset_m * lateral_candidate[0])
                            ),
                            float(
                                last_reference_pose[1]
                                + (forward_offset_m * heading_candidate[1])
                                + (lateral_offset_m * lateral_candidate[1])
                            ),
                            yaw_rad,
                        ],
                        dtype=np.float64,
                    )
                    score = _candidate_score(
                        points_local,
                        candidate_pose,
                        prior_pose,
                        coarse_level,
                        max_correspondence_m=self._max_correspondence_m,
                    )
                    top_candidates.append(score)

        if not top_candidates:
            return np.asarray(prior_pose, dtype=np.float64).copy(), float("inf")

        top_candidates.sort(key=_candidate_sort_key, reverse=True)
        refined_candidates = top_candidates[: self._correlative_search_top_k]
        best_pose = np.asarray(prior_pose, dtype=np.float64).copy()
        best_score: CandidateScore | None = None
        for candidate in refined_candidates:
            optimized_pose, _ = _optimize_pose_against_levels(
                points_local,
                candidate.pose_xyyaw,
                prior_pose,
                reference_levels,
                max_correspondence_m=self._max_correspondence_m,
                prior_weight_xy=prior_weight_xy,
                prior_weight_yaw=prior_weight_yaw,
                lidar_residual_weight=self._lidar_residual_weight,
                max_nfev=self._max_scan_optimization_evals,
            )
            optimized_score = _candidate_score(
                points_local,
                optimized_pose,
                prior_pose,
                reference_levels[-1],
                max_correspondence_m=self._max_correspondence_m,
            )
            if best_score is None or _candidate_sort_key(optimized_score) > _candidate_sort_key(best_score):
                best_score = optimized_score
                best_pose = optimized_pose
        if best_score is None:
            return best_pose, float("inf")
        return best_pose, float(best_score.median_distance_m)

    def _conservative_reference_pose(
        self,
        *,
        prior_pose: np.ndarray,
        matched_pose: np.ndarray,
    ) -> np.ndarray:
        return _blend_pose(
            prior_pose,
            matched_pose,
            blend=self._low_confidence_pose_blend,
            max_jump_xy=self._low_confidence_pose_max_jump_xy,
            max_jump_yaw=self._low_confidence_pose_max_jump_yaw,
        )

    def _should_insert_scan(
        self,
        *,
        confidence: str,
        median_residual_m: float,
        inlier_ratio: float,
    ) -> bool:
        if not self._freeze_scan_insertion_on_low_confidence:
            return True
        if confidence == "low":
            return False
        if math.isfinite(median_residual_m) and median_residual_m > self._low_confidence_residual_threshold_m:
            return False
        if float(inlier_ratio) < self._low_confidence_inlier_ratio_threshold:
            return False
        return True

    def _corrected_pose_from_raw(self, raw_pose_xyyaw: np.ndarray) -> np.ndarray:
        if not self._use_offline_correction:
            return np.asarray(raw_pose_xyyaw, dtype=np.float64).copy()
        corrected_pose = _compose_poses(self._latest_offline_correction_applied, raw_pose_xyyaw)
        if not self._offline_correction_use_yaw:
            corrected_pose[2] = float(raw_pose_xyyaw[2])
        return corrected_pose

    def _corrected_map_points(self) -> np.ndarray:
        clouds: list[np.ndarray] = []
        start_index = max(0, len(self._scan_records) - self._submap_window_scans)
        for record in self._scan_records[start_index:]:
            if not record.inserted_into_map:
                continue
            clouds.append(
                _transform_points(
                    record.points_xy,
                    self._corrected_pose_from_raw(record.raw_pose_xyyaw),
                )
            )
        if not clouds:
            return np.empty((0, 2), dtype=np.float64)
        return _voxel_downsample(np.vstack(clouds), 0.04)

    def _corrected_path_poses(self) -> np.ndarray:
        if not self._scan_records:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack(
            [self._corrected_pose_from_raw(record.raw_pose_xyyaw) for record in self._scan_records]
        ).astype(np.float64)

    def _raw_path_poses(self) -> np.ndarray:
        if not self._scan_records:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([record.raw_pose_xyyaw for record in self._scan_records]).astype(np.float64)

    def _publish_outputs(
        self,
        record: ScanRecord,
        *,
        vx_mps: float,
        vy_mps: float,
        reference_mode: str,
    ) -> None:
        stamp_msg = self.get_clock().now().to_msg()
        raw_pose = np.asarray(record.raw_pose_xyyaw, dtype=np.float64)
        corrected_pose = self._corrected_pose_from_raw(raw_pose)

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child_frame
        odom.pose.pose.position.x = float(raw_pose[0])
        odom.pose.pose.position.y = float(raw_pose[1])
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(float(raw_pose[2]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(vx_mps)
        odom.twist.twist.linear.y = float(vy_mps)
        odom.twist.twist.angular.z = float(
            self._imu_records[-1].yaw_rate_rps if self._imu_records else 0.0
        )
        self._odom_pub.publish(odom)

        pose_msg = PoseStamped()
        pose_msg.header = odom.header
        pose_msg.pose = odom.pose.pose
        self._pose_pub.publish(pose_msg)

        corrected_odom = Odometry()
        corrected_odom.header.stamp = stamp_msg
        corrected_odom.header.frame_id = self._corrected_frame
        corrected_odom.child_frame_id = self._child_frame
        corrected_odom.pose.pose.position.x = float(corrected_pose[0])
        corrected_odom.pose.pose.position.y = float(corrected_pose[1])
        corrected_odom.pose.pose.position.z = 0.0
        cqx, cqy, cqz, cqw = _yaw_to_quat(float(corrected_pose[2]))
        corrected_odom.pose.pose.orientation.x = cqx
        corrected_odom.pose.pose.orientation.y = cqy
        corrected_odom.pose.pose.orientation.z = cqz
        corrected_odom.pose.pose.orientation.w = cqw
        corrected_odom.twist.twist = odom.twist.twist
        self._corrected_odom_pub.publish(corrected_odom)

        corrected_pose_msg = PoseStamped()
        corrected_pose_msg.header = corrected_odom.header
        corrected_pose_msg.pose = corrected_odom.pose.pose
        self._corrected_pose_pub.publish(corrected_pose_msg)

        self._path_pub.publish(self._path_message_from_poses(self._raw_path_poses(), stamp_msg, self._odom_frame))
        self._corrected_path_pub.publish(
            self._path_message_from_poses(self._corrected_path_poses(), stamp_msg, self._corrected_frame)
        )

        raw_map_points = self._build_submap_points()
        corrected_map_points = self._corrected_map_points()
        self._live_map_pub.publish(self._pointcloud_message_from_xy(raw_map_points, stamp_msg, self._odom_frame))
        self._corrected_live_map_pub.publish(
            self._pointcloud_message_from_xy(corrected_map_points, stamp_msg, self._corrected_frame)
        )
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
            reference_mode=reference_mode,
        )

    def _publish_status(
        self,
        state: str,
        latest_record: ScanRecord | None,
        median_recent_residual_m: float,
        vx_mps: float,
        vy_mps: float,
        *,
        reference_mode: str,
    ) -> None:
        latest_pose = None
        if latest_record is not None:
            raw_pose = np.asarray(latest_record.raw_pose_xyyaw, dtype=np.float64)
            corrected_pose = self._corrected_pose_from_raw(raw_pose)
            latest_pose = {
                "t_s": float(latest_record.stamp_s),
                "x_m": float(raw_pose[0]),
                "y_m": float(raw_pose[1]),
                "yaw_rad": float(raw_pose[2]),
                "vx_mps": float(vx_mps),
                "vy_mps": float(vy_mps),
                "confidence": str(latest_record.confidence),
                "median_submap_residual_m": float(latest_record.median_residual_m),
                "inlier_ratio": float(latest_record.inlier_ratio),
                "scan_inserted": bool(latest_record.inserted_into_map),
            }
            latest_pose["corrected_x_m"] = float(corrected_pose[0])
            latest_pose["corrected_y_m"] = float(corrected_pose[1])
            latest_pose["corrected_yaw_rad"] = float(corrected_pose[2])
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
                "latest_inlier_ratio": float(latest_record.inlier_ratio) if latest_record is not None else 0.0,
                "latest_scan_inserted": bool(latest_record.inserted_into_map) if latest_record is not None else False,
            },
            "latest_pose": latest_pose,
            "external_prior": {
                "enabled": bool(self._use_external_prior),
                "topic": self._prior_odom_topic,
                "active": bool(self._use_external_prior and len(self._prior_odom_records) > 0),
                "weight_xy": float(self._external_prior_weight_xy),
                "weight_yaw": float(self._external_prior_weight_yaw),
            },
            "offline_correction": {
                "enabled": bool(self._use_offline_correction),
                "topic": self._offline_correction_topic,
                "use_yaw": bool(self._offline_correction_use_yaw),
                "child_frame_id": self._latest_offline_correction_child_frame_id,
                "applied_x_m": float(self._latest_offline_correction_applied[0]),
                "applied_y_m": float(self._latest_offline_correction_applied[1]),
                "applied_yaw_rad": float(self._latest_offline_correction_applied[2]),
                "applied_translation_norm_m": float(np.linalg.norm(self._latest_offline_correction_applied[:2])),
            },
            "reference": {
                "mode": reference_mode,
                "using_offline_submap": bool(reference_mode == "offline_submap"),
                "using_offline_grid": bool(reference_mode == "offline_grid"),
            },
            "parameters": {
                "submap_window_scans": self._submap_window_scans,
                "point_stride": self._point_stride,
                "max_correspondence_m": self._max_correspondence_m,
                "backend": "distance_field_online_seed",
                "imu_filter_alpha": float(self._imu_filter_alpha),
                "accel_bias_x_mps2": float(self._accel_bias_x_mps2),
                "accel_bias_y_mps2": float(self._accel_bias_y_mps2),
                "accel_bias_z_mps2": float(self._accel_bias_z_mps2),
                "yaw_bias_rps": float(self._yaw_bias_rps),
                "freeze_scan_insertion_on_low_confidence": bool(
                    self._freeze_scan_insertion_on_low_confidence
                ),
                "local_prior_weight_xy": float(self._local_prior_weight_xy),
                "local_prior_weight_yaw": float(self._local_prior_weight_yaw),
                "lidar_residual_weight": float(self._lidar_residual_weight),
                "low_confidence_residual_threshold_m": float(
                    self._low_confidence_residual_threshold_m
                ),
                "low_confidence_inlier_ratio_threshold": float(
                    self._low_confidence_inlier_ratio_threshold
                ),
            },
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._latest_status = message.data
        self._status_pub.publish(message)

    def _path_message_from_poses(self, poses_xyyaw: np.ndarray, stamp_msg, frame_id: str) -> Path:
        path_msg = Path()
        path_msg.header.stamp = stamp_msg
        path_msg.header.frame_id = frame_id
        for pose_xyyaw in poses_xyyaw[-4000:]:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp_msg
            pose_msg.header.frame_id = frame_id
            pose_msg.pose.position.x = float(pose_xyyaw[0])
            pose_msg.pose.position.y = float(pose_xyyaw[1])
            qx, qy, qz, qw = _yaw_to_quat(float(pose_xyyaw[2]))
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            path_msg.poses.append(pose_msg)
        return path_msg

    def _pointcloud_message_from_xy(
        self,
        map_points_xy: np.ndarray,
        stamp_msg,
        frame_id: str,
    ) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp = stamp_msg
        message.header.frame_id = frame_id
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

    def _pointcloud_xy_from_message(self, msg: PointCloud2) -> np.ndarray:
        if msg.width == 0 or not msg.data:
            return np.empty((0, 2), dtype=np.float64)
        if msg.point_step < 8:
            return np.empty((0, 2), dtype=np.float64)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        point_count = len(raw) // msg.point_step
        if point_count <= 0:
            return np.empty((0, 2), dtype=np.float64)
        cloud_xy = np.zeros((point_count, 2), dtype=np.float64)
        for index in range(point_count):
            offset = index * msg.point_step
            cloud_xy[index, 0] = np.frombuffer(raw[offset : offset + 4], dtype=np.float32, count=1)[0]
            cloud_xy[index, 1] = np.frombuffer(raw[offset + 4 : offset + 8], dtype=np.float32, count=1)[0]
        return _voxel_downsample(cloud_xy, 0.04)

    def _occupied_xy_from_grid(self, msg: OccupancyGrid) -> np.ndarray:
        if msg.info.width <= 0 or msg.info.height <= 0 or not msg.data:
            return np.empty((0, 2), dtype=np.float64)
        occupancy = np.asarray(msg.data, dtype=np.int16).reshape(
            (int(msg.info.height), int(msg.info.width))
        )
        occupied_y, occupied_x = np.where(occupancy >= 50)
        if occupied_x.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        resolution = float(msg.info.resolution)
        origin_x = float(msg.info.origin.position.x)
        origin_y = float(msg.info.origin.position.y)
        points_xy = np.column_stack(
            (
                origin_x + ((occupied_x.astype(np.float64) + 0.5) * resolution),
                origin_y + ((occupied_y.astype(np.float64) + 0.5) * resolution),
            )
        )
        return _voxel_downsample(points_xy.astype(np.float64), 0.04)


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
