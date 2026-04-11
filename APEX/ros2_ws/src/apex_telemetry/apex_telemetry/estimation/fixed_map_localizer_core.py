#!/usr/bin/env python3
"""Fixed-map LiDAR + IMU localization for APEX."""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares

from .planar_fusion_core import FusionStateSnapshot, LidarScanObservation, ScanEstimate


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64)


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(pose_xyyaw[2])).T) + pose_xyyaw[:2]


def _subsample_evenly(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    if points_xy.size == 0 or points_xy.shape[0] <= max_points:
        return points_xy.copy()
    indexes = np.linspace(0, points_xy.shape[0] - 1, num=max_points, dtype=np.int32)
    return points_xy[indexes]


def _median_abs_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _robust_clip_scalar(value: float, history: deque[float], mad_scale: float, abs_limit: float) -> float:
    if not history:
        return float(np.clip(value, -abs_limit, abs_limit))
    hist = np.asarray(history, dtype=np.float64)
    median = float(np.median(hist))
    mad = max(1.0e-3, 1.4826 * _median_abs_deviation(hist))
    lower = max(-abs_limit, median - (mad_scale * mad))
    upper = min(abs_limit, median + (mad_scale * mad))
    return float(np.clip(value, lower, upper))


def _blend_pose_xyyaw(prior_pose: np.ndarray, corrected_pose: np.ndarray, correction_gain: float) -> np.ndarray:
    gain = max(0.0, min(1.0, float(correction_gain)))
    blended = prior_pose.astype(np.float64).copy()
    blended[:2] = prior_pose[:2] + (gain * (corrected_pose[:2] - prior_pose[:2]))
    yaw_delta = _normalize_angle(float(corrected_pose[2] - prior_pose[2]))
    blended[2] = _normalize_angle(float(prior_pose[2]) + (gain * yaw_delta))
    return blended


def _load_visual_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                points.append([float(row["x_m"]), float(row["y_m"])])
            except Exception:
                continue
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _load_route_poses(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64)
    poses: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                poses.append([float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])])
            except Exception:
                continue
    if not poses:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(poses, dtype=np.float64)


def _weighted_circular_mean(angles_rad: np.ndarray, weights: np.ndarray) -> float:
    if angles_rad.size == 0:
        return 0.0
    sin_mean = float(np.sum(weights * np.sin(angles_rad)))
    cos_mean = float(np.sum(weights * np.cos(angles_rad)))
    return math.atan2(sin_mean, cos_mean)


def _weighted_yaw_spread(angles_rad: np.ndarray, weights: np.ndarray) -> float:
    if angles_rad.size == 0:
        return 0.0
    mean_yaw = _weighted_circular_mean(angles_rad, weights)
    deltas = np.asarray([_normalize_angle(float(value - mean_yaw)) for value in angles_rad], dtype=np.float64)
    return float(math.sqrt(max(0.0, float(np.sum(weights * np.square(deltas))))))


@dataclass(frozen=True)
class FixedMapParameters:
    fixed_map_yaml: str
    fixed_map_distance_npy: str = ""
    fixed_map_visual_points_csv: str = ""
    velocity_decay_tau_s: float = 1.1
    max_match_points: int = 120
    max_localization_iterations: int = 40
    max_correspondence_m: float = 0.40
    prior_translation_weight: float = 1.2
    prior_yaw_weight: float = 1.8
    startup_static_duration_s: float = 1.5
    startup_gyro_stddev_threshold_rps: float = 0.04
    startup_accel_stddev_threshold_mps2: float = 0.35
    imu_filter_window: int = 7
    imu_spike_mad_scale: float = 4.0
    max_accel_axis_mps2: float = 3.2
    max_yaw_rate_abs_rps: float = 2.6
    low_support_distance_m: float = 0.90
    low_support_in_bounds_ratio: float = 0.45
    low_support_near_wall_ratio: float = 0.25
    reduced_support_prior_gain: float = 1.35
    low_support_prior_gain: float = 1.80
    motion_hint_timeout_s: float = 0.45
    motion_hint_velocity_blend: float = 0.30
    motion_hint_max_speed_mps: float = 1.5
    fixed_map_route_csv: str = ""
    particle_count: int = 384
    particle_seed: int = 17
    particle_initial_xy_std_m: float = 0.28
    particle_initial_yaw_std_rad: float = 0.35
    particle_route_seed_fraction: float = 0.18
    particle_random_injection_ratio: float = 0.025
    particle_process_xy_std_m: float = 0.035
    particle_process_yaw_std_rad: float = 0.030
    particle_process_velocity_std_mps: float = 0.06
    particle_likelihood_sigma_m: float = 0.075
    particle_inlier_distance_m: float = 0.13
    particle_inlier_log_weight: float = 2.0
    particle_out_of_map_penalty: float = 2.4
    particle_resample_neff_ratio: float = 0.55
    particle_roughening_xy_std_m: float = 0.018
    particle_roughening_yaw_std_rad: float = 0.025
    particle_min_lidar_points: int = 12
    particle_min_observation_weight: float = 0.20
    particle_high_confidence_inlier_ratio: float = 0.45
    particle_medium_confidence_inlier_ratio: float = 0.28
    particle_max_high_confidence_spread_m: float = 0.55
    particle_max_medium_confidence_spread_m: float = 0.90
    particle_refine_enabled: bool = True
    particle_refine_gain: float = 0.70


class FixedMapPlanarLocalizer:
    def __init__(self, params: FixedMapParameters) -> None:
        self._params = params
        self._map_yaml_path = Path(params.fixed_map_yaml).expanduser().resolve()
        if not self._map_yaml_path.exists():
            raise FileNotFoundError(f"Missing fixed_map_yaml: {self._map_yaml_path}")
        payload = yaml.safe_load(self._map_yaml_path.read_text(encoding="utf-8")) or {}
        self._resolution_m = float(payload["resolution"])
        self._origin_xy = np.asarray(payload["origin"][:2], dtype=np.float64)
        self._distance_field_path = Path(
            params.fixed_map_distance_npy or (self._map_yaml_path.parent / str(payload["distance_field_npy"]))
        ).expanduser().resolve()
        self._visual_points_path = Path(
            params.fixed_map_visual_points_csv or (self._map_yaml_path.parent / str(payload["visual_points_csv"]))
        ).expanduser().resolve()
        self._distance_field = np.asarray(np.load(self._distance_field_path), dtype=np.float32)
        self._height = int(self._distance_field.shape[0])
        self._width = int(self._distance_field.shape[1])
        self._initial_pose = np.asarray(payload.get("initial_pose", [0.0, 0.0, 0.0]), dtype=np.float64)
        self._visual_points_xy = _load_visual_points(self._visual_points_path)

        self._state = "waiting_static_initialization"
        self._imu_initialized = False
        self._best_effort_init = False
        self._raw_imu_sample_count = 0
        self._processed_imu_sample_count = 0
        self._processed_scan_count = 0

        self._init_samples: deque[tuple[float, float, float, float, float]] = deque()
        self._gyro_bias_rps = 0.0
        self._accel_bias_xy = np.zeros(2, dtype=np.float64)
        self._static_start_s: float | None = None
        self._static_end_s: float | None = None
        self._static_sample_count = 0

        self._last_imu_t_s: float | None = None
        self._integrated_yaw_rad = float(self._initial_pose[2])
        self._yaw_offset_rad = 0.0
        self._velocity_world_xy = np.zeros(2, dtype=np.float64)
        self._latest_yaw_rate_rps = 0.0
        self._recent_gyro_z_rps: deque[float] = deque(maxlen=max(3, int(params.imu_filter_window)))
        self._recent_accel_x_mps2: deque[float] = deque(maxlen=max(3, int(params.imu_filter_window)))
        self._recent_accel_y_mps2: deque[float] = deque(maxlen=max(3, int(params.imu_filter_window)))

        self._last_estimate: ScanEstimate | None = None
        self._path_estimates: list[ScanEstimate] = []
        self._median_residual_history: list[float] = []
        self._last_support_metrics = {
            "in_bounds_ratio": 0.0,
            "near_wall_ratio": 0.0,
            "median_distance_m": float("nan"),
            "dynamic_prior_gain": 1.0,
        }

    def add_imu_sample(
        self,
        *,
        t_s: float,
        ax_mps2: float,
        ay_mps2: float,
        az_mps2: float,
        gz_rps: float,
    ) -> None:
        del az_mps2
        self._raw_imu_sample_count += 1
        if self._static_start_s is None:
            self._static_start_s = t_s
        if not self._imu_initialized:
            self._init_samples.append((t_s, ax_mps2, ay_mps2, gz_rps, 0.0))
            self._maybe_finalize_static_initialization(t_s)
            self._last_imu_t_s = t_s
            return

        if self._last_imu_t_s is None:
            self._last_imu_t_s = t_s
            return
        dt_s = max(0.0, min(0.05, float(t_s - self._last_imu_t_s)))
        self._last_imu_t_s = t_s
        raw_yaw_rate = float(gz_rps - self._gyro_bias_rps)
        raw_accel_x = float(ax_mps2 - self._accel_bias_xy[0])
        raw_accel_y = float(ay_mps2 - self._accel_bias_xy[1])
        filtered_yaw_rate = _robust_clip_scalar(
            raw_yaw_rate,
            self._recent_gyro_z_rps,
            float(self._params.imu_spike_mad_scale),
            float(self._params.max_yaw_rate_abs_rps),
        )
        filtered_accel_x = _robust_clip_scalar(
            raw_accel_x,
            self._recent_accel_x_mps2,
            float(self._params.imu_spike_mad_scale),
            float(self._params.max_accel_axis_mps2),
        )
        filtered_accel_y = _robust_clip_scalar(
            raw_accel_y,
            self._recent_accel_y_mps2,
            float(self._params.imu_spike_mad_scale),
            float(self._params.max_accel_axis_mps2),
        )
        self._recent_gyro_z_rps.append(filtered_yaw_rate)
        self._recent_accel_x_mps2.append(filtered_accel_x)
        self._recent_accel_y_mps2.append(filtered_accel_y)
        corrected_yaw_rate = filtered_yaw_rate
        self._latest_yaw_rate_rps = corrected_yaw_rate
        self._integrated_yaw_rad = _normalize_angle(self._integrated_yaw_rad + (corrected_yaw_rate * dt_s))

        accel_body = np.asarray(
            [
                filtered_accel_x,
                filtered_accel_y,
            ],
            dtype=np.float64,
        )
        accel_world = _rotation_matrix(self._integrated_yaw_rad) @ accel_body
        if dt_s > 0.0:
            decay = math.exp(-dt_s / max(0.1, self._params.velocity_decay_tau_s))
            self._velocity_world_xy = (decay * self._velocity_world_xy) + (accel_world * dt_s)
        self._processed_imu_sample_count += 1

    def _maybe_finalize_static_initialization(self, now_s: float) -> None:
        if len(self._init_samples) < 8 or self._static_start_s is None:
            return
        duration_s = float(now_s - self._static_start_s)
        if duration_s < self._params.startup_static_duration_s:
            return
        sample_array = np.asarray(self._init_samples, dtype=np.float64)
        gyro_std = float(np.std(sample_array[:, 3]))
        accel_std = float(
            max(np.std(sample_array[:, 1]), np.std(sample_array[:, 2]))
        )
        self._best_effort_init = (
            gyro_std > self._params.startup_gyro_stddev_threshold_rps
            or accel_std > self._params.startup_accel_stddev_threshold_mps2
        )
        self._gyro_bias_rps = float(np.mean(sample_array[:, 3]))
        self._accel_bias_xy = np.asarray(
            [
                float(np.mean(sample_array[:, 1])),
                float(np.mean(sample_array[:, 2])),
            ],
            dtype=np.float64,
        )
        self._imu_initialized = True
        self._state = "localizing"
        self._static_end_s = now_s
        self._static_sample_count = int(sample_array.shape[0])
        self._integrated_yaw_rad = float(self._initial_pose[2])
        self._last_imu_t_s = now_s
        self._processed_imu_sample_count = int(sample_array.shape[0])

    def _predict_pose(self, t_s: float) -> np.ndarray:
        if self._last_estimate is None:
            return self._initial_pose.copy()
        dt_s = max(0.0, min(0.30, float(t_s - self._last_estimate.t_s)))
        pose = np.asarray(
            [
                float(self._last_estimate.x_m),
                float(self._last_estimate.y_m),
                float(self._last_estimate.yaw_rad),
            ],
            dtype=np.float64,
        )
        pose[:2] += self._velocity_world_xy * dt_s
        pose[2] = _normalize_angle(self._integrated_yaw_rad + self._yaw_offset_rad)
        return pose

    def _sample_distance_field(self, world_points_xy: np.ndarray) -> np.ndarray:
        distances, _ = self._sample_distance_field_with_valid(world_points_xy)
        return distances

    def _sample_distance_field_with_valid(self, world_points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if world_points_xy.size == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=bool)
        gx = (world_points_xy[:, 0] - self._origin_xy[0]) / self._resolution_m
        gy = (world_points_xy[:, 1] - self._origin_xy[1]) / self._resolution_m
        x0 = np.floor(gx).astype(np.int32)
        y0 = np.floor(gy).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (
            (x0 >= 0)
            & (x1 < self._width)
            & (y0 >= 0)
            & (y1 < self._height)
        )
        distances = np.full((world_points_xy.shape[0],), self._params.max_correspondence_m * 2.0, dtype=np.float64)
        if not np.any(valid):
            return distances, valid
        gxv = gx[valid]
        gyv = gy[valid]
        x0v = x0[valid]
        y0v = y0[valid]
        x1v = x1[valid]
        y1v = y1[valid]
        wx = gxv - x0v
        wy = gyv - y0v
        d00 = self._distance_field[y0v, x0v]
        d10 = self._distance_field[y0v, x1v]
        d01 = self._distance_field[y1v, x0v]
        d11 = self._distance_field[y1v, x1v]
        distances_valid = (
            ((1.0 - wx) * (1.0 - wy) * d00)
            + (wx * (1.0 - wy) * d10)
            + ((1.0 - wx) * wy * d01)
            + (wx * wy * d11)
        )
        distances[valid] = distances_valid.astype(np.float64)
        return distances, valid

    def _support_metrics(self, points_local: np.ndarray, pose_xyyaw: np.ndarray) -> dict[str, float]:
        if points_local.size == 0:
            return {
                "in_bounds_ratio": 0.0,
                "near_wall_ratio": 0.0,
                "median_distance_m": float("nan"),
            }
        world_points = _transform_points(points_local, pose_xyyaw)
        distances_m, valid_mask = self._sample_distance_field_with_valid(world_points)
        in_bounds_ratio = float(np.mean(valid_mask.astype(np.float64))) if valid_mask.size > 0 else 0.0
        if not np.any(valid_mask):
            return {
                "in_bounds_ratio": in_bounds_ratio,
                "near_wall_ratio": 0.0,
                "median_distance_m": float("inf"),
            }
        valid_distances = distances_m[valid_mask]
        near_wall_ratio = float(
            np.mean((valid_distances <= float(self._params.low_support_distance_m)).astype(np.float64))
        )
        return {
            "in_bounds_ratio": in_bounds_ratio,
            "near_wall_ratio": near_wall_ratio,
            "median_distance_m": float(np.median(valid_distances)) if valid_distances.size > 0 else float("inf"),
        }

    def _dynamic_prior_gain(self, support_metrics: dict[str, float]) -> float:
        if (
            float(support_metrics.get("in_bounds_ratio", 0.0)) < float(self._params.low_support_in_bounds_ratio)
            or float(support_metrics.get("near_wall_ratio", 0.0)) < float(self._params.low_support_near_wall_ratio)
        ):
            return float(self._params.low_support_prior_gain)
        if (
            float(support_metrics.get("in_bounds_ratio", 0.0)) < 0.65
            or float(support_metrics.get("near_wall_ratio", 0.0)) < 0.40
        ):
            return float(self._params.reduced_support_prior_gain)
        return 1.0

    def _optimize_pose(self, points_local: np.ndarray, predicted_pose: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        match_points = _subsample_evenly(points_local, self._params.max_match_points)
        if match_points.shape[0] < 8:
            support_metrics = self._support_metrics(match_points, predicted_pose)
            support_metrics["dynamic_prior_gain"] = self._dynamic_prior_gain(support_metrics)
            return predicted_pose.copy(), float("nan"), support_metrics

        predicted_support_metrics = self._support_metrics(match_points, predicted_pose)
        dynamic_prior_gain = self._dynamic_prior_gain(predicted_support_metrics)
        prior_translation_weight = float(self._params.prior_translation_weight) * dynamic_prior_gain
        prior_yaw_weight = float(self._params.prior_yaw_weight) * max(1.0, 0.9 * dynamic_prior_gain)

        def _residuals(state_vec: np.ndarray) -> np.ndarray:
            world_points = _transform_points(match_points, state_vec)
            distances_m = self._sample_distance_field(world_points)
            distances_m = np.clip(distances_m, 0.0, self._params.max_correspondence_m * 1.5)
            prior = np.asarray(
                [
                    prior_translation_weight * float(state_vec[0] - predicted_pose[0]),
                    prior_translation_weight * float(state_vec[1] - predicted_pose[1]),
                    prior_yaw_weight * _normalize_angle(float(state_vec[2] - predicted_pose[2])),
                ],
                dtype=np.float64,
            )
            return np.concatenate((distances_m, prior))

        result = least_squares(
            _residuals,
            x0=predicted_pose.astype(np.float64),
            max_nfev=int(self._params.max_localization_iterations),
            loss="soft_l1",
            f_scale=max(0.05, 0.5 * float(self._params.max_correspondence_m)),
        )
        pose = result.x.astype(np.float64)
        pose[2] = _normalize_angle(float(pose[2]))
        corrected_support_metrics = self._support_metrics(match_points, pose)
        corrected_support_metrics["dynamic_prior_gain"] = dynamic_prior_gain
        median_residual_m = float(np.median(self._sample_distance_field(_transform_points(match_points, pose))))
        return pose, median_residual_m, corrected_support_metrics

    def add_scan_observation(self, scan: LidarScanObservation) -> list[ScanEstimate]:
        if not self._imu_initialized:
            return []
        predicted_pose = self._predict_pose(float(scan.t_s))
        corrected_pose, median_residual_m, support_metrics = self._optimize_pose(
            scan.sampled_points_local,
            predicted_pose,
        )
        support_gain = float(support_metrics.get("dynamic_prior_gain", 1.0))
        correction_gain = 1.0
        if support_gain >= float(self._params.low_support_prior_gain):
            correction_gain = 0.28
        elif support_gain >= float(self._params.reduced_support_prior_gain):
            correction_gain = 0.55
        corrected_pose = _blend_pose_xyyaw(predicted_pose, corrected_pose, correction_gain)
        self._last_support_metrics = dict(support_metrics)
        self._last_support_metrics["dynamic_prior_gain"] = support_gain

        if self._last_estimate is not None:
            dt_s = max(1.0e-3, float(scan.t_s - self._last_estimate.t_s))
            measured_velocity = (corrected_pose[:2] - np.asarray([self._last_estimate.x_m, self._last_estimate.y_m])) / dt_s
            self._velocity_world_xy = (0.65 * self._velocity_world_xy) + (0.35 * measured_velocity)
            self._yaw_offset_rad = _normalize_angle(float(corrected_pose[2] - self._integrated_yaw_rad))
        else:
            self._yaw_offset_rad = _normalize_angle(float(corrected_pose[2] - self._integrated_yaw_rad))

        if (
            math.isfinite(median_residual_m)
            and median_residual_m <= 0.12
            and float(support_metrics.get("in_bounds_ratio", 0.0)) >= 0.70
            and float(support_metrics.get("near_wall_ratio", 0.0)) >= 0.45
        ):
            confidence = "high"
        elif (
            math.isfinite(median_residual_m)
            and median_residual_m <= 0.20
            and float(support_metrics.get("in_bounds_ratio", 0.0)) >= 0.45
        ):
            confidence = "medium"
        else:
            confidence = "low"
        estimate = ScanEstimate(
            scan_index=int(scan.scan_index),
            stamp_sec=int(scan.stamp_sec),
            stamp_nanosec=int(scan.stamp_nanosec),
            t_s=float(scan.t_s),
            x_m=float(corrected_pose[0]),
            y_m=float(corrected_pose[1]),
            yaw_rad=float(corrected_pose[2]),
            vx_mps=float(self._velocity_world_xy[0]),
            vy_mps=float(self._velocity_world_xy[1]),
            yaw_rate_rps=float(self._latest_yaw_rate_rps),
            ax_world_mps2=0.0,
            ay_world_mps2=0.0,
            confidence=confidence,
            median_submap_residual_m=float(median_residual_m),
            median_wall_residual_m=float("nan"),
            valid_correspondence_count=int(scan.sampled_points_local.shape[0]),
            alignment_ready=True,
            imu_initialized=True,
            best_effort_init=bool(self._best_effort_init),
        )
        self._last_estimate = estimate
        self._path_estimates.append(estimate)
        self._processed_scan_count += 1
        if math.isfinite(median_residual_m):
            self._median_residual_history.append(float(median_residual_m))
        self._state = "tracking"
        return [estimate]

    def predict_estimate(self, query_t_s: float) -> ScanEstimate | None:
        if self._last_estimate is None or not self._imu_initialized:
            return None
        predicted_pose = self._predict_pose(query_t_s)
        return ScanEstimate(
            scan_index=int(self._last_estimate.scan_index),
            stamp_sec=int(query_t_s),
            stamp_nanosec=int((query_t_s - int(query_t_s)) * 1.0e9),
            t_s=float(query_t_s),
            x_m=float(predicted_pose[0]),
            y_m=float(predicted_pose[1]),
            yaw_rad=float(predicted_pose[2]),
            vx_mps=float(self._velocity_world_xy[0]),
            vy_mps=float(self._velocity_world_xy[1]),
            yaw_rate_rps=float(self._latest_yaw_rate_rps),
            ax_world_mps2=0.0,
            ay_world_mps2=0.0,
            confidence=str(self._last_estimate.confidence),
            median_submap_residual_m=float(self._last_estimate.median_submap_residual_m),
            median_wall_residual_m=float("nan"),
            valid_correspondence_count=int(self._last_estimate.valid_correspondence_count),
            alignment_ready=True,
            imu_initialized=True,
            best_effort_init=bool(self._best_effort_init),
        )

    def live_map_points_world(self, *, window_scans: int, max_points: int) -> np.ndarray:
        del window_scans
        if self._visual_points_xy.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self._last_estimate is None:
            return _subsample_evenly(self._visual_points_xy, max_points)
        pose_xy = np.asarray([self._last_estimate.x_m, self._last_estimate.y_m], dtype=np.float64)
        distances = np.linalg.norm(self._visual_points_xy - pose_xy[None, :], axis=1)
        sorted_indexes = np.argsort(distances)
        return self._visual_points_xy[sorted_indexes[: max(1, min(max_points, sorted_indexes.size))]]

    def full_map_points_world(self, *, max_points: int) -> np.ndarray:
        return _subsample_evenly(self._visual_points_xy, max_points)

    def status_snapshot(self) -> FusionStateSnapshot:
        latest_pose = None
        if self._last_estimate is not None:
            latest_pose = {
                "x_m": float(self._last_estimate.x_m),
                "y_m": float(self._last_estimate.y_m),
                "yaw_rad": float(self._last_estimate.yaw_rad),
                "vx_mps": float(self._last_estimate.vx_mps),
                "vy_mps": float(self._last_estimate.vy_mps),
                "yaw_rate_rps": float(self._last_estimate.yaw_rate_rps),
                "confidence": str(self._last_estimate.confidence),
                "stamp_s": float(self._last_estimate.t_s),
            }
        finite_residuals = [value for value in self._median_residual_history if math.isfinite(value)]
        median_residual = float(np.median(np.asarray(finite_residuals, dtype=np.float64))) if finite_residuals else float("nan")
        return FusionStateSnapshot(
            state=str(self._state),
            imu_initialized=bool(self._imu_initialized),
            alignment_ready=True,
            best_effort_init=bool(self._best_effort_init),
            raw_imu_sample_count=int(self._raw_imu_sample_count),
            processed_imu_sample_count=int(self._processed_imu_sample_count),
            pending_scan_count=0,
            processed_scan_count=int(self._processed_scan_count),
            initial_scan_count=1 if self._processed_scan_count > 0 else 0,
            alignment_yaw_rad=0.0,
            origin_projection_m=(0.0, 0.0),
            static_initialization={
                "best_effort_init": bool(self._best_effort_init),
                "static_start_s": float(self._static_start_s or 0.0),
                "static_end_s": float(self._static_end_s or 0.0),
                "static_sample_count": int(self._static_sample_count),
                "gyro_bias_z_rps": float(self._gyro_bias_rps),
                "accel_bias_x_mps2": float(self._accel_bias_xy[0]),
                "accel_bias_y_mps2": float(self._accel_bias_xy[1]),
            },
            corridor_model=None,
            quality={
                "median_submap_residual_m": median_residual,
                "median_wall_residual_m": float("nan"),
                "high_confidence_pct": 100.0
                * (
                    sum(1 for estimate in self._path_estimates if estimate.confidence == "high")
                    / max(1, len(self._path_estimates))
                ),
                "map_loaded": True,
                "visual_point_count": int(self._visual_points_xy.shape[0]),
                "map_support_in_bounds_ratio": float(self._last_support_metrics.get("in_bounds_ratio", 0.0)),
                "map_support_near_wall_ratio": float(self._last_support_metrics.get("near_wall_ratio", 0.0)),
                "map_support_median_distance_m": float(
                    self._last_support_metrics.get("median_distance_m", float("nan"))
                ),
                "dynamic_prior_gain": float(self._last_support_metrics.get("dynamic_prior_gain", 1.0)),
            },
            latest_pose=latest_pose,
            parameters=asdict(self._params),
        )


FixedMapSingleHypothesisLocalizer = FixedMapPlanarLocalizer


class FixedMapParticleLocalizer(FixedMapSingleHypothesisLocalizer):
    """Multi-hypothesis fixed-map localizer using a compact AMCL-style filter.

    The prior single-pose optimizer is still used as a final local refinement,
    but only after the particle cloud has selected a plausible basin. This keeps
    the existing ROS interface while reducing false local convergence when the
    initial pose or IMU yaw prediction is imperfect.
    """

    def __init__(self, params: FixedMapParameters) -> None:
        super().__init__(params)
        self._rng = np.random.default_rng(int(params.particle_seed))
        self._particle_count = max(32, int(params.particle_count))
        self._route_poses_xyyaw = self._load_route_seed_poses()
        self._particles_xyyaw = np.zeros((self._particle_count, 3), dtype=np.float64)
        self._weights = np.full((self._particle_count,), 1.0 / float(self._particle_count), dtype=np.float64)
        self._last_particle_metrics: dict[str, float | int | bool | str] = {
            "enabled": True,
            "particle_count": int(self._particle_count),
            "neff_ratio": 1.0,
            "xy_spread_m": 0.0,
            "yaw_spread_rad": 0.0,
            "best_inlier_ratio": 0.0,
            "best_in_bounds_ratio": 0.0,
            "best_median_distance_m": float("nan"),
            "observation_weight": 0.0,
            "confidence_score": 0.0,
            "resampled": False,
        }
        self._motion_hint_t_s: float | None = None
        self._motion_hint_velocity_world_xy = np.zeros(2, dtype=np.float64)
        self._reset_particles()

    def set_motion_hint(
        self,
        *,
        t_s: float,
        vx_mps: float,
        vy_mps: float,
    ) -> None:
        velocity_xy = np.asarray([float(vx_mps), float(vy_mps)], dtype=np.float64)
        if not np.all(np.isfinite(velocity_xy)):
            return
        speed_mps = float(np.linalg.norm(velocity_xy))
        if speed_mps > float(self._params.motion_hint_max_speed_mps):
            if speed_mps <= 1.0e-6:
                velocity_xy = np.zeros(2, dtype=np.float64)
            else:
                velocity_xy *= float(self._params.motion_hint_max_speed_mps) / speed_mps
        self._motion_hint_t_s = float(t_s)
        self._motion_hint_velocity_world_xy = velocity_xy

    def _motion_hint_is_fresh(self, t_s: float) -> bool:
        return (
            self._motion_hint_t_s is not None
            and 0.0 <= float(t_s - self._motion_hint_t_s) <= float(self._params.motion_hint_timeout_s)
        )

    def _load_route_seed_poses(self) -> np.ndarray:
        route_csv = str(self._params.fixed_map_route_csv or "").strip()
        if not route_csv:
            try:
                payload = yaml.safe_load(self._map_yaml_path.read_text(encoding="utf-8")) or {}
                route_csv = str(payload.get("optimized_keyframes_csv", "") or "")
            except Exception:
                route_csv = ""
        if not route_csv:
            return np.empty((0, 3), dtype=np.float64)
        route_path = Path(route_csv).expanduser()
        if not route_path.is_absolute():
            route_path = self._map_yaml_path.parent / route_path
        return _load_route_poses(route_path)

    def _sample_seed_poses(self, count: int, *, broad: bool = False) -> np.ndarray:
        count = max(0, int(count))
        if count == 0:
            return np.empty((0, 3), dtype=np.float64)
        route_fraction = (
            0.0
            if self._route_poses_xyyaw.size == 0
            else max(0.0, min(0.80, float(self._params.particle_route_seed_fraction)))
        )
        route_count = int(round(float(count) * route_fraction))
        initial_count = count - route_count
        chunks: list[np.ndarray] = []
        if initial_count > 0:
            initial = np.repeat(self._initial_pose.reshape(1, 3), initial_count, axis=0)
            xy_std = float(self._params.particle_initial_xy_std_m) * (2.5 if broad else 1.0)
            yaw_std = float(self._params.particle_initial_yaw_std_rad) * (2.0 if broad else 1.0)
            initial[:, :2] += self._rng.normal(0.0, xy_std, size=(initial_count, 2))
            initial[:, 2] += self._rng.normal(0.0, yaw_std, size=(initial_count,))
            chunks.append(initial)
        if route_count > 0:
            indexes = self._rng.integers(0, self._route_poses_xyyaw.shape[0], size=route_count)
            route = self._route_poses_xyyaw[indexes].astype(np.float64).copy()
            route[:, :2] += self._rng.normal(
                0.0,
                max(0.05, 0.65 * float(self._params.particle_initial_xy_std_m)),
                size=(route_count, 2),
            )
            route[:, 2] += self._rng.normal(
                0.0,
                max(0.05, 0.65 * float(self._params.particle_initial_yaw_std_rad)),
                size=(route_count,),
            )
            chunks.append(route)
        if not chunks:
            return np.empty((0, 3), dtype=np.float64)
        poses = np.vstack(chunks).astype(np.float64)
        poses[:, 2] = np.asarray([_normalize_angle(value) for value in poses[:, 2]], dtype=np.float64)
        return poses

    def _reset_particles(self) -> None:
        particles = self._sample_seed_poses(self._particle_count, broad=True)
        if particles.shape[0] < self._particle_count:
            padding = np.repeat(self._initial_pose.reshape(1, 3), self._particle_count - particles.shape[0], axis=0)
            particles = np.vstack((particles, padding))
        self._particles_xyyaw = particles[: self._particle_count].astype(np.float64, copy=True)
        self._particles_xyyaw[:, 2] = np.asarray(
            [_normalize_angle(value) for value in self._particles_xyyaw[:, 2]],
            dtype=np.float64,
        )
        self._weights = np.full((self._particle_count,), 1.0 / float(self._particle_count), dtype=np.float64)

    def _inject_random_particles(self, count: int) -> None:
        count = max(0, min(int(count), self._particle_count))
        if count <= 0:
            return
        replace_indexes = self._rng.choice(self._particle_count, size=count, replace=False)
        self._particles_xyyaw[replace_indexes] = self._sample_seed_poses(count, broad=True)
        self._weights[replace_indexes] = 1.0 / float(self._particle_count)
        self._weights /= max(1.0e-12, float(np.sum(self._weights)))

    def add_imu_sample(
        self,
        *,
        t_s: float,
        ax_mps2: float,
        ay_mps2: float,
        az_mps2: float,
        gz_rps: float,
    ) -> None:
        del az_mps2
        self._raw_imu_sample_count += 1
        if self._static_start_s is None:
            self._static_start_s = t_s
        if not self._imu_initialized:
            self._init_samples.append((t_s, ax_mps2, ay_mps2, gz_rps, 0.0))
            self._maybe_finalize_static_initialization(t_s)
            self._last_imu_t_s = t_s
            return

        if self._last_imu_t_s is None:
            self._last_imu_t_s = t_s
            return
        dt_s = max(0.0, min(0.05, float(t_s - self._last_imu_t_s)))
        self._last_imu_t_s = t_s
        raw_yaw_rate = float(gz_rps - self._gyro_bias_rps)
        filtered_yaw_rate = _robust_clip_scalar(
            raw_yaw_rate,
            self._recent_gyro_z_rps,
            float(self._params.imu_spike_mad_scale),
            float(self._params.max_yaw_rate_abs_rps),
        )
        raw_accel_x = float(ax_mps2 - self._accel_bias_xy[0])
        raw_accel_y = float(ay_mps2 - self._accel_bias_xy[1])
        self._recent_gyro_z_rps.append(filtered_yaw_rate)
        self._recent_accel_x_mps2.append(
            _robust_clip_scalar(
                raw_accel_x,
                self._recent_accel_x_mps2,
                float(self._params.imu_spike_mad_scale),
                float(self._params.max_accel_axis_mps2),
            )
        )
        self._recent_accel_y_mps2.append(
            _robust_clip_scalar(
                raw_accel_y,
                self._recent_accel_y_mps2,
                float(self._params.imu_spike_mad_scale),
                float(self._params.max_accel_axis_mps2),
            )
        )
        self._latest_yaw_rate_rps = filtered_yaw_rate
        self._integrated_yaw_rad = _normalize_angle(self._integrated_yaw_rad + (filtered_yaw_rate * dt_s))
        if dt_s > 0.0:
            # Keep translation prediction dominated by previous LiDAR-localized
            # velocity; raw accelerometer XY is kept for diagnostics/filtering
            # but deliberately not integrated as the main odometry source.
            decay = math.exp(-dt_s / max(0.1, self._params.velocity_decay_tau_s))
            self._velocity_world_xy *= decay
        self._processed_imu_sample_count += 1

    def _predict_particle_cloud(self, t_s: float) -> np.ndarray:
        predicted_pose = self._predict_pose(t_s)
        if self._last_estimate is None:
            self._particles_xyyaw[:, :2] += self._rng.normal(
                0.0,
                float(self._params.particle_process_xy_std_m),
                size=(self._particle_count, 2),
            )
            self._particles_xyyaw[:, 2] += self._rng.normal(
                0.0,
                float(self._params.particle_process_yaw_std_rad),
                size=(self._particle_count,),
            )
        else:
            last_pose = np.asarray(
                [self._last_estimate.x_m, self._last_estimate.y_m, self._last_estimate.yaw_rad],
                dtype=np.float64,
            )
            dt_s = max(0.0, min(0.35, float(t_s - self._last_estimate.t_s)))
            delta_xy = predicted_pose[:2] - last_pose[:2]
            delta_yaw = _normalize_angle(float(predicted_pose[2] - last_pose[2]))
            if self._motion_hint_is_fresh(t_s):
                hint_delta_xy = self._motion_hint_velocity_world_xy * dt_s
                blend = max(0.0, min(1.0, float(self._params.motion_hint_velocity_blend)))
                delta_xy = ((1.0 - blend) * delta_xy) + (blend * hint_delta_xy)
            xy_noise_std = math.hypot(
                float(self._params.particle_process_xy_std_m),
                float(self._params.particle_process_velocity_std_mps) * dt_s,
            )
            self._particles_xyyaw[:, :2] += delta_xy.reshape(1, 2)
            self._particles_xyyaw[:, :2] += self._rng.normal(0.0, xy_noise_std, size=(self._particle_count, 2))
            self._particles_xyyaw[:, 2] += delta_yaw
            self._particles_xyyaw[:, 2] += self._rng.normal(
                0.0,
                float(self._params.particle_process_yaw_std_rad),
                size=(self._particle_count,),
            )
        self._particles_xyyaw[:, 2] = np.asarray(
            [_normalize_angle(value) for value in self._particles_xyyaw[:, 2]],
            dtype=np.float64,
        )
        return predicted_pose

    def _particle_cloud_spread(self) -> tuple[float, float]:
        mean_xy = np.sum(self._particles_xyyaw[:, :2] * self._weights.reshape(-1, 1), axis=0)
        xy_var = float(np.sum(self._weights * np.sum(np.square(self._particles_xyyaw[:, :2] - mean_xy.reshape(1, 2)), axis=1)))
        yaw_spread = _weighted_yaw_spread(self._particles_xyyaw[:, 2], self._weights)
        return float(math.sqrt(max(0.0, xy_var))), yaw_spread

    def _score_particles(self, points_local: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        match_points = _subsample_evenly(points_local, int(self._params.max_match_points))
        if match_points.shape[0] < int(self._params.particle_min_lidar_points):
            score = np.zeros((self._particle_count,), dtype=np.float64)
            empty = np.zeros((self._particle_count,), dtype=np.float64)
            return score, {
                "inlier_ratio": empty.copy(),
                "in_bounds_ratio": empty.copy(),
                "median_distance_m": np.full((self._particle_count,), float("inf"), dtype=np.float64),
                "point_count": np.full((self._particle_count,), int(match_points.shape[0]), dtype=np.float64),
            }

        scores = np.empty((self._particle_count,), dtype=np.float64)
        inlier_ratios = np.empty_like(scores)
        in_bounds_ratios = np.empty_like(scores)
        median_distances = np.empty_like(scores)
        sigma = max(0.02, float(self._params.particle_likelihood_sigma_m))
        inlier_distance = max(0.02, float(self._params.particle_inlier_distance_m))
        max_distance = max(float(self._params.max_correspondence_m), inlier_distance)
        for index, particle in enumerate(self._particles_xyyaw):
            world_points = _transform_points(match_points, particle)
            distances_m, valid_mask = self._sample_distance_field_with_valid(world_points)
            valid_count = int(np.count_nonzero(valid_mask))
            in_bounds_ratio = float(valid_count / max(1, match_points.shape[0]))
            if valid_count:
                valid_distances = distances_m[valid_mask]
                clipped = np.minimum(valid_distances, max_distance)
                median_distance = float(np.median(valid_distances))
                mean_sq = float(np.mean(np.square(clipped / sigma)))
                inlier_ratio = float(
                    np.count_nonzero(valid_mask & (distances_m <= inlier_distance))
                    / max(1, match_points.shape[0])
                )
            else:
                median_distance = float("inf")
                mean_sq = float(np.square(max_distance / sigma))
                inlier_ratio = 0.0
            out_of_map_ratio = 1.0 - in_bounds_ratio
            scores[index] = (
                (-0.5 * mean_sq)
                + (float(self._params.particle_inlier_log_weight) * inlier_ratio)
                - (float(self._params.particle_out_of_map_penalty) * out_of_map_ratio)
            )
            inlier_ratios[index] = inlier_ratio
            in_bounds_ratios[index] = in_bounds_ratio
            median_distances[index] = median_distance
        return scores, {
            "inlier_ratio": inlier_ratios,
            "in_bounds_ratio": in_bounds_ratios,
            "median_distance_m": median_distances,
            "point_count": np.full((self._particle_count,), int(match_points.shape[0]), dtype=np.float64),
        }

    def _systematic_resample(self) -> np.ndarray:
        positions = (self._rng.random() + np.arange(self._particle_count, dtype=np.float64)) / float(self._particle_count)
        cumulative = np.cumsum(self._weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="left").astype(np.int32)

    def _update_particle_weights(
        self,
        log_likelihoods: np.ndarray,
        metrics: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, bool]:
        best_likelihood_index = int(np.argmax(log_likelihoods))
        best_inlier_ratio = float(metrics["inlier_ratio"][best_likelihood_index])
        best_in_bounds_ratio = float(metrics["in_bounds_ratio"][best_likelihood_index])
        point_count = int(metrics["point_count"][best_likelihood_index]) if metrics["point_count"].size else 0
        if point_count < int(self._params.particle_min_lidar_points):
            observation_weight = 0.0
        elif best_inlier_ratio < 0.08 or best_in_bounds_ratio < 0.35:
            observation_weight = min(0.12, float(self._params.particle_min_observation_weight))
        else:
            observation_weight = min(
                1.0,
                max(
                    float(self._params.particle_min_observation_weight),
                    best_inlier_ratio / max(1.0e-6, float(self._params.particle_high_confidence_inlier_ratio)),
                ),
            )

        if observation_weight > 0.0:
            stabilized = log_likelihoods - float(np.max(log_likelihoods))
            log_prior = np.log(np.maximum(self._weights, 1.0e-12))
            log_posterior = log_prior + (observation_weight * stabilized)
            log_posterior -= float(np.max(log_posterior))
            weights = np.exp(log_posterior)
            weights_sum = float(np.sum(weights))
            if weights_sum > 1.0e-12 and np.all(np.isfinite(weights)):
                self._weights = weights / weights_sum

        neff = 1.0 / max(1.0e-12, float(np.sum(np.square(self._weights))))
        neff_ratio = float(neff / float(self._particle_count))
        if observation_weight > 0.0:
            best_pose = self._particles_xyyaw[int(np.argmax(self._weights))].copy()
        else:
            best_pose = self._particles_xyyaw[best_likelihood_index].copy()
        resampled = False
        if (
            observation_weight >= float(self._params.particle_min_observation_weight)
            and neff_ratio < float(self._params.particle_resample_neff_ratio)
        ):
            indexes = self._systematic_resample()
            self._particles_xyyaw = self._particles_xyyaw[indexes].copy()
            self._weights.fill(1.0 / float(self._particle_count))
            self._particles_xyyaw[:, :2] += self._rng.normal(
                0.0,
                float(self._params.particle_roughening_xy_std_m),
                size=(self._particle_count, 2),
            )
            self._particles_xyyaw[:, 2] += self._rng.normal(
                0.0,
                float(self._params.particle_roughening_yaw_std_rad),
                size=(self._particle_count,),
            )
            self._particles_xyyaw[:, 2] = np.asarray(
                [_normalize_angle(value) for value in self._particles_xyyaw[:, 2]],
                dtype=np.float64,
            )
            resampled = True

        injection_count = int(round(float(self._particle_count) * float(self._params.particle_random_injection_ratio)))
        if observation_weight <= 0.12:
            injection_count = max(injection_count, int(round(0.04 * float(self._particle_count))))
        self._inject_random_particles(injection_count)
        return best_pose, observation_weight, resampled

    def _observation_metrics_for_pose(self, points_local: np.ndarray, pose_xyyaw: np.ndarray) -> dict[str, float]:
        match_points = _subsample_evenly(points_local, int(self._params.max_match_points))
        if match_points.shape[0] == 0:
            return {
                "inlier_ratio": 0.0,
                "in_bounds_ratio": 0.0,
                "median_distance_m": float("nan"),
            }
        distances_m, valid_mask = self._sample_distance_field_with_valid(_transform_points(match_points, pose_xyyaw))
        in_bounds_ratio = float(np.mean(valid_mask.astype(np.float64))) if valid_mask.size else 0.0
        inlier_ratio = float(
            np.mean((valid_mask & (distances_m <= float(self._params.particle_inlier_distance_m))).astype(np.float64))
        )
        if np.any(valid_mask):
            median_distance_m = float(np.median(distances_m[valid_mask]))
        else:
            median_distance_m = float("inf")
        return {
            "inlier_ratio": inlier_ratio,
            "in_bounds_ratio": in_bounds_ratio,
            "median_distance_m": median_distance_m,
        }

    def _refine_best_particle(self, points_local: np.ndarray, best_pose: np.ndarray, observation_weight: float) -> np.ndarray:
        if not bool(self._params.particle_refine_enabled):
            return best_pose
        if (
            observation_weight < float(self._params.particle_min_observation_weight)
            or points_local.shape[0] < int(self._params.particle_min_lidar_points)
        ):
            return best_pose
        refined_pose, median_residual_m, support_metrics = super()._optimize_pose(points_local, best_pose)
        if not np.all(np.isfinite(refined_pose)):
            return best_pose
        if math.isfinite(median_residual_m) and median_residual_m > max(0.35, 2.5 * float(self._params.particle_inlier_distance_m)):
            return best_pose
        gain = max(0.0, min(1.0, float(self._params.particle_refine_gain) * max(0.2, observation_weight)))
        self._last_support_metrics.update(support_metrics)
        return _blend_pose_xyyaw(best_pose, refined_pose, gain)

    def add_scan_observation(self, scan: LidarScanObservation) -> list[ScanEstimate]:
        if not self._imu_initialized:
            return []
        predicted_pose = self._predict_particle_cloud(float(scan.t_s))
        log_likelihoods, metrics = self._score_particles(scan.sampled_points_local)
        best_pose, observation_weight, resampled = self._update_particle_weights(log_likelihoods, metrics)
        candidate_pose = self._refine_best_particle(scan.sampled_points_local, best_pose, observation_weight)
        if observation_weight >= float(self._params.particle_min_observation_weight):
            corrected_pose = candidate_pose
        elif observation_weight > 0.0 and np.all(np.isfinite(candidate_pose)):
            # Weak LiDAR support may hint at a basin, but it should not collapse
            # the filter away from the inertial prediction.
            corrected_pose = _blend_pose_xyyaw(
                predicted_pose,
                candidate_pose,
                min(0.30, 0.75 * float(observation_weight)),
            )
        else:
            corrected_pose = predicted_pose.copy()
        observation_metrics = self._observation_metrics_for_pose(scan.sampled_points_local, corrected_pose)

        if (
            observation_weight >= float(self._params.particle_min_observation_weight)
            and np.all(np.isfinite(corrected_pose))
        ):
            replacement_count = max(2, int(round(0.04 * float(self._particle_count))))
            replacement_indexes = np.argsort(self._weights)[:replacement_count]
            self._particles_xyyaw[replacement_indexes] = corrected_pose.reshape(1, 3)
            self._particles_xyyaw[replacement_indexes, :2] += self._rng.normal(
                0.0,
                max(0.01, float(self._params.particle_roughening_xy_std_m)),
                size=(replacement_count, 2),
            )
            self._particles_xyyaw[replacement_indexes, 2] += self._rng.normal(
                0.0,
                max(0.01, float(self._params.particle_roughening_yaw_std_rad)),
                size=(replacement_count,),
            )
            self._particles_xyyaw[replacement_indexes, 2] = np.asarray(
                [_normalize_angle(value) for value in self._particles_xyyaw[replacement_indexes, 2]],
                dtype=np.float64,
            )
            self._weights[replacement_indexes] = max(float(np.max(self._weights)), 1.0 / float(self._particle_count))
            self._weights /= max(1.0e-12, float(np.sum(self._weights)))

        if self._last_estimate is not None:
            dt_s = max(1.0e-3, float(scan.t_s - self._last_estimate.t_s))
            measured_velocity = (corrected_pose[:2] - np.asarray([self._last_estimate.x_m, self._last_estimate.y_m])) / dt_s
            measured_speed = float(np.linalg.norm(measured_velocity))
            if measured_speed <= 2.0:
                self._velocity_world_xy = (0.78 * self._velocity_world_xy) + (0.22 * measured_velocity)
            if self._motion_hint_is_fresh(float(scan.t_s)):
                blend = 0.35 * max(0.0, min(1.0, float(self._params.motion_hint_velocity_blend)))
                self._velocity_world_xy = ((1.0 - blend) * self._velocity_world_xy) + (
                    blend * self._motion_hint_velocity_world_xy
                )
            self._yaw_offset_rad = _normalize_angle(float(corrected_pose[2] - self._integrated_yaw_rad))
        else:
            self._yaw_offset_rad = _normalize_angle(float(corrected_pose[2] - self._integrated_yaw_rad))

        neff_ratio = float((1.0 / max(1.0e-12, float(np.sum(np.square(self._weights))))) / float(self._particle_count))
        xy_spread_m, yaw_spread_rad = self._particle_cloud_spread()
        inlier_ratio = float(observation_metrics["inlier_ratio"])
        in_bounds_ratio = float(observation_metrics["in_bounds_ratio"])
        median_residual_m = float(observation_metrics["median_distance_m"])
        out_of_map_ratio = max(0.0, 1.0 - in_bounds_ratio)
        if math.isfinite(median_residual_m):
            distance_score = 1.0 - min(1.0, median_residual_m / max(0.05, 1.55 * float(self._params.particle_inlier_distance_m)))
        else:
            distance_score = 0.0
        inlier_score = min(1.0, inlier_ratio / max(1.0e-6, float(self._params.particle_high_confidence_inlier_ratio)))
        in_bounds_score = min(1.0, in_bounds_ratio / 0.75)
        spread_score = math.exp(
            -math.pow(
                xy_spread_m / max(0.05, float(self._params.particle_max_medium_confidence_spread_m)),
                2.0,
            )
        )
        observation_score = min(1.0, observation_weight / max(1.0e-6, float(self._params.particle_min_observation_weight)))
        confidence_score = (
            (0.34 * inlier_score)
            + (0.24 * in_bounds_score)
            + (0.24 * distance_score)
            + (0.12 * spread_score)
            + (0.06 * observation_score)
        )
        if observation_weight < float(self._params.particle_min_observation_weight):
            confidence_score *= 0.45
        if (
            math.isfinite(median_residual_m)
            and median_residual_m <= 0.11
            and inlier_ratio >= float(self._params.particle_high_confidence_inlier_ratio)
            and in_bounds_ratio >= 0.72
            and xy_spread_m <= float(self._params.particle_max_high_confidence_spread_m)
            and observation_weight >= float(self._params.particle_min_observation_weight)
        ):
            confidence = "high"
        elif (
            math.isfinite(median_residual_m)
            and median_residual_m <= 0.20
            and inlier_ratio >= float(self._params.particle_medium_confidence_inlier_ratio)
            and in_bounds_ratio >= 0.45
            and xy_spread_m <= float(self._params.particle_max_medium_confidence_spread_m)
            and confidence_score >= 0.42
        ):
            confidence = "medium"
        else:
            confidence = "low"

        self._last_support_metrics = {
            "in_bounds_ratio": in_bounds_ratio,
            "near_wall_ratio": inlier_ratio,
            "median_distance_m": median_residual_m,
            "dynamic_prior_gain": 1.0,
        }
        self._last_particle_metrics = {
            "enabled": True,
            "particle_count": int(self._particle_count),
            "neff_ratio": neff_ratio,
            "xy_spread_m": xy_spread_m,
            "yaw_spread_rad": yaw_spread_rad,
            "best_inlier_ratio": inlier_ratio,
            "best_in_bounds_ratio": in_bounds_ratio,
            "best_out_of_map_ratio": out_of_map_ratio,
            "best_median_distance_m": median_residual_m,
            "observation_weight": float(observation_weight),
            "confidence_score": float(confidence_score),
            "motion_hint_fresh": bool(self._motion_hint_is_fresh(float(scan.t_s))),
            "resampled": bool(resampled),
            "route_seed_count": int(self._route_poses_xyyaw.shape[0]),
        }

        estimate = ScanEstimate(
            scan_index=int(scan.scan_index),
            stamp_sec=int(scan.stamp_sec),
            stamp_nanosec=int(scan.stamp_nanosec),
            t_s=float(scan.t_s),
            x_m=float(corrected_pose[0]),
            y_m=float(corrected_pose[1]),
            yaw_rad=float(corrected_pose[2]),
            vx_mps=float(self._velocity_world_xy[0]),
            vy_mps=float(self._velocity_world_xy[1]),
            yaw_rate_rps=float(self._latest_yaw_rate_rps),
            ax_world_mps2=0.0,
            ay_world_mps2=0.0,
            confidence=confidence,
            median_submap_residual_m=float(median_residual_m),
            median_wall_residual_m=float("nan"),
            valid_correspondence_count=int(scan.sampled_points_local.shape[0]),
            alignment_ready=True,
            imu_initialized=True,
            best_effort_init=bool(self._best_effort_init),
        )
        self._last_estimate = estimate
        self._path_estimates.append(estimate)
        self._processed_scan_count += 1
        if math.isfinite(median_residual_m):
            self._median_residual_history.append(float(median_residual_m))
        self._state = "tracking"
        return [estimate]

    def status_snapshot(self) -> FusionStateSnapshot:
        snapshot = super().status_snapshot()
        quality = dict(snapshot.quality)
        quality.update({f"particle_{key}": value for key, value in self._last_particle_metrics.items()})
        parameters = dict(snapshot.parameters)
        parameters["localizer"] = "particle_filter"
        return replace(snapshot, quality=quality, parameters=parameters)


# Preserve the historical import name used by imu_lidar_planar_fusion_node.
FixedMapPlanarLocalizer = FixedMapParticleLocalizer
