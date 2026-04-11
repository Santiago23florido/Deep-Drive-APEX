#!/usr/bin/env python3
"""Fixed-map LiDAR + IMU localization for APEX."""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
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
