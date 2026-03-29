#!/usr/bin/env python3
"""Offline LiDAR + IMU fusion for corridor trajectory estimation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


GRAVITY_MPS2 = 9.80665

DEFAULT_MEDIAN_WINDOW = 5
DEFAULT_EMA_ALPHA = 0.25
DEFAULT_STATIC_WINDOW_S = 0.4
DEFAULT_STATIC_SEARCH_S = 2.0
DEFAULT_VELOCITY_DECAY_TAU_S = 1.1
DEFAULT_SUBMAP_WINDOW_SCANS = 6
DEFAULT_SCAN_POINT_STRIDE = 2
DEFAULT_GLOBAL_POINT_STRIDE = 3
DEFAULT_MAX_CORRESPONDENCE_M = 0.35
DEFAULT_INITIAL_SCAN_COUNT_MIN = 4
DEFAULT_CORRIDOR_BIN_M = 0.10
DEFAULT_CORRIDOR_QUANTILE = 0.18
DEFAULT_LOW_CONFIDENCE_RESIDUAL_M = 0.16

LOCAL_SUBMAP_WEIGHT = 4.0
LOCAL_WALL_WEIGHT = 2.2
LOCAL_YAW_WEIGHT = 8.5
LOCAL_MOTION_WEIGHT = 0.85

GLOBAL_SUBMAP_WEIGHT = 3.4
GLOBAL_WALL_WEIGHT = 2.0
GLOBAL_YAW_WEIGHT = 6.5
GLOBAL_VELOCITY_WEIGHT = 0.55
GLOBAL_SMOOTHNESS_WEIGHT = 0.70
GLOBAL_STATIC_ANCHOR_WEIGHT = 3.5
GLOBAL_STATIC_YAW_WEIGHT = 4.5


@dataclass(frozen=True)
class ImuSeries:
    t_s: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    az_mps2: np.ndarray
    gz_rps: np.ndarray


@dataclass(frozen=True)
class ImuProcessingResult:
    t_s: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    az_mps2: np.ndarray
    gz_rps: np.ndarray
    yaw_rad: np.ndarray
    vx_mps: np.ndarray
    vy_mps: np.ndarray
    ax_world_mps2: np.ndarray
    ay_world_mps2: np.ndarray
    bias_ax_mps2: float
    bias_ay_mps2: float
    bias_az_mps2: float
    bias_gz_rps: float
    static_start_s: float
    static_end_s: float
    static_sample_count: int
    best_effort_init: bool


@dataclass(frozen=True)
class LidarScan:
    scan_index: int
    t_s: float
    points_local: np.ndarray
    sampled_points_local: np.ndarray
    lower_wall_points_local: np.ndarray
    upper_wall_points_local: np.ndarray


@dataclass(frozen=True)
class WallModel:
    lower_coef: np.ndarray
    upper_coef: np.ndarray
    width_m: float
    corridor_yaw_rad: float


@dataclass(frozen=True)
class ScanCorrespondenceBundle:
    scan_index: int
    local_points: np.ndarray
    target_points: np.ndarray
    raw_distances_m: np.ndarray


@dataclass(frozen=True)
class ScanQuality:
    scan_index: int
    t_s: float
    median_submap_residual_m: float
    median_wall_residual_m: float
    valid_correspondence_count: int
    confidence: str


def _validate_filter_params(median_window: int, ema_alpha: float) -> None:
    if median_window < 1 or (median_window % 2) == 0:
        raise SystemExit("La ventana de mediana debe ser impar y mayor o igual a 1.")
    if not 0.0 < ema_alpha <= 1.0:
        raise SystemExit("El parametro --ema-alpha debe estar en el rango (0, 1].")


def _apply_median_filter(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size == 1:
        return values.copy()
    radius = window_size // 2
    filtered = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        filtered[index] = median(values[start:end].tolist())
    return filtered


def _apply_exponential_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, values.size):
        filtered[index] = (alpha * values[index]) + ((1.0 - alpha) * filtered[index - 1])
    return filtered


def _apply_zero_phase_low_pass(values: np.ndarray, alpha: float) -> np.ndarray:
    forward = _apply_exponential_filter(values, alpha)
    backward = _apply_exponential_filter(forward[::-1], alpha)
    return backward[::-1]


def _filter_signal(values: np.ndarray, median_window: int, ema_alpha: float) -> np.ndarray:
    return _apply_zero_phase_low_pass(
        _apply_median_filter(values, median_window),
        ema_alpha,
    )


def _load_imu_series(path: Path) -> ImuSeries:
    rows: list[tuple[float, float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                (
                    float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"])),
                    float(row["ax_mps2"]),
                    float(row["ay_mps2"]),
                    float(row["az_mps2"]),
                    float(row["gz_rps"]),
                )
            )
    if not rows:
        raise SystemExit(f"No se encontraron muestras IMU en {path}")
    data = np.asarray(rows, dtype=np.float64)
    return ImuSeries(
        t_s=data[:, 0],
        ax_mps2=data[:, 1],
        ay_mps2=data[:, 2],
        az_mps2=data[:, 3],
        gz_rps=data[:, 4],
    )


def _subsample_evenly(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.size == 0 or points.shape[0] <= max_points:
        return points.copy()
    indexes = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int32)
    return points[indexes]


def _extract_sidewall_candidates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 12:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    x_cut = float(np.quantile(points[:, 0], 0.85))
    side_points = points[points[:, 0] <= x_cut]
    if side_points.shape[0] < 12:
        side_points = points

    lower_q = float(np.quantile(side_points[:, 1], DEFAULT_CORRIDOR_QUANTILE))
    upper_q = float(np.quantile(side_points[:, 1], 1.0 - DEFAULT_CORRIDOR_QUANTILE))

    lower = side_points[side_points[:, 1] <= lower_q]
    upper = side_points[side_points[:, 1] >= upper_q]
    return _subsample_evenly(lower, 12), _subsample_evenly(upper, 12)


def _load_lidar_scans(path: Path, point_stride: int) -> list[LidarScan]:
    grouped_points: dict[int, list[tuple[float, float]]] = {}
    scan_times: dict[int, float] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_m = float(row["x_forward_m"])
            y_m = float(row["y_left_m"])
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                continue
            radial_m = math.hypot(x_m, y_m)
            if radial_m < 0.08 or radial_m > 12.0:
                continue
            scan_index = int(row["scan_index"])
            grouped_points.setdefault(scan_index, []).append((x_m, y_m))
            scan_times.setdefault(
                scan_index,
                float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"])),
            )

    scans: list[LidarScan] = []
    for scan_index in sorted(grouped_points):
        points_local = np.asarray(grouped_points[scan_index], dtype=np.float64)
        sampled = points_local[::point_stride].copy()
        lower_wall_points_local, upper_wall_points_local = _extract_sidewall_candidates(points_local)
        scans.append(
            LidarScan(
                scan_index=scan_index,
                t_s=float(scan_times[scan_index]),
                points_local=points_local,
                sampled_points_local=sampled,
                lower_wall_points_local=lower_wall_points_local,
                upper_wall_points_local=upper_wall_points_local,
            )
        )

    if not scans:
        raise SystemExit(f"No se encontraron scans LiDAR validos en {path}")
    return scans


def _window_end_index(t_s: np.ndarray, start_index: int, window_s: float) -> int:
    stop_time_s = float(t_s[start_index] + window_s)
    return int(np.searchsorted(t_s, stop_time_s, side="right"))


def _detect_static_window(
    imu: ImuSeries,
    *,
    window_s: float,
    search_s: float,
) -> tuple[int, int, bool]:
    t_s = imu.t_s
    search_end_s = float(t_s[0] + search_s)
    fallback_end_index = _window_end_index(t_s, 0, window_s)
    fallback = (0, max(fallback_end_index, 2), True)

    for start_index, sample_time_s in enumerate(t_s):
        if sample_time_s > (search_end_s - window_s):
            break
        end_index = _window_end_index(t_s, start_index, window_s)
        if (end_index - start_index) < 8:
            continue

        ax_std = float(np.std(imu.ax_mps2[start_index:end_index]))
        ay_std = float(np.std(imu.ay_mps2[start_index:end_index]))
        az_std = float(np.std(imu.az_mps2[start_index:end_index]))
        gz_std = float(np.std(imu.gz_rps[start_index:end_index]))

        if ax_std <= 0.12 and ay_std <= 0.12 and az_std <= 0.18 and gz_std <= 0.0045:
            return start_index, end_index, False

    return fallback


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    rotation = _rotation_matrix(float(pose[2]))
    return (points @ rotation.T) + pose[:2]


def _rotate_points(points: np.ndarray, theta_rad: float) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    rotation = _rotation_matrix(theta_rad)
    return points @ rotation.T


def _fit_line_robust(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    residual_threshold_m: float = 0.10,
    max_iterations: int = 4,
) -> np.ndarray:
    if x_values.size < 2:
        raise ValueError("No hay suficientes puntos para ajustar una recta.")

    mask = np.ones_like(x_values, dtype=bool)
    coefficients = np.polyfit(x_values, y_values, deg=1)
    for _ in range(max_iterations):
        if int(np.count_nonzero(mask)) < 2:
            break
        coefficients = np.polyfit(x_values[mask], y_values[mask], deg=1)
        residuals = y_values - np.polyval(coefficients, x_values)
        median_abs = float(np.median(np.abs(residuals)))
        limit = max(residual_threshold_m, 2.5 * median_abs)
        new_mask = np.abs(residuals) <= limit
        if int(np.count_nonzero(new_mask)) == int(np.count_nonzero(mask)):
            break
        mask = new_mask
    return coefficients


def _fit_corridor_quantile_lines(points: np.ndarray, bin_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] < 40:
        raise ValueError("No hay suficientes puntos LiDAR para ajustar el corredor.")

    x_min_m = float(np.min(points[:, 0]))
    x_max_m = float(np.max(points[:, 0])) + bin_m

    x_centers: list[float] = []
    lower_ys: list[float] = []
    upper_ys: list[float] = []

    for x0_m in np.arange(x_min_m, x_max_m, bin_m):
        mask = (points[:, 0] >= x0_m) & (points[:, 0] < (x0_m + bin_m))
        if int(np.count_nonzero(mask)) < 8:
            continue
        ys = points[mask, 1]
        x_centers.append(float(x0_m + (0.5 * bin_m)))
        lower_ys.append(float(np.quantile(ys, DEFAULT_CORRIDOR_QUANTILE)))
        upper_ys.append(float(np.quantile(ys, 1.0 - DEFAULT_CORRIDOR_QUANTILE)))

    if len(x_centers) < 4:
        raise ValueError("No hay suficientes bins para ajustar las paredes del corredor.")

    x_values = np.asarray(x_centers, dtype=np.float64)
    lower_values = np.asarray(lower_ys, dtype=np.float64)
    upper_values = np.asarray(upper_ys, dtype=np.float64)
    lower_coef = _fit_line_robust(x_values, lower_values)
    upper_coef = _fit_line_robust(x_values, upper_values)
    width_m = float(np.median(upper_values - lower_values))
    return lower_coef, upper_coef, width_m


def _fit_wall_model(points: np.ndarray, bin_m: float) -> WallModel:
    lower_coef, upper_coef, width_m = _fit_corridor_quantile_lines(points, bin_m)
    corridor_yaw_rad = math.atan(0.5 * (float(lower_coef[0]) + float(upper_coef[0])))
    return WallModel(
        lower_coef=lower_coef,
        upper_coef=upper_coef,
        width_m=width_m,
        corridor_yaw_rad=corridor_yaw_rad,
    )


def _estimate_initial_alignment(points: np.ndarray, bin_m: float) -> tuple[float, WallModel]:
    raw_model = _fit_wall_model(points, bin_m)
    alignment_yaw_rad = -float(raw_model.corridor_yaw_rad)
    rotated_points = _rotate_points(points, alignment_yaw_rad)
    aligned_model = _fit_wall_model(rotated_points, bin_m)
    return alignment_yaw_rad, aligned_model


def _centerline_y_m(wall_model: WallModel, x_m: float) -> float:
    return 0.5 * (
        float(np.polyval(wall_model.lower_coef, x_m)) + float(np.polyval(wall_model.upper_coef, x_m))
    )


def _process_imu(
    imu: ImuSeries,
    *,
    median_window: int,
    ema_alpha: float,
    static_window_s: float,
    static_search_s: float,
    velocity_decay_tau_s: float,
    world_yaw_offset_rad: float,
) -> ImuProcessingResult:
    static_start_index, static_end_index, best_effort_init = _detect_static_window(
        imu,
        window_s=static_window_s,
        search_s=static_search_s,
    )

    bias_ax = float(np.mean(imu.ax_mps2[static_start_index:static_end_index]))
    bias_ay = float(np.mean(imu.ay_mps2[static_start_index:static_end_index]))
    bias_az = float(np.mean(imu.az_mps2[static_start_index:static_end_index]))
    bias_gz = float(np.mean(imu.gz_rps[static_start_index:static_end_index]))

    ax_corrected = imu.ax_mps2 - bias_ax
    ay_corrected = imu.ay_mps2 - bias_ay
    az_corrected = imu.az_mps2 - bias_az
    gz_corrected = imu.gz_rps - bias_gz

    ax_filtered = _filter_signal(ax_corrected, median_window, ema_alpha)
    ay_filtered = _filter_signal(ay_corrected, median_window, ema_alpha)
    az_filtered = _filter_signal(az_corrected, median_window, ema_alpha)
    gz_filtered = _filter_signal(gz_corrected, median_window, ema_alpha)

    yaw_rad = np.zeros_like(imu.t_s)
    for index in range(1, imu.t_s.size):
        dt_s = float(imu.t_s[index] - imu.t_s[index - 1])
        yaw_rad[index] = yaw_rad[index - 1] + (
            0.5 * dt_s * (gz_filtered[index - 1] + gz_filtered[index])
        )
    yaw_rad = yaw_rad + world_yaw_offset_rad

    ax_world = np.zeros_like(ax_filtered)
    ay_world = np.zeros_like(ay_filtered)
    for index in range(ax_filtered.size):
        rotation = _rotation_matrix(float(yaw_rad[index]))
        world_accel = rotation @ np.asarray([ax_filtered[index], ay_filtered[index]], dtype=np.float64)
        ax_world[index] = world_accel[0]
        ay_world[index] = world_accel[1]

    vx_world = np.zeros_like(ax_world)
    vy_world = np.zeros_like(ay_world)
    static_end_s = float(imu.t_s[static_end_index - 1])
    for index in range(1, imu.t_s.size):
        dt_s = float(imu.t_s[index] - imu.t_s[index - 1])
        decay = math.exp(-dt_s / velocity_decay_tau_s)
        if imu.t_s[index] <= static_end_s:
            vx_world[index] = 0.0
            vy_world[index] = 0.0
            continue
        vx_world[index] = (
            decay * vx_world[index - 1]
            + (0.5 * dt_s * (ax_world[index - 1] + ax_world[index]))
        )
        vy_world[index] = (
            decay * vy_world[index - 1]
            + (0.5 * dt_s * (ay_world[index - 1] + ay_world[index]))
        )

    return ImuProcessingResult(
        t_s=imu.t_s.copy(),
        ax_mps2=ax_filtered,
        ay_mps2=ay_filtered,
        az_mps2=az_filtered,
        gz_rps=gz_filtered,
        yaw_rad=yaw_rad,
        vx_mps=vx_world,
        vy_mps=vy_world,
        ax_world_mps2=ax_world,
        ay_world_mps2=ay_world,
        bias_ax_mps2=bias_ax,
        bias_ay_mps2=bias_ay,
        bias_az_mps2=bias_az - GRAVITY_MPS2,
        bias_gz_rps=bias_gz,
        static_start_s=float(imu.t_s[static_start_index]),
        static_end_s=static_end_s,
        static_sample_count=int(static_end_index - static_start_index),
        best_effort_init=best_effort_init,
    )


def _interpolate_scan_priors(
    imu: ImuProcessingResult,
    scan_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw_prior = np.interp(scan_times_s, imu.t_s, imu.yaw_rad)
    vx_prior = np.interp(scan_times_s, imu.t_s, imu.vx_mps)
    vy_prior = np.interp(scan_times_s, imu.t_s, imu.vy_mps)
    ax_prior = np.interp(scan_times_s, imu.t_s, imu.ax_world_mps2)
    ay_prior = np.interp(scan_times_s, imu.t_s, imu.ay_world_mps2)
    return yaw_prior, np.column_stack((vx_prior, vy_prior)), np.column_stack((ax_prior, ay_prior))


def _estimate_initial_scan_count(scan_times_s: np.ndarray, static_end_s: float) -> int:
    count = int(np.count_nonzero(scan_times_s <= (static_end_s + 0.05)))
    return max(DEFAULT_INITIAL_SCAN_COUNT_MIN, count)


def _build_submap_points(
    scans: list[LidarScan],
    poses: np.ndarray,
    confidences: list[str],
    current_index: int,
    window_scans: int,
) -> np.ndarray:
    start_index = max(0, current_index - window_scans)
    submap_parts: list[np.ndarray] = []
    for scan_index in range(start_index, current_index):
        if confidences[scan_index] == "low" and scan_index >= DEFAULT_INITIAL_SCAN_COUNT_MIN:
            continue
        submap_parts.append(_transform_points(scans[scan_index].sampled_points_local, poses[scan_index]))
    if not submap_parts:
        for scan_index in range(start_index, current_index):
            submap_parts.append(_transform_points(scans[scan_index].sampled_points_local, poses[scan_index]))
    if not submap_parts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(submap_parts)


def _predict_pose(
    scan_index: int,
    poses: np.ndarray,
    scan_times_s: np.ndarray,
    velocity_priors_mps: np.ndarray,
    accel_priors_mps2: np.ndarray,
    yaw_priors_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    prev_pose = poses[scan_index - 1].copy()
    dt_s = max(1.0e-3, float(scan_times_s[scan_index] - scan_times_s[scan_index - 1]))
    if scan_index > 1:
        prev_dt_s = max(1.0e-3, float(scan_times_s[scan_index - 1] - scan_times_s[scan_index - 2]))
        prev_velocity = (poses[scan_index - 1, :2] - poses[scan_index - 2, :2]) / prev_dt_s
    else:
        prev_velocity = velocity_priors_mps[scan_index - 1].copy()
    prev_velocity = np.clip(prev_velocity, -3.0, 3.0)
    delta_prior = (prev_velocity * dt_s) + (0.5 * accel_priors_mps2[scan_index] * dt_s * dt_s)
    prediction = np.asarray(
        [
            prev_pose[0] + delta_prior[0],
            prev_pose[1] + delta_prior[1],
            yaw_priors_rad[scan_index],
        ],
        dtype=np.float64,
    )
    return prediction, delta_prior, dt_s


def _wall_residuals_for_pose(scan: LidarScan, pose: np.ndarray, wall_model: WallModel) -> np.ndarray:
    residuals: list[np.ndarray] = []
    if scan.lower_wall_points_local.size:
        lower_world = _transform_points(scan.lower_wall_points_local, pose)
        lower_fit = np.polyval(wall_model.lower_coef, lower_world[:, 0])
        residuals.append(lower_world[:, 1] - lower_fit)
    if scan.upper_wall_points_local.size:
        upper_world = _transform_points(scan.upper_wall_points_local, pose)
        upper_fit = np.polyval(wall_model.upper_coef, upper_world[:, 0])
        residuals.append(upper_world[:, 1] - upper_fit)
    if not residuals:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(residuals)


def _evaluate_pose_quality(
    scan: LidarScan,
    pose: np.ndarray,
    submap_points: np.ndarray,
    wall_model: WallModel,
    max_correspondence_m: float,
) -> tuple[float, float, int]:
    if submap_points.size == 0 or scan.sampled_points_local.size == 0:
        submap_median = float("nan")
        valid_count = 0
    else:
        tree = cKDTree(submap_points)
        world_points = _transform_points(scan.sampled_points_local, pose)
        distances, _ = tree.query(world_points, k=1)
        valid_mask = distances <= max_correspondence_m
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count:
            submap_median = float(np.median(distances[valid_mask]))
        else:
            submap_median = float("nan")

    wall_residuals = _wall_residuals_for_pose(scan, pose, wall_model)
    wall_median = float(np.median(np.abs(wall_residuals))) if wall_residuals.size else float("nan")
    return submap_median, wall_median, valid_count


def _estimate_sequential_poses(
    scans: list[LidarScan],
    yaw_priors_rad: np.ndarray,
    velocity_priors_mps: np.ndarray,
    accel_priors_mps2: np.ndarray,
    wall_model: WallModel,
    *,
    initial_yaw_rad: float,
    initial_scan_count: int,
    submap_window_scans: int,
    max_correspondence_m: float,
) -> tuple[np.ndarray, list[ScanQuality]]:
    scan_times_s = np.asarray([scan.t_s for scan in scans], dtype=np.float64)
    poses = np.zeros((len(scans), 3), dtype=np.float64)
    poses[:, 2] = yaw_priors_rad
    poses[:initial_scan_count, 2] = initial_yaw_rad
    confidences = ["high"] * len(scans)
    qualities: list[ScanQuality] = []

    for scan_index in range(initial_scan_count):
        poses[scan_index, 0] = 0.0
        poses[scan_index, 1] = 0.0
        submap_median, wall_median, valid_count = _evaluate_pose_quality(
            scans[scan_index],
            poses[scan_index],
            _transform_points(scans[0].sampled_points_local, poses[0]),
            wall_model,
            max_correspondence_m,
        )
        qualities.append(
            ScanQuality(
                scan_index=scans[scan_index].scan_index,
                t_s=scans[scan_index].t_s,
                median_submap_residual_m=submap_median,
                median_wall_residual_m=wall_median,
                valid_correspondence_count=valid_count,
                confidence="high",
            )
        )

    for scan_index in range(initial_scan_count, len(scans)):
        submap_points = _build_submap_points(
            scans,
            poses,
            confidences,
            scan_index,
            submap_window_scans,
        )
        prediction, delta_prior, _dt_s = _predict_pose(
            scan_index,
            poses,
            scan_times_s,
            velocity_priors_mps,
            accel_priors_mps2,
            yaw_priors_rad,
        )
        if submap_points.shape[0] < 10:
            poses[scan_index] = prediction
            confidences[scan_index] = "low"
            qualities.append(
                ScanQuality(
                    scan_index=scans[scan_index].scan_index,
                    t_s=scans[scan_index].t_s,
                    median_submap_residual_m=float("nan"),
                    median_wall_residual_m=float("nan"),
                    valid_correspondence_count=0,
                    confidence="low",
                )
            )
            continue

        tree = cKDTree(submap_points)
        scan = scans[scan_index]

        def residual_vector(pose_flat: np.ndarray) -> np.ndarray:
            pose = pose_flat.astype(np.float64, copy=False)
            world_points = _transform_points(scan.sampled_points_local, pose)
            distances, nearest_indexes = tree.query(world_points, k=1)

            residuals: list[np.ndarray] = []
            diffs = world_points - submap_points[nearest_indexes]
            scales = np.minimum(
                1.0,
                max_correspondence_m / np.maximum(distances, 1.0e-6),
            ).reshape(-1, 1)
            residuals.append((diffs * scales * LOCAL_SUBMAP_WEIGHT).reshape(-1))

            wall_residuals = _wall_residuals_for_pose(scan, pose, wall_model)
            if wall_residuals.size:
                residuals.append(wall_residuals * LOCAL_WALL_WEIGHT)

            motion_residual = (pose[:2] - poses[scan_index - 1, :2]) - delta_prior
            residuals.append(motion_residual * LOCAL_MOTION_WEIGHT)
            residuals.append(
                np.asarray([(pose[2] - yaw_priors_rad[scan_index]) * LOCAL_YAW_WEIGHT], dtype=np.float64)
            )
            return np.concatenate(residuals)

        lower_bounds = np.asarray(
            [
                prediction[0] - 1.5,
                prediction[1] - 1.0,
                yaw_priors_rad[scan_index] - 0.65,
            ],
            dtype=np.float64,
        )
        upper_bounds = np.asarray(
            [
                prediction[0] + 1.5,
                prediction[1] + 1.0,
                yaw_priors_rad[scan_index] + 0.65,
            ],
            dtype=np.float64,
        )

        solution = least_squares(
            residual_vector,
            x0=prediction,
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=80,
        )

        candidate_pose = solution.x if solution.success else prediction
        submap_median, wall_median, valid_count = _evaluate_pose_quality(
            scan,
            candidate_pose,
            submap_points,
            wall_model,
            max_correspondence_m,
        )

        confidence = "high"
        if (
            (not solution.success)
            or valid_count < 14
            or (math.isfinite(submap_median) and submap_median > DEFAULT_LOW_CONFIDENCE_RESIDUAL_M)
        ):
            confidence = "low"
            candidate_pose = prediction
            submap_median, wall_median, valid_count = _evaluate_pose_quality(
                scan,
                candidate_pose,
                submap_points,
                wall_model,
                max_correspondence_m,
            )

        poses[scan_index] = candidate_pose
        confidences[scan_index] = confidence
        qualities.append(
            ScanQuality(
                scan_index=scan.scan_index,
                t_s=scan.t_s,
                median_submap_residual_m=submap_median,
                median_wall_residual_m=wall_median,
                valid_correspondence_count=valid_count,
                confidence=confidence,
            )
        )

    return poses, qualities


def _collect_world_points(
    scans: list[LidarScan],
    poses: np.ndarray,
    *,
    point_stride: int,
    confidence_filter: list[str] | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for index, scan in enumerate(scans):
        if confidence_filter is not None and confidence_filter[index] == "low":
            continue
        points = scan.points_local[::point_stride]
        parts.append(_transform_points(points, poses[index]))
    if not parts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(parts)


def _build_correspondence_bundles(
    scans: list[LidarScan],
    poses: np.ndarray,
    *,
    initial_scan_count: int,
    submap_window_scans: int,
    point_stride: int,
    max_correspondence_m: float,
) -> list[ScanCorrespondenceBundle]:
    bundles: list[ScanCorrespondenceBundle] = []
    for scan_index, scan in enumerate(scans):
        if scan_index == 0:
            bundles.append(
                ScanCorrespondenceBundle(
                    scan_index=scan.scan_index,
                    local_points=np.empty((0, 2), dtype=np.float64),
                    target_points=np.empty((0, 2), dtype=np.float64),
                    raw_distances_m=np.empty((0,), dtype=np.float64),
                )
            )
            continue

        start_index = max(0, scan_index - submap_window_scans)
        submap_parts: list[np.ndarray] = []
        for source_index in range(start_index, scan_index):
            sampled_local = scans[source_index].points_local[::point_stride]
            submap_parts.append(_transform_points(sampled_local, poses[source_index]))
        if not submap_parts:
            bundles.append(
                ScanCorrespondenceBundle(
                    scan_index=scan.scan_index,
                    local_points=np.empty((0, 2), dtype=np.float64),
                    target_points=np.empty((0, 2), dtype=np.float64),
                    raw_distances_m=np.empty((0,), dtype=np.float64),
                )
            )
            continue

        submap_points = np.vstack(submap_parts)
        tree = cKDTree(submap_points)
        local_points = scan.points_local[::point_stride]
        world_points = _transform_points(local_points, poses[scan_index])
        distances, nearest_indexes = tree.query(world_points, k=1)
        valid_mask = distances <= max_correspondence_m
        if scan_index < initial_scan_count:
            valid_mask[:] = False
        bundles.append(
            ScanCorrespondenceBundle(
                scan_index=scan.scan_index,
                local_points=local_points[valid_mask].copy(),
                target_points=submap_points[nearest_indexes[valid_mask]].copy(),
                raw_distances_m=distances[valid_mask].copy(),
            )
        )
    return bundles


def _pack_poses(poses: np.ndarray) -> np.ndarray:
    return poses[1:].reshape(-1)


def _unpack_poses(flattened: np.ndarray, first_pose: np.ndarray) -> np.ndarray:
    poses = np.empty((1 + (flattened.size // 3), 3), dtype=np.float64)
    poses[0] = first_pose
    poses[1:] = flattened.reshape(-1, 3)
    return poses


def _refine_global_poses(
    initial_poses: np.ndarray,
    scans: list[LidarScan],
    wall_model: WallModel,
    yaw_priors_rad: np.ndarray,
    velocity_priors_mps: np.ndarray,
    *,
    initial_scan_count: int,
    scan_times_s: np.ndarray,
    submap_window_scans: int,
    point_stride: int,
    max_correspondence_m: float,
) -> np.ndarray:
    poses = initial_poses.copy()
    for _iteration in range(2):
        bundles = _build_correspondence_bundles(
            scans,
            poses,
            initial_scan_count=initial_scan_count,
            submap_window_scans=submap_window_scans,
            point_stride=point_stride,
            max_correspondence_m=max_correspondence_m,
        )
        first_pose = poses[0].copy()

        def residual_vector(flattened: np.ndarray) -> np.ndarray:
            current_poses = _unpack_poses(flattened, first_pose)
            residuals: list[np.ndarray] = []

            for pose_index, scan in enumerate(scans):
                pose = current_poses[pose_index]
                if pose_index > 0:
                    bundle = bundles[pose_index]
                    if bundle.local_points.size:
                        aligned_points = _transform_points(bundle.local_points, pose)
                        diffs = aligned_points - bundle.target_points
                        residuals.append((diffs * GLOBAL_SUBMAP_WEIGHT).reshape(-1))

                wall_residuals = _wall_residuals_for_pose(scan, pose, wall_model)
                if wall_residuals.size:
                    residuals.append(wall_residuals * GLOBAL_WALL_WEIGHT)

                residuals.append(
                    np.asarray([(pose[2] - yaw_priors_rad[pose_index]) * GLOBAL_YAW_WEIGHT], dtype=np.float64)
                )

                if pose_index > 0:
                    dt_s = max(1.0e-3, float(scan_times_s[pose_index] - scan_times_s[pose_index - 1]))
                    delta_prior = 0.5 * (
                        velocity_priors_mps[pose_index] + velocity_priors_mps[pose_index - 1]
                    ) * dt_s
                    delta_current = pose[:2] - current_poses[pose_index - 1, :2]
                    residuals.append((delta_current - delta_prior) * GLOBAL_VELOCITY_WEIGHT)

                if pose_index > 1:
                    smoothness = (
                        pose[:2]
                        - (2.0 * current_poses[pose_index - 1, :2])
                        + current_poses[pose_index - 2, :2]
                    )
                    residuals.append(smoothness * GLOBAL_SMOOTHNESS_WEIGHT)

                if 0 < pose_index < initial_scan_count:
                    residuals.append((pose[:2] - first_pose[:2]) * GLOBAL_STATIC_ANCHOR_WEIGHT)
                    residuals.append(
                        np.asarray([(pose[2] - first_pose[2]) * GLOBAL_STATIC_YAW_WEIGHT], dtype=np.float64)
                    )

            return np.concatenate(residuals)

        solution = least_squares(
            residual_vector,
            x0=_pack_poses(poses),
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=120,
        )
        poses = _unpack_poses(solution.x, first_pose) if solution.success else poses
    return poses


def _rotate_world_poses(poses: np.ndarray, theta_rad: float) -> np.ndarray:
    rotated = poses.copy()
    rotated[:, :2] = _rotate_points(rotated[:, :2], theta_rad)
    rotated[:, 2] = rotated[:, 2] + theta_rad
    return rotated


def _shift_world_origin(poses: np.ndarray, wall_model: WallModel) -> tuple[np.ndarray, np.ndarray]:
    shifted = poses.copy()
    origin_projection = np.asarray(
        [
            shifted[0, 0],
            _centerline_y_m(wall_model, shifted[0, 0]),
        ],
        dtype=np.float64,
    )
    shifted[:, :2] = shifted[:, :2] - origin_projection
    return shifted, origin_projection


def _compute_scan_qualities(
    scans: list[LidarScan],
    poses: np.ndarray,
    wall_model: WallModel,
    *,
    initial_scan_count: int,
    submap_window_scans: int,
    max_correspondence_m: float,
) -> list[ScanQuality]:
    qualities: list[ScanQuality] = []
    confidences = ["high"] * len(scans)
    for scan_index, scan in enumerate(scans):
        if scan_index == 0:
            submap_median = float("nan")
            wall_median = float(np.median(np.abs(_wall_residuals_for_pose(scan, poses[scan_index], wall_model))))
            valid_count = 0
        else:
            submap_points = _build_submap_points(
                scans,
                poses,
                confidences,
                scan_index,
                submap_window_scans,
            )
            submap_median, wall_median, valid_count = _evaluate_pose_quality(
                scan,
                poses[scan_index],
                submap_points,
                wall_model,
                max_correspondence_m,
            )

        confidence = "high"
        if scan_index >= initial_scan_count:
            if valid_count < 12:
                confidence = "low"
            if math.isfinite(submap_median) and submap_median > DEFAULT_LOW_CONFIDENCE_RESIDUAL_M:
                confidence = "low"
        confidences[scan_index] = confidence

        qualities.append(
            ScanQuality(
                scan_index=scan.scan_index,
                t_s=scan.t_s,
                median_submap_residual_m=submap_median,
                median_wall_residual_m=wall_median,
                valid_correspondence_count=valid_count,
                confidence=confidence,
            )
        )
    return qualities


def _write_trajectory_csv(
    path: Path,
    scans: list[LidarScan],
    poses: np.ndarray,
    qualities: list[ScanQuality],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_index",
                "t_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "confidence",
                "median_submap_residual_m",
                "median_wall_residual_m",
                "valid_correspondence_count",
            ]
        )
        for scan, pose, quality in zip(scans, poses, qualities):
            writer.writerow(
                [
                    scan.scan_index,
                    f"{scan.t_s:.9f}",
                    f"{pose[0]:.6f}",
                    f"{pose[1]:.6f}",
                    f"{pose[2]:.6f}",
                    quality.confidence,
                    "" if not math.isfinite(quality.median_submap_residual_m) else f"{quality.median_submap_residual_m:.6f}",
                    "" if not math.isfinite(quality.median_wall_residual_m) else f"{quality.median_wall_residual_m:.6f}",
                    quality.valid_correspondence_count,
                ]
            )


def _plot_overlay(
    output_path: Path,
    scans: list[LidarScan],
    poses: np.ndarray,
    wall_model: WallModel,
    qualities: list[ScanQuality],
    *,
    best_effort_init: bool,
    median_submap_residual_m: float,
    median_wall_residual_m: float,
    high_confidence_pct: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud_parts: list[np.ndarray] = []
    point_cloud_times: list[np.ndarray] = []
    for scan, pose in zip(scans, poses):
        world_points = _transform_points(scan.points_local, pose)
        point_cloud_parts.append(world_points)
        point_cloud_times.append(np.full(world_points.shape[0], scan.t_s, dtype=np.float64))

    point_cloud = np.vstack(point_cloud_parts)
    point_times = np.concatenate(point_cloud_times)
    time_min = float(np.min(point_times))
    time_max = float(np.max(point_times))
    time_norm = (
        (point_times - time_min) / max(1.0e-9, time_max - time_min)
        if point_times.size
        else np.zeros((0,), dtype=np.float64)
    )

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    scatter = ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        c=time_norm,
        s=5,
        cmap="viridis",
        alpha=0.45,
        linewidths=0.0,
        label="Nube LiDAR registrada",
    )
    fig.colorbar(scatter, ax=ax, pad=0.02, label="tiempo normalizado")

    ax.plot(poses[:, 0], poses[:, 1], color="#111111", linewidth=2.4, label="Trayectoria estimada")
    ax.scatter([poses[0, 0]], [poses[0, 1]], c="#00a676", s=80, marker="o", label="Inicio")
    ax.scatter([poses[-1, 0]], [poses[-1, 1]], c="#d7263d", s=80, marker="X", label="Fin")

    arrow_step = max(1, len(scans) // 12)
    for pose in poses[::arrow_step]:
        dx = 0.18 * math.cos(float(pose[2]))
        dy = 0.18 * math.sin(float(pose[2]))
        ax.arrow(
            float(pose[0]),
            float(pose[1]),
            dx,
            dy,
            width=0.005,
            head_width=0.06,
            head_length=0.08,
            color="#111111",
            alpha=0.85,
            length_includes_head=True,
        )

    x_line = np.linspace(float(np.min(point_cloud[:, 0])), float(np.max(point_cloud[:, 0])), num=120)
    ax.plot(
        x_line,
        np.polyval(wall_model.lower_coef, x_line),
        linestyle="--",
        linewidth=1.5,
        color="#2f6fed",
        alpha=0.8,
        label="Pared inferior estimada",
    )
    ax.plot(
        x_line,
        np.polyval(wall_model.upper_coef, x_line),
        linestyle="--",
        linewidth=1.5,
        color="#f06c00",
        alpha=0.8,
        label="Pared superior estimada",
    )

    info_lines = [
        f"scans: {len(scans)}",
        f"high confidence: {high_confidence_pct:.1f}%",
        f"mediana residual submapa: {median_submap_residual_m:.3f} m",
        f"mediana residual paredes: {median_wall_residual_m:.3f} m",
        f"best_effort_init: {best_effort_init}",
        f"ancho corredor: {wall_model.width_m:.3f} m",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#666666",
            "alpha": 0.92,
        },
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m] corredor")
    ax.set_ylabel("y [m] corredor")
    ax.set_title("Sensor fusion LiDAR + IMU en corredor estatico")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _nanmedian_or_default(values: list[float], default: float = float("nan")) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return default
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _write_summary_json(
    path: Path,
    *,
    run_dir: Path,
    imu_processing: ImuProcessingResult,
    wall_model: WallModel,
    qualities: list[ScanQuality],
    origin_projection_m: np.ndarray,
    alignment_yaw_rad: float,
    final_rotation_correction_rad: float,
    parameters: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    median_submap_residual_m = _nanmedian_or_default(
        [quality.median_submap_residual_m for quality in qualities],
        default=float("nan"),
    )
    median_wall_residual_m = _nanmedian_or_default(
        [quality.median_wall_residual_m for quality in qualities],
        default=float("nan"),
    )
    high_confidence_pct = 100.0 * (
        sum(1 for quality in qualities if quality.confidence == "high") / max(1, len(qualities))
    )

    summary = {
        "run_dir": str(run_dir),
        "used_pwm": False,
        "static_initialization": {
            "best_effort_init": imu_processing.best_effort_init,
            "static_start_s": imu_processing.static_start_s,
            "static_end_s": imu_processing.static_end_s,
            "static_sample_count": imu_processing.static_sample_count,
            "bias_ax_mps2": imu_processing.bias_ax_mps2,
            "bias_ay_mps2": imu_processing.bias_ay_mps2,
            "bias_az_mps2": imu_processing.bias_az_mps2,
            "bias_gz_rps": imu_processing.bias_gz_rps,
        },
        "corridor_model": {
            "width_m": wall_model.width_m,
            "lower_wall_coef": wall_model.lower_coef.tolist(),
            "upper_wall_coef": wall_model.upper_coef.tolist(),
            "alignment_yaw_rad": alignment_yaw_rad,
            "final_rotation_correction_rad": final_rotation_correction_rad,
            "origin_projection_m": origin_projection_m.tolist(),
        },
        "quality": {
            "median_wall_residual_m": median_wall_residual_m,
            "median_submap_residual_m": median_submap_residual_m,
            "high_confidence_pct": high_confidence_pct,
            "low_confidence_scan_count": sum(
                1 for quality in qualities if quality.confidence == "low"
            ),
        },
        "parameters": parameters,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_fusion(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    imu_path = run_dir / "imu_raw.csv"
    lidar_path = run_dir / "lidar_points.csv"
    if not imu_path.exists():
        raise SystemExit(f"No existe {imu_path}")
    if not lidar_path.exists():
        raise SystemExit(f"No existe {lidar_path}")

    scans = _load_lidar_scans(lidar_path, point_stride=args.point_stride)
    scan_times_s = np.asarray([scan.t_s for scan in scans], dtype=np.float64)

    initial_alignment_scan_count = max(DEFAULT_INITIAL_SCAN_COUNT_MIN, min(6, len(scans)))
    initial_points = np.vstack(
        [scans[index].points_local for index in range(initial_alignment_scan_count)]
    )
    alignment_yaw_rad, initial_wall_model = _estimate_initial_alignment(
        initial_points,
        bin_m=args.corridor_bin_m,
    )

    imu_raw = _load_imu_series(imu_path)
    imu_processing = _process_imu(
        imu_raw,
        median_window=args.median_window,
        ema_alpha=args.ema_alpha,
        static_window_s=args.static_window_s,
        static_search_s=args.static_search_s,
        velocity_decay_tau_s=args.velocity_decay_tau_s,
        world_yaw_offset_rad=alignment_yaw_rad,
    )

    yaw_priors_rad, velocity_priors_mps, accel_priors_mps2 = _interpolate_scan_priors(
        imu_processing,
        scan_times_s,
    )
    initial_scan_count = min(
        len(scans),
        _estimate_initial_scan_count(scan_times_s, imu_processing.static_end_s),
    )

    sequential_poses, sequential_qualities = _estimate_sequential_poses(
        scans,
        yaw_priors_rad,
        velocity_priors_mps,
        accel_priors_mps2,
        initial_wall_model,
        initial_yaw_rad=alignment_yaw_rad,
        initial_scan_count=initial_scan_count,
        submap_window_scans=args.submap_window_scans,
        max_correspondence_m=args.max_correspondence_m,
    )

    high_conf_filter = [quality.confidence for quality in sequential_qualities]
    refit_points = _collect_world_points(
        scans,
        sequential_poses,
        point_stride=args.global_point_stride,
        confidence_filter=high_conf_filter,
    )
    if refit_points.shape[0] < 60:
        refit_points = _collect_world_points(
            scans,
            sequential_poses,
            point_stride=args.global_point_stride,
            confidence_filter=None,
        )
    refined_wall_model = _fit_wall_model(refit_points, args.corridor_bin_m)

    global_poses = _refine_global_poses(
        sequential_poses,
        scans,
        refined_wall_model,
        yaw_priors_rad,
        velocity_priors_mps,
        initial_scan_count=initial_scan_count,
        scan_times_s=scan_times_s,
        submap_window_scans=args.submap_window_scans,
        point_stride=args.global_point_stride,
        max_correspondence_m=args.max_correspondence_m,
    )

    correction_points = _collect_world_points(
        scans,
        global_poses,
        point_stride=args.global_point_stride,
        confidence_filter=None,
    )
    correction_model = _fit_wall_model(correction_points, args.corridor_bin_m)
    final_rotation_correction_rad = -float(correction_model.corridor_yaw_rad)
    corrected_poses = _rotate_world_poses(global_poses, final_rotation_correction_rad)

    corrected_points = _collect_world_points(
        scans,
        corrected_poses,
        point_stride=args.global_point_stride,
        confidence_filter=None,
    )
    final_wall_model = _fit_wall_model(corrected_points, args.corridor_bin_m)
    final_poses, origin_projection_m = _shift_world_origin(corrected_poses, final_wall_model)
    final_qualities = _compute_scan_qualities(
        scans,
        final_poses,
        final_wall_model,
        initial_scan_count=initial_scan_count,
        submap_window_scans=args.submap_window_scans,
        max_correspondence_m=args.max_correspondence_m,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (run_dir / "analysis_sensor_fusion")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "sensor_fusion_trajectory.csv"
    summary_path = output_dir / "sensor_fusion_summary.json"
    overlay_path = output_dir / "sensor_fusion_overlay.png"

    _write_trajectory_csv(trajectory_path, scans, final_poses, final_qualities)

    median_submap_residual_m = _nanmedian_or_default(
        [quality.median_submap_residual_m for quality in final_qualities]
    )
    median_wall_residual_m = _nanmedian_or_default(
        [quality.median_wall_residual_m for quality in final_qualities]
    )
    high_confidence_pct = 100.0 * (
        sum(1 for quality in final_qualities if quality.confidence == "high")
        / max(1, len(final_qualities))
    )

    _plot_overlay(
        overlay_path,
        scans,
        final_poses,
        final_wall_model,
        final_qualities,
        best_effort_init=imu_processing.best_effort_init,
        median_submap_residual_m=median_submap_residual_m,
        median_wall_residual_m=median_wall_residual_m,
        high_confidence_pct=high_confidence_pct,
    )
    _write_summary_json(
        summary_path,
        run_dir=run_dir,
        imu_processing=imu_processing,
        wall_model=final_wall_model,
        qualities=final_qualities,
        origin_projection_m=origin_projection_m,
        alignment_yaw_rad=alignment_yaw_rad,
        final_rotation_correction_rad=final_rotation_correction_rad,
        parameters={
            "median_window": args.median_window,
            "ema_alpha": args.ema_alpha,
            "static_window_s": args.static_window_s,
            "static_search_s": args.static_search_s,
            "velocity_decay_tau_s": args.velocity_decay_tau_s,
            "submap_window_scans": args.submap_window_scans,
            "point_stride": args.point_stride,
            "global_point_stride": args.global_point_stride,
            "max_correspondence_m": args.max_correspondence_m,
            "initial_scan_count": initial_scan_count,
        },
    )

    return {
        "trajectory_path": trajectory_path,
        "summary_path": summary_path,
        "overlay_path": overlay_path,
        "qualities": final_qualities,
        "scan_count": len(scans),
        "median_submap_residual_m": median_submap_residual_m,
        "median_wall_residual_m": median_wall_residual_m,
        "high_confidence_pct": high_confidence_pct,
        "best_effort_init": imu_processing.best_effort_init,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directorio que contiene imu_raw.csv y lidar_points.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directorio de salida. Por defecto: <run-dir>/analysis_sensor_fusion.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=DEFAULT_MEDIAN_WINDOW,
        help=f"Ventana impar del filtro de mediana. Por defecto: {DEFAULT_MEDIAN_WINDOW}.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=DEFAULT_EMA_ALPHA,
        help=f"Alpha del filtro exponencial bidireccional. Por defecto: {DEFAULT_EMA_ALPHA}.",
    )
    parser.add_argument(
        "--static-window-s",
        type=float,
        default=DEFAULT_STATIC_WINDOW_S,
        help=f"Duracion de la ventana inicial de inmovilidad. Por defecto: {DEFAULT_STATIC_WINDOW_S}.",
    )
    parser.add_argument(
        "--static-search-s",
        type=float,
        default=DEFAULT_STATIC_SEARCH_S,
        help=f"Intervalo inicial de busqueda para la ventana estatica. Por defecto: {DEFAULT_STATIC_SEARCH_S}.",
    )
    parser.add_argument(
        "--velocity-decay-tau-s",
        type=float,
        default=DEFAULT_VELOCITY_DECAY_TAU_S,
        help=f"Constante de decaimiento para la velocidad integrada IMU. Por defecto: {DEFAULT_VELOCITY_DECAY_TAU_S}.",
    )
    parser.add_argument(
        "--submap-window-scans",
        type=int,
        default=DEFAULT_SUBMAP_WINDOW_SCANS,
        help=f"Numero de scans previos en el submapa local. Por defecto: {DEFAULT_SUBMAP_WINDOW_SCANS}.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=DEFAULT_SCAN_POINT_STRIDE,
        help=f"Submuestreo local de puntos LiDAR. Por defecto: {DEFAULT_SCAN_POINT_STRIDE}.",
    )
    parser.add_argument(
        "--global-point-stride",
        type=int,
        default=DEFAULT_GLOBAL_POINT_STRIDE,
        help=f"Submuestreo de puntos para refinamiento global. Por defecto: {DEFAULT_GLOBAL_POINT_STRIDE}.",
    )
    parser.add_argument(
        "--max-correspondence-m",
        type=float,
        default=DEFAULT_MAX_CORRESPONDENCE_M,
        help=f"Distancia maxima de correspondencia LiDAR-submapa. Por defecto: {DEFAULT_MAX_CORRESPONDENCE_M}.",
    )
    parser.add_argument(
        "--corridor-bin-m",
        type=float,
        default=DEFAULT_CORRIDOR_BIN_M,
        help=f"Bin en x usado para ajustar las paredes del corredor. Por defecto: {DEFAULT_CORRIDOR_BIN_M}.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_filter_params(args.median_window, args.ema_alpha)
    if args.point_stride < 1 or args.global_point_stride < 1:
        raise SystemExit("Los strides LiDAR deben ser enteros mayores o iguales a 1.")
    if args.submap_window_scans < 2:
        raise SystemExit("--submap-window-scans debe ser mayor o igual a 2.")
    if args.max_correspondence_m <= 0.0:
        raise SystemExit("--max-correspondence-m debe ser positivo.")
    if args.corridor_bin_m <= 0.0:
        raise SystemExit("--corridor-bin-m debe ser positivo.")

    result = _run_fusion(args)
    print(f"trayectoria_csv: {result['trajectory_path']}")
    print(f"summary_json: {result['summary_path']}")
    print(f"overlay_png: {result['overlay_path']}")
    print(f"scan_count: {result['scan_count']}")
    print(f"median_submap_residual_m: {result['median_submap_residual_m']:.6f}")
    print(f"median_wall_residual_m: {result['median_wall_residual_m']:.6f}")
    print(f"high_confidence_pct: {result['high_confidence_pct']:.2f}")
    print(f"best_effort_init: {result['best_effort_init']}")


if __name__ == "__main__":
    main()
