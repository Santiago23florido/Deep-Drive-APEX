#!/usr/bin/env python3
"""Offline full-lap reconstruction from recorded simulation datasets.

This script reconstructs a trajectory and an accumulated LiDAR map from a
recorded run directory. It uses LiDAR as the dominant geometric cue and IMU as
the rotational/dynamic prior, then scores the reconstruction against the
available Gazebo ground-truth exports for iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


GRAVITY_MPS2 = 9.80665


EXPECTED_COLUMNS = {
    "imu_raw.csv": [
        "stamp_sec",
        "stamp_nanosec",
        "ax_mps2",
        "ay_mps2",
        "az_mps2",
        "gx_rps",
        "gy_rps",
        "gz_rps",
        "qx",
        "qy",
        "qz",
        "qw",
    ],
    "lidar_points.csv": [
        "scan_index",
        "stamp_sec",
        "stamp_nanosec",
        "beam_index",
        "angle_rad",
        "range_m",
        "x_forward_m",
        "y_left_m",
    ],
    "scan_index.csv": [
        "scan_index",
        "stamp_sec",
        "stamp_nanosec",
        "frame_id",
        "angle_min_rad",
        "angle_max_rad",
        "angle_increment_rad",
        "time_increment_s",
        "scan_time_s",
        "range_min_m",
        "range_max_m",
        "beam_count",
        "valid_point_count",
    ],
    "odom_fused.csv": [
        "stamp_sec",
        "stamp_nanosec",
        "frame_id",
        "child_frame_id",
        "x_m",
        "y_m",
        "yaw_rad",
        "vx_mps",
        "vy_mps",
        "yaw_rate_rps",
    ],
    "ground_truth_path.csv": [
        "pose_index",
        "stamp_sec",
        "stamp_nanosec",
        "x_m",
        "y_m",
        "yaw_rad",
    ],
    "track_geometry.csv": [
        "x_m",
        "y_m",
    ],
}


def _stamp_from_row(row: dict[str, str]) -> float:
    return float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"]))


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


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
    inv_rot = _rotation_matrix(float(-pose[2]))
    inv_xy = -(inv_rot @ pose[:2])
    return np.asarray(
        [float(inv_xy[0]), float(inv_xy[1]), _wrap_angle(float(-pose[2]))],
        dtype=np.float64,
    )


def _relative_pose(reference_pose_xyyaw: np.ndarray, target_pose_xyyaw: np.ndarray) -> np.ndarray:
    return _compose_poses(_inverse_pose(reference_pose_xyyaw), target_pose_xyyaw)


def _scaled_pose_delta(delta_pose_xyyaw: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta = np.asarray(delta_pose_xyyaw, dtype=np.float64)
    return np.asarray(
        [alpha * float(delta[0]), alpha * float(delta[1]), alpha * float(delta[2])],
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


def _voxel_downsample(points_xy: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_xy.size == 0 or voxel_size_m <= 1.0e-6:
        return points_xy
    grid = np.floor(points_xy / voxel_size_m).astype(np.int64)
    _, unique_indexes = np.unique(grid, axis=0, return_index=True)
    unique_indexes.sort()
    return points_xy[unique_indexes]


def _path_length_m(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    deltas = np.diff(points_xy, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _median_filter(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values.copy()
    if (window_size % 2) == 0:
        raise ValueError("median window must be odd")
    radius = window_size // 2
    filtered = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        filtered[index] = statistics.median(values[start:end].tolist())
    return filtered


def _ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for index in range(1, values.size):
        filtered[index] = (alpha * values[index]) + ((1.0 - alpha) * filtered[index - 1])
    return filtered


def _zero_phase_filter(values: np.ndarray, median_window: int, ema_alpha: float) -> np.ndarray:
    medianed = _median_filter(values, median_window)
    forward = _ema_filter(medianed, ema_alpha)
    backward = _ema_filter(forward[::-1], ema_alpha)
    return backward[::-1]


def _ensure_exact_columns(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if fieldnames != expected_columns:
            raise SystemExit(
                f"{path} columns mismatch.\n"
                f"Expected: {expected_columns}\n"
                f"Found:    {fieldnames}"
            )
        return [row for row in reader]


@dataclass(frozen=True)
class ScanRecord:
    scan_index: int
    stamp_sec: int
    stamp_nanosec: int
    stamp_s: float
    frame_id: str
    points_local: np.ndarray
    sampled_points_local: np.ndarray


@dataclass(frozen=True)
class ImuSeries:
    t_s: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    az_mps2: np.ndarray
    gz_rps: np.ndarray


@dataclass(frozen=True)
class ProcessedImu:
    t_s: np.ndarray
    yaw_rad: np.ndarray
    yaw_rate_rps: np.ndarray
    bias_ax_mps2: float
    bias_ay_mps2: float
    bias_az_mps2: float
    bias_gz_rps: float
    static_start_s: float
    static_end_s: float
    static_sample_count: int
    best_effort_init: bool


@dataclass(frozen=True)
class OdomSeries:
    t_s: np.ndarray
    pose_xyyaw: np.ndarray


@dataclass(frozen=True)
class RelativeMotionEstimate:
    delta_pose_xyyaw: np.ndarray
    valid_match_count: int
    median_residual_m: float
    success: bool


@dataclass(frozen=True)
class ScanQuality:
    confidence: str
    inlier_count: int
    median_submap_residual_m: float
    source: str


@dataclass(frozen=True)
class WorldToEstimation:
    tx_m: float
    ty_m: float
    yaw_rad: float
    samples_used: int
    max_translation_drift_m: float
    max_yaw_drift_rad: float

    def transform_points(self, points_world_xy: np.ndarray) -> np.ndarray:
        return _transform_points(
            points_world_xy,
            np.asarray([self.tx_m, self.ty_m, self.yaw_rad], dtype=np.float64),
        )


def _load_imu(run_dir: Path) -> ImuSeries:
    rows = _ensure_exact_columns(run_dir / "imu_raw.csv", EXPECTED_COLUMNS["imu_raw.csv"])
    if not rows:
        raise SystemExit(f"No IMU samples found in {run_dir / 'imu_raw.csv'}")
    return ImuSeries(
        t_s=np.asarray([_stamp_from_row(row) for row in rows], dtype=np.float64),
        ax_mps2=np.asarray([float(row["ax_mps2"]) for row in rows], dtype=np.float64),
        ay_mps2=np.asarray([float(row["ay_mps2"]) for row in rows], dtype=np.float64),
        az_mps2=np.asarray([float(row["az_mps2"]) for row in rows], dtype=np.float64),
        gz_rps=np.asarray([float(row["gz_rps"]) for row in rows], dtype=np.float64),
    )


def _load_scans(run_dir: Path, point_stride: int) -> list[ScanRecord]:
    index_rows = _ensure_exact_columns(run_dir / "scan_index.csv", EXPECTED_COLUMNS["scan_index.csv"])
    lidar_rows = _ensure_exact_columns(run_dir / "lidar_points.csv", EXPECTED_COLUMNS["lidar_points.csv"])
    grouped_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in lidar_rows:
        scan_index = int(row["scan_index"])
        x_m = float(row["x_forward_m"])
        y_m = float(row["y_left_m"])
        range_m = float(row["range_m"])
        if not math.isfinite(x_m) or not math.isfinite(y_m) or not math.isfinite(range_m):
            continue
        grouped_points[scan_index].append((x_m, y_m))

    scans: list[ScanRecord] = []
    for row in index_rows:
        scan_index = int(row["scan_index"])
        points_local = np.asarray(grouped_points.get(scan_index, []), dtype=np.float64)
        if points_local.ndim != 2:
            points_local = np.empty((0, 2), dtype=np.float64)
        sampled = points_local[::point_stride].copy() if points_local.size else np.empty((0, 2), dtype=np.float64)
        scans.append(
            ScanRecord(
                scan_index=scan_index,
                stamp_sec=int(row["stamp_sec"]),
                stamp_nanosec=int(row["stamp_nanosec"]),
                stamp_s=float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"])),
                frame_id=str(row["frame_id"]),
                points_local=points_local,
                sampled_points_local=sampled,
            )
        )
    if not scans:
        raise SystemExit(f"No scans found in {run_dir}")
    return scans


def _load_odom(run_dir: Path) -> OdomSeries | None:
    path = run_dir / "odom_fused.csv"
    if not path.exists():
        return None
    rows = _ensure_exact_columns(path, EXPECTED_COLUMNS["odom_fused.csv"])
    if not rows:
        return None
    t_s = np.asarray([_stamp_from_row(row) for row in rows], dtype=np.float64)
    pose_xyyaw = np.asarray(
        [
            [float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    return OdomSeries(t_s=t_s, pose_xyyaw=pose_xyyaw)


def _load_xy_path(path: Path, expected_columns: list[str], *, with_stamp: bool) -> tuple[np.ndarray, np.ndarray | None]:
    rows = _ensure_exact_columns(path, expected_columns)
    points_xy = np.asarray(
        [[float(row["x_m"]), float(row["y_m"])] for row in rows],
        dtype=np.float64,
    )
    if not with_stamp:
        return points_xy, None
    times = np.asarray([_stamp_from_row(row) for row in rows], dtype=np.float64)
    return points_xy, times


def _load_path_with_yaw(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    path = run_dir / "ground_truth_path.csv"
    if not path.exists():
        return None, None
    rows = _ensure_exact_columns(path, EXPECTED_COLUMNS["ground_truth_path.csv"])
    if not rows:
        return None, None
    times = np.asarray([_stamp_from_row(row) for row in rows], dtype=np.float64)
    poses = np.asarray(
        [
            [float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    return poses, times


def _load_track_geometry(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "track_geometry.csv"
    if not path.exists():
        return None
    points_xy, _ = _load_xy_path(path, EXPECTED_COLUMNS["track_geometry.csv"], with_stamp=False)
    return points_xy


def _load_world_to_estimation(run_dir: Path) -> WorldToEstimation | None:
    path = run_dir / "ground_truth_status.jsonl"
    if not path.exists():
        return None
    txs: list[float] = []
    tys: list[float] = []
    yaws: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            pose_world = payload.get("pose_gt")
            pose_est = payload.get("pose_gt_estimation_frame")
            if not isinstance(pose_world, dict) or not isinstance(pose_est, dict):
                continue
            try:
                world_x = float(pose_world["x_m"])
                world_y = float(pose_world["y_m"])
                world_yaw = float(pose_world["yaw_rad"])
                est_x = float(pose_est["x_m"])
                est_y = float(pose_est["y_m"])
                est_yaw = float(pose_est["yaw_rad"])
            except Exception:
                continue
            yaw_offset = _wrap_angle(est_yaw - world_yaw)
            cos_yaw = math.cos(yaw_offset)
            sin_yaw = math.sin(yaw_offset)
            tx_m = est_x - ((cos_yaw * world_x) - (sin_yaw * world_y))
            ty_m = est_y - ((sin_yaw * world_x) + (cos_yaw * world_y))
            txs.append(tx_m)
            tys.append(ty_m)
            yaws.append(yaw_offset)
    if not txs:
        return None
    median_tx = float(np.median(np.asarray(txs, dtype=np.float64)))
    median_ty = float(np.median(np.asarray(tys, dtype=np.float64)))
    median_yaw = float(np.median(np.unwrap(np.asarray(yaws, dtype=np.float64))))
    translation_drifts = [
        math.hypot(tx - median_tx, ty - median_ty)
        for tx, ty in zip(txs, tys)
    ]
    yaw_drifts = [
        abs(_wrap_angle(yaw - median_yaw))
        for yaw in yaws
    ]
    return WorldToEstimation(
        tx_m=median_tx,
        ty_m=median_ty,
        yaw_rad=_wrap_angle(median_yaw),
        samples_used=len(txs),
        max_translation_drift_m=float(max(translation_drifts) if translation_drifts else 0.0),
        max_yaw_drift_rad=float(max(yaw_drifts) if yaw_drifts else 0.0),
    )


def _detect_static_window(
    imu: ImuSeries,
    *,
    window_s: float,
    search_s: float,
) -> tuple[int, int, bool]:
    t_s = imu.t_s
    if t_s.size < 4:
        return 0, max(1, t_s.size), True
    fallback_end = int(np.searchsorted(t_s, t_s[0] + window_s, side="right"))
    fallback_end = max(2, min(fallback_end, t_s.size))
    best_score = float("inf")
    best_window = (0, fallback_end)
    best_effort = True
    for start_index, start_t_s in enumerate(t_s):
        if start_t_s > (t_s[0] + search_s - window_s):
            break
        end_index = int(np.searchsorted(t_s, start_t_s + window_s, side="right"))
        if (end_index - start_index) < 4:
            continue
        ax_std = float(np.std(imu.ax_mps2[start_index:end_index]))
        ay_std = float(np.std(imu.ay_mps2[start_index:end_index]))
        az_std = float(np.std(imu.az_mps2[start_index:end_index]))
        gz_std = float(np.std(imu.gz_rps[start_index:end_index]))
        score = ax_std + ay_std + abs(az_std - 0.02) + (2.0 * gz_std)
        if score < best_score:
            best_score = score
            best_window = (start_index, end_index)
            best_effort = False
    return best_window[0], best_window[1], best_effort


def _process_imu(
    imu: ImuSeries,
    *,
    median_window: int,
    ema_alpha: float,
    static_window_s: float,
    static_search_s: float,
) -> ProcessedImu:
    static_start_index, static_end_index, best_effort_init = _detect_static_window(
        imu,
        window_s=static_window_s,
        search_s=static_search_s,
    )
    bias_ax = float(np.mean(imu.ax_mps2[static_start_index:static_end_index]))
    bias_ay = float(np.mean(imu.ay_mps2[static_start_index:static_end_index]))
    bias_az = float(np.mean(imu.az_mps2[static_start_index:static_end_index]))
    bias_gz = float(np.mean(imu.gz_rps[static_start_index:static_end_index]))

    gz_filtered = _zero_phase_filter(imu.gz_rps - bias_gz, median_window, ema_alpha)
    yaw_rad = np.zeros_like(imu.t_s)
    for index in range(1, imu.t_s.size):
        dt_s = max(1.0e-4, float(imu.t_s[index] - imu.t_s[index - 1]))
        yaw_rad[index] = _wrap_angle(float(yaw_rad[index - 1] + (gz_filtered[index] * dt_s)))
    return ProcessedImu(
        t_s=imu.t_s.copy(),
        yaw_rad=yaw_rad,
        yaw_rate_rps=gz_filtered,
        bias_ax_mps2=bias_ax,
        bias_ay_mps2=bias_ay,
        bias_az_mps2=bias_az - GRAVITY_MPS2,
        bias_gz_rps=bias_gz,
        static_start_s=float(imu.t_s[static_start_index]),
        static_end_s=float(imu.t_s[static_end_index - 1]),
        static_sample_count=int(static_end_index - static_start_index),
        best_effort_init=best_effort_init,
    )


def _interpolate_yaw_prior(processed_imu: ProcessedImu, scan_times_s: np.ndarray) -> np.ndarray:
    return np.interp(scan_times_s, processed_imu.t_s, processed_imu.yaw_rad).astype(np.float64, copy=False)


def _integrated_yaw_delta(processed_imu: ProcessedImu, start_s: float, end_s: float) -> float:
    yaw_start = float(np.interp(start_s, processed_imu.t_s, processed_imu.yaw_rad))
    yaw_end = float(np.interp(end_s, processed_imu.t_s, processed_imu.yaw_rad))
    return _wrap_angle(yaw_end - yaw_start)


def _estimate_relative_scan_motion(
    previous_points_xy: np.ndarray,
    current_points_xy: np.ndarray,
    *,
    yaw_delta_guess_rad: float,
    max_correspondence_m: float,
    submap_voxel_size_m: float,
) -> RelativeMotionEstimate:
    if previous_points_xy.shape[0] < 12 or current_points_xy.shape[0] < 12:
        return RelativeMotionEstimate(
            delta_pose_xyyaw=np.asarray([0.0, 0.0, _wrap_angle(yaw_delta_guess_rad)], dtype=np.float64),
            valid_match_count=0,
            median_residual_m=float("nan"),
            success=False,
        )

    target_points = _voxel_downsample(previous_points_xy, submap_voxel_size_m)
    source_points = _voxel_downsample(current_points_xy, submap_voxel_size_m)
    if target_points.shape[0] < 12 or source_points.shape[0] < 12:
        return RelativeMotionEstimate(
            delta_pose_xyyaw=np.asarray([0.0, 0.0, _wrap_angle(yaw_delta_guess_rad)], dtype=np.float64),
            valid_match_count=0,
            median_residual_m=float("nan"),
            success=False,
        )

    relative_pose = np.asarray([0.0, 0.0, _wrap_angle(yaw_delta_guess_rad)], dtype=np.float64)
    tree = cKDTree(target_points)
    valid_match_count = 0
    median_residual_m = float("nan")
    for _ in range(6):
        transformed_source = _transform_points(source_points, relative_pose)
        distances_m, nearest_indexes = tree.query(
            transformed_source,
            k=1,
            distance_upper_bound=max(max_correspondence_m * 1.25, 0.40),
        )
        valid_mask = np.isfinite(distances_m) & (distances_m < max(max_correspondence_m * 1.25, 0.40))
        valid_match_count = int(np.count_nonzero(valid_mask))
        if valid_match_count < max(10, min(40, source_points.shape[0] // 6)):
            break
        median_residual_m = float(np.median(distances_m[valid_mask]))
        incremental_pose = _best_fit_rigid_transform_2d(
            transformed_source[valid_mask],
            target_points[nearest_indexes[valid_mask]],
        )
        relative_pose = _compose_poses(incremental_pose, relative_pose)
        relative_pose[2] = _wrap_angle((0.35 * yaw_delta_guess_rad) + (0.65 * float(relative_pose[2])))

    relative_pose[0] = float(np.clip(relative_pose[0], -0.80, 0.80))
    relative_pose[1] = float(np.clip(relative_pose[1], -0.80, 0.80))
    relative_pose[2] = _wrap_angle(float(relative_pose[2]))
    return RelativeMotionEstimate(
        delta_pose_xyyaw=relative_pose,
        valid_match_count=valid_match_count,
        median_residual_m=median_residual_m,
        success=valid_match_count >= 10,
    )


def _interpolate_pose_series(series: OdomSeries | None, query_t_s: float) -> np.ndarray | None:
    if series is None or series.t_s.size == 0:
        return None
    if query_t_s < float(series.t_s[0]) or query_t_s > float(series.t_s[-1]):
        return None
    index = int(np.searchsorted(series.t_s, query_t_s, side="left"))
    if index <= 0:
        return series.pose_xyyaw[0].copy()
    if index >= series.t_s.size:
        return series.pose_xyyaw[-1].copy()
    prev_t = float(series.t_s[index - 1])
    next_t = float(series.t_s[index])
    alpha = 0.0 if next_t <= prev_t else float((query_t_s - prev_t) / (next_t - prev_t))
    prev_pose = series.pose_xyyaw[index - 1]
    next_pose = series.pose_xyyaw[index]
    xy = ((1.0 - alpha) * prev_pose[:2]) + (alpha * next_pose[:2])
    yaw_delta = _wrap_angle(float(next_pose[2] - prev_pose[2]))
    yaw = _wrap_angle(float(prev_pose[2] + (alpha * yaw_delta)))
    return np.asarray([float(xy[0]), float(xy[1]), yaw], dtype=np.float64)


def _build_initial_pose_chain(
    scans: list[ScanRecord],
    processed_imu: ProcessedImu,
    *,
    max_correspondence_m: float,
    submap_voxel_size_m: float,
) -> tuple[np.ndarray, list[RelativeMotionEstimate], np.ndarray]:
    scan_times_s = np.asarray([record.stamp_s for record in scans], dtype=np.float64)
    yaw_prior = _interpolate_yaw_prior(processed_imu, scan_times_s)
    poses = np.zeros((len(scans), 3), dtype=np.float64)
    relative_motions: list[RelativeMotionEstimate] = [
        RelativeMotionEstimate(
            delta_pose_xyyaw=np.zeros(3, dtype=np.float64),
            valid_match_count=0,
            median_residual_m=float("nan"),
            success=True,
        )
    ]
    poses[0] = np.zeros(3, dtype=np.float64)
    for index in range(1, len(scans)):
        yaw_delta_guess = _integrated_yaw_delta(
            processed_imu,
            scans[index - 1].stamp_s,
            scans[index].stamp_s,
        )
        relative = _estimate_relative_scan_motion(
            scans[index - 1].sampled_points_local,
            scans[index].sampled_points_local,
            yaw_delta_guess_rad=yaw_delta_guess,
            max_correspondence_m=max_correspondence_m,
            submap_voxel_size_m=submap_voxel_size_m,
        )
        relative_delta = relative.delta_pose_xyyaw.copy()
        relative_delta[2] = _wrap_angle((0.55 * yaw_delta_guess) + (0.45 * float(relative_delta[2])))
        poses[index] = _compose_poses(poses[index - 1], relative_delta)
        relative_motions.append(
            RelativeMotionEstimate(
                delta_pose_xyyaw=relative_delta,
                valid_match_count=relative.valid_match_count,
                median_residual_m=relative.median_residual_m,
                success=relative.success,
            )
        )
    return poses, relative_motions, yaw_prior


def _evaluate_pose_quality(
    scan_points_local: np.ndarray,
    pose_xyyaw: np.ndarray,
    submap_points_xy: np.ndarray,
    max_correspondence_m: float,
) -> tuple[int, float]:
    if scan_points_local.size == 0 or submap_points_xy.size == 0:
        return 0, float("nan")
    world_points = _transform_points(scan_points_local, pose_xyyaw)
    tree = cKDTree(submap_points_xy)
    distances_m, _ = tree.query(world_points, k=1)
    valid_mask = distances_m <= max_correspondence_m
    inlier_count = int(np.count_nonzero(valid_mask))
    median_residual_m = float(np.median(distances_m[valid_mask])) if inlier_count else float("nan")
    return inlier_count, median_residual_m


def _collect_world_points(
    scans: list[ScanRecord],
    poses_xyyaw: np.ndarray,
    indexes: Iterable[int],
    *,
    use_sampled: bool,
    confidence: list[str] | None = None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for index in indexes:
        if confidence is not None and confidence[index] == "low":
            continue
        points_local = scans[index].sampled_points_local if use_sampled else scans[index].points_local
        if points_local.size == 0:
            continue
        parts.append(_transform_points(points_local, poses_xyyaw[index]))
    if not parts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(parts)


def _window_overlap_count(window_start: int, total_scans: int, window_overlap_count: int) -> int:
    if window_start <= 0:
        return 0
    return min(window_overlap_count, total_scans)


def _estimate_global_correction(
    source_points_xy: np.ndarray,
    target_points_xy: np.ndarray,
    *,
    max_correspondence_m: float,
) -> np.ndarray:
    if source_points_xy.shape[0] < 20 or target_points_xy.shape[0] < 20:
        return np.zeros(3, dtype=np.float64)
    source = source_points_xy.copy()
    correction = np.zeros(3, dtype=np.float64)
    target_tree = cKDTree(target_points_xy)
    for _ in range(8):
        corrected_source = _transform_points(source, correction)
        distances_m, nearest_indexes = target_tree.query(
            corrected_source,
            k=1,
            distance_upper_bound=max_correspondence_m,
        )
        valid_mask = np.isfinite(distances_m) & (distances_m <= max_correspondence_m)
        if int(np.count_nonzero(valid_mask)) < 15:
            break
        incremental = _best_fit_rigid_transform_2d(
            corrected_source[valid_mask],
            target_points_xy[nearest_indexes[valid_mask]],
        )
        correction = _compose_poses(incremental, correction)
        if np.linalg.norm(incremental[:2]) < 1.0e-3 and abs(float(incremental[2])) < 1.0e-3:
            break
    correction[0] = float(np.clip(correction[0], -0.60, 0.60))
    correction[1] = float(np.clip(correction[1], -0.60, 0.60))
    correction[2] = _wrap_angle(float(np.clip(correction[2], -0.35, 0.35)))
    return correction


def _refine_window(
    scans: list[ScanRecord],
    global_poses_xyyaw: np.ndarray,
    *,
    window_id: int,
    window_start: int,
    window_end: int,
    window_overlap_count: int,
    submap_history_count: int,
    max_correspondence_m: float,
    window_registration_max_correspondence_m: float,
    lidar_weight: float,
    motion_prior_weight: float,
    yaw_prior_weight: float,
    odom_prior_xy_weight: float,
    odom_prior_yaw_weight: float,
    use_odom_prior: bool,
    yaw_prior_rad: np.ndarray,
    reference_deltas: list[RelativeMotionEstimate],
    odom_series: OdomSeries | None,
    submap_voxel_size_m: float,
) -> tuple[np.ndarray, list[ScanQuality], dict[str, float | int | list[float]]]:
    window_indexes = list(range(window_start, window_end + 1))
    window_count = len(window_indexes)
    window_overlap = _window_overlap_count(window_start, window_count, window_overlap_count)
    refined_poses = global_poses_xyyaw[window_indexes].copy()
    accumulated_anchor_indexes = range(max(0, window_start - max(window_overlap_count, submap_history_count)), window_start)
    anchor_points_xy = _collect_world_points(
        scans,
        global_poses_xyyaw,
        accumulated_anchor_indexes,
        use_sampled=True,
        confidence=None,
    )
    anchor_points_xy = _voxel_downsample(anchor_points_xy, submap_voxel_size_m)
    window_qualities: list[ScanQuality] = [
        ScanQuality(confidence="low", inlier_count=0, median_submap_residual_m=float("nan"), source="window_init")
        for _ in window_indexes
    ]
    pass_count = 3
    for _ in range(pass_count):
        max_translation_update = 0.0
        max_yaw_update = 0.0
        for local_index, global_index in enumerate(window_indexes):
            current_scan = scans[global_index]
            if local_index == 0 and window_start > 0:
                continue

            submap_parts: list[np.ndarray] = []
            if anchor_points_xy.size:
                submap_parts.append(anchor_points_xy)
            history_start = max(0, local_index - submap_history_count)
            for previous_local_index in range(history_start, local_index):
                previous_points = scans[window_indexes[previous_local_index]].sampled_points_local
                if previous_points.size == 0:
                    continue
                submap_parts.append(_transform_points(previous_points, refined_poses[previous_local_index]))
            if not submap_parts:
                continue
            submap_points_xy = _voxel_downsample(np.vstack(submap_parts), submap_voxel_size_m)
            if submap_points_xy.shape[0] < 12 or current_scan.sampled_points_local.shape[0] < 12:
                continue

            reference_pose = refined_poses[local_index].copy()
            if local_index > 0:
                reference_pose = _compose_poses(
                    refined_poses[local_index - 1],
                    reference_deltas[global_index].delta_pose_xyyaw,
                )
            odom_prior_pose = (
                _interpolate_pose_series(odom_series, current_scan.stamp_s)
                if use_odom_prior
                else None
            )
            if odom_prior_pose is not None and local_index == 0 and window_start == 0:
                reference_pose[:2] = 0.5 * (reference_pose[:2] + odom_prior_pose[:2])

            tree = cKDTree(submap_points_xy)

            def residual_vector(pose_flat: np.ndarray) -> np.ndarray:
                pose = pose_flat.astype(np.float64, copy=False)
                world_points = _transform_points(current_scan.sampled_points_local, pose)
                distances_m, nearest_indexes = tree.query(world_points, k=1)
                residuals: list[np.ndarray] = []
                diffs = world_points - submap_points_xy[nearest_indexes]
                scales = np.minimum(
                    1.0,
                    max_correspondence_m / np.maximum(distances_m, 1.0e-6),
                ).reshape(-1, 1)
                residuals.append((diffs * scales * lidar_weight).reshape(-1))
                if global_index > 0:
                    prev_pose = refined_poses[local_index - 1] if local_index > 0 else global_poses_xyyaw[global_index - 1]
                    relative = _relative_pose(prev_pose, pose)
                    reference_delta = reference_deltas[global_index].delta_pose_xyyaw
                    motion_residual = np.asarray(
                        [
                            (relative[0] - reference_delta[0]) * motion_prior_weight,
                            (relative[1] - reference_delta[1]) * motion_prior_weight,
                        ],
                        dtype=np.float64,
                    )
                    residuals.append(motion_residual)
                residuals.append(
                    np.asarray(
                        [(_wrap_angle(float(pose[2] - yaw_prior_rad[global_index]))) * yaw_prior_weight],
                        dtype=np.float64,
                    )
                )
                if odom_prior_pose is not None:
                    residuals.append(
                        np.asarray(
                            [
                                (pose[0] - odom_prior_pose[0]) * odom_prior_xy_weight,
                                (pose[1] - odom_prior_pose[1]) * odom_prior_xy_weight,
                                (_wrap_angle(float(pose[2] - odom_prior_pose[2]))) * odom_prior_yaw_weight,
                            ],
                            dtype=np.float64,
                        )
                    )
                return np.concatenate(residuals)

            lower_bounds = np.asarray(
                [
                    reference_pose[0] - 0.75,
                    reference_pose[1] - 0.75,
                    reference_pose[2] - 0.60,
                ],
                dtype=np.float64,
            )
            upper_bounds = np.asarray(
                [
                    reference_pose[0] + 0.75,
                    reference_pose[1] + 0.75,
                    reference_pose[2] + 0.60,
                ],
                dtype=np.float64,
            )
            initial_guess = np.asarray(
                [
                    float(np.clip(reference_pose[0], lower_bounds[0] + 1.0e-6, upper_bounds[0] - 1.0e-6)),
                    float(np.clip(reference_pose[1], lower_bounds[1] + 1.0e-6, upper_bounds[1] - 1.0e-6)),
                    float(np.clip(reference_pose[2], lower_bounds[2] + 1.0e-6, upper_bounds[2] - 1.0e-6)),
                ],
                dtype=np.float64,
            )
            solution = least_squares(
                residual_vector,
                x0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                loss="soft_l1",
                f_scale=0.08,
                max_nfev=30,
            )
            candidate_pose = solution.x.astype(np.float64, copy=False) if solution.success else reference_pose
            inlier_count, median_residual_m = _evaluate_pose_quality(
                current_scan.sampled_points_local,
                candidate_pose,
                submap_points_xy,
                max_correspondence_m,
            )
            confidence = "high"
            if (not solution.success) or inlier_count < 14 or (
                math.isfinite(median_residual_m) and median_residual_m > 0.18
            ):
                confidence = "low"
                candidate_pose = reference_pose
                inlier_count, median_residual_m = _evaluate_pose_quality(
                    current_scan.sampled_points_local,
                    candidate_pose,
                    submap_points_xy,
                    max_correspondence_m,
                )
            translation_update = float(np.linalg.norm(candidate_pose[:2] - refined_poses[local_index, :2]))
            yaw_update = abs(_wrap_angle(float(candidate_pose[2] - refined_poses[local_index, 2])))
            max_translation_update = max(max_translation_update, translation_update)
            max_yaw_update = max(max_yaw_update, yaw_update)
            refined_poses[local_index] = candidate_pose
            window_qualities[local_index] = ScanQuality(
                confidence=confidence,
                inlier_count=inlier_count,
                median_submap_residual_m=median_residual_m,
                source=f"window_{window_id}",
            )
        if max_translation_update < 0.01 and max_yaw_update < 0.01:
            break

    accumulated_points_xy = _collect_world_points(
        scans,
        global_poses_xyyaw,
        range(0, window_start),
        use_sampled=True,
        confidence=None,
    )
    window_points_xy = np.vstack(
        [
            _transform_points(scans[global_index].sampled_points_local, refined_poses[local_index])
            for local_index, global_index in enumerate(window_indexes)
            if scans[global_index].sampled_points_local.size
        ]
    ) if any(scans[index].sampled_points_local.size for index in window_indexes) else np.empty((0, 2), dtype=np.float64)
    accumulated_points_xy = _voxel_downsample(accumulated_points_xy, submap_voxel_size_m)
    window_points_xy = _voxel_downsample(window_points_xy, submap_voxel_size_m)
    applied_correction = np.zeros(3, dtype=np.float64)
    if window_start > 0 and accumulated_points_xy.shape[0] >= 20 and window_points_xy.shape[0] >= 20:
        applied_correction = _estimate_global_correction(
            window_points_xy,
            accumulated_points_xy,
            max_correspondence_m=window_registration_max_correspondence_m,
        )
        if np.linalg.norm(applied_correction[:2]) > 1.0e-6 or abs(float(applied_correction[2])) > 1.0e-6:
            last_index = max(1, window_count - 1)
            for local_index in range(window_count):
                blend = float(local_index / last_index)
                refined_poses[local_index] = _compose_poses(
                    _scaled_pose_delta(applied_correction, blend),
                    refined_poses[local_index],
                )

    median_residuals = [
        quality.median_submap_residual_m
        for quality in window_qualities
        if math.isfinite(quality.median_submap_residual_m)
    ]
    inlier_counts = [quality.inlier_count for quality in window_qualities]
    metrics = {
        "window_id": int(window_id),
        "start_scan_index": int(window_start),
        "end_scan_index": int(window_end),
        "overlap_count": int(window_overlap),
        "refined_scan_count": int(window_count),
        "low_confidence_scan_count": int(sum(quality.confidence == "low" for quality in window_qualities)),
        "median_residual_m": float(np.median(median_residuals)) if median_residuals else float("nan"),
        "median_inlier_count": float(np.median(np.asarray(inlier_counts, dtype=np.float64))) if inlier_counts else 0.0,
        "submap_point_count": int(window_points_xy.shape[0]),
        "applied_window_correction_xyyaw": [
            float(applied_correction[0]),
            float(applied_correction[1]),
            float(applied_correction[2]),
        ],
    }
    return refined_poses, window_qualities, metrics


def _build_reconstructed_map_points(
    scans: list[ScanRecord],
    poses_xyyaw: np.ndarray,
    *,
    confidence: list[str],
    map_voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[tuple[int, int], list[float]] = {}
    hit_counts: dict[tuple[int, int], int] = defaultdict(int)
    for scan_index, scan in enumerate(scans):
        if confidence[scan_index] != "high" or scan.points_local.size == 0:
            continue
        world_points = _transform_points(scan.points_local, poses_xyyaw[scan_index])
        for point in world_points:
            key = (
                int(math.floor(float(point[0]) / map_voxel_size_m)),
                int(math.floor(float(point[1]) / map_voxel_size_m)),
            )
            if key not in buckets:
                buckets[key] = [0.0, 0.0]
            buckets[key][0] += float(point[0])
            buckets[key][1] += float(point[1])
            hit_counts[key] += 1
    if not buckets:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    keys = sorted(buckets)
    points_xy = np.asarray(
        [
            [
                buckets[key][0] / hit_counts[key],
                buckets[key][1] / hit_counts[key],
            ]
            for key in keys
        ],
        dtype=np.float64,
    )
    counts = np.asarray([hit_counts[key] for key in keys], dtype=np.int64)
    return points_xy, counts


def _compute_path_error_metrics(
    estimate_times_s: np.ndarray,
    estimate_poses_xyyaw: np.ndarray,
    reference_times_s: np.ndarray | None,
    reference_poses_xyyaw: np.ndarray | None,
) -> dict[str, float | int] | None:
    if reference_times_s is None or reference_poses_xyyaw is None:
        return None
    if reference_times_s.size < 2 or estimate_times_s.size == 0:
        return None
    valid_mask = (estimate_times_s >= float(reference_times_s[0])) & (
        estimate_times_s <= float(reference_times_s[-1])
    )
    if int(np.count_nonzero(valid_mask)) < 2:
        return None
    matched_times = estimate_times_s[valid_mask]
    ref_x = np.interp(matched_times, reference_times_s, reference_poses_xyyaw[:, 0])
    ref_y = np.interp(matched_times, reference_times_s, reference_poses_xyyaw[:, 1])
    reference_matched_xy = np.column_stack((ref_x, ref_y))
    estimate_matched_xy = estimate_poses_xyyaw[valid_mask, :2]
    errors_m = np.linalg.norm(estimate_matched_xy - reference_matched_xy, axis=1)
    return {
        "position_rmse_m": float(math.sqrt(float(np.mean(np.square(errors_m))))),
        "position_mae_m": float(np.mean(errors_m)),
        "position_max_m": float(np.max(errors_m)),
        "endpoint_error_m": float(errors_m[-1]),
        "matched_sample_count": int(matched_times.size),
        "matched_time_span_s": float(matched_times[-1] - matched_times[0]),
        "reconstructed_length_on_gt_interval_m": float(_path_length_m(estimate_matched_xy)),
        "ground_truth_length_m": float(_path_length_m(reference_poses_xyyaw[:, :2])),
        "length_ratio_vs_ground_truth": float(
            _path_length_m(estimate_matched_xy) / max(_path_length_m(reference_poses_xyyaw[:, :2]), 1.0e-9)
        ),
    }


def _compute_map_similarity_metrics(
    reference_xy: np.ndarray | None,
    estimate_xy: np.ndarray | None,
    *,
    threshold_m: float,
) -> dict[str, float | int] | None:
    if reference_xy is None or estimate_xy is None:
        return None
    if reference_xy.shape[0] == 0 or estimate_xy.shape[0] == 0:
        return None
    reference_tree = cKDTree(reference_xy)
    estimate_tree = cKDTree(estimate_xy)
    est_to_ref_m, _ = reference_tree.query(estimate_xy, k=1)
    ref_to_est_m, _ = estimate_tree.query(reference_xy, k=1)
    return {
        "chamfer_m": float(np.mean(est_to_ref_m) + np.mean(ref_to_est_m)),
        "mean_estimate_to_track_m": float(np.mean(est_to_ref_m)),
        "mean_track_to_estimate_m": float(np.mean(ref_to_est_m)),
        "track_coverage_ratio": float(np.count_nonzero(ref_to_est_m <= threshold_m) / reference_xy.shape[0]),
        "estimate_precision_ratio": float(np.count_nonzero(est_to_ref_m <= threshold_m) / estimate_xy.shape[0]),
        "reference_point_count": int(reference_xy.shape[0]),
        "estimate_point_count": int(estimate_xy.shape[0]),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_reconstructed_path_csv(
    output_path: Path,
    scans: list[ScanRecord],
    poses_xyyaw: np.ndarray,
    *,
    window_ids: list[int],
    confidence: list[str],
    inlier_counts: list[int],
    median_residuals_m: list[float],
    sources: list[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_index",
                "stamp_sec",
                "stamp_nanosec",
                "x_m",
                "y_m",
                "yaw_rad",
                "window_id",
                "confidence",
                "inlier_count",
                "median_submap_residual_m",
                "source",
            ]
        )
        for scan, pose, window_id, conf, inliers, residual_m, source in zip(
            scans,
            poses_xyyaw,
            window_ids,
            confidence,
            inlier_counts,
            median_residuals_m,
            sources,
        ):
            writer.writerow(
                [
                    int(scan.scan_index),
                    int(scan.stamp_sec),
                    int(scan.stamp_nanosec),
                    f"{float(pose[0]):.9f}",
                    f"{float(pose[1]):.9f}",
                    f"{float(pose[2]):.9f}",
                    int(window_id),
                    conf,
                    int(inliers),
                    "" if not math.isfinite(residual_m) else f"{float(residual_m):.9f}",
                    source,
                ]
            )


def _write_reconstructed_map_csv(output_path: Path, points_xy: np.ndarray, hit_counts: np.ndarray) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "hit_count"])
        for point, hit_count in zip(points_xy, hit_counts):
            writer.writerow(
                [
                    f"{float(point[0]):.9f}",
                    f"{float(point[1]):.9f}",
                    int(hit_count),
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline lap reconstruction from imu_raw.csv + lidar_points.csv",
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing the recorded dataset.")
    parser.add_argument("--output-subdir", default="offline_reconstruction")
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--window-scan-count", type=int, default=48)
    parser.add_argument("--window-overlap-count", type=int, default=16)
    parser.add_argument("--submap-history-count", type=int, default=10)
    parser.add_argument("--submap-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--map-voxel-size-m", type=float, default=0.04)
    parser.add_argument("--max-correspondence-m", type=float, default=0.35)
    parser.add_argument("--window-registration-max-correspondence-m", type=float, default=0.40)
    parser.add_argument("--imu-median-window", type=int, default=5)
    parser.add_argument("--imu-ema-alpha", type=float, default=0.25)
    parser.add_argument("--static-window-s", type=float, default=0.4)
    parser.add_argument("--static-search-s", type=float, default=2.0)
    parser.add_argument("--lidar-weight", type=float, default=1.0)
    parser.add_argument("--motion-prior-weight", type=float, default=0.18)
    parser.add_argument("--yaw-prior-weight", type=float, default=0.45)
    parser.add_argument("--odom-prior-xy-weight", type=float, default=0.10)
    parser.add_argument("--odom-prior-yaw-weight", type=float, default=0.18)
    parser.add_argument("--map-score-threshold-m", type=float, default=0.12)
    parser.add_argument("--use-odom-prior", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.point_stride < 1:
        raise SystemExit("--point-stride must be >= 1")
    if args.window_scan_count < 8:
        raise SystemExit("--window-scan-count must be >= 8")
    if not 0 <= args.window_overlap_count < args.window_scan_count:
        raise SystemExit("--window-overlap-count must satisfy 0 <= overlap < window_scan_count")
    if args.submap_history_count < 2:
        raise SystemExit("--submap-history-count must be >= 2")
    if args.imu_median_window < 1 or (args.imu_median_window % 2) == 0:
        raise SystemExit("--imu-median-window must be odd and >= 1")
    if not 0.0 < args.imu_ema_alpha <= 1.0:
        raise SystemExit("--imu-ema-alpha must be in the range (0, 1]")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    output_dir = (run_dir / args.output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time_s = time.perf_counter()
    warnings: list[str] = []

    imu = _load_imu(run_dir)
    scans = _load_scans(run_dir, point_stride=args.point_stride)
    odom_series = _load_odom(run_dir)
    world_to_estimation = _load_world_to_estimation(run_dir)

    processed_imu = _process_imu(
        imu,
        median_window=args.imu_median_window,
        ema_alpha=args.imu_ema_alpha,
        static_window_s=args.static_window_s,
        static_search_s=args.static_search_s,
    )

    initial_poses_xyyaw, reference_deltas, yaw_prior_rad = _build_initial_pose_chain(
        scans,
        processed_imu,
        max_correspondence_m=args.max_correspondence_m,
        submap_voxel_size_m=args.submap_voxel_size_m,
    )
    refined_poses_xyyaw = initial_poses_xyyaw.copy()
    window_ids = [-1 for _ in scans]
    confidence = ["high" for _ in scans]
    inlier_counts = [relative.valid_match_count for relative in reference_deltas]
    median_residuals_m = [relative.median_residual_m for relative in reference_deltas]
    sources = ["initial_chain" for _ in scans]
    window_metrics: list[dict[str, float | int | list[float]]] = []

    step = args.window_scan_count - args.window_overlap_count
    window_id = 0
    for window_start in range(0, len(scans), step):
        window_end = min(len(scans) - 1, window_start + args.window_scan_count - 1)
        window_begin_s = time.perf_counter()
        window_refined, window_qualities, metrics = _refine_window(
            scans,
            refined_poses_xyyaw,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
            window_overlap_count=args.window_overlap_count,
            submap_history_count=args.submap_history_count,
            max_correspondence_m=args.max_correspondence_m,
            window_registration_max_correspondence_m=args.window_registration_max_correspondence_m,
            lidar_weight=args.lidar_weight,
            motion_prior_weight=args.motion_prior_weight,
            yaw_prior_weight=args.yaw_prior_weight,
            odom_prior_xy_weight=args.odom_prior_xy_weight,
            odom_prior_yaw_weight=args.odom_prior_yaw_weight,
            use_odom_prior=bool(args.use_odom_prior),
            yaw_prior_rad=yaw_prior_rad,
            reference_deltas=reference_deltas,
            odom_series=odom_series,
            submap_voxel_size_m=args.submap_voxel_size_m,
        )
        for local_index, global_index in enumerate(range(window_start, window_end + 1)):
            refined_poses_xyyaw[global_index] = window_refined[local_index]
            window_ids[global_index] = window_id
            confidence[global_index] = window_qualities[local_index].confidence
            inlier_counts[global_index] = window_qualities[local_index].inlier_count
            median_residuals_m[global_index] = window_qualities[local_index].median_submap_residual_m
            sources[global_index] = window_qualities[local_index].source
        metrics["duration_ms"] = float(1000.0 * (time.perf_counter() - window_begin_s))
        window_metrics.append(metrics)
        window_id += 1
        if window_end >= (len(scans) - 1):
            break

    reconstructed_map_points_xy, reconstructed_map_hit_counts = _build_reconstructed_map_points(
        scans,
        refined_poses_xyyaw,
        confidence=confidence,
        map_voxel_size_m=args.map_voxel_size_m,
    )

    reconstructed_path_csv = output_dir / "reconstructed_path.csv"
    reconstructed_map_csv = output_dir / "reconstructed_map_points.csv"
    window_metrics_json = output_dir / "window_metrics.json"
    reconstruction_summary_json = output_dir / "reconstruction_summary.json"

    _write_reconstructed_path_csv(
        reconstructed_path_csv,
        scans,
        refined_poses_xyyaw,
        window_ids=window_ids,
        confidence=confidence,
        inlier_counts=inlier_counts,
        median_residuals_m=median_residuals_m,
        sources=sources,
    )
    _write_reconstructed_map_csv(
        reconstructed_map_csv,
        reconstructed_map_points_xy,
        reconstructed_map_hit_counts,
    )
    window_metrics_json.write_text(json.dumps(window_metrics, indent=2, default=_json_default), encoding="utf-8")

    scan_times_s = np.asarray([scan.stamp_s for scan in scans], dtype=np.float64)
    ground_truth_world_poses, ground_truth_times_s = _load_path_with_yaw(run_dir)
    ground_truth_estimation_poses = None
    if ground_truth_world_poses is not None:
        if world_to_estimation is not None:
            ground_truth_estimation_xy = world_to_estimation.transform_points(ground_truth_world_poses[:, :2])
            ground_truth_estimation_poses = np.column_stack(
                (
                    ground_truth_estimation_xy,
                    np.asarray(
                        [_wrap_angle(float(yaw + world_to_estimation.yaw_rad)) for yaw in ground_truth_world_poses[:, 2]],
                        dtype=np.float64,
                    ),
                )
            )
        else:
            warnings.append("ground_truth_status.jsonl missing or invalid; path/map evaluation kept in world frame is unavailable")
    else:
        warnings.append("ground_truth_path.csv missing; trajectory scoring unavailable")

    track_geometry_world_xy = _load_track_geometry(run_dir)
    track_geometry_estimation_xy = None
    if track_geometry_world_xy is not None:
        if world_to_estimation is not None:
            track_geometry_estimation_xy = world_to_estimation.transform_points(track_geometry_world_xy)
        else:
            warnings.append("track_geometry.csv available but ground_truth_status.jsonl missing; map similarity unavailable")
    else:
        warnings.append("track_geometry.csv missing; map similarity unavailable")

    path_error_metrics = _compute_path_error_metrics(
        scan_times_s,
        refined_poses_xyyaw,
        ground_truth_times_s,
        ground_truth_estimation_poses,
    )
    if path_error_metrics is None:
        warnings.append("trajectory scoring against ground_truth_path.csv could not be computed")

    map_similarity_metrics = _compute_map_similarity_metrics(
        track_geometry_estimation_xy,
        reconstructed_map_points_xy,
        threshold_m=args.map_score_threshold_m,
    )
    if map_similarity_metrics is None:
        warnings.append("map similarity against track_geometry.csv could not be computed")

    baseline_odom_summary: dict[str, object] | None = None
    if odom_series is not None:
        baseline_path_metrics = _compute_path_error_metrics(
            odom_series.t_s,
            odom_series.pose_xyyaw,
            ground_truth_times_s,
            ground_truth_estimation_poses,
        )
        baseline_odom_summary = {
            "available": True,
            "sample_count": int(odom_series.t_s.size),
            "path_length_m": float(_path_length_m(odom_series.pose_xyyaw[:, :2])),
            "closure_gap_m": float(
                np.linalg.norm(odom_series.pose_xyyaw[-1, :2] - odom_series.pose_xyyaw[0, :2])
            ),
            "closure_yaw_gap_rad": float(
                abs(_wrap_angle(float(odom_series.pose_xyyaw[-1, 2] - odom_series.pose_xyyaw[0, 2])))
            ),
            "vs_ground_truth_path": baseline_path_metrics,
        }
    else:
        warnings.append("odom_fused.csv missing; odometry baseline unavailable")
        baseline_odom_summary = {"available": False}

    gt_closure_gap_m = None
    gt_path_length_m = None
    if ground_truth_estimation_poses is not None:
        gt_closure_gap_m = float(
            np.linalg.norm(ground_truth_estimation_poses[-1, :2] - ground_truth_estimation_poses[0, :2])
        )
        gt_path_length_m = float(_path_length_m(ground_truth_estimation_poses[:, :2]))

    reconstruction_metrics = {
        "scan_count": int(len(scans)),
        "window_count": int(len(window_metrics)),
        "high_confidence_scan_count": int(sum(conf == "high" for conf in confidence)),
        "low_confidence_scan_count": int(sum(conf == "low" for conf in confidence)),
        "reconstructed_length_m": float(_path_length_m(refined_poses_xyyaw[:, :2])),
        "closure_gap_m": float(np.linalg.norm(refined_poses_xyyaw[-1, :2] - refined_poses_xyyaw[0, :2])),
        "closure_yaw_gap_rad": float(abs(_wrap_angle(float(refined_poses_xyyaw[-1, 2] - refined_poses_xyyaw[0, 2])))),
        "map_point_count": int(reconstructed_map_points_xy.shape[0]),
        "duration_s": float(time.perf_counter() - start_time_s),
    }
    evaluation_summary = {
        "path_vs_ground_truth": path_error_metrics,
        "map_vs_track_geometry": map_similarity_metrics,
        "ground_truth_length_m": gt_path_length_m,
        "ground_truth_closure_gap_m": gt_closure_gap_m,
    }
    input_summary = {
        "run_dir": str(run_dir),
        "scan_count": int(len(scans)),
        "imu_sample_count": int(imu.t_s.size),
        "odom_sample_count": int(odom_series.t_s.size) if odom_series is not None else 0,
        "ground_truth_path_sample_count": int(ground_truth_times_s.size) if ground_truth_times_s is not None else 0,
        "track_geometry_point_count": int(track_geometry_world_xy.shape[0]) if track_geometry_world_xy is not None else 0,
        "world_to_estimation": (
            {
                "tx_m": float(world_to_estimation.tx_m),
                "ty_m": float(world_to_estimation.ty_m),
                "yaw_rad": float(world_to_estimation.yaw_rad),
                "samples_used": int(world_to_estimation.samples_used),
                "max_translation_drift_m": float(world_to_estimation.max_translation_drift_m),
                "max_yaw_drift_rad": float(world_to_estimation.max_yaw_drift_rad),
            }
            if world_to_estimation is not None
            else None
        ),
    }
    parameters_summary = {
        "point_stride": int(args.point_stride),
        "window_scan_count": int(args.window_scan_count),
        "window_overlap_count": int(args.window_overlap_count),
        "submap_history_count": int(args.submap_history_count),
        "submap_voxel_size_m": float(args.submap_voxel_size_m),
        "map_voxel_size_m": float(args.map_voxel_size_m),
        "max_correspondence_m": float(args.max_correspondence_m),
        "window_registration_max_correspondence_m": float(args.window_registration_max_correspondence_m),
        "imu_median_window": int(args.imu_median_window),
        "imu_ema_alpha": float(args.imu_ema_alpha),
        "static_window_s": float(args.static_window_s),
        "static_search_s": float(args.static_search_s),
        "lidar_weight": float(args.lidar_weight),
        "motion_prior_weight": float(args.motion_prior_weight),
        "yaw_prior_weight": float(args.yaw_prior_weight),
        "odom_prior_xy_weight": float(args.odom_prior_xy_weight),
        "odom_prior_yaw_weight": float(args.odom_prior_yaw_weight),
        "use_odom_prior": bool(args.use_odom_prior),
        "map_score_threshold_m": float(args.map_score_threshold_m),
        "imu_biases": {
            "ax_mps2": float(processed_imu.bias_ax_mps2),
            "ay_mps2": float(processed_imu.bias_ay_mps2),
            "az_minus_g_mps2": float(processed_imu.bias_az_mps2),
            "gz_rps": float(processed_imu.bias_gz_rps),
        },
        "imu_static_initialization": {
            "static_start_s": float(processed_imu.static_start_s),
            "static_end_s": float(processed_imu.static_end_s),
            "static_sample_count": int(processed_imu.static_sample_count),
            "best_effort_init": bool(processed_imu.best_effort_init),
        },
    }
    reconstruction_summary = {
        "input_summary": input_summary,
        "parameters": parameters_summary,
        "reconstruction_metrics": reconstruction_metrics,
        "evaluation": evaluation_summary,
        "baseline_odom": baseline_odom_summary,
        "warnings": warnings,
    }
    reconstruction_summary_json.write_text(
        json.dumps(reconstruction_summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"run_dir: {run_dir}")
    print(f"output_dir: {output_dir}")
    print(f"scan_count: {len(scans)}")
    print(f"window_count: {len(window_metrics)}")
    print(f"high_confidence_scan_count: {reconstruction_metrics['high_confidence_scan_count']}")
    print(f"low_confidence_scan_count: {reconstruction_metrics['low_confidence_scan_count']}")
    print(f"reconstructed_length_m: {reconstruction_metrics['reconstructed_length_m']:.6f}")
    print(f"closure_gap_m: {reconstruction_metrics['closure_gap_m']:.6f}")
    if path_error_metrics is not None:
        print(f"path_rmse_m: {path_error_metrics['position_rmse_m']:.6f}")
    if map_similarity_metrics is not None:
        print(f"map_chamfer_m: {map_similarity_metrics['chamfer_m']:.6f}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
