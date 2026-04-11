#!/usr/bin/env python3
"""Build a general offline fixed track map with pose-graph optimization.

This mapper is designed for curved tracks. It operates only on reduced
keyframes, uses scan-to-submap matching on multiresolution distance fields,
verifies loop closures with local geometric refinement, and evaluates the final
map against the Gazebo world geometry as an oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    convolve,
    distance_transform_edt,
    generate_binary_structure,
    label,
)
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _write_status(status_path: Path | None, payload: dict[str, object]) -> None:
    if status_path is None:
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_duration_s(duration_s: float) -> str:
    total_seconds = int(max(0.0, round(float(duration_s))))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    return np.asarray(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=np.float64,
    )


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(pose_xyyaw[2])).T) + pose_xyyaw[:2]


def _apply_rigid_transform(points_xy: np.ndarray, yaw_rad: float, translation_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(yaw_rad)).T) + translation_xy[:2]


def _compose_pose(base_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    rotation = _rotation_matrix(float(base_pose[2]))
    translated = base_pose[:2] + (rotation @ np.asarray(delta_pose[:2], dtype=np.float64))
    return np.asarray(
        [
            float(translated[0]),
            float(translated[1]),
            _normalize_angle(float(base_pose[2]) + float(delta_pose[2])),
        ],
        dtype=np.float64,
    )


def _relative_pose(source_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    delta_world = np.asarray(target_pose[:2], dtype=np.float64) - np.asarray(source_pose[:2], dtype=np.float64)
    local_translation = _rotation_matrix(-float(source_pose[2])) @ delta_world
    return np.asarray(
        [
            float(local_translation[0]),
            float(local_translation[1]),
            _normalize_angle(float(target_pose[2]) - float(source_pose[2])),
        ],
        dtype=np.float64,
    )


def _relative_pose_error(source_pose: np.ndarray, target_pose: np.ndarray, measured_delta: np.ndarray) -> np.ndarray:
    current_delta = _relative_pose(source_pose, target_pose)
    error = current_delta - measured_delta
    error[2] = _normalize_angle(float(error[2]))
    return error


def _invert_relative_pose(delta_pose: np.ndarray) -> np.ndarray:
    rotation = _rotation_matrix(-float(delta_pose[2]))
    inv_translation = -(rotation @ np.asarray(delta_pose[:2], dtype=np.float64))
    return np.asarray(
        [
            float(inv_translation[0]),
            float(inv_translation[1]),
            _normalize_angle(-float(delta_pose[2])),
        ],
        dtype=np.float64,
    )


def _subsample_evenly(points_xy: np.ndarray, max_points: int) -> np.ndarray:
    if points_xy.size == 0 or points_xy.shape[0] <= max_points:
        return points_xy.copy()
    indexes = np.linspace(0, points_xy.shape[0] - 1, num=max_points, dtype=np.int32)
    return points_xy[indexes]


def _voxel_downsample(points_xy: np.ndarray, voxel_size_m: float, max_points: int) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if voxel_size_m > 1.0e-6:
        grid = np.floor(points_xy / voxel_size_m).astype(np.int64)
        _, unique_indexes = np.unique(grid, axis=0, return_index=True)
        unique_indexes.sort()
        points_xy = points_xy[unique_indexes]
    return _subsample_evenly(points_xy, max_points)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rigid_transform_from_correspondences(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray]:
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return 0.0, np.zeros(2, dtype=np.float64)
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid
    covariance = centered_source.T @ centered_target
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    rotation = vh_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vh_mat[-1, :] *= -1.0
        rotation = vh_mat.T @ u_mat.T
    yaw_rad = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    translation = target_centroid - (source_centroid @ rotation.T)
    return yaw_rad, translation.astype(np.float64)


@dataclass(frozen=True)
class OdomPriorSample:
    t_s: float
    pose_xyyaw: np.ndarray
    velocity_xy: np.ndarray
    yaw_rate_rps: float
    quality_label: str
    quality_scale_xy: float
    quality_scale_yaw: float


@dataclass(frozen=True)
class ImuSample:
    t_s: float
    accel_body_xy: np.ndarray
    accel_norm_mps2: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class ScanFrame:
    scan_index: int
    t_s: float
    points_local: np.ndarray
    prior_pose_xyyaw: np.ndarray
    prior_velocity_xy: np.ndarray
    prior_yaw_rate_rps: float
    prior_quality_label: str
    prior_quality_scale_xy: float
    prior_quality_scale_yaw: float


@dataclass(frozen=True)
class Keyframe:
    keyframe_index: int
    scan_index: int
    t_s: float
    points_local: np.ndarray
    prior_pose_xyyaw: np.ndarray
    prior_velocity_xy: np.ndarray
    prior_yaw_rate_rps: float
    prior_quality_label: str
    prior_quality_scale_xy: float
    prior_quality_scale_yaw: float
    descriptor_polar: np.ndarray
    descriptor_angular: np.ndarray
    descriptor_scan_context: np.ndarray
    descriptor_ring_key: np.ndarray


@dataclass(frozen=True)
class PoseGraphEdge:
    edge_type: str
    source_index: int
    target_index: int
    measured_delta_xyyaw: np.ndarray
    weight_xy: float
    weight_yaw: float
    residual_m: float
    score: float = 0.0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "edge_type": self.edge_type,
            "source_index": int(self.source_index),
            "target_index": int(self.target_index),
            "measured_delta_xyyaw": [float(v) for v in self.measured_delta_xyyaw.tolist()],
            "weight_xy": float(self.weight_xy),
            "weight_yaw": float(self.weight_yaw),
            "residual_m": float(self.residual_m),
            "score": float(self.score),
        }


@dataclass(frozen=True)
class DistanceFieldLevel:
    resolution_m: float
    origin_xy: np.ndarray
    distance_field_m: np.ndarray
    occupancy: np.ndarray
    max_distance_m: float


@dataclass(frozen=True)
class SubmapDescriptorEntry:
    center_index: int
    local_points_xy: np.ndarray
    bev_descriptor: np.ndarray
    bev_key: np.ndarray


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    radius = max(0, int(window) // 2)
    output = np.zeros_like(values, dtype=np.float64)
    for index in range(values.shape[0]):
        start = max(0, index - radius)
        end = min(values.shape[0], index + radius + 1)
        output[index] = float(np.median(values[start:end]))
    return output

def _median_abs_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _safe_direction(delta_xy: np.ndarray, fallback_yaw_rad: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta_xy))
    if norm > 1.0e-9:
        return delta_xy / norm
    return np.asarray([math.cos(fallback_yaw_rad), math.sin(fallback_yaw_rad)], dtype=np.float64)


def _blend_angles(source_yaw_rad: float, target_yaw_rad: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    delta = _normalize_angle(float(target_yaw_rad) - float(source_yaw_rad))
    return _normalize_angle(float(source_yaw_rad) + (alpha * delta))


def _quality_from_scores(
    *,
    accel_score: float,
    yaw_accel_score: float,
    imu_accel_score: float,
    imu_gyro_score: float,
    raw_accel_mps2: float,
    raw_yaw_accel_rps2: float,
) -> tuple[str, float, float, float]:
    max_score = max(accel_score, yaw_accel_score, imu_accel_score, imu_gyro_score)
    severe = (
        abs(raw_accel_mps2) > 4.5
        or abs(raw_yaw_accel_rps2) > 6.0
        or max_score > 6.0
    )
    reduced = (
        abs(raw_accel_mps2) > 2.6
        or abs(raw_yaw_accel_rps2) > 3.2
        or max_score > 3.0
    )
    if severe:
        return ("low", 0.30, 0.45, 0.85)
    if reduced:
        return ("reduced", 0.60, 0.75, 0.55)
    return ("high", 1.00, 1.00, 0.20)


def _load_imu_samples(path: Path) -> list[ImuSample]:
    if not path.exists():
        return []
    rows: list[ImuSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                t_s = float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"]))
                accel_body_xy = np.asarray(
                    [
                        float(row.get("ax_mps2", 0.0)),
                        float(row.get("ay_mps2", 0.0)),
                    ],
                    dtype=np.float64,
                )
                yaw_rate_rps = float(row.get("gz_rps", 0.0))
            except Exception:
                continue
            rows.append(
                ImuSample(
                    t_s=t_s,
                    accel_body_xy=accel_body_xy,
                    accel_norm_mps2=float(np.linalg.norm(accel_body_xy)),
                    yaw_rate_rps=yaw_rate_rps,
                )
            )
    return rows


def _load_odom_prior(path: Path) -> list[OdomPriorSample]:
    if not path.exists():
        return []
    rows: list[OdomPriorSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                t_s = float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"]))
                pose = np.asarray(
                    [
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["yaw_rad"]),
                    ],
                    dtype=np.float64,
                )
                velocity = np.asarray(
                    [
                        float(row.get("vx_mps", 0.0)),
                        float(row.get("vy_mps", 0.0)),
                    ],
                    dtype=np.float64,
                )
                yaw_rate = float(row.get("yaw_rate_rps", 0.0))
            except Exception:
                continue
            rows.append(
                OdomPriorSample(
                    t_s=t_s,
                    pose_xyyaw=pose,
                    velocity_xy=velocity,
                    yaw_rate_rps=yaw_rate,
                    quality_label="high",
                    quality_scale_xy=1.0,
                    quality_scale_yaw=1.0,
                )
            )
    if not rows:
        return []
    origin_pose = rows[0].pose_xyyaw
    normalized_rows: list[OdomPriorSample] = []
    for sample in rows:
        normalized_rows.append(
            OdomPriorSample(
                t_s=sample.t_s,
                pose_xyyaw=_relative_pose(origin_pose, sample.pose_xyyaw),
                velocity_xy=sample.velocity_xy.copy(),
                yaw_rate_rps=sample.yaw_rate_rps,
                quality_label=sample.quality_label,
                quality_scale_xy=sample.quality_scale_xy,
                quality_scale_yaw=sample.quality_scale_yaw,
            )
        )
    return normalized_rows


def _associate_imu_for_time(samples: list[ImuSample], t_s: float) -> ImuSample | None:
    if not samples:
        return None
    times = np.asarray([sample.t_s for sample in samples], dtype=np.float64)
    insert_index = int(np.searchsorted(times, t_s, side="left"))
    candidate_indexes: list[int] = []
    if insert_index < len(samples):
        candidate_indexes.append(insert_index)
    if insert_index > 0:
        candidate_indexes.append(insert_index - 1)
    if not candidate_indexes:
        return samples[0]
    best_index = min(candidate_indexes, key=lambda idx: abs(times[idx] - t_s))
    return samples[best_index]


def _preprocess_odom_prior(
    odom_samples: list[OdomPriorSample],
    imu_samples: list[ImuSample],
) -> list[OdomPriorSample]:
    if len(odom_samples) <= 2:
        return odom_samples

    times = np.asarray([sample.t_s for sample in odom_samples], dtype=np.float64)
    raw_pose = np.vstack([sample.pose_xyyaw for sample in odom_samples]).astype(np.float64)
    raw_delta_xy = np.diff(raw_pose[:, :2], axis=0)
    raw_delta_yaw = np.asarray(
        [
            _normalize_angle(float(raw_pose[index + 1, 2] - raw_pose[index, 2]))
            for index in range(raw_pose.shape[0] - 1)
        ],
        dtype=np.float64,
    )
    dt_s = np.clip(np.diff(times), 1.0e-3, 0.25)
    raw_speed_mps = np.linalg.norm(raw_delta_xy, axis=1) / dt_s
    raw_yaw_rate_rps = raw_delta_yaw / dt_s
    filtered_speed_mps = _rolling_median(raw_speed_mps, 5)
    filtered_yaw_rate_rps = _rolling_median(raw_yaw_rate_rps, 5)

    raw_accel = np.diff(raw_speed_mps, prepend=raw_speed_mps[0]) / np.clip(
        np.concatenate(([dt_s[0]], dt_s[:-1])),
        1.0e-3,
        0.25,
    )
    raw_yaw_accel = np.diff(raw_yaw_rate_rps, prepend=raw_yaw_rate_rps[0]) / np.clip(
        np.concatenate(([dt_s[0]], dt_s[:-1])),
        1.0e-3,
        0.25,
    )
    accel_median = _rolling_median(raw_accel, 7)
    yaw_accel_median = _rolling_median(raw_yaw_accel, 7)
    accel_mad = max(0.12, 1.4826 * _median_abs_deviation(raw_accel))
    yaw_accel_mad = max(0.10, 1.4826 * _median_abs_deviation(raw_yaw_accel))

    imu_accel_norm = np.zeros((len(odom_samples) - 1,), dtype=np.float64)
    imu_yaw_rate = np.zeros((len(odom_samples) - 1,), dtype=np.float64)
    for index in range(1, len(odom_samples)):
        imu_sample = _associate_imu_for_time(imu_samples, float(times[index]))
        if imu_sample is None:
            continue
        imu_accel_norm[index - 1] = float(imu_sample.accel_norm_mps2)
        imu_yaw_rate[index - 1] = float(imu_sample.yaw_rate_rps)
    filtered_imu_accel_norm = _rolling_median(imu_accel_norm, 7)
    filtered_imu_yaw_rate = _rolling_median(imu_yaw_rate, 7)
    imu_accel_mad = max(0.15, 1.4826 * _median_abs_deviation(imu_accel_norm))
    imu_gyro_mad = max(0.05, 1.4826 * _median_abs_deviation(imu_yaw_rate))

    filtered_pose = np.zeros_like(raw_pose)
    filtered_pose[0] = raw_pose[0]
    filtered_velocity_world = np.zeros((len(odom_samples), 2), dtype=np.float64)
    filtered_yaw_rate = np.zeros((len(odom_samples),), dtype=np.float64)
    quality_labels = ["high"] * len(odom_samples)
    quality_scales_xy = np.ones((len(odom_samples),), dtype=np.float64)
    quality_scales_yaw = np.ones((len(odom_samples),), dtype=np.float64)

    previous_speed_mps = float(filtered_speed_mps[0]) if filtered_speed_mps.size > 0 else 0.0
    previous_yaw_rate_rps = float(filtered_yaw_rate_rps[0]) if filtered_yaw_rate_rps.size > 0 else 0.0
    max_linear_accel_mps2 = 2.8
    max_yaw_accel_rps2 = 3.8

    for index in range(1, len(odom_samples)):
        interval_index = index - 1
        delta_xy = raw_delta_xy[interval_index]
        dt = float(dt_s[interval_index])
        accel_score = abs(raw_accel[interval_index] - accel_median[interval_index]) / accel_mad
        yaw_accel_score = abs(raw_yaw_accel[interval_index] - yaw_accel_median[interval_index]) / yaw_accel_mad
        imu_accel_score = abs(imu_accel_norm[interval_index] - filtered_imu_accel_norm[interval_index]) / imu_accel_mad
        imu_gyro_score = abs(imu_yaw_rate[interval_index] - filtered_imu_yaw_rate[interval_index]) / imu_gyro_mad
        quality_label, quality_scale_xy, quality_scale_yaw, filter_alpha = _quality_from_scores(
            accel_score=float(accel_score),
            yaw_accel_score=float(yaw_accel_score),
            imu_accel_score=float(imu_accel_score),
            imu_gyro_score=float(imu_gyro_score),
            raw_accel_mps2=float(raw_accel[interval_index]),
            raw_yaw_accel_rps2=float(raw_yaw_accel[interval_index]),
        )

        target_speed_mps = (1.0 - filter_alpha) * float(raw_speed_mps[interval_index]) + (
            filter_alpha * float(filtered_speed_mps[interval_index])
        )
        target_speed_mps = np.clip(
            target_speed_mps,
            previous_speed_mps - (max_linear_accel_mps2 * dt),
            previous_speed_mps + (max_linear_accel_mps2 * dt),
        )
        target_yaw_rate_rps = (0.65 * float(filtered_yaw_rate_rps[interval_index])) + (
            0.35 * float(filtered_imu_yaw_rate[interval_index])
        )
        target_yaw_rate_rps = np.clip(
            target_yaw_rate_rps,
            previous_yaw_rate_rps - (max_yaw_accel_rps2 * dt),
            previous_yaw_rate_rps + (max_yaw_accel_rps2 * dt),
        )

        direction_xy = _safe_direction(delta_xy, float(filtered_pose[index - 1, 2]))
        filtered_delta_xy = direction_xy * (target_speed_mps * dt)
        filtered_delta_yaw = target_yaw_rate_rps * dt
        filtered_pose[index, :2] = filtered_pose[index - 1, :2] + filtered_delta_xy
        raw_next_yaw = float(raw_pose[index, 2])
        predicted_yaw = _normalize_angle(float(filtered_pose[index - 1, 2]) + float(filtered_delta_yaw))
        filtered_pose[index, 2] = _blend_angles(predicted_yaw, raw_next_yaw, 0.15 if quality_label == "high" else 0.05)
        filtered_velocity_world[index] = direction_xy * float(target_speed_mps)
        filtered_yaw_rate[index] = float(target_yaw_rate_rps)
        quality_labels[index] = quality_label
        quality_scales_xy[index] = float(quality_scale_xy)
        quality_scales_yaw[index] = float(quality_scale_yaw)
        previous_speed_mps = float(target_speed_mps)
        previous_yaw_rate_rps = float(target_yaw_rate_rps)

    filtered_rows: list[OdomPriorSample] = []
    for index, sample in enumerate(odom_samples):
        filtered_rows.append(
            OdomPriorSample(
                t_s=sample.t_s,
                pose_xyyaw=filtered_pose[index].copy(),
                velocity_xy=filtered_velocity_world[index].copy(),
                yaw_rate_rps=float(filtered_yaw_rate[index]),
                quality_label=quality_labels[index],
                quality_scale_xy=float(quality_scales_xy[index]),
                quality_scale_yaw=float(quality_scales_yaw[index]),
            )
        )
    return filtered_rows


def _load_lidar_scans(path: Path) -> list[tuple[int, float, np.ndarray]]:
    scans: dict[int, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scan_index = int(row["scan_index"])
                t_s = float(row["stamp_sec"]) + (1.0e-9 * float(row["stamp_nanosec"]))
                x_forward_m = float(row["x_forward_m"])
                y_left_m = float(row["y_left_m"])
            except Exception:
                continue
            payload = scans.setdefault(
                scan_index,
                {
                    "t_s": t_s,
                    "points": [],
                },
            )
            payload["points"].append((x_forward_m, y_left_m))
    ordered: list[tuple[int, float, np.ndarray]] = []
    for scan_index in sorted(scans):
        payload = scans[scan_index]
        points = np.asarray(payload["points"], dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        ordered.append((scan_index, float(payload["t_s"]), points))
    return ordered


def _associate_prior_for_time(samples: list[OdomPriorSample], t_s: float) -> OdomPriorSample | None:
    if not samples:
        return None
    times = np.asarray([sample.t_s for sample in samples], dtype=np.float64)
    insert_index = int(np.searchsorted(times, t_s, side="left"))
    candidate_indexes: list[int] = []
    if insert_index < len(samples):
        candidate_indexes.append(insert_index)
    if insert_index > 0:
        candidate_indexes.append(insert_index - 1)
    if not candidate_indexes:
        return samples[0]
    best_index = min(candidate_indexes, key=lambda idx: abs(times[idx] - t_s))
    return samples[best_index]


def _build_scan_frames(
    lidar_scans: list[tuple[int, float, np.ndarray]],
    odom_prior: list[OdomPriorSample],
) -> list[ScanFrame]:
    frames: list[ScanFrame] = []
    for ordinal_index, (scan_index, t_s, points_local) in enumerate(lidar_scans):
        prior_sample = _associate_prior_for_time(odom_prior, t_s)
        if prior_sample is None:
            prior_pose = np.asarray([0.05 * ordinal_index, 0.0, 0.0], dtype=np.float64)
            prior_velocity = np.asarray([0.05, 0.0], dtype=np.float64)
            prior_yaw_rate = 0.0
        else:
            prior_pose = prior_sample.pose_xyyaw.copy()
            prior_velocity = prior_sample.velocity_xy.copy()
            prior_yaw_rate = float(prior_sample.yaw_rate_rps)
            prior_quality_label = str(prior_sample.quality_label)
            prior_quality_scale_xy = float(prior_sample.quality_scale_xy)
            prior_quality_scale_yaw = float(prior_sample.quality_scale_yaw)
        frames.append(
            ScanFrame(
                scan_index=scan_index,
                t_s=t_s,
                points_local=points_local,
                prior_pose_xyyaw=prior_pose,
                prior_velocity_xy=prior_velocity,
                prior_yaw_rate_rps=prior_yaw_rate,
                prior_quality_label=prior_quality_label if prior_sample is not None else "high",
                prior_quality_scale_xy=prior_quality_scale_xy if prior_sample is not None else 1.0,
                prior_quality_scale_yaw=prior_quality_scale_yaw if prior_sample is not None else 1.0,
            )
        )
    return frames


def _compute_polar_descriptor(points_local: np.ndarray, bins: int = 72) -> np.ndarray:
    descriptor = np.zeros((bins,), dtype=np.float64)
    if points_local.size == 0:
        return descriptor
    angles = np.mod(np.arctan2(points_local[:, 1], points_local[:, 0]), 2.0 * math.pi)
    radii = np.linalg.norm(points_local, axis=1)
    weights = 1.0 / np.clip(0.20 + radii, 0.20, None)
    indexes = np.floor((angles / (2.0 * math.pi)) * bins).astype(np.int32)
    indexes = np.clip(indexes, 0, bins - 1)
    np.add.at(descriptor, indexes, weights)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1.0e-9:
        descriptor /= norm
    return descriptor


def _compute_angular_descriptor(points_local: np.ndarray, bins: int = 36) -> np.ndarray:
    descriptor = np.zeros((bins,), dtype=np.float64)
    if points_local.shape[0] < 2:
        return descriptor
    centroid = np.mean(points_local, axis=0)
    centered = points_local - centroid
    angles = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2.0 * math.pi)
    radii = np.linalg.norm(centered, axis=1)
    weights = np.clip(radii, 0.05, None)
    indexes = np.floor((angles / (2.0 * math.pi)) * bins).astype(np.int32)
    indexes = np.clip(indexes, 0, bins - 1)
    np.add.at(descriptor, indexes, weights)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1.0e-9:
        descriptor /= norm
    return descriptor


def _compute_scan_context_descriptor(
    points_local: np.ndarray,
    *,
    radial_bins: int = 12,
    angular_bins: int = 36,
    max_radius_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    descriptor = np.zeros((radial_bins, angular_bins), dtype=np.float64)
    if points_local.size == 0:
        return descriptor, np.zeros((radial_bins,), dtype=np.float64)
    radii = np.linalg.norm(points_local, axis=1)
    valid_mask = (radii >= 0.15) & (radii <= max_radius_m)
    if not np.any(valid_mask):
        return descriptor, np.zeros((radial_bins,), dtype=np.float64)
    filtered_points = points_local[valid_mask]
    filtered_radii = radii[valid_mask]
    filtered_angles = np.mod(np.arctan2(filtered_points[:, 1], filtered_points[:, 0]), 2.0 * math.pi)
    ring_indexes = np.floor((filtered_radii / max_radius_m) * radial_bins).astype(np.int32)
    sector_indexes = np.floor((filtered_angles / (2.0 * math.pi)) * angular_bins).astype(np.int32)
    ring_indexes = np.clip(ring_indexes, 0, radial_bins - 1)
    sector_indexes = np.clip(sector_indexes, 0, angular_bins - 1)
    weights = 1.0 / np.clip(0.25 + filtered_radii, 0.25, None)
    np.add.at(descriptor, (ring_indexes, sector_indexes), weights)
    descriptor = np.log1p(descriptor)
    descriptor_norm = float(np.linalg.norm(descriptor))
    if descriptor_norm > 1.0e-9:
        descriptor /= descriptor_norm
    ring_key = np.mean(descriptor, axis=1)
    ring_norm = float(np.linalg.norm(ring_key))
    if ring_norm > 1.0e-9:
        ring_key /= ring_norm
    return descriptor.astype(np.float64), ring_key.astype(np.float64)


def _best_scan_context_alignment(
    target_descriptor: np.ndarray,
    source_descriptor: np.ndarray,
) -> tuple[float, int]:
    if target_descriptor.size == 0 or source_descriptor.size == 0:
        return (0.0, 0)
    if target_descriptor.shape != source_descriptor.shape:
        return (0.0, 0)
    angular_bins = int(target_descriptor.shape[1])
    target_flat = target_descriptor.reshape(-1)
    target_norm = float(np.linalg.norm(target_flat))
    if target_norm <= 1.0e-9:
        return (0.0, 0)
    best_score = -1.0
    best_shift = 0
    for shift in range(angular_bins):
        shifted = np.roll(source_descriptor, shift=shift, axis=1)
        shifted_flat = shifted.reshape(-1)
        shifted_norm = float(np.linalg.norm(shifted_flat))
        if shifted_norm <= 1.0e-9:
            continue
        score = float(np.dot(target_flat, shifted_flat) / (target_norm * shifted_norm))
        if score > best_score:
            best_score = score
            best_shift = shift
    return (max(0.0, best_score), best_shift)


def _compute_bev_descriptor(
    points_local: np.ndarray,
    *,
    grid_size_m: float = 8.0,
    grid_cells: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    descriptor = np.zeros((grid_cells, grid_cells), dtype=np.float64)
    if points_local.size == 0:
        return descriptor, np.zeros((grid_cells,), dtype=np.float64)
    half_extent = 0.5 * float(grid_size_m)
    valid_mask = (
        (points_local[:, 0] >= -half_extent)
        & (points_local[:, 0] <= half_extent)
        & (points_local[:, 1] >= -half_extent)
        & (points_local[:, 1] <= half_extent)
    )
    if not np.any(valid_mask):
        return descriptor, np.zeros((grid_cells,), dtype=np.float64)
    filtered = points_local[valid_mask]
    cell_size = float(grid_size_m) / float(grid_cells)
    gx = np.floor((filtered[:, 0] + half_extent) / cell_size).astype(np.int32)
    gy = np.floor((filtered[:, 1] + half_extent) / cell_size).astype(np.int32)
    gx = np.clip(gx, 0, grid_cells - 1)
    gy = np.clip(gy, 0, grid_cells - 1)
    radii = np.linalg.norm(filtered, axis=1)
    weights = 1.0 / np.clip(0.35 + radii, 0.35, None)
    np.add.at(descriptor, (gy, gx), weights)
    descriptor = np.log1p(descriptor)
    descriptor_norm = float(np.linalg.norm(descriptor))
    if descriptor_norm > 1.0e-9:
        descriptor /= descriptor_norm
    bev_key = np.concatenate(
        (
            np.mean(descriptor, axis=0),
            np.mean(descriptor, axis=1),
        )
    )
    bev_key_norm = float(np.linalg.norm(bev_key))
    if bev_key_norm > 1.0e-9:
        bev_key /= bev_key_norm
    return descriptor.astype(np.float64), bev_key.astype(np.float64)


def _points_in_center_frame(
    center_pose: np.ndarray,
    other_pose: np.ndarray,
    points_local: np.ndarray,
) -> np.ndarray:
    if points_local.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    relative = _relative_pose(center_pose, other_pose)
    return _apply_rigid_transform(points_local, float(relative[2]), relative[:2])


def _build_submap_descriptor_cache(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    *,
    radius: int,
    output_dir: Path | None = None,
) -> list[SubmapDescriptorEntry]:
    cache_entries: list[SubmapDescriptorEntry] = []
    cache_dir = None if output_dir is None else (output_dir / "submap_cache")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for center_index, keyframe in enumerate(keyframes):
        collected_local: list[np.ndarray] = []
        center_pose = reference_poses[center_index]
        for relative_offset in range(-radius, radius + 1):
            sample_index = center_index + relative_offset
            if sample_index < 0 or sample_index >= len(keyframes):
                continue
            sample_points = keyframes[sample_index].points_local
            if sample_points.size == 0:
                continue
            local_points = _points_in_center_frame(center_pose, reference_poses[sample_index], sample_points)
            if local_points.size == 0:
                continue
            collected_local.append(local_points)
        if collected_local:
            local_points_xy = np.vstack(collected_local)
            local_points_xy = _voxel_downsample(local_points_xy, 0.05, 320)
        else:
            local_points_xy = np.empty((0, 2), dtype=np.float64)
        bev_descriptor, bev_key = _compute_bev_descriptor(local_points_xy)
        entry = SubmapDescriptorEntry(
            center_index=center_index,
            local_points_xy=local_points_xy,
            bev_descriptor=bev_descriptor,
            bev_key=bev_key,
        )
        cache_entries.append(entry)
        if cache_dir is not None:
            np.savez_compressed(
                cache_dir / f"submap_{center_index:04d}.npz",
                local_points_xy=local_points_xy,
                bev_descriptor=bev_descriptor,
                bev_key=bev_key,
            )
    if cache_dir is not None:
        summary_payload = {
            "count": len(cache_entries),
            "radius": int(radius),
            "files": [f"submap_{index:04d}.npz" for index in range(len(cache_entries))],
        }
        (cache_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return cache_entries


def _quality_selection_multiplier(label: str) -> float:
    if label == "high":
        return 1.0
    if label == "reduced":
        return 1.25
    return 1.75


def _keyframe_priority_score(keyframe: Keyframe, index: int, total_count: int) -> float:
    edge_bonus = 2.0 if index in (0, max(0, total_count - 1)) else 0.0
    quality_bonus = {
        "high": 1.0,
        "reduced": 0.65,
        "low": 0.25,
    }.get(keyframe.prior_quality_label, 0.25)
    shape_bonus = min(0.45, 0.0035 * float(keyframe.points_local.shape[0]))
    return (
        edge_bonus
        + quality_bonus
        + (0.8 * float(keyframe.prior_quality_scale_xy))
        + (0.9 * float(keyframe.prior_quality_scale_yaw))
        + shape_bonus
    )


def _select_keyframes(
    frames: list[ScanFrame],
    *,
    distance_threshold_m: float,
    yaw_threshold_rad: float,
    time_threshold_s: float,
    voxel_size_m: float,
    max_points_per_keyframe: int,
    max_keyframes: int,
) -> list[Keyframe]:
    if not frames:
        return []
    selected: list[Keyframe] = []
    last_pose = frames[0].prior_pose_xyyaw
    last_t_s = frames[0].t_s
    for frame in frames:
        selection_multiplier = _quality_selection_multiplier(frame.prior_quality_label)
        translation_m = float(np.linalg.norm(frame.prior_pose_xyyaw[:2] - last_pose[:2]))
        yaw_delta_rad = abs(_normalize_angle(float(frame.prior_pose_xyyaw[2] - last_pose[2])))
        time_delta_s = float(frame.t_s - last_t_s)
        if (
            not selected
            or translation_m >= (distance_threshold_m * selection_multiplier)
            or yaw_delta_rad >= (yaw_threshold_rad * selection_multiplier)
            or time_delta_s >= (time_threshold_s * selection_multiplier)
            or (
                frame.prior_quality_label != "low"
                and (
                    translation_m >= (0.75 * distance_threshold_m)
                    or yaw_delta_rad >= (0.75 * yaw_threshold_rad)
                )
            )
        ):
            points_local = _voxel_downsample(frame.points_local, voxel_size_m, max_points_per_keyframe)
            descriptor_scan_context, descriptor_ring_key = _compute_scan_context_descriptor(points_local)
            selected.append(
                Keyframe(
                    keyframe_index=len(selected),
                    scan_index=frame.scan_index,
                    t_s=frame.t_s,
                    points_local=points_local,
                    prior_pose_xyyaw=frame.prior_pose_xyyaw.copy(),
                    prior_velocity_xy=frame.prior_velocity_xy.copy(),
                    prior_yaw_rate_rps=frame.prior_yaw_rate_rps,
                    prior_quality_label=frame.prior_quality_label,
                    prior_quality_scale_xy=frame.prior_quality_scale_xy,
                    prior_quality_scale_yaw=frame.prior_quality_scale_yaw,
                    descriptor_polar=_compute_polar_descriptor(points_local),
                    descriptor_angular=_compute_angular_descriptor(points_local),
                    descriptor_scan_context=descriptor_scan_context,
                    descriptor_ring_key=descriptor_ring_key,
                )
            )
            last_pose = frame.prior_pose_xyyaw
            last_t_s = frame.t_s
    if selected[-1].scan_index != frames[-1].scan_index:
        tail_frame = frames[-1]
        points_local = _voxel_downsample(tail_frame.points_local, voxel_size_m, max_points_per_keyframe)
        descriptor_scan_context, descriptor_ring_key = _compute_scan_context_descriptor(points_local)
        selected.append(
            Keyframe(
                keyframe_index=len(selected),
                scan_index=tail_frame.scan_index,
                t_s=tail_frame.t_s,
                points_local=points_local,
                prior_pose_xyyaw=tail_frame.prior_pose_xyyaw.copy(),
                prior_velocity_xy=tail_frame.prior_velocity_xy.copy(),
                prior_yaw_rate_rps=tail_frame.prior_yaw_rate_rps,
                prior_quality_label=tail_frame.prior_quality_label,
                prior_quality_scale_xy=tail_frame.prior_quality_scale_xy,
                prior_quality_scale_yaw=tail_frame.prior_quality_scale_yaw,
                descriptor_polar=_compute_polar_descriptor(points_local),
                descriptor_angular=_compute_angular_descriptor(points_local),
                descriptor_scan_context=descriptor_scan_context,
                descriptor_ring_key=descriptor_ring_key,
            )
        )
    if len(selected) <= max_keyframes:
        return selected

    bucket_edges = np.linspace(0, len(selected), num=max_keyframes + 1, dtype=np.int32)
    keep_indexes: set[int] = {0, len(selected) - 1}
    for bucket_index in range(max_keyframes):
        start_index = int(bucket_edges[bucket_index])
        end_index = int(bucket_edges[bucket_index + 1])
        if end_index <= start_index:
            continue
        candidate_indexes = list(range(start_index, end_index))
        best_index = max(
            candidate_indexes,
            key=lambda idx: _keyframe_priority_score(selected[idx], idx, len(selected)),
        )
        keep_indexes.add(int(best_index))
    if len(keep_indexes) > max_keyframes:
        ranked_keep = sorted(
            keep_indexes,
            key=lambda idx: (
                1 if idx in (0, len(selected) - 1) else 0,
                _keyframe_priority_score(selected[idx], idx, len(selected)),
            ),
            reverse=True,
        )
        keep_indexes = set(ranked_keep[:max_keyframes])
    thinned: list[Keyframe] = []
    for new_index, old_index in enumerate(sorted(keep_indexes)):
        keyframe = selected[old_index]
        thinned.append(
            Keyframe(
                keyframe_index=new_index,
                scan_index=keyframe.scan_index,
                t_s=keyframe.t_s,
                points_local=keyframe.points_local.copy(),
                prior_pose_xyyaw=keyframe.prior_pose_xyyaw.copy(),
                prior_velocity_xy=keyframe.prior_velocity_xy.copy(),
                prior_yaw_rate_rps=keyframe.prior_yaw_rate_rps,
                prior_quality_label=keyframe.prior_quality_label,
                prior_quality_scale_xy=keyframe.prior_quality_scale_xy,
                prior_quality_scale_yaw=keyframe.prior_quality_scale_yaw,
                descriptor_polar=keyframe.descriptor_polar.copy(),
                descriptor_angular=keyframe.descriptor_angular.copy(),
                descriptor_scan_context=keyframe.descriptor_scan_context.copy(),
                descriptor_ring_key=keyframe.descriptor_ring_key.copy(),
            )
        )
    return thinned


def _build_submap_points(
    keyframes: list[Keyframe],
    poses_xyyaw: np.ndarray,
    *,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    world_points: list[np.ndarray] = []
    for keyframe_index in range(start_index, end_index):
        points_local = keyframes[keyframe_index].points_local
        if points_local.size == 0:
            continue
        world_points.append(_transform_points(points_local, poses_xyyaw[keyframe_index]))
    if not world_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(world_points)


def _build_hit_count_grid(
    map_points_xy: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    min_extent_cells: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if map_points_xy.size == 0:
        hit_counts = np.zeros((min_extent_cells, min_extent_cells), dtype=np.int32)
        origin_xy = np.asarray(
            [
                -0.5 * min_extent_cells * resolution_m,
                -0.5 * min_extent_cells * resolution_m,
            ],
            dtype=np.float64,
        )
        return hit_counts, origin_xy

    min_xy = np.min(map_points_xy, axis=0) - margin_m
    max_xy = np.max(map_points_xy, axis=0) + margin_m
    width = max(min_extent_cells, int(math.ceil((max_xy[0] - min_xy[0]) / resolution_m)) + 1)
    height = max(min_extent_cells, int(math.ceil((max_xy[1] - min_xy[1]) / resolution_m)) + 1)
    hit_counts = np.zeros((height, width), dtype=np.int32)
    gx = np.floor((map_points_xy[:, 0] - min_xy[0]) / resolution_m).astype(np.int32)
    gy = np.floor((map_points_xy[:, 1] - min_xy[1]) / resolution_m).astype(np.int32)
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
    max_distance_m = max(0.6, 5.0 * resolution_m)
    return DistanceFieldLevel(
        resolution_m=float(resolution_m),
        origin_xy=origin_xy,
        distance_field_m=np.asarray(distance_field, dtype=np.float32),
        occupancy=np.asarray(occupancy, dtype=bool),
        max_distance_m=float(max_distance_m),
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


def _build_multires_levels(
    submap_points_xy: np.ndarray,
    *,
    resolutions_m: tuple[float, ...],
    margin_m: float,
) -> list[DistanceFieldLevel]:
    return [
        _build_distance_field_level(
            submap_points_xy,
            resolution_m=float(resolution_m),
            margin_m=margin_m,
        )
        for resolution_m in resolutions_m
    ]


def _limit_pose_correction(
    reference_pose: np.ndarray,
    candidate_pose: np.ndarray,
    *,
    max_translation_m: float,
    max_yaw_rad: float,
) -> np.ndarray:
    limited = candidate_pose.astype(np.float64).copy()
    delta_xy = limited[:2] - reference_pose[:2]
    delta_norm = float(np.linalg.norm(delta_xy))
    if delta_norm > max_translation_m and delta_norm > 1.0e-9:
        limited[:2] = reference_pose[:2] + (delta_xy * (max_translation_m / delta_norm))
    delta_yaw = _normalize_angle(float(limited[2] - reference_pose[2]))
    delta_yaw = max(-max_yaw_rad, min(max_yaw_rad, delta_yaw))
    limited[2] = _normalize_angle(float(reference_pose[2]) + delta_yaw)
    return limited


def _multires_distance_field_residuals(
    candidate_pose: np.ndarray,
    *,
    points_local: np.ndarray,
    level: DistanceFieldLevel,
    max_correspondence_m: float,
    prior_pose: np.ndarray,
    prior_weight_xy: float,
    prior_weight_yaw: float,
) -> np.ndarray:
    candidate = candidate_pose.astype(np.float64).copy()
    candidate[2] = _normalize_angle(float(candidate[2]))
    world_points = _transform_points(points_local, candidate)
    distances_m = _sample_distance_field_bilinear(
        level,
        world_points,
        outside_distance_m=max(level.max_distance_m, max_correspondence_m * 2.0),
    )
    distances_m = np.clip(distances_m, 0.0, max(level.max_distance_m, max_correspondence_m * 2.0))
    distances_m *= 0.35
    prior_residual = np.asarray(
        [
            prior_weight_xy * float(candidate[0] - prior_pose[0]),
            prior_weight_xy * float(candidate[1] - prior_pose[1]),
            prior_weight_yaw * _normalize_angle(float(candidate[2] - prior_pose[2])),
        ],
        dtype=np.float64,
    )
    return np.concatenate((distances_m, prior_residual))


def _optimize_pose_against_levels(
    points_local: np.ndarray,
    initial_pose: np.ndarray,
    prior_pose: np.ndarray,
    levels: list[DistanceFieldLevel],
    *,
    max_correspondence_m: float,
    prior_weight_xy: float,
    prior_weight_yaw: float,
    loss: str,
    max_nfev: int,
) -> tuple[np.ndarray, float]:
    if points_local.shape[0] < 8 or not levels:
        return initial_pose.copy(), float("nan")
    state = initial_pose.astype(np.float64).copy()
    for level in levels:
        result = least_squares(
            lambda state_vec: _multires_distance_field_residuals(
                state_vec,
                points_local=points_local,
                level=level,
                max_correspondence_m=max_correspondence_m,
                prior_pose=prior_pose,
                prior_weight_xy=prior_weight_xy,
                prior_weight_yaw=prior_weight_yaw,
            ),
            x0=state,
            loss=loss,
            f_scale=max(0.03, 0.5 * max_correspondence_m),
            max_nfev=max_nfev,
        )
        state = result.x.astype(np.float64)
        state[2] = _normalize_angle(float(state[2]))
    final_level = levels[-1]
    final_distances = _sample_distance_field_bilinear(
        final_level,
        _transform_points(points_local, state),
        outside_distance_m=max(final_level.max_distance_m, max_correspondence_m * 2.0),
    )
    final_distances = np.clip(final_distances, 0.0, max(final_level.max_distance_m, max_correspondence_m * 2.0))
    return state, float(np.median(final_distances))


def _match_statistics_against_level(
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
            "p90_distance_m": float("inf"),
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
        "p90_distance_m": float(np.percentile(distances, 90.0)),
    }


def _heading_difference_rad(source_yaw_rad: float, target_yaw_rad: float) -> float:
    return abs(_normalize_angle(float(target_yaw_rad) - float(source_yaw_rad)))


def _compute_segment_descriptors(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    *,
    radius: int,
) -> list[np.ndarray]:
    descriptors: list[np.ndarray] = []
    for center_index, keyframe in enumerate(keyframes):
        center_pose = reference_poses[center_index]
        polar_acc = np.zeros_like(keyframe.descriptor_polar)
        angular_acc = np.zeros_like(keyframe.descriptor_angular)
        path_features: list[float] = []
        normalizer = max(0.6, float(radius) * 0.30)
        yaw_normalizer = max(math.radians(15.0), float(radius) * math.radians(4.0))
        for relative_offset in range(-radius, radius + 1):
            sample_index = center_index + relative_offset
            if 0 <= sample_index < len(keyframes):
                sample_keyframe = keyframes[sample_index]
                polar_acc += sample_keyframe.descriptor_polar
                angular_acc += sample_keyframe.descriptor_angular
                rel_pose = _relative_pose(center_pose, reference_poses[sample_index])
                path_features.extend(
                    [
                        float(rel_pose[0] / normalizer),
                        float(rel_pose[1] / normalizer),
                        float(rel_pose[2] / yaw_normalizer),
                    ]
                )
            else:
                path_features.extend([0.0, 0.0, 0.0])
        polar_norm = float(np.linalg.norm(polar_acc))
        if polar_norm > 1.0e-9:
            polar_acc /= polar_norm
        angular_norm = float(np.linalg.norm(angular_acc))
        if angular_norm > 1.0e-9:
            angular_acc /= angular_norm
        descriptor = np.concatenate(
            (
                polar_acc,
                angular_acc,
                np.asarray(path_features, dtype=np.float64),
            )
        )
        descriptor_norm = float(np.linalg.norm(descriptor))
        if descriptor_norm > 1.0e-9:
            descriptor /= descriptor_norm
        descriptors.append(descriptor.astype(np.float64))
    return descriptors


def _validate_bidirectional_geometric_edge(
    *,
    edge_type: str,
    similarity_score: float,
    source_index: int,
    target_index: int,
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    submap_keyframes: int,
    max_correspondence_m: float,
    prior_weight_xy: float,
    prior_weight_yaw: float,
    loss: str,
    max_nfev: int,
    max_translation_correction_m: float,
    max_yaw_correction_rad: float,
    max_residual_m: float,
    min_inlier_ratio: float,
    min_support_ratio: float,
    max_bidirectional_translation_error_m: float,
    max_bidirectional_yaw_error_rad: float,
    base_weight_xy: float,
    base_weight_yaw: float,
    forward_initial_pose: np.ndarray | None = None,
    reverse_initial_pose: np.ndarray | None = None,
) -> PoseGraphEdge | None:
    resolutions_m = (0.20, 0.10, 0.05)
    half_window = max(1, submap_keyframes // 2)

    source_submap = _build_submap_points(
        keyframes,
        reference_poses,
        start_index=max(0, source_index - half_window),
        end_index=min(len(keyframes), source_index + half_window + 1),
    )
    source_levels = _build_multires_levels(
        source_submap,
        resolutions_m=resolutions_m,
        margin_m=max(0.8, 2.0 * max_correspondence_m),
    )
    refined_target_pose, residual_m = _optimize_pose_against_levels(
        keyframes[target_index].points_local,
        reference_poses[target_index] if forward_initial_pose is None else forward_initial_pose,
        reference_poses[target_index],
        source_levels,
        max_correspondence_m=max_correspondence_m,
        prior_weight_xy=prior_weight_xy,
        prior_weight_yaw=prior_weight_yaw,
        loss=loss,
        max_nfev=max_nfev,
    )
    if not math.isfinite(residual_m) or residual_m > max_residual_m:
        return None
    pose_delta_m = float(np.linalg.norm(refined_target_pose[:2] - reference_poses[target_index, :2]))
    yaw_delta_rad = _heading_difference_rad(refined_target_pose[2], reference_poses[target_index, 2])
    if pose_delta_m > max_translation_correction_m or yaw_delta_rad > max_yaw_correction_rad:
        return None
    forward_stats = _match_statistics_against_level(
        keyframes[target_index].points_local,
        refined_target_pose,
        source_levels[-1],
        max_correspondence_m=max_correspondence_m,
    )
    if (
        forward_stats["inlier_ratio"] < min_inlier_ratio
        or forward_stats["support_ratio"] < min_support_ratio
    ):
        return None

    target_submap = _build_submap_points(
        keyframes,
        reference_poses,
        start_index=max(0, target_index - half_window),
        end_index=min(len(keyframes), target_index + half_window + 1),
    )
    target_levels = _build_multires_levels(
        target_submap,
        resolutions_m=resolutions_m,
        margin_m=max(0.8, 2.0 * max_correspondence_m),
    )
    refined_source_pose, reverse_residual_m = _optimize_pose_against_levels(
        keyframes[source_index].points_local,
        reference_poses[source_index] if reverse_initial_pose is None else reverse_initial_pose,
        reference_poses[source_index],
        target_levels,
        max_correspondence_m=max_correspondence_m,
        prior_weight_xy=prior_weight_xy,
        prior_weight_yaw=prior_weight_yaw,
        loss=loss,
        max_nfev=max_nfev,
    )
    if not math.isfinite(reverse_residual_m) or reverse_residual_m > max_residual_m:
        return None
    reverse_pose_delta_m = float(np.linalg.norm(refined_source_pose[:2] - reference_poses[source_index, :2]))
    reverse_yaw_delta_rad = _heading_difference_rad(refined_source_pose[2], reference_poses[source_index, 2])
    if reverse_pose_delta_m > max_translation_correction_m or reverse_yaw_delta_rad > max_yaw_correction_rad:
        return None
    reverse_stats = _match_statistics_against_level(
        keyframes[source_index].points_local,
        refined_source_pose,
        target_levels[-1],
        max_correspondence_m=max_correspondence_m,
    )
    if (
        reverse_stats["inlier_ratio"] < min_inlier_ratio
        or reverse_stats["support_ratio"] < min_support_ratio
    ):
        return None

    forward_delta = _relative_pose(reference_poses[source_index], refined_target_pose)
    reverse_delta = _relative_pose(reference_poses[target_index], refined_source_pose)
    reverse_as_forward = _invert_relative_pose(reverse_delta)
    bidirectional_translation_error_m = float(
        np.linalg.norm(forward_delta[:2] - reverse_as_forward[:2])
    )
    bidirectional_yaw_error_rad = _heading_difference_rad(
        forward_delta[2],
        reverse_as_forward[2],
    )
    if (
        bidirectional_translation_error_m > max_bidirectional_translation_error_m
        or bidirectional_yaw_error_rad > max_bidirectional_yaw_error_rad
    ):
        return None

    quality_scale_xy = float(
        min(
            keyframes[source_index].prior_quality_scale_xy,
            keyframes[target_index].prior_quality_scale_xy,
        )
    )
    quality_scale_yaw = float(
        min(
            keyframes[source_index].prior_quality_scale_yaw,
            keyframes[target_index].prior_quality_scale_yaw,
        )
    )
    reliability_scale = min(
        1.0,
        0.55
        + (0.45 * similarity_score)
        + (0.25 * forward_stats["inlier_ratio"])
        + (0.20 * reverse_stats["inlier_ratio"]),
    )
    return PoseGraphEdge(
        edge_type=edge_type,
        source_index=source_index,
        target_index=target_index,
        measured_delta_xyyaw=forward_delta,
        weight_xy=max(1.0, base_weight_xy * quality_scale_xy * reliability_scale),
        weight_yaw=max(1.2, base_weight_yaw * quality_scale_yaw * reliability_scale),
        residual_m=0.5 * (float(residual_m) + float(reverse_residual_m)),
        score=float(similarity_score),
    )


def _sequential_refine_keyframes(
    keyframes: list[Keyframe],
    *,
    submap_keyframes: int,
    max_correspondence_m: float,
    local_registration: str,
    loss: str,
    status_callback=None,
) -> tuple[np.ndarray, list[PoseGraphEdge], list[PoseGraphEdge]]:
    poses = np.zeros((len(keyframes), 3), dtype=np.float64)
    local_edges: list[PoseGraphEdge] = []
    odom_edges: list[PoseGraphEdge] = []
    if not keyframes:
        return poses, local_edges, odom_edges
    poses[0] = keyframes[0].prior_pose_xyyaw.copy()
    resolutions_m = (0.20, 0.10, 0.05)
    for keyframe_index in range(1, len(keyframes)):
        if status_callback is not None and (
            keyframe_index == 1
            or (keyframe_index % 10) == 0
            or keyframe_index == (len(keyframes) - 1)
        ):
            status_callback("sequential_refine", keyframe_index, len(keyframes))
        prior_delta = _relative_pose(
            keyframes[keyframe_index - 1].prior_pose_xyyaw,
            keyframes[keyframe_index].prior_pose_xyyaw,
        )
        predicted_pose = _compose_pose(poses[keyframe_index - 1], prior_delta)
        submap_start = max(0, keyframe_index - submap_keyframes)
        submap_points = _build_submap_points(
            keyframes,
            poses,
            start_index=submap_start,
            end_index=keyframe_index,
        )
        if local_registration == "multires_distance_field":
            quality_scale_xy = float(
                min(
                    keyframes[keyframe_index - 1].prior_quality_scale_xy,
                    keyframes[keyframe_index].prior_quality_scale_xy,
                )
            )
            quality_scale_yaw = float(
                min(
                    keyframes[keyframe_index - 1].prior_quality_scale_yaw,
                    keyframes[keyframe_index].prior_quality_scale_yaw,
                )
            )
            levels = _build_multires_levels(
                submap_points,
                resolutions_m=resolutions_m,
                margin_m=max(0.6, 2.0 * max_correspondence_m),
            )
            refined_pose, residual_m = _optimize_pose_against_levels(
                keyframes[keyframe_index].points_local,
                predicted_pose,
                predicted_pose,
                levels,
                max_correspondence_m=max_correspondence_m,
                prior_weight_xy=max(0.30, 1.0 * quality_scale_xy),
                prior_weight_yaw=max(0.45, 1.6 * quality_scale_yaw),
                loss="soft_l1",
                max_nfev=35,
            )
            refined_pose = _limit_pose_correction(
                predicted_pose,
                refined_pose,
                max_translation_m=0.20,
                max_yaw_rad=math.radians(10.0),
            )
        else:
            raise ValueError(f"Unsupported local_registration={local_registration!r}")
        poses[keyframe_index] = refined_pose
        odom_edges.append(
            PoseGraphEdge(
                edge_type="odom_prior",
                source_index=keyframe_index - 1,
                target_index=keyframe_index,
                measured_delta_xyyaw=prior_delta,
                weight_xy=max(0.25, 1.0 * quality_scale_xy),
                weight_yaw=max(0.35, 1.5 * quality_scale_yaw),
                residual_m=0.0,
            )
        )
        local_quality_scale = 1.0
        if math.isfinite(residual_m):
            if residual_m > 0.18:
                local_quality_scale = 0.45
            elif residual_m > 0.10:
                local_quality_scale = 0.75
        local_edges.append(
            PoseGraphEdge(
                edge_type="local_scan_match",
                source_index=keyframe_index - 1,
                target_index=keyframe_index,
                measured_delta_xyyaw=_relative_pose(poses[keyframe_index - 1], refined_pose),
                weight_xy=3.0 * local_quality_scale,
                weight_yaw=4.0 * local_quality_scale,
                residual_m=0.0 if not math.isfinite(residual_m) else residual_m,
            )
        )
    return poses, local_edges, odom_edges


def _candidate_loop_closure_indexes(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    target_index: int,
    *,
    min_index_gap: int,
    max_candidates: int,
    descriptor_mode: str,
    submap_descriptor_cache: list[SubmapDescriptorEntry] | None = None,
) -> list[tuple[float, int, float]]:
    if descriptor_mode not in {"polar_occupancy", "scan_context_2d"}:
        raise ValueError(f"Unsupported loop_closure_descriptor={descriptor_mode!r}")
    target_keyframe = keyframes[target_index]
    candidate_scores: list[tuple[float, int, float]] = []
    target_ref_xy = reference_poses[target_index, :2]
    target_ref_yaw = float(reference_poses[target_index, 2])
    for source_index in range(0, target_index - min_index_gap):
        source_keyframe = keyframes[source_index]
        prior_distance_m = float(
            np.linalg.norm(target_ref_xy - reference_poses[source_index, :2])
        )
        heading_diff_rad = _heading_difference_rad(target_ref_yaw, reference_poses[source_index, 2])
        polar_similarity = _cosine_similarity(
            target_keyframe.descriptor_polar,
            source_keyframe.descriptor_polar,
        )
        angular_similarity = _cosine_similarity(
            target_keyframe.descriptor_angular,
            source_keyframe.descriptor_angular,
        )
        yaw_offset_rad = 0.0
        if descriptor_mode == "scan_context_2d":
            if prior_distance_m > 5.2 or heading_diff_rad > math.radians(80.0):
                continue
            ring_similarity = _cosine_similarity(
                target_keyframe.descriptor_ring_key,
                source_keyframe.descriptor_ring_key,
            )
            if ring_similarity <= 0.24:
                continue
            scan_context_similarity, sector_shift = _best_scan_context_alignment(
                target_keyframe.descriptor_scan_context,
                source_keyframe.descriptor_scan_context,
            )
            sector_count = max(1, int(target_keyframe.descriptor_scan_context.shape[1]))
            yaw_offset_rad = _normalize_angle(-((2.0 * math.pi * float(sector_shift)) / float(sector_count)))
            yaw_consistency_rad = _heading_difference_rad(
                _normalize_angle(float(reference_poses[source_index, 2]) + yaw_offset_rad),
                target_ref_yaw,
            )
            if yaw_consistency_rad > math.radians(34.0):
                continue
            bev_similarity = 0.0
            bev_key_similarity = 0.0
            if submap_descriptor_cache is not None:
                target_submap = submap_descriptor_cache[target_index]
                source_submap = submap_descriptor_cache[source_index]
                bev_similarity = _cosine_similarity(
                    target_submap.bev_descriptor.reshape(-1),
                    source_submap.bev_descriptor.reshape(-1),
                )
                bev_key_similarity = _cosine_similarity(
                    target_submap.bev_key,
                    source_submap.bev_key,
                )
                if bev_similarity <= 0.08 and bev_key_similarity <= 0.16:
                    continue
            combined_similarity = (
                (0.38 * scan_context_similarity)
                + (0.20 * ring_similarity)
                + (0.18 * bev_similarity)
                + (0.08 * bev_key_similarity)
                + (0.10 * polar_similarity)
                + (0.06 * angular_similarity)
            )
            if prior_distance_m <= 2.8:
                combined_similarity += 0.06
            if yaw_consistency_rad <= math.radians(18.0):
                combined_similarity += 0.08
            elif yaw_consistency_rad <= math.radians(28.0):
                combined_similarity += 0.03
        else:
            if prior_distance_m > 4.2 or heading_diff_rad > math.radians(50.0):
                continue
            combined_similarity = (0.7 * polar_similarity) + (0.3 * angular_similarity)
            if prior_distance_m <= 2.6:
                combined_similarity += 0.15
            if heading_diff_rad <= math.radians(20.0):
                combined_similarity += 0.08
        candidate_scores.append((combined_similarity, source_index, yaw_offset_rad))
    candidate_scores.sort(key=lambda item: item[0], reverse=True)
    return candidate_scores[:max_candidates]


def _detect_loop_closures(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    *,
    submap_keyframes: int,
    min_index_gap: int,
    max_correspondence_m: float,
    max_edges_per_keyframe: int,
    descriptor_mode: str,
    submap_descriptor_cache: list[SubmapDescriptorEntry] | None = None,
    status_callback=None,
) -> list[PoseGraphEdge]:
    loop_edges: list[PoseGraphEdge] = []
    if len(keyframes) < (min_index_gap + 2):
        return loop_edges
    resolutions_m = (0.20, 0.10, 0.05)
    for target_index in range(min_index_gap + 1, len(keyframes)):
        if status_callback is not None and (
            target_index == (min_index_gap + 1)
            or (target_index % 10) == 0
            or target_index == (len(keyframes) - 1)
        ):
            status_callback("loop_closure_search", target_index, len(keyframes))
        accepted = 0
        candidate_indexes = _candidate_loop_closure_indexes(
            keyframes,
            reference_poses,
            target_index,
            min_index_gap=min_index_gap,
            max_candidates=4 if descriptor_mode == "scan_context_2d" else 6,
            descriptor_mode=descriptor_mode,
            submap_descriptor_cache=submap_descriptor_cache,
        )
        for similarity_score, source_index, yaw_offset_rad in candidate_indexes:
            similarity_threshold = 0.18 if descriptor_mode == "polar_occupancy" else 0.36
            if similarity_score <= similarity_threshold:
                continue
            forward_initial_pose = None
            reverse_initial_pose = None
            if descriptor_mode == "scan_context_2d":
                expected_target_yaw = _normalize_angle(float(reference_poses[source_index, 2]) + yaw_offset_rad)
                expected_source_yaw = _normalize_angle(float(reference_poses[target_index, 2]) - yaw_offset_rad)
                forward_initial_pose = reference_poses[target_index].copy()
                reverse_initial_pose = reference_poses[source_index].copy()
                forward_initial_pose[2] = _blend_angles(float(reference_poses[target_index, 2]), expected_target_yaw, 0.80)
                reverse_initial_pose[2] = _blend_angles(float(reference_poses[source_index, 2]), expected_source_yaw, 0.80)
            edge = _validate_bidirectional_geometric_edge(
                edge_type="loop_closure",
                similarity_score=similarity_score,
                source_index=source_index,
                target_index=target_index,
                keyframes=keyframes,
                reference_poses=reference_poses,
                submap_keyframes=submap_keyframes,
                max_correspondence_m=max_correspondence_m,
                prior_weight_xy=max(0.25, 0.60 * keyframes[target_index].prior_quality_scale_xy),
                prior_weight_yaw=max(0.35, 0.90 * keyframes[target_index].prior_quality_scale_yaw),
                loss="soft_l1",
                max_nfev=26,
                max_translation_correction_m=0.80 if descriptor_mode == "polar_occupancy" else 0.60,
                max_yaw_correction_rad=math.radians(20.0 if descriptor_mode == "polar_occupancy" else 15.0),
                max_residual_m=0.08 if descriptor_mode == "polar_occupancy" else 0.065,
                min_inlier_ratio=0.55 if descriptor_mode == "polar_occupancy" else 0.62,
                min_support_ratio=0.32 if descriptor_mode == "polar_occupancy" else 0.38,
                max_bidirectional_translation_error_m=0.45 if descriptor_mode == "polar_occupancy" else 0.28,
                max_bidirectional_yaw_error_rad=math.radians(10.0 if descriptor_mode == "polar_occupancy" else 7.0),
                base_weight_xy=4.0 if descriptor_mode == "polar_occupancy" else 3.1,
                base_weight_yaw=5.0 if descriptor_mode == "polar_occupancy" else 4.1,
                forward_initial_pose=forward_initial_pose,
                reverse_initial_pose=reverse_initial_pose,
            )
            if edge is None:
                continue
            loop_edges.append(edge)
            accepted += 1
            max_edges_this_keyframe = 1 if descriptor_mode == "scan_context_2d" else max_edges_per_keyframe
            if accepted >= max_edges_this_keyframe:
                break
    return loop_edges


def _candidate_segment_alignment_indexes(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    segment_descriptors: list[np.ndarray],
    target_index: int,
    *,
    min_index_gap: int,
    max_candidates: int,
) -> list[tuple[float, int]]:
    target_xy = reference_poses[target_index, :2]
    target_yaw = float(reference_poses[target_index, 2])
    target_descriptor = segment_descriptors[target_index]
    scores: list[tuple[float, int]] = []
    for source_index in range(0, target_index - min_index_gap):
        distance_m = float(np.linalg.norm(target_xy - reference_poses[source_index, :2]))
        if distance_m > 4.8:
            continue
        heading_diff_rad = _heading_difference_rad(target_yaw, reference_poses[source_index, 2])
        if heading_diff_rad > math.radians(50.0):
            continue
        similarity = _cosine_similarity(target_descriptor, segment_descriptors[source_index])
        if distance_m <= 2.8:
            similarity += 0.08
        if heading_diff_rad <= math.radians(18.0):
            similarity += 0.06
        scores.append((similarity, source_index))
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores[:max_candidates]


def _detect_segment_consistency_edges(
    keyframes: list[Keyframe],
    reference_poses: np.ndarray,
    *,
    submap_keyframes: int,
    min_index_gap: int,
    max_correspondence_m: float,
    max_edges_per_keyframe: int,
    status_callback=None,
) -> list[PoseGraphEdge]:
    if len(keyframes) < (min_index_gap + 4):
        return []
    segment_descriptors = _compute_segment_descriptors(
        keyframes,
        reference_poses,
        radius=2,
    )
    segment_edges: list[PoseGraphEdge] = []
    for target_index in range(min_index_gap + 2, len(keyframes) - 2):
        if status_callback is not None and (
            target_index == (min_index_gap + 2)
            or (target_index % 10) == 0
            or target_index == (len(keyframes) - 3)
        ):
            status_callback("segment_consistency_search", target_index, len(keyframes))
        accepted = 0
        candidates = _candidate_segment_alignment_indexes(
            keyframes,
            reference_poses,
            segment_descriptors,
            target_index,
            min_index_gap=min_index_gap,
            max_candidates=3,
        )
        for similarity_score, source_index in candidates:
            if similarity_score <= 0.52:
                continue
            edge = _validate_bidirectional_geometric_edge(
                edge_type="segment_consistency",
                similarity_score=similarity_score,
                source_index=source_index,
                target_index=target_index,
                keyframes=keyframes,
                reference_poses=reference_poses,
                submap_keyframes=max(submap_keyframes, 12),
                max_correspondence_m=max_correspondence_m,
                prior_weight_xy=max(0.30, 0.70 * keyframes[target_index].prior_quality_scale_xy),
                prior_weight_yaw=max(0.45, 1.00 * keyframes[target_index].prior_quality_scale_yaw),
                loss="soft_l1",
                max_nfev=26,
                max_translation_correction_m=0.75,
                max_yaw_correction_rad=math.radians(18.0),
                max_residual_m=0.07,
                min_inlier_ratio=0.60,
                min_support_ratio=0.35,
                max_bidirectional_translation_error_m=0.35,
                max_bidirectional_yaw_error_rad=math.radians(8.0),
                base_weight_xy=4.8,
                base_weight_yaw=5.8,
            )
            if edge is None:
                continue
            segment_edges.append(edge)
            accepted += 1
            if accepted >= max_edges_per_keyframe:
                break
    return segment_edges


def _optimize_pose_graph(
    initial_poses: np.ndarray,
    edges: list[PoseGraphEdge],
    *,
    loss: str,
) -> np.ndarray:
    if initial_poses.shape[0] <= 1 or not edges:
        return initial_poses.copy()
    x0 = initial_poses.reshape(-1).astype(np.float64)

    def _residuals(flat_state: np.ndarray) -> np.ndarray:
        poses = flat_state.reshape((-1, 3))
        residuals: list[float] = [
            20.0 * float(poses[0, 0] - initial_poses[0, 0]),
            20.0 * float(poses[0, 1] - initial_poses[0, 1]),
            24.0 * _normalize_angle(float(poses[0, 2] - initial_poses[0, 2])),
        ]
        for edge in edges:
            error_xyyaw = _relative_pose_error(
                poses[edge.source_index],
                poses[edge.target_index],
                edge.measured_delta_xyyaw,
            )
            residuals.append(float(edge.weight_xy * error_xyyaw[0]))
            residuals.append(float(edge.weight_xy * error_xyyaw[1]))
            residuals.append(float(edge.weight_yaw * error_xyyaw[2]))
        return np.asarray(residuals, dtype=np.float64)

    result = least_squares(
        _residuals,
        x0=x0,
        loss=loss,
        f_scale=0.10,
        max_nfev=45,
    )
    optimized = result.x.reshape((-1, 3)).astype(np.float64)
    optimized[:, 2] = np.vectorize(_normalize_angle)(optimized[:, 2])
    return optimized


def _build_map_points(keyframes: list[Keyframe], poses_xyyaw: np.ndarray) -> np.ndarray:
    world_points: list[np.ndarray] = []
    for keyframe_index, keyframe in enumerate(keyframes):
        if keyframe.points_local.size == 0:
            continue
        world_points.append(_transform_points(keyframe.points_local, poses_xyyaw[keyframe_index]))
    if not world_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(world_points)


def _derive_map_points(
    keyframes: list[Keyframe],
    poses_xyyaw: np.ndarray,
    local_edges: list[PoseGraphEdge],
) -> np.ndarray:
    local_residuals_by_target: dict[int, float] = {
        int(edge.target_index): float(edge.residual_m)
        for edge in local_edges
        if edge.edge_type == "local_scan_match" and math.isfinite(edge.residual_m)
    }
    world_points: list[np.ndarray] = []
    for keyframe_index, keyframe in enumerate(keyframes):
        if keyframe.points_local.size == 0:
            continue
        local_residual_m = local_residuals_by_target.get(keyframe_index, 0.0)
        support_score = min(
            float(keyframe.prior_quality_scale_xy),
            float(keyframe.prior_quality_scale_yaw),
        )
        if math.isfinite(local_residual_m):
            if local_residual_m > 0.20:
                support_score *= 0.35
            elif local_residual_m > 0.12:
                support_score *= 0.60
            elif local_residual_m > 0.08:
                support_score *= 0.82
        point_stride = 1
        if keyframe.prior_quality_label == "low":
            point_stride = 2
        if support_score < 0.45:
            if (keyframe_index % 4) != 0:
                continue
            point_stride = max(point_stride, 4)
        elif support_score < 0.65:
            if (keyframe_index % 3) != 0:
                point_stride = max(point_stride, 2)
        elif support_score < 0.85:
            point_stride = max(point_stride, 2)
        points_local = keyframe.points_local[::point_stride]
        if points_local.size == 0:
            continue
        world_points.append(_transform_points(points_local, poses_xyyaw[keyframe_index]))
    if not world_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(world_points)


def _remove_small_components(mask: np.ndarray, min_area_cells: int) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    structure = generate_binary_structure(2, 2)
    labels, count = label(mask, structure=structure)
    if count <= 0:
        return mask.astype(bool)
    output = np.zeros_like(mask, dtype=bool)
    component_areas = np.bincount(labels.reshape(-1))
    for component_index in range(1, count + 1):
        area = int(component_areas[component_index]) if component_index < component_areas.shape[0] else 0
        if area >= min_area_cells:
            output |= labels == component_index
    if not np.any(output):
        largest_index = int(np.argmax(component_areas[1:]) + 1) if component_areas.shape[0] > 1 else 0
        if largest_index > 0:
            output = labels == largest_index
    return output


def _build_occupancy_grid(
    map_points_xy: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hit_counts, origin_xy = _build_hit_count_grid(
        map_points_xy,
        resolution_m=resolution_m,
        margin_m=margin_m,
        min_extent_cells=48,
    )
    support = convolve((hit_counts > 0).astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant", cval=0)
    occupancy = (hit_counts >= 2) | ((hit_counts > 0) & (support >= 3))
    occupancy = binary_closing(occupancy, structure=np.ones((3, 3), dtype=bool), iterations=1)
    occupancy = binary_dilation(occupancy, iterations=1)
    occupancy = _remove_small_components(occupancy, min_area_cells=6)
    distance_field = distance_transform_edt(~occupancy) * resolution_m
    return (
        occupancy.astype(bool),
        distance_field.astype(np.float32),
        origin_xy.astype(np.float64),
        hit_counts.astype(np.int32),
    )


def _occupied_cell_centers(mask: np.ndarray, origin_xy: np.ndarray, resolution_m: float) -> np.ndarray:
    occupied_y, occupied_x = np.nonzero(mask)
    if occupied_x.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    points_x = origin_xy[0] + ((occupied_x.astype(np.float64) + 0.5) * resolution_m)
    points_y = origin_xy[1] + ((occupied_y.astype(np.float64) + 0.5) * resolution_m)
    return np.column_stack((points_x, points_y))


def _build_visual_points(occupancy: np.ndarray, origin_xy: np.ndarray, resolution_m: float) -> np.ndarray:
    if not np.any(occupancy):
        return np.empty((0, 2), dtype=np.float64)
    eroded = binary_erosion(
        occupancy,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        iterations=1,
        border_value=0,
    )
    boundary = occupancy & ~eroded
    if not np.any(boundary):
        boundary = occupancy.copy()
    return _occupied_cell_centers(boundary, origin_xy, resolution_m)


def _write_pgm(path: Path, occupancy: np.ndarray) -> None:
    image = np.where(occupancy, 0, 254).astype(np.uint8)
    pgm = np.flipud(image)
    with path.open("wb") as handle:
        handle.write(f"P5\n{pgm.shape[1]} {pgm.shape[0]}\n255\n".encode("ascii"))
        handle.write(pgm.tobytes())


def _write_keyframes_csv(path: Path, keyframes: list[Keyframe], poses_xyyaw: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "keyframe_index",
                "scan_index",
                "stamp_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "prior_x_m",
                "prior_y_m",
                "prior_yaw_rad",
            ]
        )
        for keyframe_index, keyframe in enumerate(keyframes):
            pose = poses_xyyaw[keyframe_index]
            writer.writerow(
                [
                    keyframe_index,
                    keyframe.scan_index,
                    f"{keyframe.t_s:.9f}",
                    f"{pose[0]:.6f}",
                    f"{pose[1]:.6f}",
                    f"{pose[2]:.6f}",
                    f"{keyframe.prior_pose_xyyaw[0]:.6f}",
                    f"{keyframe.prior_pose_xyyaw[1]:.6f}",
                    f"{keyframe.prior_pose_xyyaw[2]:.6f}",
                ]
            )


def _write_visual_points_csv(path: Path, points_xy: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m"])
        for point in points_xy:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}"])


def _write_pose_graph_edges(path: Path, edges: list[PoseGraphEdge]) -> None:
    payload = [edge.to_json_dict() for edge in edges]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_map_yaml(
    path: Path,
    *,
    image_name: str,
    resolution_m: float,
    origin_xy: np.ndarray,
    distance_field_name: str,
    visual_points_name: str,
    optimized_keyframes_name: str,
    initial_pose_xyyaw: np.ndarray,
) -> None:
    payload = {
        "image": image_name,
        "resolution": float(resolution_m),
        "origin": [float(origin_xy[0]), float(origin_xy[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "distance_field_npy": distance_field_name,
        "visual_points_csv": visual_points_name,
        "optimized_keyframes_csv": optimized_keyframes_name,
        "initial_pose": [
            float(initial_pose_xyyaw[0]),
            float(initial_pose_xyyaw[1]),
            float(initial_pose_xyyaw[2]),
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _parse_pose(element: ET.Element | None) -> tuple[float, float, float]:
    if element is None or not (element.text or "").strip():
        return 0.0, 0.0, 0.0
    parts = [float(value) for value in (element.text or "").split()]
    while len(parts) < 6:
        parts.append(0.0)
    return parts[0], parts[1], parts[5]


def _compose_pose_2d(
    base: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    cos_yaw = math.cos(base[2])
    sin_yaw = math.sin(base[2])
    x = base[0] + (cos_yaw * offset[0]) - (sin_yaw * offset[1])
    y = base[1] + (sin_yaw * offset[0]) + (cos_yaw * offset[1])
    yaw = base[2] + offset[2]
    return x, y, yaw


def _rectangle_segments(
    *,
    center_x: float,
    center_y: float,
    yaw_rad: float,
    size_x: float,
    size_y: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    half_x = 0.5 * size_x
    half_y = 0.5 * size_y
    local_corners = [
        (half_x, half_y),
        (-half_x, half_y),
        (-half_x, -half_y),
        (half_x, -half_y),
    ]
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_corners: list[tuple[float, float]] = []
    for x_local, y_local in local_corners:
        world_corners.append(
            (
                center_x + (cos_yaw * x_local) - (sin_yaw * y_local),
                center_y + (sin_yaw * x_local) + (cos_yaw * y_local),
            )
        )
    return [
        (world_corners[0], world_corners[1]),
        (world_corners[1], world_corners[2]),
        (world_corners[2], world_corners[3]),
        (world_corners[3], world_corners[0]),
    ]


def _sample_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    sample_step_m: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for a, b in segments:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        samples = max(2, int(math.ceil(length / sample_step_m)))
        for idx in range(samples):
            ratio = idx / float(samples - 1)
            points.append(
                (
                    a[0] + (ratio * (b[0] - a[0])),
                    a[1] + (ratio * (b[1] - a[1])),
                )
            )
    return points


def _load_world_geometry(world_path: Path, *, sample_step_m: float = 0.05) -> np.ndarray:
    if not world_path.exists():
        return np.empty((0, 2), dtype=np.float64)
    root = ET.fromstring(world_path.read_text(encoding="utf-8"))
    sampled_points: list[tuple[float, float]] = []
    for world in root.findall("world"):
        for model in world.findall("model"):
            if (model.findtext("static", default="false").strip().lower()) != "true":
                continue
            model_pose = _parse_pose(model.find("pose"))
            for link in model.findall("link"):
                link_pose = _compose_pose_2d(model_pose, _parse_pose(link.find("pose")))
                for collision in link.findall("collision"):
                    size_text = collision.findtext("geometry/box/size", default="").strip()
                    if not size_text:
                        continue
                    size = [float(value) for value in size_text.split()]
                    if len(size) < 3 or size[2] < 0.08:
                        continue
                    collision_pose = _compose_pose_2d(
                        link_pose,
                        _parse_pose(collision.find("pose")),
                    )
                    rect_segments = _rectangle_segments(
                        center_x=collision_pose[0],
                        center_y=collision_pose[1],
                        yaw_rad=collision_pose[2],
                        size_x=size[0],
                        size_y=size[1],
                    )
                    sampled_points.extend(_sample_segments(rect_segments, sample_step_m))
    if not sampled_points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(sampled_points, dtype=np.float64)


def _principal_axis_angle(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0
    centered = points_xy - np.mean(points_xy, axis=0)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _estimate_alignment_to_world(
    predicted_visual_points_xy: np.ndarray,
    gt_points_xy: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    if predicted_visual_points_xy.shape[0] < 8 or gt_points_xy.shape[0] < 8:
        return 0.0, np.zeros(2, dtype=np.float64), float("inf")

    pred_sample = _subsample_evenly(predicted_visual_points_xy, 4000)
    gt_sample = _subsample_evenly(gt_points_xy, 4000)
    gt_tree = cKDTree(gt_sample)
    pred_angle = _principal_axis_angle(pred_sample)
    gt_angle = _principal_axis_angle(gt_sample)
    initial_yaws = [gt_angle - pred_angle + (0.5 * math.pi * k) for k in range(4)]

    best_yaw = 0.0
    best_translation = np.zeros(2, dtype=np.float64)
    best_chamfer = float("inf")

    for initial_yaw in initial_yaws:
        rotation = _rotation_matrix(initial_yaw)
        translation = np.mean(gt_sample, axis=0) - (np.mean(pred_sample, axis=0) @ rotation.T)
        current_yaw = initial_yaw
        current_translation = translation.astype(np.float64)
        for _ in range(10):
            transformed = _apply_rigid_transform(pred_sample, current_yaw, current_translation)
            _, indexes = gt_tree.query(transformed, k=1)
            matched = gt_sample[indexes]
            refined_yaw, refined_translation = _rigid_transform_from_correspondences(pred_sample, matched)
            delta_yaw = abs(_normalize_angle(refined_yaw - current_yaw))
            delta_translation = float(np.linalg.norm(refined_translation - current_translation))
            current_yaw = refined_yaw
            current_translation = refined_translation
            if delta_yaw <= math.radians(0.05) and delta_translation <= 1.0e-3:
                break

        transformed_pred = _apply_rigid_transform(pred_sample, current_yaw, current_translation)
        pred_tree = cKDTree(transformed_pred)
        d_pred_to_gt, _ = gt_tree.query(transformed_pred, k=1)
        d_gt_to_pred, _ = pred_tree.query(gt_sample, k=1)
        chamfer = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
        if chamfer < best_chamfer:
            best_chamfer = chamfer
            best_yaw = current_yaw
            best_translation = current_translation.copy()

    return best_yaw, best_translation, best_chamfer


def _rasterize_points_to_grid(
    points_xy: np.ndarray,
    *,
    origin_xy: np.ndarray,
    resolution_m: float,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape_hw, dtype=bool)
    if points_xy.size == 0:
        return grid
    gx = np.floor((points_xy[:, 0] - origin_xy[0]) / resolution_m).astype(np.int32)
    gy = np.floor((points_xy[:, 1] - origin_xy[1]) / resolution_m).astype(np.int32)
    valid = (
        (gx >= 0)
        & (gx < shape_hw[1])
        & (gy >= 0)
        & (gy < shape_hw[0])
    )
    grid[gy[valid], gx[valid]] = True
    return grid


def _evaluate_against_world(
    *,
    world_path: Path,
    occupancy: np.ndarray,
    origin_xy: np.ndarray,
    resolution_m: float,
    visual_points_xy: np.ndarray,
    optimized_poses_xyyaw: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    gt_points_xy = _load_world_geometry(world_path, sample_step_m=0.05)
    if gt_points_xy.size == 0:
        return {
            "world_path": str(world_path),
            "available": False,
        }

    predicted_occ_points_xy = _occupied_cell_centers(occupancy, origin_xy, resolution_m)
    align_yaw, align_translation, icp_chamfer = _estimate_alignment_to_world(
        visual_points_xy if visual_points_xy.shape[0] > 0 else predicted_occ_points_xy,
        gt_points_xy,
    )
    aligned_pred_occ_points = _apply_rigid_transform(predicted_occ_points_xy, align_yaw, align_translation)
    aligned_pred_visual_points = _apply_rigid_transform(
        visual_points_xy if visual_points_xy.shape[0] > 0 else predicted_occ_points_xy,
        align_yaw,
        align_translation,
    )

    min_xy = np.minimum(np.min(gt_points_xy, axis=0), np.min(aligned_pred_occ_points, axis=0)) - (2.0 * resolution_m)
    max_xy = np.maximum(np.max(gt_points_xy, axis=0), np.max(aligned_pred_occ_points, axis=0)) + (2.0 * resolution_m)
    width = max(32, int(math.ceil((max_xy[0] - min_xy[0]) / resolution_m)) + 1)
    height = max(32, int(math.ceil((max_xy[1] - min_xy[1]) / resolution_m)) + 1)

    gt_grid = _rasterize_points_to_grid(
        gt_points_xy,
        origin_xy=min_xy,
        resolution_m=resolution_m,
        shape_hw=(height, width),
    )
    pred_grid = _rasterize_points_to_grid(
        aligned_pred_occ_points,
        origin_xy=min_xy,
        resolution_m=resolution_m,
        shape_hw=(height, width),
    )
    gt_dilated = binary_dilation(gt_grid, iterations=1)
    pred_dilated = binary_dilation(pred_grid, iterations=1)
    intersection = int(np.count_nonzero(pred_grid & gt_grid))
    union = int(np.count_nonzero(pred_grid | gt_grid))
    raw_iou = float(intersection / union) if union > 0 else 0.0
    dilated_intersection = int(np.count_nonzero(pred_dilated & gt_dilated))
    dilated_union = int(np.count_nonzero(pred_dilated | gt_dilated))
    dilated_iou = float(dilated_intersection / dilated_union) if dilated_union > 0 else 0.0
    precision_den = int(np.count_nonzero(pred_grid))
    recall_den = int(np.count_nonzero(gt_grid))
    wall_precision = (
        float(np.count_nonzero(pred_grid & gt_dilated) / precision_den)
        if precision_den > 0
        else 0.0
    )
    wall_recall = (
        float(np.count_nonzero(gt_grid & pred_dilated) / recall_den)
        if recall_den > 0
        else 0.0
    )

    pred_tree = cKDTree(aligned_pred_visual_points) if aligned_pred_visual_points.shape[0] > 0 else None
    gt_tree = cKDTree(gt_points_xy)
    pred_to_gt = gt_tree.query(aligned_pred_visual_points, k=1)[0] if aligned_pred_visual_points.shape[0] > 0 else np.asarray([float("inf")])
    gt_to_pred = pred_tree.query(gt_points_xy, k=1)[0] if pred_tree is not None else np.asarray([float("inf")])
    chamfer_distance_m = float(np.mean(pred_to_gt) + np.mean(gt_to_pred))
    closure_gap_m = float(np.linalg.norm(optimized_poses_xyyaw[-1, :2] - optimized_poses_xyyaw[0, :2]))
    closure_yaw_gap_deg = math.degrees(
        abs(_normalize_angle(float(optimized_poses_xyyaw[-1, 2] - optimized_poses_xyyaw[0, 2])))
    )

    overlay_path = output_dir / "mapping_vs_gazebo_overlay.png"
    csv_path = output_dir / "mapping_vs_gazebo_metrics.csv"
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 1] = gt_grid.astype(np.float32)
    rgb[..., 0] = pred_grid.astype(np.float32)
    rgb[..., 2] = (gt_grid & pred_grid).astype(np.float32)
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    axis.imshow(
        np.flipud(rgb),
        extent=[min_xy[0], max_xy[0], min_xy[1], max_xy[1]],
        interpolation="nearest",
    )
    axis.set_title(
        "Offline Map vs Gazebo\n"
        f"IoU={raw_iou:.3f} dilIoU={dilated_iou:.3f} "
        f"prec={wall_precision:.3f} rec={wall_recall:.3f} "
        f"chamfer={chamfer_distance_m:.3f}m"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(overlay_path, dpi=160)
    plt.close(figure)

    csv_path.write_text(
        "metric,value\n"
        f"raw_iou,{raw_iou:.6f}\n"
        f"dilated_iou,{dilated_iou:.6f}\n"
        f"wall_precision,{wall_precision:.6f}\n"
        f"wall_recall,{wall_recall:.6f}\n"
        f"chamfer_distance_m,{chamfer_distance_m:.6f}\n"
        f"closure_gap_m,{closure_gap_m:.6f}\n"
        f"closure_yaw_gap_deg,{closure_yaw_gap_deg:.6f}\n"
        f"icp_alignment_chamfer_m,{icp_chamfer:.6f}\n",
        encoding="utf-8",
    )

    return {
        "available": True,
        "world_path": str(world_path),
        "raw_iou": raw_iou,
        "dilated_iou": dilated_iou,
        "wall_precision": wall_precision,
        "wall_recall": wall_recall,
        "chamfer_distance_m": chamfer_distance_m,
        "closure_gap_m": closure_gap_m,
        "closure_yaw_gap_deg": closure_yaw_gap_deg,
        "icp_alignment_chamfer_m": icp_chamfer,
        "evaluation_alignment_xyyaw": [
            float(align_translation[0]),
            float(align_translation[1]),
            float(align_yaw),
        ],
        "files": {
            "mapping_vs_gazebo_overlay_png": str(overlay_path),
            "mapping_vs_gazebo_metrics_csv": str(csv_path),
        },
    }


def _candidate_self_consistency(
    *,
    occupancy: np.ndarray,
    hit_counts: np.ndarray,
    optimized_poses_xyyaw: np.ndarray,
    local_edges: list[PoseGraphEdge],
    loop_edges: list[PoseGraphEdge],
    segment_edges: list[PoseGraphEdge],
) -> dict[str, object]:
    occupied_cells = int(np.count_nonzero(occupancy))
    supported_cells = int(np.count_nonzero(hit_counts >= 2))
    labels, component_count = label(occupancy, structure=generate_binary_structure(2, 2))
    del labels
    support_ratio = float(supported_cells / occupied_cells) if occupied_cells > 0 else 0.0
    closure_gap_m = float(np.linalg.norm(optimized_poses_xyyaw[-1, :2] - optimized_poses_xyyaw[0, :2]))
    local_residuals = np.asarray(
        [edge.residual_m for edge in local_edges if math.isfinite(edge.residual_m)],
        dtype=np.float64,
    )
    loop_residuals = np.asarray(
        [edge.residual_m for edge in loop_edges if math.isfinite(edge.residual_m)],
        dtype=np.float64,
    )
    segment_residuals = np.asarray(
        [edge.residual_m for edge in segment_edges if math.isfinite(edge.residual_m)],
        dtype=np.float64,
    )
    return {
        "support_ratio": support_ratio,
        "component_count": int(component_count),
        "closure_gap_m": closure_gap_m,
        "median_local_residual_m": float(np.median(local_residuals)) if local_residuals.size > 0 else float("inf"),
        "median_loop_residual_m": float(np.median(loop_residuals)) if loop_residuals.size > 0 else float("inf"),
        "median_segment_residual_m": float(np.median(segment_residuals)) if segment_residuals.size > 0 else float("inf"),
        "accepted_loop_count": len(loop_edges),
        "accepted_segment_count": len(segment_edges),
    }


def _candidate_self_score(consistency: dict[str, object]) -> tuple[float, float, float, float, float]:
    loop_residual = float(consistency.get("median_loop_residual_m", 1.0e9) or 1.0e9)
    if not math.isfinite(loop_residual):
        loop_residual = 1.0e9
    segment_residual = float(consistency.get("median_segment_residual_m", 1.0e9) or 1.0e9)
    if not math.isfinite(segment_residual):
        segment_residual = 1.0e9
    return (
        float(consistency.get("support_ratio", 0.0) or 0.0),
        float(consistency.get("accepted_segment_count", 0) or 0),
        -float(consistency.get("component_count", 1.0e9) or 1.0e9),
        -min(
            float(consistency.get("median_local_residual_m", 1.0e9) or 1.0e9),
            segment_residual,
        ),
        -loop_residual,
    )


def _candidate_score_tuple(evaluation_payload: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(evaluation_payload.get("dilated_iou", 0.0) or 0.0),
        -float(evaluation_payload.get("chamfer_distance_m", 1.0e9) or 1.0e9),
        -float(evaluation_payload.get("closure_gap_m", 1.0e9) or 1.0e9),
    )


def _run_mapping(args: argparse.Namespace) -> dict[str, object]:
    overall_start = time.monotonic()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (run_dir / "fixed_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_json.resolve() if args.status_json is not None else None
    evaluation_json_path = (
        args.evaluation_json.resolve()
        if args.evaluation_json is not None
        else (output_dir / "mapping_evaluation.json")
    )

    def _emit_status(stage: str, **extra: object) -> None:
        payload = {
            "stage": stage,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "elapsed_s": float(time.monotonic() - overall_start),
            "local_registration": args.local_registration,
            "loop_closure_descriptor": args.loop_closure_descriptor,
            "optimizer_loss": args.optimizer_loss,
        }
        payload.update(extra)
        _write_status(status_path, payload)

    lidar_path = run_dir / "lidar_points.csv"
    odom_prior_path = run_dir / "odom_fused.csv"
    imu_path = run_dir / "imu_raw.csv"

    _emit_status("loading_lidar")
    print("[general_mapper] cargando scans LiDAR...", flush=True)
    lidar_scans = _load_lidar_scans(lidar_path)
    if not lidar_scans:
        raise SystemExit(f"No se encontraron scans válidos en {lidar_path}")
    print(f"[general_mapper] scans cargados: {len(lidar_scans)}", flush=True)

    _emit_status("loading_imu_and_odom_prior", scan_count=len(lidar_scans))
    print("[general_mapper] cargando IMU y prior de odometría...", flush=True)
    imu_samples = _load_imu_samples(imu_path)
    odom_prior_raw = _load_odom_prior(odom_prior_path)
    odom_prior = _preprocess_odom_prior(odom_prior_raw, imu_samples)
    print(
        f"[general_mapper] muestras IMU={len(imu_samples)} prior_raw={len(odom_prior_raw)} prior_filtrado={len(odom_prior)}",
        flush=True,
    )

    scan_frames = _build_scan_frames(lidar_scans, odom_prior)
    _emit_status(
        "selecting_keyframes",
        scan_count=len(scan_frames),
        odom_prior_count=len(odom_prior),
        imu_sample_count=len(imu_samples),
    )
    print("[general_mapper] seleccionando keyframes...", flush=True)
    keyframes = _select_keyframes(
        scan_frames,
        distance_threshold_m=float(args.keyframe_distance_m),
        yaw_threshold_rad=math.radians(float(args.keyframe_yaw_deg)),
        time_threshold_s=float(args.keyframe_time_s),
        voxel_size_m=float(args.voxel_size_m),
        max_points_per_keyframe=int(args.max_points_per_keyframe),
        max_keyframes=int(args.max_keyframes),
    )
    print(f"[general_mapper] keyframes seleccionados: {len(keyframes)}", flush=True)

    if len(keyframes) < 2:
        raise SystemExit("No hay suficientes keyframes para construir el mapa.")
    prior_poses = np.vstack([keyframe.prior_pose_xyyaw for keyframe in keyframes]).astype(np.float64)
    quality_counts = {
        label: sum(1 for keyframe in keyframes if keyframe.prior_quality_label == label)
        for label in ("high", "reduced", "low")
    }
    print(
        "[general_mapper] calidad prior keyframes: high=%d reduced=%d low=%d"
        % (
            quality_counts["high"],
            quality_counts["reduced"],
            quality_counts["low"],
        ),
        flush=True,
    )

    _emit_status(
        "sequential_refine",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=0,
        loop_closure_count=0,
    )
    print("[general_mapper] refinando trayectoria secuencial...", flush=True)
    sequential_poses, local_edges, odom_edges = _sequential_refine_keyframes(
        keyframes,
        submap_keyframes=int(args.submap_keyframes),
        max_correspondence_m=float(args.max_correspondence_m),
        local_registration=str(args.local_registration),
        loss=str(args.optimizer_loss),
        status_callback=lambda stage, index, total: _emit_status(
            stage,
            scan_count=len(scan_frames),
            keyframe_count=len(keyframes),
            progress_index=int(index),
            progress_total=int(total),
            edge_count=0,
            loop_closure_count=0,
        ),
    )

    _emit_status(
        "building_submap_cache",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(local_edges) + len(odom_edges),
        loop_closure_count=0,
    )
    print("[general_mapper] construyendo cache de submapas...", flush=True)
    submap_descriptor_cache = _build_submap_descriptor_cache(
        keyframes,
        sequential_poses,
        radius=2,
        output_dir=output_dir,
    )

    _emit_status(
        "loop_closure_search",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(local_edges) + len(odom_edges),
        loop_closure_count=0,
    )
    print("[general_mapper] detectando loop closures...", flush=True)
    loop_edges = _detect_loop_closures(
        keyframes,
        sequential_poses,
        submap_keyframes=int(args.submap_keyframes),
        min_index_gap=int(args.loop_closure_min_separation),
        max_correspondence_m=float(args.max_correspondence_m),
        max_edges_per_keyframe=int(args.max_loop_closures_per_keyframe),
        descriptor_mode=str(args.loop_closure_descriptor),
        submap_descriptor_cache=submap_descriptor_cache,
        status_callback=lambda stage, index, total: _emit_status(
            stage,
            scan_count=len(scan_frames),
            keyframe_count=len(keyframes),
            progress_index=int(index),
            progress_total=int(total),
            edge_count=len(local_edges) + len(odom_edges),
            loop_closure_count=0,
        ),
    )
    print(f"[general_mapper] loop closures aceptados: {len(loop_edges)}", flush=True)

    _emit_status(
        "segment_consistency_search",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(local_edges) + len(odom_edges) + len(loop_edges),
        loop_closure_count=len(loop_edges),
        segment_consistency_count=0,
    )
    print("[general_mapper] detectando consistencia por segmentos...", flush=True)
    segment_edges = _detect_segment_consistency_edges(
        keyframes,
        sequential_poses,
        submap_keyframes=int(args.submap_keyframes),
        min_index_gap=max(20, int(args.loop_closure_min_separation)),
        max_correspondence_m=float(args.max_correspondence_m),
        max_edges_per_keyframe=1,
        status_callback=lambda stage, index, total: _emit_status(
            stage,
            scan_count=len(scan_frames),
            keyframe_count=len(keyframes),
            progress_index=int(index),
            progress_total=int(total),
            edge_count=len(local_edges) + len(odom_edges) + len(loop_edges),
            loop_closure_count=len(loop_edges),
            segment_consistency_count=0,
        ),
    )
    print(f"[general_mapper] aristas de segmento aceptadas: {len(segment_edges)}", flush=True)

    _emit_status(
        "pose_graph_optimization",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(odom_edges) + len(local_edges) + len(loop_edges) + len(segment_edges),
        loop_closure_count=len(loop_edges),
        segment_consistency_count=len(segment_edges),
    )
    print("[general_mapper] optimizando pose graph...", flush=True)
    optimized_pose_graph_poses = _optimize_pose_graph(
        sequential_poses,
        odom_edges + local_edges + loop_edges + segment_edges,
        loss=str(args.optimizer_loss),
    )

    _emit_status(
        "rasterizing_map",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(odom_edges) + len(local_edges) + len(loop_edges) + len(segment_edges),
        loop_closure_count=len(loop_edges),
        segment_consistency_count=len(segment_edges),
    )
    print("[general_mapper] rasterizando mapa fijo...", flush=True)
    candidate_poses_xyyaw: dict[str, np.ndarray] = {
        "prior_baseline": prior_poses,
        "sequential_refined": sequential_poses,
        "loop_pose_graph": optimized_pose_graph_poses,
    }
    candidate_maps: dict[str, dict[str, np.ndarray]] = {}
    candidate_self_consistency: dict[str, dict[str, object]] = {}
    for candidate_name, candidate_poses in candidate_poses_xyyaw.items():
        map_points_xy = _derive_map_points(keyframes, candidate_poses, local_edges)
        occupancy, distance_field, origin_xy, hit_counts = _build_occupancy_grid(
            map_points_xy,
            resolution_m=float(args.map_resolution_m),
            margin_m=float(args.map_margin_m),
        )
        visual_points_xy = _build_visual_points(occupancy, origin_xy, float(args.map_resolution_m))
        candidate_self_consistency[candidate_name] = _candidate_self_consistency(
            occupancy=occupancy,
            hit_counts=hit_counts,
            optimized_poses_xyyaw=candidate_poses,
            local_edges=local_edges,
            loop_edges=loop_edges if candidate_name == "loop_pose_graph" else [],
            segment_edges=segment_edges if candidate_name == "loop_pose_graph" else [],
        )
        candidate_maps[candidate_name] = {
            "occupancy": occupancy,
            "distance_field": distance_field,
            "origin_xy": origin_xy,
            "hit_counts": hit_counts,
            "visual_points_xy": visual_points_xy,
        }

    keyframes_csv = output_dir / "optimized_keyframes.csv"
    edges_json = output_dir / "pose_graph_edges.json"
    pgm_path = output_dir / "fixed_map.pgm"
    yaml_path = output_dir / "fixed_map.yaml"
    distance_path = output_dir / "fixed_map_distance.npy"
    visual_points_csv = output_dir / "fixed_map_visual_points.csv"
    summary_json = output_dir / "mapping_summary.json"

    _emit_status(
        "evaluating_against_world",
        scan_count=len(scan_frames),
        keyframe_count=len(keyframes),
        edge_count=len(odom_edges) + len(local_edges) + len(loop_edges) + len(segment_edges),
        loop_closure_count=len(loop_edges),
        segment_consistency_count=len(segment_edges),
    )
    evaluation_payload: dict[str, object] = {"available": False}
    candidate_evaluations: dict[str, dict[str, object]] = {}
    selected_candidate_name = "sequential_refined"
    if args.evaluation_world is not None:
        print("[general_mapper] evaluando contra geometría de Gazebo...", flush=True)
        for candidate_name, candidate_poses in candidate_poses_xyyaw.items():
            candidate_output_dir = output_dir / f"candidate_{candidate_name}"
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            candidate_map = candidate_maps[candidate_name]
            candidate_evaluations[candidate_name] = _evaluate_against_world(
                world_path=args.evaluation_world,
                occupancy=candidate_map["occupancy"],
                origin_xy=candidate_map["origin_xy"],
                resolution_m=float(args.map_resolution_m),
                visual_points_xy=candidate_map["visual_points_xy"],
                optimized_poses_xyyaw=candidate_poses,
                output_dir=candidate_output_dir,
            )
        selected_candidate_name = max(
            candidate_evaluations,
            key=lambda name: _candidate_score_tuple(candidate_evaluations[name]),
        )
        evaluation_payload = candidate_evaluations[selected_candidate_name]
    else:
        selected_candidate_name = max(
            candidate_self_consistency,
            key=lambda name: _candidate_self_score(candidate_self_consistency[name]),
        )

    selected_poses = candidate_poses_xyyaw[selected_candidate_name]
    selected_map = candidate_maps[selected_candidate_name]
    occupancy = selected_map["occupancy"]
    distance_field = selected_map["distance_field"]
    origin_xy = selected_map["origin_xy"]
    visual_points_xy = selected_map["visual_points_xy"]
    hit_counts = selected_map["hit_counts"]

    _write_keyframes_csv(keyframes_csv, keyframes, selected_poses)
    _write_pose_graph_edges(edges_json, odom_edges + local_edges + loop_edges + segment_edges)
    _write_pgm(pgm_path, occupancy)
    np.save(distance_path, distance_field)
    _write_visual_points_csv(visual_points_csv, visual_points_xy)
    _write_map_yaml(
        yaml_path,
        image_name=pgm_path.name,
        resolution_m=float(args.map_resolution_m),
        origin_xy=origin_xy,
        distance_field_name=distance_path.name,
        visual_points_name=visual_points_csv.name,
        optimized_keyframes_name=keyframes_csv.name,
        initial_pose_xyyaw=selected_poses[0],
    )
    evaluation_json_path.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
    if evaluation_payload.get("available"):
        overlay_src = Path(str(((evaluation_payload.get("files") or {}).get("mapping_vs_gazebo_overlay_png"))))
        metrics_src = Path(str(((evaluation_payload.get("files") or {}).get("mapping_vs_gazebo_metrics_csv"))))
        if overlay_src.is_file():
            shutil.copy2(overlay_src, output_dir / "mapping_vs_gazebo_overlay.png")
        if metrics_src.is_file():
            shutil.copy2(metrics_src, output_dir / "mapping_vs_gazebo_metrics.csv")

    summary_payload = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "scan_count": len(scan_frames),
        "keyframe_count": len(keyframes),
        "edge_count": len(odom_edges) + len(local_edges) + len(loop_edges) + len(segment_edges),
        "loop_closure_count": len(loop_edges),
        "segment_consistency_count": len(segment_edges),
        "occupied_cell_count": int(np.count_nonzero(occupancy)),
        "visual_point_count": int(visual_points_xy.shape[0]),
        "map_resolution_m": float(args.map_resolution_m),
        "processing_elapsed_s": float(time.monotonic() - overall_start),
        "local_registration": str(args.local_registration),
        "loop_closure_descriptor": str(args.loop_closure_descriptor),
        "optimizer_loss": str(args.optimizer_loss),
        "prior_quality_counts": quality_counts,
        "selected_candidate": selected_candidate_name,
        "candidate_evaluations": candidate_evaluations,
        "candidate_self_consistency": candidate_self_consistency,
        "evaluation": evaluation_payload,
        "files": {
            "fixed_map_yaml": str(yaml_path),
            "fixed_map_pgm": str(pgm_path),
            "fixed_map_distance_npy": str(distance_path),
            "fixed_map_visual_points_csv": str(visual_points_csv),
            "optimized_keyframes_csv": str(keyframes_csv),
            "pose_graph_edges_json": str(edges_json),
            "mapping_evaluation_json": str(evaluation_json_path),
            "mapping_vs_gazebo_overlay_png": str(output_dir / "mapping_vs_gazebo_overlay.png"),
            "mapping_vs_gazebo_metrics_csv": str(output_dir / "mapping_vs_gazebo_metrics.csv"),
            "submap_cache_dir": str(output_dir / "submap_cache"),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _emit_status("done", **summary_payload)

    print(
        "[general_mapper] mapa listo: keyframes=%d loop_closures=%d segment_edges=%d occupied_cells=%d elapsed=%s"
        % (
            len(keyframes),
            len(loop_edges),
            len(segment_edges),
            int(np.count_nonzero(occupancy)),
            _format_duration_s(time.monotonic() - overall_start),
        ),
        flush=True,
    )
    print(f"[general_mapper] candidato seleccionado: {selected_candidate_name}", flush=True)
    print(
        "[general_mapper] consistencia propia: support_ratio=%.3f components=%s local_residual=%.3f"
        % (
            float(candidate_self_consistency[selected_candidate_name].get("support_ratio", 0.0)),
            int(candidate_self_consistency[selected_candidate_name].get("component_count", 0)),
            float(candidate_self_consistency[selected_candidate_name].get("median_local_residual_m", 0.0)),
        ),
        flush=True,
    )
    if evaluation_payload.get("available", False):
        print(
            "[general_mapper] evaluación Gazebo: dilated_iou=%.3f precision=%.3f recall=%.3f chamfer=%.3fm closure_gap=%.3fm"
            % (
                float(evaluation_payload.get("dilated_iou", 0.0)),
                float(evaluation_payload.get("wall_precision", 0.0)),
                float(evaluation_payload.get("wall_recall", 0.0)),
                float(evaluation_payload.get("chamfer_distance_m", 0.0)),
                float(evaluation_payload.get("closure_gap_m", 0.0)),
            ),
            flush=True,
        )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--status-json")
    parser.add_argument("--local-registration", default="multires_distance_field")
    parser.add_argument("--loop-closure-descriptor", default="scan_context_2d")
    parser.add_argument("--optimizer-loss", default="cauchy")
    parser.add_argument("--evaluation-world")
    parser.add_argument("--evaluation-json")
    parser.add_argument("--keyframe-distance-m", type=float, default=0.16)
    parser.add_argument("--keyframe-yaw-deg", type=float, default=6.0)
    parser.add_argument("--keyframe-time-s", type=float, default=1.0)
    parser.add_argument("--voxel-size-m", type=float, default=0.03)
    parser.add_argument("--max-points-per-keyframe", type=int, default=160)
    parser.add_argument("--max-keyframes", type=int, default=220)
    parser.add_argument("--submap-keyframes", type=int, default=10)
    parser.add_argument("--max-correspondence-m", type=float, default=0.30)
    parser.add_argument("--loop-closure-min-separation", type=int, default=25)
    parser.add_argument("--max-loop-closures-per-keyframe", type=int, default=2)
    parser.add_argument("--map-resolution-m", type=float, default=0.04)
    parser.add_argument("--map-margin-m", type=float, default=0.80)
    args = parser.parse_args()

    args.run_dir = Path(args.run_dir).expanduser().resolve()
    args.output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None and str(args.output_dir).strip()
        else None
    )
    args.status_json = (
        Path(args.status_json).expanduser().resolve()
        if args.status_json is not None and str(args.status_json).strip()
        else None
    )
    args.evaluation_world = (
        Path(args.evaluation_world).expanduser().resolve()
        if args.evaluation_world is not None and str(args.evaluation_world).strip()
        else None
    )
    args.evaluation_json = (
        Path(args.evaluation_json).expanduser().resolve()
        if args.evaluation_json is not None and str(args.evaluation_json).strip()
        else None
    )
    _run_mapping(args)


if __name__ == "__main__":
    main()
