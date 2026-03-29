#!/usr/bin/env python3
"""Causal planar LiDAR + IMU fusion primitives used by the online ROS node."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

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
DEFAULT_MAX_CORRESPONDENCE_M = 0.35
DEFAULT_INITIAL_SCAN_COUNT_MIN = 4
DEFAULT_CORRIDOR_BIN_M = 0.10
DEFAULT_CORRIDOR_QUANTILE = 0.18
DEFAULT_LOW_CONFIDENCE_RESIDUAL_M = 0.16
DEFAULT_MAX_INITIAL_ALIGNMENT_SCANS = 6
DEFAULT_MIN_VALID_CORRESPONDENCES = 14

LOCAL_SUBMAP_WEIGHT = 4.0
LOCAL_WALL_WEIGHT = 2.2
LOCAL_YAW_WEIGHT = 8.5
LOCAL_MOTION_WEIGHT = 0.85


@dataclass(frozen=True)
class FusionParameters:
    median_window: int = DEFAULT_MEDIAN_WINDOW
    ema_alpha: float = DEFAULT_EMA_ALPHA
    static_window_s: float = DEFAULT_STATIC_WINDOW_S
    static_search_s: float = DEFAULT_STATIC_SEARCH_S
    velocity_decay_tau_s: float = DEFAULT_VELOCITY_DECAY_TAU_S
    submap_window_scans: int = DEFAULT_SUBMAP_WINDOW_SCANS
    point_stride: int = DEFAULT_SCAN_POINT_STRIDE
    max_correspondence_m: float = DEFAULT_MAX_CORRESPONDENCE_M
    initial_scan_count_min: int = DEFAULT_INITIAL_SCAN_COUNT_MIN
    max_initial_alignment_scans: int = DEFAULT_MAX_INITIAL_ALIGNMENT_SCANS
    corridor_bin_m: float = DEFAULT_CORRIDOR_BIN_M
    low_confidence_residual_m: float = DEFAULT_LOW_CONFIDENCE_RESIDUAL_M
    min_valid_correspondence_count: int = DEFAULT_MIN_VALID_CORRESPONDENCES
    max_scan_optimization_evals: int = 80

    def __post_init__(self) -> None:
        if self.median_window < 1 or (self.median_window % 2) == 0:
            raise ValueError("median_window must be odd and >= 1")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.static_window_s <= 0.0:
            raise ValueError("static_window_s must be positive")
        if self.static_search_s < self.static_window_s:
            raise ValueError("static_search_s must be >= static_window_s")
        if self.velocity_decay_tau_s <= 0.0:
            raise ValueError("velocity_decay_tau_s must be positive")
        if self.submap_window_scans < 2:
            raise ValueError("submap_window_scans must be >= 2")
        if self.point_stride < 1:
            raise ValueError("point_stride must be >= 1")
        if self.max_correspondence_m <= 0.0:
            raise ValueError("max_correspondence_m must be positive")
        if self.initial_scan_count_min < 2:
            raise ValueError("initial_scan_count_min must be >= 2")
        if self.max_initial_alignment_scans < self.initial_scan_count_min:
            raise ValueError("max_initial_alignment_scans must be >= initial_scan_count_min")
        if self.corridor_bin_m <= 0.0:
            raise ValueError("corridor_bin_m must be positive")
        if self.low_confidence_residual_m <= 0.0:
            raise ValueError("low_confidence_residual_m must be positive")
        if self.min_valid_correspondence_count < 1:
            raise ValueError("min_valid_correspondence_count must be >= 1")


@dataclass(frozen=True)
class ImuSeries:
    t_s: np.ndarray
    ax_mps2: np.ndarray
    ay_mps2: np.ndarray
    az_mps2: np.ndarray
    gz_rps: np.ndarray


@dataclass(frozen=True)
class MotionState:
    t_s: float
    yaw_rad: float
    yaw_rate_rps: float
    velocity_mps: np.ndarray
    accel_world_mps2: np.ndarray


@dataclass(frozen=True)
class WallModel:
    lower_coef: np.ndarray
    upper_coef: np.ndarray
    width_m: float
    corridor_yaw_rad: float


@dataclass(frozen=True)
class LidarScanObservation:
    scan_index: int
    stamp_sec: int
    stamp_nanosec: int
    t_s: float
    points_local: np.ndarray
    sampled_points_local: np.ndarray
    lower_wall_points_local: np.ndarray
    upper_wall_points_local: np.ndarray


@dataclass(frozen=True)
class ScanEstimate:
    scan_index: int
    stamp_sec: int
    stamp_nanosec: int
    t_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float
    ax_world_mps2: float
    ay_world_mps2: float
    confidence: str
    median_submap_residual_m: float
    median_wall_residual_m: float
    valid_correspondence_count: int
    alignment_ready: bool
    imu_initialized: bool
    best_effort_init: bool


@dataclass(frozen=True)
class FusionStateSnapshot:
    state: str
    imu_initialized: bool
    alignment_ready: bool
    best_effort_init: bool
    raw_imu_sample_count: int
    processed_imu_sample_count: int
    pending_scan_count: int
    processed_scan_count: int
    initial_scan_count: int
    alignment_yaw_rad: float
    origin_projection_m: tuple[float, float]
    static_initialization: dict[str, Any]
    corridor_model: dict[str, Any] | None
    quality: dict[str, Any]
    latest_pose: dict[str, Any] | None
    parameters: dict[str, Any]


class _MedianEmaFilter:
    def __init__(self, window_size: int, alpha: float) -> None:
        self._window = deque(maxlen=window_size)
        self._alpha = float(alpha)
        self._ema_value: float | None = None

    def reset(self) -> None:
        self._window.clear()
        self._ema_value = None

    def update(self, value: float) -> float:
        self._window.append(float(value))
        median_value = float(median(self._window))
        if self._ema_value is None:
            self._ema_value = median_value
        else:
            self._ema_value = (
                (self._alpha * median_value) + ((1.0 - self._alpha) * self._ema_value)
            )
        return self._ema_value


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


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


def _rotate_points(points: np.ndarray, theta_rad: float) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    rotation = _rotation_matrix(theta_rad)
    return points @ rotation.T


def _rotate_vector(vector_xy: np.ndarray, theta_rad: float) -> np.ndarray:
    return (_rotation_matrix(theta_rad) @ np.asarray(vector_xy, dtype=np.float64).reshape(2)).reshape(
        2
    )


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    rotation = _rotation_matrix(float(pose[2]))
    return (points @ rotation.T) + pose[:2]


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


def scan_observation_from_ranges(
    *,
    scan_index: int,
    stamp_sec: int,
    stamp_nanosec: int,
    angle_min_rad: float,
    angle_increment_rad: float,
    ranges: list[float] | tuple[float, ...],
    range_min_m: float,
    range_max_m: float,
    point_stride: int,
) -> LidarScanObservation:
    points: list[tuple[float, float]] = []
    angle_rad = float(angle_min_rad)
    for raw_range in ranges:
        range_m = float(raw_range)
        if math.isfinite(range_m) and range_min_m <= range_m <= range_max_m:
            x_scan_m = range_m * math.cos(angle_rad)
            y_scan_m = range_m * math.sin(angle_rad)
            points.append((-x_scan_m, y_scan_m))
        angle_rad += float(angle_increment_rad)

    points_local = np.asarray(points, dtype=np.float64)
    if points_local.size == 0:
        points_local = np.empty((0, 2), dtype=np.float64)
    sampled_points_local = points_local[::point_stride].copy()
    lower_wall_points_local, upper_wall_points_local = _extract_sidewall_candidates(points_local)
    return LidarScanObservation(
        scan_index=scan_index,
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        t_s=float(stamp_sec) + (1.0e-9 * float(stamp_nanosec)),
        points_local=points_local,
        sampled_points_local=sampled_points_local,
        lower_wall_points_local=lower_wall_points_local,
        upper_wall_points_local=upper_wall_points_local,
    )


def _fit_line_robust(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    residual_threshold_m: float = 0.10,
    max_iterations: int = 4,
) -> np.ndarray:
    if x_values.size < 2:
        raise ValueError("not enough points to fit a line")

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
        raise ValueError("not enough LiDAR points to fit corridor walls")

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
        raise ValueError("not enough corridor bins to fit walls")

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


def _build_submap_points(
    scans: list[LidarScanObservation],
    poses: list[np.ndarray],
    confidences: list[str],
    current_index: int,
    window_scans: int,
    initial_scan_count: int,
) -> np.ndarray:
    start_index = max(0, current_index - window_scans)
    submap_parts: list[np.ndarray] = []
    for scan_index in range(start_index, current_index):
        if confidences[scan_index] == "low" and scan_index >= initial_scan_count:
            continue
        submap_parts.append(_transform_points(scans[scan_index].sampled_points_local, poses[scan_index]))
    if not submap_parts:
        for scan_index in range(start_index, current_index):
            submap_parts.append(
                _transform_points(scans[scan_index].sampled_points_local, poses[scan_index])
            )
    if not submap_parts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(submap_parts)


def _wall_residuals_for_pose(
    scan: LidarScanObservation,
    pose: np.ndarray,
    wall_model: WallModel,
) -> np.ndarray:
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
    scan: LidarScanObservation,
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
        submap_median = (
            float(np.median(distances[valid_mask])) if valid_count else float("nan")
        )

    wall_residuals = _wall_residuals_for_pose(scan, pose, wall_model)
    wall_median = float(np.median(np.abs(wall_residuals))) if wall_residuals.size else float("nan")
    return submap_median, wall_median, valid_count


def _nanmedian_or_default(values: list[float], default: float = float("nan")) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return default
    return float(np.median(np.asarray(finite, dtype=np.float64)))


class OnlinePlanarFusion:
    """Causal local scan-to-submap fusion using raw IMU + LiDAR scans."""

    def __init__(self, params: FusionParameters) -> None:
        self._params = params

        self._raw_imu_t_s: list[float] = []
        self._raw_ax_mps2: list[float] = []
        self._raw_ay_mps2: list[float] = []
        self._raw_az_mps2: list[float] = []
        self._raw_gz_rps: list[float] = []
        self._raw_imu_processed_index = 0

        self._imu_initialized = False
        self._best_effort_init = False
        self._bias_ax_mps2 = 0.0
        self._bias_ay_mps2 = 0.0
        self._bias_az_mps2 = 0.0
        self._bias_gz_rps = 0.0
        self._static_start_s = 0.0
        self._static_end_s = 0.0
        self._static_sample_count = 0

        self._ax_filter = _MedianEmaFilter(params.median_window, params.ema_alpha)
        self._ay_filter = _MedianEmaFilter(params.median_window, params.ema_alpha)
        self._az_filter = _MedianEmaFilter(params.median_window, params.ema_alpha)
        self._gz_filter = _MedianEmaFilter(params.median_window, params.ema_alpha)

        self._imu_t_s: list[float] = []
        self._yaw_body_rad: list[float] = []
        self._yaw_rate_rps: list[float] = []
        self._vx_start_mps: list[float] = []
        self._vy_start_mps: list[float] = []
        self._ax_start_mps2: list[float] = []
        self._ay_start_mps2: list[float] = []

        self._pending_scans: deque[LidarScanObservation] = deque()
        self._processed_scans: list[LidarScanObservation] = []
        self._poses_internal: list[np.ndarray] = []
        self._scan_estimates: list[ScanEstimate] = []
        self._confidences: list[str] = []

        self._alignment_ready = False
        self._alignment_yaw_rad = 0.0
        self._origin_projection_m = np.zeros(2, dtype=np.float64)
        self._wall_model: WallModel | None = None
        self._initial_scan_count = 0

    def add_imu_sample(
        self,
        *,
        t_s: float,
        ax_mps2: float,
        ay_mps2: float,
        az_mps2: float,
        gz_rps: float,
    ) -> None:
        if self._raw_imu_t_s and t_s <= self._raw_imu_t_s[-1]:
            return

        self._raw_imu_t_s.append(float(t_s))
        self._raw_ax_mps2.append(float(ax_mps2))
        self._raw_ay_mps2.append(float(ay_mps2))
        self._raw_az_mps2.append(float(az_mps2))
        self._raw_gz_rps.append(float(gz_rps))

        if not self._imu_initialized:
            self._try_initialize_imu()
            return

        self._process_pending_raw_imu()

    def add_scan_observation(self, scan: LidarScanObservation) -> list[ScanEstimate]:
        self._pending_scans.append(scan)
        return self._process_pending_scans()

    def status_snapshot(self) -> FusionStateSnapshot:
        latest_estimate = self._scan_estimates[-1] if self._scan_estimates else None
        high_confidence_pct = 100.0 * (
            sum(1 for estimate in self._scan_estimates if estimate.confidence == "high")
            / max(1, len(self._scan_estimates))
        )
        median_submap_residual_m = _nanmedian_or_default(
            [estimate.median_submap_residual_m for estimate in self._scan_estimates]
        )
        median_wall_residual_m = _nanmedian_or_default(
            [estimate.median_wall_residual_m for estimate in self._scan_estimates]
        )

        if not self._imu_initialized:
            state = "waiting_static_initialization"
        elif not self._alignment_ready:
            state = "waiting_alignment"
        else:
            state = "tracking"

        corridor_model_payload: dict[str, Any] | None = None
        if self._wall_model is not None:
            corridor_model_payload = {
                "width_m": self._wall_model.width_m,
                "lower_wall_coef": self._wall_model.lower_coef.tolist(),
                "upper_wall_coef": self._wall_model.upper_coef.tolist(),
                "alignment_yaw_rad": self._alignment_yaw_rad,
                "origin_projection_m": self._origin_projection_m.tolist(),
            }

        latest_pose_payload: dict[str, Any] | None = None
        if latest_estimate is not None:
            latest_pose_payload = {
                "scan_index": latest_estimate.scan_index,
                "t_s": latest_estimate.t_s,
                "x_m": latest_estimate.x_m,
                "y_m": latest_estimate.y_m,
                "yaw_rad": latest_estimate.yaw_rad,
                "vx_mps": latest_estimate.vx_mps,
                "vy_mps": latest_estimate.vy_mps,
                "yaw_rate_rps": latest_estimate.yaw_rate_rps,
                "confidence": latest_estimate.confidence,
                "median_submap_residual_m": latest_estimate.median_submap_residual_m,
                "median_wall_residual_m": latest_estimate.median_wall_residual_m,
                "valid_correspondence_count": latest_estimate.valid_correspondence_count,
            }

        return FusionStateSnapshot(
            state=state,
            imu_initialized=self._imu_initialized,
            alignment_ready=self._alignment_ready,
            best_effort_init=self._best_effort_init,
            raw_imu_sample_count=len(self._raw_imu_t_s),
            processed_imu_sample_count=len(self._imu_t_s),
            pending_scan_count=len(self._pending_scans),
            processed_scan_count=len(self._processed_scans),
            initial_scan_count=self._initial_scan_count,
            alignment_yaw_rad=self._alignment_yaw_rad,
            origin_projection_m=(
                float(self._origin_projection_m[0]),
                float(self._origin_projection_m[1]),
            ),
            static_initialization={
                "best_effort_init": self._best_effort_init,
                "static_start_s": self._static_start_s,
                "static_end_s": self._static_end_s,
                "static_sample_count": self._static_sample_count,
                "bias_ax_mps2": self._bias_ax_mps2,
                "bias_ay_mps2": self._bias_ay_mps2,
                "bias_az_mps2": self._bias_az_mps2 - GRAVITY_MPS2,
                "bias_gz_rps": self._bias_gz_rps,
            },
            corridor_model=corridor_model_payload,
            quality={
                "high_confidence_pct": high_confidence_pct,
                "low_confidence_scan_count": sum(
                    1 for estimate in self._scan_estimates if estimate.confidence == "low"
                ),
                "median_submap_residual_m": median_submap_residual_m,
                "median_wall_residual_m": median_wall_residual_m,
            },
            latest_pose=latest_pose_payload,
            parameters=asdict(self._params),
        )

    def _try_initialize_imu(self) -> None:
        if len(self._raw_imu_t_s) < 8:
            return

        elapsed_s = float(self._raw_imu_t_s[-1] - self._raw_imu_t_s[0])
        if elapsed_s < self._params.static_window_s:
            return

        imu_series = ImuSeries(
            t_s=np.asarray(self._raw_imu_t_s, dtype=np.float64),
            ax_mps2=np.asarray(self._raw_ax_mps2, dtype=np.float64),
            ay_mps2=np.asarray(self._raw_ay_mps2, dtype=np.float64),
            az_mps2=np.asarray(self._raw_az_mps2, dtype=np.float64),
            gz_rps=np.asarray(self._raw_gz_rps, dtype=np.float64),
        )
        static_start_index, static_end_index, best_effort_init = _detect_static_window(
            imu_series,
            window_s=self._params.static_window_s,
            search_s=self._params.static_search_s,
        )
        search_complete = elapsed_s >= self._params.static_search_s
        if best_effort_init and not search_complete:
            return

        self._bias_ax_mps2 = float(np.mean(imu_series.ax_mps2[static_start_index:static_end_index]))
        self._bias_ay_mps2 = float(np.mean(imu_series.ay_mps2[static_start_index:static_end_index]))
        self._bias_az_mps2 = float(np.mean(imu_series.az_mps2[static_start_index:static_end_index]))
        self._bias_gz_rps = float(np.mean(imu_series.gz_rps[static_start_index:static_end_index]))
        self._best_effort_init = bool(best_effort_init)
        self._static_start_s = float(imu_series.t_s[static_start_index])
        self._static_end_s = float(imu_series.t_s[static_end_index - 1])
        self._static_sample_count = int(static_end_index - static_start_index)
        self._imu_initialized = True

        self._ax_filter.reset()
        self._ay_filter.reset()
        self._az_filter.reset()
        self._gz_filter.reset()
        self._imu_t_s.clear()
        self._yaw_body_rad.clear()
        self._yaw_rate_rps.clear()
        self._vx_start_mps.clear()
        self._vy_start_mps.clear()
        self._ax_start_mps2.clear()
        self._ay_start_mps2.clear()
        self._raw_imu_processed_index = 0
        self._process_pending_raw_imu()

    def _process_pending_raw_imu(self) -> None:
        while self._raw_imu_processed_index < len(self._raw_imu_t_s):
            index = self._raw_imu_processed_index
            self._process_single_imu_sample(
                t_s=self._raw_imu_t_s[index],
                ax_raw_mps2=self._raw_ax_mps2[index],
                ay_raw_mps2=self._raw_ay_mps2[index],
                az_raw_mps2=self._raw_az_mps2[index],
                gz_raw_rps=self._raw_gz_rps[index],
            )
            self._raw_imu_processed_index += 1

    def _process_single_imu_sample(
        self,
        *,
        t_s: float,
        ax_raw_mps2: float,
        ay_raw_mps2: float,
        az_raw_mps2: float,
        gz_raw_rps: float,
    ) -> None:
        ax_filtered = self._ax_filter.update(ax_raw_mps2 - self._bias_ax_mps2)
        ay_filtered = self._ay_filter.update(ay_raw_mps2 - self._bias_ay_mps2)
        az_filtered = self._az_filter.update(az_raw_mps2 - self._bias_az_mps2)
        gz_filtered = self._gz_filter.update(gz_raw_rps - self._bias_gz_rps)

        if not self._imu_t_s:
            yaw_body_rad = 0.0
            ax_start, ay_start = _rotate_vector(
                np.asarray([ax_filtered, ay_filtered], dtype=np.float64),
                yaw_body_rad,
            )
            vx_start = 0.0
            vy_start = 0.0
        else:
            dt_s = max(1.0e-6, float(t_s - self._imu_t_s[-1]))
            yaw_body_rad = self._yaw_body_rad[-1] + (
                0.5 * dt_s * (self._yaw_rate_rps[-1] + gz_filtered)
            )
            yaw_body_rad = _normalize_angle(yaw_body_rad)
            ax_start, ay_start = _rotate_vector(
                np.asarray([ax_filtered, ay_filtered], dtype=np.float64),
                yaw_body_rad,
            )
            if t_s <= self._static_end_s:
                vx_start = 0.0
                vy_start = 0.0
            else:
                decay = math.exp(-dt_s / self._params.velocity_decay_tau_s)
                vx_start = (
                    decay * self._vx_start_mps[-1]
                    + (0.5 * dt_s * (self._ax_start_mps2[-1] + ax_start))
                )
                vy_start = (
                    decay * self._vy_start_mps[-1]
                    + (0.5 * dt_s * (self._ay_start_mps2[-1] + ay_start))
                )

        self._imu_t_s.append(float(t_s))
        self._yaw_body_rad.append(float(yaw_body_rad))
        self._yaw_rate_rps.append(float(gz_filtered))
        self._vx_start_mps.append(float(vx_start))
        self._vy_start_mps.append(float(vy_start))
        self._ax_start_mps2.append(float(ax_start))
        self._ay_start_mps2.append(float(ay_start))

    def _interpolate_motion_state(self, t_s: float) -> MotionState:
        if not self._imu_t_s:
            velocity = np.zeros(2, dtype=np.float64)
            accel = np.zeros(2, dtype=np.float64)
            return MotionState(
                t_s=t_s,
                yaw_rad=self._alignment_yaw_rad,
                yaw_rate_rps=0.0,
                velocity_mps=velocity,
                accel_world_mps2=accel,
            )

        times = np.asarray(self._imu_t_s, dtype=np.float64)
        yaw_body = float(np.interp(t_s, times, np.asarray(self._yaw_body_rad, dtype=np.float64)))
        yaw_rate = float(np.interp(t_s, times, np.asarray(self._yaw_rate_rps, dtype=np.float64)))
        vx_start = float(np.interp(t_s, times, np.asarray(self._vx_start_mps, dtype=np.float64)))
        vy_start = float(np.interp(t_s, times, np.asarray(self._vy_start_mps, dtype=np.float64)))
        ax_start = float(np.interp(t_s, times, np.asarray(self._ax_start_mps2, dtype=np.float64)))
        ay_start = float(np.interp(t_s, times, np.asarray(self._ay_start_mps2, dtype=np.float64)))

        velocity = np.asarray([vx_start, vy_start], dtype=np.float64)
        accel = np.asarray([ax_start, ay_start], dtype=np.float64)
        yaw_rad = yaw_body
        if self._alignment_ready:
            velocity = _rotate_vector(velocity, self._alignment_yaw_rad)
            accel = _rotate_vector(accel, self._alignment_yaw_rad)
            yaw_rad = _normalize_angle(yaw_body + self._alignment_yaw_rad)

        return MotionState(
            t_s=t_s,
            yaw_rad=yaw_rad,
            yaw_rate_rps=yaw_rate,
            velocity_mps=velocity,
            accel_world_mps2=accel,
        )

    def _process_pending_scans(self) -> list[ScanEstimate]:
        outputs: list[ScanEstimate] = []
        if not self._imu_initialized:
            return outputs

        if not self._alignment_ready:
            self._try_initialize_alignment()
            if not self._alignment_ready:
                return outputs

        while self._pending_scans:
            scan = self._pending_scans.popleft()
            outputs.append(self._process_single_scan(scan))
        return outputs

    def _try_initialize_alignment(self) -> None:
        if len(self._pending_scans) < self._params.initial_scan_count_min:
            return

        latest_scan_t_s = self._pending_scans[-1].t_s
        static_phase_observed = latest_scan_t_s > (self._static_end_s + 0.05)
        if (
            not static_phase_observed
            and len(self._pending_scans) < self._params.max_initial_alignment_scans
        ):
            return

        initial_alignment_scan_count = max(
            self._params.initial_scan_count_min,
            min(self._params.max_initial_alignment_scans, len(self._pending_scans)),
        )
        initial_scans = list(self._pending_scans)[:initial_alignment_scan_count]
        initial_points = np.vstack([scan.points_local for scan in initial_scans if scan.points_local.size])
        if initial_points.shape[0] < 40:
            return

        try:
            alignment_yaw_rad, wall_model = _estimate_initial_alignment(
                initial_points,
                self._params.corridor_bin_m,
            )
        except ValueError:
            return

        self._alignment_yaw_rad = float(alignment_yaw_rad)
        self._wall_model = wall_model
        self._origin_projection_m = np.asarray(
            [0.0, _centerline_y_m(wall_model, 0.0)],
            dtype=np.float64,
        )

        scan_times_s = np.asarray([scan.t_s for scan in self._pending_scans], dtype=np.float64)
        initial_scan_count = int(np.count_nonzero(scan_times_s <= (self._static_end_s + 0.05)))
        self._initial_scan_count = min(
            len(self._pending_scans),
            max(self._params.initial_scan_count_min, initial_scan_count),
        )
        self._alignment_ready = True

    def _predict_pose(self, scan_t_s: float, motion_state: MotionState) -> tuple[np.ndarray, np.ndarray]:
        prev_pose = self._poses_internal[-1].copy()
        prev_scan = self._processed_scans[-1]
        dt_s = max(1.0e-3, float(scan_t_s - prev_scan.t_s))
        if len(self._poses_internal) > 1:
            prev_prev_scan = self._processed_scans[-2]
            prev_dt_s = max(1.0e-3, float(prev_scan.t_s - prev_prev_scan.t_s))
            prev_velocity = (self._poses_internal[-1][:2] - self._poses_internal[-2][:2]) / prev_dt_s
        else:
            prev_motion_state = self._interpolate_motion_state(prev_scan.t_s)
            prev_velocity = prev_motion_state.velocity_mps.copy()
        prev_velocity = np.clip(prev_velocity, -3.0, 3.0)
        delta_prior = (
            (prev_velocity * dt_s)
            + (0.5 * motion_state.accel_world_mps2 * dt_s * dt_s)
        )
        prediction = np.asarray(
            [
                prev_pose[0] + delta_prior[0],
                prev_pose[1] + delta_prior[1],
                motion_state.yaw_rad,
            ],
            dtype=np.float64,
        )
        return prediction, delta_prior

    def _process_single_scan(self, scan: LidarScanObservation) -> ScanEstimate:
        if self._wall_model is None:
            raise RuntimeError("wall model must be initialized before processing scans")

        processed_index = len(self._processed_scans)
        motion_state = self._interpolate_motion_state(scan.t_s)

        if processed_index < self._initial_scan_count:
            pose = np.asarray([0.0, 0.0, self._alignment_yaw_rad], dtype=np.float64)
            if processed_index == 0:
                submap_median = float("nan")
                wall_median = float(
                    np.median(np.abs(_wall_residuals_for_pose(scan, pose, self._wall_model)))
                )
                valid_count = 0
            else:
                reference_submap = _transform_points(
                    self._processed_scans[0].sampled_points_local,
                    self._poses_internal[0],
                )
                submap_median, wall_median, valid_count = _evaluate_pose_quality(
                    scan,
                    pose,
                    reference_submap,
                    self._wall_model,
                    self._params.max_correspondence_m,
                )
            confidence = "high"
        else:
            submap_points = _build_submap_points(
                self._processed_scans,
                self._poses_internal,
                self._confidences,
                processed_index,
                self._params.submap_window_scans,
                self._initial_scan_count,
            )
            prediction, delta_prior = self._predict_pose(scan.t_s, motion_state)

            if submap_points.shape[0] < 10:
                pose = prediction
                confidence = "low"
                submap_median, wall_median, valid_count = _evaluate_pose_quality(
                    scan,
                    pose,
                    submap_points,
                    self._wall_model,
                    self._params.max_correspondence_m,
                )
            else:
                tree = cKDTree(submap_points)

                def residual_vector(pose_flat: np.ndarray) -> np.ndarray:
                    pose_candidate = pose_flat.astype(np.float64, copy=False)
                    world_points = _transform_points(scan.sampled_points_local, pose_candidate)
                    distances, nearest_indexes = tree.query(world_points, k=1)

                    residuals: list[np.ndarray] = []
                    diffs = world_points - submap_points[nearest_indexes]
                    scales = np.minimum(
                        1.0,
                        self._params.max_correspondence_m / np.maximum(distances, 1.0e-6),
                    ).reshape(-1, 1)
                    residuals.append((diffs * scales * LOCAL_SUBMAP_WEIGHT).reshape(-1))

                    wall_residuals = _wall_residuals_for_pose(scan, pose_candidate, self._wall_model)
                    if wall_residuals.size:
                        residuals.append(wall_residuals * LOCAL_WALL_WEIGHT)

                    motion_residual = (pose_candidate[:2] - self._poses_internal[-1][:2]) - delta_prior
                    residuals.append(motion_residual * LOCAL_MOTION_WEIGHT)
                    residuals.append(
                        np.asarray(
                            [(pose_candidate[2] - motion_state.yaw_rad) * LOCAL_YAW_WEIGHT],
                            dtype=np.float64,
                        )
                    )
                    return np.concatenate(residuals)

                lower_bounds = np.asarray(
                    [
                        prediction[0] - 1.5,
                        prediction[1] - 1.0,
                        motion_state.yaw_rad - 0.65,
                    ],
                    dtype=np.float64,
                )
                upper_bounds = np.asarray(
                    [
                        prediction[0] + 1.5,
                        prediction[1] + 1.0,
                        motion_state.yaw_rad + 0.65,
                    ],
                    dtype=np.float64,
                )

                solution = least_squares(
                    residual_vector,
                    x0=prediction,
                    bounds=(lower_bounds, upper_bounds),
                    loss="soft_l1",
                    f_scale=0.08,
                    max_nfev=self._params.max_scan_optimization_evals,
                )

                candidate_pose = solution.x if solution.success else prediction
                submap_median, wall_median, valid_count = _evaluate_pose_quality(
                    scan,
                    candidate_pose,
                    submap_points,
                    self._wall_model,
                    self._params.max_correspondence_m,
                )

                confidence = "high"
                if (
                    (not solution.success)
                    or valid_count < self._params.min_valid_correspondence_count
                    or (
                        math.isfinite(submap_median)
                        and submap_median > self._params.low_confidence_residual_m
                    )
                ):
                    confidence = "low"
                    candidate_pose = prediction
                    submap_median, wall_median, valid_count = _evaluate_pose_quality(
                        scan,
                        candidate_pose,
                        submap_points,
                        self._wall_model,
                        self._params.max_correspondence_m,
                    )

                pose = candidate_pose

        self._processed_scans.append(scan)
        self._poses_internal.append(pose)
        self._confidences.append(confidence)

        published_pose = pose.copy()
        published_pose[:2] = published_pose[:2] - self._origin_projection_m
        estimate = ScanEstimate(
            scan_index=scan.scan_index,
            stamp_sec=scan.stamp_sec,
            stamp_nanosec=scan.stamp_nanosec,
            t_s=scan.t_s,
            x_m=float(published_pose[0]),
            y_m=float(published_pose[1]),
            yaw_rad=_normalize_angle(float(published_pose[2])),
            vx_mps=float(motion_state.velocity_mps[0]),
            vy_mps=float(motion_state.velocity_mps[1]),
            yaw_rate_rps=float(motion_state.yaw_rate_rps),
            ax_world_mps2=float(motion_state.accel_world_mps2[0]),
            ay_world_mps2=float(motion_state.accel_world_mps2[1]),
            confidence=confidence,
            median_submap_residual_m=float(submap_median),
            median_wall_residual_m=float(wall_median),
            valid_correspondence_count=int(valid_count),
            alignment_ready=self._alignment_ready,
            imu_initialized=self._imu_initialized,
            best_effort_init=self._best_effort_init,
        )
        self._scan_estimates.append(estimate)
        return estimate
