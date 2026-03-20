#!/usr/bin/env python3
"""LiDAR-driven reconnaissance navigation for slow autonomous mapping laps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_LEGACY_STEER_TABLE = np.array(
    [
        [0.0, 0.000],
        [10.0, 0.197],
        [20.0, 0.460],
        [30.0, 0.790],
        [40.0, 0.880],
        [50.0, 0.920],
    ],
    dtype=np.float32,
)
_LEGACY_STEERING_GAIN_BASELINE = 0.35
_LEGACY_AVOID_CORNER_SCALE_FACTOR = 1.2


@dataclass(frozen=True)
class ReconCommand:
    speed_pct: float
    steering_deg: float
    target_heading_deg: float
    gap_heading_deg: float
    centering_heading_deg: float
    centering_weight: float
    front_clearance_m: float
    left_clearance_m: float
    right_clearance_m: float
    left_right_delta_m: float


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _circular_convolution_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()

    kernel = np.ones(int(window), dtype=np.float32)
    half = int(window) // 2

    padded_values = np.concatenate((values[-half:], values, values[:half])).astype(np.float32)
    return np.convolve(padded_values, kernel, mode="valid") / float(window)


def _calculate_hitbox_polar(
    half_width_m: float,
    front_overhang_m: float,
    rear_overhang_m: float,
) -> np.ndarray:
    rad_angles = np.linspace(0.0, 2.0 * np.pi, num=360, endpoint=False)
    distances = np.zeros(360, dtype=np.float32)

    for index, theta in enumerate(rad_angles):
        c = math.cos(theta)
        s = math.sin(theta)
        candidates = []

        if abs(c) > 1e-9:
            x_side = half_width_m if c > 0.0 else -half_width_m
            t_x = x_side / c
            if t_x >= 0.0:
                y_at_x = s * t_x
                if -rear_overhang_m <= y_at_x <= front_overhang_m:
                    candidates.append(t_x)

        if abs(s) > 1e-9:
            y_side = front_overhang_m if s > 0.0 else -rear_overhang_m
            t_y = y_side / s
            if t_y >= 0.0:
                x_at_y = c * t_y
                if -half_width_m <= x_at_y <= half_width_m:
                    candidates.append(t_y)

        distances[index] = min(candidates) if candidates else 0.0

    return np.roll(distances, -90)


def _piecewise_lerp(value: float, table: np.ndarray) -> float:
    scalar = max(0.0, float(value))
    if scalar <= float(table[0, 0]):
        return float(table[0, 1])

    for index in range(1, table.shape[0]):
        x0 = float(table[index - 1, 0])
        x1 = float(table[index, 0])
        if scalar <= x1:
            y0 = float(table[index - 1, 1])
            y1 = float(table[index, 1])
            if x1 <= x0:
                return y1
            ratio = (scalar - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)

    return float(table[-1, 1])


class ReconNavigator:
    def __init__(
        self,
        *,
        steering_limit_deg: float,
        steering_gain: float,
        fov_half_angle_deg: float,
        smoothing_window: int,
        stop_distance_m: float,
        slow_distance_m: float,
        min_speed_pct: float,
        max_speed_pct: float,
        front_window_deg: int,
        side_window_deg: int,
        center_angle_penalty_per_deg: float,
        wall_centering_gain_deg_per_m: float,
        wall_centering_limit_deg: float,
        wall_centering_base_weight: float,
        turn_speed_reduction: float,
        min_turn_speed_factor: float,
        vehicle_half_width_m: float,
        vehicle_front_overhang_m: float,
        vehicle_rear_overhang_m: float,
    ) -> None:
        self._steering_limit_deg = max(1.0, float(steering_limit_deg))
        self._steering_gain = max(0.0, float(steering_gain))
        self._steering_gain_scale = (
            self._steering_gain / _LEGACY_STEERING_GAIN_BASELINE
            if _LEGACY_STEERING_GAIN_BASELINE > 0.0
            else 1.0
        )
        self._fov_half_angle_deg = max(15.0, min(170.0, float(fov_half_angle_deg)))
        self._field_of_view_deg = int(round(self._fov_half_angle_deg * 2.0))
        self._convolution_size = max(1, int(smoothing_window))
        if self._convolution_size % 2 == 0:
            self._convolution_size += 1
        self._stop_distance_m = float(stop_distance_m)
        self._slow_distance_m = max(self._stop_distance_m + 0.05, float(slow_distance_m))
        self._min_speed_pct = max(0.0, float(min_speed_pct))
        self._max_speed_pct = max(self._min_speed_pct, float(max_speed_pct))
        self._front_window_deg = max(1, int(front_window_deg))
        self._side_window_deg = max(5, int(side_window_deg))
        self._center_angle_penalty_per_deg = float(center_angle_penalty_per_deg)
        self._wall_centering_gain_deg_per_m = float(wall_centering_gain_deg_per_m)
        self._wall_centering_limit_deg = float(wall_centering_limit_deg)
        self._wall_centering_base_weight = float(wall_centering_base_weight)
        self._avoid_corner_max_angle_deg = max(
            8.0,
            min(30.0, float(self._side_window_deg) / 3.0),
        )
        self._avoid_corner_min_distance_m = max(1.0, self._slow_distance_m * 2.5)
        self._turn_speed_reduction = max(0.0, min(1.0, float(turn_speed_reduction)))
        self._min_turn_speed_factor = max(0.1, min(1.0, float(min_turn_speed_factor)))
        self._hitbox = _calculate_hitbox_polar(
            half_width_m=float(vehicle_half_width_m),
            front_overhang_m=float(vehicle_front_overhang_m),
            rear_overhang_m=float(vehicle_rear_overhang_m),
        )

    def compute_command(self, scan_ranges_m: np.ndarray) -> ReconCommand:
        ranges = np.asarray(scan_ranges_m, dtype=np.float32).copy()
        if ranges.size == 0:
            return ReconCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges < 0.0] = 0.0

        shrunk = self._shrink_scan(ranges)
        smoothed = _circular_convolution_average(shrunk, self._convolution_size)

        front_clearance_m = self._window_min(
            shrunk,
            -self._front_window_deg,
            self._front_window_deg,
        )
        left_clearance_m = self._window_mean(shrunk, 20, self._side_window_deg)
        right_clearance_m = self._window_mean(shrunk, -self._side_window_deg, -20)

        gap_heading_deg = self._select_heading_deg(smoothed)
        target_heading_deg, avoid_corner_delta_deg = self._apply_avoid_corner(
            gap_heading_deg,
            shrunk,
        )
        steering_deg = self._compute_steering_deg(target_heading_deg)

        speed_pct = self._compute_speed_pct(front_clearance_m, steering_deg)
        return ReconCommand(
            speed_pct=speed_pct,
            steering_deg=steering_deg,
            target_heading_deg=target_heading_deg,
            gap_heading_deg=gap_heading_deg,
            centering_heading_deg=avoid_corner_delta_deg,
            centering_weight=0.0,
            front_clearance_m=front_clearance_m,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_right_delta_m=(left_clearance_m - right_clearance_m),
        )

    def _shrink_scan(self, ranges: np.ndarray) -> np.ndarray:
        shrunk = ranges.copy()
        valid = shrunk > 0.0
        shrunk[valid] = np.maximum(0.0, shrunk[valid] - self._hitbox[valid])
        return shrunk

    def _select_heading_deg(self, smoothed: np.ndarray) -> float:
        sample_angles_deg = np.arange(smoothed.size, dtype=np.float32) * (360.0 / smoothed.size)
        centered_deg = np.vectorize(_normalize_angle_deg)(sample_angles_deg).astype(np.float32)
        sector_mask = np.abs(centered_deg) <= min(
            self._fov_half_angle_deg,
            0.5 * self._field_of_view_deg,
        )
        sector_ranges = smoothed[sector_mask]
        sector_angles = centered_deg[sector_mask]

        if sector_ranges.size == 0 or np.count_nonzero(sector_ranges > 0.0) == 0:
            return 0.0

        scores = np.where(sector_ranges > 0.0, sector_ranges, -np.inf)
        best_idx = int(np.argmax(scores))
        return float(sector_angles[best_idx])

    def _apply_avoid_corner(
        self,
        gap_heading_deg: float,
        shrunk_ranges: np.ndarray,
    ) -> tuple[float, float]:
        if shrunk_ranges.size == 0:
            return 0.0, 0.0

        bin_step_deg = 360.0 / shrunk_ranges.size
        max_offset_bins = max(1, int(round(self._avoid_corner_max_angle_deg / bin_step_deg)))
        positive_hit_deg = 0.0
        negative_hit_deg = 0.0

        for offset_bins in range(max_offset_bins, 0, -1):
            offset_deg = offset_bins * bin_step_deg
            positive_distance = self._distance_at_angle(shrunk_ranges, gap_heading_deg + offset_deg)
            negative_distance = self._distance_at_angle(shrunk_ranges, gap_heading_deg - offset_deg)

            if (
                positive_hit_deg == 0.0
                and 0.0 < positive_distance < self._avoid_corner_min_distance_m
            ):
                positive_hit_deg = offset_deg

            if (
                negative_hit_deg == 0.0
                and 0.0 < negative_distance < self._avoid_corner_min_distance_m
            ):
                negative_hit_deg = offset_deg

        delta_deg = 0.0
        if positive_hit_deg > negative_hit_deg:
            delta_deg = -_LEGACY_AVOID_CORNER_SCALE_FACTOR * (
                self._avoid_corner_max_angle_deg - negative_hit_deg
            )
        elif positive_hit_deg < negative_hit_deg:
            delta_deg = _LEGACY_AVOID_CORNER_SCALE_FACTOR * (
                self._avoid_corner_max_angle_deg - positive_hit_deg
            )

        return _normalize_angle_deg(gap_heading_deg + delta_deg), float(delta_deg)

    def _distance_at_angle(self, ranges: np.ndarray, angle_deg: float) -> float:
        normalized_deg = _normalize_angle_deg(angle_deg)
        if normalized_deg < 0.0:
            normalized_deg += 360.0
        index = int(round(normalized_deg * ranges.size / 360.0)) % ranges.size
        return float(ranges[index])

    def _compute_steering_deg(self, heading_deg: float) -> float:
        effective_heading_deg = abs(float(heading_deg)) * self._steering_gain_scale
        steering_mag = _piecewise_lerp(
            effective_heading_deg,
            np.column_stack(
                (
                    _LEGACY_STEER_TABLE[:, 0],
                    _LEGACY_STEER_TABLE[:, 1] * self._steering_limit_deg,
                )
            ),
        )
        steering_deg = math.copysign(steering_mag, float(heading_deg))
        return float(
            max(
                -self._steering_limit_deg,
                min(self._steering_limit_deg, steering_deg),
            )
        )

    def _window_mean(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(np.mean(values)) if values else 0.0

    def _window_min(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(min(values)) if values else 0.0

    def _window_values(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> list[float]:
        start = int(start_deg)
        end = int(end_deg)
        if start <= end:
            degrees = range(start, end + 1)
        else:
            degrees = list(range(start, 181)) + list(range(-180, end + 1))

        values = []
        for degree in degrees:
            index = int(round(degree)) % ranges.size
            if ranges[index] > 0.0:
                values.append(float(ranges[index]))
        return values

    def _compute_speed_pct(self, front_clearance_m: float, steering_deg: float) -> float:
        if front_clearance_m <= 0.0 or front_clearance_m <= self._stop_distance_m:
            return 0.0

        if front_clearance_m >= self._slow_distance_m:
            distance_factor = 1.0
        else:
            distance_factor = (
                (front_clearance_m - self._stop_distance_m)
                / (self._slow_distance_m - self._stop_distance_m)
            )

        steering_ratio = min(1.0, abs(steering_deg) / self._steering_limit_deg)
        turn_factor = max(
            self._min_turn_speed_factor,
            1.0 - self._turn_speed_reduction * steering_ratio,
        )

        speed_pct = self._min_speed_pct + (
            (self._max_speed_pct - self._min_speed_pct) * distance_factor
        )
        speed_pct *= turn_factor
        return float(max(0.0, min(self._max_speed_pct, speed_pct)))
