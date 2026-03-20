#!/usr/bin/env python3
"""LiDAR-driven reconnaissance navigation for slow autonomous mapping laps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReconCommand:
    speed_pct: float
    steering_deg: float
    steering_pre_servo_deg: float
    target_heading_deg: float
    gap_heading_deg: float
    front_turn_heading_deg: float
    centering_heading_deg: float
    avoidance_heading_deg: float
    centering_weight: float
    front_clearance_m: float
    effective_front_clearance_m: float
    front_clearance_fallback_used: bool
    front_left_clearance_m: float
    front_right_clearance_m: float
    left_clearance_m: float
    right_clearance_m: float
    left_min_m: float
    right_min_m: float
    left_right_delta_m: float
    active_heading_source: str


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _circular_weighted_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()

    kernel = np.ones(int(window), dtype=np.float32)
    half = int(window) // 2

    padded_values = np.concatenate((values[-half:], values, values[:half]))
    valid = (padded_values > 0.0).astype(np.float32)
    safe_values = np.where(padded_values > 0.0, padded_values, 0.0)

    weighted_sum = np.convolve(safe_values, kernel, mode="valid")
    weight = np.convolve(valid, kernel, mode="valid")
    return np.divide(
        weighted_sum,
        weight,
        out=np.zeros_like(weighted_sum),
        where=weight > 0.0,
    )


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


def signbit(value: float) -> int:
    if value > 1e-6:
        return 1
    if value < -1e-6:
        return -1
    return 0


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
        wall_avoid_distance_m: float,
        wall_avoid_gain_deg_per_m: float,
        wall_avoid_limit_deg: float,
        gap_escape_heading_threshold_deg: float,
        gap_escape_release_distance_m: float,
        gap_escape_weight: float,
        turn_speed_reduction: float,
        min_turn_speed_factor: float,
        vehicle_half_width_m: float,
        vehicle_front_overhang_m: float,
        vehicle_rear_overhang_m: float,
    ) -> None:
        self._steering_limit_deg = max(1.0, float(steering_limit_deg))
        self._steering_gain = float(steering_gain)
        self._fov_half_angle_deg = max(15.0, min(170.0, float(fov_half_angle_deg)))
        self._smoothing_window = max(1, int(smoothing_window))
        self._stop_distance_m = float(stop_distance_m)
        self._slow_distance_m = max(self._stop_distance_m + 0.05, float(slow_distance_m))
        self._min_speed_pct = max(0.0, float(min_speed_pct))
        self._max_speed_pct = max(self._min_speed_pct, float(max_speed_pct))
        self._front_window_deg = max(1, int(front_window_deg))
        self._side_window_deg = max(5, int(side_window_deg))
        self._center_angle_penalty_per_deg = max(0.0, float(center_angle_penalty_per_deg))
        self._wall_centering_gain_deg_per_m = max(0.0, float(wall_centering_gain_deg_per_m))
        self._wall_centering_limit_deg = max(0.0, float(wall_centering_limit_deg))
        self._wall_centering_base_weight = max(
            0.0, min(1.0, float(wall_centering_base_weight))
        )
        self._wall_avoid_distance_m = max(0.0, float(wall_avoid_distance_m))
        self._wall_avoid_gain_deg_per_m = max(0.0, float(wall_avoid_gain_deg_per_m))
        self._wall_avoid_limit_deg = max(0.0, float(wall_avoid_limit_deg))
        self._gap_escape_heading_threshold_deg = max(
            0.0, float(gap_escape_heading_threshold_deg)
        )
        self._gap_escape_release_distance_m = max(
            0.0, float(gap_escape_release_distance_m)
        )
        self._gap_escape_weight = max(0.0, min(1.0, float(gap_escape_weight)))
        self._turn_speed_reduction = max(0.0, min(1.0, float(turn_speed_reduction)))
        self._min_turn_speed_factor = max(0.1, min(1.0, float(min_turn_speed_factor)))
        self._turn_commit_heading_threshold_deg = 8.0
        self._turn_commit_release_heading_deg = 18.0
        self._turn_commit_min_heading_deg = 8.0
        self._turn_commit_hold_cycles = 12
        self._turn_commit_sign = 0
        self._turn_commit_cycles_remaining = 0
        self._hitbox = _calculate_hitbox_polar(
            half_width_m=float(vehicle_half_width_m),
            front_overhang_m=float(vehicle_front_overhang_m),
            rear_overhang_m=float(vehicle_rear_overhang_m),
        )

    def compute_command(self, scan_ranges_m: np.ndarray) -> ReconCommand:
        ranges = np.asarray(scan_ranges_m, dtype=np.float32).copy()
        if ranges.size == 0:
            return self._build_empty_command()

        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges < 0.0] = 0.0

        shrunk = self._shrink_scan(ranges)
        smoothed = _circular_weighted_average(shrunk, self._smoothing_window)

        front_clearance_m = self._window_mean(
            smoothed,
            -self._front_window_deg,
            self._front_window_deg,
        )
        front_side_window_deg = int(min(self._fov_half_angle_deg, 55.0))
        front_left_clearance_m = self._window_mean(smoothed, 5, front_side_window_deg)
        front_right_clearance_m = self._window_mean(smoothed, -front_side_window_deg, -5)
        left_clearance_m = self._window_mean(smoothed, 20, self._side_window_deg)
        right_clearance_m = self._window_mean(smoothed, -self._side_window_deg, -20)
        left_min_m = self._window_min(smoothed, 20, self._side_window_deg)
        right_min_m = self._window_min(smoothed, -self._side_window_deg, -20)

        gap_heading_deg, gap_available = self._select_heading_deg(smoothed)
        centering_heading_deg = self._compute_centering_heading_deg(
            left_clearance_m,
            right_clearance_m,
        )
        front_turn_heading_deg = self._compute_front_turn_heading_deg(
            front_left_clearance_m,
            front_right_clearance_m,
        )
        avoidance_heading_deg, avoidance_active = self._compute_avoidance_heading_deg(
            left_min_m,
            right_min_m,
        )

        target_heading_deg, centering_weight, active_heading_source = self._select_target_heading_deg(
            gap_heading_deg=gap_heading_deg,
            gap_available=gap_available,
            front_clearance_m=front_clearance_m,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            front_turn_heading_deg=front_turn_heading_deg,
            centering_heading_deg=centering_heading_deg,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        )
        target_heading_deg, active_heading_source = self._apply_turn_commit(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
        )

        steering_pre_servo_deg = max(
            -self._steering_limit_deg,
            min(self._steering_limit_deg, target_heading_deg * self._steering_gain),
        )
        effective_front_clearance_m, front_clearance_fallback_used = (
            self._resolve_speed_front_clearance_m(
                front_clearance_m=front_clearance_m,
                gap_available=gap_available,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
        )
        speed_pct = self._compute_speed_pct(effective_front_clearance_m, steering_pre_servo_deg)
        return ReconCommand(
            speed_pct=speed_pct,
            steering_deg=steering_pre_servo_deg,
            steering_pre_servo_deg=steering_pre_servo_deg,
            target_heading_deg=target_heading_deg,
            gap_heading_deg=gap_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            centering_weight=centering_weight,
            front_clearance_m=front_clearance_m,
            effective_front_clearance_m=effective_front_clearance_m,
            front_clearance_fallback_used=front_clearance_fallback_used,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            left_right_delta_m=(left_clearance_m - right_clearance_m),
            active_heading_source=active_heading_source,
        )

    def _build_empty_command(self) -> ReconCommand:
        return ReconCommand(
            speed_pct=0.0,
            steering_deg=0.0,
            steering_pre_servo_deg=0.0,
            target_heading_deg=0.0,
            gap_heading_deg=0.0,
            front_turn_heading_deg=0.0,
            centering_heading_deg=0.0,
            avoidance_heading_deg=0.0,
            centering_weight=0.0,
            front_clearance_m=0.0,
            effective_front_clearance_m=0.0,
            front_clearance_fallback_used=False,
            front_left_clearance_m=0.0,
            front_right_clearance_m=0.0,
            left_clearance_m=0.0,
            right_clearance_m=0.0,
            left_min_m=0.0,
            right_min_m=0.0,
            left_right_delta_m=0.0,
            active_heading_source="fallback",
        )

    def _shrink_scan(self, ranges: np.ndarray) -> np.ndarray:
        shrunk = ranges.copy()
        valid = shrunk > 0.0
        shrunk[valid] = np.maximum(0.0, shrunk[valid] - self._hitbox[valid])
        return shrunk

    def _select_heading_deg(self, smoothed: np.ndarray) -> tuple[float, bool]:
        indices = np.arange(smoothed.size, dtype=np.int32)
        centered_deg = np.vectorize(_normalize_angle_deg)(indices).astype(np.float32)
        sector_mask = np.abs(centered_deg) <= self._fov_half_angle_deg
        sector_ranges = smoothed[sector_mask]
        sector_angles = centered_deg[sector_mask]

        if sector_ranges.size == 0 or np.count_nonzero(sector_ranges > 0.0) == 0:
            return 0.0, False

        scores = sector_ranges - self._center_angle_penalty_per_deg * np.abs(sector_angles)
        scores = np.where(sector_ranges > 0.0, scores, -np.inf)
        best_idx = int(np.argmax(scores))
        return float(sector_angles[best_idx]), True

    def _compute_centering_heading_deg(
        self,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> float:
        if left_clearance_m <= 0.0 or right_clearance_m <= 0.0:
            return 0.0

        heading_deg = (left_clearance_m - right_clearance_m) * self._wall_centering_gain_deg_per_m
        return float(
            max(
                -self._wall_centering_limit_deg,
                min(self._wall_centering_limit_deg, heading_deg),
            )
        )

    def _compute_front_turn_heading_deg(
        self,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
    ) -> float:
        if front_left_clearance_m <= 0.0 or front_right_clearance_m <= 0.0:
            return 0.0

        heading_deg = (front_left_clearance_m - front_right_clearance_m) * 35.0
        return float(
            max(
                -30.0,
                min(30.0, heading_deg),
            )
        )

    def _compute_avoidance_heading_deg(
        self,
        left_min_m: float,
        right_min_m: float,
    ) -> tuple[float, bool]:
        if self._wall_avoid_distance_m <= 0.0:
            return 0.0, False

        left_deficit_m = 0.0
        right_deficit_m = 0.0
        if left_min_m > 0.0:
            left_deficit_m = max(0.0, self._wall_avoid_distance_m - left_min_m)
        if right_min_m > 0.0:
            right_deficit_m = max(0.0, self._wall_avoid_distance_m - right_min_m)

        if left_deficit_m <= 0.0 and right_deficit_m <= 0.0:
            return 0.0, False

        heading_deg = (right_deficit_m - left_deficit_m) * self._wall_avoid_gain_deg_per_m
        heading_deg = float(
            max(
                -self._wall_avoid_limit_deg,
                min(self._wall_avoid_limit_deg, heading_deg),
            )
        )
        return heading_deg, True

    def _resolve_speed_front_clearance_m(
        self,
        *,
        front_clearance_m: float,
        gap_available: bool,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
    ) -> tuple[float, bool]:
        if front_clearance_m > 0.0:
            return front_clearance_m, False

        # Some scenes produce sparse or missing returns directly ahead even when
        # lateral windows clearly show free space and a navigable gap. In that
        # case, move at the minimum exploration speed instead of treating the
        # missing frontal reading as a hard stop.
        side_signal_available = any(
            value > 0.0
            for value in (left_clearance_m, right_clearance_m, left_min_m, right_min_m)
        )
        if gap_available and side_signal_available:
            fallback_front_clearance_m = min(
                self._slow_distance_m,
                self._stop_distance_m + 0.20,
            )
            return fallback_front_clearance_m, True

        return front_clearance_m, False

    def _select_target_heading_deg(
        self,
        *,
        gap_heading_deg: float,
        gap_available: bool,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        front_turn_heading_deg: float,
        centering_heading_deg: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, float, str]:
        if avoidance_active:
            if self._should_use_front_turn(
                front_clearance_m=front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                front_turn_heading_deg=front_turn_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
            ):
                return front_turn_heading_deg, 0.0, "front_turn"
            if self._should_use_gap_escape(
                gap_heading_deg=gap_heading_deg,
                gap_available=gap_available,
                avoidance_heading_deg=avoidance_heading_deg,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            ):
                return self._blend_gap_escape_heading_deg(
                    gap_heading_deg=gap_heading_deg,
                    avoidance_heading_deg=avoidance_heading_deg,
                ), 0.0, "gap_escape"
            return avoidance_heading_deg, 0.0, "avoidance"

        if left_clearance_m <= 0.0 or right_clearance_m <= 0.0:
            if gap_available:
                return gap_heading_deg, 0.0, "gap"
            return 0.0, 0.0, "fallback"

        target_heading_deg, centering_weight = self._blend_heading_deg(
            gap_heading_deg,
            centering_heading_deg,
            left_clearance_m,
            right_clearance_m,
        )

        gap_component = abs((1.0 - centering_weight) * gap_heading_deg)
        centering_component = abs(centering_weight * centering_heading_deg)
        if centering_component > gap_component and centering_weight > 0.0:
            source = "centering"
        elif gap_available:
            source = "gap"
        else:
            source = "fallback"
        return target_heading_deg, centering_weight, source

    def _blend_heading_deg(
        self,
        gap_heading_deg: float,
        centering_heading_deg: float,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> tuple[float, float]:
        if left_clearance_m <= 0.0 or right_clearance_m <= 0.0:
            return gap_heading_deg, 0.0

        side_ratio = min(left_clearance_m, right_clearance_m) / max(left_clearance_m, right_clearance_m)
        centering_weight = self._wall_centering_base_weight * side_ratio
        blended = (
            (1.0 - centering_weight) * gap_heading_deg
            + centering_weight * centering_heading_deg
        )
        return (
            float(
                max(
                    -self._fov_half_angle_deg,
                    min(self._fov_half_angle_deg, blended),
                )
            ),
            float(centering_weight),
        )

    def _should_use_gap_escape(
        self,
        *,
        gap_heading_deg: float,
        gap_available: bool,
        avoidance_heading_deg: float,
        left_min_m: float,
        right_min_m: float,
    ) -> bool:
        if not gap_available:
            return False
        if abs(gap_heading_deg) < self._gap_escape_heading_threshold_deg:
            return False
        if signbit(gap_heading_deg) == signbit(avoidance_heading_deg):
            return False

        side_min_values = [value for value in (left_min_m, right_min_m) if value > 0.0]
        if not side_min_values:
            return False
        nearest_side_m = min(side_min_values)
        if nearest_side_m < self._gap_escape_release_distance_m:
            return False
        return True

    def _should_use_front_turn(
        self,
        *,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        front_turn_heading_deg: float,
        avoidance_heading_deg: float,
    ) -> bool:
        if front_left_clearance_m <= 0.0 or front_right_clearance_m <= 0.0:
            return False
        if front_clearance_m > self._stop_distance_m:
            return False
        if abs(front_turn_heading_deg) < 6.0:
            return False
        front_sign = signbit(front_turn_heading_deg)
        avoidance_sign = signbit(avoidance_heading_deg)
        if avoidance_sign == 0:
            return True
        if front_sign != avoidance_sign:
            return True
        return abs(front_turn_heading_deg) >= (abs(avoidance_heading_deg) + 4.0)

    def _blend_gap_escape_heading_deg(
        self,
        *,
        gap_heading_deg: float,
        avoidance_heading_deg: float,
    ) -> float:
        blended = (
            (1.0 - self._gap_escape_weight) * avoidance_heading_deg
            + self._gap_escape_weight * gap_heading_deg
        )
        return float(
            max(
                -self._fov_half_angle_deg,
                min(self._fov_half_angle_deg, blended),
            )
        )

    def _apply_turn_commit(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
    ) -> tuple[float, str]:
        target_sign = signbit(target_heading_deg)

        if self._turn_commit_cycles_remaining > 0 and self._turn_commit_sign != 0:
            if target_sign == 0:
                target_heading_deg = self._turn_commit_sign * self._turn_commit_min_heading_deg
                active_heading_source = "turn_commit"
            elif target_sign != self._turn_commit_sign:
                if abs(target_heading_deg) < self._turn_commit_release_heading_deg:
                    target_heading_deg = self._turn_commit_sign * max(
                        abs(target_heading_deg),
                        self._turn_commit_min_heading_deg,
                    )
                    active_heading_source = "turn_commit"
                else:
                    self._turn_commit_sign = 0
                    self._turn_commit_cycles_remaining = 0

            if self._turn_commit_sign != 0:
                self._turn_commit_cycles_remaining -= 1
                if self._turn_commit_cycles_remaining <= 0:
                    self._turn_commit_sign = 0

        target_sign = signbit(target_heading_deg)
        if (
            target_sign != 0
            and abs(target_heading_deg) >= self._turn_commit_heading_threshold_deg
            and active_heading_source in {"front_turn", "gap_escape"}
        ):
            self._turn_commit_sign = target_sign
            self._turn_commit_cycles_remaining = self._turn_commit_hold_cycles

        return target_heading_deg, active_heading_source

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

    def _window_mean(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(np.mean(values)) if values else 0.0

    def _window_min(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(min(values)) if values else 0.0

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
