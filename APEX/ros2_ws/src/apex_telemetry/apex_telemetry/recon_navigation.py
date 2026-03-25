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
    corridor_axis_heading_deg: float
    corridor_center_heading_deg: float
    corridor_balance_ratio: float
    corridor_available: bool
    wall_follow_heading_deg: float
    wall_follow_active: bool
    wall_follow_anchor_side: str
    startup_candidate_heading_deg: float
    startup_candidate_source: str
    startup_hold_active: bool
    startup_latched_sign: int
    startup_latch_cycles_remaining: int
    left_wall_heading_deg: float
    right_wall_heading_deg: float
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
        corridor_balance_ratio_threshold: float,
        corridor_front_min_clearance_m: float,
        corridor_side_min_clearance_m: float,
        corridor_front_turn_weight: float,
        corridor_override_margin_deg: float,
        corridor_min_heading_deg: float,
        corridor_wall_start_deg: int,
        corridor_wall_end_deg: int,
        corridor_wall_min_points: int,
        wall_follow_target_distance_m: float,
        wall_follow_gain_deg_per_m: float,
        wall_follow_limit_deg: float,
        wall_follow_activation_heading_deg: float,
        wall_follow_release_balance_ratio: float,
        wall_follow_min_cycles: int,
        wall_follow_max_clearance_m: float,
        wall_follow_front_turn_weight: float,
        startup_consensus_min_heading_deg: float,
        startup_valid_cycles_required: int,
        startup_gap_lockout_cycles: int,
        startup_latch_cycles: int,
        ambiguity_probe_speed_pct: float,
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
        self._corridor_balance_ratio_threshold = max(
            0.0, min(1.0, float(corridor_balance_ratio_threshold))
        )
        self._corridor_front_min_clearance_m = max(
            0.0, float(corridor_front_min_clearance_m)
        )
        self._corridor_side_min_clearance_m = max(0.0, float(corridor_side_min_clearance_m))
        self._corridor_front_turn_weight = max(
            0.0, min(1.0, float(corridor_front_turn_weight))
        )
        self._corridor_centering_weight = 1.0 - self._corridor_front_turn_weight
        self._corridor_override_margin_deg = max(
            0.0, float(corridor_override_margin_deg)
        )
        self._corridor_min_heading_deg = max(0.0, float(corridor_min_heading_deg))
        self._corridor_wall_start_deg = max(5, int(corridor_wall_start_deg))
        self._corridor_wall_end_deg = max(
            self._corridor_wall_start_deg + 5,
            int(corridor_wall_end_deg),
        )
        self._corridor_wall_min_points = max(4, int(corridor_wall_min_points))
        self._wall_follow_target_distance_m = max(
            0.05, float(wall_follow_target_distance_m)
        )
        self._wall_follow_gain_deg_per_m = max(0.0, float(wall_follow_gain_deg_per_m))
        self._wall_follow_limit_deg = max(0.0, float(wall_follow_limit_deg))
        self._wall_follow_activation_heading_deg = max(
            0.0, float(wall_follow_activation_heading_deg)
        )
        self._wall_follow_release_balance_ratio = max(
            0.0, min(1.0, float(wall_follow_release_balance_ratio))
        )
        self._wall_follow_min_cycles = max(1, int(wall_follow_min_cycles))
        self._wall_follow_max_clearance_m = max(
            self._wall_follow_target_distance_m,
            float(wall_follow_max_clearance_m),
        )
        self._wall_follow_front_turn_weight = max(
            0.0, min(1.0, float(wall_follow_front_turn_weight))
        )
        self._wall_follow_base_weight = 1.0 - self._wall_follow_front_turn_weight
        self._wall_follow_support_min_factor = 0.65
        self._startup_consensus_min_heading_deg = max(
            0.5, float(startup_consensus_min_heading_deg)
        )
        self._startup_valid_cycles_required = max(1, int(startup_valid_cycles_required))
        self._startup_gap_lockout_cycles = max(1, int(startup_gap_lockout_cycles))
        self._startup_latch_cycles = max(0, int(startup_latch_cycles))
        self._ambiguity_probe_speed_pct = max(0.0, float(ambiguity_probe_speed_pct))
        self._turn_speed_reduction = max(0.0, min(1.0, float(turn_speed_reduction)))
        self._min_turn_speed_factor = max(0.1, min(1.0, float(min_turn_speed_factor)))
        self._turn_commit_heading_threshold_deg = 8.0
        self._turn_commit_min_heading_deg = 8.0
        self._turn_commit_hold_cycles = 16
        self._turn_commit_sign = 0
        self._turn_commit_cycles_remaining = 0
        self._curve_entry_bias_hold_cycles = 20
        self._curve_entry_bias_sign = 0
        self._curve_entry_bias_cycles_remaining = 0
        self._curve_entry_bias_override_sign = 0
        self._curve_entry_bias_override_streak = 0
        # The current exercise only needs stable corridor tracking. Keep the
        # controller in a simple centerline-following mode and avoid layered
        # state machines that have been causing premature turns before curves.
        self._simple_corridor_tracking_mode = True
        self._simple_centerline_heading_limit_deg = min(
            self._wall_centering_limit_deg,
            self._turn_commit_min_heading_deg + 6.0,
        )
        self._simple_centerline_local_limit_deg = min(
            self._simple_centerline_heading_limit_deg,
            self._turn_commit_min_heading_deg + 2.0,
        )
        self._simple_centerline_preview_limit_deg = self._simple_centerline_heading_limit_deg
        self._simple_centerline_axis_blend_gain = 0.20
        self._simple_centerline_balance_blend_range = 0.35
        self._simple_centerline_slew_step_deg = 2.5
        self._reference_curve_confirm_heading_deg = max(
            6.0,
            self._startup_consensus_min_heading_deg * 3.0,
        )
        self._reference_curve_center_confirm_heading_deg = max(
            10.0,
            self._reference_curve_confirm_heading_deg + 2.0,
        )
        self._reference_curve_entry_limit_deg = min(
            8.0,
            max(5.0, self._simple_centerline_local_limit_deg * 0.70),
        )
        self._reference_target_angle_limit_deg = min(50.0, self._fov_half_angle_deg)
        self._reference_convolution_size = max(2, min(7, self._front_window_deg // 4))
        if self._reference_convolution_size % 2 == 0:
            self._reference_convolution_size = min(7, self._reference_convolution_size + 1)
        self._reference_avoid_corner_max_angle = max(
            4, min(12, self._front_window_deg - 4)
        )
        self._reference_avoid_corner_min_distance_m = max(
            self._stop_distance_m + 0.10,
            min(self._slow_distance_m, 0.80),
        )
        self._reference_avoid_corner_scale_factor = 1.2
        self._reference_steer_factor = np.array(
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
        self._reference_steer_factor[:, 1] *= self._steering_limit_deg
        self._hitbox = _calculate_hitbox_polar(
            half_width_m=float(vehicle_half_width_m),
            front_overhang_m=float(vehicle_front_overhang_m),
            rear_overhang_m=float(vehicle_rear_overhang_m),
        )
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self._turn_commit_sign = 0
        self._turn_commit_cycles_remaining = 0
        self._wall_follow_active = False
        self._wall_follow_anchor_side = ""
        self._wall_follow_turn_sign = 0
        self._wall_follow_cycles_active = 0
        self._startup_valid_cycles = 0
        self._startup_consensus_sign = 0
        self._startup_consensus_streak = 0
        self._startup_latched_sign = 0
        self._startup_latch_cycles_remaining = 0
        self._startup_complete = False
        self._curve_confirmation_sign = 0
        self._curve_confirmation_streak = 0
        self._curve_confirmed_sign = 0
        self._curve_confirmed_cycles_remaining = 0
        self._curve_entry_bias_sign = 0
        self._curve_entry_bias_cycles_remaining = 0
        self._curve_entry_bias_override_sign = 0
        self._curve_entry_bias_override_streak = 0
        self._simple_last_heading_deg = 0.0

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
        corridor_wall_end_deg = min(self._side_window_deg, self._corridor_wall_end_deg)
        left_wall_points_xy = self._window_points(
            smoothed,
            self._corridor_wall_start_deg,
            corridor_wall_end_deg,
        )
        right_wall_points_xy = self._window_points(
            smoothed,
            -corridor_wall_end_deg,
            -self._corridor_wall_start_deg,
        )
        left_wall_heading_deg, left_wall_points = self._fit_wall_heading_deg(left_wall_points_xy)
        right_wall_heading_deg, right_wall_points = self._fit_wall_heading_deg(right_wall_points_xy)
        simple_preview_heading_deg, simple_preview_available = (
            self._compute_simple_centerline_preview_heading_deg(
                left_wall_points_xy=left_wall_points_xy,
                right_wall_points_xy=right_wall_points_xy,
            )
        )

        gap_heading_deg, gap_available = self._select_heading_deg(smoothed)
        centering_heading_deg = self._compute_centering_heading_deg(
            left_clearance_m,
            right_clearance_m,
        )
        front_turn_heading_deg = self._compute_front_turn_heading_deg(
            front_left_clearance_m,
            front_right_clearance_m,
        )
        (
            corridor_axis_heading_deg,
            corridor_center_heading_deg,
            corridor_balance_ratio,
            corridor_available,
        ) = (
            self._compute_corridor_center_heading_deg(
                left_wall_heading_deg=left_wall_heading_deg,
                right_wall_heading_deg=right_wall_heading_deg,
                left_wall_points=left_wall_points,
                right_wall_points=right_wall_points,
                front_turn_heading_deg=front_turn_heading_deg,
                centering_heading_deg=centering_heading_deg,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
            )
        )
        avoidance_heading_deg, avoidance_active = self._compute_avoidance_heading_deg(
            left_min_m,
            right_min_m,
        )
        effective_front_clearance_m, front_clearance_fallback_used = (
            self._resolve_speed_front_clearance_m(
                front_clearance_m=front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                gap_available=gap_available,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
        )

        if self._simple_corridor_tracking_mode:
            self._deactivate_wall_follow()
            self._turn_commit_sign = 0
            self._turn_commit_cycles_remaining = 0
            self._startup_latched_sign = 0
            self._startup_latch_cycles_remaining = 0
            self._startup_complete = True
            self._curve_confirmation_sign = 0
            self._curve_confirmation_streak = 0
            self._curve_confirmed_sign = 0
            self._curve_confirmed_cycles_remaining = 0
            (
                target_heading_deg,
                centering_weight,
                active_heading_source,
            ) = self._select_simple_corridor_tracking_heading_deg(
                shrunk_scan=shrunk,
                front_clearance_m=front_clearance_m,
                front_clearance_fallback_used=front_clearance_fallback_used,
                gap_heading_deg=gap_heading_deg,
                gap_available=gap_available,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_balance_ratio=corridor_balance_ratio,
                corridor_available=corridor_available,
                preview_heading_deg=simple_preview_heading_deg,
                preview_available=simple_preview_available,
                centering_heading_deg=centering_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )
            target_heading_deg, active_heading_source = self._apply_simple_reference_safety_limit(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                preview_heading_deg=simple_preview_heading_deg,
                preview_available=simple_preview_available,
                front_turn_heading_deg=front_turn_heading_deg,
                centering_heading_deg=centering_heading_deg,
            )
            target_heading_deg = self._apply_simple_heading_slew(target_heading_deg)
            if abs(target_heading_deg) < 0.5 and active_heading_source == "fallback":
                target_heading_deg = 0.0
            wall_follow_active = False
            wall_follow_anchor_side = "none"
            wall_follow_heading_deg = 0.0
            startup_candidate_heading_deg = 0.0
            startup_candidate_source = "none"
            startup_hold_active = False
            startup_latched_sign = 0
            startup_latch_cycles_remaining = 0
            steering_pre_servo_deg = self._compute_reference_steering_deg(target_heading_deg)
        else:
            (
                target_heading_deg,
                centering_weight,
                active_heading_source,
                wall_follow_heading_deg,
                wall_follow_active,
                wall_follow_anchor_side,
                startup_candidate_heading_deg,
                startup_candidate_source,
                startup_hold_active,
                startup_latched_sign,
                startup_latch_cycles_remaining,
            ) = self._select_target_heading_deg(
                gap_heading_deg=gap_heading_deg,
                gap_available=gap_available,
                simple_preview_heading_deg=simple_preview_heading_deg,
                simple_preview_available=simple_preview_available,
                front_clearance_m=effective_front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_balance_ratio=corridor_balance_ratio,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                left_wall_heading_deg=left_wall_heading_deg,
                right_wall_heading_deg=right_wall_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )
            target_heading_deg, active_heading_source = self._apply_curve_geometry_guard(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                front_clearance_m=effective_front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )
            target_heading_deg, active_heading_source = self._apply_turn_commit(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
            )
            target_heading_deg, active_heading_source = self._apply_curve_geometry_guard(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                front_clearance_m=effective_front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )
            self._update_curve_confirmation_state(
                front_clearance_m=effective_front_clearance_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
            )
            target_heading_deg, active_heading_source = self._apply_curve_entry_guard(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                front_clearance_m=effective_front_clearance_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
            )
            wall_follow_active = self._wall_follow_active
            wall_follow_anchor_side = self._wall_follow_anchor_side or "none"
            if not wall_follow_active:
                wall_follow_heading_deg = 0.0
            steering_gain = self._compute_adaptive_steering_gain(
                active_heading_source=active_heading_source,
                target_heading_deg=target_heading_deg,
                front_clearance_m=effective_front_clearance_m,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                wall_follow_heading_deg=wall_follow_heading_deg,
            )
            steering_pre_servo_deg = max(
                -self._steering_limit_deg,
                min(self._steering_limit_deg, target_heading_deg * steering_gain),
            )
        motion_front_clearance_m = self._compute_motion_front_clearance_m(
            effective_front_clearance_m=effective_front_clearance_m,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            wall_follow_active=wall_follow_active,
        )
        speed_pct = self._compute_speed_pct(
            motion_front_clearance_m,
            steering_pre_servo_deg,
            target_heading_deg,
        )
        if active_heading_source == "ambiguity_probe":
            speed_pct = min(speed_pct, self._compute_probe_speed_limit_pct())
        return ReconCommand(
            speed_pct=speed_pct,
            steering_deg=steering_pre_servo_deg,
            steering_pre_servo_deg=steering_pre_servo_deg,
            target_heading_deg=target_heading_deg,
            gap_heading_deg=gap_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_balance_ratio=corridor_balance_ratio,
            corridor_available=corridor_available,
            wall_follow_heading_deg=wall_follow_heading_deg,
            wall_follow_active=wall_follow_active,
            wall_follow_anchor_side=wall_follow_anchor_side,
            startup_candidate_heading_deg=startup_candidate_heading_deg,
            startup_candidate_source=startup_candidate_source,
            startup_hold_active=startup_hold_active,
            startup_latched_sign=startup_latched_sign,
            startup_latch_cycles_remaining=startup_latch_cycles_remaining,
            left_wall_heading_deg=left_wall_heading_deg,
            right_wall_heading_deg=right_wall_heading_deg,
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
            corridor_axis_heading_deg=0.0,
            corridor_center_heading_deg=0.0,
            corridor_balance_ratio=0.0,
            corridor_available=False,
            wall_follow_heading_deg=0.0,
            wall_follow_active=False,
            wall_follow_anchor_side="none",
            startup_candidate_heading_deg=0.0,
            startup_candidate_source="none",
            startup_hold_active=False,
            startup_latched_sign=0,
            startup_latch_cycles_remaining=0,
            left_wall_heading_deg=0.0,
            right_wall_heading_deg=0.0,
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

    def _compute_corridor_center_heading_deg(
        self,
        *,
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
        left_wall_points: int,
        right_wall_points: int,
        front_turn_heading_deg: float,
        centering_heading_deg: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> tuple[float, float, float, bool]:
        if min(
            front_left_clearance_m,
            front_right_clearance_m,
            left_clearance_m,
            right_clearance_m,
        ) <= 0.0:
            return 0.0, 0.0, 0.0, False

        side_ratio = min(left_clearance_m, right_clearance_m) / max(
            left_clearance_m,
            right_clearance_m,
        )
        front_ratio = min(front_left_clearance_m, front_right_clearance_m) / max(
            front_left_clearance_m,
            front_right_clearance_m,
        )
        corridor_balance_ratio = float(min(side_ratio, front_ratio))
        corridor_walls_available = min(left_wall_points, right_wall_points) >= (
            self._corridor_wall_min_points
        )
        corridor_available = (
            corridor_walls_available
            and corridor_balance_ratio >= self._corridor_balance_ratio_threshold
            and min(front_left_clearance_m, front_right_clearance_m)
            >= self._corridor_front_min_clearance_m
            and min(left_clearance_m, right_clearance_m) >= self._corridor_side_min_clearance_m
        )
        if not corridor_available:
            return 0.0, 0.0, corridor_balance_ratio, False

        corridor_axis_heading_deg = self._average_heading_deg(
            left_wall_heading_deg,
            right_wall_heading_deg,
        )
        corridor_heading_deg = (
            self._corridor_front_turn_weight
            * self._average_heading_deg(corridor_axis_heading_deg, front_turn_heading_deg)
            + self._corridor_centering_weight * centering_heading_deg
        )
        corridor_heading_deg = float(max(-30.0, min(30.0, corridor_heading_deg)))
        return corridor_axis_heading_deg, corridor_heading_deg, corridor_balance_ratio, True

    def _compute_wall_heading_deg(
        self,
        ranges: np.ndarray,
        start_deg: int,
        end_deg: int,
    ) -> tuple[float, int]:
        points = self._window_points(ranges, start_deg, end_deg)
        heading_deg, point_count = self._fit_wall_heading_deg(points)
        return heading_deg, point_count

    def _fit_wall_heading_deg(
        self,
        points: list[tuple[float, float]],
    ) -> tuple[float, int]:
        if len(points) < self._corridor_wall_min_points:
            return 0.0, len(points)

        pts = np.asarray(points, dtype=np.float32)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        covariance = centered.T @ centered
        if not np.isfinite(covariance).all():
            return 0.0, len(points)

        eigvals, eigvecs = np.linalg.eigh(covariance)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        if axis[0] < 0.0:
            axis = -axis
        heading_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
        heading_deg = max(-45.0, min(45.0, heading_deg))
        return heading_deg, len(points)

    def _average_heading_deg(self, heading_a_deg: float, heading_b_deg: float) -> float:
        rad_a = math.radians(heading_a_deg)
        rad_b = math.radians(heading_b_deg)
        x = math.cos(rad_a) + math.cos(rad_b)
        y = math.sin(rad_a) + math.sin(rad_b)
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            return 0.0
        return float(math.degrees(math.atan2(y, x)))

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
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        gap_available: bool,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
    ) -> tuple[float, bool]:
        if front_clearance_m > 0.0:
            return front_clearance_m, False

        front_side_candidates = [
            value
            for value in (front_left_clearance_m, front_right_clearance_m)
            if value > 0.0
        ]
        if front_side_candidates:
            return min(front_side_candidates), True

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
        simple_preview_heading_deg: float,
        simple_preview_available: bool,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_balance_ratio: float,
        corridor_available: bool,
        centering_heading_deg: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[
        float,
        float,
        str,
        float,
        bool,
        str,
        float,
        str,
        bool,
        int,
        int,
    ]:
        if self._simple_corridor_tracking_mode:
            self._deactivate_wall_follow()
            return (
                0.0,
                0.0,
                "fallback",
                0.0,
                False,
                "none",
                0.0,
                "none",
                False,
                0,
                0,
            )

        (
            target_heading_deg,
            centering_weight,
            active_heading_source,
        ) = self._select_base_target_heading_deg(
            gap_heading_deg=gap_heading_deg,
            gap_available=gap_available,
            front_clearance_m=front_clearance_m,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_balance_ratio=corridor_balance_ratio,
            corridor_available=corridor_available,
            centering_heading_deg=centering_heading_deg,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        )
        (
            target_heading_deg,
            active_heading_source,
            startup_hold_active,
            startup_candidate_heading_deg,
            startup_candidate_source,
        ) = self._apply_startup_hold(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            gap_heading_deg=gap_heading_deg,
            gap_available=gap_available,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
        )
        if startup_hold_active:
            self._deactivate_wall_follow()
            return (
                0.0,
                0.0,
                "startup_hold",
                0.0,
                False,
                "none",
                startup_candidate_heading_deg,
                startup_candidate_source,
                True,
                self._startup_latched_sign,
                self._startup_latch_cycles_remaining,
            )

        self._update_wall_follow_state(
            base_heading_deg=target_heading_deg,
            base_source=active_heading_source,
            front_clearance_m=front_clearance_m,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
            corridor_balance_ratio=corridor_balance_ratio,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            left_wall_heading_deg=left_wall_heading_deg,
            right_wall_heading_deg=right_wall_heading_deg,
        )

        wall_follow_heading_deg = self._compute_wall_follow_heading_deg(
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            base_heading_deg=target_heading_deg,
        )

        if self._wall_follow_active and abs(wall_follow_heading_deg) > 0.0:
            target_heading_deg = wall_follow_heading_deg
            centering_weight = 0.0
            active_heading_source = "wall_follow"

        target_heading_deg, active_heading_source = self._apply_startup_latch(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
        )

        return (
            target_heading_deg,
            centering_weight,
            active_heading_source,
            wall_follow_heading_deg,
            self._wall_follow_active,
            self._wall_follow_anchor_side or "none",
            startup_candidate_heading_deg,
            startup_candidate_source,
            False,
            self._startup_latched_sign,
            self._startup_latch_cycles_remaining,
        )

    def _select_simple_corridor_tracking_heading_deg(
        self,
        *,
        shrunk_scan: np.ndarray,
        front_clearance_m: float,
        front_clearance_fallback_used: bool,
        gap_heading_deg: float,
        gap_available: bool,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_balance_ratio: float,
        corridor_available: bool,
        preview_heading_deg: float,
        preview_available: bool,
        centering_heading_deg: float,
        front_turn_heading_deg: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, float, str]:
        side_signal_available = left_clearance_m > 0.0 and right_clearance_m > 0.0
        centerline_heading_deg = 0.0
        if side_signal_available:
            centerline_heading_deg = self._compute_simple_centerline_heading_deg(
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                corridor_balance_ratio=corridor_balance_ratio,
                corridor_available=corridor_available,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
            )
            if preview_available and abs(preview_heading_deg) > 0.0:
                centerline_sign = signbit(centerline_heading_deg)
                preview_sign = signbit(preview_heading_deg)
                if centerline_sign == 0 or preview_sign == 0 or preview_sign == centerline_sign:
                    centerline_heading_deg = (0.60 * centerline_heading_deg) + (
                        0.40 * preview_heading_deg
                    )

        reference_heading_deg, reference_available = self._compute_reference_target_heading_deg(
            shrunk_scan=shrunk_scan,
        )
        if not reference_available:
            if side_signal_available:
                return float(centerline_heading_deg), 0.0, "reference_centerline"
            if avoidance_active:
                return float(avoidance_heading_deg), 0.0, "avoidance"
            if gap_available:
                return float(gap_heading_deg), 0.0, "gap"
            return 0.0, 0.0, "fallback"

        target_heading_deg = reference_heading_deg
        active_heading_source = "reference_free_space"
        support_heading_deg = max(
            abs(front_turn_heading_deg),
            abs(corridor_axis_heading_deg),
            abs(corridor_center_heading_deg),
            abs(preview_heading_deg) if preview_available else 0.0,
        )

        if side_signal_available:
            centerline_sign = signbit(centerline_heading_deg)
            reference_sign = signbit(reference_heading_deg)
            if centerline_sign == 0:
                active_heading_source = "reference_free_space"
            elif reference_sign == 0 or reference_sign == centerline_sign:
                curve_blend = 0.55 + 0.25 * min(1.0, support_heading_deg / 12.0)
                target_heading_deg = ((1.0 - curve_blend) * centerline_heading_deg) + (
                    curve_blend * reference_heading_deg
                )
                active_heading_source = "reference_blend"
            elif support_heading_deg < 6.0:
                target_heading_deg = 0.65 * centerline_heading_deg
                active_heading_source = "reference_straight_guard"
            else:
                target_heading_deg = (0.85 * reference_heading_deg) + (
                    0.15 * centerline_heading_deg
                )
                active_heading_source = "reference_curve_commit"

        if avoidance_active:
            target_sign = signbit(target_heading_deg)
            avoidance_sign = signbit(avoidance_heading_deg)
            if target_sign != 0 and avoidance_sign == target_sign:
                target_heading_deg = (0.85 * target_heading_deg) + (
                    0.15 * avoidance_heading_deg
                )
            elif target_sign == 0 and avoidance_sign != 0:
                target_heading_deg = 0.35 * avoidance_heading_deg
                active_heading_source = "avoidance"

        target_heading_deg, active_heading_source = self._apply_reference_curve_entry_guard(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            front_clearance_m=front_clearance_m,
            front_clearance_fallback_used=front_clearance_fallback_used,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_balance_ratio=corridor_balance_ratio,
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
        )
        target_heading_deg = float(
            max(
                -self._reference_target_angle_limit_deg,
                min(self._reference_target_angle_limit_deg, target_heading_deg),
            )
        )
        return float(target_heading_deg), 0.0, active_heading_source

    def _apply_reference_curve_entry_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_clearance_m: float,
        front_clearance_fallback_used: bool,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_balance_ratio: float,
        preview_heading_deg: float,
        preview_available: bool,
    ) -> tuple[float, str]:
        if not active_heading_source.startswith("reference"):
            return target_heading_deg, active_heading_source

        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source

        preview_sign = signbit(preview_heading_deg) if preview_available else 0
        axis_sign = signbit(corridor_axis_heading_deg)
        center_sign = signbit(corridor_center_heading_deg)

        preview_support = (
            preview_sign == target_sign
            and abs(preview_heading_deg) >= self._reference_curve_confirm_heading_deg
        )
        axis_support = (
            axis_sign == target_sign
            and abs(corridor_axis_heading_deg) >= self._reference_curve_confirm_heading_deg
        )
        center_support = (
            center_sign == target_sign
            and abs(corridor_center_heading_deg) >= self._reference_curve_center_confirm_heading_deg
            and corridor_balance_ratio >= 0.35
        )
        opposite_support = (
            (preview_sign == -target_sign and abs(preview_heading_deg) >= 4.0)
            or (
                axis_sign == -target_sign
                and abs(corridor_axis_heading_deg) >= self._reference_curve_confirm_heading_deg
            )
        )
        curve_confirmed = (preview_support or axis_support or center_support) and not opposite_support
        if curve_confirmed:
            return target_heading_deg, active_heading_source

        # When the frontal sector is sparse on this car, the reference free-space
        # target can "see" the side opening of the upcoming curve too early.
        # Until the corridor preview/axis confirms that turn direction, keep only
        # a shallow approach heading instead of releasing the full free-space turn.
        if front_clearance_m > 0.0 and not front_clearance_fallback_used:
            if abs(target_heading_deg) <= self._reference_curve_entry_limit_deg:
                return target_heading_deg, active_heading_source

        support_limit_deg = 0.0
        if preview_sign == target_sign:
            support_limit_deg = max(support_limit_deg, abs(preview_heading_deg) * 0.60)
        if axis_sign == target_sign:
            support_limit_deg = max(support_limit_deg, abs(corridor_axis_heading_deg) * 0.30)
        if center_sign == target_sign:
            support_limit_deg = max(support_limit_deg, abs(corridor_center_heading_deg) * 0.55)

        limit_deg = min(
            self._reference_curve_entry_limit_deg,
            max(2.0, support_limit_deg),
        )
        guarded_heading_deg = target_sign * min(abs(target_heading_deg), limit_deg)
        if opposite_support and support_limit_deg <= 0.0:
            return 0.0, "reference_entry_guard"
        return float(guarded_heading_deg), "reference_entry_guard"

    def _compute_simple_centerline_heading_deg(
        self,
        *,
        corridor_axis_heading_deg: float,
        corridor_balance_ratio: float,
        corridor_available: bool,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> float:
        clearance_sum_m = left_clearance_m + right_clearance_m
        if clearance_sum_m <= 0.0:
            return 0.0

        lateral_offset_ratio = (left_clearance_m - right_clearance_m) / clearance_sum_m
        heading_deg = lateral_offset_ratio * self._simple_centerline_local_limit_deg
        target_sign = signbit(heading_deg)

        if not corridor_available:
            return float(
                max(
                    -self._simple_centerline_local_limit_deg,
                    min(self._simple_centerline_local_limit_deg, heading_deg),
                )
            )

        balance_excess = corridor_balance_ratio - self._corridor_balance_ratio_threshold
        axis_blend = max(
            0.0,
            min(1.0, balance_excess / self._simple_centerline_balance_blend_range),
        )
        axis_sign = signbit(corridor_axis_heading_deg)
        if axis_sign != 0 and (target_sign == 0 or axis_sign == target_sign):
            axis_heading_deg = axis_sign * min(
                abs(corridor_axis_heading_deg),
                self._simple_centerline_local_limit_deg,
            )
            heading_deg += (
                self._simple_centerline_axis_blend_gain * axis_blend * axis_heading_deg
            )

        return float(
            max(
                -self._simple_centerline_local_limit_deg,
                min(self._simple_centerline_local_limit_deg, heading_deg),
            )
        )

    def _compute_simple_centerline_preview_heading_deg(
        self,
        *,
        left_wall_points_xy: list[tuple[float, float]],
        right_wall_points_xy: list[tuple[float, float]],
    ) -> tuple[float, bool]:
        left_fit = self._fit_wall_line(left_wall_points_xy)
        right_fit = self._fit_wall_line(right_wall_points_xy)
        if left_fit is None or right_fit is None:
            return 0.0, False

        left_slope, left_intercept, left_x_min_m, left_x_max_m = left_fit
        right_slope, right_intercept, right_x_min_m, right_x_max_m = right_fit
        overlap_x_max_m = min(left_x_max_m, right_x_max_m)
        if overlap_x_max_m < 0.30:
            return 0.0, False

        lookahead_x_m = min(0.75, max(0.35, overlap_x_max_m * 0.6))
        left_y_m = (left_slope * lookahead_x_m) + left_intercept
        right_y_m = (right_slope * lookahead_x_m) + right_intercept
        if left_y_m <= right_y_m:
            return 0.0, False

        center_y_m = 0.5 * (left_y_m + right_y_m)
        center_heading_deg = math.degrees(math.atan2(center_y_m, lookahead_x_m))
        left_heading_deg = math.degrees(math.atan(left_slope))
        right_heading_deg = math.degrees(math.atan(right_slope))
        axis_heading_deg = self._average_heading_deg(left_heading_deg, right_heading_deg)

        preview_heading_deg = center_heading_deg
        if signbit(axis_heading_deg) == signbit(center_heading_deg):
            preview_heading_deg = (0.7 * center_heading_deg) + (0.3 * axis_heading_deg)

        preview_heading_deg = float(
            max(
                -self._simple_centerline_preview_limit_deg,
                min(self._simple_centerline_preview_limit_deg, preview_heading_deg),
            )
        )
        return preview_heading_deg, abs(preview_heading_deg) >= 1.0

    def _compute_reference_target_heading_deg(
        self,
        *,
        shrunk_scan: np.ndarray,
    ) -> tuple[float, bool]:
        if shrunk_scan.size == 0:
            return 0.0, False

        kernel = np.ones(self._reference_convolution_size, dtype=np.float32)
        kernel /= float(self._reference_convolution_size)
        half = self._reference_convolution_size // 2
        padded = np.concatenate((shrunk_scan[-half:], shrunk_scan, shrunk_scan[:half]))
        filtered_scan = np.convolve(padded, kernel, mode="valid")

        indices = np.arange(filtered_scan.size, dtype=np.int32)
        centered_deg = np.vectorize(_normalize_angle_deg)(indices).astype(np.float32)
        sector_mask = np.abs(centered_deg) <= self._fov_half_angle_deg
        sector_filtered = filtered_scan[sector_mask]
        sector_angles = centered_deg[sector_mask]
        sector_ranges = shrunk_scan[sector_mask]
        if sector_filtered.size == 0 or np.count_nonzero(sector_ranges > 0.0) == 0:
            return 0.0, False

        scores = np.where(sector_ranges > 0.0, sector_filtered, -np.inf)
        best_idx = int(np.argmax(scores))
        target_heading_deg = float(sector_angles[best_idx])
        target_heading_deg = self._apply_reference_corner_avoidance(
            target_heading_deg=target_heading_deg,
            shrunk_scan=shrunk_scan,
        )
        return float(target_heading_deg), True

    def _apply_reference_corner_avoidance(
        self,
        *,
        target_heading_deg: float,
        shrunk_scan: np.ndarray,
    ) -> float:
        left_block_deg = 0
        right_block_deg = 0
        target_index = int(round(target_heading_deg)) % shrunk_scan.size

        for offset_deg in range(self._reference_avoid_corner_max_angle, 0, -1):
            left_distance_m = float(shrunk_scan[(target_index + offset_deg) % shrunk_scan.size])
            right_distance_m = float(shrunk_scan[(target_index - offset_deg) % shrunk_scan.size])

            if (
                left_block_deg == 0
                and 0.0 < left_distance_m < self._reference_avoid_corner_min_distance_m
            ):
                left_block_deg = offset_deg
            if (
                right_block_deg == 0
                and 0.0 < right_distance_m < self._reference_avoid_corner_min_distance_m
            ):
                right_block_deg = offset_deg

        delta_deg = 0.0
        if left_block_deg == right_block_deg:
            delta_deg = 0.0
        elif left_block_deg != 0 and (
            right_block_deg == 0 or left_block_deg < right_block_deg
        ):
            delta_deg = -self._reference_avoid_corner_scale_factor * (
                self._reference_avoid_corner_max_angle - left_block_deg
            )
        elif right_block_deg != 0 and (
            left_block_deg == 0 or right_block_deg < left_block_deg
        ):
            delta_deg = self._reference_avoid_corner_scale_factor * (
                self._reference_avoid_corner_max_angle - right_block_deg
            )

        target_heading_deg = float(
            max(
                -self._reference_target_angle_limit_deg,
                min(
                    self._reference_target_angle_limit_deg,
                    target_heading_deg + delta_deg,
                ),
            )
        )
        return target_heading_deg

    def _apply_simple_reference_safety_limit(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        preview_heading_deg: float,
        preview_available: bool,
        front_turn_heading_deg: float,
        centering_heading_deg: float,
    ) -> tuple[float, str]:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source

        if target_sign > 0:
            turn_side_front_clearance_m = front_left_clearance_m
            opposite_front_clearance_m = front_right_clearance_m
            turn_side_min_m = left_min_m
        else:
            turn_side_front_clearance_m = front_right_clearance_m
            opposite_front_clearance_m = front_left_clearance_m
            turn_side_min_m = right_min_m

        support_heading_deg = 0.0
        for candidate_heading_deg in (
            preview_heading_deg if preview_available else 0.0,
            corridor_axis_heading_deg,
            corridor_center_heading_deg,
            front_turn_heading_deg,
            centering_heading_deg,
        ):
            if signbit(candidate_heading_deg) == target_sign:
                support_heading_deg = max(support_heading_deg, abs(candidate_heading_deg))

        if (
            turn_side_min_m > 0.0
            and turn_side_min_m <= (self._stop_distance_m + 0.05)
            and support_heading_deg < 6.0
        ):
            return 0.0, "reference_wall_limit"

        if (
            turn_side_front_clearance_m > 0.0
            and opposite_front_clearance_m > 0.0
            and turn_side_front_clearance_m < (opposite_front_clearance_m - 0.08)
            and support_heading_deg < 8.0
        ):
            limited_heading_deg = target_sign * min(abs(target_heading_deg), 4.0)
            return float(limited_heading_deg), "reference_wall_limit"

        return target_heading_deg, active_heading_source

    def _compute_reference_steering_deg(self, target_heading_deg: float) -> float:
        magnitude_deg = min(self._reference_target_angle_limit_deg, abs(target_heading_deg))
        steering_deg = self._lerp_table(magnitude_deg, self._reference_steer_factor)
        return float(math.copysign(steering_deg, target_heading_deg))

    def _lerp_table(self, value: float, factor: np.ndarray) -> float:
        indices = np.nonzero(value < factor[:, 0])[0]
        if len(indices) == 0:
            return float(factor[-1, 1])

        index = int(indices[0])
        if index == 0:
            return float(factor[0, 1])

        delta = factor[index] - factor[index - 1]
        scale = (value - factor[index - 1, 0]) / delta[0]
        return float(factor[index - 1, 1] + scale * delta[1])

    def _fit_wall_line(
        self,
        points_xy: list[tuple[float, float]],
    ) -> Optional[tuple[float, float, float, float]]:
        if len(points_xy) < self._corridor_wall_min_points:
            return None

        pts = np.asarray(points_xy, dtype=np.float32)
        forward = pts[:, 0]
        lateral = pts[:, 1]
        valid = forward > 0.05
        if np.count_nonzero(valid) < self._corridor_wall_min_points:
            return None

        forward = forward[valid]
        lateral = lateral[valid]
        if np.ptp(forward) < 0.08:
            return None

        design = np.column_stack((forward, np.ones_like(forward)))
        slope, intercept = np.linalg.lstsq(design, lateral, rcond=None)[0]
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None

        return (
            float(slope),
            float(intercept),
            float(np.min(forward)),
            float(np.max(forward)),
        )

    def _apply_simple_heading_slew(self, heading_deg: float) -> float:
        delta_deg = float(heading_deg) - self._simple_last_heading_deg
        max_delta_deg = self._simple_centerline_slew_step_deg
        if delta_deg > max_delta_deg:
            heading_deg = self._simple_last_heading_deg + max_delta_deg
        elif delta_deg < -max_delta_deg:
            heading_deg = self._simple_last_heading_deg - max_delta_deg
        self._simple_last_heading_deg = float(heading_deg)
        return float(heading_deg)

    def _apply_simple_near_wall_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        corridor_center_heading_deg: float,
        preview_heading_deg: float,
        preview_available: bool,
        centering_heading_deg: float,
    ) -> tuple[float, str]:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source

        if target_sign > 0:
            turn_side_front_clearance_m = front_left_clearance_m
            opposite_front_clearance_m = front_right_clearance_m
            turn_side_min_m = left_min_m
            opposite_side_min_m = right_min_m
        else:
            turn_side_front_clearance_m = front_right_clearance_m
            opposite_front_clearance_m = front_left_clearance_m
            turn_side_min_m = right_min_m
            opposite_side_min_m = left_min_m

        turn_toward_near_wall = False
        if (
            turn_side_min_m > 0.0
            and turn_side_min_m < max(self._wall_avoid_distance_m, self._stop_distance_m + 0.10)
        ):
            turn_toward_near_wall = True
        if (
            turn_side_min_m > 0.0
            and opposite_side_min_m > 0.0
            and turn_side_min_m < (opposite_side_min_m - 0.04)
        ):
            turn_toward_near_wall = True
        if not turn_toward_near_wall:
            return target_heading_deg, active_heading_source

        preview_sign = signbit(preview_heading_deg)
        if (
            preview_available
            and preview_sign == target_sign
            and abs(preview_heading_deg) >= max(3.0, self._startup_consensus_min_heading_deg)
            and turn_side_front_clearance_m > (self._stop_distance_m + 0.08)
            and (
                opposite_front_clearance_m <= 0.0
                or turn_side_front_clearance_m >= (opposite_front_clearance_m - 0.03)
            )
        ):
            limited_heading_deg = target_sign * min(
                abs(target_heading_deg),
                max(abs(preview_heading_deg), abs(corridor_center_heading_deg), 4.0),
            )
            return float(limited_heading_deg), active_heading_source

        safe_limit_deg = self._compute_recenter_heading_limit_deg(
            target_sign=target_sign,
            corridor_center_heading_deg=corridor_center_heading_deg,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=0.0,
        )
        limited_heading_deg = target_sign * min(abs(target_heading_deg), safe_limit_deg)
        if abs(limited_heading_deg) < self._startup_consensus_min_heading_deg:
            return 0.0, "entry_wall_safety"
        return float(limited_heading_deg), "entry_wall_safety"

    def _apply_simple_curve_side_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        preview_heading_deg: float,
        preview_available: bool,
        front_turn_heading_deg: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
    ) -> tuple[float, str]:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source

        previous_sign = 0
        if abs(self._simple_last_heading_deg) >= self._startup_consensus_min_heading_deg:
            previous_sign = signbit(self._simple_last_heading_deg)
        if previous_sign == 0 or previous_sign == target_sign:
            return target_heading_deg, active_heading_source

        preview_sign = 0
        if preview_available and abs(preview_heading_deg) >= max(
            3.0, self._startup_consensus_min_heading_deg
        ):
            preview_sign = signbit(preview_heading_deg)

        axis_sign = 0
        if abs(corridor_axis_heading_deg) >= 5.0:
            axis_sign = signbit(corridor_axis_heading_deg)

        support_previous_sign = preview_sign == previous_sign or axis_sign == previous_sign
        support_target_sign = preview_sign == target_sign or axis_sign == target_sign
        if not support_previous_sign or support_target_sign:
            return target_heading_deg, active_heading_source

        support_heading_deg = 0.0
        if preview_sign == previous_sign:
            support_heading_deg = max(support_heading_deg, abs(preview_heading_deg) * 0.45)
        if axis_sign == previous_sign:
            support_heading_deg = max(
                support_heading_deg,
                max(2.5, abs(corridor_axis_heading_deg) * 0.22),
            )
        if signbit(corridor_center_heading_deg) == previous_sign:
            support_heading_deg = max(
                support_heading_deg,
                abs(corridor_center_heading_deg) * 0.30,
            )

        guarded_heading_deg = previous_sign * min(
            self._simple_centerline_local_limit_deg,
            max(2.0, support_heading_deg),
        )
        return float(guarded_heading_deg), "curve_side_guard"

    def _select_base_target_heading_deg(
        self,
        *,
        gap_heading_deg: float,
        gap_available: bool,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_balance_ratio: float,
        corridor_available: bool,
        centering_heading_deg: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, float, str]:
        if not avoidance_active and left_clearance_m > 0.0 and right_clearance_m > 0.0:
            if self._should_use_ambiguity_probe(
                gap_heading_deg=gap_heading_deg,
                gap_available=gap_available,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
            ):
                return self._compute_ambiguity_probe_heading_deg(
                    centering_heading_deg=centering_heading_deg,
                ), 0.0, "ambiguity_probe"

        if corridor_available and self._should_use_corridor_center(
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_balance_ratio=corridor_balance_ratio,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        ):
            return corridor_center_heading_deg, 0.0, "corridor_center"

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

    def _direction_probe_threshold_deg(self) -> float:
        return max(
            self._startup_consensus_min_heading_deg * 2.0,
            self._turn_commit_min_heading_deg * 0.5,
        )

    def _compute_ambiguity_probe_heading_deg(
        self,
        *,
        centering_heading_deg: float,
    ) -> float:
        # In ambiguity_probe we explicitly avoid committing to a turn sign.
        # The goal is to creep forward until a real directional signal appears.
        _ = centering_heading_deg
        return 0.0

    def _should_use_ambiguity_probe(
        self,
        *,
        gap_heading_deg: float,
        gap_available: bool,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> bool:
        if not gap_available:
            return False
        if left_clearance_m <= 0.0 or right_clearance_m <= 0.0:
            return False
        if avoidance_active and abs(avoidance_heading_deg) >= self._direction_probe_threshold_deg():
            return False

        directional_strength_deg = max(
            abs(front_turn_heading_deg),
            abs(corridor_center_heading_deg) if corridor_available else 0.0,
        )
        if directional_strength_deg >= self._direction_probe_threshold_deg():
            return False

        centering_probe_limit_deg = max(1.5, self._direction_probe_threshold_deg() * 0.75)
        if abs(centering_heading_deg) >= centering_probe_limit_deg:
            return False

        return abs(gap_heading_deg) >= (self._direction_probe_threshold_deg() * 3.0)

    def _compute_probe_speed_limit_pct(self) -> float:
        return float(
            max(0.0, min(self._max_speed_pct, self._ambiguity_probe_speed_pct))
        )

    def _curve_entry_support_heading_deg(
        self,
        *,
        target_sign: int,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> float:
        if target_sign == 0:
            return 0.0

        axis_sign = signbit(corridor_axis_heading_deg) if corridor_available else 0
        center_sign = signbit(corridor_center_heading_deg) if corridor_available else 0
        front_sign = signbit(front_turn_heading_deg)
        axis_presence_threshold_deg = max(
            0.75,
            self._startup_consensus_min_heading_deg * 0.5,
        )
        aligned_support_threshold_deg = max(
            self._startup_consensus_min_heading_deg * 2.0,
            self._turn_commit_min_heading_deg * 0.75,
        )
        axis_only_threshold_deg = max(
            self._startup_consensus_min_heading_deg * 2.0,
            self._turn_commit_min_heading_deg * 0.5,
        )

        if (
            axis_sign != 0
            and axis_sign != target_sign
            and abs(corridor_axis_heading_deg) >= axis_presence_threshold_deg
        ):
            return 0.0

        support_heading_deg = 0.0
        if axis_sign == target_sign and abs(corridor_axis_heading_deg) >= axis_presence_threshold_deg:
            support_heading_deg = corridor_axis_heading_deg
            if (
                front_sign == target_sign
                and abs(front_turn_heading_deg) >= aligned_support_threshold_deg
            ):
                support_heading_deg = target_sign * max(
                    abs(support_heading_deg),
                    abs(front_turn_heading_deg),
                )
            if (
                center_sign == target_sign
                and abs(corridor_center_heading_deg) >= (self._startup_consensus_min_heading_deg * 2.0)
            ):
                support_heading_deg = target_sign * max(
                    abs(support_heading_deg),
                    abs(corridor_center_heading_deg),
                )
            if abs(corridor_axis_heading_deg) >= axis_only_threshold_deg:
                support_heading_deg = target_sign * max(
                    abs(support_heading_deg),
                    abs(corridor_axis_heading_deg),
                )
            return float(support_heading_deg)

        if axis_sign == 0:
            if (
                front_sign == target_sign
                and center_sign == target_sign
                and abs(front_turn_heading_deg) >= self._turn_commit_min_heading_deg
                and abs(corridor_center_heading_deg) >= (self._startup_consensus_min_heading_deg * 2.0)
            ):
                return float(
                    target_sign
                    * max(abs(front_turn_heading_deg), abs(corridor_center_heading_deg))
                )

        return 0.0

    def _compute_recenter_heading_limit_deg(
        self,
        *,
        target_sign: int,
        corridor_center_heading_deg: float,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
    ) -> float:
        candidates_deg: list[float] = []
        if signbit(corridor_center_heading_deg) == target_sign:
            candidates_deg.append(abs(corridor_center_heading_deg))
        if signbit(centering_heading_deg) == target_sign:
            candidates_deg.append(abs(centering_heading_deg) * 0.5)
        if signbit(avoidance_heading_deg) == target_sign:
            candidates_deg.append(abs(avoidance_heading_deg) * 0.5)

        base_limit_deg = max(2.0, self._startup_consensus_min_heading_deg * 1.5)
        if not candidates_deg:
            return float(base_limit_deg)
        return float(
            max(
                base_limit_deg,
                min(self._turn_commit_min_heading_deg * 0.75, max(candidates_deg)),
            )
        )

    def _apply_curve_geometry_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, str]:
        if active_heading_source in {"", "ambiguity_probe", "startup_hold"}:
            return target_heading_deg, active_heading_source
        if front_clearance_m <= 0.0 or front_clearance_m >= self._slow_distance_m:
            return target_heading_deg, active_heading_source

        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source

        support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=target_sign,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        if abs(support_heading_deg) > 0.0:
            return self._apply_near_wall_turn_guard(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )

        if self._wall_follow_active:
            self._deactivate_wall_follow()
        if self._startup_latched_sign == target_sign:
            self._startup_latched_sign = 0
            self._startup_latch_cycles_remaining = 0
        if self._turn_commit_sign == target_sign:
            self._turn_commit_sign = 0
            self._turn_commit_cycles_remaining = 0

        alignment_heading_deg = self._compute_alignment_heading_deg(
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        )
        if abs(alignment_heading_deg) < abs(target_heading_deg) or active_heading_source in {
            "wall_follow",
            "turn_commit",
            "startup_latch",
            "curve_entry_guard",
            "curve_entry_bias",
            "front_turn",
            "corridor_center",
        }:
            return self._apply_near_wall_turn_guard(
                target_heading_deg=float(alignment_heading_deg),
                active_heading_source="alignment_guard",
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                corridor_axis_heading_deg=corridor_axis_heading_deg,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                corridor_available=corridor_available,
                centering_heading_deg=centering_heading_deg,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
            )
        return self._apply_near_wall_turn_guard(
            target_heading_deg=float(alignment_heading_deg),
            active_heading_source=active_heading_source,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        )

    def _compute_alignment_heading_deg(
        self,
        *,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> float:
        centering_sign = signbit(centering_heading_deg)
        avoidance_sign = signbit(avoidance_heading_deg) if avoidance_active else 0

        if avoidance_sign != 0:
            alignment_sign = avoidance_sign
        else:
            alignment_sign = centering_sign
        if alignment_sign == 0:
            return 0.0

        alignment_candidates_deg: list[float] = []
        if centering_sign == alignment_sign:
            alignment_candidates_deg.append(abs(centering_heading_deg) * 0.25)
        if avoidance_sign == alignment_sign:
            alignment_candidates_deg.append(abs(avoidance_heading_deg) * 0.4)

        if not alignment_candidates_deg:
            return 0.0

        alignment_limit_deg = max(
            2.0,
            self._startup_consensus_min_heading_deg * 1.5,
        )
        return float(alignment_sign * min(alignment_limit_deg, max(alignment_candidates_deg)))

    def _apply_near_wall_turn_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, str]:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source
        if active_heading_source in {"", "ambiguity_probe", "startup_hold"}:
            return target_heading_deg, active_heading_source
        if self._is_curve_confirmed(
            target_heading_deg=target_heading_deg,
            front_clearance_m=0.0,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        ):
            return target_heading_deg, active_heading_source

        if target_sign > 0:
            turn_side_front_clearance_m = front_left_clearance_m
            opposite_front_clearance_m = front_right_clearance_m
            turn_side_min_m = left_min_m
            opposite_side_min_m = right_min_m
        else:
            turn_side_front_clearance_m = front_right_clearance_m
            opposite_front_clearance_m = front_left_clearance_m
            turn_side_min_m = right_min_m
            opposite_side_min_m = left_min_m

        turn_toward_near_wall = False
        if avoidance_active and signbit(avoidance_heading_deg) == -target_sign:
            turn_toward_near_wall = True
        if (
            turn_side_min_m > 0.0
            and turn_side_min_m < max(self._wall_avoid_distance_m, self._stop_distance_m + 0.10)
        ):
            turn_toward_near_wall = True
        if (
            turn_side_min_m > 0.0
            and opposite_side_min_m > 0.0
            and turn_side_min_m < (opposite_side_min_m - 0.04)
        ):
            turn_toward_near_wall = True
        if not turn_toward_near_wall:
            return target_heading_deg, active_heading_source

        support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=target_sign,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        open_margin_m = 0.0
        if turn_side_front_clearance_m > 0.0 and opposite_front_clearance_m > 0.0:
            open_margin_m = turn_side_front_clearance_m - opposite_front_clearance_m

        support_override_active = (
            abs(support_heading_deg) >= self._curve_confirmation_strong_threshold_deg()
            and turn_side_front_clearance_m > (self._stop_distance_m + 0.12)
            and open_margin_m >= 0.14
        )
        if support_override_active:
            return target_heading_deg, active_heading_source

        if self._wall_follow_active and self._wall_follow_turn_sign == target_sign:
            self._deactivate_wall_follow()
        if self._startup_latched_sign == target_sign:
            self._startup_latched_sign = 0
            self._startup_latch_cycles_remaining = 0
        if self._turn_commit_sign == target_sign:
            self._turn_commit_sign = 0
            self._turn_commit_cycles_remaining = 0

        guarded_heading_deg = 0.0
        avoidance_sign = signbit(avoidance_heading_deg)
        if avoidance_sign != 0 and avoidance_sign != target_sign:
            guarded_heading_deg = avoidance_sign * min(
                abs(avoidance_heading_deg),
                self._compute_recenter_heading_limit_deg(
                    target_sign=avoidance_sign,
                    corridor_center_heading_deg=corridor_center_heading_deg,
                    centering_heading_deg=centering_heading_deg,
                    avoidance_heading_deg=avoidance_heading_deg,
                ),
            )

        if abs(guarded_heading_deg) >= self._startup_consensus_min_heading_deg:
            return float(guarded_heading_deg), "entry_wall_safety"
        return 0.0, "entry_wall_safety"

    def _curve_confirmation_entry_distance_m(self) -> float:
        return max(
            self._stop_distance_m + 0.12,
            min(self._slow_distance_m, self._stop_distance_m + 0.30),
        )

    def _curve_entry_bias_entry_distance_m(self) -> float:
        return max(
            self._stop_distance_m + 0.18,
            min(self._slow_distance_m + 0.12, self._stop_distance_m + 0.50),
        )

    def _curve_entry_bias_seed_threshold_deg(self) -> float:
        return max(
            self._startup_consensus_min_heading_deg * 2.0,
            self._turn_commit_min_heading_deg * 0.6,
        )

    def _curve_entry_bias_override_threshold_deg(self) -> float:
        return max(
            self._curve_confirmation_strong_threshold_deg(),
            self._turn_commit_heading_threshold_deg + 2.0,
        )

    def _curve_confirmation_strong_threshold_deg(self) -> float:
        return max(
            self._wall_follow_activation_heading_deg,
            self._turn_commit_min_heading_deg * 1.25,
        )

    def _curve_entry_open_margin_m(
        self,
        *,
        turn_sign: int,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
    ) -> float:
        if turn_sign > 0:
            turn_side_clearance_m = front_left_clearance_m
            opposite_clearance_m = front_right_clearance_m
        elif turn_sign < 0:
            turn_side_clearance_m = front_right_clearance_m
            opposite_clearance_m = front_left_clearance_m
        else:
            return 0.0

        if turn_side_clearance_m <= 0.0:
            return 0.0
        if opposite_clearance_m <= 0.0:
            return turn_side_clearance_m
        return turn_side_clearance_m - opposite_clearance_m

    def _compute_curve_entry_bias_candidate(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> tuple[int, float, float]:
        seed_threshold_deg = self._curve_entry_bias_seed_threshold_deg()
        target_sign = signbit(target_heading_deg)
        front_sign = signbit(front_turn_heading_deg)
        corridor_sign = signbit(corridor_center_heading_deg) if corridor_available else 0

        candidate_sign = 0
        candidate_strength_deg = 0.0
        if (
            front_sign != 0
            and corridor_sign == front_sign
            and max(abs(front_turn_heading_deg), abs(corridor_center_heading_deg)) >= seed_threshold_deg
        ):
            candidate_sign = front_sign
            candidate_strength_deg = max(
                abs(front_turn_heading_deg),
                abs(corridor_center_heading_deg),
            )
        elif (
            front_sign != 0
            and abs(front_turn_heading_deg) >= self._curve_entry_bias_override_threshold_deg()
        ):
            candidate_sign = front_sign
            candidate_strength_deg = abs(front_turn_heading_deg)
        elif (
            corridor_sign != 0
            and abs(corridor_center_heading_deg) >= self._curve_entry_bias_override_threshold_deg()
        ):
            candidate_sign = corridor_sign
            candidate_strength_deg = abs(corridor_center_heading_deg)
        elif (
            target_sign != 0
            and active_heading_source in {"corridor_center", "front_turn", "curve_entry_guard", "wall_follow"}
            and abs(target_heading_deg) >= seed_threshold_deg
        ):
            candidate_sign = target_sign
            candidate_strength_deg = abs(target_heading_deg)

        if candidate_sign == 0:
            return 0, 0.0, 0.0

        open_margin_m = self._curve_entry_open_margin_m(
            turn_sign=candidate_sign,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
        )
        return candidate_sign, candidate_strength_deg, open_margin_m

    def _update_curve_entry_bias_state(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> None:
        if self._curve_confirmed_sign != 0 and self._curve_confirmed_cycles_remaining > 0:
            self._curve_entry_bias_sign = self._curve_confirmed_sign
            self._curve_entry_bias_cycles_remaining = max(
                self._curve_entry_bias_cycles_remaining,
                self._curve_entry_bias_hold_cycles,
            )
            self._curve_entry_bias_override_sign = 0
            self._curve_entry_bias_override_streak = 0
            return

        if (
            front_clearance_m <= 0.0
            or front_clearance_m > self._curve_entry_bias_entry_distance_m()
        ):
            if self._curve_entry_bias_cycles_remaining > 0:
                self._curve_entry_bias_cycles_remaining -= 1
                if self._curve_entry_bias_cycles_remaining <= 0:
                    self._curve_entry_bias_sign = 0
            self._curve_entry_bias_override_sign = 0
            self._curve_entry_bias_override_streak = 0
            return

        (
            candidate_sign,
            candidate_strength_deg,
            open_margin_m,
        ) = self._compute_curve_entry_bias_candidate(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )

        if candidate_sign == 0:
            if self._curve_entry_bias_cycles_remaining > 0:
                self._curve_entry_bias_cycles_remaining -= 1
                if self._curve_entry_bias_cycles_remaining <= 0:
                    self._curve_entry_bias_sign = 0
            self._curve_entry_bias_override_sign = 0
            self._curve_entry_bias_override_streak = 0
            return

        if (
            self._curve_entry_bias_sign == 0
            or self._curve_entry_bias_cycles_remaining <= 0
        ):
            if open_margin_m >= 0.05:
                self._curve_entry_bias_sign = candidate_sign
                self._curve_entry_bias_cycles_remaining = self._curve_entry_bias_hold_cycles
            self._curve_entry_bias_override_sign = 0
            self._curve_entry_bias_override_streak = 0
            return

        if candidate_sign == self._curve_entry_bias_sign:
            self._curve_entry_bias_cycles_remaining = self._curve_entry_bias_hold_cycles
            self._curve_entry_bias_override_sign = 0
            self._curve_entry_bias_override_streak = 0
            return

        if (
            candidate_strength_deg >= self._curve_entry_bias_override_threshold_deg()
            and open_margin_m >= 0.18
        ):
            if self._curve_entry_bias_override_sign == candidate_sign:
                self._curve_entry_bias_override_streak += 1
            else:
                self._curve_entry_bias_override_sign = candidate_sign
                self._curve_entry_bias_override_streak = 1
            if self._curve_entry_bias_override_streak >= 2:
                self._curve_entry_bias_sign = candidate_sign
                self._curve_entry_bias_cycles_remaining = self._curve_entry_bias_hold_cycles
                self._curve_entry_bias_override_sign = 0
                self._curve_entry_bias_override_streak = 0
            return

        if self._curve_entry_bias_cycles_remaining > 0:
            self._curve_entry_bias_cycles_remaining -= 1
            if self._curve_entry_bias_cycles_remaining <= 0:
                self._curve_entry_bias_sign = 0
        self._curve_entry_bias_override_sign = 0
        self._curve_entry_bias_override_streak = 0

    def _apply_curve_entry_bias(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> tuple[float, str]:
        if self._curve_entry_bias_sign == 0 or self._curve_entry_bias_cycles_remaining <= 0:
            return target_heading_deg, active_heading_source
        if active_heading_source in {"", "ambiguity_probe", "startup_hold"}:
            return target_heading_deg, active_heading_source
        if (
            front_clearance_m <= 0.0
            or front_clearance_m > self._curve_entry_bias_entry_distance_m()
        ):
            return target_heading_deg, active_heading_source

        target_sign = signbit(target_heading_deg)
        if target_sign == 0 or target_sign == self._curve_entry_bias_sign:
            return target_heading_deg, active_heading_source
        if (
            self._curve_confirmed_sign != 0
            and self._curve_confirmed_sign == target_sign
            and self._curve_confirmed_cycles_remaining > 0
        ):
            return target_heading_deg, active_heading_source

        (
            candidate_sign,
            candidate_strength_deg,
            open_margin_m,
        ) = self._compute_curve_entry_bias_candidate(
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        if (
            candidate_sign == target_sign
            and candidate_strength_deg >= self._curve_entry_bias_override_threshold_deg()
            and open_margin_m >= 0.18
        ):
            return target_heading_deg, active_heading_source

        if active_heading_source == "wall_follow":
            self._deactivate_wall_follow()
        held_heading_deg = self._curve_entry_bias_sign * max(
            self._startup_consensus_min_heading_deg * 3.0,
            self._turn_commit_min_heading_deg * 0.75,
        )
        return held_heading_deg, "curve_entry_bias"

    def _update_curve_confirmation_state(
        self,
        *,
        front_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> None:
        positive_support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=1,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        negative_support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=-1,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )

        support_heading_deg = 0.0
        if abs(positive_support_heading_deg) >= (
            abs(negative_support_heading_deg) + self._startup_consensus_min_heading_deg
        ):
            support_heading_deg = positive_support_heading_deg
        elif abs(negative_support_heading_deg) >= (
            abs(positive_support_heading_deg) + self._startup_consensus_min_heading_deg
        ):
            support_heading_deg = negative_support_heading_deg
        support_sign = signbit(support_heading_deg)
        strong_threshold_deg = self._curve_confirmation_strong_threshold_deg()
        entry_distance_m = self._curve_confirmation_entry_distance_m()
        strong_support_active = (
            support_sign != 0
            and front_clearance_m > 0.0
            and front_clearance_m <= entry_distance_m
            and abs(support_heading_deg) >= strong_threshold_deg
        )

        if strong_support_active:
            if support_sign == self._curve_confirmation_sign:
                self._curve_confirmation_streak += 1
            else:
                self._curve_confirmation_sign = support_sign
                self._curve_confirmation_streak = 1
        else:
            self._curve_confirmation_sign = 0
            self._curve_confirmation_streak = 0

        if (
            strong_support_active
            and self._curve_confirmation_streak >= 2
        ):
            self._curve_confirmed_sign = support_sign
            self._curve_confirmed_cycles_remaining = max(
                6,
                self._turn_commit_hold_cycles,
            )
            return

        if self._curve_confirmed_cycles_remaining > 0:
            if strong_support_active and support_sign == self._curve_confirmed_sign:
                self._curve_confirmed_cycles_remaining = max(
                    self._curve_confirmed_cycles_remaining,
                    max(6, self._turn_commit_hold_cycles),
                )
            else:
                self._curve_confirmed_cycles_remaining -= 1
                if self._curve_confirmed_cycles_remaining <= 0:
                    self._curve_confirmed_sign = 0

    def _is_curve_confirmed(
        self,
        *,
        target_heading_deg: float,
        front_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> bool:
        target_sign = signbit(target_heading_deg)
        return (
            target_sign != 0
            and self._curve_confirmed_cycles_remaining > 0
            and self._curve_confirmed_sign == target_sign
        )

    def _compute_curve_preentry_limit_deg(
        self,
        *,
        target_heading_deg: float,
        front_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
    ) -> float:
        base_limit_deg = max(
            self._startup_consensus_min_heading_deg * 3.0,
            self._turn_commit_min_heading_deg * 0.75,
        )
        max_limit_deg = min(
            self._wall_follow_limit_deg,
            max(base_limit_deg + 6.0, self._turn_commit_min_heading_deg * 1.5),
        )
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return base_limit_deg

        support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=target_sign,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        if abs(support_heading_deg) <= 0.0:
            return base_limit_deg

        support_cap_deg = min(max_limit_deg, abs(support_heading_deg) * 0.75)
        if front_clearance_m <= 0.0 or front_clearance_m >= self._slow_distance_m:
            return max(base_limit_deg, support_cap_deg)

        proximity_factor = (
            (self._slow_distance_m - front_clearance_m)
            / (self._slow_distance_m - self._stop_distance_m)
        )
        support_factor = min(
            1.0,
            abs(support_heading_deg)
            / max(
                self._curve_confirmation_strong_threshold_deg(),
                self._turn_commit_min_heading_deg,
            ),
        )
        ramp_factor = max(0.0, min(1.0, max(proximity_factor * 0.7, support_factor)))
        dynamic_limit_deg = base_limit_deg + (
            (max(base_limit_deg, support_cap_deg) - base_limit_deg) * ramp_factor
        )
        return float(max(base_limit_deg, min(max_limit_deg, dynamic_limit_deg)))

    def _apply_curve_entry_guard(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        front_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
    ) -> tuple[float, str]:
        if active_heading_source in {"", "ambiguity_probe", "startup_hold"}:
            return target_heading_deg, active_heading_source

        if abs(target_heading_deg) <= 0.0:
            return target_heading_deg, active_heading_source

        if self._is_curve_confirmed(
            target_heading_deg=target_heading_deg,
            front_clearance_m=front_clearance_m,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        ):
            return target_heading_deg, active_heading_source

        preentry_limit_deg = self._compute_curve_preentry_limit_deg(
            target_heading_deg=target_heading_deg,
            front_clearance_m=front_clearance_m,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        recenter_limit_deg = self._compute_recenter_heading_limit_deg(
            target_sign=signbit(target_heading_deg),
            corridor_center_heading_deg=corridor_center_heading_deg,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
        )
        effective_limit_deg = max(preentry_limit_deg, recenter_limit_deg)
        if abs(target_heading_deg) <= effective_limit_deg:
            return target_heading_deg, active_heading_source

        guarded_heading_deg = signbit(target_heading_deg) * effective_limit_deg
        if active_heading_source == "wall_follow":
            self._deactivate_wall_follow()
            return guarded_heading_deg, "curve_entry_guard"
        return guarded_heading_deg, "curve_entry_guard"

    def _compute_adaptive_steering_gain(
        self,
        *,
        active_heading_source: str,
        target_heading_deg: float,
        front_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        wall_follow_heading_deg: float,
    ) -> float:
        if active_heading_source in {"", "ambiguity_probe", "startup_hold"}:
            return self._steering_gain

        directional_signal_deg = max(
            abs(target_heading_deg),
            abs(front_turn_heading_deg),
            abs(corridor_center_heading_deg),
            abs(wall_follow_heading_deg),
        )
        if directional_signal_deg < self._turn_commit_min_heading_deg:
            return self._steering_gain

        engage_distance_m = max(
            self._stop_distance_m + 0.20,
            min(self._slow_distance_m, self._stop_distance_m + 0.45),
        )
        if (
            front_clearance_m <= 0.0
            or front_clearance_m >= engage_distance_m
            or engage_distance_m <= self._stop_distance_m
        ):
            return self._steering_gain

        proximity_factor = (
            (engage_distance_m - front_clearance_m)
            / (engage_distance_m - self._stop_distance_m)
        )
        signal_factor = min(
            1.0,
            directional_signal_deg / max(self._wall_follow_limit_deg, self._turn_commit_min_heading_deg),
        )
        gain_boost = 1.0 + (1.4 * proximity_factor * signal_factor)
        return float(min(1.0, self._steering_gain * gain_boost))

    def _update_wall_follow_state(
        self,
        *,
        base_heading_deg: float,
        base_source: str,
        front_clearance_m: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        corridor_balance_ratio: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
    ) -> None:
        if self._wall_follow_active:
            self._wall_follow_cycles_active += 1
            anchor_support_clearance_m = self._get_wall_follow_anchor_support_clearance_m(
                left_clearance_m=left_clearance_m,
                right_clearance_m=right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
            if (
                anchor_support_clearance_m <= 0.0
                or anchor_support_clearance_m > self._wall_follow_max_clearance_m
            ):
                self._deactivate_wall_follow()
                return
            opposite_support_detected = (
                signbit(front_turn_heading_deg) != 0
                and signbit(front_turn_heading_deg) != self._wall_follow_turn_sign
            ) or (
                signbit(base_heading_deg) != 0
                and signbit(base_heading_deg) != self._wall_follow_turn_sign
            )
            if (
                opposite_support_detected
                and anchor_support_clearance_m <= (self._wall_follow_target_distance_m + 0.14)
            ):
                self._deactivate_wall_follow()
                return
            if (
                self._wall_follow_cycles_active >= self._wall_follow_min_cycles
                and corridor_available
                and corridor_balance_ratio >= self._wall_follow_release_balance_ratio
                and front_clearance_m >= self._slow_distance_m
            ):
                self._deactivate_wall_follow()
                return
            if (
                self._wall_follow_cycles_active >= self._wall_follow_min_cycles
                and abs(front_turn_heading_deg) < (self._wall_follow_activation_heading_deg * 0.5)
                and abs(base_heading_deg) < (self._wall_follow_activation_heading_deg * 0.5)
            ):
                self._deactivate_wall_follow()
                return

        if self._wall_follow_active:
            return

        activation_source = base_source
        activation_heading_deg = max(abs(base_heading_deg), abs(front_turn_heading_deg))
        activation_sign = signbit(base_heading_deg)
        if activation_sign == 0:
            activation_sign = signbit(front_turn_heading_deg)
        if self._turn_commit_cycles_remaining > 0 and self._turn_commit_sign != 0:
            activation_source = "turn_commit"
            activation_sign = self._turn_commit_sign
            activation_heading_deg = max(
                activation_heading_deg,
                self._turn_commit_min_heading_deg,
            )
        if activation_sign == 0:
            return
        if activation_source not in {
            "front_turn",
            "corridor_center",
            "gap_escape",
            "startup_latch",
            "turn_commit",
        }:
            return
        if activation_heading_deg < self._wall_follow_activation_heading_deg:
            return
        if (
            self._curve_confirmed_cycles_remaining <= 0
            or self._curve_confirmed_sign != activation_sign
        ):
            return
        support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=activation_sign,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        if (
            activation_source in {"startup_latch", "turn_commit"}
            and abs(support_heading_deg)
            < max(
                self._startup_consensus_min_heading_deg * 2.0,
                self._turn_commit_min_heading_deg * 0.75,
            )
        ):
            return

        anchor_side = self._select_wall_follow_anchor_side(
            turn_sign=activation_sign,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
        )
        if anchor_side == "":
            return

        self._wall_follow_active = True
        self._wall_follow_anchor_side = anchor_side
        self._wall_follow_turn_sign = activation_sign
        self._wall_follow_cycles_active = 0

    def _select_wall_follow_anchor_side(
        self,
        *,
        turn_sign: int,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
    ) -> str:
        preferred_side = ""
        if turn_sign > 0:
            preferred_side = "left"
        elif turn_sign < 0:
            preferred_side = "right"

        if preferred_side == "left":
            preferred_clearance_m = left_min_m if left_min_m > 0.0 else left_clearance_m
            if 0.0 < preferred_clearance_m <= self._wall_follow_max_clearance_m:
                return "left"
        elif preferred_side == "right":
            preferred_clearance_m = right_min_m if right_min_m > 0.0 else right_clearance_m
            if 0.0 < preferred_clearance_m <= self._wall_follow_max_clearance_m:
                return "right"
        if preferred_side != "":
            return ""

        candidates: list[tuple[float, str]] = []
        if left_min_m > 0.0:
            candidates.append((left_min_m, "left"))
        elif left_clearance_m > 0.0:
            candidates.append((left_clearance_m, "left"))
        if right_min_m > 0.0:
            candidates.append((right_min_m, "right"))
        elif right_clearance_m > 0.0:
            candidates.append((right_clearance_m, "right"))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _get_wall_follow_anchor_clearance_m(
        self,
        *,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> float:
        if self._wall_follow_anchor_side == "left":
            return left_clearance_m
        if self._wall_follow_anchor_side == "right":
            return right_clearance_m
        return 0.0

    def _get_wall_follow_anchor_support_clearance_m(
        self,
        *,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
    ) -> float:
        if self._wall_follow_anchor_side == "left":
            if left_min_m > 0.0:
                return left_min_m
            return left_clearance_m
        if self._wall_follow_anchor_side == "right":
            if right_min_m > 0.0:
                return right_min_m
            return right_clearance_m
        return 0.0

    def _compute_wall_follow_heading_deg(
        self,
        *,
        left_clearance_m: float,
        right_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        base_heading_deg: float,
    ) -> float:
        if not self._wall_follow_active or self._wall_follow_turn_sign == 0:
            return 0.0

        anchor_clearance_m = self._get_wall_follow_anchor_clearance_m(
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
        )
        if anchor_clearance_m <= 0.0:
            return 0.0

        if self._wall_follow_anchor_side == "left":
            wall_heading_deg = (
                anchor_clearance_m - self._wall_follow_target_distance_m
            ) * self._wall_follow_gain_deg_per_m
        else:
            wall_heading_deg = -(
                anchor_clearance_m - self._wall_follow_target_distance_m
            ) * self._wall_follow_gain_deg_per_m

        support_heading_deg = 0.0
        for candidate in (
            front_turn_heading_deg,
            corridor_center_heading_deg,
            base_heading_deg,
        ):
            if signbit(candidate) == self._wall_follow_turn_sign and abs(candidate) > abs(
                support_heading_deg
            ):
                support_heading_deg = candidate

        blended_heading_deg = (
            self._wall_follow_base_weight * wall_heading_deg
            + self._wall_follow_front_turn_weight * support_heading_deg
        )
        if (
            signbit(wall_heading_deg) != 0
            and signbit(wall_heading_deg) != self._wall_follow_turn_sign
            and signbit(support_heading_deg) == self._wall_follow_turn_sign
        ):
            blended_heading_deg = (
                self._wall_follow_front_turn_weight * support_heading_deg
                + (self._wall_follow_base_weight * 0.35) * wall_heading_deg
            )
        blended_heading_deg = float(
            max(-self._wall_follow_limit_deg, min(self._wall_follow_limit_deg, blended_heading_deg))
        )

        min_locked_heading_deg = min(
            self._wall_follow_limit_deg,
            max(self._turn_commit_min_heading_deg, self._wall_follow_activation_heading_deg * 0.6),
        )
        if signbit(support_heading_deg) == self._wall_follow_turn_sign and abs(support_heading_deg) > 0.0:
            min_locked_heading_deg = min(
                self._wall_follow_limit_deg,
                max(
                    min_locked_heading_deg,
                    abs(support_heading_deg) * self._wall_follow_support_min_factor,
                ),
            )
        if signbit(blended_heading_deg) != self._wall_follow_turn_sign:
            blended_heading_deg = self._wall_follow_turn_sign * min_locked_heading_deg
        else:
            blended_heading_deg = self._wall_follow_turn_sign * max(
                abs(blended_heading_deg),
                min_locked_heading_deg,
            )

        return float(
            max(-self._wall_follow_limit_deg, min(self._wall_follow_limit_deg, blended_heading_deg))
        )

    def _deactivate_wall_follow(self) -> None:
        self._wall_follow_active = False
        self._wall_follow_anchor_side = ""
        self._wall_follow_turn_sign = 0
        self._wall_follow_cycles_active = 0

    def _apply_startup_hold(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        gap_heading_deg: float,
        gap_available: bool,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> tuple[float, str, bool, float, str]:
        if self._startup_complete:
            return target_heading_deg, active_heading_source, False, 0.0, "none"

        nav_signal_available = any(
            abs(value) > 0.0
            for value in (
                left_clearance_m,
                right_clearance_m,
                front_turn_heading_deg,
                corridor_center_heading_deg,
                gap_heading_deg if gap_available else 0.0,
            )
        )
        if nav_signal_available:
            self._startup_valid_cycles += 1

        (
            startup_candidate_heading_deg,
            startup_candidate_source,
        ) = self._select_startup_candidate_heading_deg(
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
            centering_heading_deg=centering_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
        )
        startup_candidate_sign = signbit(startup_candidate_heading_deg)
        if startup_candidate_sign != 0:
            if startup_candidate_sign == self._startup_consensus_sign:
                self._startup_consensus_streak += 1
            else:
                self._startup_consensus_sign = startup_candidate_sign
                self._startup_consensus_streak = 1
        else:
            self._startup_consensus_sign = 0
            self._startup_consensus_streak = 0

        if (
            self._startup_latched_sign == 0
            and startup_candidate_sign != 0
            and abs(startup_candidate_heading_deg)
            >= max(
                self._startup_consensus_min_heading_deg,
                self._turn_commit_min_heading_deg * 0.45,
            )
            and self._startup_consensus_streak >= self._startup_valid_cycles_required
        ):
            self._startup_latched_sign = startup_candidate_sign
            self._startup_latch_cycles_remaining = self._startup_latch_cycles
            seeded_heading_deg = startup_candidate_sign * max(
                self._turn_commit_min_heading_deg,
                abs(startup_candidate_heading_deg),
            )
            return (
                seeded_heading_deg,
                "startup_latch",
                False,
                startup_candidate_heading_deg,
                startup_candidate_source,
            )

        if self._startup_latched_sign != 0:
            return (
                target_heading_deg,
                active_heading_source,
                False,
                startup_candidate_heading_deg,
                startup_candidate_source,
            )

        if active_heading_source in {"gap", "gap_escape"}:
            return (
                0.0,
                "startup_hold",
                True,
                startup_candidate_heading_deg,
                startup_candidate_source,
            )

        if self._startup_valid_cycles < self._startup_gap_lockout_cycles:
            return (
                0.0,
                "startup_hold",
                True,
                startup_candidate_heading_deg,
                startup_candidate_source,
            )

        self._startup_complete = True
        return (
            target_heading_deg,
            active_heading_source,
            False,
            startup_candidate_heading_deg,
            startup_candidate_source,
        )

    def _select_startup_candidate_heading_deg(
        self,
        *,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_available: bool,
        centering_heading_deg: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> tuple[float, str]:
        strong_candidates: list[tuple[float, str]] = []
        front_curve_support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=signbit(front_turn_heading_deg),
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        corridor_curve_support_heading_deg = self._curve_entry_support_heading_deg(
            target_sign=signbit(corridor_center_heading_deg),
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_available=corridor_available,
        )
        if (
            signbit(front_curve_support_heading_deg) == signbit(front_turn_heading_deg)
            and abs(front_turn_heading_deg) >= self._startup_consensus_min_heading_deg
        ):
            strong_candidates.append((front_turn_heading_deg, "front_turn"))
        if (
            corridor_available
            and signbit(corridor_curve_support_heading_deg) == signbit(corridor_center_heading_deg)
            and abs(corridor_center_heading_deg) >= self._startup_consensus_min_heading_deg
        ):
            strong_candidates.append((corridor_center_heading_deg, "corridor_center"))

        if strong_candidates:
            return self._choose_startup_candidate(strong_candidates)

        fallback_candidates: list[tuple[float, str]] = []
        if avoidance_active and abs(avoidance_heading_deg) >= self._startup_consensus_min_heading_deg:
            fallback_candidates.append((avoidance_heading_deg, "avoidance"))
        if (
            left_clearance_m > 0.0
            and right_clearance_m > 0.0
            and abs(centering_heading_deg) >= self._startup_consensus_min_heading_deg
        ):
            fallback_candidates.append((centering_heading_deg, "centering"))
        if fallback_candidates:
            return self._choose_startup_candidate(fallback_candidates)

        return 0.0, "none"

    def _choose_startup_candidate(
        self,
        candidates: list[tuple[float, str]],
    ) -> tuple[float, str]:
        positive = [candidate for candidate in candidates if signbit(candidate[0]) > 0]
        negative = [candidate for candidate in candidates if signbit(candidate[0]) < 0]
        if positive and negative:
            strongest_positive = max(positive, key=lambda item: abs(item[0]))
            strongest_negative = max(negative, key=lambda item: abs(item[0]))
            if abs(strongest_positive[0]) >= (
                abs(strongest_negative[0]) + self._startup_consensus_min_heading_deg
            ):
                return strongest_positive
            if abs(strongest_negative[0]) >= (
                abs(strongest_positive[0]) + self._startup_consensus_min_heading_deg
            ):
                return strongest_negative
            return 0.0, "none"

        if positive:
            return max(positive, key=lambda item: abs(item[0]))
        if negative:
            return max(negative, key=lambda item: abs(item[0]))
        return 0.0, "none"

    def _apply_startup_latch(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
    ) -> tuple[float, str]:
        if self._startup_latched_sign == 0 or self._startup_latch_cycles_remaining <= 0:
            return target_heading_deg, active_heading_source

        target_sign = signbit(target_heading_deg)
        if target_sign == self._startup_latched_sign:
            target_heading_deg = self._startup_latched_sign * max(
                abs(target_heading_deg),
                self._turn_commit_min_heading_deg,
            )
        else:
            target_heading_deg = self._startup_latched_sign * self._turn_commit_min_heading_deg
            active_heading_source = "startup_latch"

        self._startup_latch_cycles_remaining -= 1
        if self._startup_latch_cycles_remaining <= 0:
            self._startup_complete = True

        return target_heading_deg, active_heading_source

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

    def _should_use_corridor_center(
        self,
        *,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_balance_ratio: float,
        left_min_m: float,
        right_min_m: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> bool:
        if abs(corridor_center_heading_deg) < self._corridor_min_heading_deg:
            return False
        if corridor_balance_ratio < self._corridor_balance_ratio_threshold:
            return False
        axis_sign = signbit(corridor_axis_heading_deg)
        center_sign = signbit(corridor_center_heading_deg)
        if (
            axis_sign != 0
            and center_sign != 0
            and axis_sign != center_sign
            and abs(corridor_axis_heading_deg)
            >= max(0.75, self._startup_consensus_min_heading_deg * 0.5)
        ):
            return False
        if not avoidance_active:
            return True

        side_min_values = [value for value in (left_min_m, right_min_m) if value > 0.0]
        nearest_side_m = min(side_min_values) if side_min_values else float("inf")
        if nearest_side_m < max(0.10, self._wall_avoid_distance_m * 0.35):
            return signbit(corridor_center_heading_deg) == signbit(avoidance_heading_deg)

        if signbit(corridor_center_heading_deg) != signbit(avoidance_heading_deg):
            return abs(corridor_center_heading_deg) >= (
                abs(avoidance_heading_deg) + self._corridor_override_margin_deg
            )

        return True

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
                target_heading_deg = self._turn_commit_sign * max(
                    abs(target_heading_deg),
                    self._turn_commit_min_heading_deg,
                )
                active_heading_source = "turn_commit"

            self._turn_commit_cycles_remaining -= 1
            if self._turn_commit_cycles_remaining <= 0:
                self._turn_commit_sign = 0

        target_sign = signbit(target_heading_deg)
        if (
            target_sign != 0
            and abs(target_heading_deg) >= self._turn_commit_heading_threshold_deg
            and active_heading_source in {"front_turn", "gap_escape", "corridor_center", "wall_follow"}
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

    def _window_points(
        self,
        ranges: np.ndarray,
        start_deg: int,
        end_deg: int,
    ) -> list[tuple[float, float]]:
        start = int(start_deg)
        end = int(end_deg)
        if start <= end:
            degrees = range(start, end + 1)
        else:
            degrees = list(range(start, 181)) + list(range(-180, end + 1))

        points: list[tuple[float, float]] = []
        for degree in degrees:
            index = int(round(degree)) % ranges.size
            distance = float(ranges[index])
            if distance <= 0.0:
                continue
            rad = math.radians(float(degree))
            forward_m = distance * math.cos(rad)
            lateral_m = distance * math.sin(rad)
            points.append((forward_m, lateral_m))
        return points

    def _window_mean(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(np.mean(values)) if values else 0.0

    def _window_min(self, ranges: np.ndarray, start_deg: int, end_deg: int) -> float:
        values = self._window_values(ranges, start_deg, end_deg)
        return float(min(values)) if values else 0.0

    def _compute_speed_pct(
        self,
        front_clearance_m: float,
        steering_deg: float,
        target_heading_deg: float,
    ) -> float:
        if front_clearance_m <= 0.0 or front_clearance_m <= self._stop_distance_m:
            return 0.0

        angle_magnitude_deg = max(abs(target_heading_deg), abs(steering_deg))
        speed_pct = self._max_speed_pct * math.exp(-0.03 * angle_magnitude_deg)
        speed_pct = max(self._min_speed_pct, min(self._max_speed_pct, speed_pct))

        if front_clearance_m >= self._slow_distance_m:
            distance_factor = 1.0
        else:
            distance_factor = (
                (front_clearance_m - self._stop_distance_m)
                / (self._slow_distance_m - self._stop_distance_m)
            )

        speed_pct *= distance_factor
        if speed_pct > 0.0:
            speed_pct = max(self._min_speed_pct, speed_pct)
        return float(max(0.0, min(self._max_speed_pct, speed_pct)))

    def _compute_motion_front_clearance_m(
        self,
        *,
        effective_front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        target_heading_deg: float,
        active_heading_source: str,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        wall_follow_active: bool,
    ) -> float:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return effective_front_clearance_m

        if active_heading_source not in {
            "curve_entry_guard",
            "corridor_center",
            "front_turn",
            "reference_blend",
            "reference_centerline",
            "reference_curve_commit",
            "reference_free_space",
            "turn_commit",
            "wall_follow",
        }:
            return effective_front_clearance_m

        directional_front_clearance_m = (
            front_left_clearance_m if target_sign > 0 else front_right_clearance_m
        )
        opposite_front_clearance_m = (
            front_right_clearance_m if target_sign > 0 else front_left_clearance_m
        )
        if directional_front_clearance_m <= 0.0:
            return effective_front_clearance_m

        support_candidates = [abs(target_heading_deg)]
        if signbit(front_turn_heading_deg) == target_sign:
            support_candidates.append(abs(front_turn_heading_deg))
        if signbit(corridor_center_heading_deg) == target_sign:
            support_candidates.append(abs(corridor_center_heading_deg))
        support_heading_deg = max(support_candidates)

        # Once a curve is genuinely supported, prefer the turn-side front clearance
        # over the worst-case frontal minimum so we do not stop exactly at the curve entry.
        if support_heading_deg < self._turn_commit_min_heading_deg:
            return effective_front_clearance_m
        if directional_front_clearance_m <= (self._stop_distance_m + 0.05):
            return effective_front_clearance_m
        if (
            not wall_follow_active
            and opposite_front_clearance_m > 0.0
            and directional_front_clearance_m < (opposite_front_clearance_m + 0.10)
        ):
            return effective_front_clearance_m

        return max(effective_front_clearance_m, directional_front_clearance_m)
