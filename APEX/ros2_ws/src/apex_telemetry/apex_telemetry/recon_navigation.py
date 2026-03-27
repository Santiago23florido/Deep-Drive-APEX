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
    nav_mode: str
    corridor_confidence: float
    curve_confidence: float
    preview_heading_deg: float
    corridor_curvature_sign: int
    corridor_curvature_confidence: float
    committed_turn_sign: int
    gate_curve_sign: int
    curve_capture_active: bool
    curve_capture_reason: str
    curve_severity_score: float
    curve_steering_floor_deg: float
    curve_speed_cap_pct: float
    curve_release_reason: str
    same_sign_trim_active: bool
    free_space_candidate_heading_deg: float
    sign_veto_reason: str
    near_wall_mode: str
    straight_veto_active: bool
    startup_adapt_active: bool
    curve_intent_score: float
    curve_intent_sign: int
    curve_evidence_strength: float
    curve_decay_active: bool
    premature_curve_veto: bool
    pre_curve_bias_veto: bool
    straight_corridor_score: float
    curve_gate_open: bool
    curve_gate_reason: str
    geometry_agreement_score: float
    curve_confirm_distance_m: float
    curve_confirm_yaw_deg: float


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


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


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
        self._adaptive_startup_cycles = 12
        self._adaptive_startup_distance_m = 0.75
        self._adaptive_state_alpha = 0.12
        self._adaptive_curve_entry_conf_threshold = 0.48
        self._adaptive_curve_commit_conf_threshold = 0.62
        self._adaptive_curve_release_conf_threshold = 0.20
        self._adaptive_curve_release_cycles = 6
        self._adaptive_curve_intent_entry_score = 0.24
        self._adaptive_curve_intent_follow_score = 0.52
        self._adaptive_curve_intent_release_score = 0.10
        self._adaptive_curve_intent_decay_keep = 0.58
        self._adaptive_curve_intent_decay_drop = 0.38
        self._adaptive_curve_intent_decay_switch = 0.28
        self._adaptive_curve_intent_switch_score = 0.46
        self._adaptive_near_wall_distance_m = max(
            self._stop_distance_m + 0.08,
            self._wall_avoid_distance_m * 0.85,
        )
        self._adaptive_curve_geometry_min_heading_deg = max(
            4.0,
            self._corridor_min_heading_deg + 2.0,
        )
        self._adaptive_curve_gate_required_cycles = 4
        self._adaptive_curve_hold_cycles = 10
        self._adaptive_curve_hold_release_cycles = 3
        self._adaptive_curve_hold_min_evidence = 0.22
        self._adaptive_curve_gate_far_balance_ratio = 0.78
        self._adaptive_curve_gate_min_agreement_score = 0.62
        self._adaptive_curve_straight_heading_cap_deg = 1.5
        self._adaptive_curve_preconfirm_heading_cap_deg = 3.0
        self._adaptive_curve_confirm_distance_threshold_m = 0.10
        self._adaptive_curve_confirm_yaw_threshold_deg = 1.5
        self._adaptive_curve_motion_release_distance_m = 0.08
        self._adaptive_launch_speed_floor_pct = max(self._min_speed_pct, 25.0)
        self._adaptive_curve_speed_floor_pct = max(
            0.0,
            self._adaptive_launch_speed_floor_pct * max(self._min_turn_speed_factor, 0.70),
        )
        self._adaptive_curve_capture_cycles = 5
        self._adaptive_curve_capture_release_cycles = 3
        self._adaptive_curve_capture_steering_floor_deg = max(
            6.0,
            0.45 * self._steering_limit_deg,
        )
        self._adaptive_curve_follow_steering_floor_deg = max(
            8.0,
            0.60 * self._steering_limit_deg,
        )
        self._adaptive_curve_max_steering_floor_deg = 0.70 * self._steering_limit_deg
        self._adaptive_curve_capture_speed_cap_min_pct = 22.0
        self._adaptive_curve_follow_speed_cap_min_pct = 18.0
        self._adaptive_curve_entry_steering_gain_floor = max(
            self._steering_gain,
            min(0.56, self._steering_gain + 0.18),
        )
        self._adaptive_curve_follow_steering_gain_floor = max(
            self._adaptive_curve_entry_steering_gain_floor,
            min(0.74, self._steering_gain + 0.33),
        )
        self._adaptive_straight_heading_limit_deg = 3.0
        self._adaptive_straight_centerline_limit_deg = 1.25
        self._adaptive_straight_axis_guidance_limit_deg = 1.35
        self._adaptive_straight_parallel_score_threshold = 0.32
        self._adaptive_straight_wall_symmetry_min = 0.72
        self._adaptive_straight_wall_pair_min_heading_deg = 10.0
        self._adaptive_midpoint_band_m = 0.14
        self._adaptive_min_midpoint_span_m = 0.22
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
        self._adaptive_cycle_count = 0
        self._adaptive_front_fallback_rate = 0.0
        self._adaptive_corridor_width_m = 0.0
        self._adaptive_lateral_margin_m = 0.0
        self._adaptive_preview_release_gain = 0.72
        self._adaptive_corridor_confidence_ema = 0.0
        self._adaptive_curve_confidence_ema = 0.0
        self._adaptive_last_vote_sign = 0
        self._adaptive_last_vote_streak = 0
        self._adaptive_committed_turn_sign = 0
        self._adaptive_committed_yaw_progress_deg = 0.0
        self._adaptive_release_streak = 0
        self._adaptive_opposite_sign_streak = 0
        self._adaptive_curve_intent_score = 0.0
        self._adaptive_curve_intent_sign = 0
        self._adaptive_curve_evidence_strength = 0.0
        self._adaptive_curve_decay_active = False
        self._adaptive_premature_curve_veto = False
        self._adaptive_curve_gate_open_streak = 0
        self._adaptive_curve_gate_open = False
        self._adaptive_curve_gate_reason = "init"
        self._adaptive_curve_geometry_agreement_score = 0.0
        self._adaptive_curve_confirm_sign = 0
        self._adaptive_curve_confirm_distance_m = 0.0
        self._adaptive_curve_confirm_yaw_deg = 0.0
        self._adaptive_curve_hold_sign = 0
        self._adaptive_curve_hold_cycles_remaining = 0
        self._adaptive_curve_hold_release_streak = 0
        self._adaptive_curve_capture_sign = 0
        self._adaptive_curve_capture_cycles_remaining = 0
        self._adaptive_curve_capture_release_streak = 0
        self._adaptive_curve_capture_reason = "inactive"
        self._adaptive_curve_release_reason = "none"
        self._adaptive_curve_heading_memory_deg = 0.0
        self._adaptive_last_pose_yaw_deg = None
        self._adaptive_last_pose_distance_m = None
        self._adaptive_last_pose_yaw_delta_deg = 0.0
        self._adaptive_last_nav_mode = "startup_adapt"

    def compute_command(
        self,
        scan_ranges_m: np.ndarray,
        *,
        pose_yaw_deg: float | None = None,
        pose_yaw_change_deg: float | None = None,
        pose_distance_from_phase_start_m: float | None = None,
        pose_lateral_drift_m: float | None = None,
    ) -> ReconCommand:
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

        yaw_delta_deg, distance_delta_m = self._update_adaptive_pose_feedback(
            pose_yaw_deg=pose_yaw_deg,
            pose_distance_from_phase_start_m=pose_distance_from_phase_start_m,
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
                nav_mode,
                corridor_confidence,
                curve_confidence,
                corridor_curvature_sign,
                corridor_curvature_confidence,
                committed_turn_sign,
                gate_curve_sign,
                curve_capture_active,
                curve_capture_reason,
                curve_severity_score,
                curve_steering_floor_deg,
                curve_speed_cap_pct,
                curve_release_reason,
                same_sign_trim_active,
                free_space_candidate_heading_deg,
                sign_veto_reason,
                near_wall_mode,
                straight_veto_active,
                startup_adapt_active,
                curve_intent_score,
                curve_intent_sign,
                curve_evidence_strength,
                curve_decay_active,
                premature_curve_veto,
                pre_curve_bias_veto,
                straight_corridor_score,
                curve_gate_open,
                curve_gate_reason,
                geometry_agreement_score,
                curve_confirm_distance_m,
                curve_confirm_yaw_deg,
            ) = self._compute_adaptive_simple_command(
                shrunk_scan=shrunk,
                front_clearance_m=front_clearance_m,
                effective_front_clearance_m=effective_front_clearance_m,
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
                left_wall_heading_deg=left_wall_heading_deg,
                right_wall_heading_deg=right_wall_heading_deg,
                left_wall_points=left_wall_points,
                right_wall_points=right_wall_points,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
                pose_yaw_change_deg=pose_yaw_change_deg,
                pose_distance_from_phase_start_m=pose_distance_from_phase_start_m,
                pose_lateral_drift_m=pose_lateral_drift_m,
                yaw_delta_deg=yaw_delta_deg,
                distance_delta_m=distance_delta_m,
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
                nav_mode=nav_mode,
                target_heading_deg=target_heading_deg,
                front_clearance_m=effective_front_clearance_m,
                front_turn_heading_deg=front_turn_heading_deg,
                corridor_center_heading_deg=corridor_center_heading_deg,
                wall_follow_heading_deg=wall_follow_heading_deg,
                committed_turn_sign=committed_turn_sign,
                curve_confirm_distance_m=curve_confirm_distance_m,
            )
            steering_pre_servo_deg = max(
                -self._steering_limit_deg,
                min(self._steering_limit_deg, target_heading_deg * steering_gain),
            )
            nav_mode = "legacy"
            corridor_confidence = 0.0
            curve_confidence = 0.0
            corridor_curvature_sign = 0
            corridor_curvature_confidence = 0.0
            committed_turn_sign = self._turn_commit_sign
            free_space_candidate_heading_deg = gap_heading_deg if gap_available else 0.0
            sign_veto_reason = "none"
            near_wall_mode = "none"
            straight_veto_active = False
            startup_adapt_active = False
            curve_intent_score = 0.0
            curve_intent_sign = 0
            curve_evidence_strength = 0.0
            curve_decay_active = False
            premature_curve_veto = False
            pre_curve_bias_veto = False
            straight_corridor_score = 0.0
            curve_gate_open = False
            curve_gate_reason = "legacy"
            geometry_agreement_score = 0.0
            curve_confirm_distance_m = 0.0
            curve_confirm_yaw_deg = 0.0
            gate_curve_sign = 0
            curve_capture_active = False
            curve_capture_reason = "inactive"
            curve_severity_score = 0.0
            curve_steering_floor_deg = 0.0
            curve_speed_cap_pct = 0.0
            curve_release_reason = "none"
            same_sign_trim_active = False
        motion_front_clearance_m = self._compute_motion_front_clearance_m(
            effective_front_clearance_m=effective_front_clearance_m,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            target_heading_deg=target_heading_deg,
            active_heading_source=active_heading_source,
            nav_mode=nav_mode,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            wall_follow_active=wall_follow_active,
            corridor_confidence=corridor_confidence,
        )
        if (
            curve_steering_floor_deg > 0.0
            and nav_mode in {"curve_capture", "curve_entry", "curve_follow"}
        ):
            steering_sign = signbit(steering_pre_servo_deg)
            if steering_sign == 0:
                steering_sign = (
                    committed_turn_sign
                    if committed_turn_sign != 0
                    else (
                        gate_curve_sign
                        if gate_curve_sign != 0
                        else (curve_intent_sign if curve_intent_sign != 0 else signbit(target_heading_deg))
                    )
                )
            if steering_sign != 0:
                steering_floor_deg = curve_steering_floor_deg
                if same_sign_trim_active:
                    steering_floor_deg = max(2.0, steering_floor_deg * 0.75)
                steering_pre_servo_deg = steering_sign * _clamp(
                    abs(steering_pre_servo_deg),
                    steering_floor_deg,
                    self._steering_limit_deg,
                )
        speed_pct = self._compute_speed_pct(
            motion_front_clearance_m,
            steering_pre_servo_deg,
            target_heading_deg,
            nav_mode=nav_mode,
            pose_distance_from_phase_start_m=pose_distance_from_phase_start_m,
            committed_turn_sign=committed_turn_sign,
            curve_confirm_distance_m=curve_confirm_distance_m,
        )
        if (
            curve_speed_cap_pct > 0.0
            and nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and (pose_distance_from_phase_start_m or 0.0) >= self._adaptive_curve_motion_release_distance_m
        ):
            speed_pct = min(speed_pct, curve_speed_cap_pct)
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
            nav_mode=nav_mode,
            corridor_confidence=corridor_confidence,
            curve_confidence=curve_confidence,
            preview_heading_deg=simple_preview_heading_deg if simple_preview_available else 0.0,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
            committed_turn_sign=committed_turn_sign,
            gate_curve_sign=gate_curve_sign,
            curve_capture_active=curve_capture_active,
            curve_capture_reason=curve_capture_reason,
            curve_severity_score=curve_severity_score,
            curve_steering_floor_deg=curve_steering_floor_deg,
            curve_speed_cap_pct=curve_speed_cap_pct,
            curve_release_reason=curve_release_reason,
            same_sign_trim_active=same_sign_trim_active,
            free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            sign_veto_reason=sign_veto_reason,
            near_wall_mode=near_wall_mode,
            straight_veto_active=straight_veto_active,
            startup_adapt_active=startup_adapt_active,
            curve_intent_score=curve_intent_score,
            curve_intent_sign=curve_intent_sign,
            curve_evidence_strength=curve_evidence_strength,
            curve_decay_active=curve_decay_active,
            premature_curve_veto=premature_curve_veto,
            pre_curve_bias_veto=pre_curve_bias_veto,
            straight_corridor_score=straight_corridor_score,
            curve_gate_open=curve_gate_open,
            curve_gate_reason=curve_gate_reason,
            geometry_agreement_score=geometry_agreement_score,
            curve_confirm_distance_m=curve_confirm_distance_m,
            curve_confirm_yaw_deg=curve_confirm_yaw_deg,
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
            nav_mode="idle",
            corridor_confidence=0.0,
            curve_confidence=0.0,
            preview_heading_deg=0.0,
            corridor_curvature_sign=0,
            corridor_curvature_confidence=0.0,
            committed_turn_sign=0,
            gate_curve_sign=0,
            curve_capture_active=False,
            curve_capture_reason="inactive",
            curve_severity_score=0.0,
            curve_steering_floor_deg=0.0,
            curve_speed_cap_pct=0.0,
            curve_release_reason="none",
            same_sign_trim_active=False,
            free_space_candidate_heading_deg=0.0,
            sign_veto_reason="none",
            near_wall_mode="none",
            straight_veto_active=False,
            startup_adapt_active=False,
            curve_intent_score=0.0,
            curve_intent_sign=0,
            curve_evidence_strength=0.0,
            curve_decay_active=False,
            premature_curve_veto=False,
            pre_curve_bias_veto=False,
            straight_corridor_score=0.0,
            curve_gate_open=False,
            curve_gate_reason="idle",
            geometry_agreement_score=0.0,
            curve_confirm_distance_m=0.0,
            curve_confirm_yaw_deg=0.0,
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
        if len(front_side_candidates) == 2:
            front_side_min_m = min(front_side_candidates)
            front_side_max_m = max(front_side_candidates)

            side_clearance_candidates = [
                value for value in (left_clearance_m, right_clearance_m) if value > 0.0
            ]
            if len(side_clearance_candidates) == 2 and gap_available:
                front_balance_ratio = front_side_min_m / max(front_side_max_m, 1e-6)
                side_balance_ratio = min(side_clearance_candidates) / max(
                    max(side_clearance_candidates), 1e-6
                )
                front_open_sign = signbit(front_left_clearance_m - front_right_clearance_m)
                side_open_sign = signbit(left_clearance_m - right_clearance_m)

                # When the frontal center window is sparse, the worst front-side
                # sector alone is too pessimistic for an off-center car in a
                # still-open corridor. In that case, use a conservative
                # centerline proxy instead of treating the tighter front corner
                # as a hard frontal stop.
                if (
                    front_open_sign != 0
                    and front_open_sign == side_open_sign
                    and front_balance_ratio <= 0.55
                    and side_balance_ratio <= 0.55
                    and front_side_max_m > (self._stop_distance_m + 0.12)
                ):
                    centerline_proxy_front_clearance_m = 0.5 * (
                        front_side_min_m + front_side_max_m
                    )
                    return min(self._slow_distance_m, centerline_proxy_front_clearance_m), True

            return front_side_min_m, True

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

    def _update_adaptive_pose_feedback(
        self,
        *,
        pose_yaw_deg: float | None,
        pose_distance_from_phase_start_m: float | None,
    ) -> tuple[float, float]:
        yaw_delta_deg = 0.0
        distance_delta_m = 0.0
        if pose_yaw_deg is not None:
            pose_yaw_deg = float(pose_yaw_deg)
            if self._adaptive_last_pose_yaw_deg is not None:
                yaw_delta_deg = _normalize_angle_deg(
                    pose_yaw_deg - self._adaptive_last_pose_yaw_deg
                )
            self._adaptive_last_pose_yaw_deg = pose_yaw_deg
        self._adaptive_last_pose_yaw_delta_deg = float(yaw_delta_deg)

        if pose_distance_from_phase_start_m is not None:
            pose_distance_from_phase_start_m = float(pose_distance_from_phase_start_m)
            if self._adaptive_last_pose_distance_m is not None:
                distance_delta_m = (
                    pose_distance_from_phase_start_m - self._adaptive_last_pose_distance_m
                )
            self._adaptive_last_pose_distance_m = pose_distance_from_phase_start_m

        if (
            self._adaptive_curve_intent_sign != 0
            and self._adaptive_curve_intent_score
            >= self._adaptive_curve_intent_entry_score
        ):
            signed_yaw_delta_deg = yaw_delta_deg * self._adaptive_curve_intent_sign
            if signed_yaw_delta_deg >= -1.5:
                self._adaptive_committed_yaw_progress_deg = max(
                    0.0,
                    self._adaptive_committed_yaw_progress_deg + signed_yaw_delta_deg,
                )
        else:
            self._adaptive_committed_yaw_progress_deg *= 0.5

        return float(yaw_delta_deg), float(distance_delta_m)

    def _update_startup_adaptation_state(
        self,
        *,
        front_clearance_fallback_used: bool,
        corridor_balance_ratio: float,
        left_clearance_m: float,
        right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        pose_distance_from_phase_start_m: float | None,
    ) -> bool:
        self._adaptive_cycle_count += 1
        alpha = self._adaptive_state_alpha
        fallback_value = 1.0 if front_clearance_fallback_used else 0.0
        self._adaptive_front_fallback_rate = (
            (1.0 - alpha) * self._adaptive_front_fallback_rate
            + alpha * fallback_value
        )

        if left_clearance_m > 0.0 and right_clearance_m > 0.0:
            corridor_width_m = left_clearance_m + right_clearance_m
            if self._adaptive_corridor_width_m <= 0.0:
                self._adaptive_corridor_width_m = corridor_width_m
            else:
                self._adaptive_corridor_width_m = (
                    (1.0 - alpha) * self._adaptive_corridor_width_m
                    + alpha * corridor_width_m
                )

        side_min_candidates = [value for value in (left_min_m, right_min_m) if value > 0.0]
        if side_min_candidates:
            lateral_margin_m = min(side_min_candidates)
            if self._adaptive_lateral_margin_m <= 0.0:
                self._adaptive_lateral_margin_m = lateral_margin_m
            else:
                self._adaptive_lateral_margin_m = (
                    (1.0 - alpha) * self._adaptive_lateral_margin_m
                    + alpha * lateral_margin_m
                )

        balance_norm = _clamp(
            (corridor_balance_ratio - self._corridor_balance_ratio_threshold) / 0.55,
            0.0,
            1.0,
        )
        self._adaptive_preview_release_gain = _clamp(
            0.55 + (0.22 * balance_norm) - (0.18 * self._adaptive_front_fallback_rate),
            0.55,
            0.85,
        )

        adapt_distance_m = max(
            0.55,
            min(
                1.0,
                max(
                    self._adaptive_startup_distance_m,
                    self._adaptive_corridor_width_m * 0.45 if self._adaptive_corridor_width_m > 0.0 else 0.0,
                ),
            ),
        )
        startup_adapt_active = self._adaptive_cycle_count <= self._adaptive_startup_cycles
        if pose_distance_from_phase_start_m is not None:
            startup_adapt_active = startup_adapt_active or (
                float(pose_distance_from_phase_start_m) < adapt_distance_m
            )

        return bool(startup_adapt_active)

    def _compute_adaptive_corridor_confidence(
        self,
        *,
        corridor_available: bool,
        corridor_balance_ratio: float,
        left_wall_points: int,
        right_wall_points: int,
        left_clearance_m: float,
        right_clearance_m: float,
        preview_heading_deg: float,
        preview_available: bool,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        front_clearance_fallback_used: bool,
    ) -> float:
        wall_points_conf = _clamp(
            min(left_wall_points, right_wall_points)
            / max(1.0, float(self._corridor_wall_min_points * 2)),
            0.0,
            1.0,
        )
        coverage_conf = 0.0
        if left_clearance_m > 0.0 and right_clearance_m > 0.0:
            coverage_conf = 1.0
        elif left_clearance_m > 0.0 or right_clearance_m > 0.0:
            coverage_conf = 0.45

        balance_conf = _clamp(
            (corridor_balance_ratio - 0.05) / 0.55,
            0.0,
            1.0,
        )

        pos_score = 0.0
        neg_score = 0.0
        for heading_deg, weight in (
            (preview_heading_deg if preview_available else 0.0, 1.0),
            (corridor_center_heading_deg, 0.9),
            (corridor_axis_heading_deg, 0.8),
        ):
            heading_sign = signbit(heading_deg)
            if heading_sign == 0:
                continue
            score = weight * _clamp(abs(heading_deg) / 12.0, 0.0, 1.0)
            if heading_sign > 0:
                pos_score += score
            else:
                neg_score += score
        total_score = pos_score + neg_score
        sign_consistency = max(pos_score, neg_score) / total_score if total_score > 0.0 else 0.0

        fallback_penalty = 0.25 if front_clearance_fallback_used else 0.0
        raw_confidence = (
            0.10
            + (0.24 * coverage_conf)
            + (0.22 * wall_points_conf)
            + (0.22 * balance_conf)
            + (0.22 * sign_consistency)
            - fallback_penalty
        )
        if corridor_available:
            raw_confidence = max(raw_confidence, 0.48)

        corridor_confidence = _clamp(raw_confidence, 0.0, 1.0)
        self._adaptive_corridor_confidence_ema = (
            0.75 * self._adaptive_corridor_confidence_ema
            + 0.25 * corridor_confidence
        )
        return float(_clamp(self._adaptive_corridor_confidence_ema, 0.0, 1.0))

    def _interpolate_wall_lateral_m(
        self,
        *,
        points_xy: list[tuple[float, float]],
        target_forward_m: float,
    ) -> tuple[float, int] | None:
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
        distance = np.abs(forward - float(target_forward_m))
        close_mask = distance <= self._adaptive_midpoint_band_m
        if np.count_nonzero(close_mask) < 2:
            nearest_indices = np.argsort(distance)[: min(6, distance.size)]
            if nearest_indices.size < 2:
                return None
            sample_forward = forward[nearest_indices]
            sample_lateral = lateral[nearest_indices]
            weights = 1.0 / np.maximum(distance[nearest_indices], 0.03)
            support = int(nearest_indices.size)
        else:
            sample_forward = forward[close_mask]
            sample_lateral = lateral[close_mask]
            weights = 1.0 / np.maximum(distance[close_mask], 0.02)
            support = int(np.count_nonzero(close_mask))

        if sample_forward.size < 2:
            return None
        weighted_lateral_m = float(np.average(sample_lateral, weights=weights))
        return weighted_lateral_m, support

    def _compute_corridor_curvature_features(
        self,
        *,
        left_wall_points_xy: list[tuple[float, float]],
        right_wall_points_xy: list[tuple[float, float]],
    ) -> tuple[int, float, float, float, bool]:
        if (
            len(left_wall_points_xy) < self._corridor_wall_min_points
            or len(right_wall_points_xy) < self._corridor_wall_min_points
        ):
            return 0, 0.0, 0.0, 0.0, False

        left_forward = np.asarray([point[0] for point in left_wall_points_xy], dtype=np.float32)
        right_forward = np.asarray([point[0] for point in right_wall_points_xy], dtype=np.float32)
        left_forward = left_forward[left_forward > 0.05]
        right_forward = right_forward[right_forward > 0.05]
        if left_forward.size < self._corridor_wall_min_points or right_forward.size < self._corridor_wall_min_points:
            return 0, 0.0, 0.0, 0.0, False

        overlap_min_m = max(0.20, float(max(np.min(left_forward), np.min(right_forward))))
        overlap_max_m = float(min(np.max(left_forward), np.max(right_forward)))
        if overlap_max_m - overlap_min_m < self._adaptive_min_midpoint_span_m:
            return 0, 0.0, 0.0, 0.0, False

        near_forward_m = _clamp(overlap_min_m + 0.10, 0.25, overlap_max_m - 0.12)
        far_forward_m = _clamp(overlap_max_m - 0.08, near_forward_m + 0.16, overlap_max_m)
        if far_forward_m - near_forward_m < self._adaptive_min_midpoint_span_m:
            return 0, 0.0, 0.0, 0.0, False

        left_near = self._interpolate_wall_lateral_m(
            points_xy=left_wall_points_xy,
            target_forward_m=near_forward_m,
        )
        right_near = self._interpolate_wall_lateral_m(
            points_xy=right_wall_points_xy,
            target_forward_m=near_forward_m,
        )
        left_far = self._interpolate_wall_lateral_m(
            points_xy=left_wall_points_xy,
            target_forward_m=far_forward_m,
        )
        right_far = self._interpolate_wall_lateral_m(
            points_xy=right_wall_points_xy,
            target_forward_m=far_forward_m,
        )
        if left_near is None or right_near is None or left_far is None or right_far is None:
            return 0, 0.0, 0.0, 0.0, False

        left_near_y_m, left_near_support = left_near
        right_near_y_m, right_near_support = right_near
        left_far_y_m, left_far_support = left_far
        right_far_y_m, right_far_support = right_far
        if left_near_y_m <= right_near_y_m or left_far_y_m <= right_far_y_m:
            return 0, 0.0, 0.0, 0.0, False

        near_center_y_m = 0.5 * (left_near_y_m + right_near_y_m)
        far_center_y_m = 0.5 * (left_far_y_m + right_far_y_m)
        near_heading_deg = float(math.degrees(math.atan2(near_center_y_m, near_forward_m)))
        far_heading_deg = float(math.degrees(math.atan2(far_center_y_m, far_forward_m)))
        tangent_heading_deg = float(
            math.degrees(
                math.atan2(
                    far_center_y_m - near_center_y_m,
                    max(1e-3, far_forward_m - near_forward_m),
                )
            )
        )
        curvature_delta_deg = _normalize_angle_deg(far_heading_deg - near_heading_deg)
        curvature_sign = signbit(tangent_heading_deg)
        if curvature_sign == 0 and abs(curvature_delta_deg) >= 1.5:
            curvature_sign = signbit(curvature_delta_deg)

        support_confidence = _clamp(
            min(
                left_near_support,
                right_near_support,
                left_far_support,
                right_far_support,
            )
            / max(2.0, float(self._corridor_wall_min_points)),
            0.0,
            1.0,
        )
        geometry_confidence = _clamp(
            max(
                abs(tangent_heading_deg) / 10.0,
                abs(curvature_delta_deg) / 8.0,
            ),
            0.0,
            1.0,
        )
        curvature_confidence = support_confidence * geometry_confidence
        if curvature_confidence < 0.12:
            return 0, 0.0, tangent_heading_deg, curvature_delta_deg, True

        return (
            curvature_sign,
            float(_clamp(curvature_confidence, 0.0, 1.0)),
            tangent_heading_deg,
            curvature_delta_deg,
            True,
        )

    def _compute_weighted_curve_vote(
        self,
        *,
        preview_heading_deg: float,
        preview_available: bool,
        corridor_axis_heading_deg: float,
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
        corridor_curvature_heading_deg: float,
        corridor_center_heading_deg: float,
        front_turn_heading_deg: float,
        free_space_candidate_heading_deg: float,
        free_space_available: bool,
        corridor_available: bool,
        corridor_confidence: float,
        left_clearance_m: float,
        right_clearance_m: float,
    ) -> tuple[int, float, float]:
        axis_sign = signbit(corridor_axis_heading_deg)
        left_wall_sign = signbit(left_wall_heading_deg)
        right_wall_sign = signbit(right_wall_heading_deg)
        wall_geometry_available = (
            left_clearance_m > 0.0
            and right_clearance_m > 0.0
            and left_wall_sign != 0
            and right_wall_sign != 0
        )
        pos_score = 0.0
        neg_score = 0.0
        pos_components = 0
        neg_components = 0
        max_support_heading_deg = 0.0

        def add_score(candidate_sign: int, score: float, heading_deg: float) -> None:
            nonlocal pos_score, neg_score, pos_components, neg_components, max_support_heading_deg
            if candidate_sign == 0 or score <= 0.0:
                return
            max_support_heading_deg = max(max_support_heading_deg, abs(heading_deg))
            if candidate_sign > 0:
                pos_score += score
                pos_components += 1
            else:
                neg_score += score
                neg_components += 1

        if wall_geometry_available and left_wall_sign == right_wall_sign:
            wall_pair_strength_deg = min(abs(left_wall_heading_deg), abs(right_wall_heading_deg))
            if wall_pair_strength_deg >= (0.85 * self._adaptive_curve_geometry_min_heading_deg):
                add_score(
                    left_wall_sign,
                    0.44 * _clamp(wall_pair_strength_deg / 14.0, 0.0, 1.0),
                    wall_pair_strength_deg,
                )

        if axis_sign != 0 and abs(corridor_axis_heading_deg) >= self._adaptive_curve_geometry_min_heading_deg:
            add_score(
                axis_sign,
                0.34 * _clamp(abs(corridor_axis_heading_deg) / 14.0, 0.0, 1.0),
                corridor_axis_heading_deg,
            )

        center_sign = signbit(corridor_center_heading_deg)
        if (
            corridor_available
            and corridor_confidence >= 0.26
            and center_sign != 0
            and abs(corridor_center_heading_deg) >= max(5.0, self._adaptive_curve_geometry_min_heading_deg - 0.5)
        ):
            add_score(
                center_sign,
                0.24
                * _clamp(abs(corridor_center_heading_deg) / 12.0, 0.0, 1.0)
                * _clamp(corridor_confidence, 0.26, 1.0),
                corridor_center_heading_deg,
            )

        front_turn_sign = signbit(front_turn_heading_deg)
        if (
            front_turn_sign != 0
            and abs(front_turn_heading_deg) >= 4.0
            and (
                corridor_available
                or preview_available
                or abs(free_space_candidate_heading_deg) >= self._reference_curve_confirm_heading_deg
            )
        ):
            add_score(
                front_turn_sign,
                0.18 * _clamp(abs(front_turn_heading_deg) / 12.0, 0.0, 1.0),
                front_turn_heading_deg,
            )

        if (
            corridor_curvature_sign != 0
            and corridor_curvature_confidence >= 0.35
            and abs(corridor_curvature_heading_deg) >= self._corridor_min_heading_deg
        ):
            add_score(
                corridor_curvature_sign,
                0.30
                * _clamp(abs(corridor_curvature_heading_deg) / 12.0, 0.0, 1.0)
                * _clamp(corridor_curvature_confidence, 0.35, 1.0),
                corridor_curvature_heading_deg,
            )

        if (
            preview_available
            and abs(preview_heading_deg) >= max(4.5, self._reference_curve_confirm_heading_deg * 0.75)
        ):
            add_score(
                signbit(preview_heading_deg),
                0.16 * _clamp(abs(preview_heading_deg) / 14.0, 0.0, 1.0),
                preview_heading_deg,
            )

        if free_space_available and abs(free_space_candidate_heading_deg) >= self._reference_curve_confirm_heading_deg:
            add_score(
                signbit(free_space_candidate_heading_deg),
                0.08 * _clamp(abs(free_space_candidate_heading_deg) / 16.0, 0.0, 1.0),
                free_space_candidate_heading_deg,
            )

        total_score = pos_score + neg_score
        if total_score <= 1e-6:
            return 0, 0.0, 0.0

        dominant_score = max(pos_score, neg_score)
        vote_consistency = dominant_score / total_score
        dominant_components = pos_components if pos_score > neg_score else neg_components
        if (
            dominant_score < 0.24
            or vote_consistency < 0.64
            or max_support_heading_deg < self._adaptive_curve_geometry_min_heading_deg
            or dominant_components < 2
        ):
            return 0, float(vote_consistency), float(max_support_heading_deg)

        curve_sign = 1 if pos_score > neg_score else -1
        return curve_sign, float(vote_consistency), float(max_support_heading_deg)

    def _infer_curve_gate_sign(
        self,
        *,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
        corridor_curvature_heading_deg: float,
        preview_heading_deg: float,
        preview_available: bool,
        free_space_candidate_heading_deg: float,
        free_space_available: bool,
    ) -> int:
        pos_score = 0.0
        neg_score = 0.0

        def add_score(
            heading_deg: float,
            *,
            weight: float,
            min_heading_deg: float,
            confidence: float = 1.0,
        ) -> None:
            nonlocal pos_score, neg_score
            heading_sign = signbit(heading_deg)
            if heading_sign == 0:
                return
            magnitude = _clamp(abs(heading_deg) / max(min_heading_deg, 1.0), 0.0, 1.4)
            score = weight * magnitude * _clamp(confidence, 0.0, 1.0)
            if score <= 0.0:
                return
            if heading_sign > 0:
                pos_score += score
            else:
                neg_score += score

        add_score(
            corridor_center_heading_deg,
            weight=0.34,
            min_heading_deg=max(5.0, self._adaptive_curve_geometry_min_heading_deg - 0.5),
        )
        add_score(
            front_turn_heading_deg,
            weight=0.26,
            min_heading_deg=4.0,
        )
        add_score(
            corridor_axis_heading_deg,
            weight=0.22,
            min_heading_deg=max(6.0, self._adaptive_curve_geometry_min_heading_deg),
        )
        if corridor_curvature_sign != 0 and corridor_curvature_confidence > 0.0:
            add_score(
                corridor_curvature_heading_deg,
                weight=0.18,
                min_heading_deg=max(4.0, self._corridor_min_heading_deg),
                confidence=_clamp(corridor_curvature_confidence, 0.0, 1.0),
            )
        if preview_available:
            add_score(
                preview_heading_deg,
                weight=0.10,
                min_heading_deg=4.0,
            )
        if free_space_available:
            add_score(
                free_space_candidate_heading_deg,
                weight=0.08,
                min_heading_deg=5.0,
            )

        total_score = pos_score + neg_score
        if total_score <= 1e-6:
            return 0
        dominant_score = max(pos_score, neg_score)
        support_ratio = dominant_score / total_score
        if dominant_score < 0.20 or support_ratio < 0.66:
            return 0
        return 1 if pos_score > neg_score else -1

    def _infer_preview_dominant_curve_sign(
        self,
        *,
        preview_heading_deg: float,
        preview_available: bool,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        corridor_axis_heading_deg: float,
        corridor_confidence: float,
        effective_front_clearance_m: float,
        gate_curve_sign: int,
    ) -> int:
        if not preview_available:
            return 0
        preview_sign = signbit(preview_heading_deg)
        if preview_sign == 0:
            return 0
        if abs(preview_heading_deg) < max(10.0, self._reference_curve_confirm_heading_deg):
            return 0
        if corridor_confidence < 0.20:
            return 0
        if (
            effective_front_clearance_m > 0.0
            and effective_front_clearance_m <= (self._stop_distance_m + 0.08)
        ):
            return 0

        front_turn_sign = signbit(front_turn_heading_deg)
        front_turn_supports_preview = (
            front_turn_sign == preview_sign
            or abs(front_turn_heading_deg)
            <= max(6.0, 0.45 * abs(preview_heading_deg))
        )
        if not front_turn_supports_preview:
            return 0

        center_sign = signbit(corridor_center_heading_deg)
        axis_sign = signbit(corridor_axis_heading_deg)
        opposing_corridor = (
            center_sign != 0
            and axis_sign != 0
            and center_sign == axis_sign
            and center_sign != preview_sign
        )
        if not opposing_corridor and gate_curve_sign == 0:
            return 0

        if gate_curve_sign != 0 and gate_curve_sign == preview_sign:
            return preview_sign

        return preview_sign

    def _update_adaptive_curve_capture_state(
        self,
        *,
        gate_curve_sign: int,
        curve_gate_open: bool,
        geometry_agreement_score: float,
        curve_confidence: float,
        curve_intent_sign: int,
        curve_intent_score: float,
        committed_turn_sign: int,
        corridor_confidence: float,
        corridor_axis_heading_deg: float,
        front_turn_heading_deg: float,
        preview_heading_deg: float,
        corridor_center_heading_deg: float,
        straight_heading_deg: float,
        effective_front_clearance_m: float,
    ) -> tuple[int, bool, str, str]:
        capture_reason = self._adaptive_curve_capture_reason
        release_reason = "none"
        exit_alignment = (
            abs(preview_heading_deg) < 4.5
            and abs(corridor_center_heading_deg) < 4.5
            and abs(straight_heading_deg) < 2.5
        )
        support_active = (
            curve_gate_open
            and gate_curve_sign != 0
            and geometry_agreement_score >= self._adaptive_curve_gate_min_agreement_score
            and curve_confidence >= max(0.30, self._adaptive_curve_entry_conf_threshold * 0.60)
        )
        preview_support_count = 0
        if (
            gate_curve_sign != 0
            and corridor_confidence >= 0.30
            and (
                effective_front_clearance_m <= 0.0
                or effective_front_clearance_m > (self._stop_distance_m + 0.08)
            )
        ):
            if (
                signbit(preview_heading_deg) == gate_curve_sign
                and abs(preview_heading_deg) >= max(4.5, self._reference_curve_confirm_heading_deg * 0.75)
            ):
                preview_support_count += 1
            if signbit(front_turn_heading_deg) == gate_curve_sign and abs(front_turn_heading_deg) >= 4.0:
                preview_support_count += 1
            if (
                signbit(corridor_center_heading_deg) == gate_curve_sign
                and abs(corridor_center_heading_deg) >= max(4.5, self._adaptive_curve_geometry_min_heading_deg - 0.5)
            ):
                preview_support_count += 1
            if (
                signbit(corridor_axis_heading_deg) == gate_curve_sign
                and abs(corridor_axis_heading_deg) >= max(5.5, self._adaptive_curve_geometry_min_heading_deg)
            ):
                preview_support_count += 1
        preview_support_active = (
            not support_active
            and gate_curve_sign != 0
            and preview_support_count >= 3
        )
        preview_dominant_support_active = (
            not support_active
            and not preview_support_active
            and gate_curve_sign != 0
            and signbit(preview_heading_deg) == gate_curve_sign
            and abs(preview_heading_deg) >= max(10.0, self._reference_curve_confirm_heading_deg)
            and corridor_confidence >= 0.20
            and (
                effective_front_clearance_m <= 0.0
                or effective_front_clearance_m > (self._stop_distance_m + 0.08)
            )
            and (
                signbit(front_turn_heading_deg) == gate_curve_sign
                or abs(front_turn_heading_deg) <= max(6.0, 0.45 * abs(preview_heading_deg))
            )
        )
        support_active = support_active or preview_support_active or preview_dominant_support_active

        if committed_turn_sign != 0:
            if self._adaptive_curve_capture_sign != 0:
                release_reason = "confirmed"
            self._adaptive_curve_capture_sign = 0
            self._adaptive_curve_capture_cycles_remaining = 0
            self._adaptive_curve_capture_release_streak = 0
            self._adaptive_curve_capture_reason = "inactive"
            self._adaptive_curve_release_reason = release_reason
            return 0, False, "inactive", release_reason

        if self._adaptive_curve_capture_sign == 0 and support_active:
            self._adaptive_curve_capture_sign = gate_curve_sign
            self._adaptive_curve_capture_cycles_remaining = self._adaptive_curve_capture_cycles
            self._adaptive_curve_capture_release_streak = 0
            if preview_dominant_support_active:
                capture_reason = "preview_dominant_alignment"
            elif preview_support_active:
                capture_reason = "preview_alignment"
            else:
                capture_reason = "gate_open"
        elif self._adaptive_curve_capture_sign != 0:
            if support_active and gate_curve_sign == self._adaptive_curve_capture_sign:
                self._adaptive_curve_capture_cycles_remaining = self._adaptive_curve_capture_cycles
                self._adaptive_curve_capture_release_streak = 0
                if preview_dominant_support_active:
                    capture_reason = "preview_dominant_hold"
                elif preview_support_active:
                    capture_reason = "preview_hold"
                else:
                    capture_reason = "gate_hold"
            elif support_active and gate_curve_sign != self._adaptive_curve_capture_sign:
                self._adaptive_curve_capture_release_streak += 1
                capture_reason = "sign_contradiction"
            elif self._adaptive_curve_capture_cycles_remaining > 0:
                self._adaptive_curve_capture_cycles_remaining -= 1
                capture_reason = "min_hold"
            elif (
                curve_intent_sign == self._adaptive_curve_capture_sign
                and curve_intent_score > self._adaptive_curve_intent_release_score
            ):
                self._adaptive_curve_capture_release_streak = 0
                capture_reason = "intent_hold"
            elif exit_alignment:
                self._adaptive_curve_capture_release_streak += 1
                capture_reason = "recentered"
            else:
                self._adaptive_curve_capture_release_streak += 1
                capture_reason = "geometry_lost"

            if self._adaptive_curve_capture_release_streak >= self._adaptive_curve_capture_release_cycles:
                release_reason = capture_reason
                self._adaptive_curve_capture_sign = 0
                self._adaptive_curve_capture_cycles_remaining = 0
                self._adaptive_curve_capture_release_streak = 0
                capture_reason = "inactive"

        self._adaptive_curve_capture_reason = capture_reason
        self._adaptive_curve_release_reason = release_reason
        capture_active = self._adaptive_curve_capture_sign != 0
        return (
            self._adaptive_curve_capture_sign,
            capture_active,
            capture_reason,
            release_reason,
        )

    def _preview_curve_follow_ready(
        self,
        *,
        curve_capture_active: bool,
        curve_capture_sign: int,
        curve_capture_reason: str,
        curve_gate_open: bool,
        curve_gate_reason: str,
        curve_confidence: float,
        geometry_agreement_score: float,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        corridor_confidence: float,
        preview_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        effective_front_clearance_m: float,
    ) -> bool:
        if not curve_capture_active or curve_capture_sign == 0:
            return False
        if curve_capture_reason not in {
            "preview_alignment",
            "preview_hold",
            "preview_dominant_alignment",
            "preview_dominant_hold",
            "gate_open",
            "gate_hold",
            "near_wall_gate_hold",
        }:
            return False
        if curve_intent_sign != curve_capture_sign:
            return False
        if curve_intent_score < self._adaptive_curve_intent_entry_score:
            return False
        if curve_evidence_strength < self._adaptive_curve_hold_min_evidence:
            return False
        if corridor_confidence < 0.20:
            return False
        if (
            effective_front_clearance_m > 0.0
            and effective_front_clearance_m <= (self._stop_distance_m + 0.05)
        ):
            return False

        preview_support = (
            signbit(preview_heading_deg) == curve_capture_sign
            and abs(preview_heading_deg)
            >= max(10.0, self._reference_curve_confirm_heading_deg)
        )
        front_turn_support = (
            signbit(front_turn_heading_deg) == curve_capture_sign
            and abs(front_turn_heading_deg)
            >= max(6.0, 0.75 * self._reference_curve_confirm_heading_deg)
        )
        center_support = (
            signbit(corridor_center_heading_deg) == curve_capture_sign
            and abs(corridor_center_heading_deg)
            >= max(3.5, self._adaptive_curve_geometry_min_heading_deg - 1.0)
        )
        gate_like_support = (
            curve_gate_open
            or curve_gate_reason in {"open", "open_candidate", "sign_disagreement"}
            or geometry_agreement_score
            >= max(0.45, 0.75 * self._adaptive_curve_gate_min_agreement_score)
            or curve_confidence >= max(0.42, self._adaptive_curve_entry_conf_threshold)
        )
        return bool(preview_support and front_turn_support and (center_support or gate_like_support))

    def _compute_curve_severity_score(
        self,
        *,
        turn_sign: int,
        corridor_center_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_curvature_confidence: float,
        geometry_agreement_score: float,
        left_min_m: float,
        right_min_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
    ) -> float:
        if turn_sign == 0:
            return 0.0

        inside_min_m = left_min_m if turn_sign > 0 else right_min_m
        outside_min_m = right_min_m if turn_sign > 0 else left_min_m
        inside_front_m = front_left_clearance_m if turn_sign > 0 else front_right_clearance_m
        outside_front_m = front_right_clearance_m if turn_sign > 0 else front_left_clearance_m

        asymmetry_score = 0.0
        if inside_min_m > 0.0 and outside_min_m > 0.0:
            asymmetry_score = _clamp(
                (outside_min_m - inside_min_m) / max(outside_min_m, 0.10),
                0.0,
                1.0,
            )
        front_bias_score = 0.0
        if inside_front_m > 0.0 and outside_front_m > 0.0:
            front_bias_score = _clamp(
                (outside_front_m - inside_front_m) / max(outside_front_m, 0.10),
                0.0,
                1.0,
            )

        return float(
            _clamp(
                0.26 * _clamp(abs(corridor_center_heading_deg) / 16.0, 0.0, 1.0)
                + 0.24 * _clamp(abs(front_turn_heading_deg) / 14.0, 0.0, 1.0)
                + 0.20 * _clamp(corridor_curvature_confidence, 0.0, 1.0)
                + 0.20 * _clamp(geometry_agreement_score, 0.0, 1.0)
                + 0.06 * asymmetry_score
                + 0.04 * front_bias_score,
                0.0,
                1.0,
            )
        )

    def _compute_curve_steering_floor_deg(
        self,
        *,
        nav_mode: str,
        turn_sign: int,
        curve_severity_score: float,
        left_min_m: float,
        right_min_m: float,
    ) -> float:
        if turn_sign == 0:
            return 0.0

        if nav_mode == "curve_capture":
            base_floor_deg = self._adaptive_curve_capture_steering_floor_deg
        else:
            base_floor_deg = self._adaptive_curve_follow_steering_floor_deg
        base_floor_deg += (
            self._adaptive_curve_max_steering_floor_deg - base_floor_deg
        ) * _clamp(curve_severity_score, 0.0, 1.0)

        inside_min_m = left_min_m if turn_sign > 0 else right_min_m
        if inside_min_m > 0.0 and inside_min_m < self._adaptive_near_wall_distance_m:
            trim_factor = _clamp(
                inside_min_m / max(self._adaptive_near_wall_distance_m, 0.05),
                0.45,
                1.0,
            )
            base_floor_deg *= trim_factor

        min_floor_deg = 3.0 if nav_mode == "curve_capture" else 4.0
        return float(_clamp(base_floor_deg, min_floor_deg, self._steering_limit_deg))

    def _compute_curve_speed_cap_pct(
        self,
        *,
        nav_mode: str,
        curve_severity_score: float,
        inside_front_clearance_m: float,
    ) -> float:
        if nav_mode not in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}:
            return 0.0

        if nav_mode == "curve_capture":
            speed_cap_pct = self._adaptive_launch_speed_floor_pct - (
                3.0 * _clamp(curve_severity_score, 0.0, 1.0)
            )
            speed_cap_pct = _clamp(
                speed_cap_pct,
                self._adaptive_curve_capture_speed_cap_min_pct,
                self._adaptive_launch_speed_floor_pct,
            )
        else:
            follow_max_pct = min(self._adaptive_launch_speed_floor_pct, 22.0)
            speed_cap_pct = follow_max_pct - (4.0 * _clamp(curve_severity_score, 0.0, 1.0))
            speed_cap_pct = _clamp(
                speed_cap_pct,
                self._adaptive_curve_follow_speed_cap_min_pct,
                follow_max_pct,
            )

        if inside_front_clearance_m > 0.0 and inside_front_clearance_m < self._slow_distance_m:
            clearance_factor = _clamp(
                (inside_front_clearance_m - self._stop_distance_m)
                / max(self._slow_distance_m - self._stop_distance_m, 0.05),
                0.45,
                1.0,
            )
            speed_cap_pct = max(
                self._adaptive_curve_follow_speed_cap_min_pct,
                speed_cap_pct * clearance_factor,
            )

        return float(_clamp(speed_cap_pct, 0.0, self._max_speed_pct))

    def _compute_premature_curve_veto(
        self,
        *,
        preview_heading_deg: float,
        preview_available: bool,
        front_turn_heading_deg: float,
        free_space_candidate_heading_deg: float,
        corridor_axis_heading_deg: float,
        corridor_available: bool,
        corridor_confidence: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
    ) -> bool:
        local_pos_score = 0.0
        local_neg_score = 0.0
        for heading_deg, weight in (
            (preview_heading_deg if preview_available else 0.0, 0.50),
            (front_turn_heading_deg, 0.35),
            (free_space_candidate_heading_deg, 0.15),
        ):
            heading_sign = signbit(heading_deg)
            if heading_sign == 0:
                continue
            score = weight * _clamp(abs(heading_deg) / 18.0, 0.0, 1.0)
            if heading_sign > 0:
                local_pos_score += score
            else:
                local_neg_score += score

        local_total_score = local_pos_score + local_neg_score
        if local_total_score <= 0.28:
            return False
        if not corridor_available or corridor_confidence < 0.30:
            return False
        if abs(corridor_axis_heading_deg) >= self._adaptive_curve_geometry_min_heading_deg:
            return False
        if (
            corridor_curvature_sign != 0
            and corridor_curvature_confidence >= 0.35
        ):
            return False
        return True

    def _compute_pre_curve_bias_veto(
        self,
        *,
        preview_heading_deg: float,
        preview_available: bool,
        front_turn_heading_deg: float,
        avoidance_heading_deg: float,
        corridor_axis_heading_deg: float,
        corridor_available: bool,
        corridor_confidence: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
        straight_corridor_score: float,
        front_clearance_fallback_used: bool,
    ) -> bool:
        if not corridor_available:
            return False
        if corridor_confidence < 0.28 and straight_corridor_score < 0.16:
            return False
        if abs(corridor_axis_heading_deg) >= self._adaptive_curve_geometry_min_heading_deg:
            return False
        if (
            corridor_curvature_sign != 0
            and corridor_curvature_confidence >= 0.35
        ):
            return False

        local_bias_deg = 0.0
        if preview_available:
            local_bias_deg = max(local_bias_deg, abs(preview_heading_deg))
        local_bias_deg = max(local_bias_deg, abs(front_turn_heading_deg))
        local_bias_deg = max(local_bias_deg, abs(avoidance_heading_deg))
        if local_bias_deg < 3.0:
            return False

        if front_clearance_fallback_used:
            return True

        return (
            straight_corridor_score >= (0.55 * self._adaptive_straight_parallel_score_threshold)
            and local_bias_deg >= 4.0
        )

    def _compute_straight_corridor_score(
        self,
        *,
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
        corridor_axis_heading_deg: float,
        corridor_available: bool,
        corridor_confidence: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
    ) -> float:
        if not corridor_available or corridor_confidence < 0.18:
            return 0.0
        left_sign = signbit(left_wall_heading_deg)
        right_sign = signbit(right_wall_heading_deg)
        if left_sign == 0 or right_sign == 0 or left_sign == right_sign:
            return 0.0

        left_abs_deg = abs(left_wall_heading_deg)
        right_abs_deg = abs(right_wall_heading_deg)
        wall_pair_strength_deg = min(left_abs_deg, right_abs_deg)
        if wall_pair_strength_deg < self._adaptive_straight_wall_pair_min_heading_deg:
            return 0.0

        symmetry_ratio = wall_pair_strength_deg / max(left_abs_deg, right_abs_deg)
        if symmetry_ratio < self._adaptive_straight_wall_symmetry_min:
            return 0.0

        axis_small_factor = _clamp(
            1.0
            - (
                abs(corridor_axis_heading_deg)
                / max(self._adaptive_curve_geometry_min_heading_deg, 5.0)
            ),
            0.0,
            1.0,
        )
        if axis_small_factor <= 0.0:
            return 0.0

        axis_sign = signbit(corridor_axis_heading_deg)
        if (
            corridor_curvature_sign != 0
            and corridor_curvature_confidence >= 0.32
            and axis_sign == corridor_curvature_sign
            and abs(corridor_axis_heading_deg)
            >= (0.75 * self._adaptive_curve_geometry_min_heading_deg)
        ):
            return 0.0

        wall_factor = _clamp(wall_pair_strength_deg / 28.0, 0.0, 1.0)
        confidence_factor = _clamp((corridor_confidence - 0.18) / 0.42, 0.0, 1.0)
        return float(
            _clamp(
                symmetry_ratio * wall_factor * axis_small_factor * max(0.55, confidence_factor),
                0.0,
                1.0,
            )
        )

    def _compute_adaptive_curve_confidence(
        self,
        *,
        curve_sign: int,
        vote_consistency: float,
        max_support_heading_deg: float,
        corridor_confidence: float,
        yaw_delta_deg: float,
        pose_yaw_change_deg: float | None,
    ) -> float:
        if curve_sign == 0:
            self._adaptive_curve_confidence_ema *= 0.55
            return float(self._adaptive_curve_confidence_ema)

        magnitude_conf = _clamp(max_support_heading_deg / 18.0, 0.0, 1.0)
        stability_conf = 0.0
        if curve_sign == self._adaptive_last_vote_sign:
            stability_conf = _clamp(self._adaptive_last_vote_streak / 4.0, 0.0, 1.0)

        yaw_alignment_conf = 0.0
        if abs(yaw_delta_deg) >= 0.15:
            yaw_alignment_conf = 1.0 if signbit(yaw_delta_deg) == curve_sign else 0.0
        elif pose_yaw_change_deg is not None and abs(pose_yaw_change_deg) >= 3.0:
            yaw_alignment_conf = 1.0 if signbit(pose_yaw_change_deg) == curve_sign else 0.0

        raw_confidence = (
            0.34 * vote_consistency
            + 0.30 * magnitude_conf
            + 0.18 * stability_conf
            + 0.10 * yaw_alignment_conf
            + 0.08 * corridor_confidence
        )
        curve_confidence = _clamp(raw_confidence, 0.0, 1.0)
        self._adaptive_curve_confidence_ema = (
            0.60 * self._adaptive_curve_confidence_ema
            + 0.40 * curve_confidence
        )
        return float(_clamp(self._adaptive_curve_confidence_ema, 0.0, 1.0))

    def _reset_adaptive_curve_confirmation(self) -> None:
        self._adaptive_curve_confirm_sign = 0
        self._adaptive_curve_confirm_distance_m = 0.0
        self._adaptive_curve_confirm_yaw_deg = 0.0
        self._adaptive_committed_turn_sign = 0
        self._adaptive_committed_yaw_progress_deg = 0.0

    def _clear_adaptive_curve_tracking(self, *, clear_heading_memory: bool) -> None:
        self._adaptive_curve_intent_sign = 0
        self._adaptive_curve_intent_score = 0.0
        self._adaptive_curve_evidence_strength = 0.0
        self._adaptive_curve_decay_active = False
        self._adaptive_last_vote_sign = 0
        self._adaptive_last_vote_streak = 0
        self._adaptive_premature_curve_veto = False
        self._adaptive_curve_hold_sign = 0
        self._adaptive_curve_hold_cycles_remaining = 0
        self._adaptive_curve_hold_release_streak = 0
        self._adaptive_curve_capture_sign = 0
        self._adaptive_curve_capture_cycles_remaining = 0
        self._adaptive_curve_capture_release_streak = 0
        self._adaptive_curve_capture_reason = "inactive"
        self._adaptive_curve_release_reason = "none"
        self._reset_adaptive_curve_confirmation()
        if clear_heading_memory:
            self._adaptive_curve_heading_memory_deg = 0.0

    def _update_adaptive_curve_hold_state(
        self,
        *,
        curve_gate_open: bool,
        corridor_confidence: float,
        curve_heading_deg: float,
        committed_turn_sign: int,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        preview_heading_deg: float,
        corridor_center_heading_deg: float,
        straight_heading_deg: float,
    ) -> tuple[int, bool]:
        exit_alignment = (
            abs(preview_heading_deg) < 4.5
            and abs(corridor_center_heading_deg) < 4.5
            and abs(straight_heading_deg) < 2.5
        )
        support_sign = (
            committed_turn_sign
            if committed_turn_sign != 0
            else (
                curve_intent_sign
                if curve_intent_sign != 0
                else self._adaptive_curve_hold_sign
            )
        )
        support_heading_deg = float(curve_heading_deg)
        if (
            self._adaptive_curve_hold_sign != 0
            and signbit(self._adaptive_curve_heading_memory_deg)
            == self._adaptive_curve_hold_sign
            and abs(self._adaptive_curve_heading_memory_deg) > abs(support_heading_deg)
        ):
            support_heading_deg = float(self._adaptive_curve_heading_memory_deg)

        strong_curve_support = bool(
            support_sign != 0
            and signbit(support_heading_deg) == support_sign
            and (
                committed_turn_sign == support_sign
                or (
                    curve_intent_sign == support_sign
                    and curve_intent_score >= self._adaptive_curve_intent_entry_score
                    and curve_evidence_strength >= self._adaptive_curve_hold_min_evidence
                    and corridor_confidence >= 0.36
                    and (curve_gate_open or abs(support_heading_deg) >= 4.5)
                )
            )
        )
        weak_same_sign_support = bool(
            self._adaptive_curve_hold_sign != 0
            and support_sign == self._adaptive_curve_hold_sign
            and signbit(support_heading_deg) == self._adaptive_curve_hold_sign
            and abs(support_heading_deg) >= 3.0
            and curve_intent_score > self._adaptive_curve_intent_release_score
            and corridor_confidence >= 0.30
        )

        if strong_curve_support:
            self._adaptive_curve_hold_sign = support_sign
            self._adaptive_curve_hold_cycles_remaining = self._adaptive_curve_hold_cycles
            self._adaptive_curve_hold_release_streak = 0
        elif self._adaptive_curve_hold_sign != 0:
            if weak_same_sign_support and not exit_alignment:
                self._adaptive_curve_hold_cycles_remaining = max(
                    0,
                    self._adaptive_curve_hold_cycles_remaining - 1,
                )
                self._adaptive_curve_hold_release_streak = 0
            else:
                if exit_alignment and curve_evidence_strength < 0.20:
                    self._adaptive_curve_hold_release_streak += 1
                    decay_cycles = 2
                else:
                    self._adaptive_curve_hold_release_streak = 0
                    decay_cycles = 1
                self._adaptive_curve_hold_cycles_remaining = max(
                    0,
                    self._adaptive_curve_hold_cycles_remaining - decay_cycles,
                )

            if (
                self._adaptive_curve_hold_cycles_remaining <= 0
                or self._adaptive_curve_hold_release_streak
                >= self._adaptive_curve_hold_release_cycles
            ):
                self._adaptive_curve_hold_sign = 0
                self._adaptive_curve_hold_cycles_remaining = 0
                self._adaptive_curve_hold_release_streak = 0

        curve_hold_active = bool(
            self._adaptive_curve_hold_sign != 0
            and self._adaptive_curve_hold_cycles_remaining > 0
        )
        return self._adaptive_curve_hold_sign, curve_hold_active

    def _update_adaptive_curve_gate(
        self,
        *,
        corridor_available: bool,
        corridor_confidence: float,
        corridor_balance_ratio: float,
        left_wall_points: int,
        right_wall_points: int,
        left_clearance_m: float,
        right_clearance_m: float,
        curve_sign: int,
        vote_consistency: float,
        max_support_heading_deg: float,
        corridor_axis_heading_deg: float,
        corridor_center_heading_deg: float,
        front_turn_heading_deg: float,
        corridor_curvature_sign: int,
        corridor_curvature_confidence: float,
        corridor_curvature_heading_deg: float,
    ) -> tuple[bool, str, float]:
        geometry_agreement_score = 0.0
        gate_reason = "no_curve_geometry"

        wall_coverage_ready = (
            corridor_available
            and corridor_confidence >= 0.28
            and min(left_wall_points, right_wall_points) >= self._corridor_wall_min_points
            and left_clearance_m > 0.0
            and right_clearance_m > 0.0
        )
        if not wall_coverage_ready:
            self._adaptive_curve_gate_open_streak = 0
            self._adaptive_curve_gate_open = False
            self._adaptive_curve_gate_reason = "insufficient_wall_coverage"
            self._adaptive_curve_geometry_agreement_score = 0.0
            return False, self._adaptive_curve_gate_reason, 0.0

        if curve_sign == 0 or max_support_heading_deg < self._adaptive_curve_geometry_min_heading_deg:
            self._adaptive_curve_gate_open_streak = 0
            self._adaptive_curve_gate_open = False
            self._adaptive_curve_gate_reason = gate_reason
            self._adaptive_curve_geometry_agreement_score = 0.0
            return False, gate_reason, 0.0

        support_score = 0.0
        opposition_score = 0.0
        support_components = 0

        def add_component(
            heading_deg: float,
            *,
            weight: float,
            min_heading_deg: float,
            confidence: float = 1.0,
        ) -> None:
            nonlocal support_score, opposition_score, support_components
            heading_sign = signbit(heading_deg)
            if heading_sign == 0:
                return
            magnitude = _clamp(abs(heading_deg) / max(min_heading_deg, 1.0), 0.0, 1.5)
            score = weight * magnitude * _clamp(confidence, 0.0, 1.0)
            if score <= 0.0:
                return
            if heading_sign == curve_sign:
                support_score += score
                support_components += 1
            else:
                opposition_score += score

        add_component(
            corridor_axis_heading_deg,
            weight=0.34,
            min_heading_deg=max(6.0, self._adaptive_curve_geometry_min_heading_deg),
        )
        add_component(
            corridor_center_heading_deg,
            weight=0.30,
            min_heading_deg=max(5.0, self._adaptive_curve_geometry_min_heading_deg - 0.5),
        )
        add_component(
            front_turn_heading_deg,
            weight=0.20,
            min_heading_deg=4.0,
        )
        if corridor_curvature_sign != 0 and corridor_curvature_confidence > 0.0:
            add_component(
                corridor_curvature_heading_deg,
                weight=0.16,
                min_heading_deg=max(4.0, self._corridor_min_heading_deg),
                confidence=_clamp(corridor_curvature_confidence, 0.0, 1.0),
            )

        total_score = support_score + opposition_score
        support_ratio = support_score / total_score if total_score > 1e-6 else 0.0
        support_count_factor = _clamp(support_components / 3.0, 0.0, 1.0)

        visibility_bonus = 0.0
        if corridor_balance_ratio < self._adaptive_curve_gate_far_balance_ratio:
            visibility_bonus += 0.32 * _clamp(
                (self._adaptive_curve_gate_far_balance_ratio - corridor_balance_ratio) / 0.30,
                0.0,
                1.0,
            )
        if (
            signbit(front_turn_heading_deg) == curve_sign
            and abs(front_turn_heading_deg) >= 3.0
        ):
            visibility_bonus += 0.18 * _clamp(abs(front_turn_heading_deg) / 8.0, 0.0, 1.0)
        if (
            corridor_curvature_sign == curve_sign
            and corridor_curvature_confidence >= 0.35
            and abs(corridor_curvature_heading_deg) >= max(4.0, self._corridor_min_heading_deg)
        ):
            visibility_bonus += 0.20 * _clamp(corridor_curvature_confidence, 0.35, 1.0)
        if (
            signbit(corridor_center_heading_deg) == curve_sign
            and abs(corridor_center_heading_deg) >= max(7.0, self._adaptive_curve_geometry_min_heading_deg)
        ):
            visibility_bonus += 0.12 * _clamp(abs(corridor_center_heading_deg) / 12.0, 0.0, 1.0)

        geometry_agreement_score = _clamp(
            support_ratio
            * _clamp(0.28 + (0.22 * support_count_factor) + visibility_bonus, 0.0, 1.0)
            * _clamp(vote_consistency, 0.0, 1.0),
            0.0,
            1.0,
        )

        gate_candidate_open = (
            support_components >= 2
            and support_ratio >= 0.72
            and geometry_agreement_score >= self._adaptive_curve_gate_min_agreement_score
        )

        if opposition_score >= 0.18 and support_ratio < 0.78:
            gate_reason = "sign_disagreement"
        elif corridor_balance_ratio >= self._adaptive_curve_gate_far_balance_ratio and visibility_bonus < 0.22:
            gate_reason = "far_balanced_corridor"
        elif support_components < 2:
            gate_reason = "insufficient_geometry_support"
        elif geometry_agreement_score < self._adaptive_curve_gate_min_agreement_score:
            gate_reason = "geometry_not_stable"
        else:
            gate_reason = "open_candidate"

        if gate_candidate_open:
            self._adaptive_curve_gate_open_streak += 1
        else:
            self._adaptive_curve_gate_open_streak = 0

        curve_gate_open = (
            self._adaptive_curve_gate_open_streak >= self._adaptive_curve_gate_required_cycles
        )
        self._adaptive_curve_gate_open = curve_gate_open
        self._adaptive_curve_gate_reason = "open" if curve_gate_open else gate_reason
        self._adaptive_curve_geometry_agreement_score = float(geometry_agreement_score)
        return (
            curve_gate_open,
            self._adaptive_curve_gate_reason,
            float(geometry_agreement_score),
        )

    def _update_adaptive_curve_confirmation(
        self,
        *,
        curve_gate_open: bool,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        curve_confidence: float,
        corridor_available: bool,
        corridor_confidence: float,
        yaw_delta_deg: float,
        distance_delta_m: float,
    ) -> tuple[int, float, float, bool]:
        confirmation_sign = 0
        if (
            curve_intent_sign != 0
            and curve_intent_score >= self._adaptive_curve_intent_entry_score
            and curve_evidence_strength >= 0.22
            and corridor_available
            and corridor_confidence >= 0.28
            and (
                curve_gate_open
                or self._adaptive_committed_turn_sign == curve_intent_sign
            )
        ):
            confirmation_sign = curve_intent_sign

        if confirmation_sign == 0:
            self._reset_adaptive_curve_confirmation()
            return 0, 0.0, 0.0, False

        if self._adaptive_curve_confirm_sign != confirmation_sign:
            self._adaptive_curve_confirm_sign = confirmation_sign
            self._adaptive_curve_confirm_distance_m = 0.0
            self._adaptive_curve_confirm_yaw_deg = 0.0

        signed_yaw_delta_deg = float(yaw_delta_deg) * confirmation_sign
        if signed_yaw_delta_deg >= -0.5:
            self._adaptive_curve_confirm_yaw_deg = max(
                0.0,
                self._adaptive_curve_confirm_yaw_deg + max(0.0, signed_yaw_delta_deg),
            )
        else:
            self._adaptive_curve_confirm_yaw_deg *= 0.7

        if (
            distance_delta_m > 0.0
            and curve_confidence >= self._adaptive_curve_entry_conf_threshold * 0.75
            and curve_evidence_strength >= 0.22
        ):
            self._adaptive_curve_confirm_distance_m = min(
                self._adaptive_curve_confirm_distance_threshold_m + 0.20,
                self._adaptive_curve_confirm_distance_m + min(float(distance_delta_m), 0.12),
            )
        elif distance_delta_m < -0.02:
            self._adaptive_curve_confirm_distance_m *= 0.6

        curve_confirmed = bool(
            self._adaptive_curve_confirm_yaw_deg >= self._adaptive_curve_confirm_yaw_threshold_deg
            or (
                self._adaptive_curve_confirm_distance_m
                >= self._adaptive_curve_confirm_distance_threshold_m
                and corridor_confidence >= 0.36
                and curve_confidence >= self._adaptive_curve_entry_conf_threshold * 0.85
            )
        )
        self._adaptive_committed_turn_sign = confirmation_sign if curve_confirmed else 0
        return (
            self._adaptive_committed_turn_sign,
            float(self._adaptive_curve_confirm_distance_m),
            float(self._adaptive_curve_confirm_yaw_deg),
            curve_confirmed,
        )

    def _update_adaptive_curve_intent(
        self,
        *,
        curve_sign: int,
        curve_confidence: float,
        vote_consistency: float,
        max_support_heading_deg: float,
        corridor_available: bool,
        corridor_confidence: float,
        premature_curve_veto: bool,
        curve_entry_blocked: bool,
        pose_yaw_change_deg: float | None,
    ) -> tuple[int, float, float, bool]:
        if curve_sign != 0:
            if curve_sign == self._adaptive_last_vote_sign:
                self._adaptive_last_vote_streak += 1
            else:
                self._adaptive_last_vote_sign = curve_sign
                self._adaptive_last_vote_streak = 1
        else:
            self._adaptive_last_vote_sign = 0
            self._adaptive_last_vote_streak = 0

        evidence_strength = 0.0
        if (
            curve_sign != 0
            and not premature_curve_veto
            and not curve_entry_blocked
            and corridor_available
        ):
            evidence_strength = _clamp(
                0.58 * curve_confidence
                + 0.26 * vote_consistency
                + 0.16 * _clamp(max_support_heading_deg / 14.0, 0.0, 1.0),
                0.0,
                1.0,
            )
        elif (
            curve_sign != 0
            and not premature_curve_veto
            and not curve_entry_blocked
            and corridor_confidence >= 0.28
        ):
            evidence_strength = _clamp(
                0.48 * curve_confidence
                + 0.22 * vote_consistency
                + 0.12 * _clamp(max_support_heading_deg / 14.0, 0.0, 1.0),
                0.0,
                0.72,
            )

        if (
            curve_sign != 0
            and not curve_entry_blocked
            and pose_yaw_change_deg is not None
            and abs(pose_yaw_change_deg) >= 2.5
            and signbit(pose_yaw_change_deg) == curve_sign
        ):
            evidence_strength = _clamp(evidence_strength + 0.06, 0.0, 1.0)

        current_sign = self._adaptive_curve_intent_sign
        current_score = self._adaptive_curve_intent_score
        curve_decay_active = False

        if curve_sign == 0 or evidence_strength <= 1e-6:
            decay_factor = (
                self._adaptive_curve_intent_decay_keep
                if corridor_available
                else self._adaptive_curve_intent_decay_drop
            )
            current_score *= decay_factor
            curve_decay_active = current_sign != 0 and current_score > 0.0
        elif current_sign == 0 or current_sign == curve_sign:
            if current_sign == 0:
                current_sign = curve_sign
                current_score *= 0.35
            rise = 0.20 + (0.34 * evidence_strength)
            current_score = current_score + ((1.0 - current_score) * rise)
        else:
            current_score *= self._adaptive_curve_intent_decay_switch
            curve_decay_active = True
            if (
                evidence_strength >= self._adaptive_curve_intent_switch_score
                or current_score <= self._adaptive_curve_intent_release_score
            ):
                current_sign = curve_sign
                current_score = min(0.55, 0.18 + (0.44 * evidence_strength))
                curve_decay_active = False

        if current_score < self._adaptive_curve_intent_release_score:
            if curve_sign != 0 and evidence_strength >= self._adaptive_curve_intent_entry_score:
                current_sign = curve_sign
                current_score = max(current_score, 0.18 + (0.30 * evidence_strength))
            else:
                current_sign = 0
                current_score = 0.0
                curve_decay_active = False

        self._adaptive_curve_intent_sign = current_sign
        self._adaptive_curve_intent_score = float(_clamp(current_score, 0.0, 1.0))
        self._adaptive_curve_evidence_strength = float(_clamp(evidence_strength, 0.0, 1.0))
        self._adaptive_curve_decay_active = bool(curve_decay_active)
        self._adaptive_premature_curve_veto = bool(premature_curve_veto)
        self._adaptive_committed_turn_sign = (
            current_sign
            if self._adaptive_curve_intent_score >= self._adaptive_curve_intent_entry_score
            else 0
        )
        if self._adaptive_committed_turn_sign == 0:
            self._adaptive_committed_yaw_progress_deg *= 0.5

        return (
            current_sign,
            self._adaptive_curve_intent_score,
            self._adaptive_curve_evidence_strength,
            self._adaptive_curve_decay_active,
        )

    def _select_adaptive_nav_mode(
        self,
        *,
        startup_adapt_active: bool,
        corridor_available: bool,
        corridor_confidence: float,
        curve_confidence: float,
        curve_gate_open: bool,
        curve_confirmed: bool,
        curve_hold_active: bool,
        curve_capture_active: bool,
        curve_capture_sign: int,
        preview_curve_follow_ready: bool,
        curve_release_reason: str,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        curve_decay_active: bool,
        premature_curve_veto: bool,
        nearest_side_m: float,
        preview_heading_deg: float,
        corridor_center_heading_deg: float,
        straight_heading_deg: float,
    ) -> str:
        near_wall = 0.0 < nearest_side_m < self._adaptive_near_wall_distance_m
        entry_active = (
            curve_intent_sign != 0
            and curve_intent_score >= self._adaptive_curve_intent_entry_score
        )
        follow_active = (
            curve_intent_sign != 0
            and curve_intent_score >= self._adaptive_curve_intent_follow_score
        )
        exit_alignment = (
            abs(preview_heading_deg) < 4.5
            and abs(corridor_center_heading_deg) < 4.5
            and abs(straight_heading_deg) < 2.5
        )
        decaying_curve_active = (
            curve_decay_active
            and curve_intent_sign != 0
            and curve_intent_score > self._adaptive_curve_intent_release_score
        )
        capture_active = (
            curve_capture_active
            and curve_capture_sign != 0
            and not curve_confirmed
        )

        if capture_active:
            if curve_confirmed or preview_curve_follow_ready:
                return "curve_follow"
            if follow_active:
                return "curve_entry"
            if curve_release_reason in {"sign_contradiction", "recentered"} and exit_alignment:
                return "curve_exit"
            return "curve_capture"
        if decaying_curve_active and (curve_hold_active or exit_alignment):
            return "curve_exit"
        if curve_hold_active and not exit_alignment:
            if curve_confirmed:
                return "curve_follow"
            if curve_gate_open and entry_active:
                return "curve_entry"
            return "curve_exit"
        if not curve_gate_open and not curve_confirmed:
            if startup_adapt_active:
                return "startup_adapt"
            if near_wall and corridor_confidence < 0.25:
                return "near_wall_recovery"
            return "straight_follow"
        if startup_adapt_active and not entry_active and curve_evidence_strength < 0.40:
            return "startup_adapt"
        if near_wall and not entry_active and curve_evidence_strength < 0.22:
            return "near_wall_recovery"

        if follow_active:
            if (
                curve_decay_active
                and (curve_evidence_strength < 0.22 or not corridor_available)
                and exit_alignment
            ):
                return "curve_exit"
            if near_wall and curve_confidence < 0.30 and corridor_confidence < 0.30:
                return "near_wall_recovery"
            return "curve_follow" if curve_confirmed else "curve_entry"
        if entry_active:
            if curve_hold_active and not exit_alignment:
                if curve_confirmed:
                    return "curve_follow"
                if curve_gate_open:
                    return "curve_entry"
                return "curve_exit"
            if premature_curve_veto or curve_evidence_strength < 0.22 or not curve_gate_open:
                return "near_wall_recovery" if near_wall and corridor_confidence < 0.25 else "straight_follow"
            return "curve_entry"
        if (
            curve_decay_active
            and curve_intent_sign != 0
            and curve_intent_score > self._adaptive_curve_intent_release_score
            and exit_alignment
        ):
            return "curve_exit"
        if corridor_confidence < 0.18 and near_wall:
            return "near_wall_recovery"
        return "straight_follow"

    def _compute_straight_follow_heading_deg(
        self,
        *,
        centerline_heading_deg: float,
        preview_heading_deg: float,
        preview_available: bool,
        corridor_axis_heading_deg: float,
        corridor_available: bool,
        straight_corridor_score: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
    ) -> tuple[float, bool]:
        heading_deg = float(centerline_heading_deg)
        straight_veto_active = False
        straight_score = _clamp(straight_corridor_score, 0.0, 1.0)
        preview_weight = 0.22 * (1.0 - (0.90 * straight_score))
        if preview_available and signbit(preview_heading_deg) == signbit(heading_deg):
            heading_deg = ((1.0 - preview_weight) * heading_deg) + (
                preview_weight * preview_heading_deg
            )
        elif abs(heading_deg) < 1e-6 and preview_available and abs(preview_heading_deg) <= 4.0:
            heading_deg = (0.25 * (1.0 - (0.85 * straight_score))) * preview_heading_deg

        axis_sign = signbit(corridor_axis_heading_deg)
        heading_limit_deg = self._adaptive_straight_heading_limit_deg
        if corridor_available and straight_score >= self._adaptive_straight_parallel_score_threshold:
            axis_guidance_deg = 0.0
            if axis_sign != 0:
                axis_guidance_deg = axis_sign * min(
                    self._adaptive_straight_axis_guidance_limit_deg,
                    0.45 * abs(corridor_axis_heading_deg),
                )
            damping_blend = 0.62 + (0.18 * straight_score)
            heading_deg = ((1.0 - damping_blend) * heading_deg) + (
                damping_blend * axis_guidance_deg
            )
            heading_limit_deg = (
                (1.0 - straight_score) * self._adaptive_straight_heading_limit_deg
                + (straight_score * self._adaptive_straight_centerline_limit_deg)
            )
            straight_veto_active = True

        heading_sign = signbit(heading_deg)
        if (
            corridor_available
            and axis_sign != 0
            and heading_sign != 0
            and axis_sign != heading_sign
            and abs(corridor_axis_heading_deg)
            >= max(self._adaptive_curve_geometry_min_heading_deg, abs(heading_deg) + 2.0)
        ):
            heading_deg = axis_sign * min(3.5, 0.18 * abs(corridor_axis_heading_deg))
            straight_veto_active = True

        if avoidance_active:
            if signbit(heading_deg) == 0:
                heading_deg = 0.25 * avoidance_heading_deg
            elif signbit(heading_deg) == signbit(avoidance_heading_deg):
                heading_deg = (0.85 * heading_deg) + (0.15 * avoidance_heading_deg)

        return (
            _clamp(
                heading_deg,
                -heading_limit_deg,
                heading_limit_deg,
            ),
            straight_veto_active,
        )

    def _compute_corridor_curve_heading_deg(
        self,
        *,
        curve_sign: int,
        preview_heading_deg: float,
        preview_available: bool,
        corridor_axis_heading_deg: float,
        corridor_curvature_heading_deg: float,
        corridor_curvature_confidence: float,
        corridor_center_heading_deg: float,
        front_turn_heading_deg: float,
    ) -> float:
        if curve_sign == 0:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for heading_deg, weight in (
            (
                corridor_curvature_heading_deg,
                0.42 * _clamp(corridor_curvature_confidence, 0.25, 1.0),
            ),
            (corridor_axis_heading_deg, 0.30),
            (corridor_center_heading_deg, 0.18),
            (front_turn_heading_deg, 0.07),
            (preview_heading_deg if preview_available else 0.0, 0.03),
        ):
            if signbit(heading_deg) != curve_sign:
                continue
            weighted_sum += weight * abs(heading_deg)
            total_weight += weight

        if total_weight <= 1e-6:
            return 0.0
        return float(curve_sign * (weighted_sum / total_weight))

    def _compute_adaptive_curve_adjustment_deg(
        self,
        *,
        turn_sign: int,
        corridor_confidence: float,
        curve_confidence: float,
        curve_heading_deg: float,
        corridor_curvature_heading_deg: float,
        corridor_curvature_confidence: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
    ) -> float:
        if turn_sign == 0:
            return 0.0

        if turn_sign > 0:
            turn_front_clearance_m = front_left_clearance_m
            turn_side_min_m = left_min_m
        else:
            turn_front_clearance_m = front_right_clearance_m
            turn_side_min_m = right_min_m
        geometry_gain_deg = 0.0
        if corridor_curvature_confidence > 0.0:
            geometry_gain_deg += (
                5.0
                * _clamp(abs(corridor_curvature_heading_deg) / 18.0, 0.0, 1.0)
                * _clamp(corridor_curvature_confidence, 0.0, 1.0)
            )
        geometry_gain_deg += (
            2.5
            * _clamp(abs(curve_heading_deg) / 12.0, 0.0, 1.0)
            * _clamp(0.35 + (0.40 * corridor_confidence) + (0.25 * curve_confidence), 0.0, 1.0)
        )

        inside_penalty_deg = 0.0
        if turn_side_min_m > 0.0:
            inside_penalty_deg += (
                4.0
                * _clamp(
                    (self._adaptive_near_wall_distance_m - turn_side_min_m)
                    / max(self._adaptive_near_wall_distance_m, 0.05),
                    0.0,
                    1.0,
                )
            )
        if turn_front_clearance_m > 0.0:
            inside_penalty_deg += (
                2.5
                * _clamp(
                    (self._wall_avoid_distance_m - turn_front_clearance_m)
                    / max(self._wall_avoid_distance_m, 0.05),
                    0.0,
                    1.0,
                )
            )

        return float(_clamp(geometry_gain_deg - inside_penalty_deg, -5.0, 6.0))

    def _compute_adaptive_target_heading_deg(
        self,
        *,
        nav_mode: str,
        startup_adapt_active: bool,
        corridor_confidence: float,
        curve_confidence: float,
        curve_confirmed: bool,
        curve_sign: int,
        gate_curve_sign: int,
        committed_turn_sign: int,
        curve_capture_sign: int,
        curve_capture_reason: str,
        curve_hold_sign: int,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        curve_decay_active: bool,
        curve_severity_score: float,
        curve_steering_floor_deg: float,
        straight_heading_deg: float,
        curve_heading_deg: float,
        corridor_curvature_heading_deg: float,
        corridor_curvature_confidence: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        free_space_candidate_heading_deg: float,
        free_space_available: bool,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        effective_front_clearance_m: float,
    ) -> tuple[float, str]:
        heading_deg = 0.0
        active_heading_source = "fallback"
        live_curve_sign = (
            gate_curve_sign
            if gate_curve_sign != 0
            else (curve_sign if curve_sign != 0 else signbit(curve_heading_deg))
        )
        prefer_live_curve_sign = not curve_confirmed and live_curve_sign != 0
        free_space_curve_locked = (
            nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and (
                committed_turn_sign != 0
                or curve_capture_sign != 0
                or curve_hold_sign != 0
            )
        )

        if nav_mode == "startup_adapt":
            heading_deg = _clamp(
                straight_heading_deg,
                -self._adaptive_curve_straight_heading_cap_deg,
                self._adaptive_curve_straight_heading_cap_deg,
            )
            active_heading_source = "startup_adapt"
        elif nav_mode == "straight_follow":
            heading_deg = straight_heading_deg
            if (
                free_space_available
                and corridor_confidence < 0.35
                and (
                    signbit(free_space_candidate_heading_deg) == signbit(heading_deg)
                    or signbit(heading_deg) == 0
                )
            ):
                heading_deg = (0.80 * heading_deg) + (
                    0.20 * _clamp(free_space_candidate_heading_deg, -6.0, 6.0)
                )
                active_heading_source = "free_space_support"
            else:
                active_heading_source = "corridor_straight"
        elif nav_mode == "curve_entry":
            turn_sign = (
                committed_turn_sign
                if committed_turn_sign != 0
                else (
                    curve_capture_sign
                    if curve_capture_sign != 0
                    else (
                    live_curve_sign
                    if prefer_live_curve_sign
                    else (
                        curve_intent_sign
                        if curve_intent_sign != 0
                        else (curve_hold_sign if curve_hold_sign != 0 else curve_sign)
                    )
                    )
                )
            )
            curve_adjustment_deg = self._compute_adaptive_curve_adjustment_deg(
                turn_sign=turn_sign,
                corridor_confidence=corridor_confidence,
                curve_confidence=curve_confidence,
                curve_heading_deg=curve_heading_deg,
                corridor_curvature_heading_deg=corridor_curvature_heading_deg,
                corridor_curvature_confidence=corridor_curvature_confidence,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
            entry_limit_deg = _clamp(
                4.0
                + (3.0 * curve_confidence)
                + (2.0 * curve_evidence_strength)
                + (1.5 * curve_intent_score)
                + (2.0 * (1.0 - self._adaptive_front_fallback_rate)),
                4.0,
                12.0,
            )
            entry_heading_deg = max(
                abs(curve_heading_deg),
                4.0 + (1.8 * curve_evidence_strength),
            )
            entry_heading_deg = _clamp(
                entry_heading_deg + curve_adjustment_deg,
                3.5,
                entry_limit_deg,
            )
            if not curve_confirmed:
                entry_heading_deg = min(
                    entry_heading_deg,
                    self._adaptive_curve_preconfirm_heading_cap_deg,
                )
            heading_deg = turn_sign * entry_heading_deg
            if (
                curve_confirmed
                and
                free_space_available
                and not free_space_curve_locked
                and signbit(free_space_candidate_heading_deg) == turn_sign
                and corridor_confidence < 0.55
            ):
                heading_deg = (0.86 * heading_deg) + (
                    0.14 * _clamp(free_space_candidate_heading_deg, -10.0, 10.0)
                )
                active_heading_source = "free_space_support"
            else:
                active_heading_source = "corridor_curve"
        elif nav_mode == "curve_capture":
            turn_sign = (
                curve_capture_sign
                if curve_capture_sign != 0
                else (
                    live_curve_sign
                    if live_curve_sign != 0
                    else (curve_intent_sign if curve_intent_sign != 0 else curve_sign)
                )
            )
            curve_adjustment_deg = self._compute_adaptive_curve_adjustment_deg(
                turn_sign=turn_sign,
                corridor_confidence=corridor_confidence,
                curve_confidence=curve_confidence,
                curve_heading_deg=curve_heading_deg,
                corridor_curvature_heading_deg=corridor_curvature_heading_deg,
                corridor_curvature_confidence=corridor_curvature_confidence,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
            capture_heading_deg = max(
                abs(curve_heading_deg),
                4.5 + (3.0 * curve_severity_score),
                0.90 * curve_steering_floor_deg,
            )
            capture_heading_deg = _clamp(
                capture_heading_deg + curve_adjustment_deg,
                max(4.5, 0.80 * curve_steering_floor_deg),
                min(self._reference_target_angle_limit_deg, 12.0 + (3.0 * curve_severity_score)),
            )
            if not curve_confirmed:
                capture_heading_deg = min(
                    max(capture_heading_deg, curve_steering_floor_deg),
                    max(curve_steering_floor_deg, self._adaptive_curve_preconfirm_heading_cap_deg + (4.0 * curve_severity_score)),
                )
            if curve_capture_reason in {"preview_dominant_alignment", "preview_dominant_hold"}:
                midpoint_blend = 0.58
                midpoint_heading_deg = _clamp(straight_heading_deg, -2.5, 2.5)
                guided_heading_deg = ((1.0 - midpoint_blend) * (turn_sign * capture_heading_deg)) + (
                    midpoint_blend * midpoint_heading_deg
                )
                guarded_cap_deg = min(
                    max(4.0, 0.70 * curve_steering_floor_deg),
                    6.0,
                )
                capture_heading_deg = _clamp(
                    abs(guided_heading_deg),
                    2.5,
                    guarded_cap_deg,
                )
            heading_deg = turn_sign * capture_heading_deg
            active_heading_source = "corridor_curve_capture"
        elif nav_mode == "curve_follow":
            turn_sign = (
                committed_turn_sign
                if committed_turn_sign != 0
                else (
                    curve_capture_sign
                    if curve_capture_sign != 0
                    else (
                    live_curve_sign
                    if prefer_live_curve_sign
                    else (
                        curve_intent_sign
                        if curve_intent_sign != 0
                        else (curve_hold_sign if curve_hold_sign != 0 else curve_sign)
                    )
                    )
                )
            )
            curve_adjustment_deg = self._compute_adaptive_curve_adjustment_deg(
                turn_sign=turn_sign,
                corridor_confidence=corridor_confidence,
                curve_confidence=curve_confidence,
                curve_heading_deg=curve_heading_deg,
                corridor_curvature_heading_deg=corridor_curvature_heading_deg,
                corridor_curvature_confidence=corridor_curvature_confidence,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
            )
            follow_limit_deg = _clamp(
                6.0
                + (6.0 * curve_evidence_strength)
                + (4.0 * curve_intent_score)
                + (2.0 * curve_confidence),
                6.0,
                18.0,
            )
            follow_heading_deg = max(
                abs(curve_heading_deg),
                4.5 + (2.5 * curve_evidence_strength),
            )
            follow_heading_deg = _clamp(
                follow_heading_deg + curve_adjustment_deg,
                4.0,
                follow_limit_deg,
            )
            heading_deg = turn_sign * follow_heading_deg
            if (
                curve_confirmed
                and
                free_space_available
                and not free_space_curve_locked
                and signbit(free_space_candidate_heading_deg) == turn_sign
                and corridor_confidence < 0.45
            ):
                heading_deg = (0.88 * heading_deg) + (
                    0.12 * _clamp(free_space_candidate_heading_deg, -12.0, 12.0)
                )
                active_heading_source = "free_space_support"
            else:
                active_heading_source = "corridor_curve"
        elif nav_mode == "curve_exit":
            turn_sign = (
                committed_turn_sign
                if committed_turn_sign != 0
                else (
                    curve_capture_sign
                    if curve_capture_sign != 0
                    else (
                    live_curve_sign
                    if prefer_live_curve_sign
                    else (
                        curve_hold_sign
                        if curve_hold_sign != 0
                        else (curve_intent_sign if curve_intent_sign != 0 else curve_sign)
                    )
                    )
                )
            )
            base_curve_heading_deg = curve_heading_deg
            if (
                turn_sign != 0
                and signbit(self._adaptive_curve_heading_memory_deg) == turn_sign
                and abs(self._adaptive_curve_heading_memory_deg) > abs(base_curve_heading_deg)
            ):
                base_curve_heading_deg = self._adaptive_curve_heading_memory_deg
            exit_curve_weight = _clamp(
                0.34 + (0.26 * curve_intent_score) - (0.08 if curve_decay_active else 0.0),
                0.24,
                0.68,
            )
            exit_heading_deg = (exit_curve_weight * base_curve_heading_deg) + (
                (1.0 - exit_curve_weight) * straight_heading_deg
            )
            if signbit(exit_heading_deg) == 0 and turn_sign != 0:
                exit_heading_deg = turn_sign * min(abs(base_curve_heading_deg), 4.5)
            if turn_sign != 0 and signbit(exit_heading_deg) != turn_sign:
                exit_heading_deg = turn_sign * min(abs(exit_heading_deg), 4.0)
            heading_deg = _clamp(exit_heading_deg, -9.0, 9.0)
            active_heading_source = "corridor_exit"
        else:
            if avoidance_active:
                heading_deg = _clamp(avoidance_heading_deg, -4.0, 4.0)
                active_heading_source = "near_wall_limit"
            elif corridor_confidence >= 0.24:
                heading_deg = _clamp(0.70 * straight_heading_deg, -2.5, 2.5)
                active_heading_source = "recenter_min"
            else:
                heading_deg = 0.0
                active_heading_source = "near_wall_limit"

        if (
            nav_mode == "near_wall_recovery"
            and effective_front_clearance_m > 0.0
            and effective_front_clearance_m <= (self._stop_distance_m + 0.05)
            and avoidance_active
            and corridor_confidence < 0.30
        ):
            heading_deg = _clamp(avoidance_heading_deg, -4.0, 4.0)
            active_heading_source = "collision_escape"

        return float(heading_deg), active_heading_source

    def _update_adaptive_curve_heading_memory(
        self,
        *,
        curve_sign: int,
        curve_heading_deg: float,
        curve_intent_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
    ) -> float:
        current_heading_deg = float(self._adaptive_curve_heading_memory_deg)
        current_sign = signbit(current_heading_deg)
        support_sign = curve_intent_sign if curve_intent_sign != 0 else curve_sign
        support_heading_deg = 0.0
        if (
            support_sign != 0
            and signbit(curve_heading_deg) == support_sign
            and (
                curve_evidence_strength >= 0.24
                or curve_intent_score >= self._adaptive_curve_intent_entry_score
            )
        ):
            support_heading_deg = support_sign * max(abs(curve_heading_deg), 4.5)
            if current_sign == support_sign:
                current_heading_deg = (0.68 * current_heading_deg) + (0.32 * support_heading_deg)
            else:
                current_heading_deg = support_heading_deg
        elif current_sign != 0:
            decay = 0.80 if curve_intent_score > self._adaptive_curve_intent_release_score else 0.55
            current_heading_deg *= decay
            if abs(current_heading_deg) < 2.0:
                current_heading_deg = 0.0

        self._adaptive_curve_heading_memory_deg = float(
            _clamp(
                current_heading_deg,
                -self._reference_target_angle_limit_deg,
                self._reference_target_angle_limit_deg,
            )
        )
        return self._adaptive_curve_heading_memory_deg

    def _apply_adaptive_sign_commit_veto(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        nav_mode: str,
        committed_turn_sign: int,
        gate_curve_sign: int,
        curve_capture_active: bool,
        curve_capture_sign: int,
        curve_hold_active: bool,
        curve_hold_sign: int,
        corridor_confidence: float,
        curve_confidence: float,
        curve_sign: int,
        curve_intent_score: float,
        curve_evidence_strength: float,
        free_space_candidate_heading_deg: float,
    ) -> tuple[float, str, str]:
        sign_veto_reason = "none"
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return target_heading_deg, active_heading_source, sign_veto_reason

        protected_turn_sign = committed_turn_sign
        if protected_turn_sign == 0 and curve_capture_active and curve_capture_sign != 0:
            protected_turn_sign = curve_capture_sign

        if (
            protected_turn_sign != 0
            and target_sign != protected_turn_sign
            and nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and active_heading_source != "collision_escape"
        ):
            sign_veto_reason = "capture_sign_veto"
            corrected_heading_deg = protected_turn_sign * _clamp(
                max(abs(target_heading_deg), 2.0),
                2.0,
                6.0,
            )
            return corrected_heading_deg, active_heading_source, sign_veto_reason

        if (
            committed_turn_sign != 0
            and target_sign != committed_turn_sign
            and nav_mode in {"curve_entry", "curve_follow", "curve_exit"}
        ):
            sign_veto_reason = "commit_sign_veto"
            return 0.0, active_heading_source, sign_veto_reason

        free_space_curve_locked = (
            nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and (
                protected_turn_sign != 0
                or (curve_hold_active and curve_hold_sign != 0)
            )
        )
        if free_space_curve_locked:
            return target_heading_deg, active_heading_source, sign_veto_reason

        free_space_sign = signbit(free_space_candidate_heading_deg)
        reference_curve_sign = protected_turn_sign
        if reference_curve_sign == 0 and curve_hold_active and curve_hold_sign != 0:
            reference_curve_sign = curve_hold_sign
        if reference_curve_sign == 0 and gate_curve_sign != 0:
            reference_curve_sign = gate_curve_sign
        if reference_curve_sign == 0 and curve_intent_score >= self._adaptive_curve_intent_release_score:
            reference_curve_sign = curve_sign
        if (
            free_space_sign != 0
            and target_sign == free_space_sign
            and reference_curve_sign != 0
            and target_sign != reference_curve_sign
            and corridor_confidence >= 0.45
            and curve_confidence >= 0.35
        ):
            if (
                curve_hold_active
                and curve_hold_sign == target_sign
                and nav_mode in {"curve_entry", "curve_follow", "curve_exit"}
                and (
                    committed_turn_sign == target_sign
                    or curve_intent_score > self._adaptive_curve_intent_release_score
                    or curve_evidence_strength >= 0.20
                )
            ):
                return target_heading_deg, active_heading_source, sign_veto_reason
            sign_veto_reason = "free_space_opposite_corridor"
            return 0.0, active_heading_source, sign_veto_reason

        return target_heading_deg, active_heading_source, sign_veto_reason

    def _apply_adaptive_near_wall_safety(
        self,
        *,
        target_heading_deg: float,
        active_heading_source: str,
        nav_mode: str,
        committed_turn_sign: int,
        curve_capture_active: bool,
        curve_capture_sign: int,
        curve_steering_floor_deg: float,
        corridor_confidence: float,
        curve_confidence: float,
        straight_corridor_score: float,
        effective_front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        left_min_m: float,
        right_min_m: float,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        pre_curve_bias_veto: bool,
    ) -> tuple[float, str, str, str, bool]:
        near_wall_mode = "none"
        same_sign_trim_active = False
        target_sign = signbit(target_heading_deg)
        nearest_side_m = min(
            (value for value in (left_min_m, right_min_m) if value > 0.0),
            default=0.0,
        )
        imminent_side_risk = (
            nearest_side_m > 0.0
            and nearest_side_m <= max(0.10, self._stop_distance_m * 0.55)
        )

        straight_like = nav_mode in {"startup_adapt", "straight_follow"} and not curve_capture_active
        corridor_straight_supported = (
            straight_like
            and (
                straight_corridor_score >= (0.55 * self._adaptive_straight_parallel_score_threshold)
                or corridor_confidence >= 0.32
            )
        )
        front_buffer_clear = (
            effective_front_clearance_m <= 0.0
            or effective_front_clearance_m > (self._stop_distance_m + 0.12)
        )

        if straight_like and corridor_straight_supported and front_buffer_clear and not imminent_side_risk:
            trim_limit_deg = 0.55 + (0.50 * (1.0 - _clamp(straight_corridor_score, 0.0, 1.0)))
            trim_heading_deg = (
                _clamp(0.18 * avoidance_heading_deg, -trim_limit_deg, trim_limit_deg)
                if avoidance_active
                else 0.0
            )
            trimmed_heading_deg = _clamp(
                target_heading_deg + trim_heading_deg,
                -max(1.1, trim_limit_deg),
                max(1.1, trim_limit_deg),
            )
            trimmed_sign = signbit(trimmed_heading_deg)
            if (
                pre_curve_bias_veto
                or (
                    target_sign == 0
                    and trimmed_sign != 0
                    and abs(trimmed_heading_deg) > 0.35
                )
                or (
                    target_sign != 0
                    and abs(target_heading_deg) < 1.8
                    and trimmed_sign != 0
                    and trimmed_sign != target_sign
                )
            ):
                trimmed_heading_deg = 0.0 if abs(target_heading_deg) < 0.75 else _clamp(
                    target_heading_deg,
                    -0.75,
                    0.75,
                )
                near_wall_mode = "trim"
                return float(trimmed_heading_deg), active_heading_source, "pre_curve_bias_veto", near_wall_mode, same_sign_trim_active

            if abs(trimmed_heading_deg - target_heading_deg) > 0.05:
                near_wall_mode = "trim"
            return float(trimmed_heading_deg), active_heading_source, "none", near_wall_mode, same_sign_trim_active

        if (
            nav_mode == "near_wall_recovery"
            and corridor_confidence < 0.30
            and curve_confidence < 0.30
        ):
            if (
                effective_front_clearance_m > 0.0
                and effective_front_clearance_m <= (self._stop_distance_m + 0.03)
                and avoidance_active
            ):
                return (
                    _clamp(avoidance_heading_deg, -4.0, 4.0),
                    "collision_escape",
                    "collision_escape",
                    "escape",
                    same_sign_trim_active,
                )
            local_heading_deg = (
                _clamp(avoidance_heading_deg, -2.5, 2.5) if avoidance_active else 0.0
            )
            near_wall_mode = "trim" if abs(local_heading_deg) > 0.05 else "none"
            return float(local_heading_deg), "near_wall_recovery", "none", near_wall_mode, same_sign_trim_active

        if target_sign == 0:
            return target_heading_deg, active_heading_source, "none", near_wall_mode, same_sign_trim_active

        protected_turn_sign = committed_turn_sign
        if protected_turn_sign == 0 and curve_capture_active and curve_capture_sign != 0:
            protected_turn_sign = curve_capture_sign

        if target_sign > 0:
            turn_side_front_clearance_m = front_left_clearance_m
            turn_side_min_m = left_min_m
            opposite_side_min_m = right_min_m
        else:
            turn_side_front_clearance_m = front_right_clearance_m
            turn_side_min_m = right_min_m
            opposite_side_min_m = left_min_m

        if (
            turn_side_min_m > 0.0
            and turn_side_min_m < self._adaptive_near_wall_distance_m
            and not (
                protected_turn_sign == target_sign
                and nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
                and (curve_confidence >= 0.30 or corridor_confidence >= 0.55)
            )
        ):
            near_wall_mode = "trim"
            if avoidance_active and signbit(avoidance_heading_deg) == -target_sign:
                safe_heading_deg = _clamp(avoidance_heading_deg, -4.0, 4.0)
                if signbit(safe_heading_deg) != 0:
                    return safe_heading_deg, "near_wall_limit", "near_wall_limit", near_wall_mode, same_sign_trim_active
            limited_heading_deg = target_sign * min(abs(target_heading_deg), 3.0)
            if (
                opposite_side_min_m > 0.0
                and turn_side_min_m < (opposite_side_min_m - 0.04)
                and abs(limited_heading_deg) < self._startup_consensus_min_heading_deg
            ):
                return 0.0, "near_wall_limit", "near_wall_limit", near_wall_mode, same_sign_trim_active
            return float(limited_heading_deg), "near_wall_limit", "near_wall_limit", near_wall_mode, same_sign_trim_active

        if (
            protected_turn_sign != 0
            and target_sign == protected_turn_sign
            and nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and turn_side_min_m > 0.0
            and turn_side_min_m < self._adaptive_near_wall_distance_m
        ):
            same_sign_trim_active = True
            trim_scale = _clamp(
                turn_side_min_m / max(self._adaptive_near_wall_distance_m, 0.05),
                0.45,
                1.0,
            )
            keep_heading_deg = max(1.5, curve_steering_floor_deg * (0.65 * trim_scale))
            if imminent_side_risk:
                keep_heading_deg = max(1.5, keep_heading_deg * 0.75)
            limited_heading_deg = protected_turn_sign * _clamp(
                abs(target_heading_deg),
                keep_heading_deg,
                max(abs(target_heading_deg), keep_heading_deg),
            )
            near_wall_mode = "trim"
            return (
                float(limited_heading_deg),
                "near_wall_limit",
                "near_wall_limit_same_sign",
                near_wall_mode,
                same_sign_trim_active,
            )

        if (
            turn_side_front_clearance_m > 0.0
            and turn_side_front_clearance_m <= (self._stop_distance_m + 0.03)
            and avoidance_active
            and signbit(avoidance_heading_deg) == -target_sign
            and corridor_confidence < 0.40
        ):
            return (
                _clamp(avoidance_heading_deg, -4.0, 4.0),
                "collision_escape",
                "collision_escape",
                "escape",
                same_sign_trim_active,
            )

        return target_heading_deg, active_heading_source, "none", near_wall_mode, same_sign_trim_active

    def _compute_adaptive_simple_command(
        self,
        *,
        shrunk_scan: np.ndarray,
        front_clearance_m: float,
        effective_front_clearance_m: float,
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
        left_wall_heading_deg: float,
        right_wall_heading_deg: float,
        left_wall_points: int,
        right_wall_points: int,
        avoidance_heading_deg: float,
        avoidance_active: bool,
        pose_yaw_change_deg: float | None,
        pose_distance_from_phase_start_m: float | None,
        pose_lateral_drift_m: float | None,
        yaw_delta_deg: float,
        distance_delta_m: float,
    ) -> tuple[
        float,
        float,
        str,
        str,
        float,
        float,
        int,
        float,
        int,
        float,
        str,
        str,
        bool,
        bool,
        float,
        int,
        float,
        bool,
        bool,
        bool,
        float,
        bool,
        str,
        float,
        float,
        float,
    ]:
        _ = gap_heading_deg
        _ = gap_available
        _ = pose_lateral_drift_m

        free_space_candidate_heading_deg, free_space_available = (
            self._compute_reference_target_heading_deg(shrunk_scan=shrunk_scan)
        )
        (
            corridor_curvature_sign,
            corridor_curvature_confidence,
            corridor_curvature_heading_deg,
            _corridor_curvature_delta_deg,
            corridor_curvature_available,
        ) = self._compute_corridor_curvature_features(
            left_wall_points_xy=self._window_points(
                shrunk_scan,
                self._corridor_wall_start_deg,
                min(self._side_window_deg, self._corridor_wall_end_deg),
            ),
            right_wall_points_xy=self._window_points(
                shrunk_scan,
                -min(self._side_window_deg, self._corridor_wall_end_deg),
                -self._corridor_wall_start_deg,
            ),
        )

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
            if preview_available:
                centerline_sign = signbit(centerline_heading_deg)
                preview_sign = signbit(preview_heading_deg)
                axis_sign = signbit(corridor_axis_heading_deg)
                preview_blend = 0.25 + (0.20 * _clamp((self._adaptive_preview_release_gain - 0.55) / 0.30, 0.0, 1.0))
                if (
                    axis_sign != 0
                    and centerline_sign != 0
                    and axis_sign != centerline_sign
                    and abs(corridor_axis_heading_deg) >= max(4.0, abs(centerline_heading_deg) + 2.0)
                ):
                    preview_blend = 0.0
                if centerline_sign == 0:
                    centerline_heading_deg = (1.0 - preview_blend) * centerline_heading_deg + (
                        preview_blend * preview_heading_deg
                    )
                elif preview_sign == 0 or preview_sign == centerline_sign:
                    centerline_heading_deg = (1.0 - preview_blend) * centerline_heading_deg + (
                        preview_blend * preview_heading_deg
                    )

        startup_adapt_active = self._update_startup_adaptation_state(
            front_clearance_fallback_used=front_clearance_fallback_used,
            corridor_balance_ratio=corridor_balance_ratio,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            pose_distance_from_phase_start_m=pose_distance_from_phase_start_m,
        )
        corridor_confidence = self._compute_adaptive_corridor_confidence(
            corridor_available=corridor_available,
            corridor_balance_ratio=corridor_balance_ratio,
            left_wall_points=left_wall_points,
            right_wall_points=right_wall_points,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_clearance_fallback_used=front_clearance_fallback_used,
        )
        straight_corridor_score = self._compute_straight_corridor_score(
            left_wall_heading_deg=left_wall_heading_deg,
            right_wall_heading_deg=right_wall_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
        )
        if side_signal_available and straight_corridor_score > 0.0:
            axis_sign = signbit(corridor_axis_heading_deg)
            axis_guidance_deg = 0.0
            if axis_sign != 0:
                axis_guidance_deg = axis_sign * min(
                    self._adaptive_straight_axis_guidance_limit_deg,
                    0.45 * abs(corridor_axis_heading_deg),
                )
            damping_blend = 0.60 + (0.20 * straight_corridor_score)
            centerline_heading_deg = ((1.0 - damping_blend) * centerline_heading_deg) + (
                damping_blend * axis_guidance_deg
            )
            centerline_limit_deg = (
                (1.0 - straight_corridor_score) * self._simple_centerline_local_limit_deg
                + (straight_corridor_score * self._adaptive_straight_centerline_limit_deg)
            )
            centerline_heading_deg = _clamp(
                centerline_heading_deg,
                -centerline_limit_deg,
                centerline_limit_deg,
            )
        curve_sign, vote_consistency, max_support_heading_deg = self._compute_weighted_curve_vote(
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            left_wall_heading_deg=left_wall_heading_deg,
            right_wall_heading_deg=right_wall_heading_deg,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
            corridor_curvature_heading_deg=corridor_curvature_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            free_space_available=free_space_available,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
        )
        curve_confidence = self._compute_adaptive_curve_confidence(
            curve_sign=curve_sign,
            vote_consistency=vote_consistency,
            max_support_heading_deg=max_support_heading_deg,
            corridor_confidence=corridor_confidence,
            yaw_delta_deg=yaw_delta_deg,
            pose_yaw_change_deg=pose_yaw_change_deg,
        )
        (
            curve_gate_open,
            curve_gate_reason,
            geometry_agreement_score,
        ) = self._update_adaptive_curve_gate(
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            corridor_balance_ratio=corridor_balance_ratio,
            left_wall_points=left_wall_points,
            right_wall_points=right_wall_points,
            left_clearance_m=left_clearance_m,
            right_clearance_m=right_clearance_m,
            curve_sign=curve_sign,
            vote_consistency=vote_consistency,
            max_support_heading_deg=max_support_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
            corridor_curvature_heading_deg=corridor_curvature_heading_deg,
        )
        inferred_gate_curve_sign = self._infer_curve_gate_sign(
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
            corridor_curvature_heading_deg=corridor_curvature_heading_deg,
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            free_space_available=free_space_available,
        )
        preview_dominant_curve_sign = self._infer_preview_dominant_curve_sign(
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_confidence=corridor_confidence,
            effective_front_clearance_m=effective_front_clearance_m,
            gate_curve_sign=inferred_gate_curve_sign,
        )
        gate_curve_sign = curve_sign
        if gate_curve_sign == 0:
            gate_curve_sign = (
                preview_dominant_curve_sign
                if preview_dominant_curve_sign != 0
                else inferred_gate_curve_sign
            )
        elif (
            inferred_gate_curve_sign != 0
            and inferred_gate_curve_sign != gate_curve_sign
            and not curve_gate_open
            and curve_confidence < self._adaptive_curve_commit_conf_threshold
        ):
            gate_curve_sign = inferred_gate_curve_sign
        if (
            preview_dominant_curve_sign != 0
            and preview_dominant_curve_sign != gate_curve_sign
            and not curve_gate_open
            and self._adaptive_committed_turn_sign == 0
        ):
            gate_curve_sign = preview_dominant_curve_sign

        straight_heading_deg, straight_veto_active = self._compute_straight_follow_heading_deg(
            centerline_heading_deg=centerline_heading_deg,
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_available=corridor_available,
            straight_corridor_score=straight_corridor_score,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
        )
        straight_corridor_veto = (
            straight_corridor_score >= self._adaptive_straight_parallel_score_threshold
            or (
                corridor_available
                and corridor_confidence >= 0.42
                and abs(corridor_axis_heading_deg) < self._adaptive_curve_geometry_min_heading_deg
                and (
                    corridor_curvature_sign == 0
                    or corridor_curvature_confidence < 0.35
                )
            )
        )
        premature_curve_veto = self._compute_premature_curve_veto(
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            front_turn_heading_deg=front_turn_heading_deg,
            free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
        )
        pre_curve_bias_veto = self._compute_pre_curve_bias_veto(
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            front_turn_heading_deg=front_turn_heading_deg,
            avoidance_heading_deg=avoidance_heading_deg,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            corridor_curvature_sign=corridor_curvature_sign,
            corridor_curvature_confidence=corridor_curvature_confidence,
            straight_corridor_score=straight_corridor_score,
            front_clearance_fallback_used=front_clearance_fallback_used,
        )
        premature_curve_veto = premature_curve_veto or (
            curve_sign != 0 and straight_corridor_veto
        )
        premature_curve_veto = premature_curve_veto or pre_curve_bias_veto
        committed_turn_sign = self._adaptive_committed_turn_sign
        curve_entry_blocked = bool(
            premature_curve_veto
            or (not curve_gate_open and committed_turn_sign == 0)
        )
        (
            curve_intent_sign,
            curve_intent_score,
            curve_evidence_strength,
            curve_decay_active,
        ) = self._update_adaptive_curve_intent(
            curve_sign=gate_curve_sign if gate_curve_sign != 0 else curve_sign,
            curve_confidence=curve_confidence,
            vote_consistency=vote_consistency,
            max_support_heading_deg=max_support_heading_deg,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            premature_curve_veto=premature_curve_veto,
            curve_entry_blocked=curve_entry_blocked,
            pose_yaw_change_deg=pose_yaw_change_deg,
        )
        (
            curve_capture_sign,
            curve_capture_active,
            curve_capture_reason,
            curve_release_reason,
        ) = self._update_adaptive_curve_capture_state(
            gate_curve_sign=gate_curve_sign,
            curve_gate_open=curve_gate_open,
            geometry_agreement_score=geometry_agreement_score,
            curve_confidence=curve_confidence,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            committed_turn_sign=committed_turn_sign,
            corridor_confidence=corridor_confidence,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            preview_heading_deg=preview_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            straight_heading_deg=straight_heading_deg,
            effective_front_clearance_m=effective_front_clearance_m,
        )
        if curve_capture_active and curve_capture_sign != 0 and committed_turn_sign == 0:
            gate_curve_sign = curve_capture_sign
        gate_curve_bootstrap_active = (
            curve_capture_active
            and committed_turn_sign == 0
            and curve_intent_sign == 0
            and curve_capture_sign != 0
        )
        if gate_curve_bootstrap_active:
            curve_intent_sign = curve_capture_sign
            curve_intent_score = max(curve_intent_score, self._adaptive_curve_intent_entry_score)
            curve_evidence_strength = max(
                curve_evidence_strength,
                self._adaptive_curve_hold_min_evidence,
            )
            curve_decay_active = False
            self._adaptive_curve_intent_sign = curve_intent_sign
            self._adaptive_curve_intent_score = float(curve_intent_score)
            self._adaptive_curve_evidence_strength = float(curve_evidence_strength)
            self._adaptive_curve_decay_active = False
        (
            committed_turn_sign,
            curve_confirm_distance_m,
            curve_confirm_yaw_deg,
            curve_confirmed,
        ) = self._update_adaptive_curve_confirmation(
            curve_gate_open=curve_gate_open,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
            curve_confidence=curve_confidence,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            yaw_delta_deg=yaw_delta_deg,
            distance_delta_m=distance_delta_m,
        )
        if committed_turn_sign == 0 and curve_entry_blocked:
            self._adaptive_curve_confirm_distance_m = 0.0
            self._adaptive_curve_confirm_yaw_deg = 0.0
        if committed_turn_sign != 0 and curve_capture_active:
            curve_capture_sign, curve_capture_active, curve_capture_reason, curve_release_reason = (
                self._update_adaptive_curve_capture_state(
                    gate_curve_sign=gate_curve_sign,
                    curve_gate_open=curve_gate_open,
                    geometry_agreement_score=geometry_agreement_score,
                    curve_confidence=curve_confidence,
                    curve_intent_sign=curve_intent_sign,
                    curve_intent_score=curve_intent_score,
                    committed_turn_sign=committed_turn_sign,
                    corridor_confidence=corridor_confidence,
                    corridor_axis_heading_deg=corridor_axis_heading_deg,
                    front_turn_heading_deg=front_turn_heading_deg,
                    preview_heading_deg=preview_heading_deg,
                    corridor_center_heading_deg=corridor_center_heading_deg,
                    straight_heading_deg=straight_heading_deg,
                    effective_front_clearance_m=effective_front_clearance_m,
                )
            )

        resolved_curve_sign = committed_turn_sign
        if resolved_curve_sign == 0:
            if curve_capture_sign != 0:
                resolved_curve_sign = curve_capture_sign
            elif gate_curve_sign != 0 and curve_intent_sign != gate_curve_sign:
                resolved_curve_sign = gate_curve_sign
            elif curve_sign != 0 and curve_intent_sign != curve_sign:
                resolved_curve_sign = curve_sign
            elif curve_intent_sign != 0:
                resolved_curve_sign = curve_intent_sign
            else:
                resolved_curve_sign = gate_curve_sign if gate_curve_sign != 0 else curve_sign
        curve_heading_deg = self._compute_corridor_curve_heading_deg(
            curve_sign=resolved_curve_sign,
            preview_heading_deg=preview_heading_deg,
            preview_available=preview_available,
            corridor_axis_heading_deg=corridor_axis_heading_deg,
            corridor_curvature_heading_deg=corridor_curvature_heading_deg,
            corridor_curvature_confidence=corridor_curvature_confidence,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
        )
        self._update_adaptive_curve_heading_memory(
            curve_sign=gate_curve_sign if gate_curve_sign != 0 else curve_sign,
            curve_heading_deg=curve_heading_deg,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
        )
        curve_hold_sign, curve_hold_active = self._update_adaptive_curve_hold_state(
            curve_gate_open=curve_gate_open,
            corridor_confidence=corridor_confidence,
            curve_heading_deg=curve_heading_deg,
            committed_turn_sign=committed_turn_sign,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
            preview_heading_deg=preview_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            straight_heading_deg=straight_heading_deg,
        )
        curve_severity_score = self._compute_curve_severity_score(
            turn_sign=resolved_curve_sign,
            corridor_center_heading_deg=corridor_center_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_curvature_confidence=corridor_curvature_confidence,
            geometry_agreement_score=geometry_agreement_score,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
        )
        curve_steering_floor_deg = 0.0
        curve_speed_cap_pct = 0.0
        nearest_side_m = min(
            value for value in (left_min_m, right_min_m) if value > 0.0
        ) if any(value > 0.0 for value in (left_min_m, right_min_m)) else 0.0
        preview_curve_follow_ready = self._preview_curve_follow_ready(
            curve_capture_active=curve_capture_active,
            curve_capture_sign=curve_capture_sign,
            curve_capture_reason=curve_capture_reason,
            curve_gate_open=curve_gate_open,
            curve_gate_reason=curve_gate_reason,
            curve_confidence=curve_confidence,
            geometry_agreement_score=geometry_agreement_score,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
            corridor_confidence=corridor_confidence,
            preview_heading_deg=preview_heading_deg,
            front_turn_heading_deg=front_turn_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            effective_front_clearance_m=effective_front_clearance_m,
        )
        nav_mode = self._select_adaptive_nav_mode(
            startup_adapt_active=startup_adapt_active,
            corridor_available=corridor_available,
            corridor_confidence=corridor_confidence,
            curve_confidence=curve_confidence,
            curve_gate_open=curve_gate_open,
            curve_confirmed=curve_confirmed,
            curve_hold_active=curve_hold_active,
            curve_capture_active=curve_capture_active,
            curve_capture_sign=curve_capture_sign,
            preview_curve_follow_ready=preview_curve_follow_ready,
            curve_release_reason=curve_release_reason,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
            curve_decay_active=curve_decay_active,
            premature_curve_veto=premature_curve_veto,
            nearest_side_m=nearest_side_m,
            preview_heading_deg=preview_heading_deg,
            corridor_center_heading_deg=corridor_center_heading_deg,
            straight_heading_deg=straight_heading_deg,
        )
        if (
            (straight_corridor_veto or premature_curve_veto or (not curve_gate_open and not curve_confirmed))
            and nav_mode == "curve_entry"
        ):
            nav_mode = "straight_follow"
            straight_veto_active = True

        curve_dynamics_mode = (
            "curve_capture"
            if nav_mode in {"curve_capture", "curve_entry"}
            else ("curve_follow" if nav_mode in {"curve_follow", "curve_exit"} else nav_mode)
        )
        curve_steering_floor_deg = self._compute_curve_steering_floor_deg(
            nav_mode=curve_dynamics_mode,
            turn_sign=resolved_curve_sign,
            curve_severity_score=curve_severity_score,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
        )
        inside_front_clearance_m = (
            front_left_clearance_m if resolved_curve_sign > 0 else front_right_clearance_m
        )
        curve_speed_cap_pct = self._compute_curve_speed_cap_pct(
            nav_mode=curve_dynamics_mode,
            curve_severity_score=curve_severity_score,
            inside_front_clearance_m=inside_front_clearance_m,
        )
        preview_midpoint_guard_active = (
            nav_mode == "curve_capture"
            and curve_capture_reason in {"preview_dominant_alignment", "preview_dominant_hold"}
            and not preview_curve_follow_ready
        )
        if preview_midpoint_guard_active:
            curve_steering_floor_deg = min(
                curve_steering_floor_deg,
                max(3.5, 0.55 * curve_steering_floor_deg),
            )
            if curve_speed_cap_pct > 0.0:
                curve_speed_cap_pct = min(curve_speed_cap_pct, 22.0)
            else:
                curve_speed_cap_pct = 22.0

        target_heading_deg, active_heading_source = self._compute_adaptive_target_heading_deg(
            nav_mode=nav_mode,
            startup_adapt_active=startup_adapt_active,
            corridor_confidence=corridor_confidence,
            curve_confidence=curve_confidence,
            curve_confirmed=curve_confirmed,
            curve_sign=curve_sign,
            gate_curve_sign=gate_curve_sign,
            committed_turn_sign=committed_turn_sign,
            curve_capture_sign=curve_capture_sign,
            curve_capture_reason=curve_capture_reason,
            curve_hold_sign=curve_hold_sign,
            curve_intent_sign=curve_intent_sign,
            curve_intent_score=curve_intent_score,
            curve_evidence_strength=curve_evidence_strength,
            curve_decay_active=curve_decay_active,
            curve_severity_score=curve_severity_score,
            curve_steering_floor_deg=curve_steering_floor_deg,
            straight_heading_deg=straight_heading_deg,
            curve_heading_deg=curve_heading_deg,
            corridor_curvature_heading_deg=corridor_curvature_heading_deg,
            corridor_curvature_confidence=corridor_curvature_confidence,
            front_left_clearance_m=front_left_clearance_m,
            front_right_clearance_m=front_right_clearance_m,
            left_min_m=left_min_m,
            right_min_m=right_min_m,
            free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            free_space_available=free_space_available,
            avoidance_heading_deg=avoidance_heading_deg,
            avoidance_active=avoidance_active,
            effective_front_clearance_m=effective_front_clearance_m,
        )
        if (
            (straight_corridor_veto or premature_curve_veto)
            and committed_turn_sign == 0
            and not curve_capture_active
            and nav_mode not in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
        ):
            target_heading_deg = straight_heading_deg
            active_heading_source = "corridor_straight_veto"
            straight_veto_active = True

        target_heading_deg, active_heading_source, sign_veto_reason = (
            self._apply_adaptive_sign_commit_veto(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                nav_mode=nav_mode,
                committed_turn_sign=committed_turn_sign,
                gate_curve_sign=gate_curve_sign,
                curve_capture_active=curve_capture_active,
                curve_capture_sign=curve_capture_sign,
                curve_hold_active=curve_hold_active,
                curve_hold_sign=curve_hold_sign,
                corridor_confidence=corridor_confidence,
                curve_confidence=curve_confidence,
                curve_sign=curve_sign,
                curve_intent_score=curve_intent_score,
                curve_evidence_strength=curve_evidence_strength,
                free_space_candidate_heading_deg=free_space_candidate_heading_deg,
            )
        )
        if premature_curve_veto and sign_veto_reason == "none":
            sign_veto_reason = "premature_curve_veto"
        elif straight_corridor_veto and sign_veto_reason == "none":
            sign_veto_reason = "straight_corridor_veto"
        (
            target_heading_deg,
            active_heading_source,
            wall_safety_reason,
            near_wall_mode,
            same_sign_trim_active,
        ) = (
            self._apply_adaptive_near_wall_safety(
                target_heading_deg=target_heading_deg,
                active_heading_source=active_heading_source,
                nav_mode=nav_mode,
                committed_turn_sign=committed_turn_sign,
                curve_capture_active=curve_capture_active,
                curve_capture_sign=curve_capture_sign,
                curve_steering_floor_deg=curve_steering_floor_deg,
                corridor_confidence=corridor_confidence,
                curve_confidence=curve_confidence,
                straight_corridor_score=straight_corridor_score,
                effective_front_clearance_m=effective_front_clearance_m,
                front_left_clearance_m=front_left_clearance_m,
                front_right_clearance_m=front_right_clearance_m,
                left_min_m=left_min_m,
                right_min_m=right_min_m,
                avoidance_heading_deg=avoidance_heading_deg,
                avoidance_active=avoidance_active,
                pre_curve_bias_veto=pre_curve_bias_veto,
            )
        )
        if wall_safety_reason != "none":
            sign_veto_reason = wall_safety_reason
        if pre_curve_bias_veto and sign_veto_reason == "none":
            sign_veto_reason = "pre_curve_bias_veto"

        if (
            wall_safety_reason == "near_wall_limit"
            and committed_turn_sign == 0
            and nav_mode in {"startup_adapt", "straight_follow", "curve_capture", "curve_entry", "curve_follow"}
        ):
            live_curve_sign = (
                resolved_curve_sign
                if resolved_curve_sign != 0
                else (curve_sign if curve_sign != 0 else signbit(target_heading_deg))
            )
            gate_supported_nearwall_curve = (
                curve_gate_open
                and live_curve_sign != 0
                and geometry_agreement_score >= self._adaptive_curve_gate_min_agreement_score
                and curve_confidence >= max(0.30, self._adaptive_curve_entry_conf_threshold * 0.60)
                and max(curve_evidence_strength, curve_intent_score) >= 0.18
            )
            if gate_supported_nearwall_curve:
                nav_mode = "curve_capture"
                target_heading_deg = live_curve_sign * _clamp(
                    max(abs(target_heading_deg), 2.5),
                    2.5,
                    4.0,
                )
                active_heading_source = "corridor_curve_nearwall_hold"
                sign_veto_reason = "near_wall_limit_curve_hold"
                curve_capture_active = True
                curve_capture_sign = live_curve_sign
                curve_capture_reason = "near_wall_gate_hold"
                curve_intent_sign = live_curve_sign
                curve_intent_score = max(
                    curve_intent_score,
                    self._adaptive_curve_intent_entry_score,
                )
                curve_evidence_strength = max(
                    curve_evidence_strength,
                    self._adaptive_curve_hold_min_evidence,
                )
                curve_decay_active = False
                self._adaptive_curve_intent_sign = curve_intent_sign
                self._adaptive_curve_intent_score = float(curve_intent_score)
                self._adaptive_curve_evidence_strength = float(curve_evidence_strength)
                self._adaptive_curve_decay_active = False
            else:
                nav_mode = "startup_adapt" if startup_adapt_active else "straight_follow"
                straight_cap_deg = min(1.5, self._adaptive_curve_straight_heading_cap_deg)
                target_heading_deg = _clamp(straight_heading_deg, -straight_cap_deg, straight_cap_deg)
                active_heading_source = "corridor_straight_nearwall_veto"
                sign_veto_reason = "near_wall_limit_curve_veto"
                straight_veto_active = True
                self._clear_adaptive_curve_tracking(clear_heading_memory=True)
                curve_intent_sign = self._adaptive_curve_intent_sign
                curve_intent_score = self._adaptive_curve_intent_score
                curve_evidence_strength = self._adaptive_curve_evidence_strength
                curve_decay_active = self._adaptive_curve_decay_active
                committed_turn_sign = self._adaptive_committed_turn_sign
                curve_confirm_distance_m = self._adaptive_curve_confirm_distance_m
                curve_confirm_yaw_deg = self._adaptive_curve_confirm_yaw_deg

        if active_heading_source == "collision_escape" and signbit(target_heading_deg) != 0:
            nav_mode = "near_wall_recovery"
            if curve_capture_active:
                self._adaptive_curve_capture_sign = 0
                self._adaptive_curve_capture_cycles_remaining = 0
                self._adaptive_curve_capture_release_streak = 0
                self._adaptive_curve_capture_reason = "inactive"
                self._adaptive_curve_release_reason = "collision_escape"
                curve_capture_active = False
                curve_capture_sign = 0
                curve_capture_reason = "inactive"
                curve_release_reason = "collision_escape"

        if active_heading_source == "fallback" and side_signal_available:
            target_heading_deg = straight_heading_deg
            active_heading_source = "corridor_straight"

        if (
            not curve_gate_open
            and committed_turn_sign == 0
            and nav_mode in {"startup_adapt", "straight_follow"}
        ):
            target_heading_deg = _clamp(
                target_heading_deg,
                -self._adaptive_curve_straight_heading_cap_deg,
                self._adaptive_curve_straight_heading_cap_deg,
            )
            if abs(target_heading_deg) > 0.0 and active_heading_source == "free_space_support":
                active_heading_source = "corridor_straight_gate"

        if (
            nav_mode in {"startup_adapt", "straight_follow"}
            and not corridor_curvature_available
            and corridor_available
            and abs(corridor_axis_heading_deg) < self._adaptive_curve_geometry_min_heading_deg
        ):
            straight_veto_active = True

        if (
            nav_mode in {"startup_adapt", "straight_follow", "near_wall_recovery"}
            and committed_turn_sign == 0
            and not curve_capture_active
            and (
                not curve_gate_open
                or curve_evidence_strength < 0.16
                or curve_intent_score <= self._adaptive_curve_intent_release_score
            )
        ):
            self._clear_adaptive_curve_tracking(clear_heading_memory=True)
            curve_intent_sign = self._adaptive_curve_intent_sign
            curve_intent_score = self._adaptive_curve_intent_score
            curve_evidence_strength = self._adaptive_curve_evidence_strength
            curve_decay_active = self._adaptive_curve_decay_active
            committed_turn_sign = self._adaptive_committed_turn_sign
            curve_confirm_distance_m = self._adaptive_curve_confirm_distance_m
            curve_confirm_yaw_deg = self._adaptive_curve_confirm_yaw_deg

        self._adaptive_last_nav_mode = nav_mode
        return (
            float(_clamp(target_heading_deg, -self._reference_target_angle_limit_deg, self._reference_target_angle_limit_deg)),
            0.0,
            active_heading_source,
            nav_mode,
            corridor_confidence,
            curve_confidence,
            corridor_curvature_sign,
            corridor_curvature_confidence,
            committed_turn_sign,
            gate_curve_sign,
            curve_capture_active,
            curve_capture_reason,
            curve_severity_score,
            curve_steering_floor_deg,
            curve_speed_cap_pct,
            curve_release_reason,
            same_sign_trim_active,
            free_space_candidate_heading_deg if free_space_available else 0.0,
            sign_veto_reason,
            near_wall_mode,
            straight_veto_active,
            startup_adapt_active,
            curve_intent_score,
            curve_intent_sign,
            curve_evidence_strength,
            curve_decay_active,
            premature_curve_veto,
            pre_curve_bias_veto,
            straight_corridor_score,
            curve_gate_open,
            curve_gate_reason,
            geometry_agreement_score,
            curve_confirm_distance_m,
            curve_confirm_yaw_deg,
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
        axis_sign = signbit(corridor_axis_heading_deg)
        axis_mag_deg = abs(corridor_axis_heading_deg)
        lateral_limit_deg = self._simple_centerline_local_limit_deg
        if corridor_available and axis_mag_deg < self._adaptive_curve_geometry_min_heading_deg:
            lateral_limit_deg = min(lateral_limit_deg, 1.5)

        heading_deg = lateral_offset_ratio * lateral_limit_deg
        target_sign = signbit(heading_deg)

        if not corridor_available:
            return float(
                max(
                    -lateral_limit_deg,
                    min(lateral_limit_deg, heading_deg),
                )
            )

        balance_excess = corridor_balance_ratio - self._corridor_balance_ratio_threshold
        axis_blend = max(
            0.0,
            min(1.0, balance_excess / self._simple_centerline_balance_blend_range),
        )
        if axis_sign != 0 and (target_sign == 0 or axis_sign == target_sign):
            axis_heading_deg = axis_sign * min(
                abs(corridor_axis_heading_deg),
                self._simple_centerline_local_limit_deg,
            )
            heading_deg += (
                self._simple_centerline_axis_blend_gain * axis_blend * axis_heading_deg
            )
        elif axis_sign != 0 and target_sign != 0 and axis_sign != target_sign:
            heading_deg = axis_sign * min(axis_mag_deg * 0.12, 1.0)

        return float(
            max(
                -lateral_limit_deg,
                min(lateral_limit_deg, heading_deg),
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
        nav_mode: str,
        target_heading_deg: float,
        front_clearance_m: float,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        wall_follow_heading_deg: float,
        committed_turn_sign: int,
        curve_confirm_distance_m: float,
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
        steering_gain = float(min(1.0, self._steering_gain * gain_boost))

        if nav_mode in {"curve_capture", "curve_entry", "curve_exit"}:
            gain_floor = self._adaptive_curve_entry_steering_gain_floor
            if nav_mode == "curve_capture":
                gain_floor = max(gain_floor, self._adaptive_curve_entry_steering_gain_floor + 0.06)
            if curve_confirm_distance_m >= self._adaptive_curve_motion_release_distance_m:
                gain_floor = max(
                    gain_floor,
                    self._adaptive_curve_entry_steering_gain_floor + 0.04,
                )
            steering_gain = max(steering_gain, gain_floor)
        elif nav_mode == "curve_follow" or committed_turn_sign != 0:
            gain_floor = self._adaptive_curve_follow_steering_gain_floor
            if curve_confirm_distance_m >= self._adaptive_curve_confirm_distance_threshold_m:
                gain_floor = min(0.82, gain_floor + 0.06)
            steering_gain = max(steering_gain, gain_floor)

        return float(min(1.0, steering_gain))

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
        *,
        nav_mode: str,
        pose_distance_from_phase_start_m: float | None,
        committed_turn_sign: int,
        curve_confirm_distance_m: float,
    ) -> float:
        if front_clearance_m <= 0.0 or front_clearance_m <= self._stop_distance_m:
            return 0.0

        pose_progress_m = pose_distance_from_phase_start_m or 0.0
        launch_floor_active = pose_progress_m < self._adaptive_curve_motion_release_distance_m
        max_speed_pct = self._max_speed_pct
        min_speed_pct = self._min_speed_pct
        if launch_floor_active:
            min_speed_pct = max(min_speed_pct, self._adaptive_launch_speed_floor_pct)
            max_speed_pct = max(max_speed_pct, min_speed_pct)

        angle_magnitude_deg = max(abs(target_heading_deg), abs(steering_deg))
        speed_pct = max_speed_pct * math.exp(-0.03 * angle_magnitude_deg)
        if (
            nav_mode in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"}
            and pose_progress_m >= self._adaptive_curve_motion_release_distance_m
            and (
                committed_turn_sign != 0
                or curve_confirm_distance_m >= self._adaptive_curve_motion_release_distance_m
            )
        ):
            min_speed_pct = min(min_speed_pct, self._adaptive_curve_speed_floor_pct)
        speed_pct = max(min_speed_pct, min(max_speed_pct, speed_pct))

        if front_clearance_m >= self._slow_distance_m:
            distance_factor = 1.0
        else:
            distance_factor = (
                (front_clearance_m - self._stop_distance_m)
                / (self._slow_distance_m - self._stop_distance_m)
            )

        speed_pct *= distance_factor
        if speed_pct > 0.0:
            speed_pct = max(min_speed_pct, speed_pct)
        return float(max(0.0, min(max_speed_pct, speed_pct)))

    def _compute_motion_front_clearance_m(
        self,
        *,
        effective_front_clearance_m: float,
        front_left_clearance_m: float,
        front_right_clearance_m: float,
        target_heading_deg: float,
        active_heading_source: str,
        nav_mode: str,
        front_turn_heading_deg: float,
        corridor_center_heading_deg: float,
        wall_follow_active: bool,
        corridor_confidence: float,
    ) -> float:
        target_sign = signbit(target_heading_deg)
        if target_sign == 0:
            return effective_front_clearance_m

        if nav_mode not in {"curve_capture", "curve_entry", "curve_follow", "curve_exit"} and active_heading_source not in {
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
        min_support_heading_deg = self._turn_commit_min_heading_deg
        if nav_mode in {"curve_capture", "curve_entry"}:
            min_support_heading_deg = max(6.0, 0.75 * self._turn_commit_min_heading_deg)
        if corridor_confidence < 0.35:
            min_support_heading_deg = max(min_support_heading_deg, 9.0)
        if support_heading_deg < min_support_heading_deg:
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
