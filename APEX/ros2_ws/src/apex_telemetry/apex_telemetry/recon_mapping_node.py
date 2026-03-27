#!/usr/bin/env python3
"""Automatic reconnaissance lap and diagnostic routines for APEX."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import signal
import subprocess
import threading
import time
from importlib import import_module
from typing import Any, Optional, TextIO

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from .actuation import MaverickESCMotor, SteeringServo
from .recon_navigation import ReconCommand, ReconNavigator, ReconStateEstimate


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class DiagnosticStep:
    name: str
    steering_deg: float
    speed_pct: float
    duration_s: float


@dataclass
class PhaseStats:
    phase_name: str
    phase_started_at: Optional[Time]
    cycles: int = 0
    scan_samples: int = 0
    pose_samples: int = 0
    front_clearance_sum: float = 0.0
    min_front_clearance_m: float = float("inf")
    speed_sum: float = 0.0
    max_abs_target_heading_deg: float = 0.0
    max_abs_steering_deg: float = 0.0
    max_abs_lateral_drift_m: float = 0.0
    max_abs_yaw_change_deg: float = 0.0
    max_distance_from_phase_start_m: float = 0.0


class ReconMappingNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_recon_mapping_node")

        self._declare_parameters()

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._scan_timeout_s = max(0.1, float(self.get_parameter("scan_timeout_s").value))
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._heading_offset_deg = int(self.get_parameter("heading_offset_deg").value)
        self._enable_autostop_on_lap = bool(self.get_parameter("enable_autostop_on_lap").value)
        self._lap_min_duration_s = max(5.0, float(self.get_parameter("lap_min_duration_s").value))
        self._lap_min_path_length_m = max(1.0, float(self.get_parameter("lap_min_path_length_m").value))
        self._lap_depart_radius_m = max(0.2, float(self.get_parameter("lap_depart_radius_m").value))
        self._lap_return_radius_m = max(0.2, float(self.get_parameter("lap_return_radius_m").value))
        self._recovery_trigger_distance_m = max(
            0.1, float(self.get_parameter("recovery_trigger_distance_m").value)
        )
        self._recovery_trigger_cycles = max(
            1, int(self.get_parameter("recovery_trigger_cycles").value)
        )
        self._recovery_reverse_speed_pct = -abs(
            float(self.get_parameter("recovery_reverse_speed_pct").value)
        )
        self._recovery_duration_s = max(0.1, float(self.get_parameter("recovery_duration_s").value))
        self._save_map_on_completion = bool(self.get_parameter("save_map_on_completion").value)
        self._map_output_prefix = str(self.get_parameter("map_output_prefix").value)
        self._reset_map_on_start = bool(self.get_parameter("reset_map_on_start").value)
        self._reset_service_name = str(self.get_parameter("reset_service_name").value)
        self._clear_previous_saved_map_files = bool(
            self.get_parameter("clear_previous_saved_map_files").value
        )

        self._steering_center_trim_dc = float(
            self.get_parameter("steering_center_trim_dc").value
        )
        self._steering_direction_sign = float(
            self.get_parameter("steering_direction_sign").value
        )
        self._steering_gain = float(self.get_parameter("steering_gain").value)
        self._wall_centering_gain_deg_per_m = float(
            self.get_parameter("wall_centering_gain_deg_per_m").value
        )
        self._wall_centering_base_weight = float(
            self.get_parameter("wall_centering_base_weight").value
        )
        self._wall_avoid_distance_m = float(self.get_parameter("wall_avoid_distance_m").value)
        self._wall_avoid_gain_deg_per_m = float(
            self.get_parameter("wall_avoid_gain_deg_per_m").value
        )
        self._wall_avoid_limit_deg = float(self.get_parameter("wall_avoid_limit_deg").value)
        self._stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self._slow_distance_m = float(self.get_parameter("slow_distance_m").value)
        self._front_window_deg = max(1, int(self.get_parameter("front_window_deg").value))

        self._diagnostic_mode = str(self.get_parameter("diagnostic_mode").value).strip().lower()
        self._diagnostic_fixed_speed_pct = float(
            self.get_parameter("diagnostic_fixed_speed_pct").value
        )
        self._diagnostic_step_duration_s = max(
            0.1, float(self.get_parameter("diagnostic_step_duration_s").value)
        )
        self._diagnostic_log_every_n_cycles = max(
            1, int(self.get_parameter("diagnostic_log_every_n_cycles").value)
        )
        self._diagnostic_log_path = str(self.get_parameter("diagnostic_log_path").value).strip()
        self._diagnostic_file_flush_every_n_records = max(
            1, int(self.get_parameter("diagnostic_file_flush_every_n_records").value)
        )
        self._diagnostic_overwrite_log_on_start = bool(
            self.get_parameter("diagnostic_overwrite_log_on_start").value
        )
        self._diagnostic_recon_timeout_s = max(
            1.0, float(self.get_parameter("diagnostic_recon_timeout_s").value)
        )
        self._diagnostic_max_recoveries = max(
            1, int(self.get_parameter("diagnostic_max_recoveries").value)
        )
        self._diagnostic_min_progress_m = max(
            0.0, float(self.get_parameter("diagnostic_min_progress_m").value)
        )
        self._state_velocity_topic = str(self.get_parameter("state_velocity_topic").value)
        self._state_acceleration_topic = str(
            self.get_parameter("state_acceleration_topic").value
        )
        self._state_angular_velocity_topic = str(
            self.get_parameter("state_angular_velocity_topic").value
        )
        self._state_timeout_s = max(0.0, float(self.get_parameter("state_timeout_s").value))
        self._diagnostic_enabled = self._diagnostic_mode != "off"

        self._navigator = ReconNavigator(
            steering_limit_deg=float(self.get_parameter("steering_limit_deg").value),
            steering_gain=self._steering_gain,
            fov_half_angle_deg=float(self.get_parameter("fov_half_angle_deg").value),
            smoothing_window=int(self.get_parameter("smoothing_window").value),
            stop_distance_m=self._stop_distance_m,
            slow_distance_m=self._slow_distance_m,
            min_speed_pct=float(self.get_parameter("explore_min_speed_pct").value),
            max_speed_pct=float(self.get_parameter("explore_max_speed_pct").value),
            front_window_deg=self._front_window_deg,
            side_window_deg=int(self.get_parameter("side_window_deg").value),
            center_angle_penalty_per_deg=float(
                self.get_parameter("center_angle_penalty_per_deg").value
            ),
            wall_centering_gain_deg_per_m=self._wall_centering_gain_deg_per_m,
            wall_centering_limit_deg=float(
                self.get_parameter("wall_centering_limit_deg").value
            ),
            wall_centering_base_weight=self._wall_centering_base_weight,
            wall_avoid_distance_m=self._wall_avoid_distance_m,
            wall_avoid_gain_deg_per_m=self._wall_avoid_gain_deg_per_m,
            wall_avoid_limit_deg=self._wall_avoid_limit_deg,
            gap_escape_heading_threshold_deg=float(
                self.get_parameter("gap_escape_heading_threshold_deg").value
            ),
            gap_escape_release_distance_m=float(
                self.get_parameter("gap_escape_release_distance_m").value
            ),
            gap_escape_weight=float(self.get_parameter("gap_escape_weight").value),
            corridor_balance_ratio_threshold=float(
                self.get_parameter("corridor_balance_ratio_threshold").value
            ),
            corridor_front_min_clearance_m=float(
                self.get_parameter("corridor_front_min_clearance_m").value
            ),
            corridor_side_min_clearance_m=float(
                self.get_parameter("corridor_side_min_clearance_m").value
            ),
            corridor_front_turn_weight=float(
                self.get_parameter("corridor_front_turn_weight").value
            ),
            corridor_override_margin_deg=float(
                self.get_parameter("corridor_override_margin_deg").value
            ),
            corridor_min_heading_deg=float(
                self.get_parameter("corridor_min_heading_deg").value
            ),
            corridor_wall_start_deg=int(
                self.get_parameter("corridor_wall_start_deg").value
            ),
            corridor_wall_end_deg=int(
                self.get_parameter("corridor_wall_end_deg").value
            ),
            corridor_wall_min_points=int(
                self.get_parameter("corridor_wall_min_points").value
            ),
            wall_follow_target_distance_m=float(
                self.get_parameter("wall_follow_target_distance_m").value
            ),
            wall_follow_gain_deg_per_m=float(
                self.get_parameter("wall_follow_gain_deg_per_m").value
            ),
            wall_follow_limit_deg=float(
                self.get_parameter("wall_follow_limit_deg").value
            ),
            wall_follow_activation_heading_deg=float(
                self.get_parameter("wall_follow_activation_heading_deg").value
            ),
            wall_follow_release_balance_ratio=float(
                self.get_parameter("wall_follow_release_balance_ratio").value
            ),
            wall_follow_min_cycles=int(
                self.get_parameter("wall_follow_min_cycles").value
            ),
            wall_follow_max_clearance_m=float(
                self.get_parameter("wall_follow_max_clearance_m").value
            ),
            wall_follow_front_turn_weight=float(
                self.get_parameter("wall_follow_front_turn_weight").value
            ),
            startup_consensus_min_heading_deg=float(
                self.get_parameter("startup_consensus_min_heading_deg").value
            ),
            startup_valid_cycles_required=int(
                self.get_parameter("startup_valid_cycles_required").value
            ),
            startup_gap_lockout_cycles=int(
                self.get_parameter("startup_gap_lockout_cycles").value
            ),
            startup_latch_cycles=int(
                self.get_parameter("startup_latch_cycles").value
            ),
            ambiguity_probe_speed_pct=float(
                self.get_parameter("ambiguity_probe_speed_pct").value
            ),
            turn_speed_reduction=float(self.get_parameter("turn_speed_reduction").value),
            min_turn_speed_factor=float(self.get_parameter("min_turn_speed_factor").value),
            vehicle_half_width_m=float(self.get_parameter("vehicle_half_width_m").value),
            vehicle_front_overhang_m=float(self.get_parameter("vehicle_front_overhang_m").value),
            vehicle_rear_overhang_m=float(self.get_parameter("vehicle_rear_overhang_m").value),
            trajectory_horizon_m=float(self.get_parameter("trajectory_horizon_m").value),
            trajectory_lookahead_min_m=float(
                self.get_parameter("trajectory_lookahead_min_m").value
            ),
            trajectory_lookahead_max_m=float(
                self.get_parameter("trajectory_lookahead_max_m").value
            ),
            trajectory_curvature_slew_per_cycle=float(
                self.get_parameter("trajectory_curvature_slew_per_cycle").value
            ),
            trajectory_track_memory_alpha=float(
                self.get_parameter("trajectory_track_memory_alpha").value
            ),
            trajectory_exit_heading_threshold_deg=float(
                self.get_parameter("trajectory_exit_heading_threshold_deg").value
            ),
            trajectory_curve_speed_gain=float(
                self.get_parameter("trajectory_curve_speed_gain").value
            ),
            trajectory_state_aux_weight=float(
                self.get_parameter("trajectory_state_aux_weight").value
            ),
            trajectory_min_confidence=float(
                self.get_parameter("trajectory_min_confidence").value
            ),
            trajectory_flip_hold_cycles=int(
                self.get_parameter("trajectory_flip_hold_cycles").value
            ),
            trajectory_entry_heading_threshold_deg=float(
                self.get_parameter("trajectory_entry_heading_threshold_deg").value
            ),
            trajectory_min_radius_m=float(
                self.get_parameter("trajectory_min_radius_m").value
            ),
            trajectory_curve_heading_limit_deg=float(
                self.get_parameter("trajectory_curve_heading_limit_deg").value
            ),
        )

        self._motor = MaverickESCMotor(
            channel=int(self.get_parameter("motor_channel").value),
            frequency_hz=float(self.get_parameter("motor_frequency_hz").value),
            dc_min=float(self.get_parameter("motor_dc_min").value),
            dc_max=float(self.get_parameter("motor_dc_max").value),
            neutral_dc=float(self.get_parameter("motor_neutral_dc").value),
            reverse_brake_dc=float(self.get_parameter("reverse_brake_dc").value),
            reverse_brake_hold_s=float(self.get_parameter("reverse_brake_hold_s").value),
            reverse_neutral_hold_s=float(self.get_parameter("reverse_neutral_hold_s").value),
            reverse_exit_hold_s=float(self.get_parameter("reverse_exit_hold_s").value),
            logger=self.get_logger(),
        )
        self._steer = SteeringServo(
            channel=int(self.get_parameter("steering_channel").value),
            frequency_hz=float(self.get_parameter("steering_frequency_hz").value),
            limit_deg=float(self.get_parameter("steering_limit_deg").value),
            dc_min=float(self.get_parameter("steering_dc_min").value),
            dc_max=float(self.get_parameter("steering_dc_max").value),
            center_trim_dc=self._steering_center_trim_dc,
            direction_sign=self._steering_direction_sign,
            logger=self.get_logger(),
        )

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_scan: Optional[np.ndarray] = None
        self._latest_scan_stamp = None
        self._latest_state_velocity: Optional[tuple[float, float, float]] = None
        self._latest_state_velocity_stamp: Optional[Time] = None
        self._latest_state_acceleration: Optional[tuple[float, float, float]] = None
        self._latest_state_acceleration_stamp: Optional[Time] = None
        self._latest_state_angular_velocity: Optional[tuple[float, float, float]] = None
        self._latest_state_angular_velocity_stamp: Optional[Time] = None
        self._last_command: Optional[ReconCommand] = None
        self._blocked_cycles = 0
        self._recovery_deadline = None
        self._recovery_mode = "reverse"
        self._curve_recovery_duration_s = min(self._recovery_duration_s, 0.35)
        self._curve_recovery_speed_pct = max(20.0, self._navigator._min_speed_pct)
        self._mapping_started_at = None
        self._start_pose_xy = None
        self._last_pose_xy = None
        self._path_length_m = 0.0
        self._departed_start_zone = False
        self._completed_lap = False
        self._map_save_thread = None

        self._startup_reset_completed = not self._reset_map_on_start
        self._startup_reset_wait_started_at = None
        self._startup_reset_last_wait_log_s = -10.0
        self._startup_reset_future = None
        self._startup_reset_retry_after = None

        self._reset_service_type = None
        self._reset_client = None
        if self._reset_map_on_start:
            self._reset_service_type, self._reset_client = self._create_reset_client()

        self._phase_sequence = self._build_phase_sequence()
        self._active_phase_index = -1
        self._active_phase_name: Optional[str] = None
        self._active_step_name = "idle"
        self._active_steps: list[DiagnosticStep] = []
        self._active_step_index = -1
        self._active_step_started_at = None
        self._phase_started_at = None
        self._phase_cycle = 0
        self._phase_stats: Optional[PhaseStats] = None
        self._phase_start_pose = None
        self._phase_start_pose_source: Optional[str] = None
        self._diagnostic_completed = False
        self._recovery_events = 0
        self._diagnostic_log_handle: Optional[TextIO] = None
        self._diagnostic_pending_flush_records = 0

        if self._diagnostic_enabled:
            self._diagnostic_log_handle = self._open_diagnostic_log_handle()

        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Vector3Stamped,
            self._state_velocity_topic,
            self._state_velocity_cb,
            20,
        )
        self.create_subscription(
            Vector3Stamped,
            self._state_acceleration_topic,
            self._state_acceleration_cb,
            20,
        )
        self.create_subscription(
            Vector3Stamped,
            self._state_angular_velocity_topic,
            self._state_angular_velocity_cb,
            20,
        )
        self.create_timer(1.0 / self._control_rate_hz, self._control_loop)

        self.get_logger().info(
            "ReconMappingNode ready (scan=%s, control_rate=%.1f Hz, mode=%s)"
            % (self._scan_topic, self._control_rate_hz, self._diagnostic_mode)
        )
        self._emit_diag_config()

    def _declare_parameters(self) -> None:
        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("scan_timeout_s", 0.6)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("heading_offset_deg", -89)

        self.declare_parameter("steering_channel", 1)
        self.declare_parameter("steering_frequency_hz", 50.0)
        self.declare_parameter("steering_limit_deg", 18.0)
        self.declare_parameter("steering_dc_min", 5.0)
        self.declare_parameter("steering_dc_max", 8.6)
        self.declare_parameter("steering_center_trim_dc", 0.0)
        self.declare_parameter("steering_direction_sign", 1.0)

        self.declare_parameter("motor_channel", 0)
        self.declare_parameter("motor_frequency_hz", 50.0)
        self.declare_parameter("motor_dc_min", 5.0)
        self.declare_parameter("motor_dc_max", 10.0)
        self.declare_parameter("motor_neutral_dc", 7.5)
        self.declare_parameter("reverse_brake_dc", 6.9)
        self.declare_parameter("reverse_brake_hold_s", 0.12)
        self.declare_parameter("reverse_neutral_hold_s", 0.12)
        self.declare_parameter("reverse_exit_hold_s", 0.15)

        self.declare_parameter("explore_min_speed_pct", 18.0)
        self.declare_parameter("explore_max_speed_pct", 24.0)
        self.declare_parameter("recovery_reverse_speed_pct", 30.0)
        self.declare_parameter("recovery_duration_s", 0.8)
        self.declare_parameter("recovery_trigger_distance_m", 0.30)
        self.declare_parameter("recovery_trigger_cycles", 4)

        self.declare_parameter("stop_distance_m", 0.35)
        self.declare_parameter("slow_distance_m", 0.90)
        self.declare_parameter("front_window_deg", 12)
        self.declare_parameter("side_window_deg", 70)
        self.declare_parameter("fov_half_angle_deg", 90.0)
        self.declare_parameter("steering_gain", 0.35)
        self.declare_parameter("center_angle_penalty_per_deg", 0.012)
        self.declare_parameter("wall_centering_gain_deg_per_m", 18.0)
        self.declare_parameter("wall_centering_limit_deg", 18.0)
        self.declare_parameter("wall_centering_base_weight", 0.35)
        self.declare_parameter("wall_avoid_distance_m", 0.45)
        self.declare_parameter("wall_avoid_gain_deg_per_m", 25.0)
        self.declare_parameter("wall_avoid_limit_deg", 20.0)
        self.declare_parameter("gap_escape_heading_threshold_deg", 45.0)
        self.declare_parameter("gap_escape_release_distance_m", 0.18)
        self.declare_parameter("gap_escape_weight", 0.35)
        self.declare_parameter("corridor_balance_ratio_threshold", 0.20)
        self.declare_parameter("corridor_front_min_clearance_m", 0.08)
        self.declare_parameter("corridor_side_min_clearance_m", 0.05)
        self.declare_parameter("corridor_front_turn_weight", 0.70)
        self.declare_parameter("corridor_override_margin_deg", 4.0)
        self.declare_parameter("corridor_min_heading_deg", 2.0)
        self.declare_parameter("corridor_wall_start_deg", 25)
        self.declare_parameter("corridor_wall_end_deg", 85)
        self.declare_parameter("corridor_wall_min_points", 6)
        self.declare_parameter("wall_follow_target_distance_m", 0.24)
        self.declare_parameter("wall_follow_gain_deg_per_m", 55.0)
        self.declare_parameter("wall_follow_limit_deg", 22.0)
        self.declare_parameter("wall_follow_activation_heading_deg", 10.0)
        self.declare_parameter("wall_follow_release_balance_ratio", 0.72)
        self.declare_parameter("wall_follow_min_cycles", 10)
        self.declare_parameter("wall_follow_max_clearance_m", 1.25)
        self.declare_parameter("wall_follow_front_turn_weight", 0.35)
        self.declare_parameter("startup_consensus_min_heading_deg", 2.0)
        self.declare_parameter("startup_valid_cycles_required", 3)
        self.declare_parameter("startup_gap_lockout_cycles", 8)
        self.declare_parameter("startup_latch_cycles", 18)
        self.declare_parameter("ambiguity_probe_speed_pct", 15.0)
        self.declare_parameter("turn_speed_reduction", 0.35)
        self.declare_parameter("min_turn_speed_factor", 0.70)
        self.declare_parameter("smoothing_window", 9)
        self.declare_parameter("vehicle_half_width_m", 0.11)
        self.declare_parameter("vehicle_front_overhang_m", 0.11)
        self.declare_parameter("vehicle_rear_overhang_m", 0.31)

        self.declare_parameter("enable_autostop_on_lap", True)
        self.declare_parameter("lap_min_duration_s", 25.0)
        self.declare_parameter("lap_min_path_length_m", 8.0)
        self.declare_parameter("lap_depart_radius_m", 1.2)
        self.declare_parameter("lap_return_radius_m", 0.6)
        self.declare_parameter("save_map_on_completion", True)
        self.declare_parameter("map_output_prefix", "/work/ros2_ws/maps/apex_recon_map")
        self.declare_parameter("reset_map_on_start", True)
        self.declare_parameter("reset_service_name", "/slam_toolbox/reset")
        self.declare_parameter("clear_previous_saved_map_files", True)

        self.declare_parameter("diagnostic_mode", "calibration")
        self.declare_parameter("diagnostic_fixed_speed_pct", 10.0)
        self.declare_parameter("diagnostic_step_duration_s", 1.2)
        self.declare_parameter("diagnostic_log_every_n_cycles", 1)
        self.declare_parameter("diagnostic_log_path", "/work/ros2_ws/logs/recon_diagnostic.log")
        self.declare_parameter("diagnostic_file_flush_every_n_records", 5)
        self.declare_parameter("diagnostic_overwrite_log_on_start", True)
        self.declare_parameter("diagnostic_recon_timeout_s", 25.0)
        self.declare_parameter("diagnostic_max_recoveries", 4)
        self.declare_parameter("diagnostic_min_progress_m", 0.30)
        self.declare_parameter("state_velocity_topic", "/apex/kinematics/velocity")
        self.declare_parameter("state_acceleration_topic", "/apex/kinematics/acceleration")
        self.declare_parameter("state_angular_velocity_topic", "/apex/kinematics/angular_velocity")
        self.declare_parameter("state_timeout_s", 0.5)
        self.declare_parameter("trajectory_horizon_m", 0.95)
        self.declare_parameter("trajectory_lookahead_min_m", 0.35)
        self.declare_parameter("trajectory_lookahead_max_m", 0.85)
        self.declare_parameter("trajectory_curvature_slew_per_cycle", 0.22)
        self.declare_parameter("trajectory_track_memory_alpha", 0.55)
        self.declare_parameter("trajectory_exit_heading_threshold_deg", 2.0)
        self.declare_parameter("trajectory_curve_speed_gain", 1.2)
        self.declare_parameter("trajectory_state_aux_weight", 0.30)
        self.declare_parameter("trajectory_min_confidence", 0.28)
        self.declare_parameter("trajectory_flip_hold_cycles", 8)
        self.declare_parameter("trajectory_entry_heading_threshold_deg", 4.0)
        self.declare_parameter("trajectory_min_radius_m", 1.35)
        self.declare_parameter("trajectory_curve_heading_limit_deg", 18.0)

    def _create_reset_client(self):
        try:
            slam_toolbox_srv = import_module("slam_toolbox.srv")
            reset_service_type = getattr(slam_toolbox_srv, "Reset")
        except Exception as exc:
            raise RuntimeError(
                "slam_toolbox Reset service type is unavailable; reconnaissance mapping "
                "requires slam_toolbox to be installed in the runtime container"
            ) from exc

        return reset_service_type, self.create_client(reset_service_type, self._reset_service_name)

    def _scan_cb(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = 0.0
        ranges[ranges < max(0.0, float(msg.range_min))] = 0.0
        ranges[ranges > float(msg.range_max)] = 0.0
        self._latest_scan = ranges
        self._latest_scan_stamp = self.get_clock().now()

    def _msg_time_or_now(self, msg: Vector3Stamped) -> Time:
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            return Time.from_msg(msg.header.stamp)
        return self.get_clock().now()

    def _state_velocity_cb(self, msg: Vector3Stamped) -> None:
        self._latest_state_velocity = (
            float(msg.vector.x),
            float(msg.vector.y),
            float(msg.vector.z),
        )
        self._latest_state_velocity_stamp = self._msg_time_or_now(msg)

    def _state_acceleration_cb(self, msg: Vector3Stamped) -> None:
        self._latest_state_acceleration = (
            float(msg.vector.x),
            float(msg.vector.y),
            float(msg.vector.z),
        )
        self._latest_state_acceleration_stamp = self._msg_time_or_now(msg)

    def _state_angular_velocity_cb(self, msg: Vector3Stamped) -> None:
        self._latest_state_angular_velocity = (
            float(msg.vector.x),
            float(msg.vector.y),
            float(msg.vector.z),
        )
        self._latest_state_angular_velocity_stamp = self._msg_time_or_now(msg)

    def _fresh_state_tuple(
        self,
        stamp: Optional[Time],
        value: Optional[tuple[float, float, float]],
    ) -> Optional[tuple[float, float, float]]:
        if stamp is None or value is None:
            return None
        if self._state_timeout_s > 0.0:
            age_s = (self.get_clock().now() - stamp).nanoseconds * 1e-9
            if age_s > self._state_timeout_s:
                return None
        return value

    def _build_phase_sequence(self) -> list[str]:
        mode = self._diagnostic_mode
        if mode == "off":
            return ["recon_debug"]
        if mode == "calibration":
            return ["steering_static", "straight_open_loop"]
        if mode in {
            "steering_static",
            "straight_open_loop",
            "steering_sign_check",
            "nav_dryrun",
            "recon_debug",
        }:
            return [mode]
        if mode == "all":
            return [
                "steering_static",
                "straight_open_loop",
                "steering_sign_check",
                "nav_dryrun",
                "recon_debug",
            ]
        self.get_logger().warning("Unknown diagnostic_mode=%s, falling back to calibration" % mode)
        self._diagnostic_mode = "calibration"
        self._diagnostic_enabled = True
        return ["steering_static", "straight_open_loop"]

    def _steps_for_phase(self, phase_name: str) -> list[DiagnosticStep]:
        duration = self._diagnostic_step_duration_s
        speed = self._diagnostic_fixed_speed_pct
        if phase_name == "steering_static":
            return [
                DiagnosticStep("steer_left_static", -10.0, 0.0, duration),
                DiagnosticStep("steer_center_static", 0.0, 0.0, duration),
                DiagnosticStep("steer_right_static", 10.0, 0.0, duration),
                DiagnosticStep("steer_center_static_end", 0.0, 0.0, duration),
            ]
        if phase_name == "straight_open_loop":
            return [DiagnosticStep("straight_open_loop", 0.0, speed, duration)]
        if phase_name == "steering_sign_check":
            return [
                DiagnosticStep("sign_left", -8.0, speed, duration),
                DiagnosticStep("sign_center_mid", 0.0, 0.0, duration),
                DiagnosticStep("sign_right", 8.0, speed, duration),
                DiagnosticStep("sign_center_end", 0.0, 0.0, duration),
            ]
        return []

    def _start_next_phase(self, now: Time) -> None:
        self._active_phase_index += 1
        if self._active_phase_index >= len(self._phase_sequence):
            self._diagnostic_completed = True
            self._active_phase_name = None
            self._active_step_name = "complete"
            self._apply_command(0.0, 0.0)
            return

        self._active_phase_name = self._phase_sequence[self._active_phase_index]
        self._active_step_name = "idle"
        self._active_steps = self._steps_for_phase(self._active_phase_name)
        self._active_step_index = -1
        self._active_step_started_at = None
        self._phase_started_at = now
        self._phase_cycle = 0
        self._phase_stats = PhaseStats(
            phase_name=self._active_phase_name,
            phase_started_at=now,
        )
        self._navigator.reset_runtime_state()
        self._phase_start_pose, self._phase_start_pose_source = self._lookup_pose_with_source()
        self._blocked_cycles = 0
        self._recovery_deadline = None
        self._recovery_mode = "reverse"
        self._recovery_events = 0
        self._mapping_started_at = None
        if self._active_phase_name == "recon_debug":
            self._reset_lap_tracking()
        self.get_logger().info("Starting diagnostic phase %s" % self._active_phase_name)

    def _end_current_phase(self, reason: str, now: Time) -> None:
        if self._active_phase_name is None or self._phase_stats is None:
            return

        summary = {
            "reason": reason,
            "duration_s": self._duration_since(self._phase_stats.phase_started_at, now),
            "cycles": self._phase_stats.cycles,
            "avg_front_clearance_m": self._safe_avg(
                self._phase_stats.front_clearance_sum,
                self._phase_stats.scan_samples,
            ),
            "min_front_clearance_m": (
                None
                if self._phase_stats.scan_samples == 0
                else self._phase_stats.min_front_clearance_m
            ),
            "avg_speed_pct": self._safe_avg(
                self._phase_stats.speed_sum,
                self._phase_stats.cycles,
            ),
            "max_abs_target_heading_deg": self._phase_stats.max_abs_target_heading_deg,
            "max_abs_steering_deg": self._phase_stats.max_abs_steering_deg,
            "max_abs_lateral_drift_m": self._phase_stats.max_abs_lateral_drift_m,
            "max_abs_yaw_change_deg": self._phase_stats.max_abs_yaw_change_deg,
            "max_distance_from_phase_start_m": self._phase_stats.max_distance_from_phase_start_m,
        }
        if self._diagnostic_enabled:
            self._emit_diag("DIAG_SUMMARY", summary)

        self._apply_command(0.0, 0.0)
        self._active_phase_name = None
        self._active_step_name = "idle"
        self._active_steps = []
        self._active_step_index = -1
        self._active_step_started_at = None
        self._phase_started_at = None
        self._phase_cycle = 0
        self._phase_stats = None
        self._phase_start_pose = None
        self._phase_start_pose_source = None

        self._start_next_phase(now)

    def _control_loop(self) -> None:
        now = self.get_clock().now()

        if self._diagnostic_completed:
            self._apply_command(0.0, 0.0)
            return

        if self._active_phase_name is None:
            self._start_next_phase(now)
            if self._diagnostic_completed:
                return

        if self._active_phase_name != "nav_dryrun" and not self._ensure_startup_map_reset():
            self._apply_command(0.0, 0.0)
            return

        if self._active_phase_name == "recon_debug":
            self._run_recon_debug(now)
        elif self._active_phase_name == "nav_dryrun":
            self._run_nav_dryrun(now)
        else:
            self._run_scripted_phase(now)

    def _run_scripted_phase(self, now: Time) -> None:
        if not self._active_steps:
            self._end_current_phase("no_steps", now)
            return

        if self._active_step_index < 0:
            self._active_step_index = 0
            self._active_step_started_at = now
            self._active_step_name = self._active_steps[0].name

        current_step = self._active_steps[self._active_step_index]
        pose_diag = self._build_pose_diag()
        scan_command, front_min_m, scan_age_s = self._build_scan_diagnostics(
            pose_diag=pose_diag
        )
        self._apply_command(current_step.speed_pct, current_step.steering_deg)

        self._phase_cycle += 1
        self._update_phase_stats(
            applied_speed_pct=current_step.speed_pct,
            applied_steering_deg=current_step.steering_deg,
            nav_command=scan_command,
            pose_diag=pose_diag,
        )
        self._maybe_emit_cycle_logs(
            nav_command=scan_command,
            front_min_m=front_min_m,
            pose_diag=pose_diag,
            scan_age_s=scan_age_s,
        )

        if self._active_step_started_at is None:
            return

        if self._duration_since(self._active_step_started_at, now) < current_step.duration_s:
            return

        self._active_step_index += 1
        if self._active_step_index >= len(self._active_steps):
            self._end_current_phase("script_complete", now)
            return

        self._active_step_started_at = now
        self._active_step_name = self._active_steps[self._active_step_index].name

    def _run_recon_debug(self, now: Time) -> None:
        if self._completed_lap:
            self._end_current_phase("lap_complete", now)
            return

        scan, age_s = self._get_fresh_scan()
        if scan is None:
            if age_s is not None and age_s > self._scan_timeout_s:
                self.get_logger().warning("Stopping: LiDAR scan timeout %.2fs" % age_s)
            self._apply_command(0.0, 0.0)
            self._phase_cycle += 1
            self._maybe_emit_cycle_logs(
                nav_command=None,
                front_min_m=None,
                pose_diag=self._build_pose_diag(),
                scan_age_s=age_s,
            )
            return

        if self._mapping_started_at is None:
            self._mapping_started_at = now
            self.get_logger().info("Reconnaissance lap started")
        elif self._duration_since(self._mapping_started_at, now) >= self._diagnostic_recon_timeout_s:
            self.get_logger().warning(
                "Ending recon_debug after %.1fs without lap completion"
                % self._diagnostic_recon_timeout_s
            )
            self._end_current_phase("recon_timeout", now)
            return

        pose_diag = self._build_pose_diag()
        state_estimate = self._build_state_estimate(pose_diag)

        if self._recovery_deadline is not None and now < self._recovery_deadline:
            scan_command, front_min_m, scan_age_s = self._build_scan_diagnostics(
                scan,
                scan_age_s=age_s,
                pose_diag=pose_diag,
                state_estimate=state_estimate,
            )
            if self._recovery_mode == "curve_crawl":
                self._active_step_name = "recovery_curve_crawl"
                applied_speed_pct, applied_steering_deg = self._build_curve_recovery_command(
                    scan_command
                )
            else:
                self._active_step_name = "recovery_reverse"
                applied_speed_pct = self._recovery_reverse_speed_pct
                applied_steering_deg = 0.0
            self._apply_command(applied_speed_pct, applied_steering_deg)
            self._phase_cycle += 1
            self._update_phase_stats(
                applied_speed_pct=applied_speed_pct,
                applied_steering_deg=applied_steering_deg,
                nav_command=scan_command,
                pose_diag=pose_diag,
            )
            self._maybe_emit_cycle_logs(
                nav_command=scan_command,
                front_min_m=front_min_m,
                pose_diag=pose_diag,
                scan_age_s=scan_age_s,
            )
            return

        self._recovery_deadline = None
        self._recovery_mode = "reverse"
        self._active_step_name = "autonomous"
        command = self._navigator.compute_command(
            scan,
            state_estimate=state_estimate,
        )
        self._last_command = command

        if 0.0 < command.effective_front_clearance_m <= self._recovery_trigger_distance_m:
            self._blocked_cycles += 1
        else:
            self._blocked_cycles = 0

        if self._blocked_cycles >= self._recovery_trigger_cycles:
            self._blocked_cycles = 0
            self._recovery_events += 1
            self._recovery_mode = self._select_recovery_mode(command)
            recovery_duration_s = (
                self._curve_recovery_duration_s
                if self._recovery_mode == "curve_crawl"
                else self._recovery_duration_s
            )
            self._recovery_deadline = now + Duration(seconds=recovery_duration_s)
            if self._recovery_mode == "curve_crawl":
                self.get_logger().warning(
                    "Front blocked in curve (%.2fm, source=%s). Starting forward curve recovery for %.2fs"
                    % (
                        command.effective_front_clearance_m,
                        command.active_heading_source,
                        recovery_duration_s,
                    )
                )
            else:
                self.get_logger().warning(
                    "Front blocked (%.2fm). Starting recovery reverse for %.2fs"
                    % (command.effective_front_clearance_m, recovery_duration_s)
                )
            progress_m = 0.0
            if self._phase_stats is not None:
                progress_m = self._phase_stats.max_distance_from_phase_start_m
            if (
                self._recovery_events >= self._diagnostic_max_recoveries
                and progress_m < self._diagnostic_min_progress_m
            ):
                self.get_logger().warning(
                    "Ending recon_debug after %d recovery events with only %.2fm progress"
                    % (self._recovery_events, progress_m)
                )
                self._end_current_phase("stuck_recovery_loop", now)
                return
            if self._recovery_mode == "curve_crawl":
                self._active_step_name = "recovery_curve_crawl"
                applied_speed_pct, applied_steering_deg = self._build_curve_recovery_command(
                    command
                )
            else:
                self._active_step_name = "recovery_reverse"
                applied_speed_pct = self._recovery_reverse_speed_pct
                applied_steering_deg = 0.0
            self._apply_command(applied_speed_pct, applied_steering_deg)
            self._phase_cycle += 1
            self._update_phase_stats(
                applied_speed_pct=applied_speed_pct,
                applied_steering_deg=applied_steering_deg,
                nav_command=command,
                pose_diag=pose_diag,
            )
            self._maybe_emit_cycle_logs(
                nav_command=command,
                front_min_m=self._compute_front_min(scan),
                pose_diag=pose_diag,
                scan_age_s=age_s,
            )
            return

        self._apply_command(command.speed_pct, command.steering_deg)
        self._update_lap_tracking()

        self._phase_cycle += 1
        self._update_phase_stats(
            applied_speed_pct=command.speed_pct,
            applied_steering_deg=command.steering_deg,
            nav_command=command,
            pose_diag=pose_diag,
        )
        self._maybe_emit_cycle_logs(
            nav_command=command,
            front_min_m=self._compute_front_min(scan),
            pose_diag=pose_diag,
            scan_age_s=age_s,
        )

    def _run_nav_dryrun(self, now: Time) -> None:
        scan, age_s = self._get_fresh_scan()
        self._active_step_name = "dryrun"

        if self._mapping_started_at is None:
            self._mapping_started_at = now
            self.get_logger().info("Navigation dry-run started")
        elif self._duration_since(self._mapping_started_at, now) >= self._diagnostic_recon_timeout_s:
            self.get_logger().warning(
                "Ending nav_dryrun after %.1fs"
                % self._diagnostic_recon_timeout_s
            )
            self._end_current_phase("nav_dryrun_timeout", now)
            return

        if scan is None:
            if age_s is not None and age_s > self._scan_timeout_s:
                self.get_logger().warning("Dry-run: LiDAR scan timeout %.2fs" % age_s)
            self._phase_cycle += 1
            self._maybe_emit_cycle_logs(
                nav_command=None,
                front_min_m=None,
                pose_diag=self._build_pose_diag(),
                scan_age_s=age_s,
            )
            return

        pose_diag = self._build_pose_diag()
        state_estimate = self._build_state_estimate(pose_diag)
        command = self._navigator.compute_command(
            scan,
            state_estimate=state_estimate,
        )
        self._last_command = command

        self._phase_cycle += 1
        self._update_phase_stats(
            applied_speed_pct=0.0,
            applied_steering_deg=command.steering_pre_servo_deg,
            nav_command=command,
            pose_diag=pose_diag,
        )
        self._maybe_emit_cycle_logs(
            nav_command=command,
            front_min_m=self._compute_front_min(scan),
            pose_diag=pose_diag,
            scan_age_s=age_s,
        )

    def _select_recovery_mode(self, command: ReconCommand) -> str:
        if self._is_curve_recovery_context(command):
            return "curve_crawl"
        return "reverse"

    def _is_curve_recovery_context(self, command: ReconCommand) -> bool:
        if command.nav_mode in {"curve_entry", "curve_follow", "curve_exit"}:
            return True

        if command.curve_intent_score < 0.20 and command.curve_evidence_strength < 0.18:
            return False

        if command.wall_follow_active:
            return True

        if command.active_heading_source in {
            "curve_entry_guard",
            "corridor_center",
            "front_turn",
            "reference_blend",
            "reference_centerline",
            "reference_curve_commit",
            "reference_free_space",
            "turn_commit",
            "wall_follow",
            "startup_latch",
        }:
            return True

        support_heading_deg = max(
            abs(command.front_turn_heading_deg),
            abs(command.corridor_center_heading_deg),
            abs(command.target_heading_deg),
        )
        return support_heading_deg >= 8.0

    def _build_curve_recovery_command(
        self, command: Optional[ReconCommand]
    ) -> tuple[float, float]:
        if command is None:
            return 0.0, 0.0

        speed_pct = max(self._curve_recovery_speed_pct, float(command.speed_pct))
        steering_deg = float(command.steering_deg)
        if abs(steering_deg) >= 2.0:
            return speed_pct, steering_deg

        fallback_heading_deg = 0.0
        for candidate in (
            command.target_heading_deg,
            command.wall_follow_heading_deg,
            command.front_turn_heading_deg,
            command.corridor_center_heading_deg,
        ):
            if abs(candidate) > abs(fallback_heading_deg):
                fallback_heading_deg = float(candidate)

        if abs(fallback_heading_deg) < 1e-6:
            return speed_pct, 0.0

        fallback_limit_deg = min(12.0, max(6.0, abs(fallback_heading_deg)))
        steering_deg = math.copysign(fallback_limit_deg, fallback_heading_deg)
        return speed_pct, steering_deg

    def _apply_command(self, speed_pct: float, steering_deg: float) -> None:
        self._steer.set_angle_deg(steering_deg)
        self._motor.set_speed_pct(speed_pct)

    def _build_scan_diagnostics(
        self,
        scan_override: Optional[np.ndarray] = None,
        *,
        scan_age_s: Optional[float] = None,
        pose_diag: Optional[dict[str, Optional[float]]] = None,
        state_estimate: Optional[ReconStateEstimate] = None,
    ) -> tuple[Optional[ReconCommand], Optional[float], Optional[float]]:
        scan = scan_override
        if scan is None:
            scan, scan_age_s = self._get_fresh_scan()
        elif scan_age_s is None and self._latest_scan_stamp is not None:
            scan_age_s = (self.get_clock().now() - self._latest_scan_stamp).nanoseconds * 1e-9
        if scan is None:
            return None, None, scan_age_s
        if pose_diag is None:
            pose_diag = self._build_pose_diag()
        if state_estimate is None:
            state_estimate = self._build_state_estimate(pose_diag)
        command = self._navigator.compute_command(
            scan,
            state_estimate=state_estimate,
        )
        return command, self._compute_front_min(scan), scan_age_s

    def _get_fresh_scan(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        if self._latest_scan is None or self._latest_scan_stamp is None:
            return None, None

        age_s = (self.get_clock().now() - self._latest_scan_stamp).nanoseconds * 1e-9
        if age_s > self._scan_timeout_s:
            return None, age_s
        return self._latest_scan.copy(), age_s

    def _compute_front_min(self, scan_ranges: np.ndarray) -> Optional[float]:
        values = []
        for degree in range(-self._front_window_deg, self._front_window_deg + 1):
            distance = float(scan_ranges[int(round(degree)) % scan_ranges.size])
            if distance > 0.0:
                values.append(distance)
        return min(values) if values else None

    def _build_pose_diag(self) -> dict[str, Optional[float]]:
        pose, pose_source = self._lookup_pose_with_source()
        if pose is None:
            return {
                "x": None,
                "y": None,
                "yaw_deg": None,
                "distance_from_phase_start_m": None,
                "lateral_drift_m": None,
                "yaw_change_deg": None,
                "pose_source": None,
            }

        if self._phase_start_pose is None or self._phase_start_pose_source != pose_source:
            self._phase_start_pose = pose
            self._phase_start_pose_source = pose_source

        x, y, yaw = pose
        x0, y0, yaw0 = self._phase_start_pose
        dx = x - x0
        dy = y - y0
        distance = math.hypot(dx, dy)
        lateral = (-math.sin(yaw0) * dx) + (math.cos(yaw0) * dy)
        yaw_change_deg = _normalize_angle_deg(math.degrees(yaw - yaw0))
        return {
            "x": x,
            "y": y,
            "yaw_deg": math.degrees(yaw),
            "distance_from_phase_start_m": distance,
            "lateral_drift_m": lateral,
            "yaw_change_deg": yaw_change_deg,
            "pose_source": pose_source,
        }

    def _build_state_estimate(
        self,
        pose_diag: dict[str, Optional[float]],
    ) -> ReconStateEstimate:
        velocity = self._fresh_state_tuple(
            self._latest_state_velocity_stamp,
            self._latest_state_velocity,
        )
        acceleration = self._fresh_state_tuple(
            self._latest_state_acceleration_stamp,
            self._latest_state_acceleration,
        )
        angular_velocity = self._fresh_state_tuple(
            self._latest_state_angular_velocity_stamp,
            self._latest_state_angular_velocity,
        )

        speed_mps = 0.0
        accel_mps2 = 0.0
        yaw_rate_rps = 0.0
        slip_proxy = 0.0

        if velocity is not None:
            speed_mps = math.hypot(velocity[0], velocity[1])
        if acceleration is not None:
            accel_mps2 = math.hypot(acceleration[0], acceleration[1])
        if angular_velocity is not None:
            yaw_rate_rps = float(angular_velocity[2])

        yaw_deg = pose_diag["yaw_deg"]
        if velocity is not None and yaw_deg is not None and speed_mps >= 0.05:
            velocity_heading_deg = math.degrees(math.atan2(velocity[1], velocity[0]))
            slip_proxy = _normalize_angle_deg(velocity_heading_deg - yaw_deg)

        valid = (
            pose_diag["x"] is not None
            or velocity is not None
            or acceleration is not None
            or angular_velocity is not None
        )
        return ReconStateEstimate(
            x_m=float(pose_diag["x"] or 0.0),
            y_m=float(pose_diag["y"] or 0.0),
            yaw_deg=float(yaw_deg or 0.0),
            yaw_rate_rps=float(yaw_rate_rps),
            speed_mps=float(speed_mps),
            accel_mps2=float(accel_mps2),
            lateral_drift_m=float(pose_diag["lateral_drift_m"] or 0.0),
            slip_proxy=float(slip_proxy),
            pose_source=str(pose_diag["pose_source"] or "none"),
            distance_from_phase_start_m=float(
                pose_diag["distance_from_phase_start_m"] or 0.0
            ),
            yaw_change_deg=float(pose_diag["yaw_change_deg"] or 0.0),
            valid=bool(valid),
        )

    def _lookup_pose_with_source(self) -> tuple[Optional[tuple[float, float, float]], Optional[str]]:
        for source_frame, source_name in (
            (self._map_frame, "map"),
            (self._odom_frame, "odom"),
        ):
            try:
                transform = self._tf_buffer.lookup_transform(
                    source_frame,
                    self._base_frame,
                    Time(),
                )
            except Exception:
                continue

            rotation = transform.transform.rotation
            yaw = _yaw_from_quat(rotation.x, rotation.y, rotation.z, rotation.w)
            return (
                (
                    float(transform.transform.translation.x),
                    float(transform.transform.translation.y),
                    float(yaw),
                ),
                source_name,
            )

        return None, None

    def _lookup_pose(self) -> Optional[tuple[float, float, float]]:
        pose, _ = self._lookup_pose_with_source()
        if pose is None:
            return None

        return pose

    def _update_phase_stats(
        self,
        *,
        applied_speed_pct: float,
        applied_steering_deg: float,
        nav_command: Optional[ReconCommand],
        pose_diag: dict[str, Optional[float]],
    ) -> None:
        if self._phase_stats is None:
            return

        self._phase_stats.cycles += 1
        self._phase_stats.speed_sum += float(applied_speed_pct)
        self._phase_stats.max_abs_steering_deg = max(
            self._phase_stats.max_abs_steering_deg,
            abs(float(applied_steering_deg)),
        )

        if nav_command is not None:
            self._phase_stats.scan_samples += 1
            self._phase_stats.front_clearance_sum += nav_command.front_clearance_m
            self._phase_stats.min_front_clearance_m = min(
                self._phase_stats.min_front_clearance_m,
                nav_command.front_clearance_m,
            )
            self._phase_stats.max_abs_target_heading_deg = max(
                self._phase_stats.max_abs_target_heading_deg,
                abs(nav_command.target_heading_deg),
            )

        if pose_diag["distance_from_phase_start_m"] is not None:
            self._phase_stats.pose_samples += 1
            self._phase_stats.max_distance_from_phase_start_m = max(
                self._phase_stats.max_distance_from_phase_start_m,
                abs(float(pose_diag["distance_from_phase_start_m"])),
            )
        if pose_diag["lateral_drift_m"] is not None:
            self._phase_stats.max_abs_lateral_drift_m = max(
                self._phase_stats.max_abs_lateral_drift_m,
                abs(float(pose_diag["lateral_drift_m"])),
            )
        if pose_diag["yaw_change_deg"] is not None:
            self._phase_stats.max_abs_yaw_change_deg = max(
                self._phase_stats.max_abs_yaw_change_deg,
                abs(float(pose_diag["yaw_change_deg"])),
            )

    def _maybe_emit_cycle_logs(
        self,
        *,
        nav_command: Optional[ReconCommand],
        front_min_m: Optional[float],
        pose_diag: dict[str, Optional[float]],
        scan_age_s: Optional[float],
    ) -> None:
        if not self._diagnostic_enabled:
            return
        if self._phase_cycle % self._diagnostic_log_every_n_cycles != 0:
            return

        self._emit_diag("DIAG_POSE", pose_diag)
        self._emit_diag("DIAG_STEER", self._steer.get_state())
        self._emit_diag("DIAG_MOTOR", self._motor.get_state())

        if nav_command is None:
            self._emit_diag(
                "DIAG_SCAN",
                {
                    "heading_offset_deg": self._heading_offset_deg,
                    "scan_age_s": scan_age_s,
                    "front_clearance_m": None,
                    "effective_front_clearance_m": None,
                    "front_clearance_fallback_used": None,
                    "front_left_clearance_m": None,
                    "front_right_clearance_m": None,
                    "corridor_balance_ratio": None,
                    "corridor_available": None,
                    "left_clearance_m": None,
                    "right_clearance_m": None,
                    "left_min_m": None,
                    "right_min_m": None,
                    "front_min_m": front_min_m,
                    "left_right_delta_m": None,
                },
            )
            self._emit_diag(
                "DIAG_NAV",
                {
                    "nav_mode": None,
                    "corridor_confidence": None,
                    "curve_confidence": None,
                    "preview_heading_deg": None,
                    "corridor_curvature_sign": None,
                    "corridor_curvature_confidence": None,
                    "committed_turn_sign": None,
                    "gate_curve_sign": None,
                    "curve_capture_active": None,
                    "curve_capture_reason": None,
                    "curve_severity_score": None,
                    "curve_steering_floor_deg": None,
                    "curve_speed_cap_pct": None,
                    "curve_release_reason": None,
                    "same_sign_trim_active": None,
                    "free_space_candidate_heading_deg": None,
                    "sign_veto_reason": None,
                    "straight_veto_active": None,
                    "startup_adapt_active": None,
                    "curve_intent_score": None,
                    "curve_intent_sign": None,
                    "curve_evidence_strength": None,
                    "curve_decay_active": None,
                    "premature_curve_veto": None,
                    "pre_curve_bias_veto": None,
                    "near_wall_mode": None,
                    "straight_corridor_score": None,
                    "curve_gate_open": None,
                    "curve_gate_reason": None,
                    "geometry_agreement_score": None,
                    "curve_confirm_distance_m": None,
                    "curve_confirm_yaw_deg": None,
                    "trajectory_phase": None,
                    "lookahead_x_m": None,
                    "lookahead_y_m": None,
                    "signed_curvature": None,
                    "radius_m": None,
                    "target_speed_pct": None,
                    "track_confidence": None,
                    "state_speed_mps": None,
                    "state_yaw_rate": None,
                    "slip_proxy": None,
                    "trajectory_flip_blocked": None,
                    "pose_source": pose_diag["pose_source"],
                    "gap_heading_deg": None,
                    "front_turn_heading_deg": None,
                    "corridor_axis_heading_deg": None,
                    "corridor_center_heading_deg": None,
                    "wall_follow_heading_deg": None,
                    "wall_follow_active": None,
                    "wall_follow_anchor_side": None,
                    "startup_candidate_heading_deg": None,
                    "startup_candidate_source": None,
                    "startup_hold_active": None,
                    "startup_latched_sign": None,
                    "startup_latch_cycles_remaining": None,
                    "left_wall_heading_deg": None,
                    "right_wall_heading_deg": None,
                    "centering_heading_deg": None,
                    "avoidance_heading_deg": None,
                    "centering_weight": None,
                    "active_heading_source": None,
                    "target_heading_deg": None,
                    "steering_pre_servo_deg": None,
                    "steering_deg": None,
                    "speed_pct": None,
                },
            )
        else:
            self._emit_diag(
                "DIAG_SCAN",
                {
                    "heading_offset_deg": self._heading_offset_deg,
                    "scan_age_s": scan_age_s,
                    "front_clearance_m": nav_command.front_clearance_m,
                    "effective_front_clearance_m": nav_command.effective_front_clearance_m,
                    "front_clearance_fallback_used": nav_command.front_clearance_fallback_used,
                    "front_left_clearance_m": nav_command.front_left_clearance_m,
                    "front_right_clearance_m": nav_command.front_right_clearance_m,
                    "corridor_balance_ratio": nav_command.corridor_balance_ratio,
                    "corridor_available": nav_command.corridor_available,
                    "left_clearance_m": nav_command.left_clearance_m,
                    "right_clearance_m": nav_command.right_clearance_m,
                    "left_min_m": nav_command.left_min_m,
                    "right_min_m": nav_command.right_min_m,
                    "front_min_m": front_min_m,
                    "left_right_delta_m": nav_command.left_right_delta_m,
                },
            )
            self._emit_diag(
                "DIAG_NAV",
                {
                    "nav_mode": nav_command.nav_mode,
                    "corridor_confidence": nav_command.corridor_confidence,
                    "curve_confidence": nav_command.curve_confidence,
                    "preview_heading_deg": nav_command.preview_heading_deg,
                    "corridor_curvature_sign": nav_command.corridor_curvature_sign,
                    "corridor_curvature_confidence": (
                        nav_command.corridor_curvature_confidence
                    ),
                    "committed_turn_sign": nav_command.committed_turn_sign,
                    "gate_curve_sign": nav_command.gate_curve_sign,
                    "curve_capture_active": nav_command.curve_capture_active,
                    "curve_capture_reason": nav_command.curve_capture_reason,
                    "curve_severity_score": nav_command.curve_severity_score,
                    "curve_steering_floor_deg": nav_command.curve_steering_floor_deg,
                    "curve_speed_cap_pct": nav_command.curve_speed_cap_pct,
                    "curve_release_reason": nav_command.curve_release_reason,
                    "same_sign_trim_active": nav_command.same_sign_trim_active,
                    "free_space_candidate_heading_deg": (
                        nav_command.free_space_candidate_heading_deg
                    ),
                    "sign_veto_reason": nav_command.sign_veto_reason,
                    "straight_veto_active": nav_command.straight_veto_active,
                    "startup_adapt_active": nav_command.startup_adapt_active,
                    "curve_intent_score": nav_command.curve_intent_score,
                    "curve_intent_sign": nav_command.curve_intent_sign,
                    "curve_evidence_strength": nav_command.curve_evidence_strength,
                    "curve_decay_active": nav_command.curve_decay_active,
                    "premature_curve_veto": nav_command.premature_curve_veto,
                    "pre_curve_bias_veto": nav_command.pre_curve_bias_veto,
                    "near_wall_mode": nav_command.near_wall_mode,
                    "straight_corridor_score": nav_command.straight_corridor_score,
                    "curve_gate_open": nav_command.curve_gate_open,
                    "curve_gate_reason": nav_command.curve_gate_reason,
                    "geometry_agreement_score": nav_command.geometry_agreement_score,
                    "curve_confirm_distance_m": nav_command.curve_confirm_distance_m,
                    "curve_confirm_yaw_deg": nav_command.curve_confirm_yaw_deg,
                    "trajectory_phase": nav_command.trajectory_phase,
                    "lookahead_x_m": nav_command.lookahead_x_m,
                    "lookahead_y_m": nav_command.lookahead_y_m,
                    "signed_curvature": nav_command.signed_curvature,
                    "radius_m": nav_command.radius_m,
                    "target_speed_pct": nav_command.target_speed_pct,
                    "track_confidence": nav_command.track_confidence,
                    "state_speed_mps": nav_command.state_speed_mps,
                    "state_yaw_rate": nav_command.state_yaw_rate,
                    "slip_proxy": nav_command.slip_proxy,
                    "trajectory_flip_blocked": nav_command.trajectory_flip_blocked,
                    "pose_source": pose_diag["pose_source"],
                    "gap_heading_deg": nav_command.gap_heading_deg,
                    "front_turn_heading_deg": nav_command.front_turn_heading_deg,
                    "corridor_axis_heading_deg": nav_command.corridor_axis_heading_deg,
                    "corridor_center_heading_deg": nav_command.corridor_center_heading_deg,
                    "wall_follow_heading_deg": nav_command.wall_follow_heading_deg,
                    "wall_follow_active": nav_command.wall_follow_active,
                    "wall_follow_anchor_side": nav_command.wall_follow_anchor_side,
                    "startup_candidate_heading_deg": nav_command.startup_candidate_heading_deg,
                    "startup_candidate_source": nav_command.startup_candidate_source,
                    "startup_hold_active": nav_command.startup_hold_active,
                    "startup_latched_sign": nav_command.startup_latched_sign,
                    "startup_latch_cycles_remaining": nav_command.startup_latch_cycles_remaining,
                    "left_wall_heading_deg": nav_command.left_wall_heading_deg,
                    "right_wall_heading_deg": nav_command.right_wall_heading_deg,
                    "centering_heading_deg": nav_command.centering_heading_deg,
                    "avoidance_heading_deg": nav_command.avoidance_heading_deg,
                    "centering_weight": nav_command.centering_weight,
                    "active_heading_source": nav_command.active_heading_source,
                    "target_heading_deg": nav_command.target_heading_deg,
                    "steering_pre_servo_deg": nav_command.steering_pre_servo_deg,
                    "steering_deg": nav_command.steering_deg,
                    "speed_pct": nav_command.speed_pct,
                },
            )

    def _emit_stop_diag(self, stage: str) -> None:
        if not self._diagnostic_enabled:
            return
        payload = {
            "stage": stage,
            "commanded_speed_pct": 0.0,
            "commanded_steering_deg": 0.0,
        }
        try:
            payload.update(
                {
                    "motor_speed_pct": self._motor.get_state().get("speed_pct"),
                    "motor_pwm_dc": self._motor.get_state().get("pwm_dc"),
                    "motor_pwm_enabled": self._motor.get_pwm_state().get("enabled"),
                    "motor_pwm_duty_cycle_ns": self._motor.get_pwm_state().get("duty_cycle_ns"),
                    "motor_pwm_period_ns": self._motor.get_pwm_state().get("period_ns"),
                    "steer_requested_deg": self._steer.get_state().get("requested_deg"),
                    "steer_pwm_dc": self._steer.get_state().get("pwm_dc"),
                    "steer_pwm_enabled": self._steer.get_pwm_state().get("enabled"),
                }
            )
        except Exception:
            pass
        self._emit_diag("DIAG_STOP", payload)

    def _emit_diag_config(self) -> None:
        if not self._diagnostic_enabled:
            return
        self._emit_diag(
            "DIAG_CONFIG",
            {
                "heading_offset_deg": self._heading_offset_deg,
                "steering_center_trim_dc": self._steering_center_trim_dc,
                "steering_direction_sign": self._steering_direction_sign,
                "diagnostic_mode": self._diagnostic_mode,
                "diagnostic_fixed_speed_pct": self._diagnostic_fixed_speed_pct,
                "diagnostic_step_duration_s": self._diagnostic_step_duration_s,
                "diagnostic_log_every_n_cycles": self._diagnostic_log_every_n_cycles,
                "diagnostic_log_path": self._diagnostic_log_path,
                "diagnostic_file_flush_every_n_records": (
                    self._diagnostic_file_flush_every_n_records
                ),
                "diagnostic_overwrite_log_on_start": self._diagnostic_overwrite_log_on_start,
                "diagnostic_recon_timeout_s": self._diagnostic_recon_timeout_s,
                "diagnostic_max_recoveries": self._diagnostic_max_recoveries,
                "diagnostic_min_progress_m": self._diagnostic_min_progress_m,
                "steering_gain": self._steering_gain,
                "wall_centering_gain_deg_per_m": self._wall_centering_gain_deg_per_m,
                "wall_centering_base_weight": self._wall_centering_base_weight,
                "wall_avoid_distance_m": self._wall_avoid_distance_m,
                "wall_avoid_gain_deg_per_m": self._wall_avoid_gain_deg_per_m,
                "wall_avoid_limit_deg": self._wall_avoid_limit_deg,
                "gap_escape_heading_threshold_deg": (
                    self._navigator._gap_escape_heading_threshold_deg
                ),
                "gap_escape_release_distance_m": (
                    self._navigator._gap_escape_release_distance_m
                ),
                "gap_escape_weight": self._navigator._gap_escape_weight,
                "corridor_balance_ratio_threshold": (
                    self._navigator._corridor_balance_ratio_threshold
                ),
                "corridor_front_min_clearance_m": (
                    self._navigator._corridor_front_min_clearance_m
                ),
                "corridor_side_min_clearance_m": (
                    self._navigator._corridor_side_min_clearance_m
                ),
                "corridor_front_turn_weight": self._navigator._corridor_front_turn_weight,
                "corridor_override_margin_deg": (
                    self._navigator._corridor_override_margin_deg
                ),
                "corridor_min_heading_deg": self._navigator._corridor_min_heading_deg,
                "corridor_wall_start_deg": self._navigator._corridor_wall_start_deg,
                "corridor_wall_end_deg": self._navigator._corridor_wall_end_deg,
                "corridor_wall_min_points": self._navigator._corridor_wall_min_points,
                "wall_follow_target_distance_m": self._navigator._wall_follow_target_distance_m,
                "wall_follow_gain_deg_per_m": self._navigator._wall_follow_gain_deg_per_m,
                "wall_follow_limit_deg": self._navigator._wall_follow_limit_deg,
                "wall_follow_activation_heading_deg": (
                    self._navigator._wall_follow_activation_heading_deg
                ),
                "wall_follow_release_balance_ratio": (
                    self._navigator._wall_follow_release_balance_ratio
                ),
                "wall_follow_min_cycles": self._navigator._wall_follow_min_cycles,
                "wall_follow_max_clearance_m": self._navigator._wall_follow_max_clearance_m,
                "wall_follow_front_turn_weight": (
                    self._navigator._wall_follow_front_turn_weight
                ),
                "startup_consensus_min_heading_deg": (
                    self._navigator._startup_consensus_min_heading_deg
                ),
                "startup_valid_cycles_required": (
                    self._navigator._startup_valid_cycles_required
                ),
                "startup_gap_lockout_cycles": self._navigator._startup_gap_lockout_cycles,
                "startup_latch_cycles": self._navigator._startup_latch_cycles,
                "ambiguity_probe_speed_pct": self._navigator._ambiguity_probe_speed_pct,
                "stop_distance_m": self._stop_distance_m,
                "slow_distance_m": self._slow_distance_m,
                "state_velocity_topic": self._state_velocity_topic,
                "state_acceleration_topic": self._state_acceleration_topic,
                "state_angular_velocity_topic": self._state_angular_velocity_topic,
                "state_timeout_s": self._state_timeout_s,
                "trajectory_horizon_m": self._navigator._trajectory_horizon_m,
                "trajectory_lookahead_min_m": self._navigator._trajectory_lookahead_min_m,
                "trajectory_lookahead_max_m": self._navigator._trajectory_lookahead_max_m,
                "trajectory_curvature_slew_per_cycle": (
                    self._navigator._trajectory_curvature_slew_per_cycle
                ),
                "trajectory_track_memory_alpha": (
                    self._navigator._trajectory_track_memory_alpha
                ),
                "trajectory_exit_heading_threshold_deg": (
                    self._navigator._trajectory_exit_heading_threshold_deg
                ),
                "trajectory_curve_speed_gain": self._navigator._trajectory_curve_speed_gain,
                "trajectory_state_aux_weight": self._navigator._trajectory_state_aux_weight,
                "trajectory_min_confidence": self._navigator._trajectory_min_confidence,
                "trajectory_flip_hold_cycles": self._navigator._trajectory_flip_hold_cycles,
                "trajectory_entry_heading_threshold_deg": (
                    self._navigator._trajectory_entry_heading_threshold_deg
                ),
                "trajectory_min_radius_m": self._navigator._trajectory_min_radius_m,
                "trajectory_curve_heading_limit_deg": (
                    self._navigator._trajectory_curve_heading_limit_deg
                ),
            },
        )

    def _emit_diag(self, tag: str, payload: dict[str, Any]) -> None:
        base = {
            "phase": self._active_phase_name,
            "step": self._active_step_name,
            "cycle": self._phase_cycle,
        }
        base.update(payload)
        line = "%s %s" % (tag, json.dumps(base, sort_keys=True))
        self.get_logger().info(line)
        self._write_diagnostic_log_line(
            line,
            force_flush=tag in {"DIAG_CONFIG", "DIAG_SUMMARY", "DIAG_STOP"},
        )

    def _open_diagnostic_log_handle(self) -> Optional[TextIO]:
        if not self._diagnostic_log_path:
            return None

        try:
            os.makedirs(os.path.dirname(self._diagnostic_log_path), exist_ok=True)
            mode = "w" if self._diagnostic_overwrite_log_on_start else "a"
            handle = open(
                self._diagnostic_log_path,
                mode,
                encoding="utf-8",
                buffering=1,
            )
            return handle
        except Exception as exc:
            self.get_logger().warning(
                "Failed to open diagnostic log file %s: %s"
                % (self._diagnostic_log_path, str(exc))
            )
            return None

    def _write_diagnostic_log_line(self, line: str, *, force_flush: bool = False) -> None:
        handle = self._diagnostic_log_handle
        if handle is None:
            return

        try:
            handle.write(line + "\n")
            self._diagnostic_pending_flush_records += 1
            if (
                force_flush
                or self._diagnostic_pending_flush_records
                >= self._diagnostic_file_flush_every_n_records
            ):
                handle.flush()
                os.fsync(handle.fileno())
                self._diagnostic_pending_flush_records = 0
        except Exception as exc:
            self.get_logger().warning(
                "Failed to persist diagnostic log line to %s: %s"
                % (self._diagnostic_log_path, str(exc))
            )
            self._close_diagnostic_log_handle()

    def _close_diagnostic_log_handle(self) -> None:
        handle = self._diagnostic_log_handle
        if handle is None:
            return

        try:
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
        self._diagnostic_log_handle = None
        self._diagnostic_pending_flush_records = 0

    def _reset_lap_tracking(self) -> None:
        self._mapping_started_at = None
        self._start_pose_xy = None
        self._last_pose_xy = None
        self._path_length_m = 0.0
        self._departed_start_zone = False
        self._completed_lap = False

    def _ensure_startup_map_reset(self) -> bool:
        if self._startup_reset_completed:
            return True

        now = self.get_clock().now()

        if self._startup_reset_retry_after is not None and now < self._startup_reset_retry_after:
            return False

        if self._startup_reset_wait_started_at is None:
            self._startup_reset_wait_started_at = now

        if self._startup_reset_future is None:
            if self._reset_client is None or not self._reset_client.wait_for_service(timeout_sec=0.0):
                wait_s = (now - self._startup_reset_wait_started_at).nanoseconds * 1e-9
                if wait_s - self._startup_reset_last_wait_log_s >= 2.0:
                    self._startup_reset_last_wait_log_s = wait_s
                    self.get_logger().info(
                        "Waiting for %s before starting reconnaissance mapping"
                        % self._reset_service_name
                    )
                return False

            request = self._reset_service_type.Request()
            request.pause_new_measurements = False
            self._startup_reset_future = self._reset_client.call_async(request)
            self.get_logger().info(
                "Requesting SLAM reset on %s before reconnaissance start"
                % self._reset_service_name
            )
            return False

        if not self._startup_reset_future.done():
            return False

        exception = self._startup_reset_future.exception()
        if exception is not None:
            self.get_logger().warning("SLAM reset call failed: %s" % str(exception))
            self._startup_reset_future = None
            self._startup_reset_retry_after = now + Duration(seconds=1.0)
            return False

        response = self._startup_reset_future.result()
        if response is None or int(response.result) != 0:
            result_code = int(response.result) if response is not None else -1
            self.get_logger().warning(
                "SLAM reset returned code %d, retrying in 1s" % result_code
            )
            self._startup_reset_future = None
            self._startup_reset_retry_after = now + Duration(seconds=1.0)
            return False

        self._startup_reset_future = None
        self._startup_reset_retry_after = None
        self._startup_reset_completed = True
        self._remove_previous_saved_map_files()
        self.get_logger().info("SLAM map reset completed; reconnaissance can start")
        return True

    def _update_lap_tracking(self) -> None:
        pose = self._lookup_pose()
        if pose is None:
            return

        pose_xy = (pose[0], pose[1])
        if self._start_pose_xy is None:
            self._start_pose_xy = pose_xy
            self._last_pose_xy = pose_xy
            self.get_logger().info(
                "Map-frame start pose locked at x=%.2f y=%.2f" % (pose_xy[0], pose_xy[1])
            )
            return

        if self._last_pose_xy is not None:
            step_distance = math.hypot(
                pose_xy[0] - self._last_pose_xy[0],
                pose_xy[1] - self._last_pose_xy[1],
            )
            if 0.0 < step_distance < 1.0:
                self._path_length_m += step_distance
        self._last_pose_xy = pose_xy

        distance_to_start = math.hypot(
            pose_xy[0] - self._start_pose_xy[0],
            pose_xy[1] - self._start_pose_xy[1],
        )
        if distance_to_start >= self._lap_depart_radius_m:
            self._departed_start_zone = True

        if not self._enable_autostop_on_lap or not self._departed_start_zone:
            return

        elapsed_s = (
            (self.get_clock().now() - self._mapping_started_at).nanoseconds * 1e-9
            if self._mapping_started_at is not None
            else 0.0
        )
        if (
            elapsed_s >= self._lap_min_duration_s
            and self._path_length_m >= self._lap_min_path_length_m
            and distance_to_start <= self._lap_return_radius_m
        ):
            self._completed_lap = True
            self._apply_command(0.0, 0.0)
            self.get_logger().info(
                "Reconnaissance lap complete: elapsed=%.1fs path=%.1fm return=%.2fm"
                % (elapsed_s, self._path_length_m, distance_to_start)
            )
            if self._save_map_on_completion:
                self._start_map_save()

    def _start_map_save(self) -> None:
        if self._map_save_thread is not None and self._map_save_thread.is_alive():
            return

        self._map_save_thread = threading.Thread(target=self._save_map_worker, daemon=True)
        self._map_save_thread.start()

    def _save_map_worker(self) -> None:
        output_prefix = self._map_output_prefix
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        self.get_logger().info("Saving map to %s.[yaml|pgm]" % output_prefix)

        try:
            result = subprocess.run(
                ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", output_prefix],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            if result.returncode == 0:
                self.get_logger().info("Map saved successfully to %s" % output_prefix)
            else:
                self.get_logger().error(
                    "Map save failed (code=%d): %s"
                    % (result.returncode, (result.stderr or result.stdout).strip())
                )
        except Exception as exc:
            self.get_logger().error("Map save exception: %s" % str(exc))

    def _remove_previous_saved_map_files(self) -> None:
        if not self._clear_previous_saved_map_files:
            return

        removed_any = False
        for suffix in (".yaml", ".pgm"):
            path = f"{self._map_output_prefix}{suffix}"
            if os.path.exists(path):
                os.remove(path)
                removed_any = True

        if removed_any:
            self.get_logger().info(
                "Removed previous saved reconnaissance map artifacts for %s"
                % self._map_output_prefix
            )

    @staticmethod
    def _safe_avg(total: float, count: int) -> Optional[float]:
        if count <= 0:
            return None
        return total / float(count)

    @staticmethod
    def _duration_since(start_time: Optional[Time], now: Time) -> float:
        if start_time is None:
            return 0.0
        return (now - start_time).nanoseconds * 1e-9

    def shutdown(self) -> None:
        try:
            self._emit_stop_diag("begin")
        except Exception as exc:
            self.get_logger().error("Failed to emit DIAG_STOP begin: %s" % str(exc))

        shutdown_steps = (
            ("apply_neutral_command", lambda: self._apply_command(0.0, 0.0)),
            ("apply_neutral_command", lambda: self._apply_command(0.0, 0.0)),
            ("apply_neutral_command", lambda: self._apply_command(0.0, 0.0)),
            ("motor_hold_neutral", lambda: self._motor.hold_neutral(disable_pwm=False)),
            ("steer_center", self._steer.center),
            ("steer_stop", self._steer.stop),
        )

        for step_name, step_fn in shutdown_steps:
            try:
                step_fn()
            except Exception as exc:
                self.get_logger().error(
                    "Shutdown step %s failed: %s" % (step_name, str(exc))
                )
            if step_name == "apply_neutral_command":
                time.sleep(0.05)
            elif step_name == "steer_center":
                time.sleep(0.05)

        try:
            self._emit_stop_diag("final")
        except Exception as exc:
            self.get_logger().error("Failed to emit DIAG_STOP final: %s" % str(exc))
        finally:
            self._close_diagnostic_log_handle()


def main() -> None:
    rclpy.init()
    node = ReconMappingNode()
    shutdown_requested = False

    def _handle_shutdown_signal(signum, _frame) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        try:
            node.get_logger().warning(
                "Received signal %d, stopping actuators and shutting down" % int(signum)
            )
        except Exception:
            pass
        try:
            node.shutdown()
        except Exception as exc:
            try:
                node.get_logger().error("Shutdown exception: %s" % str(exc))
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown_requested:
            node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
