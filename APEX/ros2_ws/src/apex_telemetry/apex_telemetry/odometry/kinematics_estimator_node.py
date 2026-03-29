#!/usr/bin/env python3
"""Integrate IMU acceleration + gyro into kinematics for APEX telemetry."""

from __future__ import annotations

import json
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger


GRAVITY_MPS2 = 9.80665


def _rpy_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[float, float, float, float]:
    half_roll = 0.5 * roll_rad
    half_pitch = 0.5 * pitch_rad
    half_yaw = 0.5 * yaw_rad
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        (sr * cp * cy) - (cr * sp * sy),
        (cr * sp * cy) + (sr * cp * sy),
        (cr * cp * sy) - (sr * sp * cy),
        (cr * cp * cy) + (sr * sp * sy),
    )


class KinematicsEstimatorNode(Node):
    """Compute kinematics from IMU acceleration and yaw rate."""

    def __init__(self) -> None:
        super().__init__("kinematics_estimator_node")

        self.declare_parameter("input_topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("gyro_input_topic", "/apex/imu/angular_velocity/raw")
        self.declare_parameter("acceleration_topic", "/apex/kinematics/acceleration")
        self.declare_parameter("velocity_topic", "/apex/kinematics/velocity")
        self.declare_parameter("position_topic", "/apex/kinematics/position")
        self.declare_parameter("angular_velocity_topic", "/apex/kinematics/angular_velocity")
        self.declare_parameter("heading_topic", "/apex/kinematics/heading")
        self.declare_parameter("corrected_imu_topic", "/apex/imu/data_corrected")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("use_message_time", True)
        self.declare_parameter("max_dt_s", 0.1)
        self.declare_parameter("accel_low_pass_alpha", 0.35)
        self.declare_parameter("deadband_mps2", 0.03)
        self.declare_parameter("gyro_low_pass_alpha", 0.45)
        self.declare_parameter("gyro_deadband_rps", 0.005)
        self.declare_parameter("integrate_accel_in_world_frame", True)
        self.declare_parameter("use_gyro_yaw", True)
        self.declare_parameter("gravity_compensation_enabled", True)
        self.declare_parameter("gravity_axis", "z")
        self.declare_parameter("planar_mount_yaw_deg", 0.0)
        self.declare_parameter("accel_bias_x", 0.0)
        self.declare_parameter("accel_bias_y", 0.0)
        self.declare_parameter("accel_bias_z", 0.0)
        self.declare_parameter("gyro_bias_x", 0.0)
        self.declare_parameter("gyro_bias_y", 0.0)
        self.declare_parameter("gyro_bias_z", 0.0)
        self.declare_parameter("status_topic", "/apex/kinematics/status")
        self.declare_parameter("startup_static_calibration_enabled", True)
        self.declare_parameter("startup_static_calibration_duration_s", 2.5)
        self.declare_parameter("calibration_accel_stddev_threshold_mps2", 0.08)
        self.declare_parameter("calibration_gyro_stddev_threshold_rps", 0.015)
        self.declare_parameter("calibration_gravity_norm_tolerance_mps2", 0.35)
        self.declare_parameter("calibration_best_effort_enabled", True)
        self.declare_parameter("calibration_max_total_duration_s", 6.0)
        self.declare_parameter("calibration_fallback_accel_stddev_threshold_mps2", 0.22)
        self.declare_parameter("calibration_fallback_gyro_stddev_threshold_rps", 0.035)
        self.declare_parameter("calibration_fallback_gravity_norm_tolerance_mps2", 0.80)
        self.declare_parameter("stationary_accel_threshold_mps2", 0.10)
        self.declare_parameter("stationary_gyro_threshold_rps", 0.02)
        self.declare_parameter("stationary_speed_threshold_mps", 0.05)
        self.declare_parameter("stationary_hold_s", 0.35)
        self.declare_parameter("stationary_release_accel_multiplier", 1.10)
        self.declare_parameter("stationary_release_speed_multiplier", 1.50)
        self.declare_parameter("stationary_release_gyro_multiplier", 1.25)
        self.declare_parameter("zero_velocity_update_enabled", True)
        self.declare_parameter("velocity_decay_enabled", True)
        self.declare_parameter("velocity_decay_tau_s", 0.8)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._use_message_time = bool(self.get_parameter("use_message_time").value)
        self._max_dt_s = max(0.001, float(self.get_parameter("max_dt_s").value))
        self._accel_alpha = min(
            1.0, max(0.0, float(self.get_parameter("accel_low_pass_alpha").value))
        )
        self._accel_deadband = max(0.0, float(self.get_parameter("deadband_mps2").value))
        self._gyro_alpha = min(
            1.0, max(0.0, float(self.get_parameter("gyro_low_pass_alpha").value))
        )
        self._gyro_deadband = max(0.0, float(self.get_parameter("gyro_deadband_rps").value))
        self._integrate_accel_world = bool(self.get_parameter("integrate_accel_in_world_frame").value)
        self._use_gyro_yaw = bool(self.get_parameter("use_gyro_yaw").value)
        self._gravity_enabled = bool(self.get_parameter("gravity_compensation_enabled").value)
        self._gravity_axis = str(self.get_parameter("gravity_axis").value).lower()
        self._planar_mount_yaw_deg = float(self.get_parameter("planar_mount_yaw_deg").value)
        self._planar_mount_yaw_rad = math.radians(self._planar_mount_yaw_deg)
        self._bias_x = float(self.get_parameter("accel_bias_x").value)
        self._bias_y = float(self.get_parameter("accel_bias_y").value)
        self._bias_z = float(self.get_parameter("accel_bias_z").value)
        self._gyro_bias_x = float(self.get_parameter("gyro_bias_x").value)
        self._gyro_bias_y = float(self.get_parameter("gyro_bias_y").value)
        self._gyro_bias_z = float(self.get_parameter("gyro_bias_z").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._calibration_accel_stddev_threshold_mps2 = max(
            0.0, float(self.get_parameter("calibration_accel_stddev_threshold_mps2").value)
        )
        self._calibration_gyro_stddev_threshold_rps = max(
            0.0, float(self.get_parameter("calibration_gyro_stddev_threshold_rps").value)
        )
        self._calibration_gravity_norm_tolerance_mps2 = max(
            0.0, float(self.get_parameter("calibration_gravity_norm_tolerance_mps2").value)
        )
        self._startup_static_calibration_enabled = bool(
            self.get_parameter("startup_static_calibration_enabled").value
        )
        self._startup_static_calibration_duration_s = max(
            0.0, float(self.get_parameter("startup_static_calibration_duration_s").value)
        )
        self._calibration_best_effort_enabled = bool(
            self.get_parameter("calibration_best_effort_enabled").value
        )
        self._calibration_max_total_duration_s = max(
            self._startup_static_calibration_duration_s,
            float(self.get_parameter("calibration_max_total_duration_s").value),
        )
        self._calibration_fallback_accel_stddev_threshold_mps2 = max(
            self._calibration_accel_stddev_threshold_mps2,
            float(self.get_parameter("calibration_fallback_accel_stddev_threshold_mps2").value),
        )
        self._calibration_fallback_gyro_stddev_threshold_rps = max(
            self._calibration_gyro_stddev_threshold_rps,
            float(self.get_parameter("calibration_fallback_gyro_stddev_threshold_rps").value),
        )
        self._calibration_fallback_gravity_norm_tolerance_mps2 = max(
            self._calibration_gravity_norm_tolerance_mps2,
            float(self.get_parameter("calibration_fallback_gravity_norm_tolerance_mps2").value),
        )
        self._stationary_accel_threshold_mps2 = max(
            0.0, float(self.get_parameter("stationary_accel_threshold_mps2").value)
        )
        self._stationary_gyro_threshold_rps = max(
            0.0, float(self.get_parameter("stationary_gyro_threshold_rps").value)
        )
        self._stationary_speed_threshold_mps = max(
            0.0, float(self.get_parameter("stationary_speed_threshold_mps").value)
        )
        self._stationary_hold_s = max(0.0, float(self.get_parameter("stationary_hold_s").value))
        self._stationary_release_accel_multiplier = max(
            1.0, float(self.get_parameter("stationary_release_accel_multiplier").value)
        )
        self._stationary_release_speed_multiplier = max(
            1.0, float(self.get_parameter("stationary_release_speed_multiplier").value)
        )
        self._stationary_release_gyro_multiplier = max(
            1.0, float(self.get_parameter("stationary_release_gyro_multiplier").value)
        )
        self._zero_velocity_update_enabled = bool(
            self.get_parameter("zero_velocity_update_enabled").value
        )
        self._velocity_decay_enabled = bool(self.get_parameter("velocity_decay_enabled").value)
        self._velocity_decay_tau_s = max(0.05, float(self.get_parameter("velocity_decay_tau_s").value))

        input_topic = str(self.get_parameter("input_topic").value)
        gyro_input_topic = str(self.get_parameter("gyro_input_topic").value)
        accel_topic = str(self.get_parameter("acceleration_topic").value)
        velocity_topic = str(self.get_parameter("velocity_topic").value)
        position_topic = str(self.get_parameter("position_topic").value)
        angular_velocity_topic = str(self.get_parameter("angular_velocity_topic").value)
        heading_topic = str(self.get_parameter("heading_topic").value)
        corrected_imu_topic = str(self.get_parameter("corrected_imu_topic").value)

        self._sub = self.create_subscription(Vector3Stamped, input_topic, self._accel_callback, 50)
        self._gyro_sub = self.create_subscription(
            Vector3Stamped, gyro_input_topic, self._gyro_callback, 50
        )
        self._pub_accel = self.create_publisher(Vector3Stamped, accel_topic, 20)
        self._pub_vel = self.create_publisher(Vector3Stamped, velocity_topic, 20)
        self._pub_pos = self.create_publisher(PointStamped, position_topic, 20)
        self._pub_ang_vel = self.create_publisher(Vector3Stamped, angular_velocity_topic, 20)
        self._pub_heading = self.create_publisher(Vector3Stamped, heading_topic, 20)
        self._pub_corrected_imu = self.create_publisher(Imu, corrected_imu_topic, 20)
        self._pub_status = self.create_publisher(String, self._status_topic, 20)
        self._reset_srv = self.create_service(Trigger, "reset_kinematics", self._handle_reset)
        self._recalibrate_srv = self.create_service(
            Trigger,
            "recalibrate_kinematics_static",
            self._handle_recalibrate,
        )

        self._last_time: Optional[Time] = None
        self._last_gyro_time: Optional[Time] = None
        self._f_ax = 0.0
        self._f_ay = 0.0
        self._f_az = 0.0
        self._f_gx = 0.0
        self._f_gy = 0.0
        self._f_gz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._calibration_active = False
        self._calibration_complete = not self._startup_static_calibration_enabled
        self._calibration_start_time: Optional[Time] = None
        self._calibration_overall_start_time: Optional[Time] = None
        self._calibration_accel_sum = [0.0, 0.0, 0.0]
        self._calibration_gyro_sum = [0.0, 0.0, 0.0]
        self._calibration_accel_sum_sq = [0.0, 0.0, 0.0]
        self._calibration_gyro_sum_sq = [0.0, 0.0, 0.0]
        self._calibration_accel_count = 0
        self._calibration_gyro_count = 0
        self._calibration_accel_stddev_mps2 = 0.0
        self._calibration_gyro_stddev_rps = 0.0
        self._calibration_retry_count = 0
        self._calibration_best_effort_used = False
        self._level_roll_rad = 0.0
        self._level_pitch_rad = 0.0
        self._session_accel_bias_x = 0.0
        self._session_accel_bias_y = 0.0
        self._session_accel_bias_z = 0.0
        self._session_gyro_bias_x = 0.0
        self._session_gyro_bias_y = 0.0
        self._session_gyro_bias_z = 0.0
        self._session_gravity_mps2 = GRAVITY_MPS2
        self._raw_accel_planar_mps2 = 0.0
        self._corrected_accel_planar_mps2 = 0.0
        self._corr_ax_body = 0.0
        self._corr_ay_body = 0.0
        self._corr_az_body = 0.0
        self._stationary_detected = False
        self._stationary_candidate_started_at: Optional[Time] = None
        self._zupt_applied = False
        self._velocity_decay_active = False
        self._odom_translation_confidence = 0.0
        self._nonfinite_warn_count = 0

        if self._startup_static_calibration_enabled:
            self._begin_static_calibration("startup")

        self.get_logger().info(
            "KinematicsEstimatorNode started (accel_in=%s gyro_in=%s accel=%s vel=%s pos=%s heading=%s status=%s planar_mount_yaw=%.1f deg)"
            % (
                input_topic,
                gyro_input_topic,
                accel_topic,
                velocity_topic,
                position_topic,
                heading_topic,
                self._status_topic,
                self._planar_mount_yaw_deg,
            )
        )

    def _apply_accel_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self._accel_deadband else value

    def _apply_gyro_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self._gyro_deadband else value

    def _begin_static_calibration(self, reason: str) -> None:
        preserve_overall_window = reason in {"quality_retry", "nonfinite_calibration_retry"}
        self._reset_state_vectors()
        self._calibration_active = True
        self._calibration_complete = False
        self._calibration_start_time = None
        if not preserve_overall_window:
            self._calibration_overall_start_time = None
            self._calibration_retry_count = 0
            self._calibration_best_effort_used = False
        else:
            self._calibration_retry_count += 1
        self._calibration_accel_sum = [0.0, 0.0, 0.0]
        self._calibration_gyro_sum = [0.0, 0.0, 0.0]
        self._calibration_accel_sum_sq = [0.0, 0.0, 0.0]
        self._calibration_gyro_sum_sq = [0.0, 0.0, 0.0]
        self._calibration_accel_count = 0
        self._calibration_gyro_count = 0
        self._calibration_accel_stddev_mps2 = 0.0
        self._calibration_gyro_stddev_rps = 0.0
        self._stationary_detected = True
        self._zupt_applied = True
        self._velocity_decay_active = False
        self._odom_translation_confidence = 0.0
        self.get_logger().info(
            "Static calibration started (%s, %.2fs window, %.2fs max total, retry=%d)"
            % (
                reason,
                self._startup_static_calibration_duration_s,
                self._calibration_max_total_duration_s,
                self._calibration_retry_count,
            )
        )

    def _reset_state_vectors(self) -> None:
        self._last_time = None
        self._last_gyro_time = None
        self._f_ax = 0.0
        self._f_ay = 0.0
        self._f_az = 0.0
        self._f_gx = 0.0
        self._f_gy = 0.0
        self._f_gz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._raw_accel_planar_mps2 = 0.0
        self._corrected_accel_planar_mps2 = 0.0
        self._corr_ax_body = 0.0
        self._corr_ay_body = 0.0
        self._corr_az_body = 0.0
        self._stationary_candidate_started_at = None

    @staticmethod
    def _values_are_finite(*values: float) -> bool:
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        return float(value) if math.isfinite(value) else 0.0

    def _warn_nonfinite(self, context: str, *values: float) -> None:
        self._nonfinite_warn_count += 1
        if self._nonfinite_warn_count <= 5 or self._nonfinite_warn_count % 25 == 0:
            formatted = ", ".join("nan" if math.isnan(v) else str(v) for v in values)
            self.get_logger().warning(
                "Non-finite IMU kinematics state in %s; resetting dynamic state (%s)"
                % (context, formatted)
            )

    def _reset_dynamic_state_after_nonfinite(self, context: str, *values: float) -> None:
        self._warn_nonfinite(context, *values)
        self._reset_state_vectors()
        self._stationary_detected = False
        self._zupt_applied = False
        self._velocity_decay_active = False
        self._odom_translation_confidence = 0.0

    def _state_is_finite(self) -> bool:
        return self._values_are_finite(
            self._f_ax,
            self._f_ay,
            self._f_az,
            self._f_gx,
            self._f_gy,
            self._f_gz,
            self._vx,
            self._vy,
            self._vz,
            self._px,
            self._py,
            self._pz,
            self._yaw,
            self._yaw_rate,
            self._raw_accel_planar_mps2,
            self._corrected_accel_planar_mps2,
            self._session_gravity_mps2,
            self._level_roll_rad,
            self._level_pitch_rad,
        )

    @staticmethod
    def _rotate_to_level(
        ax: float,
        ay: float,
        az: float,
        roll_rad: float,
        pitch_rad: float,
    ) -> tuple[float, float, float]:
        sin_roll = math.sin(roll_rad)
        cos_roll = math.cos(roll_rad)
        sin_pitch = math.sin(pitch_rad)
        cos_pitch = math.cos(pitch_rad)
        leveled_x = (cos_pitch * ax) + (sin_pitch * sin_roll * ay) + (sin_pitch * cos_roll * az)
        leveled_y = (cos_roll * ay) - (sin_roll * az)
        leveled_z = (-sin_pitch * ax) + (cos_pitch * sin_roll * ay) + (cos_pitch * cos_roll * az)
        return float(leveled_x), float(leveled_y), float(leveled_z)

    def _maybe_finish_static_calibration(self, now_t: Time) -> bool:
        if not self._calibration_active:
            return False
        if self._calibration_start_time is None:
            return False
        if self._calibration_overall_start_time is None:
            self._calibration_overall_start_time = self._calibration_start_time
        elapsed_s = (now_t - self._calibration_start_time).nanoseconds * 1e-9
        total_elapsed_s = (now_t - self._calibration_overall_start_time).nanoseconds * 1e-9
        if elapsed_s < self._startup_static_calibration_duration_s:
            return False
        if self._calibration_accel_count < 10 or self._calibration_gyro_count < 10:
            return False

        mean_ax = self._calibration_accel_sum[0] / float(self._calibration_accel_count)
        mean_ay = self._calibration_accel_sum[1] / float(self._calibration_accel_count)
        mean_az = self._calibration_accel_sum[2] / float(self._calibration_accel_count)
        mean_gx = self._calibration_gyro_sum[0] / float(self._calibration_gyro_count)
        mean_gy = self._calibration_gyro_sum[1] / float(self._calibration_gyro_count)
        mean_gz = self._calibration_gyro_sum[2] / float(self._calibration_gyro_count)
        accel_var = [
            max(
                0.0,
                (self._calibration_accel_sum_sq[index] / float(self._calibration_accel_count))
                - (mean_val * mean_val),
            )
            for index, mean_val in enumerate((mean_ax, mean_ay, mean_az))
        ]
        gyro_var = [
            max(
                0.0,
                (self._calibration_gyro_sum_sq[index] / float(self._calibration_gyro_count))
                - (mean_val * mean_val),
            )
            for index, mean_val in enumerate((mean_gx, mean_gy, mean_gz))
        ]
        self._calibration_accel_stddev_mps2 = float(
            math.sqrt(sum(accel_var) / float(len(accel_var)))
        )
        self._calibration_gyro_stddev_rps = float(
            math.sqrt(sum(gyro_var) / float(len(gyro_var)))
        )
        gravity_norm_mps2 = math.sqrt((mean_ax * mean_ax) + (mean_ay * mean_ay) + (mean_az * mean_az))
        strict_reject = (
            self._calibration_accel_stddev_mps2 > self._calibration_accel_stddev_threshold_mps2
            or self._calibration_gyro_stddev_rps > self._calibration_gyro_stddev_threshold_rps
            or abs(gravity_norm_mps2 - GRAVITY_MPS2) > self._calibration_gravity_norm_tolerance_mps2
        )
        fallback_ok = (
            self._calibration_accel_stddev_mps2
            <= self._calibration_fallback_accel_stddev_threshold_mps2
            and self._calibration_gyro_stddev_rps
            <= self._calibration_fallback_gyro_stddev_threshold_rps
            and abs(gravity_norm_mps2 - GRAVITY_MPS2)
            <= self._calibration_fallback_gravity_norm_tolerance_mps2
        )
        if strict_reject:
            if (
                self._calibration_best_effort_enabled
                and total_elapsed_s >= self._calibration_max_total_duration_s
                and fallback_ok
            ):
                self._calibration_best_effort_used = True
                self.get_logger().warning(
                    "Static calibration accepted in best-effort mode after %.2fs "
                    "(accel_std=%.4f gyro_std=%.5f gravity_norm=%.4f)"
                    % (
                        total_elapsed_s,
                        self._calibration_accel_stddev_mps2,
                        self._calibration_gyro_stddev_rps,
                        gravity_norm_mps2,
                    )
                )
            else:
                self.get_logger().warning(
                    "Static calibration rejected (accel_std=%.4f gyro_std=%.5f gravity_norm=%.4f total=%.2fs); retrying"
                    % (
                        self._calibration_accel_stddev_mps2,
                        self._calibration_gyro_stddev_rps,
                        gravity_norm_mps2,
                        total_elapsed_s,
                    )
                )
                self._begin_static_calibration("quality_retry")
                return False

        adjusted_ax = mean_ax - self._bias_x
        adjusted_ay = mean_ay - self._bias_y
        adjusted_az = mean_az - self._bias_z
        horizontal_norm = math.hypot(adjusted_ay, adjusted_az)
        self._level_roll_rad = math.atan2(adjusted_ay, adjusted_az)
        self._level_pitch_rad = math.atan2(-adjusted_ax, max(horizontal_norm, 1e-9))

        leveled_x, leveled_y, leveled_z = self._rotate_to_level(
            adjusted_ax,
            adjusted_ay,
            adjusted_az,
            self._level_roll_rad,
            self._level_pitch_rad,
        )
        self._session_accel_bias_x = float(leveled_x)
        self._session_accel_bias_y = float(leveled_y)
        self._session_accel_bias_z = 0.0
        self._session_gravity_mps2 = max(1e-6, float(leveled_z))
        self._session_gyro_bias_x = float(mean_gx - self._gyro_bias_x)
        self._session_gyro_bias_y = float(mean_gy - self._gyro_bias_y)
        self._session_gyro_bias_z = float(mean_gz - self._gyro_bias_z)
        if not self._values_are_finite(
            self._level_roll_rad,
            self._level_pitch_rad,
            self._session_accel_bias_x,
            self._session_accel_bias_y,
            self._session_gravity_mps2,
            self._session_gyro_bias_x,
            self._session_gyro_bias_y,
            self._session_gyro_bias_z,
        ):
            self._reset_dynamic_state_after_nonfinite(
                "static_calibration",
                self._level_roll_rad,
                self._level_pitch_rad,
                self._session_accel_bias_x,
                self._session_accel_bias_y,
                self._session_gravity_mps2,
                self._session_gyro_bias_x,
                self._session_gyro_bias_y,
                self._session_gyro_bias_z,
            )
            self._begin_static_calibration("nonfinite_calibration_retry")
            return False

        self._calibration_active = False
        self._calibration_complete = True
        self._reset_state_vectors()
        self._stationary_detected = True
        self._zupt_applied = True
        self._odom_translation_confidence = 1.0
        self.get_logger().info(
            "Static calibration complete (roll=%.2f deg pitch=%.2f deg gravity=%.4f m/s^2 gyro_z=%.5f rad/s)"
            % (
                math.degrees(self._level_roll_rad),
                math.degrees(self._level_pitch_rad),
                self._session_gravity_mps2,
                self._session_gyro_bias_z,
            )
        )
        return True

    def _calibration_collect_accel(self, ax: float, ay: float, az: float, now_t: Time) -> None:
        if not self._calibration_active:
            return
        if self._calibration_start_time is None:
            self._calibration_start_time = now_t
        if self._calibration_overall_start_time is None:
            self._calibration_overall_start_time = now_t
        self._calibration_accel_sum[0] += ax
        self._calibration_accel_sum[1] += ay
        self._calibration_accel_sum[2] += az
        self._calibration_accel_sum_sq[0] += ax * ax
        self._calibration_accel_sum_sq[1] += ay * ay
        self._calibration_accel_sum_sq[2] += az * az
        self._calibration_accel_count += 1

    def _calibration_collect_gyro(self, gx: float, gy: float, gz: float, now_t: Time) -> None:
        if not self._calibration_active:
            return
        if self._calibration_start_time is None:
            self._calibration_start_time = now_t
        if self._calibration_overall_start_time is None:
            self._calibration_overall_start_time = now_t
        self._calibration_gyro_sum[0] += gx
        self._calibration_gyro_sum[1] += gy
        self._calibration_gyro_sum[2] += gz
        self._calibration_gyro_sum_sq[0] += gx * gx
        self._calibration_gyro_sum_sq[1] += gy * gy
        self._calibration_gyro_sum_sq[2] += gz * gz
        self._calibration_gyro_count += 1

    def _correct_accel_sample(
        self,
        raw_ax: float,
        raw_ay: float,
        raw_az: float,
    ) -> tuple[float, float, float]:
        adjusted_ax = raw_ax - self._bias_x
        adjusted_ay = raw_ay - self._bias_y
        adjusted_az = raw_az - self._bias_z
        leveled_ax, leveled_ay, leveled_az = self._rotate_to_level(
            adjusted_ax,
            adjusted_ay,
            adjusted_az,
            self._level_roll_rad,
            self._level_pitch_rad,
        )
        corrected_ax = leveled_ax - self._session_accel_bias_x
        corrected_ay = leveled_ay - self._session_accel_bias_y
        corrected_az = leveled_az - self._session_accel_bias_z
        corrected_ax, corrected_ay, corrected_az = self._apply_gravity_compensation(
            corrected_ax,
            corrected_ay,
            corrected_az,
        )
        corrected_ax, corrected_ay = self._rotate_planar_mount(corrected_ax, corrected_ay)
        self._corr_ax_body = float(corrected_ax)
        self._corr_ay_body = float(corrected_ay)
        self._corr_az_body = float(corrected_az)
        return float(corrected_ax), float(corrected_ay), float(corrected_az)

    def _apply_gravity_compensation(self, ax: float, ay: float, az: float) -> tuple[float, float, float]:
        if not self._gravity_enabled:
            return ax, ay, az

        gravity_mps2 = self._session_gravity_mps2 if self._calibration_complete else GRAVITY_MPS2
        if self._gravity_axis == "x":
            ax -= gravity_mps2
        elif self._gravity_axis == "y":
            ay -= gravity_mps2
        else:
            az -= gravity_mps2
        return ax, ay, az

    def _rotate_planar_mount(self, ax: float, ay: float) -> tuple[float, float]:
        if abs(self._planar_mount_yaw_rad) <= 1e-9:
            return ax, ay
        c = math.cos(self._planar_mount_yaw_rad)
        s = math.sin(self._planar_mount_yaw_rad)
        rotated_x = (c * ax) - (s * ay)
        rotated_y = (s * ax) + (c * ay)
        return float(rotated_x), float(rotated_y)

    def _correct_gyro_sample(self, raw_gx: float, raw_gy: float, raw_gz: float) -> tuple[float, float, float]:
        gx = raw_gx - self._gyro_bias_x - self._session_gyro_bias_x
        gy = raw_gy - self._gyro_bias_y - self._session_gyro_bias_y
        gz = raw_gz - self._gyro_bias_z - self._session_gyro_bias_z
        return float(gx), float(gy), float(gz)

    def _msg_time_or_now(self, msg: Vector3Stamped) -> Time:
        if self._use_message_time and (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0):
            return Time.from_msg(msg.header.stamp)
        return self.get_clock().now()

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _rotate_body_to_world(self, ax: float, ay: float) -> tuple[float, float]:
        if not self._integrate_accel_world:
            return ax, ay

        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        return (c * ax - s * ay, s * ax + c * ay)

    def _update_stationary_state(
        self,
        now_t: Time,
        filtered_planar_accel_mps2: float,
        corrected_planar_accel_mps2: float,
    ) -> None:
        if not self._values_are_finite(
            filtered_planar_accel_mps2,
            corrected_planar_accel_mps2,
            self._vx,
            self._vy,
            self._yaw_rate,
        ):
            self._reset_dynamic_state_after_nonfinite(
                "stationary_state",
                filtered_planar_accel_mps2,
                corrected_planar_accel_mps2,
                self._vx,
                self._vy,
                self._yaw_rate,
            )
            return
        speed_xy = math.hypot(self._vx, self._vy)
        motion_accel_mps2 = max(filtered_planar_accel_mps2, corrected_planar_accel_mps2)
        enter_motionless = (
            filtered_planar_accel_mps2 <= self._stationary_accel_threshold_mps2
            and abs(self._yaw_rate) <= self._stationary_gyro_threshold_rps
            and speed_xy <= self._stationary_speed_threshold_mps
        )
        release_motionless = (
            motion_accel_mps2
            > (self._stationary_accel_threshold_mps2 * self._stationary_release_accel_multiplier)
            or abs(self._yaw_rate)
            > (self._stationary_gyro_threshold_rps * self._stationary_release_gyro_multiplier)
            or speed_xy
            > (self._stationary_speed_threshold_mps * self._stationary_release_speed_multiplier)
        )

        if self._stationary_detected:
            if release_motionless:
                self._stationary_detected = False
                self._stationary_candidate_started_at = None
            return

        if enter_motionless:
            if self._stationary_candidate_started_at is None:
                self._stationary_candidate_started_at = now_t
                return
            held_s = (now_t - self._stationary_candidate_started_at).nanoseconds * 1e-9
            if held_s >= self._stationary_hold_s:
                self._stationary_detected = True
            return

        self._stationary_candidate_started_at = None

    def _should_decay_velocity(self, planar_accel_mps2: float) -> bool:
        if not self._values_are_finite(planar_accel_mps2, self._vx, self._vy, self._yaw_rate):
            self._reset_dynamic_state_after_nonfinite(
                "velocity_decay_gate",
                planar_accel_mps2,
                self._vx,
                self._vy,
                self._yaw_rate,
            )
            return False
        if not self._velocity_decay_enabled:
            return False
        if self._stationary_detected:
            return False
        speed_xy = math.hypot(self._vx, self._vy)
        if speed_xy <= 1e-6:
            return False
        if speed_xy > (self._stationary_speed_threshold_mps * 3.0):
            return False
        if planar_accel_mps2 > (self._stationary_accel_threshold_mps2 * 0.75):
            return False
        if abs(self._yaw_rate) > (self._stationary_gyro_threshold_rps * 1.50):
            return False
        return True

    def _apply_velocity_decay(self, dt: float) -> None:
        if not self._values_are_finite(dt, self._vx, self._vy, self._vz):
            self._reset_dynamic_state_after_nonfinite("velocity_decay", dt, self._vx, self._vy, self._vz)
            return
        decay = math.exp(-dt / self._velocity_decay_tau_s)
        self._vx *= decay
        self._vy *= decay
        self._vz *= decay
        if math.hypot(self._vx, self._vy) < (self._stationary_speed_threshold_mps * 0.25):
            self._vx = 0.0
            self._vy = 0.0
        self._velocity_decay_active = True

    def _update_translation_confidence(self) -> None:
        if not self._calibration_complete or self._calibration_active:
            self._odom_translation_confidence = 0.0
            return
        if not self._values_are_finite(
            self._corrected_accel_planar_mps2,
            self._vx,
            self._vy,
            self._yaw_rate,
        ):
            self._reset_dynamic_state_after_nonfinite(
                "translation_confidence",
                self._corrected_accel_planar_mps2,
                self._vx,
                self._vy,
                self._yaw_rate,
            )
            self._odom_translation_confidence = 0.0
            return
        if self._stationary_detected:
            self._odom_translation_confidence = 1.0
            return
        confidence = 1.0
        if self._corrected_accel_planar_mps2 < self._stationary_accel_threshold_mps2:
            confidence *= 0.55
        if self._velocity_decay_active:
            confidence *= 0.55
        speed_xy = math.hypot(self._vx, self._vy)
        if speed_xy <= self._stationary_speed_threshold_mps:
            confidence *= 0.75
        self._odom_translation_confidence = max(0.0, min(1.0, confidence))

    def _publish_status(self, stamp_time: Time) -> None:
        payload = {
            "calibration_active": self._calibration_active,
            "calibration_complete": self._calibration_complete,
            "calibration_retry_count": self._calibration_retry_count,
            "calibration_best_effort_used": self._calibration_best_effort_used,
            "stationary_detected": self._stationary_detected,
            "zupt_applied": self._zupt_applied,
            "velocity_decay_active": self._velocity_decay_active,
            "odom_translation_confidence": self._finite_or_zero(self._odom_translation_confidence),
            "raw_accel_planar_mps2": self._finite_or_zero(self._raw_accel_planar_mps2),
            "corrected_accel_planar_mps2": self._finite_or_zero(self._corrected_accel_planar_mps2),
            "speed_mps": self._finite_or_zero(math.hypot(self._vx, self._vy)),
            "yaw_rate_rps": self._finite_or_zero(self._yaw_rate),
            "level_roll_deg": self._finite_or_zero(math.degrees(self._level_roll_rad)),
            "level_pitch_deg": self._finite_or_zero(math.degrees(self._level_pitch_rad)),
            "planar_mount_yaw_deg": self._planar_mount_yaw_deg,
            "session_gravity_mps2": self._finite_or_zero(self._session_gravity_mps2),
            "effective_gyro_bias_z": self._finite_or_zero(self._gyro_bias_z + self._session_gyro_bias_z),
            "calibration_accel_stddev_mps2": self._finite_or_zero(self._calibration_accel_stddev_mps2),
            "calibration_gyro_stddev_rps": self._finite_or_zero(self._calibration_gyro_stddev_rps),
            "calibration_accel_samples": self._calibration_accel_count,
            "calibration_gyro_samples": self._calibration_gyro_count,
        }
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._pub_status.publish(msg)

    def _gyro_callback(self, msg: Vector3Stamped) -> None:
        now_t = self._msg_time_or_now(msg)

        raw_gx = float(msg.vector.x)
        raw_gy = float(msg.vector.y)
        raw_gz = float(msg.vector.z)
        if not self._values_are_finite(raw_gx, raw_gy, raw_gz):
            self._reset_dynamic_state_after_nonfinite("gyro_raw", raw_gx, raw_gy, raw_gz)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return
        self._calibration_collect_gyro(raw_gx, raw_gy, raw_gz, now_t)
        self._maybe_finish_static_calibration(now_t)
        if self._calibration_active:
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        gx, gy, gz = self._correct_gyro_sample(raw_gx, raw_gy, raw_gz)
        if not self._values_are_finite(gx, gy, gz):
            self._reset_dynamic_state_after_nonfinite("gyro_corrected", gx, gy, gz)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        if self._last_gyro_time is None:
            self._f_gx, self._f_gy, self._f_gz = gx, gy, gz
            self._yaw_rate = self._apply_gyro_deadband(gz)
            if not self._values_are_finite(self._f_gx, self._f_gy, self._f_gz, self._yaw_rate):
                self._reset_dynamic_state_after_nonfinite(
                    "gyro_initialize",
                    self._f_gx,
                    self._f_gy,
                    self._f_gz,
                    self._yaw_rate,
                )
            self._last_gyro_time = now_t
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        dt = (now_t - self._last_gyro_time).nanoseconds * 1e-9
        self._last_gyro_time = now_t
        if dt <= 0.0:
            return
        if dt > self._max_dt_s:
            dt = self._max_dt_s

        self._f_gx = self._gyro_alpha * gx + (1.0 - self._gyro_alpha) * self._f_gx
        self._f_gy = self._gyro_alpha * gy + (1.0 - self._gyro_alpha) * self._f_gy
        self._f_gz = self._gyro_alpha * gz + (1.0 - self._gyro_alpha) * self._f_gz

        self._f_gx = self._apply_gyro_deadband(self._f_gx)
        self._f_gy = self._apply_gyro_deadband(self._f_gy)
        self._f_gz = self._apply_gyro_deadband(self._f_gz)
        self._yaw_rate = self._f_gz
        if not self._values_are_finite(self._f_gx, self._f_gy, self._f_gz, self._yaw_rate):
            self._reset_dynamic_state_after_nonfinite(
                "gyro_filter",
                self._f_gx,
                self._f_gy,
                self._f_gz,
                self._yaw_rate,
            )
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        if self._use_gyro_yaw:
            self._yaw = self._normalize_angle(self._yaw + self._yaw_rate * dt)
            if not self._values_are_finite(self._yaw):
                self._reset_dynamic_state_after_nonfinite("yaw_integrate", self._yaw_rate, dt, self._yaw)
                self._publish_orientation(now_t)
                self._publish_status(now_t)
                return

        self._publish_orientation(now_t)
        self._publish_status(now_t)

    def _accel_callback(self, msg: Vector3Stamped) -> None:
        now_t = self._msg_time_or_now(msg)

        raw_ax = float(msg.vector.x)
        raw_ay = float(msg.vector.y)
        raw_az = float(msg.vector.z)
        if not self._values_are_finite(raw_ax, raw_ay, raw_az):
            self._reset_dynamic_state_after_nonfinite("accel_raw", raw_ax, raw_ay, raw_az)
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return
        self._raw_accel_planar_mps2 = math.hypot(raw_ax, raw_ay)
        self._calibration_collect_accel(raw_ax, raw_ay, raw_az, now_t)
        self._maybe_finish_static_calibration(now_t)
        if self._calibration_active:
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        ax_body, ay_body, az = self._correct_accel_sample(raw_ax, raw_ay, raw_az)
        if not self._values_are_finite(ax_body, ay_body, az):
            self._reset_dynamic_state_after_nonfinite("accel_corrected", ax_body, ay_body, az)
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return
        self._corrected_accel_planar_mps2 = math.hypot(ax_body, ay_body)
        ax, ay = self._rotate_body_to_world(ax_body, ay_body)
        if not self._values_are_finite(self._corrected_accel_planar_mps2, ax, ay):
            self._reset_dynamic_state_after_nonfinite(
                "accel_world_rotate",
                self._corrected_accel_planar_mps2,
                ax,
                ay,
            )
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        if self._last_time is None:
            self._f_ax, self._f_ay, self._f_az = ax, ay, az
            self._last_time = now_t
            if not self._values_are_finite(self._f_ax, self._f_ay, self._f_az):
                self._reset_dynamic_state_after_nonfinite(
                    "accel_initialize",
                    self._f_ax,
                    self._f_ay,
                    self._f_az,
                )
            self._update_translation_confidence()
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        dt = (now_t - self._last_time).nanoseconds * 1e-9
        self._last_time = now_t
        if dt <= 0.0:
            return
        if dt > self._max_dt_s:
            dt = self._max_dt_s

        # Low-pass filter to reduce high-frequency IMU noise before integration.
        self._f_ax = self._accel_alpha * ax + (1.0 - self._accel_alpha) * self._f_ax
        self._f_ay = self._accel_alpha * ay + (1.0 - self._accel_alpha) * self._f_ay
        self._f_az = self._accel_alpha * az + (1.0 - self._accel_alpha) * self._f_az

        self._f_ax = self._apply_accel_deadband(self._f_ax)
        self._f_ay = self._apply_accel_deadband(self._f_ay)
        self._f_az = self._apply_accel_deadband(self._f_az)
        if not self._values_are_finite(self._f_ax, self._f_ay, self._f_az):
            self._reset_dynamic_state_after_nonfinite("accel_filter", self._f_ax, self._f_ay, self._f_az)
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            self._publish_status(now_t)
            return

        filtered_planar_mps2 = math.hypot(self._f_ax, self._f_ay)
        self._zupt_applied = False
        self._velocity_decay_active = False
        self._update_stationary_state(now_t, filtered_planar_mps2, self._corrected_accel_planar_mps2)
        if self._stationary_detected and self._zero_velocity_update_enabled:
            self._vx = 0.0
            self._vy = 0.0
            self._vz = 0.0
            self._zupt_applied = True
        else:
            if self._should_decay_velocity(filtered_planar_mps2):
                self._apply_velocity_decay(dt)

            # Integrate acceleration -> velocity -> position.
            self._vx += self._f_ax * dt
            self._vy += self._f_ay * dt
            self._vz += self._f_az * dt

            self._px += self._vx * dt
            self._py += self._vy * dt
            self._pz += self._vz * dt
            if not self._state_is_finite():
                self._reset_dynamic_state_after_nonfinite(
                    "accel_integrate",
                    self._vx,
                    self._vy,
                    self._vz,
                    self._px,
                    self._py,
                    self._pz,
                    self._yaw,
                    self._yaw_rate,
                )

        self._update_translation_confidence()

        self._publish_kinematics(now_t)
        self._publish_orientation(now_t)
        self._publish_status(now_t)

    def _publish_kinematics(self, stamp_time: Time) -> None:
        stamp = stamp_time.to_msg()

        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = self._frame_id
        accel_msg.vector.x = self._finite_or_zero(self._f_ax)
        accel_msg.vector.y = self._finite_or_zero(self._f_ay)
        accel_msg.vector.z = self._finite_or_zero(self._f_az)
        self._pub_accel.publish(accel_msg)

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self._frame_id
        vel_msg.vector.x = self._finite_or_zero(self._vx)
        vel_msg.vector.y = self._finite_or_zero(self._vy)
        vel_msg.vector.z = self._finite_or_zero(self._vz)
        self._pub_vel.publish(vel_msg)

        pos_msg = PointStamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = self._frame_id
        pos_msg.point.x = self._finite_or_zero(self._px)
        pos_msg.point.y = self._finite_or_zero(self._py)
        pos_msg.point.z = self._finite_or_zero(self._pz)
        self._pub_pos.publish(pos_msg)

    def _publish_orientation(self, stamp_time: Time) -> None:
        stamp = stamp_time.to_msg()

        angular_msg = Vector3Stamped()
        angular_msg.header.stamp = stamp
        angular_msg.header.frame_id = self._frame_id
        angular_msg.vector.x = self._finite_or_zero(self._f_gx)
        angular_msg.vector.y = self._finite_or_zero(self._f_gy)
        angular_msg.vector.z = self._finite_or_zero(self._yaw_rate)
        self._pub_ang_vel.publish(angular_msg)

        heading_msg = Vector3Stamped()
        heading_msg.header.stamp = stamp
        heading_msg.header.frame_id = self._frame_id
        heading_msg.vector.x = 0.0
        heading_msg.vector.y = 0.0
        heading_msg.vector.z = self._finite_or_zero(self._yaw)
        self._pub_heading.publish(heading_msg)

        qx, qy, qz, qw = _rpy_to_quat(
            self._finite_or_zero(self._level_roll_rad),
            self._finite_or_zero(self._level_pitch_rad),
            self._finite_or_zero(self._yaw),
        )
        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self._frame_id
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.orientation_covariance[0] = 0.01
        imu_msg.orientation_covariance[4] = 0.01
        imu_msg.orientation_covariance[8] = 0.04
        imu_msg.angular_velocity.x = self._finite_or_zero(self._f_gx)
        imu_msg.angular_velocity.y = self._finite_or_zero(self._f_gy)
        imu_msg.angular_velocity.z = self._finite_or_zero(self._yaw_rate)
        imu_msg.angular_velocity_covariance[0] = 0.01
        imu_msg.angular_velocity_covariance[4] = 0.01
        imu_msg.angular_velocity_covariance[8] = 0.02
        imu_msg.linear_acceleration.x = self._finite_or_zero(self._corr_ax_body)
        imu_msg.linear_acceleration.y = self._finite_or_zero(self._corr_ay_body)
        imu_msg.linear_acceleration.z = self._finite_or_zero(self._corr_az_body)
        imu_msg.linear_acceleration_covariance[0] = 0.02
        imu_msg.linear_acceleration_covariance[4] = 0.02
        imu_msg.linear_acceleration_covariance[8] = 0.04
        self._pub_corrected_imu.publish(imu_msg)

    def _handle_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._reset_state_vectors()
        if self._startup_static_calibration_enabled:
            self._begin_static_calibration("reset_service")
            response.message = "Kinematic state reset; static calibration restarted."
        else:
            self._stationary_detected = False
            self._zupt_applied = False
            self._velocity_decay_active = False
            self._odom_translation_confidence = 0.0
            response.message = "Kinematic state reset."
        response.success = True
        self.get_logger().info(response.message)
        return response

    def _handle_recalibrate(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self._begin_static_calibration("recalibrate_service")
        response.success = True
        response.message = "Static recalibration started."
        self.get_logger().info(response.message)
        return response


def main() -> None:
    rclpy.init()
    node = KinematicsEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
