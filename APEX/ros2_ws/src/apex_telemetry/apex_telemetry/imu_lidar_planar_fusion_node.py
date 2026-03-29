#!/usr/bin/env python3
"""Planar LiDAR + IMU odometry fusion with explicit drift correction."""

from __future__ import annotations

import json
import math
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _rotate_planar(x: float, y: float, yaw_rad: float) -> tuple[float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return ((c * x) - (s * y), (s * x) + (c * y))


def _stamp_to_time(node: Node, stamp) -> Time:
    if stamp.sec != 0 or stamp.nanosec != 0:
        return Time.from_msg(stamp)
    return node.get_clock().now()


def _stamp_key(stamp) -> int:
    return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)


class ImuLidarPlanarFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_lidar_planar_fusion_node")

        self.declare_parameter("accel_topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("gyro_topic", "/apex/imu/angular_velocity/raw")
        self.declare_parameter("lidar_relative_odom_topic", "/apex/lidar/relative_odom")
        self.declare_parameter("status_topic", "/apex/kinematics/status")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("fusion_status_topic", "/apex/odometry/fusion_status")
        self.declare_parameter("odom_frame_id", "odom_lidar_local")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("use_message_time", True)
        self.declare_parameter("accel_input_is_world_frame", True)
        self.declare_parameter("gyro_input_is_corrected", True)
        self.declare_parameter("planar_mount_yaw_deg", -90.0)
        self.declare_parameter("accel_bias_x", 0.0)
        self.declare_parameter("accel_bias_y", 0.0)
        self.declare_parameter("gyro_bias_z", 0.0)
        self.declare_parameter("max_dt_s", 0.05)
        self.declare_parameter("max_planar_accel_mps2", 4.5)
        self.declare_parameter("max_speed_mps", 2.5)
        self.declare_parameter("velocity_leak_tau_s", 4.0)
        self.declare_parameter("stationary_zero_gain", 8.0)
        self.declare_parameter("stationary_bias_alpha", 0.05)
        self.declare_parameter("launch_suppression_s", 0.25)
        self.declare_parameter("launch_min_accel_scale", 0.25)
        self.declare_parameter("launch_speed_cap_mps", 0.65)
        self.declare_parameter("lidar_pose_timeout_s", 0.35)
        self.declare_parameter("status_timeout_s", 0.75)
        self.declare_parameter("status_motion_release_accel_mps2", 0.10)
        self.declare_parameter("status_motion_release_speed_mps", 0.05)
        self.declare_parameter("status_motion_release_yaw_rate_rps", 0.03)
        self.declare_parameter("lidar_position_gain", 0.35)
        self.declare_parameter("lidar_yaw_gain", 0.75)
        self.declare_parameter("lidar_velocity_gain", 0.8)
        self.declare_parameter("lidar_min_translation_delta_m", 0.01)
        self.declare_parameter("lidar_min_translation_speed_mps", 0.06)
        self.declare_parameter("lidar_reacquire_ramp_s", 0.40)
        self.declare_parameter("lidar_reacquire_position_gain_start", 0.12)
        self.declare_parameter("lidar_reacquire_velocity_gain_start", 0.20)
        self.declare_parameter("innovation_bias_gain", 0.10)
        self.declare_parameter("gyro_bias_gain", 0.08)
        self.declare_parameter("lidar_relative_valid_quality_min", 0.08)
        self.declare_parameter("lidar_relative_nominal_covariance_x_m2", 0.01)
        self.declare_parameter("lidar_relative_nominal_covariance_y_m2", 0.01)
        self.declare_parameter("lidar_relative_nominal_yaw_covariance_rad2", 0.04)
        self.declare_parameter("base_pose_covariance_x_m2", 0.02)
        self.declare_parameter("base_pose_covariance_y_m2", 0.02)
        self.declare_parameter("base_yaw_covariance_rad2", 0.03)
        self.declare_parameter("base_twist_covariance_xy_m2ps2", 0.20)
        self.declare_parameter("stale_covariance_scale", 25.0)

        self._accel_topic = str(self.get_parameter("accel_topic").value)
        self._gyro_topic = str(self.get_parameter("gyro_topic").value)
        self._lidar_relative_odom_topic = str(self.get_parameter("lidar_relative_odom_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._fusion_status_topic = str(self.get_parameter("fusion_status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._publish_rate_hz = max(5.0, float(self.get_parameter("publish_rate_hz").value))
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._use_message_time = bool(self.get_parameter("use_message_time").value)
        self._accel_input_is_world_frame = bool(
            self.get_parameter("accel_input_is_world_frame").value
        )
        self._gyro_input_is_corrected = bool(
            self.get_parameter("gyro_input_is_corrected").value
        )
        self._planar_mount_yaw_rad = math.radians(
            float(self.get_parameter("planar_mount_yaw_deg").value)
        )
        self._base_accel_bias_x = float(self.get_parameter("accel_bias_x").value)
        self._base_accel_bias_y = float(self.get_parameter("accel_bias_y").value)
        self._base_gyro_bias_z = float(self.get_parameter("gyro_bias_z").value)
        self._max_dt_s = max(1e-3, float(self.get_parameter("max_dt_s").value))
        self._max_planar_accel = max(0.1, float(self.get_parameter("max_planar_accel_mps2").value))
        self._max_speed_mps = max(0.1, float(self.get_parameter("max_speed_mps").value))
        self._velocity_leak_tau_s = max(0.0, float(self.get_parameter("velocity_leak_tau_s").value))
        self._stationary_zero_gain = max(0.0, float(self.get_parameter("stationary_zero_gain").value))
        self._stationary_bias_alpha = _clamp(
            float(self.get_parameter("stationary_bias_alpha").value), 0.0, 1.0
        )
        self._launch_suppression_s = max(
            0.0, float(self.get_parameter("launch_suppression_s").value)
        )
        self._launch_min_accel_scale = _clamp(
            float(self.get_parameter("launch_min_accel_scale").value), 0.0, 1.0
        )
        self._launch_speed_cap_mps = max(
            0.0, float(self.get_parameter("launch_speed_cap_mps").value)
        )
        self._lidar_pose_timeout_s = max(
            0.05, float(self.get_parameter("lidar_pose_timeout_s").value)
        )
        self._status_timeout_s = max(0.1, float(self.get_parameter("status_timeout_s").value))
        self._status_motion_release_accel_mps2 = max(
            0.0, float(self.get_parameter("status_motion_release_accel_mps2").value)
        )
        self._status_motion_release_speed_mps = max(
            0.0, float(self.get_parameter("status_motion_release_speed_mps").value)
        )
        self._status_motion_release_yaw_rate_rps = max(
            0.0, float(self.get_parameter("status_motion_release_yaw_rate_rps").value)
        )
        self._lidar_position_gain = _clamp(
            float(self.get_parameter("lidar_position_gain").value), 0.0, 1.0
        )
        self._lidar_yaw_gain = _clamp(
            float(self.get_parameter("lidar_yaw_gain").value), 0.0, 1.0
        )
        self._lidar_velocity_gain = _clamp(
            float(self.get_parameter("lidar_velocity_gain").value), 0.0, 1.0
        )
        self._lidar_min_translation_delta_m = max(
            0.0, float(self.get_parameter("lidar_min_translation_delta_m").value)
        )
        self._lidar_min_translation_speed_mps = max(
            0.0, float(self.get_parameter("lidar_min_translation_speed_mps").value)
        )
        self._lidar_reacquire_ramp_s = max(
            0.0, float(self.get_parameter("lidar_reacquire_ramp_s").value)
        )
        self._lidar_reacquire_position_gain_start = _clamp(
            float(self.get_parameter("lidar_reacquire_position_gain_start").value), 0.0, 1.0
        )
        self._lidar_reacquire_velocity_gain_start = _clamp(
            float(self.get_parameter("lidar_reacquire_velocity_gain_start").value), 0.0, 1.0
        )
        self._innovation_bias_gain = max(
            0.0, float(self.get_parameter("innovation_bias_gain").value)
        )
        self._gyro_bias_gain = max(0.0, float(self.get_parameter("gyro_bias_gain").value))
        self._lidar_relative_valid_quality_min = _clamp(
            float(self.get_parameter("lidar_relative_valid_quality_min").value), 0.0, 1.0
        )
        self._lidar_relative_nominal_cov_x = max(
            1e-6, float(self.get_parameter("lidar_relative_nominal_covariance_x_m2").value)
        )
        self._lidar_relative_nominal_cov_y = max(
            1e-6, float(self.get_parameter("lidar_relative_nominal_covariance_y_m2").value)
        )
        self._lidar_relative_nominal_yaw_cov = max(
            1e-6, float(self.get_parameter("lidar_relative_nominal_yaw_covariance_rad2").value)
        )
        self._base_pose_cov_x = max(
            1e-6, float(self.get_parameter("base_pose_covariance_x_m2").value)
        )
        self._base_pose_cov_y = max(
            1e-6, float(self.get_parameter("base_pose_covariance_y_m2").value)
        )
        self._base_yaw_cov = max(
            1e-6, float(self.get_parameter("base_yaw_covariance_rad2").value)
        )
        self._base_twist_cov = max(
            1e-6, float(self.get_parameter("base_twist_covariance_xy_m2ps2").value)
        )
        self._stale_cov_scale = max(
            1.0, float(self.get_parameter("stale_covariance_scale").value)
        )

        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._bax = 0.0
        self._bay = 0.0
        self._bgz = 0.0
        self._calibration_complete = False
        self._calibration_active = False
        self._stationary_detected = False
        self._velocity_decay_active = False
        self._odom_translation_confidence = 0.0
        self._status_corrected_accel_planar_mps2 = 0.0
        self._status_speed_mps = 0.0
        self._status_yaw_rate_rps = 0.0
        self._last_status_payload: Dict[str, object] = {}
        self._last_status_receipt: Optional[Time] = None
        self._launch_started_at: Optional[Time] = None
        self._last_imu_time: Optional[Time] = None
        self._latest_imu_receipt: Optional[Time] = None
        self._last_lidar_pose: Optional[Tuple[float, float, float, Time]] = None
        self._last_lidar_receipt: Optional[Time] = None
        self._last_lidar_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_lidar_innovation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_lidar_delta_m = 0.0
        self._last_lidar_delta_yaw = 0.0
        self._last_lidar_speed_mps = 0.0
        self._last_lidar_translation_observable = False
        self._lidar_position_update_suppressed = False
        self._last_lidar_position_gain = self._lidar_position_gain
        self._last_lidar_yaw_gain = self._lidar_yaw_gain
        self._last_lidar_velocity_gain = self._lidar_velocity_gain
        self._last_lidar_relative_valid = False
        self._last_lidar_relative_quality = 0.0
        self._lidar_observable_since: Optional[Time] = None
        self._imu_samples_processed = 0
        self._lidar_updates_applied = 0

        self._pending_accel: Dict[int, Tuple[float, float, Time]] = {}
        self._pending_gyro: Dict[int, Tuple[float, Time]] = {}

        self.create_subscription(Vector3Stamped, self._accel_topic, self._accel_cb, 50)
        self.create_subscription(Vector3Stamped, self._gyro_topic, self._gyro_cb, 50)
        self.create_subscription(Odometry, self._lidar_relative_odom_topic, self._lidar_odom_cb, 20)
        self.create_subscription(String, self._status_topic, self._status_cb, 20)

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._status_pub = self.create_publisher(String, self._fusion_status_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_outputs)

        self.get_logger().info(
            "ImuLidarPlanarFusionNode started (accel=%s gyro=%s lidar_relative=%s status=%s odom=%s)"
            % (
                self._accel_topic,
                self._gyro_topic,
                self._lidar_relative_odom_topic,
                self._status_topic,
                self._odom_topic,
            )
        )

    def _reset_fusion_state(self, reason: str) -> None:
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._bax = 0.0
        self._bay = 0.0
        self._bgz = 0.0
        self._velocity_decay_active = False
        self._odom_translation_confidence = 0.0
        self._launch_started_at = None
        self._last_imu_time = None
        self._latest_imu_receipt = None
        self._last_lidar_pose = None
        self._last_lidar_receipt = None
        self._last_lidar_velocity = (0.0, 0.0, 0.0)
        self._last_lidar_innovation = (0.0, 0.0, 0.0)
        self._last_lidar_delta_m = 0.0
        self._last_lidar_delta_yaw = 0.0
        self._last_lidar_speed_mps = 0.0
        self._last_lidar_translation_observable = False
        self._lidar_position_update_suppressed = False
        self._last_lidar_position_gain = self._lidar_position_gain
        self._last_lidar_yaw_gain = self._lidar_yaw_gain
        self._last_lidar_velocity_gain = self._lidar_velocity_gain
        self._last_lidar_relative_valid = False
        self._last_lidar_relative_quality = 0.0
        self._lidar_observable_since = None
        self._imu_samples_processed = 0
        self._lidar_updates_applied = 0
        self._pending_accel.clear()
        self._pending_gyro.clear()
        self.get_logger().info("Fusion state reset (%s)." % reason)

    def _status_is_fresh(self, now_t: Time) -> bool:
        if self._last_status_receipt is None:
            return False
        age_s = max(0.0, (now_t - self._last_status_receipt).nanoseconds * 1e-9)
        return age_s <= self._status_timeout_s

    def _accel_cb(self, msg: Vector3Stamped) -> None:
        stamp_key = _stamp_key(msg.header.stamp)
        sample_time = _stamp_to_time(self, msg.header.stamp) if self._use_message_time else self.get_clock().now()
        self._pending_accel[stamp_key] = (float(msg.vector.x), float(msg.vector.y), sample_time)
        self._trim_pending(self._pending_accel)
        self._consume_pending_sample(stamp_key)

    def _gyro_cb(self, msg: Vector3Stamped) -> None:
        stamp_key = _stamp_key(msg.header.stamp)
        sample_time = _stamp_to_time(self, msg.header.stamp) if self._use_message_time else self.get_clock().now()
        self._pending_gyro[stamp_key] = (float(msg.vector.z), sample_time)
        self._trim_pending(self._pending_gyro)
        self._consume_pending_sample(stamp_key)

    def _trim_pending(self, pending: Dict[int, object]) -> None:
        if len(pending) <= 16:
            return
        for stale_key in sorted(pending.keys())[:-16]:
            pending.pop(stale_key, None)

    def _consume_pending_sample(self, stamp_key: int) -> None:
        accel_sample = self._pending_accel.get(stamp_key)
        gyro_sample = self._pending_gyro.get(stamp_key)
        if accel_sample is None or gyro_sample is None:
            return

        self._pending_accel.pop(stamp_key, None)
        self._pending_gyro.pop(stamp_key, None)

        raw_ax, raw_ay, accel_time = accel_sample
        raw_gz, gyro_time = gyro_sample
        sample_time = accel_time if accel_time.nanoseconds >= gyro_time.nanoseconds else gyro_time
        self._process_imu_sample(sample_time, raw_ax, raw_ay, raw_gz)

    def _status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return

        now_t = self.get_clock().now()
        prev_stationary = self._stationary_detected
        prev_calibration_active = self._calibration_active

        self._last_status_payload = payload
        self._last_status_receipt = now_t
        self._calibration_active = bool(payload.get("calibration_active", False))
        self._calibration_complete = bool(payload.get("calibration_complete", False))
        self._stationary_detected = bool(payload.get("stationary_detected", False))
        self._velocity_decay_active = bool(payload.get("velocity_decay_active", False))
        self._odom_translation_confidence = _clamp(
            float(payload.get("odom_translation_confidence", 0.0) or 0.0), 0.0, 1.0
        )
        self._status_corrected_accel_planar_mps2 = max(
            0.0, float(payload.get("corrected_accel_planar_mps2", 0.0) or 0.0)
        )
        self._status_speed_mps = max(0.0, float(payload.get("speed_mps", 0.0) or 0.0))
        self._status_yaw_rate_rps = float(payload.get("yaw_rate_rps", 0.0) or 0.0)

        mount_yaw_deg = payload.get("planar_mount_yaw_deg")
        if mount_yaw_deg is not None:
            self._planar_mount_yaw_rad = math.radians(float(mount_yaw_deg))

        if self._calibration_active and not prev_calibration_active:
            self._reset_fusion_state("calibration_active")

        if prev_stationary and not self._stationary_detected:
            self._launch_started_at = now_t
        elif self._stationary_detected:
            self._launch_started_at = None

    def _status_motion_override_active(self, now_t: Time) -> bool:
        if not self._status_is_fresh(now_t):
            return False
        return (
            self._status_corrected_accel_planar_mps2 >= self._status_motion_release_accel_mps2
            or self._status_speed_mps >= self._status_motion_release_speed_mps
            or abs(self._status_yaw_rate_rps) >= self._status_motion_release_yaw_rate_rps
        )

    def _process_imu_sample(self, sample_time: Time, raw_ax: float, raw_ay: float, raw_gz: float) -> None:
        now_t = self.get_clock().now()
        self._latest_imu_receipt = now_t
        if not self._calibration_complete:
            self._last_imu_time = sample_time
            return

        if self._last_imu_time is None:
            self._last_imu_time = sample_time
            return

        dt = (sample_time - self._last_imu_time).nanoseconds * 1e-9
        self._last_imu_time = sample_time
        if dt <= 0.0 or dt > self._max_dt_s:
            return

        if self._accel_input_is_world_frame:
            accel_world_meas_x = raw_ax
            accel_world_meas_y = raw_ay
        else:
            adj_ax = raw_ax - self._base_accel_bias_x
            adj_ay = raw_ay - self._base_accel_bias_y
            accel_body_meas_x, accel_body_meas_y = _rotate_planar(
                adj_ax, adj_ay, self._planar_mount_yaw_rad
            )
            accel_world_meas_x, accel_world_meas_y = _rotate_planar(
                accel_body_meas_x,
                accel_body_meas_y,
                self._yaw + (0.5 * self._yaw_rate * dt),
            )
        yaw_rate_meas = raw_gz if self._gyro_input_is_corrected else (raw_gz - self._base_gyro_bias_z)

        status_motion_override = self._status_motion_override_active(now_t)
        stationary_detected = (
            self._stationary_detected
            and self._status_is_fresh(now_t)
            and not status_motion_override
        )

        if status_motion_override and self._launch_started_at is None:
            self._launch_started_at = sample_time

        if stationary_detected:
            alpha = self._stationary_bias_alpha
            self._bax = ((1.0 - alpha) * self._bax) + (alpha * accel_world_meas_x)
            self._bay = ((1.0 - alpha) * self._bay) + (alpha * accel_world_meas_y)
            self._bgz = ((1.0 - alpha) * self._bgz) + (alpha * yaw_rate_meas)
            zero_decay = max(0.0, 1.0 - (self._stationary_zero_gain * dt))
            self._vx *= zero_decay
            self._vy *= zero_decay
            if math.hypot(self._vx, self._vy) < 0.01:
                self._vx = 0.0
                self._vy = 0.0
            self._yaw_rate = 0.0
            self._imu_samples_processed += 1
            return

        accel_world_x = accel_world_meas_x - self._bax
        accel_world_y = accel_world_meas_y - self._bay
        yaw_rate = yaw_rate_meas - self._bgz

        suppression_scale = 1.0
        if self._launch_started_at is not None and self._launch_suppression_s > 0.0:
            launch_elapsed_s = max(
                0.0, (sample_time - self._launch_started_at).nanoseconds * 1e-9
            )
            if launch_elapsed_s < self._launch_suppression_s:
                progress = launch_elapsed_s / self._launch_suppression_s
                suppression_scale = self._launch_min_accel_scale + (
                    (1.0 - self._launch_min_accel_scale) * progress
                )
            else:
                self._launch_started_at = None

        accel_gain = suppression_scale
        accel_mag = math.hypot(accel_world_x, accel_world_y)
        if accel_mag > self._max_planar_accel:
            scale = self._max_planar_accel / max(accel_mag, 1e-6)
            accel_world_x *= scale
            accel_world_y *= scale
        accel_world_x *= accel_gain
        accel_world_y *= accel_gain

        self._yaw = _normalize_angle(self._yaw + (yaw_rate * dt))
        self._yaw_rate = yaw_rate
        self._vx += accel_world_x * dt
        self._vy += accel_world_y * dt

        if self._velocity_leak_tau_s > 0.0:
            leak = math.exp(-dt / self._velocity_leak_tau_s)
            self._vx *= leak
            self._vy *= leak

        speed_xy = math.hypot(self._vx, self._vy)
        speed_cap = self._max_speed_mps
        if self._launch_started_at is not None and self._launch_speed_cap_mps > 0.0:
            speed_cap = min(speed_cap, self._launch_speed_cap_mps)
        if speed_xy > speed_cap > 0.0:
            scale = speed_cap / max(speed_xy, 1e-6)
            self._vx *= scale
            self._vy *= scale

        self._x += self._vx * dt
        self._y += self._vy * dt
        self._imu_samples_processed += 1

    def _lidar_quality_from_covariance(
        self,
        cov_x_m2: float,
        cov_y_m2: float,
        cov_yaw_rad2: float,
    ) -> float:
        if not (
            math.isfinite(cov_x_m2)
            and math.isfinite(cov_y_m2)
            and math.isfinite(cov_yaw_rad2)
            and cov_x_m2 > 0.0
            and cov_y_m2 > 0.0
            and cov_yaw_rad2 > 0.0
        ):
            return 0.0
        quality_x = _clamp(self._lidar_relative_nominal_cov_x / cov_x_m2, 0.0, 1.0)
        quality_y = _clamp(self._lidar_relative_nominal_cov_y / cov_y_m2, 0.0, 1.0)
        quality_yaw = _clamp(self._lidar_relative_nominal_yaw_cov / cov_yaw_rad2, 0.0, 1.0)
        return _clamp((quality_x * quality_y * quality_yaw) ** (1.0 / 3.0), 0.0, 1.0)

    def _lidar_odom_cb(self, msg: Odometry) -> None:
        now_t = self.get_clock().now()
        pose_time = _stamp_to_time(self, msg.header.stamp) if self._use_message_time else now_t
        lidar_x = float(msg.pose.pose.position.x)
        lidar_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        lidar_yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        cov_x = float(msg.pose.covariance[0]) if len(msg.pose.covariance) >= 1 else float("inf")
        cov_y = float(msg.pose.covariance[7]) if len(msg.pose.covariance) >= 8 else float("inf")
        cov_yaw = float(msg.pose.covariance[35]) if len(msg.pose.covariance) >= 36 else float("inf")
        lidar_quality = self._lidar_quality_from_covariance(cov_x, cov_y, cov_yaw)
        lidar_valid = (
            lidar_quality >= self._lidar_relative_valid_quality_min
            and math.isfinite(lidar_x)
            and math.isfinite(lidar_y)
            and math.isfinite(lidar_yaw)
        )

        pred_x = self._x
        pred_y = self._y
        pred_yaw = self._yaw
        pred_vx = self._vx
        pred_vy = self._vy
        pred_yaw_rate = self._yaw_rate
        self._last_lidar_receipt = now_t
        self._lidar_updates_applied += 1

        innovation_x = lidar_x - pred_x
        innovation_y = lidar_y - pred_y
        innovation_yaw = _normalize_angle(lidar_yaw - pred_yaw)
        position_gain = 0.0
        yaw_gain = 0.0
        velocity_gain = 0.0
        translation_observable = False
        lidar_delta_m = 0.0
        lidar_delta_yaw = 0.0
        lidar_speed_mps = 0.0

        if self._last_lidar_pose is None:
            if lidar_valid:
                self._x = lidar_x
                self._y = lidar_y
                self._yaw = lidar_yaw
                position_gain = 1.0
                yaw_gain = 1.0
                velocity_gain = self._lidar_velocity_gain * max(lidar_quality, 0.25)
            self._last_lidar_innovation = (innovation_x, innovation_y, innovation_yaw)
            self._last_lidar_pose = (lidar_x, lidar_y, lidar_yaw, pose_time)
            self._last_lidar_translation_observable = False
            self._lidar_position_update_suppressed = not lidar_valid
            self._last_lidar_position_gain = position_gain
            self._last_lidar_yaw_gain = yaw_gain
            self._last_lidar_velocity_gain = velocity_gain
            self._last_lidar_delta_m = 0.0
            self._last_lidar_delta_yaw = 0.0
            self._last_lidar_speed_mps = 0.0
            self._last_lidar_relative_valid = lidar_valid
            self._last_lidar_relative_quality = lidar_quality
            self._lidar_observable_since = pose_time if lidar_valid else None
            return

        prev_x, prev_y, prev_yaw, prev_time = self._last_lidar_pose
        dt = (pose_time - prev_time).nanoseconds * 1e-9
        if lidar_valid and 1e-3 < dt <= 1.0:
            dx = lidar_x - prev_x
            dy = lidar_y - prev_y
            dyaw = _normalize_angle(lidar_yaw - prev_yaw)
            vx_lidar = dx / dt
            vy_lidar = dy / dt
            yaw_rate_lidar = dyaw / dt
            lidar_delta_m = math.hypot(dx, dy)
            lidar_delta_yaw = dyaw
            lidar_speed_mps = math.hypot(vx_lidar, vy_lidar)
            translation_observable = (
                lidar_delta_m >= self._lidar_min_translation_delta_m
                or lidar_speed_mps >= self._lidar_min_translation_speed_mps
            )
            self._last_lidar_velocity = (vx_lidar, vy_lidar, yaw_rate_lidar)

            if translation_observable:
                if not self._last_lidar_translation_observable or self._lidar_observable_since is None:
                    self._lidar_observable_since = pose_time
                ramp_alpha = 1.0
                if self._lidar_reacquire_ramp_s > 0.0 and self._lidar_observable_since is not None:
                    observable_dt_s = max(
                        0.0, (pose_time - self._lidar_observable_since).nanoseconds * 1e-9
                    )
                    ramp_alpha = _clamp(observable_dt_s / self._lidar_reacquire_ramp_s, 0.0, 1.0)
                position_gain = self._lidar_reacquire_position_gain_start + (
                    (self._lidar_position_gain - self._lidar_reacquire_position_gain_start) * ramp_alpha
                )
                velocity_gain = self._lidar_reacquire_velocity_gain_start + (
                    (self._lidar_velocity_gain - self._lidar_reacquire_velocity_gain_start) * ramp_alpha
                )
                quality_gain = max(lidar_quality, self._lidar_relative_valid_quality_min)
                position_gain *= quality_gain
                velocity_gain *= quality_gain
                self._vx = pred_vx + (velocity_gain * (vx_lidar - pred_vx))
                self._vy = pred_vy + (velocity_gain * (vy_lidar - pred_vy))
                vel_err_world_x = pred_vx - vx_lidar
                vel_err_world_y = pred_vy - vy_lidar
                accel_bias_limit = self._max_planar_accel * 0.5
                bias_ax_delta = _clamp(
                    vel_err_world_x / max(dt, 1e-3), -accel_bias_limit, accel_bias_limit
                )
                bias_ay_delta = _clamp(
                    vel_err_world_y / max(dt, 1e-3), -accel_bias_limit, accel_bias_limit
                )
                self._bax += self._innovation_bias_gain * bias_ax_delta
                self._bay += self._innovation_bias_gain * bias_ay_delta
                self._bax = _clamp(self._bax, -self._max_planar_accel, self._max_planar_accel)
                self._bay = _clamp(self._bay, -self._max_planar_accel, self._max_planar_accel)
                self._x = pred_x + (position_gain * innovation_x)
                self._y = pred_y + (position_gain * innovation_y)
                self._lidar_position_update_suppressed = False
            else:
                self._lidar_observable_since = None
                self._lidar_position_update_suppressed = True

            yaw_gain = self._lidar_yaw_gain * max(lidar_quality, 0.25)
            yaw_rate_err = pred_yaw_rate - yaw_rate_lidar
            self._bgz += self._gyro_bias_gain * yaw_rate_err
            self._bgz = _clamp(self._bgz, -2.0, 2.0)
        else:
            self._lidar_observable_since = None
            self._last_lidar_velocity = (0.0, 0.0, 0.0)
            self._lidar_position_update_suppressed = True

        self._yaw = _normalize_angle(pred_yaw + (yaw_gain * innovation_yaw))
        self._last_lidar_innovation = (innovation_x, innovation_y, innovation_yaw)
        self._last_lidar_pose = (lidar_x, lidar_y, lidar_yaw, pose_time)
        self._last_lidar_translation_observable = translation_observable
        self._last_lidar_delta_m = lidar_delta_m
        self._last_lidar_delta_yaw = lidar_delta_yaw
        self._last_lidar_speed_mps = lidar_speed_mps
        self._last_lidar_position_gain = position_gain
        self._last_lidar_yaw_gain = yaw_gain
        self._last_lidar_velocity_gain = velocity_gain
        self._last_lidar_relative_valid = lidar_valid
        self._last_lidar_relative_quality = lidar_quality

    def _publish_outputs(self) -> None:
        now_t = self.get_clock().now()
        lidar_age_s = float("inf")
        if self._last_lidar_receipt is not None:
            lidar_age_s = max(0.0, (now_t - self._last_lidar_receipt).nanoseconds * 1e-9)
        lidar_pose_fresh = lidar_age_s <= self._lidar_pose_timeout_s

        imu_age_s = float("inf")
        if self._latest_imu_receipt is not None:
            imu_age_s = max(0.0, (now_t - self._latest_imu_receipt).nanoseconds * 1e-9)

        status_age_s = float("inf")
        if self._last_status_receipt is not None:
            status_age_s = max(0.0, (now_t - self._last_status_receipt).nanoseconds * 1e-9)

        cov_scale = 1.0 if lidar_pose_fresh else self._stale_cov_scale
        qx, qy, qz, qw = _yaw_to_quat(self._yaw)
        stamp = now_t.to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self._vx
        odom.twist.twist.linear.y = self._vy
        odom.twist.twist.angular.z = self._yaw_rate
        pose_covariance = [0.0] * 36
        pose_covariance[0] = self._base_pose_cov_x * cov_scale
        pose_covariance[7] = self._base_pose_cov_y * cov_scale
        pose_covariance[14] = 1e6
        pose_covariance[21] = 1e6
        pose_covariance[28] = 1e6
        pose_covariance[35] = self._base_yaw_cov * cov_scale
        odom.pose.covariance = pose_covariance
        twist_covariance = [0.0] * 36
        twist_covariance[0] = self._base_twist_cov * cov_scale
        twist_covariance[7] = self._base_twist_cov * cov_scale
        twist_covariance[35] = self._base_yaw_cov * cov_scale
        odom.twist.covariance = twist_covariance
        self._odom_pub.publish(odom)

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf)

        status_msg = String()
        status_msg.data = json.dumps(
            {
                "calibration_complete": self._calibration_complete,
                "stationary_detected": self._stationary_detected,
                "status_fresh": self._status_is_fresh(now_t),
                "status_motion_override_active": self._status_motion_override_active(now_t),
                "velocity_decay_active": self._velocity_decay_active,
                "odom_translation_confidence": self._odom_translation_confidence,
                "lidar_pose_fresh": lidar_pose_fresh,
                "lidar_age_s": lidar_age_s,
                "lidar_relative_valid": self._last_lidar_relative_valid,
                "lidar_relative_quality": self._last_lidar_relative_quality,
                "imu_age_s": imu_age_s,
                "status_age_s": status_age_s,
                "status_corrected_accel_planar_mps2": self._status_corrected_accel_planar_mps2,
                "status_speed_mps": self._status_speed_mps,
                "status_yaw_rate_rps": self._status_yaw_rate_rps,
                "x_m": self._x,
                "y_m": self._y,
                "vx_mps": self._vx,
                "vy_mps": self._vy,
                "yaw_deg": math.degrees(self._yaw),
                "yaw_rate_rps": self._yaw_rate,
                "bax_mps2": self._bax,
                "bay_mps2": self._bay,
                "bgz_rps": self._bgz,
                "imu_samples_processed": self._imu_samples_processed,
                "lidar_updates_applied": self._lidar_updates_applied,
                "launch_suppression_active": self._launch_started_at is not None,
                "last_lidar_velocity_mps": {
                    "x": self._last_lidar_velocity[0],
                    "y": self._last_lidar_velocity[1],
                    "yaw_rate": self._last_lidar_velocity[2],
                },
                "last_lidar_translation_observable": self._last_lidar_translation_observable,
                "lidar_position_update_suppressed": self._lidar_position_update_suppressed,
                "prediction_only_active": self._lidar_position_update_suppressed,
                "last_lidar_position_gain": self._last_lidar_position_gain,
                "last_lidar_yaw_gain": self._last_lidar_yaw_gain,
                "last_lidar_velocity_gain": self._last_lidar_velocity_gain,
                "last_lidar_delta_m": self._last_lidar_delta_m,
                "last_lidar_delta_yaw_deg": math.degrees(self._last_lidar_delta_yaw),
                "last_lidar_speed_mps": self._last_lidar_speed_mps,
                "last_lidar_innovation": {
                    "x_m": self._last_lidar_innovation[0],
                    "y_m": self._last_lidar_innovation[1],
                    "yaw_deg": math.degrees(self._last_lidar_innovation[2]),
                },
            },
            sort_keys=True,
        )
        self._status_pub.publish(status_msg)


def main() -> None:
    rclpy.init()
    node = ImuLidarPlanarFusionNode()
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
