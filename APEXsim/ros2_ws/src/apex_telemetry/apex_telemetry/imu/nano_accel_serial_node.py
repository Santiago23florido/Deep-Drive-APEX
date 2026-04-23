#!/usr/bin/env python3
"""Read Arduino Nano IMU stream from serial and publish ROS2 messages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
import re
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    import serial
except Exception:  # pragma: no cover - runtime dependency check
    serial = None


@dataclass(frozen=True)
class SerialConnectProfile:
    name: str
    toggle_dtr: bool
    dtr_low_s: float
    settle_s: float
    flush_input_on_connect: bool


class NanoAccelSerialNode(Node):
    """Bridge raw serial IMU stream from Arduino Nano into ROS2."""

    def __init__(self) -> None:
        super().__init__("nano_accel_serial_node")

        self.declare_parameter("transport_backend", "serial")
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout_s", 0.5)
        self.declare_parameter("reconnect_delay_s", 1.0)
        self.declare_parameter("connect_toggle_dtr", True)
        self.declare_parameter("connect_dtr_low_s", 0.2)
        self.declare_parameter("connect_settle_s", 2.0)
        self.declare_parameter("flush_input_on_connect", True)
        self.declare_parameter("no_data_reconnect_s", 3.5)
        self.declare_parameter("fallback_profiles_enabled", True)
        self.declare_parameter("connection_profile_name", "configured")
        self.declare_parameter("log_no_data_every_s", 5.0)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("gyro_topic", "/apex/imu/angular_velocity/raw")
        self.declare_parameter("imu_topic", "/apex/imu/data_raw")
        self.declare_parameter("publish_imu_topic", True)
        self.declare_parameter("allow_legacy_3axis_lines", False)
        self.declare_parameter("log_every_n_invalid", 25)
        self.declare_parameter("log_every_n_no_gyro", 100)
        self.declare_parameter("max_abs_accel_mps2", 30.0)
        self.declare_parameter("max_abs_gyro_rps", 15.0)
        self.declare_parameter("max_accel_norm_mps2", 30.0)
        self.declare_parameter("sim_imu_topic", "/apex/sim/imu")
        self.declare_parameter("sim_publish_latency_s", 0.004)
        self.declare_parameter("sim_random_seed", 1337)
        self.declare_parameter("sim_accel_noise_stddev_mps2", 0.08)
        self.declare_parameter("sim_gyro_noise_stddev_rps", 0.012)
        self.declare_parameter("sim_accel_drift_stddev_mps2_per_s", 0.003)
        self.declare_parameter("sim_gyro_drift_stddev_rps_per_s", 0.0008)
        self.declare_parameter("sim_accel_bias_x_mps2", 0.0)
        self.declare_parameter("sim_accel_bias_y_mps2", 0.0)
        self.declare_parameter("sim_accel_bias_z_mps2", 0.0)
        self.declare_parameter("sim_gyro_bias_x_rps", 0.0)
        self.declare_parameter("sim_gyro_bias_y_rps", 0.0)
        self.declare_parameter("sim_gyro_bias_z_rps", 0.0)
        self.declare_parameter("sim_startup_accel_bias_x_mps2", 0.0)
        self.declare_parameter("sim_startup_accel_bias_y_mps2", 0.0)
        self.declare_parameter("sim_startup_accel_bias_z_mps2", 0.0)
        self.declare_parameter("sim_startup_gyro_bias_x_rps", 0.0)
        self.declare_parameter("sim_startup_gyro_bias_y_rps", 0.0)
        self.declare_parameter("sim_startup_gyro_bias_z_rps", 0.0)
        self.declare_parameter("sim_startup_bias_hold_s", 0.0)
        self.declare_parameter("sim_initial_roll_deg", 0.0)
        self.declare_parameter("sim_initial_pitch_deg", 0.0)
        self.declare_parameter("sim_initial_yaw_deg", 0.0)

        self._transport_backend = (
            str(self.get_parameter("transport_backend").value).strip().lower() or "serial"
        )
        self._serial_port = str(self.get_parameter("serial_port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._serial_timeout_s = max(0.05, float(self.get_parameter("serial_timeout_s").value))
        self._reconnect_delay_s = max(0.1, float(self.get_parameter("reconnect_delay_s").value))
        self._connect_toggle_dtr = bool(self.get_parameter("connect_toggle_dtr").value)
        self._connect_dtr_low_s = max(0.0, float(self.get_parameter("connect_dtr_low_s").value))
        self._connect_settle_s = max(0.0, float(self.get_parameter("connect_settle_s").value))
        self._flush_input_on_connect = bool(self.get_parameter("flush_input_on_connect").value)
        self._no_data_reconnect_s = max(0.5, float(self.get_parameter("no_data_reconnect_s").value))
        self._fallback_profiles_enabled = bool(
            self.get_parameter("fallback_profiles_enabled").value
        )
        self._connection_profile_name = str(
            self.get_parameter("connection_profile_name").value
        )
        self._log_no_data_every_s = max(
            0.5, float(self.get_parameter("log_no_data_every_s").value)
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._log_every_n_invalid = max(1, int(self.get_parameter("log_every_n_invalid").value))
        self._log_every_n_no_gyro = max(1, int(self.get_parameter("log_every_n_no_gyro").value))
        self._publish_imu_topic = bool(self.get_parameter("publish_imu_topic").value)
        self._allow_legacy_3axis_lines = bool(
            self.get_parameter("allow_legacy_3axis_lines").value
        )
        self._max_abs_accel_mps2 = max(
            1.0, float(self.get_parameter("max_abs_accel_mps2").value)
        )
        self._max_abs_gyro_rps = max(
            0.1, float(self.get_parameter("max_abs_gyro_rps").value)
        )
        self._max_accel_norm_mps2 = max(
            self._max_abs_accel_mps2,
            float(self.get_parameter("max_accel_norm_mps2").value),
        )
        self._sim_imu_topic = str(self.get_parameter("sim_imu_topic").value)
        self._sim_publish_latency_s = max(
            0.0, float(self.get_parameter("sim_publish_latency_s").value)
        )
        self._sim_rng = random.Random(int(self.get_parameter("sim_random_seed").value))
        self._sim_accel_noise_stddev_mps2 = max(
            0.0, float(self.get_parameter("sim_accel_noise_stddev_mps2").value)
        )
        self._sim_gyro_noise_stddev_rps = max(
            0.0, float(self.get_parameter("sim_gyro_noise_stddev_rps").value)
        )
        self._sim_accel_drift_stddev_mps2_per_s = max(
            0.0, float(self.get_parameter("sim_accel_drift_stddev_mps2_per_s").value)
        )
        self._sim_gyro_drift_stddev_rps_per_s = max(
            0.0, float(self.get_parameter("sim_gyro_drift_stddev_rps_per_s").value)
        )
        self._sim_accel_bias = [
            float(self.get_parameter("sim_accel_bias_x_mps2").value),
            float(self.get_parameter("sim_accel_bias_y_mps2").value),
            float(self.get_parameter("sim_accel_bias_z_mps2").value),
        ]
        self._sim_gyro_bias = [
            float(self.get_parameter("sim_gyro_bias_x_rps").value),
            float(self.get_parameter("sim_gyro_bias_y_rps").value),
            float(self.get_parameter("sim_gyro_bias_z_rps").value),
        ]
        self._sim_startup_accel_bias = [
            float(self.get_parameter("sim_startup_accel_bias_x_mps2").value),
            float(self.get_parameter("sim_startup_accel_bias_y_mps2").value),
            float(self.get_parameter("sim_startup_accel_bias_z_mps2").value),
        ]
        self._sim_startup_gyro_bias = [
            float(self.get_parameter("sim_startup_gyro_bias_x_rps").value),
            float(self.get_parameter("sim_startup_gyro_bias_y_rps").value),
            float(self.get_parameter("sim_startup_gyro_bias_z_rps").value),
        ]
        self._sim_startup_bias_hold_s = max(
            0.0, float(self.get_parameter("sim_startup_bias_hold_s").value)
        )
        self._sim_initial_roll_rad = math.radians(
            float(self.get_parameter("sim_initial_roll_deg").value)
        )
        self._sim_initial_pitch_rad = math.radians(
            float(self.get_parameter("sim_initial_pitch_deg").value)
        )
        self._sim_initial_yaw_rad = math.radians(
            float(self.get_parameter("sim_initial_yaw_deg").value)
        )

        self._accel_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("topic").value),
            20,
        )
        self._gyro_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("gyro_topic").value),
            20,
        )
        self._imu_pub = (
            self.create_publisher(
                Imu,
                str(self.get_parameter("imu_topic").value),
                20,
            )
            if self._publish_imu_topic
            else None
        )

        self._invalid_counter = 0
        self._no_gyro_counter = 0
        self._preferred_profile_index = 0
        self._connection_profiles = self._build_connection_profiles()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._sim_queue: deque[tuple[float, tuple[float, float, float, float, float, float, bool]]] = deque()
        self._sim_accel_drift = [0.0, 0.0, 0.0]
        self._sim_gyro_drift = [0.0, 0.0, 0.0]
        self._sim_last_sample_monotonic: float | None = None
        self._sim_start_monotonic = time.monotonic()
        self._sim_rotation_matrix = self._rotation_matrix_rpy(
            self._sim_initial_roll_rad,
            self._sim_initial_pitch_rad,
            self._sim_initial_yaw_rad,
        )
        self._sim_flush_timer = None

        if self._transport_backend == "serial":
            self._worker_thread = threading.Thread(target=self._serial_worker, daemon=True)
            self._worker_thread.start()
        elif self._transport_backend == "sim_imu":
            self.create_subscription(Imu, self._sim_imu_topic, self._sim_imu_cb, 50)
            self._sim_flush_timer = self.create_timer(0.002, self._flush_sim_queue)
        else:
            raise ValueError(
                "Unsupported transport_backend=%r (expected 'serial' or 'sim_imu')"
                % self._transport_backend
            )

        self.get_logger().info(
            "NanoAccelSerialNode started (backend=%s port=%s sim_imu=%s baudrate=%d accel=%s gyro=%s imu=%s profiles=%s)"
            % (
                self._transport_backend,
                self._serial_port,
                self._sim_imu_topic,
                self._baudrate,
                self._accel_pub.topic_name,
                self._gyro_pub.topic_name,
                self._imu_pub.topic_name if self._imu_pub is not None else "(disabled)",
                ",".join(profile.name for profile in self._connection_profiles),
            )
        )

    @staticmethod
    def _rotation_matrix_rpy(
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        cr = math.cos(roll_rad)
        sr = math.sin(roll_rad)
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)
        return (
            ((cy * cp), (cy * sp * sr) - (sy * cr), (cy * sp * cr) + (sy * sr)),
            ((sy * cp), (sy * sp * sr) + (cy * cr), (sy * sp * cr) - (cy * sr)),
            ((-sp), cp * sr, cp * cr),
        )

    @staticmethod
    def _rotate_vector(
        matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        vector: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            (matrix[0][0] * vector[0]) + (matrix[0][1] * vector[1]) + (matrix[0][2] * vector[2]),
            (matrix[1][0] * vector[0]) + (matrix[1][1] * vector[1]) + (matrix[1][2] * vector[2]),
            (matrix[2][0] * vector[0]) + (matrix[2][1] * vector[1]) + (matrix[2][2] * vector[2]),
        )

    def _update_sim_drift(self, *, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        accel_sigma = self._sim_accel_drift_stddev_mps2_per_s * math.sqrt(dt_s)
        gyro_sigma = self._sim_gyro_drift_stddev_rps_per_s * math.sqrt(dt_s)
        if accel_sigma > 0.0:
            for idx in range(3):
                self._sim_accel_drift[idx] += self._sim_rng.gauss(0.0, accel_sigma)
        if gyro_sigma > 0.0:
            for idx in range(3):
                self._sim_gyro_drift[idx] += self._sim_rng.gauss(0.0, gyro_sigma)

    def _apply_sim_imu_effects(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        *,
        now_monotonic: float,
    ) -> tuple[float, float, float, float, float, float]:
        if self._sim_last_sample_monotonic is None:
            dt_s = 0.0
        else:
            dt_s = max(0.0, now_monotonic - self._sim_last_sample_monotonic)
        self._sim_last_sample_monotonic = now_monotonic
        self._update_sim_drift(dt_s=dt_s)

        accel = self._rotate_vector(self._sim_rotation_matrix, (ax, ay, az))
        gyro = self._rotate_vector(self._sim_rotation_matrix, (gx, gy, gz))

        startup_scale = 0.0
        if self._sim_startup_bias_hold_s > 1.0e-6:
            elapsed = max(0.0, now_monotonic - self._sim_start_monotonic)
            if elapsed < self._sim_startup_bias_hold_s:
                startup_scale = 1.0 - (elapsed / self._sim_startup_bias_hold_s)

        accel_out = []
        gyro_out = []
        for idx in range(3):
            accel_out.append(
                accel[idx]
                + self._sim_accel_bias[idx]
                + self._sim_accel_drift[idx]
                + (startup_scale * self._sim_startup_accel_bias[idx])
                + self._sim_rng.gauss(0.0, self._sim_accel_noise_stddev_mps2)
            )
            gyro_out.append(
                gyro[idx]
                + self._sim_gyro_bias[idx]
                + self._sim_gyro_drift[idx]
                + (startup_scale * self._sim_startup_gyro_bias[idx])
                + self._sim_rng.gauss(0.0, self._sim_gyro_noise_stddev_rps)
            )

        return (
            accel_out[0],
            accel_out[1],
            accel_out[2],
            gyro_out[0],
            gyro_out[1],
            gyro_out[2],
        )

    def _sim_imu_cb(self, msg: Imu) -> None:
        now_monotonic = time.monotonic()
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        gx = float(msg.angular_velocity.x)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)

        ax, ay, az, gx, gy, gz = self._apply_sim_imu_effects(
            ax,
            ay,
            az,
            gx,
            gy,
            gz,
            now_monotonic=now_monotonic,
        )
        if not self._sample_is_physically_plausible(ax, ay, az, gx, gy, gz, True):
            self._invalid_counter += 1
            if self._invalid_counter % self._log_every_n_invalid == 0:
                self.get_logger().warning(
                    (
                        "Skipping implausible simulated IMU sample: "
                        "accel=(%.3f, %.3f, %.3f) gyro=(%.3f, %.3f, %.3f)"
                    )
                    % (ax, ay, az, gx, gy, gz)
                )
            return

        ready_monotonic = now_monotonic + self._sim_publish_latency_s
        self._sim_queue.append((ready_monotonic, (ax, ay, az, gx, gy, gz, True)))

    def _flush_sim_queue(self) -> None:
        now_monotonic = time.monotonic()
        while self._sim_queue and self._sim_queue[0][0] <= now_monotonic:
            _, sample = self._sim_queue.popleft()
            self._publish_sample(*sample)

    def _append_profile_once(
        self,
        profiles: list[SerialConnectProfile],
        profile: SerialConnectProfile,
    ) -> None:
        signature = (
            bool(profile.toggle_dtr),
            round(float(profile.dtr_low_s), 4),
            round(float(profile.settle_s), 4),
            bool(profile.flush_input_on_connect),
        )
        for existing in profiles:
            existing_signature = (
                bool(existing.toggle_dtr),
                round(float(existing.dtr_low_s), 4),
                round(float(existing.settle_s), 4),
                bool(existing.flush_input_on_connect),
            )
            if existing_signature == signature:
                return
        profiles.append(profile)

    def _build_connection_profiles(self) -> list[SerialConnectProfile]:
        configured = SerialConnectProfile(
            name=self._connection_profile_name or "configured",
            toggle_dtr=self._connect_toggle_dtr,
            dtr_low_s=self._connect_dtr_low_s,
            settle_s=self._connect_settle_s,
            flush_input_on_connect=self._flush_input_on_connect,
        )
        profiles: list[SerialConnectProfile] = [configured]
        if not self._fallback_profiles_enabled:
            return profiles

        if self._flush_input_on_connect:
            self._append_profile_once(
                profiles,
                SerialConnectProfile(
                    name="configured_no_flush",
                    toggle_dtr=self._connect_toggle_dtr,
                    dtr_low_s=self._connect_dtr_low_s,
                    settle_s=self._connect_settle_s,
                    flush_input_on_connect=False,
                ),
            )

        self._append_profile_once(
            profiles,
            SerialConnectProfile(
                name="passive_no_flush",
                toggle_dtr=False,
                dtr_low_s=0.0,
                settle_s=0.0,
                flush_input_on_connect=False,
            ),
        )
        self._append_profile_once(
            profiles,
            SerialConnectProfile(
                name="passive_flush",
                toggle_dtr=False,
                dtr_low_s=0.0,
                settle_s=max(0.0, min(0.75, self._connect_settle_s)),
                flush_input_on_connect=True,
            ),
        )
        self._append_profile_once(
            profiles,
            SerialConnectProfile(
                name="dtr_handshake",
                toggle_dtr=True,
                dtr_low_s=max(0.2, self._connect_dtr_low_s),
                settle_s=max(2.0, self._connect_settle_s),
                flush_input_on_connect=True,
            ),
        )
        return profiles

    def _parse_line(
        self, line: str
    ) -> Optional[Tuple[float, float, float, float, float, float, bool]]:
        """Parse one serial line into accel+gyro values (SI units)."""
        text = line.strip()
        if not text:
            return None

        # Accept optional prefixes such as "ACC:" or "INFO:".
        if ":" in text:
            text = text.split(":", 1)[1].strip()

        parts = [p for p in re.split(r"[,\s;]+", text) if p]
        if len(parts) < 3:
            return None

        try:
            ax = float(parts[0])
            ay = float(parts[1])
            az = float(parts[2])
            if len(parts) >= 6:
                gx = float(parts[3])
                gy = float(parts[4])
                gz = float(parts[5])
                if not all(math.isfinite(v) for v in (ax, ay, az, gx, gy, gz)):
                    return None
                return (ax, ay, az, gx, gy, gz, True)
            if not self._allow_legacy_3axis_lines:
                return None
            if not all(math.isfinite(v) for v in (ax, ay, az)):
                return None
            return (ax, ay, az, 0.0, 0.0, 0.0, False)
        except ValueError:
            return None

    def _sample_is_physically_plausible(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        has_gyro: bool,
    ) -> bool:
        accel = (ax, ay, az)
        if any(abs(value) > self._max_abs_accel_mps2 for value in accel):
            return False
        accel_norm = math.sqrt((ax * ax) + (ay * ay) + (az * az))
        if accel_norm > self._max_accel_norm_mps2:
            return False
        if has_gyro and any(abs(value) > self._max_abs_gyro_rps for value in (gx, gy, gz)):
            return False
        return True

    def _publish_sample(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        has_gyro: bool,
    ) -> None:
        stamp = self.get_clock().now().to_msg()

        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = self._frame_id
        accel_msg.vector.x = ax
        accel_msg.vector.y = ay
        accel_msg.vector.z = az
        self._accel_pub.publish(accel_msg)

        if has_gyro:
            gyro_msg = Vector3Stamped()
            gyro_msg.header.stamp = stamp
            gyro_msg.header.frame_id = self._frame_id
            gyro_msg.vector.x = gx
            gyro_msg.vector.y = gy
            gyro_msg.vector.z = gz
            self._gyro_pub.publish(gyro_msg)
        else:
            self._no_gyro_counter += 1
            if self._no_gyro_counter % self._log_every_n_no_gyro == 0:
                self.get_logger().warning(
                    "Received legacy 3-axis line without gyro values; yaw rate unavailable."
                )

        if self._imu_pub is not None:
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = self._frame_id
            imu_msg.orientation_covariance[0] = -1.0
            imu_msg.angular_velocity.x = gx if has_gyro else 0.0
            imu_msg.angular_velocity.y = gy if has_gyro else 0.0
            imu_msg.angular_velocity.z = gz if has_gyro else 0.0
            imu_msg.linear_acceleration.x = ax
            imu_msg.linear_acceleration.y = ay
            imu_msg.linear_acceleration.z = az
            self._imu_pub.publish(imu_msg)

    def _sleep_with_stop(self, duration_s: float) -> None:
        if duration_s <= 0.0:
            return
        deadline = time.monotonic() + duration_s
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.1, remaining))

    def _prepare_serial_connection(
        self,
        ser: serial.Serial,
        profile: SerialConnectProfile,
    ) -> None:
        # Arduino ACM devices often reset on open/DTR transitions. Make that
        # reset explicit, then wait for the firmware to start streaming before
        # we begin parsing lines.
        if profile.toggle_dtr:
            try:
                ser.setDTR(False)
            except Exception:
                pass
            self._sleep_with_stop(profile.dtr_low_s)

        if profile.flush_input_on_connect:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass

        if profile.toggle_dtr:
            try:
                ser.setDTR(True)
            except Exception:
                pass

        if profile.toggle_dtr and profile.settle_s > 0.0:
            self.get_logger().info(
                "Waiting %.2fs for Arduino serial to settle after DTR handshake (profile=%s)."
                % (profile.settle_s, profile.name)
            )
        self._sleep_with_stop(profile.settle_s)

        if profile.flush_input_on_connect:
            try:
                ser.reset_input_buffer()
            except Exception:
                pass

    def _serial_worker(self) -> None:
        if serial is None:
            self.get_logger().error("pyserial is not available. Install python3-serial.")
            return

        profile_index = self._preferred_profile_index
        attempt_count = 0
        while not self._stop_event.is_set():
            ser = None
            profile = self._connection_profiles[profile_index]
            try:
                ser = serial.Serial()
                ser.port = self._serial_port
                ser.baudrate = self._baudrate
                ser.timeout = self._serial_timeout_s
                ser.write_timeout = self._serial_timeout_s
                if not profile.toggle_dtr:
                    try:
                        ser.dtr = False
                    except Exception:
                        pass
                ser.open()
                attempt_count += 1
                self.get_logger().info(
                    "Connected to Arduino serial %s @ %d (profile=%s attempt=%d)"
                    % (self._serial_port, self._baudrate, profile.name, attempt_count)
                )
                self._prepare_serial_connection(ser, profile)

                connect_t = time.monotonic()
                last_no_data_log_t = connect_t
                last_sample_t = connect_t
                valid_samples = 0
                invalid_since_connect = 0

                while not self._stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        now_t = time.monotonic()
                        if now_t - last_no_data_log_t >= self._log_no_data_every_s:
                            self.get_logger().warning(
                                "No serial IMU samples received on %s for %.1fs after connect."
                                % (self._serial_port, now_t - last_no_data_log_t)
                            )
                            last_no_data_log_t = now_t
                        if (now_t - last_sample_t) >= self._no_data_reconnect_s:
                            raise RuntimeError(
                                "No valid serial IMU samples received for %.2fs on profile=%s; reconnecting"
                                % (now_t - last_sample_t, profile.name)
                            )
                        continue

                    line = raw.decode("utf-8", errors="replace").strip()
                    parsed = self._parse_line(line)
                    if parsed is None:
                        invalid_since_connect += 1
                        self._invalid_counter += 1
                        if self._invalid_counter % self._log_every_n_invalid == 0:
                            self.get_logger().warning(
                                "Skipping invalid serial line on profile=%s: '%s'"
                                % (profile.name, line)
                            )
                        if (
                            invalid_since_connect >= self._log_every_n_invalid
                            and (time.monotonic() - connect_t) >= self._no_data_reconnect_s
                            and valid_samples == 0
                        ):
                            raise RuntimeError(
                                "Only invalid serial lines seen for %.2fs on profile=%s; reconnecting"
                                % (time.monotonic() - connect_t, profile.name)
                            )
                        continue

                    ax, ay, az, gx, gy, gz, has_gyro = parsed
                    if not self._sample_is_physically_plausible(ax, ay, az, gx, gy, gz, has_gyro):
                        invalid_since_connect += 1
                        self._invalid_counter += 1
                        if self._invalid_counter % self._log_every_n_invalid == 0:
                            self.get_logger().warning(
                                (
                                    "Skipping implausible IMU sample on profile=%s: "
                                    "accel=(%.3f, %.3f, %.3f) gyro=(%.3f, %.3f, %.3f)"
                                )
                                % (profile.name, ax, ay, az, gx, gy, gz)
                            )
                        if (
                            invalid_since_connect >= self._log_every_n_invalid
                            and (time.monotonic() - connect_t) >= self._no_data_reconnect_s
                            and valid_samples == 0
                        ):
                            raise RuntimeError(
                                "Only implausible serial IMU samples seen for %.2fs on profile=%s; reconnecting"
                                % (time.monotonic() - connect_t, profile.name)
                            )
                        continue
                    invalid_since_connect = 0
                    valid_samples += 1
                    last_no_data_log_t = time.monotonic()
                    last_sample_t = last_no_data_log_t
                    if valid_samples == 1:
                        self.get_logger().info(
                            "First valid Arduino IMU sample received on %s (profile=%s)."
                            % (self._serial_port, profile.name)
                        )
                        self._preferred_profile_index = profile_index
                    self._publish_sample(ax, ay, az, gx, gy, gz, has_gyro)
            except Exception as exc:
                self.get_logger().warning(
                    "Serial connection error on %s (profile=%s): %s"
                    % (self._serial_port, profile.name, str(exc))
                )
                next_profile_index = (profile_index + 1) % len(self._connection_profiles)
                wrapped = next_profile_index == self._preferred_profile_index
                profile_index = next_profile_index
                self._sleep_with_stop(self._reconnect_delay_s if wrapped else 0.2)
            finally:
                if ser is not None:
                    try:
                        if ser.is_open:
                            ser.close()
                    except Exception:
                        pass

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        if self._sim_flush_timer is not None:
            try:
                self._sim_flush_timer.cancel()
            except Exception:
                pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = NanoAccelSerialNode()
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
