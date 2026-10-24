#!/usr/bin/env python3
"""ROS2 node: read RPLidar on Raspberry and publish LaserScan.

Backends (``source_backend``):
* ``rplidar_express`` (default): the A2M8 in the Express scan (4 kHz) at a
  regulated 10 Hz, one scan per complete revolution, stamped at the first
  sample of the revolution from the LiDAR's own sample clock
  (``rplidar_express.py``). Besides the three APEX streams it publishes
  ``scan_raw_topic`` in the layout of the simulation (``APEX_real``), which
  is what the learned odometry consumes.
* ``rplidar``: the previous driver (python ``rplidar`` library, 2 kHz legacy
  scan, stamped on reception, ~13 Hz). Kept to compare.
* ``sim_scan``: re-publish a simulated scan.
"""

from __future__ import annotations

from collections import deque
import glob
import math
import os
import random
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RosTime
from sensor_msgs.msg import LaserScan

from .lidar_scan_buffer import LidarScanBuffer
from .rplidar_express import (
    ExpressScanPipeline,
    HEALTH_STATUS,
    MotorRegulator,
    RPLidarExpressDriver,
    StampedRevolution,
    SupplyGuard,
    bin_revolution,
    find_undervoltage_alarm,
    legacy_measurements,
)


def _try_import_rplidar():
    """Import rplidar, with fallback to the dedicated APEX venv site-packages."""
    try:
        from rplidar import RPLidar as _RPLidar, RPLidarException as _RPLidarException

        return _RPLidar, _RPLidarException
    except Exception:
        # Fallback for cases where ROS launches with system python instead of venv python.
        candidates = sorted(glob.glob("/opt/apex_venv/lib/python*/site-packages"))
        for path in candidates:
            if path not in sys.path:
                sys.path.append(path)

        try:
            from rplidar import RPLidar as _RPLidar, RPLidarException as _RPLidarException

            return _RPLidar, _RPLidarException
        except Exception:
            return None, Exception


RPLidar, RPLidarException = _try_import_rplidar()


class RPLidarPublisherNode(Node):
    """Publish RPLidar scans on `/lidar/scan`."""

    def __init__(self) -> None:
        super().__init__("apex_rplidar_publisher")

        self.declare_parameter("source_backend", "rplidar_express")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("topic", "/lidar/scan")
        self.declare_parameter("localization_topic", "/lidar/scan_localization")
        self.declare_parameter("slam_topic", "/lidar/scan_slam")
        self.declare_parameter("max_buf_meas", 2000)
        self.declare_parameter("localization_samples", 720)
        self.declare_parameter("localization_missing_as_inf", True)
        self.declare_parameter("frame_id", "laser")
        self.declare_parameter("heading_offset_deg", -89)
        self.declare_parameter("fov_filter_deg", 180)
        self.declare_parameter("point_timeout_ms", 1000)
        self.declare_parameter("fill_missing_bins", False)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 12.0)
        # --- rplidar_express backend (real_vehicle/docs/sensor_timing_requirements.md)
        # Scan in the simulation's layout for the learned odometry: 360 bins
        # from -180 deg counter-clockwise, nearest return per bin, REP-117.
        self.declare_parameter("scan_raw_topic", "/lidar/scan_raw")
        self.declare_parameter("publish_scan_raw", True)
        self.declare_parameter("raw_range_min_m", 0.15)  # A2M8 datasheet, as APEX_real
        self.declare_parameter("raw_range_max_m", 12.0)
        self.declare_parameter("target_rate_hz", 10.0)  # the rate the learned odometry was trained at
        self.declare_parameter("motor_pwm", 520)  # start value; ~660 gave 13 Hz on this unit
        self.declare_parameter("regulate_motor", True)
        # Soft start: the PWM climbs from motor_start_pwm to motor_pwm over motor_ramp_s.
        self.declare_parameter("motor_start_pwm", 0)
        self.declare_parameter("motor_ramp_s", 6.0)
        self.declare_parameter("motor_settle_s", 2.0)
        # The car's Pi 5 has reset while the motor started: wait for power_clear_s
        # without under-voltage first (at most power_wait_max_s), and hold the ramp
        # while the flag is up (at most power_hold_max_s across the ramp).
        self.declare_parameter("power_guard", True)
        self.declare_parameter("power_clear_s", 1.0)
        self.declare_parameter("power_wait_max_s", 20.0)
        self.declare_parameter("power_hold_max_s", 10.0)
        run_dir = os.environ.get("APEX_RACE_RUN_DIR", "")
        self.declare_parameter("power_status_file", os.path.join(run_dir, "power.csv") if run_dir else "")
        # After a failure the driver waits 1, 2, 4 ... up to this before reopening the LiDAR.
        self.declare_parameter("retry_max_s", 30.0)
        # Minimum USB latency, subtracted from the stamps (calibrate on the car).
        self.declare_parameter("stamp_correction_ms", 0.0)
        self.declare_parameter("drop_incomplete_revolutions", True)
        self.declare_parameter("timing_log_every_s", 10.0)
        self.declare_parameter("sim_scan_topic", "/apex/sim/scan")
        self.declare_parameter("sim_publish_latency_s", 0.03)
        self.declare_parameter("sim_random_seed", 4242)
        self.declare_parameter("sim_range_noise_stddev_m", 0.02)
        self.declare_parameter("sim_dropout_ratio", 0.01)
        self.declare_parameter("sim_inf_ratio", 0.02)
        self.declare_parameter("sim_heading_jitter_deg", 0.0)
        self.declare_parameter("sim_sector_dropout_center_deg", 0.0)
        self.declare_parameter("sim_sector_dropout_width_deg", 0.0)
        self.declare_parameter("sim_sector_dropout_ratio", 0.0)
        self.declare_parameter("sim_sector_range_bias_m", 0.0)
        self.declare_parameter("sim_startup_heading_offset_deg", 0.0)
        self.declare_parameter("sim_startup_heading_offset_hold_s", 0.0)

        self._source_backend = (
            str(self.get_parameter("source_backend").value).strip().lower() or "rplidar"
        )
        self._port = str(self.get_parameter("port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._topic = str(self.get_parameter("topic").value)
        self._localization_topic = str(self.get_parameter("localization_topic").value)
        self._slam_topic = str(self.get_parameter("slam_topic").value)
        self._max_buf_meas = max(500, int(self.get_parameter("max_buf_meas").value))
        self._localization_samples = max(
            360,
            int(self.get_parameter("localization_samples").value),
        )
        self._localization_missing_as_inf = bool(
            self.get_parameter("localization_missing_as_inf").value
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._range_min = float(self.get_parameter("range_min").value)
        self._range_max = float(self.get_parameter("range_max").value)
        self._heading_offset_deg = int(self.get_parameter("heading_offset_deg").value)
        self._fov_filter_deg = int(self.get_parameter("fov_filter_deg").value)
        self._scan_raw_topic = str(self.get_parameter("scan_raw_topic").value)
        self._publish_scan_raw = bool(self.get_parameter("publish_scan_raw").value)
        self._raw_range_min = float(self.get_parameter("raw_range_min_m").value)
        self._raw_range_max = float(self.get_parameter("raw_range_max_m").value)
        self._target_rate_hz = float(self.get_parameter("target_rate_hz").value)
        self._motor_pwm = int(self.get_parameter("motor_pwm").value)
        self._regulate_motor = bool(self.get_parameter("regulate_motor").value)
        self._motor_start_pwm = int(self.get_parameter("motor_start_pwm").value)
        self._motor_ramp_s = max(0.0, float(self.get_parameter("motor_ramp_s").value))
        self._motor_settle_s = max(0.0, float(self.get_parameter("motor_settle_s").value))
        self._retry_max_s = max(1.0, float(self.get_parameter("retry_max_s").value))
        self._power_guard = bool(self.get_parameter("power_guard").value)
        self._power_clear_s = max(0.0, float(self.get_parameter("power_clear_s").value))
        self._power_wait_max_s = max(0.0, float(self.get_parameter("power_wait_max_s").value))
        self._power_hold_max_s = max(0.0, float(self.get_parameter("power_hold_max_s").value))
        self._power_status_file = str(self.get_parameter("power_status_file").value)
        self._stamp_correction_s = float(self.get_parameter("stamp_correction_ms").value) * 1e-3
        self._drop_incomplete = bool(self.get_parameter("drop_incomplete_revolutions").value)
        self._timing_log_every_s = max(1.0, float(self.get_parameter("timing_log_every_s").value))
        self._sim_scan_topic = str(self.get_parameter("sim_scan_topic").value)
        self._sim_publish_latency_s = max(
            0.0, float(self.get_parameter("sim_publish_latency_s").value)
        )
        self._sim_rng = random.Random(int(self.get_parameter("sim_random_seed").value))
        self._sim_range_noise_stddev_m = max(
            0.0, float(self.get_parameter("sim_range_noise_stddev_m").value)
        )
        self._sim_dropout_ratio = max(
            0.0, min(1.0, float(self.get_parameter("sim_dropout_ratio").value))
        )
        self._sim_inf_ratio = max(
            0.0, min(1.0, float(self.get_parameter("sim_inf_ratio").value))
        )
        self._sim_heading_jitter_deg = max(
            0.0, float(self.get_parameter("sim_heading_jitter_deg").value)
        )
        self._sim_sector_dropout_center_deg = float(
            self.get_parameter("sim_sector_dropout_center_deg").value
        )
        self._sim_sector_dropout_width_deg = max(
            0.0, float(self.get_parameter("sim_sector_dropout_width_deg").value)
        )
        self._sim_sector_dropout_ratio = max(
            0.0, min(1.0, float(self.get_parameter("sim_sector_dropout_ratio").value))
        )
        self._sim_sector_range_bias_m = float(
            self.get_parameter("sim_sector_range_bias_m").value
        )
        self._sim_startup_heading_offset_deg = float(
            self.get_parameter("sim_startup_heading_offset_deg").value
        )
        self._sim_startup_heading_offset_hold_s = max(
            0.0, float(self.get_parameter("sim_startup_heading_offset_hold_s").value)
        )

        self._scan_buffer = LidarScanBuffer(
            samples=360,
            heading_offset_deg=self._heading_offset_deg,
            fov_filter_deg=self._fov_filter_deg,
            point_timeout_ms=int(self.get_parameter("point_timeout_ms").value),
            fill_missing_bins=bool(self.get_parameter("fill_missing_bins").value),
        )

        self._publisher = self.create_publisher(LaserScan, self._topic, qos_profile_sensor_data)
        self._localization_publisher = self.create_publisher(
            LaserScan,
            self._localization_topic,
            qos_profile_sensor_data,
        )
        self._slam_publisher = self.create_publisher(
            LaserScan,
            self._slam_topic,
            qos_profile_sensor_data,
        )
        # Same QoS as the simulation's scan (the learned odometry subscribes reliable).
        self._scan_raw_publisher = self.create_publisher(
            LaserScan,
            self._scan_raw_topic,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._sim_queue: deque[
            tuple[float, object, LaserScan, list[tuple[float, float, float]]]
        ] = deque()
        self._sim_flush_timer = None
        self._sim_start_monotonic = time.monotonic()

        self._lidar = None
        self._last_publish_time_by_stream = {
            "navigation": None,
            "localization": None,
        }

        if self._source_backend == "rplidar_express":
            self._worker = threading.Thread(target=self._express_loop, daemon=True)
            self._worker.start()
        elif self._source_backend == "rplidar":
            if RPLidar is None:
                raise RuntimeError("rplidar module not found. Install `rplidar-roboticia`.")
            self._worker = threading.Thread(target=self._scan_loop, daemon=True)
            self._worker.start()
        elif self._source_backend == "sim_scan":
            self.create_subscription(
                LaserScan,
                self._sim_scan_topic,
                self._sim_scan_cb,
                qos_profile_sensor_data,
            )
            self._sim_flush_timer = self.create_timer(0.005, self._flush_sim_queue)
        else:
            raise ValueError(
                "Unsupported source_backend=%r (expected 'rplidar_express', 'rplidar' or 'sim_scan')"
                % self._source_backend
            )

        self.get_logger().info(
            "LiDAR publisher started (backend=%s port=%s sim_scan=%s topic=%s localization_topic=%s slam_topic=%s "
            "baudrate=%d localization_samples=%d max_buf_meas=%d)"
            % (
                self._source_backend,
                self._port,
                self._sim_scan_topic,
                self._topic,
                self._localization_topic,
                self._slam_topic,
                self._baudrate,
                self._localization_samples,
                self._max_buf_meas,
            )
        )

    def _effective_sim_heading_offset_deg(self) -> float:
        heading_offset_deg = 0.0
        if self._sim_startup_heading_offset_hold_s > 1.0e-6:
            elapsed = max(0.0, time.monotonic() - self._sim_start_monotonic)
            if elapsed < self._sim_startup_heading_offset_hold_s:
                heading_offset_deg += self._sim_startup_heading_offset_deg * (
                    1.0 - (elapsed / self._sim_startup_heading_offset_hold_s)
                )
        if self._sim_heading_jitter_deg > 1.0e-9:
            heading_offset_deg += self._sim_rng.gauss(0.0, self._sim_heading_jitter_deg)
        return heading_offset_deg

    @staticmethod
    def _wrap_deg(angle_deg: float) -> float:
        wrapped = math.fmod(angle_deg, 360.0)
        if wrapped < 0.0:
            wrapped += 360.0
        return wrapped

    def _angle_in_dropout_sector(self, angle_deg: float) -> bool:
        if self._sim_sector_dropout_width_deg <= 1.0e-6:
            return False
        center = self._wrap_deg(self._sim_sector_dropout_center_deg)
        angle = self._wrap_deg(angle_deg)
        diff = abs(((angle - center) + 180.0) % 360.0 - 180.0)
        return diff <= (0.5 * self._sim_sector_dropout_width_deg)

    @staticmethod
    def _sample_scan_range(msg: LaserScan, angle_rad: float) -> float:
        samples = len(msg.ranges)
        if samples == 0:
            return math.inf
        angle = angle_rad
        full_turn = 2.0 * math.pi
        while angle < msg.angle_min:
            angle += full_turn
        while angle > (msg.angle_min + full_turn):
            angle -= full_turn
        index = int(round((angle - msg.angle_min) / max(1.0e-9, msg.angle_increment)))
        index %= samples
        try:
            return float(msg.ranges[index])
        except Exception:
            return math.inf

    def _corrupt_sim_ranges(self, msg: LaserScan) -> list[tuple[float, float, float]]:
        samples = 360
        heading_error_deg = self._effective_sim_heading_offset_deg()
        measurements: list[tuple[float, float, float]] = []
        for index in range(samples):
            # APEX localization/fusion uses a non-standard scan convention where
            # "forward" lands around 180 deg after heading correction.
            #
            # The Gazebo lidar bridge already publishes a standard 360 scan,
            # but under a simulator-specific frame id. Convert the desired
            # APEX-facing bin back into that raw scan angle, then encode it
            # into the raw RPLidar angle space so the existing heading_offset
            # correction reproduces the expected APEX-facing scan layout.
            apex_output_angle_deg = float(index)
            sample_apex_angle_deg = self._wrap_deg(apex_output_angle_deg - heading_error_deg)
            sample_angle_deg = self._wrap_deg(180.0 - sample_apex_angle_deg)
            sample_angle_rad = math.radians(sample_angle_deg)
            distance_m = self._sample_scan_range(msg, sample_angle_rad)

            if not math.isfinite(distance_m):
                continue
            if distance_m < self._range_min or distance_m > self._range_max:
                continue

            if self._sim_range_noise_stddev_m > 1.0e-9:
                distance_m += self._sim_rng.gauss(0.0, self._sim_range_noise_stddev_m)
            if self._angle_in_dropout_sector(apex_output_angle_deg):
                distance_m += self._sim_sector_range_bias_m
                if self._sim_rng.random() < self._sim_sector_dropout_ratio:
                    continue
            if self._sim_rng.random() < self._sim_dropout_ratio:
                continue
            if self._sim_rng.random() < self._sim_inf_ratio:
                continue
            if distance_m < self._range_min or distance_m > self._range_max:
                continue

            raw_angle_deg = self._wrap_deg(
                apex_output_angle_deg - float(self._heading_offset_deg)
            )
            measurements.append((0.0, raw_angle_deg, distance_m * 1000.0))
        return measurements

    def _corrupt_sim_distance(self, distance_m: float, output_angle_deg: float) -> float:
        if not math.isfinite(distance_m):
            return math.inf
        if distance_m < self._range_min or distance_m > self._range_max:
            return math.inf

        if self._sim_range_noise_stddev_m > 1.0e-9:
            distance_m += self._sim_rng.gauss(0.0, self._sim_range_noise_stddev_m)
        if self._angle_in_dropout_sector(output_angle_deg):
            distance_m += self._sim_sector_range_bias_m
            if self._sim_rng.random() < self._sim_sector_dropout_ratio:
                return math.inf
        if self._sim_rng.random() < self._sim_dropout_ratio:
            return math.inf
        if self._sim_rng.random() < self._sim_inf_ratio:
            return math.inf
        if distance_m < self._range_min or distance_m > self._range_max:
            return math.inf
        return distance_m

    def _build_sim_localization_ranges_from_msg(self, msg: LaserScan) -> np.ndarray:
        samples = self._localization_samples
        missing_value = math.inf if self._localization_missing_as_inf else 0.0
        ranges = np.full(samples, missing_value, dtype=np.float32)
        heading_error_deg = self._effective_sim_heading_offset_deg()

        for idx in range(samples):
            apex_output_angle_deg = float(idx) * 360.0 / float(samples)
            sample_apex_angle_deg = self._wrap_deg(apex_output_angle_deg - heading_error_deg)
            sample_raw_angle_deg = self._wrap_deg(180.0 - sample_apex_angle_deg)
            distance_m = self._sample_scan_range(msg, math.radians(sample_raw_angle_deg))
            distance_m = self._corrupt_sim_distance(distance_m, apex_output_angle_deg)
            if math.isfinite(distance_m):
                ranges[idx] = distance_m

        if 0 < self._fov_filter_deg < 360:
            half_fov = self._fov_filter_deg / 2.0
            angles = np.arange(samples, dtype=np.float32) * (360.0 / float(samples))
            diffs = np.mod(angles, 360.0)
            keep = (diffs <= half_fov) | (diffs >= 360.0 - half_fov)
            ranges[~keep] = missing_value

        return ranges

    def _build_sim_slam_ranges_from_msg(self, msg: LaserScan) -> np.ndarray:
        samples = 360
        ranges = np.full(samples, math.inf, dtype=np.float32)
        for idx in range(samples):
            laser_angle_deg = -180.0 + float(idx)
            sample_raw_angle_deg = self._wrap_deg(laser_angle_deg)
            distance_m = self._sample_scan_range(msg, math.radians(sample_raw_angle_deg))
            if (
                math.isfinite(distance_m)
                and self._range_min <= distance_m <= self._range_max
            ):
                ranges[idx] = distance_m
        return ranges

    def _sim_scan_cb(self, msg: LaserScan) -> None:
        ready_monotonic = time.monotonic() + self._sim_publish_latency_s
        self._sim_queue.append(
            (ready_monotonic, msg.header.stamp, msg, self._corrupt_sim_ranges(msg))
        )

    def _flush_sim_queue(self) -> None:
        now_monotonic = time.monotonic()
        while self._sim_queue and self._sim_queue[0][0] <= now_monotonic:
            _, stamp_msg, raw_msg, scan = self._sim_queue.popleft()
            ranges = self._scan_buffer.update_from_rplidar_scan(scan)
            localization_ranges = self._build_sim_localization_ranges_from_msg(raw_msg)
            slam_ranges = self._build_sim_slam_ranges_from_msg(raw_msg)
            self._publisher.publish(
                self._build_scan_msg(
                    ranges,
                    stream_key="navigation",
                    stamp_msg=stamp_msg,
                )
            )
            self._localization_publisher.publish(
                self._build_scan_msg(
                    localization_ranges,
                    stream_key="localization",
                    stamp_msg=stamp_msg,
                )
            )
            self._slam_publisher.publish(
                self._build_scan_msg(
                    slam_ranges,
                    stream_key="slam",
                    stamp_msg=stamp_msg,
                    angle_min_rad=-math.pi,
                )
            )

    def _connect_lidar(self, baudrate: int) -> None:
        self._lidar = RPLidar(self._port, baudrate=baudrate)

        if hasattr(self._lidar, "connect"):
            self._lidar.connect()

        if hasattr(self._lidar, "_serial") and self._lidar._serial is not None:
            self._lidar._serial.reset_input_buffer()
            self._lidar._serial.reset_output_buffer()

        self._lidar.start_motor()
        self._lidar.start()
        self.get_logger().info(
            "RPLidar connected and scanning (port=%s baudrate=%d)."
            % (self._port, baudrate)
        )

    def _disconnect_lidar(self) -> None:
        if self._lidar is None:
            return

        try:
            self._lidar.stop()
        except Exception:
            pass

        try:
            self._lidar.stop_motor()
        except Exception:
            pass

        try:
            self._lidar.disconnect()
        except Exception:
            pass

        self._lidar = None

    def _build_scan_msg(
        self,
        ranges: np.ndarray,
        *,
        stream_key: str,
        stamp_msg=None,
        angle_min_rad: float = 0.0,
        scan_time_s: float | None = None,
    ) -> LaserScan:
        now = stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg()
        current_t = time.time()

        last_publish_time = self._last_publish_time_by_stream.get(stream_key)
        if scan_time_s is not None:
            scan_time = float(scan_time_s)  # the revolution's own period
        elif last_publish_time is None:
            scan_time = 0.0
        else:
            scan_time = max(0.0, current_t - last_publish_time)
        self._last_publish_time_by_stream[stream_key] = current_t

        samples = len(ranges)
        angle_increment = (2.0 * math.pi) / float(samples)

        msg = LaserScan()
        msg.header.stamp = now
        msg.header.frame_id = self._frame_id
        msg.angle_min = float(angle_min_rad)
        msg.angle_max = float(angle_min_rad) + ((samples - 1) * angle_increment)
        msg.angle_increment = angle_increment
        msg.time_increment = scan_time / float(samples) if samples > 0 else 0.0
        msg.scan_time = scan_time
        msg.range_min = self._range_min
        msg.range_max = self._range_max
        msg.ranges = ranges.tolist()
        msg.intensities = []
        return msg

    def _build_localization_ranges(self, scan: list[tuple[float, float, float]]) -> np.ndarray:
        # Use only the current revolution for localization. Persisting bins across
        # scans is helpful for navigation heuristics but smears geometry for scan
        # matching and can suppress translational odometry.
        missing_value = math.inf if self._localization_missing_as_inf else 0.0
        ranges = np.full(self._localization_samples, missing_value, dtype=np.float32)
        for measurement in scan:
            if len(measurement) < 3:
                continue
            angle_deg = float(measurement[1])
            distance_m = float(measurement[2]) / 1000.0
            if not np.isfinite(distance_m) or distance_m <= 0.0:
                continue
            idx = int(round(angle_deg * self._localization_samples / 360.0)) % self._localization_samples
            current = float(ranges[idx])
            if not np.isfinite(current) or current <= 0.0 or distance_m < current:
                ranges[idx] = distance_m

        shift = int(round(self._heading_offset_deg * self._localization_samples / 360.0))
        shift %= self._localization_samples
        ranges = np.roll(ranges, shift)

        if 0 < self._fov_filter_deg < 360:
            half_fov = self._fov_filter_deg / 2.0
            angles = np.arange(self._localization_samples, dtype=np.float32)
            diffs = np.mod(angles, 360.0)
            keep = (diffs <= half_fov) | (diffs >= 360.0 - half_fov)
            ranges[~keep] = missing_value

        return ranges

    def _build_slam_ranges(self, scan: list[tuple[float, float, float]]) -> np.ndarray:
        samples = 360
        ranges = np.full(samples, math.inf, dtype=np.float32)
        for measurement in scan:
            if len(measurement) < 3:
                continue
            raw_angle_deg = float(measurement[1])
            distance_m = float(measurement[2]) / 1000.0
            if not np.isfinite(distance_m) or distance_m <= 0.0:
                continue
            apex_angle_deg = self._wrap_deg(raw_angle_deg + float(self._heading_offset_deg))
            standard_angle_deg = self._wrap_deg(180.0 - apex_angle_deg)
            normalized_deg = ((standard_angle_deg + 180.0) % 360.0) - 180.0
            idx = int(round(normalized_deg + 180.0)) % samples
            current = float(ranges[idx])
            if not np.isfinite(current) or distance_m < current:
                ranges[idx] = distance_m
        return ranges

    def _scan_loop(self) -> None:
        while not self._stop_event.is_set() and rclpy.ok():
            baud = self._baudrate
            try:
                self._connect_lidar(baud)
                for scan in self._lidar.iter_scans(max_buf_meas=self._max_buf_meas):
                    if self._stop_event.is_set() or not rclpy.ok():
                        break

                    stamp_msg = self.get_clock().now().to_msg()
                    ranges = self._scan_buffer.update_from_rplidar_scan(scan)
                    localization_ranges = self._build_localization_ranges(scan)
                    self._publisher.publish(
                        self._build_scan_msg(
                            ranges,
                            stream_key="navigation",
                            stamp_msg=stamp_msg,
                        )
                    )
                    self._localization_publisher.publish(
                        self._build_scan_msg(
                            localization_ranges,
                            stream_key="localization",
                            stamp_msg=stamp_msg,
                        )
                    )
                    self._slam_publisher.publish(
                        self._build_scan_msg(
                            self._build_slam_ranges(scan),
                            stream_key="slam",
                            stamp_msg=stamp_msg,
                            angle_min_rad=-math.pi,
                        )
                    )

            except (RPLidarException, OSError, ValueError) as exc:
                self.get_logger().error(
                    "RPLidar read error (port=%s baudrate=%d): %s"
                    % (self._port, baud, str(exc))
                )
                time.sleep(1.0)
            except Exception as exc:
                self.get_logger().error(
                    "Unexpected LiDAR failure (port=%s baudrate=%d): %s"
                    % (self._port, baud, str(exc))
                )
                time.sleep(1.0)
            finally:
                self._disconnect_lidar()

    # --- rplidar_express backend -------------------------------------------
    def _express_loop(self) -> None:
        failures = 0
        while not self._stop_event.is_set() and rclpy.ok():
            session_start = time.monotonic()
            driver = RPLidarExpressDriver(
                self._port,
                self._baudrate,
                timeout_s=1.0,
                now_ns=lambda: self.get_clock().now().nanoseconds,
            )
            try:
                driver.open()
                info = driver.get_info()
                self.get_logger().info(
                    "RPLIDAR model 0x%02X firmware %s hardware %d S/N %s (port=%s baudrate=%d)"
                    % (info.model, info.firmware, info.hardware, info.serial, self._port, self._baudrate)
                )
                if info.supports_sensitivity:
                    self.get_logger().info(
                        "Firmware %s >= 1.24: the 8 kHz Sensitivity scan is available too "
                        "(simulation profile APEX_real_sensitivity). This node runs the Express scan (APEX_real)."
                        % info.firmware
                    )
                status, error_code = driver.get_health()
                if status == 2:
                    raise RuntimeError("RPLIDAR health ERROR (code 0x%04X): power-cycle the LiDAR" % error_code)
                if status != 0:
                    self.get_logger().warning(
                        "RPLIDAR health %s (code 0x%04X)" % (HEALTH_STATUS.get(status, status), error_code)
                    )
                std_us, express_us = driver.get_sample_rate()
                pwm_flag = driver.supports_motor_pwm()
                pwm_supported = pwm_flag is not False
                if pwm_flag is None:
                    self.get_logger().info(
                        "The LiDAR does not answer the motor-control query; regulating the PWM anyway "
                        "(the timing log shows whether the rate follows it)."
                    )
                elif not pwm_flag:
                    self.get_logger().error(
                        "The LiDAR adapter reports no PWM motor control: the rotation rate cannot be set "
                        "and will not be %.1f Hz." % self._target_rate_hz
                    )
                regulator = MotorRegulator(
                    self._target_rate_hz,
                    self._motor_pwm,
                    enabled=self._regulate_motor and pwm_supported,
                )
                guard = SupplyGuard(
                    find_undervoltage_alarm() if self._power_guard else None,
                    clear_s=self._power_clear_s,
                    wait_max_s=self._power_wait_max_s,
                    hold_max_s=self._power_hold_max_s,
                    status_path=self._power_status_file if self._power_guard else None,
                    sleep=self._sleep_with_stop,
                )
                if guard.status_path:
                    self.get_logger().info("Power guard: current under-voltage bit from host power log; hwmon fallback if stale")
                if guard.alarm_path is None and not guard.status_path:
                    self.get_logger().info("No under-voltage flag to watch: plain soft start")
                elif not guard.wait_clear():
                    self.get_logger().warning(
                        "The Pi stayed in under-voltage for %.0f s; starting the motor anyway "
                        "(a reset now means the supply cannot carry the LiDAR motor)" % self._power_wait_max_s
                    )
                self.get_logger().info(
                    "Motor soft start: PWM %d -> %d over %.1f s"
                    % (min(self._motor_start_pwm, regulator.pwm), regulator.pwm, self._motor_ramp_s)
                )
                ramp_t0 = time.monotonic()
                driver.start_motor(
                    regulator.pwm,
                    start_pwm=self._motor_start_pwm,
                    ramp_s=self._motor_ramp_s,
                    wait=guard.ramp_wait,
                )
                if guard.alarm_path is not None:
                    self.get_logger().info(
                        "Motor ramp done in %.1f s; under-voltage seen in %d polls"
                        % (time.monotonic() - ramp_t0, guard.alarms)
                    )
                self._sleep_with_stop(self._motor_settle_s)
                pipeline = ExpressScanPipeline(
                    express_us * 1e-6,
                    self._baudrate,
                    stamp_correction_s=self._stamp_correction_s,
                    drop_incomplete=self._drop_incomplete,
                )
                driver.start_express()
                self.get_logger().info(
                    "Express scan started: %d us per sample (standard %d us), target %.1f Hz, motor PWM %d%s"
                    % (
                        express_us,
                        std_us,
                        self._target_rate_hz,
                        regulator.pwm,
                        " (regulated)" if regulator.enabled else " (fixed)",
                    )
                )
                last_data = time.monotonic()
                self._express_stats_reset()
                while not self._stop_event.is_set() and rclpy.ok():
                    data, host_ns = driver.read_chunk()
                    now = time.monotonic()
                    if not data:
                        if now - last_data > 2.0:
                            raise RuntimeError("no data from the LiDAR for 2 s")
                        continue
                    last_data = now
                    for stamped in pipeline.feed(data, host_ns):
                        self._publish_express_revolution(stamped)
                        new_pwm = regulator.update(stamped.period_s)
                        if new_pwm is not None:
                            driver.set_pwm(new_pwm)
                    self._maybe_log_express(pipeline, regulator)
            except Exception as exc:
                if time.monotonic() - session_start > 60.0:
                    failures = 0
                failures += 1
                delay = min(self._retry_max_s, 2.0 ** (failures - 1))
                self.get_logger().error(
                    "RPLIDAR Express failure (port=%s baudrate=%d): %s; retry %d in %.0f s"
                    % (self._port, self._baudrate, exc, failures, delay)
                )
                driver.close()
                self._sleep_with_stop(delay)
            finally:
                driver.close()

    def _sleep_with_stop(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _publish_express_revolution(self, stamped: StampedRevolution) -> None:
        rev = stamped.revolution
        stamp_msg = RosTime(nanoseconds=int(stamped.stamp_ns)).to_msg()
        period = float(stamped.period_s)
        if self._publish_scan_raw:
            ranges = bin_revolution(
                rev.angles_deg,
                rev.dist_mm,
                heading_offset_deg=self._heading_offset_deg,
                range_min_m=self._raw_range_min,
                range_max_m=self._raw_range_max,
            )
            beams = len(ranges)
            inc = 2.0 * math.pi / beams
            msg = LaserScan()
            msg.header.stamp = stamp_msg
            msg.header.frame_id = self._frame_id
            msg.angle_min = -math.pi
            msg.angle_max = -math.pi + inc * (beams - 1)
            msg.angle_increment = inc
            # Nominal time per bin; the firing order (clockwise from +89 deg) is
            # part of the LiDAR calibration, as in the simulation.
            msg.time_increment = period / beams
            msg.scan_time = period
            msg.range_min = self._raw_range_min
            msg.range_max = self._raw_range_max
            msg.ranges = ranges.tolist()
            self._scan_raw_publisher.publish(msg)
            self._express_stats["valid_bins"] += int(np.isfinite(ranges).sum())
        scan = legacy_measurements(rev.angles_deg, rev.dist_mm)
        ranges = self._scan_buffer.update_from_rplidar_scan(scan)
        self._publisher.publish(
            self._build_scan_msg(ranges, stream_key="navigation", stamp_msg=stamp_msg, scan_time_s=period)
        )
        self._localization_publisher.publish(
            self._build_scan_msg(
                self._build_localization_ranges(scan),
                stream_key="localization",
                stamp_msg=stamp_msg,
                scan_time_s=period,
            )
        )
        self._slam_publisher.publish(
            self._build_scan_msg(
                self._build_slam_ranges(scan),
                stream_key="slam",
                stamp_msg=stamp_msg,
                angle_min_rad=-math.pi,
                scan_time_s=period,
            )
        )
        self._express_stats["revs"] += 1
        self._express_stats["samples"] += rev.n_samples
        self._express_stats["period_sum"] += period

    def _express_stats_reset(self) -> None:
        self._express_stats = {"t0": time.monotonic(), "revs": 0, "samples": 0, "period_sum": 0.0, "valid_bins": 0}

    def _maybe_log_express(self, pipeline: ExpressScanPipeline, regulator: MotorRegulator) -> None:
        st = self._express_stats
        if time.monotonic() - st["t0"] < self._timing_log_every_s:
            return
        revs = max(1, st["revs"])
        rate = revs / st["period_sum"] if st["period_sum"] > 0.0 else float("nan")
        c = pipeline.counters
        clock = pipeline.mapper.stats()
        self.get_logger().info(
            "LiDAR timing: %.2f Hz (target %.1f, PWM %d), %.0f samples/rev, %.0f valid bins/rev, "
            "published %d, dropped incomplete %d irregular %d, checksum errors %d, lost capsules %d, "
            "restarts %d, sample clock drift %+.0f ppm, reception latency above envelope p50 %.2f ms p95 %.2f ms"
            % (
                rate,
                self._target_rate_hz,
                regulator.pwm,
                st["samples"] / revs,
                st["valid_bins"] / revs,
                c.published,
                c.dropped_incomplete,
                c.dropped_irregular,
                pipeline.stream.checksum_errors,
                pipeline.decoder.lost_capsules,
                c.restarts,
                clock.drift_ppm,
                clock.latency_p50_ms,
                clock.latency_p95_ms,
            )
        )
        if math.isfinite(rate) and abs(rate - self._target_rate_hz) > 0.03 * self._target_rate_hz:
            self.get_logger().warning(
                "LiDAR at %.2f Hz, more than 3 %% off the %.1f Hz the odometry was trained for"
                % (rate, self._target_rate_hz)
            )
        self._express_stats_reset()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._sim_flush_timer is not None:
            try:
                self._sim_flush_timer.cancel()
            except Exception:
                pass
        self._disconnect_lidar()


def main() -> None:
    rclpy.init(args=None)
    node = RPLidarPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
