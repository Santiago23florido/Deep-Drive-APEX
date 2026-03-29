#!/usr/bin/env python3
"""ROS2 node: read RPLidar on Raspberry and publish LaserScan."""

from __future__ import annotations

import glob
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .lidar_scan_buffer import LidarScanBuffer


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

        if RPLidar is None:
            raise RuntimeError("rplidar module not found. Install `rplidar-roboticia`.")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("topic", "/lidar/scan")
        self.declare_parameter("localization_topic", "/lidar/scan_localization")
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

        self._port = str(self.get_parameter("port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._topic = str(self.get_parameter("topic").value)
        self._localization_topic = str(self.get_parameter("localization_topic").value)
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
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._scan_loop, daemon=True)

        self._lidar = None
        self._last_publish_time_by_stream = {
            "navigation": None,
            "localization": None,
        }

        self.get_logger().info(
            "LiDAR publisher started (port=%s topic=%s localization_topic=%s baudrate=%d "
            "localization_samples=%d max_buf_meas=%d)"
            % (
                self._port,
                self._topic,
                self._localization_topic,
                self._baudrate,
                self._localization_samples,
                self._max_buf_meas,
            )
        )
        self._worker.start()

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
    ) -> LaserScan:
        now = stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg()
        current_t = time.time()

        last_publish_time = self._last_publish_time_by_stream.get(stream_key)
        if last_publish_time is None:
            scan_time = 0.0
        else:
            scan_time = max(0.0, current_t - last_publish_time)
        self._last_publish_time_by_stream[stream_key] = current_t

        samples = len(ranges)
        angle_increment = (2.0 * math.pi) / float(samples)

        msg = LaserScan()
        msg.header.stamp = now
        msg.header.frame_id = self._frame_id
        msg.angle_min = 0.0
        msg.angle_max = (samples - 1) * angle_increment
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

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
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
