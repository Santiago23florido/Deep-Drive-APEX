#!/usr/bin/env python3
"""ROS2 node for Raspberry Pi: reads RPLidar and publishes LaserScan."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .lidar_scan_buffer import LidarScanBuffer

try:
    from rplidar import RPLidar, RPLidarException
except Exception:
    RPLidar = None
    RPLidarException = Exception


class RPLidarPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("rpi_lidar_publisher")

        if RPLidar is None:
            raise RuntimeError("No se pudo importar rplidar. Instala `rplidar-roboticia`.")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 256000)
        self.declare_parameter("topic", "/lidar/scan")
        self.declare_parameter("frame_id", "laser")
        # Keep defaults aligned with the original full_soft LiDAR configuration.
        self.declare_parameter("heading_offset_deg", -89)
        self.declare_parameter("fov_filter_deg", 180)
        self.declare_parameter("point_timeout_ms", 1000)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 12.0)

        self._port = str(self.get_parameter("port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._topic = str(self.get_parameter("topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._range_min = float(self.get_parameter("range_min").value)
        self._range_max = float(self.get_parameter("range_max").value)

        self._scan_buffer = LidarScanBuffer(
            samples=360,
            heading_offset_deg=int(self.get_parameter("heading_offset_deg").value),
            fov_filter_deg=int(self.get_parameter("fov_filter_deg").value),
            point_timeout_ms=int(self.get_parameter("point_timeout_ms").value),
        )

        self._publisher = self.create_publisher(LaserScan, self._topic, qos_profile_sensor_data)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._scan_loop, daemon=True)

        self._lidar = None
        self._last_publish_time = None

        self.get_logger().info(
            f"Publicando LiDAR en {self._topic} | port={self._port} baudrate={self._baudrate}"
        )
        self._worker.start()

    def _connect_lidar(self) -> None:
        self._lidar = RPLidar(self._port, baudrate=self._baudrate)

        if hasattr(self._lidar, "connect"):
            self._lidar.connect()

        if hasattr(self._lidar, "_serial") and self._lidar._serial is not None:
            self._lidar._serial.reset_input_buffer()
            self._lidar._serial.reset_output_buffer()

        self._lidar.start_motor()
        self._lidar.start()
        self.get_logger().info("RPLidar conectado y escaneando.")

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

    def _build_scan_msg(self, ranges: np.ndarray) -> LaserScan:
        now = self.get_clock().now().to_msg()
        current_t = time.time()

        if self._last_publish_time is None:
            scan_time = 0.0
        else:
            scan_time = max(0.0, current_t - self._last_publish_time)

        self._last_publish_time = current_t

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

    def _scan_loop(self) -> None:
        while not self._stop_event.is_set() and rclpy.ok():
            try:
                self._connect_lidar()

                for scan in self._lidar.iter_scans(max_buf_meas=500):
                    if self._stop_event.is_set() or not rclpy.ok():
                        break

                    ranges = self._scan_buffer.update_from_rplidar_scan(scan)
                    msg = self._build_scan_msg(ranges)
                    self._publisher.publish(msg)

            except (RPLidarException, OSError, ValueError) as exc:
                self.get_logger().error(f"Error leyendo RPLidar: {exc}")
                time.sleep(1.0)
            except Exception as exc:
                self.get_logger().error(f"Fallo inesperado del LiDAR: {exc}")
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
