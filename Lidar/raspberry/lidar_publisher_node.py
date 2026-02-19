#!/usr/bin/env python3
"""ROS2 node for Raspberry Pi: reads RPLidar and publishes LaserScan."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

_THIS_DIR = Path(__file__).resolve().parent
_LIDAR_ROOT = _THIS_DIR.parent
if str(_LIDAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIDAR_ROOT))

from common import LidarScanBuffer

try:
    from rplidar import RPLidar, RPLidarException
except Exception:
    RPLidar = None
    RPLidarException = Exception


class RPLidarPublisherNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("rpi_lidar_publisher")

        if RPLidar is None:
            raise RuntimeError("No se pudo importar rplidar. Instala el paquete en la Raspberry.")

        self._port = args.port
        self._baudrate = args.baudrate
        self._topic = args.topic
        self._frame_id = args.frame_id
        self._range_min = args.range_min
        self._range_max = args.range_max

        self._scan_buffer = LidarScanBuffer(
            samples=360,
            heading_offset_deg=args.heading_offset_deg,
            fov_filter_deg=args.fov_filter_deg,
            point_timeout_ms=args.point_timeout_ms,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nodo ROS2 publisher para RPLidar en Raspberry Pi")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Puerto serial del LiDAR")
    parser.add_argument("--baudrate", type=int, default=256000, help="Baudrate serial")
    parser.add_argument("--topic", default="/lidar/scan", help="Topico LaserScan a publicar")
    parser.add_argument("--frame-id", default="laser", help="frame_id del LaserScan")
    parser.add_argument("--heading-offset-deg", type=int, default=0, help="Offset angular en grados")
    parser.add_argument("--fov-filter-deg", type=int, default=360, help="Campo de vision util")
    parser.add_argument("--point-timeout-ms", type=int, default=200, help="Timeout por angulo")
    parser.add_argument("--range-min", type=float, default=0.05, help="Distancia minima valida (m)")
    parser.add_argument("--range-max", type=float, default=12.0, help="Distancia maxima valida (m)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)

    node = RPLidarPublisherNode(args)
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
