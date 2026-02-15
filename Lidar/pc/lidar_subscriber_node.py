#!/usr/bin/env python3
"""ROS2 node for PC: subscribes to LaserScan and prints measurements."""

from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def _safe_value(ranges: np.ndarray, idx: int) -> float:
    value = float(ranges[idx % ranges.size])
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return value


class LidarSubscriberNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pc_lidar_subscriber")

        self._topic = args.topic
        self._full = args.full
        self._print_every_s = max(0.1, float(args.print_every_s))
        self._last_print = 0.0

        self.create_subscription(LaserScan, self._topic, self._scan_cb, qos_profile_sensor_data)
        self.get_logger().info(f"Suscrito a {self._topic}")

    def _scan_cb(self, msg: LaserScan) -> None:
        now = time.time()
        if now - self._last_print < self._print_every_s:
            return

        self._last_print = now

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            self.get_logger().warning("Scan vacio")
            return

        valid = ranges[np.isfinite(ranges) & (ranges > 0.0)]
        if valid.size == 0:
            self.get_logger().warning("Sin puntos validos en este scan")
            return

        front = _safe_value(ranges, 0)
        left = _safe_value(ranges, 90)
        right = _safe_value(ranges, 270)

        self.get_logger().info(
            "Puntos=%d | min=%.3fm avg=%.3fm max=%.3fm | front=%.3fm left=%.3fm right=%.3fm"
            % (
                valid.size,
                float(np.min(valid)),
                float(np.mean(valid)),
                float(np.max(valid)),
                front,
                left,
                right,
            )
        )

        if self._full:
            rounded = np.round(ranges, 3)
            self.get_logger().info(f"ranges[360]: {rounded.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nodo ROS2 subscriber de LaserScan para PC")
    parser.add_argument("--topic", default="/lidar/scan", help="Topico LaserScan de la Raspberry")
    parser.add_argument(
        "--print-every-s",
        type=float,
        default=0.5,
        help="Periodo minimo entre impresiones en consola",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Imprime el arreglo completo de 360 mediciones",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)

    node = LidarSubscriberNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
