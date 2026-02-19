#!/usr/bin/env python3
"""ROS 2 node for PC: subscribes to LaserScan and prints/plots measurements."""

from __future__ import annotations

import argparse
import time
from typing import Optional

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


def _safe_value_by_angle(msg: LaserScan, ranges: np.ndarray, target_deg: float) -> float:
    if ranges.size == 0 or msg.angle_increment == 0.0:
        return 0.0

    target_rad = np.deg2rad(target_deg)
    idx = int(round((target_rad - msg.angle_min) / msg.angle_increment))
    return _safe_value(ranges, idx)


class PointCloudPlotter:
    """Live XY point-cloud view from LaserScan ranges."""

    def __init__(self, max_range_m: float, draw_every_s: float) -> None:
        self._draw_every_s = max(0.02, float(draw_every_s))
        self._last_draw = 0.0
        self._base_limit_m = max(0.5, float(max_range_m))

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for --plot. Install it in your PC venv with "
                "`python -m pip install matplotlib`."
            ) from exc

        self._plt = plt
        self._plt.ion()
        self._fig, self._ax = self._plt.subplots(figsize=(7, 7))
        self._scatter = self._ax.scatter([], [], s=7, c="tab:cyan", alpha=0.85)
        self._ax.scatter([0.0], [0.0], s=35, c="tab:red", label="LiDAR")
        self._ax.set_title("Live LiDAR Point Cloud (XY)")
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.grid(True, alpha=0.25)
        self._ax.legend(loc="upper right")
        self._set_axes_limit(self._base_limit_m)
        self._fig.tight_layout()
        self._draw()

    def _set_axes_limit(self, limit_m: float) -> None:
        self._ax.set_xlim(-limit_m, limit_m)
        self._ax.set_ylim(-limit_m, limit_m)

    def _draw(self) -> None:
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def update(self, angles: np.ndarray, ranges: np.ndarray) -> None:
        now = time.time()
        if now - self._last_draw < self._draw_every_s:
            return
        self._last_draw = now

        if ranges.size == 0:
            self._scatter.set_offsets(np.empty((0, 2), dtype=np.float32))
            self._draw()
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.column_stack((x, y))
        self._scatter.set_offsets(points)

        dynamic_limit = float(np.percentile(ranges, 95)) * 1.2
        if np.isfinite(dynamic_limit):
            self._set_axes_limit(max(self._base_limit_m, max(1.0, dynamic_limit)))

        self._draw()

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)


class LidarSubscriberNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pc_lidar_subscriber")

        self._topic = args.topic
        self._full = args.full
        self._print_every_s = max(0.1, float(args.print_every_s))
        self._last_print = 0.0
        self._plotter: Optional[PointCloudPlotter] = None

        if args.plot:
            self._plotter = PointCloudPlotter(
                max_range_m=args.plot_max_range,
                draw_every_s=args.plot_every_s,
            )
            self.get_logger().info("Live point-cloud plotting enabled.")

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

        valid_mask = np.isfinite(ranges) & (ranges > 0.0)
        if msg.range_min > 0.0:
            valid_mask = valid_mask & (ranges >= msg.range_min)
        if msg.range_max > 0.0:
            valid_mask = valid_mask & (ranges <= msg.range_max)

        valid = ranges[valid_mask]
        if valid.size == 0:
            self.get_logger().warning("Sin puntos validos en este scan")
            return

        if self._plotter is not None:
            angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            self._plotter.update(angles[valid_mask], valid)

        front = _safe_value_by_angle(msg, ranges, 0.0)
        left = _safe_value_by_angle(msg, ranges, 90.0)
        right = _safe_value_by_angle(msg, ranges, -90.0)

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

    def shutdown(self) -> None:
        if self._plotter is not None:
            self._plotter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 LaserScan subscriber for PC/WSL")
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a live 2D XY point-cloud window",
    )
    parser.add_argument(
        "--plot-max-range",
        type=float,
        default=3.0,
        help="Minimum axis range (meters) for the plot",
    )
    parser.add_argument(
        "--plot-every-s",
        type=float,
        default=0.1,
        help="Minimum plotting period (seconds)",
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
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
