#!/usr/bin/env python3
"""Capture static LiDAR data and export both point cloud CSV and snapshot CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class StaticLidarCaptureNode(Node):
    def __init__(
        self,
        scan_topic: str,
        points_output: str,
        snapshot_output: str,
        duration_s: float,
    ) -> None:
        super().__init__("static_lidar_capture")
        self._duration_s = max(0.2, float(duration_s))
        self._start_monotonic = time.monotonic()
        self._scan_count = 0
        self._best_scan_rows: list[list[object]] = []
        self._best_scan_index = -1

        os.makedirs(os.path.dirname(points_output), exist_ok=True)
        os.makedirs(os.path.dirname(snapshot_output), exist_ok=True)

        self._points_file = open(points_output, "w", newline="", encoding="utf-8")
        self._points_writer = csv.writer(self._points_file)
        self._points_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "scan_index",
                "point_index",
                "angle_rad",
                "range_m",
                "x_scan_m",
                "y_scan_m",
                "x_forward_m",
                "y_left_m",
            ]
        )
        self._snapshot_output = snapshot_output

        self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            "Static LiDAR capture started (scan=%s duration=%.2fs)"
            % (scan_topic, self._duration_s)
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        scan_index = self._scan_count
        angle = float(msg.angle_min)
        scan_rows: list[list[object]] = []

        for point_index, raw_range in enumerate(msg.ranges):
            range_m = float(raw_range)
            if math.isfinite(range_m) and float(msg.range_min) <= range_m <= float(msg.range_max):
                x_scan_m = range_m * math.cos(angle)
                y_scan_m = range_m * math.sin(angle)
                x_forward_m = -x_scan_m
                y_left_m = y_scan_m
                row = [
                    stamp_sec,
                    stamp_nanosec,
                    scan_index,
                    point_index,
                    angle,
                    range_m,
                    x_scan_m,
                    y_scan_m,
                    x_forward_m,
                    y_left_m,
                ]
                self._points_writer.writerow(row)
                scan_rows.append(row)
            angle += float(msg.angle_increment)

        self._scan_count += 1
        if scan_rows:
            self._points_file.flush()
            if len(scan_rows) > len(self._best_scan_rows):
                self._best_scan_rows = scan_rows
                self._best_scan_index = scan_index

    def _write_snapshot(self) -> None:
        with open(self._snapshot_output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["angle_deg", "range_m", "count"])
            for row in self._best_scan_rows:
                writer.writerow([f"{math.degrees(float(row[4])):.6f}", f"{float(row[5]):.6f}", 1])

    def _tick(self) -> None:
        if (time.monotonic() - self._start_monotonic) < self._duration_s:
            return

        self._points_file.flush()
        self._points_file.close()

        if not self._best_scan_rows:
            self.get_logger().error("Static LiDAR capture finished without any valid scan")
            raise SystemExit(2)

        self._write_snapshot()
        self.get_logger().info(
            "Static LiDAR capture finished (scans=%d best_scan_index=%d best_points=%d)"
            % (self._scan_count, self._best_scan_index, len(self._best_scan_rows))
        )
        raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--points-output", required=True)
    parser.add_argument("--snapshot-output", required=True)
    parser.add_argument("--duration-s", type=float, default=4.0)
    args = parser.parse_args()

    rclpy.init()
    node = StaticLidarCaptureNode(
        scan_topic=args.scan_topic,
        points_output=args.points_output,
        snapshot_output=args.snapshot_output,
        duration_s=args.duration_s,
    )
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
