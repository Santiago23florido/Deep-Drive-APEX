#!/usr/bin/env python3
"""Capture raw IMU and LiDAR scan points to CSV for a short straight test."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


class RectSensorCaptureNode(Node):
    def __init__(
        self,
        imu_topic: str,
        scan_topic: str,
        imu_output: str,
        lidar_output: str,
        duration_s: float,
    ) -> None:
        super().__init__("rect_sensorfus_capture")
        self._duration_s = max(0.1, float(duration_s))
        self._start_monotonic = time.monotonic()
        self._imu_count = 0
        self._scan_count = 0

        os.makedirs(os.path.dirname(imu_output), exist_ok=True)
        os.makedirs(os.path.dirname(lidar_output), exist_ok=True)

        self._imu_file = open(imu_output, "w", newline="", encoding="utf-8")
        self._lidar_file = open(lidar_output, "w", newline="", encoding="utf-8")
        self._imu_writer = csv.writer(self._imu_file)
        self._lidar_writer = csv.writer(self._lidar_file)

        self._imu_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "gx_rps",
                "gy_rps",
                "gz_rps",
            ]
        )
        self._lidar_writer.writerow(
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

        self.create_subscription(
            Imu,
            imu_topic,
            self._imu_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            "Rect sensor capture started (imu=%s scan=%s duration=%.2fs)"
            % (imu_topic, scan_topic, self._duration_s)
        )

    def _imu_callback(self, msg: Imu) -> None:
        self._imu_writer.writerow(
            [
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                float(msg.linear_acceleration.x),
                float(msg.linear_acceleration.y),
                float(msg.linear_acceleration.z),
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ]
        )
        self._imu_count += 1
        if self._imu_count % 50 == 0:
            self._imu_file.flush()

    def _scan_callback(self, msg: LaserScan) -> None:
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        scan_index = self._scan_count
        angle = float(msg.angle_min)
        wrote_points = False
        for point_index, raw_range in enumerate(msg.ranges):
            range_m = float(raw_range)
            if not math.isfinite(range_m):
                angle += float(msg.angle_increment)
                continue
            if range_m < float(msg.range_min) or range_m > float(msg.range_max):
                angle += float(msg.angle_increment)
                continue
            x_scan_m = range_m * math.cos(angle)
            y_scan_m = range_m * math.sin(angle)
            x_forward_m = -x_scan_m
            y_left_m = y_scan_m
            self._lidar_writer.writerow(
                [
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
            )
            wrote_points = True
            angle += float(msg.angle_increment)
        self._scan_count += 1
        if wrote_points:
            self._lidar_file.flush()

    def _tick(self) -> None:
        if (time.monotonic() - self._start_monotonic) < self._duration_s:
            return
        self.get_logger().info(
            "Rect sensor capture finished (imu_rows=%d lidar_scans=%d)"
            % (self._imu_count, self._scan_count)
        )
        self._imu_file.flush()
        self._lidar_file.flush()
        self._imu_file.close()
        self._lidar_file.close()
        raise SystemExit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-topic", default="/apex/imu/data_raw")
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--imu-output", required=True)
    parser.add_argument("--lidar-output", required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = RectSensorCaptureNode(
        imu_topic=args.imu_topic,
        scan_topic=args.scan_topic,
        imu_output=args.imu_output,
        lidar_output=args.lidar_output,
        duration_s=args.duration_s,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
