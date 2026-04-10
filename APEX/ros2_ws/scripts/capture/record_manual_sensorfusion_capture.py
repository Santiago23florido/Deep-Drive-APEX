#!/usr/bin/env python3
"""Capture raw IMU and LiDAR continuously until interrupted."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


class ManualSensorCaptureNode(Node):
    def __init__(
        self,
        *,
        imu_topic: str,
        scan_topic: str,
        odom_topic: str | None,
        imu_output: Path,
        lidar_output: Path,
        odom_output: Path | None,
        summary_json: Path,
        status_json: Path | None,
        max_total_bytes: int | None,
        min_free_disk_bytes: int | None,
    ) -> None:
        super().__init__("manual_sensorfusion_capture")
        self._start_monotonic = time.monotonic()
        self._stop_requested = False
        self._imu_count = 0
        self._scan_count = 0
        self._lidar_point_count = 0
        self._odom_count = 0
        self._summary_json = summary_json
        self._status_json = status_json
        self._finalized = False
        self._end_cause = "finished"
        self._output_root_dir = imu_output.parent
        self._max_total_bytes = (
            max(1, int(max_total_bytes)) if max_total_bytes is not None and int(max_total_bytes) > 0 else None
        )
        self._min_free_disk_bytes = (
            max(1, int(min_free_disk_bytes))
            if min_free_disk_bytes is not None and int(min_free_disk_bytes) > 0
            else None
        )
        self._last_csv_total_bytes = 0
        self._last_free_disk_bytes: int | None = None

        imu_output.parent.mkdir(parents=True, exist_ok=True)
        lidar_output.parent.mkdir(parents=True, exist_ok=True)
        if odom_output is not None:
            odom_output.parent.mkdir(parents=True, exist_ok=True)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        if status_json is not None:
            status_json.parent.mkdir(parents=True, exist_ok=True)

        self._imu_file = imu_output.open("w", newline="", encoding="utf-8")
        self._lidar_file = lidar_output.open("w", newline="", encoding="utf-8")
        self._odom_file = (
            odom_output.open("w", newline="", encoding="utf-8")
            if odom_output is not None
            else None
        )
        self._imu_writer = csv.writer(self._imu_file)
        self._lidar_writer = csv.writer(self._lidar_file)
        self._odom_writer = csv.writer(self._odom_file) if self._odom_file is not None else None

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
        if self._odom_writer is not None:
            self._odom_writer.writerow(
                [
                    "stamp_sec",
                    "stamp_nanosec",
                    "x_m",
                    "y_m",
                    "yaw_rad",
                    "vx_mps",
                    "vy_mps",
                    "yaw_rate_rps",
                ]
            )

        self.create_subscription(Imu, imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        if odom_topic and self._odom_writer is not None:
            self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._write_live_status)
        self.get_logger().info(
            "Manual sensor capture started (imu=%s scan=%s odom=%s imu_csv=%s lidar_csv=%s odom_csv=%s)"
            % (
                imu_topic,
                scan_topic,
                str(odom_topic or ""),
                str(imu_output),
                str(lidar_output),
                str(odom_output) if odom_output is not None else "",
            )
        )

    def request_stop(self, cause: str | None = None) -> None:
        if cause:
            self._end_cause = str(cause)
        self._stop_requested = True

    def _current_csv_total_bytes(self) -> int:
        total_bytes = 0
        for handle in (self._imu_file, self._lidar_file, self._odom_file):
            if handle is None or handle.closed:
                continue
            try:
                handle.flush()
            except Exception:
                pass
            try:
                total_bytes += int(os.fstat(handle.fileno()).st_size)
            except Exception:
                pass
        self._last_csv_total_bytes = total_bytes
        return total_bytes

    def _free_disk_bytes(self) -> int | None:
        try:
            stats = os.statvfs(str(self._output_root_dir))
        except Exception:
            return None
        free_bytes = int(stats.f_bavail) * int(stats.f_frsize)
        self._last_free_disk_bytes = free_bytes
        return free_bytes

    def _check_limits(self) -> None:
        if self._finalized:
            return
        total_bytes = self._current_csv_total_bytes()
        free_bytes = self._free_disk_bytes()
        if self._max_total_bytes is not None and total_bytes >= self._max_total_bytes:
            if not self._stop_requested:
                self.get_logger().error(
                    "Stopping manual sensor capture because csv_total_bytes=%d reached max_total_bytes=%d"
                    % (total_bytes, self._max_total_bytes)
                )
            self.request_stop("storage_limit")
            return
        if self._min_free_disk_bytes is not None and free_bytes is not None and free_bytes <= self._min_free_disk_bytes:
            if not self._stop_requested:
                self.get_logger().error(
                    "Stopping manual sensor capture because free_disk_bytes=%d fell below min_free_disk_bytes=%d"
                    % (free_bytes, self._min_free_disk_bytes)
                )
            self.request_stop("low_disk_space")

    def _imu_cb(self, msg: Imu) -> None:
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
            self._check_limits()

    def _scan_cb(self, msg: LaserScan) -> None:
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        scan_index = self._scan_count
        angle_rad = float(msg.angle_min)
        wrote_points = False
        for point_index, raw_range in enumerate(msg.ranges):
            range_m = float(raw_range)
            if not math.isfinite(range_m):
                angle_rad += float(msg.angle_increment)
                continue
            if range_m < float(msg.range_min) or range_m > float(msg.range_max):
                angle_rad += float(msg.angle_increment)
                continue
            x_scan_m = range_m * math.cos(angle_rad)
            y_scan_m = range_m * math.sin(angle_rad)
            self._lidar_writer.writerow(
                [
                    stamp_sec,
                    stamp_nanosec,
                    scan_index,
                    point_index,
                    angle_rad,
                    range_m,
                    x_scan_m,
                    y_scan_m,
                    -x_scan_m,
                    y_scan_m,
                ]
            )
            wrote_points = True
            self._lidar_point_count += 1
            angle_rad += float(msg.angle_increment)
        self._scan_count += 1
        if wrote_points:
            self._lidar_file.flush()
            self._check_limits()

    def _odom_cb(self, msg: Odometry) -> None:
        if self._odom_writer is None:
            return
        q = msg.pose.pose.orientation
        yaw_rad = math.atan2(
            2.0 * ((float(q.w) * float(q.z)) + (float(q.x) * float(q.y))),
            1.0 - (2.0 * ((float(q.y) * float(q.y)) + (float(q.z) * float(q.z)))),
        )
        self._odom_writer.writerow(
            [
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw_rad,
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.angular.z),
            ]
        )
        self._odom_count += 1
        if (self._odom_count % 25) == 0 and self._odom_file is not None:
            self._odom_file.flush()
            self._check_limits()

    def _tick(self) -> None:
        if not self._stop_requested:
            return
        self.finalize()
        raise SystemExit(0)

    def _write_live_status(self) -> None:
        if self._status_json is None or self._finalized:
            return
        duration_s = time.monotonic() - self._start_monotonic
        payload = {
            "duration_s": duration_s,
            "imu_row_count": self._imu_count,
            "scan_count": self._scan_count,
            "lidar_point_count": self._lidar_point_count,
            "odom_row_count": self._odom_count,
            "csv_total_bytes": self._current_csv_total_bytes(),
            "max_total_bytes": self._max_total_bytes,
            "free_disk_bytes": self._free_disk_bytes(),
            "min_free_disk_bytes": self._min_free_disk_bytes,
            "end_cause": self._end_cause,
            "state": "stopping" if self._stop_requested else "capturing",
        }
        try:
            self._status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        self._check_limits()

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        duration_s = time.monotonic() - self._start_monotonic
        self._current_csv_total_bytes()
        self._free_disk_bytes()
        try:
            self._imu_file.flush()
            self._lidar_file.flush()
            self._imu_file.close()
            self._lidar_file.close()
            if self._odom_file is not None:
                self._odom_file.flush()
                self._odom_file.close()
        except Exception:
            pass
        payload = {
            "duration_s": duration_s,
            "imu_row_count": self._imu_count,
            "scan_count": self._scan_count,
            "lidar_point_count": self._lidar_point_count,
            "odom_row_count": self._odom_count,
            "csv_total_bytes": self._last_csv_total_bytes,
            "max_total_bytes": self._max_total_bytes,
            "free_disk_bytes": self._last_free_disk_bytes,
            "min_free_disk_bytes": self._min_free_disk_bytes,
            "end_cause": self._end_cause,
            "state": "finished",
        }
        self._summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._status_json is not None:
            try:
                self._status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
        self.get_logger().info(
            "Manual sensor capture finished (cause=%s duration=%.2fs imu=%d scans=%d points=%d odom=%d csv_total_bytes=%d)"
            % (
                self._end_cause,
                duration_s,
                self._imu_count,
                self._scan_count,
                self._lidar_point_count,
                self._odom_count,
                self._last_csv_total_bytes,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-topic", default="/apex/imu/data_raw")
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--odom-topic", default="/apex/odometry/imu_lidar_fused")
    parser.add_argument("--imu-output", required=True)
    parser.add_argument("--lidar-output", required=True)
    parser.add_argument("--odom-output", default="")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--max-total-bytes", type=int, default=0)
    parser.add_argument("--min-free-disk-bytes", type=int, default=0)
    args = parser.parse_args()

    rclpy.init()
    node = ManualSensorCaptureNode(
        imu_topic=args.imu_topic,
        scan_topic=args.scan_topic,
        odom_topic=(str(args.odom_topic).strip() or None),
        imu_output=Path(args.imu_output).expanduser().resolve(),
        lidar_output=Path(args.lidar_output).expanduser().resolve(),
        odom_output=(
            Path(args.odom_output).expanduser().resolve()
            if str(args.odom_output).strip()
            else None
        ),
        summary_json=Path(args.summary_json).expanduser().resolve(),
        status_json=(
            Path(args.status_json).expanduser().resolve() if str(args.status_json).strip() else None
        ),
        max_total_bytes=(int(args.max_total_bytes) if int(args.max_total_bytes) > 0 else None),
        min_free_disk_bytes=(
            int(args.min_free_disk_bytes) if int(args.min_free_disk_bytes) > 0 else None
        ),
    )

    def _signal_handler(_signum, _frame) -> None:
        node.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.finalize()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
