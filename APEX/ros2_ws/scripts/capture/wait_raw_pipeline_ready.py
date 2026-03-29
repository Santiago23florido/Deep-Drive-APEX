#!/usr/bin/env python3
"""Wait for stable raw IMU and LiDAR streams before starting a capture run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


class RawPipelineReadyNode(Node):
    def __init__(
        self,
        imu_topic: str,
        scan_topic: str,
        timeout_s: float,
        min_imu_messages: int,
        min_scan_messages: int,
        json_output: Path | None,
    ) -> None:
        super().__init__("wait_raw_pipeline_ready")
        self._start_monotonic = time.monotonic()
        self._last_status_log_t = self._start_monotonic
        self._timeout_s = max(0.5, float(timeout_s))
        self._min_imu_messages = max(1, int(min_imu_messages))
        self._min_scan_messages = max(1, int(min_scan_messages))
        self._json_output = json_output
        self._imu_count = 0
        self._scan_count = 0
        self._imu_first_t: float | None = None
        self._imu_last_t: float | None = None
        self._scan_first_t: float | None = None
        self._scan_last_t: float | None = None

        self.create_subscription(Imu, imu_topic, self._imu_callback, qos_profile_sensor_data)
        self.create_subscription(LaserScan, scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            "Waiting for raw pipeline readiness (imu=%s scan=%s timeout=%.2fs imu_min=%d scan_min=%d)"
            % (
                imu_topic,
                scan_topic,
                self._timeout_s,
                self._min_imu_messages,
                self._min_scan_messages,
            )
        )

    def _imu_callback(self, _msg: Imu) -> None:
        now_t = time.monotonic()
        if self._imu_first_t is None:
            self._imu_first_t = now_t
        self._imu_last_t = now_t
        self._imu_count += 1

    def _scan_callback(self, _msg: LaserScan) -> None:
        now_t = time.monotonic()
        if self._scan_first_t is None:
            self._scan_first_t = now_t
        self._scan_last_t = now_t
        self._scan_count += 1

    def _is_ready(self) -> bool:
        return (
            self._imu_count >= self._min_imu_messages
            and self._scan_count >= self._min_scan_messages
        )

    def _summary(self, *, ready: bool) -> dict[str, object]:
        elapsed_s = time.monotonic() - self._start_monotonic
        return {
            "ready": ready,
            "elapsed_s": elapsed_s,
            "imu": {
                "count": self._imu_count,
                "required": self._min_imu_messages,
                "first_receipt_offset_s": None
                if self._imu_first_t is None
                else self._imu_first_t - self._start_monotonic,
                "last_receipt_offset_s": None
                if self._imu_last_t is None
                else self._imu_last_t - self._start_monotonic,
            },
            "lidar": {
                "count": self._scan_count,
                "required": self._min_scan_messages,
                "first_receipt_offset_s": None
                if self._scan_first_t is None
                else self._scan_first_t - self._start_monotonic,
                "last_receipt_offset_s": None
                if self._scan_last_t is None
                else self._scan_last_t - self._start_monotonic,
            },
        }

    def _write_summary(self, *, ready: bool) -> None:
        if self._json_output is None:
            return
        self._json_output.parent.mkdir(parents=True, exist_ok=True)
        with self._json_output.open("w", encoding="utf-8") as handle:
            json.dump(self._summary(ready=ready), handle, indent=2)

    def _tick(self) -> None:
        now_t = time.monotonic()
        elapsed_s = now_t - self._start_monotonic

        if self._is_ready():
            self._write_summary(ready=True)
            self.get_logger().info(
                "Raw pipeline ready after %.2fs (imu=%d lidar=%d)"
                % (elapsed_s, self._imu_count, self._scan_count)
            )
            raise SystemExit(0)

        if (now_t - self._last_status_log_t) >= 1.0:
            self._last_status_log_t = now_t
            self.get_logger().info(
                "Still waiting for readiness after %.2fs (imu=%d/%d lidar=%d/%d)"
                % (
                    elapsed_s,
                    self._imu_count,
                    self._min_imu_messages,
                    self._scan_count,
                    self._min_scan_messages,
                )
            )

        if elapsed_s >= self._timeout_s:
            self._write_summary(ready=False)
            self.get_logger().error(
                "Timed out waiting for raw pipeline readiness (imu=%d/%d lidar=%d/%d)"
                % (
                    self._imu_count,
                    self._min_imu_messages,
                    self._scan_count,
                    self._min_scan_messages,
                )
            )
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-topic", default="/apex/imu/data_raw")
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--min-imu-messages", type=int, default=5)
    parser.add_argument("--min-scan-messages", type=int, default=3)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    json_output = Path(args.json_output) if args.json_output else None

    rclpy.init()
    node = RawPipelineReadyNode(
        imu_topic=args.imu_topic,
        scan_topic=args.scan_topic,
        timeout_s=args.timeout_s,
        min_imu_messages=args.min_imu_messages,
        min_scan_messages=args.min_scan_messages,
        json_output=json_output,
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
