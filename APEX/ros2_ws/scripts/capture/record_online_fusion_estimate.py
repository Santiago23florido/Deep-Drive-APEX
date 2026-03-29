#!/usr/bin/env python3
"""Record online IMU+LiDAR fusion outputs for one straight-run validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


class OnlineFusionRecorder(Node):
    def __init__(
        self,
        *,
        odom_topic: str,
        status_topic: str,
        output_dir: Path,
        duration_s: float,
    ) -> None:
        super().__init__("record_online_fusion_estimate")
        self._duration_s = max(0.5, float(duration_s))
        self._start_monotonic = time.monotonic()
        self._status_payload: dict[str, Any] = {}
        self._status_count = 0
        self._odom_count = 0
        self._last_position: tuple[float, float] | None = None
        self._distance_m = 0.0
        self._finalized = False

        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_path = self._output_dir / "online_fusion_trajectory.csv"
        self._summary_path = self._output_dir / "online_fusion_summary.json"
        self._trajectory_file = self._trajectory_path.open("w", newline="", encoding="utf-8")
        self._trajectory_writer = csv.writer(self._trajectory_file)
        self._trajectory_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "x_m",
                "y_m",
                "yaw_rad",
                "vx_mps",
                "vy_mps",
                "yaw_rate_rps",
                "confidence",
                "median_submap_residual_m",
                "median_wall_residual_m",
                "valid_correspondence_count",
                "alignment_ready",
                "imu_initialized",
                "best_effort_init",
            ]
        )

        self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.create_subscription(String, status_topic, self._status_cb, 20)
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            "Online fusion recorder started (odom=%s status=%s duration=%.2fs output=%s)"
            % (odom_topic, status_topic, self._duration_s, str(self._output_dir))
        )

    def _status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._status_payload = payload
        self._status_count += 1

    def _odom_cb(self, msg: Odometry) -> None:
        x_m = float(msg.pose.pose.position.x)
        y_m = float(msg.pose.pose.position.y)
        yaw_rad = _quat_to_yaw(
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        )
        vx_mps = float(msg.twist.twist.linear.x)
        vy_mps = float(msg.twist.twist.linear.y)
        yaw_rate_rps = float(msg.twist.twist.angular.z)

        latest_pose = self._status_payload.get("latest_pose") or {}
        confidence = str(latest_pose.get("confidence", "unknown"))
        median_submap_residual_m = latest_pose.get("median_submap_residual_m", "")
        median_wall_residual_m = latest_pose.get("median_wall_residual_m", "")
        valid_correspondence_count = latest_pose.get("valid_correspondence_count", "")
        alignment_ready = bool(self._status_payload.get("alignment_ready", False))
        imu_initialized = bool(self._status_payload.get("imu_initialized", False))
        best_effort_init = bool(self._status_payload.get("best_effort_init", False))

        if self._last_position is not None:
            self._distance_m += math.hypot(x_m - self._last_position[0], y_m - self._last_position[1])
        self._last_position = (x_m, y_m)

        self._trajectory_writer.writerow(
            [
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                f"{x_m:.6f}",
                f"{y_m:.6f}",
                f"{yaw_rad:.6f}",
                f"{vx_mps:.6f}",
                f"{vy_mps:.6f}",
                f"{yaw_rate_rps:.6f}",
                confidence,
                median_submap_residual_m,
                median_wall_residual_m,
                valid_correspondence_count,
                int(alignment_ready),
                int(imu_initialized),
                int(best_effort_init),
            ]
        )
        self._trajectory_file.flush()
        self._odom_count += 1

    def _write_summary(self) -> None:
        trajectory_rows = []
        with self._trajectory_path.open(newline="", encoding="utf-8") as handle:
            trajectory_rows = list(csv.DictReader(handle))

        final_pose = {
            "x_m": None,
            "y_m": None,
            "yaw_rad": None,
        }
        high_confidence_count = 0
        if trajectory_rows:
            last_row = trajectory_rows[-1]
            final_pose = {
                "x_m": float(last_row["x_m"]),
                "y_m": float(last_row["y_m"]),
                "yaw_rad": float(last_row["yaw_rad"]),
            }
            high_confidence_count = sum(
                1 for row in trajectory_rows if str(row["confidence"]).strip().lower() == "high"
            )

        summary = {
            "trajectory_csv": str(self._trajectory_path),
            "trajectory_row_count": len(trajectory_rows),
            "status_message_count": self._status_count,
            "final_pose": final_pose,
            "distance_m": self._distance_m,
            "high_confidence_pct": (
                100.0 * high_confidence_count / max(1, len(trajectory_rows))
            ),
            "static_initialization": self._status_payload.get("static_initialization"),
            "corridor_model": self._status_payload.get("corridor_model"),
            "quality": self._status_payload.get("quality"),
            "parameters": self._status_payload.get("parameters"),
            "state": self._status_payload.get("state"),
            "alignment_ready": bool(self._status_payload.get("alignment_ready", False)),
            "imu_initialized": bool(self._status_payload.get("imu_initialized", False)),
            "best_effort_init": bool(self._status_payload.get("best_effort_init", False)),
        }
        self._summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            if not self._trajectory_file.closed:
                self._trajectory_file.flush()
                self._trajectory_file.close()
        except Exception:
            pass

        try:
            self._write_summary()
        except Exception as exc:
            self.get_logger().error(f"Failed to write online fusion summary: {exc}")
            raise

        self.get_logger().info(
            "Online fusion recorder finished (odom_rows=%d status_msgs=%d)"
            % (self._odom_count, self._status_count)
        )

    def _tick(self) -> None:
        if (time.monotonic() - self._start_monotonic) < self._duration_s:
            return
        self.finalize()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/apex/odometry/imu_lidar_fused")
    parser.add_argument("--status-topic", default="/apex/estimation/status")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    rclpy.init()
    node = OnlineFusionRecorder(
        odom_topic=args.odom_topic,
        status_topic=args.status_topic,
        output_dir=output_dir,
        duration_s=args.duration_s,
    )
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
