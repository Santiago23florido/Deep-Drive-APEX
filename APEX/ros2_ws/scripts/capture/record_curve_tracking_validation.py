#!/usr/bin/env python3
"""Record planner/tracker outputs for one autonomous curve-entry validation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path as FsPath
from typing import Any

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


TERMINAL_STATES = {
    "goal_reached",
    "aborted_low_confidence",
    "aborted_path_loss",
    "aborted_odom_timeout",
    "timeout",
    "planner_failed",
}


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


class CurveTrackingValidationRecorder(Node):
    def __init__(
        self,
        *,
        path_topic: str,
        planner_status_topic: str,
        tracker_status_topic: str,
        bridge_status_topic: str,
        odom_topic: str,
        scan_topic: str,
        output_dir: FsPath,
        timeout_s: float,
        lidar_offset_x_m: float,
        lidar_offset_y_m: float,
    ) -> None:
        super().__init__("record_curve_tracking_validation")

        self._timeout_s = max(1.0, float(timeout_s))
        self._start_monotonic = time.monotonic()
        self._terminal_state_seen_at: float | None = None
        self._finalized = False

        self._planner_status: dict[str, Any] = {}
        self._tracker_status: dict[str, Any] = {}
        self._bridge_status: dict[str, Any] = {}
        self._path_points: list[dict[str, float]] = []
        self._path_written = False
        self._track_started = False
        self._initial_pose: dict[str, float] | None = None
        self._final_pose: dict[str, float] | None = None
        self._trajectory_row_count = 0
        self._scan_count = 0
        self._lidar_point_count = 0
        self._latest_odom_message: Odometry | None = None
        self._lidar_offset_x_m = float(lidar_offset_x_m)
        self._lidar_offset_y_m = float(lidar_offset_y_m)

        self._output_dir = output_dir
        self._analysis_dir = self._output_dir / "analysis_curve_tracking"
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        self._lidar_points_csv = self._output_dir / "lidar_points.csv"
        self._planned_path_csv = self._analysis_dir / "planned_path.csv"
        self._planned_path_json = self._analysis_dir / "planned_path.json"
        self._tracking_csv = self._analysis_dir / "tracking_trajectory.csv"
        self._summary_json = self._analysis_dir / "tracking_summary.json"
        self._status_log = self._analysis_dir / "controller_status.log"
        self._bridge_status_log = self._analysis_dir / "drive_bridge_status.log"

        self._tracking_handle = self._tracking_csv.open("w", newline="", encoding="utf-8")
        self._tracking_writer = csv.writer(self._tracking_handle)
        self._tracking_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "x_m",
                "y_m",
                "yaw_rad",
                "vx_mps",
                "vy_mps",
                "yaw_rate_rps",
                "tracker_state",
                "goal_distance_m",
                "path_deviation_m",
                "fusion_confidence",
                "desired_speed_pct",
                "applied_speed_pct",
                "desired_steering_deg",
                "applied_steering_deg",
                "bridge_state",
            ]
        )
        self._status_handle = self._status_log.open("w", encoding="utf-8")
        self._bridge_handle = self._bridge_status_log.open("w", encoding="utf-8")
        self._lidar_handle = self._lidar_points_csv.open("w", newline="", encoding="utf-8")
        self._lidar_writer = csv.writer(self._lidar_handle)
        self._lidar_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "scan_index",
                "point_index",
                "angle_rad",
                "range_m",
                "x_forward_m",
                "y_left_m",
                "x_world_m",
                "y_world_m",
            ]
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(NavPath, path_topic, self._path_cb, latched_qos)
        self.create_subscription(String, planner_status_topic, self._planner_status_cb, latched_qos)
        self.create_subscription(String, tracker_status_topic, self._tracker_status_cb, latched_qos)
        self.create_subscription(String, bridge_status_topic, self._bridge_status_cb, latched_qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            "Curve tracking recorder started (timeout=%.2fs output=%s)"
            % (self._timeout_s, str(self._analysis_dir))
        )

    def _path_cb(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        self._path_points = []
        for index, pose in enumerate(msg.poses):
            self._path_points.append(
                {
                    "index": float(index),
                    "x_m": float(pose.pose.position.x),
                    "y_m": float(pose.pose.position.y),
                    "yaw_rad": _quat_to_yaw(
                        float(pose.pose.orientation.x),
                        float(pose.pose.orientation.y),
                        float(pose.pose.orientation.z),
                        float(pose.pose.orientation.w),
                    ),
                }
            )
        if not self._path_written:
            with self._planned_path_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "x_m", "y_m", "yaw_rad"])
                for row in self._path_points:
                    writer.writerow([int(row["index"]), row["x_m"], row["y_m"], row["yaw_rad"]])
            self._path_written = True

    def _planner_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._planner_status = payload
        if bool(payload.get("ready", False)):
            self._track_started = True

    def _tracker_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        self._tracker_status = payload
        self._status_handle.write(
            json.dumps(
                {
                    "t_monotonic_s": time.monotonic() - self._start_monotonic,
                    "payload": payload,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._status_handle.flush()

        state = str(payload.get("state", ""))
        if state == "tracking":
            self._track_started = True
        elif state in TERMINAL_STATES and self._terminal_state_seen_at is None:
            self._terminal_state_seen_at = time.monotonic()

    def _bridge_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._bridge_status = payload
        self._bridge_handle.write(
            json.dumps(
                {
                    "t_monotonic_s": time.monotonic() - self._start_monotonic,
                    "payload": payload,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._bridge_handle.flush()
        if float(payload.get("applied_speed_pct", 0.0) or 0.0) > 1.0e-6:
            self._track_started = True

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom_message = msg
        tracker_state = str(self._tracker_status.get("state", ""))
        if tracker_state == "tracking":
            self._track_started = True
        if not self._track_started:
            return

        pose = {
            "x_m": float(msg.pose.pose.position.x),
            "y_m": float(msg.pose.pose.position.y),
            "yaw_rad": _quat_to_yaw(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ),
            "vx_mps": float(msg.twist.twist.linear.x),
            "vy_mps": float(msg.twist.twist.linear.y),
            "yaw_rate_rps": float(msg.twist.twist.angular.z),
        }
        if self._initial_pose is None:
            self._initial_pose = dict(pose)
        self._final_pose = dict(pose)

        self._tracking_writer.writerow(
            [
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                pose["x_m"],
                pose["y_m"],
                pose["yaw_rad"],
                pose["vx_mps"],
                pose["vy_mps"],
                pose["yaw_rate_rps"],
                tracker_state,
                self._tracker_status.get("goal_distance_m", ""),
                self._tracker_status.get("path_deviation_m", ""),
                self._tracker_status.get("fusion_confidence", ""),
                self._bridge_status.get("desired_speed_pct", ""),
                self._bridge_status.get("applied_speed_pct", ""),
                self._bridge_status.get("desired_steering_deg", ""),
                self._bridge_status.get("applied_steering_deg", ""),
                self._bridge_status.get("state", ""),
            ]
        )
        self._tracking_handle.flush()
        self._trajectory_row_count += 1

    def _scan_cb(self, msg: LaserScan) -> None:
        odom_msg = self._latest_odom_message
        if odom_msg is None:
            return

        base_x_m = float(odom_msg.pose.pose.position.x)
        base_y_m = float(odom_msg.pose.pose.position.y)
        yaw_rad = _quat_to_yaw(
            float(odom_msg.pose.pose.orientation.x),
            float(odom_msg.pose.pose.orientation.y),
            float(odom_msg.pose.pose.orientation.z),
            float(odom_msg.pose.pose.orientation.w),
        )
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

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

            x_forward_m = -(range_m * math.cos(angle_rad))
            y_left_m = range_m * math.sin(angle_rad)
            point_body_x_m = self._lidar_offset_x_m + x_forward_m
            point_body_y_m = self._lidar_offset_y_m + y_left_m
            x_world_m = base_x_m + (cos_yaw * point_body_x_m) - (sin_yaw * point_body_y_m)
            y_world_m = base_y_m + (sin_yaw * point_body_x_m) + (cos_yaw * point_body_y_m)

            self._lidar_writer.writerow(
                [
                    stamp_sec,
                    stamp_nanosec,
                    scan_index,
                    point_index,
                    angle_rad,
                    range_m,
                    x_forward_m,
                    y_left_m,
                    x_world_m,
                    y_world_m,
                ]
            )
            self._lidar_point_count += 1
            wrote_points = True
            angle_rad += float(msg.angle_increment)

        self._scan_count += 1
        if wrote_points:
            self._lidar_handle.flush()

    def _write_planned_path_json(self) -> None:
        payload = {
            "planner_status": self._planner_status,
            "path_point_count": len(self._path_points),
            "path_xy_yaw": [
                [row["x_m"], row["y_m"], row["yaw_rad"]] for row in self._path_points
            ],
        }
        self._planned_path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_summary(self, end_cause: str) -> None:
        target_pose = None
        if self._path_points:
            last = self._path_points[-1]
            target_pose = {
                "x_m": float(last["x_m"]),
                "y_m": float(last["y_m"]),
                "yaw_rad": float(last["yaw_rad"]),
            }

        final_error_x_m = None
        final_error_y_m = None
        final_error_yaw_deg = None
        goal_distance_m = None
        if self._final_pose is not None and target_pose is not None:
            final_error_x_m = float(self._final_pose["x_m"] - target_pose["x_m"])
            final_error_y_m = float(self._final_pose["y_m"] - target_pose["y_m"])
            final_error_yaw_deg = abs(
                math.degrees(
                    math.atan2(
                        math.sin(float(self._final_pose["yaw_rad"] - target_pose["yaw_rad"])),
                        math.cos(float(self._final_pose["yaw_rad"] - target_pose["yaw_rad"])),
                    )
                )
            )
            goal_distance_m = math.hypot(final_error_x_m, final_error_y_m)

        summary = {
            "run_dir": str(self._output_dir),
            "initial_pose": self._initial_pose,
            "final_pose": self._final_pose,
            "target_pose": target_pose,
            "final_error_x_m": final_error_x_m,
            "final_error_y_m": final_error_y_m,
            "final_error_yaw_deg": final_error_yaw_deg,
            "goal_distance_m": goal_distance_m,
            "time_total_s": time.monotonic() - self._start_monotonic,
            "end_cause": end_cause,
            "path_point_count": len(self._path_points),
            "trajectory_row_count": self._trajectory_row_count,
            "lidar_scan_count": self._scan_count,
            "lidar_point_count": self._lidar_point_count,
            "planner_status": self._planner_status,
            "tracker_status": self._tracker_status,
            "bridge_status": self._bridge_status,
        }
        self._summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def finalize(self, *, end_cause: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            if not self._tracking_handle.closed:
                self._tracking_handle.flush()
                self._tracking_handle.close()
        except Exception:
            pass
        try:
            if not self._status_handle.closed:
                self._status_handle.flush()
                self._status_handle.close()
        except Exception:
            pass
        try:
            if not self._bridge_handle.closed:
                self._bridge_handle.flush()
                self._bridge_handle.close()
        except Exception:
            pass
        try:
            if not self._lidar_handle.closed:
                self._lidar_handle.flush()
                self._lidar_handle.close()
        except Exception:
            pass
        self._write_planned_path_json()
        self._write_summary(end_cause=end_cause)
        self.get_logger().info("Curve tracking recorder finished (cause=%s)" % end_cause)

    def _tick(self) -> None:
        elapsed_s = time.monotonic() - self._start_monotonic
        if self._terminal_state_seen_at is not None:
            if (time.monotonic() - self._terminal_state_seen_at) >= 0.25:
                self.finalize(end_cause=str(self._tracker_status.get("state", "unknown_terminal")))
                if rclpy.ok():
                    rclpy.shutdown()
                return

        if elapsed_s >= self._timeout_s:
            self.finalize(end_cause="timeout")
            if rclpy.ok():
                rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-topic", default="/apex/planning/curve_entry_path")
    parser.add_argument("--planner-status-topic", default="/apex/planning/curve_entry_status")
    parser.add_argument("--tracker-status-topic", default="/apex/tracking/status")
    parser.add_argument("--bridge-status-topic", default="/apex/vehicle/drive_bridge_status")
    parser.add_argument("--odom-topic", default="/apex/odometry/imu_lidar_fused")
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--lidar-offset-x-m", type=float, default=0.18)
    parser.add_argument("--lidar-offset-y-m", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = FsPath(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = CurveTrackingValidationRecorder(
        path_topic=args.path_topic,
        planner_status_topic=args.planner_status_topic,
        tracker_status_topic=args.tracker_status_topic,
        bridge_status_topic=args.bridge_status_topic,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        output_dir=output_dir,
        timeout_s=args.timeout_s,
        lidar_offset_x_m=args.lidar_offset_x_m,
        lidar_offset_y_m=args.lidar_offset_y_m,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.finalize(end_cause=str(node._tracker_status.get("state", "interrupted")))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
