#!/usr/bin/env python3
"""Record planner/tracker outputs for one recognition_tour validation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


TERMINAL_STATES = {
    "loop_closed",
    "timeout",
    "aborted_low_confidence",
    "aborted_path_loss",
    "aborted_odom_timeout",
    "planner_failed",
}


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


class RecognitionTourRecorder(Node):
    def __init__(
        self,
        *,
        path_topic: str,
        route_topic: str,
        planner_status_topic: str,
        tracker_status_topic: str,
        bridge_status_topic: str,
        odom_topic: str,
        scan_topic: str,
        output_dir: Path,
        timeout_s: float,
        lidar_offset_x_m: float,
        lidar_offset_y_m: float,
    ) -> None:
        super().__init__("record_recognition_tour")

        self._timeout_s = max(1.0, float(timeout_s))
        self._start_monotonic = time.monotonic()
        self._terminal_state_seen_at: float | None = None
        self._finalized = False

        self._planner_status: dict[str, Any] = {}
        self._tracker_status: dict[str, Any] = {}
        self._bridge_status: dict[str, Any] = {}
        self._latest_odom_message: Odometry | None = None
        self._initial_pose: dict[str, float] | None = None
        self._final_pose: dict[str, float] | None = None
        self._path_count = 0
        self._route_points: list[dict[str, float]] = []
        self._route_received = False
        self._track_started = False
        self._trajectory_row_count = 0
        self._scan_count = 0
        self._lidar_point_count = 0
        self._lidar_offset_x_m = float(lidar_offset_x_m)
        self._lidar_offset_y_m = float(lidar_offset_y_m)

        self._output_dir = output_dir
        self._analysis_dir = self._output_dir / "analysis_recognition_tour"
        self._analysis_dir.mkdir(parents=True, exist_ok=True)
        self._lidar_points_csv = self._output_dir / "lidar_points.csv"
        self._trajectory_csv = self._analysis_dir / "recognition_tour_trajectory.csv"
        self._route_csv = self._analysis_dir / "recognition_tour_route.csv"
        self._route_json = self._analysis_dir / "recognition_tour_route.json"
        self._summary_json = self._analysis_dir / "recognition_tour_summary.json"
        self._status_log = self._analysis_dir / "recognition_tour_status.log"
        self._bridge_status_log = self._analysis_dir / "drive_bridge_status.log"
        self._local_path_history_jsonl = self._analysis_dir / "recognition_tour_local_path_history.jsonl"

        self._trajectory_handle = self._trajectory_csv.open("w", newline="", encoding="utf-8")
        self._trajectory_writer = csv.writer(self._trajectory_handle)
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
                "planner_state",
                "tracker_state",
                "path_age_s",
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
        self._path_history_handle = self._local_path_history_jsonl.open("w", encoding="utf-8")
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
        self.create_subscription(NavPath, route_topic, self._route_cb, latched_qos)
        self.create_subscription(String, planner_status_topic, self._planner_status_cb, latched_qos)
        self.create_subscription(String, tracker_status_topic, self._tracker_status_cb, latched_qos)
        self.create_subscription(String, bridge_status_topic, self._bridge_status_cb, 20)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            "Recognition tour recorder started (timeout=%.2fs output=%s)"
            % (self._timeout_s, str(self._analysis_dir))
        )

    def _path_cb(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        payload = {
            "t_monotonic_s": time.monotonic() - self._start_monotonic,
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
            "path_xy_yaw": [],
        }
        for pose in msg.poses:
            payload["path_xy_yaw"].append(
                [
                    float(pose.pose.position.x),
                    float(pose.pose.position.y),
                    _quat_to_yaw(
                        float(pose.pose.orientation.x),
                        float(pose.pose.orientation.y),
                        float(pose.pose.orientation.z),
                        float(pose.pose.orientation.w),
                    ),
                ]
            )
        self._path_history_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._path_history_handle.flush()
        self._path_count += 1

    def _route_cb(self, msg: NavPath) -> None:
        if not msg.poses:
            return
        self._route_points = []
        for index, pose in enumerate(msg.poses):
            self._route_points.append(
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
        with self._route_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "x_m", "y_m", "yaw_rad"])
            for row in self._route_points:
                writer.writerow([int(row["index"]), row["x_m"], row["y_m"], row["yaw_rad"]])
        self._route_received = True

    def _planner_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._planner_status = payload
        self._status_handle.write(
            json.dumps(
                {
                    "t_monotonic_s": time.monotonic() - self._start_monotonic,
                    "source": "planner",
                    "payload": payload,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._status_handle.flush()

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
                    "source": "tracker",
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

        if not self._track_started:
            return

        self._trajectory_writer.writerow(
            [
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                pose["x_m"],
                pose["y_m"],
                pose["yaw_rad"],
                pose["vx_mps"],
                pose["vy_mps"],
                pose["yaw_rate_rps"],
                self._planner_status.get("state", ""),
                self._tracker_status.get("state", ""),
                self._planner_status.get("local_path_age_s", ""),
                self._tracker_status.get("path_deviation_m", ""),
                self._tracker_status.get("fusion_confidence", ""),
                self._bridge_status.get("desired_speed_pct", ""),
                self._bridge_status.get("applied_speed_pct", ""),
                self._bridge_status.get("desired_steering_deg", ""),
                self._bridge_status.get("applied_steering_deg", ""),
                self._bridge_status.get("state", ""),
            ]
        )
        self._trajectory_handle.flush()
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

    def _write_route_json(self) -> None:
        payload = {
            "planner_status": self._planner_status,
            "route_point_count": len(self._route_points),
            "path_xy_yaw": [
                [row["x_m"], row["y_m"], row["yaw_rad"]] for row in self._route_points
            ],
        }
        self._route_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_summary(self, end_cause: str) -> None:
        summary = {
            "run_dir": str(self._output_dir),
            "initial_pose": self._initial_pose,
            "final_pose": self._final_pose,
            "time_total_s": time.monotonic() - self._start_monotonic,
            "end_cause": end_cause,
            "trajectory_row_count": self._trajectory_row_count,
            "local_path_message_count": self._path_count,
            "route_point_count": len(self._route_points),
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
        for handle in (
            self._trajectory_handle,
            self._status_handle,
            self._bridge_handle,
            self._path_history_handle,
            self._lidar_handle,
        ):
            try:
                if not handle.closed:
                    handle.flush()
                    handle.close()
            except Exception:
                pass
        self._write_route_json()
        self._write_summary(end_cause=end_cause)
        self.get_logger().info("Recognition tour recorder finished (cause=%s)" % end_cause)

    def _tick(self) -> None:
        elapsed_s = time.monotonic() - self._start_monotonic
        if self._terminal_state_seen_at is not None:
            if (time.monotonic() - self._terminal_state_seen_at) >= 0.75:
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
    parser.add_argument("--path-topic", default="/apex/planning/recognition_tour_local_path")
    parser.add_argument("--route-topic", default="/apex/planning/recognition_tour_route")
    parser.add_argument("--planner-status-topic", default="/apex/planning/recognition_tour_status")
    parser.add_argument("--tracker-status-topic", default="/apex/tracking/recognition_tour_status")
    parser.add_argument("--bridge-status-topic", default="/apex/vehicle/drive_bridge_status")
    parser.add_argument("--odom-topic", default="/apex/odometry/imu_lidar_fused")
    parser.add_argument("--scan-topic", default="/lidar/scan_localization")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--lidar-offset-x-m", type=float, default=0.18)
    parser.add_argument("--lidar-offset-y-m", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = RecognitionTourRecorder(
        path_topic=args.path_topic,
        route_topic=args.route_topic,
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
