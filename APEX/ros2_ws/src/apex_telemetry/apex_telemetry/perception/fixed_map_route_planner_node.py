#!/usr/bin/env python3
"""Publish the planned fixed-map route for closed-loop following."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _load_route_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    poses: list[list[float]] = []
    clearances: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                poses.append([float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])])
                clearances.append(float(row.get("clearance_m") or "nan"))
            except Exception:
                continue
    if not poses:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(poses, dtype=np.float64), np.asarray(clearances, dtype=np.float64)


def _path_length_m(poses_xyyaw: np.ndarray) -> float:
    if poses_xyyaw.shape[0] < 2:
        return 0.0
    diffs = np.diff(poses_xyyaw[:, :2], axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


class FixedMapRoutePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("fixed_map_route_planner_node")

        self.declare_parameter("fixed_map_dir", "")
        self.declare_parameter("route_csv", "")
        self.declare_parameter("build_status_json", "")
        self.declare_parameter("frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("path_topic", "/apex/planning/fixed_map_path")
        self.declare_parameter("status_topic", "/apex/planning/fixed_map_status")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("clearance_min_m", 0.15)

        fixed_map_dir_text = str(self.get_parameter("fixed_map_dir").value or "").strip()
        route_csv_text = str(self.get_parameter("route_csv").value or "").strip()
        build_status_json_text = str(self.get_parameter("build_status_json").value or "").strip()
        fixed_map_dir = Path(fixed_map_dir_text).expanduser()
        route_csv = Path(route_csv_text).expanduser() if route_csv_text else (fixed_map_dir / "fixed_route_path.csv")
        build_status_json = (
            Path(build_status_json_text).expanduser()
            if build_status_json_text
            else (fixed_map_dir / "fixed_map_build_status.json")
        )
        self._route_csv = route_csv.resolve()
        self._build_status_json = build_status_json.resolve()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._publish_rate_hz = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self._clearance_min_m = max(0.0, float(self.get_parameter("clearance_min_m").value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._path_pub = self.create_publisher(
            NavPath,
            str(self.get_parameter("path_topic").value),
            latched_qos,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            latched_qos,
        )

        self._poses_xyyaw = np.empty((0, 3), dtype=np.float64)
        self._clearances_m = np.empty((0,), dtype=np.float64)
        self._path_msg: NavPath | None = None
        self._status_payload: dict[str, object] = {"state": "initializing", "ready": False}
        self._load_route()
        self.create_timer(1.0 / self._publish_rate_hz, self._publish)

        self.get_logger().info(
            "FixedMapRoutePlannerNode started (route=%s path=%s status=%s)"
            % (
                self._route_csv,
                str(self.get_parameter("path_topic").value),
                str(self.get_parameter("status_topic").value),
            )
        )

    def _load_route(self) -> None:
        if not self._route_csv.exists():
            self._status_payload = {
                "state": "error",
                "ready": False,
                "error": "missing_route_csv",
                "route_csv": str(self._route_csv),
            }
            self.get_logger().error(f"Missing fixed route CSV: {self._route_csv}")
            return

        self._poses_xyyaw, self._clearances_m = _load_route_csv(self._route_csv)
        if self._poses_xyyaw.shape[0] < 2:
            self._status_payload = {
                "state": "error",
                "ready": False,
                "error": "route_too_short",
                "route_csv": str(self._route_csv),
                "path_point_count": int(self._poses_xyyaw.shape[0]),
            }
            self.get_logger().error(f"Fixed route is too short: {self._route_csv}")
            return

        finite_clearance = self._clearances_m[np.isfinite(self._clearances_m)]
        min_clearance_m = float(np.min(finite_clearance)) if finite_clearance.size else float("nan")
        if finite_clearance.size and min_clearance_m + 1.0e-9 < self._clearance_min_m:
            self._status_payload = {
                "state": "error",
                "ready": False,
                "error": "route_clearance_below_minimum",
                "route_csv": str(self._route_csv),
                "clearance_min_m": min_clearance_m,
                "required_clearance_m": self._clearance_min_m,
            }
            self.get_logger().error(
                "Fixed route clearance %.3f m is below %.3f m"
                % (min_clearance_m, self._clearance_min_m)
            )
            return

        build_status: dict[str, object] = {}
        if self._build_status_json.exists():
            try:
                payload = json.loads(self._build_status_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    build_status = payload
            except Exception:
                build_status = {}

        route_length_m = _path_length_m(self._poses_xyyaw)
        self._path_msg = self._build_path_msg()
        self._status_payload = {
            "state": "tracking",
            "ready": True,
            "route_source": "fixed_map",
            "local_path_source": "fixed_map",
            "continuation_source": "tracking",
            "route_csv": str(self._route_csv),
            "path_point_count": int(self._poses_xyyaw.shape[0]),
            "path_length_m": route_length_m,
            "path_forward_span_m": route_length_m,
            "local_path_age_s": 0.0,
            "clearance_min_m": min_clearance_m,
            "required_clearance_m": self._clearance_min_m,
            "start_error_m": float(build_status.get("start_error_m", 0.0) or 0.0),
            "goal_error_m": float(build_status.get("goal_error_m", 0.0) or 0.0),
            "fixed_map_build": build_status,
        }

    def _build_path_msg(self) -> NavPath:
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        for pose in self._poses_xyyaw:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self._frame_id
            pose_msg.pose.position.x = float(pose[0])
            pose_msg.pose.position.y = float(pose[1])
            pose_msg.pose.position.z = 0.0
            qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            msg.poses.append(pose_msg)
        return msg

    def _publish(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        if self._path_msg is not None:
            self._path_msg.header.stamp = now_msg
            for pose in self._path_msg.poses:
                pose.header.stamp = now_msg
            self._path_pub.publish(self._path_msg)

        payload = dict(self._status_payload)
        payload["local_path_age_s"] = 0.0
        payload["stamp_s"] = float(now_msg.sec) + (1.0e-9 * float(now_msg.nanosec))
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FixedMapRoutePlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
