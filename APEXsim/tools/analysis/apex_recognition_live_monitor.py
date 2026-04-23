#!/usr/bin/env python3
"""Lightweight PC-side monitor for live recognition-tour topics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class _PoseState:
    x_m: float = float("nan")
    y_m: float = float("nan")
    yaw_deg: float = float("nan")
    stamp_s: float = float("nan")


class ApexRecognitionLiveMonitor(Node):
    def __init__(self) -> None:
        super().__init__("apex_recognition_live_monitor_pc")

        self.declare_parameter("pose_topic", "/apex/estimation/current_pose")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("path_topic", "/apex/estimation/path")
        self.declare_parameter("route_topic", "/apex/planning/recognition_tour_route")
        self.declare_parameter("local_path_topic", "/apex/planning/recognition_tour_local_path")
        self.declare_parameter("map_topic", "/apex/estimation/live_map_points")
        self.declare_parameter("full_map_topic", "/apex/estimation/full_map_points")
        self.declare_parameter("report_period_s", 1.0)

        self._pose = _PoseState()
        self._odom = _PoseState()
        self._path_pose_count = 0
        self._route_pose_count = 0
        self._local_path_pose_count = 0
        self._map_point_count = 0
        self._map_stamp_s = float("nan")
        self._full_map_point_count = 0
        self._full_map_stamp_s = float("nan")

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("pose_topic").value),
            self._pose_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("path_topic").value),
            self._path_cb,
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("route_topic").value),
            self._route_cb,
            10,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("local_path_topic").value),
            self._local_path_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("map_topic").value),
            self._map_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("full_map_topic").value),
            self._full_map_cb,
            10,
        )

        report_period_s = max(0.25, float(self.get_parameter("report_period_s").value))
        self.create_timer(report_period_s, self._report)
        self.get_logger().info("APEX recognition live monitor started")

    def _stamp_s(self, stamp) -> float:
        return float(stamp.sec) + (1.0e-9 * float(stamp.nanosec))

    def _pose_cb(self, msg: PoseStamped) -> None:
        yaw_rad = _quat_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self._pose = _PoseState(
            x_m=float(msg.pose.position.x),
            y_m=float(msg.pose.position.y),
            yaw_deg=math.degrees(yaw_rad),
            stamp_s=self._stamp_s(msg.header.stamp),
        )

    def _odom_cb(self, msg: Odometry) -> None:
        yaw_rad = _quat_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._odom = _PoseState(
            x_m=float(msg.pose.pose.position.x),
            y_m=float(msg.pose.pose.position.y),
            yaw_deg=math.degrees(yaw_rad),
            stamp_s=self._stamp_s(msg.header.stamp),
        )

    def _path_cb(self, msg: Path) -> None:
        self._path_pose_count = len(msg.poses)

    def _route_cb(self, msg: Path) -> None:
        self._route_pose_count = len(msg.poses)

    def _local_path_cb(self, msg: Path) -> None:
        self._local_path_pose_count = len(msg.poses)

    def _map_cb(self, msg: PointCloud2) -> None:
        self._map_point_count = int(msg.width) * int(msg.height)
        self._map_stamp_s = self._stamp_s(msg.header.stamp)

    def _full_map_cb(self, msg: PointCloud2) -> None:
        self._full_map_point_count = int(msg.width) * int(msg.height)
        self._full_map_stamp_s = self._stamp_s(msg.header.stamp)

    def _report(self) -> None:
        pose = self._pose
        odom = self._odom
        self.get_logger().info(
            (
                "live pose=(%.2f, %.2f, %.1fdeg) odom=(%.2f, %.2f, %.1fdeg) "
                "map_pts=%d full_map_pts=%d route_pts=%d local_path_pts=%d fused_path_pts=%d"
            )
            % (
                pose.x_m,
                pose.y_m,
                pose.yaw_deg,
                odom.x_m,
                odom.y_m,
                odom.yaw_deg,
                self._map_point_count,
                self._full_map_point_count,
                self._route_pose_count,
                self._local_path_pose_count,
                self._path_pose_count,
            )
        )


def main() -> None:
    rclpy.init()
    node = ApexRecognitionLiveMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
