#!/usr/bin/env python3
"""Publish LiDAR-local pose as PoseWithCovarianceStamped from TF."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class LidarPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("lidar_pose_bridge")

        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("pose_topic", "/apex/lidar/pose_local")
        self.declare_parameter("pose_frame_id", "odom_lidar_local")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("scan_timeout_s", 0.25)
        self.declare_parameter("tf_timeout_s", 0.05)
        self.declare_parameter("covariance_x_m2", 0.01)
        self.declare_parameter("covariance_y_m2", 0.01)
        self.declare_parameter("covariance_yaw_rad2", 0.04)
        self.declare_parameter("stale_covariance_scale", 50.0)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._pose_frame = str(self.get_parameter("pose_frame_id").value)
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._scan_timeout_s = max(0.05, float(self.get_parameter("scan_timeout_s").value))
        self._tf_timeout_s = max(0.0, float(self.get_parameter("tf_timeout_s").value))
        self._cov_x = max(1e-6, float(self.get_parameter("covariance_x_m2").value))
        self._cov_y = max(1e-6, float(self.get_parameter("covariance_y_m2").value))
        self._cov_yaw = max(1e-6, float(self.get_parameter("covariance_yaw_rad2").value))
        self._stale_cov_scale = max(1.0, float(self.get_parameter("stale_covariance_scale").value))

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_scan_stamp: Time | None = None

        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, self._pose_topic, 20)
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_pose)

        self.get_logger().info(
            "LidarPoseBridge started (scan=%s pose=%s frame=%s->%s)"
            % (self._scan_topic, self._pose_topic, self._pose_frame, self._base_frame)
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            self._latest_scan_stamp = Time.from_msg(msg.header.stamp)
        else:
            self._latest_scan_stamp = self.get_clock().now()

    def _publish_pose(self) -> None:
        now_t = self.get_clock().now()
        scan_age_s = float("inf")
        if self._latest_scan_stamp is not None:
            scan_age_s = max(0.0, (now_t - self._latest_scan_stamp).nanoseconds * 1e-9)

        try:
            transform = self._tf_buffer.lookup_transform(
                self._pose_frame,
                self._base_frame,
                Time(),
                timeout=Duration(seconds=self._tf_timeout_s),
            )
        except Exception:
            return

        if transform.header.stamp.sec != 0 or transform.header.stamp.nanosec != 0:
            stamp = transform.header.stamp
        elif self._latest_scan_stamp is not None:
            stamp = self._latest_scan_stamp.to_msg()
        else:
            stamp = now_t.to_msg()

        cov_scale = 1.0 if scan_age_s <= self._scan_timeout_s else self._stale_cov_scale
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._pose_frame
        msg.pose.pose.position.x = float(transform.transform.translation.x)
        msg.pose.pose.position.y = float(transform.transform.translation.y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = transform.transform.rotation
        covariance = [0.0] * 36
        covariance[0] = self._cov_x * cov_scale
        covariance[7] = self._cov_y * cov_scale
        covariance[14] = 1e6
        covariance[21] = 1e6
        covariance[28] = 1e6
        covariance[35] = self._cov_yaw * cov_scale
        msg.pose.covariance = covariance
        self._pose_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = LidarPoseBridge()
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
