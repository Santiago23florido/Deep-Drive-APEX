#!/usr/bin/env python3
"""Publish `/odom` from APEX kinematics topics (position + velocity)."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class KinematicsOdometryNode(Node):
    """Bridge kinematics outputs into ROS2 odometry for SLAM input."""

    def __init__(self) -> None:
        super().__init__("kinematics_odometry_node")

        self.declare_parameter("position_topic", "/apex/kinematics/position")
        self.declare_parameter("velocity_topic", "/apex/kinematics/velocity")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("use_velocity_heading", True)
        self.declare_parameter("min_speed_for_heading_mps", 0.05)

        pos_topic = str(self.get_parameter("position_topic").value)
        vel_topic = str(self.get_parameter("velocity_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)

        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._use_velocity_heading = bool(self.get_parameter("use_velocity_heading").value)
        self._min_speed_for_heading = max(0.0, float(self.get_parameter("min_speed_for_heading_mps").value))

        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw = 0.0

        self.create_subscription(PointStamped, pos_topic, self._position_cb, 20)
        self.create_subscription(Vector3Stamped, vel_topic, self._velocity_cb, 20)

        self._odom_pub = self.create_publisher(Odometry, odom_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(period, self._publish_odom)

        self.get_logger().info(
            "KinematicsOdometryNode started (pos=%s vel=%s odom=%s)"
            % (pos_topic, vel_topic, odom_topic)
        )

    def _position_cb(self, msg: PointStamped) -> None:
        self._px = float(msg.point.x)
        self._py = float(msg.point.y)
        self._pz = float(msg.point.z)

    def _velocity_cb(self, msg: Vector3Stamped) -> None:
        self._vx = float(msg.vector.x)
        self._vy = float(msg.vector.y)
        self._vz = float(msg.vector.z)

        if self._use_velocity_heading:
            speed_xy = math.hypot(self._vx, self._vy)
            if speed_xy >= self._min_speed_for_heading:
                self._yaw = math.atan2(self._vy, self._vx)

    def _publish_odom(self) -> None:
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quat(self._yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = self._px
        odom.pose.pose.position.y = self._py
        odom.pose.pose.position.z = self._pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self._vx
        odom.twist.twist.linear.y = self._vy
        odom.twist.twist.linear.z = self._vz
        self._odom_pub.publish(odom)

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = self._px
            tf.transform.translation.y = self._py
            tf.transform.translation.z = self._pz
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = KinematicsOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
