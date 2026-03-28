#!/usr/bin/env python3
"""Publish `/odom` from APEX kinematics topics (position, velocity, heading)."""

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
        self.declare_parameter("heading_topic", "/apex/kinematics/heading")
        self.declare_parameter("angular_velocity_topic", "/apex/kinematics/angular_velocity")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_raw")
        self.declare_parameter("odom_frame_id", "odom_imu_raw")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("translation_mode", "full")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("use_heading_topic", True)
        self.declare_parameter("heading_timeout_s", 0.5)
        self.declare_parameter("integrate_yaw_from_rate_when_heading_missing", True)
        self.declare_parameter("use_velocity_heading", True)
        self.declare_parameter("min_speed_for_heading_mps", 0.05)

        pos_topic = str(self.get_parameter("position_topic").value)
        vel_topic = str(self.get_parameter("velocity_topic").value)
        heading_topic = str(self.get_parameter("heading_topic").value)
        angular_velocity_topic = str(self.get_parameter("angular_velocity_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)

        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._base_frame = str(self.get_parameter("base_frame_id").value)
        self._translation_mode = str(self.get_parameter("translation_mode").value).strip().lower()
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._use_heading_topic = bool(self.get_parameter("use_heading_topic").value)
        self._heading_timeout_s = max(0.0, float(self.get_parameter("heading_timeout_s").value))
        self._integrate_from_rate = bool(
            self.get_parameter("integrate_yaw_from_rate_when_heading_missing").value
        )
        self._use_velocity_heading = bool(self.get_parameter("use_velocity_heading").value)
        self._min_speed_for_heading = max(0.0, float(self.get_parameter("min_speed_for_heading_mps").value))

        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._last_heading_update = None
        self._last_publish_time = None

        self.create_subscription(PointStamped, pos_topic, self._position_cb, 20)
        self.create_subscription(Vector3Stamped, vel_topic, self._velocity_cb, 20)
        self.create_subscription(Vector3Stamped, heading_topic, self._heading_cb, 20)
        self.create_subscription(Vector3Stamped, angular_velocity_topic, self._angular_velocity_cb, 20)

        self._odom_pub = self.create_publisher(Odometry, odom_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(period, self._publish_odom)

        self.get_logger().info(
            "KinematicsOdometryNode started (pos=%s vel=%s heading=%s ang_vel=%s odom=%s mode=%s tf=%s)"
            % (
                pos_topic,
                vel_topic,
                heading_topic,
                angular_velocity_topic,
                odom_topic,
                self._translation_mode,
                self._publish_tf,
            )
        )

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _heading_is_fresh(self) -> bool:
        if not self._use_heading_topic or self._last_heading_update is None:
            return False
        if self._heading_timeout_s <= 0.0:
            return True
        age_s = (self.get_clock().now() - self._last_heading_update).nanoseconds * 1e-9
        return age_s <= self._heading_timeout_s

    def _position_cb(self, msg: PointStamped) -> None:
        self._px = float(msg.point.x)
        self._py = float(msg.point.y)
        self._pz = float(msg.point.z)

    def _velocity_cb(self, msg: Vector3Stamped) -> None:
        self._vx = float(msg.vector.x)
        self._vy = float(msg.vector.y)
        self._vz = float(msg.vector.z)

        if self._use_velocity_heading and not self._heading_is_fresh():
            speed_xy = math.hypot(self._vx, self._vy)
            if speed_xy >= self._min_speed_for_heading:
                self._yaw = math.atan2(self._vy, self._vx)

    def _heading_cb(self, msg: Vector3Stamped) -> None:
        if not self._use_heading_topic:
            return
        self._yaw = self._normalize_angle(float(msg.vector.z))
        self._last_heading_update = self.get_clock().now()

    def _angular_velocity_cb(self, msg: Vector3Stamped) -> None:
        self._yaw_rate = float(msg.vector.z)

    def _publish_odom(self) -> None:
        now_t = self.get_clock().now()
        if (
            self._integrate_from_rate
            and not self._heading_is_fresh()
            and self._last_publish_time is not None
        ):
            dt = (now_t - self._last_publish_time).nanoseconds * 1e-9
            if dt > 0.0:
                self._yaw = self._normalize_angle(self._yaw + self._yaw_rate * dt)
        self._last_publish_time = now_t

        stamp = now_t.to_msg()
        qx, qy, qz, qw = yaw_to_quat(self._yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        if self._translation_mode == "zero":
            px = 0.0
            py = 0.0
            pz = 0.0
            vx = 0.0
            vy = 0.0
            vz = 0.0
        else:
            px = self._px
            py = self._py
            pz = self._pz
            vx = self._vx
            vy = self._vy
            vz = self._vz

        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz
        odom.twist.twist.angular.z = self._yaw_rate
        self._odom_pub.publish(odom)

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = px
            tf.transform.translation.y = py
            tf.transform.translation.z = pz
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
