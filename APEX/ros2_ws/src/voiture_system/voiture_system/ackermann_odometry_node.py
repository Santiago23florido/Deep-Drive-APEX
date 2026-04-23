#!/usr/bin/env python3
"""Ackermann odometry node for real RC vehicle."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    half = 0.5 * yaw
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class AckermannOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("ackermann_odometry_node")

        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("rear_axle_to_com_m", 0.15)
        self.declare_parameter("front_half_track_m", 0.05)

        self.declare_parameter("speed_topic", "/vehicle/speed_mps")
        self.declare_parameter("steering_topic", "/vehicle/steering_angle_cmd_rad")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")

        self._wheelbase_m = max(1e-6, float(self.get_parameter("wheelbase_m").value))
        self._rear_axle_to_com_m = max(0.0, float(self.get_parameter("rear_axle_to_com_m").value))
        self._front_half_track_m = max(0.0, float(self.get_parameter("front_half_track_m").value))

        self._speed_mps = 0.0
        self._steering_rad = 0.0

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self._last_time = self.get_clock().now()

        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self._base_frame_id = str(self.get_parameter("base_frame_id").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        self.create_subscription(Float64, str(self.get_parameter("speed_topic").value), self._on_speed, 10)
        self.create_subscription(Float64, str(self.get_parameter("steering_topic").value), self._on_steering, 10)

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate_hz, self._step)

        self.get_logger().info(
            "AckermannOdometryNode started (L=%.3f, Lr=%.3f, front_half_track=%.3f)"
            % (self._wheelbase_m, self._rear_axle_to_com_m, self._front_half_track_m)
        )

    def _on_speed(self, msg: Float64) -> None:
        self._speed_mps = float(msg.data)

    def _on_steering(self, msg: Float64) -> None:
        self._steering_rad = float(msg.data)

    def _step(self) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self._last_time = now

        v_rear = float(self._speed_mps)
        delta = float(self._steering_rad)

        yaw_rate = 0.0
        if abs(self._wheelbase_m) > 1e-9:
            yaw_rate = v_rear * math.tan(delta) / self._wheelbase_m

        # COM velocity from rear-axle reference with a rigid-body offset.
        vx_body = v_rear
        vy_body = self._rear_axle_to_com_m * yaw_rate

        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw

        self._x += vx_world * dt
        self._y += vy_world * dt
        self._yaw = _wrap_angle(self._yaw + yaw_rate * dt)

        q = _yaw_to_quaternion(self._yaw)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self._odom_frame_id
        odom.child_frame_id = self._base_frame_id
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = yaw_rate
        self._odom_pub.publish(odom)

        if self._tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = odom.header.stamp
            tf.header.frame_id = self._odom_frame_id
            tf.child_frame_id = self._base_frame_id
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.translation.z = 0.0
            tf.transform.rotation = q
            self._tf_broadcaster.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = AckermannOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

