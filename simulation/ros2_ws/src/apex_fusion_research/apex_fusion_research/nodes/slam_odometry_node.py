"""Odometry prior (TF ``noisy/odom -> noisy/base_link``) for the damaged-sensor SLAM.

Input : ``/apex/fusion/ins/odom`` (pure INS from the noisy IMU).
Output: TF ``<odom_frame_id> -> <base_frame_id>`` and ``nav_msgs/Odometry`` on
        ``/apex/fusion/slam_noisy/odom``, computed by
        :func:`apex_fusion_research.core.slam_odometry.planar_base_pose`.
Nothing is published until the INS finishes its alignment.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from ..core.slam_odometry import SlamOdometryConfig, planar_base_pose
from ._common import ConfigParameters


class SlamOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_slam_odometry")
        self.declare_parameter("ins_topic", "/apex/fusion/ins/odom")
        self.declare_parameter("odom_topic", "/apex/fusion/slam_noisy/odom")
        gp = self.get_parameter
        self._params = ConfigParameters(self, SlamOdometryConfig)
        self._tf = TransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, str(gp("odom_topic").value), 50)
        self.create_subscription(Odometry, str(gp("ins_topic").value), self._on_ins, 200)
        self._announced = False

    def _on_ins(self, msg: Odometry) -> None:
        cfg: SlamOdometryConfig = self._params.config
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        x, y, yaw = planar_base_pose(np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z]), cfg)
        qz, qw = math.sin(0.5 * yaw), math.cos(0.5 * yaw)

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = cfg.odom_frame_id
        tf.child_frame_id = cfg.base_frame_id
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._tf.sendTransform(tf)

        odom = Odometry()
        odom.header = tf.header
        odom.child_frame_id = cfg.base_frame_id
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.angular.z = msg.twist.twist.angular.z
        self._odom_pub.publish(odom)
        if not self._announced:
            self._announced = True
            self.get_logger().info(f"publishing {cfg.odom_frame_id} -> {cfg.base_frame_id} (mode '{cfg.mode}')")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamOdometryNode()
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
