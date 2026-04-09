#!/usr/bin/env python3
"""Bridge Gazebo ground truth odometry into slam_toolbox-friendly frames."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class ApexGroundTruthTfBridge(Node):
    def __init__(self) -> None:
        super().__init__("apex_ground_truth_tf_bridge")

        self.declare_parameter("source_odom_topic", "/apex/sim/ground_truth/odom")
        self.declare_parameter("ideal_odom_topic", "/apex/sim/ideal_odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        # The next stage only needs to swap this source for the fused IMU +
        # LiDAR odometry topic while keeping the odom -> base_link contract.

        self._source_odom_topic = str(self.get_parameter("source_odom_topic").value)
        self._ideal_odom_topic = str(self.get_parameter("ideal_odom_topic").value)
        self._odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self._child_frame_id = str(self.get_parameter("child_frame_id").value)

        self._odom_pub = self.create_publisher(Odometry, self._ideal_odom_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, self._source_odom_topic, self._odom_cb, 20)

        self.get_logger().info(
            "ApexGroundTruthTfBridge started (source=%s ideal=%s odom_frame=%s child=%s)"
            % (
                self._source_odom_topic,
                self._ideal_odom_topic,
                self._odom_frame_id,
                self._child_frame_id,
            )
        )

    def _odom_cb(self, msg: Odometry) -> None:
        bridged = Odometry()
        bridged.header = msg.header
        # slam_toolbox follows the odom -> base_link TF chain, so the perfect
        # Gazebo pose is exposed in that standard frame pair for the ideal stage.
        bridged.header.frame_id = self._odom_frame_id
        bridged.child_frame_id = self._child_frame_id
        bridged.pose = msg.pose
        bridged.twist = msg.twist
        self._odom_pub.publish(bridged)

        transform = TransformStamped()
        transform.header = bridged.header
        transform.child_frame_id = bridged.child_frame_id
        transform.transform.translation.x = bridged.pose.pose.position.x
        transform.transform.translation.y = bridged.pose.pose.position.y
        transform.transform.translation.z = bridged.pose.pose.position.z
        transform.transform.rotation = bridged.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ApexGroundTruthTfBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
