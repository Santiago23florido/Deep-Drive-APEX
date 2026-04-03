#!/usr/bin/env python3
"""Online causal LiDAR + IMU planar fusion for short corridor runs."""

from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .planar_fusion_core import FusionParameters, OnlinePlanarFusion, scan_observation_from_ranges


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class ImuLidarPlanarFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_lidar_planar_fusion_node")

        self.declare_parameter("imu_topic", "/apex/imu/data_raw")
        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("path_topic", "/apex/estimation/path")
        self.declare_parameter("status_topic", "/apex/estimation/status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("status_publish_rate_hz", 2.0)
        self.declare_parameter("path_max_poses", 4000)

        self.declare_parameter("median_window", 5)
        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("static_window_s", 0.4)
        self.declare_parameter("static_search_s", 2.0)
        self.declare_parameter("velocity_decay_tau_s", 1.1)
        self.declare_parameter("submap_window_scans", 6)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("max_correspondence_m", 0.35)
        self.declare_parameter("initial_scan_count_min", 4)
        self.declare_parameter("max_initial_alignment_scans", 6)
        self.declare_parameter("corridor_bin_m", 0.10)
        self.declare_parameter("low_confidence_residual_m", 0.16)
        self.declare_parameter("min_valid_correspondence_count", 14)
        self.declare_parameter("max_scan_optimization_evals", 80)

        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._child_frame = str(self.get_parameter("child_frame_id").value)
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._status_publish_rate_hz = max(
            0.5, float(self.get_parameter("status_publish_rate_hz").value)
        )
        self._path_max_poses = max(32, int(self.get_parameter("path_max_poses").value))
        self._point_stride = int(self.get_parameter("point_stride").value)

        params = FusionParameters(
            median_window=int(self.get_parameter("median_window").value),
            ema_alpha=float(self.get_parameter("ema_alpha").value),
            static_window_s=float(self.get_parameter("static_window_s").value),
            static_search_s=float(self.get_parameter("static_search_s").value),
            velocity_decay_tau_s=float(self.get_parameter("velocity_decay_tau_s").value),
            submap_window_scans=int(self.get_parameter("submap_window_scans").value),
            point_stride=int(self.get_parameter("point_stride").value),
            max_correspondence_m=float(self.get_parameter("max_correspondence_m").value),
            initial_scan_count_min=int(self.get_parameter("initial_scan_count_min").value),
            max_initial_alignment_scans=int(
                self.get_parameter("max_initial_alignment_scans").value
            ),
            corridor_bin_m=float(self.get_parameter("corridor_bin_m").value),
            low_confidence_residual_m=float(
                self.get_parameter("low_confidence_residual_m").value
            ),
            min_valid_correspondence_count=int(
                self.get_parameter("min_valid_correspondence_count").value
            ),
            max_scan_optimization_evals=int(
                self.get_parameter("max_scan_optimization_evals").value
            ),
        )
        self._fusion = OnlinePlanarFusion(params)

        self._scan_counter = 0
        self._last_status_payload = ""
        self._last_estimate = None
        self._path_msg = Path()
        self._path_msg.header.frame_id = self._odom_frame

        self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._path_pub = self.create_publisher(Path, self._path_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        self.create_timer(1.0 / self._status_publish_rate_hz, self._publish_status)

        self.get_logger().info(
            "ImuLidarPlanarFusionNode started (imu=%s scan=%s odom=%s path=%s status=%s)"
            % (
                self._imu_topic,
                self._scan_topic,
                self._odom_topic,
                self._path_topic,
                self._status_topic,
            )
        )

    def _imu_cb(self, msg: Imu) -> None:
        t_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        self._fusion.add_imu_sample(
            t_s=t_s,
            ax_mps2=float(msg.linear_acceleration.x),
            ay_mps2=float(msg.linear_acceleration.y),
            az_mps2=float(msg.linear_acceleration.z),
            gz_rps=float(msg.angular_velocity.z),
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        scan = scan_observation_from_ranges(
            scan_index=self._scan_counter,
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
            angle_min_rad=float(msg.angle_min),
            angle_increment_rad=float(msg.angle_increment),
            ranges=list(msg.ranges),
            range_min_m=float(msg.range_min),
            range_max_m=float(msg.range_max),
            point_stride=self._point_stride,
        )
        self._scan_counter += 1
        if scan.points_local.shape[0] < 8:
            self._publish_status()
            return

        estimates = self._fusion.add_scan_observation(scan)
        for estimate in estimates:
            self._last_estimate = estimate
            self._publish_estimate(estimate)
        self._publish_status()

    def _publish_estimate(self, estimate) -> None:
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(estimate.stamp_sec)
        stamp.nanosec = int(estimate.stamp_nanosec)
        qx, qy, qz, qw = _yaw_to_quat(estimate.yaw_rad)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child_frame
        odom.pose.pose.position.x = float(estimate.x_m)
        odom.pose.pose.position.y = float(estimate.y_m)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(estimate.vx_mps)
        odom.twist.twist.linear.y = float(estimate.vy_mps)
        odom.twist.twist.angular.z = float(estimate.yaw_rate_rps)

        high_confidence = estimate.confidence == "high"
        pose_cov_xy = 0.025 if high_confidence else 0.18
        yaw_cov = 0.035 if high_confidence else 0.20
        twist_cov_xy = 0.06 if high_confidence else 0.25
        odom.pose.covariance[0] = pose_cov_xy
        odom.pose.covariance[7] = pose_cov_xy
        odom.pose.covariance[35] = yaw_cov
        odom.twist.covariance[0] = twist_cov_xy
        odom.twist.covariance[7] = twist_cov_xy
        odom.twist.covariance[35] = yaw_cov
        self._odom_pub.publish(odom)

        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose_stamped)
        if len(self._path_msg.poses) > self._path_max_poses:
            del self._path_msg.poses[: len(self._path_msg.poses) - self._path_max_poses]
        self._path_pub.publish(self._path_msg)

        if self._tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = self._child_frame
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = 0.0
            transform.transform.rotation = odom.pose.pose.orientation
            self._tf_broadcaster.sendTransform(transform)

    def _publish_status(self) -> None:
        snapshot = self._fusion.status_snapshot()
        payload = {
            "state": snapshot.state,
            "imu_initialized": snapshot.imu_initialized,
            "alignment_ready": snapshot.alignment_ready,
            "best_effort_init": snapshot.best_effort_init,
            "raw_imu_sample_count": snapshot.raw_imu_sample_count,
            "processed_imu_sample_count": snapshot.processed_imu_sample_count,
            "pending_scan_count": snapshot.pending_scan_count,
            "processed_scan_count": snapshot.processed_scan_count,
            "initial_scan_count": snapshot.initial_scan_count,
            "alignment_yaw_rad": snapshot.alignment_yaw_rad,
            "origin_projection_m": list(snapshot.origin_projection_m),
            "static_initialization": snapshot.static_initialization,
            "corridor_model": snapshot.corridor_model,
            "quality": snapshot.quality,
            "latest_pose": snapshot.latest_pose,
            "parameters": snapshot.parameters,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._last_status_payload = message.data
        self._status_pub.publish(message)


def main() -> None:
    rclpy.init()
    node = ImuLidarPlanarFusionNode()
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
