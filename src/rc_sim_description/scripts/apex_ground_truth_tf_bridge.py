#!/usr/bin/env python3
"""Bridge Gazebo ground truth odometry into slam_toolbox-friendly frames."""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


@dataclass
class PendingOdomPublish:
    release_time_sec: float
    odom: Odometry
    transform: TransformStamped


@dataclass
class DegradedPlanarState:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    stamp_sec: float


class ApexGroundTruthTfBridge(Node):
    def __init__(self) -> None:
        super().__init__("apex_ground_truth_tf_bridge")

        self.declare_parameter("source_odom_topic", "/apex/sim/ground_truth/odom")
        self.declare_parameter("ideal_odom_topic", "/apex/sim/ideal_odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        # Keep the baseline ideal path untouched unless the launch explicitly
        # enables one of these odom degradations for SLAM robustness sweeps.
        self.declare_parameter("degrade_odom", False)
        self.declare_parameter("odom_position_noise_std", 0.0)
        self.declare_parameter("odom_yaw_noise_std", 0.0)
        self.declare_parameter("odom_velocity_noise_std", 0.0)
        self.declare_parameter("odom_yaw_bias_per_sec", 0.0)
        self.declare_parameter("odom_latency_sec", 0.0)
        # The next stage only needs to swap this source for the fused IMU +
        # LiDAR odometry topic while keeping the odom -> base_link contract.

        self._source_odom_topic = str(self.get_parameter("source_odom_topic").value)
        self._ideal_odom_topic = str(self.get_parameter("ideal_odom_topic").value)
        self._odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self._child_frame_id = str(self.get_parameter("child_frame_id").value)
        self._degrade_odom = bool(self.get_parameter("degrade_odom").value)
        self._odom_position_noise_std = max(
            0.0,
            float(self.get_parameter("odom_position_noise_std").value),
        )
        self._odom_yaw_noise_std = max(
            0.0,
            float(self.get_parameter("odom_yaw_noise_std").value),
        )
        self._odom_velocity_noise_std = max(
            0.0,
            float(self.get_parameter("odom_velocity_noise_std").value),
        )
        self._odom_yaw_bias_per_sec = float(
            self.get_parameter("odom_yaw_bias_per_sec").value
        )
        self._odom_latency_sec = max(
            0.0,
            float(self.get_parameter("odom_latency_sec").value),
        )
        self._rng = random.Random()
        self._pending_queue: deque[PendingOdomPublish] = deque()
        self._degraded_state: DegradedPlanarState | None = None

        self._odom_pub = self.create_publisher(Odometry, self._ideal_odom_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, self._source_odom_topic, self._odom_cb, 20)
        self.create_timer(0.01, self._flush_pending_queue)

        self.get_logger().info(
            (
                "ApexGroundTruthTfBridge started (source=%s ideal=%s odom_frame=%s "
                "child=%s degrade=%s pos_std=%.4f yaw_std=%.4f vel_std=%.4f "
                "yaw_bias_per_sec=%.4f latency=%.4f)"
            )
            % (
                self._source_odom_topic,
                self._ideal_odom_topic,
                self._odom_frame_id,
                self._child_frame_id,
                str(self._degrade_odom).lower(),
                self._odom_position_noise_std,
                self._odom_yaw_noise_std,
                self._odom_velocity_noise_std,
                self._odom_yaw_bias_per_sec,
                self._odom_latency_sec,
            )
        )

    @staticmethod
    def _stamp_to_sec(msg: Odometry) -> float:
        return float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))

    @staticmethod
    def _wrap_angle_rad(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    @staticmethod
    def _quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * ((w * x) + (y * z))
        cosr_cosp = 1.0 - (2.0 * ((x * x) + (y * y)))
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * ((w * y) - (z * x))
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * ((w * z) + (x * y))
        cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = (cr * cp * cy) + (sr * sp * sy)
        x = (sr * cp * cy) - (cr * sp * sy)
        y = (cr * sp * cy) + (sr * cp * sy)
        z = (cr * cp * sy) - (sr * sp * cy)
        return x, y, z, w

    def _extract_true_planar_state(self, msg: Odometry) -> DegradedPlanarState:
        orientation = msg.pose.pose.orientation
        roll, pitch, yaw = self._quaternion_to_rpy(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        stamp_sec = self._stamp_to_sec(msg)
        if stamp_sec <= 0.0:
            stamp_sec = float(self.get_clock().now().nanoseconds) * 1.0e-9
        return DegradedPlanarState(
            x_m=float(msg.pose.pose.position.x),
            y_m=float(msg.pose.pose.position.y),
            z_m=float(msg.pose.pose.position.z),
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            stamp_sec=stamp_sec,
        )

    def _build_bridged_odom(self, msg: Odometry) -> Odometry:
        bridged = Odometry()
        bridged.header = msg.header
        # slam_toolbox follows the odom -> base_link TF chain, so the perfect
        # Gazebo pose is exposed in that standard frame pair for the ideal stage.
        bridged.header.frame_id = self._odom_frame_id
        bridged.child_frame_id = self._child_frame_id
        bridged.pose = copy.deepcopy(msg.pose)
        bridged.twist = copy.deepcopy(msg.twist)

        if not self._degrade_odom:
            self._degraded_state = None
            return bridged

        true_state = self._extract_true_planar_state(msg)
        if self._degraded_state is None or true_state.stamp_sec <= self._degraded_state.stamp_sec:
            self._degraded_state = true_state

        degraded_state = self._degraded_state
        dt_sec = max(0.0, true_state.stamp_sec - degraded_state.stamp_sec)

        measured_vx_mps = float(msg.twist.twist.linear.x)
        measured_vy_mps = float(msg.twist.twist.linear.y)
        measured_vz_mps = float(msg.twist.twist.linear.z)
        measured_wx_rps = float(msg.twist.twist.angular.x)
        measured_wy_rps = float(msg.twist.twist.angular.y)
        measured_wz_rps = float(msg.twist.twist.angular.z)

        if self._odom_velocity_noise_std > 0.0:
            measured_vx_mps += self._rng.gauss(0.0, self._odom_velocity_noise_std)
            measured_vy_mps += self._rng.gauss(0.0, self._odom_velocity_noise_std)
            measured_vz_mps += self._rng.gauss(0.0, self._odom_velocity_noise_std)
            measured_wx_rps += self._rng.gauss(0.0, self._odom_velocity_noise_std)
            measured_wy_rps += self._rng.gauss(0.0, self._odom_velocity_noise_std)
            measured_wz_rps += self._rng.gauss(0.0, self._odom_velocity_noise_std)

        # Model gyro drift as a constant bias on yaw rate before integration.
        measured_wz_rps += self._odom_yaw_bias_per_sec

        yaw_noise_rad = (
            self._rng.gauss(0.0, self._odom_yaw_noise_std)
            if self._odom_yaw_noise_std > 0.0
            else 0.0
        )
        integrated_yaw_rad = self._wrap_angle_rad(
            degraded_state.yaw_rad + (measured_wz_rps * dt_sec) + yaw_noise_rad
        )
        yaw_mid_rad = self._wrap_angle_rad(
            degraded_state.yaw_rad + (0.5 * measured_wz_rps * dt_sec)
        )

        integrated_x_m = degraded_state.x_m + (
            ((math.cos(yaw_mid_rad) * measured_vx_mps) - (math.sin(yaw_mid_rad) * measured_vy_mps))
            * dt_sec
        )
        integrated_y_m = degraded_state.y_m + (
            ((math.sin(yaw_mid_rad) * measured_vx_mps) + (math.cos(yaw_mid_rad) * measured_vy_mps))
            * dt_sec
        )
        if self._odom_position_noise_std > 0.0:
            integrated_x_m += self._rng.gauss(0.0, self._odom_position_noise_std)
            integrated_y_m += self._rng.gauss(0.0, self._odom_position_noise_std)

        self._degraded_state = DegradedPlanarState(
            x_m=integrated_x_m,
            y_m=integrated_y_m,
            z_m=true_state.z_m,
            roll_rad=true_state.roll_rad,
            pitch_rad=true_state.pitch_rad,
            yaw_rad=integrated_yaw_rad,
            stamp_sec=true_state.stamp_sec,
        )

        bridged.pose.pose.position.x = self._degraded_state.x_m
        bridged.pose.pose.position.y = self._degraded_state.y_m
        bridged.pose.pose.position.z = self._degraded_state.z_m
        qx, qy, qz, qw = self._quaternion_from_rpy(
            self._degraded_state.roll_rad,
            self._degraded_state.pitch_rad,
            self._degraded_state.yaw_rad,
        )
        bridged.pose.pose.orientation.x = qx
        bridged.pose.pose.orientation.y = qy
        bridged.pose.pose.orientation.z = qz
        bridged.pose.pose.orientation.w = qw

        bridged.twist.twist.linear.x = measured_vx_mps
        bridged.twist.twist.linear.y = measured_vy_mps
        bridged.twist.twist.linear.z = measured_vz_mps
        bridged.twist.twist.angular.x = measured_wx_rps
        bridged.twist.twist.angular.y = measured_wy_rps
        bridged.twist.twist.angular.z = measured_wz_rps

        return bridged

    def _build_transform(self, bridged: Odometry) -> TransformStamped:
        transform = TransformStamped()
        transform.header = bridged.header
        transform.child_frame_id = bridged.child_frame_id
        transform.transform.translation.x = bridged.pose.pose.position.x
        transform.transform.translation.y = bridged.pose.pose.position.y
        transform.transform.translation.z = bridged.pose.pose.position.z
        transform.transform.rotation = bridged.pose.pose.orientation
        return transform

    def _publish_bridged_odom(self, bridged: Odometry, transform: TransformStamped) -> None:
        self._odom_pub.publish(bridged)
        self._tf_broadcaster.sendTransform(transform)

    def _flush_pending_queue(self) -> None:
        if not self._pending_queue:
            return
        now_sec = float(self.get_clock().now().nanoseconds) * 1.0e-9
        while self._pending_queue and self._pending_queue[0].release_time_sec <= now_sec:
            pending = self._pending_queue.popleft()
            self._publish_bridged_odom(pending.odom, pending.transform)

    def _odom_cb(self, msg: Odometry) -> None:
        bridged = self._build_bridged_odom(msg)
        transform = self._build_transform(bridged)

        # Latency is applied only to the degraded mode so the pure ideal path
        # remains a byte-for-byte baseline for regression and SLAM comparison.
        if self._degrade_odom and self._odom_latency_sec > 0.0:
            release_time_sec = (
                float(self.get_clock().now().nanoseconds) * 1.0e-9
            ) + self._odom_latency_sec
            self._pending_queue.append(
                PendingOdomPublish(
                    release_time_sec=release_time_sec,
                    odom=bridged,
                    transform=transform,
                )
            )
            return

        self._publish_bridged_odom(bridged, transform)


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
