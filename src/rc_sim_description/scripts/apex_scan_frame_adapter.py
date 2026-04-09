#!/usr/bin/env python3
"""Normalize and geometrically align Gazebo ideal LaserScan for slam_toolbox."""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class ApexScanFrameAdapter(Node):
    def __init__(self) -> None:
        super().__init__("apex_scan_frame_adapter")

        self.declare_parameter("source_scan_topic", "/apex/sim/scan")
        self.declare_parameter("target_scan_topic", "/apex/sim/scan_ideal")
        self.declare_parameter("target_frame_id", "laser")
        self.declare_parameter("enable_tf_yaw_correction", True)

        self._source_scan_topic = str(self.get_parameter("source_scan_topic").value)
        self._target_scan_topic = str(self.get_parameter("target_scan_topic").value)
        self._target_frame_id = str(self.get_parameter("target_frame_id").value)
        self._enable_tf_yaw_correction = bool(
            self.get_parameter("enable_tf_yaw_correction").value
        )
        self._cached_yaw_offsets_rad: dict[str, float] = {}
        self._warned_missing_tf_frames: set[str] = set()
        self._reported_aligned_frames: set[str] = set()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        self._pub = self.create_publisher(
            LaserScan,
            self._target_scan_topic,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            self._source_scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "ApexScanFrameAdapter started (source=%s target=%s frame=%s tf_yaw_correction=%s)"
            % (
                self._source_scan_topic,
                self._target_scan_topic,
                self._target_frame_id,
                str(self._enable_tf_yaw_correction).lower(),
            )
        )

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * ((w * z) + (x * y))
        cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
        return math.atan2(siny_cosp, cosy_cosp)

    def _lookup_yaw_offset_rad(self, source_frame_id: str, stamp) -> float:
        if source_frame_id == self._target_frame_id:
            return 0.0
        if source_frame_id in self._cached_yaw_offsets_rad:
            return self._cached_yaw_offsets_rad[source_frame_id]
        if not self._enable_tf_yaw_correction:
            return 0.0

        try:
            transform = self._tf_buffer.lookup_transform(
                self._target_frame_id,
                source_frame_id,
                Time.from_msg(stamp),
            )
        except TransformException as exc:
            if source_frame_id not in self._warned_missing_tf_frames:
                self._warned_missing_tf_frames.add(source_frame_id)
                self.get_logger().warn(
                    "No TF from scan frame '%s' to target frame '%s'; publishing with frame rename only. (%s)"
                    % (source_frame_id, self._target_frame_id, str(exc))
                )
            return 0.0

        rotation = transform.transform.rotation
        yaw_offset_rad = self._yaw_from_quaternion(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        self._cached_yaw_offsets_rad[source_frame_id] = yaw_offset_rad

        translation = transform.transform.translation
        if source_frame_id not in self._reported_aligned_frames:
            self._reported_aligned_frames.add(source_frame_id)
            self.get_logger().info(
                "Aligned scan frame '%s' to '%s' with yaw %.2f deg and translation (%.3f, %.3f, %.3f)"
                % (
                    source_frame_id,
                    self._target_frame_id,
                    math.degrees(yaw_offset_rad),
                    translation.x,
                    translation.y,
                    translation.z,
                )
            )
        return yaw_offset_rad

    @staticmethod
    def _roll_list(values: list[float], shift_bins: int) -> list[float]:
        if not values or shift_bins == 0:
            return list(values)
        shift_bins %= len(values)
        if shift_bins == 0:
            return list(values)
        return list(values[-shift_bins:]) + list(values[:-shift_bins])

    def _scan_cb(self, msg: LaserScan) -> None:
        source_frame_id = str(msg.header.frame_id).strip()
        yaw_offset_rad = self._lookup_yaw_offset_rad(source_frame_id, msg.header.stamp)

        out = LaserScan()
        out.header = msg.header
        out.header.frame_id = self._target_frame_id
        # Gazebo can expose a sensor-scoped scan frame that differs from the
        # URDF laser link. When TF is available, rotate the scan into the real
        # target frame instead of only renaming the header.
        angle_increment = float(msg.angle_increment)
        shift_bins = 0
        residual_yaw_rad = yaw_offset_rad
        if abs(angle_increment) > 1.0e-9 and abs(yaw_offset_rad) > 1.0e-9:
            shift_bins = int(round(yaw_offset_rad / angle_increment))
            residual_yaw_rad = yaw_offset_rad - (shift_bins * angle_increment)

        out.angle_min = float(msg.angle_min) + residual_yaw_rad
        out.angle_max = float(msg.angle_max) + residual_yaw_rad
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = self._roll_list(list(msg.ranges), shift_bins)
        out.intensities = self._roll_list(list(msg.intensities), shift_bins)
        self._pub.publish(out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ApexScanFrameAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
