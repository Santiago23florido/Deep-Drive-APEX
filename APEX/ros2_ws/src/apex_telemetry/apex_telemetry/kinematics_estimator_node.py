#!/usr/bin/env python3
"""Integrate IMU acceleration + gyro into kinematics for APEX telemetry."""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Trigger


GRAVITY_MPS2 = 9.80665


class KinematicsEstimatorNode(Node):
    """Compute kinematics from IMU acceleration and yaw rate."""

    def __init__(self) -> None:
        super().__init__("kinematics_estimator_node")

        self.declare_parameter("input_topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("gyro_input_topic", "/apex/imu/angular_velocity/raw")
        self.declare_parameter("acceleration_topic", "/apex/kinematics/acceleration")
        self.declare_parameter("velocity_topic", "/apex/kinematics/velocity")
        self.declare_parameter("position_topic", "/apex/kinematics/position")
        self.declare_parameter("angular_velocity_topic", "/apex/kinematics/angular_velocity")
        self.declare_parameter("heading_topic", "/apex/kinematics/heading")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("use_message_time", True)
        self.declare_parameter("max_dt_s", 0.1)
        self.declare_parameter("accel_low_pass_alpha", 0.35)
        self.declare_parameter("deadband_mps2", 0.03)
        self.declare_parameter("gyro_low_pass_alpha", 0.45)
        self.declare_parameter("gyro_deadband_rps", 0.005)
        self.declare_parameter("integrate_accel_in_world_frame", True)
        self.declare_parameter("use_gyro_yaw", True)
        self.declare_parameter("gravity_compensation_enabled", True)
        self.declare_parameter("gravity_axis", "z")
        self.declare_parameter("accel_bias_x", 0.0)
        self.declare_parameter("accel_bias_y", 0.0)
        self.declare_parameter("accel_bias_z", 0.0)
        self.declare_parameter("gyro_bias_x", 0.0)
        self.declare_parameter("gyro_bias_y", 0.0)
        self.declare_parameter("gyro_bias_z", 0.0)

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._use_message_time = bool(self.get_parameter("use_message_time").value)
        self._max_dt_s = max(0.001, float(self.get_parameter("max_dt_s").value))
        self._accel_alpha = min(
            1.0, max(0.0, float(self.get_parameter("accel_low_pass_alpha").value))
        )
        self._accel_deadband = max(0.0, float(self.get_parameter("deadband_mps2").value))
        self._gyro_alpha = min(
            1.0, max(0.0, float(self.get_parameter("gyro_low_pass_alpha").value))
        )
        self._gyro_deadband = max(0.0, float(self.get_parameter("gyro_deadband_rps").value))
        self._integrate_accel_world = bool(self.get_parameter("integrate_accel_in_world_frame").value)
        self._use_gyro_yaw = bool(self.get_parameter("use_gyro_yaw").value)
        self._gravity_enabled = bool(self.get_parameter("gravity_compensation_enabled").value)
        self._gravity_axis = str(self.get_parameter("gravity_axis").value).lower()
        self._bias_x = float(self.get_parameter("accel_bias_x").value)
        self._bias_y = float(self.get_parameter("accel_bias_y").value)
        self._bias_z = float(self.get_parameter("accel_bias_z").value)
        self._gyro_bias_x = float(self.get_parameter("gyro_bias_x").value)
        self._gyro_bias_y = float(self.get_parameter("gyro_bias_y").value)
        self._gyro_bias_z = float(self.get_parameter("gyro_bias_z").value)

        input_topic = str(self.get_parameter("input_topic").value)
        gyro_input_topic = str(self.get_parameter("gyro_input_topic").value)
        accel_topic = str(self.get_parameter("acceleration_topic").value)
        velocity_topic = str(self.get_parameter("velocity_topic").value)
        position_topic = str(self.get_parameter("position_topic").value)
        angular_velocity_topic = str(self.get_parameter("angular_velocity_topic").value)
        heading_topic = str(self.get_parameter("heading_topic").value)

        self._sub = self.create_subscription(Vector3Stamped, input_topic, self._accel_callback, 50)
        self._gyro_sub = self.create_subscription(
            Vector3Stamped, gyro_input_topic, self._gyro_callback, 50
        )
        self._pub_accel = self.create_publisher(Vector3Stamped, accel_topic, 20)
        self._pub_vel = self.create_publisher(Vector3Stamped, velocity_topic, 20)
        self._pub_pos = self.create_publisher(PointStamped, position_topic, 20)
        self._pub_ang_vel = self.create_publisher(Vector3Stamped, angular_velocity_topic, 20)
        self._pub_heading = self.create_publisher(Vector3Stamped, heading_topic, 20)
        self._reset_srv = self.create_service(Trigger, "reset_kinematics", self._handle_reset)

        self._last_time: Optional[Time] = None
        self._last_gyro_time: Optional[Time] = None
        self._f_ax = 0.0
        self._f_ay = 0.0
        self._f_az = 0.0
        self._f_gx = 0.0
        self._f_gy = 0.0
        self._f_gz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0

        self.get_logger().info(
            "KinematicsEstimatorNode started (accel_in=%s gyro_in=%s accel=%s vel=%s pos=%s heading=%s)"
            % (
                input_topic,
                gyro_input_topic,
                accel_topic,
                velocity_topic,
                position_topic,
                heading_topic,
            )
        )

    def _apply_accel_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self._accel_deadband else value

    def _apply_gyro_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self._gyro_deadband else value

    def _apply_gravity_compensation(self, ax: float, ay: float, az: float) -> tuple[float, float, float]:
        if not self._gravity_enabled:
            return ax, ay, az

        if self._gravity_axis == "x":
            ax -= GRAVITY_MPS2
        elif self._gravity_axis == "y":
            ay -= GRAVITY_MPS2
        else:
            az -= GRAVITY_MPS2
        return ax, ay, az

    def _msg_time_or_now(self, msg: Vector3Stamped) -> Time:
        if self._use_message_time and (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0):
            return Time.from_msg(msg.header.stamp)
        return self.get_clock().now()

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    def _rotate_body_to_world(self, ax: float, ay: float) -> tuple[float, float]:
        if not self._integrate_accel_world:
            return ax, ay

        c = math.cos(self._yaw)
        s = math.sin(self._yaw)
        return (c * ax - s * ay, s * ax + c * ay)

    def _gyro_callback(self, msg: Vector3Stamped) -> None:
        now_t = self._msg_time_or_now(msg)

        gx = float(msg.vector.x) - self._gyro_bias_x
        gy = float(msg.vector.y) - self._gyro_bias_y
        gz = float(msg.vector.z) - self._gyro_bias_z

        if self._last_gyro_time is None:
            self._f_gx, self._f_gy, self._f_gz = gx, gy, gz
            self._yaw_rate = self._apply_gyro_deadband(gz)
            self._last_gyro_time = now_t
            self._publish_orientation(now_t)
            return

        dt = (now_t - self._last_gyro_time).nanoseconds * 1e-9
        self._last_gyro_time = now_t
        if dt <= 0.0:
            return
        if dt > self._max_dt_s:
            dt = self._max_dt_s

        self._f_gx = self._gyro_alpha * gx + (1.0 - self._gyro_alpha) * self._f_gx
        self._f_gy = self._gyro_alpha * gy + (1.0 - self._gyro_alpha) * self._f_gy
        self._f_gz = self._gyro_alpha * gz + (1.0 - self._gyro_alpha) * self._f_gz

        self._f_gx = self._apply_gyro_deadband(self._f_gx)
        self._f_gy = self._apply_gyro_deadband(self._f_gy)
        self._f_gz = self._apply_gyro_deadband(self._f_gz)
        self._yaw_rate = self._f_gz

        if self._use_gyro_yaw:
            self._yaw = self._normalize_angle(self._yaw + self._yaw_rate * dt)

        self._publish_orientation(now_t)

    def _accel_callback(self, msg: Vector3Stamped) -> None:
        now_t = self._msg_time_or_now(msg)

        ax_body = float(msg.vector.x) - self._bias_x
        ay_body = float(msg.vector.y) - self._bias_y
        az = float(msg.vector.z) - self._bias_z

        ax_body, ay_body, az = self._apply_gravity_compensation(ax_body, ay_body, az)
        ax, ay = self._rotate_body_to_world(ax_body, ay_body)

        if self._last_time is None:
            self._f_ax, self._f_ay, self._f_az = ax, ay, az
            self._last_time = now_t
            self._publish_kinematics(now_t)
            self._publish_orientation(now_t)
            return

        dt = (now_t - self._last_time).nanoseconds * 1e-9
        self._last_time = now_t
        if dt <= 0.0:
            return
        if dt > self._max_dt_s:
            dt = self._max_dt_s

        # Low-pass filter to reduce high-frequency IMU noise before integration.
        self._f_ax = self._accel_alpha * ax + (1.0 - self._accel_alpha) * self._f_ax
        self._f_ay = self._accel_alpha * ay + (1.0 - self._accel_alpha) * self._f_ay
        self._f_az = self._accel_alpha * az + (1.0 - self._accel_alpha) * self._f_az

        self._f_ax = self._apply_accel_deadband(self._f_ax)
        self._f_ay = self._apply_accel_deadband(self._f_ay)
        self._f_az = self._apply_accel_deadband(self._f_az)

        # Integrate acceleration -> velocity -> position.
        self._vx += self._f_ax * dt
        self._vy += self._f_ay * dt
        self._vz += self._f_az * dt

        self._px += self._vx * dt
        self._py += self._vy * dt
        self._pz += self._vz * dt

        self._publish_kinematics(now_t)
        self._publish_orientation(now_t)

    def _publish_kinematics(self, stamp_time: Time) -> None:
        stamp = stamp_time.to_msg()

        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = self._frame_id
        accel_msg.vector.x = self._f_ax
        accel_msg.vector.y = self._f_ay
        accel_msg.vector.z = self._f_az
        self._pub_accel.publish(accel_msg)

        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self._frame_id
        vel_msg.vector.x = self._vx
        vel_msg.vector.y = self._vy
        vel_msg.vector.z = self._vz
        self._pub_vel.publish(vel_msg)

        pos_msg = PointStamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = self._frame_id
        pos_msg.point.x = self._px
        pos_msg.point.y = self._py
        pos_msg.point.z = self._pz
        self._pub_pos.publish(pos_msg)

    def _publish_orientation(self, stamp_time: Time) -> None:
        stamp = stamp_time.to_msg()

        angular_msg = Vector3Stamped()
        angular_msg.header.stamp = stamp
        angular_msg.header.frame_id = self._frame_id
        angular_msg.vector.x = self._f_gx
        angular_msg.vector.y = self._f_gy
        angular_msg.vector.z = self._yaw_rate
        self._pub_ang_vel.publish(angular_msg)

        heading_msg = Vector3Stamped()
        heading_msg.header.stamp = stamp
        heading_msg.header.frame_id = self._frame_id
        heading_msg.vector.x = 0.0
        heading_msg.vector.y = 0.0
        heading_msg.vector.z = self._yaw
        self._pub_heading.publish(heading_msg)

    def _handle_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._f_ax = 0.0
        self._f_ay = 0.0
        self._f_az = 0.0
        self._f_gx = 0.0
        self._f_gy = 0.0
        self._f_gz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._yaw = 0.0
        self._yaw_rate = 0.0
        self._last_time = None
        self._last_gyro_time = None

        response.success = True
        response.message = "Kinematic state reset."
        self.get_logger().info(response.message)
        return response


def main() -> None:
    rclpy.init()
    node = KinematicsEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
