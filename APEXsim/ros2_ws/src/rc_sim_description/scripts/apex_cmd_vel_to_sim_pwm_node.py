#!/usr/bin/env python3
"""Convert manual cmd_vel commands into Gazebo PWM topics for ideal sim mapping."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64


class ApexCmdVelToSimPwmNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_cmd_vel_to_sim_pwm_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("motor_pwm_topic", "/apex/sim/pwm/motor_dc")
        self.declare_parameter("steering_pwm_topic", "/apex/sim/pwm/steering_dc")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("max_linear_speed_mps", 1.50)
        self.declare_parameter("steering_limit_deg", 18.0)
        self.declare_parameter("steering_left_ratio", 1.0)
        self.declare_parameter("steering_right_ratio", 0.96)
        self.declare_parameter("steering_dc_min", 5.0)
        self.declare_parameter("steering_dc_max", 8.6)
        self.declare_parameter("steering_center_trim_dc", 1.4)
        self.declare_parameter("steering_direction_sign", -1.0)
        self.declare_parameter("steering_min_authority_ratio", 0.90)
        self.declare_parameter("motor_neutral_dc", 7.5)
        self.declare_parameter("motor_forward_deadband_dc", 7.72)
        self.declare_parameter("motor_forward_top_dc", 8.55)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._motor_pwm_topic = str(self.get_parameter("motor_pwm_topic").value)
        self._steering_pwm_topic = str(self.get_parameter("steering_pwm_topic").value)
        self._publish_rate_hz = max(5.0, float(self.get_parameter("publish_rate_hz").value))
        self._command_timeout_s = max(0.05, float(self.get_parameter("command_timeout_s").value))
        self._wheelbase_m = max(1.0e-3, float(self.get_parameter("wheelbase_m").value))
        self._max_linear_speed_mps = max(
            0.05, float(self.get_parameter("max_linear_speed_mps").value)
        )
        self._steering_limit_deg = max(
            1.0, float(self.get_parameter("steering_limit_deg").value)
        )
        self._steering_left_ratio = max(
            0.1, float(self.get_parameter("steering_left_ratio").value)
        )
        self._steering_right_ratio = max(
            0.1, float(self.get_parameter("steering_right_ratio").value)
        )
        self._steering_dc_min = float(self.get_parameter("steering_dc_min").value)
        self._steering_dc_max = float(self.get_parameter("steering_dc_max").value)
        self._steering_center_trim_dc = float(
            self.get_parameter("steering_center_trim_dc").value
        )
        self._steering_direction_sign = (
            -1.0 if float(self.get_parameter("steering_direction_sign").value) < 0.0 else 1.0
        )
        self._steering_min_authority_ratio = max(
            0.0, min(1.0, float(self.get_parameter("steering_min_authority_ratio").value))
        )
        self._motor_neutral_dc = float(self.get_parameter("motor_neutral_dc").value)
        self._motor_forward_deadband_dc = float(
            self.get_parameter("motor_forward_deadband_dc").value
        )
        self._motor_forward_top_dc = float(self.get_parameter("motor_forward_top_dc").value)

        raw_center = (
            0.5 * (self._steering_dc_min + self._steering_dc_max)
            + self._steering_center_trim_dc
        )
        half_span = 0.5 * (self._steering_dc_max - self._steering_dc_min)
        required_half_span = self._steering_min_authority_ratio * half_span
        center_min = self._steering_dc_min + required_half_span
        center_max = self._steering_dc_max - required_half_span
        if center_min <= center_max:
            self._steering_dc_center = min(max(raw_center, center_min), center_max)
        else:
            self._steering_dc_center = 0.5 * (self._steering_dc_min + self._steering_dc_max)
        self._steering_variation_per_deg = (
            0.5 * (self._steering_dc_max - self._steering_dc_min) / self._steering_limit_deg
        )

        self._last_linear_x_mps = 0.0
        self._last_angular_z_rps = 0.0
        self._last_command_stamp = self.get_clock().now()

        self._motor_pub = self.create_publisher(Float64, self._motor_pwm_topic, 20)
        self._steering_pub = self.create_publisher(Float64, self._steering_pwm_topic, 20)
        self.create_subscription(Twist, self._cmd_vel_topic, self._cmd_vel_cb, 20)
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_pwm)

        self.get_logger().info(
            "ApexCmdVelToSimPwmNode started (cmd=%s motor_pwm=%s steering_pwm=%s)"
            % (
                self._cmd_vel_topic,
                self._motor_pwm_topic,
                self._steering_pwm_topic,
            )
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._last_linear_x_mps = float(msg.linear.x)
        self._last_angular_z_rps = float(msg.angular.z)
        self._last_command_stamp = self.get_clock().now()

    def _command_is_stale(self) -> bool:
        age_s = (self.get_clock().now() - self._last_command_stamp).nanoseconds * 1.0e-9
        return age_s > self._command_timeout_s

    def _steering_pwm_from_cmd(self, linear_x_mps: float, angular_z_rps: float) -> float:
        if abs(self._steering_variation_per_deg) <= 1.0e-9:
            return self._steering_dc_center

        if abs(linear_x_mps) <= 1.0e-4 or abs(angular_z_rps) <= 1.0e-4:
            steering_deg = 0.0
        else:
            steering_deg = math.degrees(
                math.atan((self._wheelbase_m * angular_z_rps) / max(1.0e-6, linear_x_mps))
            )
        steering_deg = max(-self._steering_limit_deg, min(self._steering_limit_deg, steering_deg))
        if steering_deg >= 0.0:
            pre_ratio_deg = steering_deg / self._steering_left_ratio
        else:
            pre_ratio_deg = steering_deg / self._steering_right_ratio
        signed_deg = pre_ratio_deg / self._steering_direction_sign
        duty_cycle_pct = self._steering_dc_center + (signed_deg * self._steering_variation_per_deg)
        return max(self._steering_dc_min, min(self._steering_dc_max, duty_cycle_pct))

    def _motor_pwm_from_cmd(self, linear_x_mps: float) -> float:
        forward_speed_mps = max(0.0, linear_x_mps)
        if forward_speed_mps <= 1.0e-4:
            return self._motor_neutral_dc
        ratio = max(0.0, min(1.0, forward_speed_mps / self._max_linear_speed_mps))
        return self._motor_forward_deadband_dc + (
            ratio * (self._motor_forward_top_dc - self._motor_forward_deadband_dc)
        )

    def _publish_pwm(self) -> None:
        if self._command_is_stale():
            linear_x_mps = 0.0
            angular_z_rps = 0.0
        else:
            linear_x_mps = self._last_linear_x_mps
            angular_z_rps = self._last_angular_z_rps

        motor_msg = Float64()
        motor_msg.data = float(self._motor_pwm_from_cmd(linear_x_mps))
        self._motor_pub.publish(motor_msg)

        steering_msg = Float64()
        steering_msg.data = float(self._steering_pwm_from_cmd(linear_x_mps, angular_z_rps))
        self._steering_pub.publish(steering_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ApexCmdVelToSimPwmNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
