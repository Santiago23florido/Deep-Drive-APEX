#!/usr/bin/env python3
"""Ackermann drive node for real RC hardware (ESC + steering servo)."""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Float64

from .real_motor_driver import RealMotorDriver
from .real_steering_driver import RealSteeringDriver


class AckermannDriveNode(Node):
    def __init__(self) -> None:
        super().__init__("ackermann_drive_node")

        # Vehicle geometry and limits.
        self.declare_parameter("wheelbase_m", 0.30)  # 2 * 0.15m (COM->rear)
        self.declare_parameter("rear_axle_to_com_m", 0.15)
        self.declare_parameter("front_half_track_m", 0.05)
        self.declare_parameter("max_steering_deg", 18.0)
        self.declare_parameter("steering_sign", 1.0)
        # 220 rpm, wheel radius 0.025 m -> v_max ~= 0.576 m/s
        self.declare_parameter("max_linear_speed_mps", 5.0)
        self.declare_parameter("speed_limit_pct", 95.0)
        self.declare_parameter("min_effective_speed_norm", 0.28)

        # Input / output topics.
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("applied_speed_topic", "/vehicle/motor_speed_cmd")
        self.declare_parameter("applied_steering_topic", "/vehicle/steering_angle_cmd_rad")
        self.declare_parameter("rear_wheel_speed_topic", "/rear_wheel_speed")
        self.declare_parameter("steering_angle_topic", "/steering_angle")

        # Safety / timing.
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("control_rate_hz", 50.0)

        # ESC calibration (full_soft defaults).
        self.declare_parameter("motor_pwm_channel", 0)
        self.declare_parameter("motor_pwm_frequency_hz", 50.0)
        self.declare_parameter("esc_dc_min", 5.0)
        self.declare_parameter("esc_dc_max", 10.0)

        # Steering calibration (full_soft defaults).
        self.declare_parameter("steering_pwm_channel", 1)
        self.declare_parameter("steering_pwm_frequency_hz", 50.0)
        self.declare_parameter("dc_steer_min", 5.0)
        self.declare_parameter("dc_steer_max", 8.6)

        self._wheelbase_m = max(1e-4, float(self.get_parameter("wheelbase_m").value))
        self._max_steering_rad = math.radians(abs(float(self.get_parameter("max_steering_deg").value)))
        self._steering_sign = -1.0 if float(self.get_parameter("steering_sign").value) < 0.0 else 1.0
        self._max_linear_speed_mps = max(1e-4, abs(float(self.get_parameter("max_linear_speed_mps").value)))
        self._speed_limit_pct = max(1.0, min(100.0, float(self.get_parameter("speed_limit_pct").value)))
        self._min_effective_speed_norm = max(0.0, min(3.0, float(self.get_parameter("min_effective_speed_norm").value)))

        self._command_timeout_s = max(0.1, float(self.get_parameter("command_timeout_s").value))
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))

        self._motor = RealMotorDriver(
            pwm_channel=int(self.get_parameter("motor_pwm_channel").value),
            pwm_frequency_hz=float(self.get_parameter("motor_pwm_frequency_hz").value),
            esc_dc_min=float(self.get_parameter("esc_dc_min").value),
            esc_dc_max=float(self.get_parameter("esc_dc_max").value),
        )
        self._steering = RealSteeringDriver(
            pwm_channel=int(self.get_parameter("steering_pwm_channel").value),
            pwm_frequency_hz=float(self.get_parameter("steering_pwm_frequency_hz").value),
            steering_limit_deg=abs(float(self.get_parameter("max_steering_deg").value)),
            dc_steer_min=float(self.get_parameter("dc_steer_min").value),
            dc_steer_max=float(self.get_parameter("dc_steer_max").value),
        )

        self._pub_applied_speed = self.create_publisher(Float64, str(self.get_parameter("applied_speed_topic").value), 10)
        self._pub_applied_steer = self.create_publisher(Float64, str(self.get_parameter("applied_steering_topic").value), 10)
        self._pub_rear_speed = self.create_publisher(Float64, str(self.get_parameter("rear_wheel_speed_topic").value), 10)
        self._pub_steer_angle = self.create_publisher(Float64, str(self.get_parameter("steering_angle_topic").value), 10)

        self._last_cmd_time = 0.0
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        self._last_applied_speed_norm = 0.0
        self._last_applied_steer_rad = 0.0

        self.create_subscription(Twist, str(self.get_parameter("cmd_vel_topic").value), self._on_cmd_vel, 10)
        self._timer = self.create_timer(1.0 / self._control_rate_hz, self._control_step)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            (
                "AckermannDriveNode started "
                "(wheelbase=%.3fm, max_steer=%.1fdeg, steering_sign=%.1f, "
                "max_linear=%.3f m/s, speed_limit_pct=%.1f, min_effective=%.2f)"
            )
            % (
                self._wheelbase_m,
                math.degrees(self._max_steering_rad),
                self._steering_sign,
                self._max_linear_speed_mps,
                self._speed_limit_pct,
                self._min_effective_speed_norm,
            )
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_v = float(msg.linear.x)
        self._cmd_w = float(msg.angular.z)
        self._last_cmd_time = time.time()

    def _compute_steering(self, v_mps: float, yaw_rate: float) -> float:
        if abs(v_mps) < 1e-4:
            return 0.0
        steer = self._steering_sign * math.atan(self._wheelbase_m * yaw_rate / v_mps)
        return max(-self._max_steering_rad, min(self._max_steering_rad, steer))

    def _to_motor_speed_norm(self, v_mps: float) -> float:
        speed_norm = 3.0 * (v_mps / self._max_linear_speed_mps)
        speed_norm *= self._speed_limit_pct / 100.0
        # Deadband compensation across full range:
        # keep command shape but shift any non-zero command above neutral zone.
        if abs(speed_norm) > 1e-4:
            mag = min(3.0, abs(speed_norm))
            if self._min_effective_speed_norm > 0.0:
                mag = self._min_effective_speed_norm + (mag / 3.0) * (3.0 - self._min_effective_speed_norm)
            speed_norm = math.copysign(mag, speed_norm)
        return max(-3.0, min(3.0, speed_norm))

    def _on_set_parameters(self, params: list) -> SetParametersResult:
        for p in params:
            if p.name == "speed_limit_pct":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="speed_limit_pct must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="speed_limit_pct must be > 0")
                self._speed_limit_pct = max(1.0, min(100.0, value))
                self.get_logger().info("Updated speed_limit_pct=%.1f" % self._speed_limit_pct)
            elif p.name == "max_linear_speed_mps":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="max_linear_speed_mps must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="max_linear_speed_mps must be > 0")
                self._max_linear_speed_mps = value
                self.get_logger().info("Updated max_linear_speed_mps=%.3f" % self._max_linear_speed_mps)
            elif p.name == "steering_sign":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="steering_sign must be numeric")
                if not math.isfinite(value) or abs(value) < 1e-6:
                    return SetParametersResult(successful=False, reason="steering_sign must be non-zero")
                self._steering_sign = -1.0 if value < 0.0 else 1.0
                self.get_logger().info("Updated steering_sign=%.1f" % self._steering_sign)
            elif p.name == "min_effective_speed_norm":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="min_effective_speed_norm must be numeric")
                if not math.isfinite(value) or value < 0.0 or value > 3.0:
                    return SetParametersResult(successful=False, reason="min_effective_speed_norm must be in [0,3]")
                self._min_effective_speed_norm = value
                self.get_logger().info("Updated min_effective_speed_norm=%.2f" % self._min_effective_speed_norm)
        return SetParametersResult(successful=True)

    def _publish_applied(self, speed_norm: float, steer_rad: float) -> None:
        msg_speed = Float64()
        msg_speed.data = float(speed_norm)
        self._pub_applied_speed.publish(msg_speed)

        msg_steer = Float64()
        msg_steer.data = float(steer_rad)
        self._pub_applied_steer.publish(msg_steer)

        # Compatibility topics used by existing stack.
        self._pub_rear_speed.publish(msg_speed)
        self._pub_steer_angle.publish(msg_steer)

    def _control_step(self) -> None:
        now = time.time()
        timed_out = (now - self._last_cmd_time) > self._command_timeout_s
        v = 0.0 if timed_out else self._cmd_v
        w = 0.0 if timed_out else self._cmd_w

        steer_rad = self._compute_steering(v, w)
        speed_norm = self._to_motor_speed_norm(v)

        self._last_applied_speed_norm = self._motor.set_speed(speed_norm)
        applied_deg = self._steering.set_steering_angle_deg(math.degrees(steer_rad))
        self._last_applied_steer_rad = math.radians(applied_deg)

        self._publish_applied(self._last_applied_speed_norm, self._last_applied_steer_rad)

    def destroy_node(self) -> bool:
        try:
            self._motor.stop()
        finally:
            self._steering.stop()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = AckermannDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
