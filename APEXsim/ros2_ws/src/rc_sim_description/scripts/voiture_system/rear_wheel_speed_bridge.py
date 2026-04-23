#!/usr/bin/env python3

import math
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

def _add_algorithms_to_path() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "algorithms"),
        os.path.join(here, "..", "algorithms"),
    ]
    for candidate in [os.path.abspath(p) for p in candidates]:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


_add_algorithms_to_path()

from interface_motor import RearWheelSpeedBridgeCore


class RearWheelSpeedBridge(Node):
    def __init__(self) -> None:
        super().__init__("rear_wheel_speed_bridge")
        self.declare_parameter("rear_wheel_speed_topic", "/rear_wheel_speed")
        self.declare_parameter(
            "rear_left_cmd_topic",
            "/model/rc_car/joint/rear_left_wheel_joint/cmd_vel",
        )
        self.declare_parameter(
            "rear_right_cmd_topic",
            "/model/rc_car/joint/rear_right_wheel_joint/cmd_vel",
        )
        self.declare_parameter("steering_angle_topic", "/steering_angle")
        self.declare_parameter(
            "front_left_steer_topic",
            "/model/rc_car/joint/front_left_wheel_steer_joint/cmd_pos",
        )
        self.declare_parameter(
            "front_right_steer_topic",
            "/model/rc_car/joint/front_right_wheel_steer_joint/cmd_pos",
        )
        self.declare_parameter("wheel_diameter", 0.065)
        self.declare_parameter("wheel_base", 0.32)
        self.declare_parameter("track_width", 0.29)
        self.declare_parameter("steering_limit", 0.6)
        self.declare_parameter("lp_enable", False)
        self.declare_parameter("lp_alpha", 0.2)
        self.declare_parameter("publish_rate", 30.0)

        speed_topic = (
            self.get_parameter("rear_wheel_speed_topic")
            .get_parameter_value()
            .string_value
        )
        rear_left_cmd_topic = (
            self.get_parameter("rear_left_cmd_topic")
            .get_parameter_value()
            .string_value
        )
        rear_right_cmd_topic = (
            self.get_parameter("rear_right_cmd_topic")
            .get_parameter_value()
            .string_value
        )
        steering_topic = (
            self.get_parameter("steering_angle_topic")
            .get_parameter_value()
            .string_value
        )
        front_left_steer_topic = (
            self.get_parameter("front_left_steer_topic")
            .get_parameter_value()
            .string_value
        )
        front_right_steer_topic = (
            self.get_parameter("front_right_steer_topic")
            .get_parameter_value()
            .string_value
        )
        self._wheel_diameter = (
            self.get_parameter("wheel_diameter").get_parameter_value().double_value
        )
        self._wheel_base = (
            self.get_parameter("wheel_base").get_parameter_value().double_value
        )
        self._track_width = (
            self.get_parameter("track_width").get_parameter_value().double_value
        )
        self._steering_limit = (
            self.get_parameter("steering_limit").get_parameter_value().double_value
        )
        self._lp_enable = bool(self.get_parameter("lp_enable").value)
        self._lp_alpha = float(self.get_parameter("lp_alpha").value)
        self._publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )

        self._bridge = RearWheelSpeedBridgeCore(
            rear_left_cmd_topic=rear_left_cmd_topic,
            rear_right_cmd_topic=rear_right_cmd_topic,
            front_left_cmd_topic=front_left_steer_topic,
            front_right_cmd_topic=front_right_steer_topic,
        )

        self._last_speed = None
        self._last_delta = 0.0
        self._prev_delta = None
        self.create_subscription(Float64, speed_topic, self._on_speed, 10)
        self.create_subscription(Float64, steering_topic, self._on_steering, 10)

        if self._publish_rate > 0.0:
            self._timer = self.create_timer(
                1.0 / self._publish_rate, self._republish
            )
        else:
            self._timer = None

    def _on_speed(self, msg: Float64) -> None:
        self._last_speed = msg.data
        self._publish_from_state()

    def _on_steering(self, msg: Float64) -> None:
        self._last_delta = msg.data
        self._publish_from_state()

    def _republish(self) -> None:
        if self._last_speed is not None:
            self._publish_from_state()

    def _publish_from_state(self) -> None:
        if self._last_speed is None:
            return
        v = float(self._last_speed)
        delta = float(self._last_delta)
        if self._steering_limit > 0.0:
            limit = abs(self._steering_limit)
            delta = max(-limit, min(delta, limit))
        if self._lp_enable:
            delta = self._low_pass_delta(delta)
        omega_rl, omega_rr = self._rear_wheel_omegas(v, delta)
        self._bridge.publish_left_right(omega_rl, omega_rr)
        left_delta, right_delta = self._ackermann_angles(delta)
        if self._steering_limit > 0.0:
            limit = abs(self._steering_limit)
            left_delta = max(-limit, min(left_delta, limit))
            right_delta = max(-limit, min(right_delta, limit))
        self._bridge.publish_front_steer(left_delta, right_delta)

    def _to_rad_per_sec(self, linear_speed_m_s: float) -> float:
        diameter = self._wheel_diameter if self._wheel_diameter > 0.0 else 0.065
        radius = diameter / 2.0
        return linear_speed_m_s / radius

    def _rear_wheel_omegas(self, v: float, delta: float) -> tuple[float, float]:
        if self._wheel_base <= 0.0:
            psi_dot = 0.0
        else:
            psi_dot = (v / self._wheel_base) * math.tan(delta)
        half_track = self._track_width / 2.0
        v_rr = v + psi_dot * half_track
        v_rl = v - psi_dot * half_track
        omega_rr = self._to_rad_per_sec(v_rr)
        omega_rl = self._to_rad_per_sec(v_rl)
        return omega_rl, omega_rr

    def _ackermann_angles(self, delta: float) -> tuple[float, float]:
        if self._wheel_base <= 0.0:
            return 0.0, 0.0
        tan_val = math.tan(delta)
        if abs(tan_val) < 1e-6:
            return 0.0, 0.0
        radius = self._wheel_base / tan_val
        half_track = self._track_width / 2.0
        inner = radius - half_track
        outer = radius + half_track
        inner = self._avoid_zero(inner, radius)
        outer = self._avoid_zero(outer, radius)
        left = math.atan(self._wheel_base / inner)
        right = math.atan(self._wheel_base / outer)
        return left, right

    def _low_pass_delta(self, delta: float) -> float:
        alpha = max(0.0, min(self._lp_alpha, 1.0))
        if self._prev_delta is None:
            self._prev_delta = delta
            return delta
        filtered = self._prev_delta + alpha * (delta - self._prev_delta)
        self._prev_delta = filtered
        return filtered

    @staticmethod
    def _avoid_zero(value: float, sign_hint: float) -> float:
        if abs(value) < 1e-6:
            sign = 1.0 if sign_hint >= 0.0 else -1.0
            return sign * 1e-6
        return value


def main() -> None:
    rclpy.init()
    node = RearWheelSpeedBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
