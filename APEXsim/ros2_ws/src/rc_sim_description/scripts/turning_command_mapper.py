#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TurningCommandMapper(Node):
    def __init__(self) -> None:
        super().__init__("turning_command_mapper")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("turning_mode", "steering_angle")
        self.declare_parameter("compute_ackermann", False)
        self.declare_parameter("compute_rear_wheels", False)
        self.declare_parameter("publish_rear_speed", True)
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("L", 0.32)
        self.declare_parameter("W", 0.29)
        self.declare_parameter("r", 0.06)
        self.declare_parameter("delta_max", 0.6)
        self.declare_parameter("v_eps", 1e-3)
        self.declare_parameter("tan_eps", 1e-4)
        self.declare_parameter("radius_is_signed", True)
        self.declare_parameter("radius_sign", 1.0)
        self.declare_parameter("zero_speed_behavior", "zero")
        self.declare_parameter("steering_angle_unit", "deg")
        self.declare_parameter("delta_topic", "/steering_angle")
        self.declare_parameter("delta_left_topic", "/steering_angle_left")
        self.declare_parameter("delta_right_topic", "/steering_angle_right")
        self.declare_parameter("rear_speed_topic", "/rear_wheel_speed")
        self.declare_parameter("omega_rl_topic", "/rear_left_wheel_omega")
        self.declare_parameter("omega_rr_topic", "/rear_right_wheel_omega")
        self.declare_parameter("psi_dot_topic", "/yaw_rate")
        self.declare_parameter("radius_topic", "/turn_radius")

        cmd_topic = self.get_parameter("cmd_topic").value
        self._turning_mode = str(self.get_parameter("turning_mode").value).lower()
        self._compute_ackermann = bool(self.get_parameter("compute_ackermann").value)
        self._compute_rear_wheels = bool(
            self.get_parameter("compute_rear_wheels").value
        )
        self._publish_rear_speed = bool(
            self.get_parameter("publish_rear_speed").value
        )
        self._publish_debug = bool(self.get_parameter("publish_debug").value)

        self._L = float(self.get_parameter("L").value)
        self._W = float(self.get_parameter("W").value)
        self._r = float(self.get_parameter("r").value)
        self._delta_max = abs(float(self.get_parameter("delta_max").value))
        self._v_eps = abs(float(self.get_parameter("v_eps").value))
        self._tan_eps = abs(float(self.get_parameter("tan_eps").value))
        self._radius_is_signed = bool(self.get_parameter("radius_is_signed").value)
        self._radius_sign = float(self.get_parameter("radius_sign").value)
        self._zero_speed_behavior = str(
            self.get_parameter("zero_speed_behavior").value
        ).lower()
        self._steering_angle_unit = str(
            self.get_parameter("steering_angle_unit").value
        ).lower()

        delta_topic = self.get_parameter("delta_topic").value
        delta_left_topic = self.get_parameter("delta_left_topic").value
        delta_right_topic = self.get_parameter("delta_right_topic").value
        rear_speed_topic = self.get_parameter("rear_speed_topic").value
        omega_rl_topic = self.get_parameter("omega_rl_topic").value
        omega_rr_topic = self.get_parameter("omega_rr_topic").value
        psi_dot_topic = self.get_parameter("psi_dot_topic").value
        radius_topic = self.get_parameter("radius_topic").value

        self._delta_pub = self.create_publisher(Float64, delta_topic, 10)
        self._delta_left_pub = None
        self._delta_right_pub = None
        self._rear_speed_pub = None
        self._omega_rl_pub = None
        self._omega_rr_pub = None
        self._psi_dot_pub = None
        self._radius_pub = None

        if self._compute_ackermann:
            self._delta_left_pub = self.create_publisher(Float64, delta_left_topic, 10)
            self._delta_right_pub = self.create_publisher(Float64, delta_right_topic, 10)

        if self._compute_rear_wheels:
            self._omega_rl_pub = self.create_publisher(Float64, omega_rl_topic, 10)
            self._omega_rr_pub = self.create_publisher(Float64, omega_rr_topic, 10)

        if self._publish_rear_speed:
            self._rear_speed_pub = self.create_publisher(Float64, rear_speed_topic, 10)

        if self._publish_debug:
            self._psi_dot_pub = self.create_publisher(Float64, psi_dot_topic, 10)
            self._radius_pub = self.create_publisher(Float64, radius_topic, 10)

        self._last_valid_delta = 0.0
        self._invalid_mode_logged = False
        self._invalid_unit_logged = False

        self.create_subscription(Twist, cmd_topic, self._on_cmd, 10)

    def _on_cmd(self, msg: Twist) -> None:
        v = float(msg.linear.x)
        turning = float(msg.angular.z)

        delta, valid = self._compute_delta(v, turning)
        delta = self._clamp(delta, -self._delta_max, self._delta_max)
        if math.isfinite(delta) and valid:
            self._last_valid_delta = delta

        self._publish_float(self._delta_pub, delta)

        if self._publish_rear_speed:
            rear_speed = self._rear_speed_from_v(v)
            self._publish_float(self._rear_speed_pub, rear_speed)

        need_radius = self._compute_ackermann or self._publish_debug
        need_yaw = self._compute_rear_wheels or self._publish_debug

        radius = None
        if need_radius:
            radius = self._radius_from_delta(delta)

        if self._compute_ackermann:
            left, right = self._ackermann_angles(radius)
            self._publish_float(self._delta_left_pub, left)
            self._publish_float(self._delta_right_pub, right)

        psi_dot = None
        if need_yaw:
            psi_dot = self._yaw_rate(v, delta)

        if self._compute_rear_wheels:
            omega_rl, omega_rr = self._rear_wheel_omegas(v, psi_dot)
            self._publish_float(self._omega_rl_pub, omega_rl)
            self._publish_float(self._omega_rr_pub, omega_rr)

        if self._publish_debug:
            if psi_dot is not None:
                self._publish_float(self._psi_dot_pub, psi_dot)
            if radius is not None:
                self._publish_float(self._radius_pub, radius)

    def _compute_delta(self, v: float, turning: float) -> tuple[float, bool]:
        if self._turning_mode == "curvature":
            return math.atan(self._L * turning), True
        if self._turning_mode == "radius":
            radius = turning
            if not self._radius_is_signed:
                sign = 1.0 if self._radius_sign >= 0.0 else -1.0
                radius = abs(radius) * sign
            if abs(radius) < self._tan_eps:
                if radius == 0.0 and not self._radius_is_signed:
                    sign = 1.0 if self._radius_sign >= 0.0 else -1.0
                else:
                    sign = 1.0 if radius >= 0.0 else -1.0
                radius = sign * max(self._tan_eps, 1e-6)
            return math.atan(self._L / radius), True
        if self._turning_mode in ("steering_angle", "steering", "delta"):
            if self._steering_angle_unit in ("deg", "degree", "degrees"):
                return math.radians(turning), True
            if self._steering_angle_unit in ("rad", "radian", "radians"):
                return turning, True
            if not self._invalid_unit_logged:
                self.get_logger().error(
                    f"Unknown steering_angle_unit '{self._steering_angle_unit}', "
                    "expected 'deg' or 'rad'."
                )
                self._invalid_unit_logged = True
            return turning, True
        if self._turning_mode == "yaw_rate":
            if abs(v) > self._v_eps:
                return math.atan(self._L * turning / v), True
            if self._zero_speed_behavior == "hold":
                return self._last_valid_delta, False
            return 0.0, False

        if not self._invalid_mode_logged:
            self.get_logger().error(
                f"Unknown turning_mode '{self._turning_mode}', "
                "expected 'curvature', 'radius', 'steering_angle', or 'yaw_rate'."
            )
            self._invalid_mode_logged = True
        return 0.0, False

    def _radius_from_delta(self, delta: float) -> float:
        if self._L <= 0.0:
            return math.inf
        tan_val = math.tan(delta)
        if abs(tan_val) <= self._tan_eps:
            return math.inf
        return self._L / tan_val

    def _ackermann_angles(self, radius: float) -> tuple[float, float]:
        if radius is None or math.isinf(radius):
            return 0.0, 0.0
        half_track = self._W / 2.0
        inner = radius - half_track
        outer = radius + half_track
        inner = self._avoid_zero(inner, radius)
        outer = self._avoid_zero(outer, radius)
        left = math.atan(self._L / inner)
        right = math.atan(self._L / outer)
        return left, right

    def _yaw_rate(self, v: float, delta: float) -> float:
        if self._L <= 0.0 or abs(v) <= self._v_eps:
            return 0.0
        return (v / self._L) * math.tan(delta)

    def _rear_wheel_omegas(self, v: float, psi_dot: float) -> tuple[float, float]:
        half_track = self._W / 2.0
        v_rr = v + half_track * psi_dot
        v_rl = v - half_track * psi_dot
        if self._r <= 0.0:
            return 0.0, 0.0
        omega_rr = v_rr / self._r
        omega_rl = v_rl / self._r
        return omega_rl, omega_rr

    def _rear_speed_from_v(self, v: float) -> float:
        if self._r <= 0.0:
            return 0.0
        return v / self._r

    def _avoid_zero(self, value: float, sign_hint: float) -> float:
        if abs(value) < self._tan_eps:
            sign = 1.0 if sign_hint >= 0.0 else -1.0
            return sign * max(self._tan_eps, 1e-6)
        return value

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    @staticmethod
    def _publish_float(publisher: Publisher, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = TurningCommandMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
