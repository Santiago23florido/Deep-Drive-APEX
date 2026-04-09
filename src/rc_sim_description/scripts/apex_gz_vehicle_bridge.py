#!/usr/bin/env python3
"""Drive the Gazebo RC car model from the real APEX PWM outputs."""

from __future__ import annotations

import json
import math
import time

import rclpy
from gz.msgs10 import double_pb2
from gz.transport13 import Node as GzNode
from rclpy.node import Node
from std_msgs.msg import Float64, String


class ApexGzVehicleBridge(Node):
    def __init__(self) -> None:
        super().__init__("apex_gz_vehicle_bridge")

        self.declare_parameter("publish_rate_hz", 120.0)
        self.declare_parameter("wheel_radius_m", 0.06)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("track_width_m", 0.29)
        self.declare_parameter("steering_limit_deg", 18.0)
        self.declare_parameter("steering_rate_limit_deg_per_s", 75.0)
        self.declare_parameter("steering_response_tau_s", 0.08)
        self.declare_parameter("steering_left_ratio", 1.0)
        self.declare_parameter("steering_right_ratio", 0.96)
        self.declare_parameter("steering_dc_min", 5.0)
        self.declare_parameter("steering_dc_max", 8.6)
        self.declare_parameter("steering_center_trim_dc", 1.4)
        self.declare_parameter("steering_direction_sign", -1.0)
        self.declare_parameter("steering_min_authority_ratio", 0.90)
        self.declare_parameter("steering_pwm_topic", "/apex/sim/pwm/steering_dc")
        self.declare_parameter("motor_pwm_topic", "/apex/sim/pwm/motor_dc")
        self.declare_parameter("status_topic", "/apex/sim/vehicle_state")
        self.declare_parameter("rear_left_cmd_topic", "/model/rc_car/joint/rear_left_wheel_joint/cmd_vel")
        self.declare_parameter("rear_right_cmd_topic", "/model/rc_car/joint/rear_right_wheel_joint/cmd_vel")
        self.declare_parameter(
            "front_left_cmd_topic",
            "/model/rc_car/joint/front_left_wheel_steer_joint/cmd_pos",
        )
        self.declare_parameter(
            "front_right_cmd_topic",
            "/model/rc_car/joint/front_right_wheel_steer_joint/cmd_pos",
        )
        self.declare_parameter("motor_neutral_dc", 7.5)
        self.declare_parameter("motor_forward_deadband_dc", 7.72)
        self.declare_parameter("motor_forward_top_dc", 8.55)
        self.declare_parameter("motor_max_forward_speed_mps", 0.55)
        self.declare_parameter("motor_min_effective_speed_mps", 0.07)
        self.declare_parameter("motor_accel_limit_mps2", 0.85)
        self.declare_parameter("motor_decel_limit_mps2", 1.20)
        self.declare_parameter("motor_response_tau_s", 0.22)

        self._publish_rate_hz = max(10.0, float(self.get_parameter("publish_rate_hz").value))
        self._wheel_radius_m = max(1.0e-3, float(self.get_parameter("wheel_radius_m").value))
        self._wheelbase_m = max(1.0e-3, float(self.get_parameter("wheelbase_m").value))
        self._track_width_m = max(1.0e-3, float(self.get_parameter("track_width_m").value))
        self._steering_limit_deg = max(1.0, float(self.get_parameter("steering_limit_deg").value))
        self._steering_limit_rad = math.radians(self._steering_limit_deg)
        self._steering_rate_limit_deg_per_s = max(
            0.0, float(self.get_parameter("steering_rate_limit_deg_per_s").value)
        )
        self._steering_response_tau_s = max(
            1.0e-3, float(self.get_parameter("steering_response_tau_s").value)
        )
        self._steering_left_ratio = max(0.1, float(self.get_parameter("steering_left_ratio").value))
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
        self._steering_pwm_topic = str(self.get_parameter("steering_pwm_topic").value)
        self._motor_pwm_topic = str(self.get_parameter("motor_pwm_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._motor_neutral_dc = float(self.get_parameter("motor_neutral_dc").value)
        self._motor_forward_deadband_dc = float(
            self.get_parameter("motor_forward_deadband_dc").value
        )
        self._motor_forward_top_dc = float(self.get_parameter("motor_forward_top_dc").value)
        self._motor_max_forward_speed_mps = max(
            0.01, float(self.get_parameter("motor_max_forward_speed_mps").value)
        )
        self._motor_min_effective_speed_mps = max(
            0.0, float(self.get_parameter("motor_min_effective_speed_mps").value)
        )
        self._motor_accel_limit_mps2 = max(
            0.0, float(self.get_parameter("motor_accel_limit_mps2").value)
        )
        self._motor_decel_limit_mps2 = max(
            0.0, float(self.get_parameter("motor_decel_limit_mps2").value)
        )
        self._motor_response_tau_s = max(
            1.0e-3, float(self.get_parameter("motor_response_tau_s").value)
        )

        raw_center = 0.5 * (self._steering_dc_min + self._steering_dc_max) + self._steering_center_trim_dc
        half_span = 0.5 * (self._steering_dc_max - self._steering_dc_min)
        required_half_span = self._steering_min_authority_ratio * half_span
        dc_center_min = self._steering_dc_min + required_half_span
        dc_center_max = self._steering_dc_max - required_half_span
        if dc_center_min <= dc_center_max:
            self._steering_dc_center = min(max(raw_center, dc_center_min), dc_center_max)
        else:
            self._steering_dc_center = 0.5 * (self._steering_dc_min + self._steering_dc_max)
        self._steering_variation_per_deg = 0.5 * (
            self._steering_dc_max - self._steering_dc_min
        ) / self._steering_limit_deg

        self._requested_steering_dc = self._steering_dc_center
        self._requested_motor_dc = self._motor_neutral_dc
        self._target_steering_deg = 0.0
        self._applied_steering_deg = 0.0
        self._target_speed_mps = 0.0
        self._applied_speed_mps = 0.0
        self._last_step_monotonic = time.monotonic()

        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self.create_subscription(Float64, self._steering_pwm_topic, self._steering_pwm_cb, 20)
        self.create_subscription(Float64, self._motor_pwm_topic, self._motor_pwm_cb, 20)

        self._gz_node = GzNode()
        self._rear_left_pub = self._gz_node.advertise(
            str(self.get_parameter("rear_left_cmd_topic").value),
            double_pb2.Double,
        )
        self._rear_right_pub = self._gz_node.advertise(
            str(self.get_parameter("rear_right_cmd_topic").value),
            double_pb2.Double,
        )
        self._front_left_pub = self._gz_node.advertise(
            str(self.get_parameter("front_left_cmd_topic").value),
            double_pb2.Double,
        )
        self._front_right_pub = self._gz_node.advertise(
            str(self.get_parameter("front_right_cmd_topic").value),
            double_pb2.Double,
        )

        self.create_timer(1.0 / self._publish_rate_hz, self._step)
        self.get_logger().info(
            "ApexGzVehicleBridge started (motor_pwm=%s steering_pwm=%s status=%s)"
            % (self._motor_pwm_topic, self._steering_pwm_topic, self._status_topic)
        )

    def _steering_pwm_cb(self, msg: Float64) -> None:
        self._requested_steering_dc = float(msg.data)

    def _motor_pwm_cb(self, msg: Float64) -> None:
        self._requested_motor_dc = float(msg.data)

    @staticmethod
    def _first_order(current: float, target: float, tau_s: float, dt_s: float) -> float:
        if dt_s <= 0.0:
            return target
        alpha = 1.0 - math.exp(-dt_s / max(1.0e-6, tau_s))
        return current + alpha * (target - current)

    @staticmethod
    def _rate_limit(current: float, target: float, rate_per_s: float, dt_s: float) -> float:
        if rate_per_s <= 0.0 or dt_s <= 0.0:
            return target
        max_delta = rate_per_s * dt_s
        delta = target - current
        if abs(delta) <= max_delta:
            return target
        return current + math.copysign(max_delta, delta)

    def _steering_target_from_pwm(self, duty_cycle_pct: float) -> float:
        duty_cycle_pct = max(self._steering_dc_min, min(self._steering_dc_max, duty_cycle_pct))
        if abs(self._steering_variation_per_deg) <= 1.0e-9:
            return 0.0
        signed_deg = (duty_cycle_pct - self._steering_dc_center) / self._steering_variation_per_deg
        requested_deg = signed_deg * self._steering_direction_sign
        requested_deg = max(-self._steering_limit_deg, min(self._steering_limit_deg, requested_deg))
        if requested_deg >= 0.0:
            requested_deg *= self._steering_left_ratio
        else:
            requested_deg *= self._steering_right_ratio
        return max(-self._steering_limit_deg, min(self._steering_limit_deg, requested_deg))

    def _speed_target_from_pwm(self, duty_cycle_pct: float) -> float:
        duty_cycle_pct = float(duty_cycle_pct)
        if duty_cycle_pct <= self._motor_forward_deadband_dc:
            return 0.0

        denom = max(1.0e-6, self._motor_forward_top_dc - self._motor_forward_deadband_dc)
        ratio = max(0.0, min(1.0, (duty_cycle_pct - self._motor_forward_deadband_dc) / denom))
        speed_mps = ratio * self._motor_max_forward_speed_mps
        if speed_mps > 1.0e-6:
            speed_mps = max(speed_mps, self._motor_min_effective_speed_mps)
        return speed_mps

    def _ackermann_angles(self, center_angle_rad: float) -> tuple[float, float]:
        tan_val = math.tan(center_angle_rad)
        if abs(tan_val) < 1.0e-6:
            return 0.0, 0.0
        radius = self._wheelbase_m / tan_val
        half_track = 0.5 * self._track_width_m
        left_angle = math.atan(self._wheelbase_m / (radius - half_track))
        right_angle = math.atan(self._wheelbase_m / (radius + half_track))
        return left_angle, right_angle

    def _rear_wheel_omegas(self, speed_mps: float, steering_rad: float) -> tuple[float, float]:
        if abs(steering_rad) < 1.0e-6:
            wheel_omega = speed_mps / self._wheel_radius_m
            return wheel_omega, wheel_omega
        yaw_rate = (speed_mps / self._wheelbase_m) * math.tan(steering_rad)
        half_track = 0.5 * self._track_width_m
        left_speed = speed_mps - (yaw_rate * half_track)
        right_speed = speed_mps + (yaw_rate * half_track)
        return left_speed / self._wheel_radius_m, right_speed / self._wheel_radius_m

    @staticmethod
    def _publish_double(publisher, value: float) -> None:
        msg = double_pb2.Double()
        msg.data = float(value)
        publisher.publish(msg)

    def _step(self) -> None:
        now_monotonic = time.monotonic()
        dt_s = max(1.0 / self._publish_rate_hz, now_monotonic - self._last_step_monotonic)
        self._last_step_monotonic = now_monotonic

        self._target_steering_deg = self._steering_target_from_pwm(self._requested_steering_dc)
        steering_after_tau = self._first_order(
            self._applied_steering_deg,
            self._target_steering_deg,
            self._steering_response_tau_s,
            dt_s,
        )
        self._applied_steering_deg = self._rate_limit(
            self._applied_steering_deg,
            steering_after_tau,
            self._steering_rate_limit_deg_per_s,
            dt_s,
        )
        self._applied_steering_deg = max(
            -self._steering_limit_deg,
            min(self._steering_limit_deg, self._applied_steering_deg),
        )

        self._target_speed_mps = self._speed_target_from_pwm(self._requested_motor_dc)
        speed_after_tau = self._first_order(
            self._applied_speed_mps,
            self._target_speed_mps,
            self._motor_response_tau_s,
            dt_s,
        )
        rate_limit = (
            self._motor_accel_limit_mps2
            if speed_after_tau >= self._applied_speed_mps
            else self._motor_decel_limit_mps2
        )
        self._applied_speed_mps = self._rate_limit(
            self._applied_speed_mps,
            speed_after_tau,
            rate_limit,
            dt_s,
        )
        if abs(self._target_speed_mps) <= 1.0e-6 and abs(self._applied_speed_mps) < 1.0e-3:
            self._applied_speed_mps = 0.0

        steering_rad = math.radians(self._applied_steering_deg)
        left_steer_rad, right_steer_rad = self._ackermann_angles(steering_rad)
        rear_left_omega, rear_right_omega = self._rear_wheel_omegas(
            self._applied_speed_mps,
            steering_rad,
        )

        self._publish_double(self._rear_left_pub, rear_left_omega)
        self._publish_double(self._rear_right_pub, rear_right_omega)
        self._publish_double(self._front_left_pub, left_steer_rad)
        self._publish_double(self._front_right_pub, right_steer_rad)

        status_msg = String()
        status_msg.data = json.dumps(
            {
                "requested_motor_dc": self._requested_motor_dc,
                "requested_steering_dc": self._requested_steering_dc,
                "target_speed_mps": self._target_speed_mps,
                "applied_speed_mps": self._applied_speed_mps,
                "target_steering_deg": self._target_steering_deg,
                "applied_steering_deg": self._applied_steering_deg,
                "front_left_steering_rad": left_steer_rad,
                "front_right_steering_rad": right_steer_rad,
                "rear_left_wheel_radps": rear_left_omega,
                "rear_right_wheel_radps": rear_right_omega,
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status_msg)


def main() -> None:
    rclpy.init()
    node = ApexGzVehicleBridge()
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
