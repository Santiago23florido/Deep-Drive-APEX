#!/usr/bin/env python3
"""Bridge `/apex/cmd_vel_track` into direct APEX ESC + steering commands."""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64, String

from .actuation import MaverickESCMotor, SteeringServo


class CmdVelToApexActuationNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_to_apex_actuation_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("applied_speed_topic", "/apex/vehicle/applied_speed_pct")
        self.declare_parameter("applied_steering_topic", "/apex/vehicle/applied_steering_deg")
        self.declare_parameter("status_topic", "/apex/vehicle/drive_bridge_status")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("command_timeout_s", 0.35)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("max_steering_deg", 18.0)
        self.declare_parameter("max_linear_speed_mps", 0.55)
        self.declare_parameter("min_effective_speed_pct", 40.0)
        self.declare_parameter("max_speed_pct", 55.0)
        self.declare_parameter("speed_pct_ramp_duration_s", 1.0)
        self.declare_parameter("speed_pct_ramp_exp_k", 4.0)
        self.declare_parameter("speed_pct_ramp_down_per_s", 24.0)
        self.declare_parameter("launch_boost_speed_pct", 0.0)
        self.declare_parameter("launch_boost_hold_s", 0.0)
        self.declare_parameter("steering_rate_limit_deg_per_s", 90.0)
        self.declare_parameter("active_brake_on_zero", False)

        self.declare_parameter("steering_channel", 1)
        self.declare_parameter("steering_frequency_hz", 50.0)
        self.declare_parameter("steering_limit_deg", 18.0)
        self.declare_parameter("steering_dc_min", 5.0)
        self.declare_parameter("steering_dc_max", 8.6)
        self.declare_parameter("steering_center_trim_dc", 1.4)
        self.declare_parameter("steering_direction_sign", -1.0)
        self.declare_parameter("steering_min_authority_ratio", 0.90)

        self.declare_parameter("motor_channel", 0)
        self.declare_parameter("motor_frequency_hz", 50.0)
        self.declare_parameter("motor_dc_min", 5.0)
        self.declare_parameter("motor_dc_max", 10.0)
        self.declare_parameter("motor_neutral_dc", 7.5)
        self.declare_parameter("reverse_brake_dc", 6.9)
        self.declare_parameter("reverse_brake_hold_s", 0.12)
        self.declare_parameter("reverse_neutral_hold_s", 0.12)
        self.declare_parameter("reverse_exit_hold_s", 0.15)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._applied_speed_topic = str(self.get_parameter("applied_speed_topic").value)
        self._applied_steering_topic = str(self.get_parameter("applied_steering_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._command_timeout_s = max(0.05, float(self.get_parameter("command_timeout_s").value))
        self._wheelbase_m = max(1e-3, float(self.get_parameter("wheelbase_m").value))
        self._max_steering_deg = max(1.0, float(self.get_parameter("max_steering_deg").value))
        self._max_linear_speed_mps = max(0.05, float(self.get_parameter("max_linear_speed_mps").value))
        self._min_effective_speed_pct = max(
            0.0, min(100.0, float(self.get_parameter("min_effective_speed_pct").value))
        )
        self._max_speed_pct = max(
            self._min_effective_speed_pct,
            min(100.0, float(self.get_parameter("max_speed_pct").value)),
        )
        self._speed_pct_ramp_duration_s = max(
            0.0, float(self.get_parameter("speed_pct_ramp_duration_s").value)
        )
        self._speed_pct_ramp_exp_k = max(
            1.0e-3, float(self.get_parameter("speed_pct_ramp_exp_k").value)
        )
        self._speed_pct_ramp_down_per_s = max(
            0.0, float(self.get_parameter("speed_pct_ramp_down_per_s").value)
        )
        self._launch_boost_speed_pct = max(
            0.0, min(100.0, float(self.get_parameter("launch_boost_speed_pct").value))
        )
        self._launch_boost_hold_s = max(
            0.0, float(self.get_parameter("launch_boost_hold_s").value)
        )
        self._steering_rate_limit_deg_per_s = max(
            0.0, float(self.get_parameter("steering_rate_limit_deg_per_s").value)
        )
        self._active_brake_on_zero = bool(self.get_parameter("active_brake_on_zero").value)

        self._motor = MaverickESCMotor(
            channel=int(self.get_parameter("motor_channel").value),
            frequency_hz=float(self.get_parameter("motor_frequency_hz").value),
            dc_min=float(self.get_parameter("motor_dc_min").value),
            dc_max=float(self.get_parameter("motor_dc_max").value),
            neutral_dc=float(self.get_parameter("motor_neutral_dc").value),
            reverse_brake_dc=float(self.get_parameter("reverse_brake_dc").value),
            reverse_brake_hold_s=float(self.get_parameter("reverse_brake_hold_s").value),
            reverse_neutral_hold_s=float(self.get_parameter("reverse_neutral_hold_s").value),
            reverse_exit_hold_s=float(self.get_parameter("reverse_exit_hold_s").value),
            logger=self.get_logger(),
        )
        self._steering = SteeringServo(
            channel=int(self.get_parameter("steering_channel").value),
            frequency_hz=float(self.get_parameter("steering_frequency_hz").value),
            limit_deg=float(self.get_parameter("steering_limit_deg").value),
            dc_min=float(self.get_parameter("steering_dc_min").value),
            dc_max=float(self.get_parameter("steering_dc_max").value),
            center_trim_dc=float(self.get_parameter("steering_center_trim_dc").value),
            direction_sign=float(self.get_parameter("steering_direction_sign").value),
            min_authority_ratio=float(self.get_parameter("steering_min_authority_ratio").value),
            logger=self.get_logger(),
        )
        self._steering.center()
        self._motor.hold_neutral(disable_pwm=False)

        self.create_subscription(Twist, self._cmd_vel_topic, self._cmd_cb, 20)
        self._speed_pub = self.create_publisher(Float64, self._applied_speed_topic, 20)
        self._steering_pub = self.create_publisher(Float64, self._applied_steering_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self.create_timer(1.0 / self._control_rate_hz, self._control_step)

        self._cmd_linear_x_mps = 0.0
        self._cmd_angular_z_rps = 0.0
        self._last_cmd_monotonic = 0.0
        self._last_applied_speed_pct = 0.0
        self._last_applied_steering_deg = 0.0
        self._last_requested_steering_deg = 0.0
        self._last_desired_speed_pct = 0.0
        self._last_desired_steering_deg = 0.0
        self._last_steering_saturated = False
        self._last_control_monotonic = time.monotonic()
        self._speed_ramp_start_monotonic: float | None = None
        self._speed_ramp_start_pct = 0.0
        self._speed_ramp_target_pct = 0.0
        self._launch_boost_until_monotonic = 0.0

        self.get_logger().info(
            "CmdVelToApexActuationNode started (cmd=%s max_linear=%.2f m/s speed_pct=%.1f..%.1f)"
            % (
                self._cmd_vel_topic,
                self._max_linear_speed_mps,
                self._min_effective_speed_pct,
                self._max_speed_pct,
            )
        )

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_linear_x_mps = max(0.0, float(msg.linear.x))
        self._cmd_angular_z_rps = float(msg.angular.z)
        self._last_cmd_monotonic = time.monotonic()

    def _map_linear_speed_to_pct(self, linear_x_mps: float) -> float:
        if linear_x_mps <= 1e-4:
            return 0.0
        ratio = max(0.0, min(1.0, linear_x_mps / self._max_linear_speed_mps))
        return self._min_effective_speed_pct + (
            ratio * (self._max_speed_pct - self._min_effective_speed_pct)
        )

    def _compute_steering_deg(self, linear_x_mps: float, angular_z_rps: float) -> float:
        if linear_x_mps <= 1e-4:
            return 0.0
        steering_rad = math.atan((self._wheelbase_m * angular_z_rps) / linear_x_mps)
        return math.degrees(steering_rad)

    def _start_speed_ramp(self, *, target_speed_pct: float, now_monotonic: float) -> None:
        self._speed_ramp_start_monotonic = now_monotonic
        self._launch_boost_until_monotonic = 0.0
        if (
            target_speed_pct > 1.0e-6
            and self._last_applied_speed_pct <= 1.0e-6
            and max(self._min_effective_speed_pct, self._launch_boost_speed_pct) > 1.0e-6
        ):
            launch_start_pct = self._min_effective_speed_pct
            if self._launch_boost_speed_pct > 1.0e-6:
                launch_start_pct = max(launch_start_pct, self._launch_boost_speed_pct)
                if self._launch_boost_hold_s > 1.0e-6:
                    self._launch_boost_until_monotonic = (
                        now_monotonic + self._launch_boost_hold_s
                    )
            # Do not ramp up from a value that is already below the ESC's
            # effective movement threshold; otherwise the vehicle can sit still
            # for most of the launch ramp even though motion has been commanded.
            self._speed_ramp_start_pct = min(100.0, launch_start_pct)
        else:
            self._speed_ramp_start_pct = self._last_applied_speed_pct
        self._speed_ramp_target_pct = target_speed_pct

    def _compute_exponential_ramp_speed(self, now_monotonic: float) -> tuple[float, float]:
        if self._speed_ramp_start_monotonic is None or self._speed_pct_ramp_duration_s <= 1e-6:
            return self._speed_ramp_target_pct, 1.0

        elapsed_s = max(0.0, now_monotonic - self._speed_ramp_start_monotonic)
        progress = min(1.0, elapsed_s / self._speed_pct_ramp_duration_s)
        if progress >= 1.0:
            self._speed_ramp_start_monotonic = None
            return self._speed_ramp_target_pct, 1.0

        exp_k = self._speed_pct_ramp_exp_k
        normalized = (1.0 - math.exp(-exp_k * progress)) / (1.0 - math.exp(-exp_k))
        speed_pct = self._speed_ramp_start_pct + (
            normalized * (self._speed_ramp_target_pct - self._speed_ramp_start_pct)
        )
        return speed_pct, progress

    @staticmethod
    def _apply_rate_limit(
        *,
        current_value: float,
        target_value: float,
        rate_up_per_s: float,
        rate_down_per_s: float,
        dt_s: float,
    ) -> float:
        if dt_s <= 0.0:
            return target_value
        delta = target_value - current_value
        if delta >= 0.0:
            max_delta = rate_up_per_s * dt_s
            if rate_up_per_s <= 0.0:
                return target_value
        else:
            max_delta = rate_down_per_s * dt_s
            if rate_down_per_s <= 0.0:
                return target_value
        if abs(delta) <= max_delta:
            return target_value
        return current_value + math.copysign(max_delta, delta)

    def _publish_state(self, *, timed_out: bool) -> None:
        speed_msg = Float64()
        speed_msg.data = float(self._last_applied_speed_pct)
        self._speed_pub.publish(speed_msg)

        steer_msg = Float64()
        steer_msg.data = float(self._last_applied_steering_deg)
        self._steering_pub.publish(steer_msg)

        status = String()
        status.data = json.dumps(
            {
                "state": "timeout_hold" if timed_out else "tracking",
                "timed_out": bool(timed_out),
                "desired_linear_x_mps": self._cmd_linear_x_mps,
                "desired_angular_z_rps": self._cmd_angular_z_rps,
                "desired_speed_pct": self._last_desired_speed_pct,
                "desired_steering_deg": self._last_desired_steering_deg,
                "requested_steering_deg": self._last_requested_steering_deg,
                "speed_ramp_start_pct": self._speed_ramp_start_pct,
                "speed_ramp_target_pct": self._speed_ramp_target_pct,
                "speed_ramp_active": self._speed_ramp_start_monotonic is not None,
                "launch_boost_active": self._launch_boost_until_monotonic > time.monotonic(),
                "launch_boost_speed_pct": self._launch_boost_speed_pct,
                "applied_speed_pct": self._last_applied_speed_pct,
                "applied_steering_deg": self._last_applied_steering_deg,
                "steering_saturated": self._last_steering_saturated,
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status)

    def _control_step(self) -> None:
        now_monotonic = time.monotonic()
        dt_s = max(1.0 / self._control_rate_hz, now_monotonic - self._last_control_monotonic)
        self._last_control_monotonic = now_monotonic
        timed_out = (now_monotonic - self._last_cmd_monotonic) > self._command_timeout_s
        if timed_out:
            desired_linear_x_mps = 0.0
            desired_angular_z_rps = 0.0
        else:
            desired_linear_x_mps = self._cmd_linear_x_mps
            desired_angular_z_rps = self._cmd_angular_z_rps

        desired_steering_deg = self._compute_steering_deg(
            desired_linear_x_mps,
            desired_angular_z_rps,
        )
        desired_speed_pct = self._map_linear_speed_to_pct(desired_linear_x_mps)

        requested_steering_deg = self._apply_rate_limit(
            current_value=self._last_requested_steering_deg,
            target_value=desired_steering_deg,
            rate_up_per_s=self._steering_rate_limit_deg_per_s,
            rate_down_per_s=self._steering_rate_limit_deg_per_s,
            dt_s=dt_s,
        )
        if desired_speed_pct > 1e-6:
            if (
                self._speed_ramp_start_monotonic is None
                or abs(desired_speed_pct - self._speed_ramp_target_pct) > 1.0e-6
            ):
                self._start_speed_ramp(
                    target_speed_pct=desired_speed_pct,
                    now_monotonic=now_monotonic,
                )
            applied_speed_pct, _ = self._compute_exponential_ramp_speed(now_monotonic)
            if (
                self._launch_boost_until_monotonic > now_monotonic
                and self._speed_ramp_start_pct > desired_speed_pct
            ):
                applied_speed_pct = max(applied_speed_pct, self._speed_ramp_start_pct)
        else:
            self._speed_ramp_start_monotonic = None
            self._speed_ramp_start_pct = self._last_applied_speed_pct
            self._speed_ramp_target_pct = 0.0
            self._launch_boost_until_monotonic = 0.0
            if self._active_brake_on_zero and self._last_applied_speed_pct > 1.0:
                requested_steering_deg = 0.0
                self._steering.set_angle_deg(0.0)
                self._motor.brake_to_neutral()
                applied_speed_pct = 0.0
            else:
                applied_speed_pct = self._apply_rate_limit(
                    current_value=self._last_applied_speed_pct,
                    target_value=0.0,
                    rate_up_per_s=self._speed_pct_ramp_down_per_s,
                    rate_down_per_s=self._speed_pct_ramp_down_per_s,
                    dt_s=dt_s,
                )

        self._steering.set_angle_deg(requested_steering_deg)
        steering_state = self._steering.get_state()
        applied_steering_deg = float(
            steering_state.get("applied_deg", requested_steering_deg)
        )
        self._motor.set_speed_pct(applied_speed_pct)
        self._last_desired_steering_deg = desired_steering_deg
        self._last_desired_speed_pct = desired_speed_pct
        self._last_requested_steering_deg = requested_steering_deg
        self._last_applied_steering_deg = applied_steering_deg
        self._last_applied_speed_pct = applied_speed_pct
        self._last_steering_saturated = abs(applied_steering_deg - requested_steering_deg) > 1.0e-3
        self._publish_state(timed_out=timed_out)

    def destroy_node(self) -> bool:
        try:
            self._motor.hold_neutral(disable_pwm=False)
        except Exception:
            pass
        try:
            self._steering.center()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CmdVelToApexActuationNode()
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
