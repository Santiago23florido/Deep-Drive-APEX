#!/usr/bin/env python3
"""Bridge `/apex/cmd_vel_track` into direct APEX ESC + steering commands.

Two profiles, selected by ``actuation_profile``:

``legacy`` (the default, unchanged)
    What the recognition-tour stack was tuned with: a step to
    ``min_effective_speed_pct`` at any non-zero command, an exponential ramp, a
    launch boost and a ramp-down, plus direct gamepad steering override.

``affine``
    What the race stack needs. ``linear.x`` maps to ESC percent through one
    affine function per direction (:mod:`speed_map`) with no ramp, no boost and
    no deadband compensation, because ``apex_race.core.plant_id`` identifies the
    deadband, the gain and the actuator lag from the car's own odometry while it
    drives, and a bridge that also compensated them would be what gets
    identified. Direction changes run as a non-blocking phase machine instead of
    sleeping inside the control callback, and the servo holds its angle at a
    stop instead of snapping to centre.

In both profiles the computed steering angle is now clamped to
``max_steering_deg`` before it reaches the servo. It was not, and the car logged
46-48 deg requests against an 18 deg mechanical limit
(``data/reports/recognition_tour_diagnostic_report.md``).
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64, String

from .actuation import MaverickESCMotor, SteeringServo
from .speed_map import SpeedMap, steer_deg_from_twist


class CmdVelToApexActuationNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_to_apex_actuation_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("applied_speed_topic", "/apex/vehicle/applied_speed_pct")
        self.declare_parameter("applied_steering_topic", "/apex/vehicle/applied_steering_deg")
        self.declare_parameter("status_topic", "/apex/vehicle/drive_bridge_status")
        self.declare_parameter("manual_control_status_topic", "/apex/manual_control/status")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("command_timeout_s", 0.35)
        self.declare_parameter("manual_status_stale_s", 0.50)
        self.declare_parameter("use_direct_manual_steering", True)
        self.declare_parameter("actuation_profile", "legacy")  # legacy | affine
        # `affine` profile only: the clean m/s -> ESC percent map. See speed_map.py.
        self.declare_parameter("esc_zero_pct", 15.0)
        self.declare_parameter("speed_pct_per_mps", 33.0)
        self.declare_parameter("esc_zero_rev_pct", 18.0)
        self.declare_parameter("reverse_pct_per_mps", 33.0)
        self.declare_parameter("decode_v_min_mps", 0.02)  # below this the servo holds its angle
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("max_steering_deg", 18.0)
        self.declare_parameter("max_steering_left_deg", 0.0)
        self.declare_parameter("max_steering_right_deg", 0.0)
        self.declare_parameter("max_linear_speed_mps", 0.55)
        self.declare_parameter("min_effective_speed_pct", 40.0)
        self.declare_parameter("max_speed_pct", 55.0)
        self.declare_parameter("min_effective_reverse_speed_pct", 25.0)
        self.declare_parameter("max_reverse_speed_pct", 25.0)
        self.declare_parameter("speed_pct_ramp_duration_s", 1.0)
        self.declare_parameter("speed_pct_ramp_exp_k", 4.0)
        self.declare_parameter("speed_pct_ramp_down_per_s", 24.0)
        self.declare_parameter("launch_boost_speed_pct", 0.0)
        self.declare_parameter("launch_boost_hold_s", 0.0)
        self.declare_parameter("steering_rate_limit_deg_per_s", 90.0)
        self.declare_parameter("active_brake_on_zero", False)
        self.declare_parameter("actuation_backend", "sysfs_pwm")
        self.declare_parameter("sim_motor_pwm_topic", "/apex/sim/pwm/motor_dc")
        self.declare_parameter("sim_steering_pwm_topic", "/apex/sim/pwm/steering_dc")

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
        self._manual_control_status_topic = str(
            self.get_parameter("manual_control_status_topic").value
        )
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._command_timeout_s = max(0.05, float(self.get_parameter("command_timeout_s").value))
        self._manual_status_stale_s = max(
            0.05, float(self.get_parameter("manual_status_stale_s").value)
        )
        self._use_direct_manual_steering = bool(
            self.get_parameter("use_direct_manual_steering").value
        )
        self._wheelbase_m = max(1e-3, float(self.get_parameter("wheelbase_m").value))
        self._max_steering_deg = max(1.0, float(self.get_parameter("max_steering_deg").value))
        self._max_steering_left_deg = float(self.get_parameter("max_steering_left_deg").value) or self._max_steering_deg
        self._max_steering_right_deg = float(self.get_parameter("max_steering_right_deg").value) or self._max_steering_deg
        if not all(0.0 < v < 89.0 for v in (self._max_steering_left_deg, self._max_steering_right_deg)):
            raise ValueError("steering limits must be positive and below 89 degrees")
        self._asymmetric_steering = self._max_steering_left_deg != self._max_steering_right_deg
        self._max_linear_speed_mps = max(0.05, float(self.get_parameter("max_linear_speed_mps").value))
        self._min_effective_speed_pct = max(
            0.0, min(100.0, float(self.get_parameter("min_effective_speed_pct").value))
        )
        self._max_speed_pct = max(
            self._min_effective_speed_pct,
            min(100.0, float(self.get_parameter("max_speed_pct").value)),
        )
        self._min_effective_reverse_speed_pct = max(
            0.0,
            min(
                100.0,
                float(self.get_parameter("min_effective_reverse_speed_pct").value),
            ),
        )
        self._max_reverse_speed_pct = max(
            self._min_effective_reverse_speed_pct,
            min(100.0, float(self.get_parameter("max_reverse_speed_pct").value)),
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
        self._actuation_profile = str(self.get_parameter("actuation_profile").value).strip().lower()
        if self._actuation_profile not in ("legacy", "affine"):
            raise ValueError(f"actuation_profile must be legacy or affine, not {self._actuation_profile!r}")
        self._affine = self._actuation_profile == "affine"
        self._decode_v_min_mps = max(0.0, float(self.get_parameter("decode_v_min_mps").value))
        self._speed_map: SpeedMap | None = None
        if self._affine:
            self._speed_map = SpeedMap(
                esc_zero_pct=float(self.get_parameter("esc_zero_pct").value),
                pct_per_mps=float(self.get_parameter("speed_pct_per_mps").value),
                max_pct=self._max_speed_pct,
                esc_zero_rev_pct=float(self.get_parameter("esc_zero_rev_pct").value),
                rev_pct_per_mps=float(self.get_parameter("reverse_pct_per_mps").value),
                max_rev_pct=self._max_reverse_speed_pct,
            )
            # The shaper owns the launch and the identifier owns the deadband:
            # anything here that depends on the command history would be
            # identified instead of the car.
            for name, value in (("speed_pct_ramp_duration_s", self._speed_pct_ramp_duration_s),
                                ("launch_boost_speed_pct", self._launch_boost_speed_pct),
                                ("speed_pct_ramp_down_per_s", self._speed_pct_ramp_down_per_s)):
                if value > 1.0e-6:
                    self.get_logger().warning(
                        "affine profile: %s=%.3f is ignored (the adaptive layer owns it)" % (name, value)
                    )
            self._speed_pct_ramp_duration_s = 0.0
            self._launch_boost_speed_pct = 0.0
            self._speed_pct_ramp_down_per_s = 0.0
            if self._use_direct_manual_steering:
                self.get_logger().warning("affine profile: use_direct_manual_steering is forced off")
                self._use_direct_manual_steering = False
            if self._active_brake_on_zero:
                # brake_to_neutral() sleeps 0.24 s inside the control callback.
                self.get_logger().warning("affine profile: active_brake_on_zero is forced off")
                self._active_brake_on_zero = False
        self._actuation_backend = str(self.get_parameter("actuation_backend").value).strip().lower()
        self._sim_motor_pwm_topic = str(self.get_parameter("sim_motor_pwm_topic").value)
        self._sim_steering_pwm_topic = str(self.get_parameter("sim_steering_pwm_topic").value)

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
            backend=self._actuation_backend,
            node=self,
            publish_topic=self._sim_motor_pwm_topic,
            async_reverse=self._affine,
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
            backend=self._actuation_backend,
            node=self,
            publish_topic=self._sim_steering_pwm_topic,
            asymmetric_limits=self._asymmetric_steering,
        )
        if self._asymmetric_steering:
            authority = self._steering.get_state()
            if (self._max_steering_left_deg > authority["authority_left_deg"] + 1e-9
                    or self._max_steering_right_deg > authority["authority_right_deg"] + 1e-9):
                raise ValueError("directional steering limits exceed the configured servo PWM window")
        self._steering.center()
        self._motor.hold_neutral(disable_pwm=False)

        # Command reception must keep running while PWM I/O or status delivery
        # waits. The race bridge uses two executor threads, with short coherent
        # input snapshots; actuator operations still serialize on one group.
        self._command_lock = threading.RLock()
        self._command_group = MutuallyExclusiveCallbackGroup() if self._affine else None
        self._actuation_group = MutuallyExclusiveCallbackGroup() if self._affine else None
        self.create_subscription(Twist, self._cmd_vel_topic, self._observed_cmd_cb, 20,
                                 callback_group=self._command_group)
        self.create_subscription(
            String,
            self._manual_control_status_topic,
            self._manual_status_cb,
            20,
            callback_group=self._command_group,
        )
        self._speed_pub = self.create_publisher(Float64, self._applied_speed_topic, 20)
        self._steering_pub = self.create_publisher(Float64, self._applied_steering_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self.create_timer(1.0 / self._control_rate_hz, self._control_step,
                          callback_group=self._actuation_group)

        self._cmd_linear_x_mps = 0.0
        self._cmd_angular_z_rps = 0.0
        self._last_cmd_monotonic = 0.0
        self._last_applied_speed_pct = 0.0
        self._last_applied_steering_deg = 0.0
        self._last_requested_steering_deg = 0.0
        self._last_desired_speed_pct = 0.0
        self._last_desired_steering_deg = 0.0
        self._last_manual_status: dict[str, Any] = {}
        self._last_manual_status_monotonic = 0.0
        self._last_steering_saturated = False
        self._last_control_monotonic = time.monotonic()
        self._speed_ramp_start_monotonic: float | None = None
        self._speed_ramp_start_pct = 0.0
        self._speed_ramp_target_pct = 0.0
        self._launch_boost_until_monotonic = 0.0
        self._cmd_seq = 0
        self._cmd_gap_ms_max = 0.0
        self._previous_cmd_info: dict[str, int] | None = None
        self._steering_held = False

        self.get_logger().info(
            "CmdVelToApexActuationNode started (cmd=%s max_linear=%.2f m/s forward_pct=%.1f..%.1f reverse_pct=%.1f..%.1f)"
            % (
                self._cmd_vel_topic,
                self._max_linear_speed_mps,
                self._min_effective_speed_pct,
                self._max_speed_pct,
                self._min_effective_reverse_speed_pct,
                self._max_reverse_speed_pct,
            )
        )

    def _observed_cmd_cb(self, msg: Twist, info: dict) -> None:
        """Keep DDS timing evidence when the existing command deadline fails."""
        callback_ns = time.time_ns()
        previous = self._previous_cmd_info
        current = {
            "source_timestamp": int(info.get("source_timestamp", 0)),
            "received_timestamp": int(info.get("received_timestamp", 0)),
            "callback_timestamp": callback_ns,
        }
        last_cmd = self._last_cmd_monotonic
        self._cmd_cb(msg)
        self._previous_cmd_info = current
        gap_ms = 1e3 * (self._last_cmd_monotonic - last_cmd) if last_cmd > 0.0 else 0.0
        if gap_ms >= 100.0 and previous is not None:
            self.get_logger().warning("command delivery gap: " + json.dumps({
                "cmd_seq": self._cmd_seq,
                "callback_gap_ms": round(gap_ms, 3),
                "source_gap_ms": round((current["source_timestamp"] - previous["source_timestamp"]) / 1e6, 3),
                "receive_gap_ms": round((current["received_timestamp"] - previous["received_timestamp"]) / 1e6, 3),
                "source_to_receive_ms": round((current["received_timestamp"] - current["source_timestamp"]) / 1e6, 3),
                "receive_to_callback_ms": round((callback_ns - current["received_timestamp"]) / 1e6, 3),
                "desired_linear_x_mps": self._cmd_linear_x_mps,
            }, separators=(",", ":")))

    def _cmd_cb(self, msg: Twist) -> None:
        with self._command_lock:
            self._cmd_linear_x_mps = float(msg.linear.x)
            self._cmd_angular_z_rps = float(msg.angular.z)
            now = time.monotonic()
            # Measure actual reception gaps, independently of PWM/status work.
            if self._last_cmd_monotonic > 0.0:
                self._cmd_gap_ms_max = max(self._cmd_gap_ms_max, 1.0e3 * (now - self._last_cmd_monotonic))
            self._cmd_seq += 1
            self._last_cmd_monotonic = now

    def _manual_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        with self._command_lock:
            self._last_manual_status = payload
            self._last_manual_status_monotonic = time.monotonic()

    def _manual_status_is_fresh(self, now_monotonic: float) -> bool:
        if self._last_manual_status_monotonic <= 0.0:
            return False
        return (now_monotonic - self._last_manual_status_monotonic) <= self._manual_status_stale_s

    def _direct_manual_steering_deg(self, now_monotonic: float) -> float | None:
        if not self._use_direct_manual_steering:
            return None
        with self._command_lock:
            if not self._manual_status_is_fresh(now_monotonic):
                return None
            status = self._last_manual_status.copy()
        if not bool(status.get("bridge_connected", False)):
            return None
        if not bool(status.get("controller_connected", False)):
            return None
        if not bool(status.get("enabled", False)):
            return None
        try:
            steering_deg = float(status.get("steering_deg", 0.0))
        except Exception:
            return None
        return max(-self._max_steering_right_deg, min(self._max_steering_left_deg, steering_deg))

    def _map_linear_speed_to_pct(self, linear_x_mps: float) -> float:
        if self._speed_map is not None:
            return self._speed_map.pct(linear_x_mps)
        if abs(linear_x_mps) <= 1e-4:
            return 0.0
        ratio = max(0.0, min(1.0, abs(linear_x_mps) / self._max_linear_speed_mps))
        if linear_x_mps > 0.0:
            return self._min_effective_speed_pct + (
                ratio * (self._max_speed_pct - self._min_effective_speed_pct)
            )
        return -(
            self._min_effective_reverse_speed_pct
            + (ratio * (self._max_reverse_speed_pct - self._min_effective_reverse_speed_pct))
        )

    def _compute_steering_deg(self, linear_x_mps: float, angular_z_rps: float) -> float | None:
        """Steering angle [deg], clamped; ``None`` means hold the current angle.

        The clamp is the fix for the defect in the diagnostic report: the
        computed branch was unbounded and the servo's duty saturation hid it.
        """
        v_min = self._decode_v_min_mps if self._affine else 1.0e-4
        deg, saturated = steer_deg_from_twist(
            linear_x_mps, angular_z_rps, self._wheelbase_m, self._max_steering_deg, v_min,
            max_left_deg=self._max_steering_left_deg, max_right_deg=self._max_steering_right_deg,
        )
        if saturated:
            self.get_logger().warning(
                "steering request beyond left/right %.1f/%.1f deg, clamped"
                % (self._max_steering_left_deg, self._max_steering_right_deg),
                throttle_duration_sec=5.0,
            )
        if deg is None:
            # Below the decode speed the command carries no steering
            # information. The simulated bridge holds the angle; snapping to
            # centre at the end of every stop is an input nobody asked for.
            return None if self._affine else 0.0
        return deg

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
        with self._command_lock:
            command_state = {
                "cmd_seq": self._cmd_seq,
                "cmd_gap_ms_max": round(self._cmd_gap_ms_max, 2),
                "desired_linear_x_mps": self._cmd_linear_x_mps,
                "desired_angular_z_rps": self._cmd_angular_z_rps,
            }
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
                "actuation_backend": self._actuation_backend,
                "actuation_profile": self._actuation_profile,
                "speed_map": self._speed_map.describe() if self._speed_map is not None else None,
                "max_steering_deg": self._max_steering_deg,
                "max_steering_left_deg": self._max_steering_left_deg,
                "max_steering_right_deg": self._max_steering_right_deg,
                "steering_authority_deg": self._steering.authority_deg,
                "steering_held": self._steering_held,
                "decode_v_min_mps": self._decode_v_min_mps,
                **command_state,
                "motor_state": self._motor.get_state(),
                "timed_out": bool(timed_out),
                "desired_speed_pct": self._last_desired_speed_pct,
                "desired_steering_deg": self._last_desired_steering_deg,
                "requested_steering_deg": self._last_requested_steering_deg,
                "min_effective_speed_pct": self._min_effective_speed_pct,
                "max_speed_pct": self._max_speed_pct,
                "min_effective_reverse_speed_pct": self._min_effective_reverse_speed_pct,
                "max_reverse_speed_pct": self._max_reverse_speed_pct,
                "max_linear_speed_mps": self._max_linear_speed_mps,
                "speed_ramp_start_pct": self._speed_ramp_start_pct,
                "speed_ramp_target_pct": self._speed_ramp_target_pct,
                "speed_ramp_active": self._speed_ramp_start_monotonic is not None,
                "launch_boost_active": self._launch_boost_until_monotonic > time.monotonic(),
                "launch_boost_speed_pct": self._launch_boost_speed_pct,
                "applied_speed_pct": self._last_applied_speed_pct,
                "applied_steering_deg": self._last_applied_steering_deg,
                "steering_saturated": self._last_steering_saturated,
                "motor_pwm_state": self._motor.get_pwm_state(),
                "steering_pwm_state": self._steering.get_pwm_state(),
                "steering_servo_state": self._steering.get_state(),
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status)

    def _control_step(self) -> None:
        now_monotonic = time.monotonic()
        dt_s = max(1.0 / self._control_rate_hz, now_monotonic - self._last_control_monotonic)
        self._last_control_monotonic = now_monotonic
        with self._command_lock:
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
        manual_steering_deg = self._direct_manual_steering_deg(now_monotonic)
        if manual_steering_deg is not None:
            desired_steering_deg = manual_steering_deg
        self._steering_held = desired_steering_deg is None
        if desired_steering_deg is None:
            desired_steering_deg = self._last_requested_steering_deg
        desired_speed_pct = self._map_linear_speed_to_pct(desired_linear_x_mps)

        requested_steering_deg = self._apply_rate_limit(
            current_value=self._last_requested_steering_deg,
            target_value=desired_steering_deg,
            rate_up_per_s=self._steering_rate_limit_deg_per_s,
            rate_down_per_s=self._steering_rate_limit_deg_per_s,
            dt_s=dt_s,
        )
        if self._affine:
            # One affine map, no history: this is the whole point of the profile.
            self._speed_ramp_start_monotonic = None
            self._speed_ramp_start_pct = self._last_applied_speed_pct
            self._speed_ramp_target_pct = desired_speed_pct
            self._launch_boost_until_monotonic = 0.0
            applied_speed_pct = desired_speed_pct
        elif desired_speed_pct > 1e-6:
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
        elif desired_speed_pct < -1e-6:
            self._speed_ramp_start_monotonic = None
            self._speed_ramp_start_pct = self._last_applied_speed_pct
            self._speed_ramp_target_pct = desired_speed_pct
            self._launch_boost_until_monotonic = 0.0
            applied_speed_pct = desired_speed_pct
        else:
            self._speed_ramp_start_monotonic = None
            self._speed_ramp_start_pct = self._last_applied_speed_pct
            self._speed_ramp_target_pct = 0.0
            self._launch_boost_until_monotonic = 0.0
            if self._active_brake_on_zero and abs(self._last_applied_speed_pct) > 1.0:
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
        self._motor.tick()  # advance the non-blocking direction change, if any
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
    executor = MultiThreadedExecutor(num_threads=2) if node._affine else None
    try:
        if executor is None:
            rclpy.spin(node)
        else:
            executor.add_node(node)
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
