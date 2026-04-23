#!/usr/bin/env python3
"""Publish manual cmd_vel commands from an Xbox-style joystick via pygame."""

from __future__ import annotations

import json
import math
import os

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402


def _apply_deadband(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / max(1.0e-6, 1.0 - deadband)
    return math.copysign(min(1.0, scaled), value)


class ApexXboxManualTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_xbox_manual_teleop_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("status_topic", "/apex/sim/manual_control/status")
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("device_name_contains", "xbox")
        self.declare_parameter("joystick_index", -1)
        self.declare_parameter("linear_axis_index", 1)
        self.declare_parameter("steering_axis_index", 0)
        self.declare_parameter("linear_axis_invert", True)
        self.declare_parameter("steering_axis_invert", True)
        self.declare_parameter("axis_deadband", 0.10)
        self.declare_parameter("max_linear_speed_mps", 0.60)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("max_steering_deg", 18.0)
        self.declare_parameter("enable_button_index", -1)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._publish_rate_hz = max(2.0, float(self.get_parameter("publish_rate_hz").value))
        self._device_name_contains = (
            str(self.get_parameter("device_name_contains").value).strip().lower()
        )
        self._joystick_index = int(self.get_parameter("joystick_index").value)
        self._linear_axis_index = int(self.get_parameter("linear_axis_index").value)
        self._steering_axis_index = int(self.get_parameter("steering_axis_index").value)
        self._linear_axis_invert = bool(self.get_parameter("linear_axis_invert").value)
        self._steering_axis_invert = bool(self.get_parameter("steering_axis_invert").value)
        self._axis_deadband = max(0.0, min(0.4, float(self.get_parameter("axis_deadband").value)))
        self._max_linear_speed_mps = max(
            0.05, float(self.get_parameter("max_linear_speed_mps").value)
        )
        self._wheelbase_m = max(1.0e-3, float(self.get_parameter("wheelbase_m").value))
        self._max_steering_deg = max(1.0, float(self.get_parameter("max_steering_deg").value))
        self._enable_button_index = int(self.get_parameter("enable_button_index").value)

        pygame.init()
        pygame.joystick.init()

        self._joystick: pygame.joystick.Joystick | None = None
        self._last_missing_log_t = 0.0

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self.create_timer(1.0 / self._publish_rate_hz, self._tick)

        self.get_logger().info(
            "ApexXboxManualTeleopNode started (cmd=%s device~=%s index=%d)"
            % (self._cmd_vel_topic, self._device_name_contains or "<any>", self._joystick_index)
        )

    def _find_joystick(self) -> pygame.joystick.Joystick | None:
        pygame.joystick.quit()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count <= 0:
            return None
        if self._joystick_index >= 0 and self._joystick_index < count:
            joystick = pygame.joystick.Joystick(self._joystick_index)
            joystick.init()
            return joystick
        for index in range(count):
            joystick = pygame.joystick.Joystick(index)
            joystick.init()
            name = joystick.get_name().strip().lower()
            if not self._device_name_contains or self._device_name_contains in name:
                return joystick
            joystick.quit()
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        return joystick

    def _publish_zero(self, *, reason: str) -> None:
        msg = Twist()
        self._cmd_pub.publish(msg)
        status = String()
        status.data = json.dumps(
            {
                "state": reason,
                "device_connected": False,
                "linear_x_mps": 0.0,
                "angular_z_rps": 0.0,
                "steering_deg": 0.0,
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status)

    def _tick(self) -> None:
        pygame.event.pump()
        if self._joystick is None or not self._joystick.get_init():
            self._joystick = self._find_joystick()
            if self._joystick is None:
                now_s = self.get_clock().now().nanoseconds * 1.0e-9
                if (now_s - self._last_missing_log_t) >= 1.5:
                    self._last_missing_log_t = now_s
                    self.get_logger().warning("No joystick detected yet; holding zero cmd_vel.")
                self._publish_zero(reason="waiting_joystick")
                return
            self.get_logger().info(
                "Using joystick %d: %s"
                % (self._joystick.get_instance_id(), self._joystick.get_name())
            )

        if self._linear_axis_index >= self._joystick.get_numaxes():
            self.get_logger().error("linear_axis_index is out of range for this joystick")
            self._publish_zero(reason="axis_error")
            return
        if self._steering_axis_index >= self._joystick.get_numaxes():
            self.get_logger().error("steering_axis_index is out of range for this joystick")
            self._publish_zero(reason="axis_error")
            return

        linear_axis = float(self._joystick.get_axis(self._linear_axis_index))
        steering_axis = float(self._joystick.get_axis(self._steering_axis_index))
        if self._linear_axis_invert:
            linear_axis *= -1.0
        if self._steering_axis_invert:
            steering_axis *= -1.0

        linear_axis = _apply_deadband(linear_axis, self._axis_deadband)
        steering_axis = _apply_deadband(steering_axis, self._axis_deadband)

        enabled = True
        if self._enable_button_index >= 0:
            enabled = (
                self._enable_button_index < self._joystick.get_numbuttons()
                and bool(self._joystick.get_button(self._enable_button_index))
            )

        forward_ratio = max(0.0, min(1.0, linear_axis))
        steering_ratio = max(-1.0, min(1.0, steering_axis))
        linear_x_mps = forward_ratio * self._max_linear_speed_mps if enabled else 0.0
        steering_deg = steering_ratio * self._max_steering_deg if enabled else 0.0
        if linear_x_mps <= 1.0e-4:
            angular_z_rps = 0.0
        else:
            angular_z_rps = (
                linear_x_mps * math.tan(math.radians(steering_deg)) / self._wheelbase_m
            )

        msg = Twist()
        msg.linear.x = float(linear_x_mps)
        msg.angular.z = float(angular_z_rps)
        self._cmd_pub.publish(msg)

        status = String()
        status.data = json.dumps(
            {
                "state": "manual_enabled" if enabled else "waiting_enable_button",
                "device_connected": True,
                "device_name": self._joystick.get_name(),
                "linear_axis": linear_axis,
                "steering_axis": steering_axis,
                "linear_x_mps": linear_x_mps,
                "angular_z_rps": angular_z_rps,
                "steering_deg": steering_deg,
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status)

    def destroy_node(self) -> bool:
        try:
            self._publish_zero(reason="shutdown")
        except Exception:
            pass
        try:
            if self._joystick is not None:
                self._joystick.quit()
        except Exception:
            pass
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ApexXboxManualTeleopNode()
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
