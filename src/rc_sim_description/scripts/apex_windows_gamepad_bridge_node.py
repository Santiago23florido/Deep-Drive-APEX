#!/usr/bin/env python3
"""Receive manual driving commands from a Windows gamepad bridge over TCP."""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String


class ApexWindowsGamepadBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_windows_gamepad_bridge_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("status_topic", "/apex/sim/manual_control/status")
        self.declare_parameter("manual_status_topic", "/apex/manual_control/status")
        self.declare_parameter("session_toggle_topic", "/apex/manual_control/session_toggle")
        self.declare_parameter("listen_host", "0.0.0.0")
        self.declare_parameter("listen_port", 8765)
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("command_timeout_s", 0.50)
        self.declare_parameter("start_button_field", "start_pressed")
        self.declare_parameter("enable_session_toggle_input", True)
        self.declare_parameter("require_session_toggle_to_drive", True)
        self.declare_parameter("linear_gain", 3.0)
        self.declare_parameter("angular_gain", 3.0)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._manual_status_topic = str(self.get_parameter("manual_status_topic").value)
        self._session_toggle_topic = str(self.get_parameter("session_toggle_topic").value)
        self._listen_host = str(self.get_parameter("listen_host").value).strip() or "0.0.0.0"
        self._listen_port = int(self.get_parameter("listen_port").value)
        self._publish_rate_hz = max(2.0, float(self.get_parameter("publish_rate_hz").value))
        self._command_timeout_s = max(
            0.05, float(self.get_parameter("command_timeout_s").value)
        )
        self._start_button_field = (
            str(self.get_parameter("start_button_field").value).strip() or "start_pressed"
        )
        self._enable_session_toggle_input = bool(
            self.get_parameter("enable_session_toggle_input").value
        )
        self._require_session_toggle_to_drive = bool(
            self.get_parameter("require_session_toggle_to_drive").value
        )
        self._linear_gain = float(self.get_parameter("linear_gain").value)
        self._angular_gain = float(self.get_parameter("angular_gain").value)

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._server_socket: socket.socket | None = None
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)

        self._session_toggle_count = 0
        self._run_active = not self._require_session_toggle_to_drive
        self._latest_command: dict[str, Any] = self._zero_status("waiting_windows_bridge")
        self._last_command_monotonic = 0.0
        self._client_label = ""
        self._client_connected = False
        self._last_start_pressed = False

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._manual_status_pub = self.create_publisher(String, self._manual_status_topic, 10)
        self._toggle_pub = self.create_publisher(Bool, self._session_toggle_topic, 10)
        self.create_timer(1.0 / self._publish_rate_hz, self._publish_latest)

        self._server_thread.start()
        self.get_logger().info(
            "Listening for Windows gamepad bridge on %s:%d"
            % (self._listen_host, self._listen_port)
        )

    def _zero_status(self, state: str) -> dict[str, Any]:
        return {
            "state": state,
            "bridge_connected": False,
            "controller_connected": False,
            "enabled": False,
            "linear_x_mps": 0.0,
            "angular_z_rps": 0.0,
            "steering_deg": 0.0,
            "device_name": "",
            "start_pressed": False,
            "session_toggle_count": self._session_toggle_count,
            "run_active": self._run_active,
        }

    def _server_loop(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._listen_host, self._listen_port))
        server.listen(1)
        server.settimeout(0.5)
        self._server_socket = server

        while not self._shutdown.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            client_label = f"{addr[0]}:{addr[1]}"
            with self._lock:
                self._client_label = client_label
                self._client_connected = True
                self._last_start_pressed = False
            self.get_logger().info(f"Windows bridge connected: {client_label}")

            try:
                self._handle_client(conn)
            finally:
                with self._lock:
                    self._client_connected = False
                    self._client_label = ""
                    self._last_start_pressed = False
                self.get_logger().info(f"Windows bridge disconnected: {client_label}")

        try:
            server.close()
        except Exception:
            pass

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        buffer = ""
        with conn:
            while not self._shutdown.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    self._consume_line(line.strip())

    def _consume_line(self, line: str) -> None:
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring invalid JSON from Windows bridge.")
            return

        start_pressed = False
        if self._enable_session_toggle_input:
            start_pressed = bool(
                payload.get(
                    self._start_button_field,
                    payload.get("start_pressed", payload.get("start", False)),
                )
            )
        raw_enabled = bool(payload.get("enabled", True))
        status = {
            "state": "manual_enabled" if raw_enabled else "waiting_enable",
            "bridge_connected": True,
            "controller_connected": bool(payload.get("controller_connected", True)),
            "enabled": raw_enabled,
            "input_enabled": raw_enabled,
            "linear_x_mps": float(payload.get("linear_x_mps", 0.0)),
            "angular_z_rps": float(payload.get("angular_z_rps", 0.0)),
            "steering_deg": float(payload.get("steering_deg", 0.0)),
            "device_name": str(payload.get("device_name", "")),
            "raw_linear_axis": float(payload.get("raw_linear_axis", 0.0)),
            "raw_steering_axis": float(payload.get("raw_steering_axis", 0.0)),
            "current_linear_cap_mps": float(payload.get("current_linear_cap_mps", 0.0)),
            "windows_bridge_stamp_ms": int(payload.get("stamp_ms", 0)),
            "start_pressed": start_pressed,
            "session_toggle_count": self._session_toggle_count,
            "run_active": self._run_active,
        }
        if not status["controller_connected"]:
            status["state"] = "waiting_controller"
        if not status["enabled"]:
            status["linear_x_mps"] = 0.0
            status["angular_z_rps"] = 0.0
            status["steering_deg"] = 0.0

        publish_toggle = False
        with self._lock:
            if start_pressed and not self._last_start_pressed:
                self._session_toggle_count += 1
                if self._require_session_toggle_to_drive:
                    self._run_active = not self._run_active
                status["run_active"] = self._run_active
                status["session_toggle_count"] = self._session_toggle_count
                publish_toggle = True
            else:
                status["run_active"] = self._run_active
                status["session_toggle_count"] = self._session_toggle_count
            self._last_start_pressed = start_pressed
            self._latest_command = status
            self._last_command_monotonic = time.monotonic()

        drive_enabled = bool(status["enabled"]) and (
            bool(status["run_active"]) or not self._require_session_toggle_to_drive
        )
        status["enabled"] = drive_enabled
        if self._require_session_toggle_to_drive and not bool(status["run_active"]):
            status["state"] = "waiting_start_button"
        elif not drive_enabled and bool(status["controller_connected"]) and bool(status["input_enabled"]):
            status["state"] = "waiting_start_button"
        if not drive_enabled:
            status["linear_x_mps"] = 0.0
            status["angular_z_rps"] = 0.0
            status["steering_deg"] = 0.0

        if publish_toggle:
            pulse = Bool()
            pulse.data = True
            self._toggle_pub.publish(pulse)

    def _publish_latest(self) -> None:
        now = time.monotonic()
        with self._lock:
            status = dict(self._latest_command)
            last_msg_age = now - self._last_command_monotonic if self._last_command_monotonic else None
            client_label = self._client_label
            bridge_connected = self._client_connected
            toggle_count = self._session_toggle_count
            run_active = self._run_active

        if last_msg_age is None or last_msg_age > self._command_timeout_s:
            status = self._zero_status(
                "waiting_windows_bridge" if not bridge_connected else "waiting_command"
            )
            status["session_toggle_count"] = toggle_count
        status["run_active"] = run_active
        status["bridge_connected"] = bridge_connected
        status["bridge_client"] = client_label
        status["command_age_s"] = round(last_msg_age, 3) if last_msg_age is not None else None
        if self._require_session_toggle_to_drive and not bool(status.get("run_active", False)):
            status["enabled"] = False
            if bridge_connected and bool(status.get("controller_connected", False)):
                status["state"] = "waiting_start_button"

        msg = Twist()
        if (
            status.get("bridge_connected")
            and status.get("controller_connected")
            and status.get("enabled")
        ):
            msg.linear.x = float(status.get("linear_x_mps", 0.0)) * self._linear_gain
            msg.angular.z = float(status.get("angular_z_rps", 0.0)) * self._angular_gain
        self._cmd_pub.publish(msg)

        status_msg = String()
        status_msg.data = json.dumps(status, separators=(",", ":"))
        self._status_pub.publish(status_msg)
        self._manual_status_pub.publish(status_msg)

    def destroy_node(self) -> bool:
        try:
            self._shutdown.set()
            if self._server_socket is not None:
                self._server_socket.close()
        except Exception:
            pass
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ApexWindowsGamepadBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
