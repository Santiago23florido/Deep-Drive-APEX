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
from std_msgs.msg import String


class ApexWindowsGamepadBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_windows_gamepad_bridge_node")

        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("status_topic", "/apex/sim/manual_control/status")
        self.declare_parameter("listen_host", "0.0.0.0")
        self.declare_parameter("listen_port", 8765)
        self.declare_parameter("publish_rate_hz", 25.0)
        self.declare_parameter("command_timeout_s", 0.50)

        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._listen_host = str(self.get_parameter("listen_host").value).strip() or "0.0.0.0"
        self._listen_port = int(self.get_parameter("listen_port").value)
        self._publish_rate_hz = max(2.0, float(self.get_parameter("publish_rate_hz").value))
        self._command_timeout_s = max(
            0.05, float(self.get_parameter("command_timeout_s").value)
        )

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._server_socket: socket.socket | None = None
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)

        self._latest_command: dict[str, Any] = self._zero_status("waiting_windows_bridge")
        self._last_command_monotonic = 0.0
        self._client_label = ""
        self._client_connected = False

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)
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
            self.get_logger().info(f"Windows bridge connected: {client_label}")

            try:
                self._handle_client(conn)
            finally:
                with self._lock:
                    self._client_connected = False
                    self._client_label = ""
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

        status = {
            "state": "manual_enabled" if bool(payload.get("enabled", True)) else "waiting_enable",
            "bridge_connected": True,
            "controller_connected": bool(payload.get("controller_connected", True)),
            "enabled": bool(payload.get("enabled", True)),
            "linear_x_mps": float(payload.get("linear_x_mps", 0.0)),
            "angular_z_rps": float(payload.get("angular_z_rps", 0.0)),
            "steering_deg": float(payload.get("steering_deg", 0.0)),
            "device_name": str(payload.get("device_name", "")),
            "raw_linear_axis": float(payload.get("raw_linear_axis", 0.0)),
            "raw_steering_axis": float(payload.get("raw_steering_axis", 0.0)),
            "current_linear_cap_mps": float(payload.get("current_linear_cap_mps", 0.0)),
            "windows_bridge_stamp_ms": int(payload.get("stamp_ms", 0)),
        }
        if not status["controller_connected"]:
            status["state"] = "waiting_controller"
        if not status["enabled"]:
            status["linear_x_mps"] = 0.0
            status["angular_z_rps"] = 0.0
            status["steering_deg"] = 0.0

        with self._lock:
            self._latest_command = status
            self._last_command_monotonic = time.monotonic()

    def _publish_latest(self) -> None:
        now = time.monotonic()
        with self._lock:
            status = dict(self._latest_command)
            last_msg_age = now - self._last_command_monotonic if self._last_command_monotonic else None
            client_label = self._client_label
            bridge_connected = self._client_connected

        if last_msg_age is None or last_msg_age > self._command_timeout_s:
            status = self._zero_status(
                "waiting_windows_bridge" if not bridge_connected else "waiting_command"
            )
        status["bridge_connected"] = bridge_connected
        status["bridge_client"] = client_label
        status["command_age_s"] = round(last_msg_age, 3) if last_msg_age is not None else None

        msg = Twist()
        if (
            status.get("bridge_connected")
            and status.get("controller_connected")
            and status.get("enabled")
        ):
            msg.linear.x = float(status.get("linear_x_mps", 0.0))
            msg.angular.z = float(status.get("angular_z_rps", 0.0))
        self._cmd_pub.publish(msg)

        status_msg = String()
        status_msg.data = json.dumps(status, separators=(",", ":"))
        self._status_pub.publish(status_msg)

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
