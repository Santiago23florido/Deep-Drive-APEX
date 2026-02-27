#!/usr/bin/env python3
"""Read Arduino Nano acceleration from serial and publish ROS2 messages."""

from __future__ import annotations

import re
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node

try:
    import serial
except Exception:  # pragma: no cover - runtime dependency check
    serial = None


class NanoAccelSerialNode(Node):
    """Bridge raw serial acceleration from Arduino Nano into ROS2."""

    def __init__(self) -> None:
        super().__init__("nano_accel_serial_node")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout_s", 0.5)
        self.declare_parameter("reconnect_delay_s", 1.0)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("log_every_n_invalid", 25)

        self._serial_port = str(self.get_parameter("serial_port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._serial_timeout_s = max(0.05, float(self.get_parameter("serial_timeout_s").value))
        self._reconnect_delay_s = max(0.1, float(self.get_parameter("reconnect_delay_s").value))
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._log_every_n_invalid = max(1, int(self.get_parameter("log_every_n_invalid").value))

        self._publisher = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("topic").value),
            20,
        )

        self._invalid_counter = 0
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            "NanoAccelSerialNode started (port=%s, baudrate=%d, topic=%s)"
            % (self._serial_port, self._baudrate, self._publisher.topic_name)
        )

    def _parse_line(self, line: str) -> Optional[Tuple[float, float, float]]:
        """Parse one serial line into acceleration xyz floats (m/s^2)."""
        text = line.strip()
        if not text:
            return None

        # Accept optional prefixes such as "ACC:" or "INFO:".
        if ":" in text:
            text = text.split(":", 1)[1].strip()

        parts = [p for p in re.split(r"[,\s;]+", text) if p]
        if len(parts) < 3:
            return None

        try:
            ax = float(parts[0])
            ay = float(parts[1])
            az = float(parts[2])
            return (ax, ay, az)
        except ValueError:
            return None

    def _publish_accel(self, ax: float, ay: float, az: float) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.vector.x = ax
        msg.vector.y = ay
        msg.vector.z = az
        self._publisher.publish(msg)

    def _serial_worker(self) -> None:
        if serial is None:
            self.get_logger().error("pyserial is not available. Install python3-serial.")
            return

        while not self._stop_event.is_set():
            ser = None
            try:
                ser = serial.Serial(
                    self._serial_port,
                    self._baudrate,
                    timeout=self._serial_timeout_s,
                )
                self.get_logger().info(
                    "Connected to Arduino serial %s @ %d"
                    % (self._serial_port, self._baudrate)
                )

                while not self._stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode("utf-8", errors="replace").strip()
                    parsed = self._parse_line(line)
                    if parsed is None:
                        self._invalid_counter += 1
                        if self._invalid_counter % self._log_every_n_invalid == 0:
                            self.get_logger().warning(
                                "Skipping invalid serial line: '%s'" % line
                            )
                        continue

                    ax, ay, az = parsed
                    self._publish_accel(ax, ay, az)
            except Exception as exc:
                self.get_logger().warning(
                    "Serial connection error on %s: %s"
                    % (self._serial_port, str(exc))
                )
                time.sleep(self._reconnect_delay_s)
            finally:
                if ser is not None:
                    try:
                        if ser.is_open:
                            ser.close()
                    except Exception:
                        pass

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = NanoAccelSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
