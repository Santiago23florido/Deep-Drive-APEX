#!/usr/bin/env python3
"""Read Arduino serial telemetry and publish ROS topics."""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

try:
    import serial
except Exception:
    serial = None


class SerialStateNode(Node):
    def __init__(self) -> None:
        super().__init__("serial_state_node")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("ticks_to_meter", 213.0)
        self.declare_parameter("connect_retry_s", 1.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("line_separator", "/")

        self.declare_parameter("speed_ticks_topic", "/vehicle/speed_ticks")
        self.declare_parameter("speed_mps_topic", "/vehicle/speed_mps")
        self.declare_parameter("measured_wheelspeed_topic", "/measured_wheelspeed")
        self.declare_parameter("ultrasonic_topic", "/vehicle/ultrasonic_cm")
        self.declare_parameter("battery_topic", "/vehicle/battery_v")

        self._serial_port = str(self.get_parameter("serial_port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._ticks_to_meter = max(1e-6, float(self.get_parameter("ticks_to_meter").value))
        self._connect_retry_s = max(0.1, float(self.get_parameter("connect_retry_s").value))
        self._line_separator = str(self.get_parameter("line_separator").value)

        self._pub_speed_ticks = self.create_publisher(Float64, str(self.get_parameter("speed_ticks_topic").value), 10)
        self._pub_speed_mps = self.create_publisher(Float64, str(self.get_parameter("speed_mps_topic").value), 10)
        self._pub_measured = self.create_publisher(Float64, str(self.get_parameter("measured_wheelspeed_topic").value), 10)
        self._pub_ultra = self.create_publisher(Float64, str(self.get_parameter("ultrasonic_topic").value), 10)
        self._pub_batt = self.create_publisher(Float64, str(self.get_parameter("battery_topic").value), 10)

        self._lock = threading.Lock()
        self._speed_ticks = 0.0
        self._ultrasonic_cm = 0.0
        self._battery_v = 0.0
        self._last_update = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._thread.start()

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_state)

    def _serial_worker(self) -> None:
        if serial is None:
            self.get_logger().error("pyserial not installed; serial state node cannot run.")
            return

        while not self._stop_event.is_set():
            ser: Optional[serial.Serial] = None
            try:
                ser = serial.Serial(self._serial_port, self._baudrate, timeout=1.0)
                self.get_logger().info(f"Connected to serial {self._serial_port} @ {self._baudrate}")

                while not self._stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    parsed = self._parse_line(line)
                    if parsed is None:
                        continue
                    speed_ticks, ultrasonic_cm, battery_v = parsed
                    with self._lock:
                        self._speed_ticks = speed_ticks
                        self._ultrasonic_cm = ultrasonic_cm
                        self._battery_v = battery_v
                        self._last_update = time.time()
            except Exception as exc:
                self.get_logger().warning(f"Serial read error ({self._serial_port}): {exc}")
                time.sleep(self._connect_retry_s)
            finally:
                try:
                    if ser is not None and ser.is_open:
                        ser.close()
                except Exception:
                    pass

    def _parse_line(self, line: str) -> Optional[tuple[float, float, float]]:
        parts = line.split(self._line_separator)
        if len(parts) != 3:
            return None
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return None

    def _publish_state(self) -> None:
        with self._lock:
            speed_ticks = float(self._speed_ticks)
            ultrasonic_cm = float(self._ultrasonic_cm)
            battery_v = float(self._battery_v)

        speed_mps = speed_ticks / self._ticks_to_meter

        msg_ticks = Float64()
        msg_ticks.data = speed_ticks
        self._pub_speed_ticks.publish(msg_ticks)

        msg_speed = Float64()
        msg_speed.data = speed_mps
        self._pub_speed_mps.publish(msg_speed)
        self._pub_measured.publish(msg_speed)

        msg_ultra = Float64()
        msg_ultra.data = ultrasonic_cm
        self._pub_ultra.publish(msg_ultra)

        msg_batt = Float64()
        msg_batt.data = battery_v
        self._pub_batt.publish(msg_batt)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SerialStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

