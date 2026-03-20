#!/usr/bin/env python3
"""Read Arduino Nano IMU stream from serial and publish ROS2 messages."""

from __future__ import annotations

import re
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Imu

try:
    import serial
except Exception:  # pragma: no cover - runtime dependency check
    serial = None


class NanoAccelSerialNode(Node):
    """Bridge raw serial IMU stream from Arduino Nano into ROS2."""

    def __init__(self) -> None:
        super().__init__("nano_accel_serial_node")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout_s", 0.5)
        self.declare_parameter("reconnect_delay_s", 1.0)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("topic", "/apex/imu/acceleration/raw")
        self.declare_parameter("gyro_topic", "/apex/imu/angular_velocity/raw")
        self.declare_parameter("imu_topic", "/apex/imu/data_raw")
        self.declare_parameter("publish_imu_topic", True)
        self.declare_parameter("log_every_n_invalid", 25)
        self.declare_parameter("log_every_n_no_gyro", 100)

        self._serial_port = str(self.get_parameter("serial_port").value)
        self._baudrate = int(self.get_parameter("baudrate").value)
        self._serial_timeout_s = max(0.05, float(self.get_parameter("serial_timeout_s").value))
        self._reconnect_delay_s = max(0.1, float(self.get_parameter("reconnect_delay_s").value))
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._log_every_n_invalid = max(1, int(self.get_parameter("log_every_n_invalid").value))
        self._log_every_n_no_gyro = max(1, int(self.get_parameter("log_every_n_no_gyro").value))
        self._publish_imu_topic = bool(self.get_parameter("publish_imu_topic").value)

        self._accel_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("topic").value),
            20,
        )
        self._gyro_pub = self.create_publisher(
            Vector3Stamped,
            str(self.get_parameter("gyro_topic").value),
            20,
        )
        self._imu_pub = (
            self.create_publisher(
                Imu,
                str(self.get_parameter("imu_topic").value),
                20,
            )
            if self._publish_imu_topic
            else None
        )

        self._invalid_counter = 0
        self._no_gyro_counter = 0
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(
            "NanoAccelSerialNode started (port=%s, baudrate=%d, accel=%s, gyro=%s, imu=%s)"
            % (
                self._serial_port,
                self._baudrate,
                self._accel_pub.topic_name,
                self._gyro_pub.topic_name,
                self._imu_pub.topic_name if self._imu_pub is not None else "(disabled)",
            )
        )

    def _parse_line(
        self, line: str
    ) -> Optional[Tuple[float, float, float, float, float, float, bool]]:
        """Parse one serial line into accel+gyro values (SI units)."""
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
            if len(parts) >= 6:
                gx = float(parts[3])
                gy = float(parts[4])
                gz = float(parts[5])
                return (ax, ay, az, gx, gy, gz, True)
            return (ax, ay, az, 0.0, 0.0, 0.0, False)
        except ValueError:
            return None

    def _publish_sample(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        has_gyro: bool,
    ) -> None:
        stamp = self.get_clock().now().to_msg()

        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = stamp
        accel_msg.header.frame_id = self._frame_id
        accel_msg.vector.x = ax
        accel_msg.vector.y = ay
        accel_msg.vector.z = az
        self._accel_pub.publish(accel_msg)

        if has_gyro:
            gyro_msg = Vector3Stamped()
            gyro_msg.header.stamp = stamp
            gyro_msg.header.frame_id = self._frame_id
            gyro_msg.vector.x = gx
            gyro_msg.vector.y = gy
            gyro_msg.vector.z = gz
            self._gyro_pub.publish(gyro_msg)
        else:
            self._no_gyro_counter += 1
            if self._no_gyro_counter % self._log_every_n_no_gyro == 0:
                self.get_logger().warning(
                    "Received legacy 3-axis line without gyro values; yaw rate unavailable."
                )

        if self._imu_pub is not None:
            imu_msg = Imu()
            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = self._frame_id
            imu_msg.orientation_covariance[0] = -1.0
            imu_msg.angular_velocity.x = gx if has_gyro else 0.0
            imu_msg.angular_velocity.y = gy if has_gyro else 0.0
            imu_msg.angular_velocity.z = gz if has_gyro else 0.0
            imu_msg.linear_acceleration.x = ax
            imu_msg.linear_acceleration.y = ay
            imu_msg.linear_acceleration.z = az
            self._imu_pub.publish(imu_msg)

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

                    ax, ay, az, gx, gy, gz, has_gyro = parsed
                    self._publish_sample(ax, ay, az, gx, gy, gz, has_gyro)
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
