#!/usr/bin/env python3
"""Record the car's IMU and LiDAR as the learned odometry receives them.

Writes the simulation's formats (``simulation/.../real_sensor_node.py``), so
the same tools read a car recording and a simulated run:

* ``imu_raw.csv``: stamp_sec, stamp_nanosec, ax, ay, az (m/s^2), gx, gy, gz (rad/s)
  from ``/apex/imu/data_raw``;
* ``lidar_scans.csv``: stamp_ns, scan_time_s, r0..r359 from ``/lidar/scan_raw``
  (one row per revolution, REP-117 inf values);
* ``capture_meta.json``: topics, duration, message counts.

Audit the timing afterwards:

    python3 simulation/tools/analysis/sensor_timing_audit.py <out_dir> --profile APEX_real

Usage (on the car, with the pipeline running):

    python3 record_real2sim_sensors.py --out /work/repo/real_vehicle/data/real2sim_capture/<name> --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan


class Real2SimSensorRecorder(Node):
    def __init__(self, out_dir: str, imu_topic: str, scan_topic: str, duration_s: float, best_effort: bool = False) -> None:
        super().__init__("real2sim_sensor_recorder")
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir
        self._duration_s = float(duration_s)
        self._t0 = time.monotonic()
        self._counts = {"imu": 0, "scans": 0}
        self._meta = {"imu_topic": imu_topic, "scan_topic": scan_topic, "started_unix_s": time.time()}
        self._meta.update(started_monotonic_ns=time.monotonic_ns(), best_effort=best_effort)
        self._timing_file = open(os.path.join(out_dir, "sensor_receipt_timing.csv"), "w", newline="", encoding="utf-8")
        self._timing = csv.writer(self._timing_file)
        self._timing.writerow(["sensor", "stamp_ns", "receipt_ros_ns", "receipt_monotonic_ns"])

        self._imu_file = open(os.path.join(out_dir, "imu_raw.csv"), "w", newline="", encoding="utf-8")
        self._imu = csv.writer(self._imu_file)
        self._imu.writerow(["stamp_sec", "stamp_nanosec", "ax_mps2", "ay_mps2", "az_mps2", "gx_rps", "gy_rps", "gz_rps"])
        self._scan_file = open(os.path.join(out_dir, "lidar_scans.csv"), "w", newline="", encoding="utf-8")
        self._scan = csv.writer(self._scan_file)
        self._scan_header_written = False

        reliability = ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE
        reliable = QoSProfile(depth=200, reliability=reliability)
        self.create_subscription(Imu, imu_topic, self._on_imu, reliable)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, QoSProfile(depth=20, reliability=reliability))
        self.create_timer(0.5, self._tick)
        self.done = False
        self.get_logger().info(f"Recording {imu_topic} and {scan_topic} for {duration_s:.0f} s into {out_dir}")

    def _on_imu(self, m: Imu) -> None:
        self._timing.writerow(["imu", Time.from_msg(m.header.stamp).nanoseconds,
                               self.get_clock().now().nanoseconds, time.monotonic_ns()])
        a, g = m.linear_acceleration, m.angular_velocity
        self._imu.writerow(
            [m.header.stamp.sec, m.header.stamp.nanosec, *(f"{v:.6f}" for v in (a.x, a.y, a.z, g.x, g.y, g.z))]
        )
        self._counts["imu"] += 1

    def _on_scan(self, m: LaserScan) -> None:
        self._timing.writerow(["lidar", Time.from_msg(m.header.stamp).nanoseconds,
                               self.get_clock().now().nanoseconds, time.monotonic_ns()])
        if not self._scan_header_written:
            self._scan.writerow(["stamp_ns", "scan_time_s"] + [f"r{i}" for i in range(len(m.ranges))])
            self._scan_header_written = True
            self._meta.update(angle_min=m.angle_min, angle_max=m.angle_max, angle_increment=m.angle_increment,
                              time_increment=m.time_increment, range_min=m.range_min, range_max=m.range_max,
                              beams=len(m.ranges), frame_id=m.header.frame_id)
        stamp_ns = Time.from_msg(m.header.stamp).nanoseconds
        self._scan.writerow([stamp_ns, f"{m.scan_time:.6f}", *(f"{r:.4f}" if math.isfinite(r) else ("inf" if r > 0 else "-inf") for r in m.ranges)])
        self._counts["scans"] += 1

    def _tick(self) -> None:
        if time.monotonic() - self._t0 >= self._duration_s:
            self.done = True

    def close(self) -> None:
        self._timing_file.close()
        self._imu_file.close()
        self._scan_file.close()
        self._meta.update(duration_s=time.monotonic() - self._t0, counts=self._counts)
        with open(os.path.join(self._out_dir, "capture_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, indent=2)
        self.get_logger().info(f"Recorded {self._counts['imu']} IMU samples and {self._counts['scans']} scans")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--imu-topic", default="/apex/imu/data_raw")
    parser.add_argument("--scan-topic", default="/lidar/scan_raw")
    parser.add_argument("--best-effort", action="store_true", help="Passive capture compatible with sensor-data publishers")
    args = parser.parse_args()
    rclpy.init()
    node = Real2SimSensorRecorder(args.out, args.imu_topic, args.scan_topic, args.duration, args.best_effort)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
