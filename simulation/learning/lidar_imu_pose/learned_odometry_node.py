#!/usr/bin/env python3
"""ROS 2 node of the learned LiDAR-inertial odometry (``live_odometry``).

Runs with the Python that has torch (``simulation/learning/.venv``, which
also sees ROS Jazzy's rclpy once ``setup.bash`` is sourced).

Inputs (the interfaces of the real car):
  ``/apex/imu/data_raw``   sensor_msgs/Imu, raw specific force + rate, stamped at sampling
  ``/lidar/scan_raw``      sensor_msgs/LaserScan, one revolution, stamped at its first sample
The LiDAR calibration (rotation direction, sync angle, mount, datasheet noise)
and the nominal IMU rate come from a sensor profile of
``tools/pose_dataset/config/sensors.yaml`` (the car's calibration file).

Outputs:
  TF ``odom_learned -> base_link``           at every scan stamp (for the SLAM)
  ``/apex/odometry/learned``                 nav_msgs/Odometry at the scan stamps, covariance from the network sigma
  ``/apex/odometry/learned_predicted``       nav_msgs/Odometry at the IMU rate, propagated to the latest IMU sample (control)
  ``/apex/learned_odometry/scan_deskewed``   LaserScan de-skewed to its stamp (SLAM input)
  ``/apex/learned_odometry/status``          JSON: readiness, latency, compute time, biases, input health
and a CSV log of every estimate (evaluation).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))

import rclpy  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402
import torch  # noqa: E402

from live_odometry import Estimate, LidarCalibration, LiveOdometry  # noqa: E402
from pose_dataset.sensors import load_sensor_profiles  # noqa: E402
from sqlite_windows import imu_group_delay_s  # noqa: E402

DEFAULT_CKPT = HERE.parent / "outputs" / "real2sim_v2_fast" / "hybrid_submap_v3" / "best_model.pt"  # latest trained (fast-motion correction)


def _stamp(t_ns: int) -> TimeMsg:
    return TimeMsg(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))


class LearnedOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_learned_odometry")
        dp = self.declare_parameter
        dp("checkpoint", str(DEFAULT_CKPT))
        dp("sensor_profile", "APEX_real")
        dp("sensors_yaml", str(HERE.parents[1] / "tools" / "pose_dataset" / "config" / "sensors.yaml"))
        # One car = one scan interval at a time: the CPU beats the GPU (kernel
        # launches, 3x3 solves): ~20 ms per scan with 4 threads vs ~110 ms.
        dp("device", "cpu")
        dp("num_threads", 4)
        dp("imu_wait_s", 0.03)
        dp("imu_topic", "/apex/imu/data_raw")
        dp("scan_topic", "/lidar/scan_raw")
        dp("odom_topic", "/apex/odometry/learned")
        dp("predicted_topic", "/apex/odometry/learned_predicted")
        dp("deskewed_topic", "/apex/learned_odometry/scan_deskewed")
        dp("status_topic", "/apex/learned_odometry/status")
        dp("odom_frame", "odom_learned")
        dp("base_frame", "base_link")
        dp("laser_frame", "laser")
        dp("publish_tf", True)
        dp("ready_scans", 10)  # estimates before the pose is declared usable (IMU bias settling at rest)
        dp("log_csv", "")
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        prof = load_sensor_profiles(Path(str(gp("sensors_yaml"))))[str(gp("sensor_profile"))]
        self.cal = LidarCalibration.from_profile(prof["lidar"])
        self.imu_rate = float(prof["imu"]["rate_hz"])
        torch.set_num_threads(int(gp("num_threads")))
        torch.backends.cuda.matmul.allow_tf32 = True  # as in training / evaluation (the ICP forces FP32 itself)
        torch.backends.cudnn.allow_tf32 = True
        ckpt = str(gp("checkpoint"))
        self.odo = LiveOdometry(ckpt, self.cal, device=str(gp("device")), imu_wait_s=float(gp("imu_wait_s")), imu_rate_hz=self.imu_rate,
                                imu_group_delay_s=imu_group_delay_s(prof["imu"]))
        self._warmup(ckpt)
        self.odom_frame, self.base_frame, self.laser_frame = str(gp("odom_frame")), str(gp("base_frame")), str(gp("laser_frame"))
        self.publish_tf = bool(gp("publish_tf"))
        self.ready_scans = int(gp("ready_scans"))
        reliable = QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE)
        self.tf = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, str(gp("odom_topic")), 50)
        self.pred_pub = self.create_publisher(Odometry, str(gp("predicted_topic")), 50)
        self.scan_pub = self.create_publisher(LaserScan, str(gp("deskewed_topic")), 20)
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), 10)
        self.create_subscription(Imu, str(gp("imu_topic")), self._on_imu, reliable)
        self.create_subscription(LaserScan, str(gp("scan_topic")), self._on_scan, QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
        self.create_timer(0.5, self._status)
        # Scans are processed in their own thread: the IMU propagation (the
        # pose used for control) keeps running while a scan is computed.
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._scan_worker, name="scan_worker", daemon=True)
        self._worker.start()
        self.cov = np.zeros(3)
        self.latency_ms: list[float] = []
        self.compute_ms: list[float] = []
        self.log = None
        if str(gp("log_csv")):
            Path(str(gp("log_csv"))).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(str(gp("log_csv")), "w", newline="", encoding="utf-8")
            self.log = csv.writer(self._log_file)
            self.log.writerow(["stamp_ns", "x", "y", "yaw", "dx", "dy", "dyaw", "sx", "sy", "syaw", "vx", "vy", "bias_gz", "bias_ax", "bias_ay",
                               "dt", "imu_samples", "icp_standstill", "icp_map_used", "compute_ms", "latency_ms"])
        self.get_logger().info(
            f"learned odometry: {Path(ckpt).parent.name}/{Path(ckpt).name} ({'hybrid ' + str(self.odo.icp_cfg.submap_keyframes) + '-keyframe submap ICP' if self.odo.hybrid else 'pure network'}) "
            f"on {self.odo.device}; LiDAR {self.cal.rate_hz} Hz {self.cal.scan_direction} from {self.cal.scan_start_angle_deg} deg, IMU {self.imu_rate} Hz")

    def _warmup(self, ckpt: str) -> None:
        """One synthetic interval on a throw-away instance (CUDA kernels, cuDNN plans)."""
        t0 = time.perf_counter()
        tmp = LiveOdometry(ckpt, self.cal, device=str(self.odo.device), imu_wait_s=0.0)
        ranges = np.full(self.cal.beams, 3.0, dtype=np.float32)
        for k in range(3):
            base = k * 80_000_000
            for i in range(10):
                tmp.add_imu(base + i * 8_000_000 + 1, np.zeros(3), np.array([0.0, 0.0, 9.80665]))
            tmp.add_scan(base, ranges, 0.08, arrival_ns=base)
            tmp.process(now_ns=base + 10**9)
        self.get_logger().info(f"warm-up done in {1e3 * (time.perf_counter() - t0):.0f} ms")

    # ------------------------------------------------------------- callbacks
    def _on_imu(self, m: Imu) -> None:
        t = Time.from_msg(m.header.stamp).nanoseconds
        self.odo.add_imu(t, np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]),
                         np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]))
        self._wake.set()  # a pending scan may now be covered by the IMU
        pred = self.odo.predict()
        if pred is not None and self.odo.scans > 0:
            ts, pose, v, rate = pred
            self._odom(self.pred_pub, ts, pose, v, None, rate)

    def _on_scan(self, m: LaserScan) -> None:
        t = Time.from_msg(m.header.stamp).nanoseconds
        self.odo.add_scan(t, np.asarray(m.ranges, dtype=np.float32), float(m.scan_time), arrival_ns=self.get_clock().now().nanoseconds)
        self._wake.set()

    def _scan_worker(self) -> None:
        while rclpy.ok():
            self._wake.wait(0.005)  # also polls the IMU-wait timeout of pending scans
            self._wake.clear()
            try:
                self._handle(self.odo.process(self.get_clock().now().nanoseconds))
            except Exception as exc:  # the context is shutting down
                if rclpy.ok():
                    self.get_logger().error(f"scan processing failed: {exc}")
                return

    # --------------------------------------------------------------- outputs
    def _handle(self, estimates: list[Estimate]) -> None:
        now = self.get_clock().now().nanoseconds
        for e in estimates:
            # Covariance of the pose: increment variances rotated into the odom frame and summed.
            c, s = math.cos(e.pose[2] - e.delta[2]), math.sin(e.pose[2] - e.delta[2])
            var = e.sigma.astype(float) ** 2
            self.cov += np.array([c * c * var[0] + s * s * var[1], s * s * var[0] + c * c * var[1], var[2]])
            if self.publish_tf:
                tf = TransformStamped()
                tf.header.stamp = _stamp(e.stamp_ns)
                tf.header.frame_id = self.odom_frame
                tf.child_frame_id = self.base_frame
                tf.transform.translation.x, tf.transform.translation.y = float(e.pose[0]), float(e.pose[1])
                tf.transform.rotation.z, tf.transform.rotation.w = math.sin(0.5 * e.pose[2]), math.cos(0.5 * e.pose[2])
                self.tf.sendTransform(tf)
            self._odom(self.odom_pub, e.stamp_ns, e.pose, e.velocity, e.sigma)
            scan = LaserScan()
            scan.header.stamp = _stamp(e.stamp_ns)
            scan.header.frame_id = self.laser_frame
            scan.angle_min = float(self.cal.angle_min)
            scan.angle_increment = float(self.cal.angle_increment)
            scan.angle_max = float(self.cal.angle_min + self.cal.angle_increment * (self.cal.beams - 1))
            scan.time_increment = 0.0  # de-skewed: every beam at the stamp
            scan.scan_time = float(1.0 / self.cal.rate_hz)
            scan.range_min, scan.range_max = float(self.cal.range_min), float(self.cal.range_max)
            scan.ranges = e.deskewed_ranges.astype(float).tolist()
            self.scan_pub.publish(scan)
            lat = (now - e.stamp_ns) * 1e-6
            self.latency_ms.append(lat)
            self.compute_ms.append(e.compute_ms)
            if self.log is not None:
                self.log.writerow([e.stamp_ns, *[f"{v:.6f}" for v in e.pose], *[f"{v:.6f}" for v in e.delta], *[f"{v:.6g}" for v in e.sigma],
                                   *[f"{v:.4f}" for v in e.velocity], *[f"{v:.6f}" for v in e.bias], f"{e.dt:.5f}", e.imu_samples,
                                   e.icp.get("standstill", ""), e.icp.get("map_used", ""), f"{e.compute_ms:.2f}", f"{lat:.2f}"])

    def _odom(self, pub, t_ns: int, pose: np.ndarray, v: np.ndarray, sigma, yaw_rate: float = 0.0) -> None:  # noqa: ANN001
        o = Odometry()
        o.header.stamp = _stamp(t_ns)
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y = float(pose[0]), float(pose[1])
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(0.5 * pose[2]), math.cos(0.5 * pose[2])
        o.pose.covariance[0], o.pose.covariance[7], o.pose.covariance[35] = float(self.cov[0]), float(self.cov[1]), float(self.cov[2])
        o.twist.twist.linear.x, o.twist.twist.linear.y = float(v[0]), float(v[1])
        o.twist.twist.angular.z = float(yaw_rate)
        if sigma is not None:
            o.twist.covariance[0], o.twist.covariance[7], o.twist.covariance[35] = (float(x) for x in np.asarray(sigma) ** 2)
        pub.publish(o)

    def _status(self) -> None:
        health = self.odo.health.report()
        lat = np.array(self.latency_ms[-200:]) if self.latency_ms else np.zeros(1)
        comp = np.array(self.compute_ms[-200:]) if self.compute_ms else np.zeros(1)
        payload = {"ready": self.odo.scans >= self.ready_scans, "scans": self.odo.scans, "pending": len(self.odo._pending),
                   "latency_ms_p50": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
                   "compute_ms_p50": float(np.median(comp)), "compute_ms_p95": float(np.percentile(comp, 95)),
                   "bias": self.odo.bias.tolist(), "velocity": self.odo.velocity.tolist(), "health": health}
        self.status_pub.publish(String(data=json.dumps(payload)))
        for w in health.get("warnings", []):
            self.get_logger().warn(w, throttle_duration_sec=10.0)

    def destroy_node(self) -> None:
        if self.log is not None:
            self._log_file.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LearnedOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:  # publishing while the context shuts down (SIGINT of the launch)
        if rclpy.ok():
            raise
        node.get_logger().debug(f"stopped during shutdown: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
