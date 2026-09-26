"""Real2sim sensors: Gazebo's native IMU and gpu_lidar -> the real APEX devices.

Gazebo measures, this node makes the measurement look like the real hardware
(the same realism layer as the pose-dataset generator, so the live simulation
and the training data are produced by one chain):

* Gazebo ``imu`` at 1 kHz on the sprung body, with Gazebo's native noise
  (white noise, turn-on and Gauss-Markov bias, configured in the car's SDF
  from the sensor profile) -> :class:`ImuChip` (vibration, chip low-pass,
  exact output data rate with its oscillator error, scale / misalignment /
  quantization, stamps) -> ``sensor_msgs/Imu`` at the chip rate;
* Gazebo ``gpu_lidar`` captures (fast and fine, the geometry source) + the
  1 kHz pose of the sprung body -> :class:`RollingLidar` (revolution timing,
  rotation direction and sync angle, per-sample poses, occluded sectors,
  range noise, driver binning) -> ``sensor_msgs/LaserScan`` per revolution,
  stamped at its first sample.

Every message is published when the real device would deliver it (simulation
clock >= its release time: end of the revolution / sampling instant + the
profile's publication latency). The truth at the scan instants and a 100 Hz
truth odometry are published for evaluation only.
"""

from __future__ import annotations

from collections import deque
import heapq
import itertools
import math
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
import yaml

from ..core.imu_chip import ImuChip, chip_config_from_profile
from ..core.lidar_rolling import RollingLidar, rolling_config_from_profile
from ..core.sim_paths import sim_root  # noqa: F401 (re-exported)
from ._common import json_msg

RELIABLE = QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE)


def load_profile(name: str, sensors_yaml: Path) -> dict:
    tools = str(sim_root() / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from pose_dataset.sensors import load_sensor_profiles  # noqa: PLC0415 (inheritance resolution)

    return load_sensor_profiles(sensors_yaml)[name]


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r @ v


def _rpy_quat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def sensor_mount(urdf: str, joint: str) -> tuple[str, np.ndarray, np.ndarray]:
    """(parent link, xyz, quaternion) of a fixed sensor joint of the URDF."""
    root = ET.fromstring(urdf)
    for j in root.findall("joint"):
        if j.get("name") == joint:
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            return j.find("parent").get("link"), xyz, _rpy_quat(*rpy)
    raise KeyError(f"joint {joint} not in robot_description")


class RealSensorNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_real_sensors")
        root = sim_root()
        dp = self.declare_parameter
        dp("sensor_profile", "APEX_real")
        dp("sensors_yaml", str(root / "tools" / "pose_dataset" / "config" / "sensors.yaml"))
        dp("vehicle_yaml", str(root / "tools" / "pose_dataset" / "config" / "vehicle.yaml"))
        dp("seed", 0)
        dp("robot_description", "")
        dp("model_name", "rc_car")
        dp("gz_imu_topic", "/apex/sim/imu")
        dp("gz_scan_topic", "/apex/sim/scan")
        dp("gz_pose_topic", "/model/rc_car/pose")  # link poses relative to the model (PosePublisher)
        dp("gz_odom_topic", "/model/rc_car/odometry")  # world pose of base_link (OdometryPublisher)
        dp("imu_topic", "/apex/imu/data_raw")
        dp("scan_topic", "/lidar/scan_raw")
        dp("imu_frame", "imu_link")
        dp("laser_frame", "laser")
        dp("truth_odom_topic", "/apex/sim/ground_truth/base_odom")
        dp("truth_scan_pose_topic", "/apex/sim/ground_truth/scan_pose")
        dp("truth_rate_hz", 100.0)
        dp("status_topic", "/apex/sim/real_sensors/status")
        dp("publish_period_s", 0.002)
        dp("truth_csv_dir", "")  # evaluation: truth track (100 Hz), truth at every scan, and the published IMU samples and scans
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.profile = load_profile(str(gp("sensor_profile")), Path(gp("sensors_yaml")))
        vib = yaml.safe_load(Path(gp("vehicle_yaml")).read_text(encoding="utf-8"))["body_dynamics"]["vibration"]
        seed = int(gp("seed"))
        self.chip = ImuChip(chip_config_from_profile(self.profile["imu"], vib, seed * 7919 + 1))
        cfg = rolling_config_from_profile(self.profile["lidar"], seed * 7919 + 2)
        cfg.start_grid_s = 0.001
        self.lidar = RollingLidar(cfg)
        urdf = str(gp("robot_description"))
        if not urdf:
            raise RuntimeError("robot_description (the car URDF) is required to know the LiDAR mount")
        self.lidar_parent, self.lidar_xyz, self.lidar_q = sensor_mount(urdf, "lidar_joint")
        self.model = str(gp("model_name"))
        self.imu_frame, self.laser_frame = str(gp("imu_frame")), str(gp("laser_frame"))

        self.imu_pub = self.create_publisher(Imu, str(gp("imu_topic")), RELIABLE)
        self.scan_pub = self.create_publisher(LaserScan, str(gp("scan_topic")), QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
        self.truth_pub = self.create_publisher(Odometry, str(gp("truth_odom_topic")), 50)
        self.truth_scan_pub = self.create_publisher(PoseStamped, str(gp("truth_scan_pose_topic")), 50)
        self.status_pub = self.create_publisher(String, str(gp("status_topic")), 10)
        self.truth_period_ns = int(1e9 / float(gp("truth_rate_hz")))
        self._last_truth_ns = -(2**62)

        self._lock = threading.Lock()
        self._rel_latest: tuple[np.ndarray, np.ndarray] | None = None  # sprung body pose relative to the model
        self._imu_in: list[tuple[int, list[float]]] = []
        self._pose_in: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []  # t, base p, base q, sprung p, sprung q
        self._scan_in: list[tuple[int, np.ndarray, float, float]] = []
        self._track: deque[tuple[int, float, float, float, float, float]] = deque()  # t, base x, y, yaw, speed, laser... (see _on_poses)
        self._laser: deque[tuple[int, float, float, float]] = deque()
        self._out: list = []  # heap of (release_ns, seq, kind, payload)
        self._seq = itertools.count()
        self.counts = {"imu_in": 0, "imu_out": 0, "scans_out": 0, "captures_in": 0, "poses_in": 0}
        self._csv = {}
        if str(gp("truth_csv_dir")):
            import csv  # noqa: PLC0415

            out = Path(str(gp("truth_csv_dir")))
            out.mkdir(parents=True, exist_ok=True)
            (out / "measurements_ideal").mkdir(exist_ok=True)
            fh = open(out / "measurements_ideal" / "lidar_points.csv", "w", newline="", encoding="utf-8")
            w = csv.writer(fh)
            w.writerow(["stamp_sec", "stamp_nanosec", "x_forward_m", "y_left_m"])
            self._csv["ideal_points"] = (fh, w)
            for name, header in (("truth_track", ["t_ns", "x", "y", "z", "yaw", "speed"]),
                                 ("truth_scans", ["stamp_ns", "t_start_ns", "t_end_ns", "x", "y", "yaw", "valid_bins"]),
                                 ("imu_raw", ["stamp_sec", "stamp_nanosec", "ax_mps2", "ay_mps2", "az_mps2", "gx_rps", "gy_rps", "gz_rps"]),
                                 # the LaserScans as published (noisy, what the car receives): one row per revolution
                                 ("lidar_scans", ["stamp_ns", "scan_time_s"] + [f"r{i}" for i in range(self.lidar.cfg.beams)])):
                fh = open(out / f"{name}.csv", "w", newline="", encoding="utf-8")
                w = csv.writer(fh)
                w.writerow(header)
                self._csv[name] = (fh, w)

        from gz.msgs10.imu_pb2 import IMU as GzImu  # noqa: PLC0415
        from gz.msgs10.laserscan_pb2 import LaserScan as GzScan  # noqa: PLC0415
        from gz.msgs10.odometry_pb2 import Odometry as GzOdom  # noqa: PLC0415
        from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: PLC0415
        from gz.transport13 import Node as GzNode  # noqa: PLC0415

        self._gz = GzNode()
        self._gz.subscribe(GzImu, str(gp("gz_imu_topic")), self._gz_imu)
        self._gz.subscribe(GzScan, str(gp("gz_scan_topic")), self._gz_scan)
        self._gz.subscribe(Pose_V, str(gp("gz_pose_topic")), self._gz_pose)
        self._gz.subscribe(GzOdom, str(gp("gz_odom_topic")), self._gz_odom)
        self.create_timer(float(gp("publish_period_s")), self._tick)
        self.create_timer(1.0, self._status)
        self.get_logger().info(
            f"real2sim sensors: profile {gp('sensor_profile')} (IMU {self.profile['imu']['rate_hz']} Hz, LiDAR "
            f"{self.profile['lidar']['rate_hz']} Hz x {self.profile['lidar'].get('sample_rate_hz', 'beam')} samples/s, "
            f"{self.profile['lidar'].get('scan_direction', 'ccw')} from {self.profile['lidar'].get('scan_start_angle_deg', -180)} deg), "
            f"LiDAR on {self.lidar_parent} at {self.lidar_xyz.round(4).tolist()}")

    # ------------------------------------------------------ gz-transport side
    @staticmethod
    def _stamp(m) -> int:  # noqa: ANN001
        return int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nsec)

    def _gz_imu(self, m) -> None:  # noqa: ANN001
        v = [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
        with self._lock:
            self._imu_in.append((self._stamp(m), v))

    def _gz_scan(self, m) -> None:  # noqa: ANN001
        with self._lock:
            self._scan_in.append((self._stamp(m), np.asarray(m.ranges, dtype=float), float(m.angle_min), float(m.angle_step)))

    def _gz_pose(self, m) -> None:  # noqa: ANN001
        """Sprung body pose relative to the model (PosePublisher; its Pose_V
        carries no stamp). It changes slowly (suspension mode ~3.5 Hz), so
        the latest one is composed with every stamped base pose."""
        for p in m.pose:
            if p.name.split("::")[-1] == self.lidar_parent:
                rel = (np.array([p.position.x, p.position.y, p.position.z]), np.array([p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]))
                with self._lock:
                    self._rel_latest = rel
                return

    def _gz_odom(self, m) -> None:  # noqa: ANN001
        """World pose of base_link at every physics step (OdometryPublisher)."""
        p, q = m.pose.position, m.pose.orientation
        bp, bq = np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z])
        with self._lock:
            if self._rel_latest is None:
                return
            rp, rq = self._rel_latest
            self._pose_in.append((self._stamp(m), bp, bq, bp + _quat_rot(bq, rp), _quat_mul(bq, rq)))

    # ------------------------------------------------------------ ROS side
    def _tick(self) -> None:
        with self._lock:
            poses, self._pose_in = self._pose_in, []
            imu_in, self._imu_in = self._imu_in, []
            scans_in, self._scan_in = self._scan_in, []
        self.counts["poses_in"] += len(poses)
        for t, bp, bq, sp, sq in sorted(poses, key=lambda r: r[0]):
            speed = 0.0
            if self._track:
                t0, x0, y0 = self._track[-1][0], self._track[-1][1], self._track[-1][2]
                if t > t0:
                    speed = math.hypot(bp[0] - x0, bp[1] - y0) / ((t - t0) * 1e-9)
                    speed = 0.9 * self._track[-1][4] + 0.1 * speed  # ~10 ms smoothing at 1 kHz
            self._track.append((t, bp[0], bp[1], _yaw(bq), speed, bp[2]))
            lp = sp + _quat_rot(sq, self.lidar_xyz)
            lyaw = _yaw(_quat_mul(sq, self.lidar_q))
            self._laser.append((t, lp[0], lp[1], lyaw))
            self.lidar.add_pose(t * 1e-9, lp[0], lp[1], lyaw)
            if t - self._last_truth_ns >= self.truth_period_ns:
                self._last_truth_ns = t
                self._publish_truth(t, bp, bq, speed)
        while len(self._track) > 2 and self._track[0][0] < self._track[-1][0] - 2_000_000_000:
            self._track.popleft()
        while len(self._laser) > 2 and self._laser[0][0] < self._laser[-1][0] - 2_000_000_000:
            self._laser.popleft()
        if imu_in and self._track:
            imu_in.sort(key=lambda r: r[0])
            t_ns = np.array([r[0] for r in imu_in], dtype=np.int64)
            v = np.array([r[1] for r in imu_in])
            tt = np.array([r[0] for r in self._track], dtype=np.float64)
            speed = np.interp(t_ns.astype(np.float64), tt, np.array([r[4] for r in self._track]))
            self.counts["imu_in"] += len(t_ns)
            for s in self.chip.push(t_ns, v[:, :3], v[:, 3:], speed):
                heapq.heappush(self._out, (s.release_ns, next(self._seq), "imu", s))
        for t, ranges, a_min, a_step in scans_in:
            if not self._laser or t > self._laser[-1][0]:
                with self._lock:  # pose not there yet: retry at the next tick
                    self._scan_in.append((t, ranges, a_min, a_step))
                continue
            lt = np.array([r[0] for r in self._laser], dtype=np.float64)
            x = float(np.interp(t, lt, [r[1] for r in self._laser]))
            y = float(np.interp(t, lt, [r[2] for r in self._laser]))
            yw = float(np.interp(t, lt, np.unwrap([r[3] for r in self._laser])))
            self.lidar.add_capture(t * 1e-9, ranges, a_min, a_step, (x, y, yw))
            self.counts["captures_in"] += 1
        for rev in self.lidar.poll():
            if not rev.lost:
                heapq.heappush(self._out, (rev.release_ns, next(self._seq), "scan", rev))
        now = self.get_clock().now().nanoseconds
        while self._out and self._out[0][0] <= now:
            _, _, kind, item = heapq.heappop(self._out)
            (self._publish_imu if kind == "imu" else self._publish_scan)(item)

    def _time(self, t_ns: int):  # noqa: ANN202
        from builtin_interfaces.msg import Time  # noqa: PLC0415

        return Time(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))

    def _publish_imu(self, s) -> None:  # noqa: ANN001
        msg = Imu()
        msg.header.stamp = self._time(s.stamp_ns)
        msg.header.frame_id = self.imu_frame
        msg.orientation_covariance[0] = -1.0  # a raw 6-axis IMU has no orientation
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(v) for v in s.gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(v) for v in s.accel)
        self.imu_pub.publish(msg)
        self.counts["imu_out"] += 1
        if "imu_raw" in self._csv:
            self._csv["imu_raw"][1].writerow([s.stamp_ns // 1_000_000_000, s.stamp_ns % 1_000_000_000, *[f"{v:.6f}" for v in s.accel], *[f"{v:.6f}" for v in s.gyro]])

    def _publish_scan(self, rev) -> None:  # noqa: ANN001
        cfg = self.lidar.cfg
        inc = 2.0 * math.pi / cfg.beams
        msg = LaserScan()
        msg.header.stamp = self._time(rev.stamp_ns)
        msg.header.frame_id = self.laser_frame
        msg.angle_min = float(cfg.angle_min_rad)
        msg.angle_max = float(cfg.angle_min_rad + inc * (cfg.beams - 1))
        msg.angle_increment = float(inc)
        # Nominal time per bin; the firing order (rotation direction, sync
        # angle) is part of the LiDAR calibration.
        msg.time_increment = float(rev.period_s / cfg.beams)
        msg.scan_time = float(rev.period_s)
        msg.range_min, msg.range_max = float(cfg.range_min), float(cfg.range_max)
        msg.ranges = rev.ranges.astype(float).tolist()
        self.scan_pub.publish(msg)
        self.counts["scans_out"] += 1
        if "lidar_scans" in self._csv:  # empty = no return (+inf), 0 = below range_min (-inf)
            self._csv["lidar_scans"][1].writerow([rev.stamp_ns, f"{rev.period_s:.6f}"]
                                                 + [f"{r:.4f}" if np.isfinite(r) else ("" if r > 0 else "0") for r in rev.ranges])
        # Truth at the true first sample of the revolution (evaluation only).
        if self._track:
            tt = np.array([r[0] for r in self._track], dtype=np.float64)
            x = float(np.interp(rev.t_start_ns, tt, [r[1] for r in self._track]))
            y = float(np.interp(rev.t_start_ns, tt, [r[2] for r in self._track]))
            yw = float(np.interp(rev.t_start_ns, tt, np.unwrap([r[3] for r in self._track])))
            ps = PoseStamped()
            ps.header.stamp = self._time(rev.stamp_ns)
            ps.header.frame_id = "world"
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(0.5 * yw), math.cos(0.5 * yw)
            self.truth_scan_pub.publish(ps)
            if "ideal_points" in self._csv:
                # Noise-free revolution (evaluation: part of the track the LiDAR saw).
                th = cfg.angle_min_rad + inc * np.arange(cfg.beams)
                ok = np.isfinite(rev.ideal)
                sec, nsec = rev.stamp_ns // 1_000_000_000, rev.stamp_ns % 1_000_000_000
                w = self._csv["ideal_points"][1]
                for r, a in zip(rev.ideal[ok], th[ok]):
                    w.writerow([sec, nsec, f"{r * math.cos(a):.4f}", f"{r * math.sin(a):.4f}"])
            if "truth_scans" in self._csv:
                self._csv["truth_scans"][1].writerow([rev.stamp_ns, rev.t_start_ns, rev.t_end_ns, f"{x:.5f}", f"{y:.5f}", f"{yw:.6f}", int(np.isfinite(rev.ranges).sum())])

    def _publish_truth(self, t: int, bp: np.ndarray, bq: np.ndarray, speed: float) -> None:
        o = Odometry()
        o.header.stamp = self._time(t)
        o.header.frame_id = "world"
        o.child_frame_id = "base_link_truth"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(v) for v in bp)
        o.pose.pose.orientation.w, o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z = (float(v) for v in bq)
        o.twist.twist.linear.x = float(speed)
        self.truth_pub.publish(o)
        if "truth_track" in self._csv:
            self._csv["truth_track"][1].writerow([t, f"{bp[0]:.5f}", f"{bp[1]:.5f}", f"{bp[2]:.5f}", f"{_yaw(bq):.6f}", f"{speed:.4f}"])

    def _status(self) -> None:
        self.status_pub.publish(json_msg({
            **self.counts, "pending_out": len(self._out), "revolutions": self.lidar.revolutions,
            "imu_period_s": self.chip.period_s, "imu_dropped": self.chip.samples_dropped,
            "lidar_noise_realization": self.lidar.noise.realization(),
        }))


    def destroy_node(self) -> None:
        for fh, _ in self._csv.values():
            fh.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealSensorNode()
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
