"""Visualisation: noisy LiDAR scans projected with the true and the INS pose.

Each noisy scan is transformed into the world frame twice, using (a) the
ground-truth pose and (b) the pure-INS pose at the scan timestamp. The last
``accumulate_scans`` projections are published as point clouds, so in RViz
the truth-projected cloud stays on the walls (only sensor noise is visible)
while the INS-projected cloud smears and drifts as inertial errors grow.
"""

from __future__ import annotations

import bisect
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from ..core.rotation import quat_to_rotmat
from ._common import stamp_to_sec


class _PoseBuffer:
    def __init__(self, horizon_s: float = 5.0) -> None:
        self._t: list[float] = []
        self._pose: list[tuple[np.ndarray, np.ndarray]] = []
        self._horizon = horizon_s

    def add(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        if self._t and t <= self._t[-1]:
            return
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._t.append(t)
        self._pose.append((np.array([p.x, p.y, p.z]), quat_to_rotmat(np.array([o.w, o.x, o.y, o.z]))))
        while self._t[0] < t - self._horizon:
            self._t.pop(0)
            self._pose.pop(0)

    def nearest(self, t: float, max_age_s: float):
        if not self._t:
            return None
        i = bisect.bisect_left(self._t, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self._t)), key=lambda j: abs(self._t[j] - t))
        return self._pose[best] if abs(self._t[best] - t) <= max_age_s else None


class ScanProjectorNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_scan_projector")
        self.declare_parameter("scan_topic", "/apex/fusion/scan_noisy")
        self.declare_parameter("truth_topic", "/apex/fusion/truth/odom")
        self.declare_parameter("ins_topic", "/apex/fusion/ins/odom")
        self.declare_parameter("truth_cloud_topic", "/apex/fusion/viz/scan_on_truth_pose")
        self.declare_parameter("ins_cloud_topic", "/apex/fusion/viz/scan_on_ins_pose")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("laser_offset_in_imu_xyz", [0.16, 0.0, 0.04])
        self.declare_parameter("accumulate_scans", 15)
        self.declare_parameter("max_pose_age_s", 0.1)
        gp = self.get_parameter

        self._offset = np.array(gp("laser_offset_in_imu_xyz").value, dtype=float)
        self._frame_id = str(gp("frame_id").value)
        self._max_age = float(gp("max_pose_age_s").value)
        n_acc = max(1, int(gp("accumulate_scans").value))
        self._truth = _PoseBuffer()
        self._ins = _PoseBuffer()
        self._acc = {"truth": deque(maxlen=n_acc), "ins": deque(maxlen=n_acc)}
        self._pubs = {
            "truth": self.create_publisher(PointCloud2, str(gp("truth_cloud_topic").value), 5),
            "ins": self.create_publisher(PointCloud2, str(gp("ins_cloud_topic").value), 5),
        }
        self.create_subscription(Odometry, str(gp("truth_topic").value), self._truth.add, 100)
        self.create_subscription(Odometry, str(gp("ins_topic").value), self._ins.add, 200)
        self.create_subscription(LaserScan, str(gp("scan_topic").value), self._on_scan, qos_profile_sensor_data)

    def _on_scan(self, msg: LaserScan) -> None:
        r = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(r.size)
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        pts_laser = np.column_stack((r[ok] * np.cos(angles[ok]), r[ok] * np.sin(angles[ok]), np.zeros(int(ok.sum()))))
        pts_imu = pts_laser + self._offset
        t = stamp_to_sec(msg.header.stamp)
        header = Header(stamp=msg.header.stamp, frame_id=self._frame_id)
        for key, buf in (("truth", self._truth), ("ins", self._ins)):
            pose = buf.nearest(t, self._max_age)
            if pose is None:
                continue
            p, R = pose
            self._acc[key].append(pts_imu @ R.T + p)
            cloud = np.vstack(self._acc[key]).astype(np.float32)
            self._pubs[key].publish(point_cloud2.create_cloud_xyz32(header, cloud.tolist()))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanProjectorNode()
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
