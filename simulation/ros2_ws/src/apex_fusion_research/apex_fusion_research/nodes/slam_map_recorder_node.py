"""Records the maps and trajectories of one or more 2D SLAM instances.

For every SLAM ``name`` (parallel parameter lists) the node subscribes to its
``nav_msgs/OccupancyGrid`` and samples TF ``map_frame -> base_frame``. It also
records the ground truth (IMU truth -> base frame through the lever arm) and
the real track walls (``/apex/sim/ground_truth/perfect_map_points``).

World anchoring
---------------
Each SLAM map frame is an arbitrary frame. At the first SLAM pose with a
ground-truth pose close in time, the node fixes

    T_world_map = T_world_base_truth(t0) * inv(T_map_base_slam(t0))

(saved in ``alignment.json``) and writes every map point and pose both in the
SLAM map frame and in the world frame. The anchor uses only the initial pose,
so all later errors (drift, distortion) remain visible.

Output (``<output_dir>/``)
--------------------------
* ``track_truth_points.csv``                x_m, y_m (real walls, world)
* ``truth_trajectory.csv``                  t, x, y, yaw (base_link, world)
* ``slam_<name>_trajectory.csv``            t, x_map, y_map, yaw_map, x_world, y_world, yaw_world
* ``map_<name>_points.csv``                 x_map, y_map, x_world, y_world, occupancy
* ``map_<name>.pgm/.yaml``                  map_server format
* ``snapshots/map_<name>_t<sim s>.csv``     periodic copies (map growth)
* ``alignment.json``, ``slam_summary.json``
Files are rewritten every ``snapshot_period_s`` and on shutdown.
"""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud
from tf2_ros import Buffer, TransformException, TransformListener

from ..core.map_metrics import se2_apply, se2_compose, se2_inverse
from ..core.occupancy import GridInfo, grid_to_points, write_pgm_yaml
from ..core.rotation import quat_to_euler, quat_to_rotmat
from ._common import stamp_to_sec

MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _yaw(q) -> float:
    return quat_to_euler(np.array([q.w, q.x, q.y, q.z]))[2]


class _SlamTrack:
    def __init__(self, name: str, map_topic: str, map_frame: str, base_frame: str) -> None:
        self.name = name
        self.map_topic = map_topic
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.grid: OccupancyGrid | None = None
        self.map_updates = 0
        self.first_map_t: float | None = None
        self.last_pose_t: float | None = None
        self.anchor: tuple[float, float, float] | None = None  # T_world_map
        self.pending: list[tuple[float, float, float, float]] = []
        self.writer = None
        self.file = None
        self.poses = 0


class SlamMapRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_slam_map_recorder")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("slam_names", ["good", "noisy"])
        self.declare_parameter("map_topics", ["/apex/slam/good/map", "/apex/slam/noisy/map"])
        self.declare_parameter("map_frames", ["good/map", "noisy/map"])
        self.declare_parameter("base_frames", ["good/base_link", "noisy/base_link"])
        self.declare_parameter("truth_topic", "/apex/fusion/truth/odom")
        self.declare_parameter("track_topic", "/apex/sim/ground_truth/perfect_map_points")
        self.declare_parameter("imu_offset_in_base_xyz", [0.02, 0.0, 0.08])
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("pose_rate_hz", 10.0)
        self.declare_parameter("truth_rate_hz", 20.0)
        self.declare_parameter("snapshot_period_s", 10.0)
        self.declare_parameter("keep_snapshots", True)
        self.declare_parameter("anchor_max_dt_s", 0.2)
        gp = self.get_parameter

        out = str(gp("output_dir").value).strip()
        if not out:
            raise RuntimeError("fusion_slam_map_recorder needs the 'output_dir' parameter")
        self._out = Path(out).expanduser()
        (self._out / "snapshots").mkdir(parents=True, exist_ok=True)
        self._offset = np.array(gp("imu_offset_in_base_xyz").value, dtype=float)
        self._threshold = int(gp("occupied_threshold").value)
        self._truth_period = 1.0 / float(gp("truth_rate_hz").value)
        self._keep_snapshots = bool(gp("keep_snapshots").value)
        self._anchor_max_dt = float(gp("anchor_max_dt_s").value)

        names = list(gp("slam_names").value)
        topics = list(gp("map_topics").value)
        frames = list(gp("map_frames").value)
        bases = list(gp("base_frames").value)
        if not (len(names) == len(topics) == len(frames) == len(bases)):
            raise RuntimeError("slam_names, map_topics, map_frames and base_frames must have equal length")
        self._slams = [_SlamTrack(*args) for args in zip(names, topics, frames, bases)]

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        for slam in self._slams:
            self.create_subscription(OccupancyGrid, slam.map_topic, lambda m, s=slam: self._on_map(s, m), MAP_QOS)
        self.create_subscription(Odometry, str(gp("truth_topic").value), self._on_truth, 50)
        self.create_subscription(PointCloud, str(gp("track_topic").value), self._on_track, 2)
        self.create_timer(1.0 / float(gp("pose_rate_hz").value), self._sample_slam_poses)
        self.create_timer(float(gp("snapshot_period_s").value), self._export_maps)

        self._truth_t: list[float] = []
        self._truth_pose: list[tuple[float, float, float]] = []
        self._last_truth_csv_t = -1e9
        self._truth_file = open(self._out / "truth_trajectory.csv", "w", newline="", encoding="utf-8")
        self._truth_writer = csv.writer(self._truth_file)
        self._truth_writer.writerow(["t", "x", "y", "yaw"])
        self._track_written = False
        self._latest_sim_t = 0.0
        self._closed = False
        self.get_logger().info(f"recording SLAM maps {names} to {self._out}")

    # ---------------------------------------------------------------- inputs
    def _on_track(self, msg: PointCloud) -> None:
        if self._track_written or not msg.points:
            return
        with open(self._out / "track_truth_points.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x_m", "y_m"])
            for pt in msg.points:
                writer.writerow([f"{pt.x:.5f}", f"{pt.y:.5f}"])
        self._track_written = True
        self.get_logger().info(f"real track written ({len(msg.points)} points)")

    def _on_truth(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        q = np.array([o.w, o.x, o.y, o.z])
        base = np.array([p.x, p.y, p.z]) - quat_to_rotmat(q) @ self._offset
        pose = (float(base[0]), float(base[1]), float(quat_to_euler(q)[2]))
        if self._truth_t and t <= self._truth_t[-1]:
            return
        self._truth_t.append(t)
        self._truth_pose.append(pose)
        if len(self._truth_t) > 6000:  # ~100 s at 60 Hz is plenty for anchoring
            del self._truth_t[:1000]
            del self._truth_pose[:1000]
        self._latest_sim_t = max(self._latest_sim_t, t)
        if t - self._last_truth_csv_t >= self._truth_period:
            self._last_truth_csv_t = t
            self._truth_writer.writerow([f"{t:.4f}", f"{pose[0]:.5f}", f"{pose[1]:.5f}", f"{pose[2]:.6f}"])
            self._truth_file.flush()

    def _truth_at(self, t: float):
        i = bisect.bisect_left(self._truth_t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self._truth_t) and (best is None or abs(self._truth_t[j] - t) < abs(self._truth_t[best] - t)):
                best = j
        if best is None or abs(self._truth_t[best] - t) > self._anchor_max_dt:
            return None
        return self._truth_pose[best]

    def _on_map(self, slam: _SlamTrack, msg: OccupancyGrid) -> None:
        slam.grid = msg
        slam.map_updates += 1
        if slam.first_map_t is None:
            slam.first_map_t = stamp_to_sec(msg.header.stamp)
            self.get_logger().info(f"first map from SLAM '{slam.name}' on {slam.map_topic}")

    # ------------------------------------------------------------ SLAM poses
    def _sample_slam_poses(self) -> None:
        for slam in self._slams:
            try:
                tf = self._tf_buffer.lookup_transform(slam.map_frame, slam.base_frame, Time())
            except TransformException:
                continue
            t = stamp_to_sec(tf.header.stamp)
            if slam.last_pose_t is not None and t <= slam.last_pose_t:
                continue
            slam.last_pose_t = t
            tr = tf.transform.translation
            pose_map = (float(tr.x), float(tr.y), float(_yaw(tf.transform.rotation)))
            if slam.anchor is None:
                truth = self._truth_at(t)
                if truth is None:
                    continue
                slam.anchor = se2_compose(truth, se2_inverse(pose_map))
                self._write_alignment()
                slam.file = open(self._out / f"slam_{slam.name}_trajectory.csv", "w", newline="", encoding="utf-8")
                slam.writer = csv.writer(slam.file)
                slam.writer.writerow(["t", "x_map", "y_map", "yaw_map", "x_world", "y_world", "yaw_world"])
                self.get_logger().info(f"SLAM '{slam.name}' anchored to world at t={t:.2f} s: {tuple(round(v, 4) for v in slam.anchor)}")
            world = se2_compose(slam.anchor, pose_map)
            slam.writer.writerow([f"{t:.4f}", *(f"{v:.5f}" for v in pose_map[:2]), f"{pose_map[2]:.6f}",
                                  *(f"{v:.5f}" for v in world[:2]), f"{world[2]:.6f}"])
            slam.file.flush()
            slam.poses += 1

    def _write_alignment(self) -> None:
        payload = {
            s.name: {"map_frame": s.map_frame, "T_world_map": list(s.anchor) if s.anchor else None}
            for s in self._slams
        }
        (self._out / "alignment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ map export
    def _export_maps(self, final: bool = False) -> None:
        summary = {}
        for slam in self._slams:
            summary[slam.name] = {
                "map_topic": slam.map_topic,
                "map_updates": slam.map_updates,
                "poses_recorded": slam.poses,
                "anchored": slam.anchor is not None,
            }
            if slam.grid is None:
                continue
            g = slam.grid
            o = g.info.origin
            info = GridInfo(g.info.width, g.info.height, g.info.resolution, o.position.x, o.position.y, _yaw(o.orientation))
            xy, occ = grid_to_points(g.data, info, self._threshold)
            world = se2_apply(slam.anchor, xy) if slam.anchor is not None else np.full_like(xy, np.nan)
            rows = np.column_stack((xy, world, occ))
            header = "x_map,y_map,x_world,y_world,occupancy"
            np.savetxt(self._out / f"map_{slam.name}_points.csv", rows, delimiter=",", header=header, comments="",
                       fmt=["%.4f", "%.4f", "%.4f", "%.4f", "%d"])
            write_pgm_yaml(g.data, info, self._out / f"map_{slam.name}", self._threshold)
            if self._keep_snapshots and not final:
                t = stamp_to_sec(g.header.stamp)
                np.savetxt(self._out / "snapshots" / f"map_{slam.name}_t{t:07.1f}.csv", rows, delimiter=",",
                           header=header, comments="", fmt=["%.4f", "%.4f", "%.4f", "%.4f", "%d"])
            summary[slam.name].update(
                {"occupied_cells": int(xy.shape[0]), "width": info.width, "height": info.height,
                 "resolution": info.resolution, "last_map_stamp": stamp_to_sec(g.header.stamp)}
            )
        summary["final"] = final
        (self._out / "slam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._export_maps(final=True)
        for slam in self._slams:
            if slam.file is not None:
                slam.file.close()
        self._truth_file.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamMapRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
