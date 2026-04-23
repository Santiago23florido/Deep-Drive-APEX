#!/usr/bin/env python3
"""Publish a latched PointCloud2 built from offline sensor_fusionn outputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


def _default_reference_script() -> Path:
    env_root = os.environ.get("APEX_SIM_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve() / "tools" / "analysis" / "sensor_fusionn.py"
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "tools" / "analysis" / "sensor_fusionn.py"
        if candidate.exists():
            return candidate
    return current.parents[3] / "tools" / "analysis" / "sensor_fusionn.py"


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _grid_downsample_xy(points_xy: np.ndarray, resolution_m: float) -> np.ndarray:
    if points_xy.size == 0 or resolution_m <= 1.0e-6:
        return points_xy
    grid = np.floor(points_xy / resolution_m).astype(np.int64)
    _, unique_indexes = np.unique(grid, axis=0, return_index=True)
    unique_indexes.sort()
    return points_xy[unique_indexes]


class ApexOfflineSensorfusionMapPublisher(Node):
    def __init__(self) -> None:
        super().__init__("apex_offline_sensorfusion_map_publisher")

        default_reference = _default_reference_script()

        self.declare_parameter("run_dir", "")
        self.declare_parameter("reference_script_path", str(default_reference))
        self.declare_parameter("trajectory_csv", "")
        self.declare_parameter("summary_json", "")
        self.declare_parameter("lidar_points_csv", "")
        self.declare_parameter("frame_id", "offline_map")
        self.declare_parameter("child_frame_id", "offline_base_link")
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("grid_resolution_m", 0.0)
        self.declare_parameter("reload_on_change", False)
        self.declare_parameter("reload_period_s", 1.0)
        self.declare_parameter("allow_missing_inputs", False)
        self.declare_parameter("map_topic", "/apex/sim/offline_map_points")
        self.declare_parameter("path_topic", "/apex/sim/offline_map_path")
        self.declare_parameter("odom_topic", "/apex/sim/offline_map_odom")
        self.declare_parameter("status_topic", "/apex/sim/offline_map_status")

        run_dir = Path(str(self.get_parameter("run_dir").value or "")).expanduser()
        if not str(run_dir):
            raise RuntimeError("run_dir parameter is required")
        self._run_dir = run_dir.resolve()
        self._reference_script_path = Path(
            str(self.get_parameter("reference_script_path").value)
        ).expanduser().resolve()
        self._trajectory_csv = self._resolve_optional_path(
            self.get_parameter("trajectory_csv").value,
            self._run_dir / "analysis_sensor_fusion" / "sensor_fusion_trajectory.csv",
        )
        self._summary_json = self._resolve_optional_path(
            self.get_parameter("summary_json").value,
            self._run_dir / "analysis_sensor_fusion" / "sensor_fusion_summary.json",
        )
        self._lidar_points_csv = self._resolve_optional_path(
            self.get_parameter("lidar_points_csv").value,
            self._run_dir / "lidar_points.csv",
        )
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._child_frame_id = str(self.get_parameter("child_frame_id").value)
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._grid_resolution_m = max(0.0, float(self.get_parameter("grid_resolution_m").value))
        self._reload_on_change = bool(self.get_parameter("reload_on_change").value)
        self._reload_period_s = max(0.2, float(self.get_parameter("reload_period_s").value))
        self._allow_missing_inputs = bool(self.get_parameter("allow_missing_inputs").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)

        self._fusion_module = self._load_reference_module(self._reference_script_path)
        self._last_reload_signature: tuple[float, float, float] | None = None
        self._last_loaded_scan_count = 0
        self._last_loaded_point_count = 0

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_pub = self.create_publisher(PointCloud2, self._map_topic, latched_qos)
        self._path_pub = self.create_publisher(NavPath, self._path_topic, latched_qos)
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)

        self._map_msg: PointCloud2 | None = None
        self._path_msg: NavPath | None = None
        self._odom_msg: Odometry | None = None
        self._status_msg = String()
        self._set_waiting_status("initializing")
        self.create_timer(self._reload_period_s, self._tick)
        self._reload_if_needed(force=True)
        self._republish()
        self.get_logger().info("Offline sensor fusion map publisher ready (run=%s)" % str(self._run_dir))

    @staticmethod
    def _resolve_optional_path(value, fallback: Path) -> Path:
        text = str(value or "").strip()
        return (Path(text).expanduser() if text else fallback).resolve()

    def _load_reference_module(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing reference script: {path}")
        os.environ.setdefault("MPLBACKEND", "Agg")
        spec = importlib.util.spec_from_file_location("apex_forward_raw_sensor_fusionn", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load reference module from: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _load_map_and_path(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._trajectory_csv.exists():
            raise FileNotFoundError(f"Missing trajectory CSV: {self._trajectory_csv}")
        if not self._lidar_points_csv.exists():
            raise FileNotFoundError(f"Missing lidar_points.csv: {self._lidar_points_csv}")

        poses_rows: list[list[float]] = []
        with self._trajectory_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                poses_rows.append(
                    [float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])]
                )
        poses = np.asarray(poses_rows, dtype=np.float64)
        if poses.size == 0:
            raise RuntimeError(f"No rows in trajectory CSV: {self._trajectory_csv}")

        scans = self._fusion_module._load_lidar_scans(self._lidar_points_csv, point_stride=1)
        scan_count = min(len(scans), poses.shape[0])
        scans = scans[:scan_count]
        poses = poses[:scan_count]
        map_points_xy = self._fusion_module._collect_world_points(
            scans,
            poses,
            point_stride=self._point_stride,
            confidence_filter=None,
        )
        map_points_xy = _grid_downsample_xy(map_points_xy, self._grid_resolution_m)
        return map_points_xy, poses

    def _load_summary(self) -> dict[str, object]:
        if not self._summary_json.exists():
            return {}
        try:
            payload = json.loads(self._summary_json.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _input_signature(self) -> tuple[float, float, float]:
        trajectory_mtime = self._trajectory_csv.stat().st_mtime if self._trajectory_csv.exists() else -1.0
        summary_mtime = self._summary_json.stat().st_mtime if self._summary_json.exists() else -1.0
        lidar_mtime = self._lidar_points_csv.stat().st_mtime if self._lidar_points_csv.exists() else -1.0
        return (trajectory_mtime, summary_mtime, lidar_mtime)

    def _set_waiting_status(self, reason: str) -> None:
        self._status_msg.data = json.dumps(
            {
                "run_dir": str(self._run_dir),
                "state": reason,
                "scan_count": int(self._last_loaded_scan_count),
                "map_point_count": int(self._last_loaded_point_count),
                "trajectory_csv": str(self._trajectory_csv),
            },
            separators=(",", ":"),
        )

    def _reload_if_needed(self, *, force: bool = False) -> None:
        signature = self._input_signature()
        if not force and self._reload_on_change and signature == self._last_reload_signature:
            return
        if not self._trajectory_csv.exists() or not self._lidar_points_csv.exists():
            if self._allow_missing_inputs:
                self._set_waiting_status("waiting_snapshot")
                return
            missing = self._trajectory_csv if not self._trajectory_csv.exists() else self._lidar_points_csv
            raise FileNotFoundError(f"Missing input for offline map publisher: {missing}")

        map_points_xy, poses = self._load_map_and_path()
        summary_payload = self._load_summary()
        self._map_msg = self._build_pointcloud(map_points_xy)
        self._path_msg = self._build_path(poses)
        self._odom_msg = self._build_odom(poses[-1])
        self._status_msg.data = json.dumps(
            {
                "run_dir": str(self._run_dir),
                "state": "loaded",
                "scan_count": int(poses.shape[0]),
                "map_point_count": int(map_points_xy.shape[0]),
                "summary": summary_payload,
            },
            separators=(",", ":"),
        )
        self._last_reload_signature = signature
        self._last_loaded_scan_count = int(poses.shape[0])
        self._last_loaded_point_count = int(map_points_xy.shape[0])
        self.get_logger().info(
            "Offline map reloaded (points=%d scans=%d)"
            % (self._last_loaded_point_count, self._last_loaded_scan_count)
        )

    def _republish(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        if self._map_msg is not None and self._path_msg is not None and self._odom_msg is not None:
            self._map_msg.header.stamp = now_msg
            self._path_msg.header.stamp = now_msg
            for pose in self._path_msg.poses:
                pose.header.stamp = now_msg
            self._odom_msg.header.stamp = now_msg
            self._map_pub.publish(self._map_msg)
            self._path_pub.publish(self._path_msg)
            self._odom_pub.publish(self._odom_msg)
        self._status_pub.publish(self._status_msg)

    def _tick(self) -> None:
        if self._reload_on_change:
            try:
                self._reload_if_needed(force=False)
            except FileNotFoundError:
                if self._allow_missing_inputs:
                    self._set_waiting_status("waiting_snapshot")
                else:
                    raise
        self._republish()

    def _build_pointcloud(self, map_points_xy: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = int(map_points_xy.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.width * msg.point_step
        msg.is_dense = True
        if map_points_xy.size == 0:
            msg.data = b""
            return msg
        points_xyz = np.column_stack(
            (
                map_points_xy.astype(np.float32, copy=False),
                np.zeros((map_points_xy.shape[0], 1), dtype=np.float32),
            )
        )
        msg.data = points_xyz.tobytes()
        return msg

    def _build_path(self, poses: np.ndarray) -> NavPath:
        msg = NavPath()
        msg.header.frame_id = self._frame_id
        for pose in poses:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self._frame_id
            pose_msg.pose.position.x = float(pose[0])
            pose_msg.pose.position.y = float(pose[1])
            qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            msg.poses.append(pose_msg)
        return msg

    def _build_odom(self, pose: np.ndarray) -> Odometry:
        msg = Odometry()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = self._child_frame_id
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        return msg


def main() -> None:
    rclpy.init()
    node = ApexOfflineSensorfusionMapPublisher()
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
