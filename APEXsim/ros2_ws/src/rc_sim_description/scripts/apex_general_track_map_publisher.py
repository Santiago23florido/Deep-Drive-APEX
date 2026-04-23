#!/usr/bin/env python3
"""Publish a general fixed track map as OccupancyGrid + visual points + path."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path as NavPath
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _read_binary_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise RuntimeError(f"Unsupported PGM format in {path}: {magic!r}")

        def _next_token() -> bytes:
            while True:
                line = handle.readline()
                if not line:
                    raise RuntimeError(f"Unexpected EOF while reading {path}")
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                return line

        dims = _next_token().split()
        if len(dims) != 2:
            raise RuntimeError(f"Invalid PGM dimensions in {path}")
        width = int(dims[0])
        height = int(dims[1])
        max_value = int(_next_token())
        if max_value <= 0 or max_value > 255:
            raise RuntimeError(f"Unsupported PGM max_value in {path}: {max_value}")
        data = np.frombuffer(handle.read(width * height), dtype=np.uint8)
        if data.size != (width * height):
            raise RuntimeError(f"Incomplete PGM pixel data in {path}")
    image = data.reshape((height, width))
    return np.flipud(image)


def _load_visual_points(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                points.append([float(row["x_m"]), float(row["y_m"])])
            except Exception:
                continue
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _load_path_poses(path: Path) -> np.ndarray:
    poses: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                poses.append([float(row["x_m"]), float(row["y_m"]), float(row["yaw_rad"])])
            except Exception:
                continue
    if not poses:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(poses, dtype=np.float64)


class ApexGeneralTrackMapPublisher(Node):
    def __init__(self) -> None:
        super().__init__("apex_general_track_map_publisher")

        self.declare_parameter("map_yaml", "")
        self.declare_parameter("summary_json", "")
        self.declare_parameter("frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("grid_topic", "/apex/sim/mapping_preview/grid")
        self.declare_parameter("visual_points_topic", "/apex/sim/mapping_preview/visual_points")
        self.declare_parameter("path_topic", "/apex/sim/mapping_preview/path")
        self.declare_parameter("status_topic", "/apex/sim/mapping_preview/status")
        self.declare_parameter("reload_on_change", False)
        self.declare_parameter("reload_period_s", 1.0)
        self.declare_parameter("allow_missing_inputs", False)
        self.declare_parameter("point_stride", 1)

        map_yaml_text = str(self.get_parameter("map_yaml").value or "").strip()
        if not map_yaml_text:
            raise RuntimeError("map_yaml parameter is required")
        self._map_yaml = Path(map_yaml_text).expanduser().resolve()
        self._summary_json = Path(
            str(self.get_parameter("summary_json").value or "").strip() or (self._map_yaml.parent / "mapping_summary.json")
        ).expanduser().resolve()
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._reload_on_change = bool(self.get_parameter("reload_on_change").value)
        self._reload_period_s = max(0.2, float(self.get_parameter("reload_period_s").value))
        self._allow_missing_inputs = bool(self.get_parameter("allow_missing_inputs").value)
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._grid_pub = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("grid_topic").value),
            latched_qos,
        )
        self._visual_points_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("visual_points_topic").value),
            latched_qos,
        )
        self._path_pub = self.create_publisher(
            NavPath,
            str(self.get_parameter("path_topic").value),
            latched_qos,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            latched_qos,
        )

        self._grid_msg: OccupancyGrid | None = None
        self._visual_points_msg: PointCloud2 | None = None
        self._path_msg: NavPath | None = None
        self._status_msg = String()
        self._last_signature: tuple[float, float, float, float] | None = None
        self._set_waiting_status("initializing")
        self._reload_if_needed(force=True)
        self.create_timer(self._reload_period_s, self._tick)
        self.get_logger().info(f"General track map publisher ready (map_yaml={self._map_yaml})")

    def _input_signature(self) -> tuple[float, float, float, float]:
        yaml_mtime = self._map_yaml.stat().st_mtime if self._map_yaml.exists() else -1.0
        summary_mtime = self._summary_json.stat().st_mtime if self._summary_json.exists() else -1.0
        visual_mtime = -1.0
        keyframes_mtime = -1.0
        if self._map_yaml.exists():
            payload = yaml.safe_load(self._map_yaml.read_text(encoding="utf-8")) or {}
            visual_rel = payload.get("visual_points_csv", "")
            keyframes_rel = payload.get("optimized_keyframes_csv", "")
            if visual_rel:
                visual_path = (self._map_yaml.parent / str(visual_rel)).resolve()
                visual_mtime = visual_path.stat().st_mtime if visual_path.exists() else -1.0
            if keyframes_rel:
                keyframes_path = (self._map_yaml.parent / str(keyframes_rel)).resolve()
                keyframes_mtime = keyframes_path.stat().st_mtime if keyframes_path.exists() else -1.0
        return (yaml_mtime, summary_mtime, visual_mtime, keyframes_mtime)

    def _set_waiting_status(self, state: str) -> None:
        self._status_msg.data = json.dumps(
            {
                "state": state,
                "map_yaml": str(self._map_yaml),
            },
            separators=(",", ":"),
        )

    def _reload_if_needed(self, *, force: bool) -> None:
        signature = self._input_signature()
        if not force and self._reload_on_change and signature == self._last_signature:
            return
        if not self._map_yaml.exists():
            if self._allow_missing_inputs:
                self._set_waiting_status("waiting_map_yaml")
                return
            raise FileNotFoundError(f"Missing map_yaml: {self._map_yaml}")

        map_yaml_payload = yaml.safe_load(self._map_yaml.read_text(encoding="utf-8")) or {}
        image_path = (self._map_yaml.parent / str(map_yaml_payload["image"])).resolve()
        visual_points_path = (
            self._map_yaml.parent / str(map_yaml_payload["visual_points_csv"])
        ).resolve()
        keyframes_path = (
            self._map_yaml.parent / str(map_yaml_payload["optimized_keyframes_csv"])
        ).resolve()
        if not image_path.exists() or not visual_points_path.exists() or not keyframes_path.exists():
            if self._allow_missing_inputs:
                self._set_waiting_status("waiting_map_assets")
                return
            raise FileNotFoundError(f"Missing map asset under {self._map_yaml.parent}")

        image = _read_binary_pgm(image_path)
        resolution_m = float(map_yaml_payload["resolution"])
        origin_xy = np.asarray(map_yaml_payload["origin"][:2], dtype=np.float64)
        visual_points_xy = _load_visual_points(visual_points_path)[:: self._point_stride]
        path_poses = _load_path_poses(keyframes_path)
        summary_payload = {}
        if self._summary_json.exists():
            try:
                payload = json.loads(self._summary_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    summary_payload = payload
            except Exception:
                summary_payload = {}

        self._grid_msg = self._build_grid(image, resolution_m, origin_xy)
        self._visual_points_msg = self._build_pointcloud(visual_points_xy)
        self._path_msg = self._build_path(path_poses)
        self._status_msg.data = json.dumps(
            {
                "state": "loaded",
                "map_yaml": str(self._map_yaml),
                "occupied_cells": int(np.count_nonzero(image < 128)),
                "visual_point_count": int(visual_points_xy.shape[0]),
                "keyframe_count": int(path_poses.shape[0]),
                "summary": summary_payload,
            },
            separators=(",", ":"),
        )
        self._last_signature = signature
        self.get_logger().info(
            "General track map reloaded (occupied=%d visual_points=%d keyframes=%d)"
            % (
                int(np.count_nonzero(image < 128)),
                int(visual_points_xy.shape[0]),
                int(path_poses.shape[0]),
            )
        )

    def _build_grid(self, image: np.ndarray, resolution_m: float, origin_xy: np.ndarray) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.frame_id = self._frame_id
        meta = MapMetaData()
        meta.resolution = float(resolution_m)
        meta.width = int(image.shape[1])
        meta.height = int(image.shape[0])
        meta.origin.position.x = float(origin_xy[0])
        meta.origin.position.y = float(origin_xy[1])
        meta.origin.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(0.0)
        meta.origin.orientation.x = qx
        meta.origin.orientation.y = qy
        meta.origin.orientation.z = qz
        meta.origin.orientation.w = qw
        grid.info = meta
        occupancy_values = np.where(image < 128, 100, 0).astype(np.int8)
        grid.data = occupancy_values.reshape(-1).tolist()
        return grid

    def _build_pointcloud(self, points_xy: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = int(points_xy.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        if points_xy.size == 0:
            msg.data = b""
            return msg
        points_xyz = np.column_stack(
            (
                points_xy.astype(np.float32, copy=False),
                np.zeros((points_xy.shape[0], 1), dtype=np.float32),
            )
        )
        msg.data = points_xyz.tobytes()
        return msg

    def _build_path(self, poses_xyyaw: np.ndarray) -> NavPath:
        path = NavPath()
        path.header.frame_id = self._frame_id
        for pose in poses_xyyaw:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self._frame_id
            pose_msg.pose.position.x = float(pose[0])
            pose_msg.pose.position.y = float(pose[1])
            qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            path.poses.append(pose_msg)
        return path

    def _tick(self) -> None:
        if self._reload_on_change:
            try:
                self._reload_if_needed(force=False)
            except FileNotFoundError:
                if self._allow_missing_inputs:
                    self._set_waiting_status("waiting_map_assets")
                else:
                    raise
        self._republish()

    def _republish(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        if self._grid_msg is not None:
            self._grid_msg.header.stamp = now_msg
            self._grid_msg.info.map_load_time = now_msg
            self._grid_pub.publish(self._grid_msg)
        if self._visual_points_msg is not None:
            self._visual_points_msg.header.stamp = now_msg
            self._visual_points_pub.publish(self._visual_points_msg)
        if self._path_msg is not None:
            self._path_msg.header.stamp = now_msg
            for pose in self._path_msg.poses:
                pose.header.stamp = now_msg
            self._path_pub.publish(self._path_msg)
        self._status_pub.publish(self._status_msg)


def main() -> None:
    rclpy.init()
    node = ApexGeneralTrackMapPublisher()
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
