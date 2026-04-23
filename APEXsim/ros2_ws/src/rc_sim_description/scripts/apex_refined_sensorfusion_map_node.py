#!/usr/bin/env python3
"""Sim-only RViz map refiner based on the APEX forward_raw batch sensor fusion logic."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
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


class ApexRefinedSensorfusionMapNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_refined_sensorfusion_map_node")

        default_reference = _default_reference_script()

        self.declare_parameter("reference_script_path", str(default_reference))
        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("map_topic", "/apex/sim/refined_map_points")
        self.declare_parameter("path_topic", "/apex/sim/refined_map_path")
        self.declare_parameter("odom_out_topic", "/apex/sim/refined_map_odom")
        self.declare_parameter("status_topic", "/apex/sim/refined_map_status")
        self.declare_parameter("frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("child_frame_id", "base_link_refined")
        self.declare_parameter("publish_period_s", 1.2)
        self.declare_parameter("min_scans", 16)
        self.declare_parameter("max_scans", 240)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("global_point_stride", 3)
        self.declare_parameter("submap_window_scans", 8)
        self.declare_parameter("optimization_window_scans", 48)
        self.declare_parameter("min_new_scans_per_update", 6)
        self.declare_parameter("max_accumulated_points", 120000)
        self.declare_parameter("min_merge_translation_m", 0.025)
        self.declare_parameter("min_merge_yaw_rad", 0.02)
        self.declare_parameter("max_correspondence_m", 0.35)
        self.declare_parameter("corridor_bin_m", 0.10)
        self.declare_parameter("disable_wall_regularization", True)

        self._reference_script_path = Path(
            str(self.get_parameter("reference_script_path").value)
        ).expanduser().resolve()
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._odom_out_topic = str(self.get_parameter("odom_out_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._child_frame_id = str(self.get_parameter("child_frame_id").value)
        self._publish_period_s = max(0.2, float(self.get_parameter("publish_period_s").value))
        self._min_scans = max(4, int(self.get_parameter("min_scans").value))
        self._max_scans = max(self._min_scans, int(self.get_parameter("max_scans").value))
        self._point_stride = max(1, int(self.get_parameter("point_stride").value))
        self._global_point_stride = max(1, int(self.get_parameter("global_point_stride").value))
        self._submap_window_scans = max(2, int(self.get_parameter("submap_window_scans").value))
        self._optimization_window_scans = max(
            self._min_scans,
            int(self.get_parameter("optimization_window_scans").value),
        )
        self._min_new_scans_per_update = max(
            1,
            int(self.get_parameter("min_new_scans_per_update").value),
        )
        self._max_accumulated_points = max(
            2000,
            int(self.get_parameter("max_accumulated_points").value),
        )
        self._min_merge_translation_m = max(
            0.0,
            float(self.get_parameter("min_merge_translation_m").value),
        )
        self._min_merge_yaw_rad = max(0.0, float(self.get_parameter("min_merge_yaw_rad").value))
        self._max_correspondence_m = max(
            0.05, float(self.get_parameter("max_correspondence_m").value)
        )
        self._corridor_bin_m = max(0.02, float(self.get_parameter("corridor_bin_m").value))
        self._disable_wall_regularization = bool(
            self.get_parameter("disable_wall_regularization").value
        )

        self._fusion_module = self._load_reference_module(self._reference_script_path)
        if self._disable_wall_regularization:
            self._fusion_module.LOCAL_WALL_WEIGHT = 0.0
            self._fusion_module.GLOBAL_WALL_WEIGHT = 0.0

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._map_pub = self.create_publisher(PointCloud2, self._map_topic, latched_qos)
        self._path_pub = self.create_publisher(NavPath, self._path_topic, latched_qos)
        self._odom_pub = self.create_publisher(Odometry, self._odom_out_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)

        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)
        self.create_timer(self._publish_period_s, self._kick_processing)

        self._lock = threading.Lock()
        self._latest_odom_pose: np.ndarray | None = None
        self._latest_odom_vel: np.ndarray | None = None
        self._latest_odom_yaw_rate_rps = 0.0
        self._latest_odom_t_s: float | None = None
        self._scan_records: list[dict[str, object]] = []
        self._scan_counter = 0
        self._last_processed_scan_count = 0
        self._processing_thread: threading.Thread | None = None
        self._latest_status_payload = ""
        self._last_merged_scan_index = -1
        self._last_merged_pose: np.ndarray | None = None
        self._accumulated_map_chunks: list[np.ndarray] = []
        self._accumulated_point_count = 0
        self._refined_pose_history: dict[int, np.ndarray] = {}

        self.get_logger().info(
            "ApexRefinedSensorfusionMapNode started (scan=%s odom=%s map=%s path=%s ref=%s)"
            % (
                self._scan_topic,
                self._odom_topic,
                self._map_topic,
                self._path_topic,
                str(self._reference_script_path),
            )
        )

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

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw_rad = math.atan2(
            2.0 * ((float(q.w) * float(q.z)) + (float(q.x) * float(q.y))),
            1.0 - (2.0 * ((float(q.y) * float(q.y)) + (float(q.z) * float(q.z)))),
        )
        pose = np.asarray(
            [
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw_rad,
            ],
            dtype=np.float64,
        )
        vel = np.asarray(
            [
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
            ],
            dtype=np.float64,
        )
        t_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        with self._lock:
            self._latest_odom_pose = pose
            self._latest_odom_vel = vel
            self._latest_odom_yaw_rate_rps = float(msg.twist.twist.angular.z)
            self._latest_odom_t_s = t_s

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            if self._latest_odom_pose is None or self._latest_odom_t_s is None:
                return
            pose = self._latest_odom_pose.copy()
            vel = self._latest_odom_vel.copy() if self._latest_odom_vel is not None else np.zeros(2)
            yaw_rate_rps = float(self._latest_odom_yaw_rate_rps)

        stamp_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        points_local: list[tuple[float, float]] = []
        angle_rad = float(msg.angle_min)
        for raw_range in msg.ranges:
            range_m = float(raw_range)
            if math.isfinite(range_m) and float(msg.range_min) <= range_m <= float(msg.range_max):
                x_scan_m = range_m * math.cos(angle_rad)
                y_scan_m = range_m * math.sin(angle_rad)
                points_local.append((-x_scan_m, y_scan_m))
            angle_rad += float(msg.angle_increment)

        points_array = np.asarray(points_local, dtype=np.float64)
        if points_array.size == 0:
            points_array = np.empty((0, 2), dtype=np.float64)
        sampled_points = points_array[:: self._point_stride].copy()
        lower_wall_points, upper_wall_points = self._fusion_module._extract_sidewall_candidates(
            points_array
        )
        scan = self._fusion_module.LidarScan(
            scan_index=self._scan_counter,
            t_s=stamp_s,
            points_local=points_array,
            sampled_points_local=sampled_points,
            lower_wall_points_local=lower_wall_points,
            upper_wall_points_local=upper_wall_points,
        )
        self._scan_counter += 1

        with self._lock:
            self._scan_records.append(
                {
                    "scan": scan,
                    "pose": pose,
                    "velocity": vel,
                    "yaw_rate_rps": yaw_rate_rps,
                }
            )
            if len(self._scan_records) > self._max_scans:
                del self._scan_records[: len(self._scan_records) - self._max_scans]

    def _kick_processing(self) -> None:
        with self._lock:
            scan_count = len(self._scan_records)
        if scan_count < self._min_scans:
            return
        if (scan_count - self._last_processed_scan_count) < self._min_new_scans_per_update:
            return
        if scan_count == self._last_processed_scan_count:
            return
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return
        self._processing_thread = threading.Thread(target=self._process_snapshot, daemon=True)
        self._processing_thread.start()

    def _dummy_wall_model(self, scans: list[object], poses: np.ndarray):
        points = self._fusion_module._collect_world_points(
            scans,
            poses,
            point_stride=self._global_point_stride,
            confidence_filter=None,
        )
        if points.shape[0] >= 40:
            try:
                return self._fusion_module._fit_wall_model(points, self._corridor_bin_m)
            except Exception:
                pass
        if points.shape[0]:
            lower = float(np.quantile(points[:, 1], 0.10))
            upper = float(np.quantile(points[:, 1], 0.90))
        else:
            lower = -0.55
            upper = 0.55
        width = max(0.2, upper - lower)
        return self._fusion_module.WallModel(
            lower_coef=np.asarray([0.0, lower], dtype=np.float64),
            upper_coef=np.asarray([0.0, upper], dtype=np.float64),
            width_m=float(width),
            corridor_yaw_rad=0.0,
        )

    def _process_snapshot(self) -> None:
        try:
            start_t = time.perf_counter()
            with self._lock:
                snapshot = list(self._scan_records)
                last_merged_scan_index = int(self._last_merged_scan_index)
            if len(snapshot) < self._min_scans:
                return

            window_snapshot = snapshot[-self._optimization_window_scans :]
            scans = [record["scan"] for record in window_snapshot]
            initial_poses = np.vstack([record["pose"] for record in window_snapshot]).astype(np.float64)
            velocity_priors = np.vstack([record["velocity"] for record in window_snapshot]).astype(np.float64)
            yaw_priors = initial_poses[:, 2].copy()
            scan_times_s = np.asarray([float(scan.t_s) for scan in scans], dtype=np.float64)

            if velocity_priors.shape[0] > 1:
                velocity_priors[0] = velocity_priors[1]

            wall_model = self._dummy_wall_model(scans, initial_poses)

            refined_poses = self._fusion_module._refine_global_poses(
                initial_poses,
                scans,
                wall_model,
                yaw_priors,
                velocity_priors,
                initial_scan_count=1,
                scan_times_s=scan_times_s,
                submap_window_scans=self._submap_window_scans,
                point_stride=self._global_point_stride,
                max_correspondence_m=self._max_correspondence_m,
            )

            qualities = self._fusion_module._compute_scan_qualities(
                scans,
                refined_poses,
                wall_model,
                initial_scan_count=1,
                submap_window_scans=self._submap_window_scans,
                max_correspondence_m=self._max_correspondence_m,
            )
            confidence_filter = [quality.confidence for quality in qualities]
            map_points_xy = self._fusion_module._collect_world_points(
                scans,
                refined_poses,
                point_stride=self._global_point_stride,
                confidence_filter=confidence_filter,
            )
            if map_points_xy.shape[0] < 60:
                map_points_xy = self._fusion_module._collect_world_points(
                    scans,
                    refined_poses,
                    point_stride=self._global_point_stride,
                    confidence_filter=None,
                )

            newest_scan_index = last_merged_scan_index
            appended_points: list[np.ndarray] = []
            for local_index, record in enumerate(window_snapshot):
                scan = record["scan"]
                scan_index = int(scan.scan_index)
                refined_pose = refined_poses[local_index].copy()
                self._refined_pose_history[scan_index] = refined_pose
                if scan_index <= last_merged_scan_index:
                    continue
                newest_scan_index = max(newest_scan_index, scan_index)
                if not self._should_merge_pose(refined_pose):
                    continue
                points_local = scan.points_local[:: self._global_point_stride]
                appended_points.append(self._fusion_module._transform_points(points_local, refined_pose))
                self._last_merged_pose = refined_pose

            if appended_points:
                new_chunk = np.vstack(appended_points)
                self._append_map_chunk(new_chunk)

            refined_path_poses = self._build_refined_path_poses()
            accumulated_map_points = self._stack_accumulated_map_points()

            self._publish_outputs(
                map_points_xy=accumulated_map_points if accumulated_map_points.size else map_points_xy,
                refined_poses=refined_path_poses if refined_path_poses.size else refined_poses,
                latest_pose=refined_poses[-1],
                qualities=qualities,
                processing_ms=1000.0 * (time.perf_counter() - start_t),
            )
            self._last_processed_scan_count = len(snapshot)
            self._last_merged_scan_index = newest_scan_index
        except Exception as exc:
            self.get_logger().warn(f"Refined map optimization failed: {repr(exc)}")

    def _append_map_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        points = np.asarray(chunk, dtype=np.float64)
        self._accumulated_map_chunks.append(points)
        self._accumulated_point_count += int(points.shape[0])
        while self._accumulated_map_chunks and self._accumulated_point_count > self._max_accumulated_points:
            removed = self._accumulated_map_chunks.pop(0)
            self._accumulated_point_count -= int(removed.shape[0])

    def _stack_accumulated_map_points(self) -> np.ndarray:
        if not self._accumulated_map_chunks:
            return np.empty((0, 2), dtype=np.float64)
        if len(self._accumulated_map_chunks) == 1:
            return self._accumulated_map_chunks[0]
        return np.vstack(self._accumulated_map_chunks)

    def _build_refined_path_poses(self) -> np.ndarray:
        if not self._refined_pose_history:
            return np.empty((0, 3), dtype=np.float64)
        ordered_indexes = sorted(self._refined_pose_history)
        return np.vstack([self._refined_pose_history[index] for index in ordered_indexes]).astype(
            np.float64
        )

    def _should_merge_pose(self, pose: np.ndarray) -> bool:
        if self._last_merged_pose is None:
            return True
        translation_m = float(np.linalg.norm(pose[:2] - self._last_merged_pose[:2]))
        yaw_delta_rad = math.atan2(
            math.sin(float(pose[2] - self._last_merged_pose[2])),
            math.cos(float(pose[2] - self._last_merged_pose[2])),
        )
        return (
            translation_m >= self._min_merge_translation_m
            or abs(yaw_delta_rad) >= self._min_merge_yaw_rad
        )

    def _publish_outputs(
        self,
        *,
        map_points_xy: np.ndarray,
        refined_poses: np.ndarray,
        latest_pose: np.ndarray,
        qualities: list[object],
        processing_ms: float,
    ) -> None:
        now_msg = self.get_clock().now().to_msg()
        self._map_pub.publish(self._pointcloud_message_from_xy(map_points_xy, now_msg))

        path_msg = NavPath()
        path_msg.header.stamp = now_msg
        path_msg.header.frame_id = self._frame_id
        for pose in refined_poses:
            pose_msg = PoseStamped()
            pose_msg.header = path_msg.header
            pose_msg.pose.position.x = float(pose[0])
            pose_msg.pose.position.y = float(pose[1])
            qx, qy, qz, qw = _yaw_to_quat(float(pose[2]))
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            path_msg.poses.append(pose_msg)
        self._path_pub.publish(path_msg)

        odom = Odometry()
        odom.header = path_msg.header
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x = float(latest_pose[0])
        odom.pose.pose.position.y = float(latest_pose[1])
        qx, qy, qz, qw = _yaw_to_quat(float(latest_pose[2]))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom)

        median_submap = self._nanmedian(
            [float(quality.median_submap_residual_m) for quality in qualities]
        )
        high_confidence_pct = 100.0 * (
            sum(1 for quality in qualities if str(quality.confidence) == "high")
            / max(1, len(qualities))
        )
        status_payload = {
            "scan_count": len(refined_poses),
            "map_point_count": int(map_points_xy.shape[0]),
            "high_confidence_pct": high_confidence_pct,
            "median_submap_residual_m": median_submap,
            "processing_ms": processing_ms,
            "latest_pose": {
                "x_m": float(latest_pose[0]),
                "y_m": float(latest_pose[1]),
                "yaw_rad": float(latest_pose[2]),
            },
        }
        status = String()
        status.data = json.dumps(status_payload, separators=(",", ":"))
        self._status_pub.publish(status)
        self._latest_status_payload = status.data

    @staticmethod
    def _nanmedian(values: list[float]) -> float:
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            return float("nan")
        return float(np.median(np.asarray(finite, dtype=np.float64)))

    def _pointcloud_message_from_xy(self, map_points_xy: np.ndarray, stamp_msg) -> PointCloud2:
        message = PointCloud2()
        message.header.stamp = stamp_msg
        message.header.frame_id = self._frame_id
        message.height = 1
        message.width = int(map_points_xy.shape[0])
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = message.point_step * message.width
        message.is_dense = True
        if map_points_xy.size == 0:
            message.data = b""
            return message
        points_xyz = np.column_stack(
            (
                map_points_xy.astype(np.float32, copy=False),
                np.zeros((map_points_xy.shape[0], 1), dtype=np.float32),
            )
        )
        message.data = points_xyz.tobytes()
        return message


def main() -> None:
    rclpy.init()
    node = ApexRefinedSensorfusionMapNode()
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
