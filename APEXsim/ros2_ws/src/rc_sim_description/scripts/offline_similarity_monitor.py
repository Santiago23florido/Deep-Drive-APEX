#!/usr/bin/env python3
"""Sim-only similarity monitor for online/offline map quality against Gazebo ground truth."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud, PointCloud2
from std_msgs.msg import String


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * ((w * z) + (x * y)),
        1.0 - (2.0 * ((y * y) + (z * z))),
    )


def _rotation_matrix(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)


def _transform_points(points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (points_xy @ _rotation_matrix(float(pose_xyyaw[2])).T) + pose_xyyaw[:2]


def _path_to_xy(path_msg: NavPath) -> np.ndarray:
    if not path_msg.poses:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [
            [float(pose.pose.position.x), float(pose.pose.position.y)]
            for pose in path_msg.poses
        ],
        dtype=np.float64,
    )


def _path_length_m(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    deltas = np.diff(points_xy, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _pointcloud2_xy(msg: PointCloud2) -> np.ndarray:
    if msg.width == 0 or msg.point_step <= 0 or not msg.data:
        return np.empty((0, 2), dtype=np.float64)
    offset_x = None
    offset_y = None
    for field in msg.fields:
        if field.name == "x":
            offset_x = int(field.offset)
        elif field.name == "y":
            offset_y = int(field.offset)
    if offset_x is None or offset_y is None:
        return np.empty((0, 2), dtype=np.float64)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    point_count = int(msg.width * max(1, msg.height))
    if raw.size < (point_count * msg.point_step):
        return np.empty((0, 2), dtype=np.float64)
    raw = raw.reshape((point_count, msg.point_step))
    xs = raw[:, offset_x : offset_x + 4].copy().view(np.float32).reshape(-1)
    ys = raw[:, offset_y : offset_y + 4].copy().view(np.float32).reshape(-1)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((xs[finite], ys[finite])).astype(np.float64, copy=False)


def _pointcloud_xy(msg: PointCloud) -> np.ndarray:
    if not msg.points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [[float(point.x), float(point.y)] for point in msg.points],
        dtype=np.float64,
    )


def _map_metrics(reference_xy: np.ndarray, estimate_xy: np.ndarray, coverage_threshold_m: float) -> dict[str, float]:
    if reference_xy.shape[0] == 0 or estimate_xy.shape[0] == 0:
        return {}
    ref_tree = cKDTree(reference_xy)
    est_tree = cKDTree(estimate_xy)
    est_to_ref_m, _ = ref_tree.query(estimate_xy, k=1)
    ref_to_est_m, _ = est_tree.query(reference_xy, k=1)
    coverage_ratio = float(np.count_nonzero(ref_to_est_m <= coverage_threshold_m) / reference_xy.shape[0])
    return {
        "chamfer_m": float(np.mean(est_to_ref_m) + np.mean(ref_to_est_m)),
        "mean_estimate_to_reference_m": float(np.mean(est_to_ref_m)),
        "mean_reference_to_estimate_m": float(np.mean(ref_to_est_m)),
        "coverage_ratio": coverage_ratio,
    }


def _path_metrics(reference_xy: np.ndarray, estimate_xy: np.ndarray) -> dict[str, float]:
    if reference_xy.shape[0] == 0 or estimate_xy.shape[0] == 0:
        return {}
    reference_length_m = _path_length_m(reference_xy)
    estimate_length_m = _path_length_m(estimate_xy)
    endpoint_error_m = float(np.linalg.norm(reference_xy[-1] - estimate_xy[-1]))
    return {
        "reference_length_m": reference_length_m,
        "estimate_length_m": estimate_length_m,
        "length_ratio": (
            float(estimate_length_m / reference_length_m)
            if reference_length_m > 1.0e-6
            else float("nan")
        ),
        "endpoint_error_m": endpoint_error_m,
    }


@dataclass(frozen=True)
class WorldToEstimation:
    tx_m: float
    ty_m: float
    yaw_rad: float

    def transform_points(self, points_world_xy: np.ndarray) -> np.ndarray:
        return _transform_points(
            points_world_xy,
            np.asarray([self.tx_m, self.ty_m, self.yaw_rad], dtype=np.float64),
        )


class OfflineSimilarityMonitor(Node):
    def __init__(self) -> None:
        super().__init__("offline_similarity_monitor")

        self.declare_parameter("ground_truth_status_topic", "/apex/sim/ground_truth/status")
        self.declare_parameter("perfect_map_topic", "/apex/sim/ground_truth/perfect_map_points")
        self.declare_parameter("ground_truth_path_topic", "/apex/sim/ground_truth/path")
        self.declare_parameter("offline_map_topic", "/apex/sim/offline_refined_map")
        self.declare_parameter("offline_path_topic", "/apex/sim/offline_refined_path")
        self.declare_parameter("online_map_topic", "/apex/estimation/live_map_points")
        self.declare_parameter("online_path_topic", "/apex/estimation/path")
        self.declare_parameter("status_topic", "/apex/sim/offline_similarity_status")
        self.declare_parameter("coverage_threshold_m", 0.12)
        self.declare_parameter("publish_period_sec", 1.0)

        self._ground_truth_status_topic = str(self.get_parameter("ground_truth_status_topic").value)
        self._coverage_threshold_m = max(
            0.01, float(self.get_parameter("coverage_threshold_m").value)
        )
        self._publish_period_s = max(0.2, float(self.get_parameter("publish_period_sec").value))

        self._latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            self._latched_qos,
        )

        self._world_to_estimation: WorldToEstimation | None = None
        self._perfect_map_world_xy = np.empty((0, 2), dtype=np.float64)
        self._ground_truth_path_world_xy = np.empty((0, 2), dtype=np.float64)
        self._offline_map_xy = np.empty((0, 2), dtype=np.float64)
        self._offline_path_xy = np.empty((0, 2), dtype=np.float64)
        self._online_map_xy = np.empty((0, 2), dtype=np.float64)
        self._online_path_xy = np.empty((0, 2), dtype=np.float64)

        self.create_subscription(
            String,
            self._ground_truth_status_topic,
            self._ground_truth_status_cb,
            20,
        )
        self.create_subscription(
            PointCloud,
            str(self.get_parameter("perfect_map_topic").value),
            self._perfect_map_cb,
            2,
        )
        self.create_subscription(
            NavPath,
            str(self.get_parameter("ground_truth_path_topic").value),
            self._ground_truth_path_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("offline_map_topic").value),
            self._offline_map_cb,
            self._latched_qos,
        )
        self.create_subscription(
            NavPath,
            str(self.get_parameter("offline_path_topic").value),
            self._offline_path_cb,
            self._latched_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("online_map_topic").value),
            self._online_map_cb,
            self._latched_qos,
        )
        self.create_subscription(
            NavPath,
            str(self.get_parameter("online_path_topic").value),
            self._online_path_cb,
            20,
        )

        self.create_timer(self._publish_period_s, self._publish_status)
        self.get_logger().info(
            "OfflineSimilarityMonitor started (gt_status=%s perfect_map=%s gt_path=%s offline_map=%s offline_path=%s)"
            % (
                self._ground_truth_status_topic,
                str(self.get_parameter("perfect_map_topic").value),
                str(self.get_parameter("ground_truth_path_topic").value),
                str(self.get_parameter("offline_map_topic").value),
                str(self.get_parameter("offline_path_topic").value),
            )
        )

    def _ground_truth_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        pose_world = payload.get("pose_gt", {})
        pose_estimation = payload.get("pose_gt_estimation_frame", {})
        if not isinstance(pose_world, dict) or not isinstance(pose_estimation, dict):
            return
        required_world = {"x_m", "y_m", "yaw_rad"}
        required_est = {"x_m", "y_m", "yaw_rad"}
        if not required_world.issubset(pose_world) or not required_est.issubset(pose_estimation):
            return
        world_yaw = float(pose_world["yaw_rad"])
        estimation_yaw = float(pose_estimation["yaw_rad"])
        yaw_offset = estimation_yaw - world_yaw
        cos_yaw = math.cos(yaw_offset)
        sin_yaw = math.sin(yaw_offset)
        world_x = float(pose_world["x_m"])
        world_y = float(pose_world["y_m"])
        estimation_x = float(pose_estimation["x_m"])
        estimation_y = float(pose_estimation["y_m"])
        tx_m = estimation_x - ((cos_yaw * world_x) - (sin_yaw * world_y))
        ty_m = estimation_y - ((sin_yaw * world_x) + (cos_yaw * world_y))
        self._world_to_estimation = WorldToEstimation(tx_m=tx_m, ty_m=ty_m, yaw_rad=yaw_offset)

    def _perfect_map_cb(self, msg: PointCloud) -> None:
        self._perfect_map_world_xy = _pointcloud_xy(msg)

    def _ground_truth_path_cb(self, msg: NavPath) -> None:
        self._ground_truth_path_world_xy = _path_to_xy(msg)

    def _offline_map_cb(self, msg: PointCloud2) -> None:
        self._offline_map_xy = _pointcloud2_xy(msg)

    def _offline_path_cb(self, msg: NavPath) -> None:
        self._offline_path_xy = _path_to_xy(msg)

    def _online_map_cb(self, msg: PointCloud2) -> None:
        self._online_map_xy = _pointcloud2_xy(msg)

    def _online_path_cb(self, msg: NavPath) -> None:
        self._online_path_xy = _path_to_xy(msg)

    def _publish_status(self) -> None:
        payload: dict[str, object] = {
            "state": "waiting_for_inputs",
            "coverage_threshold_m": self._coverage_threshold_m,
            "has_world_to_estimation": self._world_to_estimation is not None,
            "has_perfect_map": bool(self._perfect_map_world_xy.shape[0]),
            "has_ground_truth_path": bool(self._ground_truth_path_world_xy.shape[0]),
            "has_offline_map": bool(self._offline_map_xy.shape[0]),
            "has_offline_path": bool(self._offline_path_xy.shape[0]),
            "has_online_map": bool(self._online_map_xy.shape[0]),
            "has_online_path": bool(self._online_path_xy.shape[0]),
        }

        if self._world_to_estimation is not None:
            perfect_map_xy = self._world_to_estimation.transform_points(self._perfect_map_world_xy)
            ground_truth_path_xy = self._world_to_estimation.transform_points(
                self._ground_truth_path_world_xy
            )
            offline_map_metrics = _map_metrics(
                perfect_map_xy,
                self._offline_map_xy,
                self._coverage_threshold_m,
            )
            offline_path_metrics = _path_metrics(ground_truth_path_xy, self._offline_path_xy)
            online_map_metrics = _map_metrics(
                perfect_map_xy,
                self._online_map_xy,
                self._coverage_threshold_m,
            )
            online_path_metrics = _path_metrics(ground_truth_path_xy, self._online_path_xy)

            payload["state"] = "ready"
            payload["counts"] = {
                "perfect_map_point_count": int(perfect_map_xy.shape[0]),
                "ground_truth_path_pose_count": int(ground_truth_path_xy.shape[0]),
                "offline_map_point_count": int(self._offline_map_xy.shape[0]),
                "offline_path_pose_count": int(self._offline_path_xy.shape[0]),
                "online_map_point_count": int(self._online_map_xy.shape[0]),
                "online_path_pose_count": int(self._online_path_xy.shape[0]),
            }
            payload["offline"] = {
                "map": offline_map_metrics,
                "path": offline_path_metrics,
            }
            payload["online"] = {
                "map": online_map_metrics,
                "path": online_path_metrics,
            }
            comparison: dict[str, float] = {}
            offline_chamfer = offline_map_metrics.get("chamfer_m")
            online_chamfer = online_map_metrics.get("chamfer_m")
            if offline_chamfer is not None and online_chamfer is not None:
                comparison["map_chamfer_gain_m"] = float(online_chamfer - offline_chamfer)
            offline_endpoint = offline_path_metrics.get("endpoint_error_m")
            online_endpoint = online_path_metrics.get("endpoint_error_m")
            if offline_endpoint is not None and online_endpoint is not None:
                comparison["endpoint_error_gain_m"] = float(online_endpoint - offline_endpoint)
            payload["comparison"] = comparison

        status_msg = String()
        status_msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(status_msg)


def main() -> None:
    rclpy.init()
    node = OfflineSimilarityMonitor()
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
