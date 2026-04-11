#!/usr/bin/env python3
"""Publish Gazebo ground truth and direct comparison metrics for APEX sim."""

from __future__ import annotations

import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point32, PoseStamped, Quaternion
from gz.msgs10 import model_pb2, pose_v_pb2
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String


def _quat_from_yaw(yaw_rad: float) -> Quaternion:
    quat = Quaternion()
    quat.x = 0.0
    quat.y = 0.0
    quat.z = math.sin(0.5 * yaw_rad)
    quat.w = math.cos(0.5 * yaw_rad)
    return quat


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle_rad(angle_rad: float) -> float:
    wrapped = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def _compose_pose_2d(
    base: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    cos_yaw = math.cos(base[2])
    sin_yaw = math.sin(base[2])
    x = base[0] + (cos_yaw * offset[0]) - (sin_yaw * offset[1])
    y = base[1] + (sin_yaw * offset[0]) + (cos_yaw * offset[1])
    yaw = base[2] + offset[2]
    return x, y, yaw


def _parse_pose(element: ET.Element | None) -> tuple[float, float, float]:
    if element is None or not (element.text or "").strip():
        return 0.0, 0.0, 0.0
    parts = [float(value) for value in (element.text or "").split()]
    while len(parts) < 6:
        parts.append(0.0)
    return parts[0], parts[1], parts[5]


def _distance_point_to_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx) + (aby * aby)
    if denom <= 1.0e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    proj_x = ax + (t * abx)
    proj_y = ay + (t * aby)
    return math.hypot(px - proj_x, py - proj_y)


def _ray_segment_intersection_distance(
    origin: tuple[float, float],
    direction: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ox, oy = origin
    dx, dy = direction
    ax, ay = a
    bx, by = b
    sx = bx - ax
    sy = by - ay
    cross = (dx * sy) - (dy * sx)
    if abs(cross) <= 1.0e-9:
        return float("inf")
    rel_x = ax - ox
    rel_y = ay - oy
    t = ((rel_x * sy) - (rel_y * sx)) / cross
    u = ((rel_x * dy) - (rel_y * dx)) / cross
    if t < 0.0 or u < 0.0 or u > 1.0:
        return float("inf")
    return t


class ApexGroundTruthNode(Node):
    def __init__(self) -> None:
        super().__init__("apex_ground_truth_node")

        self.declare_parameter("world_name", "default")
        self.declare_parameter("model_name", "rc_car")
        self.declare_parameter("world_path", "")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("map_publish_rate_hz", 1.0)
        self.declare_parameter("rear_axle_offset_x_m", -0.15)
        self.declare_parameter("odom_topic", "/apex/sim/ground_truth/odom")
        self.declare_parameter("path_topic", "/apex/sim/ground_truth/path")
        self.declare_parameter("perfect_map_topic", "/apex/sim/ground_truth/perfect_map_points")
        self.declare_parameter("status_topic", "/apex/sim/ground_truth/status")
        self.declare_parameter("fusion_odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter(
            "local_path_topic",
            "/apex/planning/recognition_tour_local_path",
        )
        self.declare_parameter("drive_bridge_status_topic", "/apex/vehicle/drive_bridge_status")
        self.declare_parameter("vehicle_state_topic", "/apex/sim/vehicle_state")
        self.declare_parameter("map_sample_step_m", 0.05)

        self._world_name = str(self.get_parameter("world_name").value)
        self._model_name = str(self.get_parameter("model_name").value)
        self._world_path = str(self.get_parameter("world_path").value)
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._map_publish_rate_hz = max(
            0.1, float(self.get_parameter("map_publish_rate_hz").value)
        )
        self._rear_axle_offset_x_m = float(self.get_parameter("rear_axle_offset_x_m").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._perfect_map_topic = str(self.get_parameter("perfect_map_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._fusion_odom_topic = str(self.get_parameter("fusion_odom_topic").value)
        self._local_path_topic = str(self.get_parameter("local_path_topic").value)
        self._drive_bridge_status_topic = str(
            self.get_parameter("drive_bridge_status_topic").value
        )
        self._vehicle_state_topic = str(self.get_parameter("vehicle_state_topic").value)
        self._map_sample_step_m = max(0.01, float(self.get_parameter("map_sample_step_m").value))

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._path_pub = self.create_publisher(NavPath, self._path_topic, 10)
        self._perfect_map_pub = self.create_publisher(PointCloud, self._perfect_map_topic, 2)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self.create_subscription(Odometry, self._fusion_odom_topic, self._fusion_odom_cb, 20)
        self.create_subscription(NavPath, self._local_path_topic, self._local_path_cb, 10)
        self.create_subscription(String, self._drive_bridge_status_topic, self._json_topic_cb, 20)
        self.create_subscription(String, self._vehicle_state_topic, self._vehicle_state_cb, 20)

        self._lock = threading.Lock()
        self._base_pose: tuple[float, float, float] | None = None
        self._joint_state: dict[str, float] = {}
        self._fusion_odom: Odometry | None = None
        self._local_path_xy: list[tuple[float, float]] = []
        self._drive_bridge_status: dict[str, float | str | bool | dict] = {}
        self._vehicle_state: dict[str, float | str | bool | dict] = {}
        self._gt_path = NavPath()
        self._gt_path.header.frame_id = "world"
        self._last_pose_publish: tuple[float, float, float] | None = None
        self._last_pose_publish_t: float | None = None
        self._world_to_estimation: tuple[float, float, float] | None = None

        self._segments, self._map_points = self._load_world_geometry(self._world_path)
        self._gz_node = GzNode()
        pose_topic = f"/world/{self._world_name}/pose/info"
        joint_topic = f"/world/{self._world_name}/model/{self._model_name}/joint_state"
        self._gz_node.subscribe(pose_v_pb2.Pose_V, pose_topic, self._pose_cb)
        self._gz_node.subscribe(model_pb2.Model, joint_topic, self._joint_state_cb)

        self.create_timer(1.0 / self._publish_rate_hz, self._publish_step)
        self.create_timer(1.0 / self._map_publish_rate_hz, self._publish_map)
        self.get_logger().info(
            "ApexGroundTruthNode started (world=%s model=%s world_path=%s)"
            % (self._world_name, self._model_name, self._world_path)
        )

    def _json_topic_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._drive_bridge_status = payload

    def _vehicle_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._vehicle_state = payload

    def _fusion_odom_cb(self, msg: Odometry) -> None:
        self._fusion_odom = msg

    def _local_path_cb(self, msg: NavPath) -> None:
        path_xy: list[tuple[float, float]] = []
        for pose in msg.poses:
            path_xy.append((float(pose.pose.position.x), float(pose.pose.position.y)))
        self._local_path_xy = path_xy

    def _pose_cb(self, msg: pose_v_pb2.Pose_V) -> None:
        base_pose = None
        model_pose = None
        for pose in msg.pose:
            yaw_rad = _quat_to_yaw(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
            if pose.name == f"{self._model_name}::base_link":
                base_pose = (
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_rad,
                )
                break
            if pose.name == self._model_name:
                model_pose = (
                    float(pose.position.x),
                    float(pose.position.y),
                    yaw_rad,
                )
        with self._lock:
            self._base_pose = base_pose if base_pose is not None else model_pose

    def _joint_state_cb(self, msg: model_pb2.Model) -> None:
        joint_state: dict[str, float] = {}
        for joint in msg.joint:
            joint_state[f"{joint.name}.position"] = float(joint.axis1.position)
            joint_state[f"{joint.name}.velocity"] = float(joint.axis1.velocity)
        with self._lock:
            self._joint_state = joint_state

    def _load_world_geometry(
        self,
        world_path: str,
    ) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[float, float]]]:
        if not world_path:
            return [], []
        path = Path(world_path).expanduser().resolve()
        if not path.exists():
            return [], []
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
        except Exception:
            return [], []

        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        sampled_points: list[tuple[float, float]] = []
        for world in root.findall("world"):
            for model in world.findall("model"):
                if (model.findtext("static", default="false").strip().lower()) != "true":
                    continue
                model_pose = _parse_pose(model.find("pose"))
                for link in model.findall("link"):
                    link_pose = _compose_pose_2d(model_pose, _parse_pose(link.find("pose")))
                    for collision in link.findall("collision"):
                        size_text = collision.findtext("geometry/box/size", default="").strip()
                        if not size_text:
                            continue
                        size = [float(value) for value in size_text.split()]
                        if len(size) < 3 or size[2] < 0.08:
                            continue
                        collision_pose = _compose_pose_2d(
                            link_pose,
                            _parse_pose(collision.find("pose")),
                        )
                        rect_segments = self._rectangle_segments(
                            center_x=collision_pose[0],
                            center_y=collision_pose[1],
                            yaw_rad=collision_pose[2],
                            size_x=size[0],
                            size_y=size[1],
                        )
                        segments.extend(rect_segments)
                        sampled_points.extend(self._sample_segments(rect_segments))
        return segments, sampled_points

    @staticmethod
    def _rectangle_segments(
        *,
        center_x: float,
        center_y: float,
        yaw_rad: float,
        size_x: float,
        size_y: float,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        half_x = 0.5 * size_x
        half_y = 0.5 * size_y
        local_corners = [
            (half_x, half_y),
            (-half_x, half_y),
            (-half_x, -half_y),
            (half_x, -half_y),
        ]
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        world_corners: list[tuple[float, float]] = []
        for x_local, y_local in local_corners:
            world_corners.append(
                (
                    center_x + (cos_yaw * x_local) - (sin_yaw * y_local),
                    center_y + (sin_yaw * x_local) + (cos_yaw * y_local),
                )
            )
        return [
            (world_corners[0], world_corners[1]),
            (world_corners[1], world_corners[2]),
            (world_corners[2], world_corners[3]),
            (world_corners[3], world_corners[0]),
        ]

    def _sample_segments(
        self,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for a, b in segments:
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            samples = max(2, int(math.ceil(length / self._map_sample_step_m)))
            for idx in range(samples):
                ratio = idx / float(samples - 1)
                points.append(
                    (
                        a[0] + (ratio * (b[0] - a[0])),
                        a[1] + (ratio * (b[1] - a[1])),
                    )
                )
        return points

    def _publish_map(self) -> None:
        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        for x_m, y_m in self._map_points:
            point = Point32()
            point.x = float(x_m)
            point.y = float(y_m)
            point.z = 0.0
            msg.points.append(point)
        self._perfect_map_pub.publish(msg)

    def _rear_axle_pose(self, base_pose: tuple[float, float, float]) -> tuple[float, float, float]:
        return _compose_pose_2d(base_pose, (self._rear_axle_offset_x_m, 0.0, 0.0))

    def _ensure_world_to_estimation_alignment(
        self,
        base_pose_world: tuple[float, float, float],
    ) -> None:
        if self._world_to_estimation is not None or self._fusion_odom is None:
            return
        fused_yaw = _quat_to_yaw(
            float(self._fusion_odom.pose.pose.orientation.x),
            float(self._fusion_odom.pose.pose.orientation.y),
            float(self._fusion_odom.pose.pose.orientation.z),
            float(self._fusion_odom.pose.pose.orientation.w),
        )
        yaw_offset = fused_yaw - base_pose_world[2]
        cos_yaw = math.cos(yaw_offset)
        sin_yaw = math.sin(yaw_offset)
        tx = float(self._fusion_odom.pose.pose.position.x) - (
            (cos_yaw * base_pose_world[0]) - (sin_yaw * base_pose_world[1])
        )
        ty = float(self._fusion_odom.pose.pose.position.y) - (
            (sin_yaw * base_pose_world[0]) + (cos_yaw * base_pose_world[1])
        )
        self._world_to_estimation = (tx, ty, yaw_offset)

    def _transform_world_pose_to_estimation(
        self,
        base_pose_world: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        self._ensure_world_to_estimation_alignment(base_pose_world)
        if self._world_to_estimation is None:
            return None
        tx, ty, yaw_offset = self._world_to_estimation
        cos_yaw = math.cos(yaw_offset)
        sin_yaw = math.sin(yaw_offset)
        return (
            tx + (cos_yaw * base_pose_world[0]) - (sin_yaw * base_pose_world[1]),
            ty + (sin_yaw * base_pose_world[0]) + (cos_yaw * base_pose_world[1]),
            base_pose_world[2] + yaw_offset,
        )

    def _clearances(
        self,
        origin: tuple[float, float],
        yaw_rad: float,
    ) -> tuple[float, float, float]:
        left_dir = (-math.sin(yaw_rad), math.cos(yaw_rad))
        right_dir = (math.sin(yaw_rad), -math.cos(yaw_rad))
        left = float("inf")
        right = float("inf")
        nearest = float("inf")
        for segment in self._segments:
            left = min(left, _ray_segment_intersection_distance(origin, left_dir, *segment))
            right = min(right, _ray_segment_intersection_distance(origin, right_dir, *segment))
            nearest = min(nearest, _distance_point_to_segment(origin, *segment))
        return left, right, nearest

    def _path_error_m(self, point: tuple[float, float]) -> float:
        if len(self._local_path_xy) < 2:
            return float("nan")
        return min(
            _distance_point_to_segment(point, self._local_path_xy[idx], self._local_path_xy[idx + 1])
            for idx in range(len(self._local_path_xy) - 1)
        )

    def _publish_step(self) -> None:
        with self._lock:
            base_pose = self._base_pose
            joint_state = dict(self._joint_state)
        if base_pose is None:
            return

        now_msg = self.get_clock().now().to_msg()
        now_monotonic = time.monotonic()
        rear_axle_pose = self._rear_axle_pose(base_pose)
        estimation_pose = self._transform_world_pose_to_estimation(base_pose)
        estimation_rear_axle_pose = (
            self._rear_axle_pose(estimation_pose) if estimation_pose is not None else None
        )

        vx_world = 0.0
        vy_world = 0.0
        yaw_rate_rps = 0.0
        if self._last_pose_publish is not None and self._last_pose_publish_t is not None:
            dt_s = max(1.0e-6, now_monotonic - self._last_pose_publish_t)
            vx_world = (base_pose[0] - self._last_pose_publish[0]) / dt_s
            vy_world = (base_pose[1] - self._last_pose_publish[1]) / dt_s
            yaw_rate_rps = _wrap_angle_rad(base_pose[2] - self._last_pose_publish[2]) / dt_s
        self._last_pose_publish = base_pose
        self._last_pose_publish_t = now_monotonic

        cos_yaw = math.cos(base_pose[2])
        sin_yaw = math.sin(base_pose[2])
        vx_body = (cos_yaw * vx_world) + (sin_yaw * vy_world)
        vy_body = (-sin_yaw * vx_world) + (cos_yaw * vy_world)

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(base_pose[0])
        odom.pose.pose.position.y = float(base_pose[1])
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _quat_from_yaw(base_pose[2])
        odom.twist.twist.linear.x = float(vx_body)
        odom.twist.twist.linear.y = float(vy_body)
        odom.twist.twist.angular.z = float(yaw_rate_rps)
        self._odom_pub.publish(odom)

        if not self._gt_path.poses or (
            math.hypot(
                base_pose[0] - self._gt_path.poses[-1].pose.position.x,
                base_pose[1] - self._gt_path.poses[-1].pose.position.y,
            )
            >= 0.02
        ):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = now_msg
            pose_stamped.header.frame_id = "world"
            pose_stamped.pose.position.x = float(base_pose[0])
            pose_stamped.pose.position.y = float(base_pose[1])
            pose_stamped.pose.orientation = _quat_from_yaw(base_pose[2])
            self._gt_path.poses.append(pose_stamped)
            if len(self._gt_path.poses) > 5000:
                self._gt_path.poses = self._gt_path.poses[-5000:]
        self._gt_path.header.stamp = now_msg
        self._path_pub.publish(self._gt_path)

        left_clearance_m, right_clearance_m, nearest_wall_m = self._clearances(
            (rear_axle_pose[0], rear_axle_pose[1]),
            rear_axle_pose[2],
        )
        steering_deg = float(self._vehicle_state.get("applied_steering_deg", 0.0) or 0.0)
        if abs(steering_deg) <= 1.0e-3:
            inner_clearance_m = min(left_clearance_m, right_clearance_m)
            outer_clearance_m = max(left_clearance_m, right_clearance_m)
        elif steering_deg > 0.0:
            inner_clearance_m = left_clearance_m
            outer_clearance_m = right_clearance_m
        else:
            inner_clearance_m = right_clearance_m
            outer_clearance_m = left_clearance_m

        fusion_error = {}
        if self._fusion_odom is not None and estimation_pose is not None:
            fused_yaw = _quat_to_yaw(
                float(self._fusion_odom.pose.pose.orientation.x),
                float(self._fusion_odom.pose.pose.orientation.y),
                float(self._fusion_odom.pose.pose.orientation.z),
                float(self._fusion_odom.pose.pose.orientation.w),
            )
            dx = estimation_pose[0] - float(self._fusion_odom.pose.pose.position.x)
            dy = estimation_pose[1] - float(self._fusion_odom.pose.pose.position.y)
            fusion_error = {
                "x_m": dx,
                "y_m": dy,
                "pos_m": math.hypot(dx, dy),
                "yaw_rad": _wrap_angle_rad(estimation_pose[2] - fused_yaw),
            }

        gt_path_error_m = (
            self._path_error_m((estimation_rear_axle_pose[0], estimation_rear_axle_pose[1]))
            if estimation_rear_axle_pose is not None
            else float("nan")
        )
        status = String()
        status.data = json.dumps(
            {
                "pose_gt": {
                    "x_m": base_pose[0],
                    "y_m": base_pose[1],
                    "yaw_rad": base_pose[2],
                },
                "rear_axle_gt": {
                    "x_m": rear_axle_pose[0],
                    "y_m": rear_axle_pose[1],
                    "yaw_rad": rear_axle_pose[2],
                },
                "pose_gt_estimation_frame": (
                    {
                        "x_m": estimation_pose[0],
                        "y_m": estimation_pose[1],
                        "yaw_rad": estimation_pose[2],
                    }
                    if estimation_pose is not None
                    else {}
                ),
                "twist_gt": {
                    "vx_mps": vx_body,
                    "vy_mps": vy_body,
                    "yaw_rate_rps": yaw_rate_rps,
                },
                "steering_real_deg": steering_deg,
                "front_left_steering_rad": joint_state.get("front_left_wheel_steer_joint.position"),
                "front_right_steering_rad": joint_state.get("front_right_wheel_steer_joint.position"),
                "clearance_left_m": left_clearance_m,
                "clearance_right_m": right_clearance_m,
                "clearance_inner_m": inner_clearance_m,
                "clearance_outer_m": outer_clearance_m,
                "nearest_wall_m": nearest_wall_m,
                "fusion_error": fusion_error,
                "local_path_error_m": gt_path_error_m,
                "drive_bridge_status": self._drive_bridge_status,
                "vehicle_state": self._vehicle_state,
            },
            separators=(",", ":"),
        )
        self._status_pub.publish(status)


def main() -> None:
    rclpy.init()
    node = ApexGroundTruthNode()
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
