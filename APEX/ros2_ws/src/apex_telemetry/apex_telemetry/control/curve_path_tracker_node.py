#!/usr/bin/env python3
"""Track one planned curve-entry path using Pure Pursuit over fused odometry."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * ((w * z) + (x * y))
    cosy_cosp = 1.0 - (2.0 * ((y * y) + (z * z)))
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


PLANNER_WAIT_STATES = {
    "waiting_fusion",
    "waiting_odom",
    "waiting_static",
    "collecting_snapshot",
}

PLANNER_FAILURE_STATES = {
    "no_curve_detected",
    "missing_odom",
    "error",
}


class CurvePathTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("curve_path_tracker_node")

        self.declare_parameter("path_topic", "/apex/planning/curve_entry_path")
        self.declare_parameter("planning_status_topic", "/apex/planning/curve_entry_status")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("fusion_status_topic", "/apex/estimation/status")
        self.declare_parameter("arm_topic", "/apex/tracking/arm")
        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("status_topic", "/apex/tracking/status")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("rear_axle_offset_x_m", -0.15)
        self.declare_parameter("rear_axle_offset_y_m", 0.0)
        self.declare_parameter("min_lookahead_m", 0.25)
        self.declare_parameter("max_lookahead_m", 0.55)
        self.declare_parameter("lookahead_speed_gain", 0.9)
        self.declare_parameter("min_linear_speed_mps", 0.22)
        self.declare_parameter("max_linear_speed_mps", 0.38)
        self.declare_parameter("curvature_speed_gain", 1.8)
        self.declare_parameter("max_lateral_accel_mps2", 0.08)
        self.declare_parameter("slowdown_distance_m", 0.55)
        self.declare_parameter("goal_tolerance_m", 0.18)
        self.declare_parameter("goal_yaw_tolerance_rad", 0.35)
        self.declare_parameter("goal_line_stop_margin_m", 0.00)
        self.declare_parameter("goal_line_activation_tail_points", 8)
        self.declare_parameter("max_path_deviation_m", 0.45)
        self.declare_parameter("startup_ramp_duration_s", 1.5)
        self.declare_parameter("startup_speed_scale_min", 0.65)
        self.declare_parameter("low_confidence_abort_hold_s", 0.75)
        self.declare_parameter("global_timeout_s", 12.0)
        self.declare_parameter("odom_timeout_s", 0.5)

        self._path_topic = str(self.get_parameter("path_topic").value)
        self._planning_status_topic = str(self.get_parameter("planning_status_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._fusion_status_topic = str(self.get_parameter("fusion_status_topic").value)
        self._arm_topic = str(self.get_parameter("arm_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._wheelbase_m = max(1e-3, float(self.get_parameter("wheelbase_m").value))
        self._rear_axle_offset = np.asarray(
            [
                float(self.get_parameter("rear_axle_offset_x_m").value),
                float(self.get_parameter("rear_axle_offset_y_m").value),
            ],
            dtype=np.float64,
        )
        self._min_lookahead_m = max(0.05, float(self.get_parameter("min_lookahead_m").value))
        self._max_lookahead_m = max(
            self._min_lookahead_m, float(self.get_parameter("max_lookahead_m").value)
        )
        self._lookahead_speed_gain = max(
            0.0, float(self.get_parameter("lookahead_speed_gain").value)
        )
        self._min_linear_speed_mps = max(
            0.0, float(self.get_parameter("min_linear_speed_mps").value)
        )
        self._max_linear_speed_mps = max(
            self._min_linear_speed_mps,
            float(self.get_parameter("max_linear_speed_mps").value),
        )
        self._curvature_speed_gain = max(
            0.0, float(self.get_parameter("curvature_speed_gain").value)
        )
        self._max_lateral_accel_mps2 = max(
            1.0e-4, float(self.get_parameter("max_lateral_accel_mps2").value)
        )
        self._slowdown_distance_m = max(
            self._min_lookahead_m, float(self.get_parameter("slowdown_distance_m").value)
        )
        self._goal_tolerance_m = max(0.05, float(self.get_parameter("goal_tolerance_m").value))
        self._goal_yaw_tolerance_rad = max(
            0.0, float(self.get_parameter("goal_yaw_tolerance_rad").value)
        )
        self._goal_line_stop_margin_m = float(
            self.get_parameter("goal_line_stop_margin_m").value
        )
        self._goal_line_activation_tail_points = max(
            1, int(self.get_parameter("goal_line_activation_tail_points").value)
        )
        self._max_path_deviation_m = max(
            self._goal_tolerance_m, float(self.get_parameter("max_path_deviation_m").value)
        )
        self._startup_ramp_duration_s = max(
            0.0, float(self.get_parameter("startup_ramp_duration_s").value)
        )
        self._startup_speed_scale_min = max(
            0.0,
            min(1.0, float(self.get_parameter("startup_speed_scale_min").value)),
        )
        self._low_conf_abort_hold_s = max(
            0.1, float(self.get_parameter("low_confidence_abort_hold_s").value)
        )
        self._global_timeout_s = max(1.0, float(self.get_parameter("global_timeout_s").value))
        self._odom_timeout_s = max(0.05, float(self.get_parameter("odom_timeout_s").value))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        arm_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Path, self._path_topic, self._path_cb, latched_qos)
        self.create_subscription(
            String, self._planning_status_topic, self._planning_status_cb, latched_qos
        )
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)
        self.create_subscription(String, self._fusion_status_topic, self._fusion_status_cb, 20)
        self.create_subscription(Bool, self._arm_topic, self._arm_cb, arm_qos)

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 20)
        self._status_pub = self.create_publisher(String, self._status_topic, latched_qos)
        self.create_timer(1.0 / self._control_rate_hz, self._control_step)

        self._path_points_xy: np.ndarray | None = None
        self._path_yaw: np.ndarray | None = None
        self._path_s: np.ndarray | None = None
        self._planning_status: dict[str, object] = {}
        self._fusion_status: dict[str, object] = {}
        self._latest_odom: dict[str, float] | None = None
        self._armed = False
        self._state = "waiting_path"
        self._terminal_cause: str | None = None
        self._tracking_started_monotonic: float | None = None
        self._low_conf_since_monotonic: float | None = None
        self._status_payload: dict[str, object] = {
            "state": self._state,
            "terminal": False,
            "cause": None,
        }

        self.get_logger().info(
            "CurvePathTrackerNode started (path=%s odom=%s arm=%s cmd=%s)"
            % (self._path_topic, self._odom_topic, self._arm_topic, self._cmd_vel_topic)
        )

    def _path_cb(self, msg: Path) -> None:
        if not msg.poses:
            return
        points = []
        yaw = []
        for pose in msg.poses:
            points.append([float(pose.pose.position.x), float(pose.pose.position.y)])
            yaw.append(
                _quat_to_yaw(
                    float(pose.pose.orientation.x),
                    float(pose.pose.orientation.y),
                    float(pose.pose.orientation.z),
                    float(pose.pose.orientation.w),
                )
            )
        self._path_points_xy = np.asarray(points, dtype=np.float64)
        self._path_yaw = np.asarray(yaw, dtype=np.float64)
        self._path_s = np.zeros((self._path_points_xy.shape[0],), dtype=np.float64)
        if self._path_points_xy.shape[0] > 1:
            diffs = np.diff(self._path_points_xy, axis=0)
            segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
            self._path_s[1:] = np.cumsum(segment_lengths)

    def _planning_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._planning_status = payload

    def _fusion_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._fusion_status = payload

    def _arm_cb(self, msg: Bool) -> None:
        self._armed = bool(msg.data)

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = {
            "stamp_s": float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec)),
            "x_m": float(msg.pose.pose.position.x),
            "y_m": float(msg.pose.pose.position.y),
            "yaw_rad": _quat_to_yaw(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ),
            "vx_mps": float(msg.twist.twist.linear.x),
            "vy_mps": float(msg.twist.twist.linear.y),
            "yaw_rate_rps": float(msg.twist.twist.angular.z),
        }

    def _publish_cmd(self, linear_x_mps: float, angular_z_rps: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x_mps)
        msg.angular.z = float(angular_z_rps)
        self._cmd_pub.publish(msg)

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self._status_payload, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _set_terminal(self, cause: str) -> None:
        if self._terminal_cause is not None:
            return
        self._terminal_cause = cause
        self._state = cause
        self._publish_cmd(0.0, 0.0)
        self.get_logger().warn(f"Curve path tracking terminated: {cause}")

    def _fusion_confidence(self) -> str:
        latest_pose = self._fusion_status.get("latest_pose") or {}
        if latest_pose:
            return str(latest_pose.get("confidence", "unknown"))
        return "unknown"

    def _tracking_allowed(self) -> bool:
        return bool(
            self._planning_status.get("ready", False)
            and str(self._planning_status.get("state", "")) == "ready"
            and bool(self._fusion_status.get("alignment_ready", False))
            and str(self._fusion_status.get("state", "")) == "tracking"
        )

    def _startup_ramp_scale(self, now_monotonic: float) -> float:
        if self._tracking_started_monotonic is None or self._startup_ramp_duration_s <= 1e-6:
            return 1.0
        elapsed_s = max(0.0, now_monotonic - self._tracking_started_monotonic)
        ratio = min(1.0, elapsed_s / self._startup_ramp_duration_s)
        return self._startup_speed_scale_min + (
            ratio * (1.0 - self._startup_speed_scale_min)
        )

    def _rear_axle_pose(self) -> tuple[np.ndarray, float]:
        if self._latest_odom is None:
            raise RuntimeError("missing odometry")
        yaw = float(self._latest_odom["yaw_rad"])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rear_xy = np.asarray(
            [
                float(self._latest_odom["x_m"])
                + (cos_yaw * self._rear_axle_offset[0])
                - (sin_yaw * self._rear_axle_offset[1]),
                float(self._latest_odom["y_m"])
                + (sin_yaw * self._rear_axle_offset[0])
                + (cos_yaw * self._rear_axle_offset[1]),
            ],
            dtype=np.float64,
        )
        return rear_xy, yaw

    def _control_step(self) -> None:
        now_monotonic = time.monotonic()

        if self._terminal_cause is not None:
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        if self._latest_odom is None:
            self._state = "waiting_odom"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        if self._path_points_xy is None or self._path_s is None or self._path_yaw is None:
            self._state = "waiting_path"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        planner_state = str(self._planning_status.get("state", ""))
        if planner_state and planner_state != "ready":
            if planner_state in PLANNER_FAILURE_STATES and self._terminal_cause is None:
                self._set_terminal("planner_failed")
            elif planner_state not in PLANNER_WAIT_STATES and self._terminal_cause is None:
                self._set_terminal("planner_failed")
            self._state = planner_state
            self._status_payload = {
                "state": self._state,
                "terminal": self._terminal_cause is not None,
                "cause": self._terminal_cause,
                "planner_state": planner_state,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        if not self._tracking_allowed():
            self._state = "waiting_fusion"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
                "fusion_state": self._fusion_status.get("state"),
                "armed": self._armed,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        if not self._armed:
            self._state = "waiting_arm"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
                "planner_ready": bool(self._planning_status.get("ready", False)),
                "fusion_state": self._fusion_status.get("state"),
                "armed": False,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        confidence = self._fusion_confidence()
        if confidence != "high":
            if self._low_conf_since_monotonic is None:
                self._low_conf_since_monotonic = now_monotonic
            elif (now_monotonic - self._low_conf_since_monotonic) >= self._low_conf_abort_hold_s:
                self._set_terminal("aborted_low_confidence")
        else:
            self._low_conf_since_monotonic = None

        if self._tracking_started_monotonic is None:
            self._tracking_started_monotonic = now_monotonic
        elif (now_monotonic - self._tracking_started_monotonic) >= self._global_timeout_s:
            self._set_terminal("timeout")

        odom_age_s = max(0.0, time.time() - float(self._latest_odom["stamp_s"]))
        if odom_age_s > self._odom_timeout_s:
            self._set_terminal("aborted_odom_timeout")

        if self._terminal_cause is not None:
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        rear_xy, rear_yaw = self._rear_axle_pose()
        deltas = self._path_points_xy - rear_xy.reshape(1, 2)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        nearest_index = int(np.argmin(distances))
        path_deviation_m = float(distances[nearest_index])
        if path_deviation_m > self._max_path_deviation_m:
            self._set_terminal("aborted_path_loss")

        goal_point = self._path_points_xy[-1]
        goal_yaw_rad = float(self._path_yaw[-1])
        goal_tangent = np.asarray(
            [math.cos(goal_yaw_rad), math.sin(goal_yaw_rad)],
            dtype=np.float64,
        )
        goal_normal_left = np.asarray(
            [-math.sin(goal_yaw_rad), math.cos(goal_yaw_rad)],
            dtype=np.float64,
        )
        goal_delta_xy = rear_xy - goal_point
        goal_line_projection_m = float(np.dot(goal_delta_xy, goal_tangent))
        goal_line_lateral_error_m = float(np.dot(goal_delta_xy, goal_normal_left))
        goal_line_crossed = bool(
            nearest_index >= max(0, self._path_points_xy.shape[0] - self._goal_line_activation_tail_points)
            and goal_line_projection_m >= self._goal_line_stop_margin_m
        )
        goal_distance_m = float(np.linalg.norm(goal_point - rear_xy))
        goal_yaw_error_rad = abs(
            _normalize_angle(goal_yaw_rad - float(rear_yaw))
        )
        if goal_line_crossed:
            self._set_terminal("goal_line_crossed")
        elif goal_distance_m <= self._goal_tolerance_m and goal_yaw_error_rad <= self._goal_yaw_tolerance_rad:
            self._set_terminal("goal_reached")

        if self._terminal_cause is not None:
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "goal_distance_m": goal_distance_m,
                "path_deviation_m": path_deviation_m,
                "goal_line_projection_m": goal_line_projection_m,
                "goal_line_lateral_error_m": goal_line_lateral_error_m,
                "goal_line_crossed": goal_line_crossed,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        speed_now_mps = math.hypot(
            float(self._latest_odom["vx_mps"]),
            float(self._latest_odom["vy_mps"]),
        )
        lookahead_m = min(
            self._max_lookahead_m,
            max(self._min_lookahead_m, self._min_lookahead_m + (self._lookahead_speed_gain * speed_now_mps)),
        )
        target_s = float(self._path_s[nearest_index] + lookahead_m)
        target_index = int(np.searchsorted(self._path_s, target_s, side="left"))
        target_index = min(target_index, self._path_points_xy.shape[0] - 1)

        target_xy = self._path_points_xy[target_index]
        dx_world = float(target_xy[0] - rear_xy[0])
        dy_world = float(target_xy[1] - rear_xy[1])
        cos_yaw = math.cos(rear_yaw)
        sin_yaw = math.sin(rear_yaw)
        dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
        dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)

        while dx_local <= 1e-3 and target_index < (self._path_points_xy.shape[0] - 1):
            target_index += 1
            target_xy = self._path_points_xy[target_index]
            dx_world = float(target_xy[0] - rear_xy[0])
            dy_world = float(target_xy[1] - rear_xy[1])
            dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
            dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)

        if dx_local <= 1e-3:
            self._set_terminal("aborted_path_loss")
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "goal_distance_m": goal_distance_m,
                "path_deviation_m": path_deviation_m,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        lookahead_actual_m = max(
            self._min_lookahead_m,
            math.hypot(dx_local, dy_local),
        )
        curvature = (2.0 * dy_local) / max(1e-6, lookahead_actual_m * lookahead_actual_m)

        curvature_abs = abs(curvature)
        curvature_speed_limit_mps = self._max_linear_speed_mps / (
            1.0 + (self._curvature_speed_gain * curvature_abs)
        )
        if curvature_abs > 1.0e-6:
            lateral_accel_speed_limit_mps = math.sqrt(
                self._max_lateral_accel_mps2 / curvature_abs
            )
        else:
            lateral_accel_speed_limit_mps = self._max_linear_speed_mps

        linear_x_mps = min(
            self._max_linear_speed_mps,
            curvature_speed_limit_mps,
            lateral_accel_speed_limit_mps,
        )
        goal_speed_limit_mps = self._max_linear_speed_mps
        if goal_distance_m < self._slowdown_distance_m:
            goal_speed_limit_mps = self._max_linear_speed_mps * max(
                0.15, goal_distance_m / self._slowdown_distance_m
            )
            linear_x_mps = min(linear_x_mps, goal_speed_limit_mps)
        startup_ramp_scale = self._startup_ramp_scale(now_monotonic)
        linear_x_mps *= startup_ramp_scale
        min_linear_speed_mps = self._min_linear_speed_mps * startup_ramp_scale
        linear_x_mps = max(
            min_linear_speed_mps if goal_distance_m > self._goal_tolerance_m else 0.0,
            min(self._max_linear_speed_mps, linear_x_mps),
        )
        angular_z_rps = linear_x_mps * curvature

        self._state = "tracking"
        self._status_payload = {
            "state": self._state,
            "terminal": False,
            "cause": None,
            "armed": True,
            "tracking_started": True,
            "elapsed_s": now_monotonic - self._tracking_started_monotonic,
            "planner_ready": bool(self._planning_status.get("ready", False)),
            "fusion_state": self._fusion_status.get("state"),
            "fusion_confidence": confidence,
            "closest_path_index": nearest_index,
            "target_path_index": target_index,
            "goal_distance_m": goal_distance_m,
            "goal_yaw_error_rad": goal_yaw_error_rad,
            "goal_line_projection_m": goal_line_projection_m,
            "goal_line_lateral_error_m": goal_line_lateral_error_m,
            "goal_line_crossed": goal_line_crossed,
            "path_deviation_m": path_deviation_m,
            "startup_ramp_scale": startup_ramp_scale,
            "lookahead_m": lookahead_actual_m,
            "curvature_m_inv": curvature,
            "curvature_speed_limit_mps": curvature_speed_limit_mps,
            "lateral_accel_speed_limit_mps": lateral_accel_speed_limit_mps,
            "goal_speed_limit_mps": goal_speed_limit_mps,
            "cmd_linear_x_mps": linear_x_mps,
            "cmd_angular_z_rps": angular_z_rps,
            "rear_axle_pose": {
                "x_m": float(rear_xy[0]),
                "y_m": float(rear_xy[1]),
                "yaw_rad": float(rear_yaw),
            },
            "target_point": {
                "x_m": float(target_xy[0]),
                "y_m": float(target_xy[1]),
            },
        }
        self._publish_cmd(linear_x_mps, angular_z_rps)
        self._publish_status()


def main() -> None:
    rclpy.init()
    node = CurvePathTrackerNode()
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
