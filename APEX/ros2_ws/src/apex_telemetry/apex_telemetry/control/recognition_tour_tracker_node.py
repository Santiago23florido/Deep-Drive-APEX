#!/usr/bin/env python3
"""Track a continuously refreshed local path for full-lap recognition."""

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

from .curve_path_tracker_node import _normalize_angle, _quat_to_yaw


PLANNER_WAIT_STATES = {
    "waiting_fusion",
    "waiting_odom",
    "waiting_local_path",
    "holding_last_path",
}

PLANNER_TERMINAL_STATES = {
    "loop_closed",
    "timeout",
}

PLANNER_FAILURE_STATES = {
    "error",
}


def _estimate_path_curvature(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float64)
    point_count = path_xy.shape[0]
    curvature = np.zeros((point_count,), dtype=np.float64)
    if point_count < 3:
        return curvature
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    for index in range(1, point_count - 1):
        ds = max(1.0e-6, 0.5 * (seg_lengths[index - 1] + seg_lengths[index]))
        dtheta = math.atan2(
            math.sin(float(headings[index] - headings[index - 1])),
            math.cos(float(headings[index] - headings[index - 1])),
        )
        curvature[index] = dtheta / ds
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


class RecognitionTourTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("recognition_tour_tracker_node")

        self.declare_parameter("path_topic", "/apex/planning/recognition_tour_local_path")
        self.declare_parameter("planning_status_topic", "/apex/planning/recognition_tour_status")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("fusion_status_topic", "/apex/estimation/status")
        self.declare_parameter("arm_topic", "/apex/tracking/arm")
        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("status_topic", "/apex/tracking/recognition_tour_status")
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("wheelbase_m", 0.30)
        self.declare_parameter("steering_limit_deg", 18.0)
        self.declare_parameter("rear_axle_offset_x_m", -0.15)
        self.declare_parameter("rear_axle_offset_y_m", 0.0)
        self.declare_parameter("min_lookahead_m", 0.40)
        self.declare_parameter("max_lookahead_m", 0.90)
        self.declare_parameter("lookahead_speed_gain", 0.70)
        self.declare_parameter("sharp_turn_lookahead_min_m", 0.26)
        self.declare_parameter("lookahead_curvature_gain", 0.70)
        self.declare_parameter("lookahead_curvature_window_points", 10)
        self.declare_parameter("sharp_turn_speed_boost_start_deg", 24.0)
        self.declare_parameter("sharp_turn_speed_boost_full_deg", 38.0)
        self.declare_parameter("sharp_turn_speed_boost_min_speed_mps", 0.12)
        self.declare_parameter("min_linear_speed_mps", 0.08)
        self.declare_parameter("max_linear_speed_mps", 0.24)
        self.declare_parameter("curvature_speed_gain", 1.8)
        self.declare_parameter("max_lateral_accel_mps2", 0.08)
        self.declare_parameter("path_end_slowdown_distance_m", 0.60)
        self.declare_parameter("path_end_goal_tolerance_m", 0.18)
        self.declare_parameter("path_end_yaw_tolerance_rad", 0.45)
        self.declare_parameter("path_end_line_stop_margin_m", 0.00)
        self.declare_parameter("path_end_line_activation_tail_points", 8)
        self.declare_parameter("max_path_deviation_m", 0.45)
        self.declare_parameter("angular_cmd_ema_alpha", 0.12)
        self.declare_parameter("curvature_deadband_m_inv", 0.08)
        self.declare_parameter("startup_ramp_duration_s", 1.5)
        self.declare_parameter("startup_speed_scale_min", 0.65)
        self.declare_parameter("low_confidence_abort_hold_s", 0.75)
        self.declare_parameter("low_confidence_speed_scale", 0.60)
        self.declare_parameter("path_stale_max_age_s", 0.30)
        self.declare_parameter("path_stale_abort_hold_s", 0.40)
        self.declare_parameter("no_forward_target_abort_hold_s", 0.45)
        self.declare_parameter("waiting_path_refresh_hold_s", 0.55)
        self.declare_parameter("waiting_path_refresh_near_end_distance_m", 0.75)
        self.declare_parameter("path_projection_heading_tolerance_rad", 1.10)
        self.declare_parameter("path_deviation_abort_margin_m", 0.15)
        self.declare_parameter("path_refresh_min_forward_speed_mps", 0.06)
        self.declare_parameter("steering_saturation_start_ratio", 0.85)
        self.declare_parameter("steering_saturation_speed_scale_min", 0.58)
        self.declare_parameter("global_timeout_s", 60.0)
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
        self._steering_limit_deg = max(1.0, float(self.get_parameter("steering_limit_deg").value))
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
        self._lookahead_speed_gain = max(0.0, float(self.get_parameter("lookahead_speed_gain").value))
        self._sharp_turn_lookahead_min_m = max(
            0.05, float(self.get_parameter("sharp_turn_lookahead_min_m").value)
        )
        self._lookahead_curvature_gain = max(
            0.0, float(self.get_parameter("lookahead_curvature_gain").value)
        )
        self._lookahead_curvature_window_points = max(
            1, int(self.get_parameter("lookahead_curvature_window_points").value)
        )
        self._min_linear_speed_mps = max(0.0, float(self.get_parameter("min_linear_speed_mps").value))
        self._max_linear_speed_mps = max(
            self._min_linear_speed_mps,
            float(self.get_parameter("max_linear_speed_mps").value),
        )
        self._sharp_turn_speed_boost_start_deg = max(
            0.0, float(self.get_parameter("sharp_turn_speed_boost_start_deg").value)
        )
        self._sharp_turn_speed_boost_full_deg = max(
            self._sharp_turn_speed_boost_start_deg + 1.0e-3,
            float(self.get_parameter("sharp_turn_speed_boost_full_deg").value),
        )
        self._sharp_turn_speed_boost_min_speed_mps = max(
            self._min_linear_speed_mps,
            float(self.get_parameter("sharp_turn_speed_boost_min_speed_mps").value),
        )
        self._curvature_speed_gain = max(0.0, float(self.get_parameter("curvature_speed_gain").value))
        self._max_lateral_accel_mps2 = max(
            1.0e-4, float(self.get_parameter("max_lateral_accel_mps2").value)
        )
        self._path_end_slowdown_distance_m = max(
            self._min_lookahead_m,
            float(self.get_parameter("path_end_slowdown_distance_m").value),
        )
        self._path_end_goal_tolerance_m = max(
            0.05, float(self.get_parameter("path_end_goal_tolerance_m").value)
        )
        self._path_end_yaw_tolerance_rad = max(
            0.0, float(self.get_parameter("path_end_yaw_tolerance_rad").value)
        )
        self._path_end_line_stop_margin_m = float(
            self.get_parameter("path_end_line_stop_margin_m").value
        )
        self._path_end_line_activation_tail_points = max(
            1, int(self.get_parameter("path_end_line_activation_tail_points").value)
        )
        self._max_path_deviation_m = max(
            0.05, float(self.get_parameter("max_path_deviation_m").value)
        )
        self._angular_cmd_ema_alpha = max(
            0.0, min(1.0, float(self.get_parameter("angular_cmd_ema_alpha").value))
        )
        self._curvature_deadband_m_inv = max(
            0.0, float(self.get_parameter("curvature_deadband_m_inv").value)
        )
        self._startup_ramp_duration_s = max(
            0.0, float(self.get_parameter("startup_ramp_duration_s").value)
        )
        self._startup_speed_scale_min = max(
            0.0, min(1.0, float(self.get_parameter("startup_speed_scale_min").value))
        )
        self._low_conf_abort_hold_s = max(
            0.1, float(self.get_parameter("low_confidence_abort_hold_s").value)
        )
        self._low_confidence_speed_scale = max(
            0.1, min(1.0, float(self.get_parameter("low_confidence_speed_scale").value))
        )
        self._path_stale_max_age_s = max(
            0.05, float(self.get_parameter("path_stale_max_age_s").value)
        )
        self._path_stale_abort_hold_s = max(
            self._path_stale_max_age_s,
            float(self.get_parameter("path_stale_abort_hold_s").value),
        )
        self._no_forward_target_abort_hold_s = max(
            0.05, float(self.get_parameter("no_forward_target_abort_hold_s").value)
        )
        self._waiting_path_refresh_hold_s = max(
            0.05, float(self.get_parameter("waiting_path_refresh_hold_s").value)
        )
        self._waiting_path_refresh_near_end_distance_m = max(
            self._path_end_goal_tolerance_m,
            float(self.get_parameter("waiting_path_refresh_near_end_distance_m").value),
        )
        self._path_projection_heading_tolerance_rad = max(
            0.1, float(self.get_parameter("path_projection_heading_tolerance_rad").value)
        )
        self._path_deviation_abort_margin_m = max(
            0.0, float(self.get_parameter("path_deviation_abort_margin_m").value)
        )
        self._path_refresh_min_forward_speed_mps = max(
            0.0, float(self.get_parameter("path_refresh_min_forward_speed_mps").value)
        )
        self._steering_saturation_start_ratio = max(
            0.1, min(1.0, float(self.get_parameter("steering_saturation_start_ratio").value))
        )
        self._steering_saturation_speed_scale_min = max(
            0.2, min(1.0, float(self.get_parameter("steering_saturation_speed_scale_min").value))
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
        self._path_curvature_m_inv: np.ndarray | None = None
        self._path_received_monotonic: float | None = None
        self._planning_status: dict[str, object] = {}
        self._fusion_status: dict[str, object] = {}
        self._latest_odom: dict[str, float] | None = None
        self._armed = False
        self._state = "waiting_path"
        self._terminal_cause: str | None = None
        self._tracking_started_monotonic: float | None = None
        self._low_conf_since_monotonic: float | None = None
        self._path_stale_since_monotonic: float | None = None
        self._no_forward_target_since_monotonic: float | None = None
        self._waiting_path_refresh_since_monotonic: float | None = None
        self._last_path_loss_reason: str | None = None
        self._filtered_angular_z_rps = 0.0
        self._status_payload: dict[str, object] = {
            "state": self._state,
            "terminal": False,
            "cause": None,
        }

        self.get_logger().info(
            "RecognitionTourTrackerNode started (path=%s odom=%s arm=%s cmd=%s)"
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
        path_points_xy = np.asarray(points, dtype=np.float64)
        path_yaw = np.asarray(yaw, dtype=np.float64)
        if path_points_xy.shape[0] < 2:
            return
        diffs = np.diff(path_points_xy, axis=0)
        path_s = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
        self._path_points_xy = path_points_xy
        self._path_yaw = path_yaw
        self._path_s = path_s
        self._path_curvature_m_inv = _estimate_path_curvature(path_points_xy)
        self._path_received_monotonic = time.monotonic()

    def _planning_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._planning_status = payload

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

    def _fusion_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if isinstance(payload, dict):
            self._fusion_status = payload

    def _arm_cb(self, msg: Bool) -> None:
        self._armed = bool(msg.data)

    def _publish_cmd(self, linear_x_mps: float, angular_z_rps: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_x_mps)
        cmd.angular.z = float(angular_z_rps)
        self._cmd_pub.publish(cmd)

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self._status_payload, separators=(",", ":"))
        self._status_pub.publish(msg)

    def _set_terminal(self, cause: str) -> None:
        if self._terminal_cause is not None:
            return
        self._terminal_cause = str(cause)
        self._state = self._terminal_cause

    def _fusion_confidence(self) -> str:
        latest_pose = self._fusion_status.get("latest_pose") or {}
        return str(latest_pose.get("confidence", "unknown"))

    def _tracking_allowed(self) -> bool:
        return bool(
            self._fusion_status.get("alignment_ready", False)
            and str(self._fusion_status.get("state", "")) == "tracking"
        )

    def _rear_axle_pose(self) -> tuple[np.ndarray, float]:
        if self._latest_odom is None:
            raise RuntimeError("missing odometry")
        yaw_rad = float(self._latest_odom["yaw_rad"])
        base_xy = np.asarray(
            [self._latest_odom["x_m"], self._latest_odom["y_m"]],
            dtype=np.float64,
        )
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        rear_xy = base_xy + np.asarray(
            [
                (cos_yaw * self._rear_axle_offset[0]) - (sin_yaw * self._rear_axle_offset[1]),
                (sin_yaw * self._rear_axle_offset[0]) + (cos_yaw * self._rear_axle_offset[1]),
            ],
            dtype=np.float64,
        )
        return rear_xy, yaw_rad

    def _startup_ramp_scale(self, now_monotonic: float) -> float:
        if self._tracking_started_monotonic is None or self._startup_ramp_duration_s <= 1.0e-6:
            return 1.0
        elapsed_s = max(0.0, now_monotonic - self._tracking_started_monotonic)
        ratio = min(1.0, elapsed_s / self._startup_ramp_duration_s)
        return self._startup_speed_scale_min + (
            ratio * (1.0 - self._startup_speed_scale_min)
        )

    def _path_age_s(self) -> float | None:
        planner_age_s: float | None = None
        value = self._planning_status.get("local_path_age_s")
        if value is not None:
            try:
                planner_age_s = max(0.0, float(value))
            except (TypeError, ValueError):
                planner_age_s = None

        received_age_s: float | None = None
        if self._path_received_monotonic is not None:
            received_age_s = max(0.0, time.monotonic() - self._path_received_monotonic)

        if planner_age_s is None:
            return received_age_s
        if received_age_s is None:
            return planner_age_s
        return max(planner_age_s, received_age_s)

    def _project_onto_path(self, point_xy: np.ndarray) -> dict[str, float | int | np.ndarray] | None:
        if self._path_points_xy is None or self._path_s is None or self._path_yaw is None:
            return None
        if self._path_points_xy.shape[0] < 2:
            return None

        seg_start_xy = self._path_points_xy[:-1]
        seg_end_xy = self._path_points_xy[1:]
        seg_vec_xy = seg_end_xy - seg_start_xy
        seg_len_sq = np.sum(seg_vec_xy * seg_vec_xy, axis=1)
        valid_mask = seg_len_sq > 1.0e-9
        if not np.any(valid_mask):
            return None

        rel_xy = point_xy.reshape(1, 2) - seg_start_xy
        projection_alpha = np.zeros_like(seg_len_sq, dtype=np.float64)
        projection_alpha[valid_mask] = np.clip(
            np.sum(rel_xy[valid_mask] * seg_vec_xy[valid_mask], axis=1) / seg_len_sq[valid_mask],
            0.0,
            1.0,
        )
        projection_xy = seg_start_xy + (projection_alpha.reshape(-1, 1) * seg_vec_xy)
        projection_distance = np.hypot(
            projection_xy[:, 0] - float(point_xy[0]),
            projection_xy[:, 1] - float(point_xy[1]),
        )
        projection_index = int(np.argmin(projection_distance))
        segment_heading_rad = math.atan2(
            float(seg_vec_xy[projection_index, 1]),
            float(seg_vec_xy[projection_index, 0]),
        )
        segment_length_m = math.sqrt(max(1.0e-12, float(seg_len_sq[projection_index])))
        projection_s_m = float(self._path_s[projection_index]) + (
            float(projection_alpha[projection_index]) * segment_length_m
        )
        closest_vertex_index = int(
            min(
                self._path_points_xy.shape[0] - 1,
                projection_index + (1 if projection_alpha[projection_index] >= 0.5 else 0),
            )
        )
        return {
            "segment_index": projection_index,
            "closest_vertex_index": closest_vertex_index,
            "projection_xy": projection_xy[projection_index].copy(),
            "projection_s_m": projection_s_m,
            "distance_m": float(projection_distance[projection_index]),
            "heading_rad": float(segment_heading_rad),
        }

    def _status_waiting_path_refresh(
        self,
        *,
        now_monotonic: float,
        planner_state: str,
        path_age_s: float | None,
        odom_age_s: float | None,
        path_is_stale: bool,
        path_deviation_m: float,
        distance_to_path_end_m: float,
        path_projection_valid: bool,
        path_projection_s_m: float,
        path_heading_error_rad: float,
        path_loss_reason: str,
        linear_x_mps: float,
        angular_z_rps: float,
    ) -> bool:
        planner_ready = bool(self._planning_status.get("ready", False))
        recoverable = (
            planner_ready
            and not path_is_stale
            and path_projection_valid
            and distance_to_path_end_m <= self._waiting_path_refresh_near_end_distance_m
            and path_heading_error_rad <= self._path_projection_heading_tolerance_rad
        )
        if not recoverable:
            self._waiting_path_refresh_since_monotonic = None
            self._last_path_loss_reason = path_loss_reason
            return False

        if self._waiting_path_refresh_since_monotonic is None:
            self._waiting_path_refresh_since_monotonic = now_monotonic
        waiting_elapsed_s = now_monotonic - self._waiting_path_refresh_since_monotonic
        if waiting_elapsed_s >= self._waiting_path_refresh_hold_s:
            self._waiting_path_refresh_since_monotonic = None
            self._last_path_loss_reason = path_loss_reason
            return False

        self._state = "waiting_path_refresh"
        self._last_path_loss_reason = path_loss_reason
        self._status_payload = {
            "state": self._state,
            "terminal": False,
            "cause": None,
            "planner_state": planner_state,
            "planner_ready": planner_ready,
            "path_age_s": path_age_s,
            "odom_age_s": odom_age_s,
            "path_stale": path_is_stale,
            "path_deviation_m": path_deviation_m,
            "distance_to_path_end_m": distance_to_path_end_m,
            "path_projection_valid": path_projection_valid,
            "path_projection_s_m": path_projection_s_m,
            "path_loss_reason": path_loss_reason,
            "waiting_path_refresh_active": True,
            "waiting_path_refresh_elapsed_s": waiting_elapsed_s,
            "waiting_path_refresh_hold_s": self._waiting_path_refresh_hold_s,
            "path_heading_error_rad": path_heading_error_rad,
        }
        self._publish_cmd(linear_x_mps, angular_z_rps)
        self._publish_status()
        return True

    def _control_step(self) -> None:
        now_monotonic = time.monotonic()

        if self._path_points_xy is None or self._path_s is None or self._path_yaw is None:
            self._no_forward_target_since_monotonic = None
            self._waiting_path_refresh_since_monotonic = None
            self._state = "waiting_path"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
            }
            self._publish_cmd(0.0, 0.0)
            self._filtered_angular_z_rps = 0.0
            self._publish_status()
            return

        planner_state = str(self._planning_status.get("state", ""))
        if planner_state in PLANNER_TERMINAL_STATES:
            self._set_terminal(planner_state)
        elif planner_state in PLANNER_FAILURE_STATES:
            self._set_terminal("planner_failed")

        if not self._tracking_allowed():
            self._no_forward_target_since_monotonic = None
            self._waiting_path_refresh_since_monotonic = None
            self._state = "waiting_fusion"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
                "fusion_state": self._fusion_status.get("state"),
                "armed": self._armed,
            }
            self._publish_cmd(0.0, 0.0)
            self._filtered_angular_z_rps = 0.0
            self._publish_status()
            return

        if not self._armed:
            self._no_forward_target_since_monotonic = None
            self._waiting_path_refresh_since_monotonic = None
            self._state = "waiting_arm"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
                "planner_state": planner_state,
                "planner_ready": bool(self._planning_status.get("ready", False)),
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
            self._filtered_angular_z_rps = 0.0
        elif (now_monotonic - self._tracking_started_monotonic) >= self._global_timeout_s:
            self._set_terminal("timeout")

        odom_age_s: float | None = None
        if self._latest_odom is None:
            self._set_terminal("aborted_odom_timeout")
        else:
            odom_age_s = max(0.0, time.time() - float(self._latest_odom["stamp_s"]))
            if odom_age_s > self._odom_timeout_s:
                self._set_terminal("aborted_odom_timeout")

        path_age_s = self._path_age_s()
        path_is_stale = path_age_s is not None and path_age_s > self._path_stale_max_age_s
        if path_is_stale:
            if self._path_stale_since_monotonic is None:
                self._path_stale_since_monotonic = now_monotonic
            elif (now_monotonic - self._path_stale_since_monotonic) >= self._path_stale_abort_hold_s:
                self._set_terminal("aborted_path_loss")
        else:
            self._path_stale_since_monotonic = None

        if self._terminal_cause is not None:
            self._no_forward_target_since_monotonic = None
            self._waiting_path_refresh_since_monotonic = None
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "planner_state": planner_state,
                "path_age_s": path_age_s,
                "odom_age_s": odom_age_s,
                "path_loss_reason": self._last_path_loss_reason,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        rear_xy, rear_yaw = self._rear_axle_pose()
        projection = self._project_onto_path(rear_xy)
        if projection is None:
            self._last_path_loss_reason = "projection_invalid"
            self._set_terminal("aborted_path_loss")
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "planner_state": planner_state,
                "odom_age_s": odom_age_s,
                "path_projection_valid": False,
                "path_loss_reason": self._last_path_loss_reason,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        nearest_index = int(projection["closest_vertex_index"])
        path_deviation_m = float(projection["distance_m"])
        path_projection_s_m = float(projection["projection_s_m"])
        path_projection_valid = True
        path_projection_heading_rad = float(projection["heading_rad"])
        path_heading_error_rad = abs(_normalize_angle(path_projection_heading_rad - float(rear_yaw)))
        distance_to_path_end_m = max(0.0, float(self._path_s[-1] - path_projection_s_m))
        path_deviation_margin_m = self._max_path_deviation_m + self._path_deviation_abort_margin_m

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
            nearest_index >= max(
                0,
                self._path_points_xy.shape[0] - self._path_end_line_activation_tail_points,
            )
            and goal_line_projection_m >= self._path_end_line_stop_margin_m
        )
        goal_distance_m = float(np.linalg.norm(goal_point - rear_xy))
        goal_yaw_error_rad = abs(_normalize_angle(goal_yaw_rad - float(rear_yaw)))
        goal_line_alignment_ready = goal_yaw_error_rad <= self._path_end_yaw_tolerance_rad
        remaining_path_m = distance_to_path_end_m
        local_path_goal_ready = bool(
            (goal_line_crossed and goal_line_alignment_ready)
            or (
                goal_distance_m <= self._path_end_goal_tolerance_m
                and goal_yaw_error_rad <= self._path_end_yaw_tolerance_rad
            )
        )

        if local_path_goal_ready and self._terminal_cause is None:
            self._no_forward_target_since_monotonic = None
            self._waiting_path_refresh_since_monotonic = None
            self._state = "waiting_path_refresh"
            self._status_payload = {
                "state": self._state,
                "terminal": False,
                "cause": None,
                "planner_state": planner_state,
                "planner_ready": bool(self._planning_status.get("ready", False)),
                "path_age_s": path_age_s,
                "odom_age_s": odom_age_s,
                "path_stale": path_is_stale,
                "path_deviation_m": path_deviation_m,
                "goal_distance_m": goal_distance_m,
                "remaining_path_m": remaining_path_m,
                "goal_yaw_error_rad": goal_yaw_error_rad,
                "goal_line_projection_m": goal_line_projection_m,
                "goal_line_lateral_error_m": goal_line_lateral_error_m,
                "goal_line_crossed": goal_line_crossed,
                "goal_line_alignment_ready": goal_line_alignment_ready,
                "path_projection_valid": path_projection_valid,
                "path_projection_s_m": path_projection_s_m,
                "distance_to_path_end_m": distance_to_path_end_m,
                "path_loss_reason": None,
                "waiting_path_refresh_active": True,
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        speed_now_mps = math.hypot(
            float(self._latest_odom["vx_mps"]),
            float(self._latest_odom["vy_mps"]),
        )
        base_lookahead_m = min(
            self._max_lookahead_m,
            max(
                self._min_lookahead_m,
                self._min_lookahead_m + (self._lookahead_speed_gain * speed_now_mps),
            ),
        )
        ahead_path_curvature_m_inv = 0.0
        if self._path_curvature_m_inv is not None and self._path_curvature_m_inv.size > 0:
            curvature_window_end = min(
                self._path_curvature_m_inv.shape[0],
                nearest_index + self._lookahead_curvature_window_points,
            )
            curvature_window = np.abs(
                self._path_curvature_m_inv[nearest_index:curvature_window_end]
            )
            if curvature_window.size > 0:
                ahead_path_curvature_m_inv = float(np.max(curvature_window))
        lookahead_scale = 1.0 / (
            1.0 + (self._lookahead_curvature_gain * ahead_path_curvature_m_inv)
        )
        lookahead_m = min(
            self._max_lookahead_m,
            max(
                self._sharp_turn_lookahead_min_m,
                base_lookahead_m * lookahead_scale,
            ),
        )
        target_s = float(path_projection_s_m + lookahead_m)
        target_index = int(np.searchsorted(self._path_s, target_s, side="left"))
        target_index = min(target_index, self._path_points_xy.shape[0] - 1)

        target_xy = self._path_points_xy[target_index]
        dx_world = float(target_xy[0] - rear_xy[0])
        dy_world = float(target_xy[1] - rear_xy[1])
        cos_yaw = math.cos(rear_yaw)
        sin_yaw = math.sin(rear_yaw)
        dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
        dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)

        while dx_local <= 1.0e-3 and target_index < (self._path_points_xy.shape[0] - 1):
            target_index += 1
            target_xy = self._path_points_xy[target_index]
            dx_world = float(target_xy[0] - rear_xy[0])
            dy_world = float(target_xy[1] - rear_xy[1])
            dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
            dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)

        if dx_local <= 1.0e-3:
            if path_is_stale:
                self._no_forward_target_since_monotonic = None
                self._waiting_path_refresh_since_monotonic = None
                self._state = "holding_last_path"
                self._status_payload = {
                    "state": self._state,
                    "terminal": False,
                    "cause": None,
                    "planner_state": planner_state,
                    "path_age_s": path_age_s,
                    "odom_age_s": odom_age_s,
                    "path_stale": True,
                }
                self._publish_cmd(0.0, 0.0)
                self._publish_status()
                return
            recovery_linear_x_mps = min(
                0.10,
                max(self._path_refresh_min_forward_speed_mps, 0.75 * self._min_linear_speed_mps),
            )
            if self._status_waiting_path_refresh(
                now_monotonic=now_monotonic,
                planner_state=planner_state,
                path_age_s=path_age_s,
                odom_age_s=odom_age_s,
                path_is_stale=path_is_stale,
                path_deviation_m=path_deviation_m,
                distance_to_path_end_m=distance_to_path_end_m,
                path_projection_valid=path_projection_valid,
                path_projection_s_m=path_projection_s_m,
                path_heading_error_rad=path_heading_error_rad,
                path_loss_reason="projection_behind_target",
                linear_x_mps=recovery_linear_x_mps,
                angular_z_rps=0.0,
            ):
                return
            if self._no_forward_target_since_monotonic is None:
                self._no_forward_target_since_monotonic = now_monotonic
            no_forward_target_elapsed_s = (
                now_monotonic - self._no_forward_target_since_monotonic
            )
            planner_ready = bool(self._planning_status.get("ready", False))
            if planner_ready and no_forward_target_elapsed_s < self._no_forward_target_abort_hold_s:
                self._state = "waiting_forward_path"
                self._status_payload = {
                    "state": self._state,
                    "terminal": False,
                    "cause": None,
                    "planner_state": planner_state,
                    "planner_ready": planner_ready,
                    "path_age_s": path_age_s,
                    "odom_age_s": odom_age_s,
                    "path_stale": False,
                    "no_forward_target_elapsed_s": no_forward_target_elapsed_s,
                    "no_forward_target_abort_hold_s": self._no_forward_target_abort_hold_s,
                    "path_deviation_m": path_deviation_m,
                }
                self._publish_cmd(0.0, 0.0)
                self._publish_status()
                return
            self._no_forward_target_since_monotonic = None
            self._set_terminal("aborted_path_loss")
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "planner_state": planner_state,
                "path_age_s": path_age_s,
                "odom_age_s": odom_age_s,
                "path_deviation_m": path_deviation_m,
                "path_projection_valid": path_projection_valid,
                "path_projection_s_m": path_projection_s_m,
                "distance_to_path_end_m": distance_to_path_end_m,
                "path_loss_reason": "projection_behind_target",
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return

        self._no_forward_target_since_monotonic = None
        self._waiting_path_refresh_since_monotonic = None
        lookahead_actual_m = max(self._sharp_turn_lookahead_min_m, math.hypot(dx_local, dy_local))
        raw_curvature = (2.0 * dy_local) / max(1.0e-6, lookahead_actual_m * lookahead_actual_m)
        max_track_curvature_m_inv = math.tan(math.radians(self._steering_limit_deg)) / self._wheelbase_m
        desired_steering_deg = math.degrees(math.atan(self._wheelbase_m * raw_curvature))
        steering_saturation_ratio = abs(desired_steering_deg) / max(1.0e-6, self._steering_limit_deg)
        if steering_saturation_ratio > self._steering_saturation_start_ratio:
            saturation_alpha = min(
                1.0,
                (
                    (steering_saturation_ratio - self._steering_saturation_start_ratio)
                    / max(1.0e-6, 1.0 - self._steering_saturation_start_ratio)
                ),
            )
            saturation_lookahead_m = max(
                self._sharp_turn_lookahead_min_m,
                lookahead_actual_m * (1.0 - (0.35 * saturation_alpha)),
            )
            saturation_target_s = float(path_projection_s_m + saturation_lookahead_m)
            saturation_target_index = int(np.searchsorted(self._path_s, saturation_target_s, side="left"))
            saturation_target_index = min(saturation_target_index, self._path_points_xy.shape[0] - 1)
            saturation_target_xy = self._path_points_xy[saturation_target_index]
            dx_world = float(saturation_target_xy[0] - rear_xy[0])
            dy_world = float(saturation_target_xy[1] - rear_xy[1])
            dx_local = (cos_yaw * dx_world) + (sin_yaw * dy_world)
            dy_local = (-sin_yaw * dx_world) + (cos_yaw * dy_world)
            if dx_local > 1.0e-3:
                target_index = saturation_target_index
                target_xy = saturation_target_xy
                lookahead_actual_m = max(
                    self._sharp_turn_lookahead_min_m,
                    math.hypot(dx_local, dy_local),
                )
                raw_curvature = (2.0 * dy_local) / max(
                    1.0e-6, lookahead_actual_m * lookahead_actual_m
                )
                desired_steering_deg = math.degrees(math.atan(self._wheelbase_m * raw_curvature))
                steering_saturation_ratio = abs(desired_steering_deg) / max(
                    1.0e-6, self._steering_limit_deg
                )

        curvature = raw_curvature
        if abs(curvature) < self._curvature_deadband_m_inv:
            curvature = 0.0

        curvature_abs = abs(raw_curvature)
        desired_steering_abs_deg = abs(desired_steering_deg)
        steering_saturated = desired_steering_abs_deg > (self._steering_limit_deg + 1.0e-3)
        commanded_curvature = float(
            np.clip(curvature, -max_track_curvature_m_inv, max_track_curvature_m_inv)
        )
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
        if goal_distance_m < self._path_end_slowdown_distance_m:
            goal_speed_limit_mps = self._max_linear_speed_mps * max(
                0.15, goal_distance_m / self._path_end_slowdown_distance_m
            )
            linear_x_mps = min(linear_x_mps, goal_speed_limit_mps)
        linear_x_mps *= self._startup_ramp_scale(now_monotonic)
        if confidence != "high":
            linear_x_mps *= self._low_confidence_speed_scale
        min_linear_speed_floor_mps = self._min_linear_speed_mps * self._startup_ramp_scale(
            now_monotonic
        )
        sharp_turn_speed_floor_mps = 0.0
        if (
            confidence == "high"
            and goal_distance_m > self._path_end_slowdown_distance_m
            and desired_steering_abs_deg > self._sharp_turn_speed_boost_start_deg
        ):
            steer_alpha = min(
                1.0,
                max(
                    0.0,
                    (
                        (desired_steering_abs_deg - self._sharp_turn_speed_boost_start_deg)
                        / (
                            self._sharp_turn_speed_boost_full_deg
                            - self._sharp_turn_speed_boost_start_deg
                        )
                    ),
                ),
            )
            sharp_turn_speed_floor_mps = self._startup_ramp_scale(now_monotonic) * (
                self._min_linear_speed_mps
                + (
                    steer_alpha
                    * (self._sharp_turn_speed_boost_min_speed_mps - self._min_linear_speed_mps)
                )
            )
            min_linear_speed_floor_mps = max(
                min_linear_speed_floor_mps,
                sharp_turn_speed_floor_mps,
            )
        if remaining_path_m <= self._path_end_goal_tolerance_m:
            min_linear_speed_floor_mps = 0.0
        linear_x_mps = max(min_linear_speed_floor_mps, min(self._max_linear_speed_mps, linear_x_mps))
        if steering_saturation_ratio > self._steering_saturation_start_ratio:
            saturation_alpha = min(
                1.0,
                (
                    (steering_saturation_ratio - self._steering_saturation_start_ratio)
                    / max(1.0e-6, 1.5 - self._steering_saturation_start_ratio)
                ),
            )
            saturation_speed_scale = 1.0 - (
                saturation_alpha * (1.0 - self._steering_saturation_speed_scale_min)
            )
            linear_x_mps = max(
                min_linear_speed_floor_mps,
                linear_x_mps * saturation_speed_scale,
            )

        raw_angular_z_rps = linear_x_mps * commanded_curvature
        if linear_x_mps <= 1.0e-6 or abs(raw_angular_z_rps) <= 1.0e-6:
            angular_z_rps = raw_angular_z_rps
            self._filtered_angular_z_rps = raw_angular_z_rps
        elif abs(self._filtered_angular_z_rps) <= 1.0e-6:
            angular_z_rps = raw_angular_z_rps
            self._filtered_angular_z_rps = raw_angular_z_rps
        else:
            angular_z_rps = (
                (self._angular_cmd_ema_alpha * raw_angular_z_rps)
                + ((1.0 - self._angular_cmd_ema_alpha) * self._filtered_angular_z_rps)
            )
            self._filtered_angular_z_rps = angular_z_rps

        filtered_curvature_m_inv = (
            angular_z_rps / linear_x_mps if abs(linear_x_mps) > 1.0e-6 else 0.0
        )
        self._state = "tracking" if not path_is_stale else "holding_last_path"
        self._status_payload = {
            "state": self._state,
            "terminal": False,
            "cause": None,
            "armed": True,
            "tracking_started": True,
            "elapsed_s": now_monotonic - float(self._tracking_started_monotonic or now_monotonic),
            "planner_state": planner_state,
            "planner_ready": bool(self._planning_status.get("ready", False)),
            "path_age_s": path_age_s,
            "odom_age_s": odom_age_s,
            "path_stale": path_is_stale,
            "fusion_state": self._fusion_status.get("state"),
            "fusion_confidence": confidence,
            "closest_path_index": nearest_index,
            "target_path_index": target_index,
            "path_deviation_m": path_deviation_m,
            "path_projection_valid": path_projection_valid,
            "path_projection_s_m": path_projection_s_m,
            "distance_to_path_end_m": distance_to_path_end_m,
            "path_loss_reason": None,
            "waiting_path_refresh_active": False,
            "goal_distance_m": goal_distance_m,
            "remaining_path_m": remaining_path_m,
            "goal_yaw_error_rad": goal_yaw_error_rad,
            "goal_line_projection_m": goal_line_projection_m,
            "goal_line_lateral_error_m": goal_line_lateral_error_m,
            "goal_line_crossed": goal_line_crossed,
            "goal_line_alignment_ready": goal_line_alignment_ready,
            "base_lookahead_m": base_lookahead_m,
            "ahead_path_curvature_m_inv": ahead_path_curvature_m_inv,
            "lookahead_m": lookahead_actual_m,
            "raw_curvature_m_inv": raw_curvature,
            "curvature_m_inv": curvature,
            "commanded_curvature_m_inv": commanded_curvature,
            "filtered_curvature_m_inv": filtered_curvature_m_inv,
            "desired_steering_deg": desired_steering_deg,
            "steering_saturated": steering_saturated,
            "steering_saturation_ratio": steering_saturation_ratio,
            "path_projection_heading_error_rad": path_heading_error_rad,
            "curvature_deadband_m_inv": self._curvature_deadband_m_inv,
            "curvature_speed_limit_mps": curvature_speed_limit_mps,
            "lateral_accel_speed_limit_mps": lateral_accel_speed_limit_mps,
            "goal_speed_limit_mps": goal_speed_limit_mps,
            "sharp_turn_speed_floor_mps": sharp_turn_speed_floor_mps,
            "cmd_linear_x_mps": linear_x_mps,
            "raw_cmd_angular_z_rps": raw_angular_z_rps,
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
        if path_deviation_m > path_deviation_margin_m:
            recovery_linear_x_mps = min(
                0.10,
                max(self._path_refresh_min_forward_speed_mps, 0.75 * self._min_linear_speed_mps),
            )
            if self._status_waiting_path_refresh(
                now_monotonic=now_monotonic,
                planner_state=planner_state,
                path_age_s=path_age_s,
                odom_age_s=odom_age_s,
                path_is_stale=path_is_stale,
                path_deviation_m=path_deviation_m,
                distance_to_path_end_m=distance_to_path_end_m,
                path_projection_valid=path_projection_valid,
                path_projection_s_m=path_projection_s_m,
                path_heading_error_rad=path_heading_error_rad,
                path_loss_reason="path_deviation_margin_exceeded",
                linear_x_mps=recovery_linear_x_mps,
                angular_z_rps=0.0,
            ):
                return
            self._set_terminal("aborted_path_loss")
            self._status_payload = {
                "state": self._state,
                "terminal": True,
                "cause": self._terminal_cause,
                "planner_state": planner_state,
                "path_age_s": path_age_s,
                "odom_age_s": odom_age_s,
                "path_deviation_m": path_deviation_m,
                "path_projection_valid": path_projection_valid,
                "path_projection_s_m": path_projection_s_m,
                "distance_to_path_end_m": distance_to_path_end_m,
                "path_loss_reason": "path_deviation_margin_exceeded",
            }
            self._publish_cmd(0.0, 0.0)
            self._publish_status()
            return
        self._publish_cmd(linear_x_mps, angular_z_rps)
        self._publish_status()


def main() -> None:
    rclpy.init()
    node = RecognitionTourTrackerNode()
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
