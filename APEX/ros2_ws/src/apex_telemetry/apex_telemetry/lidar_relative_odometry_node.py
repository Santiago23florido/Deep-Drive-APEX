#!/usr/bin/env python3
"""Lightweight 2D LiDAR-relative odometry for short planar motions."""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _rotate_planar(x: float, y: float, yaw_rad: float) -> tuple[float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return ((c * x) - (s * y), (s * x) + (c * y))


class LidarRelativeOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_relative_odometry_node")

        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/lidar/relative_odom")
        self.declare_parameter("pose_topic", "/apex/lidar/pose_local")
        self.declare_parameter("status_topic", "/apex/kinematics/status")
        self.declare_parameter("odom_frame_id", "odom_lidar_local")
        self.declare_parameter("child_frame_id", "laser")
        self.declare_parameter("min_usable_range_m", 0.08)
        self.declare_parameter("max_usable_range_m", 8.0)
        self.declare_parameter("max_points", 160)
        self.declare_parameter("max_iterations", 12)
        self.declare_parameter("max_correspondence_distance_m", 0.12)
        self.declare_parameter("max_step_translation_m", 0.08)
        self.declare_parameter("max_step_yaw_rad", 0.06)
        self.declare_parameter("min_valid_points", 40)
        self.declare_parameter("min_match_count", 24)
        self.declare_parameter("min_match_ratio", 0.24)
        self.declare_parameter("max_rmse_m", 0.045)
        self.declare_parameter("quality_error_scale_m", 0.024)
        self.declare_parameter("min_quality", 0.20)
        self.declare_parameter("min_translation_observable_m", 0.012)
        self.declare_parameter("min_rotation_observable_rad", 0.01)
        self.declare_parameter("min_consecutive_observable_updates", 2)
        self.declare_parameter("status_timeout_s", 0.75)
        self.declare_parameter("status_motion_release_accel_mps2", 0.08)
        self.declare_parameter("status_motion_release_speed_mps", 0.04)
        self.declare_parameter("status_motion_release_yaw_rate_rps", 0.03)
        self.declare_parameter("nominal_covariance_x_m2", 0.01)
        self.declare_parameter("nominal_covariance_y_m2", 0.01)
        self.declare_parameter("nominal_covariance_yaw_rad2", 0.04)
        self.declare_parameter("invalid_covariance_scale", 50.0)
        self.declare_parameter("stale_twist_covariance_scale", 25.0)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._child_frame = str(self.get_parameter("child_frame_id").value)
        self._min_usable_range = max(0.01, float(self.get_parameter("min_usable_range_m").value))
        self._max_usable_range = max(
            self._min_usable_range + 0.1,
            float(self.get_parameter("max_usable_range_m").value),
        )
        self._max_points = max(48, int(self.get_parameter("max_points").value))
        self._max_iterations = max(1, int(self.get_parameter("max_iterations").value))
        self._max_corr_distance = max(
            0.01, float(self.get_parameter("max_correspondence_distance_m").value)
        )
        self._max_step_translation = max(
            0.01, float(self.get_parameter("max_step_translation_m").value)
        )
        self._max_step_yaw = max(0.01, float(self.get_parameter("max_step_yaw_rad").value))
        self._min_valid_points = max(12, int(self.get_parameter("min_valid_points").value))
        self._min_match_count = max(8, int(self.get_parameter("min_match_count").value))
        self._min_match_ratio = _clamp(
            float(self.get_parameter("min_match_ratio").value), 0.0, 1.0
        )
        self._max_rmse = max(1e-3, float(self.get_parameter("max_rmse_m").value))
        self._quality_error_scale = max(
            1e-3, float(self.get_parameter("quality_error_scale_m").value)
        )
        self._min_quality = _clamp(float(self.get_parameter("min_quality").value), 0.0, 1.0)
        self._min_translation_observable = max(
            0.0, float(self.get_parameter("min_translation_observable_m").value)
        )
        self._min_rotation_observable = max(
            0.0, float(self.get_parameter("min_rotation_observable_rad").value)
        )
        self._min_consecutive_observable_updates = max(
            1, int(self.get_parameter("min_consecutive_observable_updates").value)
        )
        self._status_timeout_s = max(0.1, float(self.get_parameter("status_timeout_s").value))
        self._status_motion_release_accel_mps2 = max(
            0.0, float(self.get_parameter("status_motion_release_accel_mps2").value)
        )
        self._status_motion_release_speed_mps = max(
            0.0, float(self.get_parameter("status_motion_release_speed_mps").value)
        )
        self._status_motion_release_yaw_rate_rps = max(
            0.0, float(self.get_parameter("status_motion_release_yaw_rate_rps").value)
        )
        self._nominal_cov_x = max(
            1e-6, float(self.get_parameter("nominal_covariance_x_m2").value)
        )
        self._nominal_cov_y = max(
            1e-6, float(self.get_parameter("nominal_covariance_y_m2").value)
        )
        self._nominal_cov_yaw = max(
            1e-6, float(self.get_parameter("nominal_covariance_yaw_rad2").value)
        )
        self._invalid_cov_scale = max(
            1.0, float(self.get_parameter("invalid_covariance_scale").value)
        )
        self._stale_twist_cov_scale = max(
            1.0, float(self.get_parameter("stale_twist_covariance_scale").value)
        )

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, self._pose_topic, 20)
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(String, self._status_topic, self._status_cb, 20)

        self._prev_points: Optional[np.ndarray] = None
        self._prev_stamp = None
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_step_translation = np.zeros(2, dtype=np.float64)
        self._last_step_yaw = 0.0
        self._last_quality = 0.0
        self._last_valid = False
        self._last_translation_observable = False
        self._last_match_ratio = 0.0
        self._last_rmse_m = 0.0
        self._last_delta_m = 0.0
        self._last_delta_yaw = 0.0
        self._observable_streak = 0
        self._calibration_active = False
        self._stationary_detected = False
        self._status_corrected_accel_planar_mps2 = 0.0
        self._status_speed_mps = 0.0
        self._status_yaw_rate_rps = 0.0
        self._last_status_receipt = None

        self.get_logger().info(
            "LidarRelativeOdometryNode started (scan=%s odom=%s pose=%s status=%s max_points=%d)"
            % (
                self._scan_topic,
                self._odom_topic,
                self._pose_topic,
                self._status_topic,
                self._max_points,
            )
        )

    def _reset_relative_state(self, reason: str) -> None:
        self._prev_points = None
        self._prev_stamp = None
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_step_translation = np.zeros(2, dtype=np.float64)
        self._last_step_yaw = 0.0
        self._last_quality = 0.0
        self._last_valid = False
        self._last_translation_observable = False
        self._last_match_ratio = 0.0
        self._last_rmse_m = 0.0
        self._last_delta_m = 0.0
        self._last_delta_yaw = 0.0
        self._observable_streak = 0
        self.get_logger().info("LiDAR relative odom reset (%s)." % reason)

    def _status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        prev_calibration_active = self._calibration_active
        self._last_status_receipt = self.get_clock().now()
        self._calibration_active = bool(payload.get("calibration_active", False))
        self._stationary_detected = bool(payload.get("stationary_detected", False))
        self._status_corrected_accel_planar_mps2 = max(
            0.0, float(payload.get("corrected_accel_planar_mps2", 0.0) or 0.0)
        )
        self._status_speed_mps = max(0.0, float(payload.get("speed_mps", 0.0) or 0.0))
        self._status_yaw_rate_rps = abs(float(payload.get("yaw_rate_rps", 0.0) or 0.0))
        if self._calibration_active and not prev_calibration_active:
            self._reset_relative_state("calibration_active")

    def _status_is_fresh(self) -> bool:
        if self._last_status_receipt is None:
            return False
        age_s = max(
            0.0, (self.get_clock().now() - self._last_status_receipt).nanoseconds * 1e-9
        )
        return age_s <= self._status_timeout_s

    def _motion_override_active(self) -> bool:
        if not self._status_is_fresh():
            return False
        return (
            self._status_corrected_accel_planar_mps2 >= self._status_motion_release_accel_mps2
            or self._status_speed_mps >= self._status_motion_release_speed_mps
            or self._status_yaw_rate_rps >= self._status_motion_release_yaw_rate_rps
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        points = self._scan_to_points(msg)
        stamp = msg.header.stamp

        if self._calibration_active:
            self._prev_points = None
            self._prev_stamp = None
            self._publish_outputs(
                stamp,
                valid=False,
                quality=0.0,
                translation_observable=False,
                step_translation=np.zeros(2, dtype=np.float64),
                step_yaw=0.0,
                dt_s=0.0,
                rmse_m=float("inf"),
                match_ratio=0.0,
            )
            return

        if points.shape[0] < self._min_valid_points:
            self._publish_outputs(
                stamp,
                valid=False,
                quality=0.0,
                translation_observable=False,
                step_translation=np.zeros(2, dtype=np.float64),
                step_yaw=0.0,
                dt_s=0.0,
                rmse_m=float("inf"),
                match_ratio=0.0,
            )
            self._prev_points = points if points.size else None
            self._prev_stamp = stamp
            return

        if self._prev_points is None or self._prev_stamp is None:
            self._prev_points = points
            self._prev_stamp = stamp
            self._publish_outputs(
                stamp,
                valid=False,
                quality=0.0,
                translation_observable=False,
                step_translation=np.zeros(2, dtype=np.float64),
                step_yaw=0.0,
                dt_s=0.0,
                rmse_m=float("inf"),
                match_ratio=0.0,
            )
            return

        prev_time_s = float(self._prev_stamp.sec) + (float(self._prev_stamp.nanosec) * 1e-9)
        curr_time_s = float(stamp.sec) + (float(stamp.nanosec) * 1e-9)
        dt_s = max(0.0, curr_time_s - prev_time_s)

        valid, quality, step_translation, step_yaw, rmse_m, match_ratio = self._estimate_step(
            source_points=points,
            target_points=self._prev_points,
        )
        translation_observable = valid and (
            float(np.linalg.norm(step_translation)) >= self._min_translation_observable
            or abs(step_yaw) >= self._min_rotation_observable
        )
        motion_allowed = True
        if self._status_is_fresh() and self._stationary_detected and not self._motion_override_active():
            motion_allowed = False
        if translation_observable and motion_allowed:
            self._observable_streak += 1
        else:
            self._observable_streak = 0
        accepted = valid and translation_observable and motion_allowed
        accepted = accepted and self._observable_streak >= self._min_consecutive_observable_updates

        if accepted:
            world_dx, world_dy = _rotate_planar(
                float(step_translation[0]),
                float(step_translation[1]),
                self._yaw,
            )
            self._x += world_dx
            self._y += world_dy
            self._yaw = _normalize_angle(self._yaw + step_yaw)
            self._last_step_translation = step_translation
            self._last_step_yaw = step_yaw
        else:
            self._last_step_translation = np.zeros(2, dtype=np.float64)
            self._last_step_yaw = 0.0
            step_translation = np.zeros(2, dtype=np.float64)
            step_yaw = 0.0
            valid = False

        self._publish_outputs(
            stamp,
            valid=valid,
            quality=quality,
            translation_observable=translation_observable,
            step_translation=step_translation,
            step_yaw=step_yaw,
            dt_s=dt_s,
            rmse_m=rmse_m,
            match_ratio=match_ratio,
        )

        self._prev_points = points
        self._prev_stamp = stamp

    def _scan_to_points(self, msg: LaserScan) -> np.ndarray:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        angles = msg.angle_min + (np.arange(ranges.size, dtype=np.float64) * msg.angle_increment)
        max_range = min(float(msg.range_max), self._max_usable_range)
        min_range = max(float(msg.range_min), self._min_usable_range)
        valid = np.isfinite(ranges)
        valid &= ranges >= min_range
        valid &= ranges <= max_range
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)

        ranges = ranges[valid].astype(np.float64)
        angles = angles[valid]
        points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        if points.shape[0] > self._max_points:
            indices = np.linspace(0, points.shape[0] - 1, self._max_points, dtype=np.int32)
            points = points[indices]
        return points

    def _estimate_step(
        self,
        *,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> tuple[bool, float, np.ndarray, float, float, float]:
        if (
            source_points.shape[0] < self._min_valid_points
            or target_points.shape[0] < self._min_valid_points
        ):
            return False, 0.0, np.zeros(2, dtype=np.float64), 0.0, float("inf"), 0.0

        estimate_translation = self._last_step_translation.copy()
        estimate_yaw = float(self._last_step_yaw)

        best_rmse = float("inf")
        best_match_ratio = 0.0

        for _ in range(self._max_iterations):
            transformed = self._apply_transform(source_points, estimate_translation, estimate_yaw)
            distances, indices = self._nearest_neighbor(transformed, target_points)
            match_mask = distances <= self._max_corr_distance
            match_count = int(np.count_nonzero(match_mask))
            if match_count < self._min_match_count:
                return False, 0.0, np.zeros(2, dtype=np.float64), 0.0, float("inf"), 0.0

            matched_source = transformed[match_mask]
            matched_target = target_points[indices[match_mask]]
            step_translation, step_yaw = self._best_fit_transform(matched_source, matched_target)

            estimate_translation = self._compose_translation(
                step_translation,
                estimate_translation,
                step_yaw,
            )
            estimate_yaw = _normalize_angle(estimate_yaw + step_yaw)

            best_rmse = float(np.sqrt(np.mean(np.square(distances[match_mask]))))
            best_match_ratio = match_count / float(max(1, min(source_points.shape[0], target_points.shape[0])))

            if (
                float(np.linalg.norm(step_translation)) < 5e-4
                and abs(step_yaw) < 5e-4
            ):
                break

        step_translation = estimate_translation
        step_yaw = estimate_yaw
        step_delta_m = float(np.linalg.norm(step_translation))
        if (
            not np.isfinite(step_delta_m)
            or not np.isfinite(step_yaw)
            or step_delta_m > self._max_step_translation
            or abs(step_yaw) > self._max_step_yaw
            or best_match_ratio < self._min_match_ratio
            or best_rmse > self._max_rmse
        ):
            return False, 0.0, np.zeros(2, dtype=np.float64), 0.0, best_rmse, best_match_ratio

        geometry_score = self._geometry_score(target_points)
        error_score = math.exp(-best_rmse / self._quality_error_scale)
        quality = _clamp(best_match_ratio * error_score * geometry_score, 0.0, 1.0)
        if quality < self._min_quality:
            return False, quality, np.zeros(2, dtype=np.float64), 0.0, best_rmse, best_match_ratio
        return True, quality, step_translation, step_yaw, best_rmse, best_match_ratio

    def _apply_transform(
        self,
        points: np.ndarray,
        translation: np.ndarray,
        yaw_rad: float,
    ) -> np.ndarray:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (points @ rotation.T) + translation.reshape(1, 2)

    def _nearest_neighbor(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # The clouds are deliberately downsampled to keep this brute-force match
        # stable and cheap enough without pulling runtime dependencies into the
        # Raspberry image.
        deltas = source_points[:, None, :] - target_points[None, :, :]
        sq_distances = np.sum(np.square(deltas), axis=2)
        indices = np.argmin(sq_distances, axis=1)
        distances = np.sqrt(sq_distances[np.arange(sq_distances.shape[0]), indices])
        return distances, indices

    def _best_fit_transform(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        covariance = source_centered.T @ target_centered
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        rotation = vt_mat.T @ u_mat.T
        if np.linalg.det(rotation) < 0.0:
            vt_mat[-1, :] *= -1.0
            rotation = vt_mat.T @ u_mat.T
        translation = target_centroid - (rotation @ source_centroid)
        yaw_rad = math.atan2(rotation[1, 0], rotation[0, 0])
        return translation, yaw_rad

    def _compose_translation(
        self,
        delta_translation: np.ndarray,
        current_translation: np.ndarray,
        delta_yaw: float,
    ) -> np.ndarray:
        c = math.cos(delta_yaw)
        s = math.sin(delta_yaw)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (rotation @ current_translation) + delta_translation

    def _geometry_score(self, points: np.ndarray) -> float:
        if points.shape[0] < 4:
            return 0.0
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered
        eigenvalues = np.linalg.eigvalsh(covariance)
        major = float(max(eigenvalues[-1], 1e-9))
        minor = float(max(eigenvalues[0], 1e-9))
        return _clamp(math.sqrt(minor / major), 0.15, 1.0)

    def _publish_outputs(
        self,
        stamp,
        *,
        valid: bool,
        quality: float,
        translation_observable: bool,
        step_translation: np.ndarray,
        step_yaw: float,
        dt_s: float,
        rmse_m: float,
        match_ratio: float,
    ) -> None:
        self._last_valid = valid
        self._last_quality = quality
        self._last_translation_observable = translation_observable
        self._last_match_ratio = match_ratio
        self._last_rmse_m = rmse_m if math.isfinite(rmse_m) else float("inf")
        self._last_delta_m = float(np.linalg.norm(step_translation))
        self._last_delta_yaw = float(step_yaw)

        qx, qy, qz, qw = _yaw_to_quat(self._yaw)
        if valid:
            quality_floor = max(quality, 0.05)
            cov_scale = 1.0 / quality_floor
            twist_cov_scale = 1.0 / quality_floor
        else:
            cov_scale = self._invalid_cov_scale
            twist_cov_scale = self._invalid_cov_scale * self._stale_twist_cov_scale

        pose_covariance = [0.0] * 36
        pose_covariance[0] = self._nominal_cov_x * cov_scale
        pose_covariance[7] = self._nominal_cov_y * cov_scale
        pose_covariance[14] = 1e6
        pose_covariance[21] = 1e6
        pose_covariance[28] = 1e6
        pose_covariance[35] = self._nominal_cov_yaw * cov_scale

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child_frame
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.pose.covariance = pose_covariance
        if valid and dt_s > 1e-3:
            odom.twist.twist.linear.x = float(step_translation[0] / dt_s)
            odom.twist.twist.linear.y = float(step_translation[1] / dt_s)
            odom.twist.twist.angular.z = float(step_yaw / dt_s)
        twist_covariance = [0.0] * 36
        twist_covariance[0] = self._nominal_cov_x * twist_cov_scale
        twist_covariance[7] = self._nominal_cov_y * twist_cov_scale
        twist_covariance[35] = self._nominal_cov_yaw * twist_cov_scale
        odom.twist.covariance = twist_covariance
        self._odom_pub.publish(odom)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._odom_frame
        pose_msg.pose.pose = odom.pose.pose
        pose_msg.pose.covariance = pose_covariance
        self._pose_pub.publish(pose_msg)


def main() -> None:
    rclpy.init()
    node = LidarRelativeOdometryNode()
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
