#!/usr/bin/env python3
"""Adaptive LiDAR track controller.

Single control algorithm with continuous behavior:
- Early mapping: mostly reactive corridor following.
- As map grows: progressively more anticipative steering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64


def _wrap_rad(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass
class SectorDistances:
    front: float
    left: float
    right: float
    front_left: float
    front_right: float


class AdaptiveTrackControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("adaptive_track_controller_node")

        self.declare_parameter("scan_topic", "/lidar/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("scan_timeout_s", 0.5)

        # Main speed parameters.
        self.declare_parameter("max_speed_mps", 0.576)
        self.declare_parameter("min_speed_mps", 0.50)
        self.declare_parameter("speed_limit_pct", 100.0)  # Keep at 100; limit wheels in drive node.
        self.declare_parameter("forward_speed_gain", 1.5)
        self.declare_parameter("max_yaw_rate_rad_s", 1.2)
        self.declare_parameter("speed_curve_gain", 0.6)
        self.declare_parameter("stop_distance_m", 0.05)
        self.declare_parameter("slow_distance_m", 0.20)
        self.declare_parameter("motion_min_speed_mps", 0.50)
        self.declare_parameter("motion_min_front_m", 0.12)
        self.declare_parameter("yaw_deadband_rad_s", 0.08)
        self.declare_parameter("yaw_smoothing_alpha", 0.75)
        self.declare_parameter("straight_front_threshold_m", 0.35)
        self.declare_parameter("straight_balance_threshold_m", 0.12)
        self.declare_parameter("straight_heading_threshold_rad", 0.20)
        self.declare_parameter("straight_steer_scale", 0.30)

        # Reactive + anticipative gains.
        self.declare_parameter("k_center", 0.9)
        self.declare_parameter("k_avoid", 0.20)
        self.declare_parameter("k_heading", 1.0)
        self.declare_parameter("k_curve", 0.55)
        self.declare_parameter("heading_bias_deg", 55.0)

        # Map-confidence normalization.
        self.declare_parameter("map_known_ratio_low", 0.03)
        self.declare_parameter("map_known_ratio_high", 0.28)
        self.declare_parameter("map_conf_alpha", 0.9)

        # Recovery.
        self.declare_parameter("block_cycles_threshold", 6)
        self.declare_parameter("recovery_cycles", 8)
        self.declare_parameter("recovery_speed_mps", 0.25)
        self.declare_parameter("recovery_yaw_rate_rad_s", 0.8)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._scan_timeout_s = max(0.1, float(self.get_parameter("scan_timeout_s").value))

        self._max_speed_mps = max(0.05, float(self.get_parameter("max_speed_mps").value))
        self._min_speed_mps = max(0.0, float(self.get_parameter("min_speed_mps").value))
        self._speed_limit_pct = max(1.0, min(100.0, float(self.get_parameter("speed_limit_pct").value)))
        self._forward_speed_gain = max(0.0, float(self.get_parameter("forward_speed_gain").value))
        self._max_yaw_rate = max(0.1, float(self.get_parameter("max_yaw_rate_rad_s").value))
        self._speed_curve_gain = max(0.0, float(self.get_parameter("speed_curve_gain").value))
        self._stop_distance = max(0.05, float(self.get_parameter("stop_distance_m").value))
        self._slow_distance = max(self._stop_distance + 0.05, float(self.get_parameter("slow_distance_m").value))
        self._motion_min_speed_mps = max(0.0, float(self.get_parameter("motion_min_speed_mps").value))
        self._motion_min_front_m = max(self._stop_distance + 0.01, float(self.get_parameter("motion_min_front_m").value))
        self._yaw_deadband = max(0.0, float(self.get_parameter("yaw_deadband_rad_s").value))
        self._yaw_smoothing_alpha = min(0.999, max(0.0, float(self.get_parameter("yaw_smoothing_alpha").value)))
        self._straight_front_threshold = max(0.0, float(self.get_parameter("straight_front_threshold_m").value))
        self._straight_balance_threshold = max(0.0, float(self.get_parameter("straight_balance_threshold_m").value))
        self._straight_heading_threshold = max(0.0, float(self.get_parameter("straight_heading_threshold_rad").value))
        self._straight_steer_scale = min(1.0, max(0.0, float(self.get_parameter("straight_steer_scale").value)))

        self._k_center = float(self.get_parameter("k_center").value)
        self._k_avoid = float(self.get_parameter("k_avoid").value)
        self._k_heading = float(self.get_parameter("k_heading").value)
        self._k_curve = float(self.get_parameter("k_curve").value)
        self._heading_bias_deg = max(10.0, float(self.get_parameter("heading_bias_deg").value))

        self._map_ratio_low = max(0.0, float(self.get_parameter("map_known_ratio_low").value))
        self._map_ratio_high = max(self._map_ratio_low + 1e-3, float(self.get_parameter("map_known_ratio_high").value))
        self._map_conf_alpha = max(0.0, min(0.999, float(self.get_parameter("map_conf_alpha").value)))

        self._block_cycles_threshold = max(1, int(self.get_parameter("block_cycles_threshold").value))
        self._recovery_cycles = max(1, int(self.get_parameter("recovery_cycles").value))
        self._recovery_speed = max(0.05, float(self.get_parameter("recovery_speed_mps").value))
        self._recovery_yaw_rate = max(0.1, float(self.get_parameter("recovery_yaw_rate_rad_s").value))

        self._scan: Optional[LaserScan] = None
        self._last_scan_time: Optional[rclpy.time.Time] = None
        self._map_conf = 0.0

        self._blocked_counter = 0
        self._recovery_counter = 0
        self._recovery_turn_sign = 1.0
        self._last_w_cmd = 0.0

        self._pub_cmd = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._pub_conf = self.create_publisher(Float64, "/vehicle/track_controller/map_confidence", 10)
        self._pub_front = self.create_publisher(Float64, "/vehicle/track_controller/front_distance", 10)

        # Match LiDAR publisher QoS (best-effort sensor data) to avoid incompatibility.
        self.create_subscription(LaserScan, self._scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, self._map_topic, self._on_map, 10)
        self._timer = self.create_timer(1.0 / self._control_rate_hz, self._on_timer)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            (
                "AdaptiveTrackController started | speed_limit_pct=%.1f forward_gain=%.2f "
                "max_speed=%.2f m/s yaw_deadband=%.2f smoothing=%.2f"
            )
            % (
                self._speed_limit_pct,
                self._forward_speed_gain,
                self._max_speed_mps,
                self._yaw_deadband,
                self._yaw_smoothing_alpha,
            )
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg
        self._last_scan_time = self.get_clock().now()

    def _on_map(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16)
        if data.size == 0:
            return
        known = float(np.count_nonzero(data >= 0))
        known_ratio = known / float(data.size)

        conf = (known_ratio - self._map_ratio_low) / (self._map_ratio_high - self._map_ratio_low)
        conf = max(0.0, min(1.0, conf))
        self._map_conf = self._map_conf_alpha * self._map_conf + (1.0 - self._map_conf_alpha) * conf

    @staticmethod
    def _sector_stat(ranges: np.ndarray, mask: np.ndarray, percentile: float, fallback: float) -> float:
        values = ranges[mask]
        if values.size == 0:
            return fallback
        return float(np.percentile(values, percentile))

    def _extract_sector_distances(self, scan: LaserScan) -> tuple[np.ndarray, np.ndarray, SectorDistances]:
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return ranges, np.zeros((0,), dtype=np.float32), SectorDistances(0.0, 0.0, 0.0, 0.0, 0.0)

        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
        angles = _wrap_rad(angles)

        valid = np.isfinite(ranges) & (ranges > 0.0)
        if scan.range_min > 0.0 and np.isfinite(scan.range_min):
            valid &= ranges >= float(scan.range_min)
        if scan.range_max > 0.0 and np.isfinite(scan.range_max):
            valid &= ranges <= float(scan.range_max)

        safe_ranges = np.where(valid, ranges, np.nan)

        deg = np.degrees(angles)
        front_mask = valid & (np.abs(deg) <= 12.0)
        left_mask = valid & (deg >= 60.0) & (deg <= 110.0)
        right_mask = valid & (deg <= -60.0) & (deg >= -110.0)
        fl_mask = valid & (deg >= 15.0) & (deg <= 50.0)
        fr_mask = valid & (deg <= -15.0) & (deg >= -50.0)

        fallback = 10.0
        sectors = SectorDistances(
            front=self._sector_stat(safe_ranges, front_mask, 20.0, fallback),
            left=self._sector_stat(safe_ranges, left_mask, 50.0, fallback),
            right=self._sector_stat(safe_ranges, right_mask, 50.0, fallback),
            front_left=self._sector_stat(safe_ranges, fl_mask, 35.0, fallback),
            front_right=self._sector_stat(safe_ranges, fr_mask, 35.0, fallback),
        )
        return safe_ranges, angles, sectors

    def _compute_target_heading(self, ranges: np.ndarray, angles: np.ndarray) -> float:
        if ranges.size == 0:
            return 0.0

        valid = np.isfinite(ranges)
        if not np.any(valid):
            return 0.0

        forward = valid & (np.abs(np.degrees(angles)) <= 90.0)
        if not np.any(forward):
            return 0.0

        fw_ranges = ranges[forward]
        fw_angles = angles[forward]
        bias = np.exp(-np.abs(np.degrees(fw_angles)) / self._heading_bias_deg)
        score = fw_ranges * bias
        idx = int(np.nanargmax(score))
        return float(fw_angles[idx])

    def _compute_control(self, scan: LaserScan) -> tuple[float, float, float]:
        ranges, angles, d = self._extract_sector_distances(scan)

        center_err = d.left - d.right
        avoid_err = (1.0 / max(0.05, d.front_right)) - (1.0 / max(0.05, d.front_left))
        curve_hint = d.front_left - d.front_right
        target_heading = self._compute_target_heading(ranges, angles)

        w_reactive = self._k_center * center_err + self._k_avoid * avoid_err
        w_predictive = self._k_heading * target_heading + self._k_curve * curve_hint

        # Continuous adaptation with map confidence (0: reactive, 1: more predictive).
        lam = float(self._map_conf)
        w_cmd = (1.0 - lam) * w_reactive + lam * (w_reactive + w_predictive)

        # Straight stabilizer: when corridor looks straight/balanced, damp steering commands.
        if (
            d.front >= self._straight_front_threshold
            and abs(center_err) <= self._straight_balance_threshold
            and abs(target_heading) <= self._straight_heading_threshold
        ):
            w_cmd *= self._straight_steer_scale

        w_cmd = max(-self._max_yaw_rate, min(self._max_yaw_rate, w_cmd))

        max_speed_eff = self._max_speed_mps * (self._speed_limit_pct / 100.0)
        if d.front <= self._stop_distance:
            return 0.0, w_cmd, d.front

        speed_ratio = (d.front - self._stop_distance) / (self._slow_distance - self._stop_distance)
        speed_ratio = max(0.0, min(1.0, speed_ratio))
        v_clear = speed_ratio * max_speed_eff

        # Faster on straights, slower on hard curves.
        curvature_scale = math.exp(-self._speed_curve_gain * abs(w_cmd))
        v_cmd = v_clear * curvature_scale
        # Keep stronger cruise while there is enough free space ahead.
        if d.front >= self._motion_min_front_m:
            v_cmd = max(v_cmd, self._motion_min_speed_mps)
        else:
            v_cmd = max(v_cmd, self._min_speed_mps * speed_ratio)
        # Global longitudinal gain to tune "push" without changing geometry terms.
        v_cmd *= self._forward_speed_gain
        v_cmd = max(0.0, min(v_cmd, max_speed_eff))

        return v_cmd, w_cmd, d.front

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self._pub_cmd.publish(msg)

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        if self._scan is None or self._last_scan_time is None:
            self._publish_cmd(0.0, 0.0)
            return

        age_s = (now - self._last_scan_time).nanoseconds * 1e-9
        if age_s > self._scan_timeout_s:
            self._publish_cmd(0.0, 0.0)
            return

        v_cmd, w_cmd, d_front = self._compute_control(self._scan)

        if d_front <= self._stop_distance:
            self._blocked_counter += 1
        else:
            self._blocked_counter = 0

        if self._recovery_counter > 0:
            v_cmd = -self._recovery_speed
            w_cmd = self._recovery_turn_sign * self._recovery_yaw_rate
            self._recovery_counter -= 1
            self._last_w_cmd = w_cmd
        elif self._blocked_counter >= self._block_cycles_threshold:
            # Pick direction with more free space.
            _, _, d = self._extract_sector_distances(self._scan)
            self._recovery_turn_sign = 1.0 if d.left >= d.right else -1.0
            self._recovery_counter = self._recovery_cycles
            self._blocked_counter = 0
            v_cmd = -self._recovery_speed
            w_cmd = self._recovery_turn_sign * self._recovery_yaw_rate
            self._last_w_cmd = w_cmd
        else:
            # Low-pass filter + deadband to suppress scan-to-scan steering noise.
            w_cmd = self._yaw_smoothing_alpha * self._last_w_cmd + (1.0 - self._yaw_smoothing_alpha) * w_cmd
            if abs(w_cmd) < self._yaw_deadband:
                w_cmd = 0.0
            self._last_w_cmd = w_cmd

        self._publish_cmd(v_cmd, w_cmd)

        msg_conf = Float64()
        msg_conf.data = float(self._map_conf)
        self._pub_conf.publish(msg_conf)

        msg_front = Float64()
        msg_front.data = float(d_front)
        self._pub_front.publish(msg_front)

    def _on_set_parameters(self, params: list) -> SetParametersResult:
        for p in params:
            if p.name == "speed_limit_pct":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="speed_limit_pct must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="speed_limit_pct must be > 0")
                self._speed_limit_pct = max(1.0, min(100.0, value))
                self.get_logger().info("Updated speed_limit_pct=%.1f" % self._speed_limit_pct)
            elif p.name == "forward_speed_gain":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="forward_speed_gain must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="forward_speed_gain must be >= 0")
                self._forward_speed_gain = value
                self.get_logger().info("Updated forward_speed_gain=%.2f" % self._forward_speed_gain)
            elif p.name == "min_speed_mps":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="min_speed_mps must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="min_speed_mps must be >= 0")
                self._min_speed_mps = value
                if self._max_speed_mps < self._min_speed_mps:
                    self._max_speed_mps = self._min_speed_mps
                self.get_logger().info("Updated min_speed_mps=%.3f" % self._min_speed_mps)
            elif p.name == "max_speed_mps":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="max_speed_mps must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="max_speed_mps must be > 0")
                self._max_speed_mps = max(value, self._min_speed_mps)
                self.get_logger().info("Updated max_speed_mps=%.3f" % self._max_speed_mps)
            elif p.name == "max_yaw_rate_rad_s":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="max_yaw_rate_rad_s must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="max_yaw_rate_rad_s must be > 0")
                self._max_yaw_rate = value
                self.get_logger().info("Updated max_yaw_rate_rad_s=%.3f" % self._max_yaw_rate)
            elif p.name == "yaw_deadband_rad_s":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="yaw_deadband_rad_s must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="yaw_deadband_rad_s must be >= 0")
                self._yaw_deadband = value
                self.get_logger().info("Updated yaw_deadband_rad_s=%.3f" % self._yaw_deadband)
            elif p.name == "yaw_smoothing_alpha":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="yaw_smoothing_alpha must be numeric")
                if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                    return SetParametersResult(successful=False, reason="yaw_smoothing_alpha must be in [0,1)")
                self._yaw_smoothing_alpha = value
                self.get_logger().info("Updated yaw_smoothing_alpha=%.3f" % self._yaw_smoothing_alpha)
            elif p.name == "straight_front_threshold_m":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="straight_front_threshold_m must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="straight_front_threshold_m must be >= 0")
                self._straight_front_threshold = value
                self.get_logger().info("Updated straight_front_threshold_m=%.3f" % self._straight_front_threshold)
            elif p.name == "straight_balance_threshold_m":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="straight_balance_threshold_m must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="straight_balance_threshold_m must be >= 0")
                self._straight_balance_threshold = value
                self.get_logger().info("Updated straight_balance_threshold_m=%.3f" % self._straight_balance_threshold)
            elif p.name == "straight_heading_threshold_rad":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="straight_heading_threshold_rad must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="straight_heading_threshold_rad must be >= 0")
                self._straight_heading_threshold = value
                self.get_logger().info("Updated straight_heading_threshold_rad=%.3f" % self._straight_heading_threshold)
            elif p.name == "straight_steer_scale":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="straight_steer_scale must be numeric")
                if not math.isfinite(value) or value < 0.0 or value > 1.0:
                    return SetParametersResult(successful=False, reason="straight_steer_scale must be in [0,1]")
                self._straight_steer_scale = value
                self.get_logger().info("Updated straight_steer_scale=%.3f" % self._straight_steer_scale)
            elif p.name == "speed_curve_gain":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="speed_curve_gain must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="speed_curve_gain must be >= 0")
                self._speed_curve_gain = value
                self.get_logger().info("Updated speed_curve_gain=%.3f" % self._speed_curve_gain)
            elif p.name == "stop_distance_m":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="stop_distance_m must be numeric")
                if not math.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="stop_distance_m must be > 0")
                self._stop_distance = value
                self._slow_distance = max(self._slow_distance, self._stop_distance + 0.05)
                self._motion_min_front_m = max(self._motion_min_front_m, self._stop_distance + 0.01)
                self.get_logger().info("Updated stop_distance_m=%.3f" % self._stop_distance)
            elif p.name == "slow_distance_m":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="slow_distance_m must be numeric")
                if not math.isfinite(value) or value <= self._stop_distance:
                    return SetParametersResult(
                        successful=False, reason="slow_distance_m must be > stop_distance_m"
                    )
                self._slow_distance = value
                self.get_logger().info("Updated slow_distance_m=%.3f" % self._slow_distance)
            elif p.name == "motion_min_speed_mps":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="motion_min_speed_mps must be numeric")
                if not math.isfinite(value) or value < 0.0:
                    return SetParametersResult(successful=False, reason="motion_min_speed_mps must be >= 0")
                self._motion_min_speed_mps = value
                self.get_logger().info("Updated motion_min_speed_mps=%.3f" % self._motion_min_speed_mps)
            elif p.name == "motion_min_front_m":
                try:
                    value = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason="motion_min_front_m must be numeric")
                if not math.isfinite(value) or value <= self._stop_distance:
                    return SetParametersResult(successful=False, reason="motion_min_front_m must be > stop_distance_m")
                self._motion_min_front_m = value
                self.get_logger().info("Updated motion_min_front_m=%.3f" % self._motion_min_front_m)
        return SetParametersResult(successful=True)


def main() -> None:
    rclpy.init()
    node = AdaptiveTrackControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
