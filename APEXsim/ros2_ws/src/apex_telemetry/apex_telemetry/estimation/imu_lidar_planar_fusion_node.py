#!/usr/bin/env python3
"""Online causal LiDAR + IMU planar fusion for short corridor runs."""

from __future__ import annotations

import json
import math
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .fixed_map_localizer_core import FixedMapParameters, FixedMapPlanarLocalizer
from .planar_fusion_core import FusionParameters, OnlinePlanarFusion, scan_observation_from_ranges


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


class ImuLidarPlanarFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("imu_lidar_planar_fusion_node")

        self.declare_parameter("imu_topic", "/apex/imu/data_raw")
        self.declare_parameter("scan_topic", "/lidar/scan_localization")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("path_topic", "/apex/estimation/path")
        self.declare_parameter("pose_topic", "/apex/estimation/current_pose")
        self.declare_parameter("live_map_topic", "/apex/estimation/live_map_points")
        self.declare_parameter("full_map_topic", "/apex/estimation/full_map_points")
        self.declare_parameter("status_topic", "/apex/estimation/status")
        self.declare_parameter("odom_frame_id", "odom_imu_lidar_fused")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("status_publish_rate_hz", 2.0)
        self.declare_parameter("publish_predicted_odom_between_scans", True)
        self.declare_parameter("predicted_odom_rate_hz", 30.0)
        self.declare_parameter("max_prediction_horizon_s", 0.20)
        self.declare_parameter("path_max_poses", 4000)
        self.declare_parameter("live_map_publish_rate_hz", 1.0)
        self.declare_parameter("live_map_window_scans", 120)
        self.declare_parameter("live_map_max_points", 6000)
        self.declare_parameter("full_map_publish_rate_hz", 0.5)
        self.declare_parameter("full_map_max_points", 30000)
        self.declare_parameter("estimation_backend", "online_submap")

        self.declare_parameter("median_window", 5)
        self.declare_parameter("ema_alpha", 0.25)
        self.declare_parameter("static_window_s", 0.4)
        self.declare_parameter("static_search_s", 2.0)
        self.declare_parameter("velocity_decay_tau_s", 1.1)
        self.declare_parameter("submap_window_scans", 6)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("max_correspondence_m", 0.35)
        self.declare_parameter("initial_scan_count_min", 4)
        self.declare_parameter("max_initial_alignment_scans", 6)
        self.declare_parameter("corridor_bin_m", 0.10)
        self.declare_parameter("low_confidence_residual_m", 0.16)
        self.declare_parameter("min_valid_correspondence_count", 14)
        self.declare_parameter("max_scan_optimization_evals", 80)
        self.declare_parameter("fixed_map_yaml", "")
        self.declare_parameter("fixed_map_distance_npy", "")
        self.declare_parameter("fixed_map_visual_points_csv", "")
        self.declare_parameter("fixed_map_route_csv", "")
        self.declare_parameter("fixed_map_max_match_points", 120)
        self.declare_parameter("fixed_map_max_localization_iterations", 40)
        self.declare_parameter("fixed_map_prior_translation_weight", 1.0)
        self.declare_parameter("fixed_map_prior_yaw_weight", 1.4)
        self.declare_parameter("fixed_map_startup_static_duration_s", 1.5)
        self.declare_parameter("fixed_map_startup_gyro_stddev_threshold_rps", 0.04)
        self.declare_parameter("fixed_map_startup_accel_stddev_threshold_mps2", 0.35)
        self.declare_parameter("fixed_map_imu_filter_window", 7)
        self.declare_parameter("fixed_map_imu_spike_mad_scale", 4.0)
        self.declare_parameter("fixed_map_max_accel_axis_mps2", 3.2)
        self.declare_parameter("fixed_map_max_yaw_rate_abs_rps", 2.6)
        self.declare_parameter("fixed_map_low_support_distance_m", 0.90)
        self.declare_parameter("fixed_map_low_support_in_bounds_ratio", 0.45)
        self.declare_parameter("fixed_map_low_support_near_wall_ratio", 0.25)
        self.declare_parameter("fixed_map_reduced_support_prior_gain", 1.35)
        self.declare_parameter("fixed_map_low_support_prior_gain", 1.80)
        self.declare_parameter("fixed_map_motion_odom_topic", "/apex/odometry/imu_raw")
        self.declare_parameter("fixed_map_motion_hint_timeout_s", 0.45)
        self.declare_parameter("fixed_map_motion_hint_velocity_blend", 0.30)
        self.declare_parameter("fixed_map_motion_hint_max_speed_mps", 1.5)
        self.declare_parameter("fixed_map_particle_count", 384)
        self.declare_parameter("fixed_map_particle_seed", 17)
        self.declare_parameter("fixed_map_particle_initial_xy_std_m", 0.28)
        self.declare_parameter("fixed_map_particle_initial_yaw_std_rad", 0.35)
        self.declare_parameter("fixed_map_particle_route_seed_fraction", 0.18)
        self.declare_parameter("fixed_map_particle_random_injection_ratio", 0.025)
        self.declare_parameter("fixed_map_particle_process_xy_std_m", 0.035)
        self.declare_parameter("fixed_map_particle_process_yaw_std_rad", 0.030)
        self.declare_parameter("fixed_map_particle_process_velocity_std_mps", 0.06)
        self.declare_parameter("fixed_map_particle_likelihood_sigma_m", 0.075)
        self.declare_parameter("fixed_map_particle_inlier_distance_m", 0.13)
        self.declare_parameter("fixed_map_particle_inlier_log_weight", 2.0)
        self.declare_parameter("fixed_map_particle_out_of_map_penalty", 2.4)
        self.declare_parameter("fixed_map_particle_resample_neff_ratio", 0.55)
        self.declare_parameter("fixed_map_particle_roughening_xy_std_m", 0.018)
        self.declare_parameter("fixed_map_particle_roughening_yaw_std_rad", 0.025)
        self.declare_parameter("fixed_map_particle_min_lidar_points", 12)
        self.declare_parameter("fixed_map_particle_min_observation_weight", 0.20)
        self.declare_parameter("fixed_map_particle_high_confidence_inlier_ratio", 0.45)
        self.declare_parameter("fixed_map_particle_medium_confidence_inlier_ratio", 0.28)
        self.declare_parameter("fixed_map_particle_max_high_confidence_spread_m", 0.55)
        self.declare_parameter("fixed_map_particle_max_medium_confidence_spread_m", 0.90)
        self.declare_parameter("fixed_map_particle_refine_enabled", True)
        self.declare_parameter("fixed_map_particle_refine_gain", 0.70)

        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._live_map_topic = str(self.get_parameter("live_map_topic").value)
        self._full_map_topic = str(self.get_parameter("full_map_topic").value)
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._odom_frame = str(self.get_parameter("odom_frame_id").value)
        self._child_frame = str(self.get_parameter("child_frame_id").value)
        self._estimation_backend = str(self.get_parameter("estimation_backend").value).strip().lower()
        if self._estimation_backend not in {"online_submap", "fixed_map"}:
            self._estimation_backend = "online_submap"
        self._publish_tf = bool(self.get_parameter("publish_tf").value)
        self._status_publish_rate_hz = max(
            0.5, float(self.get_parameter("status_publish_rate_hz").value)
        )
        self._publish_predicted_odom_between_scans = bool(
            self.get_parameter("publish_predicted_odom_between_scans").value
        )
        self._predicted_odom_rate_hz = max(
            1.0, float(self.get_parameter("predicted_odom_rate_hz").value)
        )
        self._max_prediction_horizon_s = max(
            0.02, float(self.get_parameter("max_prediction_horizon_s").value)
        )
        self._path_max_poses = max(32, int(self.get_parameter("path_max_poses").value))
        self._live_map_publish_rate_hz = max(
            0.2, float(self.get_parameter("live_map_publish_rate_hz").value)
        )
        self._live_map_window_scans = max(
            4, int(self.get_parameter("live_map_window_scans").value)
        )
        self._live_map_max_points = max(
            64, int(self.get_parameter("live_map_max_points").value)
        )
        self._full_map_publish_rate_hz = max(
            0.1, float(self.get_parameter("full_map_publish_rate_hz").value)
        )
        self._full_map_max_points = max(
            256, int(self.get_parameter("full_map_max_points").value)
        )
        self._point_stride = int(self.get_parameter("point_stride").value)

        if self._estimation_backend == "fixed_map":
            fixed_map_params = FixedMapParameters(
                fixed_map_yaml=str(self.get_parameter("fixed_map_yaml").value),
                fixed_map_distance_npy=str(self.get_parameter("fixed_map_distance_npy").value),
                fixed_map_visual_points_csv=str(
                    self.get_parameter("fixed_map_visual_points_csv").value
                ),
                fixed_map_route_csv=str(self.get_parameter("fixed_map_route_csv").value),
                velocity_decay_tau_s=float(self.get_parameter("velocity_decay_tau_s").value),
                max_match_points=int(self.get_parameter("fixed_map_max_match_points").value),
                max_localization_iterations=int(
                    self.get_parameter("fixed_map_max_localization_iterations").value
                ),
                max_correspondence_m=float(self.get_parameter("max_correspondence_m").value),
                prior_translation_weight=float(
                    self.get_parameter("fixed_map_prior_translation_weight").value
                ),
                prior_yaw_weight=float(self.get_parameter("fixed_map_prior_yaw_weight").value),
                startup_static_duration_s=float(
                    self.get_parameter("fixed_map_startup_static_duration_s").value
                ),
                startup_gyro_stddev_threshold_rps=float(
                    self.get_parameter("fixed_map_startup_gyro_stddev_threshold_rps").value
                ),
                startup_accel_stddev_threshold_mps2=float(
                    self.get_parameter("fixed_map_startup_accel_stddev_threshold_mps2").value
                ),
                imu_filter_window=int(self.get_parameter("fixed_map_imu_filter_window").value),
                imu_spike_mad_scale=float(
                    self.get_parameter("fixed_map_imu_spike_mad_scale").value
                ),
                max_accel_axis_mps2=float(
                    self.get_parameter("fixed_map_max_accel_axis_mps2").value
                ),
                max_yaw_rate_abs_rps=float(
                    self.get_parameter("fixed_map_max_yaw_rate_abs_rps").value
                ),
                low_support_distance_m=float(
                    self.get_parameter("fixed_map_low_support_distance_m").value
                ),
                low_support_in_bounds_ratio=float(
                    self.get_parameter("fixed_map_low_support_in_bounds_ratio").value
                ),
                low_support_near_wall_ratio=float(
                    self.get_parameter("fixed_map_low_support_near_wall_ratio").value
                ),
                reduced_support_prior_gain=float(
                    self.get_parameter("fixed_map_reduced_support_prior_gain").value
                ),
                low_support_prior_gain=float(
                    self.get_parameter("fixed_map_low_support_prior_gain").value
                ),
                motion_hint_timeout_s=float(
                    self.get_parameter("fixed_map_motion_hint_timeout_s").value
                ),
                motion_hint_velocity_blend=float(
                    self.get_parameter("fixed_map_motion_hint_velocity_blend").value
                ),
                motion_hint_max_speed_mps=float(
                    self.get_parameter("fixed_map_motion_hint_max_speed_mps").value
                ),
                particle_count=int(self.get_parameter("fixed_map_particle_count").value),
                particle_seed=int(self.get_parameter("fixed_map_particle_seed").value),
                particle_initial_xy_std_m=float(
                    self.get_parameter("fixed_map_particle_initial_xy_std_m").value
                ),
                particle_initial_yaw_std_rad=float(
                    self.get_parameter("fixed_map_particle_initial_yaw_std_rad").value
                ),
                particle_route_seed_fraction=float(
                    self.get_parameter("fixed_map_particle_route_seed_fraction").value
                ),
                particle_random_injection_ratio=float(
                    self.get_parameter("fixed_map_particle_random_injection_ratio").value
                ),
                particle_process_xy_std_m=float(
                    self.get_parameter("fixed_map_particle_process_xy_std_m").value
                ),
                particle_process_yaw_std_rad=float(
                    self.get_parameter("fixed_map_particle_process_yaw_std_rad").value
                ),
                particle_process_velocity_std_mps=float(
                    self.get_parameter("fixed_map_particle_process_velocity_std_mps").value
                ),
                particle_likelihood_sigma_m=float(
                    self.get_parameter("fixed_map_particle_likelihood_sigma_m").value
                ),
                particle_inlier_distance_m=float(
                    self.get_parameter("fixed_map_particle_inlier_distance_m").value
                ),
                particle_inlier_log_weight=float(
                    self.get_parameter("fixed_map_particle_inlier_log_weight").value
                ),
                particle_out_of_map_penalty=float(
                    self.get_parameter("fixed_map_particle_out_of_map_penalty").value
                ),
                particle_resample_neff_ratio=float(
                    self.get_parameter("fixed_map_particle_resample_neff_ratio").value
                ),
                particle_roughening_xy_std_m=float(
                    self.get_parameter("fixed_map_particle_roughening_xy_std_m").value
                ),
                particle_roughening_yaw_std_rad=float(
                    self.get_parameter("fixed_map_particle_roughening_yaw_std_rad").value
                ),
                particle_min_lidar_points=int(
                    self.get_parameter("fixed_map_particle_min_lidar_points").value
                ),
                particle_min_observation_weight=float(
                    self.get_parameter("fixed_map_particle_min_observation_weight").value
                ),
                particle_high_confidence_inlier_ratio=float(
                    self.get_parameter("fixed_map_particle_high_confidence_inlier_ratio").value
                ),
                particle_medium_confidence_inlier_ratio=float(
                    self.get_parameter("fixed_map_particle_medium_confidence_inlier_ratio").value
                ),
                particle_max_high_confidence_spread_m=float(
                    self.get_parameter("fixed_map_particle_max_high_confidence_spread_m").value
                ),
                particle_max_medium_confidence_spread_m=float(
                    self.get_parameter("fixed_map_particle_max_medium_confidence_spread_m").value
                ),
                particle_refine_enabled=bool(
                    self.get_parameter("fixed_map_particle_refine_enabled").value
                ),
                particle_refine_gain=float(
                    self.get_parameter("fixed_map_particle_refine_gain").value
                ),
            )
            self._fusion = FixedMapPlanarLocalizer(fixed_map_params)
        else:
            params = FusionParameters(
                median_window=int(self.get_parameter("median_window").value),
                ema_alpha=float(self.get_parameter("ema_alpha").value),
                static_window_s=float(self.get_parameter("static_window_s").value),
                static_search_s=float(self.get_parameter("static_search_s").value),
                velocity_decay_tau_s=float(self.get_parameter("velocity_decay_tau_s").value),
                submap_window_scans=int(self.get_parameter("submap_window_scans").value),
                point_stride=int(self.get_parameter("point_stride").value),
                max_correspondence_m=float(self.get_parameter("max_correspondence_m").value),
                initial_scan_count_min=int(self.get_parameter("initial_scan_count_min").value),
                max_initial_alignment_scans=int(
                    self.get_parameter("max_initial_alignment_scans").value
                ),
                corridor_bin_m=float(self.get_parameter("corridor_bin_m").value),
                low_confidence_residual_m=float(
                    self.get_parameter("low_confidence_residual_m").value
                ),
                min_valid_correspondence_count=int(
                    self.get_parameter("min_valid_correspondence_count").value
                ),
                max_scan_optimization_evals=int(
                    self.get_parameter("max_scan_optimization_evals").value
                ),
            )
            self._fusion = OnlinePlanarFusion(params)

        self._scan_counter = 0
        self._last_status_payload = ""
        self._last_estimate = None
        self._last_published_pose_payload: dict[str, float | str | bool] | None = None
        self._last_published_source = "none"
        self._last_prediction_age_s = 0.0
        self._last_corrected_header = None
        self._path_msg = Path()
        self._path_msg.header.frame_id = self._odom_frame

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self._fixed_map_motion_odom_topic = ""
        if self._estimation_backend == "fixed_map":
            self._fixed_map_motion_odom_topic = str(
                self.get_parameter("fixed_map_motion_odom_topic").value
            ).strip()
            if self._fixed_map_motion_odom_topic:
                self.create_subscription(
                    Odometry,
                    self._fixed_map_motion_odom_topic,
                    self._motion_odom_cb,
                    qos_profile_sensor_data,
                )

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 20)
        self._path_pub = self.create_publisher(Path, self._path_topic, 20)
        self._pose_pub = self.create_publisher(PoseStamped, self._pose_topic, 20)
        self._live_map_pub = self.create_publisher(PointCloud2, self._live_map_topic, latched_qos)
        self._full_map_pub = self.create_publisher(PointCloud2, self._full_map_topic, latched_qos)
        self._status_pub = self.create_publisher(String, self._status_topic, 20)
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None

        self.create_timer(1.0 / self._status_publish_rate_hz, self._publish_status)
        self.create_timer(1.0 / self._live_map_publish_rate_hz, self._publish_live_map)
        self.create_timer(1.0 / self._full_map_publish_rate_hz, self._publish_full_map)
        if self._publish_predicted_odom_between_scans:
            self.create_timer(1.0 / self._predicted_odom_rate_hz, self._publish_predicted_odom)

        self.get_logger().info(
            "ImuLidarPlanarFusionNode started (backend=%s imu=%s scan=%s odom=%s path=%s pose=%s map=%s full_map=%s status=%s)"
            % (
                self._estimation_backend,
                self._imu_topic,
                self._scan_topic,
                self._odom_topic,
                self._path_topic,
                self._pose_topic,
                self._live_map_topic,
                self._full_map_topic,
                self._status_topic,
            )
        )

    def _imu_cb(self, msg: Imu) -> None:
        t_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        self._fusion.add_imu_sample(
            t_s=t_s,
            ax_mps2=float(msg.linear_acceleration.x),
            ay_mps2=float(msg.linear_acceleration.y),
            az_mps2=float(msg.linear_acceleration.z),
            gz_rps=float(msg.angular_velocity.z),
        )

    def _motion_odom_cb(self, msg: Odometry) -> None:
        if not hasattr(self._fusion, "set_motion_hint"):
            return
        t_s = float(msg.header.stamp.sec) + (1.0e-9 * float(msg.header.stamp.nanosec))
        if t_s <= 0.0:
            now_msg = self.get_clock().now().to_msg()
            t_s = float(now_msg.sec) + (1.0e-9 * float(now_msg.nanosec))
        self._fusion.set_motion_hint(
            t_s=t_s,
            vx_mps=float(msg.twist.twist.linear.x),
            vy_mps=float(msg.twist.twist.linear.y),
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        scan = scan_observation_from_ranges(
            scan_index=self._scan_counter,
            stamp_sec=int(msg.header.stamp.sec),
            stamp_nanosec=int(msg.header.stamp.nanosec),
            angle_min_rad=float(msg.angle_min),
            angle_increment_rad=float(msg.angle_increment),
            ranges=list(msg.ranges),
            range_min_m=float(msg.range_min),
            range_max_m=float(msg.range_max),
            point_stride=self._point_stride,
        )
        self._scan_counter += 1
        if scan.points_local.shape[0] < 8:
            self._publish_status()
            return

        estimates = self._fusion.add_scan_observation(scan)
        for estimate in estimates:
            self._last_estimate = estimate
            self._publish_estimate(estimate, predicted=False, prediction_age_s=0.0)
        self._publish_status()

    def _publish_estimate(
        self,
        estimate,
        *,
        predicted: bool,
        prediction_age_s: float,
    ) -> None:
        if predicted:
            now_msg = self.get_clock().now().to_msg()
            stamp_sec = int(now_msg.sec)
            stamp_nanosec = int(now_msg.nanosec)
            stamp_s = float(stamp_sec) + (1.0e-9 * float(stamp_nanosec))
            x_m = float(estimate.x_m) + (float(estimate.vx_mps) * prediction_age_s)
            y_m = float(estimate.y_m) + (float(estimate.vy_mps) * prediction_age_s)
            yaw_rad = float(estimate.yaw_rad) + (float(estimate.yaw_rate_rps) * prediction_age_s)
        else:
            stamp_sec = int(estimate.stamp_sec)
            stamp_nanosec = int(estimate.stamp_nanosec)
            stamp_s = float(stamp_sec) + (1.0e-9 * float(stamp_nanosec))
            x_m = float(estimate.x_m)
            y_m = float(estimate.y_m)
            yaw_rad = float(estimate.yaw_rad)

        qx, qy, qz, qw = _yaw_to_quat(yaw_rad)

        odom = Odometry()
        odom.header.stamp.sec = stamp_sec
        odom.header.stamp.nanosec = stamp_nanosec
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child_frame
        odom.pose.pose.position.x = x_m
        odom.pose.pose.position.y = y_m
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(estimate.vx_mps)
        odom.twist.twist.linear.y = float(estimate.vy_mps)
        odom.twist.twist.angular.z = float(estimate.yaw_rate_rps)

        high_confidence = estimate.confidence == "high"
        pose_cov_xy = 0.025 if high_confidence else 0.18
        yaw_cov = 0.035 if high_confidence else 0.20
        twist_cov_xy = 0.06 if high_confidence else 0.25
        if predicted:
            pose_cov_xy += (0.35 * prediction_age_s)
            yaw_cov += (0.45 * prediction_age_s)
            twist_cov_xy += (0.18 * prediction_age_s)
        odom.pose.covariance[0] = pose_cov_xy
        odom.pose.covariance[7] = pose_cov_xy
        odom.pose.covariance[35] = yaw_cov
        odom.twist.covariance[0] = twist_cov_xy
        odom.twist.covariance[7] = twist_cov_xy
        odom.twist.covariance[35] = yaw_cov
        self._odom_pub.publish(odom)

        self._last_published_source = "predicted" if predicted else "corrected"
        self._last_prediction_age_s = float(max(0.0, prediction_age_s))
        if not predicted:
            self._last_corrected_header = odom.header
        self._last_published_pose_payload = {
            "x_m": x_m,
            "y_m": y_m,
            "yaw_rad": float(yaw_rad),
            "vx_mps": float(estimate.vx_mps),
            "vy_mps": float(estimate.vy_mps),
            "yaw_rate_rps": float(estimate.yaw_rate_rps),
            "confidence": str(estimate.confidence),
            "stamp_s": stamp_s,
            "predicted": bool(predicted),
            "prediction_age_s": float(max(0.0, prediction_age_s)),
        }

        if predicted:
            if self._tf_broadcaster is not None:
                transform = TransformStamped()
                transform.header = odom.header
                transform.child_frame_id = self._child_frame
                transform.transform.translation.x = odom.pose.pose.position.x
                transform.transform.translation.y = odom.pose.pose.position.y
                transform.transform.translation.z = 0.0
                transform.transform.rotation = odom.pose.pose.orientation
                self._tf_broadcaster.sendTransform(transform)
            return

        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self._pose_pub.publish(pose_stamped)
        self._path_msg.header.stamp = odom.header.stamp
        self._path_msg.poses.append(pose_stamped)
        if len(self._path_msg.poses) > self._path_max_poses:
            del self._path_msg.poses[: len(self._path_msg.poses) - self._path_max_poses]
        self._path_pub.publish(self._path_msg)

        if self._tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = self._child_frame
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = 0.0
            transform.transform.rotation = odom.pose.pose.orientation
            self._tf_broadcaster.sendTransform(transform)

    def _publish_predicted_odom(self) -> None:
        if self._last_estimate is None:
            return
        now_msg = self.get_clock().now().to_msg()
        now_s = float(now_msg.sec) + (1.0e-9 * float(now_msg.nanosec))
        prediction_age_s = max(0.0, now_s - float(self._last_estimate.t_s))
        if prediction_age_s <= 1.0e-3 or prediction_age_s > self._max_prediction_horizon_s:
            return
        self._publish_estimate(
            self._last_estimate,
            predicted=True,
            prediction_age_s=prediction_age_s,
        )

    def _publish_live_map(self) -> None:
        if self._last_corrected_header is None:
            return
        map_points_xy = self._fusion.live_map_points_world(
            window_scans=self._live_map_window_scans,
            max_points=self._live_map_max_points,
        )
        message = self._pointcloud_message_from_xy(map_points_xy)
        self._live_map_pub.publish(message)

    def _publish_full_map(self) -> None:
        if self._last_corrected_header is None:
            return
        map_points_xy = self._fusion.full_map_points_world(
            max_points=self._full_map_max_points,
        )
        message = self._pointcloud_message_from_xy(map_points_xy)
        self._full_map_pub.publish(message)

    def _pointcloud_message_from_xy(self, map_points_xy: np.ndarray) -> PointCloud2:
        message = PointCloud2()
        message.header = self._last_corrected_header
        message.header.frame_id = self._odom_frame
        message.height = 1
        message.width = int(map_points_xy.shape[0])
        message.is_bigendian = False
        message.is_dense = True
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.point_step = 12
        message.row_step = message.point_step * message.width
        if map_points_xy.size == 0:
            message.data = b""
        else:
            cloud = np.zeros(
                (map_points_xy.shape[0],),
                dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
            )
            cloud["x"] = map_points_xy[:, 0].astype(np.float32)
            cloud["y"] = map_points_xy[:, 1].astype(np.float32)
            cloud["z"] = 0.0
            message.data = cloud.tobytes()
        return message

    def _publish_status(self) -> None:
        snapshot = self._fusion.status_snapshot()
        payload = {
            "estimation_backend": self._estimation_backend,
            "state": snapshot.state,
            "imu_initialized": snapshot.imu_initialized,
            "alignment_ready": snapshot.alignment_ready,
            "best_effort_init": snapshot.best_effort_init,
            "raw_imu_sample_count": snapshot.raw_imu_sample_count,
            "processed_imu_sample_count": snapshot.processed_imu_sample_count,
            "pending_scan_count": snapshot.pending_scan_count,
            "processed_scan_count": snapshot.processed_scan_count,
            "initial_scan_count": snapshot.initial_scan_count,
            "alignment_yaw_rad": snapshot.alignment_yaw_rad,
            "origin_projection_m": list(snapshot.origin_projection_m),
            "static_initialization": snapshot.static_initialization,
            "corridor_model": snapshot.corridor_model,
            "quality": snapshot.quality,
            "latest_pose": snapshot.latest_pose,
            "published_pose": self._last_published_pose_payload,
            "published_source": self._last_published_source,
            "odom_prediction_age_s": self._last_prediction_age_s,
            "parameters": snapshot.parameters,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self._last_status_payload = message.data
        self._status_pub.publish(message)


def main() -> None:
    rclpy.init()
    node = ImuLidarPlanarFusionNode()
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
