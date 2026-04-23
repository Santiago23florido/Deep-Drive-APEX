from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_scenario(config_path: str, scenario_name: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    defaults = payload.get("defaults", {})
    scenarios = payload.get("scenarios", {})
    if not isinstance(defaults, dict) or not isinstance(scenarios, dict):
        return {}
    scenario = scenarios.get(scenario_name, {})
    if not isinstance(scenario, dict):
        return dict(defaults)
    return _deep_merge(defaults, scenario)


def _node_ros_parameters(all_params: dict[str, Any], node_name: str) -> dict[str, Any]:
    node_entry = all_params.get(node_name, {})
    if not isinstance(node_entry, dict):
        return {}
    ros_params = node_entry.get("ros__parameters", {})
    if not isinstance(ros_params, dict):
        return {}
    return dict(ros_params)


def _prepare_launch(context, *args, **kwargs):
    del args, kwargs
    rc_share = Path(get_package_share_directory("rc_sim_description"))
    apex_share = Path(get_package_share_directory("apex_telemetry"))
    slam_share = Path(get_package_share_directory("slam_toolbox"))
    sim_root = Path(
        os.environ.get("APEX_SIM_ROOT", str(rc_share.parents[4]))
    ).expanduser().resolve()

    scenario_name = LaunchConfiguration("scenario").perform(context)
    scenario_config_path = LaunchConfiguration("scenario_config").perform(context)
    params_file = Path(LaunchConfiguration("params_file").perform(context)).expanduser().resolve()
    slam_params_file = Path(
        LaunchConfiguration("slam_params_file").perform(context)
    ).expanduser().resolve()
    use_slam = (
        LaunchConfiguration("use_slam").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    estimation_mode_raw = (
        LaunchConfiguration("estimation_mode").perform(context).strip().lower()
    )
    mapping_mode = (
        LaunchConfiguration("mapping_mode").perform(context).strip().lower()
        or "current"
    )
    distortion_profile_raw = (
        LaunchConfiguration("distortion_profile").perform(context).strip().lower()
    )
    refinement_mode_raw = (
        LaunchConfiguration("refinement_mode").perform(context).strip().lower()
    )
    offline_replay_mode = (
        LaunchConfiguration("offline_replay_mode").perform(context).strip().lower()
        or "live_buffer"
    )
    sim_max_linear_speed_mps_value = (
        LaunchConfiguration("sim_max_linear_speed_mps").perform(context).strip()
    )
    use_ideal_pose_for_slam = (
        LaunchConfiguration("use_ideal_pose_for_slam").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    use_ideal_lidar_for_slam = (
        LaunchConfiguration("use_ideal_lidar_for_slam").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    degrade_odom = (
        LaunchConfiguration("degrade_odom").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    odom_position_noise_std = float(
        LaunchConfiguration("odom_position_noise_std").perform(context).strip() or "0.0"
    )
    odom_yaw_noise_std = float(
        LaunchConfiguration("odom_yaw_noise_std").perform(context).strip() or "0.0"
    )
    odom_velocity_noise_std = float(
        LaunchConfiguration("odom_velocity_noise_std").perform(context).strip() or "0.0"
    )
    odom_yaw_bias_per_sec = float(
        LaunchConfiguration("odom_yaw_bias_per_sec").perform(context).strip() or "0.0"
    )
    odom_latency_sec = float(
        LaunchConfiguration("odom_latency_sec").perform(context).strip() or "0.0"
    )
    window_scan_count_value = LaunchConfiguration("window_scan_count").perform(context).strip()
    window_overlap_count_value = LaunchConfiguration("window_overlap_count").perform(context).strip()
    initial_scan_count_value = LaunchConfiguration("initial_scan_count").perform(context).strip()
    submap_window_scans_value = LaunchConfiguration("submap_window_scans").perform(context).strip()
    offline_point_stride_value = LaunchConfiguration("point_stride").perform(context).strip()
    offline_max_correspondence_m_value = (
        LaunchConfiguration("max_correspondence_m").perform(context).strip()
    )
    window_scan_count = int(window_scan_count_value or "48")
    window_overlap_count = int(window_overlap_count_value or "16")
    initial_scan_count = int(initial_scan_count_value or "24")
    submap_window_scans = int(submap_window_scans_value or "8")
    offline_point_stride = int(offline_point_stride_value or "2")
    offline_max_correspondence_m = float(
        offline_max_correspondence_m_value or "0.35"
    )
    offline_update_period_sec = float(
        LaunchConfiguration("offline_update_period_sec").perform(context).strip() or "0.5"
    )
    sim_max_linear_speed_mps = (
        float(sim_max_linear_speed_mps_value) if sim_max_linear_speed_mps_value else None
    )
    offline_seed_odom_topic_override = (
        LaunchConfiguration("offline_seed_odom_topic").perform(context).strip()
    )
    offline_input_dir = LaunchConfiguration("offline_input_dir").perform(context).strip()
    online_use_external_prior = (
        LaunchConfiguration("online_use_external_prior").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    online_prior_odom_topic = (
        LaunchConfiguration("online_prior_odom_topic").perform(context).strip()
    )
    online_external_prior_weight_xy = float(
        LaunchConfiguration("online_external_prior_weight_xy").perform(context).strip() or "0.45"
    )
    online_external_prior_weight_yaw = float(
        LaunchConfiguration("online_external_prior_weight_yaw").perform(context).strip() or "0.85"
    )
    online_freeze_scan_insertion_on_low_confidence = (
        LaunchConfiguration("online_freeze_scan_insertion_on_low_confidence")
        .perform(context)
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    online_low_confidence_residual_threshold_m = float(
        LaunchConfiguration("online_low_confidence_residual_threshold_m")
        .perform(context)
        .strip()
        or "0.14"
    )
    online_low_confidence_inlier_ratio_threshold = float(
        LaunchConfiguration("online_low_confidence_inlier_ratio_threshold")
        .perform(context)
        .strip()
        or "0.24"
    )
    online_use_offline_correction = (
        LaunchConfiguration("online_use_offline_correction").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    online_offline_correction_blend = float(
        LaunchConfiguration("online_offline_correction_blend").perform(context).strip() or "0.25"
    )
    online_offline_correction_max_jump_xy = float(
        LaunchConfiguration("online_offline_correction_max_jump_xy").perform(context).strip() or "0.30"
    )
    online_offline_correction_max_jump_yaw = float(
        LaunchConfiguration("online_offline_correction_max_jump_yaw").perform(context).strip() or "0.18"
    )
    online_use_offline_reference = (
        LaunchConfiguration("online_use_offline_reference").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    symmetric_steering_for_mapping = (
        LaunchConfiguration("symmetric_steering_for_mapping")
        .perform(context)
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    offline_publish_global_correction = (
        LaunchConfiguration("offline_publish_global_correction").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    record_run = (
        LaunchConfiguration("record_run").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    record_run_base_dir_value = (
        LaunchConfiguration("record_run_base_dir").perform(context).strip()
    )
    record_run_name_value = LaunchConfiguration("record_run_name").perform(context).strip()
    use_track_geometry_prior = (
        LaunchConfiguration("use_track_geometry_prior").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    track_geometry_file_value = LaunchConfiguration("track_geometry_file").perform(context).strip()
    track_geometry_weight = float(
        LaunchConfiguration("track_geometry_weight").perform(context).strip() or "0.08"
    )
    if track_geometry_file_value:
        candidate_track_geometry = Path(track_geometry_file_value).expanduser()
        if not candidate_track_geometry.is_absolute():
            config_relative = (rc_share / "config" / candidate_track_geometry).resolve()
            repo_relative = candidate_track_geometry.resolve()
            candidate_track_geometry = config_relative if config_relative.exists() else repo_relative
        track_geometry_file = str(candidate_track_geometry.resolve())
    else:
        track_geometry_file = ""
    lidar_noise_std_value = LaunchConfiguration("lidar_noise_std").perform(context).strip()
    imu_gyro_noise_stddev_rps_value = LaunchConfiguration(
        "imu_gyro_noise_stddev_rps"
    ).perform(context).strip()
    imu_accel_noise_stddev_mps2_value = LaunchConfiguration(
        "imu_accel_noise_stddev_mps2"
    ).perform(context).strip()
    if mapping_mode not in {"current", "ideal"}:
        mapping_mode = "current"
    if estimation_mode_raw in {"current", "ideal", "rf2o_ekf"}:
        estimation_mode = estimation_mode_raw
    else:
        estimation_mode = mapping_mode
    if estimation_mode not in {"current", "ideal", "rf2o_ekf"}:
        estimation_mode = "current"
    current_estimation_mode = estimation_mode == "current"
    ideal_estimation_mode = estimation_mode == "ideal"
    rf2o_ekf_estimation_mode = estimation_mode == "rf2o_ekf"
    if rf2o_ekf_estimation_mode:
        missing_packages: list[str] = []
        for package_name in ("rf2o_laser_odometry", "robot_localization"):
            try:
                get_package_prefix(package_name)
            except PackageNotFoundError:
                missing_packages.append(package_name)
        if missing_packages:
            install_hints = [
                "robot_localization: sudo apt install ros-jazzy-robot-localization",
                "rf2o_laser_odometry: add the ROS 2 package to an overlay workspace "
                "(it is not present in the current Jazzy apt sources on this machine)",
            ]
            raise RuntimeError(
                "estimation_mode=rf2o_ekf requires missing ROS 2 packages: "
                + ", ".join(missing_packages)
                + ". Install hints: "
                + " | ".join(install_hints)
            )
    if distortion_profile_raw in {"ideal", "medium"}:
        distortion_profile = distortion_profile_raw
    elif rf2o_ekf_estimation_mode:
        distortion_profile = "medium"
    else:
        distortion_profile = "ideal"
    if refinement_mode_raw not in {"none", "offline_submaps"}:
        refinement_mode = "none"
    else:
        refinement_mode = refinement_mode_raw
    offline_submaps_refinement_mode = refinement_mode == "offline_submaps"
    medium_distortion_profile = distortion_profile == "medium"
    ideal_medium_sensor_profile = ideal_estimation_mode and medium_distortion_profile
    offline_online_fusion_seed_mode = (
        offline_submaps_refinement_mode
        and ideal_medium_sensor_profile
        and not offline_seed_odom_topic_override
    )
    distance_field_online_seed_mode = offline_online_fusion_seed_mode
    if distance_field_online_seed_mode:
        # For the noisy-sensor sim profile, keep the frontend as close as
        # possible to the real-car setup: only IMU + LiDAR, no extra priors or
        # backend feedback injected into the online pose estimate.
        online_use_external_prior = False
        online_prior_odom_topic = ""
        online_use_offline_correction = False
        online_use_offline_reference = False
    if ideal_medium_sensor_profile and offline_submaps_refinement_mode:
        if window_scan_count == 48:
            window_scan_count = 48
        if window_overlap_count == 16:
            window_overlap_count = 24
        if initial_scan_count == 24:
            initial_scan_count = 24
        if submap_window_scans == 8:
            submap_window_scans = 10
        if offline_point_stride == 2:
            offline_point_stride = 1
        if abs(offline_max_correspondence_m - 0.35) <= 1.0e-9:
            offline_max_correspondence_m = 0.30
    slam_uses_ideal_pose = (
        not rf2o_ekf_estimation_mode
        and (ideal_estimation_mode or use_ideal_pose_for_slam)
    )
    slam_uses_ideal_lidar = (
        not rf2o_ekf_estimation_mode
        and (ideal_estimation_mode or use_ideal_lidar_for_slam)
    )
    if offline_seed_odom_topic_override:
        offline_seed_odom_topic = offline_seed_odom_topic_override
    elif rf2o_ekf_estimation_mode:
        offline_seed_odom_topic = "/apex/sim/odom_fused"
    elif offline_online_fusion_seed_mode:
        # In the ideal+medium offline case, seed the submap optimizer from the
        # same online IMU+LiDAR fusion used on the real car, not from perfect
        # simulated odometry.
        offline_seed_odom_topic = "/apex/odometry/imu_lidar_fused"
    elif slam_uses_ideal_pose:
        offline_seed_odom_topic = "/apex/sim/ideal_odom"
    elif current_estimation_mode:
        offline_seed_odom_topic = "/apex/odometry/imu_lidar_fused"
    else:
        offline_seed_odom_topic = ""
    if offline_seed_odom_topic == "/apex/odometry/imu_lidar_fused":
        offline_seed_status_topic = "/apex/estimation/status"
    else:
        offline_seed_status_topic = ""
    if medium_distortion_profile and (rf2o_ekf_estimation_mode or ideal_estimation_mode):
        default_lidar_noise_std = 0.010
        default_imu_gyro_noise_stddev_rps = 0.014
        default_imu_accel_noise_stddev_mps2 = 0.09
    else:
        default_lidar_noise_std = 0.0
        default_imu_gyro_noise_stddev_rps = 0.002
        default_imu_accel_noise_stddev_mps2 = 0.02
    lidar_noise_std = float(lidar_noise_std_value or str(default_lidar_noise_std))
    imu_gyro_noise_stddev_rps = float(
        imu_gyro_noise_stddev_rps_value or str(default_imu_gyro_noise_stddev_rps)
    )
    imu_accel_noise_stddev_mps2 = float(
        imu_accel_noise_stddev_mps2_value or str(default_imu_accel_noise_stddev_mps2)
    )
    control_mode = (
        LaunchConfiguration("control_mode").perform(context).strip().lower()
        or "recognition_tour"
    )
    gazebo_gui = (
        LaunchConfiguration("gazebo_gui").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    use_refined_visual_map = (
        LaunchConfiguration("use_refined_visual_map").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    fixed_map_run_value = LaunchConfiguration("fixed_map_run").perform(context).strip()
    fixed_map_run_dir = (
        Path(fixed_map_run_value).expanduser().resolve() if fixed_map_run_value else None
    )
    rviz_config_value = LaunchConfiguration("rviz_config").perform(context).strip()
    if rviz_config_value:
        rviz_config = Path(rviz_config_value).expanduser().resolve()
    elif offline_submaps_refinement_mode:
        rviz_config = (rc_share / "rviz" / "apex_offline_submap_refiner.rviz").resolve()
    elif rf2o_ekf_estimation_mode:
        rviz_config = (rc_share / "rviz" / "apex_manual_mapping_rf2o_ekf.rviz").resolve()
    elif ideal_estimation_mode:
        rviz_config = (rc_share / "rviz" / "apex_manual_mapping_ideal.rviz").resolve()
    else:
        if fixed_map_run_dir is not None:
            default_rviz_name = "apex_fixed_map_live.rviz"
        elif use_refined_visual_map:
            default_rviz_name = "apex_recognition_refined_map_live.rviz"
        elif control_mode in {"manual_xbox", "manual_windows_bridge"}:
            default_rviz_name = "apex_manual_mapping_live.rviz"
        else:
            default_rviz_name = (
                "apex_recognition_slam_live.rviz" if use_slam else "apex_recognition_live.rviz"
            )
        rviz_config = sim_root / "rviz" / default_rviz_name
        if not rviz_config.exists():
            rviz_config = sim_root / "rviz" / "apex_recognition_live.rviz"
        if not rviz_config.exists():
            rviz_config = rc_share / "config" / "rviz" / "rc_car.rviz"
    rviz_config = rviz_config.expanduser().resolve()
    scenario = _load_scenario(scenario_config_path, scenario_name)

    try:
        base_params = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    except Exception:
        base_params = {}
    params_overrides = scenario.get("apex_params_overrides", {})
    if not isinstance(params_overrides, dict):
        params_overrides = {}
    merged_params = _deep_merge(base_params, params_overrides)
    if fixed_map_run_dir is not None:
        fixed_map_dir = fixed_map_run_dir / "fixed_map"
        fixed_map_yaml = fixed_map_dir / "fixed_map.yaml"
        fixed_map_distance_npy = fixed_map_dir / "fixed_map_distance.npy"
        fixed_map_visual_points_csv = fixed_map_dir / "fixed_map_visual_points.csv"
        merged_params = _deep_merge(
            merged_params,
            {
                "imu_lidar_planar_fusion_node": {
                    "ros__parameters": {
                        "estimation_backend": "fixed_map",
                        "fixed_map_yaml": str(fixed_map_yaml),
                        "fixed_map_distance_npy": str(fixed_map_distance_npy),
                        "fixed_map_visual_points_csv": str(fixed_map_visual_points_csv),
                        "publish_tf": True,
                    }
                }
            },
        )
    if control_mode in {"manual_xbox", "manual_windows_bridge"}:
        merged_params = _deep_merge(
            merged_params,
            {
                "cmd_vel_to_apex_actuation_node": {
                    "ros__parameters": {
                        "max_linear_speed_mps": 1.50,
                        "min_effective_speed_pct": 8.8,
                        "max_speed_pct": 42.0,
                        "launch_boost_speed_pct": 8.8,
                        "launch_boost_hold_s": 0.08,
                    }
                }
            },
        )
    if offline_online_fusion_seed_mode:
        merged_params = _deep_merge(
            merged_params,
            {
                "imu_lidar_planar_fusion_node": {
                    "ros__parameters": {
                        "submap_window_scans": 10,
                        "point_stride": 1,
                        "max_correspondence_m": 0.28,
                        "max_initial_alignment_scans": 8,
                        "min_valid_correspondence_count": 20,
                        "low_confidence_residual_m": 0.12,
                        "max_scan_optimization_evals": 100,
                        "publish_predicted_odom_between_scans": True,
                        "predicted_odom_rate_hz": 30.0,
                        "max_prediction_horizon_s": 0.18,
                    }
                }
            },
        )
    if use_slam and current_estimation_mode:
        merged_params = _deep_merge(
            merged_params,
            {
                "kinematics_odometry_node": {
                    "ros__parameters": {
                        "publish_tf": False,
                    }
                },
                "imu_lidar_planar_fusion_node": {
                    "ros__parameters": {
                        "publish_tf": True,
                    }
                },
            },
        )

    selected_slam_params_file = (
        (rc_share / "config" / "slam_toolbox_sim_ideal.yaml").resolve()
        if ideal_estimation_mode
        else (
            (rc_share / "config" / "slam_toolbox_sim_rf2o_ekf.yaml").resolve()
            if rf2o_ekf_estimation_mode
            else slam_params_file
        )
    )
    try:
        base_slam_params = yaml.safe_load(
            selected_slam_params_file.read_text(encoding="utf-8")
        ) or {}
    except Exception:
        base_slam_params = {}
    slam_params_overrides = (
        {}
        if not current_estimation_mode
        else scenario.get("slam_params_overrides", {})
    )
    if not isinstance(slam_params_overrides, dict):
        slam_params_overrides = {}
    merged_slam_params = _deep_merge(base_slam_params, slam_params_overrides)
    ideal_slam_overrides: dict[str, Any] = {"slam_toolbox": {"ros__parameters": {}}}
    ideal_slam_ros_params = ideal_slam_overrides["slam_toolbox"]["ros__parameters"]
    # In ideal mapping mode slam_toolbox consumes Gazebo's perfect scan directly
    # instead of the degraded LiDAR topics published by the telemetry pipeline.
    if slam_uses_ideal_lidar or rf2o_ekf_estimation_mode:
        ideal_slam_ros_params["scan_topic"] = "/apex/sim/scan"
    # slam_toolbox tracks motion through TF, so the ideal Gazebo ground truth is
    # bridged into the standard odom -> base_link frames for this first mapping stage.
    if slam_uses_ideal_pose or rf2o_ekf_estimation_mode:
        ideal_slam_ros_params["map_frame"] = "map"
        ideal_slam_ros_params["odom_frame"] = "odom"
        ideal_slam_ros_params["base_frame"] = "base_link"
    if ideal_slam_ros_params:
        merged_slam_params = _deep_merge(merged_slam_params, ideal_slam_overrides)

    temp_dir = Path(tempfile.gettempdir()) / "apex_sim"
    temp_dir.mkdir(parents=True, exist_ok=True)
    merged_params_path = temp_dir / f"apex_sim_{scenario_name}_params.yaml"
    merged_params_path.write_text(
        yaml.safe_dump(merged_params, sort_keys=False),
        encoding="utf-8",
    )
    merged_slam_params_path = temp_dir / f"apex_sim_{scenario_name}_slam_params.yaml"
    merged_slam_params_path.write_text(
        yaml.safe_dump(merged_slam_params, sort_keys=False),
        encoding="utf-8",
    )

    world_filename = str(scenario.get("world", "basic_track.world"))
    world_path = str((rc_share / "worlds" / world_filename).resolve())
    if record_run_base_dir_value:
        candidate_record_base_dir = Path(record_run_base_dir_value).expanduser()
        if not candidate_record_base_dir.is_absolute():
            candidate_record_base_dir = (sim_root / candidate_record_base_dir).resolve()
        record_run_base_dir = candidate_record_base_dir.resolve()
    else:
        record_run_base_dir = (sim_root / "data" / "rc_sim_description" / "runs").resolve()
    record_run_name = record_run_name_value.strip()
    if not record_run_name:
        record_run_name = f"manual_lap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_run_name = "".join(
        char if (char.isalnum() or char in {"-", "_"}) else "_"
        for char in record_run_name
    ).strip("_") or "manual_lap"
    record_run_dir = (record_run_base_dir / record_run_name).resolve()
    spawn = scenario.get("spawn", {}) if isinstance(scenario.get("spawn"), dict) else {}
    spawn_x = float(spawn.get("x", 0.0))
    spawn_y = float(spawn.get("y", -2.5))
    spawn_z = float(spawn.get("z", 0.02))
    spawn_yaw_deg = float(spawn.get("yaw_deg", 0.0))

    x_override = LaunchConfiguration("x").perform(context).strip()
    y_override = LaunchConfiguration("y").perform(context).strip()
    z_override = LaunchConfiguration("z").perform(context).strip()
    yaw_override = LaunchConfiguration("yaw_deg").perform(context).strip()
    if x_override:
        spawn_x = float(x_override)
    if y_override:
        spawn_y = float(y_override)
    if z_override:
        spawn_z = float(z_override)
    if yaw_override:
        spawn_yaw_deg = float(yaw_override)
    spawn_yaw_rad = math.radians(spawn_yaw_deg)

    robot_xacro = str((rc_share / "urdf" / "rc_car.urdf.xacro").resolve())
    pipeline_launch = str((apex_share / "launch" / "apex_pipeline.launch.py").resolve())
    reference_script_path = str((sim_root / "tools" / "analysis" / "sensor_fusionn.py").resolve())

    gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_value = f"{rc_share}:{gz_resource_path}" if gz_resource_path else str(rc_share)
    pythonpath_parts = ["/usr/lib/python3/dist-packages"]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    pythonpath_value = ":".join(pythonpath_parts)
    node_python_env = {"PYTHONPATH": pythonpath_value}
    software_gui_env: dict[str, str] = {}
    for env_name in (
        "LIBGL_ALWAYS_SOFTWARE",
        "QT_XCB_GL_INTEGRATION",
        "QT_QUICK_BACKEND",
    ):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            software_gui_env[env_name] = env_value
    sim_gui_env = dict(node_python_env)
    sim_gui_env.update(software_gui_env)

    vehicle_bridge_params = scenario.get("vehicle_bridge", {})
    if not isinstance(vehicle_bridge_params, dict):
        vehicle_bridge_params = {}
    vehicle_bridge_params = dict(vehicle_bridge_params)
    if control_mode in {"manual_xbox", "manual_windows_bridge"}:
        vehicle_bridge_params["motor_max_forward_speed_mps"] = max(
            1.50, float(vehicle_bridge_params.get("motor_max_forward_speed_mps", 0.0))
        )
        vehicle_bridge_params["motor_accel_limit_mps2"] = max(
            1.80, float(vehicle_bridge_params.get("motor_accel_limit_mps2", 0.0))
        )
        vehicle_bridge_params["motor_decel_limit_mps2"] = max(
            2.40, float(vehicle_bridge_params.get("motor_decel_limit_mps2", 0.0))
        )
    if symmetric_steering_for_mapping and (
        ideal_estimation_mode or offline_submaps_refinement_mode
    ):
        symmetric_ratio = float(vehicle_bridge_params.get("steering_left_ratio", 1.0))
        vehicle_bridge_params["steering_left_ratio"] = symmetric_ratio
        vehicle_bridge_params["steering_right_ratio"] = symmetric_ratio
    vehicle_bridge_params.setdefault("use_sim_time", True)

    ground_truth_params = scenario.get("ground_truth", {})
    if not isinstance(ground_truth_params, dict):
        ground_truth_params = {}
    ground_truth_params = dict(ground_truth_params)
    ground_truth_params.setdefault("use_sim_time", True)
    ground_truth_params.setdefault("world_name", "default")
    ground_truth_params.setdefault("model_name", "rc_car")
    ground_truth_params.setdefault("world_path", world_path)
    if distance_field_online_seed_mode:
        ground_truth_params["fusion_odom_topic"] = "/apex/odometry/imu_lidar_fused"
    cmd_vel_bridge_ros_params = _node_ros_parameters(
        merged_params,
        "cmd_vel_to_apex_actuation_node",
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                robot_xacro,
                " lidar_noise_std:=",
                str(lidar_noise_std),
                " imu_gyro_noise_stddev_rps:=",
                str(imu_gyro_noise_stddev_rps),
                " imu_accel_noise_stddev_mps2:=",
                str(imu_accel_noise_stddev_mps2),
            ]
        ),
        value_type=str,
    )
    gz_cmd = ["gz", "sim", "-r"]
    if not gazebo_gui:
        gz_cmd.append("-s")
    gz_cmd.append(world_path)
    gz_sim = ExecuteProcess(
        cmd=gz_cmd,
        output="screen",
        additional_env=sim_gui_env,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="rc_car_state_publisher",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        output="screen",
    )

    spawn_rc = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "rc_car",
            "-topic",
            "robot_description",
            "-x",
            str(spawn_x),
            "-y",
            str(spawn_y),
            "-z",
            str(spawn_z),
            "-Y",
            str(spawn_yaw_rad),
        ],
        output="screen",
    )

    sim_bridges = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="apex_sim_bridges",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/apex/sim/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/apex/sim/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        parameters=[{"use_sim_time": True}],
    )

    vehicle_bridge = Node(
        package="rc_sim_description",
        executable="apex_gz_vehicle_bridge.py",
        name="apex_gz_vehicle_bridge",
        output="screen",
        parameters=[vehicle_bridge_params],
        additional_env={"PYTHONPATH": pythonpath_value},
    )

    ground_truth = Node(
        package="rc_sim_description",
        executable="apex_ground_truth_node.py",
        name="apex_ground_truth_node",
        output="screen",
        parameters=[ground_truth_params],
        additional_env=node_python_env,
    )
    ground_truth_tf_bridge_params = {
        "use_sim_time": True,
        "source_odom_topic": "/apex/sim/ground_truth/odom",
        "ideal_odom_topic": "/apex/sim/ideal_odom",
        "odom_frame_id": "odom",
        "child_frame_id": "base_link",
    }
    if ideal_estimation_mode:
        ground_truth_tf_bridge_params.update(
            {
                "degrade_odom": degrade_odom,
                # Position noise is applied only in x/y to keep the planar
                # mapping experiments focused on odom drift, not sensor height.
                "odom_position_noise_std": odom_position_noise_std,
                # Yaw noise perturbs the pose heading before the TF is published.
                "odom_yaw_noise_std": odom_yaw_noise_std,
                # Velocity noise perturbs twist terms without touching the scan.
                "odom_velocity_noise_std": odom_velocity_noise_std,
                # Bias accumulates slowly over time to emulate heading drift.
                "odom_yaw_bias_per_sec": odom_yaw_bias_per_sec,
                # Latency delays the odom + TF publication as one unit.
                "odom_latency_sec": odom_latency_sec,
            }
        )
    ground_truth_tf_bridge = Node(
        package="rc_sim_description",
        executable="apex_ground_truth_tf_bridge.py",
        name="apex_ground_truth_tf_bridge",
        output="screen",
        parameters=[ground_truth_tf_bridge_params],
        additional_env=node_python_env,
    )
    fixed_map_publisher = None
    if fixed_map_run_dir is not None:
        fixed_map_dir = fixed_map_run_dir / "fixed_map"
        fixed_map_publisher = Node(
            package="rc_sim_description",
            executable="apex_general_track_map_publisher.py",
            name="apex_fixed_map_publisher",
            output="screen",
            parameters=[
                {
                    "map_yaml": str((fixed_map_dir / "fixed_map.yaml").resolve()),
                    "summary_json": str((fixed_map_dir / "mapping_summary.json").resolve()),
                    "frame_id": "odom_imu_lidar_fused",
                    "grid_topic": "/apex/sim/fixed_map/grid",
                    "visual_points_topic": "/apex/sim/fixed_map/visual_points",
                    "path_topic": "/apex/sim/fixed_map/path",
                    "status_topic": "/apex/sim/fixed_map/status",
                    "reload_on_change": False,
                    "allow_missing_inputs": False,
                    "point_stride": 1,
                }
            ],
            additional_env=node_python_env,
        )

    refined_visual_map = None
    if current_estimation_mode:
        refined_visual_map = Node(
            package="rc_sim_description",
            executable="apex_refined_sensorfusion_map_node.py",
            name="apex_refined_sensorfusion_map_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "reference_script_path": reference_script_path,
                    "scan_topic": "/lidar/scan_localization",
                    "odom_topic": "/apex/odometry/imu_lidar_fused",
                }
            ],
            additional_env=node_python_env,
            condition=IfCondition(LaunchConfiguration("use_refined_visual_map")),
        )

    manual_teleop = Node(
        package="rc_sim_description",
        executable="apex_xbox_manual_teleop_node.py",
        name="apex_xbox_manual_teleop_node",
        output="screen",
        parameters=[
            merged_params_path,
            {
                "cmd_vel_topic": "/apex/cmd_vel_track",
                "status_topic": "/apex/sim/manual_control/status",
            },
        ],
        additional_env=node_python_env,
    )
    manual_windows_bridge = Node(
        package="rc_sim_description",
        executable="apex_windows_gamepad_bridge_node.py",
        name="apex_windows_gamepad_bridge_node",
        output="screen",
        parameters=[
            merged_params_path,
            {
                "cmd_vel_topic": "/apex/cmd_vel_track",
                "status_topic": "/apex/sim/manual_control/status",
            },
        ],
        additional_env=node_python_env,
    )
    ideal_cmd_vel_bridge = Node(
        package="rc_sim_description",
        executable="apex_cmd_vel_to_sim_pwm_node.py",
        name="apex_cmd_vel_to_sim_pwm_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "cmd_vel_topic": "/apex/cmd_vel_track",
                "motor_pwm_topic": "/apex/sim/pwm/motor_dc",
                "steering_pwm_topic": "/apex/sim/pwm/steering_dc",
                "wheelbase_m": float(vehicle_bridge_params.get("wheelbase_m", 0.30)),
                "steering_limit_deg": float(vehicle_bridge_params.get("steering_limit_deg", 18.0)),
                "steering_left_ratio": float(vehicle_bridge_params.get("steering_left_ratio", 1.0)),
                "steering_right_ratio": float(vehicle_bridge_params.get("steering_right_ratio", 0.96)),
                "steering_dc_min": float(vehicle_bridge_params.get("steering_dc_min", 5.0)),
                "steering_dc_max": float(vehicle_bridge_params.get("steering_dc_max", 8.6)),
                "steering_center_trim_dc": float(
                    vehicle_bridge_params.get("steering_center_trim_dc", 1.4)
                ),
                "steering_direction_sign": float(
                    vehicle_bridge_params.get("steering_direction_sign", -1.0)
                ),
                "steering_min_authority_ratio": float(
                    vehicle_bridge_params.get("steering_min_authority_ratio", 0.90)
                ),
                # The direct Gazebo PWM bridge keeps manual control alive in the
                # sim-only estimation modes without pulling in the old pipeline.
                "max_linear_speed_mps": float(
                    sim_max_linear_speed_mps
                    if sim_max_linear_speed_mps is not None
                    else cmd_vel_bridge_ros_params.get(
                        "max_linear_speed_mps",
                        vehicle_bridge_params.get("motor_max_forward_speed_mps", 0.60),
                    )
                ),
                "motor_neutral_dc": float(vehicle_bridge_params.get("motor_neutral_dc", 7.5)),
                "motor_forward_deadband_dc": float(
                    vehicle_bridge_params.get("motor_forward_deadband_dc", 7.72)
                ),
                "motor_forward_top_dc": float(
                    vehicle_bridge_params.get("motor_forward_top_dc", 8.55)
                ),
            }
        ],
        additional_env=node_python_env,
    )

    enable_recognition_tour = control_mode not in {"manual_xbox", "manual_windows_bridge"}

    sim_only_cmd_vel_bridge_required = (
        ideal_estimation_mode
        or rf2o_ekf_estimation_mode
        or control_mode in {"manual_xbox", "manual_windows_bridge"}
    )

    apex_pipeline = None
    if current_estimation_mode or (offline_online_fusion_seed_mode and not distance_field_online_seed_mode):
        apex_pipeline = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pipeline_launch),
            launch_arguments={
                "params_file": str(merged_params_path),
                "use_sim_time": "true",
                "enable_laser_tf": "false",
                "imu_transport_backend": "sim_imu",
                "sim_imu_topic": "/apex/sim/imu",
                "lidar_source_backend": "sim_scan",
                "sim_scan_topic": "/apex/sim/scan",
                "actuation_backend": "sim_pwm_topic",
                "sim_motor_pwm_topic": "/apex/sim/pwm/motor_dc",
                "sim_steering_pwm_topic": "/apex/sim/pwm/steering_dc",
                "enable_imu_lidar_fusion": "true",
                "enable_imu_source": "true",
                "enable_lidar_source": "true",
                "enable_kinematics_estimator": (
                    "true" if current_estimation_mode else "false"
                ),
                "enable_kinematics_odometry": (
                    "true" if current_estimation_mode else "false"
                ),
                "enable_curve_entry_planner": (
                    "true" if current_estimation_mode and enable_recognition_tour else "false"
                ),
                "enable_path_tracker": "false",
                "enable_recognition_tour_planner": (
                    "true"
                    if current_estimation_mode and enable_recognition_tour
                    else "false"
                ),
                "enable_recognition_tour_tracker": (
                    "true"
                    if current_estimation_mode and enable_recognition_tour
                    else "false"
                ),
                "enable_cmdvel_actuation_bridge": (
                    "true" if current_estimation_mode else "false"
                ),
            }.items(),
        )

    online_distance_field_seed = None
    if distance_field_online_seed_mode:
        online_distance_field_seed = Node(
            package="rc_sim_description",
            executable="online_distance_field_seed_node.py",
            name="online_distance_field_seed_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "scan_topic": "/apex/sim/scan",
                    "imu_topic": "/apex/sim/imu",
                    "odom_topic": "/apex/odometry/imu_lidar_fused",
                    "path_topic": "/apex/estimation/path",
                    "pose_topic": "/apex/estimation/current_pose",
                    "live_map_topic": "/apex/estimation/live_map_points",
                    "corrected_odom_topic": "/apex/estimation/odom_corrected",
                    "corrected_path_topic": "/apex/estimation/path_corrected",
                    "corrected_pose_topic": "/apex/estimation/current_pose_corrected",
                    "corrected_live_map_topic": "/apex/estimation/live_map_points_corrected",
                    "status_topic": "/apex/estimation/status",
                    "odom_frame_id": "odom_imu_lidar_fused",
                    "corrected_frame_id": "map",
                    "child_frame_id": "base_link",
                    "prior_odom_topic": online_prior_odom_topic,
                    "use_external_prior": online_use_external_prior,
                    "external_prior_weight_xy": online_external_prior_weight_xy,
                    "external_prior_weight_yaw": online_external_prior_weight_yaw,
                    "freeze_scan_insertion_on_low_confidence": online_freeze_scan_insertion_on_low_confidence,
                    "low_confidence_residual_threshold_m": online_low_confidence_residual_threshold_m,
                    "low_confidence_inlier_ratio_threshold": online_low_confidence_inlier_ratio_threshold,
                    "use_offline_correction": online_use_offline_correction,
                    "offline_correction_topic": "/apex/sim/offline_global_correction",
                    "offline_correction_use_yaw": False,
                    "offline_correction_blend": online_offline_correction_blend,
                    "offline_correction_max_jump_xy": online_offline_correction_max_jump_xy,
                    "offline_correction_max_jump_yaw": online_offline_correction_max_jump_yaw,
                    "use_offline_submap_as_reference": online_use_offline_reference,
                    "use_offline_grid_as_reference": online_use_offline_reference,
                    "offline_submap_topic": "/apex/sim/offline_current_submap",
                    "offline_grid_topic": "/apex/sim/offline_refined_grid",
                    "submap_window_scans": 24,
                    "point_stride": 1,
                    "max_correspondence_m": 0.24,
                    "local_prior_weight_xy": 0.06,
                    "local_prior_weight_yaw": 0.20,
                    "lidar_residual_weight": 0.85,
                    "max_scan_optimization_evals": 100,
                    "correlative_search_forward_extent_m": 1.20,
                    "correlative_search_lateral_extent_m": 0.35,
                    "correlative_search_step_m": 0.10,
                    "correlative_search_yaw_extent_rad": 0.14,
                    "correlative_search_yaw_step_rad": 0.05,
                    "correlative_search_top_k": 5,
                    "low_confidence_pose_blend": 0.35,
                    "low_confidence_pose_max_jump_xy": 0.35,
                    "low_confidence_pose_max_jump_yaw": 0.12,
                    "yaw_bias_init_duration_s": 0.8,
                    "velocity_decay_tau_s": 0.9,
                    "live_map_max_points": 8000,
                }
            ],
            additional_env=node_python_env,
        )

    ekf_params_file = (rc_share / "config" / "ekf_sim_lidar_imu.yaml").resolve()
    rf2o_laser_odometry = None
    ekf_filter = None
    offline_submap_refiner = None
    offline_similarity_monitor = None
    run_recorder = None
    if rf2o_ekf_estimation_mode:
        rf2o_laser_odometry = Node(
            package="rf2o_laser_odometry",
            executable="rf2o_laser_odometry_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "laser_scan_topic": "/apex/sim/scan",
                    "odom_topic": "/apex/sim/rf2o_odom",
                    "publish_tf": False,
                    "base_frame_id": "base_link",
                    "odom_frame_id": "odom",
                    "init_pose_from_topic": "",
                    "freq": 20.0,
                }
            ],
        )
        ekf_filter = Node(
            package="robot_localization",
            executable="ekf_node",
            name="apex_sim_ekf_filter_node",
            output="screen",
            parameters=[str(ekf_params_file)],
            remappings=[("odometry/filtered", "/apex/sim/odom_fused")],
        )
    if offline_submaps_refinement_mode:
        offline_submap_refiner = Node(
            package="rc_sim_description",
            executable="offline_submap_refiner.py",
            name="offline_submap_refiner",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "replay_mode": offline_replay_mode,
                    "scan_topic": "/apex/sim/scan",
                    "imu_topic": "/apex/sim/imu",
                    "seed_odom_topic": offline_seed_odom_topic,
                    "seed_status_topic": offline_seed_status_topic,
                    "input_dir": offline_input_dir,
                    "frame_id": "map",
                    "child_frame_id": "offline_refined_base_link",
                    "publish_global_correction": offline_publish_global_correction,
                    "global_correction_topic": "/apex/sim/offline_global_correction",
                    "anchor_pose_topic": "/apex/sim/offline_anchor_pose",
                    "seed_odom_frame_id": "odom_imu_lidar_fused",
                    "window_scan_count": window_scan_count,
                    "window_overlap_count": window_overlap_count,
                    "initial_scan_count": initial_scan_count,
                    "submap_window_scans": submap_window_scans,
                    "point_stride": offline_point_stride,
                    "max_correspondence_m": offline_max_correspondence_m,
                    "offline_update_period_sec": offline_update_period_sec,
                    "use_track_geometry_prior": use_track_geometry_prior,
                    "track_geometry_file": track_geometry_file,
                    "track_geometry_weight": track_geometry_weight,
                }
            ],
            additional_env=node_python_env,
        )
        offline_similarity_monitor = Node(
            package="rc_sim_description",
            executable="offline_similarity_monitor.py",
            name="offline_similarity_monitor",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "ground_truth_status_topic": "/apex/sim/ground_truth/status",
                    "perfect_map_topic": "/apex/sim/ground_truth/perfect_map_points",
                    "ground_truth_path_topic": "/apex/sim/ground_truth/path",
                    "offline_map_topic": "/apex/sim/offline_refined_map",
                    "offline_path_topic": "/apex/sim/offline_refined_path",
                    "online_map_topic": "/apex/estimation/live_map_points_corrected",
                    "online_path_topic": "/apex/estimation/path_corrected",
                    "status_topic": "/apex/sim/offline_similarity_status",
                }
            ],
            additional_env=node_python_env,
        )
    if record_run:
        run_recorder = Node(
            package="rc_sim_description",
            executable="apex_sim_run_recorder.py",
            name="apex_sim_run_recorder",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "run_dir": str(record_run_dir),
                    "scan_topic": "/apex/sim/scan",
                    "imu_topic": "/apex/sim/imu",
                    "odom_topic": "/apex/odometry/imu_lidar_fused",
                    "ground_truth_odom_topic": "/apex/sim/ground_truth/odom",
                    "ground_truth_path_topic": "/apex/sim/ground_truth/path",
                    "perfect_map_topic": "/apex/sim/ground_truth/perfect_map_points",
                    "ground_truth_status_topic": "/apex/sim/ground_truth/status",
                    "online_status_topic": "/apex/estimation/status",
                    "scenario": scenario_name,
                    "control_mode": control_mode,
                    "mapping_mode": mapping_mode,
                    "estimation_mode": estimation_mode,
                    "distortion_profile": distortion_profile,
                    "refinement_mode": refinement_mode,
                    "offline_replay_mode": offline_replay_mode,
                    "world_path": world_path,
                    "write_online_status": True,
                }
            ],
            additional_env=node_python_env,
        )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str((slam_share / "launch" / "online_async_launch.py").resolve())),
        launch_arguments={
            "autostart": "true",
            "use_lifecycle_manager": "false",
            "use_sim_time": "true",
            "slam_params_file": str(merged_slam_params_path),
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_slam")),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="apex_sim_rviz",
        arguments=["-d", str(rviz_config)],
        parameters=[{"use_sim_time": True}],
        output="screen",
        additional_env=software_gui_env,
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    ideal_mode_logs = [
        LogInfo(
            msg=(
                "[apex_sim] estimation_mode=ideal uses slam params file: "
                f"{selected_slam_params_file}"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] estimation_mode=ideal uses Gazebo scan topic "
                "/apex/sim/scan directly, relies on the URDF-published "
                "base_link -> laser TF only, and trusts ideal odom for the "
                "initial mapping pass. No manual lidar static TF is launched."
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] estimation_mode=ideal bypasses the old APEX SLAM pipeline and "
                "uses apex_ground_truth_tf_bridge.py for odom -> base_link."
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] ideal odom degradation "
                f"(enabled={str(degrade_odom).lower()} pos_std={odom_position_noise_std:.4f} "
                f"yaw_std={odom_yaw_noise_std:.4f} vel_std={odom_velocity_noise_std:.4f} "
                f"yaw_bias_per_sec={odom_yaw_bias_per_sec:.4f} latency={odom_latency_sec:.4f})"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] ideal sensor noise "
                f"(distortion_profile={distortion_profile} lidar_std={lidar_noise_std:.4f} m "
                f"gyro_std={imu_gyro_noise_stddev_rps:.4f} rps "
                f"accel_std={imu_accel_noise_stddev_mps2:.4f} m/s^2)"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] steering symmetry for mapping "
                f"(enabled={str(symmetric_steering_for_mapping).lower()} "
                f"left_ratio={float(vehicle_bridge_params.get('steering_left_ratio', 1.0)):.3f} "
                f"right_ratio={float(vehicle_bridge_params.get('steering_right_ratio', 1.0)):.3f})"
            )
        ),
    ]
    rf2o_ekf_mode_logs = [
        LogInfo(
            msg=(
                "[apex_sim] estimation_mode=rf2o_ekf uses slam params file: "
                f"{selected_slam_params_file}"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] estimation_mode=rf2o_ekf uses EKF params file: "
                f"{ekf_params_file}"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] rf2o_laser_odometry consumes /apex/sim/scan and "
                "publishes /apex/sim/rf2o_odom without TF. robot_localization "
                "then fuses /apex/sim/rf2o_odom + /apex/sim/imu into "
                "/apex/sim/odom_fused and TF odom -> base_link. The RF2O "
                "executable owns two internal ROS nodes, so it is launched "
                "without a global node-name override to avoid duplicate names "
                "in the graph."
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] distortion_profile="
                f"{distortion_profile} applies local Gazebo sensor noise "
                f"(lidar_std={lidar_noise_std:.4f} m gyro_std={imu_gyro_noise_stddev_rps:.4f} rps "
                f"accel_std={imu_accel_noise_stddev_mps2:.4f} m/s^2). "
                "Pipeline-only dropout, heading jitter, and publish latency stay disabled in "
                "rf2o_ekf because this route bypasses the old degraded chain."
            )
        ),
    ]
    offline_submaps_logs = [
        LogInfo(
            msg=(
                "[apex_sim] refinement_mode=offline_submaps launches offline_submap_refiner.py "
                f"(replay_mode={offline_replay_mode} scan=/apex/sim/scan imu=/apex/sim/imu "
                f"seed_odom={offline_seed_odom_topic or '<none>'})"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] offline_submaps distortion handling "
                f"(distortion_profile={distortion_profile} "
                + (
                    "ideal sensor-noise run seeds the offline refiner from /apex/odometry/imu_lidar_fused "
                    "via an online occupancy/distance-field LiDAR+IMU seed"
                    if ideal_medium_sensor_profile
                    else "seed odom follows the active estimation route"
                )
                + ")"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] offline_submaps online-seed tuning "
                + (
                    "(online_distance_field_seed_node submap_window_scans=24 point_stride=1 "
                    "max_correspondence_m=0.24 local_prior_weight_xy=0.06 "
                    "local_prior_weight_yaw=0.20 lidar_residual_weight=0.85 "
                    "max_scan_optimization_evals=100 "
                    "yaw_bias_init_duration_s=0.8 velocity_decay_tau_s=0.9)"
                    if distance_field_online_seed_mode
                    else "(online seed tuning disabled for this mode)"
                )
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] online/offline coupling "
                f"(external_prior={str(online_use_external_prior).lower()} "
                f"prior_topic={online_prior_odom_topic or '<none>'} "
                f"offline_correction={str(online_use_offline_correction).lower()} "
                f"offline_reference={str(online_use_offline_reference).lower()} "
                f"publish_global_correction={str(offline_publish_global_correction).lower()})"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] offline track-geometry prior "
                f"(enabled={str(use_track_geometry_prior).lower()} "
                f"file={track_geometry_file or '<none>'} "
                f"weight={track_geometry_weight:.3f})"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] offline_submaps windowing "
                f"(window_scan_count={window_scan_count} overlap={window_overlap_count} "
                f"initial_scan_count={initial_scan_count} submap_window_scans={submap_window_scans} "
                f"point_stride={offline_point_stride} max_correspondence_m={offline_max_correspondence_m:.3f} "
                f"update_period_s={offline_update_period_sec:.3f})"
            )
        ),
    ]
    run_record_logs = [
        LogInfo(
            msg=(
                "[apex_sim] run recording "
                f"(enabled={str(record_run).lower()} dir={record_run_dir})"
            )
        ),
        LogInfo(
            msg=(
                "[apex_sim] run recording topics "
                "(imu=/apex/sim/imu scan=/apex/sim/scan odom=/apex/odometry/imu_lidar_fused "
                "gt_odom=/apex/sim/ground_truth/odom gt_path=/apex/sim/ground_truth/path "
                "gt_map=/apex/sim/ground_truth/perfect_map_points)"
            )
        ),
    ]

    actions = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_value),
        *(ideal_mode_logs if ideal_estimation_mode else []),
        *(rf2o_ekf_mode_logs if rf2o_ekf_estimation_mode else []),
        *(offline_submaps_logs if offline_submaps_refinement_mode else []),
        *(run_record_logs if record_run else []),
        gz_sim,
        robot_state_publisher,
        sim_bridges,
        TimerAction(period=2.0, actions=[spawn_rc]),
        TimerAction(period=2.5, actions=[vehicle_bridge, ground_truth]),
        TimerAction(period=2.8, actions=[ground_truth_tf_bridge]) if slam_uses_ideal_pose else None,
        TimerAction(period=3.0, actions=[ideal_cmd_vel_bridge]) if sim_only_cmd_vel_bridge_required else None,
        TimerAction(period=3.0, actions=[apex_pipeline]) if apex_pipeline is not None else None,
        TimerAction(period=3.2, actions=[online_distance_field_seed]) if online_distance_field_seed is not None else None,
        TimerAction(period=3.2, actions=[rf2o_laser_odometry]) if rf2o_laser_odometry is not None else None,
        TimerAction(period=3.4, actions=[ekf_filter]) if ekf_filter is not None else None,
        TimerAction(period=3.6, actions=[offline_submap_refiner]) if offline_submap_refiner is not None else None,
        TimerAction(period=3.7, actions=[offline_similarity_monitor]) if offline_similarity_monitor is not None else None,
        TimerAction(period=3.8, actions=[refined_visual_map]) if refined_visual_map is not None else None,
        TimerAction(period=3.9, actions=[run_recorder]) if run_recorder is not None else None,
        TimerAction(period=4.0, actions=[slam_toolbox]),
        TimerAction(period=4.5, actions=[rviz_node]),
    ]
    actions = [action for action in actions if action is not None]
    if fixed_map_publisher is not None:
        actions.append(TimerAction(period=3.9, actions=[fixed_map_publisher]))
    if control_mode == "manual_xbox":
        actions.append(TimerAction(period=4.1, actions=[manual_teleop]))
    if control_mode == "manual_windows_bridge":
        actions.append(TimerAction(period=4.1, actions=[manual_windows_bridge]))
    return actions


def generate_launch_description() -> LaunchDescription:
    rc_share = Path(get_package_share_directory("rc_sim_description"))
    apex_share = Path(get_package_share_directory("apex_telemetry"))
    default_scenario_config = str((rc_share / "config" / "apex_sim_scenarios.json").resolve())
    default_params_file = str((apex_share / "config" / "apex_params.yaml").resolve())
    default_slam_params_file = str((rc_share / "config" / "slam_toolbox_sim.yaml").resolve())
    return LaunchDescription(
        [
            DeclareLaunchArgument("scenario", default_value="baseline"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("rviz_config", default_value=""),
            DeclareLaunchArgument("scenario_config", default_value=default_scenario_config),
            DeclareLaunchArgument("params_file", default_value=default_params_file),
            DeclareLaunchArgument("use_slam", default_value="false"),
            DeclareLaunchArgument("use_refined_visual_map", default_value="false"),
            DeclareLaunchArgument("fixed_map_run", default_value=""),
            DeclareLaunchArgument("control_mode", default_value="recognition_tour"),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("mapping_mode", default_value="current"),
            DeclareLaunchArgument("estimation_mode", default_value=""),
            DeclareLaunchArgument("distortion_profile", default_value=""),
            DeclareLaunchArgument("refinement_mode", default_value="none"),
            DeclareLaunchArgument("offline_replay_mode", default_value="live_buffer"),
            DeclareLaunchArgument("sim_max_linear_speed_mps", default_value=""),
            DeclareLaunchArgument("window_scan_count", default_value="48"),
            DeclareLaunchArgument("window_overlap_count", default_value="16"),
            DeclareLaunchArgument("initial_scan_count", default_value="24"),
            DeclareLaunchArgument("submap_window_scans", default_value="8"),
            DeclareLaunchArgument("point_stride", default_value="2"),
            DeclareLaunchArgument("max_correspondence_m", default_value="0.35"),
            DeclareLaunchArgument("offline_update_period_sec", default_value="0.5"),
            DeclareLaunchArgument("offline_seed_odom_topic", default_value=""),
            DeclareLaunchArgument("offline_input_dir", default_value=""),
            DeclareLaunchArgument("online_use_external_prior", default_value="false"),
            DeclareLaunchArgument("online_prior_odom_topic", default_value=""),
            DeclareLaunchArgument("online_external_prior_weight_xy", default_value="0.45"),
            DeclareLaunchArgument("online_external_prior_weight_yaw", default_value="0.85"),
            DeclareLaunchArgument("online_freeze_scan_insertion_on_low_confidence", default_value="true"),
            DeclareLaunchArgument("online_low_confidence_residual_threshold_m", default_value="0.14"),
            DeclareLaunchArgument("online_low_confidence_inlier_ratio_threshold", default_value="0.24"),
            DeclareLaunchArgument("online_use_offline_correction", default_value="false"),
            DeclareLaunchArgument("online_offline_correction_blend", default_value="0.25"),
            DeclareLaunchArgument("online_offline_correction_max_jump_xy", default_value="0.30"),
            DeclareLaunchArgument("online_offline_correction_max_jump_yaw", default_value="0.18"),
            DeclareLaunchArgument("online_use_offline_reference", default_value="false"),
            DeclareLaunchArgument("symmetric_steering_for_mapping", default_value="true"),
            DeclareLaunchArgument("offline_publish_global_correction", default_value="true"),
            DeclareLaunchArgument("record_run", default_value="false"),
            DeclareLaunchArgument("record_run_base_dir", default_value=""),
            DeclareLaunchArgument("record_run_name", default_value=""),
            DeclareLaunchArgument("use_track_geometry_prior", default_value="false"),
            DeclareLaunchArgument("track_geometry_file", default_value=""),
            DeclareLaunchArgument("track_geometry_weight", default_value="0.08"),
            DeclareLaunchArgument("use_ideal_pose_for_slam", default_value="false"),
            DeclareLaunchArgument("use_ideal_lidar_for_slam", default_value="false"),
            DeclareLaunchArgument("degrade_odom", default_value="false"),
            DeclareLaunchArgument("odom_position_noise_std", default_value="0.0"),
            DeclareLaunchArgument("odom_yaw_noise_std", default_value="0.0"),
            DeclareLaunchArgument("odom_velocity_noise_std", default_value="0.0"),
            DeclareLaunchArgument("odom_yaw_bias_per_sec", default_value="0.0"),
            DeclareLaunchArgument("odom_latency_sec", default_value="0.0"),
            DeclareLaunchArgument("lidar_noise_std", default_value=""),
            DeclareLaunchArgument("imu_gyro_noise_stddev_rps", default_value=""),
            DeclareLaunchArgument("imu_accel_noise_stddev_mps2", default_value=""),
            DeclareLaunchArgument("slam_params_file", default_value=default_slam_params_file),
            DeclareLaunchArgument("x", default_value=""),
            DeclareLaunchArgument("y", default_value=""),
            DeclareLaunchArgument("z", default_value=""),
            DeclareLaunchArgument("yaw_deg", default_value=""),
            OpaqueFunction(function=_prepare_launch),
        ]
    )
