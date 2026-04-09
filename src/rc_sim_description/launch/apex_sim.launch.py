from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
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


def _prepare_launch(context, *args, **kwargs):
    del args, kwargs
    rc_share = Path(get_package_share_directory("rc_sim_description"))
    apex_share = Path(get_package_share_directory("apex_telemetry"))
    slam_share = Path(get_package_share_directory("slam_toolbox"))
    repo_root = Path(
        os.environ.get("APEX_REPO_ROOT", str(rc_share.parents[3]))
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
    mapping_mode = (
        LaunchConfiguration("mapping_mode").perform(context).strip().lower()
        or "current"
    )
    use_ideal_pose_for_slam = (
        LaunchConfiguration("use_ideal_pose_for_slam").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    use_ideal_lidar_for_slam = (
        LaunchConfiguration("use_ideal_lidar_for_slam").perform(context).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if mapping_mode not in {"current", "ideal"}:
        mapping_mode = "current"
    ideal_mapping_mode = mapping_mode == "ideal"
    slam_uses_ideal_pose = ideal_mapping_mode or use_ideal_pose_for_slam
    slam_uses_ideal_lidar = ideal_mapping_mode or use_ideal_lidar_for_slam
    control_mode = (
        LaunchConfiguration("control_mode").perform(context).strip().lower()
        or "recognition_tour"
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
    else:
        if fixed_map_run_dir is not None:
            default_rviz_name = "apex_fixed_map_live.rviz"
        elif use_refined_visual_map:
            default_rviz_name = "apex_recognition_refined_map_live.rviz"
        elif (
            control_mode in {"manual_xbox", "manual_windows_bridge"}
            and use_slam
            and ideal_mapping_mode
        ):
            default_rviz_name = "apex_recognition_slam_live.rviz"
        elif control_mode in {"manual_xbox", "manual_windows_bridge"}:
            default_rviz_name = "apex_manual_mapping_live.rviz"
        else:
            default_rviz_name = (
                "apex_recognition_slam_live.rviz" if use_slam else "apex_recognition_live.rviz"
            )
        rviz_config = repo_root / "APEX" / "rviz" / default_rviz_name
        if not rviz_config.exists():
            rviz_config = repo_root / "APEX" / "rviz" / "apex_recognition_live.rviz"
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
    if use_slam:
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
        if ideal_mapping_mode
        else slam_params_file
    )
    try:
        base_slam_params = yaml.safe_load(
            selected_slam_params_file.read_text(encoding="utf-8")
        ) or {}
    except Exception:
        base_slam_params = {}
    slam_params_overrides = scenario.get("slam_params_overrides", {})
    if not isinstance(slam_params_overrides, dict):
        slam_params_overrides = {}
    merged_slam_params = _deep_merge(base_slam_params, slam_params_overrides)
    ideal_slam_overrides: dict[str, Any] = {"slam_toolbox": {"ros__parameters": {}}}
    ideal_slam_ros_params = ideal_slam_overrides["slam_toolbox"]["ros__parameters"]
    # In ideal mapping mode slam_toolbox consumes Gazebo's perfect scan directly
    # instead of the degraded LiDAR topics published by the telemetry pipeline.
    if slam_uses_ideal_lidar:
        ideal_slam_ros_params["scan_topic"] = "/apex/sim/scan"
    # slam_toolbox tracks motion through TF, so the ideal Gazebo ground truth is
    # bridged into the standard odom -> base_link frames for this first mapping stage.
    if slam_uses_ideal_pose:
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
    reference_script_path = str((repo_root / "APEX" / "apex_forward_raw" / "sensor_fusionn.py").resolve())

    gz_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_value = f"{rc_share}:{gz_resource_path}" if gz_resource_path else str(rc_share)
    pythonpath_parts = ["/usr/lib/python3/dist-packages"]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    pythonpath_value = ":".join(pythonpath_parts)

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
    vehicle_bridge_params.setdefault("use_sim_time", True)

    ground_truth_params = scenario.get("ground_truth", {})
    if not isinstance(ground_truth_params, dict):
        ground_truth_params = {}
    ground_truth_params = dict(ground_truth_params)
    ground_truth_params.setdefault("use_sim_time", True)
    ground_truth_params.setdefault("world_name", "default")
    ground_truth_params.setdefault("model_name", "rc_car")
    ground_truth_params.setdefault("world_path", world_path)

    robot_description = ParameterValue(Command(["xacro ", robot_xacro]), value_type=str)
    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_path],
        output="screen",
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
        additional_env={"PYTHONPATH": pythonpath_value},
    )
    ground_truth_tf_bridge = Node(
        package="rc_sim_description",
        executable="apex_ground_truth_tf_bridge.py",
        name="apex_ground_truth_tf_bridge",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "source_odom_topic": "/apex/sim/ground_truth/odom",
                "ideal_odom_topic": "/apex/sim/ideal_odom",
                "odom_frame_id": "odom",
                "child_frame_id": "base_link",
            }
        ],
    )
    ideal_lidar_frame_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="apex_sim_ideal_lidar_frame_tf",
        output="screen",
        arguments=[
            "0.18",
            "0.0",
            "0.12",
            "0.0",
            "0.0",
            "0.0",
            "base_link",
            "rc_car/base_link/lidar",
        ],
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
        )

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
    )

    enable_recognition_tour = control_mode not in {"manual_xbox", "manual_windows_bridge"}

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
            # Ideal mapping keeps the actuation bridge but disables the degraded
            # IMU/LiDAR estimation chain so slam_toolbox uses Gazebo ground truth.
            "enable_imu_source": "false" if ideal_mapping_mode else "true",
            "enable_lidar_source": "false" if ideal_mapping_mode else "true",
            "enable_kinematics_estimator": "false" if ideal_mapping_mode else "true",
            "enable_kinematics_odometry": "false" if ideal_mapping_mode else "true",
            "enable_imu_lidar_fusion": "false" if ideal_mapping_mode else "true",
            "enable_curve_entry_planner": "false",
            "enable_path_tracker": "false",
            "enable_recognition_tour_planner": "true" if enable_recognition_tour else "false",
            "enable_recognition_tour_tracker": "true" if enable_recognition_tour else "false",
            "enable_cmdvel_actuation_bridge": "true",
        }.items(),
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
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    actions = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_value),
        gz_sim,
        robot_state_publisher,
        sim_bridges,
        TimerAction(period=2.0, actions=[spawn_rc]),
        TimerAction(period=2.5, actions=[vehicle_bridge, ground_truth]),
        TimerAction(period=2.8, actions=[ground_truth_tf_bridge]) if slam_uses_ideal_pose else None,
        TimerAction(period=2.9, actions=[ideal_lidar_frame_tf]) if slam_uses_ideal_lidar else None,
        TimerAction(period=3.0, actions=[apex_pipeline]),
        TimerAction(period=3.8, actions=[refined_visual_map]),
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
    repo_root = Path(
        os.environ.get("APEX_REPO_ROOT", str(rc_share.parents[3]))
    ).expanduser().resolve()
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
            DeclareLaunchArgument("mapping_mode", default_value="current"),
            DeclareLaunchArgument("use_ideal_pose_for_slam", default_value="false"),
            DeclareLaunchArgument("use_ideal_lidar_for_slam", default_value="false"),
            DeclareLaunchArgument("slam_params_file", default_value=default_slam_params_file),
            DeclareLaunchArgument("x", default_value=""),
            DeclareLaunchArgument("y", default_value=""),
            DeclareLaunchArgument("z", default_value=""),
            DeclareLaunchArgument("yaw_deg", default_value=""),
            OpaqueFunction(function=_prepare_launch),
        ]
    )
