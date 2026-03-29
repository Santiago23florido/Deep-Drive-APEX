from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    params_file = LaunchConfiguration("params_file")
    imu_filter_params_file = LaunchConfiguration("imu_filter_params_file")
    imu_filter_node = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="apex_imu_filter_madgwick",
        output="screen",
        parameters=[imu_filter_params_file],
        remappings=[
            ("imu/data_raw", "/apex/imu/data_raw"),
            ("imu/data", "/apex/imu/data_filtered"),
        ],
    )

    yaw_only_odom_node = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "apex_telemetry.kinematics_odometry_node",
            "--ros-args",
            "--params-file",
            params_file,
            "-r",
            "__node:=kinematics_odometry_yaw_only",
            "-p",
            "position_topic:=/apex/kinematics/position",
            "-p",
            "velocity_topic:=/apex/kinematics/velocity",
            "-p",
            "heading_topic:=/apex/kinematics/heading",
            "-p",
            "angular_velocity_topic:=/apex/kinematics/angular_velocity",
            "-p",
            "odom_topic:=/apex/odometry/imu_yaw_only",
            "-p",
            "odom_frame_id:=odom_imu_yaw",
            "-p",
            "base_frame_id:=base_link",
            "-p",
            "translation_mode:=zero",
            "-p",
            "publish_rate_hz:=50.0",
            "-p",
            # Keep the yaw-only odom topic for diagnostics, but do not publish a TF
            # parent for base_link. The relative LiDAR frontend is now the critical
            # geometric correction path for local odometry.
            "publish_tf:=false",
        ],
        output="screen",
    )

    lidar_relative_odometry_node = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "apex_telemetry.lidar_relative_odometry_node",
            "--ros-args",
            "--params-file",
            params_file,
        ],
        output="screen",
    )

    planar_fusion_node = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "apex_telemetry.imu_lidar_planar_fusion_node",
            "--ros-args",
            "--params-file",
            params_file,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(package_root / "config" / "apex_params.yaml"),
            ),
            DeclareLaunchArgument(
                "imu_filter_params_file",
                default_value=str(package_root / "config" / "apex_local_imu_filter.yaml"),
            ),
            DeclareLaunchArgument(
                "ekf_params_file",
                default_value=str(package_root / "config" / "apex_local_ekf.yaml"),
            ),
            DeclareLaunchArgument(
                "local_slam_params_file",
                default_value=str(package_root / "config" / "apex_local_slam_toolbox.yaml"),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            imu_filter_node,
            yaw_only_odom_node,
            lidar_relative_odometry_node,
            planar_fusion_node,
        ]
    )
