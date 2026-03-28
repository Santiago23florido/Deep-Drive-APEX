from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    params_file = LaunchConfiguration("params_file")
    imu_filter_params_file = LaunchConfiguration("imu_filter_params_file")
    local_slam_params_file = LaunchConfiguration("local_slam_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")

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

    lidar_pose_bridge_node = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "apex_telemetry.lidar_pose_bridge",
            "--ros-args",
            "--params-file",
            params_file,
        ],
        output="screen",
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
            "publish_tf:=true",
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

    local_slam_node = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="local_lidar_slam_toolbox",
        namespace="",
        output="screen",
        parameters=[
            local_slam_params_file,
            {
                "use_sim_time": use_sim_time,
                "use_lifecycle_manager": False,
            },
        ],
        remappings=[
            ("map", "/apex/local_slam/map"),
            ("map_metadata", "/apex/local_slam/map_metadata"),
        ],
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(local_slam_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(autostart),
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=local_slam_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[APEX] Local slam_toolbox is activating."),
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(local_slam_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        ),
        condition=IfCondition(autostart),
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
            DeclareLaunchArgument("autostart", default_value="true"),
            imu_filter_node,
            lidar_pose_bridge_node,
            yaw_only_odom_node,
            planar_fusion_node,
            local_slam_node,
            configure_event,
            activate_event,
        ]
    )
