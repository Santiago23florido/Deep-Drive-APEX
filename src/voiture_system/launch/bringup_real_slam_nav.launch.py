from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_slam = LaunchConfiguration("use_slam")
    use_nav2 = LaunchConfiguration("use_nav2")
    use_auto_track = LaunchConfiguration("use_auto_track")
    use_rviz = LaunchConfiguration("use_rviz")

    lidar_topic = LaunchConfiguration("lidar_topic")
    lidar_frame = LaunchConfiguration("lidar_frame")
    base_frame = LaunchConfiguration("base_frame")
    odom_frame = LaunchConfiguration("odom_frame")
    map_frame = LaunchConfiguration("map_frame")

    wheelbase_m = LaunchConfiguration("wheelbase_m")
    rear_axle_to_com_m = LaunchConfiguration("rear_axle_to_com_m")
    front_half_track_m = LaunchConfiguration("front_half_track_m")

    slam_params = LaunchConfiguration("slam_params_file")
    nav2_params = LaunchConfiguration("nav2_params_file")

    rplidar_node = Node(
        package="voiture_system",
        executable="rplidar_publisher_node",
        name="rpi_lidar_publisher",
        output="screen",
        parameters=[
            {
                "port": LaunchConfiguration("lidar_port"),
                "baudrate": LaunchConfiguration("lidar_baudrate"),
                "topic": lidar_topic,
                "frame_id": lidar_frame,
                "heading_offset_deg": LaunchConfiguration("lidar_heading_offset_deg"),
                "fov_filter_deg": LaunchConfiguration("lidar_fov_filter_deg"),
                "point_timeout_ms": LaunchConfiguration("lidar_point_timeout_ms"),
                "range_min": LaunchConfiguration("lidar_range_min"),
                "range_max": LaunchConfiguration("lidar_range_max"),
            }
        ],
    )

    serial_state_node = Node(
        package="voiture_system",
        executable="serial_state_node",
        name="serial_state_node",
        output="screen",
        parameters=[
            {
                "serial_port": LaunchConfiguration("arduino_port"),
                "baudrate": LaunchConfiguration("arduino_baudrate"),
                "ticks_to_meter": LaunchConfiguration("ticks_to_meter"),
                "publish_rate_hz": LaunchConfiguration("serial_publish_rate_hz"),
                "measured_wheelspeed_topic": "/measured_wheelspeed",
                "speed_mps_topic": "/vehicle/speed_mps",
            }
        ],
    )

    drive_node = Node(
        package="voiture_system",
        executable="ackermann_drive_node",
        name="ackermann_drive_node",
        output="screen",
        parameters=[
            {
                "wheelbase_m": wheelbase_m,
                "rear_axle_to_com_m": rear_axle_to_com_m,
                "front_half_track_m": front_half_track_m,
                "max_steering_deg": LaunchConfiguration("max_steering_deg"),
                "steering_sign": LaunchConfiguration("steering_sign"),
                "max_linear_speed_mps": LaunchConfiguration("max_linear_speed_mps"),
                "speed_limit_pct": LaunchConfiguration("speed_limit_pct"),
                "min_effective_speed_norm": LaunchConfiguration("min_effective_speed_norm"),
                "command_timeout_s": LaunchConfiguration("command_timeout_s"),
                "control_rate_hz": LaunchConfiguration("control_rate_hz"),
                "cmd_vel_topic": "/cmd_vel",
                "motor_pwm_channel": 0,
                "steering_pwm_channel": 1,
            }
        ],
    )

    auto_track_node = Node(
        package="voiture_system",
        executable="adaptive_track_controller_node",
        name="adaptive_track_controller_node",
        output="screen",
        condition=IfCondition(use_auto_track),
        parameters=[
            {
                "scan_topic": lidar_topic,
                "map_topic": "/map",
                "cmd_vel_topic": "/cmd_vel",
                "control_rate_hz": LaunchConfiguration("auto_track_rate_hz"),
                "max_speed_mps": LaunchConfiguration("auto_track_max_speed_mps"),
                "min_speed_mps": LaunchConfiguration("auto_track_min_speed_mps"),
                "speed_limit_pct": LaunchConfiguration("auto_track_speed_limit_pct"),
                "forward_speed_gain": LaunchConfiguration("auto_track_forward_speed_gain"),
                "stop_distance_m": LaunchConfiguration("auto_track_stop_distance_m"),
                "slow_distance_m": LaunchConfiguration("auto_track_slow_distance_m"),
                "max_yaw_rate_rad_s": LaunchConfiguration("auto_track_max_yaw_rate_rad_s"),
                "speed_curve_gain": LaunchConfiguration("auto_track_speed_curve_gain"),
                "motion_min_speed_mps": LaunchConfiguration("auto_track_motion_min_speed_mps"),
                "motion_min_front_m": LaunchConfiguration("auto_track_motion_min_front_m"),
                "yaw_deadband_rad_s": LaunchConfiguration("auto_track_yaw_deadband_rad_s"),
                "yaw_smoothing_alpha": LaunchConfiguration("auto_track_yaw_smoothing_alpha"),
                "straight_front_threshold_m": LaunchConfiguration("auto_track_straight_front_threshold_m"),
                "straight_balance_threshold_m": LaunchConfiguration("auto_track_straight_balance_threshold_m"),
                "straight_heading_threshold_rad": LaunchConfiguration("auto_track_straight_heading_threshold_rad"),
                "straight_steer_scale": LaunchConfiguration("auto_track_straight_steer_scale"),
            }
        ],
    )

    odom_node = Node(
        package="voiture_system",
        executable="ackermann_odometry_node",
        name="ackermann_odometry_node",
        output="screen",
        parameters=[
            {
                "wheelbase_m": wheelbase_m,
                "rear_axle_to_com_m": rear_axle_to_com_m,
                "front_half_track_m": front_half_track_m,
                "speed_topic": "/vehicle/speed_mps",
                "steering_topic": "/vehicle/steering_angle_cmd_rad",
                "odom_topic": "/odom",
                "publish_rate_hz": LaunchConfiguration("odom_rate_hz"),
                "publish_tf": True,
                "odom_frame_id": odom_frame,
                "base_frame_id": base_frame,
            }
        ],
    )

    # base_link -> laser static TF required by SLAM / Nav2.
    lidar_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        output="screen",
        arguments=[
            LaunchConfiguration("lidar_x_m"),
            LaunchConfiguration("lidar_y_m"),
            LaunchConfiguration("lidar_z_m"),
            LaunchConfiguration("lidar_roll_rad"),
            LaunchConfiguration("lidar_pitch_rad"),
            LaunchConfiguration("lidar_yaw_rad"),
            base_frame,
            lidar_frame,
        ],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"])
        ),
        condition=IfCondition(use_slam),
        launch_arguments={
            "use_sim_time": "false",
            "slam_params_file": slam_params,
        }.items(),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"])
        ),
        condition=IfCondition(use_nav2),
        launch_arguments={
            "use_sim_time": "false",
            "params_file": nav2_params,
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_slam", default_value="true"),
            DeclareLaunchArgument("use_nav2", default_value="false"),
            DeclareLaunchArgument("use_auto_track", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("lidar_topic", default_value="/lidar/scan"),
            DeclareLaunchArgument("lidar_frame", default_value="laser"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("lidar_baudrate", default_value="256000"),
            DeclareLaunchArgument("lidar_heading_offset_deg", default_value="-89"),
            DeclareLaunchArgument("lidar_fov_filter_deg", default_value="180"),
            DeclareLaunchArgument("lidar_point_timeout_ms", default_value="1000"),
            DeclareLaunchArgument("lidar_range_min", default_value="0.05"),
            DeclareLaunchArgument("lidar_range_max", default_value="12.0"),
            DeclareLaunchArgument("arduino_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("arduino_baudrate", default_value="115200"),
            DeclareLaunchArgument("ticks_to_meter", default_value="213.0"),
            DeclareLaunchArgument("serial_publish_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("wheelbase_m", default_value="0.30"),
            DeclareLaunchArgument("rear_axle_to_com_m", default_value="0.15"),
            DeclareLaunchArgument("front_half_track_m", default_value="0.05"),
            DeclareLaunchArgument("max_steering_deg", default_value="18.0"),
            DeclareLaunchArgument("steering_sign", default_value="1.0"),
            DeclareLaunchArgument("max_linear_speed_mps", default_value="0.576"),
            DeclareLaunchArgument("min_effective_speed_norm", default_value="0.45"),
            DeclareLaunchArgument(
                "speed_limit_pct",
                default_value="40.0",
                description="Global speed cap for tests (percent of max wheel speed)",
            ),
            DeclareLaunchArgument("command_timeout_s", default_value="0.5"),
            DeclareLaunchArgument("control_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("odom_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("auto_track_rate_hz", default_value="20.0"),
            DeclareLaunchArgument("auto_track_max_speed_mps", default_value="0.576"),
            DeclareLaunchArgument("auto_track_min_speed_mps", default_value="0.50"),
            DeclareLaunchArgument("auto_track_speed_limit_pct", default_value="100.0"),
            DeclareLaunchArgument("auto_track_forward_speed_gain", default_value="1.5"),
            DeclareLaunchArgument("auto_track_stop_distance_m", default_value="0.05"),
            DeclareLaunchArgument("auto_track_slow_distance_m", default_value="0.20"),
            DeclareLaunchArgument("auto_track_max_yaw_rate_rad_s", default_value="1.2"),
            DeclareLaunchArgument("auto_track_speed_curve_gain", default_value="0.6"),
            DeclareLaunchArgument("auto_track_motion_min_speed_mps", default_value="0.50"),
            DeclareLaunchArgument("auto_track_motion_min_front_m", default_value="0.12"),
            DeclareLaunchArgument("auto_track_yaw_deadband_rad_s", default_value="0.08"),
            DeclareLaunchArgument("auto_track_yaw_smoothing_alpha", default_value="0.75"),
            DeclareLaunchArgument("auto_track_straight_front_threshold_m", default_value="0.35"),
            DeclareLaunchArgument("auto_track_straight_balance_threshold_m", default_value="0.12"),
            DeclareLaunchArgument("auto_track_straight_heading_threshold_rad", default_value="0.20"),
            DeclareLaunchArgument("auto_track_straight_steer_scale", default_value="0.30"),
            DeclareLaunchArgument("lidar_x_m", default_value="0.18"),
            DeclareLaunchArgument("lidar_y_m", default_value="0.0"),
            DeclareLaunchArgument("lidar_z_m", default_value="0.12"),
            DeclareLaunchArgument("lidar_roll_rad", default_value="0.0"),
            DeclareLaunchArgument("lidar_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("lidar_yaw_rad", default_value="0.0"),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("voiture_system"), "config", "slam_toolbox_online_async.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "nav2_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("voiture_system"), "config", "nav2_ackermann.yaml"]
                ),
            ),
            rplidar_node,
            serial_state_node,
            drive_node,
            auto_track_node,
            odom_node,
            lidar_static_tf,
            slam_launch,
            nav2_launch,
            rviz_node,
        ]
    )
