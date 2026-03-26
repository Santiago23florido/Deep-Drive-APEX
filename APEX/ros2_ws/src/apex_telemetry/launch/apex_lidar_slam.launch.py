from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")

    serial_port = LaunchConfiguration("serial_port")
    serial_baudrate = LaunchConfiguration("serial_baudrate")
    lidar_port = LaunchConfiguration("lidar_port")
    lidar_baudrate = LaunchConfiguration("lidar_baudrate")

    lidar_topic = LaunchConfiguration("lidar_topic")
    lidar_frame = LaunchConfiguration("lidar_frame")
    base_frame = LaunchConfiguration("base_frame")

    accel_serial_node = Node(
        package="apex_telemetry",
        executable="nano_accel_serial_node",
        name="nano_accel_serial_node",
        output="screen",
        parameters=[
            params_file,
            {
                "serial_port": serial_port,
                "baudrate": serial_baudrate,
            },
        ],
    )

    kinematics_estimator_node = Node(
        package="apex_telemetry",
        executable="kinematics_estimator_node",
        name="kinematics_estimator_node",
        output="screen",
        parameters=[params_file],
    )

    kinematics_odom_node = Node(
        package="apex_telemetry",
        executable="kinematics_odometry_node",
        name="kinematics_odometry_node",
        output="screen",
        parameters=[params_file],
    )

    lidar_node = Node(
        package="apex_telemetry",
        executable="rplidar_publisher_node",
        name="apex_rplidar_publisher",
        output="screen",
        parameters=[
            params_file,
            {
                "port": lidar_port,
                "baudrate": lidar_baudrate,
                "topic": lidar_topic,
                "frame_id": lidar_frame,
            },
        ],
    )

    # Required TF for SLAM: base_link -> laser.
    laser_tf_node = Node(
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
        launch_arguments={
            "use_sim_time": "false",
            "slam_params_file": slam_params_file,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("apex_telemetry"), "config", "apex_params.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("apex_telemetry"), "config", "apex_slam_toolbox.yaml"]
                ),
            ),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("serial_baudrate", default_value="115200"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("lidar_baudrate", default_value="115200"),
            DeclareLaunchArgument("lidar_topic", default_value="/lidar/scan"),
            DeclareLaunchArgument("lidar_frame", default_value="laser"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("lidar_x_m", default_value="0.18"),
            DeclareLaunchArgument("lidar_y_m", default_value="0.0"),
            DeclareLaunchArgument("lidar_z_m", default_value="0.12"),
            DeclareLaunchArgument("lidar_roll_rad", default_value="0.0"),
            DeclareLaunchArgument("lidar_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("lidar_yaw_rad", default_value="0.0"),
            accel_serial_node,
            kinematics_estimator_node,
            kinematics_odom_node,
            lidar_node,
            laser_tf_node,
            slam_launch,
        ]
    )
