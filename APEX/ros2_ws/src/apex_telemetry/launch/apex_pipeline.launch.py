from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    serial_port = LaunchConfiguration("serial_port")
    serial_baudrate = LaunchConfiguration("serial_baudrate")
    lidar_port = LaunchConfiguration("lidar_port")
    lidar_baudrate = LaunchConfiguration("lidar_baudrate")
    enable_imu_lidar_fusion = LaunchConfiguration("enable_imu_lidar_fusion")

    serial_reader = Node(
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

    kinematics_estimator = Node(
        package="apex_telemetry",
        executable="kinematics_estimator_node",
        name="kinematics_estimator_node",
        output="screen",
        parameters=[params_file],
    )

    kinematics_odometry = Node(
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
            },
        ],
    )

    laser_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        output="screen",
        arguments=["0.18", "0.0", "0.12", "0.0", "0.0", "0.0", "base_link", "laser"],
    )

    imu_lidar_planar_fusion = Node(
        package="apex_telemetry",
        executable="imu_lidar_planar_fusion_node",
        name="imu_lidar_planar_fusion_node",
        output="screen",
        parameters=[params_file],
        condition=IfCondition(enable_imu_lidar_fusion),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("apex_telemetry"), "config", "apex_params.yaml"]
                ),
            ),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("serial_baudrate", default_value="115200"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("lidar_baudrate", default_value="115200"),
            DeclareLaunchArgument("enable_imu_lidar_fusion", default_value="false"),
            serial_reader,
            kinematics_estimator,
            kinematics_odometry,
            lidar_node,
            laser_tf_node,
            imu_lidar_planar_fusion,
        ]
    )
