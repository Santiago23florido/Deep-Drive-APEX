from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    serial_port = LaunchConfiguration("serial_port")
    serial_baudrate = LaunchConfiguration("serial_baudrate")

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
            serial_reader,
            kinematics_estimator,
        ]
    )
