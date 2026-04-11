from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    serial_port = LaunchConfiguration("serial_port")
    serial_baudrate = LaunchConfiguration("serial_baudrate")
    lidar_port = LaunchConfiguration("lidar_port")
    lidar_baudrate = LaunchConfiguration("lidar_baudrate")
    enable_laser_tf = LaunchConfiguration("enable_laser_tf")
    enable_imu_source = LaunchConfiguration("enable_imu_source")
    enable_lidar_source = LaunchConfiguration("enable_lidar_source")
    enable_kinematics_estimator = LaunchConfiguration("enable_kinematics_estimator")
    enable_kinematics_odometry = LaunchConfiguration("enable_kinematics_odometry")
    imu_transport_backend = LaunchConfiguration("imu_transport_backend")
    sim_imu_topic = LaunchConfiguration("sim_imu_topic")
    lidar_source_backend = LaunchConfiguration("lidar_source_backend")
    sim_scan_topic = LaunchConfiguration("sim_scan_topic")
    actuation_backend = LaunchConfiguration("actuation_backend")
    sim_motor_pwm_topic = LaunchConfiguration("sim_motor_pwm_topic")
    sim_steering_pwm_topic = LaunchConfiguration("sim_steering_pwm_topic")
    enable_imu_lidar_fusion = LaunchConfiguration("enable_imu_lidar_fusion")
    enable_curve_entry_planner = LaunchConfiguration("enable_curve_entry_planner")
    enable_path_tracker = LaunchConfiguration("enable_path_tracker")
    enable_recognition_tour_planner = LaunchConfiguration("enable_recognition_tour_planner")
    enable_recognition_tour_tracker = LaunchConfiguration("enable_recognition_tour_tracker")
    enable_cmdvel_actuation_bridge = LaunchConfiguration("enable_cmdvel_actuation_bridge")

    serial_reader = Node(
        package="apex_telemetry",
        executable="nano_accel_serial_node",
        name="nano_accel_serial_node",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "transport_backend": imu_transport_backend,
                "serial_port": serial_port,
                "baudrate": serial_baudrate,
                "sim_imu_topic": sim_imu_topic,
            },
        ],
        condition=IfCondition(enable_imu_source),
    )

    kinematics_estimator = Node(
        package="apex_telemetry",
        executable="kinematics_estimator_node",
        name="kinematics_estimator_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_kinematics_estimator),
    )

    kinematics_odometry = Node(
        package="apex_telemetry",
        executable="kinematics_odometry_node",
        name="kinematics_odometry_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_kinematics_odometry),
    )

    lidar_node = Node(
        package="apex_telemetry",
        executable="rplidar_publisher_node",
        name="apex_rplidar_publisher",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "source_backend": lidar_source_backend,
                "port": lidar_port,
                "baudrate": lidar_baudrate,
                "sim_scan_topic": sim_scan_topic,
            },
        ],
        condition=IfCondition(enable_lidar_source),
    )

    laser_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        output="screen",
        arguments=["0.18", "0.0", "0.12", "0.0", "0.0", "0.0", "base_link", "laser"],
        condition=IfCondition(enable_laser_tf),
    )

    imu_lidar_planar_fusion = Node(
        package="apex_telemetry",
        executable="imu_lidar_planar_fusion_node",
        name="imu_lidar_planar_fusion_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_imu_lidar_fusion),
    )

    curve_entry_path_planner = Node(
        package="apex_telemetry",
        executable="curve_entry_path_planner_node",
        name="curve_entry_path_planner_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_curve_entry_planner),
    )

    curve_path_tracker = Node(
        package="apex_telemetry",
        executable="curve_path_tracker_node",
        name="curve_path_tracker_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_path_tracker),
    )

    recognition_tour_planner = Node(
        package="apex_telemetry",
        executable="recognition_tour_planner_node",
        name="recognition_tour_planner_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_recognition_tour_planner),
    )

    recognition_tour_tracker = Node(
        package="apex_telemetry",
        executable="recognition_tour_tracker_node",
        name="recognition_tour_tracker_node",
        output="screen",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_recognition_tour_tracker),
    )

    cmdvel_actuation_bridge = Node(
        package="apex_telemetry",
        executable="cmd_vel_to_apex_actuation_node",
        name="cmd_vel_to_apex_actuation_node",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "actuation_backend": actuation_backend,
                "sim_motor_pwm_topic": sim_motor_pwm_topic,
                "sim_steering_pwm_topic": sim_steering_pwm_topic,
            },
        ],
        condition=IfCondition(enable_cmdvel_actuation_bridge),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("apex_telemetry"), "config", "apex_params.yaml"]
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument("serial_baudrate", default_value="115200"),
            DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("lidar_baudrate", default_value="115200"),
            DeclareLaunchArgument("enable_laser_tf", default_value="true"),
            DeclareLaunchArgument("enable_imu_source", default_value="true"),
            DeclareLaunchArgument("enable_lidar_source", default_value="true"),
            DeclareLaunchArgument("enable_kinematics_estimator", default_value="true"),
            DeclareLaunchArgument("enable_kinematics_odometry", default_value="true"),
            DeclareLaunchArgument("imu_transport_backend", default_value="serial"),
            DeclareLaunchArgument("sim_imu_topic", default_value="/apex/sim/imu"),
            DeclareLaunchArgument("lidar_source_backend", default_value="rplidar"),
            DeclareLaunchArgument("sim_scan_topic", default_value="/apex/sim/scan"),
            DeclareLaunchArgument("actuation_backend", default_value="sysfs_pwm"),
            DeclareLaunchArgument("sim_motor_pwm_topic", default_value="/apex/sim/pwm/motor_dc"),
            DeclareLaunchArgument(
                "sim_steering_pwm_topic",
                default_value="/apex/sim/pwm/steering_dc",
            ),
            DeclareLaunchArgument("enable_imu_lidar_fusion", default_value="false"),
            DeclareLaunchArgument("enable_curve_entry_planner", default_value="false"),
            DeclareLaunchArgument("enable_path_tracker", default_value="false"),
            DeclareLaunchArgument("enable_recognition_tour_planner", default_value="false"),
            DeclareLaunchArgument("enable_recognition_tour_tracker", default_value="false"),
            DeclareLaunchArgument("enable_cmdvel_actuation_bridge", default_value="false"),
            serial_reader,
            kinematics_estimator,
            kinematics_odometry,
            lidar_node,
            laser_tf_node,
            imu_lidar_planar_fusion,
            curve_entry_path_planner,
            curve_path_tracker,
            recognition_tour_planner,
            recognition_tour_tracker,
            cmdvel_actuation_bridge,
        ]
    )
