from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("rc_sim_description")

    default_world = PathJoinSubstitution(
        [pkg_share, "worlds", "basic_track.world"]
    )
    default_rviz = PathJoinSubstitution(
        [pkg_share, "config", "rviz", "rc_car.rviz"]
    )
    robot_xacro = PathJoinSubstitution(
        [pkg_share, "urdf", "rc_car.urdf.xacro"]
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    x_pos = LaunchConfiguration("x")
    y_pos = LaunchConfiguration("y")
    z_pos = LaunchConfiguration("z")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    rear_wheel_bridge = LaunchConfiguration("rear_wheel_bridge")
    control_publish_rate = LaunchConfiguration("control_publish_rate")
    wheel_base = LaunchConfiguration("wheel_base")
    track_width = LaunchConfiguration("track_width")
    steering_limit = LaunchConfiguration("steering_limit")
    lidar_bridge = LaunchConfiguration("lidar_bridge")
    lidar_reader = LaunchConfiguration("lidar_reader")
    control_node = LaunchConfiguration("control_node")
    camera_bridge = LaunchConfiguration("camera_bridge")

    # Command expects a space between executable and path
    robot_description = Command(["xacro ", robot_xacro])

    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        output="screen",
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"use_sim_time": use_sim_time}],
        name="rc_car_joint_state_publisher",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="rc_car_state_publisher",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": robot_description,
            }
        ],
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
            x_pos,
            "-y",
            y_pos,
            "-z",
            z_pos,
        ],
        output="screen",
    )

    # Small delay so gz sim is running before spawning the model
    spawn_delayed = TimerAction(period=2.0, actions=[spawn_rc])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rc_car_rviz",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(rviz),
    )

    rear_wheel_bridge_node = Node(
        package="rc_sim_description",
        executable="rear_wheel_speed_bridge.py",
        name="rear_wheel_speed_bridge",
        output="screen",
        parameters=[
            {
                "publish_rate": control_publish_rate,
                "wheel_base": wheel_base,
                "track_width": track_width,
                "steering_limit": steering_limit,
            }
        ],
        condition=IfCondition(rear_wheel_bridge),
    )


    lidar_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="scan_bridge",
        output="screen",
        arguments=["/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan"],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(lidar_bridge),
    )

    lidar_reader_node = Node(
        package="rc_sim_description",
        executable="gazebo_lidar_reader_node.py",
        name="gazebo_lidar_reader",
        output="screen",
        arguments=[
            "--topic",
            "/scan",
            "--publish-topic",
            "/lidar_processed",
            "--use-sim-time",
        ],
        condition=IfCondition(lidar_reader),
    )

    camera_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="camera_bridge",
        output="screen",
        arguments=[
            "/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(camera_bridge),
    )

    control_node = Node(
        package="rc_sim_description",
        executable="voiture_control_node.py",
        name="voiture_control_node",
        output="screen",
        parameters=[
            {
                "lidar_topic": "/lidar_processed",
                "measured_wheelspeed_topic": "/measured_wheelspeed",
                "rear_wheel_speed_topic": "/rear_wheel_speed",
                "steering_angle_topic": "/steering_angle",
                "control_rate_hz": 60.0,
            }
        ],
        condition=IfCondition(control_node),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use Gazebo simulated clock",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=default_world,
                description="Gazebo world to load",
            ),
            DeclareLaunchArgument(
                "x",
                default_value="0.0",
                description="Initial X position",
            ),
            DeclareLaunchArgument(
                "y",
                default_value="-2.5",
                description="Initial Y position",
            ),
            DeclareLaunchArgument(
                "z",
                default_value="0.02",
                description="Initial Z position",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz with the description",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz,
                description="RViz configuration file",
            ),
            DeclareLaunchArgument(
                "rear_wheel_bridge",
                default_value="true",
                description="Launch the rear wheel speed bridge node",
            ),
            DeclareLaunchArgument(
                "control_publish_rate",
                default_value="60.0",
                description="Publish rate (Hz) for control bridge",
            ),
            DeclareLaunchArgument(
                "wheel_base",
                default_value="0.32",
                description="Wheelbase for Ackermann steering (m)",
            ),
            DeclareLaunchArgument(
                "track_width",
                default_value="0.29",
                description="Track width for Ackermann steering (m)",
            ),
            DeclareLaunchArgument(
                "steering_limit",
                default_value="0.5235987756",
                description="Max steering angle (rad, 30 deg)",
            ),
            DeclareLaunchArgument(
                "lidar_bridge",
                default_value="true",
                description="Launch ros_gz_bridge for /scan LaserScan",
            ),
            DeclareLaunchArgument(
                "lidar_reader",
                default_value="true",
                description="Launch GazeboLidarReader to publish /lidar_processed",
            ),
            DeclareLaunchArgument(
                "control_node",
                default_value="true",
                description="Launch voiture_control_node",
            ),
            DeclareLaunchArgument(
                "camera_bridge",
                default_value="true",
                description="Launch ros_gz_bridge for camera topics",
            ),
            gz_sim,
            joint_state_publisher,
            robot_state_publisher,
            spawn_delayed,
            rviz_node,
            rear_wheel_bridge_node,
            lidar_bridge_node,
            lidar_reader_node,
            camera_bridge_node,
            control_node,
        ]
    )
