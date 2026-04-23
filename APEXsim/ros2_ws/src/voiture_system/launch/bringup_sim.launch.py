from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc_share = FindPackageShare('rc_sim_description')
    sys_share = FindPackageShare('voiture_system')

    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time')
    x_pos = LaunchConfiguration('x')
    y_pos = LaunchConfiguration('y')
    z_pos = LaunchConfiguration('z')

    robot_xacro = PathJoinSubstitution([desc_share, 'urdf', 'rc_car.urdf.xacro'])
    controllers_yaml = PathJoinSubstitution([sys_share, 'config', 'controllers.yaml'])

    robot_description = Command(['xacro ', robot_xacro])

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
        }],
        output='screen',
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'rc_car', '-x', x_pos, '-y', y_pos, '-z', z_pos],
        output='screen',
    )

    # Controller spawners (delay to ensure Gazebo + robot are ready)
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    rear_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['rear_wheels_velocity_controller', '--controller-manager', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    front_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['front_steer_position_controller', '--controller-manager', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    spawners_delayed = TimerAction(period=3.0, actions=[jsb_spawner, rear_spawner, front_spawner])

    high_level_node = Node(
        package='voiture_system',
        executable='high_level_controller_node',
        name='high_level_controller_node',
        parameters=[
            {'mode': 'sim'},
            {'lidar_topic': '/scan'},
            {'publish_rate_hz': 60.0},
            {'wheel_radius_m': 0.06},
            {'steer_deg_to_rad': True},
        ],
        output='screen',
    )

    bridge_node = Node(
        package='voiture_system',
        executable='vehicle_sim_bridge_node',
        name='vehicle_sim_bridge_node',
        parameters=[
            {'rear_wheel_joints': ['rear_left_wheel_joint', 'rear_right_wheel_joint']},
            {'front_steer_joints': ['front_left_wheel_steer_joint', 'front_right_wheel_steer_joint']},
            {'rear_controller_topic': '/rear_wheels_velocity_controller/commands'},
            {'front_controller_topic': '/front_steer_position_controller/commands'},
            {'publish_rate_hz': 60.0},
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value=PathJoinSubstitution([desc_share, 'worlds', 'basic_track.world'])),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.02'),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        spawners_delayed,
        high_level_node,
        bridge_node,
    ])
