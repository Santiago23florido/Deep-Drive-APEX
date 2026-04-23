from setuptools import setup

package_name = 'voiture_system'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/launch',
            [
                'launch/bringup_real_slam_nav.launch.py',
            ],
        ),
        (
            'share/' + package_name + '/config',
            [
                'config/controllers.yaml',
                'config/slam_toolbox_online_async.yaml',
                'config/nav2_ackermann.yaml',
            ],
        ),
        ('share/' + package_name + '/tools', ['tools/test_pipeline.py']),
        ('share/' + package_name + '/urdf', ['urdf/ros2_control.xacro']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@example.com',
    description='High-level controller and hardware drivers for the RC car.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'high_level_controller_node = voiture_system.high_level_controller_node:main',
            'serial_state_node = voiture_system.serial_state_node:main',
            'ackermann_drive_node = voiture_system.ackermann_drive_node:main',
            'ackermann_odometry_node = voiture_system.ackermann_odometry_node:main',
            'rplidar_publisher_node = voiture_system.rplidar_publisher_node:main',
            'adaptive_track_controller_node = voiture_system.adaptive_track_controller_node:main',
        ],
    },
)
