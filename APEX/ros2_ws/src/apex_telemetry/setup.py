from setuptools import setup
import sys


package_name = "apex_telemetry"

# Some colcon/python/setuptools combinations pass flags that older setuptools
# variants reject when invoking setup.py directly. Strip known problematic ones.
for unsupported in ("--editable", "--uninstall"):
    while unsupported in sys.argv:
        sys.argv.remove(unsupported)


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/apex_pipeline.launch.py",
                "launch/apex_lidar_slam.launch.py",
                "launch/apex_local_odom_fusion.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/apex_params.yaml",
                "config/apex_slam_toolbox.yaml",
                "config/apex_local_slam_toolbox.yaml",
                "config/apex_local_ekf.yaml",
                "config/apex_local_imu_filter.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="APEX Team",
    maintainer_email="todo@example.com",
    description="APEX telemetry pipeline nodes.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nano_accel_serial_node = apex_telemetry.nano_accel_serial_node:main",
            "kinematics_estimator_node = apex_telemetry.kinematics_estimator_node:main",
            "kinematics_odometry_node = apex_telemetry.kinematics_odometry_node:main",
            "imu_lidar_planar_fusion_node = apex_telemetry.imu_lidar_planar_fusion_node:main",
            "lidar_relative_odometry_node = apex_telemetry.lidar_relative_odometry_node:main",
            "lidar_pose_bridge = apex_telemetry.lidar_pose_bridge:main",
            "rplidar_publisher_node = apex_telemetry.rplidar_publisher_node:main",
            "recon_mapping_node = apex_telemetry.recon_mapping_node:main",
        ],
    },
)
