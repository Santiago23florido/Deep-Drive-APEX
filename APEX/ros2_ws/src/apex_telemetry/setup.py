from setuptools import find_packages, setup
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
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/apex_pipeline.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/apex_params.yaml",
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
            "nano_accel_serial_node = apex_telemetry.imu.nano_accel_serial_node:main",
            "kinematics_estimator_node = apex_telemetry.odometry.kinematics_estimator_node:main",
            "kinematics_odometry_node = apex_telemetry.odometry.kinematics_odometry_node:main",
            "rplidar_publisher_node = apex_telemetry.perception.rplidar_publisher_node:main",
        ],
    },
)
