from glob import glob

from setuptools import find_packages, setup

PACKAGE = "apex_fusion_research"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config/imu", glob("config/imu/*.yaml")),
        (f"share/{PACKAGE}/config/lidar", glob("config/lidar/*.yaml")),
        (f"share/{PACKAGE}/config/ins", glob("config/ins/*.yaml")),
        (f"share/{PACKAGE}/config/slam", glob("config/slam/*.yaml")),
        (f"share/{PACKAGE}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Santiago Florido",
    maintainer_email="sanflogom@gmail.com",
    description="LiDAR noise model, MEMS IMU model and strapdown INS for sensor-fusion research.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"imu_sensor_node = {PACKAGE}.nodes.imu_sensor_node:main",
            f"lidar_noise_node = {PACKAGE}.nodes.lidar_noise_node:main",
            f"truth_node = {PACKAGE}.nodes.truth_node:main",
            f"strapdown_ins_node = {PACKAGE}.nodes.strapdown_ins_node:main",
            f"ins_error_monitor_node = {PACKAGE}.nodes.ins_error_monitor_node:main",
            f"scan_projector_node = {PACKAGE}.nodes.scan_projector_node:main",
            f"slam_odometry_node = {PACKAGE}.nodes.slam_odometry_node:main",
            f"slam_map_recorder_node = {PACKAGE}.nodes.slam_map_recorder_node:main",
            f"real_sensor_node = {PACKAGE}.nodes.real_sensor_node:main",
            f"sim_actuation_node = {PACKAGE}.nodes.sim_actuation_node:main",
            f"track_driver_node = {PACKAGE}.nodes.track_driver_node:main",
            f"run_referee_node = {PACKAGE}.nodes.run_referee_node:main",
            f"plot_ins_drift = {PACKAGE}.tools.plot_ins_drift:main",
            f"allan_analysis = {PACKAGE}.tools.allan_analysis:main",
            f"lidar_noise_report = {PACKAGE}.tools.lidar_noise_report:main",
            f"plot_slam_maps = {PACKAGE}.tools.plot_slam_maps:main",
            f"wait_for = {PACKAGE}.tools.wait_for:main",
        ],
    },
)
