"""Multi-scenario LiDAR-IMU pose dataset generator for the APEX RC car.

The package runs Gazebo (gz-sim 8) in-process and headless, drives the car
with its normal actuators, synthesizes noisy LiDAR/IMU measurements from the
exact simulator state and stores everything in one SQLite database.

See ``tools/pose_dataset/README.md``.
"""

GENERATOR_VERSION = "1.0.0"
