# Estimation module

This directory is an architectural placeholder. The retained learned hybrid
estimator is implemented in
[`simulation/learning/lidar_imu_pose`](../../../simulation/learning/lidar_imu_pose/README.md)
and deployed separately to the Pi odometry runtime. It combines LiDAR ICP,
learned inertial correction, scan deskew and IMU propagation; `slam_toolbox`
supplies map corrections. The stationary hold covers scan estimates and the
propagated pose consumed by control.

In the [approved physical capture](../../docs/paper/20261024_005311/README.md),
hybrid processing median/P95/max was 14.165/18.128/26.490 ms. These are complete
estimator processing measurements, not isolated network inference. There is no
external real trajectory ground truth; scan consistency is not ATE/RPE. The older
IMU-integration nodes remain under `apex_telemetry` for historical workflows.
