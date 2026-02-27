# Estimation Module

This module currently contains:

- `kinematics_estimator_node`: IMU filtering/integration (acceleration + gyro yaw)
- outputs:
  - processed acceleration
  - velocity (x, y, z)
  - position (x, y, z)
  - heading yaw + angular velocity

Future expansion:
- IMU + wheel odometry fusion (EKF)
- drift correction with map constraints
