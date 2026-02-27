# Estimation Module

This module currently contains:

- `kinematics_estimator_node`: acceleration filtering and integration
- outputs:
  - processed acceleration
  - velocity (x, y, z)
  - position (x, y, z)

Future expansion:
- IMU + wheel odometry fusion (EKF)
- drift correction with map constraints
