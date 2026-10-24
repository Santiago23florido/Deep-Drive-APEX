# Simulation module

This directory is an architectural placeholder. The implemented simulation lives
in [`rc_sim_description`](../../ros2_ws/src/rc_sim_description/) and
[`apex_fusion_research`](../../ros2_ws/src/apex_fusion_research/README.md), with
learned odometry under [`learning/lidar_imu_pose`](../../learning/lidar_imu_pose/README.md).

Use [`apex_real2sim_up.sh`](../../tools/sim/apex_real2sim_up.sh) for reconnaissance,
SLAM and subsequent planned laps. The driving core includes the corrections from
[the approved real run](../../../real_vehicle/docs/paper/20261024_005311/README.md).
Historical Gazebo results and a final-version simulation campaign are separate
evidence. See the [preprint readiness report](../../../docs/reports/preprint_readiness_20261024.md).
