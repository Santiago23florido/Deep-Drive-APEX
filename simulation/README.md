# Simulation

This directory is the active development environment when the physical car is unavailable. It contains the Gazebo model and worlds, simulation-specific ROS packages, RViz layouts, tools, run data, and the earlier simulator kept for reference.

## Quick Start

```bash
cd ~/AiAtonomousRc/simulation/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
  --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
cd ~/AiAtonomousRc
./simulation/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

## Main Areas

| Path | Purpose |
| --- | --- |
| `ros2_ws/src/rc_sim_description/` | Gazebo model, worlds, bridges, ground truth, and launch files. |
| `ros2_ws/src/apex_telemetry/` | Simulation copy of the APEX pipeline. |
| `ros2_ws/src/voiture_system/` | Simulation compatibility for the vehicle stack. |
| `ros2_ws/src/apex_fusion_research/` | LiDAR/IMU error models, strapdown INS and evaluation for sensor-fusion research ([README](ros2_ws/src/apex_fusion_research/README.md)). |
| `tools/sim/` | Startup, mapping, recording, and controller wrappers. |
| `tools/analysis/` | Offline analysis. |
| `data/` | Simulation runs and generated results. |
| `legacy/Simulateur/` | Earlier simulator; not the recommended workflow. |

Build this workspace independently from `../real_vehicle/ros2_ws` to avoid package-name collisions.

See [Simulation with Gazebo](../docs/08_simulation_gazebo.md) for the full workflow.
