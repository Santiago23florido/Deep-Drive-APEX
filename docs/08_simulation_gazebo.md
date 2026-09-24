# Simulation with Gazebo

## Objective

`simulation/` supports development and validation without physical hardware. It simulates:

- Ackermann vehicle model and tracks.
- LiDAR and IMU.
- Motor and steering response.
- Pose, path, and map ground truth.
- Configurable noise, latency, drift, and edge cases.
- Recording and offline reconstruction.

## Main Packages

| Package | Role |
| --- | --- |
| `rc_sim_description` | Gazebo, URDF, worlds, bridges, and ground truth. |
| `apex_telemetry` | Perception, estimation, planner, tracker, and APEX actuation bridge. |
| `voiture_system` | Controller compatibility and alternate experiments. |
| `apex_fusion_research` | LiDAR noise model, realistic MEMS IMU model, strapdown INS and evaluation tools for LiDAR–IMU fusion research. |

## Build

```bash
cd ~/AiAtonomousRc/simulation/ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install \
  --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
```

## Recommended Entry Point

```bash
./simulation/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

The wrapper cleans stale processes, builds the simulation workspace, exports `APEX_SIM_ROOT`, and starts `apex_sim.launch.py`.

## Common Options

| Option | Effect |
| --- | --- |
| `--scenario NAME` | Select a scenario. |
| `--rviz` | Open RViz. |
| `--slam` | Enable `slam_toolbox`. |
| `--refined-map` | Display the refined map. |
| `--fixed-map-run DIR` | Localize against a saved map. |
| `--control-mode recognition_tour` | Autonomous planner/tracker. |
| `--control-mode manual_xbox` | Controller visible from Linux. |
| `--control-mode manual_windows_bridge` | Controller through the Windows bridge. |
| `--arm` | Arm the tracker after startup. |
| `--skip-build` | Skip the build step. |

## Scenarios

Defined in `simulation/ros2_ws/src/rc_sim_description/config/apex_sim_scenarios.json`:

- `baseline`
- `precision_fusion`
- `tight_right_saturation`
- `outer_long_inner_short`
- `startup_pose_jump`
- `narrowing_false_corridor`

World files live under `simulation/ros2_ws/src/rc_sim_description/worlds/`.

## Launch Flow

```text
apex_sim.launch.py
  ├── Gazebo and selected world
  ├── robot_state_publisher and vehicle spawn
  ├── ros_gz bridges
  ├── APEX pipeline in simulation mode
  ├── vehicle bridge
  ├── ground truth
  ├── optional recorder and SLAM
  └── optional RViz
```

## Sensors and Actuation

```text
Gazebo LiDAR -> /apex/sim/scan -> rplidar_publisher_node
Gazebo IMU   -> /apex/sim/imu  -> nano_accel_serial_node
/apex/cmd_vel_track -> simulated PWM -> apex_gz_vehicle_bridge
```

Estimation should not consume ground truth unless a test mode explicitly requests it.

## Ground Truth

- `/apex/sim/ground_truth/odom`
- `/apex/sim/ground_truth/path`
- `/apex/sim/ground_truth/perfect_map_points`
- `/apex/sim/ground_truth/status`

Use these topics to measure error and regressions.

## Capture and Mapping

```bash
./simulation/tools/sim/apex_recognition_tour_sim_capture.sh \
  --scenario tight_right_saturation --timeout-s 60

./simulation/tools/sim/apex_manual_mapping_up.sh \
  --scenario precision_fusion --rviz

./simulation/tools/sim/apex_manual_mapping_finish.sh
```

Results are stored under `simulation/data/`.

## Sensor-Fusion Research Stack

`apex_fusion_research` runs the simulation with ideal Gazebo sensors and adds
explicit, seeded error models: a stochastic LiDAR model and a raw MEMS IMU. It
also adds a pure strapdown INS that shows how inertial errors accumulate, along
with ground-truth evaluation and recording.

```bash
./simulation/tools/sim/apex_fusion_research_up.sh          # Gazebo GUI + RViz, car idle
./simulation/tools/sim/apex_fusion_research_up.sh --arm    # the car drives the recognition tour
ros2 run apex_fusion_research plot_ins_drift simulation/data/fusion_research/<run>
```

See the [package README](../simulation/ros2_ws/src/apex_fusion_research/README.md)
for the models, parameters and tools.

## Older Paths

`simulation/legacy/Simulateur/` is the earlier simulator. `spawn_rc_car.launch.py` is also a simpler launch than the current APEX flow. Both remain for reference or isolated experiments.

## Related Documentation

- [Quick Start](05_quick_start.md)
- [ROS Architecture](07_ros_architecture.md)
- [Mapping and Recording](18_mapping_and_recording_pipeline.md)
- [Troubleshooting](15_troubleshooting.md)
