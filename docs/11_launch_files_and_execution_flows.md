# Launch Files and Execution Flows

## Recommended Entry Points

| Use case | Entry point | Status |
| --- | --- | --- |
| APEX Gazebo simulation | `APEX/tools/sim/apex_sim_up.sh` | Recommended |
| Direct APEX Gazebo launch | `ros2 launch rc_sim_description apex_sim.launch.py` | Recommended for developers |
| Real blue-car APEX stack | `APEX/tools/core/apex_real_ready_up.sh` | Recommended |
| Real recognition-tour capture | `APEX/tools/capture/apex_recognition_tour_capture.sh` | Recommended for test runs |
| Alternate real SLAM/Nav2 stack | `ros2 launch voiture_system bringup_real_slam_nav.launch.py` | Alternate |
| Older simple Gazebo spawn | `ros2 launch rc_sim_description spawn_rc_car.launch.py` | Legacy/overlapping |
| Older Classic Gazebo path | `ros2 launch voiture_system bringup_sim.launch.py` | Legacy/alternate |

## ROS Launch Files

| Launch file | Package | What it starts | Use case |
| --- | --- | --- | --- |
| `src/rc_sim_description/launch/apex_sim.launch.py` | `rc_sim_description` | Gazebo Sim, robot model, Gazebo bridges, APEX pipeline in simulation mode, simulation vehicle bridge, optional RViz/SLAM/recording/refinement. | Current simulation. |
| `APEX/ros2_ws/src/apex_telemetry/launch/apex_pipeline.launch.py` | `apex_telemetry` | Modular APEX nodes selected by launch arguments. | Real or simulated APEX pipeline. |
| `src/voiture_system/launch/bringup_real_slam_nav.launch.py` | `voiture_system` | Alternate RPLIDAR, serial state, Ackermann drive, odometry, optional adaptive controller, SLAM toolbox, Nav2, RViz. | Alternate real-car stack. |
| `src/voiture_system/launch/bringup_sim.launch.py` | `voiture_system` | Classic Gazebo, robot state publisher, spawn entity, controller spawners, high-level controller, simulation bridge. | Older simulation path. |
| `src/rc_sim_description/launch/spawn_rc_car.launch.py` | `rc_sim_description` | Gazebo Sim, robot state/joint state publishers, spawn, bridge helpers, older controller helpers, optional RViz. | Older/simple simulation. |

## APEX Simulation Execution Flow

Command:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Flow:

```text
apex_sim_up.sh
  -> source /opt/ros/jazzy/setup.bash
  -> optionally colcon build src and APEX/ros2_ws/src
  -> source install/setup.bash
  -> ros2 launch rc_sim_description apex_sim.launch.py ...
  -> Gazebo Sim starts
  -> vehicle model is spawned
  -> /clock, /apex/sim/scan, and /apex/sim/imu are bridged
  -> apex_pipeline.launch.py is included with simulation backends
  -> tracker publishes /apex/cmd_vel_track
  -> actuation bridge publishes /apex/sim/pwm/*
  -> apex_gz_vehicle_bridge.py commands Gazebo joints
```

Useful launch arguments:

| Argument | Typical value | Purpose |
| --- | --- | --- |
| `scenario` | `baseline` | Selects world and scenario settings. |
| `rviz` | `true` or `false` | Starts RViz. |
| `control_mode` | `recognition_tour` | Selects autonomous/manual control path. |
| `use_slam` | `true` or `false` | Enables SLAM-related simulation path. |
| `mapping_mode` | varies | Enables mapping-specific behavior where supported. |
| `estimation_mode` | `current`, `ideal`, `rf2o_ekf` | Selects estimator behavior in simulation. |
| `record_run` | `true` or `false` | Starts simulation run recording where supported. |

## APEX Real Blue-Car Execution Flow

Command:

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/core/apex_real_ready_up.sh
```

Flow:

```text
apex_real_ready_up.sh
  -> sets APEX feature flags for real-ready operation
  -> calls capture/core startup helpers
  -> docker compose starts apex_pipeline
  -> start_apex_pipeline.sh starts selected ROS nodes
  -> IMU and LiDAR drivers publish sensor topics
  -> fusion publishes odometry and status
  -> recognition planner publishes route and local path
  -> recognition tracker publishes /apex/cmd_vel_track
  -> actuation bridge writes sysfs PWM
```

The real stack is controlled mostly through environment variables consumed by Docker Compose and `start_apex_pipeline.sh`.

## APEX Recognition-Tour Capture Flow

Command:

```bash
./APEX/tools/capture/apex_recognition_tour_capture.sh --run-id recognition_tour_test_01 --timeout-s 60
```

Flow:

```text
capture script
  -> configures APEX recognition-tour feature flags
  -> starts or reuses apex_pipeline
  -> performs readiness checks
  -> arms /apex/tracking/arm
  -> records selected topics
  -> writes logs, CSV files, summaries, and diagnostics
  -> stops or leaves services according to script behavior
```

Recorded topics include planner status, local path, route, fused odometry, LiDAR scans, tracker status, and drive bridge status.

## Alternate `voiture_system` Real Flow

Command:

```bash
ros2 launch voiture_system bringup_real_slam_nav.launch.py
```

Typical flow:

```text
rplidar_publisher_node
  -> /lidar/scan
serial_state_node
  -> /vehicle/speed_mps and related state
ackermann_odometry_node
  -> /odom and TF
optional slam_toolbox
  -> /map
optional Nav2 or adaptive controller
  -> /cmd_vel
ackermann_drive_node
  -> sysfs PWM motor and steering
```

This path uses standard navigation topics such as `/cmd_vel`, `/odom`, and `/map`, unlike the APEX path which uses `/apex/...` topics.

## Mapping and Offline Analysis Flows

| Flow | Entry point | Output |
| --- | --- | --- |
| Simulation run recording | `apex_sim_run_recorder.py` through launch arguments | Simulation run directories and topic logs. |
| Simulated recognition-tour capture | `APEX/tools/sim/apex_recognition_tour_sim_capture.sh` | Simulated recognition-tour logs and analysis artifacts. |
| Real recognition-tour capture | `APEX/tools/capture/apex_recognition_tour_capture.sh` | `APEX/apex_recognition_tour/<run>/`. |
| Curve-track capture | `APEX/tools/capture/apex_curve_track_capture.sh` | `APEX/apex_curve_track/<run>/`. |
| Raw/forward capture | `APEX/tools/capture/apex_rect_sensorfus_capture.sh` and related scripts | `APEX/apex_forward_raw/<run>/` or similar. |
| Offline submap refinement | `offline_submap_refiner.py` or APEX offline feature flag | Refined map artifacts. |

## Legacy and Compatibility Wrappers

The `APEX/tools` root contains some wrapper scripts that forward to organized subfolders such as:

- `APEX/tools/core`
- `APEX/tools/capture`
- `APEX/tools/sim`
- `APEX/tools/firmware`

Use the organized subfolder paths in new instructions. Treat root-level wrappers as compatibility conveniences unless a script specifically documents otherwise.

## Related Documentation

- [Quick Start](05_quick_start.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Mapping and Recording Pipeline](18_mapping_and_recording_pipeline.md)

