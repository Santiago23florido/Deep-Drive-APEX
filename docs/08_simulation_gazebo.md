# Simulation with Gazebo

## Objective

The simulation stack provides a Gazebo Sim environment for developing and validating the APEX vehicle pipeline without moving the real car. It simulates:

- Vehicle model and track worlds.
- LiDAR scans.
- IMU messages.
- Vehicle motion response to motor and steering commands.
- Optional ground truth, maps, recording, and offline refinement.

The recommended simulation path is `apex_sim.launch.py`, not the older `spawn_rc_car.launch.py`.

## Main Simulation Package

| Package | Role |
| --- | --- |
| `rc_sim_description` | Owns the Gazebo worlds, URDF/Xacro vehicle model, simulation launch file, simulation bridge, ground-truth tools, and recording tools. |
| `apex_telemetry` | Runs the same APEX sensor, estimation, planner, tracker, and actuation bridge nodes in simulation backend mode. |
| `voiture_system` | Built by the simulation wrapper for compatibility, but not the main APEX simulation architecture. |

## Recommended Entry Point

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Direct launch:

```bash
ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=true
```

Common options:

| Option | Meaning |
| --- | --- |
| `--scenario baseline` | Selects the scenario from `apex_sim_scenarios.json`. |
| `--rviz` | Starts RViz. |
| `--slam` | Enables SLAM-related simulation path. |
| `--refined-map` | Enables refined map visualization where supported. |
| `--fixed-map-run` | Uses a fixed-map run workflow where supported. |
| `--control-mode recognition_tour` | Uses the recognition-tour planner/tracker. This is the default in the current launch. |
| `--control-mode manual_xbox` | Uses manual Xbox-style control. |
| `--control-mode manual_windows_bridge` | Uses the Windows/manual bridge. |
| `--arm` | Arms the recognition-tour tracker if supported by the wrapper. |
| `--skip-build` | Skips the wrapper's build step. |

## Worlds and Scenarios

World files live in:

```text
src/rc_sim_description/worlds/
```

Confirmed worlds include:

| World | Purpose |
| --- | --- |
| `basic_track.world` | Baseline track world. |
| `flat_track.world` | Simpler flat environment. |
| `tight_right_saturation.world` | Scenario for challenging right-turn saturation. |
| `outer_long_inner_short.world` | Track geometry variant. |
| `narrowing_false_corridor.world` | Scenario with narrowing or false-corridor behavior. |

Scenario definitions live in:

```text
src/rc_sim_description/config/apex_sim_scenarios.json
```

Confirmed scenario names include:

- `baseline`
- `precision_fusion`
- `tight_right_saturation`
- `outer_long_inner_short`
- `startup_pose_jump`
- `narrowing_false_corridor`

## Robot Description

The vehicle model is:

```text
src/rc_sim_description/urdf/rc_car.urdf.xacro
```

It defines:

- Chassis geometry.
- Front and rear wheels.
- Steering joints.
- Rear axle reference.
- LiDAR frame.
- IMU frame.
- Camera frame.
- Gazebo sensor plugins.
- Gazebo joint control plugins.

Important simulated sensor topics:

| Sensor | ROS topic |
| --- | --- |
| LiDAR | `/apex/sim/scan` |
| IMU | `/apex/sim/imu` |
| Camera | `/camera/image_raw` in older/simple paths |

## Launch Flow

The current `apex_sim.launch.py` flow is:

```text
Load scenario configuration
  -> process URDF/Xacro
  -> start Gazebo Sim
  -> start robot_state_publisher
  -> spawn vehicle in Gazebo
  -> bridge Gazebo clock, LiDAR, and IMU into ROS
  -> start APEX pipeline in simulation backend mode
  -> start simulation vehicle bridge
  -> optionally start RViz, SLAM, ground truth, recorder, and map tools
```

## Gazebo and ROS Bridge

The launch uses `ros_gz_bridge parameter_bridge` for:

| Gazebo data | ROS topic | ROS type |
| --- | --- | --- |
| Clock | `/clock` | `rosgraph_msgs/msg/Clock` |
| LiDAR | `/apex/sim/scan` | `sensor_msgs/msg/LaserScan` |
| IMU | `/apex/sim/imu` | `sensor_msgs/msg/Imu` |

The APEX LiDAR and IMU nodes can use these topics as simulation backends instead of real serial devices.

## Simulation Control Flow

```text
recognition_tour_tracker_node
  -> /apex/cmd_vel_track
  -> cmd_vel_to_apex_actuation_node
  -> /apex/sim/pwm/motor_dc
  -> /apex/sim/pwm/steering_dc
  -> apex_gz_vehicle_bridge.py
  -> Gazebo wheel velocity and steering joint commands
```

This mirrors the real blue-car control path, where `cmd_vel_to_apex_actuation_node` writes to sysfs PWM instead of publishing simulated PWM topics.

## Ground Truth and Recording

Simulation-specific tools provide diagnostic information:

| Tool | Purpose |
| --- | --- |
| `apex_ground_truth_node.py` | Publishes simulated ground-truth odometry, path, map points, and status. |
| `apex_ground_truth_tf_bridge.py` | Bridges ground-truth pose into TF where enabled. |
| `apex_sim_run_recorder.py` | Records simulation runs for offline analysis. |
| `offline_submap_refiner.py` | Performs offline submap refinement. |
| `offline_similarity_monitor.py` | Monitors similarity metrics in offline workflows. |
| `apex_general_track_mapper.py` | Builds or supports general track maps. |
| `apex_general_track_map_publisher.py` | Publishes general track map data. |

Ground-truth topics include:

```text
/apex/sim/ground_truth/odom
/apex/sim/ground_truth/path
/apex/sim/ground_truth/perfect_map_points
/apex/sim/ground_truth/status
```

## Main Simulation Use Cases

| Use case | Recommended command |
| --- | --- |
| Baseline recognition-tour simulation | `./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz` |
| Arm recognition-tour tracker | `./APEX/tools/sim/apex_arm_recognition_tour.sh` |
| SLAM-enabled simulation | `./APEX/tools/sim/apex_sim_up.sh --scenario baseline --slam --rviz` |
| Simulated recognition-tour capture | `./APEX/tools/sim/apex_recognition_tour_sim_capture.sh --scenario tight_right_saturation --timeout-s 60` |
| Manual mapping | `./APEX/tools/sim/apex_manual_mapping_up.sh` |

## Older Simulation Path

`src/rc_sim_description/launch/spawn_rc_car.launch.py` is an older simple Gazebo Sim launch. It starts Gazebo, robot state publishing, vehicle spawning, and older bridge/control helper scripts. It remains useful for isolated experiments, but it overlaps with and is less current than `apex_sim.launch.py`.

The `voiture_system` package also has `bringup_sim.launch.py`, which uses Classic Gazebo and `ros2_control`. That path should be treated as alternate or legacy unless you are specifically working on the `voiture_system` architecture.

## Related Documentation

- [Quick Start](05_quick_start.md)
- [ROS Architecture](07_ros_architecture.md)
- [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
- [Mapping and Recording Pipeline](18_mapping_and_recording_pipeline.md)

