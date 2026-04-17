# ROS Architecture

This file explains the ROS 2 structure of the repository. If you are new to ROS, read [Glossary of ROS Terms](20_glossary_ros_terms.md) first.

## Workspace Layout

The repository is not a single flat ROS workspace. It contains ROS packages in two source roots:

```text
.
|-- src/
|   |-- rc_sim_description/
|   `-- voiture_system/
`-- APEX/
    `-- ros2_ws/
        `-- src/
            `-- apex_telemetry/
```

The APEX simulation wrapper builds all relevant packages together:

```bash
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
```

## ROS 2 Packages

In ROS, a package is the unit of build, installation, and discovery. This repository has three confirmed ROS 2 packages:

| Package | Build type | Role |
| --- | --- | --- |
| `rc_sim_description` | `ament_cmake` | Gazebo Sim vehicle description, worlds, simulation launch files, bridges, ground truth, run recorder, and mapping utilities. |
| `apex_telemetry` | `ament_python` | Current APEX autonomy package for IMU, LiDAR, estimation, planning, tracking, session management, and actuation bridge. |
| `voiture_system` | `ament_python` | Alternate car stack for real SLAM/Nav2 experiments, serial state, Ackermann drive, odometry, and older simulation bridge. |

## ROS Concepts in This Repository

| ROS concept | How it appears here |
| --- | --- |
| Node | A Python executable such as `recognition_tour_planner_node` or `rplidar_publisher_node`. |
| Topic | A named message stream such as `/apex/imu/data_raw` or `/apex/cmd_vel_track`. |
| Publisher | A node that writes messages to a topic. Example: `rplidar_publisher_node` publishes LaserScan messages. |
| Subscriber | A node that reads messages from a topic. Example: the fusion node subscribes to IMU and LiDAR topics. |
| Service | A request/response interface. APEX kinematics exposes reset and recalibration Trigger services. |
| Action | A long-running goal interface. No custom repository actions were found; Nav2 actions exist only when the alternate Nav2 path is launched. |
| Parameter | Runtime configuration attached to a node. Main APEX parameters are in `apex_params.yaml`. |
| TF | ROS transform tree between coordinate frames such as `base_link`, `laser`, `imu_link`, and odometry frames. |
| Launch file | A Python file that composes nodes into a runtime system. |

## Package Hierarchy

```text
Current APEX path
|-- rc_sim_description
|   |-- Gazebo worlds
|   |-- URDF/Xacro vehicle model
|   |-- Gazebo-to-ROS bridges
|   |-- simulation vehicle bridge
|   |-- ground truth and run recorder
|   `-- includes apex_telemetry launch in simulation mode
`-- apex_telemetry
    |-- sensor source nodes
    |-- kinematics and odometry nodes
    |-- IMU+LiDAR fusion node
    |-- planners
    |-- trackers
    |-- session manager
    `-- actuation bridge

Alternate path
`-- voiture_system
    |-- real RPLIDAR publisher
    |-- Arduino serial state node
    |-- Ackermann drive and odometry
    |-- adaptive track controller
    |-- optional SLAM toolbox
    `-- optional Nav2
```

## Node Categories

### Infrastructure Nodes

| Node or tool | Package | Purpose |
| --- | --- | --- |
| `robot_state_publisher` | ROS standard | Publishes TF from the URDF/Xacro robot description. |
| `joint_state_publisher` | ROS standard | Publishes joint states in older or simple simulation paths. |
| `ros_gz_bridge parameter_bridge` | `ros_gz_bridge` | Bridges Gazebo messages to ROS messages. |
| `static_transform_publisher` | `tf2_ros` | Publishes fixed transforms, for example from `base_link` to `laser`. |

### Sensor Nodes

| Node | Package | Typical role |
| --- | --- | --- |
| `nano_accel_serial_node` | `apex_telemetry` | Reads Nano IMU serial data and publishes ROS IMU and raw vector topics. |
| `rplidar_publisher_node` | `apex_telemetry` | Reads RPLIDAR data or simulated scans and publishes localization scans. |
| `rplidar_publisher_node` | `voiture_system` | Alternate RPLIDAR publisher for the `voiture_system` path. |
| `serial_state_node` | `voiture_system` | Reads Arduino state such as speed, wheel speed, steering, ultrasonic, and battery values. |

### Estimation Nodes

| Node | Package | Purpose |
| --- | --- | --- |
| `kinematics_estimator_node` | `apex_telemetry` | Estimates acceleration, velocity, position, heading, and corrected IMU data from raw IMU input. |
| `kinematics_odometry_node` | `apex_telemetry` | Converts APEX kinematics topics into odometry. |
| `imu_lidar_planar_fusion_node` | `apex_telemetry` | Fuses IMU and LiDAR data into planar odometry, path, status, and map points. |
| `ackermann_odometry_node` | `voiture_system` | Produces odometry and TF from vehicle speed and steering. |
| `robot_localization ekf_node` | External | Optional simulation estimator used by selected `apex_sim.launch.py` modes. |
| `rf2o_laser_odometry` | External | Optional simulation laser odometry source. |

### Planning and Tracking Nodes

| Node | Package | Purpose |
| --- | --- | --- |
| `recognition_tour_planner_node` | `apex_telemetry` | Builds recognition-tour route and local path from LiDAR and fused odometry. |
| `recognition_tour_tracker_node` | `apex_telemetry` | Tracks the recognition-tour local path and publishes `/apex/cmd_vel_track`. |
| `curve_entry_path_planner_node` | `apex_telemetry` | Detects and plans entry through a curve. |
| `curve_path_tracker_node` | `apex_telemetry` | Tracks the curve-entry path. |
| `adaptive_track_controller_node` | `voiture_system` | Alternate controller that can produce `/cmd_vel` from scans and map data. |

### Bridge and Hardware Nodes

| Node | Package | Purpose |
| --- | --- | --- |
| `cmd_vel_to_apex_actuation_node` | `apex_telemetry` | Converts `/apex/cmd_vel_track` into sysfs PWM or simulated PWM topics. |
| `apex_gz_vehicle_bridge.py` | `rc_sim_description` | Converts simulated PWM topics into Gazebo vehicle joint commands. |
| `ackermann_drive_node` | `voiture_system` | Converts `/cmd_vel` into sysfs PWM motor and steering commands. |
| `vehicle_sim_bridge_node` | `voiture_system` | Older simulation bridge for the alternate stack. |

### Utility and Recording Nodes

| Node | Package | Purpose |
| --- | --- | --- |
| `recognition_session_manager_node` | `apex_telemetry` | Coordinates recognition-session status and capture support. |
| `apex_sim_run_recorder.py` | `rc_sim_description` | Records simulation runs for later analysis. |
| `offline_submap_refiner.py` | `rc_sim_description` | Performs offline submap refinement for simulation workflows. |
| `apex_refined_sensorfusion_map_node.py` | `rc_sim_description` | Publishes refined sensor-fusion map output. |
| `apex_windows_gamepad_bridge_node` | `apex_telemetry` or `rc_sim_description` | Bridges Windows/manual gamepad commands into ROS topics. |

## Launch Files Compose the System

In ROS 2, a launch file starts multiple nodes and assigns parameters, remappings, namespaces, and conditions.

The most important launch files are:

| Launch file | Package | Role |
| --- | --- | --- |
| `launch/apex_sim.launch.py` | `rc_sim_description` | Current full Gazebo Sim entry point. Includes APEX pipeline in simulation mode. |
| `launch/apex_pipeline.launch.py` | `apex_telemetry` | Modular APEX pipeline for real and simulated backends. |
| `launch/bringup_real_slam_nav.launch.py` | `voiture_system` | Alternate real-car SLAM/Nav2 launch. |
| `launch/bringup_sim.launch.py` | `voiture_system` | Older Classic Gazebo and `ros2_control` simulation launch. |
| `launch/spawn_rc_car.launch.py` | `rc_sim_description` | Older simple Gazebo Sim vehicle spawn path. |

See [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md).

## Topics and Message Flow

The current APEX recognition-tour graph is organized around these topic groups:

```text
Sensor topics
  /apex/imu/data_raw
  /lidar/scan_localization

Estimation topics
  /apex/odometry/imu_raw
  /apex/odometry/imu_lidar_fused
  /apex/estimation/status
  /apex/estimation/path

Planning topics
  /apex/planning/recognition_tour_local_path
  /apex/planning/recognition_tour_route
  /apex/planning/recognition_tour_status

Tracking and command topics
  /apex/tracking/recognition_tour_status
  /apex/tracking/arm
  /apex/cmd_vel_track

Actuation topics
  /apex/vehicle/drive_bridge_status
  /apex/vehicle/applied_speed_pct
  /apex/vehicle/applied_steering_deg
  /apex/sim/pwm/motor_dc
  /apex/sim/pwm/steering_dc
```

For a full interface table, see [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md).

## Parameters

The main APEX parameter file is:

```text
APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml
```

It defines serial ports, baud rates, topic names, frames, simulation backends, fusion settings, planner and tracker tuning, actuation limits, and session-manager paths.

The alternate `voiture_system` stack uses:

```text
src/voiture_system/config/controllers.yaml
src/voiture_system/config/slam_toolbox_online_async.yaml
src/voiture_system/config/nav2_ackermann.yaml
```

## TF and Frames

TF describes how coordinate frames relate to one another.

Confirmed frames in the codebase include:

| Frame | Meaning |
| --- | --- |
| `base_link` | Main vehicle body frame. |
| `rear_axle` | Rear axle reference in the URDF/Xacro. |
| `laser` | RPLIDAR frame. |
| `imu_link` | IMU frame. |
| `camera_link` | Camera frame in the simulated robot model. |
| `odom_imu_lidar_fused` | Fused odometry frame used by the APEX fusion node. |
| `odom` | Standard odometry frame used by alternate paths and some simulation modes. |
| `map` | Map frame used by SLAM and Nav2 when enabled. |

`robot_state_publisher` publishes robot transforms from the URDF/Xacro in simulation. Real APEX launch also publishes at least a fixed laser transform using `static_transform_publisher`.

## URDF/Xacro

The current simulated vehicle description is:

```text
src/rc_sim_description/urdf/rc_car.urdf.xacro
```

It defines the chassis, wheels, LiDAR, IMU, camera, dimensions, joints, and Gazebo sensor plugins. Gazebo publishes simulated LiDAR on `/apex/sim/scan` and simulated IMU on `/apex/sim/imu`.

The alternate `voiture_system` package has additional URDF/Xacro and `ros2_control` files. One known issue is a malformed leading `v` in `src/voiture_system/urdf/ros2_control.xacro`; see [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md).

## Gazebo and ROS Integration

Gazebo Sim is not ROS itself. The bridge connects Gazebo transport topics to ROS topics:

```text
Gazebo LaserScan -> ros_gz_bridge -> sensor_msgs/msg/LaserScan
Gazebo IMU       -> ros_gz_bridge -> sensor_msgs/msg/Imu
Gazebo clock     -> ros_gz_bridge -> rosgraph_msgs/msg/Clock
```

The simulation-specific `apex_gz_vehicle_bridge.py` converts ROS simulated PWM duty-cycle topics into Gazebo joint commands.

## Current vs Alternate ROS Graphs

| Graph | Recommended for | Notes |
| --- | --- | --- |
| APEX recognition-tour graph | Current simulation and blue-car workflow. | Most actively integrated path. |
| APEX curve-entry graph | Curve detection and tracking tests. | Still APEX, but different planner/tracker pair. |
| `voiture_system` SLAM/Nav2 graph | Alternate real-car navigation experiments. | Uses `/cmd_vel`, `/odom`, `/map`, `slam_toolbox`, and Nav2. |
| Older `rc_sim_description` spawn graph | Simple Gazebo experiments. | Overlaps with APEX simulation but is less current. |

## Related Documentation

- [System Architecture](06_system_architecture.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Glossary of ROS Terms](20_glossary_ros_terms.md)

