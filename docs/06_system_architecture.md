# System Architecture

## High-Level View

The project is organized around the same autonomy loop in simulation and on the real vehicle:

```text
Sensors
  -> sensor drivers or Gazebo bridges
  -> estimation and mapping
  -> planner
  -> tracker/controller
  -> vehicle command
  -> simulator or real actuation
  -> vehicle motion
  -> sensors
```

The APEX stack keeps this structure consistent by switching backends:

- In simulation, sensors come from Gazebo and actuation is sent as simulated PWM topics.
- On the real blue car, sensors come from serial devices and actuation writes to Linux sysfs PWM.

## Main Runtime Paths

| Path | Purpose | Main entry point |
| --- | --- | --- |
| APEX simulation | Develop and test behavior in Gazebo Sim. | `APEX/tools/sim/apex_sim_up.sh` or `rc_sim_description apex_sim.launch.py` |
| APEX real blue car | Run the current real-car recognition-tour pipeline. | `APEX/tools/core/apex_real_ready_up.sh` |
| APEX capture | Record and analyze real runs. | `APEX/tools/capture/apex_recognition_tour_capture.sh` |
| Alternate ROS 2 stack | Run a separate SLAM/Nav2-oriented real or Classic Gazebo workflow. | `voiture_system bringup_real_slam_nav.launch.py` or `bringup_sim.launch.py` |

## APEX Simulation Flow

```text
Gazebo Sim vehicle
  -> simulated LaserScan /apex/sim/scan
  -> simulated IMU /apex/sim/imu
  -> ros_gz_bridge
  -> apex_telemetry sensor backends in sim mode
  -> IMU+LiDAR fusion
  -> recognition-tour planner
  -> recognition-tour tracker
  -> /apex/cmd_vel_track
  -> cmd_vel_to_apex_actuation_node in sim_pwm_topic mode
  -> /apex/sim/pwm/motor_dc and /apex/sim/pwm/steering_dc
  -> apex_gz_vehicle_bridge.py
  -> Gazebo wheel and steering joint commands
```

Simulation also publishes ground-truth data through `apex_ground_truth_node.py`, which is useful for debugging, maps, and recorded-run analysis.

## Real Blue-Car Flow

```text
Nano IMU + RPLIDAR
  -> serial drivers in apex_telemetry
  -> /apex/imu/data_raw and /lidar/scan_localization
  -> kinematics and IMU+LiDAR fusion
  -> /apex/odometry/imu_lidar_fused
  -> recognition-tour planner
  -> recognition-tour tracker
  -> /apex/cmd_vel_track
  -> cmd_vel_to_apex_actuation_node in sysfs_pwm mode
  -> ESC motor PWM and steering servo PWM
```

The real stack is normally started inside the `apex_pipeline` Docker Compose service. The container runs with host networking and privileged access to serial devices and `/sys/class/pwm`.

## Subsystem Responsibilities

| Subsystem | Simulation implementation | Real blue-car implementation |
| --- | --- | --- |
| Vehicle body and environment | Gazebo world and URDF/Xacro model in `rc_sim_description`. | Physical RC car chassis and track. |
| LiDAR source | Gazebo LaserScan bridged to ROS. | RPLIDAR serial device through APEX publisher. |
| IMU source | Gazebo IMU bridged to ROS. | Nano IMU serial stream through APEX publisher. |
| Estimation | Same APEX fusion nodes, with simulated inputs. | Same APEX fusion nodes, with real inputs. |
| Planning | Recognition-tour or curve-entry APEX planners. | Same planners, tuned for real sensor constraints. |
| Tracking | APEX path tracker nodes. | Same trackers, with real safety limits and stale-data checks. |
| Actuation bridge | Publishes simulated PWM duty-cycle topics. | Writes ESC and steering servo PWM through sysfs. |
| Vehicle response | Gazebo physics and bridge scripts. | Physical motor, steering, wheel slip, latency, and battery behavior. |

## Control Loop Overview

The recommended recognition-tour loop is:

1. Sensor drivers publish IMU and LiDAR messages.
2. Estimation nodes produce odometry, map points, and status.
3. The recognition-tour planner produces a local path and route status.
4. The recognition-tour tracker converts the local path into a velocity command.
5. The actuation bridge clamps and converts the velocity command to the backend output.
6. The simulator or real vehicle moves.
7. Sensor feedback updates the next planning cycle.

The main command topic between tracking and actuation is:

```text
/apex/cmd_vel_track
```

The main real/sim actuation split happens inside:

```text
cmd_vel_to_apex_actuation_node
```

## Sensing, Perception, Control, and Actuation

| Layer | APEX nodes and files | Typical topics |
| --- | --- | --- |
| Sensing | `nano_accel_serial_node`, `rplidar_publisher_node`, Gazebo bridges | `/apex/imu/data_raw`, `/lidar/scan_localization`, `/apex/sim/scan`, `/apex/sim/imu` |
| Estimation | `kinematics_estimator_node`, `kinematics_odometry_node`, `imu_lidar_planar_fusion_node` | `/apex/odometry/imu_raw`, `/apex/odometry/imu_lidar_fused`, `/apex/estimation/status` |
| Planning | `recognition_tour_planner_node`, `curve_entry_path_planner_node` | `/apex/planning/recognition_tour_local_path`, `/apex/planning/recognition_tour_status` |
| Tracking | `recognition_tour_tracker_node`, `curve_path_tracker_node` | `/apex/cmd_vel_track`, `/apex/tracking/recognition_tour_status` |
| Actuation | `cmd_vel_to_apex_actuation_node`, `apex_gz_vehicle_bridge.py` | `/apex/vehicle/drive_bridge_status`, `/apex/sim/pwm/*`, sysfs PWM |
| Recording and analysis | capture scripts, `apex_sim_run_recorder.py`, analysis scripts | CSV, JSON, logs, maps, trajectories |

## Alternate `voiture_system` Architecture

The `voiture_system` package represents a separate stack:

```text
RPLIDAR + Arduino serial state
  -> /lidar/scan, /vehicle/speed_mps, /rear_wheel_speed, /steering_angle
  -> SLAM or Nav2 or adaptive controller
  -> /cmd_vel
  -> ackermann_drive_node
  -> sysfs PWM motor and steering
  -> ackermann_odometry_node
  -> /odom and TF
```

This stack is useful for SLAM/Nav2 experiments and real-car bringup, but it is not the current recommended APEX recognition-tour path.

## Runtime Design Principle

The strongest architectural idea in the current codebase is backend substitution:

- Use the same planning and tracking concepts in simulation and real life.
- Substitute sensor and actuation backends at launch time.
- Capture status topics and run artifacts so failures can be diagnosed after a test.

## Related Documentation

- [ROS Architecture](07_ros_architecture.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)

