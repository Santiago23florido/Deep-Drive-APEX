# Project Overview

## Objective

This repository contains software for an autonomous RC car platform. Its current direction is the **APEX** stack, which supports:

- Gazebo Sim simulation of the RC vehicle.
- Real-vehicle operation on the current intended target platform, the blue car ("voiture blue").
- Sensor ingestion from an IMU and RPLIDAR.
- Planar localization and mapping from IMU plus LiDAR data.
- Recognition-tour and curve-entry planning.
- Path tracking and conversion of velocity commands into motor and steering actuation.
- Capture scripts and run analysis tools for development and testing.

The project solves the problem of developing, testing, and operating a small autonomous vehicle across simulation and physical hardware while preserving enough instrumentation to diagnose failures in sensing, estimation, planning, tracking, and actuation.

## Current Recommended Workflow

The current recommended workflow is APEX-first:

| Use case | Recommended path |
| --- | --- |
| Gazebo simulation | `src/rc_sim_description` plus `APEX/tools/sim/apex_sim_up.sh` and `src/rc_sim_description/launch/apex_sim.launch.py` |
| Real blue car | `APEX/docker/docker-compose.yml`, `APEX/ros2_ws/src/apex_telemetry`, and `APEX/tools/core/apex_real_ready_up.sh` |
| Recognition-tour capture | `APEX/tools/capture/apex_recognition_tour_capture.sh` |
| Offline maps and run analysis | APEX run folders under `APEX/apex_*` and simulation run data under `src/rc_sim_description/data/runs` |

The root `src/voiture_system` package is still valuable, but it represents an alternate ROS 2 SLAM/Nav2-oriented path rather than the primary APEX recognition-tour workflow.

## Simulation and Real Vehicle

The simulation path uses Gazebo Sim, a URDF/Xacro vehicle model, Gazebo sensors, `ros_gz_bridge`, and APEX planner/tracker nodes. In simulation mode, the APEX actuation bridge publishes simulated PWM duty-cycle topics instead of writing to Raspberry Pi sysfs PWM.

The real-car path runs on the Raspberry Pi 5 mounted on the blue vehicle. It uses Docker Compose to start a ROS 2 Jazzy container with host networking and privileged access to serial devices and `/sys/class/pwm`. The main hardware inputs are:

- Nano IMU over serial.
- RPLIDAR over serial.
- Optional manual/gamepad control bridge over the network.

The main hardware outputs are:

- ESC motor PWM.
- Steering servo PWM.

## Major Subsystems

| Subsystem | Responsibility | Primary code |
| --- | --- | --- |
| Vehicle model and simulation | URDF/Xacro model, Gazebo worlds, simulation bridge, ground truth, run recorder | `src/rc_sim_description` |
| APEX telemetry and autonomy | IMU, LiDAR, kinematics, fusion, planning, tracking, actuation bridge | `APEX/ros2_ws/src/apex_telemetry` |
| APEX deployment tooling | Docker, real-ready scripts, capture scripts, PC helper scripts, systemd service | `APEX/docker`, `APEX/tools`, `APEX/systemd` |
| Alternate ROS 2 car stack | RPLIDAR publisher, serial state, Ackermann drive, odometry, adaptive controller, SLAM/Nav2 launch | `src/voiture_system` |
| Historical vehicle software | Earlier non-ROS Python vehicle stack and simulator, formerly mirrored as `full_soft/` | External historical context |
| LiDAR networking utilities | Split Raspberry/PC LiDAR workflow and DDS networking notes | `Lidar` |

## Technology Stack

| Area | Technologies |
| --- | --- |
| ROS distribution | ROS 2 Jazzy is the expected modern distribution. |
| Simulation | Gazebo Sim through `ros_gz_sim` and `ros_gz_bridge`. Some older launch paths use Classic Gazebo. |
| Build system | `colcon`, `ament_cmake`, `ament_python`. |
| Robot description | URDF/Xacro. |
| Visualization | RViz 2. |
| Mapping/localization | APEX IMU+LiDAR fusion, optional `slam_toolbox`, optional `robot_localization`, optional `rf2o_laser_odometry` in simulation modes. |
| Real deployment | Docker Compose, host networking, privileged device access, Raspberry Pi 5. |
| Hardware I/O | Serial IMU, serial RPLIDAR, Linux sysfs PWM, optional Arduino serial state in the alternate stack. |
| Analysis | Python scripts, CSV logs, JSON metadata, run folders, optional offline map refinement. |

## Current Vehicle Focus

The current intended real platform is the blue car. Earlier hardware documentation from the former external `full_soft` reference listed the blue car with an RPLIDAR A2, Raspberry Pi 5, ESC, steering components, motor, and related chassis hardware. That code tree is no longer versioned in `main`; the APEX stack is the modern real-car software path for this vehicle.

Some details are inferred from the codebase because the repository contains multiple historical documentation sets. When older documentation conflicts with the APEX scripts, this documentation treats the executable APEX scripts and ROS 2 package code as the more current source of truth.

## Related Documentation

- [System Architecture](06_system_architecture.md)
- [ROS Architecture](07_ros_architecture.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md)
