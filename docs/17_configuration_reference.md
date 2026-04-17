# Configuration Reference

## Main Configuration Files

| File | Applies to | Purpose |
| --- | --- | --- |
| `APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml` | Current APEX real and simulation stack | Main node parameters for sensors, estimation, planning, tracking, actuation, and session management. |
| `src/rc_sim_description/config/apex_sim_scenarios.json` | APEX Gazebo Sim | World selection, spawn pose, vehicle model parameters, and simulation noise/override profiles. |
| `src/voiture_system/config/controllers.yaml` | Alternate `voiture_system` simulation/control | `ros2_control` controller settings. |
| `src/voiture_system/config/slam_toolbox_online_async.yaml` | Alternate `voiture_system` SLAM | SLAM toolbox settings. |
| `src/voiture_system/config/nav2_ackermann.yaml` | Alternate `voiture_system` Nav2 | Nav2 stack configuration. |
| `APEX/docker/docker-compose.yml` | Real APEX deployment | Device mappings, ROS environment, feature flags, and container runtime settings. |

## APEX Parameter File

Primary file:

```text
APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml
```

This file is the main reference for the APEX runtime. It is used by:

- `apex_pipeline.launch.py`
- `start_apex_pipeline.sh`
- Real Docker startup scripts
- Simulation launch when including the APEX pipeline

## IMU Parameters

Node:

```text
nano_accel_serial_node
```

Important parameters:

| Parameter | Typical value | Meaning |
| --- | --- | --- |
| `serial_port` | `/dev/ttyACM0` | Serial device for the Nano IMU. |
| `baud_rate` | `115200` | Serial baud rate. |
| `accel_topic` | `/apex/imu/acceleration/raw` | Raw acceleration topic. |
| `gyro_topic` | `/apex/imu/angular_velocity/raw` | Raw angular velocity topic. |
| `imu_topic` | `/apex/imu/data_raw` | Raw ROS IMU topic. |
| `frame_id` | `imu_link` | IMU frame. |
| `transport_backend` | `serial` or simulation mode | Selects real serial or simulated input. |
| `sim_imu_topic` | `/apex/sim/imu` | Simulated IMU topic. |

## LiDAR Parameters

Node:

```text
rplidar_publisher_node
```

Important parameters:

| Parameter | Typical value | Meaning |
| --- | --- | --- |
| `serial_port` | `/dev/ttyUSB0` | RPLIDAR serial device. |
| `baud_rate` | `115200` | RPLIDAR baud rate in APEX defaults. |
| `scan_topic` | `/lidar/scan` | General scan topic. |
| `localization_scan_topic` | `/lidar/scan_localization` | Scan topic used by APEX localization and planning. |
| `slam_scan_topic` | `/lidar/scan_slam` | Scan topic intended for SLAM workflows. |
| `frame_id` | `laser` | LiDAR frame. |
| `source_backend` | `rplidar` or `sim_scan` | Selects real RPLIDAR or simulation scan input. |
| `sim_scan_topic` | `/apex/sim/scan` | Simulated scan topic. |
| `heading_offset_deg` | configured value | Heading correction for LiDAR mounting. |
| `fov_deg` | configured value | Field-of-view selection. |

Baud caveat: older docs mention different LiDAR baud rates for different vehicles. Verify the actual blue-car sensor before changing defaults.

## Kinematics and Odometry Parameters

Nodes:

```text
kinematics_estimator_node
kinematics_odometry_node
```

These parameters define raw input topics, corrected IMU output, kinematics outputs, odometry topic, and status publishing. The kinematics estimator also exposes reset and static recalibration services.

Important topics include:

- `/apex/kinematics/acceleration`
- `/apex/kinematics/velocity`
- `/apex/kinematics/position`
- `/apex/kinematics/heading`
- `/apex/kinematics/status`
- `/apex/odometry/imu_raw`

## IMU+LiDAR Fusion Parameters

Node:

```text
imu_lidar_planar_fusion_node
```

Important parameters:

| Parameter area | Meaning |
| --- | --- |
| Input topics | Raw IMU and localization LiDAR scan topics. |
| Output odometry | Fused odometry topic, normally `/apex/odometry/imu_lidar_fused`. |
| Status output | Fusion status topic, normally `/apex/estimation/status`. |
| Map output | Live and full map point topics. |
| TF settings | Whether to publish fused odometry transform. |
| Frames | Fused odometry and vehicle child frames. |

## Recognition-Tour Planner Parameters

Node:

```text
recognition_tour_planner_node
```

Important parameter areas:

- LiDAR scan topic.
- Fused odometry topic.
- Fusion status topic.
- Arm topic.
- Local path topic.
- Route topic.
- Planner status topic.
- Planning rate.
- Status publish rate.
- Horizon and timeout settings.

Typical outputs:

- `/apex/planning/recognition_tour_local_path`
- `/apex/planning/recognition_tour_route`
- `/apex/planning/recognition_tour_status`

## Recognition-Tour Tracker Parameters

Node:

```text
recognition_tour_tracker_node
```

Important parameter areas:

- Local path input.
- Planner status input.
- Fused odometry input.
- Fusion status input.
- Arm topic.
- Command output topic `/apex/cmd_vel_track`.
- Path stale thresholds.
- Tracking status output.

## Actuation Bridge Parameters

Node:

```text
cmd_vel_to_apex_actuation_node
```

Important parameters:

| Parameter | Meaning |
| --- | --- |
| `cmd_vel_topic` | Input velocity command, normally `/apex/cmd_vel_track`. |
| `actuation_backend` | `sysfs_pwm` for real vehicle, `sim_pwm_topic` for simulation. |
| `motor_pwm_channel` | Sysfs PWM channel for ESC, typically `0`. |
| `steering_pwm_channel` | Sysfs PWM channel for steering, typically `1`. |
| `pwm_frequency_hz` | PWM frequency, typically `50`. |
| `motor_neutral_duty_cycle_pct` | ESC neutral duty cycle, around `7.5`. |
| `steering_limit_deg` | Steering clamp, around `18 deg`. |
| `sim_motor_pwm_topic` | Simulation motor duty-cycle topic. |
| `sim_steering_pwm_topic` | Simulation steering duty-cycle topic. |

## Simulation Scenario Configuration

File:

```text
src/rc_sim_description/config/apex_sim_scenarios.json
```

This file defines scenario-level simulation settings:

| Field type | Meaning |
| --- | --- |
| World file | Which Gazebo world to load. |
| Spawn pose | Initial vehicle position and yaw. |
| Vehicle bridge parameters | Wheel radius, wheelbase, track width, steering limits, motor model. |
| Sensor overrides | Noise, drift, latency, and distortion profiles. |
| Scenario name | User-facing scenario selected by launch argument. |

## Docker Environment Configuration

File:

```text
APEX/docker/docker-compose.yml
```

Important settings:

| Setting | Meaning |
| --- | --- |
| `ROS_DOMAIN_ID=30` | DDS domain for multi-machine discovery. |
| `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET` | DDS discovery range used by tested setups. |
| `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` | DDS middleware implementation. |
| `APEX_ENABLE_*` | Feature flags consumed by startup scripts. |
| Device mappings | IMU and LiDAR serial devices. |
| `/sys/class/pwm` volume | PWM output access. |

## Alternate `voiture_system` Configuration

`voiture_system` launch arguments configure:

- RPLIDAR port and baud.
- Arduino serial port.
- Whether SLAM is enabled.
- Whether Nav2 is enabled.
- Whether RViz is enabled.
- Controller and odometry behavior.

Important difference: the alternate real launch defaults its RPLIDAR baud differently in some paths. Validate against the actual sensor.

## Related Documentation

- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Hardware Interfaces](19_hardware_interfaces.md)

