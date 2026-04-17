# Blue Vehicle Real System

## Scope

This file documents the real-car side of the repository with the blue car ("voiture blue") as the current intended target vehicle. The current recommended real-car workflow is the APEX stack under `APEX/`, not the older `full_soft` stack.

Some hardware details come from older repository documentation, while the runtime behavior comes from the current APEX Docker, launch, configuration, and capture scripts. Where the connection is inferred from the codebase, it is labeled.

## Recommended Real-Car Stack

| Layer | Current APEX component |
| --- | --- |
| Deployment | `APEX/docker/docker-compose.yml` and `APEX/run_apex.sh` |
| Startup | `APEX/tools/core/apex_real_ready_up.sh` |
| ROS package | `APEX/ros2_ws/src/apex_telemetry` |
| Parameters | `APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml` |
| Capture | `APEX/tools/capture/apex_recognition_tour_capture.sh` |
| Shutdown | `APEX/tools/core/apex_core_down.sh` |
| Optional service | `APEX/systemd/apex-real-ready.service` |

## Vehicle Hardware

The older hardware inventory identifies the blue car with hardware including:

- Raspberry Pi 5.
- RPLIDAR A2-series sensor.
- ESC for the motor.
- Steering servo.
- RC motor and chassis components.
- Battery and switch hardware.
- Ultrasonic sensor in older hardware notes.

The current APEX runtime uses:

- Nano IMU serial stream.
- RPLIDAR serial stream.
- Linux sysfs PWM for ESC and steering.
- Optional PC/manual gamepad bridge.

Inferred from the codebase: the current APEX real-blue-car workflow does not use every older `full_soft` hardware feature directly. It focuses on IMU, LiDAR, PWM actuation, and run capture.

## Docker Runtime

The real APEX pipeline runs in a Docker Compose service named:

```text
apex_pipeline
```

The Compose file is:

```text
APEX/docker/docker-compose.yml
```

Important runtime properties:

| Property | Purpose |
| --- | --- |
| `privileged: true` | Allows access to low-level device and PWM interfaces. |
| `network_mode: host` | Allows ROS 2 DDS discovery and direct topic exchange on the host network. |
| `/dev/ttyACM0` | Default Nano IMU device mapping. |
| `/dev/ttyUSB0` | Default RPLIDAR device mapping. |
| `/sys/class/pwm:/sys/class/pwm` | PWM output for ESC and steering servo. |
| `ROS_DOMAIN_ID=30` | DDS domain used by tested multi-machine workflows. |
| `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` | DDS implementation used in the Compose environment. |

The container command is:

```text
APEX/ros2_ws/scripts/start_apex_pipeline.sh
```

## Startup Commands

On the Raspberry Pi:

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/core/apex_real_ready_up.sh
```

Stop the stack:

```bash
./tools/core/apex_core_down.sh
```

The real-ready script enables the main APEX features by default:

- IMU source.
- LiDAR source.
- Kinematics and odometry where configured.
- IMU+LiDAR fusion.
- Recognition-tour planner.
- Recognition-tour tracker.
- `cmd_vel` to actuation bridge.
- Recognition-session manager.
- Optional offline submap refinement.

## Sensor Flow

```text
Nano IMU serial device
  -> nano_accel_serial_node
  -> /apex/imu/acceleration/raw
  -> /apex/imu/angular_velocity/raw
  -> /apex/imu/data_raw

RPLIDAR serial device
  -> apex rplidar_publisher_node
  -> /lidar/scan
  -> /lidar/scan_localization
```

The APEX parameter file defaults to:

| Device | Default port | Default baud | Notes |
| --- | --- | --- | --- |
| Nano IMU | `/dev/ttyACM0` | `115200` | Current APEX default. |
| RPLIDAR | `/dev/ttyUSB0` | `115200` | APEX default. Older docs mention different car/sensor baud settings. |

Important caveat: older hardware documentation says the yellow car LiDAR used `256000` baud while the blue car used `115200`. The APEX default is `115200`; verify the actual sensor before a run.

## Estimation and Planning Flow

```text
/apex/imu/data_raw + /lidar/scan_localization
  -> imu_lidar_planar_fusion_node
  -> /apex/odometry/imu_lidar_fused
  -> /apex/estimation/status
  -> /apex/estimation/path
  -> recognition_tour_planner_node
  -> /apex/planning/recognition_tour_local_path
  -> /apex/planning/recognition_tour_route
  -> recognition_tour_tracker_node
  -> /apex/cmd_vel_track
```

The recognition-tour planner and tracker are the current APEX path for autonomous test runs.

## Actuation Flow

```text
/apex/cmd_vel_track
  -> cmd_vel_to_apex_actuation_node
  -> speed and steering clamps
  -> /apex/vehicle/applied_speed_pct
  -> /apex/vehicle/applied_steering_deg
  -> /apex/vehicle/drive_bridge_status
  -> sysfs PWM channel 0 for motor
  -> sysfs PWM channel 1 for steering
```

Default APEX actuation parameters include:

- Motor PWM channel `0`.
- Steering PWM channel `1`.
- PWM frequency `50 Hz`.
- Motor neutral duty cycle around `7.5%`.
- Steering limit around `18 deg`.

Check [Configuration Reference](17_configuration_reference.md) before changing these values.

## Recognition-Tour Capture

Run a recognition-tour capture:

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/capture/apex_recognition_tour_capture.sh --run-id recognition_tour_test_01 --timeout-s 60
```

The script configures the real pipeline, records planner/tracker/fusion/drive status, and writes run artifacts under:

```text
APEX/apex_recognition_tour/
```

Typical outputs include:

- `capture_meta.json`
- `recognition_tour_record.log`
- `docker_tail.log`
- `lidar_points.csv`
- planner status summaries
- local path history
- route and trajectory CSV files
- diagnostic reports
- drive bridge status traces

## Differences From Simulation

| Concern | Simulation | Real blue car |
| --- | --- | --- |
| Sensors | Gazebo publishes ideal or noisy simulated sensors. | Serial devices produce real data, dropouts, noise, latency, and calibration issues. |
| Actuation | Commands go to simulated PWM topics and Gazebo joints. | Commands write to ESC and steering servo PWM. |
| Ground truth | Available through simulation nodes. | Not available unless external tracking is added. |
| Safety | Vehicle cannot injure people. | Vehicle can move unexpectedly. Use physical safety procedures. |
| Time | May use simulated time. | Uses wall time. |
| Repeatability | High, scenario-controlled. | Lower, depends on battery, floor, track, sensor mounting, and network state. |

## Shared Components With Simulation

The APEX design reuses these components across simulation and real operation:

- Recognition-tour planner.
- Recognition-tour tracker.
- IMU+LiDAR fusion logic.
- Command topic `/apex/cmd_vel_track`.
- Actuation bridge interface, with backend changed by parameter.
- Status topics used by capture and diagnostics.

## Safety Checklist

Before enabling motion:

1. Confirm the car is on a safe test surface or lifted.
2. Confirm ESC neutral behavior.
3. Confirm steering center and direction.
4. Confirm LiDAR publishes stable scans.
5. Confirm IMU publishes stable data.
6. Confirm `/apex/vehicle/drive_bridge_status` reports a safe state.
7. Confirm the tracker is armed only when ready.
8. Keep a physical way to stop the car.

## Alternate Real-Car Stack

`src/voiture_system/launch/bringup_real_slam_nav.launch.py` provides a separate real-car stack with:

- RPLIDAR publisher.
- Arduino serial state.
- Ackermann drive and odometry.
- Optional adaptive track controller.
- Optional `slam_toolbox`.
- Optional Nav2.

This is documented as an alternate path, not the current recommended blue-car APEX workflow.

## Related Documentation

- [Hardware Interfaces](19_hardware_interfaces.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Mapping and Recording Pipeline](18_mapping_and_recording_pipeline.md)
- [Troubleshooting](15_troubleshooting.md)

