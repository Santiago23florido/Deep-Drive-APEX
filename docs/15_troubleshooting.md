# Troubleshooting

## First Checks

Start with these commands:

```bash
pwd
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 pkg list | grep -E 'rc_sim_description|apex_telemetry|voiture_system'
ros2 topic list
```

If you are on native Windows PowerShell and `ros2` or `colcon` is not found, use WSL2/Linux. This repository is not validated as a native Windows ROS/Gazebo project.

## Build Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `colcon: command not found` | Colcon is not installed. | Install `python3-colcon-common-extensions`. |
| `rosdep: command not found` | `python3-rosdep` missing. | Install `python3-rosdep` and run `rosdep update`. |
| `Package 'apex_telemetry' not found` | Build did not include `APEX/ros2_ws/src`. | Build with `--base-paths src APEX/ros2_ws/src`. |
| Launch file not found after build | Workspace not sourced. | Run `source install/setup.bash`. |
| Missing optional package | Optional mode enabled without dependency. | Install the missing ROS package or disable that mode. |

Recommended build command:

```bash
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
```

## ROS Environment Issues

| Symptom | Check |
| --- | --- |
| No topics from expected nodes | `ros2 node list` |
| Topic exists but has no messages | `ros2 topic hz <topic>` |
| Message type unknown | `ros2 topic info <topic>` |
| Parameter value unclear | `ros2 param list <node>` and `ros2 param get <node> <param>` |
| Service missing | `ros2 service list` |
| TF missing | `ros2 run tf2_ros tf2_echo <parent> <child>` |

Always source ROS and the workspace in every new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Gazebo Simulation Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Gazebo does not start | Missing Gazebo or ROS-Gazebo packages. | Install `ros-jazzy-ros-gz-sim` and `ros-jazzy-ros-gz-bridge`. |
| Vehicle does not spawn | Workspace not sourced or URDF/Xacro failure. | Rebuild, source workspace, check launch output. |
| `/apex/sim/scan` missing | Bridge or sensor plugin not running. | Check `ros_gz_bridge`, URDF sensor topic, and Gazebo logs. |
| `/apex/sim/imu` missing | Bridge or IMU plugin not running. | Check `ros_gz_bridge` and URDF sensor configuration. |
| Vehicle does not move | Tracker not armed, no `/apex/cmd_vel_track`, or bridge missing. | Check tracker status, PWM topics, and `apex_gz_vehicle_bridge.py`. |
| Motion direction wrong | Steering or motor mapping issue. | Check vehicle bridge parameters and simulated PWM signs. |

Useful commands:

```bash
ros2 topic hz /apex/sim/scan
ros2 topic hz /apex/sim/imu
ros2 topic echo /apex/cmd_vel_track --once
ros2 topic echo /apex/sim/pwm/motor_dc --once
ros2 topic echo /apex/sim/pwm/steering_dc --once
```

## Real Blue-Car Sensor Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No IMU data | Wrong serial port, Nano not streaming, permission issue. | Check `/dev/ttyACM0`, run Nano preflight scripts, inspect Docker device mapping. |
| No LiDAR scan | Wrong serial port or baud rate. | Check `/dev/ttyUSB0`, verify baud `115200` or sensor-specific value. |
| LiDAR scans but localization unstable | Frame, heading offset, mounting, or scan filtering mismatch. | Check `frame_id`, heading offset, LiDAR position, and scan topic. |
| Data visible on Raspberry Pi but not PC | DDS discovery/network issue. | Check `ROS_DOMAIN_ID`, firewall, host networking, and discovery range. |

Useful commands inside the relevant ROS environment:

```bash
ros2 topic hz /apex/imu/data_raw
ros2 topic echo /apex/imu/data_raw --once
ros2 topic hz /lidar/scan_localization
ros2 topic echo /lidar/scan_localization --once
```

## Real Blue-Car Actuation Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No motor or steering output | Actuation backend disabled or PWM unavailable. | Check `cmd_vel_to_apex_actuation_node`, backend parameter, Docker privileged mode, and `/sys/class/pwm`. |
| Motor moves at startup | Neutral duty cycle wrong or ESC not calibrated. | Lift car, verify neutral duty cycle, calibrate ESC. |
| Steering saturated | Planner/tracker demands exceed hardware limit. | Check `/apex/vehicle/applied_steering_deg` and steering limit parameters. |
| Commands are published but bridge inactive | Tracker not armed or stale-data safety active. | Check `/apex/tracking/arm`, tracker status, and drive bridge status. |

Useful commands:

```bash
ros2 topic echo /apex/cmd_vel_track --once
ros2 topic echo /apex/vehicle/drive_bridge_status --once
ros2 topic echo /apex/vehicle/applied_speed_pct --once
ros2 topic echo /apex/vehicle/applied_steering_deg --once
```

## Docker Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Docker permission denied | User not in Docker group. | Add user to `docker`, log out and back in. |
| Container starts but devices missing | Device path mismatch. | Check `/dev/ttyACM0`, `/dev/ttyUSB0`, and Compose device mappings. |
| PWM path missing in container | `/sys/class/pwm` not mounted or PWM overlay missing. | Check Compose volume and Raspberry Pi boot overlay. |
| ROS topics do not appear outside container | Networking or domain mismatch. | Confirm host networking and `ROS_DOMAIN_ID=30`. |

Inspect logs:

```bash
docker ps
docker logs apex_pipeline --tail 200
```

## Planner and Tracker Issues

| Symptom | Likely cause | Diagnostic topics |
| --- | --- | --- |
| Planner never produces a path | No valid scans, bad odometry, or readiness not reached. | `/lidar/scan_localization`, `/apex/estimation/status`, `/apex/planning/recognition_tour_status` |
| Tracker does not command motion | Not armed or path stale. | `/apex/tracking/arm`, `/apex/tracking/recognition_tour_status`, `/apex/cmd_vel_track` |
| Abort due to path loss | Planner/tracker timing mismatch or local path dropout. | Planner status, local path history, tracker status. |
| Steering requested beyond hardware | Planner curvature too aggressive or steering limit low. | `/apex/cmd_vel_track`, `/apex/vehicle/applied_steering_deg` |

The local diagnostic report in the repository describes one recognition-tour failure with path-loss abort and steering saturation concerns. Treat it as a run-specific clue, not a universal diagnosis.

## Alternate `voiture_system` Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Nav2 does not start | Missing Nav2 packages or lifecycle issue. | Install Nav2 packages and inspect lifecycle nodes. |
| No `/map` | SLAM disabled or no LiDAR scans. | Enable SLAM and check `/lidar/scan`. |
| No `/odom` | Serial state or Ackermann odometry not running. | Check `/vehicle/speed_mps` and odometry node. |
| `ros2_control` simulation fails | Older Classic Gazebo path or malformed Xacro. | Check `src/voiture_system/urdf/ros2_control.xacro` and use APEX simulation if possible. |

## Related Documentation

- [Installation on Linux](03_installation_linux.md)
- [Installation on Windows](04_installation_windows.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md)

