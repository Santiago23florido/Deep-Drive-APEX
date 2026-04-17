# Hardware Interfaces

## Scope

This file documents the hardware-facing interfaces found in the repository. The current focus is the APEX real-car workflow for the blue vehicle. The alternate `voiture_system` hardware path is documented separately in this file because it uses a different ROS graph.

## Blue-Car APEX Hardware Stack

| Hardware | Interface | APEX component | Main topics or outputs |
| --- | --- | --- | --- |
| Nano IMU | Serial, default `/dev/ttyACM0` | `nano_accel_serial_node` | `/apex/imu/data_raw`, `/apex/imu/acceleration/raw`, `/apex/imu/angular_velocity/raw` |
| RPLIDAR | Serial, default `/dev/ttyUSB0` | `rplidar_publisher_node` | `/lidar/scan`, `/lidar/scan_localization` |
| ESC motor control | Linux sysfs PWM | `cmd_vel_to_apex_actuation_node` | PWM channel 0, plus status topics |
| Steering servo | Linux sysfs PWM | `cmd_vel_to_apex_actuation_node` | PWM channel 1, plus status topics |
| PC/manual bridge | Network/DDS or bridge protocol | `apex_windows_gamepad_bridge_node` and PC tools | Manual status and session control topics |

## Serial Devices

APEX defaults:

| Device | Default path | Default baud | Notes |
| --- | --- | --- | --- |
| Nano IMU | `/dev/ttyACM0` | `115200` | Used by the APEX IMU source. |
| RPLIDAR | `/dev/ttyUSB0` | `115200` | APEX default for the blue-car workflow. |

Check devices on Linux:

```bash
ls -l /dev/ttyACM*
ls -l /dev/ttyUSB*
dmesg | tail -n 50
```

If permissions fail, add the user to the `dialout` group:

```bash
sudo usermod -aG dialout "$USER"
```

Then log out and back in.

## LiDAR Baud Caveat

Older hardware documentation mentions:

- Yellow car LiDAR: `256000`.
- Blue car LiDAR: `115200`.

APEX defaults to `115200`. If no scans appear, test the actual sensor baud instead of assuming the default is correct for every unit.

## PWM Actuation

The APEX actuation bridge supports two backends:

| Backend | Use case | Output |
| --- | --- | --- |
| `sysfs_pwm` | Real blue car | Writes Linux sysfs PWM for ESC and steering. |
| `sim_pwm_topic` | Gazebo simulation | Publishes `/apex/sim/pwm/motor_dc` and `/apex/sim/pwm/steering_dc`. |

Default real hardware mapping:

| Function | Channel | Notes |
| --- | --- | --- |
| Motor/ESC | PWM channel `0` | Often mapped to Raspberry Pi GPIO12 when using the documented two-channel overlay. |
| Steering servo | PWM channel `1` | Often mapped to Raspberry Pi GPIO13 when using the documented two-channel overlay. |

Older LiDAR/hardware notes mention this Raspberry Pi overlay:

```text
dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
```

Verify actual Raspberry Pi configuration before assuming these pins are active.

## Docker Device Access

`APEX/docker/docker-compose.yml` gives the `apex_pipeline` container access to:

- `/dev/ttyACM0`
- `/dev/ttyUSB0`
- `/sys/class/pwm`

It also runs with:

- `privileged: true`
- `network_mode: host`

If hardware works on the host but not in Docker, inspect the Compose mappings and the actual device names.

## ROS 2 Networking

The tested multi-machine setup uses:

```bash
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

When a PC or WSL2 machine needs to see Raspberry Pi topics:

- Put both machines on the same network.
- Keep the same `ROS_DOMAIN_ID`.
- Check firewall rules on Windows.
- Use WSL2 mirrored networking if available.
- Avoid mixing different DDS discovery assumptions.

## PC and Windows Gamepad Bridge

PC-side tools live under:

```text
APEX/tools/pc/
APEX/tools/windows/
```

These tools support manual interaction, gamepad bridging, and session monitoring. They are auxiliary to the APEX real pipeline.

## Alternate `voiture_system` Hardware Interfaces

The alternate `voiture_system` path uses:

| Hardware/input | Node | Topics |
| --- | --- | --- |
| RPLIDAR | `voiture_system/rplidar_publisher_node` | `/lidar/scan` |
| Arduino state serial | `serial_state_node` | `/vehicle/speed_mps`, `/rear_wheel_speed`, `/steering_angle`, ultrasonic and battery topics where implemented |
| Motor and steering PWM | `ackermann_drive_node` | Consumes `/cmd_vel`, publishes command/state topics |
| Odometry | `ackermann_odometry_node` | Publishes `/odom` and TF |

This stack can optionally connect to `slam_toolbox` and Nav2.

## Simulation Hardware Equivalents

In Gazebo Sim:

| Real hardware | Simulation equivalent |
| --- | --- |
| RPLIDAR | Gazebo LaserScan sensor bridged to `/apex/sim/scan`. |
| Nano IMU | Gazebo IMU sensor bridged to `/apex/sim/imu`. |
| ESC | Simulated motor duty-cycle topic and Gazebo wheel commands. |
| Steering servo | Simulated steering duty-cycle topic and Gazebo steering joint commands. |
| Track/world | Gazebo world file. |

## Pre-Run Hardware Checklist

1. Confirm serial device names.
2. Confirm IMU and LiDAR baud rates.
3. Confirm `/sys/class/pwm` is available.
4. Confirm motor neutral duty cycle.
5. Confirm steering center and direction.
6. Confirm Docker sees devices.
7. Confirm ROS topics have stable rates.
8. Keep the vehicle restrained until status topics are healthy.

## Related Documentation

- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Configuration Reference](17_configuration_reference.md)
- [Troubleshooting](15_troubleshooting.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)

