# Repository Structure

## Top-Level Tree

```text
.
|-- APEX/
|   |-- docker/
|   |-- docs/
|   |-- ros2_ws/
|   |   `-- src/apex_telemetry/
|   |-- systemd/
|   `-- tools/
|-- Lidar/
|-- docs/
|-- src/
|   |-- rc_sim_description/
|   `-- voiture_system/
|-- README.md
|-- run_basic_track.sh
`-- recognition_tour_diagnostic_report.md
```

The exact working tree may also contain generated Python dependencies, run artifacts, and untracked diagnostic files. Those are not part of the core source layout.

## Directory Roles

| Path | Role | Classification |
| --- | --- | --- |
| `APEX/` | Current APEX real-car tooling, Docker deployment, capture scripts, analysis artifacts, and APEX ROS 2 workspace. | Central |
| `APEX/ros2_ws/src/apex_telemetry/` | Main APEX ROS 2 package for sensors, estimation, planning, tracking, and actuation bridge. | Central |
| `src/rc_sim_description/` | Gazebo Sim vehicle model, worlds, launch files, simulation bridge, mapping and run-recorder scripts. | Central |
| `src/voiture_system/` | Alternate ROS 2 car stack with real SLAM/Nav2 launch and older simulation bridge. | Alternate |
| `Lidar/` | Distributed LiDAR utility stack and networking notes for Raspberry Pi plus PC/WSL setups. | Auxiliary |
| `docs/` | Repository-level documentation. This new Markdown set lives here alongside existing LaTeX/PDF/image artifacts. | Documentation |
| `README.md` | Root documentation gateway and workflow summary. | Documentation |
| `run_basic_track.sh` | Direct Gazebo world startup helper. | Utility |
| `recognition_tour_diagnostic_report.md` | Local diagnostic report for a recognition-tour run. Useful context, but not treated as committed source of truth. | Diagnostic artifact |

## Where the ROS Code Lives

There are three ROS 2 packages:

| Package | Location | Build type | Current role |
| --- | --- | --- | --- |
| `rc_sim_description` | `src/rc_sim_description` | `ament_cmake` | Gazebo Sim model, worlds, simulation launch, bridges, mapping and recording tools. |
| `apex_telemetry` | `APEX/ros2_ws/src/apex_telemetry` | `ament_python` | Current APEX real/sim autonomy nodes. |
| `voiture_system` | `src/voiture_system` | `ament_python` | Alternate ROS 2 real-car SLAM/Nav2 and older simulation path. |

The APEX simulation launch builds and uses all three package roots through:

```bash
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
```

## Gazebo-Related Assets

| Path | Content |
| --- | --- |
| `src/rc_sim_description/launch/apex_sim.launch.py` | Current recommended Gazebo Sim entry point. |
| `src/rc_sim_description/launch/spawn_rc_car.launch.py` | Older simpler Gazebo Sim path. |
| `src/rc_sim_description/urdf/rc_car.urdf.xacro` | Vehicle URDF/Xacro, sensors, links, joints, and Gazebo plugins. |
| `src/rc_sim_description/worlds/` | Track worlds such as `basic_track.world`, `tight_right_saturation.world`, and other test scenarios. |
| `src/rc_sim_description/config/apex_sim_scenarios.json` | Scenario definitions and simulation parameter overrides. |
| `src/rc_sim_description/scripts/` | Simulation bridge, ground truth, mapping, run recorder, teleop, and helper nodes. |

## Hardware and Vehicle Logic

| Path | Hardware responsibility |
| --- | --- |
| `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/nano_accel_serial_node.py` | Nano IMU serial ingestion. |
| `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/rplidar_publisher_node.py` | APEX RPLIDAR ingestion and LaserScan publishing. |
| `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/cmd_vel_to_apex_actuation_node.py` | Converts ROS velocity commands to PWM or simulated PWM output. |
| `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/actuation.py` | Low-level actuation support used by the APEX bridge. |
| `APEX/docker/docker-compose.yml` | Real-car container, device mappings, host networking, and runtime environment. |
| `src/voiture_system/voiture_system/ackermann_drive_node.py` | Alternate sysfs PWM motor and steering control. |
| `src/voiture_system/voiture_system/serial_state_node.py` | Alternate Arduino serial state ingestion. |
| `src/voiture_system/voiture_system/rplidar_publisher_node.py` | Alternate RPLIDAR publisher. |

Earlier non-ROS hardware drivers and Python control code were previously mirrored under `full_soft/`. That tree has been removed from `main`; use the current APEX and `voiture_system` code for maintained hardware behavior.

## Central, Auxiliary, and Legacy Areas

### Central

- `APEX/ros2_ws/src/apex_telemetry`
- `APEX/docker`
- `APEX/tools/core`
- `APEX/tools/capture`
- `APEX/tools/sim`
- `src/rc_sim_description`

These are the primary areas for current simulation and real blue-car work.

### Alternate

- `src/voiture_system`

This package is still ROS 2 code and can run a real-car SLAM/Nav2 stack, but it is not the primary APEX recognition-tour pipeline.

### Auxiliary

- `Lidar`
- `APEX/tools/pc`
- `APEX/tools/windows`
- `APEX/systemd`
- `APEX/docs`

These support networking, PC-side control, Windows gamepad bridging, service installation, and additional workflow notes.

### Legacy or Historical

- `src/rc_sim_description/launch/spawn_rc_car.launch.py`
- Parts of the root `README.md`
- Some older APEX README statements that describe a reduced APEX scope
- The former external `full_soft/` Python stack, removed from `main` and no longer carried as a dedicated report in this repository

These areas and references remain useful for hardware history and previous approaches, but they should not be treated as the current recommended architecture.

## Related Documentation

- [Packages and Modules](10_packages_and_modules.md)
- [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
- [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md)
