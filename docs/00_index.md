# Documentation Index

## Current implementation and publication entry points

| Topic | Current reference |
| --- | --- |
| Latest accepted physical demonstration, data and English figures | [Run 20261024_220316](../real_vehicle/docs/paper/20261024_220316/README.md) |
| Original approved physical runtime and preserved evidence | [Run 20261024_005311](../real_vehicle/docs/paper/20261024_005311/README.md) |
| Real runtime graph and deployed configuration | [Race stack](../real_vehicle/docs/race_stack.md) |
| PC SSH preflight and recognition-to-SLAM run | [Real-vehicle README commands](../real_vehicle/README.md#from-the-pc-ssh-preflight-and-recognition-to-slam-run) |
| Current simulator and historical demonstrations | [Simulation README](../simulation/README.md) |
| Remaining simulation experiments for the preprint | [Readiness report](reports/preprint_readiness_20261024.md) |

The active physical implementation is `apex_race` + `apex_telemetry`, learned
odometry and `slam_toolbox`. Older architecture pages may describe the earlier
`voiture_system` or curve-entry flows; they do not replace the approved runtime.

## Who Should Read This

This documentation is for people continuing development in simulation, developers of the SLAM-enabled physical vehicle, and maintainers who need to distinguish operational code from archived material.

## Documentation Map

| Area | Document | Content |
| --- | --- | --- |
| Overview | [Project Overview](01_project_overview.md) | Goals, scope, and current workflows. |
| Structure | [Repository Structure](02_repository_structure.md) | Simulation, real vehicle, and archive separation. |
| Setup | [Linux Installation](03_installation_linux.md) | ROS 2 Jazzy, Gazebo, and workspaces. |
| Setup | [Windows/WSL2 Installation](04_installation_windows.md) | Simulation on Windows and gamepad bridge. |
| Start | [Quick Start](05_quick_start.md) | Minimum simulation and real-vehicle commands. |
| Architecture | [System Architecture](06_system_architecture.md) | Main data flows. |
| Architecture | [ROS Architecture](07_ros_architecture.md) | Packages, nodes, TF, and topics. |
| Simulation | [Gazebo Simulation](08_simulation_gazebo.md) | Scenarios, sensors, and execution. |
| Real vehicle | [SLAM-Enabled Real Vehicle](09_blue_vehicle_real_system.md) | Approved race runtime, hardware and earlier stack distinction. |
| Code | [Packages and Modules](10_packages_and_modules.md) | Package responsibilities. |
| Execution | [Launch Files and Flows](11_launch_files_and_execution_flows.md) | Recommended and alternate entry points. |
| Interfaces | [Topics, Services, and Parameters](12_topics_services_actions_parameters.md) | Practical ROS reference. |
| Data | [Data and Runs](13_data_and_runs.md) | Runs, maps, logs, and results. |
| Development | [Developer Guide](14_developer_guide.md) | Where to change code and how to validate it. |
| Support | [Troubleshooting](15_troubleshooting.md) | Common failures. |
| Legacy | [Known Limitations and Legacy Code](16_known_limitations_and_legacy_parts.md) | Current status and known limits. |
| Configuration | [Configuration Reference](17_configuration_reference.md) | YAML, JSON, and parameters. |
| Mapping | [Mapping and Recording](18_mapping_and_recording_pipeline.md) | Capture and reconstruction. |
| Hardware | [Hardware Interfaces](19_hardware_interfaces.md) | LiDAR, Arduino, PWM, and networking. |
| Glossary | [ROS and Gazebo Terms](20_glossary_ros_terms.md) | Core definitions. |

## Recommended Reading Orders

### Continuing Without the Car

1. [Project Overview](01_project_overview.md)
2. [Linux Installation](03_installation_linux.md) or [Windows/WSL2 Installation](04_installation_windows.md)
3. [Quick Start](05_quick_start.md)
4. [Gazebo Simulation](08_simulation_gazebo.md)
5. [Mapping and Recording](18_mapping_and_recording_pipeline.md)

### Working on the SLAM-Enabled Real Vehicle

1. [SLAM-Enabled Real Vehicle](09_blue_vehicle_real_system.md)
2. [Hardware Interfaces](19_hardware_interfaces.md)
3. [ROS Architecture](07_ros_architecture.md)
4. [Configuration Reference](17_configuration_reference.md)
5. [Troubleshooting](15_troubleshooting.md)

### Inspecting the Old No-SLAM Stack

1. [Known Limitations and Legacy Code](16_known_limitations_and_legacy_parts.md)
2. `archive/legacy/full_soft/README.md`
3. `archive/legacy/full_soft/code/main.py`

## Documentation Scope

The current source of truth is the `ReestructuredApex` layout: `simulation/`, `real_vehicle/`, and `archive/`. PDF and LaTeX reports preserve historical context and may mention earlier directory names.
