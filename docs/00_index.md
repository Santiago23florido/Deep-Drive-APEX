# APEX Repository Documentation

This documentation set explains the autonomous RC car software in this repository, with the APEX stack treated as the current recommended workflow for both Gazebo simulation and the real blue vehicle ("voiture blue").

The repository contains several generations of code. The current documentation therefore separates the recommended APEX workflow from alternate, older, and auxiliary code paths:

- **Recommended current stack:** `APEX/`, `APEX/ros2_ws/src/apex_telemetry`, and the `rc_sim_description` simulation package.
- **Alternate ROS 2 stack:** `src/voiture_system`, which provides a SLAM/Nav2-oriented real-car and Classic Gazebo path.
- **Older or auxiliary code:** `full_soft/` and `Lidar/`, which contain previous non-ROS vehicle software and split LiDAR networking utilities.

## Who Should Read This

| Reader | Start here | Goal |
| --- | --- | --- |
| New user | [Project Overview](01_project_overview.md), then [Quick Start](05_quick_start.md) | Understand what the project does and run a basic workflow. |
| ROS developer | [ROS Architecture](07_ros_architecture.md), then [Packages and Modules](10_packages_and_modules.md) | Understand package boundaries, nodes, topics, launch files, and parameters. |
| Simulation user | [Simulation with Gazebo](08_simulation_gazebo.md) | Run the Gazebo Sim workflow and understand the simulated vehicle data flow. |
| Real-car operator | [Blue Vehicle Real System](09_blue_vehicle_real_system.md) | Prepare and run the APEX real-car stack on the blue vehicle. |
| Maintainer | [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md), then [Developer Guide](14_developer_guide.md) | Understand overlap, cleanup needs, and extension rules. |

## Documentation Map

| File | Purpose |
| --- | --- |
| [01_project_overview.md](01_project_overview.md) | Project objective, technology stack, and major subsystems. |
| [02_repository_structure.md](02_repository_structure.md) | Top-level folders, central code, auxiliary areas, and legacy areas. |
| [03_installation_linux.md](03_installation_linux.md) | Linux and Raspberry Pi installation guidance. |
| [04_installation_windows.md](04_installation_windows.md) | Windows guidance, with WSL2 as the recommended path. |
| [05_quick_start.md](05_quick_start.md) | Fastest commands for simulation and real-car startup. |
| [06_system_architecture.md](06_system_architecture.md) | High-level system flow across sensing, estimation, planning, control, and actuation. |
| [07_ros_architecture.md](07_ros_architecture.md) | ROS 2 architecture, packages, nodes, topics, services, TF, URDF/Xacro, and Gazebo integration. |
| [08_simulation_gazebo.md](08_simulation_gazebo.md) | Gazebo Sim workflow, worlds, robot model, bridges, and simulation control flow. |
| [09_blue_vehicle_real_system.md](09_blue_vehicle_real_system.md) | Real blue-car deployment, hardware interfaces, Docker pipeline, and capture workflow. |
| [10_packages_and_modules.md](10_packages_and_modules.md) | Package-by-package and module-by-module reference. |
| [11_launch_files_and_execution_flows.md](11_launch_files_and_execution_flows.md) | Launch files, shell entry points, and runtime execution flows. |
| [12_topics_services_actions_parameters.md](12_topics_services_actions_parameters.md) | ROS interface reference for topics, services, actions, and parameters. |
| [13_data_and_runs.md](13_data_and_runs.md) | Recorded run folders, logs, CSV files, maps, trajectories, and analysis outputs. |
| [14_developer_guide.md](14_developer_guide.md) | Extension guidance for nodes, launch files, configuration, and tests. |
| [15_troubleshooting.md](15_troubleshooting.md) | Build, ROS environment, Gazebo, Docker, networking, and hardware debugging. |
| [16_known_limitations_and_legacy_parts.md](16_known_limitations_and_legacy_parts.md) | Known limitations, ambiguous behavior, duplicate paths, and cleanup recommendations. |
| [17_configuration_reference.md](17_configuration_reference.md) | Main configuration files and parameters. |
| [18_mapping_and_recording_pipeline.md](18_mapping_and_recording_pipeline.md) | Mapping, recording, replay, and offline refinement workflows. |
| [19_hardware_interfaces.md](19_hardware_interfaces.md) | IMU, LiDAR, PWM, serial, networking, and gamepad interfaces. |
| [20_glossary_ros_terms.md](20_glossary_ros_terms.md) | Beginner-friendly ROS and Gazebo terminology used in this repository. |

## Recommended Reading Orders

### New Users

1. [Project Overview](01_project_overview.md)
2. [Installation on Linux](03_installation_linux.md) or [Installation on Windows](04_installation_windows.md)
3. [Quick Start](05_quick_start.md)
4. [Simulation with Gazebo](08_simulation_gazebo.md) or [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
5. [Troubleshooting](15_troubleshooting.md)

### Developers

1. [Repository Structure](02_repository_structure.md)
2. [System Architecture](06_system_architecture.md)
3. [ROS Architecture](07_ros_architecture.md)
4. [Packages and Modules](10_packages_and_modules.md)
5. [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
6. [Developer Guide](14_developer_guide.md)

### ROS-Focused Readers

1. [Glossary of ROS Terms](20_glossary_ros_terms.md)
2. [ROS Architecture](07_ros_architecture.md)
3. [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
4. [Configuration Reference](17_configuration_reference.md)
5. [Simulation with Gazebo](08_simulation_gazebo.md)

## Documentation Scope

This set is based on an audit of package manifests, setup files, launch files, shell scripts, Docker files, configuration files, source files, existing README files, and run artifacts in the repository. Where a behavior is not explicitly documented by the code or existing files, it is labeled as **inferred from the codebase**.

