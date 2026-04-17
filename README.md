# Deep Drive APEX

Deep Drive APEX is a ROS 2 and Gazebo-based autonomous RC car project. It includes a current APEX workflow for Gazebo simulation and the real blue vehicle ("voiture blue"), plus alternate and legacy stacks that are documented separately.

This root README is the main navigation hub. The detailed technical documentation lives in [`docs/`](docs/), while supplementary report material is grouped under [`docs/Reportes/`](docs/Reportes/).

## Documentation at a Glance

The documentation explains the project from both user and developer perspectives: installation, quick start workflows, ROS package architecture, Gazebo simulation, real blue-car operation, launch flows, topics, parameters, data recording, hardware interfaces, troubleshooting, and known legacy areas.

## Start Here

| Goal | Recommended page |
| --- | --- |
| Understand the project | [📘 Project Overview](docs/01_project_overview.md) |
| Find the full documentation map | [🧭 Documentation Index](docs/00_index.md) |
| Run something quickly | [🚀 Quick Start](docs/05_quick_start.md) |
| Work with Gazebo simulation | [🕹 Gazebo Simulation](docs/08_simulation_gazebo.md) |
| Work with the real blue car | [🚗 Blue Vehicle Real System](docs/09_blue_vehicle_real_system.md) |
| Learn the ROS graph | [🧠 ROS Architecture](docs/07_ros_architecture.md) |

## Documentation Index

### 📘 Overview

- [📘 Documentation Index](docs/00_index.md) - entry point for the full documentation set.
- [🌍 Project Overview](docs/01_project_overview.md) - objectives, scope, technology stack, and current workflow.
- [🗂 Repository Structure](docs/02_repository_structure.md) - top-level folders, ROS packages, auxiliary areas, and legacy code.

### ⚙️ Setup

- [🐧 Installation on Linux](docs/03_installation_linux.md) - Ubuntu, ROS 2 Jazzy, Gazebo, dependencies, and build steps.
- [🪟 Installation on Windows](docs/04_installation_windows.md) - WSL2 guidance and native Windows caveats.
- [🚀 Quick Start](docs/05_quick_start.md) - fastest build, simulation, and real-car commands.
- [🎛 Configuration Reference](docs/17_configuration_reference.md) - important YAML, JSON, launch, and Docker configuration.

### 🧠 Architecture

- [🏗 System Architecture](docs/06_system_architecture.md) - high-level data flow and subsystem responsibilities.
- [🧠 ROS Architecture](docs/07_ros_architecture.md) - packages, nodes, topics, TF, URDF/Xacro, and Gazebo bridges.
- [📦 Packages and Modules](docs/10_packages_and_modules.md) - package-by-package responsibilities and executables.
- [▶️ Launch Files and Execution Flows](docs/11_launch_files_and_execution_flows.md) - recommended entry points and runtime flows.
- [🔌 Topics, Services, Actions, and Parameters](docs/12_topics_services_actions_parameters.md) - ROS interface reference.

### 🕹 Simulation

- [🕹 Gazebo Simulation](docs/08_simulation_gazebo.md) - Gazebo Sim worlds, vehicle model, sensors, bridges, and control flow.
- [🗺 Mapping and Recording Pipeline](docs/18_mapping_and_recording_pipeline.md) - simulation and real capture workflows, offline refinement, and analysis.
- [📊 Data and Runs](docs/13_data_and_runs.md) - run artifacts, logs, CSV files, maps, trajectories, and diagnostics.

### 🚗 Real Vehicle

- [🚗 Blue Vehicle Real System](docs/09_blue_vehicle_real_system.md) - APEX real-car workflow for the current blue vehicle.
- [🔋 Hardware Interfaces](docs/19_hardware_interfaces.md) - IMU, LiDAR, PWM, Docker devices, networking, and gamepad bridge.

### 🛠 Developer Reference

- [🛠 Developer Guide](docs/14_developer_guide.md) - how to extend nodes, launch files, configuration, and documentation.
- [🧪 Troubleshooting](docs/15_troubleshooting.md) - build, Gazebo, Docker, ROS, networking, and hardware debugging.
- [⚠️ Known Limitations and Legacy Parts](docs/16_known_limitations_and_legacy_parts.md) - alternate stacks, older code, and cleanup recommendations.
- [📚 ROS Glossary](docs/20_glossary_ros_terms.md) - beginner-friendly ROS and Gazebo terminology.

### 📎 Reports

Supplementary French report materials are stored separately from the main documentation set:

- [📄 Simulation Status Report](docs/Reportes/simulation-status/SimulationStatus.pdf) - PDF report and LaTeX sources for the simulator status.
- [📄 Software Selection Report](docs/Reportes/software-selection/SoftwareSelection.pdf) - PDF report and LaTeX sources for simulation software selection.
- [📁 Report Folder](docs/Reportes/) - grouped report sources, PDFs, build artifacts, and related assets.

## Current Workflow Summary

| Workflow | Status | Main documentation |
| --- | --- | --- |
| APEX Gazebo simulation | Recommended | [🕹 Gazebo Simulation](docs/08_simulation_gazebo.md) |
| APEX real blue-car stack | Recommended | [🚗 Blue Vehicle Real System](docs/09_blue_vehicle_real_system.md) |
| `voiture_system` SLAM/Nav2 stack | Alternate | [📦 Packages and Modules](docs/10_packages_and_modules.md) |
| `full_soft` and older utilities | Legacy or auxiliary | [⚠️ Known Limitations and Legacy Parts](docs/16_known_limitations_and_legacy_parts.md) |

For new work, start with the APEX documentation and use the alternate or legacy sections only when maintaining those specific paths.
