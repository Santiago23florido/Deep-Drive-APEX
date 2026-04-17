# Packages and Modules

## Summary

| Package or area | Classification | Role |
| --- | --- | --- |
| `rc_sim_description` | Core simulation | Gazebo Sim model, worlds, launch, bridges, ground truth, recording, and mapping utilities. |
| `apex_telemetry` | Core APEX | Sensors, kinematics, fusion, planning, tracking, session management, and actuation bridge. |
| `voiture_system` | Alternate ROS 2 stack | RPLIDAR, serial state, Ackermann drive, odometry, adaptive control, SLAM/Nav2. |
| `APEX/tools` | Core tooling | Real startup, capture, simulation wrappers, PC tools, compatibility wrappers. |
| `APEX/docker` | Core deployment | Docker image and Compose service for real blue-car operation. |
| `full_soft` | Legacy | Older non-ROS Python vehicle stack and simulator. |
| `Lidar` | Auxiliary | Distributed LiDAR networking utilities and documentation. |

## `rc_sim_description`

| Item | Details |
| --- | --- |
| Location | `src/rc_sim_description` |
| Build type | `ament_cmake` |
| Primary use | Current Gazebo Sim workflow. |
| Classification | Core simulation package. |

### Important Files

| Path | Purpose |
| --- | --- |
| `package.xml` | Package dependencies for simulation and mapping utilities. |
| `CMakeLists.txt` | Installs launch, URDF, worlds, config, meshes, RViz, and executable scripts. |
| `launch/apex_sim.launch.py` | Recommended full Gazebo Sim launch. |
| `launch/spawn_rc_car.launch.py` | Older/simple simulation launch. |
| `urdf/rc_car.urdf.xacro` | Simulated RC car model. |
| `worlds/` | Gazebo track worlds. |
| `config/apex_sim_scenarios.json` | Scenario definitions. |
| `scripts/apex_gz_vehicle_bridge.py` | Converts simulated PWM topics to Gazebo commands. |
| `scripts/apex_ground_truth_node.py` | Publishes simulation ground truth. |
| `scripts/apex_sim_run_recorder.py` | Records simulation runs. |
| `scripts/offline_submap_refiner.py` | Offline refinement support. |

### Executables Installed by `CMakeLists.txt`

| Executable | Role |
| --- | --- |
| `apex_gz_vehicle_bridge.py` | Simulation vehicle actuation bridge. |
| `apex_cmd_vel_to_sim_pwm_node.py` | Sim-only command-to-PWM helper. |
| `apex_ground_truth_node.py` | Ground-truth publishing. |
| `apex_ground_truth_tf_bridge.py` | Ground-truth TF bridge. |
| `apex_refined_sensorfusion_map_node.py` | Refined map publishing. |
| `offline_submap_refiner.py` | Offline submap refinement. |
| `offline_similarity_monitor.py` | Offline similarity monitoring. |
| `apex_sim_run_recorder.py` | Run recording. |
| `online_distance_field_seed_node.py` | Distance-field seed support. |
| `apex_xbox_manual_teleop_node.py` | Manual teleoperation support. |
| `apex_windows_gamepad_bridge_node.py` | Windows gamepad bridge support. |
| `apex_offline_sensorfusion_map_publisher.py` | Offline map publishing. |
| `apex_general_track_mapper.py` | General track mapping. |
| `apex_general_track_map_publisher.py` | General track map publishing. |
| `rear_wheel_speed_publisher.py` | Wheel speed helper. |
| `turning_command_mapper.py` | Command mapping helper. |

### How It Connects

`apex_sim.launch.py` starts Gazebo, bridges sensors into ROS, starts simulation-specific tools, and includes the `apex_telemetry` pipeline with simulation backends.

## `apex_telemetry`

| Item | Details |
| --- | --- |
| Location | `APEX/ros2_ws/src/apex_telemetry` |
| Build type | `ament_python` |
| Primary use | Current APEX real and simulation autonomy pipeline. |
| Classification | Core APEX package. |

### Important Files

| Path | Purpose |
| --- | --- |
| `package.xml` | ROS 2 dependencies including sensor, nav, TF, std service, and Python dependencies. |
| `setup.py` | Installs console scripts. |
| `launch/apex_pipeline.launch.py` | Modular APEX pipeline launch. |
| `config/apex_params.yaml` | Main parameter reference for APEX nodes. |
| `apex_telemetry/actuation.py` | Low-level actuation helpers. |
| `apex_telemetry/*_node.py` | ROS node implementations. |

### Console Scripts

| Executable | Responsibility |
| --- | --- |
| `nano_accel_serial_node` | Nano IMU ingestion. |
| `kinematics_estimator_node` | Raw IMU integration and kinematics status. |
| `kinematics_odometry_node` | APEX kinematics to odometry. |
| `rplidar_publisher_node` | RPLIDAR or simulated LaserScan source. |
| `imu_lidar_planar_fusion_node` | IMU+LiDAR planar fusion. |
| `curve_entry_path_planner_node` | Curve-entry planner. |
| `recognition_tour_planner_node` | Recognition-tour route and local-path planner. |
| `curve_path_tracker_node` | Curve path tracker. |
| `recognition_tour_tracker_node` | Recognition-tour tracker. |
| `apex_windows_gamepad_bridge_node` | Manual/gamepad bridge. |
| `recognition_session_manager_node` | Session manager and recording coordination. |
| `cmd_vel_to_apex_actuation_node` | Converts `/apex/cmd_vel_track` to real or simulated actuation. |

### How It Connects

`apex_telemetry` can be launched directly through `apex_pipeline.launch.py`, through the real Docker startup script, or through the Gazebo simulation launch. It is the shared autonomy layer between simulation and the real blue car.

## `voiture_system`

| Item | Details |
| --- | --- |
| Location | `src/voiture_system` |
| Build type | `ament_python` |
| Primary use | Alternate ROS 2 real-car SLAM/Nav2 workflow. |
| Classification | Alternate stack. |

### Important Files

| Path | Purpose |
| --- | --- |
| `package.xml` | Dependencies for ROS 2, serial, navigation, Gazebo, and `ros2_control`. |
| `setup.py` | Installs console scripts. |
| `launch/bringup_real_slam_nav.launch.py` | Alternate real-car launch with RPLIDAR, serial state, Ackermann drive, SLAM, and Nav2 options. |
| `launch/bringup_sim.launch.py` | Older Classic Gazebo simulation launch. |
| `config/controllers.yaml` | Controller configuration for `ros2_control`. |
| `config/slam_toolbox_online_async.yaml` | SLAM toolbox configuration. |
| `config/nav2_ackermann.yaml` | Nav2 configuration. |

### Console Scripts

| Executable | Responsibility |
| --- | --- |
| `high_level_controller_node` | Higher-level control logic for the alternate stack. |
| `vehicle_sim_bridge_node` | Simulation bridge for alternate stack. |
| `serial_state_node` | Arduino serial state ingestion. |
| `ackermann_drive_node` | `/cmd_vel` to sysfs PWM motor and steering. |
| `ackermann_odometry_node` | Speed and steering to `/odom` and TF. |
| `rplidar_publisher_node` | RPLIDAR publisher for alternate stack. |
| `adaptive_track_controller_node` | Laser/map-based adaptive controller. |

### How It Connects

`bringup_real_slam_nav.launch.py` builds a standard ROS navigation graph around `/cmd_vel`, `/odom`, `/map`, and optional Nav2. It is separate from the APEX recognition-tour graph, which uses `/apex/...` topics.

## `APEX/tools`

| Subfolder | Role |
| --- | --- |
| `tools/core` | Real-ready startup, shutdown, service runner, and core operational scripts. |
| `tools/capture` | Real run capture scripts for recognition tour, curve tracking, forward/raw, and static-curve data. |
| `tools/sim` | Gazebo simulation wrappers and simulated capture helpers. |
| `tools/pc` | PC-side monitoring and manual bridge helper scripts. |
| `tools/windows` | Windows gamepad bridge assets. |
| Other `tools/*.sh` wrappers | Compatibility wrappers around the more organized subfolders. |

Some top-level wrapper scripts exist for compatibility. Prefer the organized `tools/core`, `tools/capture`, and `tools/sim` paths in new documentation and scripts.

## `APEX/docker`

| File | Role |
| --- | --- |
| `Dockerfile` | Builds a ROS 2 Jazzy-based runtime image with Python scientific and serial dependencies. |
| `docker-compose.yml` | Starts the privileged `apex_pipeline` real-car container. |

## `full_soft`

`full_soft` is an older non-ROS Python stack. It includes previous Raspberry Pi setup notes, hardware inventory, hardware overview, Python vehicle control code, older simulator assets, and configuration files such as `config.json` and `sample-config.json`.

Use it as historical reference, not as the current APEX architecture.

## `Lidar`

`Lidar` contains split Raspberry/PC LiDAR utilities and networking guidance. It is useful for ROS 2 DDS networking setup, Raspberry Pi to PC LiDAR topic sharing, and notes about device baud rates and ROS domain configuration.

It is not the current full APEX autonomy stack.

## Related Documentation

- [Repository Structure](02_repository_structure.md)
- [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
- [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md)

