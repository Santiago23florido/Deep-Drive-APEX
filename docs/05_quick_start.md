# Quick Start

This page gives the shortest practical path to run the repository. For full setup details, read [Installation on Linux](03_installation_linux.md) or [Installation on Windows](04_installation_windows.md).

## Build Once

From a Linux, WSL2, or Raspberry Pi shell with ROS 2 Jazzy installed:

```bash
cd ~/AiAtonomousRc
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src APEX/ros2_ws/src --ignore-src -r -y
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
```

Use your actual clone path instead of `~/AiAtonomousRc`.

## Start the Recommended Simulation

Run the APEX Gazebo Sim wrapper:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Expected result:

- Gazebo Sim starts with the selected track world.
- RViz starts if `--rviz` is passed.
- The simulated vehicle model is spawned.
- The APEX pipeline runs in simulation backend mode.
- Simulated LiDAR and IMU data appear on ROS topics.

Check topics:

```bash
ros2 topic list | grep apex
ros2 topic echo /apex/sim/scan --once
ros2 topic echo /apex/sim/imu --once
```

To arm the recognition-tour tracker in simulation:

```bash
./APEX/tools/sim/apex_arm_recognition_tour.sh
```

Some wrappers also support `--arm` directly:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz --arm
```

## Direct Simulation Launch

After building and sourcing the workspace, you can launch without the wrapper:

```bash
ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=true
```

Use the wrapper when you want the repository's default build, scenario, and convenience behavior.

## Start the Real Blue-Car Stack

Run this on the Raspberry Pi mounted on the vehicle, not on a native Windows PowerShell shell:

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/core/apex_real_ready_up.sh
```

This starts the Docker Compose APEX pipeline with the default real-ready feature set:

- IMU ingestion.
- LiDAR ingestion.
- IMU+LiDAR fusion.
- Recognition-tour planner and tracker.
- `cmd_vel` to PWM actuation bridge.
- Recognition session manager.
- Optional offline submap refinement.

Safety notes:

- Keep the vehicle lifted or restrained during first startup.
- Confirm ESC neutral and steering center before arming motion.
- Do not run actuation scripts near people or obstacles.

## Capture a Recognition-Tour Run

From the Raspberry Pi APEX directory:

```bash
./tools/capture/apex_recognition_tour_capture.sh --run-id recognition_tour_test_01 --timeout-s 60
```

Expected outputs are written under:

```text
APEX/apex_recognition_tour/
```

See [Mapping and Recording Pipeline](18_mapping_and_recording_pipeline.md).

## Stop the Real APEX Stack

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/core/apex_core_down.sh
```

## Useful Commands

| Task | Command |
| --- | --- |
| List APEX topics | `ros2 topic list | grep apex` |
| Check LiDAR scans | `ros2 topic hz /lidar/scan_localization` |
| Check raw IMU | `ros2 topic hz /apex/imu/data_raw` |
| Check fused odometry | `ros2 topic echo /apex/odometry/imu_lidar_fused --once` |
| Check recognition planner status | `ros2 topic echo /apex/planning/recognition_tour_status --once` |
| Check tracker status | `ros2 topic echo /apex/tracking/recognition_tour_status --once` |
| Check actuation bridge | `ros2 topic echo /apex/vehicle/drive_bridge_status --once` |
| Show simulation launch args | `ros2 launch rc_sim_description apex_sim.launch.py --show-args` |
| Show APEX pipeline launch args | `ros2 launch apex_telemetry apex_pipeline.launch.py --show-args` |

## Related Documentation

- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
- [Troubleshooting](15_troubleshooting.md)

