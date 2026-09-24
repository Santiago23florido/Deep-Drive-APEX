# APEX Simulation

## Main Startup

```bash
./simulation/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

Enable `slam_toolbox` as well:

```bash
./simulation/tools/sim/apex_sim_up.sh --scenario baseline --rviz --slam
```

Equivalent direct launch:

```bash
ros2 launch rc_sim_description apex_sim.launch.py \
  scenario:=baseline rviz:=true
```

Without `--slam`, the default RViz layout is `simulation/rviz/apex_recognition_live.rviz`. With `--slam`, it is `simulation/rviz/apex_recognition_slam_live.rviz`.

## Sensor-Fusion Research

Gazebo with ideal sensors, plus the LiDAR noise model, the raw IMU model, the
pure strapdown INS and RViz. The launcher removes Conda from the environment.

```bash
./simulation/tools/sim/apex_fusion_research_up.sh --arm
./simulation/tools/sim/apex_fusion_research_up.sh --imu-config white_noise_only --imu-seed 3 --headless
```

SLAM baseline, headless: identical `slam_toolbox` instances fed with good and
damaged sensors, CSV maps and measurements, comparison figure and metrics:

```bash
./simulation/tools/sim/apex_fusion_slam_capture.sh
```

Runs are stored under `simulation/data/fusion_research/`. See
`simulation/ros2_ws/src/apex_fusion_research/README.md`.

## Recognition Tour

Arm an already running simulation:

```bash
./simulation/tools/sim/apex_arm_recognition_tour.sh
```

Record a complete simulated run:

```bash
./simulation/tools/sim/apex_recognition_tour_sim_capture.sh \
  --scenario tight_right_saturation --timeout-s 60
```

Compare a real and simulated run:

```bash
python3 simulation/tools/analysis/compare_recognition_tour_runs.py \
  --real-run real_vehicle/data/apex_recognition_tour/<real_run> \
  --sim-run simulation/data/apex_recognition_tour/<sim_run>
```

## Manual Mapping

```bash
./simulation/tools/sim/apex_manual_mapping_up.sh \
  --scenario precision_fusion --rviz
```

Build the Windows gamepad bridge once:

```bash
./simulation/tools/windows/build_apex_xbox_bridge_sim.sh
```

Run `simulation/tools/windows/dist/apex_xbox_bridge_sim.exe` on Windows. The left-stick Y axis controls speed and the X axis controls steering.

Finalize the run and generate offline output:

```bash
./simulation/tools/sim/apex_manual_mapping_finish.sh
```

Artifacts are stored under `simulation/data/`.

## Direct WSL Launch

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/AiAtonomousRc/simulation/ros2_ws
colcon build --symlink-install \
  --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
export APEX_SIM_ROOT=~/AiAtonomousRc/simulation
ros2 launch rc_sim_description apex_sim.launch.py \
  control_mode:=manual_windows_bridge \
  use_slam:=true mapping_mode:=ideal rviz:=true
```

## Scenarios

- `baseline`
- `precision_fusion`
- `tight_right_saturation`
- `outer_long_inner_short`
- `startup_pose_jump`
- `narrowing_false_corridor`

## Key Simulation Topics

- `/apex/sim/pwm/steering_dc`
- `/apex/sim/pwm/motor_dc`
- `/apex/sim/ground_truth/odom`
- `/apex/sim/ground_truth/path`
- `/apex/sim/ground_truth/perfect_map_points`
- `/apex/sim/ground_truth/status`

## Fidelity

| Area | Current behavior |
| --- | --- |
| ROS pipeline | Starts `apex_pipeline.launch.py` with the simulation copy of `apex_params.yaml`. |
| Online fusion | Uses `imu_lidar_planar_fusion_node`; ground truth is not injected into estimation. |
| Planner and tracker | Runs the current APEX nodes and interfaces. |
| Frames and offsets | Uses `base_link`, `laser`, and `rear_axle` with configured sensor offsets. |
| Actuation | PWM conversion, trims, clamps, and ramps remain in `cmd_vel_to_apex_actuation_node`. |
| Capture | Preserves recognition-tour artifacts and adds ground-truth logs. |

## Approximations

- ESC and servo dynamics are calibratable models.
- IMU bias, drift, noise, and startup calibration are simulated.
- LiDAR latency, noise, gaps, and infinity values are modeled.
- Scenario worlds are test variants, not exact CAD copies of the physical circuit.
- Ground-truth clearances are calculated from the active SDF world.
