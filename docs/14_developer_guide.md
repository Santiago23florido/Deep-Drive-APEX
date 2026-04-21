# Developer Guide

## Development Principles

When extending the repository, prefer the current architecture:

- Use APEX for new blue-car autonomy work.
- Use `rc_sim_description` for Gazebo Sim assets and simulation-only tools.
- Use `apex_telemetry` for real/sim shared autonomy nodes.
- Use `voiture_system` only when working on the alternate SLAM/Nav2 path.
- Treat the former `full_soft/` code as external historical context. It is no longer versioned in `main`.

Do not add new behavior to multiple stacks unless the duplication is intentional and documented.

## Where to Add New Code

| New feature | Recommended location |
| --- | --- |
| New APEX ROS node | `APEX/ros2_ws/src/apex_telemetry/apex_telemetry/` |
| New APEX node executable | Add to `APEX/ros2_ws/src/apex_telemetry/setup.py` console scripts. |
| New APEX parameters | `APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml` |
| New simulation launch option | `src/rc_sim_description/launch/apex_sim.launch.py` |
| New Gazebo world | `src/rc_sim_description/worlds/` |
| New simulation scenario | `src/rc_sim_description/config/apex_sim_scenarios.json` |
| New simulation bridge/tool | `src/rc_sim_description/scripts/` and `CMakeLists.txt` install list. |
| New real capture workflow | `APEX/tools/capture/` |
| New real startup workflow | `APEX/tools/core/` |
| Alternate SLAM/Nav2 behavior | `src/voiture_system/` |

## Adding a New APEX Node

1. Create a Python module in:

   ```text
   APEX/ros2_ws/src/apex_telemetry/apex_telemetry/
   ```

2. Add a console-script entry in:

   ```text
   APEX/ros2_ws/src/apex_telemetry/setup.py
   ```

3. Add parameters to:

   ```text
   APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml
   ```

4. Add launch composition to:

   ```text
   APEX/ros2_ws/src/apex_telemetry/launch/apex_pipeline.launch.py
   ```

5. If the node should run in simulation, wire it through:

   ```text
   src/rc_sim_description/launch/apex_sim.launch.py
   ```

6. Document new topics and parameters in:

   ```text
   docs/12_topics_services_actions_parameters.md
   docs/17_configuration_reference.md
   ```

## ROS Topic Naming Guidance

Use clear names that match the existing conventions:

| Area | Existing convention |
| --- | --- |
| APEX sensor topics | `/apex/imu/...`, `/lidar/scan_localization` |
| APEX estimation | `/apex/odometry/...`, `/apex/estimation/...` |
| APEX planning | `/apex/planning/...` |
| APEX tracking | `/apex/tracking/...` |
| APEX actuation | `/apex/vehicle/...`, `/apex/cmd_vel_track` |
| Simulation-only topics | `/apex/sim/...` |
| Alternate stack | standard names such as `/cmd_vel`, `/odom`, `/map` |

Prefer explicit APEX namespaces for APEX-specific data. Use standard ROS names only when intentionally integrating with standard tools such as Nav2 or SLAM toolbox.

## Parameters and Configuration

Add parameters instead of hard-coding runtime values when a value may differ between:

- Simulation and real vehicle.
- Blue car and another vehicle.
- Different sensors or serial ports.
- Different track geometries.
- Safe test mode and full autonomous mode.

Keep parameter names stable and document defaults.

## Simulation Development

When changing simulation behavior:

1. Update the URDF/Xacro only for robot model changes.
2. Update worlds only for environment geometry changes.
3. Update `apex_sim_scenarios.json` for scenario-level configuration.
4. Keep simulation-specific bridge code in `rc_sim_description`.
5. Keep shared autonomy logic in `apex_telemetry`.

This separation prevents Gazebo-only assumptions from leaking into real-car nodes.

## Real-Car Development

When changing real-car behavior:

- Validate with the vehicle lifted before floor testing.
- Keep the actuation bridge conservative by default.
- Publish status for every safety-relevant decision.
- Record enough data to reproduce failures.
- Do not bypass command clamps or stale-data checks without documenting the test reason.
- Keep Docker Compose device paths and parameter defaults aligned.

## Testing and Validation

Minimum checks after a ROS code change:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
ros2 launch rc_sim_description apex_sim.launch.py --show-args
ros2 launch apex_telemetry apex_pipeline.launch.py --show-args
```

For a simulation-impacting change, run:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

For a real-car change, run topic-level validation before arming motion:

```bash
ros2 topic hz /apex/imu/data_raw
ros2 topic hz /lidar/scan_localization
ros2 topic echo /apex/estimation/status --once
ros2 topic echo /apex/vehicle/drive_bridge_status --once
```

## Working With Legacy Code

If you must modify older launch paths or reintroduce ideas from the former external `full_soft/` reference:

- State in commit messages or documentation that the work targets a legacy path.
- Do not make the legacy path appear to be the current recommended workflow.
- Avoid copying old hardware assumptions into APEX without validating them.
- Prefer migrating useful ideas into `apex_telemetry` or `rc_sim_description`.

## Documentation Updates

When adding or changing runtime behavior, update:

- [Launch Files and Execution Flows](11_launch_files_and_execution_flows.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Configuration Reference](17_configuration_reference.md)
- [Troubleshooting](15_troubleshooting.md), if the change creates new failure modes.

## Related Documentation

- [Repository Structure](02_repository_structure.md)
- [ROS Architecture](07_ros_architecture.md)
- [Packages and Modules](10_packages_and_modules.md)
- [Known Limitations and Legacy Parts](16_known_limitations_and_legacy_parts.md)
