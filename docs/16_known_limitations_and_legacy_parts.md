# Known Limitations and Legacy Parts

## Summary

The repository contains several overlapping generations of autonomous RC car software. The current recommended workflow is APEX, while alternate ROS paths remain in the tree and older non-ROS material is kept only as historical documentation. This file documents known limitations so developers can avoid confusing current code with historical or experimental code.

## Current Recommended vs Alternate vs Legacy

| Area | Status | Notes |
| --- | --- | --- |
| `APEX/ros2_ws/src/apex_telemetry` | Current recommended | Main APEX autonomy package for real and simulated operation. |
| `src/rc_sim_description/launch/apex_sim.launch.py` | Current recommended | Main Gazebo Sim launch. |
| `APEX/tools/core/apex_real_ready_up.sh` | Current recommended | Main real blue-car startup path. |
| `APEX/tools/capture/apex_recognition_tour_capture.sh` | Current recommended | Main real recognition-tour capture path. |
| `src/voiture_system` | Alternate | ROS 2 SLAM/Nav2 path using standard `/cmd_vel`, `/odom`, and `/map`. |
| `src/rc_sim_description/launch/spawn_rc_car.launch.py` | Legacy/overlapping | Older simple Gazebo Sim path. |
| `src/voiture_system/launch/bringup_sim.launch.py` | Legacy/alternate | Classic Gazebo and `ros2_control` path. |
| Former `full_soft/` reference | Historical external | Older non-ROS Python vehicle stack removed from `main`; only the synthesis report remains. |
| `Lidar` | Auxiliary | Useful LiDAR networking notes, not a complete APEX stack. |

## Outdated or Conflicting Documentation

Some existing README files describe earlier project states:

- Older root README revisions focused on an earlier `rc_sim_description` workflow and simple Gazebo launch path; the current root README is a documentation gateway.
- `APEX/README.md` describes a reduced APEX scope that conflicts with current scripts using fusion, planning, tracking, and actuation.
- The former `full_soft/` material described an older non-ROS stack and is no longer versioned in `main`.
- Some older documents contain non-English text or encoding artifacts.

For current workflows, prefer:

- `APEX/tools/*` scripts.
- `APEX/ros2_ws/src/apex_telemetry` source and config.
- `src/rc_sim_description/launch/apex_sim.launch.py`.
- This documentation set.

## Package Metadata TODOs

Some package manifests still contain placeholder metadata such as TODO descriptions, maintainers, or licenses. This does not necessarily break builds, but it should be cleaned before release or publication.

Known affected areas include package metadata in:

```text
src/rc_sim_description/package.xml
src/voiture_system/package.xml
```

## `voiture_system` Xacro Issue

The file:

```text
src/voiture_system/urdf/ros2_control.xacro
```

contains a malformed leading `v` before the `ros2_control` XML block. This may break Xacro processing for the older `voiture_system` simulation path. The current APEX Gazebo Sim workflow uses `src/rc_sim_description/urdf/rc_car.urdf.xacro` and is the preferred simulation path.

## LiDAR Baud-Rate Ambiguity

The current APEX parameter file defaults the RPLIDAR baud rate to `115200`. Older documentation mentions:

- Yellow car LiDAR: `256000`.
- Blue car LiDAR: `115200`.

The correct value depends on the actual sensor and adapter. Verify the device before a real run, especially if moving hardware between cars.

## Duplicate or Compatibility Wrapper Scripts

Some shell scripts under `APEX/tools/*.sh` duplicate or forward to more organized subdirectories:

```text
APEX/tools/core/
APEX/tools/capture/
APEX/tools/sim/
APEX/tools/firmware/
```

Use the subfolder scripts in new documentation. Keep compatibility wrappers only when they are needed by existing workflows.

## Large Run Artifacts

The repository includes or may generate run artifacts under:

```text
APEX/apex_recognition_tour/
APEX/apex_curve_track/
APEX/apex_forward_raw/
APEX/apex_static_curve/
src/rc_sim_description/data/runs/
```

These are useful for diagnostics but can become large and may obscure source changes. Consider moving non-baseline runs out of the repository or documenting which runs are canonical.

## Recognition-Tour Diagnostic Caveats

A local diagnostic report for a recognition-tour run notes:

- Final state similar to `aborted_path_loss`.
- Fusion appeared healthy in that run.
- A possible planner/tracker synchronization issue around local-path age.
- Steering requests around 46-48 degrees while hardware limits were around 18 degrees.
- Possible planner fallback fragility.

This is a run-specific diagnostic, not a proof that every recognition-tour run fails for the same reason. It is still useful when debugging planner/tracker timing and steering saturation.

## Windows Native Support Limitation

Native Windows ROS/Gazebo support is not complete for this repository. The practical Windows path is WSL2 with Ubuntu 24.04 and ROS 2 Jazzy. The current Windows host used during audit did not have `ros2` or `colcon` available in PowerShell.

## Simulation Limitations

Simulation is useful but cannot fully represent:

- Real tire grip and wheel slip.
- Battery voltage changes.
- ESC startup and calibration behavior.
- Steering servo backlash or mechanical limits.
- Real LiDAR reflections and dropouts.
- Serial latency and device failures.
- Network/DDS behavior during real multi-machine operation.

Use simulation to iterate quickly, then validate cautiously on the physical car.

## Real-Car Limitations

The real APEX stack depends on:

- Correct serial device paths.
- Correct sensor baud rates.
- Correct PWM overlay and sysfs PWM availability.
- Docker privileged mode.
- Physical safety and predictable test space.
- Sensor mounting alignment.
- Planner/tracker stale-data handling.

These should be checked before every autonomous run.

## Cleanup Recommendations

Recommended future cleanup:

1. Decide whether `voiture_system` is a supported alternate stack or a migration source.
2. Fix package metadata placeholders.
3. Fix or remove malformed `ros2_control.xacro` content.
4. Consolidate wrapper scripts under organized `APEX/tools` subfolders.
5. Move large non-canonical run artifacts out of source control.
6. Reconcile older README files with the APEX-first architecture.
7. Add automated checks for launch argument validity and topic availability.
8. Add explicit tests or smoke scripts for simulation launch and APEX parameter loading.

## Related Documentation

- [Repository Structure](02_repository_structure.md)
- [Packages and Modules](10_packages_and_modules.md)
- [Troubleshooting](15_troubleshooting.md)
- [Developer Guide](14_developer_guide.md)
