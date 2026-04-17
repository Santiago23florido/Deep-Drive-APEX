# Mapping and Recording Pipeline

## Purpose

The repository includes recording and analysis tools so simulation and real runs can be inspected after execution. This is important because many failures are timing or data-quality problems that are hard to understand from live terminal output alone.

The main recorded data families are:

- Real recognition-tour captures.
- Real curve-track captures.
- Raw sensor-fusion captures.
- Simulation captures.
- Offline map and submap refinement outputs.

## Real Recognition-Tour Capture

Primary script:

```text
APEX/tools/capture/apex_recognition_tour_capture.sh
```

Typical command:

```bash
cd /home/ensta/AiAtonomousRc/APEX
./tools/capture/apex_recognition_tour_capture.sh --run-id recognition_tour_test_01 --timeout-s 60
```

The script configures the APEX pipeline for recognition-tour operation and records topics such as:

- `/apex/planning/recognition_tour_local_path`
- `/apex/planning/recognition_tour_route`
- `/apex/planning/recognition_tour_status`
- `/apex/tracking/recognition_tour_status`
- `/apex/tracking/arm`
- `/apex/estimation/status`
- `/apex/odometry/imu_lidar_fused`
- `/apex/vehicle/drive_bridge_status`
- `/lidar/scan_localization`

Outputs are written under:

```text
APEX/apex_recognition_tour/
```

## Real Curve-Track Capture

Primary script:

```text
APEX/tools/capture/apex_curve_track_capture.sh
```

This capture path is used for the curve-entry planner/tracker instead of the recognition-tour planner/tracker. It records planner path/status, tracker status, fused odometry, scans, and actuation bridge status.

Outputs are written under:

```text
APEX/apex_curve_track/
```

## Raw Sensor-Fusion Capture

Relevant scripts include:

```text
APEX/tools/capture/apex_rect_sensorfus_capture.sh
APEX/tools/capture/*forward*
APEX/tools/capture/*static_curve*
```

These workflows capture lower-level data for IMU, LiDAR, PWM, and sensor-fusion debugging. They are useful when the high-level planner is not the main focus.

Typical outputs include:

- Raw IMU traces.
- LiDAR point exports.
- PWM traces.
- Event logs.
- Readiness logs.
- Fusion status.

## Simulation Recording

Simulation recording is handled by tools in `rc_sim_description`, especially:

```text
src/rc_sim_description/scripts/apex_sim_run_recorder.py
```

The simulation launch can enable recording through launch arguments where supported. Simulation recording can include:

- `/apex/sim/scan`
- `/apex/sim/imu`
- `/apex/odometry/imu_lidar_fused`
- `/apex/sim/ground_truth/odom`
- `/apex/sim/ground_truth/path`
- `/apex/sim/ground_truth/status`

Simulation run outputs may be written under:

```text
src/rc_sim_description/data/runs/
```

## Offline Submap Refinement

Relevant tools:

| Tool | Purpose |
| --- | --- |
| `offline_submap_refiner.py` | Refines submaps after a recorded run. |
| `offline_similarity_monitor.py` | Monitors similarity metrics for offline refinement. |
| `apex_refined_sensorfusion_map_node.py` | Publishes refined map output. |
| `apex_offline_sensorfusion_map_publisher.py` | Publishes offline sensor-fusion map data. |

Offline refinement is most useful when comparing recorded sensor-fusion output against improved map estimates or simulation ground truth.

## Mapping Data Flow

```text
LiDAR scans + odometry
  -> live fusion map points
  -> optional recording
  -> offline submap refinement
  -> refined map publication or analysis
```

In simulation, ground truth can be added:

```text
Gazebo ground truth
  -> perfect map points and true odometry
  -> comparison against estimated odometry and maps
```

## PC-Side Session Monitoring

`APEX/tools/pc/watch_real_recognition_session.sh` can monitor real recognition sessions and fetch completed run artifacts from the Raspberry Pi. This is useful when operating from a development PC while the APEX pipeline runs on the vehicle.

## Interpreting Analysis Outputs

When a run fails, inspect artifacts in this order:

1. Capture metadata and script command.
2. Docker logs for exceptions or device failures.
3. Sensor topic rates and scan counts.
4. Fusion status and odometry continuity.
5. Planner status and local path age.
6. Tracker status and arm state.
7. Drive bridge status and applied steering/speed.
8. Generated diagnostic report, if present.

## Recommended Recording Practices

- Use explicit run IDs.
- Record the exact command used for the run.
- Keep Docker logs with the run.
- Keep both planner and tracker status topics.
- Keep actuation bridge status for every autonomous run.
- Separate real and simulated run folders.
- Do not overwrite failed runs; they are often the most useful diagnostic data.

## Related Documentation

- [Data and Runs](13_data_and_runs.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)
- [Blue Vehicle Real System](09_blue_vehicle_real_system.md)
- [Troubleshooting](15_troubleshooting.md)

