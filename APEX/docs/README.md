# APEX Curve-Tracking Documentation

This folder documents the current curve-entry stack implemented in `APEX/ros2_ws/src/apex_telemetry/apex_telemetry`.

The goal of this documentation is to explain the code as a set of cooperating modules, not as isolated files. Each document describes:

- what the module is responsible for,
- which inputs and outputs it uses,
- the main implementation decisions,
- the theoretical ideas behind the algorithm,
- and the parameters that most strongly affect behavior.

## Document Map

- [Curve Window Detection](./curve-window-detection.md)
- [Curve Entry Path Planner](./curve-entry-path-planner.md)
- [Curve Path Tracker](./curve-path-tracker.md)
- [Trajectory Supervisor](./trajectory-supervisor.md)
- [Actuation Bridge and ESC/Servo Mapping](./actuation-bridge.md)
- [Final Tests README](./FINAL_TESTS_README.md)
- [Final Test Commands](./final-test-commands.md)

## End-to-End Pipeline

The final runtime chain is:

1. `curve_entry_path_planner_node.py`
2. `curve_window_detection.py`
3. `curve_path_tracker_node.py`
4. `cmd_vel_to_apex_actuation_node.py`
5. `actuation.py`

In practical terms:

1. A static LiDAR snapshot is collected after fusion is ready and the vehicle is considered stationary.
2. The snapshot is converted into a corridor interpretation and a curve-entry target.
3. A world-frame path is built, smoothed, and constrained to end with the heading of the second corridor.
4. Pure Pursuit tracks that path from the rear-axle frame.
5. The tracked `Twist` command is converted into steering and ESC commands.
6. Near the goal, the tracker now forces a stop by proximity, and the bridge performs an active brake-to-neutral transition.

## Current Design Intent

The present implementation is tuned for a very specific use case:

- preserve the curve-window detection heuristic,
- preserve the detected second-corridor axis and final target logic,
- make the generated path smoother,
- reduce controller oscillation,
- and stop early enough to avoid impact with a nearby wall.

That means the system intentionally mixes geometric planning and pragmatic safety logic:

- geometry determines where the path should go,
- tracking determines how to follow it,
- actuation determines how quickly the real car should react,
- and the stop logic overrides both once the goal zone is reached.

## Main Files

- `ros2_ws/src/apex_telemetry/apex_telemetry/perception/curve_window_detection.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/perception/curve_entry_path_planner_node.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/control/curve_path_tracker_node.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/actuation/cmd_vel_to_apex_actuation_node.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/actuation/actuation.py`
- `ros2_ws/src/apex_telemetry/config/apex_params.yaml`

## Recommended Reading Order

1. Read [Curve Window Detection](./curve-window-detection.md) to understand how the corridor and target are inferred from LiDAR.
2. Read [Curve Entry Path Planner](./curve-entry-path-planner.md) to understand how the local geometric path becomes a world path.
3. Read [Curve Path Tracker](./curve-path-tracker.md) to understand how the vehicle follows that path.
4. Read [Actuation Bridge and ESC/Servo Mapping](./actuation-bridge.md) to understand how software commands are translated into hardware behavior.
5. Use [Final Test Commands](./final-test-commands.md) during real validation on the Raspberry Pi and the PC.
