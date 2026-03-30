# Curve Entry Path Planner

## Responsibility

`curve_entry_path_planner_node.py` converts a local LiDAR-based entry trajectory into a world-frame path that the vehicle can track.

This node sits between perception and control.

Its job is not to detect the curve. Its job is to:

- wait until the system is ready,
- collect a stable LiDAR snapshot,
- call the detector,
- transform the local path into odometry coordinates,
- smooth the path,
- and enforce the final heading of the second corridor.

## Runtime Sequence

The planner only proceeds when all of these conditions are met:

1. Fusion is ready.
2. Odom is available.
3. The vehicle is considered stationary.
4. The vehicle has remained stationary for the configured hold time.
5. Enough scans have been collected for the snapshot.

This gating is critical because the whole curve-entry path is planned from a static snapshot. If the car moves during snapshot capture, the inferred corridor is less reliable.

## Main Stages

### 1. Static Snapshot Collection

`_scan_cb()` waits for:

- fusion readiness,
- stationary condition,
- stationary hold completion,
- and `snapshot_scan_count` scans.

Then it calls `_plan_from_snapshot()`.

### 2. Local Detection

`_plan_from_snapshot()` converts the median scan stack into points and calls:

- `detect_curve_window_points()`

The result includes a local path and all the status fields needed for analysis and plotting.

### 3. Transform to World Coordinates

The detected local path is transformed from the LiDAR frame into the odometry frame using:

- current planner pose,
- LiDAR offset,
- and current yaw.

### 4. Origin Bridge

The path does not start exactly at the tracking origin. To avoid a discontinuity, the planner inserts a short cubic Bezier connector from the tracking origin to the first path point.

This is a continuity fix. It reduces the initial steering shock the tracker would otherwise see.

### 5. Curvature Smoothing

The world path is resampled and smoothed by `_smooth_path_to_curvature_limit()`.

The idea is:

- estimate curvature,
- if curvature is too high, smooth interior points,
- resample,
- and repeat.

This is a numerical path-conditioning step, not a physical simulation.

### 6. Terminal Heading Enforcement

The path must not only reach the target point. It must also enter the second corridor with the correct yaw.

That is handled by:

- `_enforce_terminal_heading()`
- `_enforce_terminal_heading_with_constraints()`

The first function modifies the tail using a Bezier construction.
The second function wraps that process in a constraint loop to keep:

- path curvature under the configured limit,
- and path deviation from the base path under a maximum threshold.

This is important because a perfect terminal heading is not useful if it creates a path that is impossible to follow.

## Key Geometric Functions

- `_rotation()`
- `_cubic_bezier_xy()`
- `_resample_polyline_xy()`
- `_resample_polyline_xy_to_count()`
- `_estimate_path_curvature()`
- `_polyline_length_m()`
- `_path_pointwise_deviation_m()`

## Theoretical Concepts

### Curvature

Curvature is approximated from heading change over arc length:

`k = dtheta / ds`

This is the right quantity to constrain because steering demand in the bicycle model scales directly with path curvature.

### Why resampling matters

Curvature estimation is sensitive to spacing between points. Resampling makes the path numerically better behaved and makes curvature estimation more consistent.

### Why use a constrained tail heading correction

If heading is enforced naively, the endpoint may be correct but the path can become unrealistically sharp near the end. The constrained version keeps terminal orientation while limiting:

- peak curvature,
- and total deformation of the original path.

## Important Parameters

Planner parameters are configured in:

- `ros2_ws/src/apex_telemetry/config/apex_params.yaml`

The most important ones are:

- `planning_wheelbase_m`
- `planning_max_steering_deg`
- `path_curvature_limit_scale`
- `path_resample_step_m`
- `path_smoothing_alpha`
- `path_smoothing_max_iterations`
- `terminal_heading_tail_length_m`
- `terminal_heading_max_tail_fraction`
- `terminal_heading_max_path_deviation_m`
- `stationary_speed_threshold_mps`
- `stationary_yaw_rate_threshold_rps`
- `stationary_hold_s`

## Practical Interpretation

If the planned path is too aggressive:

- reduce curvature demand by increasing smoothing,
- reduce terminal tail aggressiveness,
- or reduce deformation allowed during heading enforcement.

If planning never triggers:

- check fusion state,
- check `waiting_static`,
- and check whether the car is still moving during snapshot capture.

## Outputs

The planner publishes:

- a `Path`,
- a target `PoseStamped`,
- and a detailed status JSON containing:
  - planner pose,
  - tracking origin pose,
  - target pose,
  - entry line center pose,
  - second corridor axis,
  - path metrics,
  - and curve summary.

## Relevant File

- `ros2_ws/src/apex_telemetry/apex_telemetry/perception/curve_entry_path_planner_node.py`
