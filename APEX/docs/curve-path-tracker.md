# Curve Path Tracker

## Responsibility

`curve_path_tracker_node.py` tracks the planned path and produces a `Twist` command for the actuation bridge.

It is a Pure Pursuit tracker with:

- progress management,
- path-loss abort logic,
- confidence supervision,
- startup ramping,
- speed limiting from curvature,
- angular smoothing,
- and explicit goal-stop logic.

## Control Model

The tracker uses a rear-axle bicycle model.

That means:

- the vehicle pose is evaluated at the rear axle,
- the lookahead target is chosen on the planned path,
- the target point is expressed in the rear-axle local frame,
- and curvature is computed from the Pure Pursuit relation.

## Pure Pursuit Equation

The central geometric relation is:

`k = 2 * dy / L^2`

where:

- `k` is curvature,
- `dy` is the lateral offset of the lookahead target in the vehicle frame,
- `L` is the actual lookahead distance.

This formula is what turns path geometry into steering demand.

## Main Runtime Flow

1. Wait for a path.
2. Wait for planner readiness.
3. Wait for arm state.
4. Read fused odometry.
5. Estimate rear-axle pose.
6. Find the closest progress point on the path.
7. Pick a lookahead target.
8. Compute curvature.
9. Compute a speed limit from curvature and lateral acceleration.
10. Filter angular command.
11. Publish `Twist`.

## Path Progress Logic

The tracker does not simply use the raw nearest path point at every cycle. It also stores progress and avoids moving backwards on the path unless allowed by a small tolerance.

This reduces oscillations caused by noisy nearest-point jumps.

## Speed Law

The commanded linear speed is limited by three effects:

- configured maximum speed,
- curvature-based speed reduction,
- lateral-acceleration-based speed reduction.

It is then further reduced:

- when approaching the goal,
- and during the startup ramp.

## Angular Smoothing

The angular command uses an exponential moving average:

`w_filtered = alpha * w_raw + (1 - alpha) * w_prev`

This reduces oscillatory steering corrections caused by point-to-point curvature noise.

## Goal Logic

The tracker now has three distinct terminal mechanisms:

### 1. Goal proximity stop

If the rear axle enters a configurable radius around the goal, tracking stops immediately.

This is the strongest stop condition and it does not wait for final yaw alignment.

This is intentionally safety-oriented. It exists to avoid wall impacts near the target.

### 2. Goal line crossed

If the vehicle crosses the goal line in the final tail of the path and final yaw is acceptable, tracking terminates.

### 3. Goal reached

If position and yaw are both within tolerance, tracking terminates.

## Current Safety Behavior

The current stack favors stopping before collision over preserving perfect final orientation.

That is why:

- `goal_proximity_stop_distance_m` exists,
- and why the bridge now performs an active brake-to-neutral sequence once speed command becomes zero.

## Parameters That Matter Most

- `min_lookahead_m`
- `max_lookahead_m`
- `lookahead_speed_gain`
- `curvature_speed_gain`
- `max_lateral_accel_mps2`
- `slowdown_distance_m`
- `angular_cmd_ema_alpha`
- `curvature_deadband_m_inv`
- `goal_tolerance_m`
- `goal_proximity_stop_distance_m`
- `goal_yaw_tolerance_rad`
- `max_path_deviation_m`

## Steering Limit Note

The software curvature clamp introduced during debugging has been removed.

That means the tracker now sends the curvature implied by Pure Pursuit directly, except for the optional deadband around zero curvature. Any remaining saturation now happens only at the actuation layer, where real PWM bounds still exist.

## Failure Modes

The main terminal failure causes are:

- `planner_failed`
- `aborted_low_confidence`
- `timeout`
- `aborted_odom_timeout`
- `aborted_path_loss`

The main successful or intended terminal causes are:

- `goal_proximity_stop`
- `goal_line_crossed`
- `goal_reached`

## Relevant File

- `ros2_ws/src/apex_telemetry/apex_telemetry/control/curve_path_tracker_node.py`
