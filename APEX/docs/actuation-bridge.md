# Actuation Bridge and ESC/Servo Mapping

## Responsibility

The actuation layer converts a high-level `Twist` command into:

- steering servo PWM,
- and ESC motor PWM.

This happens in two files:

- `cmd_vel_to_apex_actuation_node.py`
- `actuation.py`

## Bridge Role

`cmd_vel_to_apex_actuation_node.py` receives:

- `linear.x`
- `angular.z`

and computes:

- desired steering angle in degrees,
- desired speed percentage,
- applied steering angle after optional rate limiting,
- applied speed after ramping or braking behavior.

## Steering Conversion

The steering command uses the bicycle model:

`delta = atan(L * omega / v)`

where:

- `delta` is steering angle,
- `L` is wheelbase,
- `omega` is yaw rate,
- `v` is forward speed.

This is the correct kinematic inverse for the bicycle model used by the tracker.

## Current Steering Policy

The software angle clamp in the bridge has been removed.

That means:

- the bridge computes the steering angle requested by the controller,
- and the low-level servo object only saturates because PWM duty cycle cannot exceed hardware bounds.

This is a deliberate difference between:

- a software steering limit,
- and a hardware output saturation.

The first one changes control intent.
The second one preserves control intent but acknowledges actuator limits.

## Servo Mapping

`SteeringServo` maps steering angle to PWM duty cycle using:

- configured duty-cycle limits,
- a center trim,
- and a sign convention.

The current implementation no longer clips the requested angle in degrees before conversion. It computes the duty cycle directly and then clips the duty cycle itself to:

- `dc_min`
- `dc_max`

That means analysis logs can still reflect how much steering was requested, even if the hardware output saturated.

## Speed Mapping

The bridge maps forward speed to ESC percent with:

- `min_effective_speed_pct`
- `max_speed_pct`
- `max_linear_speed_mps`

In the current test setup both speed percentages are fixed to `20.0`, so the real vehicle runs at a single forward command level whenever `linear.x > 0`.

## Why Previous Stops Still Collided

The original stop behavior used a ramp-down to zero. That is acceptable for smooth driving, but not for safety-critical stopping near a wall.

If the tracker enters terminal mode late and the bridge still ramps down, the car can keep rolling into the obstacle.

## Current Zero-Speed Behavior

The current bridge has two safety options enabled:

- `brake_on_zero_speed_cmd`
- `center_steering_on_zero_speed_cmd`

When speed command becomes zero:

1. steering is centered immediately,
2. the speed ramp is cancelled,
3. the ESC receives an active brake pulse,
4. then the ESC is returned to neutral.

This is implemented by:

- `MaverickESCMotor.brake_to_neutral()`

The purpose is to remove residual forward motion as quickly as possible.

## ESC Theory

The ESC does not understand "meters per second". It only understands PWM duty cycle. Therefore the bridge must translate a physical command into an actuator command.

That translation is not perfectly linear in real hardware. The code currently uses a pragmatic open-loop mapping:

- low-level duty cycle bounds,
- a neutral value,
- and a reverse-brake pulse sequence.

This is enough for repeatable short experiments, but it is still open-loop actuation.

## Parameters That Matter Most

- `min_effective_speed_pct`
- `max_speed_pct`
- `speed_pct_ramp_duration_s`
- `speed_pct_ramp_down_per_s`
- `steering_rate_limit_deg_per_s`
- `brake_on_zero_speed_cmd`
- `center_steering_on_zero_speed_cmd`
- `steering_dc_min`
- `steering_dc_max`
- `steering_center_trim_dc`
- `reverse_brake_dc`
- `reverse_brake_hold_s`
- `reverse_neutral_hold_s`

## Relevant Files

- `ros2_ws/src/apex_telemetry/apex_telemetry/actuation/cmd_vel_to_apex_actuation_node.py`
- `ros2_ws/src/apex_telemetry/apex_telemetry/actuation/actuation.py`
