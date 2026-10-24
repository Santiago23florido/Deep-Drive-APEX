# Control module

This directory is an architectural placeholder. The implemented race control
lives in [`apex_race`](../../ros2_ws/src/apex_race/apex_race/core/): reactive
planning, mapped-route guidance, geometry-checked reverse recovery, plant
identification and adaptive shaping. The hardware command bridge and PWM live in
[`apex_telemetry/actuation`](../../ros2_ws/src/apex_telemetry/apex_telemetry/actuation/).

The shared real/simulation core keeps speed gain/offset at 1/0 until longitudinal
identification is sufficiently confident; steering and launch learning continue.
The [approved run](../../docs/paper/20261024_005311/README.md) completed laps in
46.837/14.594 s, with 8/1 reverse starts. See the
[race-stack guide](../../docs/race_stack.md) for current interfaces and limits.
