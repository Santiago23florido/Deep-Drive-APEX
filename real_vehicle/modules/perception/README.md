# Perception module

The retained hardware ingestion lives in
[`apex_telemetry`](../../ros2_ws/src/apex_telemetry/apex_telemetry/): WT61PC IMU
streaming at 100 Hz and A2M8 Express LiDAR at 10 Hz, with device-clock timestamp
reconstruction. Firmware lives in [`arduino`](../../arduino/).

The [approved run](../../docs/paper/20261024_005311/README.md) preserves 9,002 raw
IMU samples, 889 scans and receipt timestamps. Four LiDAR intervals exceeded
150 ms; no stale-data stop was recorded. Best-effort recording and finite files
do not prove lossless sensor transport. See the
[sensor timing guide](../../docs/sensor_timing_requirements.md) and
[runtime graph](../../docs/race_stack.md) for the current flow.
