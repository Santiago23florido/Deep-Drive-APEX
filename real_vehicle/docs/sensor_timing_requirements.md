# Sensor Timing Requirements (real2sim)

The learned LiDAR-inertial odometry is trained and validated in simulation on the car's real sensors, the RPLIDAR A2M8 and the LSM6DS3 IMU of the Arduino Nano 33 IoT, working **as the hardware allows**. The simulation models this with the sensor profile `APEX_real` in `simulation/tools/pose_dataset/config/sensors.yaml`. The default model of the simulation (`real2sim_v5_10hz`) is trained for that profile.

The car's previous software delivered these sensors with timing defects, listed below. They were software issues, not hardware limits, and the simulation does not model them.

**Status: implemented, not yet validated on the car.** Branch `feature/real-car-sensor-timing` rewrites the IMU firmware, the IMU serial node and the LiDAR driver. The work is checked in three ways:
- the tests;
- the IMU firmware compiles for the Nano 33 IoT;
- the real ROS nodes run against a fake A2M8 and a fake Nano on pseudo-terminals, and the recording passes every check of `sensor_timing_audit.py --profile APEX_real`.

The hardware checks still to do are listed in [Bring-up on the car](#bring-up-on-the-car).

## Target timing

| Sensor | Requirement | Why |
| --- | --- | --- |
| IMU | One sample per sensor output (LSM6DS3 ODR 104 Hz, 9.6 ms), no skipped or repeated samples. | The model integrates the gyro over each 100 ms scan interval and expects about 10 samples per interval. |
| IMU stamp | Time of the sample on the device (data-ready), mapped to the ROS clock, with jitter < 0.5 ms. | A stamp taken on reception adds the USB and host latency, plus its jitter, to every sample. |
| LiDAR scan | Express scan, 4000 samples/s. Every A2M8 supports it from firmware 1.17. | About 400 samples per revolution at 10 Hz, one or more in every 1° bin outside the blind sector. The compatible 2 kHz protocol leaves ~60 % of the bins empty, and the ICP finds no correspondences (see Validation). |
| LiDAR rotation | 10 Hz (600 rpm) ± 1 %. | The rate `real2sim_v5_10hz` was trained at. At 13 Hz only ~307 samples fall in a revolution, and the reactive driver and SLAM lost walls in simulation. |
| LiDAR revolution | One `LaserScan` per full revolution, never partial revolutions. | Scan intervals are the model's time steps. |
| LiDAR stamp | Time of the first sample of the revolution (sync). `scan_time` is the revolution period. | The model and the ICP de-skew every beam from that stamp. The beam order is clockwise from +89° (sync angle with `heading_offset_deg: 91`). |
| LiDAR layout | `/lidar/scan_raw`: 360 bins from −180° counter-clockwise, bin *k* centred on −180° + *k*, nearest return per bin, REP-117 (+inf no return, −inf below 0.15 m). | The simulation's layout (`lidar_rolling._bin`, `binning: min_range`). |
| Blind sector | +145°…−155° behind the car (car structure). Leave it as it is. | It is part of the hardware and is modelled. |

The Sensitivity / Boost scan (8 kHz) needs firmware ≥ 1.24 and is not assumed. The LiDAR node logs the firmware at start-up. If it is ≥ 1.24, the 8 kHz profile `APEX_real_sensitivity` becomes an option. That needs the ultra-capsule decoder, which is not implemented.

## Defects found (evidence) and their fixes

Measured with `simulation/tools/analysis/real_sensor_identification.py` and `simulation/tools/analysis/sensor_timing_audit.py` on `real_vehicle/data`.

| Defect | Evidence | Cause in the code | Fix (this branch) |
| --- | --- | --- | --- |
| IMU at 55.6 Hz instead of 104 Hz | Period 18.0 ms (p5 17.9, p95 18.1) in both IMU recordings; 4 samples per scan instead of 8. | `arduino/nano33_iot_accel_stream`: `delay(10)` plus blocking reads and prints in `loop()`. | Firmware v2: no delay. It polls the data-ready flags, reads each sample once, and runs I2C at 400 kHz. |
| IMU stamped on reception | Stamp = `now()` when the serial line is read. | `nano_accel_serial_node.py` (`_publish_sample`). | Firmware v2 sends `micros()` at data-ready plus a sample counter. The node maps the MCU clock to ROS time (`timing/device_clock.py`). |
| Irregular LiDAR intervals | Intervals of 50–60 ms (8–16 %) next to 75–80 ms; `lidar_no_partial_revolutions` fails. | `rplidar_publisher_node.py`: `iter_scans` of the python `rplidar` library, stamp = `now()` after processing, `scan_time` measured between publications. | Backend `rplidar_express`: one scan per complete revolution, from sync to sync. It is stamped at its first sample from the LiDAR's own sample clock. |
| Sparse scans | ~112 valid bins out of 360 per revolution. | The python `rplidar` library uses the legacy compatible protocol (2 kHz). | Express scan (4 kHz). |
| 13 Hz rotation | 13.03 Hz measured. | The python library sets the motor PWM to 660. | The motor PWM is regulated to `target_rate_hz` (10 Hz). |

## Implementation

All paths below are under `real_vehicle/`.

- **IMU firmware** `arduino/nano33_iot_accel_stream/nano33_iot_accel_stream.ino` (v2).
  - Line format: `ax,ay,az,gx,gy,gz,t_us,seq`, in m/s² and rad/s.
  - `t_us` is `micros()` at data-ready; `seq` counts the samples.
  - The first six fields are unchanged, so the previous host node still reads v2 lines, stamping them on reception.
  - The chip configuration is unchanged (`IMU.begin()`: 104 Hz, ±4 g, ±2000 dps), because it is the one `APEX_real` models.
- **Clock mapping** `ros2_ws/src/apex_telemetry/apex_telemetry/timing/device_clock.py`.
  - For each packet, `host_arrival = device_time + offset + latency`, with `latency ≥ 0`.
  - A line fitted over 10 s to the lower envelope of `host − device` maps device times to ROS time. That removes the USB and scheduling jitter, and the slope absorbs the clock drift.
  - It resets when the device restarts or when the host clock jumps.
- **IMU node** `imu/nano_accel_serial_node.py` (line parser in `imu/imu_line.py`).
  - It reads the port in blocks, not byte by byte with `readline`.
  - It stamps each sample at its MCU time, detects MCU restarts from `seq`, and unwraps `micros()`.
  - Every 10 s it logs the rate, the missed chip samples, the lost lines, the clock drift and the reception latency.
  - Lines without `t_us` (firmware v1) are stamped on reception, with a warning.
- **LiDAR driver** `perception/rplidar_express.py`, backend `rplidar_express` of `perception/rplidar_publisher_node.py` (now the default).
  - It speaks the RPLIDAR protocol directly. The capsule parser, the angle interpolation and the sample timing model are ported from the Slamtec SDK of `rplidar_ros` 2.1.4. The decoder matches the SDK's C++ arithmetic on 640 000 random samples.
  - Why not `rplidar_ros`: on the A series it always spins the motor at the device default PWM, so it cannot hold 10 Hz. It also stamps a scan with the host time taken before a blocking read.
  - Sample *n* is measured at device time `n × Ts`, where `Ts` = 250 µs is reported by the LiDAR. Capsule *c* reaches the host `1.5 Ts + 840 bits / baud` after its last sample (the SDK timing model) plus the USB latency. The clock mapper turns `n × Ts` into ROS time.
  - A revolution runs from one sync sample to the next.
  - Revolutions that lost a capsule, or whose sample count is off the running median by more than 15 %, are dropped whole. The odometry is trained with whole-revolution dropouts.
  - Its device-clock envelope uses a 3 s window, 6 chunks and 0.25 s refits, with the existing 3 % drift bound. This recovers timing faster after serial losses; the IMU keeps its separate clock settings. See `race_stack.md` section 5k for the real-car validation and the precision tradeoff.
  - The motor PWM is regulated to `target_rate_hz`: an integral step every 20 revolutions, with a 0.05 Hz deadband.
  - Topics:
    - `/lidar/scan`, `/lidar/scan_localization` and `/lidar/scan_slam`: the same layouts as before, now with revolution stamps.
    - `/lidar/scan_raw` (new): the simulation's layout, the input of the learned odometry.
  - CPU: about 1 % of one PC core, so a few percent of a Pi 4.
- **Recorder** `ros2_ws/scripts/capture/record_real2sim_sensors.py`. It writes `imu_raw.csv` and `lidar_scans.csv` in the simulation's formats, which the audit reads.
- **Configuration** `ros2_ws/src/apex_telemetry/config/apex_params.yaml`:

  | Parameter | Value | Meaning |
  | --- | --- | --- |
  | `source_backend` | `rplidar_express` | Driver backend. |
  | `target_rate_hz` | 10.0 | Rotation rate the motor is regulated to. |
  | `motor_pwm` | 520 | Start value of the motor PWM. |
  | `motor_start_pwm`, `motor_ramp_s` | 200, 3.0 | Soft start: the PWM climbs from 200 to `motor_pwm` over 3 s instead of stepping, to limit the motor's start-up current on the Pi's USB port. |
  | `retry_max_s` | 30 | After a failure the motor is stopped and the LiDAR reopened after 1, 2, 4 … 30 s. |
  | `regulate_motor` | true | Regulate the PWM towards `target_rate_hz`. |
  | `stamp_correction_ms` | 0 | Minimum USB latency subtracted from the stamps. |
  | `use_device_timestamps` | true | IMU: stamp each sample at its MCU time. |

  `start_apex_pipeline.sh` takes `APEX_LIDAR_BACKEND` (default `rplidar_express`).
  In race runs, `power_status_file` defaults to `power.csv` in
  `APEX_RACE_RUN_DIR`. The startup supply guard uses a fresh (at most 3 s old)
  host `get_throttled` bit 0 to distinguish current undervoltage from latched
  history. Missing, incomplete or stale rows fall back to the hwmon alarm,
  which can retain a historical fault. The extra motor-ramp hold budget is
  cumulative across all PWM steps; it is not restarted at each step. The
  separate host pre-arm power and temperature check remains required.
- **Tests** `ros2_ws/src/apex_telemetry/test/test_sensor_timing.py`. They run without ROS or hardware and cover:
  - the clock mapper, including jitter, drift, restarts and never stamping after reception;
  - IMU lines;
  - protocol framing;
  - capsule decoding against the protocol formula;
  - a synthetic A2M8 delivered byte by byte over a USB-like link, with stamps within 0.3 ms of the true first-sample time;
  - lost capsules;
  - the scan layout;
  - the motor regulator;
  - the driver against a fake port.

## Bring-up on the car

### LiDAR alone (first)

`tools/lidar/apex_lidar_only.sh` runs only the LiDAR node in the APEX image, in a detached container (`apex_lidar`): no IMU node, no actuators, and the SSH session stays free. It refuses to start while `apex_pipeline` or `apex-real-ready.service` holds the port, or while the Pi reports under-voltage. The motor starts after `APEX_LIDAR_START_DELAY_S` (5 s) with the soft start.

```bash
sudo systemctl stop apex-real-ready.service    # if installed; it restarts the full stack
./tools/core/apex_core_down.sh
./tools/lidar/apex_lidar_only.sh power         # throttled flags before the motor starts
./tools/lidar/apex_lidar_only.sh up
./tools/lidar/apex_lidar_only.sh logs          # firmware, soft start, "LiDAR timing: 10.0x Hz"
./tools/lidar/apex_lidar_only.sh check --duration 60     # car at rest in a room: PASS/FAIL per requirement
./tools/lidar/apex_lidar_only.sh check watch --expect 0  # object ahead; then --expect 90 with it on the left
./tools/lidar/apex_lidar_only.sh record lidar_static_01 60
./tools/lidar/apex_lidar_only.sh down
```

`check` (`ros2_ws/scripts/lidar_bringup_check.py`) grades `/lidar/scan_raw`: layout, 10 Hz ± 1 %, regular and complete revolutions, lost revolutions ≤ 2 %, `scan_time` equal to the revolution period, stamp at the first sample (received 0–50 ms after the end of its revolution), valid bins, and the always-empty sector behind the car. It also reports the missing-return rate outside the blind sector against the 12 % of `APEX_real`. `watch --expect DEG` prints the `heading_offset_deg` (an integer) that puts the object at DEG.

If the Pi resets or drops its network when the motor starts, the motor's start-up current is sagging the Pi's 5 V supply. The soft start limits it. If it still happens, `power` shows `under-voltage since boot 1`; power the LiDAR adapter from its auxiliary micro-USB port on a separate 5 V supply, and use a USB 2.0 (black) port. `tools/core/enable_pi_persistent_journal.sh` and `tools/core/collect_pi_reset_diagnostics.sh` keep and collect the evidence across a reset.

### Full checklist

1. **Flash the IMU firmware.**
   ```bash
   arduino-cli compile --fqbn arduino:samd:nano_33_iot real_vehicle/arduino/nano33_iot_accel_stream
   arduino-cli upload  --fqbn arduino:samd:nano_33_iot -p /dev/ttyACM0 real_vehicle/arduino/nano33_iot_accel_stream
   ```
   The library needed is `Arduino_LSM6DS3`.
2. **Start the pipeline and read the logs.**
   - LiDAR: model and firmware; `Express scan started: 250 us per sample`. After ~10 s, `LiDAR timing: 10.0x Hz`, with ~400 samples/rev and no checksum errors.
     - An error `no PWM motor control` means the adapter cannot set the speed.
     - Note the firmware: ≥ 1.24 enables the 8 kHz option.
   - IMU: `IMU timing: 104 Hz`, `chip samples missed 0`, a clock drift of a few tens of ppm, and a reception latency p50 around 1 ms.
3. **Orientation.** Put an object on the car's left: it must appear near +90° in `/lidar/scan_raw`, and an object ahead near 0°. If not, `heading_offset_deg` is wrong. The value 91 comes from the previous configuration of the car.
4. **Record and audit**: 60 s at rest, then 60 s pushing the car by hand.
   ```bash
   python3 real_vehicle/ros2_ws/scripts/capture/record_real2sim_sensors.py --out real_vehicle/data/real2sim_capture/<name> --duration 60
   python3 simulation/tools/analysis/sensor_timing_audit.py real_vehicle/data/real2sim_capture/<name> --profile APEX_real
   ```
   Every check must pass.
5. **Measure what the simulation assumed.**
   - The missing-return rate of the Express scan: the fraction of empty bins in a static scene outside the blind sector, against the 12 % of `APEX_real`.
   - Optionally, the LiDAR–IMU time offset: rotate the car in place and cross-correlate the gyro yaw rate with the scan-matching rotation. Put the result in `stamp_correction_ms`. The odometry is trained robust to ±8 ms, so this is not required.
6. **Replay the recording offline** through the learned odometry before closing the loop.

**Rollback.**
- LiDAR: `APEX_LIDAR_BACKEND=rplidar` (the previous driver); `docker-compose.yml` passes it to the container.
- IMU: the old firmware still works with the new node, stamped on reception with a warning; the new firmware works with the old node.
- The unused legacy package `voiture_system` is not modified.

## Validation

Record the sensors with `record_real2sim_sensors.py`, then audit them:

```bash
python3 simulation/tools/analysis/sensor_timing_audit.py <recording> --profile APEX_real
```

Every check must pass. Older recordings, from the recorders that write `lidar_points.csv`, fail on the IMU rate, the periods, the partial revolutions and the valid bins. Two recordings pass all of them:
- a simulated run (`simulation/tools/sim/apex_real2sim_up.sh`);
- the new drivers against the fake devices: IMU 104.3 Hz with stamp intervals of 9.587 ms ± 3 µs despite a reception jitter of ~20 ms at p95; LiDAR 10.03 Hz with 273 valid bins per revolution.

The high-speed LiDAR mode is **required**, not optional. With the 2 kHz compatible protocol a revolution has about 150 samples spread over 360 bins, so almost no bin has valid neighbours. The classical ICP that feeds the hybrid network estimates its normals from consecutive bins, and there it finds no correspondences at all: 0 on the simulated `APEX_real_compat` runs, and the odometry diverges on the IMU alone. The model is trained for the Express scan at 10 Hz (`APEX_real`). `--profile APEX_real_compat` only describes the previous driver.
