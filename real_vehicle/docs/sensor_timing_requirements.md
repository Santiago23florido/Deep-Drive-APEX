# Sensor Timing Requirements (real2sim)

The learned LiDAR-inertial odometry is trained and validated in simulation on the car's real sensors, the RPLIDAR A2M8 and the LSM6DS3 IMU of the Arduino Nano 33 IoT, working **as the hardware allows**. The simulation models this with the sensor profile `APEX_real` in `simulation/tools/pose_dataset/config/sensors.yaml`.

Today's car software delivers these sensors with timing defects, listed below. They are software issues, not hardware limits. The simulation does not model them.

**Status: recommendations only.** Nothing on the car has been changed. Apply them once the real2sim pipeline is validated in simulation, before running the learned odometry on the car.

## Target timing

| Sensor | Requirement | Why |
| --- | --- | --- |
| IMU | One sample per sensor output (LSM6DS3 ODR 104 Hz, 9.6 ms), no skipped or repeated samples. | The model integrates the gyro over each ~77 ms scan interval and expects about 8 samples per interval. |
| IMU stamp | Time of the sample on the device (data-ready), mapped to the ROS clock, with jitter < 0.5 ms. | A stamp taken on reception adds the USB and host latency, plus its jitter, to every sample. |
| LiDAR revolution | One `LaserScan` per full revolution (~13 Hz, 76.7 ms), never partial revolutions. | Scan intervals are the model's time steps. |
| LiDAR stamp | Time of the first sample of the revolution (sync pulse). `scan_time` is the revolution period. | The model and the ICP de-skew every beam from that stamp. The beam order is clockwise from +89° (sync angle with `heading_offset_deg: 91`). |
| LiDAR density | High-speed protocol (Sensitivity / Express, up to 8 kHz): about 600 samples per revolution, so every 1° bin is filled outside the blind sector. | The compatible 2 kHz protocol gives about 150 samples per revolution, so ~60 % of the bins are empty. The classical ICP then finds no correspondences (see Validation). |
| Blind sector | +145°…−155° behind the car (car structure). Leave it as it is. | It is part of the hardware and is modelled. |

## Defects found (evidence)

Measured with `simulation/tools/analysis/real_sensor_identification.py` and `simulation/tools/analysis/sensor_timing_audit.py` on `real_vehicle/data`.

| Defect | Evidence | Cause in the code | Fix |
| --- | --- | --- | --- |
| IMU at 55.6 Hz instead of 104 Hz | Period 18.0 ms (p5 17.9, p95 18.1) in both IMU recordings; 4 samples per scan instead of 8. | `arduino/nano33_iot_accel_stream`: `delay(10)` plus blocking reads and prints in `loop()`. | Read on data-ready (or the FIFO) without `delay`; stream every sample, in binary or short lines. |
| IMU stamped on reception | Stamp = `now()` when the serial line is read. | `nano_accel_serial_node.py` (stamp in `_publish`, line 518). | Send the MCU timestamp (`micros()`) with each sample and map it to ROS time (offset plus drift estimate). |
| Irregular LiDAR intervals | Intervals of 50–60 ms (8–16 %) next to 75–80 ms; `lidar_no_partial_revolutions` fails. | `rplidar_publisher_node.py`: `iter_scans` in a Python loop, stamp = `now()` after processing (line 522); `scan_time` measured between publications. | Use the Slamtec SDK / `rplidar_ros` (C++); stamp = start of revolution = end − scan duration. |
| Sparse scans | ~112 valid bins out of 360 per revolution. | The Python `rplidar` library uses the legacy compatible protocol (2 kHz). | Enable the high-speed (Sensitivity / Express) scan mode. |

## Validation

Record the sensors (the current recorders write `imu_raw.csv` and `lidar_points.csv`), then audit them:

```bash
python3 simulation/tools/analysis/sensor_timing_audit.py real_vehicle/data/<capture> --profile APEX_real
```

Every check must pass. Current recordings fail on the IMU rate, the periods, the partial revolutions and the valid bins. A simulated run (`simulation/tools/sim/apex_real2sim_up.sh`) passes all of them.

The high-speed LiDAR mode is **required**, not optional. With the 2 kHz compatible protocol a revolution has about 150 samples spread over 360 bins, so almost no bin has valid neighbours. The classical ICP that feeds the hybrid network estimates its normals from consecutive bins, and there it finds no correspondences at all: 0 on the simulated `APEX_real_compat` runs, and the odometry diverges on the IMU alone. The model is trained for the high-speed protocol (`APEX_real`). `--profile APEX_real_compat` only describes today's driver.
