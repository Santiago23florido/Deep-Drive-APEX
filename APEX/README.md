# APEX

APEX is a clean modular stack for Raspberry-based mapping.

Current scope:
- Arduino Nano 33 IoT IMU ingestion (accelerometer + gyroscope)
- LiDAR publishing (`/lidar/scan`)
- SLAM on Raspberry with `slam_toolbox` (map reconstruction)
- Remote visualization from PC
- Optional autonomous reconnaissance lap for slow wall-aware mapping

Motor control for reconnaissance is optional and only enabled when explicitly requested.

## Architecture

```text
APEX/
  arduino/
    nano33_iot_accel_stream/
  docker/
    Dockerfile
    docker-compose.yml
  ros2_ws/
    src/apex_telemetry/
      apex_telemetry/
        nano_accel_serial_node.py
        kinematics_estimator_node.py
        kinematics_odometry_node.py
        rplidar_publisher_node.py
      launch/
        apex_pipeline.launch.py
        apex_lidar_slam.launch.py
      config/
        apex_params.yaml
        apex_slam_toolbox.yaml
    scripts/start_apex_pipeline.sh
  run_apex.sh
```

## ROS Graph (Raspberry)

- `nano_accel_serial_node`: serial -> `/apex/imu/acceleration/raw` + `/apex/imu/angular_velocity/raw` + `/apex/imu/data_raw`
- `kinematics_estimator_node`: IMU raw topics -> kinematics topics + heading/yaw-rate
- `kinematics_odometry_node`: kinematics + heading/yaw-rate -> `/odom` + TF `odom -> base_link`
- `rplidar_publisher_node`: `/dev/ttyUSB*` -> `/lidar/scan`
- `slam_toolbox` (`online_async_launch.py`): `/lidar/scan` + TF -> `/map`, `/tf`
- `static_transform_publisher`: TF `base_link -> laser`

## 1) Arduino

Upload:
- `APEX/arduino/nano33_iot_accel_stream/nano33_iot_accel_stream.ino`

Library:
- `Arduino_LSM6DS3`

Output format:
- `ax,ay,az,gx,gy,gz`
- acceleration in `m/s^2`, angular velocity in `rad/s` at `115200`.

## 2) Raspberry (Docker + SLAM)

From Raspberry host:

```bash
cd ~/Voiture-Autonome/code/APEX
./run_apex.sh -d
```

`run_apex.sh` is a wrapper for Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

If your ports differ:

```bash
APEX_SERIAL_PORT=/dev/ttyACM0 \
APEX_LIDAR_PORT=/dev/ttyUSB0 \
APEX_LIDAR_BAUDRATE=115200 \
./run_apex.sh -d
```

Equivalent direct Docker command:

```bash
APEX_SERIAL_PORT=/dev/ttyACM0 \
APEX_LIDAR_PORT=/dev/ttyUSB0 \
APEX_LIDAR_BAUDRATE=115200 \
docker compose -f docker/docker-compose.yml up --build -d
```

## 2b) Raspberry (Automatic Reconnaissance Mapping)

This mode starts the same SLAM stack and additionally launches a low-speed LiDAR-based reconnaissance controller that:
- resets the current `slam_toolbox` map before the first movement command,
- begins moving automatically,
- follows open space while staying away from walls,
- performs a simple reverse recovery if blocked,
- stops after detecting a closed lap near the start pose,
- optionally saves the map to `/work/ros2_ws/maps/apex_recon_map`.

Host prerequisites:
- PWM already configured on the Raspberry (`/sys/class/pwm` available).
- Steering and motor channels matching the values in `apex_params.yaml`.

Start reconnaissance mapping:

```bash
cd ~/AiAtonomousRc/APEX
APEX_ENABLE_RECON_MAPPING=1 ./run_apex.sh -d
```

Follow logs:

```bash
docker logs -f apex_pipeline
```

Check status:

```bash
docker ps | grep apex_pipeline
docker logs -f apex_pipeline
```

## 3) Validate Topics on Raspberry

```bash
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
source /work/ros2_ws/install/setup.bash
ros2 topic list | egrep "/lidar/scan|/map|/odom|/apex"
'
```

Quick checks:

```bash
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
source /work/ros2_ws/install/setup.bash
ros2 topic hz /lidar/scan
ros2 topic hz /map
'
```

## 4) Visualize from PC

Use the same DDS settings that already worked in your project:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Start RViz2 with APEX preconfigured profile (no manual RViz setup required):

```bash
cd ~/AiAtonomousRc/APEX
chmod +x run_apex_rviz_pc.sh
./run_apex_rviz_pc.sh
```

## 5) Save Map (Raspberry)

```bash
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
source /work/ros2_ws/install/setup.bash
mkdir -p /work/ros2_ws/maps
ros2 run nav2_map_server map_saver_cli -f /work/ros2_ws/maps/apex_map
'
```

## 5b) Directed Wall-Approach Debugging

This workflow isolates three possible causes when the vehicle sees free space on one side but turns the other way:
- navigation logic sign mismatch,
- LiDAR heading offset mismatch,
- steering servo sign mismatch.

Every debug run can persist a bundle under `ros2_ws/debug_runs/<run_id>_<UTC timestamp>/` with:
- `recon_diagnostic.log`
- `docker_tail.log`
- `run_metadata.json`
- config snapshots
- rosbag2 MCAP capture

### Raspberry: Stage 1, servo static

```bash
cd ~/Voiture-Autonome/code/APEX
APEX_ENABLE_RECON_MAPPING=1 \
APEX_RECORD_DEBUG=1 \
APEX_DEBUG_RUN_ID=01_servo_static \
APEX_RECON_DIAGNOSTIC_MODE=steering_static \
./run_apex.sh -d
```

### Raspberry: Stage 2, servo sign check

```bash
cd ~/Voiture-Autonome/code/APEX
APEX_ENABLE_RECON_MAPPING=1 \
APEX_RECORD_DEBUG=1 \
APEX_DEBUG_RUN_ID=02_servo_sign_check \
APEX_RECON_DIAGNOSTIC_MODE=steering_sign_check \
APEX_RECON_FIXED_SPEED_PCT=10.0 \
./run_apex.sh -d
```

If the physical steering direction is inverted, repeat the same run with:

```bash
APEX_STEERING_DIRECTION_SIGN=-1
```

### Raspberry: Stage 3, logic dry-run with space open on the right

This computes the full navigation decision from the live LiDAR scan, but does not drive the vehicle.

```bash
cd ~/Voiture-Autonome/code/APEX
APEX_ENABLE_RECON_MAPPING=1 \
APEX_RECORD_DEBUG=1 \
APEX_DEBUG_RUN_ID=03_nav_dryrun_right_open \
APEX_RECON_DIAGNOSTIC_MODE=nav_dryrun \
./run_apex.sh -d
```

### Raspberry: Stage 4, logic dry-run with space open on the left

```bash
cd ~/Voiture-Autonome/code/APEX
APEX_ENABLE_RECON_MAPPING=1 \
APEX_RECORD_DEBUG=1 \
APEX_DEBUG_RUN_ID=04_nav_dryrun_left_open \
APEX_RECON_DIAGNOSTIC_MODE=nav_dryrun \
./run_apex.sh -d
```

### Raspberry: Stage 5, LiDAR orientation checks

Run three static captures with the wall physically placed in front, on the right, and on the left. Use different `APEX_DEBUG_RUN_ID` values such as:
- `05_orientation_front_wall`
- `05_orientation_right_wall`
- `05_orientation_left_wall`

If the scan sectors are mirrored or rotated, adjust:

```bash
APEX_LIDAR_HEADING_OFFSET_DEG=<new_offset_deg>
```

### Raspberry: Stage 6+, slow wall approach

```bash
cd ~/Voiture-Autonome/code/APEX
APEX_ENABLE_RECON_MAPPING=1 \
APEX_RECORD_DEBUG=1 \
APEX_DEBUG_RUN_ID=06_single_wall_right \
APEX_RECON_DIAGNOSTIC_MODE=recon_debug \
APEX_RECON_FIXED_SPEED_PCT=8.0 \
./run_apex.sh -d
```

Mirror the same run for the left wall:

```bash
APEX_DEBUG_RUN_ID=07_single_wall_left
```

### PC: fetch one bundle from Raspberry

```bash
cd ~/AiAtonomousRc/APEX
./tools/fetch_debug_run.sh ensta@raspberrypi.local latest
```

### PC: analyze one bundle

```bash
cd ~/AiAtonomousRc/APEX
python3 ./tools/analyze_debug_run.py ./debug_runs/<run_id>
```

Artifacts are written under:

```bash
./debug_runs/<run_id>/analysis/
```

Key files:
- `summary.md`
- `decision_timeline.csv`
- `flags.json`
- `plots/headings.svg`
- `plots/clearances.svg`

### PC: replay navigation from the recorded bag

```bash
cd ~/AiAtonomousRc/APEX
python3 ./tools/replay_nav_from_bag.py ./debug_runs/<run_id>
```

### PC: replay the run in RViz

```bash
cd ~/AiAtonomousRc/APEX
./tools/review_debug_run.sh ./debug_runs/<run_id>
```

## Quick Reset and Restart (Raspberry + PC)

Use this sequence when you want to discard the previous SLAM map and start a fresh mapping session.

Important:
- On the Raspberry host, do not run `source /opt/ros/jazzy/setup.bash`; ROS runs inside Docker.
- The reset below deletes the last saved `apex_map.yaml` and `apex_map.pgm` if they exist.

### Raspberry Terminal 1 (restart SLAM from scratch)

```bash
cd ~/Voiture-Autonome/code/APEX
docker compose -f docker/docker-compose.yml down --remove-orphans
rm -f ros2_ws/maps/apex_map.yaml ros2_ws/maps/apex_map.pgm
APEX_SERIAL_PORT=/dev/ttyACM0 \
APEX_LIDAR_PORT=/dev/ttyUSB0 \
APEX_LIDAR_BAUDRATE=115200 \
./run_apex.sh -d
```

### Raspberry Terminal 2 (optional: monitor container logs)

```bash
docker logs -f apex_pipeline
```

### PC Terminal 1 (restart RViz and display the fresh map)

```bash
cd ~/AiAtonomousRc/APEX
pkill -f rviz2 || true
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
./run_apex_rviz_pc.sh
```

### Raspberry Terminal 3 (optional: verify active topics)

```bash
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
ros2 topic list | egrep "/lidar/scan|/map|/odom|/apex"
'
```

## Notes

- With no wheel encoders and no motor model, LiDAR scan matching is the main mapping reference.
- The IMU branch (accel + gyro) is available for telemetry and auxiliary odometry.
- If you run into LiDAR serial issues, verify no other process is using `/dev/ttyUSB*`.
