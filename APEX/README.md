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
cd ~/AiAtonomousRc/APEX
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

## 5b) Static Recon Debugging Workflow

For planner debugging, do not use `./run_apex.sh` directly with `APEX_ENABLE_RECON_MAPPING=1`.
With the current defaults, `diagnostic_mode` falls back to `calibration`, so the node runs scripted calibration steps instead of a continuous reconnaissance lap.

Use this two-layer workflow instead:
- keep the container running in core mode with LiDAR + IMU + kinematics only,
- start and stop individual debug sessions inside the running container with `tools/apex_recon_start.sh`.

Every debug run persists a bundle under `ros2_ws/debug_runs/<run_id>_<UTC timestamp>/` with:
- `recon_diagnostic.log`
- `docker_tail.log`
- `run_metadata.json`
- config snapshots
- rosbag2 MCAP capture

### Raspberry: bring up the core container once

```bash
cd ~/AiAtonomousRc/APEX
rm -rf debug_runs logs ros2_ws/debug_runs ros2_ws/logs
./tools/apex_core_down.sh
APEX_SKIP_BUILD=1 ./tools/apex_core_up.sh
docker ps | grep apex_pipeline
docker logs --tail 80 apex_pipeline
```

This runs the container in core mode:
- `APEX_ENABLE_RECON_MAPPING=0`
- `APEX_ENABLE_SLAM_TOOLBOX=0`
- `APEX_RECORD_DEBUG=0`

### Raspberry: verify live sensor topics

```bash
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
ros2 topic list | egrep "/lidar/scan|/odom|/apex/kinematics"
'
docker exec -it apex_pipeline bash -lc '
source /opt/ros/jazzy/setup.bash
timeout 5 ros2 topic hz /lidar/scan
'
```

### Raspberry: run one `nav_dryrun` case

Quick wrapper:

```bash
cd ~/AiAtonomousRc/APEX
./tools/apex_nav_dryrun_case.sh 01_left_curve_center
```

Equivalent manual command:

```bash
cd ~/AiAtonomousRc/APEX
./tools/apex_recon_start.sh --run-id 01_left_curve_center --mode nav_dryrun --record-debug 1 --timeout-s 15
sleep 17
docker exec apex_pipeline /bin/bash /work/ros2_ws/scripts/apex_recon_session.sh status
ls -1t ros2_ws/debug_runs | head -n 3
```

### Raspberry: first static matrix

Use the same geometry twice, mirrored left/right, and repeat centered plus both near-wall offsets:
- `01_left_curve_center`
- `02_left_curve_near_left_wall`
- `03_left_curve_near_right_wall`
- `04_right_curve_center`
- `05_right_curve_near_left_wall`
- `06_right_curve_near_right_wall`

### Raspberry: steering and sign preflight before moving the car

```bash
cd ~/AiAtonomousRc/APEX
./tools/apex_recon_start.sh --run-id 90_steering_static --mode steering_static --record-debug 1
sleep 6
./tools/apex_recon_start.sh --restart --run-id 91_steering_sign_check --mode steering_sign_check --record-debug 1 --fixed-speed-pct 20
sleep 8
```

If the physical steering direction is inverted, repeat with:

```bash
APEX_STEERING_DIRECTION_SIGN=-1
```

### PC: fetch bundles from Raspberry

Copy all bundles:

```bash
cd ~/AiAtonomousRc/APEX
mkdir -p debug_runs
rsync -av ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/ros2_ws/debug_runs/ ./debug_runs/
ls -1t debug_runs | head -n 10
```

Fetch just the latest bundle:

```bash
cd ~/AiAtonomousRc/APEX
./tools/fetch_debug_run.sh
```

### PC: analyze one bundle

Run the full analysis stack:

```bash
cd ~/AiAtonomousRc/APEX
./tools/analyze_debug_bundle.sh ./debug_runs/<bundle_dir>
```

Equivalent manual commands:

```bash
cd ~/AiAtonomousRc/APEX
python3 ./tools/analyze_debug_run.py ./debug_runs/<bundle_dir>
python3 ./tools/replay_nav_from_bag.py ./debug_runs/<bundle_dir>
python3 ./tools/explain_recon_run.py ./debug_runs/<bundle_dir>
sed -n '1,200p' ./debug_runs/<bundle_dir>/analysis/summary.md
cat ./debug_runs/<bundle_dir>/analysis/flags.json
```

Artifacts are written under `./debug_runs/<bundle_dir>/analysis/`.

Key files:
- `summary.md`
- `flags.json`
- `decision_timeline.csv`
- `replay_nav.csv`
- `trajectory_explainer.md`

### PC: replay the run in RViz

```bash
cd ~/AiAtonomousRc/APEX
./tools/review_debug_run.sh ./debug_runs/<bundle_dir>
```

## Quick Reset and Restart (Raspberry + PC)

Use this sequence when you want to discard the previous SLAM map and start a fresh mapping session.

Important:
- On the Raspberry host, do not run `source /opt/ros/jazzy/setup.bash`; ROS runs inside Docker.
- The reset below deletes the last saved `apex_map.yaml` and `apex_map.pgm` if they exist.

### Raspberry Terminal 1 (restart SLAM from scratch)

```bash
cd ~/AiAtonomousRc/APEX
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
