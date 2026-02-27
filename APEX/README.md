# APEX

APEX is a clean modular stack for Raspberry-based mapping.

Current scope:
- Arduino Nano 33 IoT IMU ingestion (accelerometer + gyroscope)
- LiDAR publishing (`/lidar/scan`)
- SLAM on Raspberry with `slam_toolbox` (map reconstruction)
- Remote visualization from PC

No motor control is included.

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

## Notes

- With no wheel encoders and no motor model, LiDAR scan matching is the main mapping reference.
- The IMU branch (accel + gyro) is available for telemetry and auxiliary odometry.
- If you run into LiDAR serial issues, verify no other process is using `/dev/ttyUSB*`.
