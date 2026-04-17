# Glossary of ROS and Gazebo Terms

This glossary explains the ROS and Gazebo terms used in this documentation, using examples from this repository.

## ROS 2

ROS 2 is the middleware framework used to connect sensors, estimators, planners, controllers, visualization tools, and hardware drivers. This repository expects ROS 2 Jazzy for the current workflows.

## Package

A package is a buildable ROS unit. It contains code, launch files, configuration, and metadata.

Packages in this repository:

| Package | Purpose |
| --- | --- |
| `rc_sim_description` | Gazebo Sim vehicle model and simulation tools. |
| `apex_telemetry` | Current APEX autonomy nodes. |
| `voiture_system` | Alternate real-car SLAM/Nav2 stack. |

## Workspace

A workspace is a directory tree that contains ROS packages and is built with `colcon`.

This repository uses two source roots:

```text
src/
APEX/ros2_ws/src/
```

Build both roots together for the APEX workflow.

## Node

A node is a running ROS process. A node performs one job, such as reading LiDAR data, estimating odometry, or tracking a path.

Examples:

- `rplidar_publisher_node`
- `imu_lidar_planar_fusion_node`
- `recognition_tour_planner_node`
- `cmd_vel_to_apex_actuation_node`

## Topic

A topic is a named stream of messages. One node publishes messages; other nodes subscribe to them.

Examples:

| Topic | Meaning |
| --- | --- |
| `/apex/imu/data_raw` | Raw IMU messages. |
| `/lidar/scan_localization` | LiDAR scans used by APEX localization. |
| `/apex/odometry/imu_lidar_fused` | Fused odometry estimate. |
| `/apex/cmd_vel_track` | APEX tracking command. |

## Message Type

A message type defines the structure of data on a topic.

Examples:

| Type | Use |
| --- | --- |
| `sensor_msgs/msg/Imu` | IMU data. |
| `sensor_msgs/msg/LaserScan` | 2D LiDAR scan. |
| `nav_msgs/msg/Odometry` | Odometry estimate. |
| `nav_msgs/msg/Path` | Path or trajectory. |
| `geometry_msgs/msg/Twist` | Velocity command. |

## Publisher

A publisher is a node that writes messages to a topic.

Example: `nano_accel_serial_node` publishes `/apex/imu/data_raw`.

## Subscriber

A subscriber is a node that receives messages from a topic.

Example: `imu_lidar_planar_fusion_node` subscribes to IMU and LiDAR topics.

## Service

A service is a request/response API. A client sends a request and receives one response.

APEX exposes `std_srvs/srv/Trigger` services for kinematics reset and static recalibration.

## Action

An action is a long-running goal interface. It is commonly used by navigation stacks for goals that take time and can report feedback.

No custom actions were found in the repository. Nav2 actions are available only when the alternate `voiture_system` Nav2 path is enabled.

## Parameter

A parameter is runtime configuration attached to a node.

Example: `rplidar_publisher_node` has parameters for serial port, baud rate, topic names, frame ID, and backend selection.

## Launch File

A launch file starts a set of nodes with parameters and conditions.

Examples:

- `src/rc_sim_description/launch/apex_sim.launch.py`
- `APEX/ros2_ws/src/apex_telemetry/launch/apex_pipeline.launch.py`
- `src/voiture_system/launch/bringup_real_slam_nav.launch.py`

## TF

TF is the ROS transform system. It describes where coordinate frames are relative to each other.

Common frames in this repository:

| Frame | Meaning |
| --- | --- |
| `base_link` | Vehicle body frame. |
| `laser` | LiDAR frame. |
| `imu_link` | IMU frame. |
| `odom` | Odometry frame. |
| `map` | Map frame. |

## `robot_state_publisher`

`robot_state_publisher` reads the robot description and publishes TF transforms for robot links. In simulation, it publishes transforms from the `rc_car.urdf.xacro` model.

## URDF

URDF is the XML robot description format used by ROS. It describes robot links, joints, sensors, and geometry.

## Xacro

Xacro is a macro language for generating URDF. This repository uses:

```text
src/rc_sim_description/urdf/rc_car.urdf.xacro
```

## Gazebo Sim

Gazebo Sim is the simulator used by the current APEX simulation stack. It simulates the vehicle, track world, sensors, and physics.

## Classic Gazebo

Classic Gazebo is the older Gazebo generation. The `voiture_system` simulation launch uses a Classic Gazebo style path and should be treated as alternate or legacy for this repository.

## `ros_gz_bridge`

`ros_gz_bridge` connects Gazebo transport topics to ROS topics.

In this repository it bridges:

- Gazebo clock to `/clock`.
- Gazebo LiDAR to `/apex/sim/scan`.
- Gazebo IMU to `/apex/sim/imu`.

## SLAM

SLAM means simultaneous localization and mapping. The alternate `voiture_system` stack can use `slam_toolbox` to publish `/map`.

APEX also contains its own IMU+LiDAR mapping and offline refinement tools.

## Nav2

Nav2 is the ROS 2 navigation stack. It is available in the alternate `voiture_system` launch path when enabled. The current APEX recognition-tour pipeline does not primarily depend on Nav2.

## Ackermann Steering

Ackermann steering is the steering geometry used by car-like vehicles. The alternate `voiture_system` stack explicitly contains Ackermann drive and odometry nodes. APEX also controls a car-like platform, but its current control path is organized around APEX tracking commands and PWM actuation.

## Sysfs PWM

Sysfs PWM is a Linux interface under `/sys/class/pwm` for controlling PWM outputs. The real APEX actuation bridge uses it to control the ESC and steering servo.

## DDS

DDS is the communication layer underneath ROS 2. It handles discovery and message transport between nodes, including nodes on different machines.

Important environment variables in this repository include:

```bash
ROS_DOMAIN_ID=30
ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

## Related Documentation

- [ROS Architecture](07_ros_architecture.md)
- [Topics, Services, Actions, and Parameters](12_topics_services_actions_parameters.md)
- [Simulation with Gazebo](08_simulation_gazebo.md)

