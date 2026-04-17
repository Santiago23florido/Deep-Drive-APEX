# Topics, Services, Actions, and Parameters

This file is a practical ROS interface reference. Interfaces are marked as:

- **Confirmed:** found in launch files, setup files, configuration, or node source.
- **Inferred:** inferred from code relationships or launch composition.
- **Optional:** present only when a launch argument or feature flag enables a component.

## APEX Sensor Topics

| Topic | Type | Publisher | Subscriber | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `/apex/imu/data_raw` | `sensor_msgs/msg/Imu` | `nano_accel_serial_node` or simulation backend | Kinematics and fusion nodes | Raw IMU message. | Confirmed |
| `/apex/imu/acceleration/raw` | `geometry_msgs/msg/Vector3Stamped` | `nano_accel_serial_node` | Kinematics estimator | Raw acceleration vector. | Confirmed |
| `/apex/imu/angular_velocity/raw` | `geometry_msgs/msg/Vector3Stamped` | `nano_accel_serial_node` | Kinematics estimator | Raw angular velocity vector. | Confirmed |
| `/lidar/scan` | `sensor_msgs/msg/LaserScan` | APEX RPLIDAR publisher | Visualization, optional consumers | General LiDAR scan. | Confirmed |
| `/lidar/scan_localization` | `sensor_msgs/msg/LaserScan` | APEX RPLIDAR publisher | Fusion, planners | Localization scan used by APEX. | Confirmed |
| `/lidar/scan_slam` | `sensor_msgs/msg/LaserScan` | APEX RPLIDAR publisher | SLAM workflows | SLAM-oriented scan topic. | Confirmed in code |
| `/apex/sim/scan` | `sensor_msgs/msg/LaserScan` | `ros_gz_bridge` | APEX LiDAR node in sim mode | Simulated LiDAR scan from Gazebo. | Confirmed |
| `/apex/sim/imu` | `sensor_msgs/msg/Imu` | `ros_gz_bridge` | APEX IMU node in sim mode | Simulated IMU from Gazebo. | Confirmed |

## APEX Estimation Topics

| Topic | Type | Publisher | Subscriber | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `/apex/kinematics/acceleration` | `geometry_msgs/msg/Vector3Stamped` | `kinematics_estimator_node` | Diagnostics, odometry | Estimated acceleration. | Confirmed |
| `/apex/kinematics/velocity` | `geometry_msgs/msg/Vector3Stamped` | `kinematics_estimator_node` | Kinematics odometry | Estimated velocity. | Confirmed |
| `/apex/kinematics/position` | `geometry_msgs/msg/PointStamped` or equivalent stamped vector | `kinematics_estimator_node` | Kinematics odometry | Estimated position. | Inferred type |
| `/apex/kinematics/angular_velocity` | `geometry_msgs/msg/Vector3Stamped` | `kinematics_estimator_node` | Diagnostics | Estimated angular velocity. | Confirmed |
| `/apex/kinematics/heading` | `std_msgs/msg/Float64` or equivalent scalar | `kinematics_estimator_node` | Kinematics odometry | Estimated heading. | Inferred type |
| `/apex/kinematics/status` | `std_msgs/msg/String` | `kinematics_estimator_node` | Session manager, diagnostics | Kinematics health/status. | Confirmed |
| `/apex/imu/data_corrected` | `sensor_msgs/msg/Imu` | `kinematics_estimator_node` | Optional consumers | Corrected IMU data. | Confirmed |
| `/apex/odometry/imu_raw` | `nav_msgs/msg/Odometry` | `kinematics_odometry_node` | Fusion or diagnostics | IMU-derived odometry. | Confirmed |
| `/apex/odometry/imu_lidar_fused` | `nav_msgs/msg/Odometry` | `imu_lidar_planar_fusion_node` | Planners, trackers, recorders | Fused planar odometry. | Confirmed |
| `/apex/estimation/path` | `nav_msgs/msg/Path` | `imu_lidar_planar_fusion_node` | Visualization, recorders | Estimated path. | Confirmed |
| `/apex/estimation/current_pose` | `geometry_msgs/msg/PoseStamped` | `imu_lidar_planar_fusion_node` | Planners, visualization | Current fused pose. | Confirmed |
| `/apex/estimation/live_map_points` | `sensor_msgs/msg/PointCloud2` | `imu_lidar_planar_fusion_node` | Visualization, map tools | Live map points. | Confirmed |
| `/apex/estimation/full_map_points` | `sensor_msgs/msg/PointCloud2` | `imu_lidar_planar_fusion_node` | Visualization, map tools | Full accumulated map points. | Confirmed |
| `/apex/estimation/status` | `std_msgs/msg/String` | `imu_lidar_planar_fusion_node` | Planners, trackers, session manager | Fusion health/status. | Confirmed |

## APEX Planning and Tracking Topics

| Topic | Type | Publisher | Subscriber | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `/apex/planning/recognition_tour_route` | `nav_msgs/msg/Path` | `recognition_tour_planner_node` | Visualization, recorders | Global or route-level recognition-tour path. | Confirmed |
| `/apex/planning/recognition_tour_local_path` | `nav_msgs/msg/Path` | `recognition_tour_planner_node` | `recognition_tour_tracker_node` | Local path to track. | Confirmed |
| `/apex/planning/recognition_tour_status` | `std_msgs/msg/String` | `recognition_tour_planner_node` | Tracker, session manager, diagnostics | Planner status. | Confirmed |
| `/apex/tracking/recognition_tour_status` | `std_msgs/msg/String` | `recognition_tour_tracker_node` | Session manager, diagnostics | Tracker status. | Confirmed |
| `/apex/tracking/arm` | `std_msgs/msg/Bool` | Capture scripts or manual/session tools | Trackers | Arms or disarms tracking. | Confirmed |
| `/apex/planning/curve_entry_path` | `nav_msgs/msg/Path` | `curve_entry_path_planner_node` | `curve_path_tracker_node` | Curve-entry path. | Confirmed |
| `/apex/planning/curve_entry_status` | `std_msgs/msg/String` | `curve_entry_path_planner_node` | Tracker, diagnostics | Curve planner status. | Confirmed |
| `/apex/tracking/status` | `std_msgs/msg/String` | `curve_path_tracker_node` | Diagnostics | Curve tracker status. | Confirmed |

## APEX Actuation Topics

| Topic | Type | Publisher | Subscriber | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `/apex/cmd_vel_track` | `geometry_msgs/msg/Twist` | APEX trackers | `cmd_vel_to_apex_actuation_node` | Main APEX velocity command. | Confirmed |
| `/apex/vehicle/drive_bridge_status` | `std_msgs/msg/String` | `cmd_vel_to_apex_actuation_node` | Session manager, recorders | Actuation bridge state. | Confirmed |
| `/apex/vehicle/applied_speed_pct` | `std_msgs/msg/Float64` | `cmd_vel_to_apex_actuation_node` | Diagnostics | Clamped applied speed percentage. | Confirmed |
| `/apex/vehicle/applied_steering_deg` | `std_msgs/msg/Float64` | `cmd_vel_to_apex_actuation_node` | Diagnostics | Clamped applied steering angle. | Confirmed |
| `/apex/sim/pwm/motor_dc` | `std_msgs/msg/Float64` | `cmd_vel_to_apex_actuation_node` in sim mode | `apex_gz_vehicle_bridge.py` | Simulated motor duty cycle. | Confirmed |
| `/apex/sim/pwm/steering_dc` | `std_msgs/msg/Float64` | `cmd_vel_to_apex_actuation_node` in sim mode | `apex_gz_vehicle_bridge.py` | Simulated steering duty cycle. | Confirmed |

## Simulation-Specific Topics

| Topic | Type | Publisher | Purpose | Status |
| --- | --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/msg/Clock` | `ros_gz_bridge` | Simulation time. | Confirmed |
| `/apex/sim/ground_truth/odom` | `nav_msgs/msg/Odometry` | `apex_ground_truth_node.py` | Ground-truth odometry. | Confirmed |
| `/apex/sim/ground_truth/path` | `nav_msgs/msg/Path` | `apex_ground_truth_node.py` | Ground-truth path. | Confirmed |
| `/apex/sim/ground_truth/perfect_map_points` | `sensor_msgs/msg/PointCloud` | `apex_ground_truth_node.py` | Perfect map points from simulation. | Confirmed |
| `/apex/sim/ground_truth/status` | `std_msgs/msg/String` | `apex_ground_truth_node.py` | Ground-truth status. | Confirmed |
| `/apex/sim/ideal_odom` | `nav_msgs/msg/Odometry` | Simulation tools | Ideal odometry for selected modes. | Confirmed |

## Alternate `voiture_system` Topics

| Topic | Type | Publisher | Subscriber | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Nav2, adaptive controller, or manual source | `ackermann_drive_node` | Standard velocity command. | Confirmed |
| `/odom` | `nav_msgs/msg/Odometry` | `ackermann_odometry_node` | SLAM/Nav2/RViz | Vehicle odometry. | Confirmed |
| `/map` | `nav_msgs/msg/OccupancyGrid` | `slam_toolbox` | Nav2/RViz | SLAM map. | Optional |
| `/lidar/scan` | `sensor_msgs/msg/LaserScan` | `voiture_system` RPLIDAR publisher | SLAM, adaptive controller | Real LiDAR scan. | Confirmed |
| `/vehicle/speed_mps` | `std_msgs/msg/Float64` | `serial_state_node` | `ackermann_odometry_node` | Vehicle speed measurement. | Confirmed |
| `/vehicle/steering_angle_cmd_rad` | `std_msgs/msg/Float64` | `ackermann_drive_node` | `ackermann_odometry_node`, diagnostics | Commanded steering angle. | Confirmed |
| `/vehicle/motor_speed_cmd` | `std_msgs/msg/Float64` | `ackermann_drive_node` | Diagnostics | Motor command. | Confirmed |
| `/rear_wheel_speed` | `std_msgs/msg/Float64` | `ackermann_drive_node` or hardware state | Diagnostics/odometry | Rear wheel speed. | Confirmed |
| `/steering_angle` | `std_msgs/msg/Float64` | `ackermann_drive_node` or hardware state | Diagnostics | Steering angle. | Confirmed |

## Services

Confirmed APEX services:

| Service | Type | Provider | Purpose | Notes |
| --- | --- | --- | --- | --- |
| `reset_kinematics` | `std_srvs/srv/Trigger` | `kinematics_estimator_node` | Reset the kinematics estimator. | Verify fully resolved name with `ros2 service list`. |
| `recalibrate_kinematics_static` | `std_srvs/srv/Trigger` | `kinematics_estimator_node` | Recalibrate static IMU/kinematics state. | Verify fully resolved name with `ros2 service list`. |

Example:

```bash
ros2 service list | grep kinematics
ros2 service call /reset_kinematics std_srvs/srv/Trigger {}
```

If the node is launched in a namespace, the service names may be namespaced.

## Actions

No custom ROS actions were found in the repository packages.

Nav2 actions are available only when the alternate `voiture_system` Nav2 launch path is enabled. Those actions are provided by Nav2, not by custom repository code.

## Main Parameter Files

| File | Applies to | Purpose |
| --- | --- | --- |
| `APEX/ros2_ws/src/apex_telemetry/config/apex_params.yaml` | APEX real and simulated pipeline | Main APEX parameter reference. |
| `src/rc_sim_description/config/apex_sim_scenarios.json` | APEX simulation | Scenario, world, spawn, vehicle bridge, and sensor simulation settings. |
| `src/voiture_system/config/controllers.yaml` | Alternate simulation/control | `ros2_control` controller settings. |
| `src/voiture_system/config/slam_toolbox_online_async.yaml` | Alternate SLAM | SLAM toolbox settings. |
| `src/voiture_system/config/nav2_ackermann.yaml` | Alternate Nav2 | Nav2 configuration. |

## Important APEX Parameters

| Node | Parameter group | Examples |
| --- | --- | --- |
| `nano_accel_serial_node` | IMU source | `serial_port`, `baud_rate`, `imu_topic`, `frame_id`, `transport_backend`, `sim_imu_topic` |
| `rplidar_publisher_node` | LiDAR source | `serial_port`, `baud_rate`, `scan_topic`, `localization_scan_topic`, `frame_id`, `source_backend`, `sim_scan_topic` |
| `kinematics_estimator_node` | Kinematics | Raw input topics, corrected IMU topic, status topic, integration settings |
| `imu_lidar_planar_fusion_node` | Fusion | IMU topic, scan topic, odometry topic, map topics, TF settings |
| `recognition_tour_planner_node` | Planning | Scan topic, odometry topic, local path topic, route topic, status topic, planning rate and horizon |
| `recognition_tour_tracker_node` | Tracking | Local path topic, status topic, arm topic, command topic, stale-data limits |
| `cmd_vel_to_apex_actuation_node` | Actuation | `actuation_backend`, PWM channels, duty cycles, speed/steering limits, simulated PWM topics |

## Related Documentation

- [ROS Architecture](07_ros_architecture.md)
- [Configuration Reference](17_configuration_reference.md)
- [Hardware Interfaces](19_hardware_interfaces.md)

