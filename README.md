# RC Simulation Description

Description package for a simple RC car in ROS 2 with Gazebo (gz). Instructions are tailored for WSL2 on Windows with Ubuntu 24.04 (Noble) and ROS Jazzy.

## Package layout
- `urdf/`: `rc_car.urdf.xacro` with steerable front wheels.
- `launch/`: `spawn_rc_car.launch.py` starts Gazebo (gz) and spawns the robot.
- `worlds/`: `basic_track.world` with a rectangular track and arrow marker (local ground and light, no `model://`).
- `config/rviz/`: quick RViz config (optional).
- `meshes/`: placeholder for custom meshes.

## Installation (WSL2, Ubuntu 24.04, ROS Jazzy)
1) Base tools:
```
sudo apt update
sudo apt install -y curl gnupg2 lsb-release software-properties-common \
  build-essential git python3-colcon-common-extensions python3-vcstool
```

2) Add ROS 2 Jazzy apt repo:
```
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
```

3) Install ROS desktop and runtime deps:
```
sudo apt install -y ros-jazzy-desktop \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-joint-state-publisher ros-jazzy-robot-state-publisher ros-jazzy-xacro
```

4) Install rosdep (on Noble use `python3-rosdep`), init and update:
```
sudo apt install -y python3-rosdep
sudo rosdep init        # first time only
rosdep update
```

5) (Optional) Confirm `gz` is on PATH:
```
gz --help
```

6) Build the workspace (use symlink install so changes in `urdf/` are picked up without reinstalling):
```
cd ~/AiAtonomousRc
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run
Launch Gazebo and spawn the car:
```
ros2 launch rc_sim_description spawn_rc_car.launch.py
```
Defaults: `x:=0.0`, `y:=-2.5`, `z:=0.02`.

## Control nodes (nodos de control)
This package uses ROS <-> Gazebo bridges plus control nodes:
- `rear_wheel_speed_bridge.py`: subscribes to `/rear_wheel_speed` and `/steering_angle` (Float64) and forwards them to Gazebo joints. Enabled by default in `spawn_rc_car.launch.py` (`rear_wheel_bridge:=true`).
- `gazebo_lidar_reader_node.py`: subscribes to `/scan`, applies filtering, and publishes `/lidar_processed` (LaserScan). Enabled by default (`lidar_reader:=true`).
- `voiture_control_node.py`: subscribes to `/lidar_processed` and `/measured_wheelspeed`, publishes `/rear_wheel_speed` and `/steering_angle`. Enabled by default (`control_node:=true`).
- `turning_command_mapper.py`: subscribes to `/cmd_vel` (Twist). Uses `linear.x` as vehicle speed (m/s) and `angular.z` as steering angle. By default it expects degrees (`steering_angle_unit=deg`) and publishes `/rear_wheel_speed` (rad/s) and `/steering_angle` (rad).
- `rear_wheel_speed_publisher.py`: demo publisher for `/rear_wheel_speed` and `/steering_angle`.

### Example (2 terminals)
Terminal 1 (full sim):
```
source /opt/ros/jazzy/setup.bash
cd ~/AiAtonomousRc
colcon build --packages-select rc_sim_description --symlink-install
source install/setup.bash
ros2 launch rc_sim_description spawn_rc_car.launch.py
```

Terminal 2 (camera image):
```
ros2 run image_tools showimage --ros-args -r image:=/camera/image_raw
```

Note: Do not run `rear_wheel_speed_publisher.py` at the same time as
`turning_command_mapper.py` unless you want overlapping commands.

## After changes to the Xacro/URDF
If you edit `src/rc_sim_description/urdf/rc_car.urdf.xacro`, rebuild and re-source:
```
cd ~/AiAtonomousRc
colcon build --packages-select rc_sim_description --symlink-install
source install/setup.bash
```

Useful arguments:
- `x`, `y`, `z`: initial pose (keep `z` > 0 to avoid spawn collision).
- `world`: SDF world path (defaults to `worlds/basic_track.world`).
- `rviz:=true` to open RViz (default `false`, Gazebo only).

## WSL notes
- Use WSLg or an X/Wayland server to view the Gazebo GUI; on Windows 11 with WSLg it usually works out of the box.
- If you switch ROS or Gazebo distros, adjust package names accordingly.
-Use networkingMode=mirrored
