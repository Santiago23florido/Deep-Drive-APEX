# Installation on Linux

## Expected Environment

The repository is written for ROS 2 and is best used on Linux. The expected modern environment is:

- Ubuntu 24.04.
- ROS 2 Jazzy.
- Gazebo Sim with `ros_gz_sim` and `ros_gz_bridge`.
- `colcon` for workspace builds.
- Docker and Docker Compose for the real APEX blue-car deployment.

The Raspberry Pi real-car deployment uses a ROS 2 Jazzy Docker image rather than relying only on packages installed on the host.

## Prerequisites

Install core tools:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  curl \
  git \
  gnupg2 \
  lsb-release \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  software-properties-common
```

Install ROS 2 Jazzy using the official ROS 2 Ubuntu instructions for your machine. After ROS is installed, source it:

```bash
source /opt/ros/jazzy/setup.bash
```

Initialize `rosdep` if this machine has not used ROS before:

```bash
sudo rosdep init
rosdep update
```

If `sudo rosdep init` reports that the file already exists, run only `rosdep update`.

## ROS and Gazebo Dependencies

Install the dependencies used by the repository:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-rviz2 \
  ros-jazzy-slam-toolbox \
  ros-jazzy-robot-localization \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  python3-numpy \
  python3-scipy \
  python3-serial \
  python3-yaml \
  python3-pygame
```

Optional packages may be needed for specific simulation modes, such as `rf2o_laser_odometry`. Availability depends on your ROS 2 Jazzy package sources. The default APEX simulation path does not require every optional estimator.

## Clone and Build

From the repository root:

```bash
cd ~/AiAtonomousRc
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src APEX/ros2_ws/src --ignore-src -r -y
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
```

Use your actual clone path instead of `~/AiAtonomousRc`.

## Validate the ROS Workspace

Check that ROS can see the packages:

```bash
ros2 pkg list | grep -E 'rc_sim_description|apex_telemetry|voiture_system'
```

Check that the main launch files are available:

```bash
ros2 launch rc_sim_description apex_sim.launch.py --show-args
ros2 launch apex_telemetry apex_pipeline.launch.py --show-args
ros2 launch voiture_system bringup_real_slam_nav.launch.py --show-args
```

If these commands fail after a successful build, source both ROS and the workspace:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Run a Minimal Simulation

The recommended simulation wrapper is:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

You can also launch directly:

```bash
ros2 launch rc_sim_description apex_sim.launch.py scenario:=baseline rviz:=true
```

See [Quick Start](05_quick_start.md) and [Simulation with Gazebo](08_simulation_gazebo.md).

## Docker for the Real Blue Car

Install Docker on the Raspberry Pi or Linux host used for real-car deployment:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and back in so the Docker group membership applies.

The main real-car service is `APEX/docker/docker-compose.yml`. It runs a privileged container with:

- Host networking.
- `/dev/ttyACM0` for the Nano IMU by default.
- `/dev/ttyUSB0` for the RPLIDAR by default.
- `/sys/class/pwm` mounted for ESC and steering servo control.

Build and start through the APEX scripts rather than calling Docker Compose manually:

```bash
cd APEX
./tools/core/apex_real_ready_up.sh
```

Use the command only when the vehicle is physically safe to move. See [Blue Vehicle Real System](09_blue_vehicle_real_system.md).

## Common Linux Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ros2: command not found` | ROS is not installed or not sourced. | `source /opt/ros/jazzy/setup.bash`. |
| `colcon: command not found` | `python3-colcon-common-extensions` missing. | Install the package and reopen the shell. |
| Package not found after build | Workspace not sourced. | `source install/setup.bash`. |
| Gazebo opens but vehicle is missing | Workspace install path or model path not sourced. | Rebuild and source `install/setup.bash`. |
| `/dev/ttyUSB0` permission denied | User lacks serial access. | Add user to `dialout`, reconnect device, or run container with the expected device mapping. |
| PWM writes fail | `/sys/class/pwm` unavailable or wrong permissions. | Check Raspberry Pi PWM overlay and container privileged mode. |
| DDS topics do not cross machines | ROS domain, discovery range, or firewall mismatch. | Check `ROS_DOMAIN_ID`, `ROS_AUTOMATIC_DISCOVERY_RANGE`, and network firewall rules. |

## Related Documentation

- [Installation on Windows](04_installation_windows.md)
- [Quick Start](05_quick_start.md)
- [Hardware Interfaces](19_hardware_interfaces.md)
- [Troubleshooting](15_troubleshooting.md)

