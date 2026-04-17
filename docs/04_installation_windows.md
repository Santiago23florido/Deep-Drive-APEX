# Installation on Windows

## Support Level

Native Windows support is incomplete for this repository. The current codebase is a ROS 2 and Gazebo-oriented Linux project, and several important workflows depend on Linux-only interfaces:

- Gazebo Sim and ROS 2 Jazzy launch workflows.
- Docker Compose deployment on the Raspberry Pi.
- Serial devices presented as Linux device files.
- Raspberry Pi sysfs PWM under `/sys/class/pwm`.
- Shell scripts written for Bash.

The recommended Windows setup is **WSL2 with Ubuntu 24.04 and ROS 2 Jazzy**.

During repository audit on the current Windows host, `ros2` and `colcon` were not available in PowerShell. Build and runtime validation should therefore be performed inside Linux, WSL2, or on the Raspberry Pi.

## Recommended Windows Path: WSL2

Install WSL2 and Ubuntu 24.04:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if requested, then open the Ubuntu terminal.

Inside WSL2, follow [Installation on Linux](03_installation_linux.md):

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep git
source /opt/ros/jazzy/setup.bash
cd ~/AiAtonomousRc
rosdep install --from-paths src APEX/ros2_ws/src --ignore-src -r -y
colcon build --symlink-install --base-paths src APEX/ros2_ws/src --packages-select rc_sim_description apex_telemetry voiture_system
source install/setup.bash
```

## Gazebo and RViz on WSL2

For GUI applications such as Gazebo and RViz:

- Use Windows 11 with WSLg when possible.
- Keep the repository inside the WSL filesystem for better performance, for example under `~/AiAtonomousRc`.
- Avoid building from `/mnt/c/...` if performance or file watching becomes unreliable.

Run a simulation from WSL2:

```bash
./APEX/tools/sim/apex_sim_up.sh --scenario baseline --rviz
```

## Windows Gamepad and PC-Side Tools

The repository includes Windows/PC helper tooling under:

```text
APEX/tools/windows/
APEX/tools/pc/
```

These tools support manual control and session interaction from a PC, but the ROS 2 autonomy pipeline remains Linux-based. The PC bridge and Raspberry Pi must agree on DDS/network settings, especially:

```bash
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

On Windows, allow the relevant UDP traffic through the firewall. Existing LiDAR networking notes in `Lidar/README.md` mention UDP discovery ranges around 14900-15050 for the tested setup.

## Real-Car Deployment from Windows

The practical real-car workflow from a Windows development machine is:

1. Use Windows or WSL2 as the editing and monitoring environment.
2. SSH into the Raspberry Pi mounted on the blue car.
3. Run the APEX Docker Compose scripts on the Raspberry Pi.
4. Optionally run PC-side bridge or watch scripts for manual control and run retrieval.

Example:

```bash
ssh ensta@raspberrypi
cd /home/ensta/AiAtonomousRc/APEX
./tools/core/apex_real_ready_up.sh
```

Adjust hostnames and paths to your installation.

## Native PowerShell Caveats

PowerShell is useful for inspecting files, editing documentation, and using Git. It is not the recommended runtime shell for this ROS 2 repository.

Avoid trying to run these directly in native PowerShell unless you have separately installed and validated native ROS 2 tools:

```powershell
ros2 launch rc_sim_description apex_sim.launch.py
colcon build
./APEX/tools/sim/apex_sim_up.sh
```

Use WSL2 or Linux for those commands.

## Related Documentation

- [Installation on Linux](03_installation_linux.md)
- [Quick Start](05_quick_start.md)
- [Hardware Interfaces](19_hardware_interfaces.md)
- [Troubleshooting](15_troubleshooting.md)

