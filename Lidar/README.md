# LiDAR ROS 2 Split: Raspberry Publisher + WSL/PC Subscriber

This folder is used for distributed LiDAR over Wi-Fi:

```text
Lidar/
  common/
    lidar_scan_buffer.py
  raspberry/
    lidar_publisher_node.py
    run_publisher.sh
    requirements.txt
  pc/
    lidar_subscriber_node.py
    run_subscriber.sh
    requirements.txt
  legacy/
    ...
```

## Tested Working Configuration (what actually worked)

These settings were validated end-to-end (Raspberry container -> WSL subscriber):

- `ROS_DOMAIN_ID=30`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`
- Raspberry container with `--network host`
- WSL2 with `networkingMode=mirrored`
- Windows Firewall UDP rules for DDS Domain 30 (`14900-15050`)
- No `ROS_STATIC_PEERS`

## 1) Raspberry: run LiDAR publisher in Docker

### 1.1 Create/recreate the container

```bash
docker rm -f ros2_jazzy 2>/dev/null || true

docker run -d \
  --name ros2_jazzy \
  --restart unless-stopped \
  --network host \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -e ROS_DOMAIN_ID=30 \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v ~/Voiture-Autonome/code:/work/code \
  -w /work/code \
  ros:jazzy-ros-base-noble \
  sleep infinity
```

### 1.2 One-time Python setup inside the container

```bash
docker exec -it ros2_jazzy bash -lc '
apt update
apt install -y python3-venv python3-numpy python3-serial python3-yaml
python3 -m venv --system-site-packages /opt/lidar_venv
/opt/lidar_venv/bin/python -m pip install -U pip setuptools
/opt/lidar_venv/bin/pip install --no-cache-dir --retries 8 --timeout 120 rplidar-roboticia
'
```

### 1.3 Start the LiDAR publisher

Use the baudrate that works with your sensor (`115200` or `256000`):

```bash
docker exec -it ros2_jazzy bash -ic '
source /opt/ros/jazzy/setup.bash
source /opt/lidar_venv/bin/activate
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /work/code
Lidar/raspberry/run_publisher.sh --port /dev/ttyUSB0 --baudrate 115200 --topic /lidar/scan
'
```

### 1.4 Validate on Raspberry

```bash
docker exec -it ros2_jazzy bash -ic '
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=30
ros2 topic hz /lidar/scan
'
```

If you see a stable rate (for example ~13 Hz), publishing is correct.

## 2) WSL: configure network and receive LiDAR

### 2.1 Enable mirrored networking (one-time)

In `C:\Users\<your_user>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then apply:

```powershell
wsl --shutdown
```

### 2.2 Open Windows firewall for ROS 2 Domain 30 (one-time)

Run in PowerShell as Administrator:

```powershell
Get-NetFirewallRule -DisplayName "ROS2 DDS*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule

New-NetFirewallRule -DisplayName "ROS2 DDS D30 In UDP" `
  -Direction Inbound -Action Allow -Protocol UDP -LocalPort 14900-15050

New-NetFirewallRule -DisplayName "ROS2 DDS D30 Out UDP" `
  -Direction Outbound -Action Allow -Protocol UDP -RemotePort 14900-15050

wsl --shutdown
```

### 2.3 Prepare WSL Python environment (PEP668-safe)

```bash
cd ~/AiAtonomousRc
python3 -m venv .venv_lidar
source .venv_lidar/bin/activate
python -m pip install -U pip
python -m pip install -r Lidar/pc/requirements.txt
```

`.venv_lidar/` is a local virtual environment. It is intentionally ignored by Git and should be recreated on each developer machine instead of committed.

### 2.4 Start subscriber (console summary)

```bash
cd ~/AiAtonomousRc
source .venv_lidar/bin/activate
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 daemon stop
Lidar/pc/run_subscriber.sh --topic /lidar/scan
```

## 3) Live point cloud on PC (new)

The subscriber now supports a live XY point-cloud window:

```bash
cd ~/AiAtonomousRc
source .venv_lidar/bin/activate
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
Lidar/pc/run_subscriber.sh --topic /lidar/scan --plot
```

Useful plot options:

```bash
Lidar/pc/run_subscriber.sh --topic /lidar/scan --plot --plot-max-range 4.0 --plot-every-s 0.08
```

Also available:

```bash
Lidar/pc/run_subscriber.sh --topic /lidar/scan --full
```

## 4) Mapping on PC using LiDAR + Ackermann odometry

The PC subscriber supports occupancy-grid mapping with pose from `/odom`
(recommended for real car kinematics):

```bash
cd ~/AiAtonomousRc
source .venv_lidar/bin/activate
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=30
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
Lidar/pc/run_subscriber.sh --topic /lidar/scan --map --map-plot \
  --map-pose-source odom --odom-topic /odom
```

Useful mapping options:

```bash
Lidar/pc/run_subscriber.sh --topic /lidar/scan --map --map-plot \
  --map-resolution 0.05 --map-size-m 30 \
  --map-pose-source odom --odom-topic /odom \
  --map-save-path /tmp/lidar_map.npy
```

Optional fallback (if `/odom` is unavailable): use scan matching only

```bash
Lidar/pc/run_subscriber.sh --topic /lidar/scan --map --map-plot \
  --map-pose-source scanmatch
```

`--map-save-path` stores:
- probability grid as `.npy`
- basic metadata in `<file>.meta.txt`

## 5) Quick troubleshooting

- `ModuleNotFoundError: yaml` on PC: install deps inside `.venv_lidar` with `pip install -r Lidar/pc/requirements.txt`.
- `externally-managed-environment`: do not use global pip, use `.venv_lidar`.
- `Unable to resolve peer 192.168.1.X`: remove placeholder values from `ROS_STATIC_PEERS`.
- Serial errors on Raspberry (`Descriptor...`, `Check bit...`): verify port, baudrate, power, and that no other process is using `/dev/ttyUSB0`.

Publisher tuning parameters:

- `--heading-offset-deg`
- `--fov-filter-deg`
- `--point-timeout-ms`
- `--range-min`
- `--range-max`

Defaults are aligned with the historical external Python stack that was formerly mirrored as `full_soft/`:
- `--heading-offset-deg -89`
- `--fov-filter-deg 180`
- `--point-timeout-ms 1000`

## 6) Raspberry real stack (Jazzy): LiDAR + ESC + Servo + Ackermann odom + SLAM + Nav2

This repository now includes a real ROS2 stack in `src/voiture_system` that keeps the same
hardware structure documented by the historical external Python reference formerly mirrored as `full_soft/`:

- PWM overlay: `dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4`
- ESC (motor) on PWM channel `0` (GPIO12)
- Steering servo on PWM channel `1` (GPIO13)
- ESC reverse sequence preserved (brake -> neutral -> reverse)

### 6.1 One-time system packages on Raspberry

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-slam-toolbox \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nav2-regulated-pure-pursuit-controller \
  python3-serial
```

Enable PWM overlay in `/boot/firmware/config.txt`:

```ini
dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
```

Reboot after changing overlay.

### 6.2 Build workspace

```bash
cd ~/AiAtonomousRc
source /opt/ros/jazzy/setup.bash
colcon build --packages-select voiture_system --symlink-install
source install/setup.bash
```

### 6.3 Launch full real stack

```bash
ros2 launch voiture_system bringup_real_slam_nav.launch.py \
  lidar_port:=/dev/ttyUSB0 lidar_baudrate:=256000 \
  arduino_port:=/dev/ttyACM0 arduino_baudrate:=115200
```

SLAM-only mode (without Nav2 controller stack):

```bash
ros2 launch voiture_system bringup_real_slam_nav.launch.py use_nav2:=false
```

Autonomous track mode with a visible speed cap for tests:

```bash
ros2 launch voiture_system bringup_real_slam_nav.launch.py \
  use_auto_track:=true use_nav2:=false \
  speed_limit_pct:=30.0
```

`speed_limit_pct` is the easiest test knob:
- `30` = very conservative
- `50` = medium
- `70+` = aggressive

You can also change it live during tests (no relaunch required):

```bash
ros2 param set /ackermann_drive_node speed_limit_pct 30.0
ros2 param set /adaptive_track_controller_node speed_limit_pct 30.0
```

Default Ackermann geometry configured in the launch:

- `wheelbase_m = 0.30` (from 0.15 m COM->rear and symmetric wheelbase assumption)
- `rear_axle_to_com_m = 0.15`
- `front_half_track_m = 0.05` (front wheels ±5 cm from centerline)

### 6.4 High-level position control

The stack ignores the legacy movement algorithm and uses ROS2 standard components:

- `slam_toolbox` for online mapping/localization (`map -> odom`)
- `nav2` (planner + controller) for goal-based position control via `/cmd_vel`
- `ackermann_drive_node` converts `/cmd_vel` to real ESC + steering PWM commands
- `adaptive_track_controller_node` can generate `/cmd_vel` directly from LiDAR + map progress
  with a single continuous algorithm (reactive at start, more anticipative as map quality improves)

You can send goals from RViz2 (`Nav2 Goal`) once the stack is active.
