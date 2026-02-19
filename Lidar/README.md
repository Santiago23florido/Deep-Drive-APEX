# LiDAR ROS 2 Split: Raspberry (Publisher) + PC/WSL (Subscriber)

This folder is organized for distributed deployment over Wi-Fi:

```text
Lidar/
  common/
    lidar_scan_buffer.py      # Shared filtering and 360-degree mapping logic
  raspberry/
    lidar_publisher_node.py   # ROS 2 node that reads RPLidar and publishes /lidar/scan
    run_publisher.sh
    requirements.txt
  pc/
    lidar_subscriber_node.py  # ROS 2 node that subscribes and prints measurements
    run_subscriber.sh
    requirements.txt
  legacy/
    ...                       # Previous code kept for reference
```

## 1) Raspberry publisher (native ROS 2)

Requirements:
- ROS 2 installed (same distro as PC/WSL).
- LiDAR connected (for example `/dev/ttyUSB0`).
- Python dependencies (`numpy`, `rplidar-roboticia`).

Commands:

```bash
cd /home/santiago/AiAtonomousRc
python3 -m pip install -r Lidar/raspberry/requirements.txt

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

Lidar/raspberry/run_publisher.sh --port /dev/ttyUSB0 --baudrate 256000 --topic /lidar/scan
```

## 2) PC subscriber (native ROS 2)

Requirements:
- ROS 2 installed (same distro as Raspberry).
- Same Wi-Fi/LAN subnet.

Commands:

```bash
cd /home/santiago/AiAtonomousRc
python3 -m pip install -r Lidar/pc/requirements.txt

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

Lidar/pc/run_subscriber.sh --topic /lidar/scan
Lidar/pc/run_subscriber.sh --topic /lidar/scan --full
```

## 3) Full setup used to solve Step 2 with Docker on Raspberry and WSL on Windows

This section explains exactly how the container was created/initialized and how WSL was adjusted so Step 2 can receive topics.

### 3.1 Create and initialize the ROS 2 Jazzy container on Raspberry

If your project lives in `~/Voiture-Autonome/code`, keep that path mounted in the container.

```bash
mkdir -p ~/ros2_ws

docker pull ros:jazzy-ros-base-noble
docker rm -f ros2_jazzy 2>/dev/null || true

docker run -d \
  --name ros2_jazzy \
  --restart unless-stopped \
  --network host \
  --device=/dev/ttyUSB0:/dev/ttyUSB0 \
  -e ROS_DOMAIN_ID=42 \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v "$HOME/Voiture-Autonome:/work/Voiture-Autonome" \
  -w /work/Voiture-Autonome/code \
  ros:jazzy-ros-base-noble \
  sleep infinity
```

Initialize container shell environment:

```bash
docker exec ros2_jazzy bash -lc "grep -qxF 'source /opt/ros/jazzy/setup.bash' /root/.bashrc || echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc"
docker exec ros2_jazzy bash -lc "grep -qxF 'export ROS_DOMAIN_ID=42' /root/.bashrc || echo 'export ROS_DOMAIN_ID=42' >> /root/.bashrc"
docker exec ros2_jazzy bash -lc "grep -qxF 'export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET' /root/.bashrc || echo 'export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET' >> /root/.bashrc"
docker exec ros2_jazzy bash -lc "grep -qxF 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' /root/.bashrc || echo 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' >> /root/.bashrc"
```

Install Python dependencies for the Raspberry publisher inside the container:

```bash
docker exec -it ros2_jazzy bash -lc "apt update && apt install -y python3-pip"
docker exec -it ros2_jazzy bash -lc "python3 -m pip install -r /work/Voiture-Autonome/code/Lidar/raspberry/requirements.txt"
```

Start the LiDAR publisher from the container:

```bash
docker exec -it ros2_jazzy bash -ic "cd /work/Voiture-Autonome/code && Lidar/raspberry/run_publisher.sh --port /dev/ttyUSB0 --baudrate 256000 --topic /lidar/scan"
```

### 3.2 WSL adjustments to receive topics from Raspberry

1. Enable mirrored networking for WSL (`C:\Users\<you>\.wslconfig`):

```ini
[wsl2]
networkingMode=mirrored
```

2. Apply changes from PowerShell:

```powershell
wsl --shutdown
```

3. Open WSL and verify it has a LAN IP in the same subnet as Raspberry:

```bash
hostname -I
```

4. In WSL, clear stale vars and set ROS network vars:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_STATIC_PEERS FASTRTPS_DEFAULT_PROFILES_FILE ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

5. In Windows PowerShell (Admin), open DDS UDP ports for domain 42:
- DDS base for domain 42 is around `17900`.

```powershell
New-NetFirewallRule -DisplayName "ROS2 DDS D42 In UDP" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 17900-18050
New-NetFirewallRule -DisplayName "ROS2 DDS D42 Out UDP" -Direction Outbound -Action Allow -Protocol UDP -RemotePort 17900-18050
```

6. Run Step 2 subscriber from WSL:

```bash
cd ~/AiAtonomousRc
python3 -m pip install -r Lidar/pc/requirements.txt
ros2 daemon stop
Lidar/pc/run_subscriber.sh --topic /lidar/scan
```

### 3.3 Common issue seen during setup

If you get `Unable to resolve peer 192.168.1.X`, remove placeholder/static peer values:

```bash
unset ROS_STATIC_PEERS
sed -i '/ROS_STATIC_PEERS/d' ~/.bashrc
```

Do this both in WSL and in the Raspberry container.

## 4) Connectivity checks

In WSL/PC:

```bash
ros2 topic list | grep lidar
ros2 topic echo /lidar/scan --once
```

If topic discovery fails:
- Check same `ROS_DOMAIN_ID` on both sides.
- Check same `RMW_IMPLEMENTATION` on both sides.
- Ensure Raspberry and WSL LAN IPs are in the same subnet.
- Confirm container network mode is `host`.
- Confirm Windows firewall rules for DDS UDP are in place.

## 5) Important publisher parameters

- `--heading-offset-deg`: compensates LiDAR orientation.
- `--fov-filter-deg`: limits field of view (0-360).
- `--point-timeout-ms`: invalidates stale angles.
- `--range-min` and `--range-max`: LaserScan output limits.
