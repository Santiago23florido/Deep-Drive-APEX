FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-pip \
    python3-serial \
    python3-venv \
    ros-jazzy-nav2-map-server \
    ros-jazzy-slam-toolbox \
    ros-dev-tools \
  && python3 -m venv --system-site-packages /opt/apex_venv \
  && /opt/apex_venv/bin/python -m pip install -U pip setuptools \
  && /opt/apex_venv/bin/pip install --no-cache-dir rplidar-roboticia \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work/ros2_ws
