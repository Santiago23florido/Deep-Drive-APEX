FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
  find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -print0 \
    | xargs -0 -r sed -i \
      -e 's|http://ports.ubuntu.com/ubuntu-ports|https://ports.ubuntu.com/ubuntu-ports|g' \
      -e 's|http://packages.ros.org/ros2/ubuntu|https://packages.ros.org/ros2/ubuntu|g'; \
  find /etc/apt -type f -name '*.list' -print0 \
    | xargs -0 -r sed -i '/^[[:space:]]*deb-src[[:space:]].*packages\\.ros\\.org\\/ros2\\/ubuntu/d'; \
  find /etc/apt -type f -name '*.sources' -print0 \
    | xargs -0 -r sed -i 's/^Types:[[:space:]]*deb[[:space:]]\\+deb-src$/Types: deb/g'; \
  apt-get clean; \
  rm -rf /var/lib/apt/lists/* /var/lib/apt/lists/partial/*; \
  apt-get update -o Acquire::Retries=5 && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-pip \
    python3-serial \
    python3-venv \
    ros-jazzy-nav2-map-server \
    ros-jazzy-ros2bag \
    ros-jazzy-rosbag2-storage-default-plugins \
    ros-jazzy-rosbag2-storage-mcap \
    ros-dev-tools \
  && python3 -m venv --system-site-packages /opt/apex_venv \
  && /opt/apex_venv/bin/python -m pip install -U pip setuptools \
  && /opt/apex_venv/bin/pip install --no-cache-dir rplidar-roboticia \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work/ros2_ws
