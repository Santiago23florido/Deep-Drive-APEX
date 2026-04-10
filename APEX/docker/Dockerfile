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
  install_ok=0; \
  for attempt in 1 2 3 4 5; do \
    rm -rf /var/lib/apt/lists/* /var/lib/apt/lists/partial/*; \
    if apt-get update \
      -o Acquire::Retries=5 \
      -o Acquire::By-Hash=force \
      -o Acquire::http::No-Cache=true \
      -o Acquire::https::No-Cache=true \
      && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-matplotlib \
        python3-numpy \
        python3-scipy \
        python3-pip \
        python3-serial \
        python3-yaml \
        python3-venv \
        ros-jazzy-imu-filter-madgwick \
        ros-jazzy-nav2-map-server \
        ros-jazzy-robot-localization \
        ros-jazzy-ros2bag \
        ros-jazzy-rosbag2-storage-default-plugins \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-slam-toolbox; then \
      install_ok=1; \
      break; \
    fi; \
    sleep 3; \
  done; \
  test "${install_ok}" = "1"; \
  python3 -m venv --system-site-packages /opt/apex_venv \
  && /opt/apex_venv/bin/python -m pip install -U pip setuptools \
  && /opt/apex_venv/bin/pip install --no-cache-dir rplidar-roboticia \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work/ros2_ws
