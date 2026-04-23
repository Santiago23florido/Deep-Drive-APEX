#!/usr/bin/env bash
set -euo pipefail

# Ajusta este valor y usa el mismo en el PC.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"

python3 "$(dirname "$0")/lidar_publisher_node.py" "$@"
