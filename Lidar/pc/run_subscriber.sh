#!/usr/bin/env bash
set -euo pipefail

# Debe coincidir con el ROS_DOMAIN_ID de la Raspberry.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"

python3 "$(dirname "$0")/lidar_subscriber_node.py" "$@"
