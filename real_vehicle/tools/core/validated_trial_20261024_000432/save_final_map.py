"""Capture the actual latched SLAM grid after disarming, before teardown."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rclpy.init()
node = rclpy.create_node('final_map_evidence')
grid = None
graph = None

def got_grid(msg):
    global grid
    grid = msg

def got_graph(msg):
    global graph
    graph = [(m.id, m.pose.position.x, m.pose.position.y) for m in msg.markers
             if m.type == Marker.SPHERE and m.action == Marker.ADD]

qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
node.create_subscription(OccupancyGrid, '/map', got_grid, qos)
node.create_subscription(MarkerArray, '/slam_toolbox/graph_visualization', got_graph, 1)
deadline = time.monotonic() + 4
while time.monotonic() < deadline and (grid is None or graph is None):
    rclpy.spin_once(node, timeout_sec=.1)
assert grid is not None, 'no actual SLAM grid received before teardown'
meta = {'frame_id': grid.header.frame_id, 'stamp_sec': grid.header.stamp.sec,
        'stamp_nanosec': grid.header.stamp.nanosec, 'width': grid.info.width,
        'height': grid.info.height, 'resolution': grid.info.resolution,
        'origin': {'x': grid.info.origin.position.x, 'y': grid.info.origin.position.y,
                   'qx': grid.info.origin.orientation.x, 'qy': grid.info.origin.orientation.y,
                   'qz': grid.info.origin.orientation.z, 'qw': grid.info.origin.orientation.w},
        'graph_vertices': 0 if graph is None else len(graph)}
np.savez_compressed(args.output / 'slam_map_final.npz',
                    occupancy=np.asarray(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width),
                    metadata=json.dumps(meta), graph=np.asarray(graph or [], dtype=float).reshape(-1, 3))
(args.output / 'slam_map_final.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta))
node.destroy_node()
rclpy.shutdown()
