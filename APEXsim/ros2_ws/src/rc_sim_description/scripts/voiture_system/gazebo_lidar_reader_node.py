#!/usr/bin/env python3
import argparse
import os
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.dirname(__file__))
_ALGO_DIR = os.path.join(_SCRIPTS_DIR, "algorithms")
_CANDIDATES = [
    _ALGO_DIR,
]
_here = os.path.abspath(os.path.dirname(__file__))
for _ in range(6):
    _parent = os.path.dirname(_here)
    _CANDIDATES.append(os.path.join(_parent, "scripts", "algorithms"))
    _CANDIDATES.append(
        os.path.join(_parent, "src", "rc_sim_description", "scripts", "algorithms")
    )
    _here = _parent

for _path in _CANDIDATES:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from interface_lidar import GazeboLidarReader


def _parse_args():
    parser = argparse.ArgumentParser(description="Gazebo LiDAR reader wrapper")
    parser.add_argument("--topic", default="/scan", help="ROS LaserScan topic")
    parser.add_argument(
        "--publish-topic",
        default="/lidar_processed",
        help="Topic to publish processed LaserScan (empty to disable)",
    )
    parser.add_argument(
        "--publish-frame-id",
        default="",
        help="Override frame_id for published scan (empty keeps input frame)",
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        help="Use Gazebo simulated clock",
    )
    parser.add_argument(
        "--heading-offset-deg",
        type=int,
        default=0,
        help="Heading offset in degrees",
    )
    parser.add_argument(
        "--fov-filter",
        type=int,
        default=360,
        help="Field of view filter in degrees",
    )
    parser.add_argument(
        "--point-timeout-ms",
        type=int,
        default=200,
        help="Timeout per angle in ms",
    )
    parser.add_argument(
        "--log-scans",
        action="store_true",
        help="Log scan arrays to console",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = _parse_args()
    reader = GazeboLidarReader(
        topic=args.topic,
        heading_offset_deg=args.heading_offset_deg,
        fov_filter=args.fov_filter,
        point_timeout_ms=args.point_timeout_ms,
        publish_topic=args.publish_topic,
        publish_frame_id=args.publish_frame_id,
        use_sim_time=args.use_sim_time,
        log_scans=args.log_scans,
    )
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
