#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


def _try_add_algorithms_path():
    this_file = Path(__file__).resolve()
    # .../src/voiture_system/voiture_system/high_level_controller_node.py
    workspace_src = this_file.parents[3]
    algo_path = workspace_src / 'rc_sim_description' / 'scripts' / 'algorithms'
    if algo_path.exists():
        sys.path.insert(0, str(algo_path))
        return True
    return False


_have_algorithms = _try_add_algorithms_path()

SimLidarReader = None
try:
    from sim_lidar_reader import SimLidarReader as ExternalSimLidarReader
except Exception:
    ExternalSimLidarReader = None

try:
    from .sim_lidar_reader import SimLidarReader as LocalSimLidarReader
except Exception:
    LocalSimLidarReader = None

if ExternalSimLidarReader is not None:
    SimLidarReader = ExternalSimLidarReader
elif LocalSimLidarReader is not None:
    SimLidarReader = LocalSimLidarReader

try:
    from interface_lidar import RPLidarReader
except Exception:
    RPLidarReader = None

try:
    from interfaces import LiDarInterface
except Exception:
    class LiDarInterface:  # Fallback for typing
        pass

try:
    from control_direction import compute_steer_from_lidar, shrink_space
    from control_speed import compute_speed
except Exception:
    def shrink_space(raw_lidar):
        return raw_lidar

    def compute_steer_from_lidar(raw_lidar):
        return 0.0, 0.0

    def compute_speed(raw_lidar, target_angle):
        return 0.0


class HighLevelControllerNode(Node):
    def __init__(self):
        super().__init__('high_level_controller_node')

        self.declare_parameter('mode', 'sim')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('wheel_radius_m', 0.06)
        self.declare_parameter('steer_deg_to_rad', True)
        self.declare_parameter('steer_scale', 1.0)

        # SimLidarReader parameters
        self.declare_parameter('heading_offset_deg', 0)
        self.declare_parameter('fov_filter', 360)
        self.declare_parameter('point_timeout_ms', 200)
        self.declare_parameter('noise_sigma', 0.0)
        self.declare_parameter('dropout_prob', 0.0)
        self.declare_parameter('bias_m', 0.0)
        self.declare_parameter('quantization_m', 0.0)

        self.declare_parameter('cmd_speed_topic', '/car/cmd_speed_wheel_omega')
        self.declare_parameter('cmd_steer_topic', '/car/cmd_steer_angle')

        self._mode = self.get_parameter('mode').get_parameter_value().string_value
        self._lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self._publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._wheel_radius_m = self.get_parameter('wheel_radius_m').get_parameter_value().double_value
        self._steer_deg_to_rad = self.get_parameter('steer_deg_to_rad').get_parameter_value().bool_value
        self._steer_scale = self.get_parameter('steer_scale').get_parameter_value().double_value

        self._cmd_speed_topic = self.get_parameter('cmd_speed_topic').get_parameter_value().string_value
        self._cmd_steer_topic = self.get_parameter('cmd_steer_topic').get_parameter_value().string_value

        self._lidar = self._create_lidar()

        self._pub_speed = self.create_publisher(Float64, self._cmd_speed_topic, 10)
        self._pub_steer = self.create_publisher(Float64, self._cmd_steer_topic, 10)

        period = 1.0 / max(self._publish_rate_hz, 1e-3)
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info('HighLevelControllerNode started in %s mode' % self._mode)

    def _create_lidar(self):
        if self._mode == 'sim':
            if SimLidarReader is None:
                self.get_logger().error('SimLidarReader not available. Check PYTHONPATH and rc_sim_description/scripts/algorithms.')
                return None
            return SimLidarReader(
                topic=self._lidar_topic,
                heading_offset_deg=self.get_parameter('heading_offset_deg').value,
                fov_filter=self.get_parameter('fov_filter').value,
                point_timeout_ms=self.get_parameter('point_timeout_ms').value,
                noise_sigma=self.get_parameter('noise_sigma').value,
                dropout_prob=self.get_parameter('dropout_prob').value,
                bias_m=self.get_parameter('bias_m').value,
                quantization_m=self.get_parameter('quantization_m').value,
                use_sim_time=True,
            )

        if self._mode == 'real':
            if RPLidarReader is None:
                self.get_logger().error('RPLidarReader not available. Missing hardware deps?')
                return None
            return RPLidarReader()

        self.get_logger().error('Unknown mode: %s' % self._mode)
        return None

    def _on_timer(self):
        if self._lidar is None:
            return

        raw = self._lidar.get_lidar_data()
        if raw is None:
            return

        raw = np.asarray(raw, dtype=float)
        if raw.shape[0] != 360:
            self.get_logger().warning('Expected lidar size 360, got %s' % (raw.shape,))
            return

        shrinked = shrink_space(raw)
        steer_cmd, target_angle = compute_steer_from_lidar(shrinked)
        speed_mps = compute_speed(shrinked, target_angle)

        if self._wheel_radius_m <= 0.0:
            omega = 0.0
        else:
            omega = speed_mps / self._wheel_radius_m

        steer_out = steer_cmd
        if self._steer_deg_to_rad:
            steer_out = math.radians(steer_out)
        steer_out *= self._steer_scale

        msg_speed = Float64()
        msg_speed.data = float(omega)
        self._pub_speed.publish(msg_speed)

        msg_steer = Float64()
        msg_steer.data = float(steer_out)
        self._pub_steer.publish(msg_steer)

    def destroy_node(self):
        try:
            if self._lidar is not None and hasattr(self._lidar, 'stop'):
                self._lidar.stop()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = HighLevelControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
