#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import LaserScan


class TestPipeline(Node):
    def __init__(self):
        super().__init__('test_pipeline')
        self._got_scan = False
        self._got_rear = False
        self._got_front = False

        self._pub_speed = self.create_publisher(Float64, '/car/cmd_speed_wheel_omega', 10)
        self._pub_steer = self.create_publisher(Float64, '/car/cmd_steer_angle', 10)

        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, '/rear_wheels_velocity_controller/commands', self._on_rear, 10)
        self.create_subscription(Float64MultiArray, '/front_steer_position_controller/commands', self._on_front, 10)

        self._start_time = time.time()
        self._timer = self.create_timer(0.2, self._tick)

    def _on_scan(self, msg):
        if not self._got_scan:
            self.get_logger().info('Received LaserScan on /scan')
        self._got_scan = True

    def _on_rear(self, msg):
        if not self._got_rear:
            self.get_logger().info('Controller rear commands received')
        self._got_rear = True

    def _on_front(self, msg):
        if not self._got_front:
            self.get_logger().info('Controller front commands received')
        self._got_front = True

    def _tick(self):
        t = time.time() - self._start_time
        speed = Float64()
        steer = Float64()
        speed.data = 5.0
        steer.data = 0.2
        self._pub_speed.publish(speed)
        self._pub_steer.publish(steer)

        if t > 5.0:
            self.get_logger().info('Test done. scan=%s rear=%s front=%s' % (self._got_scan, self._got_rear, self._got_front))
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TestPipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
