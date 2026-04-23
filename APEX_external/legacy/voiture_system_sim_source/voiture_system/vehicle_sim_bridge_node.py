#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


class VehicleSimBridgeNode(Node):
    def __init__(self):
        super().__init__('vehicle_sim_bridge_node')

        self.declare_parameter('rear_wheel_joints', ['rear_left_wheel_joint', 'rear_right_wheel_joint'])
        self.declare_parameter('front_steer_joints', ['front_left_wheel_steer_joint', 'front_right_wheel_steer_joint'])
        self.declare_parameter('max_wheel_omega', 50.0)
        self.declare_parameter('max_steer_angle', 0.6)
        self.declare_parameter('speed_sign', 1.0)
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('publish_rate_hz', 50.0)

        self.declare_parameter('cmd_speed_topic', '/car/cmd_speed_wheel_omega')
        self.declare_parameter('cmd_steer_topic', '/car/cmd_steer_angle')

        self.declare_parameter('rear_controller_topic', '/rear_wheels_velocity_controller/commands')
        self.declare_parameter('front_controller_topic', '/front_steer_position_controller/commands')

        self._rear_wheel_joints = list(self.get_parameter('rear_wheel_joints').value)
        self._front_steer_joints = list(self.get_parameter('front_steer_joints').value)
        self._max_wheel_omega = float(self.get_parameter('max_wheel_omega').value)
        self._max_steer_angle = float(self.get_parameter('max_steer_angle').value)
        self._speed_sign = float(self.get_parameter('speed_sign').value)
        self._steer_sign = float(self.get_parameter('steer_sign').value)
        self._publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self._cmd_speed_topic = self.get_parameter('cmd_speed_topic').value
        self._cmd_steer_topic = self.get_parameter('cmd_steer_topic').value
        self._rear_controller_topic = self.get_parameter('rear_controller_topic').value
        self._front_controller_topic = self.get_parameter('front_controller_topic').value

        self._last_speed = 0.0
        self._last_steer = 0.0

        self.create_subscription(Float64, self._cmd_speed_topic, self._on_speed, 10)
        self.create_subscription(Float64, self._cmd_steer_topic, self._on_steer, 10)

        self._pub_rear = self.create_publisher(Float64MultiArray, self._rear_controller_topic, 10)
        self._pub_front = self.create_publisher(Float64MultiArray, self._front_controller_topic, 10)

        period = 1.0 / max(self._publish_rate_hz, 1e-3)
        self._timer = self.create_timer(period, self._publish_commands)

        self.get_logger().info('VehicleSimBridgeNode started')

    def _on_speed(self, msg: Float64):
        self._last_speed = float(msg.data)

    def _on_steer(self, msg: Float64):
        self._last_steer = float(msg.data)

    def _saturate(self, value, limit):
        if limit <= 0:
            return value
        return max(-limit, min(limit, value))

    def _publish_commands(self):
        omega = self._saturate(self._last_speed, self._max_wheel_omega) * self._speed_sign
        steer = self._saturate(self._last_steer, self._max_steer_angle) * self._steer_sign

        rear_msg = Float64MultiArray()
        rear_msg.data = [omega for _ in self._rear_wheel_joints]
        self._pub_rear.publish(rear_msg)

        front_msg = Float64MultiArray()
        front_msg.data = [steer for _ in self._front_steer_joints]
        self._pub_front.publish(front_msg)


def main():
    rclpy.init()
    node = VehicleSimBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
