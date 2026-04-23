#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class RearWheelSpeedPublisher(Node):
    def __init__(self) -> None:
        super().__init__("rear_wheel_speed_publisher")
        self.declare_parameter("speed", 5.0)
        self.declare_parameter("steering_angle", 0.0)
        self.declare_parameter("publish_rate", 60.0)
        self._speed_topic = "/rear_wheel_speed"
        self._steering_topic = "/steering_angle"

        self._speed_publisher = self.create_publisher(Float64, self._speed_topic, 10)
        self._steering_publisher = self.create_publisher(
            Float64, self._steering_topic, 10
        )
        self._publish_rate = self.get_parameter("publish_rate").value

        if self._publish_rate > 0.0:
            self._timer = self.create_timer(
                1.0 / self._publish_rate, self._publish
            )
        else:
            self._timer = None
            self.get_logger().warn("publish_rate <= 0.0, timer disabled")

    def _publish(self) -> None:
        speed_msg = Float64()
        speed_msg.data = float(self.get_parameter("speed").value)
        steering_msg = Float64()
        steering_msg.data = float(self.get_parameter("steering_angle").value)
        self._speed_publisher.publish(speed_msg)
        self._steering_publisher.publish(steering_msg)


def main() -> None:
    rclpy.init()
    node = RearWheelSpeedPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
