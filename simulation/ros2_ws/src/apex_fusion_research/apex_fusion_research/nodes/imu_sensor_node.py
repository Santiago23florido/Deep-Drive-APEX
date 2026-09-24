"""Realistic IMU: turns Gazebo's ideal IMU into raw MEMS measurements.

Input : ideal ``sensor_msgs/Imu`` from Gazebo (noise disabled in the URDF).
        Only angular velocity and linear acceleration (specific force) are used.
Output: raw ``sensor_msgs/Imu`` exactly like a physical 6-axis IMU delivers it:
        no orientation (``orientation_covariance[0] = -1``), corrupted angular
        rate and specific force, white-noise covariances on the diagonal.
Ground truth of the error states is published separately for evaluation of
future fusion filters (true total gyro/accelerometer bias).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from ..core.imu_error import ImuErrorConfig, ImuErrorModel
from ._common import LATCHED_QOS, ConfigParameters, json_msg, stamp_to_sec


class ImuSensorNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_imu_sensor")
        self.declare_parameter("input_topic", "/apex/sim/imu")
        self.declare_parameter("output_topic", "/apex/fusion/imu/data_raw")
        self.declare_parameter("true_gyro_bias_topic", "/apex/fusion/imu/true_gyro_bias")
        self.declare_parameter("true_accel_bias_topic", "/apex/fusion/imu/true_accel_bias")
        self.declare_parameter("realization_topic", "/apex/fusion/imu/realization")
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("nominal_rate_hz", 120.0)
        self.declare_parameter("max_dt_s", 0.5)

        gp = self.get_parameter
        self._frame_id = str(gp("frame_id").value)
        self._nominal_dt = 1.0 / float(gp("nominal_rate_hz").value)
        self._max_dt = float(gp("max_dt_s").value)

        self._params = ConfigParameters(self, ImuErrorConfig, on_change=self._rebuild)
        self._imu_pub = self.create_publisher(Imu, str(gp("output_topic").value), qos_profile_sensor_data)
        self._gyro_bias_pub = self.create_publisher(Vector3Stamped, str(gp("true_gyro_bias_topic").value), 50)
        self._accel_bias_pub = self.create_publisher(Vector3Stamped, str(gp("true_accel_bias_topic").value), 50)
        self._realization_pub = self.create_publisher(String, str(gp("realization_topic").value), LATCHED_QOS)
        self.create_subscription(Imu, str(gp("input_topic").value), self._on_imu, qos_profile_sensor_data)

        self._model: ImuErrorModel | None = None
        self._last_t: float | None = None
        self._rebuild(self._params.config)

    def _rebuild(self, config: ImuErrorConfig) -> None:
        """(Re)create the IMU: equivalent to a sensor power cycle."""
        self._model = ImuErrorModel(config)
        self._last_t = None
        payload = {"config": self._params.as_dict(), "realization": self._model.realization()}
        self._realization_pub.publish(json_msg(payload))
        self.get_logger().info(
            f"IMU realization (seed={config.seed}): gyro turn-on bias "
            f"{np.round(self._model.gyro.turn_on_bias, 5).tolist()} rad/s, accel turn-on bias "
            f"{np.round(self._model.accel.turn_on_bias, 4).tolist()} m/s^2"
        )

    def _on_imu(self, msg: Imu) -> None:
        t = stamp_to_sec(msg.header.stamp)
        if self._last_t is None:
            dt = self._nominal_dt
        else:
            dt = t - self._last_t
            if dt <= 0.0:
                return  # duplicated / out-of-order sample
            dt = min(dt, self._max_dt)
        self._last_t = t

        w_true = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        f_true = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        sample = self._model.measure(w_true, f_true, dt)

        out = Imu()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id
        out.orientation_covariance[0] = -1.0  # a raw IMU provides no orientation
        out.angular_velocity = Vector3(x=float(sample.angular_velocity[0]), y=float(sample.angular_velocity[1]), z=float(sample.angular_velocity[2]))
        out.linear_acceleration = Vector3(x=float(sample.specific_force[0]), y=float(sample.specific_force[1]), z=float(sample.specific_force[2]))
        sigma_g, sigma_a = self._model.white_noise_std(dt)
        for i in range(3):
            out.angular_velocity_covariance[4 * i] = sigma_g * sigma_g
            out.linear_acceleration_covariance[4 * i] = sigma_a * sigma_a
        self._imu_pub.publish(out)

        for pub, bias in ((self._gyro_bias_pub, sample.gyro_bias), (self._accel_bias_pub, sample.accel_bias)):
            b = Vector3Stamped()
            b.header = out.header
            b.vector = Vector3(x=float(bias[0]), y=float(bias[1]), z=float(bias[2]))
            pub.publish(b)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
