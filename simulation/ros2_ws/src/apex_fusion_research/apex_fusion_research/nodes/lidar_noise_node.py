"""LiDAR noise injector: ideal Gazebo scan -> realistic noisy scan.

Input : ideal ``sensor_msgs/LaserScan`` (Gazebo noise disabled).
Output: corrupted ``LaserScan`` with identical geometry/timestamps, produced by
        :class:`apex_fusion_research.core.lidar_noise.LidarNoiseModel`.
A JSON status topic reports the empirical fraction of each beam outcome, and
the per-realization systematic errors are published on a latched topic.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from ..core.lidar_noise import OUTCOME_NAMES, OUTCOME_NO_TRUTH, LidarNoiseConfig, LidarNoiseModel
from ._common import LATCHED_QOS, ConfigParameters, json_msg


class LidarNoiseNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_lidar_noise")
        self.declare_parameter("input_topic", "/apex/sim/scan")
        self.declare_parameter("output_topic", "/apex/fusion/scan_noisy")
        self.declare_parameter("status_topic", "/apex/fusion/lidar/status")
        self.declare_parameter("realization_topic", "/apex/fusion/lidar/realization")
        self.declare_parameter("status_period_s", 2.0)
        gp = self.get_parameter

        self._params = ConfigParameters(self, LidarNoiseConfig, on_change=self._rebuild)
        self._scan_pub = self.create_publisher(LaserScan, str(gp("output_topic").value), qos_profile_sensor_data)
        self._status_pub = self.create_publisher(String, str(gp("status_topic").value), 10)
        self._realization_pub = self.create_publisher(String, str(gp("realization_topic").value), LATCHED_QOS)
        self.create_subscription(LaserScan, str(gp("input_topic").value), self._on_scan, qos_profile_sensor_data)
        self.create_timer(float(gp("status_period_s").value), self._publish_status)

        self._model: LidarNoiseModel | None = None
        self._counts = np.zeros(len(OUTCOME_NAMES), dtype=np.int64)
        self._scans = 0
        self._residual_sq = 0.0
        self._residual_n = 0
        self._rebuild(self._params.config)

    def _rebuild(self, config: LidarNoiseConfig) -> None:
        self._model = LidarNoiseModel(config)
        self._realization_pub.publish(
            json_msg({"config": self._params.as_dict(), "realization": self._model.realization()})
        )
        self.get_logger().info(
            f"LiDAR noise realization (seed={config.seed}, enabled={config.enabled}): "
            f"{self._model.realization()}"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        res = self._model.apply(
            np.asarray(msg.ranges, dtype=float),
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
        )
        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = res.ranges.astype(np.float32).tolist()
        out.intensities = list(msg.intensities)
        self._scan_pub.publish(out)

        self._counts += np.bincount(res.outcome.astype(np.int64), minlength=len(OUTCOME_NAMES))[: len(OUTCOME_NAMES)]
        self._scans += 1
        hit = (res.outcome == 1) & np.isfinite(res.ranges)
        if np.any(hit):
            r = res.ranges[hit] - res.true_ranges[hit]
            self._residual_sq += float(np.dot(r, r))
            self._residual_n += int(r.size)

    def _publish_status(self) -> None:
        if self._scans == 0:
            return
        measured = int(self._counts.sum() - self._counts[OUTCOME_NO_TRUTH])
        fractions = {
            name: (float(self._counts[code]) / measured if measured else 0.0)
            for code, name in OUTCOME_NAMES.items()
            if code != OUTCOME_NO_TRUTH
        }
        rms = (self._residual_sq / self._residual_n) ** 0.5 if self._residual_n else 0.0
        self._status_pub.publish(
            json_msg({"scans": self._scans, "outcome_fractions": fractions, "hit_residual_rms_m": rms})
        )
        self._counts[:] = 0
        self._scans = 0
        self._residual_sq = 0.0
        self._residual_n = 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarNoiseNode()
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
