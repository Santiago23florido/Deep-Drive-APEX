"""Pure strapdown INS: integrates the raw IMU and nothing else.

State machine
-------------
``WAITING_TRUTH`` -> ``ALIGNING`` -> ``NAVIGATING``

* ``WAITING_TRUTH``: a start pose is needed (position + heading are not
  observable from a 6-axis IMU).
* ``ALIGNING`` (``alignment.mode = static_coarse``): raw IMU samples are
  accumulated while ground truth confirms the vehicle is stationary. Motion
  restarts the window. After ``alignment.duration_s`` the INS levels itself,
  estimates the gyro bias (and accelerometer bias along gravity) and starts.
  With ``alignment.mode = truth`` the full state is copied from truth at once.
* ``NAVIGATING``: open-loop strapdown integration of every IMU sample. Ground
  truth is *never* used again: all sensor errors accumulate freely.

The ``~/reset`` service (std_srvs/Trigger) restarts the alignment, which is
handy for repeated drift experiments in one simulation session.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..core.rotation import quat_from_euler, quat_to_euler
from ..core.strapdown import InsConfig, StrapdownIntegrator, static_coarse_alignment
from ._common import LATCHED_QOS, ConfigParameters, json_msg, odom_to_transform, stamp_to_sec

WAITING_TRUTH = "waiting_truth"
ALIGNING = "aligning"
NAVIGATING = "navigating"


class StrapdownInsNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_strapdown_ins")
        self.declare_parameter("imu_topic", "/apex/fusion/imu/data_raw")
        self.declare_parameter("truth_topic", "/apex/fusion/truth/odom")
        self.declare_parameter("odom_topic", "/apex/fusion/ins/odom")
        self.declare_parameter("path_topic", "/apex/fusion/ins/path")
        self.declare_parameter("status_topic", "/apex/fusion/ins/status")
        self.declare_parameter("alignment_topic", "/apex/fusion/ins/alignment")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("child_frame_id", "imu_ins")
        self.declare_parameter("path_rate_hz", 5.0)
        self.declare_parameter("path_max_poses", 20000)
        self.declare_parameter("status_period_s", 1.0)
        self.declare_parameter("gap_warning_factor", 3.0)  # warn if dt > factor * median dt
        self.declare_parameter("publish_tf", True)
        gp = self.get_parameter

        self._params = ConfigParameters(self, InsConfig, on_change=lambda _cfg: self._reset("parameters changed"))
        self._frame_id = str(gp("frame_id").value)
        self._child_frame_id = str(gp("child_frame_id").value)
        self._path_period = 1.0 / float(gp("path_rate_hz").value)
        self._path_max = int(gp("path_max_poses").value)
        self._gap_factor = float(gp("gap_warning_factor").value)

        self._odom_pub = self.create_publisher(Odometry, str(gp("odom_topic").value), 50)
        self._tf = TransformBroadcaster(self) if bool(gp("publish_tf").value) else None
        self._path_pub = self.create_publisher(NavPath, str(gp("path_topic").value), 2)
        self._status_pub = self.create_publisher(String, str(gp("status_topic").value), 10)
        self._alignment_pub = self.create_publisher(String, str(gp("alignment_topic").value), LATCHED_QOS)
        self.create_subscription(Imu, str(gp("imu_topic").value), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, str(gp("truth_topic").value), self._on_truth, 50)
        self.create_service(Trigger, "~/reset", self._on_reset_srv)
        self.create_timer(float(gp("status_period_s").value), self._publish_status)

        self._truth: Odometry | None = None
        self._reset("startup")

    # ---------------------------------------------------------------- control
    def _reset(self, reason: str) -> None:
        cfg: InsConfig = self._params.config
        self._ins = StrapdownIntegrator(cfg.strapdown)
        self._state = WAITING_TRUTH
        self._window_w: list[np.ndarray] = []
        self._window_f: list[np.ndarray] = []
        self._window_t0: float | None = None
        self._nav_t0: float | None = None
        self._last_imu_t: float | None = None
        self._dts: deque[float] = deque(maxlen=200)
        self._gaps = 0
        self._path = NavPath()
        self._path.header.frame_id = self._frame_id
        self._last_path_t = -1e9
        self._alignment_info: dict = {}
        self.get_logger().info(f"INS reset ({reason}); alignment mode '{cfg.alignment.mode}'")

    def _on_reset_srv(self, _request, response):
        self._reset("service call")
        response.success = True
        response.message = "INS reset; waiting for alignment"
        return response

    def _on_truth(self, msg: Odometry) -> None:
        self._truth = msg
        if self._state == WAITING_TRUTH:
            self._state = ALIGNING

    def _truth_is_stationary(self) -> bool:
        a = self._params.config.alignment
        tw = self._truth.twist.twist
        speed = float(np.linalg.norm([tw.linear.x, tw.linear.y, tw.linear.z]))
        rate = float(np.linalg.norm([tw.angular.x, tw.angular.y, tw.angular.z]))
        return speed <= a.max_truth_speed_mps and rate <= a.max_truth_rate_rps

    def _truth_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = self._truth.pose.pose
        q = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        p = np.array([pose.position.x, pose.position.y, pose.position.z])
        tw = self._truth.twist.twist.linear
        return q, np.array([tw.x, tw.y, tw.z]), p

    # -------------------------------------------------------------- IMU input
    def _on_imu(self, msg: Imu) -> None:
        t = stamp_to_sec(msg.header.stamp)
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        f = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self._track_timing(t)

        if self._state == ALIGNING:
            self._align(t, w, f)
        elif self._state == NAVIGATING:
            state = self._ins.propagate(t, w, f)
            self._publish_state(msg, state)

    def _track_timing(self, t: float) -> None:
        if self._last_imu_t is not None:
            dt = t - self._last_imu_t
            if dt > 0.0:
                if len(self._dts) >= 20 and dt > self._gap_factor * float(np.median(self._dts)):
                    self._gaps += 1
                    self.get_logger().warn(f"IMU gap of {dt * 1000:.1f} ms at t={t:.3f} s", throttle_duration_sec=5.0)
                self._dts.append(dt)
        self._last_imu_t = t

    def _align(self, t: float, w: np.ndarray, f: np.ndarray) -> None:
        cfg: InsConfig = self._params.config
        a = cfg.alignment
        q_true, v_true, p_true = self._truth_state()
        if a.mode == "truth":
            self._start_navigation(t, q_true, v_true, p_true, np.zeros(3), np.zeros(3), {"mode": "truth"})
            return
        if not self._truth_is_stationary():
            if self._window_w:
                self.get_logger().info("vehicle moving: static alignment window restarted", throttle_duration_sec=5.0)
            self._window_w.clear()
            self._window_f.clear()
            self._window_t0 = None
            return
        if self._window_t0 is None:
            self._window_t0 = t
        self._window_w.append(w)
        self._window_f.append(f)
        if t - self._window_t0 < a.duration_s:
            return
        res = static_coarse_alignment(
            np.array(self._window_w),
            np.array(self._window_f),
            cfg.strapdown.gravity_mps2,
            a.estimate_gyro_bias,
            a.estimate_accel_bias_along_gravity,
        )
        _, _, yaw_true = quat_to_euler(q_true)
        q0 = quat_from_euler(res.roll, res.pitch, yaw_true)
        info = {
            "mode": "static_coarse",
            "window_s": t - self._window_t0,
            "samples": res.samples,
            "roll_rad": res.roll,
            "pitch_rad": res.pitch,
            "truth_roll_pitch_yaw_rad": list(quat_to_euler(q_true)),
            "gyro_std": res.gyro_std.tolist(),
            "accel_std": res.accel_std.tolist(),
            "mean_specific_force": res.mean_specific_force.tolist(),
        }
        self._start_navigation(t, q0, np.zeros(3), p_true, res.gyro_bias, res.accel_bias, info)

    def _start_navigation(self, t, q, v, p, gyro_bias, accel_bias, info: dict) -> None:
        self._ins.initialize(t, q, v, p, gyro_bias, accel_bias)
        self._nav_t0 = t
        self._state = NAVIGATING
        info.update(
            {
                "t_start": t,
                "estimated_gyro_bias": np.asarray(gyro_bias).tolist(),
                "estimated_accel_bias": np.asarray(accel_bias).tolist(),
                "initial_position": np.asarray(p).tolist(),
                "initial_rpy_rad": list(quat_to_euler(q)),
                "config": self._params.as_dict(),
            }
        )
        self._alignment_info = info
        self._alignment_pub.publish(json_msg(info))
        self.get_logger().info(
            f"navigation started at t={t:.3f} s; gyro bias estimate "
            f"{np.round(gyro_bias, 5).tolist()} rad/s, accel bias estimate {np.round(accel_bias, 4).tolist()} m/s^2"
        )

    # ---------------------------------------------------------------- outputs
    def _publish_state(self, imu_msg: Imu, state) -> None:
        odom = Odometry()
        odom.header.stamp = imu_msg.header.stamp
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        pos = odom.pose.pose.position
        pos.x, pos.y, pos.z = map(float, state.p)
        o = odom.pose.pose.orientation
        o.w, o.x, o.y, o.z = map(float, state.q)
        lin = odom.twist.twist.linear
        lin.x, lin.y, lin.z = map(float, state.v)  # world frame, like the truth topic
        ang = odom.twist.twist.angular
        ang.x = imu_msg.angular_velocity.x - float(self._ins.gyro_bias[0])
        ang.y = imu_msg.angular_velocity.y - float(self._ins.gyro_bias[1])
        ang.z = imu_msg.angular_velocity.z - float(self._ins.gyro_bias[2])
        self._odom_pub.publish(odom)
        if self._tf is not None:
            self._tf.sendTransform(odom_to_transform(odom))

        if state.t - self._last_path_t >= self._path_period:
            self._last_path_t = state.t
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose = odom.pose.pose
            self._path.poses.append(ps)
            if len(self._path.poses) > self._path_max:
                del self._path.poses[: len(self._path.poses) - self._path_max]
            self._path.header.stamp = odom.header.stamp
            self._path_pub.publish(self._path)

    def _publish_status(self) -> None:
        payload = {
            "state": self._state,
            "alignment_mode": self._params.config.alignment.mode,
            "imu_gaps": self._gaps,
            "median_imu_dt_s": float(np.median(self._dts)) if self._dts else None,
        }
        if self._state == ALIGNING and self._window_t0 is not None and self._last_imu_t is not None:
            payload["alignment_progress_s"] = self._last_imu_t - self._window_t0
        if self._state == NAVIGATING and self._ins.state is not None:
            payload["navigation_time_s"] = self._ins.state.t - self._nav_t0
            payload["position"] = self._ins.state.p.tolist()
            payload["velocity"] = self._ins.state.v.tolist()
        self._status_pub.publish(json_msg(payload))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StrapdownInsNode()
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
