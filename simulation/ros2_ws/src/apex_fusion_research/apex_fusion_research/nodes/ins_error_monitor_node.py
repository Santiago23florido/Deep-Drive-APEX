"""Evaluates the INS against ground truth and records the experiment.

For every INS output sample the truth trajectory is interpolated at the same
timestamp (linear position/velocity, SLERP attitude) and the navigation errors
are computed:

* position error ``p_ins - p_true`` (world), horizontal norm and vertical,
* velocity error ``v_ins - v_true`` (world),
* attitude error as roll/pitch/yaw differences and as the rotation vector
  ``Log(R_ins R_true^T)`` (tilt about world x/y, heading about z).

Outputs
-------
* ``/apex/fusion/ins/error`` (std_msgs/Float64MultiArray, labelled layout)
* ``/apex/fusion/ins/error_summary`` (JSON, 1 Hz)
* ``<output_dir>/ins_error.csv`` and ``<output_dir>/metadata.json`` with the
  IMU/LiDAR realizations and the INS alignment, when ``output_dir`` is set.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

from ..core.rotation import attitude_error_rotvec, quat_slerp, quat_to_euler, wrap_angle
from ._common import LATCHED_QOS, json_msg, stamp_to_sec

ERROR_LABELS = [
    "t_nav", "e_x", "e_y", "e_z", "e_horizontal", "e_vx", "e_vy", "e_vz",
    "e_roll", "e_pitch", "e_yaw",
]
CSV_COLUMNS = (
    ["t_sim", "t_nav"]
    + [f"true_{a}" for a in ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw")]
    + [f"ins_{a}" for a in ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw")]
    + ["e_x", "e_y", "e_z", "e_horizontal", "e_3d", "e_vx", "e_vy", "e_vz"]
    + ["e_roll", "e_pitch", "e_yaw", "e_tilt_x", "e_tilt_y", "e_heading"]
    + [f"true_gyro_bias_{a}" for a in "xyz"]
    + [f"true_accel_bias_{a}" for a in "xyz"]
    + ["distance_travelled"]
)


def _odom_arrays(msg: Odometry) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    v = msg.twist.twist.linear
    return (
        stamp_to_sec(msg.header.stamp),
        np.array([p.x, p.y, p.z]),
        np.array([v.x, v.y, v.z]),
        np.array([o.w, o.x, o.y, o.z]),
    )


class InsErrorMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_ins_error_monitor")
        self.declare_parameter("truth_topic", "/apex/fusion/truth/odom")
        self.declare_parameter("ins_topic", "/apex/fusion/ins/odom")
        self.declare_parameter("ins_alignment_topic", "/apex/fusion/ins/alignment")
        self.declare_parameter("imu_realization_topic", "/apex/fusion/imu/realization")
        self.declare_parameter("lidar_realization_topic", "/apex/fusion/lidar/realization")
        self.declare_parameter("true_gyro_bias_topic", "/apex/fusion/imu/true_gyro_bias")
        self.declare_parameter("true_accel_bias_topic", "/apex/fusion/imu/true_accel_bias")
        self.declare_parameter("ins_status_topic", "/apex/fusion/ins/status")
        self.declare_parameter("error_topic", "/apex/fusion/ins/error")
        self.declare_parameter("summary_topic", "/apex/fusion/ins/error_summary")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("csv_rate_hz", 20.0)
        self.declare_parameter("truth_buffer_s", 10.0)
        self.declare_parameter("summary_period_s", 1.0)
        gp = self.get_parameter

        self._csv_period = 1.0 / float(gp("csv_rate_hz").value)
        self._buffer_s = float(gp("truth_buffer_s").value)
        self._error_pub = self.create_publisher(Float64MultiArray, str(gp("error_topic").value), 50)
        self._summary_pub = self.create_publisher(String, str(gp("summary_topic").value), 10)

        self.create_subscription(Odometry, str(gp("truth_topic").value), self._on_truth, 100)
        self.create_subscription(Odometry, str(gp("ins_topic").value), self._on_ins, 200)
        self.create_subscription(Vector3Stamped, str(gp("true_gyro_bias_topic").value), self._on_gyro_bias, 50)
        self.create_subscription(Vector3Stamped, str(gp("true_accel_bias_topic").value), self._on_accel_bias, 50)
        for key, topic_param in (
            ("ins_alignment", "ins_alignment_topic"),
            ("imu", "imu_realization_topic"),
            ("lidar", "lidar_realization_topic"),
        ):
            self.create_subscription(
                String, str(gp(topic_param).value), lambda m, k=key: self._on_metadata(k, m), LATCHED_QOS
            )
        self.create_subscription(String, str(gp("ins_status_topic").value), self._on_ins_status, 10)
        self.create_timer(float(gp("summary_period_s").value), self._publish_summary)

        self._truth_t: list[float] = []
        self._truth: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._pending: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._metadata: dict = {}
        self._nav_t0: float | None = None
        self._last_csv_t = -1e9
        self._last_true_p: np.ndarray | None = None
        self._distance = 0.0
        self._latest: dict | None = None
        self._max_horizontal = 0.0
        self._ins_state = "unknown"

        out = str(gp("output_dir").value).strip()
        self._out_dir = Path(out).expanduser() if out else None
        self._csv_writer = None
        self._csv_file = None
        if self._out_dir is not None:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._out_dir / "ins_error.csv", "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(CSV_COLUMNS)
            self.get_logger().info(f"recording INS errors to {self._out_dir}")

    # ------------------------------------------------------------------ input
    def _on_metadata(self, key: str, msg: String) -> None:
        try:
            self._metadata[key] = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if key == "ins_alignment":
            # A new alignment means a new navigation run (e.g. after ~/reset).
            self._nav_t0 = float(self._metadata[key].get("t_start", 0.0))
            self._distance = 0.0
            self._max_horizontal = 0.0
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self._out_dir is None:
            return
        (self._out_dir / "metadata.json").write_text(json.dumps(self._metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _on_ins_status(self, msg: String) -> None:
        try:
            self._ins_state = str(json.loads(msg.data).get("state", "unknown"))
        except json.JSONDecodeError:
            pass

    def _on_gyro_bias(self, msg: Vector3Stamped) -> None:
        self._gyro_bias = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def _on_accel_bias(self, msg: Vector3Stamped) -> None:
        self._accel_bias = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def _on_truth(self, msg: Odometry) -> None:
        t, p, v, q = _odom_arrays(msg)
        if self._truth_t and t <= self._truth_t[-1]:
            return
        self._truth_t.append(t)
        self._truth.append((p, v, q))
        while self._truth_t and self._truth_t[0] < t - self._buffer_s:
            self._truth_t.pop(0)
            self._truth.pop(0)
        if self._last_true_p is not None and self._nav_t0 is not None and t >= self._nav_t0:
            self._distance += float(np.linalg.norm(p[:2] - self._last_true_p[:2]))
        self._last_true_p = p
        self._process_pending()

    def _on_ins(self, msg: Odometry) -> None:
        self._pending.append(_odom_arrays(msg))
        self._process_pending()

    # ------------------------------------------------------------- evaluation
    def _interpolate_truth(self, t: float):
        i = bisect.bisect_left(self._truth_t, t)
        if i == 0 or i >= len(self._truth_t):
            return None
        t0, t1 = self._truth_t[i - 1], self._truth_t[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        (p0, v0, q0), (p1, v1, q1) = self._truth[i - 1], self._truth[i]
        return p0 + a * (p1 - p0), v0 + a * (v1 - v0), quat_slerp(q0, q1, a)

    def _process_pending(self) -> None:
        if not self._truth_t:
            return
        keep = []
        for sample in self._pending:
            t = sample[0]
            if t > self._truth_t[-1]:
                keep.append(sample)  # wait for newer truth
                continue
            truth = self._interpolate_truth(t)
            if truth is not None:
                self._evaluate(t, sample[1], sample[2], sample[3], *truth)
        self._pending = keep[-500:]

    def _evaluate(self, t, p_ins, v_ins, q_ins, p_true, v_true, q_true) -> None:
        t_nav = t - self._nav_t0 if self._nav_t0 is not None else float("nan")
        e_p = p_ins - p_true
        e_v = v_ins - v_true
        rpy_true = np.array(quat_to_euler(q_true))
        rpy_ins = np.array(quat_to_euler(q_ins))
        e_rpy = np.array([wrap_angle(a) for a in rpy_ins - rpy_true])
        e_rot = attitude_error_rotvec(q_true, q_ins)
        e_h = float(math.hypot(e_p[0], e_p[1]))
        self._max_horizontal = max(self._max_horizontal, e_h)

        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label=",".join(ERROR_LABELS), size=len(ERROR_LABELS), stride=len(ERROR_LABELS))]
        msg.data = [float(x) for x in (t_nav, *e_p, e_h, *e_v, *e_rpy)]
        self._error_pub.publish(msg)
        self._latest = {
            "t_nav_s": t_nav,
            "horizontal_error_m": e_h,
            "vertical_error_m": float(e_p[2]),
            "speed_error_mps": float(np.linalg.norm(e_v)),
            "roll_pitch_yaw_error_deg": np.degrees(e_rpy).tolist(),
            "max_horizontal_error_m": self._max_horizontal,
            "distance_travelled_m": self._distance,
        }

        if self._csv_writer is not None and t - self._last_csv_t >= self._csv_period:
            self._last_csv_t = t
            row = [t, t_nav, *p_true, *v_true, *rpy_true, *p_ins, *v_ins, *rpy_ins, *e_p, e_h,
                   float(np.linalg.norm(e_p)), *e_v, *e_rpy, *e_rot, *self._gyro_bias, *self._accel_bias,
                   self._distance]
            self._csv_writer.writerow([f"{x:.9g}" for x in row])
            self._csv_file.flush()

    def _publish_summary(self) -> None:
        # Errors are only meaningful while the INS navigates; otherwise report
        # the state alone so stale values are never mistaken for current ones.
        if self._ins_state != "navigating" or self._latest is None:
            self._summary_pub.publish(json_msg({"ins_state": self._ins_state}))
            return
        self._summary_pub.publish(json_msg({"ins_state": self._ins_state, **self._latest}))

    def destroy_node(self) -> bool:
        if self._csv_file is not None:
            self._csv_file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InsErrorMonitorNode()
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
