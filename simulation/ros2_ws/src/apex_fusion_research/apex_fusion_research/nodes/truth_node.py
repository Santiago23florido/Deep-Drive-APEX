"""6-DoF ground truth of the IMU frame, read directly from Gazebo.

The existing APEX ground-truth node is planar (x, y, yaw at the rear axle).
Inertial navigation needs the full 3D pose, so this node subscribes to the
Gazebo scene stream ``/world/<world>/dynamic_pose/info`` (~60 Hz sim time).
In that stream the model pose is expressed in the world frame while link
poses are expressed in the model frame, so the chassis pose is composed as
``T_world_link = T_world_model * T_model_link``. The IMU lever arm is then
applied and ``nav_msgs/Odometry`` is published in the world frame:

* pose       : IMU position and orientation (body -> world),
* twist.linear  : IMU velocity in the *world* frame (backward difference),
* twist.angular : body angular rate (from consecutive attitudes).
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from gz.msgs10 import pose_v_pb2
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from ..core.rotation import quat_conjugate, quat_multiply, quat_normalize, quat_to_rotmat, rotvec_from_quat
from ._common import odom_to_transform, sec_to_stamp


class FusionTruthNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_truth")
        self.declare_parameter("world_name", "default")
        self.declare_parameter("model_name", "rc_car")
        self.declare_parameter("link_name", "base_link")
        self.declare_parameter("sensor_offset_xyz", [0.02, 0.0, 0.08])  # IMU in base_link
        self.declare_parameter("pose_topic", "")  # default: /world/<world>/dynamic_pose/info
        self.declare_parameter("max_publish_rate_hz", 200.0)
        self.declare_parameter("odom_topic", "/apex/fusion/truth/odom")
        self.declare_parameter("path_topic", "/apex/fusion/truth/path")
        self.declare_parameter("path_rate_hz", 5.0)
        self.declare_parameter("path_max_poses", 20000)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("child_frame_id", "imu_truth")
        self.declare_parameter("publish_tf", True)
        gp = self.get_parameter

        self._model_name = str(gp("model_name").value)
        self._link_name = str(gp("link_name").value)
        self._link_names = {self._link_name, f"{self._model_name}::{self._link_name}"}
        self._offset = np.array(gp("sensor_offset_xyz").value, dtype=float)
        self._period = 1.0 / float(gp("max_publish_rate_hz").value)
        self._path_period = 1.0 / float(gp("path_rate_hz").value)
        self._path_max = int(gp("path_max_poses").value)
        self._frame_id = str(gp("frame_id").value)
        self._child_frame_id = str(gp("child_frame_id").value)

        self._odom_pub = self.create_publisher(Odometry, str(gp("odom_topic").value), 50)
        self._tf = TransformBroadcaster(self) if bool(gp("publish_tf").value) else None
        self._path_pub = self.create_publisher(NavPath, str(gp("path_topic").value), 2)
        self._path = NavPath()
        self._path.header.frame_id = self._frame_id
        self._lock = threading.Lock()
        self._last: tuple[float, np.ndarray, np.ndarray] | None = None
        self._last_path_t = -1e9

        topic = str(gp("pose_topic").value) or f"/world/{gp('world_name').value}/dynamic_pose/info"
        self._gz = GzNode()
        if not self._gz.subscribe(pose_v_pb2.Pose_V, topic, self._on_gz_pose):
            raise RuntimeError(f"cannot subscribe to Gazebo topic {topic}")
        self.get_logger().info(f"ground truth from {topic} ({self._model_name}/{self._link_name}), IMU offset {self._offset.tolist()}")

    def _on_gz_pose(self, msg: pose_v_pb2.Pose_V) -> None:
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nsec
        with self._lock:
            if self._last is not None and t - self._last[0] < self._period - 1e-6:
                return
            model_pose = link_pose = None
            for pose in msg.pose:
                if pose.name == self._model_name:
                    model_pose = pose
                elif pose.name in self._link_names:
                    link_pose = pose
            if model_pose is None or link_pose is None:
                return
            q_wm, p_wm = self._pose_to_np(model_pose)
            q_ml, p_ml = self._pose_to_np(link_pose)
            R_wm = quat_to_rotmat(q_wm)
            q = quat_normalize(quat_multiply(q_wm, q_ml))
            p = p_wm + R_wm @ p_ml + quat_to_rotmat(q) @ self._offset
            self._publish(t, p, q)

    @staticmethod
    def _pose_to_np(pose) -> tuple[np.ndarray, np.ndarray]:
        q = quat_normalize(np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]))
        return q, np.array([pose.position.x, pose.position.y, pose.position.z])

    def _publish(self, t: float, p: np.ndarray, q: np.ndarray) -> None:
        v = np.zeros(3)
        w = np.zeros(3)
        if self._last is not None:
            t0, p0, q0 = self._last
            dt = t - t0
            if dt > 0.0:
                v = (p - p0) / dt
                w = rotvec_from_quat(quat_multiply(quat_conjugate(q0), q)) / dt
        self._last = (t, p, q)

        odom = Odometry()
        odom.header.stamp = sec_to_stamp(t)
        odom.header.frame_id = self._frame_id
        odom.child_frame_id = self._child_frame_id
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, p)
        o = odom.pose.pose.orientation
        o.w, o.x, o.y, o.z = map(float, q)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, v)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, w)
        self._odom_pub.publish(odom)
        if self._tf is not None:
            self._tf.sendTransform(odom_to_transform(odom))

        if t - self._last_path_t >= self._path_period:
            self._last_path_t = t
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose = odom.pose.pose
            self._path.poses.append(ps)
            if len(self._path.poses) > self._path_max:
                del self._path.poses[: len(self._path.poses) - self._path_max]
            self._path.header.stamp = odom.header.stamp
            self._path_pub.publish(self._path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionTruthNode()
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
