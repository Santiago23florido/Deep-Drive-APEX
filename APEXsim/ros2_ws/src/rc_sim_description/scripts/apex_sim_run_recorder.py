#!/usr/bin/env python3
"""Record one simulation run into a reusable dataset directory."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, PointCloud
from std_msgs.msg import String


def _stamp_to_pair(stamp) -> tuple[int, int]:
    return int(stamp.sec), int(stamp.nanosec)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * ((w * z) + (x * y)),
        1.0 - (2.0 * ((y * y) + (z * z))),
    )


def _json_safe_copy(payload: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload))


class ApexSimRunRecorder(Node):
    def __init__(self) -> None:
        super().__init__("apex_sim_run_recorder")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.declare_parameter("run_dir", "")
        self.declare_parameter("scan_topic", "/apex/sim/scan")
        self.declare_parameter("imu_topic", "/apex/sim/imu")
        self.declare_parameter("odom_topic", "/apex/odometry/imu_lidar_fused")
        self.declare_parameter("ground_truth_odom_topic", "/apex/sim/ground_truth/odom")
        self.declare_parameter("ground_truth_path_topic", "/apex/sim/ground_truth/path")
        self.declare_parameter("perfect_map_topic", "/apex/sim/ground_truth/perfect_map_points")
        self.declare_parameter("ground_truth_status_topic", "/apex/sim/ground_truth/status")
        self.declare_parameter("online_status_topic", "/apex/estimation/status")
        self.declare_parameter("scenario", "")
        self.declare_parameter("control_mode", "")
        self.declare_parameter("mapping_mode", "")
        self.declare_parameter("estimation_mode", "")
        self.declare_parameter("distortion_profile", "")
        self.declare_parameter("refinement_mode", "")
        self.declare_parameter("offline_replay_mode", "")
        self.declare_parameter("world_path", "")
        self.declare_parameter("write_online_status", True)
        self.declare_parameter("status_log_every_sec", 5.0)

        run_dir_value = str(self.get_parameter("run_dir").value).strip()
        if not run_dir_value:
            raise RuntimeError("apex_sim_run_recorder requires a non-empty run_dir")
        self._run_dir = Path(run_dir_value).expanduser().resolve()
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._imu_topic = str(self.get_parameter("imu_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._ground_truth_odom_topic = str(self.get_parameter("ground_truth_odom_topic").value)
        self._ground_truth_path_topic = str(self.get_parameter("ground_truth_path_topic").value)
        self._perfect_map_topic = str(self.get_parameter("perfect_map_topic").value)
        self._ground_truth_status_topic = str(self.get_parameter("ground_truth_status_topic").value)
        self._online_status_topic = str(self.get_parameter("online_status_topic").value)
        self._write_online_status = bool(self.get_parameter("write_online_status").value)
        self._status_log_every_sec = max(
            1.0, float(self.get_parameter("status_log_every_sec").value)
        )

        self._imu_count = 0
        self._scan_count = 0
        self._lidar_point_count = 0
        self._odom_count = 0
        self._ground_truth_odom_count = 0
        self._ground_truth_status_count = 0
        self._online_status_count = 0
        self._track_geometry_count = 0
        self._latest_ground_truth_path: NavPath | None = None
        self._latest_ground_truth_status: dict[str, object] | None = None
        self._latest_online_status: dict[str, object] | None = None
        self._track_geometry_written = False
        self._closed = False

        self._imu_handle = (self._run_dir / "imu_raw.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._imu_writer = csv.writer(self._imu_handle)
        self._imu_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "gx_rps",
                "gy_rps",
                "gz_rps",
                "qx",
                "qy",
                "qz",
                "qw",
            ]
        )

        self._lidar_handle = (self._run_dir / "lidar_points.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._lidar_writer = csv.writer(self._lidar_handle)
        self._lidar_writer.writerow(
            [
                "scan_index",
                "stamp_sec",
                "stamp_nanosec",
                "beam_index",
                "angle_rad",
                "range_m",
                "x_forward_m",
                "y_left_m",
            ]
        )

        self._scan_index_handle = (self._run_dir / "scan_index.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._scan_index_writer = csv.writer(self._scan_index_handle)
        self._scan_index_writer.writerow(
            [
                "scan_index",
                "stamp_sec",
                "stamp_nanosec",
                "frame_id",
                "angle_min_rad",
                "angle_max_rad",
                "angle_increment_rad",
                "time_increment_s",
                "scan_time_s",
                "range_min_m",
                "range_max_m",
                "beam_count",
                "valid_point_count",
            ]
        )

        self._odom_handle = (self._run_dir / "odom_fused.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._odom_writer = csv.writer(self._odom_handle)
        self._odom_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "frame_id",
                "child_frame_id",
                "x_m",
                "y_m",
                "yaw_rad",
                "vx_mps",
                "vy_mps",
                "yaw_rate_rps",
            ]
        )

        self._ground_truth_odom_handle = (self._run_dir / "ground_truth_odom.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._ground_truth_odom_writer = csv.writer(self._ground_truth_odom_handle)
        self._ground_truth_odom_writer.writerow(
            [
                "stamp_sec",
                "stamp_nanosec",
                "frame_id",
                "child_frame_id",
                "x_m",
                "y_m",
                "yaw_rad",
                "vx_mps",
                "vy_mps",
                "yaw_rate_rps",
            ]
        )

        self._ground_truth_status_handle = (self._run_dir / "ground_truth_status.jsonl").open(
            "w", encoding="utf-8"
        )
        self._online_status_handle = (self._run_dir / "online_status.jsonl").open(
            "w", encoding="utf-8"
        )

        metadata = {
            "run_dir": str(self._run_dir),
            "scan_topic": self._scan_topic,
            "imu_topic": self._imu_topic,
            "odom_topic": self._odom_topic,
            "ground_truth_odom_topic": self._ground_truth_odom_topic,
            "ground_truth_path_topic": self._ground_truth_path_topic,
            "perfect_map_topic": self._perfect_map_topic,
            "ground_truth_status_topic": self._ground_truth_status_topic,
            "online_status_topic": self._online_status_topic,
            "scenario": str(self.get_parameter("scenario").value),
            "control_mode": str(self.get_parameter("control_mode").value),
            "mapping_mode": str(self.get_parameter("mapping_mode").value),
            "estimation_mode": str(self.get_parameter("estimation_mode").value),
            "distortion_profile": str(self.get_parameter("distortion_profile").value),
            "refinement_mode": str(self.get_parameter("refinement_mode").value),
            "offline_replay_mode": str(self.get_parameter("offline_replay_mode").value),
            "world_path": str(self.get_parameter("world_path").value),
        }
        (self._run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        self._latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Imu, self._imu_topic, self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self._odom_topic, self._odom_cb, 20)
        self.create_subscription(
            Odometry, self._ground_truth_odom_topic, self._ground_truth_odom_cb, 20
        )
        self.create_subscription(
            NavPath, self._ground_truth_path_topic, self._ground_truth_path_cb, 10
        )
        self.create_subscription(
            PointCloud, self._perfect_map_topic, self._perfect_map_cb, self._map_qos
        )
        self.create_subscription(
            String, self._ground_truth_status_topic, self._ground_truth_status_cb, 20
        )
        if self._write_online_status:
            self.create_subscription(String, self._online_status_topic, self._online_status_cb, 20)

        self.create_timer(self._status_log_every_sec, self._log_progress)
        self.get_logger().info(
            "ApexSimRunRecorder started (run_dir=%s scan=%s imu=%s odom=%s gt_map=%s)"
            % (
                self._run_dir,
                self._scan_topic,
                self._imu_topic,
                self._odom_topic,
                self._perfect_map_topic,
            )
        )

    def _imu_cb(self, msg: Imu) -> None:
        sec, nanosec = _stamp_to_pair(msg.header.stamp)
        self._imu_writer.writerow(
            [
                sec,
                nanosec,
                f"{float(msg.linear_acceleration.x):.9f}",
                f"{float(msg.linear_acceleration.y):.9f}",
                f"{float(msg.linear_acceleration.z):.9f}",
                f"{float(msg.angular_velocity.x):.9f}",
                f"{float(msg.angular_velocity.y):.9f}",
                f"{float(msg.angular_velocity.z):.9f}",
                f"{float(msg.orientation.x):.9f}",
                f"{float(msg.orientation.y):.9f}",
                f"{float(msg.orientation.z):.9f}",
                f"{float(msg.orientation.w):.9f}",
            ]
        )
        self._imu_count += 1
        if (self._imu_count % 50) == 0:
            self._imu_handle.flush()

    def _scan_cb(self, msg: LaserScan) -> None:
        sec, nanosec = _stamp_to_pair(msg.header.stamp)
        scan_index = self._scan_count
        valid_point_count = 0
        rows: list[list[object]] = []
        angle = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        range_min = float(msg.range_min)
        range_max = float(msg.range_max)
        for beam_index, range_value in enumerate(msg.ranges):
            range_m = float(range_value)
            if math.isfinite(range_m) and range_min <= range_m <= range_max:
                x_forward_m = range_m * math.cos(angle)
                y_left_m = range_m * math.sin(angle)
                rows.append(
                    [
                        scan_index,
                        sec,
                        nanosec,
                        beam_index,
                        f"{angle:.9f}",
                        f"{range_m:.9f}",
                        f"{x_forward_m:.9f}",
                        f"{y_left_m:.9f}",
                    ]
                )
                valid_point_count += 1
            angle += angle_increment
        if rows:
            self._lidar_writer.writerows(rows)
            self._lidar_point_count += len(rows)
        self._scan_index_writer.writerow(
            [
                scan_index,
                sec,
                nanosec,
                msg.header.frame_id,
                f"{float(msg.angle_min):.9f}",
                f"{float(msg.angle_max):.9f}",
                f"{float(msg.angle_increment):.9f}",
                f"{float(msg.time_increment):.9f}",
                f"{float(msg.scan_time):.9f}",
                f"{range_min:.9f}",
                f"{range_max:.9f}",
                len(msg.ranges),
                valid_point_count,
            ]
        )
        self._scan_count += 1
        if (self._scan_count % 5) == 0:
            self._lidar_handle.flush()
            self._scan_index_handle.flush()

    def _odom_cb(self, msg: Odometry) -> None:
        self._write_odom_row(self._odom_writer, msg)
        self._odom_count += 1
        if (self._odom_count % 20) == 0:
            self._odom_handle.flush()

    def _ground_truth_odom_cb(self, msg: Odometry) -> None:
        self._write_odom_row(self._ground_truth_odom_writer, msg)
        self._ground_truth_odom_count += 1
        if (self._ground_truth_odom_count % 20) == 0:
            self._ground_truth_odom_handle.flush()

    def _ground_truth_path_cb(self, msg: NavPath) -> None:
        self._latest_ground_truth_path = msg
        self._write_ground_truth_path()

    def _perfect_map_cb(self, msg: PointCloud) -> None:
        if self._track_geometry_written or not msg.points:
            return
        track_geometry_path = self._run_dir / "track_geometry.csv"
        with track_geometry_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x_m", "y_m"])
            for point in msg.points:
                writer.writerow([f"{float(point.x):.6f}", f"{float(point.y):.6f}"])
        self._track_geometry_count = len(msg.points)
        self._track_geometry_written = True

    def _ground_truth_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._latest_ground_truth_status = _json_safe_copy(payload)
        self._ground_truth_status_handle.write(json.dumps(payload) + "\n")
        self._ground_truth_status_count += 1
        if (self._ground_truth_status_count % 10) == 0:
            self._ground_truth_status_handle.flush()
        (self._run_dir / "ground_truth_latest_status.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _online_status_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._latest_online_status = _json_safe_copy(payload)
        self._online_status_handle.write(json.dumps(payload) + "\n")
        self._online_status_count += 1
        if (self._online_status_count % 10) == 0:
            self._online_status_handle.flush()
        (self._run_dir / "online_latest_status.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _write_odom_row(self, writer: csv.writer, msg: Odometry) -> None:
        sec, nanosec = _stamp_to_pair(msg.header.stamp)
        yaw_rad = _quat_to_yaw(
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
            float(msg.pose.pose.orientation.w),
        )
        writer.writerow(
            [
                sec,
                nanosec,
                msg.header.frame_id,
                msg.child_frame_id,
                f"{float(msg.pose.pose.position.x):.9f}",
                f"{float(msg.pose.pose.position.y):.9f}",
                f"{yaw_rad:.9f}",
                f"{float(msg.twist.twist.linear.x):.9f}",
                f"{float(msg.twist.twist.linear.y):.9f}",
                f"{float(msg.twist.twist.angular.z):.9f}",
            ]
        )

    def _write_ground_truth_path(self) -> None:
        if self._latest_ground_truth_path is None:
            return
        path = self._run_dir / "ground_truth_path.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["pose_index", "stamp_sec", "stamp_nanosec", "x_m", "y_m", "yaw_rad"]
            )
            for pose_index, pose_stamped in enumerate(self._latest_ground_truth_path.poses):
                pose_stamp = pose_stamped.header.stamp
                if int(pose_stamp.sec) == 0 and int(pose_stamp.nanosec) == 0:
                    pose_stamp = self._latest_ground_truth_path.header.stamp
                sec, nanosec = _stamp_to_pair(pose_stamp)
                yaw_rad = _quat_to_yaw(
                    float(pose_stamped.pose.orientation.x),
                    float(pose_stamped.pose.orientation.y),
                    float(pose_stamped.pose.orientation.z),
                    float(pose_stamped.pose.orientation.w),
                )
                writer.writerow(
                    [
                        pose_index,
                        sec,
                        nanosec,
                        f"{float(pose_stamped.pose.position.x):.9f}",
                        f"{float(pose_stamped.pose.position.y):.9f}",
                        f"{yaw_rad:.9f}",
                    ]
                )

    def _log_progress(self) -> None:
        self.get_logger().info(
            "Recording run %s (imu=%d scans=%d lidar_points=%d odom=%d gt_odom=%d track_points=%d)"
            % (
                self._run_dir.name,
                self._imu_count,
                self._scan_count,
                self._lidar_point_count,
                self._odom_count,
                self._ground_truth_odom_count,
                self._track_geometry_count,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._write_ground_truth_path()
        summary = {
            "run_dir": str(self._run_dir),
            "imu_sample_count": self._imu_count,
            "scan_count": self._scan_count,
            "lidar_point_count": self._lidar_point_count,
            "odom_count": self._odom_count,
            "ground_truth_odom_count": self._ground_truth_odom_count,
            "ground_truth_status_count": self._ground_truth_status_count,
            "online_status_count": self._online_status_count,
            "track_geometry_point_count": self._track_geometry_count,
            "files": {
                "imu_raw_csv": "imu_raw.csv",
                "lidar_points_csv": "lidar_points.csv",
                "scan_index_csv": "scan_index.csv",
                "odom_fused_csv": "odom_fused.csv",
                "ground_truth_odom_csv": "ground_truth_odom.csv",
                "track_geometry_csv": "track_geometry.csv",
                "ground_truth_path_csv": "ground_truth_path.csv",
                "ground_truth_status_jsonl": "ground_truth_status.jsonl",
                "online_status_jsonl": "online_status.jsonl",
                "run_metadata_json": "run_metadata.json",
                "run_summary_json": "run_summary.json",
            },
        }
        (self._run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        for handle in (
            self._imu_handle,
            self._lidar_handle,
            self._scan_index_handle,
            self._odom_handle,
            self._ground_truth_odom_handle,
            self._ground_truth_status_handle,
            self._online_status_handle,
        ):
            handle.flush()
            handle.close()
        self._closed = True

    def destroy_node(self) -> bool:
        self.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ApexSimRunRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
