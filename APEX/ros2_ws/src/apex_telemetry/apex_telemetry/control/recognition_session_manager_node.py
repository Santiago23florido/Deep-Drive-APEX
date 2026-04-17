#!/usr/bin/env python3
"""Manage real recognition-tour sessions for the Raspberry Pi pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, TextIO

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


TRACKER_TERMINAL_STATES = {
    "loop_closed",
    "timeout",
    "aborted_low_confidence",
    "aborted_path_loss",
    "aborted_odom_timeout",
    "planner_failed",
}
PLANNER_TERMINAL_STATES = {
    "loop_closed",
    "timeout",
}
PLANNER_FAILURE_STATES = {
    "error",
}
MANUAL_RELAY_STATES = {
    "idle",
    "completed",
    "error",
}
ACTIVE_SESSION_STATES = {
    "arming",
    "running",
    "stopping",
    "building_map",
}


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[Any]
    log_handle: TextIO
    log_path: Path
    started_monotonic: float


class RecognitionSessionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("recognition_session_manager_node")

        self.declare_parameter("manual_control_status_topic", "/apex/manual_control/status")
        self.declare_parameter("session_toggle_topic", "/apex/manual_control/session_toggle")
        self.declare_parameter("arm_topic", "/apex/tracking/arm")
        self.declare_parameter("cmd_vel_topic", "/apex/cmd_vel_track")
        self.declare_parameter("session_status_topic", "/apex/recognition_session/status")
        self.declare_parameter("kinematics_status_topic", "/apex/kinematics/status")
        self.declare_parameter("fusion_status_topic", "/apex/estimation/status")
        self.declare_parameter("planner_status_topic", "/apex/planning/recognition_tour_status")
        self.declare_parameter("tracker_status_topic", "/apex/tracking/recognition_tour_status")
        self.declare_parameter("drive_bridge_status_topic", "/apex/vehicle/drive_bridge_status")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("status_publish_interval_s", 0.25)
        self.declare_parameter("manual_status_stale_s", 1.0)
        self.declare_parameter("bridge_disconnect_hold_s", 0.75)
        self.declare_parameter("recorder_shutdown_grace_s", 20.0)
        self.declare_parameter("run_id_prefix", "recognition_tour")
        self.declare_parameter("run_root_dir", "/work/ros2_ws/apex_recognition_tour")
        self.declare_parameter(
            "runtime_status_path",
            "/work/repo/APEX/.apex_runtime/real_session/status.json",
        )
        self.declare_parameter(
            "sensor_capture_script_path",
            "/work/ros2_ws/scripts/capture/record_manual_sensorfusion_capture.py",
        )
        self.declare_parameter(
            "recognition_recorder_script_path",
            "/work/ros2_ws/scripts/capture/record_recognition_tour.py",
        )
        self.declare_parameter(
            "mapper_script_path",
            "/work/repo/src/rc_sim_description/scripts/apex_general_track_mapper.py",
        )
        self.declare_parameter(
            "evaluation_world",
            "/work/repo/src/rc_sim_description/worlds/basic_track.world",
        )
        self.declare_parameter("record_timeout_s", 60.0)
        self.declare_parameter("publish_tracker_arm", False)
        self.declare_parameter("sensor_capture_max_total_bytes", 134217728)
        self.declare_parameter("recognition_recorder_max_total_bytes", 134217728)
        self.declare_parameter("run_max_total_bytes", 268435456)
        self.declare_parameter("min_free_disk_bytes", 536870912)
        self.declare_parameter("storage_check_interval_s", 1.0)
        self.declare_parameter("lidar_offset_x_m", 0.18)
        self.declare_parameter("lidar_offset_y_m", 0.0)
        self.declare_parameter("rear_axle_offset_x_m", -0.15)
        self.declare_parameter("rear_axle_offset_y_m", 0.0)
        self.declare_parameter("planner_wait_ready_states", "waiting_arm,tracking")

        self._manual_control_status_topic = str(
            self.get_parameter("manual_control_status_topic").value
        )
        self._session_toggle_topic = str(self.get_parameter("session_toggle_topic").value)
        self._arm_topic = str(self.get_parameter("arm_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._session_status_topic = str(self.get_parameter("session_status_topic").value)
        self._kinematics_status_topic = str(self.get_parameter("kinematics_status_topic").value)
        self._fusion_status_topic = str(self.get_parameter("fusion_status_topic").value)
        self._planner_status_topic = str(self.get_parameter("planner_status_topic").value)
        self._tracker_status_topic = str(self.get_parameter("tracker_status_topic").value)
        self._drive_bridge_status_topic = str(
            self.get_parameter("drive_bridge_status_topic").value
        )
        self._control_rate_hz = max(2.0, float(self.get_parameter("control_rate_hz").value))
        self._status_publish_interval_s = max(
            0.1, float(self.get_parameter("status_publish_interval_s").value)
        )
        self._manual_status_stale_s = max(
            0.1, float(self.get_parameter("manual_status_stale_s").value)
        )
        self._bridge_disconnect_hold_s = max(
            0.1, float(self.get_parameter("bridge_disconnect_hold_s").value)
        )
        self._recorder_shutdown_grace_s = max(
            1.0, float(self.get_parameter("recorder_shutdown_grace_s").value)
        )
        self._run_id_prefix = str(self.get_parameter("run_id_prefix").value).strip() or "recognition_tour"
        self._run_root_dir = Path(
            str(self.get_parameter("run_root_dir").value)
        ).expanduser().resolve()
        self._runtime_status_path = Path(
            str(self.get_parameter("runtime_status_path").value)
        ).expanduser().resolve()
        self._sensor_capture_script_path = Path(
            str(self.get_parameter("sensor_capture_script_path").value)
        ).expanduser().resolve()
        self._recognition_recorder_script_path = Path(
            str(self.get_parameter("recognition_recorder_script_path").value)
        ).expanduser().resolve()
        self._mapper_script_path = Path(
            str(self.get_parameter("mapper_script_path").value)
        ).expanduser().resolve()
        self._evaluation_world = Path(
            str(self.get_parameter("evaluation_world").value)
        ).expanduser().resolve()
        self._record_timeout_s = max(5.0, float(self.get_parameter("record_timeout_s").value))
        self._publish_tracker_arm = bool(self.get_parameter("publish_tracker_arm").value)
        self._sensor_capture_max_total_bytes = max(
            0, int(self.get_parameter("sensor_capture_max_total_bytes").value)
        )
        self._recognition_recorder_max_total_bytes = max(
            0, int(self.get_parameter("recognition_recorder_max_total_bytes").value)
        )
        self._run_max_total_bytes = max(0, int(self.get_parameter("run_max_total_bytes").value))
        self._min_free_disk_bytes = max(0, int(self.get_parameter("min_free_disk_bytes").value))
        self._storage_check_interval_s = max(
            0.25, float(self.get_parameter("storage_check_interval_s").value)
        )
        self._lidar_offset_x_m = float(self.get_parameter("lidar_offset_x_m").value)
        self._lidar_offset_y_m = float(self.get_parameter("lidar_offset_y_m").value)
        self._rear_axle_offset_x_m = float(self.get_parameter("rear_axle_offset_x_m").value)
        self._rear_axle_offset_y_m = float(self.get_parameter("rear_axle_offset_y_m").value)
        self._planner_wait_ready_states = {
            item.strip()
            for item in str(self.get_parameter("planner_wait_ready_states").value).split(",")
            if item.strip()
        }

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        arm_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(String, self._manual_control_status_topic, self._manual_status_cb, 20)
        self.create_subscription(String, self._kinematics_status_topic, self._kinematics_status_cb, 20)
        self.create_subscription(String, self._fusion_status_topic, self._fusion_status_cb, 20)
        self.create_subscription(String, self._planner_status_topic, self._planner_status_cb, latched_qos)
        self.create_subscription(String, self._tracker_status_topic, self._tracker_status_cb, latched_qos)
        self.create_subscription(String, self._drive_bridge_status_topic, self._drive_bridge_status_cb, 20)
        self.create_subscription(Bool, self._session_toggle_topic, self._session_toggle_cb, arm_qos)

        self._arm_pub = self.create_publisher(Bool, self._arm_topic, 10)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 20)
        self._status_pub = self.create_publisher(String, self._session_status_topic, latched_qos)
        self.create_timer(1.0 / self._control_rate_hz, self._control_step)

        self._state = "idle"
        self._state_entered_monotonic = time.monotonic()
        self._requested_stop_cause: str | None = None
        self._current_run_id = ""
        self._current_run_dir: Path | None = None
        self._completed_run_id = ""
        self._failed_run_id = ""
        self._end_cause = ""
        self._current_session_started_monotonic: float | None = None
        self._bridge_disconnect_since_monotonic: float | None = None
        self._last_manual_status: dict[str, Any] = {}
        self._last_manual_status_monotonic = 0.0
        self._kinematics_status: dict[str, Any] = {}
        self._fusion_status: dict[str, Any] = {}
        self._planner_status: dict[str, Any] = {}
        self._tracker_status: dict[str, Any] = {}
        self._drive_bridge_status: dict[str, Any] = {}
        self._sensor_capture: ManagedProcess | None = None
        self._recognition_recorder: ManagedProcess | None = None
        self._mapper_process: ManagedProcess | None = None
        self._stopping_started_monotonic: float | None = None
        self._last_status_publish_monotonic = 0.0
        self._last_storage_check_monotonic = 0.0
        self._current_run_total_bytes = 0
        self._current_run_free_disk_bytes: int | None = None

        self._run_root_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_status_path.parent.mkdir(parents=True, exist_ok=True)
        self._publish_arm(False)
        self._publish_cmd(0.0, 0.0)
        self._flush_status(force=True)

        self.get_logger().info(
            "RecognitionSessionManagerNode started (run_root=%s runtime_status=%s)"
            % (str(self._run_root_dir), str(self._runtime_status_path))
        )

    def _manual_status_cb(self, msg: String) -> None:
        payload = self._parse_json_dict(msg.data)
        self._last_manual_status = payload
        self._last_manual_status_monotonic = time.monotonic()

    def _kinematics_status_cb(self, msg: String) -> None:
        self._kinematics_status = self._parse_json_dict(msg.data)

    def _fusion_status_cb(self, msg: String) -> None:
        self._fusion_status = self._parse_json_dict(msg.data)

    def _planner_status_cb(self, msg: String) -> None:
        self._planner_status = self._parse_json_dict(msg.data)

    def _tracker_status_cb(self, msg: String) -> None:
        self._tracker_status = self._parse_json_dict(msg.data)

    def _drive_bridge_status_cb(self, msg: String) -> None:
        self._drive_bridge_status = self._parse_json_dict(msg.data)

    def _session_toggle_cb(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        if self._state in {"idle", "completed", "error"}:
            self._requested_stop_cause = None
            self._set_state("pending_start")
            self._end_cause = ""
            self._failed_run_id = ""
            self.get_logger().info("Recognition session start requested.")
        elif self._state in {"arming", "running"}:
            self._requested_stop_cause = "manual_toggle"
            self.get_logger().info("Recognition session stop requested by controller toggle.")
        else:
            self.get_logger().info("Ignoring session toggle while state=%s" % self._state)

    def _control_step(self) -> None:
        now_monotonic = time.monotonic()

        if self._state in MANUAL_RELAY_STATES:
            linear_x_mps, angular_z_rps = self._manual_cmd()
            self._publish_cmd(linear_x_mps, angular_z_rps)
        elif self._state in {"pending_start", "stopping", "building_map"}:
            self._publish_cmd(0.0, 0.0)

        if self._state == "pending_start":
            self._publish_arm(False)
            if self._system_ready_for_start():
                self._start_session()
        elif self._state == "arming":
            self._publish_arm(self._publish_tracker_arm)
            stop_cause = self._active_stop_cause(now_monotonic)
            if stop_cause is not None:
                self._begin_stop(stop_cause)
            elif not self._publish_tracker_arm or self._arming_confirmed():
                self._set_state("running")
        elif self._state == "running":
            self._publish_arm(self._publish_tracker_arm)
            stop_cause = self._active_stop_cause(now_monotonic)
            if stop_cause is not None:
                self._begin_stop(stop_cause)
        elif self._state == "stopping":
            self._publish_arm(False)
            self._advance_stopping(now_monotonic)
        elif self._state == "building_map":
            self._publish_arm(False)
            self._advance_mapper()
        else:
            self._publish_arm(False)

        if (now_monotonic - self._last_status_publish_monotonic) >= self._status_publish_interval_s:
            self._flush_status(force=False)
            self._last_status_publish_monotonic = now_monotonic

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._state_entered_monotonic = time.monotonic()

    def _system_ready_for_start(self) -> bool:
        if not bool(self._kinematics_status.get("calibration_complete", False)):
            return False
        if bool(self._kinematics_status.get("calibration_active", False)):
            return False
        if not bool(self._fusion_status.get("alignment_ready", False)):
            return False
        if str(self._fusion_status.get("state", "")) != "tracking":
            return False
        planner_state = str(self._planner_status.get("state", ""))
        return planner_state in self._planner_wait_ready_states

    def _arming_confirmed(self) -> bool:
        planner_armed = bool(self._planner_status.get("armed", False))
        tracker_armed = bool(self._tracker_status.get("armed", False))
        tracker_state = str(self._tracker_status.get("state", ""))
        return planner_armed and (tracker_armed or tracker_state == "tracking")

    def _manual_status_is_fresh(self) -> bool:
        if self._last_manual_status_monotonic <= 0.0:
            return False
        return (time.monotonic() - self._last_manual_status_monotonic) <= self._manual_status_stale_s

    def _manual_cmd(self) -> tuple[float, float]:
        if not self._manual_status_is_fresh():
            return (0.0, 0.0)
        if not bool(self._last_manual_status.get("bridge_connected", False)):
            return (0.0, 0.0)
        if not bool(self._last_manual_status.get("controller_connected", False)):
            return (0.0, 0.0)
        if not bool(self._last_manual_status.get("enabled", False)):
            return (0.0, 0.0)
        return (
            float(self._last_manual_status.get("linear_x_mps", 0.0)),
            float(self._last_manual_status.get("angular_z_rps", 0.0)),
        )

    def _active_stop_cause(self, now_monotonic: float) -> str | None:
        if self._requested_stop_cause is not None:
            cause = self._requested_stop_cause
            self._requested_stop_cause = None
            return cause

        storage_cause = self._storage_stop_cause(now_monotonic)
        if storage_cause is not None:
            return storage_cause

        if not self._manual_status_is_fresh() or not bool(
            self._last_manual_status.get("bridge_connected", False)
        ):
            if self._bridge_disconnect_since_monotonic is None:
                self._bridge_disconnect_since_monotonic = now_monotonic
            elif (
                now_monotonic - self._bridge_disconnect_since_monotonic
            ) >= self._bridge_disconnect_hold_s:
                return "bridge_disconnect"
        else:
            self._bridge_disconnect_since_monotonic = None

        tracker_state = str(self._tracker_status.get("state", ""))
        if tracker_state in TRACKER_TERMINAL_STATES:
            return tracker_state

        planner_state = str(self._planner_status.get("state", ""))
        if planner_state in PLANNER_TERMINAL_STATES:
            return planner_state
        if planner_state in PLANNER_FAILURE_STATES:
            return "planner_failed"

        recorder_limit_cause = self._recorder_limit_cause()
        if recorder_limit_cause is not None:
            return recorder_limit_cause

        if self._sensor_capture is not None and self._sensor_capture.process.poll() is not None:
            return "sensor_capture_exit"
        if self._recognition_recorder is not None and self._recognition_recorder.process.poll() is not None:
            if tracker_state in TRACKER_TERMINAL_STATES:
                return tracker_state
            return "recognition_recorder_exit"
        return None

    def _start_session(self) -> None:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self._current_run_id = f"{self._run_id_prefix}_{timestamp}"
        self._current_run_dir = (self._run_root_dir / self._current_run_id).resolve()
        self._current_run_dir.mkdir(parents=True, exist_ok=True)
        self._completed_run_id = ""
        self._failed_run_id = ""
        self._end_cause = ""
        self._current_session_started_monotonic = time.monotonic()
        self._bridge_disconnect_since_monotonic = None
        self._stopping_started_monotonic = None
        self._last_storage_check_monotonic = 0.0
        self._current_run_total_bytes = 0
        self._current_run_free_disk_bytes = None

        bridge_min_effective_speed_pct = self._drive_bridge_min_effective_speed_pct()
        self._write_capture_meta(
            {
                "run_id": self._current_run_id,
                "mode": "recognition_tour_real_session",
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "record_timeout_s": self._record_timeout_s,
                "bridge_min_effective_speed_pct": bridge_min_effective_speed_pct,
                "sensor_capture_max_total_bytes": self._sensor_capture_max_total_bytes,
                "recognition_recorder_max_total_bytes": self._recognition_recorder_max_total_bytes,
                "run_max_total_bytes": self._run_max_total_bytes,
                "min_free_disk_bytes": self._min_free_disk_bytes,
            }
        )

        sensor_log = self._current_run_dir / "sensor_capture.log"
        recognition_log = self._current_run_dir / "recognition_tour_record.log"

        try:
            self._sensor_capture = self._start_process(
                "sensor_capture",
                [
                    os.environ.get("PYTHON", "python3"),
                    str(self._sensor_capture_script_path),
                    "--imu-topic",
                    "/apex/imu/data_raw",
                    "--scan-topic",
                    "/lidar/scan_localization",
                    "--odom-topic",
                    "/apex/odometry/imu_lidar_fused",
                    "--imu-output",
                    str(self._current_run_dir / "imu_raw.csv"),
                    "--lidar-output",
                    str(self._current_run_dir / "raw_lidar_points.csv"),
                    "--odom-output",
                    str(self._current_run_dir / "odom_fused.csv"),
                    "--summary-json",
                    str(self._current_run_dir / "sensor_capture_summary.json"),
                    "--status-json",
                    str(self._current_run_dir / "sensor_capture_status.json"),
                    "--max-total-bytes",
                    str(self._sensor_capture_max_total_bytes),
                    "--min-free-disk-bytes",
                    str(self._min_free_disk_bytes),
                ],
                sensor_log,
            )
            self._recognition_recorder = self._start_process(
                "recognition_recorder",
                [
                    os.environ.get("PYTHON", "python3"),
                    str(self._recognition_recorder_script_path),
                    "--path-topic",
                    "/apex/planning/recognition_tour_local_path",
                    "--route-topic",
                    "/apex/planning/recognition_tour_route",
                    "--fusion-status-topic",
                    "/apex/estimation/status",
                    "--planner-status-topic",
                    "/apex/planning/recognition_tour_status",
                    "--tracker-status-topic",
                    "/apex/tracking/recognition_tour_status",
                    "--bridge-status-topic",
                    str(self._drive_bridge_status_topic),
                    "--odom-topic",
                    "/apex/odometry/imu_lidar_fused",
                    "--scan-topic",
                    "/lidar/scan_localization",
                    "--output-dir",
                    str(self._current_run_dir),
                    "--timeout-s",
                    str(self._record_timeout_s),
                    "--lidar-offset-x-m",
                    str(self._lidar_offset_x_m),
                    "--lidar-offset-y-m",
                    str(self._lidar_offset_y_m),
                    "--rear-axle-offset-x-m",
                    str(self._rear_axle_offset_x_m),
                    "--rear-axle-offset-y-m",
                    str(self._rear_axle_offset_y_m),
                    "--max-total-bytes",
                    str(self._recognition_recorder_max_total_bytes),
                    "--min-free-disk-bytes",
                    str(self._min_free_disk_bytes),
                ],
                recognition_log,
            )
        except Exception as exc:
            self.get_logger().error("Failed to start recognition session recorders: %s" % exc)
            self._signal_process(self._sensor_capture, signal.SIGTERM)
            self._signal_process(self._recognition_recorder, signal.SIGTERM)
            self._close_managed_process(self._sensor_capture)
            self._close_managed_process(self._recognition_recorder)
            self._sensor_capture = None
            self._recognition_recorder = None
            self._set_error("recorder_start_failed")
            return

        self._set_state("arming" if self._publish_tracker_arm else "running")
        self.get_logger().info("Recognition session started: %s" % self._current_run_dir)
        self._flush_status(force=True)

    def _begin_stop(self, cause: str) -> None:
        if self._state not in {"arming", "running"}:
            return
        self._end_cause = str(cause)
        self._set_state("stopping")
        self._stopping_started_monotonic = time.monotonic()
        self._publish_arm(False)
        self._publish_cmd(0.0, 0.0)
        self._signal_process(self._sensor_capture, signal.SIGINT)
        self._signal_process(self._recognition_recorder, signal.SIGINT)
        self.get_logger().info("Stopping recognition session (cause=%s)" % self._end_cause)
        self._flush_status(force=True)

    def _advance_stopping(self, now_monotonic: float) -> None:
        self._publish_cmd(0.0, 0.0)
        all_exited = True
        for managed in (self._sensor_capture, self._recognition_recorder):
            if managed is None:
                continue
            if managed.process.poll() is None:
                all_exited = False
                if self._stopping_started_monotonic is not None and (
                    now_monotonic - self._stopping_started_monotonic
                ) >= self._recorder_shutdown_grace_s:
                    self._signal_process(managed, signal.SIGTERM)
        if not all_exited:
            return

        self._close_managed_process(self._sensor_capture)
        self._close_managed_process(self._recognition_recorder)
        self._sensor_capture = None
        self._recognition_recorder = None
        self._patch_recognition_summary_end_cause()
        self._start_mapper()

    def _start_mapper(self) -> None:
        if self._current_run_dir is None:
            self._set_error("mapper_missing_run_dir")
            return
        fixed_map_dir = self._current_run_dir / "fixed_map"
        mapper_log = self._current_run_dir / "mapping_build.log"
        self._mapper_process = self._start_process(
            "fixed_map_builder",
            [
                os.environ.get("PYTHON", "python3"),
                str(self._mapper_script_path),
                "--run-dir",
                str(self._current_run_dir),
                "--output-dir",
                str(fixed_map_dir),
                "--status-json",
                str(fixed_map_dir / "build_status.json"),
                "--evaluation-world",
                str(self._evaluation_world),
                "--evaluation-json",
                str(fixed_map_dir / "mapping_evaluation.json"),
            ],
            mapper_log,
        )
        self._set_state("building_map")
        self.get_logger().info("Started fixed-map build for %s" % self._current_run_dir)
        self._flush_status(force=True)

    def _advance_mapper(self) -> None:
        if self._mapper_process is None:
            self._set_error("mapper_process_missing")
            return
        exit_code = self._mapper_process.process.poll()
        if exit_code is None:
            return
        self._close_managed_process(self._mapper_process)
        self._mapper_process = None

        if exit_code != 0:
            self._set_error("fixed_map_build_failed")
            return

        if self._current_run_dir is None:
            self._set_error("completed_run_dir_missing")
            return

        fixed_map_dir = self._current_run_dir / "fixed_map"
        fixed_map_yaml = fixed_map_dir / "fixed_map.yaml"
        fixed_map_pgm = fixed_map_dir / "fixed_map.pgm"
        if not fixed_map_yaml.exists() or not fixed_map_pgm.exists():
            self._set_error("fixed_map_outputs_missing")
            return

        self._completed_run_id = self._current_run_id
        self._write_capture_meta(
            {
                "run_id": self._current_run_id,
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "end_cause": self._end_cause,
                "fixed_map_yaml": str(fixed_map_yaml),
                "fixed_map_pgm": str(fixed_map_pgm),
            },
            merge=True,
        )
        self._set_state("completed")
        self.get_logger().info(
            "Recognition session completed: %s (cause=%s)"
            % (self._completed_run_id, self._end_cause)
        )
        self._flush_status(force=True)

    def _set_error(self, cause: str) -> None:
        self._end_cause = str(cause)
        if self._current_run_id:
            self._failed_run_id = self._current_run_id
        self._set_state("error")
        self.get_logger().error("Recognition session error: %s" % self._end_cause)
        self._flush_status(force=True)

    def _start_process(self, name: str, argv: list[str], log_path: Path) -> ManagedProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            argv,
            cwd=str(self._run_root_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return ManagedProcess(
            name=name,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            started_monotonic=time.monotonic(),
        )

    def _signal_process(self, managed: ManagedProcess | None, signal_value: int) -> None:
        if managed is None:
            return
        if managed.process.poll() is not None:
            return
        try:
            managed.process.send_signal(signal_value)
        except Exception:
            pass

    def _close_managed_process(self, managed: ManagedProcess | None) -> None:
        if managed is None:
            return
        try:
            if managed.process.poll() is None:
                managed.process.wait(timeout=0.1)
        except Exception:
            pass
        try:
            managed.log_handle.flush()
            managed.log_handle.close()
        except Exception:
            pass

    def _publish_cmd(self, linear_x_mps: float, angular_z_rps: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x_mps)
        msg.angular.z = float(angular_z_rps)
        self._cmd_pub.publish(msg)

    def _publish_arm(self, armed: bool) -> None:
        msg = Bool()
        msg.data = bool(armed)
        self._arm_pub.publish(msg)

    def _capture_meta_path(self) -> Path | None:
        if self._current_run_dir is None:
            return None
        return self._current_run_dir / "capture_meta.json"

    def _write_capture_meta(self, payload: dict[str, Any], *, merge: bool = False) -> None:
        path = self._capture_meta_path()
        if path is None:
            return
        current = {}
        if merge and path.exists():
            current = self._read_json_file(path)
            if not isinstance(current, dict):
                current = {}
        current.update(payload)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def _patch_recognition_summary_end_cause(self) -> None:
        if self._current_run_dir is None or not self._end_cause:
            return
        summary_path = (
            self._current_run_dir / "analysis_recognition_tour" / "recognition_tour_summary.json"
        )
        payload = self._read_json_file(summary_path)
        if not isinstance(payload, dict):
            return
        payload["end_cause"] = self._end_cause
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _drive_bridge_min_effective_speed_pct(self) -> float | None:
        value = self._drive_bridge_status.get("min_effective_speed_pct")
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def _storage_stop_cause(self, now_monotonic: float) -> str | None:
        if self._current_run_dir is None:
            return None
        if self._run_max_total_bytes > 0 and self._current_run_total_bytes >= self._run_max_total_bytes:
            return "storage_limit"
        if (
            self._min_free_disk_bytes > 0
            and self._current_run_free_disk_bytes is not None
            and self._current_run_free_disk_bytes <= self._min_free_disk_bytes
        ):
            return "low_disk_space"
        if (now_monotonic - self._last_storage_check_monotonic) < self._storage_check_interval_s:
            return None
        self._last_storage_check_monotonic = now_monotonic
        self._current_run_total_bytes = self._measure_directory_bytes(self._current_run_dir)
        self._current_run_free_disk_bytes = self._free_disk_bytes(self._current_run_dir)
        if self._run_max_total_bytes > 0 and self._current_run_total_bytes >= self._run_max_total_bytes:
            self.get_logger().error(
                "Stopping recognition session because run_total_bytes=%d reached run_max_total_bytes=%d"
                % (self._current_run_total_bytes, self._run_max_total_bytes)
            )
            return "storage_limit"
        if (
            self._min_free_disk_bytes > 0
            and self._current_run_free_disk_bytes is not None
            and self._current_run_free_disk_bytes <= self._min_free_disk_bytes
        ):
            self.get_logger().error(
                "Stopping recognition session because free_disk_bytes=%d fell below min_free_disk_bytes=%d"
                % (self._current_run_free_disk_bytes, self._min_free_disk_bytes)
            )
            return "low_disk_space"
        return None

    def _recorder_limit_cause(self) -> str | None:
        for path in (
            self._sensor_capture_summary_path(),
            self._recognition_summary_path(),
        ):
            if path is None or not path.exists():
                continue
            payload = self._read_json_file(path)
            if not isinstance(payload, dict):
                continue
            end_cause = str(payload.get("end_cause", "")).strip()
            if end_cause in {"storage_limit", "low_disk_space"}:
                return end_cause
        return None

    def _sensor_capture_summary_path(self) -> Path | None:
        if self._current_run_dir is None:
            return None
        return self._current_run_dir / "sensor_capture_summary.json"

    def _recognition_summary_path(self) -> Path | None:
        if self._current_run_dir is None:
            return None
        return self._current_run_dir / "analysis_recognition_tour" / "recognition_tour_summary.json"

    @staticmethod
    def _measure_directory_bytes(root: Path) -> int:
        total_bytes = 0
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                total_bytes += int(entry.stat(follow_symlinks=False).st_size)
                        except FileNotFoundError:
                            continue
            except FileNotFoundError:
                continue
        return total_bytes

    @staticmethod
    def _free_disk_bytes(path: Path) -> int | None:
        try:
            stats = os.statvfs(str(path))
        except Exception:
            return None
        return int(stats.f_bavail) * int(stats.f_frsize)

    def _session_payload(self) -> dict[str, Any]:
        session_elapsed_s = None
        if self._current_session_started_monotonic is not None:
            session_elapsed_s = round(
                max(0.0, time.monotonic() - self._current_session_started_monotonic),
                3,
            )
        payload = {
            "state": self._state,
            "ready_for_start": self._system_ready_for_start(),
            "current_run_id": self._current_run_id,
            "current_run_dir": str(self._current_run_dir) if self._current_run_dir else "",
            "completed_run_id": self._completed_run_id,
            "failed_run_id": self._failed_run_id,
            "end_cause": self._end_cause,
            "session_elapsed_s": session_elapsed_s,
            "bridge_connected": bool(self._last_manual_status.get("bridge_connected", False))
            and self._manual_status_is_fresh(),
            "controller_connected": bool(self._last_manual_status.get("controller_connected", False)),
            "manual_control": {
                "state": self._last_manual_status.get("state", ""),
                "enabled": bool(self._last_manual_status.get("enabled", False)),
                "linear_x_mps": float(self._last_manual_status.get("linear_x_mps", 0.0) or 0.0),
                "angular_z_rps": float(self._last_manual_status.get("angular_z_rps", 0.0) or 0.0),
                "session_toggle_count": int(
                    self._last_manual_status.get("session_toggle_count", 0) or 0
                ),
            },
            "kinematics": {
                "calibration_complete": bool(
                    self._kinematics_status.get("calibration_complete", False)
                ),
                "calibration_active": bool(
                    self._kinematics_status.get("calibration_active", False)
                ),
            },
            "fusion": {
                "state": str(self._fusion_status.get("state", "")),
                "alignment_ready": bool(self._fusion_status.get("alignment_ready", False)),
            },
            "planner": {
                "state": str(self._planner_status.get("state", "")),
                "armed": bool(self._planner_status.get("armed", False)),
            },
            "tracker": {
                "state": str(self._tracker_status.get("state", "")),
                "armed": bool(self._tracker_status.get("armed", False)),
            },
            "drive_bridge": {
                "state": str(self._drive_bridge_status.get("state", "")),
                "timed_out": bool(self._drive_bridge_status.get("timed_out", False)),
                "min_effective_speed_pct": self._drive_bridge_min_effective_speed_pct(),
                "max_speed_pct": self._drive_bridge_status.get("max_speed_pct"),
            },
            "processes": {
                "sensor_capture_pid": (
                    int(self._sensor_capture.process.pid) if self._sensor_capture is not None else None
                ),
                "recognition_recorder_pid": (
                    int(self._recognition_recorder.process.pid)
                    if self._recognition_recorder is not None
                    else None
                ),
                "fixed_map_builder_pid": (
                    int(self._mapper_process.process.pid) if self._mapper_process is not None else None
                ),
            },
            "paths": {
                "runtime_status_path": str(self._runtime_status_path),
                "run_root_dir": str(self._run_root_dir),
                "fixed_map_yaml": (
                    str(self._current_run_dir / "fixed_map" / "fixed_map.yaml")
                    if self._current_run_dir is not None
                    else ""
                ),
                "fixed_map_pgm": (
                    str(self._current_run_dir / "fixed_map" / "fixed_map.pgm")
                    if self._current_run_dir is not None
                    else ""
                ),
            },
            "storage": {
                "run_total_bytes": self._current_run_total_bytes,
                "run_max_total_bytes": self._run_max_total_bytes,
                "free_disk_bytes": self._current_run_free_disk_bytes,
                "min_free_disk_bytes": self._min_free_disk_bytes,
                "sensor_capture_max_total_bytes": self._sensor_capture_max_total_bytes,
                "recognition_recorder_max_total_bytes": self._recognition_recorder_max_total_bytes,
            },
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return payload

    def _flush_status(self, *, force: bool) -> None:
        payload = self._session_payload()
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        self._runtime_status_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_status_path.write_text(encoded, encoding="utf-8")
        if force or (time.monotonic() - self._last_status_publish_monotonic) >= self._status_publish_interval_s:
            msg = String()
            msg.data = json.dumps(payload, separators=(",", ":"))
            self._status_pub.publish(msg)

    @staticmethod
    def _parse_json_dict(text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _read_json_file(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def destroy_node(self) -> bool:
        try:
            self._publish_arm(False)
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass
        for managed in (self._sensor_capture, self._recognition_recorder, self._mapper_process):
            self._signal_process(managed, signal.SIGTERM)
        for managed in (self._sensor_capture, self._recognition_recorder, self._mapper_process):
            self._close_managed_process(managed)
        try:
            self._flush_status(force=True)
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RecognitionSessionManagerNode()
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
