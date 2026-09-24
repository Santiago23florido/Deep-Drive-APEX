"""Gazebo simulation + sensor-fusion research stack.

Starts the standard APEX simulation (``rc_sim_description/apex_sim.launch.py``)
with *ideal* Gazebo sensors, and adds the research nodes on top of it:

    Gazebo ideal IMU  --> fusion_imu_sensor     --> /apex/fusion/imu/data_raw --> fusion_strapdown_ins
    Gazebo ideal scan --> fusion_lidar_noise    --> /apex/fusion/scan_noisy
    Gazebo poses      --> fusion_truth          --> /apex/fusion/truth/odom
    truth + INS       --> fusion_ins_error_monitor (errors, CSV, metadata)
    scan + poses      --> fusion_scan_projector (RViz point clouds)

The vehicle keeps using its own control stack; the research nodes only observe.

Example:
    ros2 launch apex_fusion_research fusion_research_sim.launch.py \
        imu_config:=/path/to/my_imu.yaml imu_seed:=3 rviz:=true
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "apex_fusion_research"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_config(value: str, share: Path, kind: str) -> Path:
    """Accept a preset name (e.g. ``consumer_mems``) or a path to a YAML file."""
    candidate = Path(value).expanduser()
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate.resolve()
    preset = share / "config" / kind / f"{value}.yaml"
    if preset.exists():
        return preset
    raise FileNotFoundError(f"{kind} config '{value}' is neither a file nor a preset in {preset.parent}")


def _prepare(context, *args, **kwargs):
    share = Path(get_package_share_directory(PKG))
    arg = lambda name: LaunchConfiguration(name).perform(context).strip()  # noqa: E731

    imu_cfg = _resolve_config(arg("imu_config"), share, "imu")
    lidar_cfg = _resolve_config(arg("lidar_config"), share, "lidar")
    ins_cfg = _resolve_config(arg("ins_config"), share, "ins")
    imu_overrides = {"seed": int(arg("imu_seed"))} if int(arg("imu_seed")) >= 0 else {}
    lidar_overrides = {"seed": int(arg("lidar_seed"))} if int(arg("lidar_seed")) >= 0 else {}

    output_dir = ""
    actions = []
    if _truthy(arg("record")):
        root = arg("output_root") or os.path.join(
            os.environ.get("APEX_SIM_ROOT", str(Path.home() / ".ros")), "data", "fusion_research"
        )
        run_name = arg("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(root).expanduser() / run_name
        (run_dir / "config").mkdir(parents=True, exist_ok=True)
        for cfg in (imu_cfg, lidar_cfg, ins_cfg):
            shutil.copy2(cfg, run_dir / "config" / f"{cfg.parent.name}_{cfg.name}")
        launch_args = {
            name: arg(name)
            for name in ("scenario", "control_mode", "imu_config", "lidar_config", "ins_config", "imu_seed", "lidar_seed")
        }
        (run_dir / "launch_args.json").write_text(json.dumps(launch_args, indent=2), encoding="utf-8")
        output_dir = str(run_dir)
        actions.append(LogInfo(msg=f"[fusion_research] recording run to {run_dir}"))

    sim_share = Path(get_package_share_directory("rc_sim_description"))
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(sim_share / "launch" / "apex_sim.launch.py")),
            launch_arguments={
                "scenario": arg("scenario"),
                "control_mode": arg("control_mode"),
                "gazebo_gui": arg("gazebo_gui"),
                "rviz": "false",
                # Ideal Gazebo sensors: all errors come from the research models.
                "lidar_noise_std": "0.0",
                "imu_gyro_noise_stddev_rps": "0.0",
                "imu_accel_noise_stddev_mps2": "0.0",
            }.items(),
        )
    )

    common = {"use_sim_time": True}
    nodes = [
        Node(package=PKG, executable="truth_node", name="fusion_truth", output="screen", parameters=[common]),
        Node(
            package=PKG, executable="imu_sensor_node", name="fusion_imu_sensor", output="screen",
            parameters=[str(imu_cfg), imu_overrides, common],
        ),
        Node(
            package=PKG, executable="lidar_noise_node", name="fusion_lidar_noise", output="screen",
            parameters=[str(lidar_cfg), lidar_overrides, common],
        ),
        Node(
            package=PKG, executable="strapdown_ins_node", name="fusion_strapdown_ins", output="screen",
            parameters=[str(ins_cfg), common],
        ),
        Node(
            package=PKG, executable="ins_error_monitor_node", name="fusion_ins_error_monitor", output="screen",
            parameters=[{"output_dir": output_dir}, common],
        ),
        Node(package=PKG, executable="scan_projector_node", name="fusion_scan_projector", output="screen", parameters=[common]),
    ]
    if _truthy(arg("rviz")):
        nodes.append(
            Node(
                package="rviz2", executable="rviz2", name="fusion_research_rviz", output="log",
                arguments=["-d", str(share / "rviz" / "fusion_research.rviz")], parameters=[common],
            )
        )
    # Give Gazebo time to spawn the vehicle before the research nodes start.
    actions.append(TimerAction(period=3.0, actions=nodes))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("scenario", default_value="baseline"),
            DeclareLaunchArgument("control_mode", default_value="recognition_tour"),
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("imu_config", default_value="consumer_mems", description="preset name or YAML path"),
            DeclareLaunchArgument("lidar_config", default_value="rplidar_like", description="preset name or YAML path"),
            DeclareLaunchArgument("ins_config", default_value="static_coarse", description="preset name or YAML path"),
            DeclareLaunchArgument("imu_seed", default_value="-1", description="-1 keeps the seed of the YAML file"),
            DeclareLaunchArgument("lidar_seed", default_value="-1", description="-1 keeps the seed of the YAML file"),
            DeclareLaunchArgument("record", default_value="true"),
            DeclareLaunchArgument("output_root", default_value=""),
            DeclareLaunchArgument("run_name", default_value=""),
            OpaqueFunction(function=_prepare),
        ]
    )
