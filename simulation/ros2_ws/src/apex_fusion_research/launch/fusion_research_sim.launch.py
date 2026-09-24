"""Gazebo simulation + sensor-fusion research stack.

Starts the standard APEX simulation (``rc_sim_description/apex_sim.launch.py``)
with *ideal* Gazebo sensors, and adds the research nodes on top of it:

    Gazebo ideal IMU  --> fusion_imu_sensor     --> /apex/fusion/imu/data_raw --> fusion_strapdown_ins
    Gazebo ideal scan --> fusion_lidar_noise    --> /apex/fusion/scan_noisy
    Gazebo poses      --> fusion_truth          --> /apex/fusion/truth/odom
    truth + INS       --> fusion_ins_error_monitor (errors, CSV, metadata)
    scan + poses      --> fusion_scan_projector (RViz point clouds)

With ``slam:=true`` identical slam_toolbox instances (same tuning) run side
by side and differ only in their inputs:

    good     : ideal LiDAR + ideal odometry (ground-truth motion prior), scan matching
               and loop closing off like the APEX ideal mapping mode           -> /apex/slam/good/map
    noisy    : noisy LiDAR + heading prior from the noisy-IMU INS             -> /apex/slam/noisy/map
    good_imu : ideal LiDAR + heading prior from an ideal-IMU INS (slam_ablation:=true)
               separates the limitation of the heading-only prior from the sensor noise
    all      --> fusion_slam_map_recorder (maps, trajectories, real track as CSV)

Each instance owns a TF tree ``<tag>/map -> <tag>/odom -> <tag>/base_link ->
<tag>/laser``. ``pipeline_slam:=true`` additionally enables the existing APEX
SLAM (``use_slam``: /lidar/scan_slam + the pipeline's fused odometry -> /map).

With ``record_measurements:=true`` the raw measurements of both sensor sets are
recorded to CSV with rc_sim_description's apex_sim_run_recorder.

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
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.events import matches_action
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

from apex_fusion_research.core.config_io import load_ros_params_yaml

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
    use_slam = _truthy(arg("slam"))
    pipeline_slam = use_slam and _truthy(arg("pipeline_slam"))
    record_measurements = _truthy(arg("record_measurements"))
    slam_cfg = _resolve_config(arg("slam_config"), share, "slam") if use_slam else None
    if use_slam:
        # The noisy scan lives in the damaged-sensor TF tree.
        lidar_overrides["output_frame_id"] = "noisy/laser"

    output_dir = ""
    actions = []
    if _truthy(arg("record")):
        root = arg("output_root") or os.path.join(
            os.environ.get("APEX_SIM_ROOT", str(Path.home() / ".ros")), "data", "fusion_research"
        )
        run_name = arg("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(root).expanduser() / run_name
        (run_dir / "config").mkdir(parents=True, exist_ok=True)
        for cfg in (imu_cfg, lidar_cfg, ins_cfg, slam_cfg):
            if cfg is not None:
                shutil.copy2(cfg, run_dir / "config" / f"{cfg.parent.name}_{cfg.name}")
        launch_args = {
            name: arg(name)
            for name in (
                "scenario", "control_mode", "imu_config", "lidar_config", "ins_config", "imu_seed", "lidar_seed",
                "slam", "slam_config", "slam_prior_mode", "slam_ablation", "pipeline_slam", "record_measurements",
            )
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
                # Optional third SLAM: the existing APEX slam_toolbox path.
                "use_slam": "true" if pipeline_slam else "false",
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
    if use_slam:
        laser = (arg("laser_offset_x"), arg("laser_offset_y"), arg("laser_offset_z"))
        prior = arg("slam_prior_mode")
        ablation = _truthy(arg("slam_ablation"))

        def relay(tag: str) -> Node:
            """Forward the ideal scan unchanged into the ``<tag>`` TF tree."""
            return Node(
                package=PKG, executable="lidar_noise_node", name=f"fusion_lidar_relay_{tag}", output="screen",
                parameters=[{
                    "enabled": False,
                    "output_topic": f"/apex/fusion/scan_{tag}",
                    "output_frame_id": f"{tag}/laser",
                    "status_topic": f"/apex/fusion/lidar_{tag}/status",
                    "realization_topic": f"/apex/fusion/lidar_{tag}/realization",
                }, common],
            )

        # Good sensors: ideal LiDAR + ideal odometry (ground-truth prior).
        nodes.append(relay("good"))
        # With a perfect prior no correction is needed; slam_toolbox's scan
        # matcher would bias the translation in the track corridors (see the
        # README), so the reference maps directly from the ideal odometry,
        # as the APEX ideal mapping mode (slam_toolbox_sim_ideal.yaml) does.
        nodes += _slam_instance("good", "/apex/fusion/scan_good", "/apex/fusion/truth/odom", "full_pose",
                                slam_cfg, laser, common,
                                overrides={"use_scan_matching": False, "do_loop_closing": False})
        # Damaged sensors: noisy LiDAR + heading prior from the noisy-IMU INS.
        nodes += _slam_instance("noisy", "/apex/fusion/scan_noisy", "/apex/fusion/ins/odom", prior,
                                slam_cfg, laser, common)
        if ablation:
            # Ideal LiDAR + heading prior from an INS on the ideal IMU (same INS config).
            nodes.append(relay("good_imu"))
            nodes.append(
                Node(
                    package=PKG, executable="strapdown_ins_node", name="fusion_strapdown_ins_good", output="screen",
                    parameters=[load_ros_params_yaml(ins_cfg), {
                        "imu_topic": "/apex/sim/imu",
                        "odom_topic": "/apex/fusion/ins_good/odom",
                        "path_topic": "/apex/fusion/ins_good/path",
                        "status_topic": "/apex/fusion/ins_good/status",
                        "alignment_topic": "/apex/fusion/ins_good/alignment",
                        "child_frame_id": "imu_ins_good",
                        "publish_tf": False,
                    }, common],
                )
            )
            nodes += _slam_instance("good_imu", "/apex/fusion/scan_good_imu", "/apex/fusion/ins_good/odom", prior,
                                    slam_cfg, laser, common)
        if output_dir:
            tags = [(t, f"/apex/slam/{t}/map", f"{t}/map", f"{t}/base_link")
                    for t in (["good", "noisy"] + (["good_imu"] if ablation else []))]
            if pipeline_slam:
                tags.append(("pipeline", "/map", "map", "base_link"))
            nodes.append(
                Node(
                    package=PKG, executable="slam_map_recorder_node", name="fusion_slam_map_recorder", output="screen",
                    parameters=[{
                        "output_dir": str(Path(output_dir) / "slam"),
                        "slam_names": [t[0] for t in tags],
                        "map_topics": [t[1] for t in tags],
                        "map_frames": [t[2] for t in tags],
                        "base_frames": [t[3] for t in tags],
                    }, common],
                )
            )
    if record_measurements and output_dir:
        nodes += _measurement_recorders(Path(output_dir), arg("scenario"), common)
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


def _slam_instance(tag: str, scan_topic: str, pose_source_topic: str, prior_mode: str, slam_cfg: Path,
                   laser_xyz: tuple, common: dict, overrides: dict | None = None) -> list:
    """One slam_toolbox instance with its own TF tree ``<tag>/...``.

    The motion prior (``<tag>/odom -> <tag>/base_link``) is built by
    fusion_slam_odometry from ``pose_source_topic`` (INS or ground truth).
    """
    slam_node = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name=f"slam_toolbox_{tag}",
        namespace="",
        output="screen",
        parameters=[
            str(slam_cfg),
            {
                "map_frame": f"{tag}/map",
                "odom_frame": f"{tag}/odom",
                "base_frame": f"{tag}/base_link",
                "scan_topic": scan_topic,
                "map_name": f"/apex/slam/{tag}/map",
                "use_lifecycle_manager": False,
                **(overrides or {}),
            },
            common,
        ],
    )
    # Same autostart sequence as slam_toolbox's online_async_launch.py.
    configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_node), transition_id=Transition.TRANSITION_CONFIGURE
        )
    )
    activate = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg=f"[fusion_research] activating slam_toolbox_{tag}"),
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )
    x, y, z = laser_xyz
    laser_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"fusion_{tag}_laser_tf",
        arguments=["--x", x, "--y", y, "--z", z, "--frame-id", f"{tag}/base_link", "--child-frame-id", f"{tag}/laser"],
        parameters=[common],
    )
    odometry = Node(
        package=PKG, executable="slam_odometry_node", name=f"fusion_slam_odometry_{tag}", output="screen",
        parameters=[{
            "ins_topic": pose_source_topic,
            "odom_topic": f"/apex/fusion/slam_{tag}/odom",
            "mode": prior_mode,
            "odom_frame_id": f"{tag}/odom",
            "base_frame_id": f"{tag}/base_link",
        }, common],
    )
    return [activate, slam_node, configure, laser_tf, odometry]


def _measurement_recorders(run_dir: Path, scenario: str, common: dict) -> list:
    """Raw measurement CSVs of both sensor sets (rc_sim_description recorder)."""
    sets = {
        "noisy": ("/apex/fusion/scan_noisy", "/apex/fusion/imu/data_raw", "/apex/fusion/ins/odom"),
        "ideal": ("/apex/sim/scan", "/apex/sim/imu", "/apex/odometry/imu_lidar_fused"),
    }
    return [
        Node(
            package="rc_sim_description",
            executable="apex_sim_run_recorder.py",
            name=f"fusion_recorder_{name}",
            output="log",
            parameters=[
                {
                    "run_dir": str(run_dir / f"measurements_{name}"),
                    "scan_topic": scan,
                    "imu_topic": imu,
                    "odom_topic": odom,
                    "scenario": scenario,
                    "mapping_mode": f"fusion_research_{name}_sensors",
                },
                common,
            ],
        )
        for name, (scan, imu, odom) in sets.items()
    ]


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
            DeclareLaunchArgument("slam", default_value="false", description="good + damaged-sensor slam_toolbox"),
            DeclareLaunchArgument("slam_config", default_value="slam_toolbox_2d", description="slam_toolbox preset or YAML"),
            DeclareLaunchArgument("slam_prior_mode", default_value="heading_only",
                                  description="damaged-sensor prior: heading_only | full_pose"),
            DeclareLaunchArgument("slam_ablation", default_value="false",
                                  description="also run ideal LiDAR + ideal-IMU heading SLAM (good_imu)"),
            DeclareLaunchArgument("pipeline_slam", default_value="false", description="also run the APEX pipeline SLAM"),
            DeclareLaunchArgument("record_measurements", default_value="false", description="raw sensor CSVs"),
            DeclareLaunchArgument("laser_offset_x", default_value="0.18", description="laser in base_link (URDF)"),
            DeclareLaunchArgument("laser_offset_y", default_value="0.0"),
            DeclareLaunchArgument("laser_offset_z", default_value="0.12"),
            DeclareLaunchArgument("record", default_value="true"),
            DeclareLaunchArgument("output_root", default_value=""),
            DeclareLaunchArgument("run_name", default_value=""),
            OpaqueFunction(function=_prepare),
        ]
    )
