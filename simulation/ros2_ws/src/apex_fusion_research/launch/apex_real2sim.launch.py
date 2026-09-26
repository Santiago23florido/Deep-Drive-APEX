"""Real2sim closed-loop run: learned odometry + SLAM, the car on an unseen track.

    Gazebo (track world, car with suspension, native gpu_lidar + 1 kHz IMU
    with Gazebo noise)
      -> apex_real_sensors: A2M8 revolutions + LSM6DS3 samples (realism layer)
      -> apex_learned_odometry (streaming network, venv Python with torch)
           TF odom_learned -> base_link, predicted pose, de-skewed scan
      -> slam_toolbox: map, TF map -> odom_learned
      -> driver:=race (default) apex_race_driver: the car knows only its odometry
           model and its own limits. Lap 1 reactive from the LiDAR while the map
           is built, lap closure on its map pose, race line planned on its map,
           pure pursuit for the next laps (everything in the map frame).
         driver:=format apex_track_driver: pure pursuit + speed plan of the
           dataset format on T_world_map * map->odom * odom pose (the seeded
           reference path of the track: validation of the estimator only)
      -> /apex/cmd_vel_track -> apex_sim_actuation (dataset ESC / servo) -> Gazebo
    apex_run_referee (truth): lap / collision / off-track verdict -> run_result.json
    fusion_slam_map_recorder: maps and trajectories for plot_slam_maps

The format (track, motion, seed, laps) is rebuilt exactly as the pose dataset
does (start pose with its seeded placement errors, actuator perturbation, true
sensor mounts drawn for that trajectory; with driver:=format also its
reference path, speed plan and driver gains). The car only knows its nominal
calibration: base_link -> laser and -> imu_link are the nominal mounts. With
driver:=race, motion and seed only draw the physical car (mounts, actuators,
spawn); the driver receives none of them.

    ros2 launch apex_fusion_research apex_real2sim.launch.py track:=val_mixed motion:=medium seed:=1 laps:=2 \
        run_dir:=/path/to/run
(normally through simulation/tools/sim/apex_real2sim_up.sh).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition

PKG = "apex_fusion_research"


def _sim_root() -> Path:
    env = os.environ.get("APEX_SIM_ROOT", "").strip()
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools" / "pose_dataset").is_dir():
            return parent
    raise RuntimeError("cannot locate simulation/ (set APEX_SIM_ROOT)")


def _axes(v) -> str:  # noqa: ANN001
    return " ".join(f"{float(x):.9g}" for x in v)


def _setup(context, *_args, **_kwargs):  # noqa: ANN001, ANN202
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = _sim_root()
    sys.path.insert(0, str(sim / "tools"))
    sys.path.insert(0, str(sim / "ros2_ws" / "src" / "apex_fusion_research"))
    from apex_fusion_research.core.imu_chip import gazebo_noise_from_profile
    from apex_fusion_research.nodes._pose_tools import run_setup
    from pose_dataset.sensors import draw_extrinsic, load_sensor_profiles, rng_for

    track, motion, seed, laps = arg("track"), arg("motion"), int(arg("seed")), float(arg("laps"))
    sensor = arg("sensor_profile")
    run_dir = Path(arg("run_dir")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rs = run_setup(track, motion, seed, laps)
    profile = load_sensor_profiles()[sensor]
    hardware = str(profile.get("hardware", sensor))
    tkey = rs["job"]["trajectory_key"]
    rng = rng_for(tkey, hardware, "mounts")  # the physical sensor set of this trajectory (as the dataset)
    lidar_ext = draw_extrinsic(profile["lidar"], rng)
    imu_ext = draw_extrinsic(profile["imu"], rng)
    gz_cfg = profile["lidar"].get("gazebo", {})
    xargs = {
        "suspension": "true", "camera": "false", "imu_clean_reference": "false", "imu_update_rate": "1000",
        "lidar_update_rate": f"{float(gz_cfg.get('capture_rate_hz', 26.0)):g}", "lidar_samples": str(int(gz_cfg.get("capture_samples", 1440))),
        "lidar_range_min": f"{float(gz_cfg.get('capture_range_min_m', 0.05)):g}", "lidar_range_max": f"{1.2 * float(profile['lidar']['range_max_m']):g}",
        "lidar_range_resolution": "0.0001",
        "lidar_xyz": _axes(lidar_ext["xyz_true"]), "lidar_rpy": _axes(lidar_ext["rpy_true_rad"]),
        "imu_xyz": _axes(imu_ext["xyz_true"]), "imu_rpy": _axes(imu_ext["rpy_true_rad"]),
        **{f"imu_{k}": _axes(v) for k, v in gazebo_noise_from_profile(profile["imu"]).items()},
    }
    xacro = sim / "ros2_ws" / "src" / "rc_sim_description" / "urdf" / "rc_car.urdf.xacro"
    urdf = subprocess.run(["xacro", str(xacro), *[f"{k}:={v}" for k, v in xargs.items()]], check=True, capture_output=True, text=True).stdout
    urdf_path = run_dir / "car.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    world = sim / "ros2_ws" / "src" / "rc_sim_description" / "worlds" / "pose_dataset" / f"{track}.world"
    sx, sy, syaw = rs["spawn"]
    # Where the car believes it starts (the origin of its reference path):
    # "true" = the path is defined from the car's actual start (as planning in
    # its own map); "nominal" = the start line, the seeded placement error of
    # the format (a few cm, up to 2 deg) stays unknown to the car.
    # driver:=race plans in its own map, whose origin is its first pose: world -> map is then the true
    # start and is used only by RViz and the evaluation, never by the car.
    start_pose = arg("start_pose")
    nx, ny, nyaw = rs["spawn"] if (start_pose == "true" or arg("driver") == "race") else rs["nominal_start"]
    spawn_z = float(rs["vcfg"]["model"]["spawn_z_m"])
    ckpt = arg("checkpoint")
    venv_python = sim / "learning" / ".venv" / "bin" / "python"
    drive = arg("drive")
    driver = arg("driver")
    if driver not in ("race", "format"):
        raise RuntimeError(f"driver must be race or format, not {driver!r}")
    car = {  # everything the race driver is told (no track, motion or seed)
        "laps": int(round(laps)), "vehicle_yaml": str(sim / "tools" / "pose_dataset" / "config" / "vehicle.yaml"),
        "reactive.w_min_m": float(arg("w_min")), "reactive.v_explore_mps": float(arg("v_explore")),
        "planner.mode": arg("race_line"), "race.v_max_mps": float(arg("v_max_race")), "race.a_lat_mps2": float(arg("a_lat")),
    }
    common = {"use_sim_time": True}
    (run_dir / "run_config.json").write_text(json.dumps({
        "track": track, "motion": motion, "seed": seed, "laps": laps, "sensor_profile": sensor, "checkpoint": ckpt, "device": arg("device"),
        "drive": drive, "driver": driver, "car": car if driver == "race" else None,
        "start_pose": start_pose, "spawn_true": rs["spawn"], "nominal_start": rs["nominal_start"], "trajectory_key": tkey,
        "lidar_mount_true": {"xyz": list(lidar_ext["xyz_true"]), "rpy": list(lidar_ext["rpy_true_rad"])},
        "imu_mount_true": {"xyz": list(imu_ext["xyz_true"]), "rpy": list(imu_ext["rpy_true_rad"])},
        "xacro_args": xargs, "plan": rs["setup"]["plan"].description, "path_length_m": rs["setup"]["path"].length,
    }, indent=1, default=float), encoding="utf-8")

    gz_cmd = ["gz", "sim", "-r", "--seed", str(seed)]
    if arg("gui").lower() not in ("1", "true", "yes"):
        gz_cmd += ["-s", "--headless-rendering"]
    actions = [
        LogInfo(msg=f"[real2sim] {tkey} ({laps:g} laps), {sensor}, drive on {drive}; run dir {run_dir}"),
        ExecuteProcess(cmd=gz_cmd + [str(world)], output="log"),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="real2sim_clock_bridge", output="log",
             arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"], parameters=[common]),
        TimerAction(period=2.0, actions=[
            Node(package="ros_gz_sim", executable="create", output="log",
                 arguments=["-name", "rc_car", "-file", str(urdf_path), "-x", str(sx), "-y", str(sy), "-z", str(spawn_z), "-Y", str(syaw)]),
        ]),
    ]
    # Nominal calibration of the car (what the real car knows).
    lx, ly, lz = (float(v) for v in profile["lidar"]["extrinsic"]["xyz"])
    ix, iy, iz = (float(v) for v in profile["imu"]["extrinsic"]["xyz"])
    tf_nodes = [
        Node(package="tf2_ros", executable="static_transform_publisher", name="real2sim_laser_tf", output="log",
             arguments=["--x", str(lx), "--y", str(ly), "--z", str(lz), "--frame-id", "base_link", "--child-frame-id", "laser"], parameters=[common]),
        Node(package="tf2_ros", executable="static_transform_publisher", name="real2sim_imu_tf", output="log",
             arguments=["--x", str(ix), "--y", str(iy), "--z", str(iz), "--frame-id", "base_link", "--child-frame-id", "imu_link"], parameters=[common]),
        Node(package="tf2_ros", executable="static_transform_publisher", name="real2sim_world_map_tf", output="log",
             arguments=["--x", str(nx), "--y", str(ny), "--yaw", str(nyaw), "--frame-id", "world", "--child-frame-id", "map"], parameters=[common]),
    ]
    slam_node = LifecycleNode(
        package="slam_toolbox", executable="async_slam_toolbox_node", name="slam_toolbox", namespace="", output="log",
        parameters=[str(sim / "ros2_ws" / "src" / PKG / "config" / "slam" / "slam_toolbox_learned.yaml"),
                    {"max_laser_range": float(profile["lidar"]["range_max_m"]), "use_lifecycle_manager": False}, common],
    )
    slam = [
        RegisterEventHandler(OnStateTransition(target_lifecycle_node=slam_node, start_state="configuring", goal_state="inactive", entities=[
            EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam_node), transition_id=Transition.TRANSITION_ACTIVATE))])),
        slam_node,
        EmitEvent(event=ChangeState(lifecycle_node_matcher=matches_action(slam_node), transition_id=Transition.TRANSITION_CONFIGURE)),
    ]
    stack = [
        *tf_nodes,
        Node(package=PKG, executable="real_sensor_node", name="apex_real_sensors", output="screen",
             parameters=[{"sensor_profile": sensor, "seed": seed, "robot_description": urdf, "truth_csv_dir": str(run_dir), **common}]),
        Node(package=PKG, executable="sim_actuation_node", name="apex_sim_actuation", output="log",
             parameters=[{"esc_tau_scale": float(rs["setup"]["actuator_perturbation"].get("esc_tau_scale", 1.0)),
                          "servo_tau_scale": float(rs["setup"]["actuator_perturbation"].get("servo_tau_scale", 1.0)), **common}]),
        ExecuteProcess(
            cmd=[str(venv_python), str(sim / "learning" / "lidar_imu_pose" / "learned_odometry_node.py"), "--ros-args",
                 "-p", f"checkpoint:={ckpt}", "-p", f"sensor_profile:={sensor}", "-p", f"device:={arg('device')}",
                 "-p", "use_sim_time:=true", "-p", f"log_csv:={run_dir / 'estimator.csv'}", "-r", "__node:=apex_learned_odometry"],
            output="screen",
        ),
        *slam,
        Node(package=PKG, executable="race_driver_node", name="apex_race_driver", output="screen",
             parameters=[{**car, "output_dir": str(run_dir / "plan"), "log_csv": str(run_dir / "driver.csv"),
                          "events_json": str(run_dir / "race_events.json"), **common}])
        if driver == "race" else
        Node(package=PKG, executable="track_driver_node", name="apex_track_driver", output="screen",
             parameters=[{"track": track, "motion": motion, "seed": seed, "laps": laps, "pose_source": drive, "start_pose": start_pose,
                          "log_csv": str(run_dir / "driver.csv"), **common}]),
        Node(package=PKG, executable="run_referee_node", name="apex_run_referee", output="screen",
             parameters=[{"track": track, "motion": motion, "seed": seed, "laps": laps, "output_dir": str(run_dir), "mode": driver, **common}]),
        Node(package=PKG, executable="slam_map_recorder_node", name="fusion_slam_map_recorder", output="log",
             parameters=[{"output_dir": str(run_dir / "slam"), "slam_names": ["learned"], "map_topics": ["/map"], "map_frames": ["map"],
                          "base_frames": ["base_link"], "truth_topic": "/apex/sim/ground_truth/base_odom", "imu_offset_in_base_xyz": [0.0, 0.0, 0.0],
                          **common}]),
    ]
    if arg("rviz").lower() in ("1", "true", "yes"):
        stack.append(Node(package="rviz2", executable="rviz2", name="real2sim_rviz", output="log",
                          arguments=["-d", arg("rviz_config") or str(sim / "ros2_ws" / "src" / PKG / "rviz" / "real2sim.rviz")], parameters=[common]))
    # The car must exist (spawn) before the sensor node composes its poses.
    actions.append(TimerAction(period=4.0, actions=stack))
    return actions


def generate_launch_description() -> LaunchDescription:
    sim = _sim_root()
    return LaunchDescription([
        DeclareLaunchArgument("track", default_value="val_mixed"),
        DeclareLaunchArgument("motion", default_value="medium"),
        DeclareLaunchArgument("seed", default_value="1"),
        DeclareLaunchArgument("laps", default_value="2"),
        DeclareLaunchArgument("sensor_profile", default_value="APEX_real"),
        DeclareLaunchArgument("checkpoint", default_value=str(sim / "learning" / "outputs" / "real2sim_v2_fast" / "hybrid_submap_v3" / "best_model.pt")),
        DeclareLaunchArgument("device", default_value="cpu", description="cpu (~20 ms per scan) | cuda"),
        DeclareLaunchArgument("driver", default_value="race", description="race (unseen track: reactive lap 1, map, race line) | format (dataset path)"),
        DeclareLaunchArgument("w_min", default_value="1.5", description="race: generic minimum lane width the car assumes [m]"),
        DeclareLaunchArgument("v_explore", default_value="1.5", description="race: top speed of the reactive lap 1 [m/s]"),
        DeclareLaunchArgument("race_line", default_value="min_curvature", description="race: min_curvature | centre"),
        DeclareLaunchArgument("v_max_race", default_value="3.5", description="race: top speed on the race line [m/s]"),
        DeclareLaunchArgument("a_lat", default_value="3.0", description="race: lateral acceleration limit of the car [m/s^2]"),
        DeclareLaunchArgument("drive", default_value="estimate", description="format only: estimate (closed loop) | truth (diagnostic)"),
        DeclareLaunchArgument("start_pose", default_value="true", description="true (path from the actual start) | nominal (start line)"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value="", description="RViz config file (default: rviz/real2sim.rviz)"),
        DeclareLaunchArgument("run_dir", default_value=str(sim / "data" / "real2sim" / "latest")),
        OpaqueFunction(function=_setup),
    ])
