"""The short segment can use full configured steering without invented confidence."""
import math
import json
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.config_io import dataclass_from_flat, load_ros_params_yaml
from apex_race.core.race_config import RaceDriverConfig, vehicle_limits
from apex_race.core.race_driver import EXPLORE, RaceDriverCore
from apex_race.core.reactive import ReactiveCommand, ReactiveDriver, pursue, two_arc_paths
from apex_race.core.vehicle_io import load_vehicle_config


def segment_config():
    directory = Path(__file__).resolve().parents[1] / 'config'
    flat = load_ros_params_yaml(directory / 'race_segment.yaml')
    cfg = dataclass_from_flat(RaceDriverConfig, flat)
    configured = cfg.vehicle
    cfg.vehicle = vehicle_limits(load_vehicle_config(directory / 'vehicle_real.yaml'), cfg.vehicle.steer_plan_deg)
    for name in ('steer_max_left_deg', 'steer_max_right_deg', 'steer_plan_left_deg', 'steer_plan_right_deg'):
        setattr(cfg.vehicle, name, getattr(configured, name))
    return cfg, flat


def test_zero_confidence_keeps_full_segment_authority_and_safety_bounds():
    cfg, flat = segment_config()
    core = RaceDriverCore(cfg, 1)
    assert core.model.confidence == 0.0
    assert math.degrees(math.atan(core.env.kappa_limit * cfg.vehicle.wheelbase_m)) == pytest.approx(12.5)
    assert cfg.vehicle.steer_plan_deg == cfg.vehicle.steer_max_deg == 12.5
    assert math.degrees(math.atan(core.env.kappa_limit_left * cfg.vehicle.wheelbase_m)) == pytest.approx(22.5)
    assert math.degrees(math.atan(core.env.kappa_limit_right * cfg.vehicle.wheelbase_m)) == pytest.approx(12.5)
    assert core.env.lookahead_add_m == core.env.lookahead_add_s == 0.0
    assert flat['require_arm'] and cfg.mission.forward_only and not cfg.adaptive.recover_enabled
    assert cfg.mission.max_command_mps == 0.40 and cfg.mission.max_distance_m == 1.0
    assert cfg.mission.max_duration_s == 5.0
    assert cfg.reactive.c_hard_m == 0.08 and cfg.reactive.wall_margin_m == 0.15
    assert not cfg.reactive.use_wall_planner
    assert not cfg.reactive.preview_while_waiting
    assert (cfg.reactive.n_k1, cfg.reactive.n_k2) == (21, 11)


def test_segment_follows_the_initial_arc_instead_of_the_opposite_second_arc():
    cfg, _ = segment_config()
    core = RaceDriverCore(cfg, 1)
    k = cfg.vehicle.curvature_limit(True, planning=True)
    x, y, _ = two_arc_paths(np.array([k]), np.array([-k]), 0.6, 0.8, 0.01, cfg.vehicle.rear_offset_m)
    path = np.column_stack((x[0, 0], y[0, 0]))
    core.phase = EXPLORE
    core.last_t = 0.0
    core.local_path_odom = path
    core.cmd = ReactiveCommand(k, 0.15, path, True, {'free': 0.8})
    v, steering, _ = core.control(1.0, (0, 0, 0), 0.15, None, True)
    assert math.degrees(steering) == pytest.approx(22.5, abs=0.01)
    assert 0 <= v <= 0.25
    assert core.model.confidence == 0.0


def test_segment_arcs_keep_wall_tracking_and_turn_before_a_corner_obstacle():
    cfg, _ = segment_config()
    right = np.column_stack((np.linspace(-0.5, 1.1, 161), np.full(161, -0.23)))
    front = np.column_stack((np.full(321, 1.1), np.linspace(-0.23, 3.0, 321)))
    cmd, state = ReactiveDriver(cfg.vehicle, cfg.reactive).step(np.vstack((right, front)), (0, 0, 0), 0, 0)
    assert cmd.diag['planner'] == 'arcs' and state.ref is not None
    assert cmd.feasible and cmd.v > 0 and cmd.kappa > 0
    assert cmd.diag['free'] >= 2.0
    assert np.linalg.norm(cmd.path[-1] - [cfg.vehicle.rear_offset_m, 0]) <= cmd.diag['free'] + 1e-9


def test_recorded_start_does_not_keep_deferring_the_left_turn():
    cfg, _ = segment_config()
    fixture = json.loads((Path(__file__).parent / 'data/segment_start_cloud.json').read_text())
    driver = ReactiveDriver(cfg.vehicle, cfg.reactive)
    cloud = np.asarray(fixture['points'])
    for _ in range(5):
        cmd, state = driver.step(cloud, (0, 0, 0), 0, 0)
    core = RaceDriverCore(cfg, 1)
    steering = math.degrees(core.steer_command(pursue(cmd.path, (cfg.vehicle.rear_offset_m, 0, 0), 0.45, cfg.vehicle.wheelbase_m)))
    assert state.ref is not None and cmd.feasible
    assert 18.0 <= steering <= cfg.vehicle.steering_limit_deg(True)
    assert cmd.diag['free'] >= 0.8 - 1e-9
