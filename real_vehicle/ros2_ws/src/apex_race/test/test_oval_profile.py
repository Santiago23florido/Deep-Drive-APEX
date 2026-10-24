"""The oval uses the existing driver, full calibrated right travel and short reverses."""
import math
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.config_io import dataclass_from_flat, load_ros_params_yaml
from apex_race.core.race_config import RaceDriverConfig, vehicle_limits
from apex_race.core.race_driver import EXPLORE, RaceDriverCore
from apex_race.core.reactive import ReactiveCommand, ReactiveDriver
from apex_race.core.vehicle_io import load_vehicle_config


def oval_core():
    directory = Path(__file__).resolve().parents[1] / 'config'
    flat = load_ros_params_yaml(directory / 'race_oval.yaml')
    cfg = dataclass_from_flat(RaceDriverConfig, flat)
    sides = cfg.vehicle
    cfg.vehicle = vehicle_limits(load_vehicle_config(directory / 'vehicle_real.yaml'), sides.steer_plan_deg)
    for name in ('steer_max_left_deg', 'steer_max_right_deg', 'steer_plan_left_deg', 'steer_plan_right_deg'):
        setattr(cfg.vehicle, name, getattr(sides, name))
    return RaceDriverCore(cfg, flat['laps'])


def blocked_core():
    core = oval_core()
    core.phase = EXPLORE
    core.cmd = ReactiveCommand(0, 0, np.zeros((0, 2)), False, {'free': .1})
    core.rear_clear = 1.0
    for t in np.arange(0, 1.2, .02):
        command = core.control(float(t), (0, 0, 0), 0, None, True)[0]
    assert core.recover == 2 and -.25 <= command < 0
    return core


def test_full_right_range_survives_the_identifier_and_adaptive_envelope():
    core = oval_core()
    assert core.laps == 2
    assert core.cfg.mission.max_distance_m == 40 and core.cfg.mission.max_duration_s == 300
    assert core.cfg.adaptive.recover_max_per_lap == 36
    assert core.cfg.adaptive.recover_max_without_progress == 12
    assert core.cfg.adaptive.recover_progress_reset_m == .75
    assert core.cfg.closure.arm_distance_m == 5
    assert not core.cfg.mission.forward_only
    for desired, expected in ((-10, -12.96), (10, 22.5)):
        core.shaper.reset()
        _, curvature = core.shaper.shape(0, desired, .2, .2, core.model, core.env)
        assert math.degrees(math.atan(curvature * .3)) == pytest.approx(expected)
    assert (core.cfg.reactive.n_k1, core.cfg.reactive.n_k2) == (21, 11)
    assert not core.cfg.reactive.preview_while_waiting
    assert core.cfg.mission.max_reverse_command_mps == .40
    assert core.cfg.adaptive.lead_preserve_turn


@pytest.mark.parametrize('rear', [.14, float('nan')])
def test_new_rear_obstacle_cuts_reverse_without_the_forward_acceleration_ramp(rear):
    core = blocked_core()
    core.rear_clear = rear
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] == 0
    assert core.recover == 0
    assert core.ident.pending[1] == 0


def test_reverse_distance_limit_stops_without_a_negative_command_tail():
    core = blocked_core()
    core.travel += .51
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] == 0
    assert core.recover == 0


def test_reverse_timeout_cuts_command_even_without_odometry_progress():
    core = blocked_core()
    assert core.control(core.recover_until, (0, 0, 0), -.2, None, True)[0] == 0
    assert core.recover == 0


def test_recovery_counts_slow_pose_motion_even_when_speed_gate_reports_zero():
    core = blocked_core()
    for t, x in ((1.22, -.2), (1.24, -.4), (1.26, -.51)):
        command = core.control(t, (x, 0, 0), 0, None, True)[0]
    assert core.travel == 0  # lap travel deliberately rejects parked speed noise
    assert core.recover_distance == pytest.approx(.51)
    assert command == 0 and core.recover == 0


def test_reverse_launch_compensation_can_reach_new_cap_without_exceeding_it():
    core = blocked_core()
    core.shaper.launch_extra = .25
    command = core.control(1.22, (0, 0, 0), 0, None, True)[0]
    assert command == pytest.approx(-.40)
    assert core.ident.pending[1] == command


@pytest.mark.parametrize('sign', [-1, 1])
def test_anticipation_cannot_reverse_a_small_requested_turn(sign):
    core = oval_core()
    core.shaper.t_prev = 0
    core.shaper.k_smooth = sign * 1.0  # a large turn is relaxing toward centre
    desired = sign * .04
    _, curvature = core.shaper.shape(.1, desired, .2, .2, core.model, core.env)
    assert sign * (desired + core.shaper.lead) >= -1e-12
    assert sign * curvature >= -1e-12  # this test uses the unbiased prior


def test_reverse_budget_is_shorter_when_rear_obstacle_is_close():
    core = oval_core()
    core.phase = EXPLORE
    core.cmd = ReactiveCommand(0, 0, np.zeros((0, 2)), False, {'free': .1})
    core.rear_clear = .28
    for t in np.arange(0, 1.2, .02):
        core.control(float(t), (0, 0, 0), 0, None, True)
    assert core.recover_distance_limit == pytest.approx(.13)
    core.travel = .14
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] == 0


def test_zero_speed_feasible_path_does_not_abort_reverse_immediately():
    core = blocked_core()
    core.cmd.feasible = True
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] < 0
    assert core.recover == 2


def test_missing_rear_cloud_does_not_allow_reverse():
    core = oval_core()
    assert core.rear_clear == core._rear_clearance(np.zeros((0, 2))) == 0


def test_reverse_permission_does_not_turn_forward_braking_feedback_into_reverse():
    core = oval_core()
    core.phase = EXPLORE
    core.local_path_odom = np.array([[-.15, 0], [1, 0]])
    core.cmd = ReactiveCommand(0, .001, core.local_path_odom, True)
    core.last_t = 0
    assert core.control(.02, (0, 0, 0), 1.0, None, True)[0] == 0


def test_both_wall_targets_allow_twenty_cm_beside_body_without_changing_collision_bounds():
    cfg = oval_core().cfg
    for side, expected in ((1, .36), (-1, .36)):
        cloud = np.column_stack((np.linspace(-1, 3, 100), np.full(100, side * .35)))
        command, state = ReactiveDriver(cfg.vehicle, cfg.reactive).step(cloud, (0, 0, 0), 0, 0)
        assert state.mode == ('L' if side > 0 else 'R')
        assert state.d_target == pytest.approx(expected)
        assert command.feasible
    assert cfg.reactive.wall_margin_m == .15
    assert cfg.reactive.c_hard_m == .08


def test_reverse_does_not_end_on_a_barely_feasible_forward_path():
    core = blocked_core()
    core.cmd.feasible, core.cmd.v = True, .15
    core.cmd.diag['free'] = .4
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] < 0
    assert core.recover == 2
    core.cmd.diag['free'] = .7
    core.control(1.24, (0, 0, 0), -.2, None, True)
    assert core.recover == 0


def test_driving_settings_match_the_physically_working_run():
    reference = json.loads((Path(__file__).parent / 'data/working_193423_config.json').read_text())
    current = json.loads(json.dumps(asdict(oval_core().cfg)))  # JSON represents tuple grids as arrays
    # New opt-in calibration switches must stay neutral in the approved profile.
    for key, value in (('longitudinal_forward_only', False), ('longitudinal_allow_overspeed', False),
                       ('launch_forward_command_scale', 1.0), ('speed_feedback_forward_scale', 1.0),
                       ('kp_speed_forward', 0.0)):
        assert current['adaptive'][key] == value
        current['adaptive'].pop(key)
    # Driving settings remain fixed; trial budgets and the minimum lap
    # arming distance changed after observing a valid return below 10 m.
    for section, key in (('adaptive', 'recover_max_per_lap'), ('adaptive', 'recover_max_without_progress'),
                         ('adaptive', 'recover_progress_reset_m'), ('mission', 'max_distance_m'),
                         ('mission', 'max_duration_s'), ('mission', 'max_reverse_command_mps'),
                         ('closure', 'arm_distance_m'), ('closure', 'overlap_m'),
                         ('race', 'guide_handover_s'), ('adaptive', 'plan_with_identified_model')):
        reference[section].pop(key, None)
        current[section].pop(key, None)
    assert current == reference
