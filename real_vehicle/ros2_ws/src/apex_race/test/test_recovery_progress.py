"""A new corner gets attempts after forward escape, never by rocking in place."""
import json
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.race_driver import EXPLORE
from apex_race.core.reactive import ReactiveCommand
from test_oval_profile import oval_core


def progressed_core():
    core = oval_core()
    core.phase = EXPLORE
    core.recover_count = core.recover_stuck_count = 12
    core.recover_progress_origin = (0., 0.)
    core._last_odom_xy = (0., 0.)
    path = np.column_stack((np.linspace(-.15, 3, 40), np.zeros(40)))
    core.cmd = ReactiveCommand(0., .17, path, True, {'free': 2.})
    core.local_path_odom = path
    core.cmd_out = (.2, 0.)
    core.last_t = core.shaper.t_prev = 0.
    core.shaper.v_out = .2
    return core


def test_forward_escape_replenishes_stuck_allowance_but_not_total_lap_count():
    core = progressed_core()
    for i in range(1, 5):
        core.control(i * .2, (i * .2, 0., 0.), .2, None, True)
    assert core.recover_stuck_count == 0
    assert core.recover_count == 12
    event = core.events['recovery_progress'][0]
    assert event['net_m'] == pytest.approx(.8)
    assert event['forward_m'] == pytest.approx(.8)
    assert event['attempts_replenished'] == 12


@pytest.mark.parametrize('speed', [0., -.2])
def test_parked_pose_changes_and_reverse_motion_do_not_replenish_attempts(speed):
    core = progressed_core()
    for i in range(1, 8):
        core.control(i * .2, (i * .2, 0., 0.), speed, None, True)
    assert core.recover_stuck_count == 12
    assert not core.events.get('recovery_progress')


def test_rocking_accumulates_distance_but_cannot_earn_another_stuck_allowance():
    core = progressed_core()
    for i in range(10):
        core.cmd_out = (.2, 0.)
        core.control(i + .1, (.4, 0., 0.), .2, None, True)
        core.control(i + .2, (0., 0., 0.), -.2, None, True)
    assert core.recover_forward_distance > .75
    assert core.recover_stuck_count == 12
    assert not core.events.get('recovery_progress')


def test_exhausted_stuck_allowance_still_blocks_repeated_attempts_at_one_place():
    core = progressed_core()
    core.cmd.v, core.cmd.feasible = 0., False
    core.block_since, core.pose_odom, core.rear_clear = 0., (0., 0., 0.), 1.
    assert core._recovery(2., 0.) is None
    assert core.recover == 0


def test_lap_ceiling_remains_even_after_successful_forward_escape():
    core = progressed_core()
    core.recover_stuck_count = 0
    core.recover_count = core.cfg.adaptive.recover_max_per_lap
    core.cmd.v, core.cmd.feasible = 0., False
    core.block_since, core.pose_odom, core.rear_clear = 0., (0., 0., 0.), 1.
    assert core._recovery(2., 0.) is None
    assert core.recover == 0


def test_recorded_last_curve_can_recover_after_prior_corners_used_twelve_attempts():
    fixture = json.loads((Path(__file__).parent / 'data/final_curve_201912.json').read_text())
    core = progressed_core()
    core.recover_stuck_count = 1  # previous corners were escaped with forward progress
    core.reactive.set_rear_offset(fixture['rear_offset_m'])
    core.pose_odom = (0., 0., 0.)
    cloud = np.array(fixture['points'])
    core.acc.cloud_in = lambda pose: cloud
    core.acc.scans.append((0, 0, cloud))
    core.cmd = ReactiveCommand(fixture['desired_kappa'], 0., np.zeros((0, 2)), True, {'free': .3})
    core.block_since = 0.
    assert core._recovery(2., 0.)[0] < 0
    assert core.recover_count == 13 and core.recover_stuck_count == 2
    assert core.recover_distance_limit == pytest.approx(.5)


@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
def test_progress_distance_must_be_finite_and_positive(value):
    core = oval_core()
    core.cfg.adaptive.recover_progress_reset_m = value
    with pytest.raises(ValueError):
        core.cfg.adaptive.validate()
