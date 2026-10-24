"""A physically completed short lap must start map planning on its first return."""
import json
from pathlib import Path

import numpy as np
import pytest

from apex_race.core.lap_closure import LapCounter
from apex_race.core.race_driver import EXPLORE, PLANNING
from apex_race.core.race_planner import Grid
from test_oval_profile import oval_core


class PendingPlan:
    def done(self):
        return False


def replay(arm_distance):
    fixture = json.loads((Path(__file__).parent / 'data/first_lap_204107.json').read_text())
    core = oval_core()
    core.cfg.closure.arm_distance_m = arm_distance
    core.cfg.mission.enabled = False
    core.cfg.adaptive.recover_enabled = False
    core.start = tuple(fixture['start'])
    core.phase = EXPLORE
    core.counter = LapCounter(core.start, core.cfg.closure, fixture['samples'][0]['t'])
    core.submit = lambda *args: PendingPlan()
    for row in fixture['samples']:
        core.travel = row['travel']
        core._last_odom_xy = None  # fixture already supplies measured cumulative travel
        core.grid = (row['t'] + .1, Grid(np.zeros((10, 10)), .1, 0., 0.))
        core.control(row['t'], tuple(row[k] for k in ('x_map', 'y_map', 'yaw_map')),
                     row['speed'], None, True)
    return core, fixture


def test_previous_ten_meter_minimum_missed_the_physically_completed_lap():
    core, _ = replay(10.)
    assert core.counter.laps == 0
    assert core.phase == EXPLORE
    assert 'planning' not in core.events


def test_first_valid_return_counts_and_submits_map_plan_after_the_overlap():
    core, fixture = replay(oval_core().cfg.closure.arm_distance_m)
    assert core.counter.laps == 1
    crossing = core.events['crossings'][0]
    assert crossing['t'] - fixture['samples'][0]['t'] == pytest.approx(48.163, abs=.01)
    assert crossing['travel'] == pytest.approx(8.341, abs=.01)
    assert abs(crossing['lateral']) < .04
    assert crossing['heading_err_deg'] < 4.
    assert core.phase == PLANNING and 'planning' in core.events
    assert core.events['planning']['t_submit'] >= core.events['closing']['t']
    assert core.close_travel + core.cfg.closure.overlap_m <= core.travel
