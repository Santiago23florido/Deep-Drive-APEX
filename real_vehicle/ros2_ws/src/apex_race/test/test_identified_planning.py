"""Keep planning, learned compensation and the phase handover in the same units."""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from apex_race.core.race_driver import CLOSING, MAP_GUIDED, PLANNING
from apex_race.core.race_planner import Grid, path_from_points
from apex_race.core.reactive import ReactiveCommand
from test_oval_profile import oval_core


def learned_core():
    core = oval_core()
    core.model = replace(core.model, gain_left=1.5904072858, gain_right=1.6, trim=-.2422761102)
    core.env = core.shaper.envelope(core.model)
    core._identified_plan_limits()
    return core


@pytest.mark.parametrize('gain', [.6, 1., 1.6])
def test_command_space_saturation_is_not_divided_by_the_learned_gain_again(gain):
    core = oval_core()
    model = replace(core.model, gain_left=gain, gain_right=gain)
    env = core.shaper.envelope(model)
    assert env.kappa_limit_right == pytest.approx(core.cfg.vehicle.curvature_limit(False))
    assert env.kappa_limit_left == pytest.approx(core.cfg.vehicle.curvature_limit(True))


def test_local_arc_geometry_uses_the_forward_of_the_same_identified_inverse():
    core = learned_core()
    left = core.model.kappa_out(core.env.kappa_limit_left, 0.)
    right = -core.model.kappa_out(-core.env.kappa_limit_right, 0.)
    assert core.reactive.planner.k1.min() == pytest.approx(-right)
    assert core.reactive.planner.k1.max() == pytest.approx(left)
    assert right > core.cfg.vehicle.curvature_limit(False)
    assert core.cfg.vehicle.steering_limit_deg(False) == 12.96  # actuator limits were not expanded


def test_global_planning_receives_identified_geometry_and_preserves_the_posterior():
    core = learned_core()
    model = core.model
    core.start = (0., 0., 0.)
    core.grid = (0., Grid(np.zeros((10, 10)), .1, 0., 0.))
    a = np.linspace(0, 2 * np.pi, 80)
    core.path_log = np.column_stack((np.sin(a), 1 - np.cos(a)))
    captured = []
    core.submit = lambda fn, *args: captured.append(args[-1])
    core._submit_plan(1.)
    assert captured[0].vehicle.curvature_limit(False, planning=True) == pytest.approx(
        core.reactive.veh.curvature_limit(False, planning=True))
    assert core.model is model


def test_physical_reverse_arc_is_inverted_to_the_existing_mechanical_command_limit():
    core = learned_core()
    core.phase = MAP_GUIDED
    k = -core.reactive.veh.curvature_limit(False, planning=True)
    core.recover, core.rear_clear = 2, 1.
    core.recover_until, core.recover_distance_limit = 10., .5
    core.recover_kappa = k
    core.steer = math.atan(k * core.cfg.vehicle.wheelbase_m)
    core.last_t = core.shaper.t_prev = 0.
    core.shaper.k_smooth = k
    _, steer, _ = core.control(.1, (0., 0., 0.), 0., None, True)
    assert core.kappa_des == pytest.approx(k)
    assert math.degrees(steer) == pytest.approx(-12.96)


def test_planning_continues_from_current_pose_and_preserves_the_model():
    core = learned_core()
    core.phase = PLANNING
    model, identifier, shaper = core.model, core.ident, core.shaper
    core.local_path_odom = np.column_stack((np.linspace(2, 5, 30), np.ones(30)))
    core.cmd = ReactiveCommand(0., .17, core.local_path_odom, True, {'free': 2.})
    core._recovery = lambda *args: None
    core.future = SimpleNamespace(done=lambda: False)
    core.last_t = core.shaper.t_prev = 0.
    core.shaper.v_out = .17
    assert core.control(1., (2., 1., 0.), .1, None, True)[0] > 0.
    assert core.model is model and core.ident is identifier and core.shaper is shaper
    a = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    path = path_from_points(np.column_stack((2 + np.cos(a), 1 + np.sin(a))))
    plan = SimpleNamespace(diag={'direct_tracking_allowed': False}, route=path, path=path)
    core.future = SimpleNamespace(done=lambda: True, result=lambda: plan)
    pose = (*path.xy[20], float(path.heading[20]))
    core.control(2., pose, .1, None, True)
    assert core.phase == MAP_GUIDED and core.guided
    assert core.model is model and core.ident is identifier and core.shaper is shaper
    guide = core._race_line_in(pose)
    np.testing.assert_allclose(guide[0], [0., 0.], atol=1e-12)
    assert core.map_route_s > .5  # entry is the current station, not route station zero
