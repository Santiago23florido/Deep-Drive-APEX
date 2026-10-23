"""Replay real blocked-corner geometry without ROS or actuator access."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apex_race.core.recovery import choose_recovery_arc, reverse_path_clear
from test_oval_profile import blocked_core, oval_core


def test_longer_backout_opens_more_forward_space_in_the_recorded_corner():
    fixture = json.loads((Path(__file__).parent / 'data/vertex_cloud.json').read_text())
    cloud, cfg = np.array(fixture['points']), oval_core().cfg
    short = choose_recovery_arc(cloud, cfg.vehicle, cfg.reactive, .25, fixture['desired_kappa'])
    long = choose_recovery_arc(cloud, cfg.vehicle, cfg.reactive, .5, fixture['desired_kappa'])
    assert short is not None and long is not None
    assert long.forward_free_m > short.forward_free_m
    assert reverse_path_clear(cloud, cfg.vehicle, cfg.reactive, long.kappa, long.distance_m)


def test_new_obstacle_in_a_curved_swept_path_stops_core_immediately():
    core = blocked_core()
    core.recover_path_clear = False
    assert core.control(1.22, (0, 0, 0), -.2, None, True)[0] == 0
    assert core.recover == 0 and core.ident.pending[1] == 0


def test_core_uses_the_geometry_selected_reverse_steering():
    core = blocked_core()
    core.recover = 0
    core.block_since = 0.
    core.cmd.kappa = -.5
    core.pose_odom = (0., 0., 0.)
    cloud = np.column_stack((np.full(201, .55), np.linspace(-2, 2, 201)))
    core.acc = SimpleNamespace(empty=False, cloud_in=lambda pose: cloud)
    arc = choose_recovery_arc(cloud, core.reactive.veh, core.reactive.cfg, .5, -.5,
                              core.cfg.adaptive.recover_clearance_m)
    speed, steer = core._recovery(2., 0.)
    assert arc is not None and abs(arc.kappa) > .05
    assert speed < 0 and core.recover == 2
    assert np.tan(steer) / core.cfg.vehicle.wheelbase_m == pytest.approx(arc.kappa)
    assert core.recover_distance_limit == pytest.approx(arc.distance_m)


def test_lateral_return_behind_bumper_does_not_block_a_checked_reverse():
    fixture = json.loads((Path(__file__).parent / 'data/side_return_blocked_start.json').read_text())
    cloud = np.array(fixture['points'])
    core = blocked_core()
    core.recover = 0
    core.block_since = 0.
    core.pose_odom = (0., 0., 0.)
    core.cmd.kappa = -core.cfg.vehicle.curvature_limit(False, planning=True)
    core.rear_clear = core._rear_clearance(cloud)
    assert core.rear_clear < .03  # the old axis projection falsely blocked this start
    core.acc = SimpleNamespace(empty=False, cloud_in=lambda pose: cloud)
    speed, _ = core._recovery(2., 0.)
    assert speed < 0 and core.recover_geometric
    assert core.recover_distance_limit == pytest.approx(.5)
    assert reverse_path_clear(cloud, core.reactive.veh, core.reactive.cfg,
                              core.recover_kappa, .5, core.cfg.adaptive.recover_clearance_m)
    assert core._recovery(2.02, 0.)[0] < 0  # do not reapply the false projection during motion


def test_rear_reserve_still_blocks_a_real_wall_across_the_reverse_path():
    cfg = oval_core().cfg
    wall = np.column_stack((np.full(201, -.85), np.linspace(-2, 2, 201)))
    assert reverse_path_clear(wall, cfg.vehicle, cfg.reactive, 0., .4)
    assert not reverse_path_clear(wall, cfg.vehicle, cfg.reactive, 0., .4, cfg.adaptive.recover_clearance_m)


def test_squeezed_front_does_not_weaken_the_hard_margin_to_a_rear_obstacle():
    cfg = oval_core().cfg
    wall = np.column_stack((np.full(201, -.45), np.linspace(-2, 2, 201)))
    cloud = np.vstack((wall, [[.284, .126]]))
    assert choose_recovery_arc(cloud, cfg.vehicle, cfg.reactive, .5, -.7, .15) is None
