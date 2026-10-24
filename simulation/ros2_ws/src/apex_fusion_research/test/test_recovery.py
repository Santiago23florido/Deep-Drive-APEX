"""Recovery uses swept geometry in either direction, without track templates."""
from dataclasses import replace
import numpy as np
import pytest

from apex_fusion_research.core.race_config import ReactiveConfig, VehicleLimits
from apex_fusion_research.core.recovery import arc_poses, choose_recovery_arc, reverse_path_clear

VEH = VehicleLimits(steer_plan_deg=22.5, steer_max_deg=22.5)
CFG = ReactiveConfig()


def front_wall():
    return np.column_stack((np.full(201, .55), np.linspace(-2, 2, 201)))


def test_reverse_geometry_is_exact_for_signed_distance_and_reference_offset():
    pose = arc_poses(np.array([0., 1.]), np.array([-.5]), VEH)
    np.testing.assert_allclose(pose[0, 0], [-.5, 0., 0.], atol=1e-12)
    assert pose[1, 0, 2] == pytest.approx(-.5)
    assert pose[1, 0, 0] == pytest.approx(VEH.rear_offset_m + np.sin(-.5) - VEH.rear_offset_m * np.cos(-.5))


def test_recovery_is_mirrored_when_the_geometry_and_vehicle_are_mirrored():
    cloud = np.vstack((front_wall(), [[-.8, .5], [-.6, .5], [-.4, .5]]))
    right = choose_recovery_arc(cloud, VEH, CFG, .5, -.8)
    mirror = cloud * [1, -1]
    left = choose_recovery_arc(mirror, VEH, CFG, .5, .8)
    assert right is not None and left is not None
    assert abs(right.kappa) > .05  # exercise curved recovery, not a vacuous straight mirror
    assert right.kappa == pytest.approx(-left.kappa)
    assert right.forward_free_m == pytest.approx(left.forward_free_m)
    assert reverse_path_clear(cloud, VEH, CFG, right.kappa, right.distance_m)
    assert reverse_path_clear(mirror, VEH, CFG, left.kappa, left.distance_m)


def test_swept_corner_obstacle_can_block_a_curved_reverse_while_straight_reverse_fits():
    cloud = np.array([[-.6, .4]])
    assert reverse_path_clear(cloud, VEH, CFG, 0., .5)
    assert not reverse_path_clear(cloud, VEH, CFG, VEH.curvature_limit(True, planning=True), .5)


def test_a_rear_wall_blocks_all_reverse_arcs():
    rear = np.column_stack((np.full(201, -.4), np.linspace(-2, 2, 201)))
    assert choose_recovery_arc(np.vstack((front_wall(), rear)), VEH, CFG, .5, -.8) is None


def test_missing_cloud_is_not_permission_to_reverse():
    assert choose_recovery_arc(np.zeros((0, 2)), VEH, CFG, .5, -.8) is None
    assert not reverse_path_clear(np.zeros((0, 2)), VEH, CFG, 0., .5)


def test_asymmetric_servo_limits_and_rear_distance_budget_are_respected():
    veh = replace(VEH, steer_plan_left_deg=22.5, steer_plan_right_deg=12.96)
    arc = choose_recovery_arc(front_wall(), veh, CFG, .28, -.7)
    assert arc is not None
    assert -veh.curvature_limit(False, planning=True) <= arc.kappa <= veh.curvature_limit(True, planning=True)
    assert .1 <= arc.distance_m <= .28


def test_new_obstacle_cuts_the_remaining_reverse_path():
    assert reverse_path_clear(front_wall(), VEH, CFG, 0., .5)
    blocked = np.vstack((front_wall(), [[-.4, 0.]]))
    assert not reverse_path_clear(blocked, VEH, CFG, 0., .5)
