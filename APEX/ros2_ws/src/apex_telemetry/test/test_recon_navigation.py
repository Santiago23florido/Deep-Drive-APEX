import unittest

import numpy as np

from apex_telemetry.recon_navigation import ReconNavigator


def _make_navigator(**overrides):
    params = {
        "steering_limit_deg": 18.0,
        "steering_gain": 0.35,
        "fov_half_angle_deg": 90.0,
        "smoothing_window": 9,
        "stop_distance_m": 0.35,
        "slow_distance_m": 0.90,
        "min_speed_pct": 18.0,
        "max_speed_pct": 24.0,
        "front_window_deg": 12,
        "side_window_deg": 70,
        "center_angle_penalty_per_deg": 0.012,
        "wall_centering_gain_deg_per_m": 45.0,
        "wall_centering_limit_deg": 18.0,
        "wall_centering_base_weight": 0.90,
        "turn_speed_reduction": 0.35,
        "min_turn_speed_factor": 0.70,
        "vehicle_half_width_m": 0.11,
        "vehicle_front_overhang_m": 0.11,
        "vehicle_rear_overhang_m": 0.31,
    }
    params.update(overrides)
    return ReconNavigator(**params)


def _full_open_scan(value=3.0):
    return np.full(360, value, dtype=np.float32)


def _apply_indices(scan, indices, value):
    for index in indices:
        scan[int(index) % scan.size] = value
    return scan


class ReconNavigatorTests(unittest.TestCase):
    def test_centered_corridor_keeps_straight_command(self):
        navigator = _make_navigator()
        command = navigator.compute_command(_full_open_scan())

        self.assertAlmostEqual(command.gap_heading_deg, 0.0, places=3)
        self.assertAlmostEqual(command.target_heading_deg, 0.0, places=3)
        self.assertAlmostEqual(command.steering_deg, 0.0, places=3)
        self.assertGreater(command.speed_pct, 0.0)

    def test_front_obstacle_stops_vehicle(self):
        navigator = _make_navigator()
        scan = _full_open_scan()
        _apply_indices(scan, list(range(350, 360)) + list(range(0, 11)), 0.2)

        command = navigator.compute_command(scan)

        self.assertEqual(command.speed_pct, 0.0)
        self.assertLess(command.front_clearance_m, 0.35)

    def test_mirrored_side_obstacles_produce_mirrored_steering(self):
        navigator = _make_navigator()

        positive_block_scan = _full_open_scan()
        _apply_indices(positive_block_scan, range(10, 70), 0.4)
        positive_block_command = navigator.compute_command(positive_block_scan)

        negative_block_scan = _full_open_scan()
        _apply_indices(negative_block_scan, range(290, 350), 0.4)
        negative_block_command = navigator.compute_command(negative_block_scan)

        self.assertNotEqual(positive_block_command.steering_deg, 0.0)
        self.assertNotEqual(negative_block_command.steering_deg, 0.0)
        self.assertLess(
            positive_block_command.steering_deg * negative_block_command.steering_deg,
            0.0,
        )

    def test_corner_avoidance_delta_changes_heading_away_from_nearby_corner(self):
        navigator = _make_navigator()

        positive_corner_scan = _full_open_scan(3.0)
        _apply_indices(positive_corner_scan, range(1, 12), 0.5)
        positive_target_heading, positive_delta = navigator._apply_avoid_corner(
            0.0,
            positive_corner_scan,
        )

        negative_corner_scan = _full_open_scan(3.0)
        _apply_indices(negative_corner_scan, list(range(349, 360)), 0.5)
        negative_target_heading, negative_delta = navigator._apply_avoid_corner(
            0.0,
            negative_corner_scan,
        )

        self.assertNotEqual(positive_delta, 0.0)
        self.assertNotEqual(negative_delta, 0.0)
        self.assertLess(positive_target_heading * negative_target_heading, 0.0)


if __name__ == "__main__":
    unittest.main()
