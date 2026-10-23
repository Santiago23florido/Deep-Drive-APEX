"""The real ROS control callback must stop even with the planning lock held."""
from types import SimpleNamespace
from collections import deque
import threading
from unittest.mock import Mock

import pytest

pytest.importorskip('rclpy')
from apex_race.nodes import race_driver_node as module


def node_at(deadline=5.0):
    # Bypass ROS graph setup: invoke the production callback with a locked
    # planner, a clock and a recorded publisher; no PWM/backend is created.
    node = module.RaceDriverNode.__new__(module.RaceDriverNode)
    node._now = lambda: 100_000_000_000
    node._arm_reason = lambda now: ''
    node._publish_cmd = Mock()
    node._hold_command = Mock()
    node._write_events = Mock()
    node._mission_deadline_s = deadline
    node._mission_time_stop = False
    node._core_lock = threading.Lock()
    node._control_served = threading.Event()
    node._control_served.set()
    node._fallback_commands = deque()
    node.core = SimpleNamespace(phase='explore', _stop_mission=Mock(), ident=SimpleNamespace(push_command=Mock()))
    return node


def test_deadline_stops_without_waiting_for_planner_lock(monkeypatch):
    node = node_at()
    node._core_lock.acquire()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 5.0)
    try:
        node._control()
    finally:
        node._core_lock.release()
    node._publish_cmd.assert_called_once_with(0.0, 0.0)
    node._hold_command.assert_not_called()
    node.core._stop_mission.assert_not_called()
    assert node._mission_time_stop


def test_before_deadline_the_existing_busy_stop_logic_still_applies(monkeypatch):
    node = node_at()
    node._core_lock.acquire()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 4.99)
    try:
        node._control()
    finally:
        node._core_lock.release()
    node._hold_command.assert_called_once()
    assert not node._mission_time_stop


def test_deadline_latches_done_once_when_core_is_available(monkeypatch):
    node = node_at()
    monkeypatch.setattr(module.time, 'monotonic', lambda: 5.1)
    node.core._stop_mission.side_effect = lambda *args: setattr(node.core, 'phase', 'done')
    node._control()
    node._control()
    assert node._publish_cmd.call_count == 2
    node.core._stop_mission.assert_called_once_with(100.0, 'duration')
    node._write_events.assert_called_once()
    assert not node._core_lock.locked()
