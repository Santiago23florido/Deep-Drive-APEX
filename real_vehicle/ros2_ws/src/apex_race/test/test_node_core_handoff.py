"""Sensor callbacks must yield to waiting control, whose actual outputs feed ID."""
from collections import deque
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

pytest.importorskip('rclpy')
from apex_race.nodes.race_driver_node import RaceDriverNode


def bare_node():
    node = RaceDriverNode.__new__(RaceDriverNode)
    node._core_lock = threading.Lock()
    node._control_served = threading.Event()
    node._control_served.set()
    node._handoff_timeout_s = .2
    node._fallback_commands = deque()
    node.hold_ms = {name: [0., 0., 0.] for name in ('scan', 'odom')}
    node.core = SimpleNamespace(ident=SimpleNamespace(push_command=Mock()))
    return node


def test_sensor_handoff_leaves_the_lock_available_for_pending_control():
    node = bare_node()
    attempting, entered, done = threading.Event(), threading.Event(), threading.Event()
    node._control_served.clear()  # a timer already missed the current scan

    def sensor():
        attempting.set()
        with node._core('scan'):
            entered.set()
        done.set()

    worker = threading.Thread(target=sensor)
    worker.start()
    try:
        assert attempting.wait(timeout=1)
        assert not entered.wait(timeout=.02)
        assert node._core_lock.acquire(blocking=False)
        node._core_lock.release()
        node._control_served.set()
        assert done.wait(timeout=1)
    finally:
        node._control_served.set()
        worker.join(timeout=1)


def test_handoff_has_a_timeout_if_control_is_not_scheduled():
    node = bare_node()
    node._handoff_timeout_s = .001
    node._control_served.clear()
    with node._core('scan'):
        assert node._core_lock.locked()


def test_actual_holds_and_watchdog_zeros_are_flushed_before_measurements():
    node = bare_node()
    node._fallback_commands.extend([(1.0, -.3, .3), (1.02, -.3, .3), (1.04, 0., 0.)])
    with node._core('odom'):
        assert node.core.ident.push_command.call_args_list == [
            ((1.0, -.3, .3),), ((1.02, -.3, .3),), ((1.04, 0., 0.),)]
        assert not node._fallback_commands
