"""Known observation sequences must never grant readiness across a fault."""
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[4] / "tools/core/power_stability.py"
spec = importlib.util.spec_from_file_location("power_stability", MODULE)
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_history_only_requires_a_full_ten_second_window():
    clock = Clock()
    rows = power.wait_for_stable_power(read=lambda: (0x50000, 60.0), clock=clock.now, sleep=clock.sleep)
    assert clock.t == 10
    assert len(rows) == 11


@pytest.mark.parametrize("bad", [(0x50005, 60.0), (0x50000, 75.0), (0x50000, float("nan"))])
def test_a_fault_at_nine_seconds_restarts_the_entire_window(bad):
    clock = Clock()
    rows = power.wait_for_stable_power(read=lambda: bad if clock.t == 9 else (0x50000, 60.0),
                                       clock=clock.now, sleep=clock.sleep)
    assert clock.t == 20
    assert rows[9]["clean_elapsed_s"] == 0


def test_persistent_undervoltage_cannot_grant_readiness_at_timeout():
    clock = Clock()
    with pytest.raises(TimeoutError):
        power.wait_for_stable_power(timeout_s=12, read=lambda: (0x50005, 50.0), clock=clock.now, sleep=clock.sleep)
    assert clock.t == 12


def test_unavailable_data_resets_readiness():
    clock = Clock()
    def read():
        if clock.t == 9:
            raise OSError("host telemetry unavailable")
        return 0x50000, 60.0
    rows = power.wait_for_stable_power(read=read, clock=clock.now, sleep=clock.sleep)
    assert clock.t == 20
    assert "error" in rows[9]


def test_a_long_unobserved_interval_cannot_count_as_clean_time():
    clock = Clock()
    calls = 0
    def read():
        nonlocal calls
        calls += 1
        if calls == 10:
            clock.t += 6
        return 0x50000, 60.0
    rows = power.wait_for_stable_power(read=read, clock=clock.now, sleep=clock.sleep)
    assert clock.t == 25
    assert rows[9]["observation_gap"]


def test_read_returning_after_deadline_does_not_grant_readiness():
    clock = Clock()
    def read():
        clock.t += 20
        return 0x50000, 60.0
    with pytest.raises(TimeoutError):
        power.wait_for_stable_power(timeout_s=12, read=read, clock=clock.now, sleep=clock.sleep)
