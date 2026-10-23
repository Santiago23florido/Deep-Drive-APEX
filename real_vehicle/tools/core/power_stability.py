"""Observe host power readiness without enabling or commanding actuators."""
from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Callable


def read_host_power() -> tuple[int, float]:
    flags = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=3)
    temperature = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=3)
    return int(flags.strip().split("=", 1)[1], 16), float(temperature.split("=", 1)[1].split("'")[0])


def wait_for_stable_power(
    *,
    stable_s: float = 10.0,
    timeout_s: float = 90.0,
    poll_s: float = 1.0,
    max_gap_s: float = 2.5,
    read: Callable[[], tuple[int, float]] = read_host_power,
    on_sample: Callable[[dict], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Require sustained clean live flags and <75 C within a bounded wait.

    Faults, unavailable readings and observation gaps reset the clean window.
    Historical flags alone are allowed. Success is readiness only: the caller
    must still perform its preflight and final power check before arming.
    """
    if not (0 < stable_s <= timeout_s and 0 < poll_s <= max_gap_s):
        raise ValueError("invalid power stability observation bounds")
    start = clock()
    clean_since = previous = None
    rows = []
    while True:
        row = {}
        try:
            flags, temp = read()
            clean = flags & 0xffff == 0 and math.isfinite(temp) and temp < 75.0
            row.update(throttled=hex(flags), temp_c=temp)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
            clean = False
            row["error"] = str(exc)
        now = clock()
        gap = previous is not None and now - previous > max_gap_s
        if not clean or gap:
            clean_since = None
        if clean and clean_since is None:
            clean_since = now
        elapsed = now - start
        clean_elapsed = 0.0 if clean_since is None else now - clean_since
        row.update(elapsed_s=elapsed, clean=clean, observation_gap=gap, clean_elapsed_s=clean_elapsed)
        rows.append(row)
        if on_sample is not None:
            on_sample(row)
        # A late read must not turn an expired observation into success.
        if elapsed > timeout_s:
            raise TimeoutError("power did not stabilize within the observation deadline")
        if clean and clean_elapsed >= stable_s:
            return rows
        if elapsed >= timeout_s:
            raise TimeoutError("power did not stabilize within the observation deadline")
        previous = now
        sleep(min(poll_s, timeout_s - elapsed))
