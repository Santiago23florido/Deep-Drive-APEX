#!/usr/bin/env python3
"""PWM helpers for Raspberry Pi sysfs PWM (compatible with full_soft behavior)."""

from __future__ import annotations

import glob
import logging
import os
import time


LOGGER = logging.getLogger(__name__)


class PWM:
    """Simple sysfs PWM wrapper.

    This keeps the same control approach used in `full_soft/VoitureAutonome/code/raspberry_pwm.py`.
    """

    def __init__(self, channel: int, frequency_hz: float = 50.0) -> None:
        self.channel = int(channel)
        self.frequency_hz = float(frequency_hz)
        self.period_ns = int(1.0e9 / self.frequency_hz)

        self.chip_path = self._find_pwm_chip()
        self.pwm_dir = os.path.join(self.chip_path, f"pwm{self.channel}")

        self._ensure_exported()
        self._configure_period()

    @staticmethod
    def _find_pwm_chip() -> str:
        candidates = sorted(glob.glob("/sys/class/pwm/pwmchip*"))
        if not candidates:
            raise FileNotFoundError("No PWM chips found under /sys/class/pwm")
        return candidates[0]

    def _write(self, path: str, value: int) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{int(value)}\n")

    def _ensure_exported(self) -> None:
        if os.path.isdir(self.pwm_dir):
            return

        export_path = os.path.join(self.chip_path, "export")
        self._write(export_path, self.channel)

        timeout_s = 5.0
        start = time.time()
        while not os.path.isdir(self.pwm_dir):
            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"PWM channel directory not created: {self.pwm_dir}")
            time.sleep(0.05)

    def _configure_period(self) -> None:
        period_path = os.path.join(self.pwm_dir, "period")
        retries = 8
        for attempt in range(retries):
            try:
                self._write(period_path, self.period_ns)
                return
            except PermissionError:
                if attempt == retries - 1:
                    raise
                LOGGER.warning("Permission denied setting PWM period, retrying (%d/%d)", attempt + 1, retries)
                time.sleep(0.1)

    def set_duty_cycle_percent(self, duty_percent: float) -> None:
        duty = max(0.0, min(100.0, float(duty_percent)))
        duty_ns = int(self.period_ns * duty / 100.0)
        self._write(os.path.join(self.pwm_dir, "duty_cycle"), duty_ns)

    def start(self, duty_percent: float) -> None:
        self.set_duty_cycle_percent(duty_percent)
        self._write(os.path.join(self.pwm_dir, "enable"), 1)

    def stop(self) -> None:
        try:
            self.set_duty_cycle_percent(0.0)
        finally:
            self._write(os.path.join(self.pwm_dir, "enable"), 0)

