#!/usr/bin/env python3
"""Minimal sysfs PWM helper for Raspberry-hosted actuation."""

from __future__ import annotations

import errno
import glob
import os
import time
from typing import Any


class HardwarePWM:
    """Simple PWM wrapper around `/sys/class/pwm`."""

    def __init__(self, channel: int, frequency_hz: float, logger) -> None:
        self._channel = int(channel)
        self._frequency_hz = float(frequency_hz)
        self._logger = logger

        self._chip_path = self._find_chip_path()
        self._pwm_dir = os.path.join(self._chip_path, f"pwm{self._channel}")
        self._period_ns = int(round(1.0e9 / self._frequency_hz))

        self._ensure_channel()
        self._write_int(os.path.join(self._pwm_dir, "period"), self._period_ns)

    def _find_chip_path(self) -> str:
        chips = sorted(glob.glob("/sys/class/pwm/pwmchip*"))
        if not chips:
            raise FileNotFoundError("No PWM chip found under /sys/class/pwm")
        return chips[0]

    def _ensure_channel(self) -> None:
        if os.path.isdir(self._pwm_dir):
            return

        export_path = os.path.join(self._chip_path, "export")
        try:
            self._write_int(export_path, self._channel)
        except OSError as exc:
            if exc.errno != errno.EINVAL:
                raise

        deadline = time.time() + 5.0
        while not os.path.isdir(self._pwm_dir):
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for PWM directory {self._pwm_dir}")
            time.sleep(0.05)

    @staticmethod
    def _write_text(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _write_int(self, path: str, value: int) -> None:
        self._write_text(path, f"{int(value)}\n")

    @staticmethod
    def _read_int(path: str) -> int | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except Exception:
            return None

    def set_duty_cycle(self, duty_cycle_pct: float) -> None:
        duty_cycle_pct = max(0.0, min(100.0, float(duty_cycle_pct)))
        active_ns = int(round(self._period_ns * duty_cycle_pct / 100.0))
        self._write_int(os.path.join(self._pwm_dir, "duty_cycle"), active_ns)

    def start(self, duty_cycle_pct: float) -> None:
        self.set_duty_cycle(duty_cycle_pct)
        self._write_int(os.path.join(self._pwm_dir, "enable"), 1)
        self._logger.debug(
            "PWM started on %s at %.3f%%" % (self._pwm_dir, float(duty_cycle_pct))
        )

    def disable(self) -> None:
        try:
            self._write_int(os.path.join(self._pwm_dir, "enable"), 0)
        except Exception:
            pass

    def stop(self) -> None:
        self.disable()

    def get_state(self) -> dict[str, Any]:
        return {
            "pwm_dir": self._pwm_dir,
            "period_ns": self._read_int(os.path.join(self._pwm_dir, "period")),
            "duty_cycle_ns": self._read_int(os.path.join(self._pwm_dir, "duty_cycle")),
            "enabled": self._read_int(os.path.join(self._pwm_dir, "enable")),
        }
