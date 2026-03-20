#!/usr/bin/env python3
"""Real ESC motor driver based on full_soft PWM logic."""

from __future__ import annotations

import logging
import time

from .hardware_pwm import PWM


LOGGER = logging.getLogger(__name__)


class RealMotorDriver:
    """ESC wrapper using normalized speed command in [-3.0, 3.0]."""

    def __init__(
        self,
        pwm_channel: int = 0,
        pwm_frequency_hz: float = 50.0,
        esc_dc_min: float = 5.0,
        esc_dc_max: float = 10.0,
    ) -> None:
        self.esc_dc_min = float(esc_dc_min)
        self.esc_dc_max = float(esc_dc_max)
        self.esc_dc_neutral = 0.5 * (self.esc_dc_min + self.esc_dc_max)

        self._pwm = PWM(channel=pwm_channel, frequency_hz=pwm_frequency_hz)
        self._pwm.start(self.esc_dc_neutral)

        self._speed_cmd = 0.0
        self._in_reverse_mode = False

    @property
    def speed_cmd(self) -> float:
        return self._speed_cmd

    def _enter_reverse_mode(self) -> None:
        # Full-soft ESC sequence: brake -> neutral before reverse.
        self._pwm.set_duty_cycle_percent(7.0)
        time.sleep(0.03)
        self._pwm.set_duty_cycle_percent(self.esc_dc_neutral)
        time.sleep(0.03)
        self._in_reverse_mode = True

    def _exit_reverse_mode(self) -> None:
        self._pwm.set_duty_cycle_percent(self.esc_dc_neutral)
        time.sleep(0.1)
        self._in_reverse_mode = False

    def set_speed(self, speed_cmd: float) -> float:
        """Set normalized speed command in [-3, 3]. Returns applied command."""
        cmd = max(-3.0, min(3.0, float(speed_cmd)))
        self._speed_cmd = cmd

        if cmd < 0.0 and not self._in_reverse_mode:
            self._enter_reverse_mode()
        elif cmd >= 0.0 and self._in_reverse_mode:
            self._exit_reverse_mode()

        if abs(cmd) < 0.1:
            duty = self.esc_dc_neutral
        elif cmd >= 0.0:
            duty = self.esc_dc_neutral + (cmd / 3.0) * (self.esc_dc_max - self.esc_dc_neutral)
        else:
            duty = self.esc_dc_min + ((cmd + 3.0) / 3.0) * (self.esc_dc_neutral - self.esc_dc_min)

        self._pwm.set_duty_cycle_percent(duty)
        return cmd

    def stop(self) -> None:
        try:
            self._pwm.set_duty_cycle_percent(self.esc_dc_neutral)
            self._in_reverse_mode = False
            self._speed_cmd = 0.0
        finally:
            self._pwm.stop()
        LOGGER.info("Motor stopped")

