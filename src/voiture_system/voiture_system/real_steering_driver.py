#!/usr/bin/env python3
"""Real steering servo driver based on full_soft PWM logic."""

from __future__ import annotations

import logging

from .hardware_pwm import PWM


LOGGER = logging.getLogger(__name__)


class RealSteeringDriver:
    """Servo wrapper using steering angle in degrees."""

    def __init__(
        self,
        pwm_channel: int = 1,
        pwm_frequency_hz: float = 50.0,
        steering_limit_deg: float = 18.0,
        dc_steer_min: float = 5.0,
        dc_steer_max: float = 8.6,
    ) -> None:
        self.steering_limit_deg = max(1e-3, abs(float(steering_limit_deg)))
        self.dc_steer_min = float(dc_steer_min)
        self.dc_steer_max = float(dc_steer_max)
        self.dc_steer_center = 0.5 * (self.dc_steer_min + self.dc_steer_max)
        self.steer_variation_rate = 0.5 * (self.dc_steer_max - self.dc_steer_min) / self.steering_limit_deg

        self._pwm = PWM(channel=pwm_channel, frequency_hz=pwm_frequency_hz)
        self._pwm.start(self.dc_steer_center)
        self._angle_deg = 0.0

    @property
    def angle_deg(self) -> float:
        return self._angle_deg

    def set_steering_angle_deg(self, angle_deg: float) -> float:
        angle = max(-self.steering_limit_deg, min(self.steering_limit_deg, float(angle_deg)))
        duty = angle * self.steer_variation_rate + self.dc_steer_center
        self._pwm.set_duty_cycle_percent(duty)
        self._angle_deg = angle
        return angle

    def stop(self) -> None:
        try:
            self._pwm.set_duty_cycle_percent(self.dc_steer_center)
            self._angle_deg = 0.0
        finally:
            self._pwm.stop()
        LOGGER.info("Steering stopped")

