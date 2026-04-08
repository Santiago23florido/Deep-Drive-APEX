#!/usr/bin/env python3
"""Low-level steering and motor actuation for the APEX reconnaissance node."""

from __future__ import annotations

import time

from .hardware_pwm import HardwarePWM


class SteeringServo:
    def __init__(
        self,
        channel: int,
        frequency_hz: float,
        limit_deg: float,
        dc_min: float,
        dc_max: float,
        center_trim_dc: float,
        direction_sign: float,
        min_authority_ratio: float,
        logger,
    ) -> None:
        self._logger = logger
        self._limit_deg = max(1.0, float(limit_deg))
        self._dc_min = float(dc_min)
        self._dc_max = float(dc_max)
        self._center_trim_dc = float(center_trim_dc)
        self._direction_sign = -1.0 if float(direction_sign) < 0.0 else 1.0
        self._min_authority_ratio = max(0.0, min(1.0, float(min_authority_ratio)))
        raw_dc_center = 0.5 * (self._dc_min + self._dc_max) + self._center_trim_dc
        half_span = 0.5 * (self._dc_max - self._dc_min)
        required_half_span = self._min_authority_ratio * half_span
        dc_center_min = self._dc_min + required_half_span
        dc_center_max = self._dc_max - required_half_span
        if dc_center_min <= dc_center_max:
            self._dc_center = min(max(raw_dc_center, dc_center_min), dc_center_max)
        else:
            self._dc_center = 0.5 * (self._dc_min + self._dc_max)
        self._center_trim_effective_dc = self._dc_center - (0.5 * (self._dc_min + self._dc_max))
        if abs(self._dc_center - raw_dc_center) > 1.0e-6:
            self._logger.warning(
                "Steering center trim %.3f%% reduced to %.3f%% to preserve %.0f%% steering authority"
                % (
                    self._center_trim_dc,
                    self._center_trim_effective_dc,
                    100.0 * self._min_authority_ratio,
                )
            )
        self._variation_per_deg = 0.5 * (self._dc_max - self._dc_min) / self._limit_deg
        self._requested_angle_deg = 0.0
        self._pre_sign_angle_deg = 0.0
        self._post_sign_angle_deg = 0.0
        self._clamped_angle_deg = 0.0
        self._applied_angle_deg = 0.0
        self._last_duty_cycle_pct = self._dc_center

        self._pwm = HardwarePWM(channel=channel, frequency_hz=frequency_hz, logger=logger)
        self._pwm.start(self._dc_center)
        self._current_angle_deg = 0.0

    def set_angle_deg(self, angle_deg: float) -> None:
        self._requested_angle_deg = float(angle_deg)
        requested_angle_deg = self._requested_angle_deg
        signed_angle_deg = requested_angle_deg * self._direction_sign
        duty_cycle = self._dc_center + signed_angle_deg * self._variation_per_deg
        duty_cycle = max(self._dc_min, min(self._dc_max, duty_cycle))
        if abs(self._variation_per_deg) > 1.0e-9:
            clamped_signed_angle_deg = (duty_cycle - self._dc_center) / self._variation_per_deg
        else:
            clamped_signed_angle_deg = 0.0
        self._pwm.set_duty_cycle(duty_cycle)
        self._pre_sign_angle_deg = requested_angle_deg
        self._post_sign_angle_deg = signed_angle_deg
        self._clamped_angle_deg = clamped_signed_angle_deg
        self._applied_angle_deg = clamped_signed_angle_deg * self._direction_sign
        self._last_duty_cycle_pct = duty_cycle
        self._current_angle_deg = self._applied_angle_deg

    def center(self) -> None:
        self.set_angle_deg(0.0)

    def stop(self) -> None:
        self.center()
        self._pwm.stop()

    def get_pwm_state(self) -> dict[str, float | int | str | None]:
        return self._pwm.get_state()

    def get_state(self) -> dict[str, float]:
        return {
            "requested_deg": self._requested_angle_deg,
            "pre_sign_deg": self._pre_sign_angle_deg,
            "post_sign_deg": self._post_sign_angle_deg,
            "clamped_deg": self._clamped_angle_deg,
            "applied_deg": self._applied_angle_deg,
            "pwm_dc": self._last_duty_cycle_pct,
            "dc_center": self._dc_center,
            "center_trim_dc": self._center_trim_dc,
            "center_trim_effective_dc": self._center_trim_effective_dc,
            "steering_direction_sign": self._direction_sign,
            "steering_limit_deg": self._limit_deg,
        }


class MaverickESCMotor:
    def __init__(
        self,
        channel: int,
        frequency_hz: float,
        dc_min: float,
        dc_max: float,
        neutral_dc: float,
        reverse_brake_dc: float,
        reverse_brake_hold_s: float,
        reverse_neutral_hold_s: float,
        reverse_exit_hold_s: float,
        logger,
    ) -> None:
        self._logger = logger
        self._dc_min = float(dc_min)
        self._dc_max = float(dc_max)
        self._neutral_dc = float(neutral_dc)
        self._reverse_brake_dc = float(reverse_brake_dc)
        self._reverse_brake_hold_s = max(0.0, float(reverse_brake_hold_s))
        self._reverse_neutral_hold_s = max(0.0, float(reverse_neutral_hold_s))
        self._reverse_exit_hold_s = max(0.0, float(reverse_exit_hold_s))
        self._neutral_hold_on_stop_s = max(0.40, self._reverse_exit_hold_s + 0.35)

        self._pwm = HardwarePWM(channel=channel, frequency_hz=frequency_hz, logger=logger)
        self._pwm.start(self._neutral_dc)
        self._in_reverse_mode = False
        self._speed_pct = 0.0
        self._last_duty_cycle_pct = self._neutral_dc

    def set_speed_pct(self, speed_pct: float) -> None:
        self._speed_pct = max(-100.0, min(100.0, float(speed_pct)))

        if self._speed_pct < 0.0 and not self._in_reverse_mode:
            self._enter_reverse_mode()
        elif self._speed_pct >= 0.0 and self._in_reverse_mode:
            self._exit_reverse_mode()

        if abs(self._speed_pct) < 1.0:
            duty_cycle = self._neutral_dc
        elif self._speed_pct > 0.0:
            duty_cycle = self._neutral_dc + (
                (self._speed_pct / 100.0) * (self._dc_max - self._neutral_dc)
            )
        else:
            duty_cycle = self._neutral_dc + (
                (self._speed_pct / 100.0) * (self._neutral_dc - self._dc_min)
            )

        self._pwm.set_duty_cycle(duty_cycle)
        self._last_duty_cycle_pct = duty_cycle
        self._logger.debug(
            "Motor command %.1f%% -> duty cycle %.3f%% (reverse=%s)"
            % (self._speed_pct, duty_cycle, self._in_reverse_mode)
        )

    def neutral(self) -> None:
        self.set_speed_pct(0.0)

    def hold_neutral(
        self,
        *,
        hold_s: float | None = None,
        disable_pwm: bool = False,
    ) -> None:
        neutral_hold_s = (
            self._neutral_hold_on_stop_s if hold_s is None else max(0.0, float(hold_s))
        )
        try:
            self._pwm.start(self._neutral_dc)
            time.sleep(max(0.05, neutral_hold_s))
        except Exception:
            pass
        self._in_reverse_mode = False
        self._speed_pct = 0.0
        self._last_duty_cycle_pct = self._neutral_dc
        if disable_pwm:
            self._pwm.disable()

    def stop(self) -> None:
        self.hold_neutral(disable_pwm=False)

    def brake_to_neutral(self) -> None:
        self._pwm.set_duty_cycle(self._reverse_brake_dc)
        self._last_duty_cycle_pct = self._reverse_brake_dc
        time.sleep(self._reverse_brake_hold_s)
        self._pwm.set_duty_cycle(self._neutral_dc)
        self._last_duty_cycle_pct = self._neutral_dc
        time.sleep(self._reverse_neutral_hold_s)
        self._in_reverse_mode = False
        self._speed_pct = 0.0

    def _enter_reverse_mode(self) -> None:
        self._logger.info(
            "Entering reverse mode (brake_dc=%.3f neutral_dc=%.3f)"
            % (self._reverse_brake_dc, self._neutral_dc)
        )
        self._pwm.set_duty_cycle(self._reverse_brake_dc)
        time.sleep(self._reverse_brake_hold_s)
        self._pwm.set_duty_cycle(self._neutral_dc)
        time.sleep(self._reverse_neutral_hold_s)
        self._in_reverse_mode = True

    def _exit_reverse_mode(self) -> None:
        self._pwm.set_duty_cycle(self._neutral_dc)
        self._last_duty_cycle_pct = self._neutral_dc
        time.sleep(self._reverse_exit_hold_s)
        self._in_reverse_mode = False

    def get_state(self) -> dict[str, float | bool]:
        return {
            "speed_pct": self._speed_pct,
            "pwm_dc": self._last_duty_cycle_pct,
            "neutral_dc": self._neutral_dc,
            "reverse_mode": self._in_reverse_mode,
        }

    def get_pwm_state(self) -> dict[str, float | int | str | None]:
        return self._pwm.get_state()
