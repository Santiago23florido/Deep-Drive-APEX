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
        logger,
        min_authority_ratio: float = 0.90,
        *,
        backend: str = "sysfs_pwm",
        node=None,
        publish_topic: str = "",
        asymmetric_limits: bool = False,
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
        # `limit_deg` is the servo's *calibration*, not a limit: it is the angle
        # reached at the edge of the duty window, so this is the hardware scale
        # (5.0..10.0 % over +-18 deg = 0.1389 %/deg). Changing it rescales how
        # far the wheels move per commanded degree.
        self._variation_per_deg = 0.5 * (self._dc_max - self._dc_min) / self._limit_deg
        # Nominal PWM-equivalent angles, not independently measured wheels.
        # With direction_sign=-1, 5..10% about 8.2% gives +23.04 left and
        # -12.96 right. Legacy callers keep the smaller symmetric limit.
        # The directional opt-in is used only with matching bridge/planner
        # limits; duty bounds and the configured centre remain unchanged.
        self._authority_deg = min(self._dc_center - self._dc_min, self._dc_max - self._dc_center) / self._variation_per_deg
        lower = (self._dc_center - self._dc_min) / self._variation_per_deg
        upper = (self._dc_max - self._dc_center) / self._variation_per_deg
        self._authority_left_deg, self._authority_right_deg = (lower, upper) if self._direction_sign < 0 else (upper, lower)
        self._asymmetric_limits = bool(asymmetric_limits)
        self._requested_angle_deg = 0.0
        self._pre_sign_angle_deg = 0.0
        self._post_sign_angle_deg = 0.0
        self._clamped_angle_deg = 0.0
        self._applied_angle_deg = 0.0
        self._last_duty_cycle_pct = self._dc_center

        self._pwm = HardwarePWM(
            channel=channel,
            frequency_hz=frequency_hz,
            logger=logger,
            backend=backend,
            node=node,
            publish_topic=publish_topic,
        )
        self._pwm.start(self._dc_center)
        self._current_angle_deg = 0.0

    def set_angle_deg(self, angle_deg: float) -> None:
        self._requested_angle_deg = float(angle_deg)
        # Clamp what was *asked for*, not only the duty it computes: see
        # `_authority_deg`.
        left = self._authority_left_deg if self._asymmetric_limits else self._authority_deg
        right = self._authority_right_deg if self._asymmetric_limits else self._authority_deg
        requested_angle_deg = max(-right, min(left, self._requested_angle_deg))
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

    @property
    def authority_deg(self) -> float:
        """Angle reachable in *both* directions; a pre-flight asserts this."""
        return self._authority_deg

    def get_state(self) -> dict[str, float]:
        return {
            "authority_deg": self._authority_deg,
            "authority_left_deg": self._authority_left_deg,
            "authority_right_deg": self._authority_right_deg,
            "asymmetric_limits": self._asymmetric_limits,
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
        *,
        backend: str = "sysfs_pwm",
        node=None,
        publish_topic: str = "",
        async_reverse: bool = False,
        clock=None,
    ) -> None:
        self._logger = logger
        # Non-blocking direction changes, driven by `tick()`. Opt-in so the
        # legacy stack keeps the behaviour it was validated with.
        self._async_reverse = bool(async_reverse)
        self._clock = clock or time.monotonic
        self._phase = "fwd"  # fwd | brake | neutral | rev | exit
        self._phase_until: float | None = None
        self._requested_pct = 0.0
        self._dc_min = float(dc_min)
        self._dc_max = float(dc_max)
        self._neutral_dc = float(neutral_dc)
        self._reverse_brake_dc = float(reverse_brake_dc)
        self._reverse_brake_hold_s = max(0.0, float(reverse_brake_hold_s))
        self._reverse_neutral_hold_s = max(0.0, float(reverse_neutral_hold_s))
        self._reverse_exit_hold_s = max(0.0, float(reverse_exit_hold_s))
        self._neutral_hold_on_stop_s = max(0.40, self._reverse_exit_hold_s + 0.35)

        self._pwm = HardwarePWM(
            channel=channel,
            frequency_hz=frequency_hz,
            logger=logger,
            backend=backend,
            node=node,
            publish_topic=publish_topic,
        )
        self._pwm.start(self._neutral_dc)
        self._in_reverse_mode = False
        self._speed_pct = 0.0
        self._last_duty_cycle_pct = self._neutral_dc

    def _duty_for(self, speed_pct: float) -> float:
        if abs(speed_pct) < 1.0:
            return self._neutral_dc
        if speed_pct > 0.0:
            return self._neutral_dc + ((speed_pct / 100.0) * (self._dc_max - self._neutral_dc))
        return self._neutral_dc + ((speed_pct / 100.0) * (self._neutral_dc - self._dc_min))

    def _write(self, duty_cycle: float) -> None:
        self._pwm.set_duty_cycle(duty_cycle)
        self._last_duty_cycle_pct = duty_cycle

    def set_speed_pct(self, speed_pct: float) -> None:
        pct = max(-100.0, min(100.0, float(speed_pct)))
        if self._async_reverse:
            self._requested_pct = pct
            self._advance()
            return

        self._speed_pct = pct
        if self._speed_pct < 0.0 and not self._in_reverse_mode:
            self._enter_reverse_mode()
        elif self._speed_pct >= 0.0 and self._in_reverse_mode:
            self._exit_reverse_mode()

        duty_cycle = self._duty_for(self._speed_pct)
        self._write(duty_cycle)
        self._logger.debug(
            "Motor command %.1f%% -> duty cycle %.3f%% (reverse=%s)"
            % (self._speed_pct, duty_cycle, self._in_reverse_mode)
        )

    def tick(self) -> None:
        """Advance the non-blocking reverse sequence; a no-op in blocking mode.

        The caller must invoke this every control cycle when ``async_reverse``
        is set, otherwise a reverse request stalls in its brake phase.
        """
        if self._async_reverse:
            self._advance()

    def _advance(self) -> None:
        """One step of the ESC's direction-change sequence, without sleeping.

        A Maverick ESC has to be walked through brake and neutral before it
        accepts reverse. The blocking version of this does
        ``sleep(0.12) + sleep(0.12)`` on entry and ``sleep(0.15)`` on exit --
        inside the control callback, which freezes the steering for a quarter of
        a second every time the driver's recovery reverses. Here the same
        sequence is a phase machine driven by the control timer.
        """
        now = self._clock()
        if self._phase_until is not None:
            if now < self._phase_until:
                return  # a transition is in progress; the duty is already set
            self._phase_until = None
            if self._phase == "brake":
                self._write(self._neutral_dc)
                self._phase = "neutral"
                self._phase_until = now + self._reverse_neutral_hold_s
                return
            if self._phase == "neutral":
                self._phase, self._in_reverse_mode = "rev", True
                self._logger.info("Reverse mode engaged")
            elif self._phase == "exit":
                self._phase, self._in_reverse_mode = "fwd", False

        req = self._requested_pct
        if req < 0.0 and self._phase == "fwd":
            self._write(self._reverse_brake_dc)
            self._phase, self._phase_until, self._speed_pct = "brake", now + self._reverse_brake_hold_s, 0.0
            self._logger.info(
                "Entering reverse mode (brake_dc=%.3f neutral_dc=%.3f)"
                % (self._reverse_brake_dc, self._neutral_dc)
            )
            return
        if req >= 0.0 and self._phase == "rev":
            self._write(self._neutral_dc)
            self._phase, self._phase_until, self._speed_pct = "exit", now + self._reverse_exit_hold_s, 0.0
            return

        self._speed_pct = req
        self._write(self._duty_for(req))

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
        self._phase, self._phase_until, self._requested_pct = "fwd", None, 0.0
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
            "requested_pct": self._requested_pct,
            "reverse_phase": self._phase,
            "async_reverse": self._async_reverse,
            "pwm_dc": self._last_duty_cycle_pct,
            "neutral_dc": self._neutral_dc,
            "reverse_mode": self._in_reverse_mode,
        }

    def get_pwm_state(self) -> dict[str, float | int | str | None]:
        return self._pwm.get_state()
