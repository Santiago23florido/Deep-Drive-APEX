#!/usr/bin/env python3
"""Minimal PWM backend for Raspberry-hosted or simulated actuation."""

from __future__ import annotations

import errno
import glob
import os
import time
from typing import Any


class HardwarePWM:
    """Simple PWM wrapper around `/sys/class/pwm`."""

    def __init__(
        self,
        channel: int,
        frequency_hz: float,
        logger,
        *,
        backend: str = "sysfs_pwm",
        node=None,
        publish_topic: str = "",
    ) -> None:
        self._channel = int(channel)
        self._frequency_hz = float(frequency_hz)
        self._logger = logger
        self._backend = str(backend).strip().lower() or "sysfs_pwm"
        self._node = node
        self._publish_topic = str(publish_topic)
        self._enabled = 0
        self._duty_cycle_pct = 0.0
        self._publisher = None

        self._period_ns = int(round(1.0e9 / self._frequency_hz))
        self._chip_path = ""
        self._pwm_dir = ""

        if self._backend == "sim_pwm_topic":
            if self._node is None or not self._publish_topic:
                raise ValueError("sim_pwm_topic backend requires a ROS node and publish_topic")
            from std_msgs.msg import Float64

            self._publisher = self._node.create_publisher(Float64, self._publish_topic, 20)
        else:
            self._chip_path = self._find_chip_path()
            self._pwm_dir = os.path.join(self._chip_path, f"pwm{self._channel}")
            self._ensure_channel()
            self._configure_period()

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

    def _try_write_int(self, path: str, value: int) -> bool:
        try:
            self._write_int(path, value)
            return True
        except Exception:
            return False

    def _configure_period(self) -> None:
        if self._backend == "sim_pwm_topic":
            return

        enable_path = os.path.join(self._pwm_dir, "enable")
        duty_cycle_path = os.path.join(self._pwm_dir, "duty_cycle")
        period_path = os.path.join(self._pwm_dir, "period")

        self._try_write_int(enable_path, 0)
        self._try_write_int(duty_cycle_path, 0)
        try:
            self._write_int(period_path, self._period_ns)
        except OSError as exc:
            if exc.errno not in {errno.EBUSY, errno.EINVAL}:
                raise
            self._logger.warning(
                "PWM channel %s was busy while setting period; disabling and retrying"
                % self._pwm_dir
            )
            self._try_write_int(enable_path, 0)
            self._try_write_int(duty_cycle_path, 0)
            time.sleep(0.05)
            self._write_int(period_path, self._period_ns)

    @staticmethod
    def _read_int(path: str) -> int | None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return int(handle.read().strip())
        except Exception:
            return None

    def set_duty_cycle(self, duty_cycle_pct: float) -> None:
        duty_cycle_pct = max(0.0, min(100.0, float(duty_cycle_pct)))
        self._duty_cycle_pct = duty_cycle_pct
        if self._backend == "sim_pwm_topic":
            if self._publisher is None:
                return
            from std_msgs.msg import Float64

            msg = Float64()
            msg.data = duty_cycle_pct
            self._publisher.publish(msg)
            return

        active_ns = int(round(self._period_ns * duty_cycle_pct / 100.0))
        self._write_int(os.path.join(self._pwm_dir, "duty_cycle"), active_ns)

    def start(self, duty_cycle_pct: float) -> None:
        self.set_duty_cycle(duty_cycle_pct)
        self._enabled = 1
        if self._backend != "sim_pwm_topic":
            self._write_int(os.path.join(self._pwm_dir, "enable"), 1)
        self._logger.debug(
            "PWM started on %s at %.3f%%"
            % (self._pwm_dir or self._publish_topic, float(duty_cycle_pct))
        )

    def disable(self) -> None:
        self._enabled = 0
        if self._backend == "sim_pwm_topic":
            return
        try:
            self._write_int(os.path.join(self._pwm_dir, "enable"), 0)
        except Exception:
            pass

    def stop(self) -> None:
        self.disable()

    def get_state(self) -> dict[str, Any]:
        duty_cycle_ns = None
        enabled = self._enabled
        if self._backend != "sim_pwm_topic":
            duty_cycle_ns = self._read_int(os.path.join(self._pwm_dir, "duty_cycle"))
            enabled = self._read_int(os.path.join(self._pwm_dir, "enable")) or 0
        return {
            "pwm_dir": self._pwm_dir,
            "publish_topic": self._publish_topic,
            "backend": self._backend,
            "period_ns": (
                self._read_int(os.path.join(self._pwm_dir, "period"))
                if self._backend != "sim_pwm_topic"
                else self._period_ns
            ),
            "duty_cycle_ns": duty_cycle_ns,
            "enabled": enabled,
            "duty_cycle_pct": self._duty_cycle_pct,
        }
