# fake_pwm.py
# Minimal fake PWM/GPIO helpers for debugging without hardware.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FakePWM:
    # Match code/raspberry_pwm.PWM signature (channel, frequency)
    channel: int
    frequency: float
    duty_cycle: float = 0.0
    running: bool = False
    history: list[tuple[float, str, float]] = field(default_factory=list)

    def __init__(
        self,
        channel: Optional[int] = None,
        frequency: Optional[float] = None,
        *,
        # Backwards-compatible aliases
        pin: Optional[int] = None,
        frequency_hz: Optional[float] = None,
        duty_cycle: float = 0.0,
    ) -> None:
        if channel is None and pin is None:
            raise TypeError("FakePWM requires 'channel' (or legacy 'pin')")
        if frequency is None and frequency_hz is None:
            raise TypeError("FakePWM requires 'frequency' (or legacy 'frequency_hz')")

        self.channel = int(channel if channel is not None else pin)  # type: ignore[arg-type]
        self.frequency = float(frequency if frequency is not None else frequency_hz)  # type: ignore[arg-type]
        self.duty_cycle = float(duty_cycle)
        self.running = False
        self.history = []

    def start(self, duty_cycle: float) -> None:
        self.running = True
        self.set_duty_cycle(duty_cycle)

    # Raspberry-style API
    def set_duty_cycle(self, duty_cycle: float) -> None:
        self.duty_cycle = float(duty_cycle)
        self._log("duty", self.duty_cycle)

    def ChangeDutyCycle(self, duty_cycle: float) -> None:
        # GPIO-style alias
        self.set_duty_cycle(duty_cycle)

    def ChangeFrequency(self, frequency_hz: float) -> None:
        # GPIO-style alias
        self.frequency = float(frequency_hz)
        self._log("freq", self.frequency)

    def stop(self) -> None:
        self.running = False
        self._log("stop", 0.0)

    def _log(self, kind: str, value: float) -> None:
        ts = time.time()
        self.history.append((ts, kind, value))
        print(f"[FakePWM] channel={self.channel} {kind}={value} running={self.running}")

    # Backwards-compatible attribute aliases
    @property
    def pin(self) -> int:
        return self.channel

    @pin.setter
    def pin(self, value: int) -> None:
        self.channel = int(value)

    @property
    def frequency_hz(self) -> float:
        return self.frequency

    @frequency_hz.setter
    def frequency_hz(self, value: float) -> None:
        self.frequency = float(value)


class FakeGPIO:
    # Common GPIO-style constants
    BCM = "BCM"
    BOARD = "BOARD"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0

    def __init__(self) -> None:
        self.mode: Optional[str] = None
        self._pins: Dict[int, int] = {}
        self._pwms: Dict[int, FakePWM] = {}

    def setmode(self, mode: str) -> None:
        self.mode = mode
        print(f"[FakeGPIO] setmode({mode})")

    def setup(self, pin: int, direction: str) -> None:
        self._pins.setdefault(pin, self.LOW)
        print(f"[FakeGPIO] setup(pin={pin}, dir={direction})")

    def output(self, pin: int, value: int) -> None:
        self._pins[pin] = int(value)
        print(f"[FakeGPIO] output(pin={pin}, value={value})")

    def input(self, pin: int) -> int:
        return self._pins.get(pin, self.LOW)

    def PWM(self, pin: int, frequency_hz: float) -> FakePWM:
        pwm = FakePWM(channel=pin, frequency=frequency_hz)
        self._pwms[pin] = pwm
        print(f"[FakeGPIO] PWM(pin={pin}, freq={frequency_hz})")
        return pwm

    def cleanup(self) -> None:
        for pwm in self._pwms.values():
            if pwm.running:
                pwm.stop()
        self._pwms.clear()
        self._pins.clear()
        print("[FakeGPIO] cleanup()")