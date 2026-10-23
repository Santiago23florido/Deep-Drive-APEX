#!/usr/bin/env python3
"""RPLIDAR A2 Express scan in pure Python: full revolutions, stamped at their first sample.

Target (``real_vehicle/docs/sensor_timing_requirements.md``): the A2M8 in the
Express scan (4000 samples/s, every A2M8 from firmware 1.17), turning at
10 Hz, one ``LaserScan`` per complete revolution, stamped at the first
sample of the revolution. The learned odometry was trained for exactly this
(simulation profile ``APEX_real``).

Why not ``rplidar_ros``: on the A series it always spins the motor at the
device default PWM (the node has no PWM parameter; this unit turned at
13 Hz with PWM 660), so it cannot hold 10 Hz, and it stamps a scan with the
host time taken before a blocking read. This module talks to the LiDAR
directly, regulates the motor PWM to the target rate and stamps every
sample from the LiDAR's own sample clock.

Protocol code ported from the Slamtec SDK shipped with rplidar_ros 2.1.4
(BSD-2-Clause), so the decoding is the SDK's, bit for bit:
* command framing and constants: ``sl_lidar_protocol.h``, ``sl_lidar_cmd.h``;
* capsule parser: ``UnpackerHandler_CapsuleNode::onData``;
* angle interpolation: ``UnpackerHandler_CapsuleNode::_onScanNodeCapsuleData``;
* sample timing: ``_getSampleDelayOffsetInExpressMode``
  (``sdk/src/dataunpacker/unpacker/handler_capsules.cpp``).

Timing. The Express scan samples at a fixed period ``Ts`` set by the LiDAR's
own oscillator (250 us at 4 kHz), so the index of a sample is a device time.
Capsule ``c`` holds samples ``32c .. 32c+31`` and its last byte reaches the
host ``1.5 Ts + 840 / baud`` after its last sample (SDK timing model) plus a
USB latency. ``DeviceClockMapper`` removes that latency jitter and the clock
drift, so a sample's stamp is ``mapper.map(index * Ts)``.
"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import glob
import math
import os
import statistics
import time
from typing import Callable, Optional

import numpy as np

from ..timing import DeviceClockMapper

# --- protocol constants (sl_lidar_protocol.h, sl_lidar_cmd.h) ---------------
SYNC_BYTE = 0xA5
ANS_SYNC_BYTE2 = 0x5A
CMD_STOP = 0x25
CMD_RESET = 0x40
CMD_GET_INFO = 0x50
CMD_GET_HEALTH = 0x52
CMD_GET_SAMPLERATE = 0x59  # fw >= 1.17
CMD_EXPRESS_SCAN = 0x82  # fw >= 1.17
CMD_SET_MOTOR_PWM = 0xF0
CMD_GET_ACC_BOARD_FLAG = 0xFF
ANS_TYPE_DEVINFO = 0x04
ANS_TYPE_DEVHEALTH = 0x06
ANS_TYPE_SAMPLE_RATE = 0x15
ANS_TYPE_ACC_BOARD_FLAG = 0xFF
ANS_TYPE_CAPSULED = 0x82
ACC_BOARD_FLAG_MOTOR_CTRL_SUPPORT = 0x1
EXP_SYNC_1 = 0xA
EXP_SYNC_2 = 0x5
EXP_SYNCBIT = 0x1 << 15
CAPSULE_BYTES = 84
SAMPLES_PER_CAPSULE = 32
MAX_MOTOR_PWM = 1023
HEALTH_STATUS = {0: "OK", 1: "WARNING", 2: "ERROR"}
# Boost / Sensitivity (8 kHz) need this firmware; the car does not assume it.
SENSITIVITY_MIN_FIRMWARE = (1, 24)


def command_packet(cmd: int, payload: bytes = b"") -> bytes:
    """Request bytes: A5 cmd [size payload checksum]."""
    if not payload:
        return bytes((SYNC_BYTE, cmd))
    body = bytes((SYNC_BYTE, cmd, len(payload))) + bytes(payload)
    checksum = 0
    for value in body:
        checksum ^= value
    return body + bytes((checksum,))


def parse_descriptor(data: bytes) -> tuple[int, int, int]:
    """(payload size, send mode, answer type) of a 7-byte response descriptor."""
    if len(data) != 7 or data[0] != SYNC_BYTE or data[1] != ANS_SYNC_BYTE2:
        raise ValueError(f"not a response descriptor: {bytes(data).hex()}")
    word = int.from_bytes(data[2:6], "little")
    return word & 0x3FFFFFFF, word >> 30, data[6]


@dataclass(frozen=True)
class DeviceInfo:
    model: int
    firmware_major: int
    firmware_minor: int
    hardware: int
    serial: str

    @classmethod
    def parse(cls, payload: bytes) -> "DeviceInfo":
        if len(payload) < 20:
            raise ValueError("device info answer shorter than 20 bytes")
        firmware = int.from_bytes(payload[1:3], "little")
        return cls(
            model=payload[0],
            firmware_major=firmware >> 8,
            firmware_minor=firmware & 0xFF,
            hardware=payload[3],
            serial=bytes(payload[4:20]).hex().upper(),
        )

    @property
    def firmware(self) -> str:
        return f"{self.firmware_major}.{self.firmware_minor:02d}"

    @property
    def supports_sensitivity(self) -> bool:
        """Boost / Sensitivity scan (8 kHz) available: firmware >= 1.24."""
        return (self.firmware_major, self.firmware_minor) >= SENSITIVITY_MIN_FIRMWARE


def parse_health(payload: bytes) -> tuple[int, int]:
    """(status 0 OK / 1 warning / 2 error, error code)."""
    return payload[0], int.from_bytes(payload[1:3], "little")


def parse_sample_rate(payload: bytes) -> tuple[int, int]:
    """(standard, express) sample duration in microseconds."""
    return int.from_bytes(payload[0:2], "little"), int.from_bytes(payload[2:4], "little")


# --- capsules -----------------------------------------------------------------
@dataclass(frozen=True)
class Capsule:
    """One Express answer packet: 32 samples (16 cabins of 2)."""

    start_angle_q6: int  # angle of the first sample, 1/64 deg (15 bits)
    new_scan: bool  # S flag: the first capsule after the scan (re)started
    cabins: tuple[tuple[int, int, int], ...]  # (distance_angle_1, distance_angle_2, offset_angles_q3)
    host_ns: int  # host time when its last byte was read

    @classmethod
    def from_bytes(cls, data: bytes, host_ns: int) -> Optional["Capsule"]:
        """The capsule, or None if the sync nibbles or the checksum are wrong."""
        if len(data) != CAPSULE_BYTES or (data[0] >> 4) != EXP_SYNC_1 or (data[1] >> 4) != EXP_SYNC_2:
            return None
        received = (data[0] & 0xF) | ((data[1] & 0xF) << 4)
        checksum = 0
        for value in data[2:]:
            checksum ^= value
        if received != checksum:
            return None
        word = int.from_bytes(data[2:4], "little")
        cabins = tuple(
            (
                int.from_bytes(data[4 + 5 * i : 6 + 5 * i], "little"),
                int.from_bytes(data[6 + 5 * i : 8 + 5 * i], "little"),
                data[8 + 5 * i],
            )
            for i in range(16)
        )
        return cls(start_angle_q6=word & 0x7FFF, new_scan=bool(word & EXP_SYNCBIT), cabins=cabins, host_ns=int(host_ns))


GAP = "gap"  # stream event: the SDK would drop the pending capsule here


class CapsuleStream:
    """Byte stream -> capsules and gaps, as ``UnpackerHandler_CapsuleNode::onData``.

    A byte that is not a sync nibble, or a capsule with a bad checksum, makes
    the SDK discard the pending capsule; this emits ``GAP`` there.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._in_gap = False
        self.checksum_errors = 0
        self.skipped_bytes = 0

    def reset(self) -> None:
        self._buf.clear()
        self._in_gap = False

    def feed(self, data: bytes, host_ns: int) -> list:
        events: list = []
        for byte in data:
            pos = len(self._buf)
            if pos == 0 and (byte >> 4) != EXP_SYNC_1:
                self._skip(events)
                continue
            if pos == 1 and (byte >> 4) != EXP_SYNC_2:
                self._buf.clear()
                self._skip(events)
                continue
            self._buf.append(byte)
            if len(self._buf) == CAPSULE_BYTES:
                capsule = Capsule.from_bytes(bytes(self._buf), host_ns)
                self._buf.clear()
                if capsule is None:
                    self.checksum_errors += 1
                    events.append(GAP)
                else:
                    self._in_gap = False
                    events.append(capsule)
        return events

    def _skip(self, events: list) -> None:
        self.skipped_bytes += 1
        if not self._in_gap:
            self._in_gap = True
            events.append(GAP)


def decode_capsule_pair(prev: Capsule, cur: Capsule) -> tuple[list[int], list[int], list[bool]]:
    """Samples of ``prev``: (angle 1/64 deg, distance 1/4 mm, sync), SDK arithmetic.

    The angles of ``prev``'s 32 samples are interpolated between its start
    angle and ``cur``'s. ``sync`` marks the sample that starts a revolution.
    Distance 0 means no return.
    """
    cur_q8 = (cur.start_angle_q6 & 0x7FFF) << 2
    prev_q8 = (prev.start_angle_q6 & 0x7FFF) << 2
    diff_q8 = cur_q8 - prev_q8
    if prev_q8 > cur_q8:
        diff_q8 += 360 << 8
    inc_q16 = diff_q8 << 3
    raw_q16 = prev_q8 << 8
    full_q16 = 360 << 16
    full_q6 = 360 << 6
    angles: list[int] = []
    dists: list[int] = []
    syncs: list[bool] = []
    for distance_angle_1, distance_angle_2, offset_q3 in prev.cabins:
        offset_1 = (offset_q3 & 0xF) | ((distance_angle_1 & 0x3) << 4)
        offset_2 = (offset_q3 >> 4) | ((distance_angle_2 & 0x3) << 4)
        for distance_angle, offset in ((distance_angle_1, offset_1), (distance_angle_2, offset_2)):
            angle_q6 = (raw_q16 - (offset << 13)) >> 10
            syncs.append(((raw_q16 + inc_q16) % full_q16) < inc_q16)
            raw_q16 += inc_q16
            if angle_q6 < 0:
                angle_q6 += full_q6
            if angle_q6 >= full_q6:
                angle_q6 -= full_q6
            angles.append(angle_q6)
            dists.append(distance_angle & 0xFFFC)
    return angles, dists, syncs


@dataclass
class DecodedCapsule:
    index: int  # capsule counter: sample k has device index 32 * index + k
    angles_deg: np.ndarray  # device angle, clockwise from the LiDAR's 0 mark
    dist_mm: np.ndarray  # 0 = no return
    sync: np.ndarray  # bool, first sample of a revolution
    after_gap: bool  # capsules were lost just before this one


class ExpressDecoder:
    """Capsules -> decoded samples with their device sample index.

    Consecutive capsules of the stream are adjacent. After a gap, the number
    of lost capsules follows from the advance of the start angle (about 29
    deg per capsule at 10 Hz), and the host arrival time resolves whole
    turns.
    """

    def __init__(self, sample_period_s: float) -> None:
        self._capsule_period_s = SAMPLES_PER_CAPSULE * float(sample_period_s)
        self._deg_per_capsule: float | None = None
        self.lost_capsules = 0
        self.restarts = 0
        self.reset()

    def reset(self) -> None:
        self._prev: Capsule | None = None
        self._prev_index = -1
        self._last: Capsule | None = None  # last good capsule (even across a gap)
        self._last_index = -1
        self._gap = False
        self._gap_since_decoded = False

    def gap(self) -> None:
        self._gap = True
        self._prev = None  # the SDK cannot interpolate across a gap

    def push(self, capsule: Capsule) -> tuple[int, Optional[DecodedCapsule], bool]:
        """(capsule index, samples of the previous capsule or None, restarted).

        ``restarted`` is True when the device restarted its scan (S flag
        after the first capsule): the device clock origin is lost.
        """
        restarted = False
        if capsule.new_scan and self._last is not None:
            self.restarts += 1
            restarted = True
            self.reset()
        if self._last is not None and not self._gap:
            step = ((capsule.start_angle_q6 - self._last.start_angle_q6) % (360 << 6)) / 64.0
            if self._deg_per_capsule is not None and step > 1.5 * self._deg_per_capsule:
                self.gap()  # whole capsules missing from the stream: no byte error to see
            else:
                self._deg_per_capsule = step if self._deg_per_capsule is None else 0.9 * self._deg_per_capsule + 0.1 * step
        if self._last is None:
            index = 0
        elif not self._gap:
            index = self._last_index + 1
        else:
            index = self._last_index + self._capsules_since(capsule)
            self.lost_capsules += max(0, index - self._last_index - 1)
        decoded = None
        if self._prev is not None:
            angles, dists, syncs = decode_capsule_pair(self._prev, capsule)
            decoded = DecodedCapsule(
                index=self._prev_index,
                angles_deg=np.asarray(angles, dtype=float) / 64.0,
                dist_mm=np.asarray(dists, dtype=float) / 4.0,
                sync=np.asarray(syncs, dtype=bool),
                after_gap=self._gap_since_decoded,
            )
            self._gap_since_decoded = False
        if self._gap:
            self._gap_since_decoded = True
        self._prev, self._prev_index = capsule, index
        self._last, self._last_index = capsule, index
        self._gap = False
        return index, decoded, restarted

    def _capsules_since(self, capsule: Capsule) -> int:
        elapsed = (capsule.host_ns - self._last.host_ns) * 1e-9 / self._capsule_period_s
        step = self._deg_per_capsule
        if not step or step <= 0.0:
            return max(1, int(round(elapsed)))
        delta = ((capsule.start_angle_q6 - self._last.start_angle_q6) % (360 << 6)) / 64.0
        turns = max(0, int(elapsed * step / 360.0) + 1)
        candidates = [int(round((delta + 360.0 * m) / step)) for m in range(turns + 1)]
        best = min((c for c in candidates if c >= 1), key=lambda c: abs(c - elapsed), default=1)
        return best


@dataclass
class Revolution:
    first_index: int  # device sample index of the sync sample that starts it
    n_samples: int  # samples up to the next sync: the period is n_samples * Ts
    angles_deg: np.ndarray
    dist_mm: np.ndarray
    complete: bool  # no capsule lost inside it


class RevolutionAssembler:
    """Decoded capsules -> revolutions, split at the SDK's sync samples."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._first: int | None = None
        self._angles: list[np.ndarray] = []
        self._dists: list[np.ndarray] = []
        self._complete = True

    def push(self, capsule: DecodedCapsule) -> list[Revolution]:
        out: list[Revolution] = []
        if capsule.after_gap:
            self._complete = False
        base = SAMPLES_PER_CAPSULE * capsule.index
        start = 0
        for k in np.flatnonzero(capsule.sync):
            if self._first is not None:
                self._angles.append(capsule.angles_deg[start:k])
                self._dists.append(capsule.dist_mm[start:k])
                out.append(
                    Revolution(
                        first_index=self._first,
                        n_samples=base + int(k) - self._first,
                        angles_deg=np.concatenate(self._angles) if self._angles else np.empty(0),
                        dist_mm=np.concatenate(self._dists) if self._dists else np.empty(0),
                        complete=self._complete,
                    )
                )
            self._first = base + int(k)
            self._angles, self._dists, self._complete = [], [], True
            start = int(k)
        if self._first is not None:
            self._angles.append(capsule.angles_deg[start:])
            self._dists.append(capsule.dist_mm[start:])
        return out


@dataclass
class StampedRevolution:
    stamp_ns: int  # first sample of the revolution, host (ROS) clock
    period_s: float  # to the first sample of the next revolution
    revolution: Revolution


@dataclass
class PipelineCounters:
    capsules: int = 0
    published: int = 0
    dropped_incomplete: int = 0
    dropped_irregular: int = 0
    restarts: int = 0


class ExpressScanPipeline:
    """Bytes read from the LiDAR -> complete revolutions stamped on the host clock.

    ``stamp_correction_s`` is subtracted from every stamp: the minimum USB
    latency that the envelope cannot see (to calibrate on the car, 0 by
    default). Revolutions with a lost capsule, or whose sample count is
    off the running median by more than ``regularity_tol``, are dropped
    (the odometry is trained with whole-revolution dropouts, not holes).
    """

    def __init__(
        self,
        sample_period_s: float,
        baudrate: int,
        *,
        stamp_correction_s: float = 0.0,
        drop_incomplete: bool = True,
        regularity_tol: float = 0.15,
        mapper: DeviceClockMapper | None = None,
    ) -> None:
        self.sample_period_s = float(sample_period_s)
        # SDK timing model: the last sample of a capsule is measured
        # filter (1 Ts) + half a sample + transmission of the 84 bytes before
        # the capsule's last byte reaches the host.
        self.capsule_delay_s = 1.5 * self.sample_period_s + CAPSULE_BYTES * 10.0 / float(baudrate)
        self.stamp_correction_ns = int(round(stamp_correction_s * 1e9))
        self.drop_incomplete = bool(drop_incomplete)
        self.regularity_tol = float(regularity_tol)
        # The LiDAR's sample period may be off its nominal value by up to a
        # few percent: allow a wide drift. Use a shorter envelope here than
        # the IMU mapper: a 10 s history retained timing offsets for seconds
        # after Express packet losses in real captures. Keep the sample-clock
        # model and whole-revolution drops; only recover its fit sooner.
        self.mapper = mapper if mapper is not None else DeviceClockMapper(
            window_s=3.0, chunks=6, refit_every_s=0.25, max_drift=0.03,
        )
        self.stream = CapsuleStream()
        self.decoder = ExpressDecoder(self.sample_period_s)
        self.assembler = RevolutionAssembler()
        self.counters = PipelineCounters()
        self._recent_samples: deque[int] = deque(maxlen=21)
        self._pending: Revolution | None = None

    def reset(self) -> None:
        """Restart after a reconnection: forget capsules, revolutions and the clock."""
        self.stream.reset()
        self.decoder.reset()
        self.assembler.reset()
        self.mapper.reset()
        self._pending = None

    def feed(self, data: bytes, host_ns: int) -> list[StampedRevolution]:
        out: list[StampedRevolution] = []
        for event in self.stream.feed(data, host_ns):
            if event is GAP:
                self.decoder.gap()
                continue
            self.counters.capsules += 1
            index, decoded, restarted = self.decoder.push(event)
            if restarted:
                self.counters.restarts += 1
                self.assembler.reset()
                self.mapper.reset()
                self._pending = None
            last_sample = SAMPLES_PER_CAPSULE * index + SAMPLES_PER_CAPSULE - 1
            self.mapper.add(last_sample * self.sample_period_s + self.capsule_delay_s, event.host_ns)
            if decoded is None:
                continue
            for rev in self.assembler.push(decoded):
                stamped = self._finish(rev)
                if stamped is not None:
                    out.append(stamped)
        return out

    def stamp_of(self, sample_index: int) -> int:
        return self.mapper.map(sample_index * self.sample_period_s) - self.stamp_correction_ns

    def _finish(self, rev: Revolution) -> Optional[StampedRevolution]:
        if not rev.complete:
            self.counters.dropped_incomplete += 1
            return None
        if len(self._recent_samples) >= 5:
            median = statistics.median(self._recent_samples)
            if abs(rev.n_samples - median) > self.regularity_tol * median:
                self.counters.dropped_irregular += 1
                self._recent_samples.append(rev.n_samples)
                return None
        self._recent_samples.append(rev.n_samples)
        stamp = self.stamp_of(rev.first_index)
        period = (self.stamp_of(rev.first_index + rev.n_samples) - stamp) * 1e-9
        self.counters.published += 1
        return StampedRevolution(stamp_ns=stamp, period_s=period, revolution=rev)


# --- motor speed -------------------------------------------------------------
class MotorRegulator:
    """Integral control of the motor PWM towards the target rotation rate.

    Every ``every_revs`` published revolutions, the mean rate is compared
    with the target; outside ``deadband_hz`` the PWM moves by
    ``gain * error / rate * pwm`` (at most ``max_step``). The motor settles
    well within the averaging window, so this does not oscillate.
    """

    def __init__(
        self,
        target_hz: float = 10.0,
        pwm: int = 520,
        *,
        enabled: bool = True,
        every_revs: int = 20,
        gain: float = 0.7,
        deadband_hz: float = 0.05,
        max_step: int = 40,
        pwm_min: int = 200,
        pwm_max: int = MAX_MOTOR_PWM,
    ) -> None:
        self.target_hz = float(target_hz)
        self.pwm = int(pwm)
        self.enabled = bool(enabled)
        self.every_revs = max(1, int(every_revs))
        self.gain = float(gain)
        self.deadband_hz = float(deadband_hz)
        self.max_step = int(max_step)
        self.pwm_min = int(pwm_min)
        self.pwm_max = int(pwm_max)
        self._rates: list[float] = []
        self.last_mean_hz = float("nan")

    def update(self, period_s: float) -> Optional[int]:
        """Feed one revolution period; returns the new PWM when it changes."""
        if period_s <= 0.0 or not math.isfinite(period_s):
            return None
        self._rates.append(1.0 / period_s)
        if len(self._rates) < self.every_revs:
            return None
        mean = sum(self._rates) / len(self._rates)
        self._rates.clear()
        self.last_mean_hz = mean
        error = self.target_hz - mean
        if not self.enabled or abs(error) <= self.deadband_hz:
            return None
        step = self.gain * error / mean * self.pwm
        step = max(-self.max_step, min(self.max_step, int(round(step))))
        new = max(self.pwm_min, min(self.pwm_max, self.pwm + step))
        if new == self.pwm:
            return None
        self.pwm = new
        return new


def motor_ramp(start_pwm: int, target_pwm: int, ramp_s: float, step_s: float = 0.1) -> list[int]:
    """PWM values of a linear soft start, one every ``step_s``, ending at ``target_pwm``.

    ``start_pwm`` itself is not included (it is applied before the ramp).
    """
    start, target = int(start_pwm), int(target_pwm)
    if ramp_s <= 0.0 or step_s <= 0.0 or start >= target:
        return [target]
    steps = max(1, int(math.ceil(ramp_s / step_s)))
    return [int(round(start + (target - start) * (k + 1) / steps)) for k in range(steps)]


def find_undervoltage_alarm(pattern: str = "/sys/class/hwmon/hwmon*") -> Optional[str]:
    """Find the rpi_volt alarm; it can reflect latched rather than current faults."""
    for hwmon in sorted(glob.glob(pattern)):
        try:
            with open(os.path.join(hwmon, "name"), encoding="ascii") as fh:
                if fh.read().strip() != "rpi_volt":
                    continue
        except OSError:
            continue
        alarm = os.path.join(hwmon, "in0_lcrit_alarm")
        if os.path.exists(alarm):
            return alarm
    return None


class SupplyGuard:
    """Keeps the motor start from pulling the Pi's 5 V rail down further.

    The car's Pi 5 supply sits at the under-voltage limit and the Pi has
    reset while the LiDAR motor started. Before the start, ``wait_clear``
    waits for a moment without under-voltage; during the ramp, ``ramp_wait``
    holds the PWM where it is while the flag is up. A fresh host power log
    supplies the current get_throttled bit, since hwmon can retain an alarm.
    The ramp has one shared hold budget, not a fresh budget for every step.
    The separate pre-arm power check still decides whether driving is allowed.
    """

    def __init__(
        self,
        alarm_path: Optional[str],
        *,
        clear_s: float = 1.0,
        wait_max_s: float = 20.0,
        hold_max_s: float = 10.0,
        poll_s: float = 0.05,
        status_path: Optional[str] = None,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.alarm_path = alarm_path
        self.clear_s = clear_s
        self.wait_max_s = wait_max_s
        self.hold_max_s = hold_max_s
        self.poll_s = poll_s
        self.status_path = status_path
        self._wall_clock = wall_clock
        self.hold_spent_s = 0.0
        self._sleep = sleep
        self._clock = clock
        self.alarms = 0  # polls that saw the flag up

    def undervoltage(self) -> Optional[bool]:
        if self.status_path:
            try:
                with open(self.status_path, encoding="ascii") as fh:
                    rows = list(csv.DictReader(fh))
                last = rows[-1]
                stamp = datetime.fromisoformat(last["utc_iso"].replace("Z", "+00:00")).timestamp()
                age = self._wall_clock() - stamp
                if 0.0 <= age <= 3.0:
                    low = bool(int(last["throttled"], 16) & 0x1)
                    self.alarms += int(low)
                    return low
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass  # missing, incomplete or stale host status: use the alarm
        if not self.alarm_path:
            return None
        try:
            with open(self.alarm_path, encoding="ascii") as fh:
                low = fh.read().strip() not in ("", "0")
        except OSError:
            return None
        self.alarms += int(low)
        return low

    def wait_clear(self) -> bool:
        """True once the flag stayed down for ``clear_s``; False if ``wait_max_s`` ran out first."""
        if self.undervoltage() is None:
            return True
        start = clear_since = self._clock()
        while True:
            now = self._clock()
            if self.undervoltage():
                clear_since = now
            elif now - clear_since >= self.clear_s:
                return True
            if now - start >= self.wait_max_s:
                return False
            self._sleep(self.poll_s)

    def ramp_wait(self, step_s: float) -> None:
        """Wait one step, sharing at most ``hold_max_s`` extra across the ramp."""
        self._sleep(step_s)
        start = self._clock()
        remaining = max(0.0, self.hold_max_s - self.hold_spent_s)
        while self.undervoltage() and self._clock() - start < remaining:
            self._sleep(max(0.0, min(self.poll_s, remaining - (self._clock() - start))))
        self.hold_spent_s += max(0.0, self._clock() - start)


# --- scan layout ---------------------------------------------------------------
def laser_angle_deg(device_deg: np.ndarray, heading_offset_deg: float) -> np.ndarray:
    """Laser-frame angle (x forward, counter-clockwise) of a device angle.

    The device measures clockwise from its 0 mark. ``heading_offset_deg`` is
    the one of the existing APEX nodes (91 on the car, apex_params.yaml):
    APEX angle = device + offset, laser angle = 180 - APEX angle, so the
    revolution starts at 180 - 91 = +89 deg and turns clockwise, as the
    simulation's ``scan_start_angle_deg: 89`` / ``scan_direction: cw``.
    """
    return 180.0 - (np.asarray(device_deg, dtype=float) + float(heading_offset_deg))


def bin_revolution(
    device_deg: np.ndarray,
    dist_mm: np.ndarray,
    *,
    heading_offset_deg: float,
    beams: int = 360,
    range_min_m: float = 0.15,
    range_max_m: float = 12.0,
) -> np.ndarray:
    """One revolution in the simulation's scan layout (``lidar_rolling._bin``).

    ``beams`` bins from angle_min = -180 deg counter-clockwise, bin k centred
    on -180 + k * 360 / beams; each bin keeps its nearest return (the
    ``binning: min_range`` of the APEX_real profile); REP-117: +inf without a
    return, -inf when every return was below ``range_min_m``.
    """
    theta = np.radians(laser_angle_deg(device_deg, heading_offset_deg))
    r = np.asarray(dist_mm, dtype=float) / 1000.0
    inc = 2.0 * math.pi / beams
    idx = np.round(np.mod(theta + math.pi, 2.0 * math.pi) / inc).astype(int) % beams
    returned = r > 0.0
    usable = returned & (r >= range_min_m) & (r <= range_max_m)
    out = np.full(beams, np.inf)
    np.minimum.at(out, idx[usable], r[usable])
    low = np.zeros(beams, dtype=bool)
    low[idx[returned & (r < range_min_m)]] = True
    out[~np.isfinite(out) & low] = -np.inf
    return out


def legacy_measurements(device_deg: np.ndarray, dist_mm: np.ndarray) -> list[tuple[float, float, float]]:
    """(quality, device angle deg, distance mm) of the returns, as the rplidar python library yields them."""
    keep = np.asarray(dist_mm) > 0.0
    return [(47.0, float(a), float(d)) for a, d in zip(np.asarray(device_deg)[keep], np.asarray(dist_mm)[keep])]


# --- serial driver -------------------------------------------------------------
class RPLidarExpressDriver:
    """Minimal A-series driver: info, health, sample rate, motor PWM, Express scan."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        timeout_s: float = 1.0,
        now_ns: Callable[[], int] = time.time_ns,
        serial_factory: Callable[..., object] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self._now_ns = now_ns
        self._serial_factory = serial_factory
        self._ser = None

    def open(self) -> None:
        if self._serial_factory is None:
            import serial  # noqa: PLC0415 - pyserial, only needed on the car

            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        else:
            self._ser = self._serial_factory(self.port, self.baudrate, timeout=self.timeout_s)
        self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser is None:
            return
        try:
            self.stop()
            self.stop_motor()
        except Exception:
            pass
        try:
            self._ser.close()
        finally:
            self._ser = None

    def _send(self, cmd: int, payload: bytes = b"") -> None:
        self._ser.write(command_packet(cmd, payload))
        self._ser.flush()

    def _read_exact(self, size: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < size:
            if time.monotonic() > deadline:
                raise TimeoutError(f"RPLIDAR: {len(data)} of {size} bytes before the timeout")
            data += self._ser.read(size - len(data))
        return bytes(data)

    def _read_descriptor(self, deadline: float) -> tuple[int, int, int]:
        # Skip whatever is still in flight until the A5 5A of the answer.
        window = bytearray()
        while True:
            window += self._read_exact(1, deadline)
            if len(window) >= 2 and window[-2] == SYNC_BYTE and window[-1] == ANS_SYNC_BYTE2:
                return parse_descriptor(bytes(window[-2:]) + self._read_exact(5, deadline))
            del window[:-1]

    def _request(self, cmd: int, answer_type: int, payload: bytes = b"") -> bytes:
        self._ser.reset_input_buffer()
        self._send(cmd, payload)
        deadline = time.monotonic() + self.timeout_s
        size, _, kind = self._read_descriptor(deadline)
        if kind != answer_type:
            raise ValueError(f"RPLIDAR: answer type 0x{kind:02X} to command 0x{cmd:02X}, expected 0x{answer_type:02X}")
        return self._read_exact(size, deadline)

    def get_info(self) -> DeviceInfo:
        return DeviceInfo.parse(self._request(CMD_GET_INFO, ANS_TYPE_DEVINFO))

    def get_health(self) -> tuple[int, int]:
        return parse_health(self._request(CMD_GET_HEALTH, ANS_TYPE_DEVHEALTH))

    def get_sample_rate(self) -> tuple[int, int]:
        return parse_sample_rate(self._request(CMD_GET_SAMPLERATE, ANS_TYPE_SAMPLE_RATE))

    def supports_motor_pwm(self) -> Optional[bool]:
        """Accessory-board flag of the A2/A3 adapter (SDK checkMotorCtrlSupport); None without an answer.

        The car's A2M8 (firmware 1.25) never answers this query, yet its motor
        follows SET_MOTOR_PWM (PWM 520 -> 10.1 Hz, 660 -> 13 Hz).
        """
        try:
            payload = self._request(CMD_GET_ACC_BOARD_FLAG, ANS_TYPE_ACC_BOARD_FLAG, (0).to_bytes(4, "little"))
        except (TimeoutError, ValueError):
            return None
        return bool(int.from_bytes(payload[0:4], "little") & ACC_BOARD_FLAG_MOTOR_CTRL_SUPPORT)

    def set_pwm(self, pwm: int) -> None:
        self._send(CMD_SET_MOTOR_PWM, int(max(0, min(MAX_MOTOR_PWM, pwm))).to_bytes(2, "little"))
        time.sleep(0.01)

    def _set_dtr(self, value: bool) -> None:
        try:
            self._ser.dtr = value
        except OSError:
            pass  # not a real serial port (tests, pseudo-terminals)

    def start_motor(
        self,
        pwm: int,
        *,
        start_pwm: int | None = None,
        ramp_s: float = 0.0,
        step_s: float = 0.1,
        wait: Callable[[float], object] = time.sleep,
    ) -> None:
        # DTR low enables the motor on the adapter; the PWM command sets its speed
        # (as the rplidar python library and the SDK do). With a ramp the motor
        # starts at ``start_pwm`` and climbs to ``pwm`` over ``ramp_s``: a step to
        # the working PWM draws the motor's start-up current from the Pi's USB
        # port at once.
        ramp = motor_ramp(start_pwm if start_pwm is not None else pwm, pwm, ramp_s, step_s)
        first = int(pwm) if len(ramp) == 1 else int(start_pwm)
        self.set_pwm(first)
        self._set_dtr(False)
        self.set_pwm(first)
        if len(ramp) == 1:
            return
        for value in ramp:
            wait(step_s)
            self.set_pwm(value)

    def stop_motor(self) -> None:
        self.set_pwm(0)
        self._set_dtr(True)

    def stop(self) -> None:
        self._send(CMD_STOP)
        time.sleep(0.1)
        self._ser.reset_input_buffer()

    def start_express(self) -> None:
        """Legacy Express scan (working mode 0): 84-byte capsules at 4 kHz on the A2M8."""
        self.stop()
        self._send(CMD_EXPRESS_SCAN, bytes(5))  # working_mode 0, working_flags 0, param 0
        size, _, kind = self._read_descriptor(time.monotonic() + self.timeout_s)
        if kind != ANS_TYPE_CAPSULED or size != CAPSULE_BYTES:
            raise ValueError(f"RPLIDAR: Express scan answered type 0x{kind:02X} size {size}")

    def read_chunk(self) -> tuple[bytes, int]:
        """Whatever arrived (waits for one byte at most ``timeout_s``) and the host time now."""
        waiting = getattr(self._ser, "in_waiting", 0)
        data = self._ser.read(waiting if waiting > 0 else 1)
        return bytes(data), self._now_ns()
