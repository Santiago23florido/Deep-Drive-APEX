"""Online identification of the car's own plant, from its odometry alone.

The car is handed a nominal description of itself (``vehicle.yaml``), but on the
real car none of those numbers is measured: the distance between the axles, the
position of the front axle, the transfer function of the steering and of the
motor, the charge of the battery and the mounting of the LiDAR are all unknown.
The only feedback is the trained odometry: one pose increment per LiDAR
revolution (10 Hz) with a standard deviation per estimate.

Nothing in that list is separately observable, and nothing in it needs to be.
Three aggregates are, and they are exactly what the control needs.

**1. The lever arm.** The estimator reports the motion of the point its nominal
LiDAR mount implies. Writing the increment of that point for a rear-axle arc of
any curvature and eliminating the unknown arc length between its two
components gives, with ``h = dyaw / 2``,

    dy - dx * tan(h)  =  2 * tan(h) * arm

exactly, for any turn angle: the curvature cancels identically
(``(1 - cos 2h) - tan(h) sin 2h == 0``). One linear regression on quantities
whose sigma the estimator publishes, with no velocity and no ``dt`` in it,
yields the distance from the odometry reference point back to the rear axle.
That single scalar answers at once "how far is the front axle", "where is the
LiDAR relative to the centre" and "what is the reference point of the
tracking", and it replaces the hard-coded ``rear_offset_m``.

**2. The curvature map.** With the arm known, the curvature the car actually
drove follows from the same increment,

    kappa = sin(dyaw) / (dx + arm * (1 - cos dyaw))

and is regressed against the curvature that was commanded, delayed by the
steering lag, as

    kappa = gain_left * max(u, 0) + gain_right * min(u, 0) + trim + slip * v^2 * u

one gain per side, a trim (a steering that is not centred) and a ``v^2`` term
for the slip of the rear axle. The command channel is already a curvature
(``angular.z / linear.x``), so the wheelbase never appears: the planners emit
``atan(L kappa)`` and the inverse divides by the same ``L``, and whatever ratio
the actuation assumes is absorbed, indistinguishably and harmlessly, into the
two gains. The wheelbase is a gauge, not a parameter, and is not estimated.

**3. The speed map.** The arc length of the same increment gives the speed, and

    v = v_gain * u_v + v_offset

carries the state of the battery in ``v_gain`` (a slow random walk, not a
constant). The command below which the motor does not start is *not* fitted
through that kink: it is bracketed from the launches themselves, where the
commanded value at which the car breaks away is an upper bound on it.

Every estimate starts at its prior with the prior's covariance, so the
recursion is the Bayesian posterior: it starts where the car's datasheet says,
moves only as fast as the sigma-weighted evidence warrants, and does not move
at all in directions nothing excited (a car that has only turned left keeps the
nominal right-hand gain, with its full prior uncertainty). That is the same
behaviour the SLAM has over lap 1, and it is why no extra shrinkage towards the
prior is needed anywhere in the controller.

Two safeguards matter as much as the mathematics:

* a floor under the measurement variance. The published sigmas are tiny
  (sub-millimetre on ``dy``), so without a floor the filter would trust the
  data a hundred times more than the model deserves and chase suspension and
  Ackermann error. The floor is measured from the residuals themselves and only
  ever rises.
* a watchdog. The one-step prediction error of the identified model is compared
  with that of the prior, continuously and for free. If the identified model is
  the worse of the two, it is thrown away and the prior restored; three times
  and the adaptation latches off for the rest of the run. The adaptive car can
  therefore never be worse than the nominal one by more than one window.

The module is pure NumPy and ROS-independent: the same code runs in the node,
in the 2D harness, in the offline replay and in the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .race_config import RaceDriverConfig

ARM, SLIP_ARM = 0, 1  # lever block
GAIN_L, GAIN_R, TRIM, SLIP = 0, 1, 2, 3  # curvature block
V_GAIN, V_OFF = 0, 1  # speed block
# Why an update was refused, for the run report.
REASONS = ("dt", "sigma", "health", "slow", "saturated", "transient", "braking", "deadband", "no_command", "latched")


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def lever_measurement(dx: float, dy: float, dyaw: float) -> tuple[float, float]:
    """``(y, regressor)`` of the lever equation, with ``y = regressor * arm``.

    Exact for any turn angle: for a rear-axle arc of curvature ``k`` the
    increment of a point ``arm`` ahead of that axle is

        dx = sin(dyaw)/k + arm (cos dyaw - 1)
        dy = (1 - cos dyaw)/k + arm sin(dyaw)

    and ``dy - dx tan(dyaw/2)`` kills the ``1/k`` term identically, because
    ``(1 - cos 2h) - tan(h) sin 2h == 0``, leaving ``2 tan(h) arm``. The
    regressor vanishes with the turn, which is why a straight line carries no
    information about the arm and is gated out.
    """
    t2 = math.tan(0.5 * dyaw)
    return dy - dx * t2, 2.0 * t2


def curvature_measurement(dx: float, dyaw: float, arm: float) -> float:
    """Curvature the rear axle drove, from the increment of the point ``arm``
    ahead of it. Exact, by inverting the same two expressions."""
    d = dx + arm * (1.0 - math.cos(dyaw))
    return math.sin(dyaw) / d if abs(d) > 1e-12 else float("nan")


@dataclass(frozen=True)
class PlantModel:
    """Frozen snapshot of the identified plant: what the control may use."""

    gain_left: float
    gain_right: float
    trim: float
    slip: float
    arm_m: float
    v_gain: float
    v_offset: float
    dead_mps: float
    dead_rev_mps: float
    kappa_sat: float
    v_ref_mps: float
    tau_steer_s: float
    tau_speed_s: float
    confidence: float
    conf_right: float

    def gain_at(self, kappa: float, speed: float) -> float:
        """Curvature delivered per unit of curvature commanded, at this speed.

        The slip term is written against the deviation from the speed the gain
        was identified at. At one single speed ``v^2 * u`` and ``u`` are the
        same regressor, so an uncentred pair would split the effect between
        them arbitrarily and the split would then be extrapolated to the race
        speed, where it is worth 16 % of gain. Centred, the gain is what was
        measured and the slip only moves when the speed actually varies.
        """
        gain = self.gain_right if kappa < 0.0 else self.gain_left
        return max(0.2, gain + self.slip * (speed * speed - self.v_ref_mps ** 2))

    def kappa_out(self, kappa_cmd: float, speed: float) -> float:
        """Curvature the plant delivers for ``kappa_cmd`` (forward map)."""
        return self.gain_at(kappa_cmd, speed) * kappa_cmd + self.trim

    def kappa_in(self, kappa_des: float, speed: float) -> float:
        """Curvature to command so that ``kappa_des`` is delivered (inverse)."""
        target = kappa_des - self.trim
        return target / self.gain_at(target, speed)

    def rear_offset_m(self) -> float:
        """Reference point of the tracking in the odometry frame."""
        return -self.arm_m

    def speed_in(self, v_des: float) -> float:
        """Speed to command so that ``v_des`` is delivered (inverse)."""
        return (v_des - math.copysign(self.v_offset, v_des)) / max(self.v_gain, 0.2)

    def dead_for(self, v_cmd: float) -> float:
        return self.dead_rev_mps if v_cmd < 0.0 else self.dead_mps


class _Block:
    """One linear-measurement Kalman filter on a parameter vector."""

    def __init__(self, x0: np.ndarray, sigma0: np.ndarray, box: np.ndarray, walk: np.ndarray, floor: float) -> None:
        self.x = np.array(x0, dtype=float)
        self.x0 = np.array(x0, dtype=float)
        self.p = np.diag(np.asarray(sigma0, dtype=float) ** 2)
        self.p0 = np.diag(self.p).copy()
        self.box = np.asarray(box, dtype=float)
        self.walk = np.asarray(walk, dtype=float)
        self.floor = float(floor)
        self.r_hat = float(floor) ** 2  # measured residual, two-sided
        self.chi = 1.0
        self.n = 0
        self.j_found = 0.0  # prediction error of the identified model
        self.j_prior = 0.0  # ... and of the prior, for the watchdog

    def drift(self, dt: float) -> None:
        """A parameter that moves (the battery) loses certainty with time."""
        if self.walk.any():
            np.fill_diagonal(self.p, np.minimum(np.diag(self.p) + (self.walk * dt) ** 2, self.p0))

    def update(self, h: np.ndarray, z: float, r_sensor: float) -> float:
        """One measurement. Returns the residual (0.0 if it was refused)."""
        e = float(z - h @ self.x)
        e_prior = float(z - h @ self.x0)
        r = max(float(r_sensor), self.floor ** 2, self.r_hat)
        ph = self.p @ h
        s = float(h @ ph) + r
        if s <= 0.0 or not math.isfinite(s) or not math.isfinite(e):
            return 0.0
        self.x = self.x + ph * (e / s)
        self.p = self.p - np.outer(ph, ph) / s
        self.p = 0.5 * (self.p + self.p.T)
        np.clip(self.x, self.box[:, 0], self.box[:, 1], out=self.x)
        np.fill_diagonal(self.p, np.clip(np.diag(self.p), 1e-12, self.p0))
        self.chi += 0.02 * (e * e / s - self.chi)
        # What the residuals actually are. The published sigmas are
        # sub-millimetre, so without this the filter would trust them a
        # hundred times more than the model deserves and chase suspension and
        # Ackermann error; with it, a model error inflates the variance it is
        # responsible for and the convergence stays honest.
        self.r_hat += 0.02 * (e * e - self.r_hat)
        self.r_hat = min(self.r_hat, 100.0 * self.floor ** 2)
        self.j_found += 0.05 * (e * e - self.j_found)
        self.j_prior += 0.05 * (e_prior * e_prior - self.j_prior)
        self.n += 1
        return e

    def reset(self) -> None:
        self.x = self.x0.copy()
        self.p = np.diag(self.p0.copy())
        self.chi, self.j_found, self.j_prior = 1.0, 0.0, 0.0
        self.r_hat = self.floor ** 2

    def conf(self, idx: list[int]) -> float:
        rel = np.sqrt(np.diag(self.p)[idx] / self.p0[idx])
        return float(max(0.0, min(1.0, 1.0 - float(rel.max()))))

    def regressed(self) -> bool:
        """Is the identified model predicting *materially* worse than the prior?

        An absolute margin is as important as the ratio. When nothing was wrong
        with the datasheet in the first place the two models are the same model,
        their errors differ only by noise, and a ratio alone would trip on that
        noise and throw away a perfectly good estimate -- which is what it did
        on the recorded runs until this margin was added.
        """
        return self.n > 100 and self.j_found > 1.3 * self.j_prior + self.floor ** 2


class PlantIdentifier:
    """Recursive estimator of :class:`PlantModel`, one update per revolution.

    Feed it the command that was published, at the control rate
    (:meth:`push_command`), and every odometry estimate as it arrives
    (:meth:`on_estimate`). The estimates are late (the revolution plus the
    network take about 0.18 s), so the commands are kept in a short history and
    averaged over the interval the estimate actually covers. Aligning them is
    not an optimisation: it is the difference between identifying the plant and
    identifying the delay.
    """

    def __init__(self, cfg: RaceDriverConfig, control_hz: float = 50.0) -> None:
        a, veh = cfg.adaptive, cfg.vehicle
        self.cfg, self.a = cfg, a
        self.lever = _Block(
            x0=[-float(veh.rear_offset_m), 0.0],
            sigma0=[a.arm_sigma, a.slip_arm_sigma],
            box=[[-veh.rear_offset_m - a.arm_box_m, -veh.rear_offset_m + a.arm_box_m], [0.0, a.slip_arm_max]],
            walk=[0.0, 0.0], floor=a.floor_arm_m,
        )
        self.steer = _Block(
            x0=[1.0, float(veh.right_ratio), 0.0, float(a.slip_prior)],
            sigma0=[a.gain_sigma, a.gain_sigma, a.trim_sigma, a.slip_sigma],
            box=[[a.gain_min, a.gain_max], [a.gain_min, a.gain_max], [-a.trim_max, a.trim_max], [a.slip_min, a.slip_max]],
            walk=[a.gain_walk_per_s, a.gain_walk_per_s, a.trim_walk_per_s, 0.0], floor=a.floor_kappa,
        )
        self.long = _Block(
            x0=[1.0, 0.0],
            sigma0=[a.vgain_sigma, a.voff_sigma],
            box=[[a.vgain_min, a.vgain_max], [-a.voff_max, a.voff_max]],
            walk=[a.vgain_walk_per_s, 0.0], floor=a.floor_v,
        )
        # Start threshold: bracketed from the launches, not fitted through the kink.
        self.dead = float(veh.min_effective_mps)
        self.dead_rev = float(a.dead_rev_prior_mps)
        self.n_dead = 0
        # Steering saturation: only ever learned from saturated samples.
        self.kappa_nom = float(veh.kappa_max)
        self.sat_seen: list[float] = []
        self.kappa_sat = float(veh.kappa_max)
        # Command history, with the command already filtered by every candidate
        # lag so that reviewing the bank costs no extra walk over the history.
        self.tau_k = tuple(float(v) for v in a.tau_steer_grid_s)
        self.tau_v = tuple(float(v) for v in a.tau_speed_grid_s)
        depth = max(8, int(round(a.history_s * control_hz)))
        self.hist = np.zeros((depth, 1 + len(self.tau_k) + len(self.tau_v)), dtype=float)
        self.hist[:, 0] = -1.0e9
        self.h_at = 0
        self.ks = np.zeros(len(self.tau_k), dtype=float)
        self.vs = np.zeros(len(self.tau_v), dtype=float)
        self.i_steer = int(np.argmin(np.abs(np.asarray(self.tau_k) - a.tau_steer_prior_s)))
        self.i_speed = int(np.argmin(np.abs(np.asarray(self.tau_v) - a.tau_speed_prior_s)))
        self.res_k = np.zeros(len(self.tau_k), dtype=float)
        self.res_v = np.zeros(len(self.tau_v), dtype=float)
        self.n_res = 0
        self.t_review: float | None = None
        # Bookkeeping, sign check and watchdog.
        self.t_cmd: float | None = None
        self.pending = (0.0, 0.0)
        self.v_cmd = 0.0
        self.kappa_cmd = 0.0
        self.v_steady_since: float | None = None
        self.prev: tuple[float, tuple[float, float, float]] | None = None
        self.prev_speed = 0.0
        self.sign_votes = [0, 0]
        self.sign_fault = False
        self.trips = 0
        self.latched = False
        self.t_trip: float | None = None
        self.rejected = dict.fromkeys(REASONS, 0)
        self.v2_ref = 0.0
        self.kappa_meas = float("nan")
        self.kappa_meas_t = -1.0e9
        self.v_meas = float("nan")

    # ------------------------------------------------------------- commands
    def push_command(self, t: float, kappa_cmd: float, v_cmd: float) -> None:
        """Record the command that was published, at the control rate.

        What the history must hold is the *delivered* command as a function of
        real time, so that averaging it over the interval of an estimate is the
        same operation the estimate itself is: the measurement is a chord over
        0.1 s, not a sample. So each row stores the exact time-average of the
        lag filter over the step that just elapsed, and is stamped with the end
        of the window during which that average was actually acting, one
        actuation period after it was published. The command published now has
        not acted yet; it becomes the target of the next step.
        """
        if self.t_cmd is None:
            self.t_cmd = t
            self.pending = (kappa_cmd, v_cmd)
            self.ks[:] = kappa_cmd
            self.vs[:] = v_cmd
            return
        dt = max(1e-4, min(0.2, t - self.t_cmd))
        self.t_cmd = t
        k_target, v_target = self.pending
        self.pending = (kappa_cmd, v_cmd)
        row = self.hist[self.h_at]
        row[0] = t + self.a.cmd_delay_s
        for i, tau in enumerate(self.tau_k):
            row[1 + i] = self._advance(self.ks, i, k_target, tau, dt)
        for i, tau in enumerate(self.tau_v):
            row[1 + len(self.tau_k) + i] = self._advance(self.vs, i, v_target, tau, dt)
        self.h_at = (self.h_at + 1) % len(self.hist)
        if self.v_steady_since is None or abs(v_cmd - self.v_cmd) > 0.02:
            self.v_steady_since = t
        self.v_cmd, self.kappa_cmd = v_cmd, kappa_cmd

    @staticmethod
    def _advance(state: np.ndarray, i: int, target: float, tau: float, dt: float) -> float:
        """Step a first-order filter and return its exact mean over the step."""
        alpha = 1.0 - math.exp(-dt / tau)
        start = float(state[i])
        state[i] = start + alpha * (target - start)
        return target + (start - target) * (tau / dt) * alpha

    def _window(self, t0: float, t1: float) -> tuple[np.ndarray, float] | None:
        """Mean of every filtered command over ``(t0, t1]``, and its spread."""
        times = self.hist[:, 0]
        sel = (times > t0) & (times <= t1)
        if not sel.any():
            older = times <= t1
            if not older.any():
                return None
            row = self.hist[int(np.argmax(np.where(older, times, -np.inf))), 1:]
            return row, 0.0
        block = self.hist[sel, 1:]
        col = block[:, self.i_steer]
        return block.mean(axis=0), float(col.max() - col.min())

    # ------------------------------------------------------------ estimates
    def on_estimate(self, stamp: float, pose: tuple[float, float, float],
                    sigma: tuple[float, float, float] | None, healthy: bool = True) -> None:
        """One odometry estimate: its pose, in the odometry frame, at its stamp.

        ``sigma`` is the standard deviation of the pose increment
        ``(dx, dy, dyaw)`` that the estimator publishes with it.
        """
        prev, self.prev = self.prev, (stamp, pose)
        if not self.a.enabled or self.latched or prev is None:
            self._refuse("latched" if self.latched else "dt")
            return
        dt = stamp - prev[0]
        if not self.a.dt_min_s <= dt <= self.a.dt_max_s:
            self._refuse("dt")
            return
        xp, yp, yawp = prev[1]
        c, s = math.cos(yawp), math.sin(yawp)
        ex, ey = pose[0] - xp, pose[1] - yp
        dx, dy = c * ex + s * ey, -s * ex + c * ey
        dyaw = _wrap(pose[2] - yawp)
        sig = (self.a.sigma_xy_nominal_m, self.a.sigma_xy_nominal_m, self.a.sigma_yaw_nominal_rad) if sigma is None else tuple(max(abs(float(v)), 1e-6) for v in sigma)
        if not healthy:
            self._refuse("health")
            return
        if sig[0] > self.a.sigma_reject * self.a.sigma_xy_nominal_m or sig[2] > self.a.sigma_reject * self.a.sigma_yaw_nominal_rad:
            self._refuse("sigma")
            return
        self.lever.drift(dt)
        self.steer.drift(dt)
        self.long.drift(dt)
        win = self._window(prev[0], stamp)
        if win is None:
            self._refuse("no_command")
            return
        eff, spread = win
        self._update_lever(dx, dy, dyaw, sig, dt)
        kappa = self._update_steer(dx, dyaw, eff, sig, dt, spread)
        self._update_long(stamp, dx, dyaw, eff[len(self.tau_k):], sig, dt)
        self._review(stamp)
        self._watchdog(stamp)
        _ = kappa

    def _refuse(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    # -------------------------------------------------------------- blocks
    def _arc(self, dx: float, dyaw: float, arm: float) -> tuple[float, float]:
        """Chord and arc length of the rear axle for this increment."""
        h = 0.5 * dyaw
        d = dx + arm * (1.0 - math.cos(dyaw))
        chord = d / max(math.cos(h), 1e-6)
        arc = chord if abs(h) < 1e-6 else chord * h / math.sin(h)
        return d, arc

    def _update_lever(self, dx: float, dy: float, dyaw: float, sig, dt: float) -> None:
        a = self.a
        h = 0.5 * dyaw
        _, arc = self._arc(dx, dyaw, self.lever.x[ARM])
        speed = arc / dt
        if abs(dyaw) < a.arm_dyaw_min_rad or abs(speed) < a.v_min_update_mps:
            return
        y, regressor = lever_measurement(dx, dy, dyaw)
        t2 = 0.5 * regressor
        # The regressor depends on dyaw as well, so the residual's sensitivity
        # to it carries the 2 * arm term too.
        d_dyaw = (dx + 2.0 * self.lever.x[ARM]) / (2.0 * max(math.cos(h) ** 2, 1e-6))
        r = sig[1] ** 2 + (t2 * sig[0]) ** 2 + (d_dyaw * sig[2]) ** 2
        v2 = speed * speed if abs(speed) >= a.slip_arm_v_min_mps else 0.0
        hv = np.array([regressor, -regressor * v2])
        self.lever.update(hv, y, r)

    def _update_steer(self, dx: float, dyaw: float, eff_all: np.ndarray, sig, dt: float, spread: float) -> float:
        a = self.a
        eff = eff_all[:len(self.tau_k)]
        v_cmd_eff = abs(float(eff_all[len(self.tau_k) + self.i_speed]))
        arm = float(self.lever.x[ARM])
        d, arc = self._arc(dx, dyaw, arm)
        speed = arc / dt
        if abs(d) < 1e-4:
            return float("nan")
        kappa = curvature_measurement(dx, dyaw, arm)
        self.kappa_meas, self.kappa_meas_t = kappa, self.t_cmd or 0.0
        u = float(eff[self.i_steer])
        # Parked pose noise has a curvature but cannot establish its sign.
        # Apply the measured-speed gate before voting or learning saturation,
        # just as for the steering gains below.
        if abs(speed) < a.v_min_steer_mps:
            self._refuse("slow")
            return kappa
        self._sign_check(u, kappa)
        side_scale = self.cfg.vehicle.curvature_limit(u >= 0.0, planning=True) / self.kappa_nom
        if abs(u) >= a.sat_frac * self.kappa_sat * side_scale:
            self._saturation(kappa, speed)
            self._refuse("saturated")
            return kappa
        if spread > a.transient_max * dt:
            # A command that moved a lot inside the interval: the lag model
            # error dominates and would bias the gains.
            self._refuse("transient")
            return kappa
        if self.steer.n == 0:
            self.v2_ref = v_cmd_eff * v_cmd_eff
        dv2 = v_cmd_eff * v_cmd_eff - self.v2_ref
        # Residual of every candidate lag, scored with the current parameters.
        for i in range(len(self.tau_k)):
            ui = float(eff[i])
            pred = (self.steer.x[GAIN_R] if ui < 0.0 else self.steer.x[GAIN_L]) * ui + self.steer.x[TRIM] + self.steer.x[SLIP] * dv2 * ui
            self.res_k[i] += (kappa - pred) ** 2
        dk_dyaw = (math.cos(dyaw) * d - math.sin(dyaw) * arm * math.sin(dyaw)) / (d * d)
        dk_dx = -math.sin(dyaw) / (d * d)
        dk_darm = -math.sin(dyaw) * (1.0 - math.cos(dyaw)) / (d * d)
        r = (dk_dyaw * sig[2]) ** 2 + (dk_dx * sig[0]) ** 2 + dk_darm ** 2 * float(self.lever.p[ARM, ARM])
        hv = np.array([max(u, 0.0), min(u, 0.0), 1.0, dv2 * u])
        self.steer.update(hv, kappa, r)
        self.v2_ref += max(1.0 / (self.steer.n + 1), 0.01) * (v_cmd_eff * v_cmd_eff - self.v2_ref)
        return kappa

    def _sign_check(self, u: float, kappa: float) -> None:
        """A reversed steering is a fault, not a parameter: the gains are
        boxed positive, so a wiring or configuration flip has to be detected
        and refused rather than fitted."""
        if self.sign_fault or sum(self.sign_votes) >= self.a.sign_votes or abs(u) < self.a.sign_kappa_min:
            return
        self.sign_votes[0 if u * kappa >= 0.0 else 1] += 1
        if sum(self.sign_votes) >= self.a.sign_votes and self.sign_votes[1] > self.sign_votes[0]:
            self.sign_fault = True

    def _saturation(self, kappa: float, speed: float) -> None:
        gain = self.steer.x[GAIN_R] if kappa < 0.0 else self.steer.x[GAIN_L]
        eff = float(gain) + float(self.steer.x[SLIP]) * (speed * speed - self.v2_ref)  # noqa: PLW2901
        side_scale = self.cfg.vehicle.curvature_limit(kappa >= 0.0, planning=True) / self.kappa_nom
        self.sat_seen.append(abs(kappa) / max(eff, 0.2) / side_scale)
        if len(self.sat_seen) > self.a.sat_window:
            self.sat_seen.pop(0)
        if len(self.sat_seen) >= self.a.sat_window // 2:
            self.kappa_sat = max(self.a.sat_floor_frac * self.kappa_nom, float(np.quantile(self.sat_seen, 0.1)))

    def _update_long(self, stamp: float, dx: float, dyaw: float, eff: np.ndarray, sig, dt: float) -> None:
        a = self.a
        _, arc = self._arc(dx, dyaw, float(self.lever.x[ARM]))
        speed = arc / dt
        self.v_meas = speed
        u = float(eff[self.i_speed])
        self._update_dead(speed, u)
        for i in range(len(self.tau_v)):
            pred = self.long.x[V_GAIN] * float(eff[i]) + self.long.x[V_OFF]
            self.res_v[i] += (speed - pred) ** 2
        self.n_res += 1
        accel = (speed - self.prev_speed) / dt
        self.prev_speed = speed
        tau = self.tau_v[self.i_speed]
        if self.v_steady_since is None or stamp - self.v_steady_since < a.steady_taus * tau:
            self._refuse("transient")
            return
        if abs(u) < self.dead + a.deadband_margin_mps:
            self._refuse("deadband")
            return
        if abs(u) + 3.0 * sig[0] / dt < abs(speed) or abs(accel) > a.accel_max_update_mps2:
            # Coasting or braking: the car is slowed by drag and by the brake,
            # neither of which is the gain this block identifies.
            self._refuse("braking")
            return
        r = (sig[0] / dt) ** 2
        self.long.update(np.array([u, 1.0]), speed, r)

    def _update_dead(self, speed: float, u: float) -> None:
        """Bracket the start threshold from the launches themselves.

        The command at which a standing car breaks away is an upper bound on
        the threshold; a command that produces no motion is a lower bound.
        Both are one-sided, which is far more robust than differentiating
        through a kink.
        """
        a = self.a
        if abs(u) < 1e-3:
            return
        rev = u < 0.0
        dead = self.dead_rev if rev else self.dead
        hi = a.dead_rev_max_mps if rev else a.dead_max_mps
        moved = abs(speed) > 0.10 and abs(self.prev_speed) < 0.05
        stalled = abs(speed) < 0.05 and abs(self.prev_speed) < 0.05
        if moved and abs(u) < dead:
            dead += a.dead_step * (abs(u) - dead)
        elif stalled and abs(u) > dead:
            dead += a.dead_step * (abs(u) - dead)
        else:
            return
        dead = max(0.0, min(hi, dead))
        if rev:
            self.dead_rev = dead
        else:
            self.dead = dead
        self.n_dead += 1

    # --------------------------------------------------------- bank, watchdog
    def _review(self, stamp: float) -> None:
        a = self.a
        if self.t_review is None:
            self.t_review = stamp
            return
        if stamp - self.t_review < a.lag_review_s or self.n_res < a.lag_window:
            return
        self.t_review = stamp
        # Switch only for a clear and sustained advantage: a lag that flaps
        # would bias the gains by several per cent every time it moved.
        for res, cur, setter in ((self.res_k, self.i_steer, "i_steer"), (self.res_v, self.i_speed, "i_speed")):
            best = int(np.argmin(res))
            if best != cur and res[best] < a.lag_advantage * res[cur]:
                setattr(self, setter, best)
        self.res_k[:] = 0.0
        self.res_v[:] = 0.0
        self.n_res = 0

    def _watchdog(self, stamp: float) -> None:
        """Throw the identified model away if the prior predicts better."""
        if not any(b.regressed() for b in (self.lever, self.steer, self.long)):
            self.t_trip = None
            return
        if self.t_trip is None:
            self.t_trip = stamp
            return
        if stamp - self.t_trip < self.a.watchdog_hold_s:
            return
        self.t_trip = None
        self.trips += 1
        for b in (self.lever, self.steer, self.long):
            b.reset()
        self.dead = float(self.cfg.vehicle.min_effective_mps)
        self.kappa_sat = self.kappa_nom
        self.sat_seen.clear()
        if self.trips >= self.a.watchdog_trips:
            self.latched = True

    # -------------------------------------------------------------- outputs
    @property
    def conf_lever(self) -> float:
        ok = self.lever.n >= self.a.n_lever and self.lever.chi <= self.a.chi_max
        return self.lever.conf([ARM]) if ok else 0.0

    @property
    def conf_steer(self) -> float:
        ok = self.steer.n >= self.a.n_steer and self.steer.chi <= self.a.chi_max
        return self.steer.conf([GAIN_L, TRIM]) if ok else 0.0

    @property
    def conf_long(self) -> float:
        return self.long.conf([V_GAIN, V_OFF]) if self.long.n >= self.a.n_long else 0.0

    @property
    def longitudinal_ready(self) -> bool:
        """Both speed coefficients need evidence before their inverse is used.

        A single reverse observation can fit an offset that nearly cancels a
        small forward target. Count alone also cannot separate gain and offset
        at one command level. Keep identifying, but retain the unit-gain,
        zero-offset prior until both standard deviations halve and residuals
        remain consistent. This is independent of lateral-chain confidence.
        """
        return (not self.latched and not self.sign_fault and self.conf_long >= 0.5
                and self.long.chi <= self.a.chi_max)

    @property
    def confidence(self) -> float:
        """Confidence of the lateral chain, which is what the envelope uses.

        The right-hand gain is deliberately left out: until the car has turned
        that way it keeps the nominal value the unmodified driver uses, so
        including it would hold the envelope closed over a track that only
        turns one way without making the car any safer than it is today.
        """
        if self.latched or self.sign_fault:
            return 0.0
        return min(self.conf_steer, self.conf_lever)

    def model(self) -> PlantModel:
        if self.latched or self.sign_fault:
            return self.prior()
        a = self.a
        speed_ready = self.longitudinal_ready
        return PlantModel(
            gain_left=float(self.steer.x[GAIN_L]), gain_right=float(self.steer.x[GAIN_R]),
            trim=float(self.steer.x[TRIM]), slip=float(self.steer.x[SLIP]), arm_m=float(self.lever.x[ARM]),
            v_gain=float(self.long.x[V_GAIN]) if speed_ready else 1.0,
            v_offset=float(self.long.x[V_OFF]) if speed_ready else 0.0,
            dead_mps=float(self.dead), dead_rev_mps=float(self.dead_rev), kappa_sat=float(self.kappa_sat),
            v_ref_mps=math.sqrt(max(self.v2_ref, 0.0)), tau_steer_s=self.tau_k[self.i_steer], tau_speed_s=self.tau_v[self.i_speed],
            confidence=self.confidence, conf_right=self.steer.conf([GAIN_R]),
        )

    def prior(self) -> PlantModel:
        """The model the car starts from (its datasheet)."""
        veh, a = self.cfg.vehicle, self.a
        return PlantModel(
            gain_left=1.0, gain_right=float(veh.right_ratio), trim=0.0, slip=0.0,
            arm_m=-float(veh.rear_offset_m), v_gain=1.0, v_offset=0.0,
            dead_mps=float(veh.min_effective_mps), dead_rev_mps=float(a.dead_rev_prior_mps),
            kappa_sat=float(veh.kappa_max), v_ref_mps=0.0, tau_steer_s=a.tau_steer_prior_s, tau_speed_s=a.tau_speed_prior_s,
            confidence=0.0, conf_right=0.0,
        )

    def fresh_kappa(self, t: float, max_age: float = 0.15) -> float:
        """The last measured curvature, or NaN if it is stale.

        Held values would be logged at the control rate and compared against a
        request that has moved on, which reads as tracking error and is not.
        """
        return self.kappa_meas if t - self.kappa_meas_t <= max_age else float("nan")

    def status(self) -> dict:
        m = self.model()
        return {
            "confidence": m.confidence, "conf_steer": self.conf_steer, "conf_lever": self.conf_lever,
            "conf_long": self.conf_long, "conf_right": m.conf_right,
            "longitudinal_applied": self.longitudinal_ready,
            "v_gain_estimated": float(self.long.x[V_GAIN]),
            "v_offset_estimated": float(self.long.x[V_OFF]),
            "gain_left": m.gain_left, "gain_right": m.gain_right, "trim": m.trim, "slip": m.slip,
            "arm_m": m.arm_m, "v_gain": m.v_gain, "v_offset": m.v_offset, "dead_mps": m.dead_mps,
            "dead_rev_mps": m.dead_rev_mps, "kappa_sat": m.kappa_sat,
            "v_ref_mps": m.v_ref_mps, "tau_steer_s": m.tau_steer_s, "tau_speed_s": m.tau_speed_s,
            "n": {"lever": self.lever.n, "steer": self.steer.n, "long": self.long.n, "dead": self.n_dead},
            "chi": {"lever": self.lever.chi, "steer": self.steer.chi, "long": self.long.chi},
            "sigma": {"arm": math.sqrt(float(self.lever.p[ARM, ARM])), "gain_left": math.sqrt(float(self.steer.p[GAIN_L, GAIN_L])),
                      "gain_right": math.sqrt(float(self.steer.p[GAIN_R, GAIN_R])), "trim": math.sqrt(float(self.steer.p[TRIM, TRIM])),
                      "v_gain": math.sqrt(float(self.long.p[V_GAIN, V_GAIN]))},
            "sign_fault": self.sign_fault, "trips": self.trips, "latched": self.latched,
            "rejected": dict(self.rejected),
        }
