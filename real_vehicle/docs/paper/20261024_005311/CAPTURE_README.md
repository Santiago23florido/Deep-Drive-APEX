# Real vehicle: longitudinal-confidence-gated repeat

ROS run **20261024_005311**, diagnostic **20261024_005311**. This is a new controller change,
separate from the earlier operator-approved 000432 capture and its Git tag.
Run IDs use Pi local time; epoch stamps are the shared time basis. Operator
date was October 24, 2026 (Europe/Paris).

## Outcome

Stop reason: **driver_done**. Exploration-to-recorded-end interval:
62.099 s. Onboard completed laps: 2.
- Lap 1: 46.837 s, 8 reverse starts.
- Lap 2: 14.594 s, 1 reverse starts.

The operator-confirmation record is retained in `raw/diagnostics/operator_feedback.json`.
Observations are included only when supplied; unknown outcomes remain explicit:
```json
{
  "physical_completed_laps": 2,
  "operator_confirmation_pending": false,
  "operator_confirmed_success": true,
  "operator_selected_as_retained_version": true,
  "physical_speed_comparison": "Operator confirmed the desired speed, like the original run.",
  "contact_with_track": "not reported",
  "original_spanish": "esta vez klo hizo a la veolocidad qwiueera como el original funciono bien esi toomo los datos eses es qel que queiro dejar",
  "recorded_utc": "2026-10-24T00:03:15.136156+00:00"
}
```
Onboard lap counts and operator observations are distinct evidence. Contact
with track boundaries must not be inferred if it was not reported.

## Exact control change

The speed Kalman block continues identifying gain and offset from odometry.
Its inverse reaches the control only after `conf_long >= 0.5` (including the
existing minimum update count) and consistent residuals (`chi <= chi_max`).
Before that, effective speed gain/offset remain 1/0, as in the original fast
run. Lateral confidence is independent. Steering adaptation, deadband/launch
learning, model weights, map mode, PWM caps, finite mission and guards are
unchanged. Shared real/SIM `plant_id.py` copies are identical. A new regression
test exercises a single reverse observation, repeated collinear observations,
identified speed parameters with zero lateral confidence, and residual rejection.
221 tests passed, 3 ROS skips and 3 slow deselections; test output is included.
Before launch all 138 selected PC/Pi files matched; only the runtime plant
identifier differs from the approved source. Exact source, weights, patch and
preflight/postflight parity are retained. The prior approved Git tag was not
changed by this experiment.

Final effective and raw speed estimates, confidence and application flag:
```json
{
  "v_gain": 1.0,
  "v_offset": 0.0,
  "v_gain_estimated": 0.6,
  "v_offset_estimated": 0.02030784288795917,
  "longitudinal_applied": false,
  "conf_long": 0.42112685594235033,
  "dead_mps": 0.06993777028695689,
  "dead_rev_mps": 0.3977056826174298,
  "confidence": 0.0,
  "n": {
    "lever": 164,
    "steer": 128,
    "long": 32,
    "dead": 38
  }
}
```
These separate what the estimator fitted from what control used. No network
weights were retrained. This engineering repeat does not establish untouched
zero-shot full-stack transfer or statistical reliability.

## Data, timing and execution

The same passive raw IMU/LiDAR and CPU/RSS capture writes trial CSV/JSON to host
tmpfs, verified inside Docker before arm. After writers stop and PWM is neutral,
the wrapper copies to SD, verifies SHA256/sizes, fsyncs, restores mounts and
publishes new paths without overwriting older evidence. Ordinary OS/Docker/ROS
console writes are not globally disabled. Postflight storage and independent
SSH hash checks are included. No per-command detailed timing probe was added.

`real_run_metrics.json` defines all windows/percentiles. Hybrid processing
includes features, ICP/network or stationary hold, deskew and integration,
excluding input waiting/output publication. Header age includes sweep acquisition
and waiting. Propagated pose freshness is not a new independent LiDAR correction.
Driver maximum command publication gap was 45.05 ms;
bridge timeout samples: 0. Pi live fault
samples: 0; these are not motor voltage/current
measurements. The recorder is best effort: complete files do not establish
lossless sensor transport. Timing anomalies remain in the raw data and metrics.

Actual executed phases are shown in the figures. Check `raw/diagnostics/plan.json`
for global candidate acceptance: map-guided local planning must not be described
as direct tracking of a rejected candidate. Coordinates, distances and speeds
are onboard estimates. No external ground truth or motor electrical sensing
was available. Scan geometry and stationary stability are correlated diagnostics
of the deployed hybrid estimator, not ATE/RPE, drift or absolute position error.

`run_comparison.csv` compares original fast, previous full capture and current
trial. Different starts, maps, online estimates and unmeasured battery state
make these engineering comparisons, not controlled causal ablations. Positive
ESC snapshot distributions are not time-weighted and are not electrical power.
Command duration accounting uses zero-order holds, not physical-motion duration.
CPU 100% means one core; RSS sums may double-count shared pages.

## English figures and reproduction

`figures/` supplies English vector PDF/SVG and 600 dpi PNG for execution,
SLAM/laps, handover, within-run lap comparison when two crossings exist,
odometry/timing and active Raspberry resources. Tables, plotted traces and
captions accompany them. Source and models are under `reproducibility/source`;
analysis requires no ROS or car. Earlier OS/image/library reports are explicitly
labeled previous; live new preflight/source/resources/final state are retained.

Run `python3 verify_archive.py` before modifying a sealed package. With versions
from `requirements-analysis.txt`, run `python3 reproduce_figures.py` offline.
Regeneration can change PDF timestamps and therefore hashes. Geometry sanity
tests are in `reproducibility/analysis/test_audit_real_run.py`.
