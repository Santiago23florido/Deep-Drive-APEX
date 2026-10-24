# Preprint readiness: simulation and physical evidence

Audit date: **October 24, 2026 (Europe/Paris)**. This is a repository-based
experimental gap report, not a conference acceptance checklist.

## Current conclusion

**Latest accepted calibrated hardware result: [run `220316`](../../real_vehicle/docs/paper/20261024_220316/README.md).**
The operator confirmed two physical laps without contact and no second-lap
reverse. Recorded durations were 29.644 / 10.875 s, with 3 / 0 reverse starts.
Selected lap-2 reference / onboard mean / MAE were 0.360 / 0.309 / 0.054 m/s.
All 144 selected runtime source/config/model files matched PC/Pi after stop.
The original approved 005311 acquisition below remains a historical result.

Longitudinal Kalman accepted 12 updates, fewer than the minimum 20; confidence
was zero and its inverse remained inactive. This supports calibrated deployment
and reference tracking against an onboard estimate, not full plant convergence,
external accuracy, a repeated-trial success rate or untouched full-stack
zero-shot transfer. Runtime `f907cce` is frozen under tag
`real-car-calibrated-20261024-220316`; the complete source/model/data ZIP is linked
from its report. The final simulator campaign below remains outstanding.

The physical implementation and its approved data are preserved. The remaining
work is chiefly a **frozen-version simulation evaluation and a reproducible
paper package**. The existing real recording can support a deployment case
study, onboard timing and internal consistency. It cannot supply external
trajectory accuracy or an untouched full-stack zero-shot benchmark.

The acquisition runtime is commit `661e33e5cf2b8037d5b6129cfd1e991e26d4bcb0`,
tag `real-car-approved-20261024-005311`. That original documentation integration
did not retune the controller or launch the vehicle. Subsequent physical
calibration, attempts and their outcomes are recorded separately above and
below; deployed neural weights remain unchanged.

## Evidence already available

| Item | Evidence | What it supports |
| --- | --- | --- |
| Approved physical trial | [Run 005311](../../real_vehicle/docs/paper/20261024_005311/README.md), operator confirmation, two onboard crossings | Successful physical two-lap operation at the operator's desired response |
| Physical lap times | 46.8366 / 14.5937 s; 8 / 1 reverse starts | Descriptive outcome of one selected development trial |
| Actual planning mode | MAP_GUIDED; optimized candidate rejected | SLAM/observed-route-guided local continuation, with recovery |
| Raw capture | 9,002 IMU, 889 scans, 9,891 receipt rows | Replay and sensor/timing diagnostics; best-effort transport |
| Runtime processing | Hybrid median/P95/max 14.165/18.128/26.490 ms | Deployed pipeline processing, not isolated neural inference |
| Freshness | Propagated pose age median/P95 15.000/23.240 ms | Age of the prediction consumed by control |
| Runtime command continuity | 45.05 ms maximum publication gap, no recorded bridge timeout/stale/core-busy stop | No such stop in this approved trial; not a proof about all earlier failures |
| Resource capture | Driver/odometry/recorder median 96.97/31.49/11.00% of one core | CPU cost on this deployed platform; limited driver-core headroom |
| Static pipeline stability | 1.866 mm maximum estimated excursion over 9.696 s, hold active throughout | Full-pipeline stationary diagnostic, not independent model-only accuracy |
| Scan consistency | Mean 0.03520 m | Correlated geometric consistency, not ATE/RPE |
| Provenance | 138 selected PC/Pi file hashes matched before and after acquisition | Exact source/configuration/model correspondence at acquisition |
| Portable evidence | 305 payload files, ZIP CRC and all payload hashes verified | Sealed downloadable data/source/model package |
| Existing simulator | [Research package](../../simulation/ros2_ws/src/apex_fusion_research/README.md), [learning catalogue](../../simulation/learning/lidar_imu_pose/README.md) | Historical Gazebo, odometry, stationary and robustness studies |

The earlier approved capture remains separate: `20261024_000432`, 83.823/42.773 s,
13/4 reverse starts. Its source/configuration matched the original fast runtime,
but an underdetermined online speed fit suppressed commands. The retained change
gates the inverse on independent longitudinal confidence. In 005311, raw
confidence was 0.421, so effective gain/offset stayed 1/0 while estimation
continued. This observation supports the mechanism; different maps, starts and
unmeasured battery conditions prevent a causal performance claim from these
three development runs alone.

![Approved real execution, onboard estimates](../../real_vehicle/docs/paper/20261024_005311/figures/real_vehicle_execution.png)

[All six English figure sets](../../real_vehicle/docs/paper/20261024_005311/README.md#data-and-english-figures)
include PDF/SVG and 600 dpi PNG. The ZIP contains their inputs and offline scripts.
No new reference trajectory, video or electrical measurement is created by this
audit.

## Simulation changes included in the integration

- The shared plant identifier retains unit speed gain/zero offset until
  `conf_long >= 0.5`, the existing minimum sample count and residual checks pass.
  Raw estimates remain visible and steering/deadband/launch adaptation continues.
- The same driving core carries directional steering authority, geometric
  recovery, finite mission bounds, corrected lap closure and continuous handover
  from the current pose. No per-oval turn schedule is introduced.
- The stationary hold covers the IMU-predicted pose as well as scan corrections,
  avoiding fictitious travel between stationary scans.
- Four simulator regressions now independently exercise the speed-confidence
  behavior, in addition to the real-package regressions and parity tests.
- Existing changes to the simulator's SLAM replay tool and recovery/mission/
  steering tests are carried to main with the branch.

The current real and simulated *core files* match mechanically. Their vehicle
parameters, launch schedules, sensor profiles and actuator interfaces differ.
Passing parity tests does not establish equivalent physical dynamics.

## What remains before a defensible preprint

The following priorities describe the proposed evidence for this project. They
are not universal publication rules or sample-size requirements.

| Priority | Remaining work | Deliverable / completion condition |
| --- | --- | --- |
| P0 | Freeze the final method and intended claim | Source commit, PT/ONNX hashes, dataset/split manifest, all configs, declared online resets/updates and recorded tuning history |
| P0 | Re-run the final controller in Gazebo | Fresh-process multi-track/seed runs, all attempts retained, reconnaissance and continuation reported separately |
| P0 | Compare relevant baselines and contributions | Same declared scenarios/limits and paired random draws where possible; per-run results and uncertainty, including failures |
| P0 | Evaluate odometry against simulator truth | Open-loop and closed-loop metrics at fixed horizons/distances; separate estimator-only and SLAM-corrected pose |
| P0 | Package simulation evidence | Raw/per-run outputs, source/config hashes, dataset/checkpoint retrieval or regeneration instructions, CSV tables and English vector figures |
| P0 | Write the method and limitations around actual execution | Latest MAP_GUIDED result has no lap-2 reverse; disclose inactive longitudinal inverse, no real ATE/RPE, hardware tuning and online identification |
| P1 | Stress timing, recovery and observability | Low speed, stop/restart, tight turns in both directions, sparse scans, delayed/lost scans, bias and plant mismatch; failure/guard counts |
| P1 | Repeatability and sensitivity | Multiple independent starts; confidence-gate/sample/excitation sensitivity; median, spread and trial-level confidence intervals |
| Optional | Distill v7 into the smaller v6 encoder | Only if compute reduction is claimed; v6 currently uses the older teacher and is not the deployed model |

A new training run is **not inherently required** for these tasks. First evaluate
the frozen deployed weights and final controller. Retraining or distillation
creates a different method/version and needs a separate evaluation.

### 1. Freeze the protocol and protect the test set

Choose a primary contribution: sensor-realistic transfer, hybrid odometry,
confidence-aware online identification, or the complete autonomy integration.
Every chosen contribution needs a corresponding measurement and comparator.

Freeze thresholds, mapping parameters and checkpoints before the final campaign.
Use validation scenarios for further changes, and explicitly mark all scenarios
already inspected during controller development. `test_unseen` is held out of
training but has repeatedly been used for controller development; it is not an
untouched test of the complete stack. The
[held-out variant definitions](../../simulation/tools/pose_dataset/ZERO_SHOT.md)
protect dataset splits, not automatically the history of all system tuning.
If untouched system generalization is claimed, reserve new layouts/seeds before
running them and do not tune after viewing their outcomes.

Record separate hashes for training/validation/test datasets and both PT/ONNX
weights. The exact deployed weights are in the real archive; the broader
training/evaluation outputs are Git-ignored and still need an explicit release.
Do not silently infer a checkpoint from a filename alone.

### 2. Run the final-version simulation campaign

Start with a small smoke set; then freeze a larger matrix. A practical initial
matrix is at least three geometries, both turn directions and ten independent
seeds per geometry, with three laps per run. This is a planning target, not a
statistical guarantee. Include narrow/high-curvature tracks, feature-poor
corridors and a layout not used to develop the controller.

Reset map, recurrent odometry state, identification and route history between
independent trials. Keep adaptation and mapping continuous between laps within
one trial. Report all failed/aborted runs and their stop reasons. Three laps
within a run are correlated observations, not three independent trials.

The 2D harness is a useful regression/development tool. Its synthetic odometry
and simplified SLAM use simulated truth; its completion is not a replacement for
the Gazebo learned-odometry closed loop. This integration ran the existing 2D
integration tests, but did **not** run a fresh multi-seed Gazebo campaign.

### 3. Baselines and ablations

| Question | Suggested comparison | Keep fixed |
| --- | --- | --- |
| Does the hybrid estimator add value? | Classical scan-to-submap ICP vs learned hybrid; optionally network-only | Data intervals, sensor perturbations and evaluation metrics |
| Does online identification help? | Full method vs `adaptive.enabled=false` | Vehicle draws, geometry, mission/actuator bounds and initial-state protocol |
| Does the confidence gate prevent false inversions? | Declared gated method vs historical ungated method, simulation/offline only | Sparse/reverse excitation and measurement noise; report raw and applied parameters |
| Does the stationary correction work? | Full deployed hold vs `--standstill-hold false` | Parked/stop/restart streams, plus driving accuracy and release delay |
| Does map guidance improve continuation? | Declared reactive-only reference vs mapped continuation | Same geometry and nominal limits; expose which plan was actually executed |
| Is real-time headroom adequate? | Fixed-rate replay/loaded runs with and without passive capture | Same inputs and scheduling; measure delays and processing separately |

Some baselines need an explicit experimental configuration or replay branch;
the table is a protocol proposal, not a claim that every switch exists today.
Do not implement a baseline by feeding truth into the normal driver. Truth belongs
only in the referee/evaluator, except for clearly labeled oracle upper bounds.
Keep ungated/unsafe experimental variants out of the approved physical runtime.

### 4. Metrics and plots

Use the implemented
[`odometry_metrics.py`](../../simulation/learning/lidar_imu_pose/odometry_metrics.py)
and [closed-loop evaluator](../../simulation/tools/analysis/evaluate_real2sim_run.py).
For simulation, report translation/speed and yaw-rate error, RPE at fixed time
horizons, segment drift at fixed distances, uncertainty coverage and
stationary false motion. State alignment conventions for any ATE metric and
separate open-loop odometry from SLAM-corrected estimates.

For control, report physical/referee completion, true footprint clearance,
contacts, lap times, reverse count/duration/distance, stalled time, interventions,
handover transients and the executed planning mode. An internally feasible map
route is not automatically a physically feasible swept vehicle path.

The real run's rejected candidate had maximum curvature **2.487 /m** against
a **1.460 /m** nominal limit and minimum mapped clearance **0.0476 m**. These are
onboard candidate/map quantities, not measured physical wheel curvature or
clearance. The fallback correctly prevented direct tracking of that candidate.
If the paper claims optimized-line execution or forward-only continuation,
demonstrate it with final-version simulation evidence; the approved physical
capture does not establish either outcome.

For runtime, distinguish acquisition age, correction processing, propagation
freshness, command publication gaps and actual actuation delay. The approved
capture has no detailed per-command source/receive/PWM trace. It cannot establish
a distribution of end-to-end motor latency. Likewise, Pi power flags do not
measure motor voltage/current or identify every earlier weak-motion cause.

### 5. Transfer wording and real evidence

A precise description is: simulation-trained hybrid odometry deployed with
unchanged weights during the final physical trial, with declared online
identification and a controller developed using hardware feedback. The v7
training design also incorporated the observed real gyro/standstill signature;
describe that sensor-model development explicitly.

Zero-shot *weight transfer*, zero-shot *complete-system evaluation* and
*real-to-sim identification* are different claims. Online adaptation can be part
of a declared zero-shot protocol, but real-world manual controller tuning cannot
be omitted from that protocol. The current developed real track provides an
engineering demonstration. It does not become an untouched test by rerunning it.

Without access to the car, publish the available onboard evidence and keep the
unavailable independent metrics explicit. Do not relabel SLAM agreement,
stationary hold, a returned start-line estimate or a selected successful run as
ground-truth odometry accuracy or a statistically established success rate.

## Commands for the next simulation work

The later [forward speed calibration report](../../real_vehicle/docs/calibration/speed_initialization_20261024/README.md)
records an opt-in hardware conversion, separate reference logging and additional
simulation development trials. Motor tracking tests pass; track-level failures,
including Gazebo localization drift and timeout, remain visible. Subsequent
physical follow-ups include initialized-profile failures and
[approved-profile run `211216`](../../real_vehicle/docs/calibration/speed_initialization_20261024/physical_trial_211216/README.md),
which completed two operator-confirmed laps after battery/boundary changes but
retained substantial nominal reference error. Those development outcomes do not
establish calibrated speed tracking, external odometry accuracy or statistical
reliability. The simulation campaign below is still required.

The [later longitudinal-only follow-up `214635`](../../real_vehicle/docs/calibration/speed_initialization_20261024/tracking_followup/README.md)
also reached the two-lap stop, with 4 / 2 reverse starts. Its declared forward
reference was 0.360 m/s; selected lap-2 speed mean / MAE were 0.256 / 0.106 m/s.
The operator reported multiple physical laps, without specifying exact count,
contact or pace. The learned longitudinal inverse remained inactive. This is
onboard-data hardware calibration with fixed neural weights, not untouched
zero-shot full-stack evidence. Final simulator validation, repeatability and
accurate longitudinal tracking remain open.

From the repository root, after preparing ROS 2/Gazebo, dependencies and v7e
weights:

```bash
# Regression suites: run separately to avoid duplicate ROS/pytest module names.
simulation/learning/.venv/bin/python -m pytest -q \
  real_vehicle/ros2_ws/src/apex_race/test \
  real_vehicle/ros2_ws/src/apex_telemetry/test \
  real_vehicle/tools/pc/test_audit_real_run.py
simulation/learning/.venv/bin/python -m pytest -q \
  simulation/ros2_ws/src/apex_fusion_research/test
simulation/learning/.venv/bin/python -m pytest -q \
  simulation/learning/lidar_imu_pose/tests
simulation/learning/.venv/bin/python -m pytest -q \
  real_vehicle/tools/pc/test_race_pi.py

# One final-version Gazebo run, not the complete statistical campaign.
./simulation/tools/sim/apex_real2sim_up.sh \
  --track test_unseen --seed 6 --laps 3 --no-rviz \
  --checkpoint "$PWD/simulation/learning/outputs/real2sim_v7e_standstill/hybrid_submap_v4/best_model.pt" \
  --icp-backend polar --net-backend onnx \
  --run-name frozen_v7e_test_unseen_s6

# Re-evaluate saved run outputs offline.
simulation/learning/.venv/bin/python simulation/tools/analysis/evaluate_real2sim_run.py \
  simulation/data/real2sim/frozen_v7e_test_unseen_s6
```

The [PC SSH workflow in the real-vehicle README](../../real_vehicle/README.md#from-the-pc-ssh-preflight-and-recognition-to-slam-run)
documents physical preflight/run separately. This audit starts no hardware trial.

## Integration validation

| Suite | Result |
| --- | --- |
| Real race, telemetry and offline audit | 267 passed |
| Simulator research package, including existing 2D integration tests | 204 passed |
| Learned odometry, ICP, ONNX, robustness and stationary behavior | 68 passed; 8 exporter warnings |
| PC SSH dispatch, preview and invalid input handling | 8 passed, mocked SSH only |
| Total | **547 passed** |
| Approved and previous ZIPs | CRC, payload sizes and every payload SHA256 verified; both ZIP hashes unchanged |

These are regression results, not fresh Gazebo campaign statistics.
[Machine-readable validation](preprint_validation_20261024.json) records commands,
counts, [test transcripts](validation_20261024/) and shared-source/archive hashes.
Historical README tables retain their
original dates; sealed capture READMEs, generated artifacts, third-party and
legacy documentation are preserved rather than rewritten as new evidence.
