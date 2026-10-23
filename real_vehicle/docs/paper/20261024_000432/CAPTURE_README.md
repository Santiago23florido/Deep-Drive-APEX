# Real-vehicle two-lap execution with RAM telemetry capture

ROS run **20261024_000432**; diagnostic ID **20261024_000431**. The operator
confirmed that the vehicle physically completed both laps and stopped. Contact
with track boundaries was not reported either way. Run IDs use the Pi timezone;
the local operator date in Europe/Paris was October 24. Epoch stamps provide
the shared timestamp basis.

## Observed execution

| Interval | Duration | Reverse starts |
|---|---:|---:|
| Reconnaissance lap | 83.823 s | 13 |
| Second lap, including SLAM handover | 42.773 s | 4 |

Stop reason was `driver_done`, after the second recognized line crossing;
execution through final stopping lasted 126.820 s.
The arm helper's 129.785 s includes initialization and discovery overhead.
The last crossing preceded `done` by 0.224 s.
Thus the recorded final stop was the requested two-lap completion, not a
recorded low-power or stale-command stop. The operator perceived low motor power.

Actual execution passed through reconnaissance, overlap, asynchronous global
planning, SLAM-guided local planning and completion. Map guidance began
91.141 s after exploration start. The global
candidate was rejected: peak curvature 2.457 /m exceeded
1.476 /m and map clearance
0.0976 m was below the configured 0.10 m gate.
The second lap used the observed SLAM route to guide live-LiDAR local arcs;
it did not directly track the rejected optimized raceline. `raceline.csv`
contains that diagnostic candidate, not a claimed executed trajectory.

## What changed and what was preserved

The successful 20261023_222959 controller, parameters, bridge mapping, model
weights, finite limits and protections were preserved. 138 selected runtime
files matched PC/Pi before launch and remained equal after completion. All
archived controller Python core/node files match that successful reference.
No model was retrained. The added change temporarily directed trial output
trees to host `/dev/shm` tmpfs, verified on both host and Docker paths before
arming, with the existing passive sensor recorder and managed resource probe.
Ordinary OS and Docker/ROS console writes were not globally disabled.

The wrapper stopped writers, confirmed neutral, copied and verified the RAM
tree, fsynced it, restored the original paths, then published the new trial
without overwriting prior data. Persistence and verification took
7.916 s
after the motor had stopped. The current 94 published trial files match the
RAM inventory; those files plus the postflight report (95 files) also match
independent hashes read back over SSH. Historical data are preserved.

Both temporary mounts are restored, no Docker container is running, motor
readback is 1500000 ns neutral and the CPU ceiling is restored to 2.4 GHz.
`metadata/final_vehicle_state.json` records the post-trial readback.

## Timing, resources and the slow movement observation

- Driver maximum command publication gap: 42.0 ms, below the unchanged 100 ms
  arm-helper limit. Bridge maximum gap over startup/execution: 71.21 ms;
  bridge timeout samples: 0.
- Maximum control-callback core-lock hold: 22.4 ms. No core-busy stops,
  pose-age stops, stale scans or scan-age stops were recorded. Busy sensor
  callbacks held previous commands; they are not counted as individual stops.
- Hybrid odometry processing median/P95/max:
  14.03/18.43/34.63 ms.
  This covers feature preparation, ICP/network or stationary hold, deskew and
  integration. It excludes waiting for inputs and output publication; it is
  not isolated neural-network inference time.
- Scan stamp to estimate output-handler entry median/P95:
  148.77/164.84 ms, including acquisition,
  waiting and processing. Propagated pose age at control P95:
  21.05 ms. Propagation is not a new independent
  LiDAR correction at every control tick.
- Mean CPU, odometry/driver/recorder/container:
  30.11/
  95.67/
  10.67/
  196.66% of one core on a four-core Pi 5 at a temporary
  2.0 GHz ceiling. Mean host utilization: 50.62% across all four cores.
  Driver CPU is close to one core, so computational headroom in that process
  is limited, although the observed command watchdogs did not trip. RSS sums
  can double-count shared pages; odometry mean RSS was 756.83 MiB.

Command time accounting gives 52.947 s of forward commands, 39.737 s of reverse
commands and 34.137 s of zero/deadband commands across execution. These are
zero-order holds of driver rows with a 0.005 m/s mapping threshold, not measured
physical-motion durations. The second lap included 9.380 s of reverse commands
and 7.297 s of zero/deadband commands. `lap_command_budget.csv` supplies exact
values; rounding and driver-row update times limit boundary precision.

Positive bridge snapshots had a median requested ESC command of
16.146, maximum
21.188, below the 28.2 ceiling
in every snapshot. Median forward PWM readback was
1580.716 microseconds (neutral: 1500).
The bridge uses `15 + 33*u` for positive command `u`; at approximately
0.035 m/s this gives 16.155. This normalized ESC percentage is not percent
electrical power or PWM duty cycle. The displayed legacy minimum of 17 is
not imposed by the active affine profile; 76.35% of positive snapshots were
below that parameter. The current physical breakaway threshold is unmeasured.
The forward ceiling was therefore not the limiting software bound in this run.

Final longitudinal identification accepted only one update, with speed offset
0.1343 m/s and gain 0.8799; overall confidence remained zero. The speed shaper
inverts this offset and adds bounded speed feedback and standstill launch
compensation. This identifies a software mechanism for small forward commands,
but does not isolate the physical reason for weak propulsion. It is not a
well-identified motor model or independent motion calibration.

No live Pi undervoltage/throttle flags were sampled; maximum logged temperature
was 72.5 degrees Celsius. Pi supply flags do not measure motor terminal
voltage or current and cannot establish battery charge or motor power. The
records support low commands and maneuver time as visible contributors to slow
progress. They do not prove that battery weakness was absent, or that all ROS
scheduling effects were absent.

The preceding SD-capture attempt stopped at a 309.55 ms source command gap and
301.5 ms control-lock hold. RAM capture avoided that interruption in this run.
This is an engineering repeat with the same code, not a randomized causal
ablation: initial placement, adaptation, maps and recovery sequences differ.
The actual SD write latency was not instrumented. Full earlier evidence,
including the original successful archive, is included intact under
`previous_sd_capture/real_vehicle_20261023_233947.zip`.

## Data completeness and ground-truth-free diagnostics

Captured 15869 raw IMU samples, 1565 raw 360-beam LiDAR scans and
17434 receipt records. All recorded counts agree with capture metadata, critical
driver/estimator values are finite, timestamps increase, and raw rows have the
expected schema. The recorder is best effort; complete files do not establish
lossless sensor transport. Non-return LiDAR ranges retain `inf` as expected.

During execution there were 12701 IMU and 1255 LiDAR records; IMU header rate
was 100.151 Hz without a gap above 15 ms. LiDAR header rate was
9.895 Hz, with six intervals above 150 ms and a largest interval
of 449.470 ms. These timing anomalies remain in the data and triggered no
recorded stale-data stop. Header age includes the roughly 100 ms sweep; it is
not an isolated ROS queue delay. Captured active estimate count was 1255.

The stationary pre-motion check spans 9.749 s and reports maximum
displacement from its first pose 1.399 mm, with stationary hold active
throughout. This is hybrid-pipeline stationary behavior, not isolated model
accuracy. Scan consistency evaluates 914 consecutive deskewed pairs with
7-neighbor planar PCA, coverage gates and a 0.20 m unmatched-point penalty;
mean score 3.269 cm. Poses and deskew depend on the same estimator.
This is correlated geometric self-consistency, not ATE/RPE, drift or absolute
position error. The identity-pose trace is a diagnostic, not a competitive
odometry baseline. No external ground truth or video was available.

The archive contains saved occupancy and graph vertices, not a full ROS bag
or serialized SLAM factor graph. Logs include learned/propagated estimates,
uncertainty, adaptation, inputs, commands, nominal steering, PWM readbacks,
recovery events, route/global candidate, all finite-trial guards, raw sensor
timing, power/temperature and passive process CPU/RSS samples. Sources, model
weights, configuration, image ID, environment, self-test reports and checksums
support reproduction. These experiments involved previous on-car engineering
adjustments and online adaptation; unchanged model weights alone do not establish
untouched zero-shot transfer of the full stack or statistical reliability.

## English figures and offline reproduction

- `figures/real_vehicle_execution.pdf`: map, estimated trajectory, commands and lap count.
- `figures/slam_map_and_laps.pdf`: reconnaissance and second-lap estimates.
- `figures/slam_handover_detail.pdf`: actual continuous handover.
- `figures/single_run_lap_comparison.pdf`: both laps from this same run.
- `figures/real_odometry_and_timing.pdf`: processing, sensor ages, consistency and stationary stability.
- `figures/raspberry_resources.pdf`: active per-process CPU and RSS.

Each plot has vector PDF/SVG and 600 dpi PNG output. Captions, plotted CSVs,
`paper_metrics.csv`, `paper_timing_table.tex`, full JSON definitions and command
time accounting are included. No car or ROS is required for plotting.

Run `python3 verify_archive.py` before modifying the package. Install the exact
versions in `requirements-analysis.txt`, then run `python3 reproduce_figures.py`.
PDF creation dates may change hashes on regeneration. The packaged geometry
sanity test is `reproducibility/analysis/test_audit_real_run.py`; it checks known
box motion and coordinate-frame invariance. Scripts for original SSH collection
are preserved for provenance and are not needed by the offline reproducer.
