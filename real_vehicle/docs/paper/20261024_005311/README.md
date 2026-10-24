# Retained real-vehicle run: original-like response with full capture

ROS run and diagnostic ID **20261024_005311**. The operator confirmed that
this attempt worked at the desired speed, like the original, and selected it
as the version to retain. The response confirmed the two-lap trial; contact
with track boundaries was not reported. Run IDs use Pi local time; the operator
date was October 24, 2026 (Europe/Paris).

![Recorded real-vehicle execution](figures/real_vehicle_execution.png)

| Trial | Reconnaissance | Second lap | Reverse starts, lap 1 / lap 2 |
|---|---:|---:|---:|
| Original fast run `20261023_222959` | 34.209 s | 14.844 s | 5 / 1 |
| Previous full capture `20261024_000432` | 83.823 s | 42.773 s | 13 / 4 |
| **Retained run `20261024_005311`** | **46.837 s** | **14.594 s** | **8 / 1** |

The recorded median forward command returned to **0.09 m/s**, with median
positive ESC command **17.97**, matching the original run. These are commanded
values, distinct from physical speed. The operator also reported the desired
physical response. Recognition still needed more recovery maneuvers than the
original; the second-lap time was approximately the same. Different initial
placement, online estimates, maps and unmeasured battery state prevent treating
these engineering repeats as a controlled causal ablation.

## Control correction

The Kalman speed identifier continues estimating motor gain and offset. The
controller uses that inverse only when independent longitudinal confidence
reaches 0.5 and residual consistency passes the existing gate. This includes
the existing minimum sample count and requires uncertainty in both parameters
to decrease. Until then, effective gain/offset remain **1/0**, preserving the
initial response used by the original fast run. Steering adaptation and motor
deadband/launch learning continue, with the same PWM/mission limits and guards.

At the end of this trial the raw gain/offset were 0.6/0.0203 with confidence
0.421; they were retained as estimates and were **not applied** to control.
The effective gain/offset remained 1/0. Full raw/applied values and the
application flag are in [the command analysis](motion_effort_metrics.json).
Shared real/SIM controller copies are byte-identical. No neural model weights
were retrained. 221 tests passed, with 3 ROS skips and 3 slow deselections;
the four new regressions also passed using the archived source itself.

The executed second-lap mode was **SLAM-guided local planning**, with one reverse
recovery. The global candidate was rejected; this is not direct tracking of
that rejected candidate. Coordinates and trajectory are onboard estimates.

## Data and English figures

- [Complete portable archive](real_vehicle_20261024_005311.zip), **10,004,574 bytes**,
  with all raw data, source/configuration/model snapshots, operator confirmation,
  tests, maps, commands, PWM, resources and offline analysis.
- [Archive SHA256](real_vehicle_20261024_005311.zip.sha256):
  `cdcda4bf8d42be5b5ab841ccaa8c70922119a268e8fce094cbc143ffacaf2721`.
- [Execution](figures/real_vehicle_execution.pdf),
  [map and laps](figures/slam_map_and_laps.pdf),
  [SLAM handover](figures/slam_handover_detail.pdf),
  [lap comparison](figures/single_run_lap_comparison.pdf),
  [odometry and timing](figures/real_odometry_and_timing.pdf),
  [Raspberry resources](figures/raspberry_resources.pdf).
  All six figures include vector PDF/SVG and 600 dpi PNG with English labels.
- [Metrics](paper_metrics.csv), [LaTeX table](paper_timing_table.tex),
  [comparison with earlier runs](run_comparison.csv),
  [full definitions](real_run_metrics.json),
  [command durations](lap_command_budget.csv) and
  [operator confirmation](operator_feedback.json).
- [Detailed capture report](CAPTURE_README.md); its `raw/`, `metadata/`,
  `figures/` and `reproducibility/` paths refer to the extracted ZIP.

The archive has **305 verified payload files**; ZIP CRC and every entry SHA256
were checked. Standalone SVG/CSV presentation copies normalize whitespace for
Git; the ZIP keeps original exported bytes. Capture includes **9,002 raw IMU
samples and 889 scans**, with schema/count/time/finite-field checks passed.
Four LiDAR intervals exceeded 150 ms and remain in the records. Recording was
best effort, so complete files do not prove lossless sensor transport.

Trial output was recorded on RAM, then copied, SHA-verified and fsynced after
stopping. 95 downloaded files matched the Pi, and all 138 selected runtime
files matched PC/Pi before and after this trial. The maximum command publication
gap was 45.05 ms, with no bridge timeout, stale-data stop or core-busy stop.
The vehicle finished in neutral, Docker stopped, original CPU ceiling restored
and RAM mounts removed. [The source manifest](approved_runtime_manifest.json)
records the retained version. It reuses the exact earlier
[trial helpers](../../../tools/core/validated_trial_20261024_000432/README.md);
their directory name records their first approved use.

## Measurement scope

The [main README](../../../../README.md) now presents the additional timing,
resource and data tables alongside the figures, and documents the
[PC SSH preflight/run entry point](../../../tools/pc/race_pi.sh). The
[preprint readiness report](../../../../docs/reports/preprint_readiness_20261024.md)
separates this approved physical evidence from the remaining simulator campaign.
The acquisition ZIP, captured source and historical test results remain unchanged.

| Additional recorded quantity | Approved run |
| --- | ---: |
| Hybrid processing median / P95 / max | 14.165 / 18.128 / 26.490 ms |
| Scan stamp to output median / P95 | 149.165 / 164.091 ms |
| Propagated pose age at control median / P95 | 15.000 / 23.240 ms |
| Driver / odometry / recorder CPU median | 96.97 / 31.49 / 11.00% of one core |
| Container CPU median | 195.95% of one core |
| Odometry RSS | 756.72 MiB |
| Stationary maximum estimated excursion / observation | 1.866 mm / 9.696 s |
| Scan consistency mean | 0.03520 m, correlated diagnostic |

The stationary figure describes the deployed pipeline with its hold active,
not a separate network-only evaluation.

Hybrid odometry processing includes feature preparation, ICP/network or
stationary hold, deskew and integration. It is not isolated network inference.
Sensor-header age includes acquisition/waiting; propagated pose freshness is
not a new LiDAR correction each tick. CPU 100% means one core; RSS sums can
double-count shared pages. No external ground truth or motor voltage/current
sensor was available. Scan consistency and stationary stability are correlated
diagnostics, not ATE/RPE or absolute position error. Prior engineering changes
and online adaptation do not establish untouched zero-shot evaluation of the
whole stack or statistical reliability.

Use the archived `verify_archive.py` before modifying the package, and
`reproduce_figures.py` with `requirements-analysis.txt` for offline reproduction.
No vehicle or ROS connection is required for the figures. The earlier approved
capture and its tag remain available separately.
