# Real-vehicle evaluation and transfer claims

This protocol separates engineering runs from an evaluation with a frozen
implementation. It is an experimental recommendation, not an IROS acceptance
checklist. Last updated: 2026-10-24 (Europe/Paris).

## Define the claim before collecting the test set

- **Simulation-to-real transfer of a frozen learned model:** preserve the exact
  weights and training provenance. Report any real-world calibration, processing
  changes and controller tuning separately. Unchanged weights alone do not prove
  that the complete autonomous stack was transferred without target-domain tuning.
- **Zero-shot transfer with online adaptation:** specify the adaptation algorithm
  and initial parameters before evaluation, reset them between independent trials,
  and allow only the declared automatic updates within a trial. Do not call this
  "no real-world adaptation." Mapping on lap 1 and retaining estimates on lap 2
  can be part of that declared protocol.
- **Real-to-sim identification:** fitting a simulator to real data is a different
  experiment. If performed later, reserve entire runs for identification and
  other runs for held-out prediction tests. Do not tune with the reported test set.

The current track has been used repeatedly to develop the controller. The
archived runs are therefore development evidence. They are not an untouched
zero-shot benchmark of the complete stack. A new configuration or track tested
after freezing the method would provide stronger generalization evidence.

## What needs the physical car today, in priority order

1. **Freeze and repeat the final method.** Keep source, configuration, model and
   image hashes; start each trial with new odometry, mapping and adaptation state.
   Aim initially for 5 independent starts per condition if time and batteries
   permit. This is a practical target, not a statistically justified universal
   sample size or an IROS requirement. Retain every attempt and stop reason.
2. **Collect raw synchronized evidence.** IMU, LiDAR, sensor header and receipt
   times, commands, applied PWM/status, predicted odometry, SLAM transforms,
   grid, observed route, candidate plan, online parameters, phase transitions,
   recovery events, power flags, CPU temperature, and the exact software.
   The runner now adds passive raw IMU/LiDAR CSVs and receipt timing to its
   existing control/estimation/map logs. These CSVs are not a full ROS bag.
3. **Use an independent reference if available.** A fixed overhead video with
   measured floor markers can provide a reference after camera calibration and
   synchronization. A photograph and measured dimensions support setup reporting
   but do not by themselves provide trajectory ground truth. The owner currently
   cannot film; do not infer reference accuracy or absence of contact from SLAM.
4. **Run a controlled comparison.** After declaring conditions, compare the full
   method with a relevant fixed-parameter controller or adaptation disabled,
   changing one contribution at a time. Use comparable starts, geometry, bounds
   and battery conditions, and alternate the order when possible. Existing runs
   with multiple simultaneous code/battery changes are not controlled ablations.
5. **Test generalization if feasible.** After freezing, use a previously untested
   arrangement, start pose or direction. Log the change. Do not tailor a turn
   schedule to an oval. Multiple starts on one developed track support
   repeatability on that track, not broad track generalization.
6. **Document the platform.** Photograph the chassis and sensor positions; record
   wheelbase, footprint, mass if measurable, nominal steering/PWM mapping, sensor
   extrinsics, surface, track dimensions, battery replacements, and operator
   interventions. Nominal command angles are not measured wheel angles; Pi
   undervoltage flags are not a measurement of motor battery voltage.

## Reportable outcomes

For each trial record physical completion, onboard recognized crossings,
lap durations, reverse count and duration/distance, stops, contacts if observed,
interventions, and failure reason. Report the number attempted as well as the
number completed. Summarize per-trial variation and uncertainty; two laps in one
run are correlated and are not two independent trials.

With the existing onboard logs one can measure timing, commanded/applied
actuation, estimates, map consistency and model-parameter evolution. Absolute
pose/speed error and external tracking accuracy require an independent reference.
For an odometry contribution, an independent reference trajectory is especially
valuable: comparing the estimator with a SLAM map that uses it is not independent
ground truth. For an identification contribution, test predictive error on
held-out command sequences; closed-loop completion alone does not validate all
identified physical parameters.

## Current limitations that must remain visible

The successful 20261023_222959 run recorded two onboard start-line crossings,
five reverse starts in lap 1 and one in lap 2. The candidate optimized line failed
curvature and clearance checks. The executed second-lap mode was **SLAM-guided
local planning**, not direct tracking of that rejected optimized line. Its saved
occupancy grid and graph vertices are not a serialized full SLAM factor graph.
There is no external reference trajectory, video, controlled ablation or
statistical success-rate estimate in that archive.

Two later development runs tested earlier availability of the observed route.
Run 20261023_230328 recorded two crossings but still reversed twice in lap 2.
Run 20261023_231131 enabled the route immediately at the first crossing, still
reversed repeatedly, and was stopped by the operator. These runs do not establish
forward-only second-lap operation. The successful 222959 controller was restored
after stopping; the exact experimental sources remain with their raw runs.

## Work that can wait until after vehicle access ends

Generate English figures and tables from preserved data, audit timing and model
provenance, replay the frozen stack, evaluate simulation independently, compute
uncertainty and write limitations. Real-world baseline or reference measurements
that were never recorded cannot be reconstructed by adjusting the simulator.

## Sources and interpretation

- [IEEE RAS: Information for IROS Associate Editors](https://www.ieee-ras.org/conferences-workshops/financially-co-sponsored/iros/information-for-iros-associate-editors/)
  identifies absence of a novel contribution and significant technical errors as
  grounds for summary rejection. It explicitly does not make real experiments a
  universal prerequisite. No fixed number of runs guarantees acceptance.
- [Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)
  demonstrates simulation-trained policies that adapt without training on the
  physical system. Training, inference-time adaptation and manual development
  should be described separately in our protocol.
- [Tan et al., Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://roboticsproceedings.org/rss14/p10.html)
  explicitly models actuator behavior and latency through system identification.
  This supports measuring the real execution chain, but is not an instruction
  to fit a zero-shot evaluation model to its test runs.

The priorities and trial counts above are recommendations for this project,
not quoted conference rules.
