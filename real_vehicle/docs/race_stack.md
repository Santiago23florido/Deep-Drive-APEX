# The race stack on the real car

Status (October 24, 2026): **two physical laps completed and confirmed by the
operator; the retained configuration is deployed and the vehicle is stopped.**
The preserved run, exact launch helpers, raw evidence and English figures are
documented in [the retained capture](paper/20261024_005311/README.md).
The operator selected this repeat at the desired original-like speed. Its
longitudinal inverse keeps unit gain/zero offset until the independent speed
identifier reaches confidence 0.5 and passes residual consistency checks.
The second lap used SLAM-guided local planning with reverse recovery; the
optimized global candidate was rejected. Later trial notes below record the
development history and do not supersede the approved runtime manifest.

What this is: the autonomy the simulation validated, running on the car.
Learned odometry (the trained LiDAR-inertial network) feeds slam_toolbox, which
builds the map; the race driver explores lap 1 reactively from the LiDAR alone,
plans a race line on its own map, and races it; and underneath all of it the
adaptive layer identifies the car's own plant -- where its axles are, how much
its servo really steers, where its motor starts -- from the odometry, and
inverts it. Nothing in the car is told the track.

The reference is `simulation/data/real2sim/final_s13`: three laps, 13.7 cm of
minimum clearance, plant identified with confidence 0.966.

---

## 1. The graph

```
WT61PC @100 Hz ──/apex/imu/data_raw──┐
A2M8 Express @10 Hz ──/lidar/scan_raw┤
                                     ├─► apex_learned_odometry   (core 1, /odom_rt)
                                     │     ONNX network + polar ICP, 1 thread
                                     │     ├─ /apex/odometry/learned            10 Hz
                                     │     ├─ /apex/odometry/learned_predicted 100 Hz
                                     │     ├─ /apex/learned_odometry/scan_deskewed
                                     │     ├─ /apex/learned_odometry/status
                                     │     └─ TF odom_learned → base_link
                                     │
                  slam_toolbox (core 2) ◄┘  /map, graph, TF map → odom_learned
                                     │
   /apex/tracking/arm ───────────────┤
                       apex_race_driver (core 3)  50 Hz control, 10 Hz reactive
                                     ├─ /apex/cmd_vel_track        ← single owner
                                     ├─ /apex/race_driver/status
                                     └─ planner worker, cores 2-3, nice 10
                                     │
              cmd_vel_to_apex_actuation_node (core 0, profile `affine`)
                                     └─ sysfs PWM ch0 ESC / ch1 servo
```

Everything legacy (`recognition_tour`, `curve_*`, `fixed_map`, the session
manager) is off in race mode, because three of those nodes also publish
`/apex/cmd_vel_track` and the car may have exactly one driver. `apex_race_up.sh`
asserts that before arming is possible.

## 2. Where the code lives, and why it is in three places

| Part | Where | Why there |
|---|---|---|
| Driver, lap-1 planner, race planner, plant identification, adaptive layer | `real_vehicle/ros2_ws/src/apex_race/apex_race/core/` | **byte-identical copies** of the simulation's modules. `test/test_core_parity.py` hashes all thirteen against the simulation tree and fails if one drifts: what Gazebo validated has to be what the car runs. |
| The estimator and the trained model | `~/odom_rt` on the Pi, mounted read-only at `/odom_rt` | It carries ~1 GB of torch, onnxruntime and numba, which will not fit in the image, and it is coupled to the weights: the de-skewing, the metric embedding and the ICP are the network's input pipeline. `sync_odom_rt_to_pi.sh` ships an explicit file list with a sha256 manifest and the git commit. |
| `Path2D`, `path_from_points`, `PurePursuit`, `_forward_backward` | `apex_race/apex_race/vendor/pose_dataset/` | The only four objects of the simulation's pose-dataset package the driving logic may import. The firewall test asserts that the vendored surface is exactly those four. |

The knowledge firewall (`test/test_car_knowledge_firewall.py`) scans every module of
the package for the track definitions, the speed calibration, the ground truth
and the simulation tree. The car has to find the track with its own LiDAR.

## 3. The car's own numbers

`config/vehicle_real.yaml` and `config/race_driver.yaml` hold the base vehicle
parameters. The approved finite trial applies `config/race_oval.yaml` and the
recorded launcher overrides. The relevant current values are:

**Steering in the approved profile: 22.5° left and 12.96° right.** The servo's
duty window is [5.0, 10.0] % about an 8.2 % centre, giving asymmetric nominal
authority. The recorded profile applies these directional limits to planning
and actuation; the base fallback remains ±12.5°. `SteeringServo` clamps
requests to its PWM authority. These are nominal command units, not readings
from a wheel-angle sensor.

**Speed: the bridge is affine now.** The legacy profile stepped to 17 % at any
non-zero command, then applied an exponential ramp, a launch boost and a
ramp-down, so what the car delivered depended on the command *history* -- and
the plant identifier would have measured the bridge's ramp instead of the car.
The `affine` profile is one straight line per direction:

```
pct(u) = 15 % + 33 %/(m/s) · u,   approved forward ceiling 28.2 % → 0.40 m/s command
reverse magnitude = 18 % + 33 %/(m/s) · |u|, ceiling 31.2 % → 0.40 m/s command
```

The 15 % offset sits deliberately *below* the believed 17 % start threshold:
the leftover 0.061 m/s is a residual deadband the identifier brackets from the
launches, and an error on that side leaves the car still rather than lurching.

**Speeds: explore 0.70, race 0.90 m/s.** Below this the adaptive controller
cannot identify anything -- the curvature map needs 0.45 m/s, the lever arm
0.30 m/s -- and the odometry network drops below its 0.3 m/s training floor.
Above it, the stopping distance stops fitting in a one-metre lane.

**`reactive.w_min_m` is a measurement, not a tuning knob.** It is how wide the
lane really is. Set it to the corridor you are driving in.

## 4. What the 2D harness says about the track

The lap-1 reactive driver had never been validated in a corridor of about one
metre. It was measured there before the car was allowed to move
(`race_harness_2d.py`, 7 seeds, three laps each, the measured 17.4 % missing
returns, the plant drawn as it may be):

| corridor | corner radius | steering | result |
|---|---|---|---|
| 1.3 m | 1.5 m | ±12.5° | **6/7 complete, clearance 15-20 cm** |
| 1.1 m | 1.5 m | ±12.5° | 5/7 |
| 1.1 m | 1.2 m | ±12.5° | marginal (4 cm) |
| 1.1 m | 1.2 m | ±18° | 2/2 complete |
| 1.1 m | 1.0 m | either | 0/2 |

The arithmetic behind it, worth keeping: the usable centre band is
`corridor − 0.514 m` (two footprint circles of 0.177 m plus `c_hard` 0.08); the
largest path radius through a corner of centreline radius `Rc` is `Rc + band/2`;
and the steering that needs is `atan(0.30 / that)`.

**Every seed that fails has confidence 0.00 at the moment it fails, and every
seed that completes reaches at least 0.28.** The failures are the plants whose
servo delivers 11-13 % less than commanded: they run wide in a corner before the
identifier has measured them. This is why the motion test is staged -- the first
steps measure the plant on a straight line, not in a corner.

Two configuration defects were found this way and fixed:

* `reactive.lane_margin_m` was 0.5 m, sized for 1.5 m/s. At 0.70 m/s the proven-lane
  rule then demanded 0.81 m of visible lane to cruise, and a corner only ever
  proves about 0.6 m ahead, so **the car stopped in every corner**. Now 0.25 m
  (0.2 s of latency plus one revolution).
* `adaptive.recover_clearance_m` was 0.25 m, which **silently disabled the reverse
  recovery** in a narrow corridor: there are always accumulated points inside the
  0.31 m half-strip behind the car, so the car stood still through 13 s of being
  stuck without ever trying to back out. Now 0.15 m; what makes the reverse safe
  is that it is bounded in distance, time and attempts per lap.

## 4b. The one thing lap 1 cannot do yet

Every harness failure happens with the identifier still at confidence 0, so the
obvious fix is to let the car start a run already knowing itself. It does not,
and neither does the simulation: `PlantIdentifier` begins every run at the
priors and nothing is persisted between runs.

Two thirds of a warm start are available through configuration today, and the
staged motion test is what fills them in: after the straight-line and
figure-of-eight steps, read the identified values off
`/apex/race_driver/status` and write them into `config/vehicle_real.yaml`:

| measured | goes to | what it is the prior of |
|---|---|---|
| `gain_right / gain_left` | `servo.right_ratio` | the right-turn gain |
| `dead_mps` | `esc.min_effective_speed_mps` | the start threshold |

The left-turn gain and the steering trim have no configurable prior, so they
are still learned from scratch on every run. Giving `PlantIdentifier` a
`warm_start()` -- a measured plant loaded as a *prior with its own sigma*, not
as a certainty, so the filter still refines it -- is the obvious next step and
belongs on the simulation side first, where the parity test can police the copy.

## 4c. The blocker: the 5 V supply

Measured on 2026-10-11, and it stops the ladder where it is:

| state | `vcgencmd get_throttled` |
|---|---|
| LiDAR stopped, governor `ondemand` | `0x50000` — historical bits only, the rail is fine |
| LiDAR motor turning | `0x50005` — **under-voltage now, throttled now** |
| LiDAR motor + governor `performance` | the motor never reaches speed; no scan is ever published |

The LiDAR node measures it itself during the soft start. With no other load the
ramp takes 11.1 s and sees under-voltage in 144 polls; with the race stack
starting beside it, 37.0 s and 598 polls. And on the run after that, with the
motor turning and the estimator warming up, **the Pi stopped responding to SSH
and to ping** — the reset at motor start that
`docs/sensor_timing_requirements.md` already describes, reproduced with the
full stack.

What changed in the code because of this:

* the race stack **waits until `/lidar/scan_raw` is actually publishing** before
  it raises the governor or starts the estimator, so the CPU is idle while the
  motor draws its worst current;
* the `performance` governor is **no longer forced**. Pinning four cores at
  2.4 GHz was enough on its own to keep the motor from reaching speed. The Pi 5
  boosts under load by itself; `APEX_SET_GOVERNOR=performance` is available for
  when the supply can take it, and it is restored on exit;
* with the actuation bridge enabled, the stack **refuses to start** on live
  under-voltage. A Pi that browns out mid-run takes the estimator, the SLAM and
  the driver with it, and the car coasts until the bridge's 0.35 s watchdog
  fires.

None of that is a fix. The fix is a 5 V supply that can carry a Pi 5 under load
and the LiDAR motor at the same time, and it is a hard gate for the motion
test. Until then the car is safe to leave alone: the pipeline container is
`restart: "no"`, the LiDAR's boot mode is off and `apex-real-ready.service` is
disabled, so nothing starts the motor by itself after a reset.

## 5. Running it

```bash
# on the workstation
./real_vehicle/tools/core/sync_apex_code_to_pi.sh      # code  -> ~/AiAtonomousRc/APEX
APEX_ODOM_CKPT_DIR=simulation/learning/outputs/real2sim_v7e_standstill/hybrid_submap_v4 \
  ./real_vehicle/tools/core/sync_odom_rt_to_pi.sh      # model -> ~/odom_rt, with a manifest

# on the car
cd ~/AiAtonomousRc/APEX
./tools/core/apex_race_up.sh sensors-only              # C: capture + timing audit
./tools/core/apex_race_up.sh dry                       # D1: everything runs, nothing moves
./tools/core/apex_race_up.sh check --duration 180      #     PASS/FAIL of the whole graph
./tools/core/apex_race_up.sh replay /work/data/replay/final_s13_sensors.npz
                                                       # D2: the full state machine, car off
./tools/core/apex_race_up.sh armed                     # the real thing; announce first
./tools/core/apex_race_up.sh arm                       # in another terminal; Ctrl-C disarms
./tools/core/apex_race_up.sh down
```

`dry` points the driver's commands at `/apex/race_driver/cmd_vel_dry` and does
not start the actuation bridge at all, so the ESC cannot be reached even by
mistake. `replay` additionally replaces the two sensor drivers with a recording,
which is the only way the whole state machine -- explore, closing, planning,
race -- runs on the Pi with the car switched off: a stationary car never leaves
`wait`, because the lap counter needs a closed lap.

## 5b. Losing the car mid-run

The car is on Wi-Fi to a laptop hotspot and it has gone off the network in the
middle of a run. Three things keep that from costing a run:

* **the stack is detached.** `apex_race_up.sh` starts the container with
  `docker compose up -d`, so an SSH session that dies takes nothing with it.
  Every run writes into its own `ros2_ws/apex_race/<timestamp>/`, and
  `apex_race_up.sh status` prints what is running and what that run has written
  so far. It is the first thing to run when the link comes back; it touches
  nothing.
* **`APEX_RACE_PERSIST=1`** gives the container Docker's `unless-stopped`
  policy, so the session comes back by itself after a reboot or a crash,
  spaced out by Docker's own backoff. Off by default, because a stack that
  cannot start must not loop.
* **`apex-net-watchdog`** (`tools/core/install_apex_net_watchdog.sh`) reconnects
  the Wi-Fi and writes one line per drop into `data/net_watchdog.csv`, together
  with the rail state at that moment and whether the stack was still running.

Its link check is deliberately *not* a ping of the gateway: this hotspot does
not answer ICMP, and the first version of the watchdog read that as a dead link
and bounced the Wi-Fi every thirty seconds. What it checks is NetworkManager's
own device state plus an address and a default route.

That log is also how the next disappearance gets diagnosed: a radio that drops
and returns leaves `miss` and `recovered` rows; a car that went down entirely
leaves none, because nothing was running to write them.

## 5c. What the car measured (static replay, 2026-10-11)

Run `20261011_101220`: the recorded sensors of `final_s13` replayed into the
real stack on the Pi, with no motor and nothing able to move.

| | |
|---|---|
| topics | all eight up, including `/map` from slam_toolbox |
| command gap | **48.8 ms** worst (gate: under 100 ms) |
| stale scans dropped | 0 |
| **reactive step** | **145 ms mean, 395 ms worst, over 371 revolutions** |
| everything else holding the core | under 1 ms (odometry 0.5, map 0.1, graph 0.2, control 0.2) |
| consequence | the control tick lost its slot 1860 times; the driver never left `explore` |

The lap-1 planner is the whole budget and it does not fit in the 100 ms a
revolution allows. The candidate grid went from 21x11 to 15x7 because of it --
the 2D harness gives the same six-of-seven seeds and the same 15-20 cm of
clearance either way, so it costs nothing that shows up in how the car drives --
and run `20261011_131021` measured the result:

| grid | reactive step | worst | command gap | control ticks lost |
|---|---|---|---|---|
| 21x11 (231 arcs) | 145.0 ms | 395 ms | 48.8 ms | 1860 |
| 15x7 (105 arcs) | **107.7 ms** | 414 ms | 70.4 ms | 1368 |

A 26 % saving, not the 2.2x the arc count suggests, because the candidate
evaluation is not the whole cost: accumulating the cloud, clustering it,
tracking the walls and building the distance field are paid once per
revolution whatever the grid. **It is still at the budget, not under it**, and
16 of that run's 138 seconds were spent below 2.4 GHz with the rail sagging, so
the true figure is somewhere under 108 ms and has to be re-measured on a healthy
supply before anything else is cut.

If it still does not fit then, the levers left are the fixed costs -- a coarser
voxel for the accumulated cloud, a shorter horizon -- or planning every other
revolution, which at 0.70 m/s means replanning every 14 cm instead of every 7.
That last one changes how the car drives and belongs in the 2D harness first.

The same run also measured the rail: **60 of 112 seconds in live under-voltage
with no motor turning**, on CPU load alone. And `in0_lcrit_alarm` latches -- it
read 0 at idle minutes earlier and 1 in all 112 samples afterwards -- which is
worth knowing because the LiDAR driver's `SupplyGuard` then treated that file
as a live signal, so once it latched the motor ramp always waited its maximum.
That is why the ramp went from 11.1 s to 37.0 s. Section 5m records the later
current-status and cumulative ramp-budget correction.

## 5d. What the dual supply changed, and the drift it left behind (2026-10-11, later)

The supply was the compute problem. With the second feed in, measured at rest
and on load: rail `0x0`, zero under-voltage polls, motor ramp 37.0 s -> 7.0 s,
LiDAR jitter 13-46 ms -> 0.1 ms, 256 valid bins instead of 244. **The reactive
step went from 145 ms to 37 ms mean (75 ms worst)**, so the grid cut in 5c was
answering a brown-out, not a budget: the 21x11 grid of the simulation is
restored. Live sensors then gave 24 of 25 on the bring-up check (estimator
latency p95 160 ms, compute p95 15 ms, command gap p99.9 35 ms, TF complete,
SLAM publishing); the one miss was this document's own governor check, now
informative. What appeared instead is *thermal* throttling (`0xe0000`, no
under-voltage bit), which a fan would answer.

**Bounded motion, authorised, under 30 cm.** The affine map is exact: 0.20 m/s
commanded drove the ESC to 21.6 %, which is 15 + 33 x 0.20 to the digit. The
odometry reported 4.1 cm and the car moved about 4 cm -- confirmed by eye -- so
**the learned odometry follows real motion**. The delivered speed peaked at
0.125 m/s against the 0.138 the believed curve predicted, and two repeats gave
7.4 and 7.7 cm.

**The drift that was left: the car walked away while parked.** 0.42 m/min with
the estimator freshly started, 0.97 m/min after a single 7 cm pulse, 4.78 m/min
after a series -- it grows with motion, not with time standing. The sensors were
clean and the classical ICP called the standstill on 1125 of 1142 revolutions,
so the invented motion was the learned part of the fusion, with `bias_ay` on its
clamp at -1.00000.

That is **fixed twice over, in the simulation tree, and validated there**.

1. **At inference** (`simulation/learning/lidar_imu_pose/standstill.py`, README
   section 15): while the classical ICP confirms a standstill, its increment is
   used and the network is not stepped. Three parked minutes went from 23.8 m of
   drift to 3 mm.
2. **In the weights** (`real2sim_v7e_standstill`, README section 16): the
   network no longer believes the car is moving. Standing still was simply not
   in the training set -- the dataset's recorded standstill is 1.5 s -- and
   neither was the WT61PC's habit of reporting exactly 0.000 rad/s at rest.
   With both in, and with the inference hold **off**, three parked minutes cost
   **0.13 m** instead of 23.8, and driving is unchanged on 108 held-out runs.

The two are independent and both ship. The architecture is v5's, so the Pi
budget is unchanged: 18 ms per scan.

**Both were deployed on 2026-10-22.** `standstill.py`, the v7 checkpoint and its
`.onnx` are in `~/odom_rt`, verified against `MANIFEST.sha256`. The real pipeline
now defaults to v7e and explicitly passes `standstill_hold:=true`; both effective
parameters were checked through ROS on the Pi. The stationary drift gate passed,
but the full bring-up check still failed; see the measurements below.

## 5e. v7 deployment and stationary check (2026-10-22)

Run `20261022_121433` used the live sensors in `dry` mode, with the actuation
bridge disabled. The 120 s stationary check passed **24/25** graded checks.
The one failed check was **two control ticks stopped by `max_scan_age_s`**.

| measurement | result |
|---|---|
| parked drift over the check | below 1 mm (reported as 0.000 m) |
| whole estimator log | 0.70 mm net drift, 1.51 mm maximum excursion; hold on 1482/1484 scans |
| estimator latency / compute p95 | 185 ms / 6 ms |
| command gap p99.9 / maximum | 38 ms / 47 ms |
| sensor rates | IMU 100.15 Hz; LiDAR median-arrival rate 10.21 Hz |
| longest LiDAR arrival gap | 465 ms; estimator stamp gap reached 500 ms |
| rail / maximum temperature | `0x50000` throughout (historical bits only), 68.6 °C |
| free disk | 1.4 GB |

There are additional unresolved findings that the checker does not grade. While
parked, the driver counted two fictitious lap crossings (6.44 m and 10.25 m of
accumulated travel) and entered `planning`; its adaptation status also latched
`sign_fault=true`. These cannot be treated as real laps or proof of an IMU axis
error. The relationship between the stable scan-stamped pose, the predicted pose
and the driver's accumulated travel needs investigation. The LiDAR log also
ended with 166 checksum errors and 205 lost capsules; their cause is unconfirmed.
Motion is not cleared until these findings are resolved.

Evidence is on the Pi under
`~/AiAtonomousRc/APEX/ros2_ws/apex_race/20261022_121433/`, including
`stationary_check.json`, `estimator.csv`, `power.csv`, `container.log`,
`race_events.json` and `odom_manifest.sha256`. Copies are in the workstation's
ignored `.codex/validation/20261022_121433/`. The stack was stopped after the
check; `apex-real-ready` remains disabled and inactive. No drive or steering
test was executed.

## 5f. Fictitious travel diagnosed and corrected (2026-10-22)

The v7 scan-stamped estimate was stationary; the 100 Hz prediction was not.
`LiveOdometry.predict()` kept integrating the IMU after the scan-side hold had
set velocity to zero. Accelerometer tilt and residual bias produced a sawtooth
between scan corrections: the first run's driver reported -0.054 to -0.276 m/s,
accumulated **10.881 m** and counted two fictitious laps. The earlier simulation
standstill tests measured scan increments and missed this prediction path.

The predictor now shares the committed scan hold and publishes zero velocity
and yaw rate with the latest IMU stamp while held. The controller no longer
adds position corrections to travelled distance when observed velocity is zero.
Separately, the identifier used to vote for reversed steering before checking
measured speed: parked pose noise could latch `sign_fault`. The existing
steering-speed gate now precedes sign voting and saturation learning. Both core
changes were made in the simulation first and copied byte-for-byte to the car.
The checker now grades predicted speed and lap count at rest and steering sign.

Pi run `20261022_125732`, live sensors, v7e/polar/ONNX, `dry`, 120 s:

| measurement | result |
|---|---|
| bring-up checker | **28/28 PASS** |
| predicted speed | p99.9 and maximum **0.0000 m/s** |
| lap counter / phase | **0 laps**, stayed in `explore` |
| steering sign / scan-age stops | **false / 0** |
| parked net drift | below 1 mm (reported 0.000 m) |
| estimator latency / compute p95 | 151 ms / 6 ms |
| command gap p99.9 / maximum | 42 ms / 44 ms |
| rail / maximum temperature | `0x50000` throughout (historical bits only), 74.7 °C |

The driver retained about 0.030 m of startup travel from before the hold was
confirmed; that value was already present before the 120 s check and did not
continue growing. Local regressions also passed: tilted-IMU prediction at rest,
prediction resuming during recorded motion, stationary sign voting, detection
of actual reversed steering in the plant test, forward/reverse distance, and
the parity checks (88 distinct targeted tests across the selected suites).

**LiDAR transport remains unresolved.** Two separate 45 s captures read the
real `/dev/ttyUSB0` directly, without ROS, SLAM, estimation or a race driver.
The regulated run had **27 checksum errors / 42 lost capsules**; the fixed-PWM
run had **30 / 45**. Corruption appeared around 19 s and again around 37 s in
both. The saved bytes contain checksum-invalid packets and discontinuities;
this is already present in the serial input, before the estimator. Motor PWM
updates are not required to reproduce it. There were no competing serial
owners, USB autosuspend was off, and the kernel log showed no USB reset during
these runs. The origin within the LiDAR, adapter, wiring, power or USB receive
path has not been isolated; do not label it a cable or supply fault yet.

The owner then disconnected and reconnected USB in the same port. Repeating
the regulated 45 s capture produced **51 checksum errors / 41 lost capsules**,
again in bursts near 19 and 38 s. The same-port reconnect did not clear the
fault. Evidence is in `data/diagnostics/20261022_lidar_reconnected_same_port/`
on the Pi and `.codex/validation/20261022_lidar_reconnected_same_port/` locally.

The subsequent USB port change (`1-2` to `3-1`, confirmed in sysfs) yielded
**6 checksum errors / 8 lost capsules** in the same regulated 45 s capture,
again near 19 and 38 s. This sample improved on 51/41 but still fails serial
integrity; it does not establish the cause of the difference. Evidence is in
`data/diagnostics/20261022_lidar_changed_port/` on the Pi and
`.codex/validation/20261022_lidar_changed_port/` locally.

The 28/28 checker result covers its ROS-level checks, not serial checksum
integrity: the full run still had scan arrival gaps of 463 ms, and its final
LiDAR timing log reported 73 checksum errors and 105 lost capsules. These
counters alone do not decide whether the sensor gate passes: the existing
requirement permits up to 2 % lost whole revolutions. Section 5h evaluates
that requirement; all remaining sensor and motion gates still apply.
Raw capture evidence and summaries are under
`data/diagnostics/20261022_movement_lidar/` on the Pi, copied locally to
`.codex/validation/20261022_movement_lidar/`. The corrected run artifacts are
under `ros2_ws/apex_race/20261022_125732/` on the Pi and
`.codex/validation/20261022_125732/` locally. The stack was stopped afterwards;
no drive or steering test was executed.

## 5g. Software-only transport diagnosis (2026-10-22)

The owner cannot make further hardware changes. A binary usbmon capture on the
current port observed zero monitor drops and no active bulk-IN transfer errors.
The 466,433 bytes read from the serial port exactly match a contiguous region
of the USB payload. Invalid packets therefore already exist before the Linux
serial driver/Python receive path; the application parser did not change those
bytes. Its XOR checksum also matches the official Slamtec unpacker algorithm.

Neither a documented software RESET nor requesting the advertised Express mode
1 explicitly cleared the errors. The firmware advertises Standard (500 us,
answer 0x81), Express (250 us, 0x82) and Boost (125 us, 0x84). Controlled raw
captures of Express and Boost gave:

| | Express | Boost |
|---|---|---|
| nominal data requirement | 10,500 bytes/s (84 bytes / 32 samples, 4 kHz) | 11,000 bytes/s (132 / 96, 8 kHz) |
| measured output | 10,472.9 bytes/s | 10,472.3 bytes/s |
| fitted interval between error bursts | 18.748844 s | 0.970815 s |

**The measurements strongly support a throughput ceiling and buffer overflow
before USB.** Assuming the same queue in both modes, the two error periods
independently imply a capacity of 10,472.696 bytes/s and a 511.915-byte queue,
matching the byte-rate measurement and a 512-byte buffer. This is an inference
from controlled load changes, not a direct read of an internal overflow flag.
The exact location within the LiDAR transmitter/adapter path is still unknown;
these results do not diagnose a defective cable. CP210x register readback
confirmed the requested 115200 baud and 8N1. Software RESET, extended Express
and Boost were diagnostic only; production configuration is unchanged.

Reducing data demand is a possible diagnostic option (Standard advertises
2 kHz, 10,000 bytes/s), not the next production change. The existing timing
requirements document records insufficient spatial coverage for ICP in that
mode. Section 5h instead evaluates continuing with Express under the existing
loss allowance. Standard has a different integrity mechanism: simply losing
the Express checksum counter would not establish that it is sound.
Evidence, including the USB/serial comparison, mode query, raw captures and
queue fit, is in `.codex/validation/20261022_lidar_software/` locally and
`APEX/data/diagnostics/20261022_lidar_software/` on the Pi. All diagnostic
containers stopped; the temporary usbmon module was unloaded. No actuation
was enabled.

## 5h. Continuing within the existing loss allowance (2026-10-22)

The owner asked whether work can continue while accepting the lost data.
**Yes, continue the no-movement validation with Express unchanged.** Treating
every checksum error as a blocking failure was stricter than the implemented
requirement: `lidar_bringup_check.py` accepts at most **2 % missing whole
revolutions**. The parser rejects damaged capsules and the assembler discards
affected revolutions; it does not publish their surviving fragments as complete
scans. The simulation's `APEX_real` profile includes 1 % whole-scan dropouts.

Offline analysis used existing real recordings, with no device opened and no
additional live run. It applied the checker's median-period / rounded missing
interval formula to production-pipeline outputs from the saved byte chunks:

| Evidence | Inferred lost revolutions | Loss fraction | Longest stamp gap |
|---|---|---|---|
| Full-stack corrected run `20261022_125732`, estimator stamps | 17 / 1455 expected intervals | 1.17 % | 450 ms |
| Raw capture after the USB port change | 2 / 444 | 0.45 % | 204 ms |
| Raw capture with USB monitoring | 6 / 445 | 1.35 % | 450 ms |
| Raw capture after software RESET | 6 / 444 | 1.35 % | 403 ms |
| Raw capture with explicit Express mode 1 | 7 / 444 | 1.58 % | 451 ms |

The full-stack row is a downstream estimate, not a separate raw-sensor check;
its log extends beyond the 120 s stationary checker window. The fractions
count missing revolutions, not checksum attempts or missing capsules. Each
listed sample is within the existing loss allowance. This clears continued
investigation of timing, estimator replay and SLAM without eliminating every
serial error; it does not declare the whole sensor ladder passed.

Bursts still matter: some intervals imply up to four missing revolutions. The
live estimator's IMU encoder retains at most 40 samples per scan interval;
the corrected run reached that cap at a 450 ms gap. Training with 1 % random
dropouts does not by itself validate every such burst during motion. Preserve
the current `max_scan_age_s: 0.6`, pose-age and command-watchdog gates, and
require zero scan-age stops for the relevant live acceptance check. The
corrected run achieved that; the earlier run had two stops. Do not raise the
thresholds or pass corrupt/partial scans to conceal losses.

The next steps remain C (joint sensor capture and timing audit), R1 (estimator
replay agreement) and R2 (stationary SLAM stability). Movement still requires
the owner's explicit confirmation and the remaining motion gates. Evidence:
`.codex/validation/20261022_lidar_software/loss_tolerance_analysis.json` and the
corrected-run artifacts above. No production code or parameters changed for
this decision; the car remained stopped.

## 5i. Joint capture and estimator replay with accepted losses (2026-10-22)

Run **`20261022_135930`** used the existing dry configuration: actual Express
LiDAR and WT61PC, v7e, polar ICP, ONNX and standstill hold. The actuation bridge
was disabled; `/apex/cmd_vel_track` did not exist. An observer subscribed before
estimator startup, retaining startup sensors plus a **120 s graded window**
after the full stack was ready. This single live session supplies both the
sensor audit and the online/offline estimator comparison.

| Check | Result |
|---|---|
| Sensor timing audit, `APEX_real_wt61` | **8/8 PASS**; IMU 100.15 Hz, LiDAR 10.00 Hz, median 10 IMU samples/scan |
| Strict LiDAR check | **9/10**; loss 1.49 % passes the <=2 % criterion |
| LiDAR end-of-sweep to observer arrival | p5/p50/p95 **22.4/36.3/74.6 ms**; p95 fails the <50 ms criterion |
| Longest raw-scan stamp gap in the graded window | 454 ms |
| Estimator replay, R1 | **PASS**, 1310 matched scans, maximum position difference **0.034 mm**, yaw 0.000074 deg; no fitted pose offset |
| Stationary behavior in the graded window | zero predicted speed in the driver log, zero additional travel, no crossings, `sign_fault=false`; net scan-pose drift 0.26 mm |
| Online map -> odom observation | 6004 transforms in the graded window; zero accumulated translation change |
| Freshness protection across the whole run | **6 control ticks stopped by scan age**; this is not six independently counted outages |
| Pi power and temperature log | `0x50000` throughout (historical bits only), maximum 71.9 C |

The capture contains 16825 IMU messages and 1544 raw scans; 1188 scans were
observed during grading. R1 starts at the recorded scan preceding the first
online estimate and uses the same checkpoint, profile and inference backends.
The online log continues after the observer finishes, so those trailing online
estimates are outside the replay comparison. The existing recorder rounds
ranges to four decimals and IMU values to six; the observed difference remains
well below the 1 mm gate.

**Step C is not cleared:** the missing-revolution fraction is acceptable, but
the delivery-latency criterion fails. The static pose and map observations do
not remove the freshness failure or replace formal R2. Next investigate timing
and recovery around serial-loss bursts using these recordings; preserve the
current age/watchdog thresholds and do not alter stamps merely to pass the
check. No additional hardware change is required to inspect this evidence.

Evidence on the Pi is `data/diagnostics/20261022_sensor_ladder/capture/`, with
the online run also under `ros2_ws/apex_race/20261022_135930/`. Local copies,
the replay input and reports are in
`.codex/validation/20261022_sensor_ladder/capture/`. The diagnostic observer and
replay scripts are adjacent; they changed no production sources or parameters.
The stack was stopped and both serial ports released. No movement was run.

## 5j. Burst frequency and a clock-recovery candidate (2026-10-22)

The graded window of `20261022_135930` contains **7 raw-scan gaps over 150 ms**,
spaced **18.58--18.77 s** apart (mean **18.68 s**). Gaps range from **203 to
454 ms**. The 74.6 ms value in section 5i is an arrival-latency p95, not a
constant delay or the duration of those gaps.

Reconstructing estimator completion times from scan stamps and logged latency
finds one interval in the graded window that exceeds the 600 ms freshness
limit. It begins at graded time 113.39 s: a 454.05 ms scan gap plus 266.20 ms
latency for the following estimate leaves the prior correction 720.25 ms old.
The excess is 120.25 ms, consistent with six nominal 50 Hz control ticks.
This supports one brief stop episode, not six independently counted outages;
exact driver stop transitions were not recorded. The following estimate's
compute time was only 13.73 ms, so its 266.20 ms latency is not explained by
network/ICP compute time alone.

One software candidate is faster recovery of the LiDAR clock mapping after
loss. The current envelope uses a 10 s window; in saved raw captures, some
bursts are followed by several seconds of elevated apparent latency. Replaying
five unchanged byte recordings with their original host arrival times compared
that mapping with **3 s / 6 chunks / 0.25 s refits**, keeping max drift 0.03:

- Four captures with baseline arrival-latency p95 of 65--74 ms decreased to
  45--47.5 ms. A fifth already near 44 ms stayed near 44 ms.
- Published revolutions and their sample indices were identical. Stamps stayed
  increasing, rates near 10 Hz and stamp-interval p5/p95 within 90--110 ms.
- Worst individual apparent delays still reached 81--104 ms. The experiment
  does not establish that the long-gap stop would disappear.

**This is a candidate, not a deployed fix.** These raw recordings have no
independent sample-time ground truth; a lower reported latency alone does not
prove more accurate timing or correct alignment with the IMU. Validate clock
accuracy/recovery and then the live arrival and freshness gates before adopting
it. Apply any change to the LiDAR mapper only, preserve whole-revolution drops
and all existing stop thresholds. No production source or parameter changed,
no extra live run was made, and no movement was commanded for this analysis.

Evidence: `.codex/validation/20261022_sensor_ladder/capture/repetition_analysis.json`
and `.codex/validation/20261022_sensor_ladder/clock_recovery_experiment.json`.

## 5k. LiDAR clock recovery implemented and validated (2026-10-22)

The owner authorized implementation if the candidate works. The default
`ExpressScanPipeline` mapper now uses **3 s / 6 chunks / 0.25 s refits** with
the existing max drift 0.03. This change is confined to LiDAR clock fitting;
the IMU mapper, scan mode, checkpoint, sample-index reconstruction,
whole-revolution discard policy and all freshness/watchdog thresholds stay
unchanged. Explicitly injected mappers remain supported.

All **29 sensor-timing tests pass**. A new test uses independently known sample
times, a 0.5 % oscillator-rate change and a damaged packet: the old mapping
retained up to 14.7 ms of calibrated timing error after four seconds; the new
mapping is within 0.5 ms. Shorter envelope windows have fewer minimum-latency
observations, so the lost-capsule test now grades the new default against
0.5 ms rather than 0.3 ms, below the profile's 1 ms timestamp jitter. The test
still checks the injected legacy mapper against its original 0.3 ms bound.
Five existing real raw captures also preserve exactly the published sample
indices and pass monotonicity/rate checks, with apparent arrival p95 below
50 ms. These offline results were followed by the real validation below.

**Pi run `20261022_160927` passes the targeted acceptance checks.** The full
live stack ran in dry mode, with actuation disabled, for a 120 s graded window:

| Measurement | Before (`20261022_135930`) | Implemented |
|---|---|---|
| Strict LiDAR check | 9/10 | **10/10** |
| End-of-sweep arrival p95 | 74.6 ms | **47.0 ms** |
| Scan-age stop control ticks, whole run | 6 | **0** |
| Lost whole revolutions | 1.49 % | **1.41 %**, within 2 % |
| Estimator latency p95 | 178.8 ms | **151.9 ms** |
| Estimator replay max position difference | 0.034 mm | **0.019 mm**, 1406 matched scans |

The sensor timing audit passes **8/8**. The new run has zero additional parked
travel, zero reported speed, net scan-pose drift 0.39 mm, and constant
map->odom translation during grading. Power flags remain `0x50000` (historical
bits only), logged maximum temperature 70.8 C. Raw gaps still reach 452 ms:
the change improves clock recovery and does not eliminate transport loss.
This is one real stationary acceptance run, not evidence of performance during
motion or a replacement for R2, S, I or D2.

The deployed source hash is
`ee203ec5129d012389e0eae4303c16b872c153400a2c3f019346496f8a7eef80`.
Only `ros2_ws/src/apex_telemetry/apex_telemetry/perception/rplidar_express.py`
was deployed. A rollback copy is in
`data/diagnostics/20261022_lidar_clock_recovery/rplidar_express.before.py` on
the Pi. Evidence is in that directory's `capture/` (also the online logs under
`ros2_ws/apex_race/20261022_160927/`) and locally under
`.codex/validation/20261022_lidar_clock_recovery/`.

**The adjustment is retained on the Pi. C and R1 now pass for this stationary
configuration.** The stack was stopped after validation, no movement occurred,
and no model was retrained. Continue with the remaining ladder without
repeating these checks unless another change or concern justifies it.

## 5l. Stationary SLAM replay on the Pi (2026-10-22)

R2's stationary stability check now passes on the actual **Pi arm64 image,
slam_toolbox 2.8.5**, using the existing `20261022_160927` capture. No additional
live capture was made. The replay container had no device or PWM mounts and
used a separate ROS domain. It used the deployed SLAM parameters, with replay
clock and queue settings only; the physical stack remained stopped.

The replay covers **141.8 s**: its observer received **1406/1406 input scans**,
recorded **1674 map->odom transforms**, and received **67 maps** with identical
geometry and occupancy hashes. Accumulated map->odom translation and rotation
were **zero**, inside the 5 cm limit. The SLAM log has no warnings or errors.
The live capture had already recorded a constant map->odom over its 120 s
graded window.

**The original scan-count criterion needed a definition correction.** `/pose`
counts accepted map updates, not received input. With
`minimum_travel_distance: 0.2`, the
[SLAM motion filter](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/src/slam_toolbox_common.cpp)
intentionally rejects further updates while this car is parked: the replay's
position span is only 1.34 mm. One update is eligible and one is produced.
Requiring 90 % of all raw scans to produce poses would be incompatible with
this deployed filter and the 0.2 s minimum interval. The table below separates
observed delivery, motion-eligible updates and transform stability; no runtime
filter or safety threshold was changed to pass it. Observer receipt does not
instrument SLAM's internal subscription callbacks. This stationary result
does not validate moving scan matching or loop closure.

`simulation/tools/analysis/replay_slam.py` now records observed scans,
map->odom and map-content metadata in addition to `/pose`, and preserves every
replay's log. This diagnostic change was exercised first locally (2.8.3) and
then on the Pi (2.8.5), because version parity matters. The final Pi artifacts
are `capture/slam_replay/pi_clock3s_static_arm64.npz`, its `.log`, and
`static_check.json` under the clock-recovery evidence directories in section
5k. SLAM configuration on the Pi is unchanged.

The full driver/planner D2 replay still remains available using the existing
`data/replay/final_s13_sensors.npz` fixture on the Pi. The owner subsequently
requested a bounded autonomous trial on a short open segment, replacing the
earlier 20 cm nudge proposal; see section 5m. This does not clear S, I, moving
SLAM or the full-lap validation. The longer sequence in section 7 remains
conditional on suitable space and authorization.

## 5m. Bounded autonomous segment and continuous clearance (2026-10-22)

The owner has approximately **1.5 m straight, 0.9 m width**, a left curve
followed by a right curve, and an initial heading that is not aligned with the
straight. He will not reposition the car manually or make further hardware
changes. He requested the full conductor / SLAM / adaptive control stack,
then explicitly confirmed the bounded actuator trial after its limits were
presented. This is authorization for that trial, not unrestricted driving.

`config/race_segment.yaml` selects the same real driver, with exploration
target 0.20 m/s, forward-only commands capped **after adaptive feedback** at
0.25 m/s, no recovery/reversing, and a latched stop at **1.0 m accumulated
odometry path or 5.0 s**, whichever is reached first. Identification receives
the bounded command actually published. The mission feature is disabled in
the normal race configuration. Distance is an odometry estimate, not an
independently measured physical bound; the timer and existing stale-pose,
stale-scan, arm and bridge watchdogs remain in effect. A separate finite arm
publisher on the Pi has an 8 s wall-clock timeout including the static start.
At this speed the identifier's existing 0.30/0.45 m/s update gates mean the
trial cannot establish full plant calibration or complete laps.

The first live dry preview (`20261022_164216`) found zero feasible paths and
published zero speed. Reconstructing its actual scans isolated a clearance
calculation defect: flooring both points and queries onto a 5 cm grid made a
near-parallel wall jump a whole cell closer at the next path sample. This
defeated the existing relative-clearance rule for an already squeezed start.
`LocalGrid` now queries continuous nearest-point distances within the same
local bounds and retains the 2.5 cm sampling allowance. Footprint, hard margin,
stopping-distance rule, steering authority and safety timeouts are unchanged.
The same recorded cloud changes from **0 to 121 feasible candidates**, with
a left-turning exit and 0.8 m free along the chosen path. This establishes an
initial exit, not that both curves can be completed.

The simulation and car core copies match byte for byte. **135 targeted tests**
pass, including a distance oracle across cell boundaries, escape from a
slanted close wall, frontal obstruction, bounded commands and latched stops.
The full non-slow simulation set passes **179 tests, 2 deselected**. No model
or training data changed.

The corrected live dry run (`20261022_174617`, bridge disabled) had 121
feasible candidates, left steering, commands within 0..0.25 m/s and a latched
time stop after 5.020 s. The strict stationary check was **28/29**, not a full
pass: one stale-scan stop tick occurred about 8 s after DONE. The corresponding
estimator gap was 404 ms with 240 ms completion latency, making the previous
correction about 643 ms old against the unchanged 600 ms gate. Command gap
was at most 36.4 ms; no core-busy stop or sign fault occurred. Transport loss
therefore still occasionally invokes the stop protection.

The actuator preflight `20261022_175158` refused arming (31/33): Docker Compose
was not forwarding `APEX_BRIDGE_MAX_STEERING_DEG`, so the bridge retained an
18 degree clamp despite a 12.5 degree launcher request. Compose now forwards
that variable and `APEX_BRIDGE_PROFILE`; readback in `20261022_175532`
confirmed **12.5 degrees <= 12.96 degrees servo authority**. That second
preflight remained disarmed (32/33) because a historical command gap of
339 ms still exceeded 100 ms; its observed check window had a maximum gap of
26 ms. Increasing estimator startup settling from 10 to 25 s did not clear
that historical gap, so it must not be presented as a proven fix. The next
preparation omits a full parameter dump, verifies the necessary parameters in
one service call and records status before and after it to locate the pause.

Evidence and the finite trial runner are under
`.codex/validation/20261022_open_segment/` locally and
`APEX/data/diagnostics/20261022_open_segment/` on the Pi. Each attempt saves
its own preflight, runtime limits and logs. A failed preflight does not publish
arm=true. No locomotion occurred in the two refused preparations above.

The confirmed trial **`20261022_175815`** then performed real locomotion after
a **33/33** preflight. The owner confirms advance and stop without touching
borders, but too little steering for the curve. Odometry reports **0.662 m net
displacement**, **0.715 m accumulated mission path** and **7.18 degrees left**.
Motor output stayed within 0..23.25 %, with no reverse sequence. The planner
lost all feasible forward candidates about **1.84 s** into exploration and
commanded zero, followed by coasting. There were **0 stale-scan, stale-pose or
core-busy stops and no sign fault**. Final held scan velocity was exactly zero;
peak temperature 73 C, rail flags historical-only `0x50000`.

This was **not a live pass of the mission DONE latch**: the last core tick was
at 4.961 s of exploration before the finite arm expired, leaving EXPLORE and
disarmed. This run stopped by blocked-path control and disarm. Also, after
preflight but before arm, bridge telemetry reported a **276.97 ms** historical
receive gap. Its high-water value did not increase during captured motion;
driver publication maximum was 36.14 ms. This does not establish <100 ms
bridge delay for the entire run. The runner now rechecks that counter
immediately before arm and refuses such a condition. The association between
parameter/CLI inspection and preparation pauses still needs isolation.

The bridge applied the requested steering, but the request fell to **3--4
degrees** during much of the advance. Real-input replay also found pursuit
targets beyond the planner's collision-checked free prefix. Both reactive
planners now return only that prefix, and pursuit holds its endpoint rather
than extrapolating the last tangent. On one actual frame the target changes
from 5.83 to 8.09 degrees; that local comparison does not predict a successful
complete curve. **213 tests pass (181 non-slow simulation plus 32 parity), 2
slow tests deselected**. Hard margins, steering caps and stop gates remain.
The deployment hash for that version of `reactive.py` was
`17c36d5679583b0f89e0ac93107eec5d23a67d90f92966d79438e0ebbe6db9dc`.
At 12.5 degrees and 0.30 m wheelbase, nominal minimum rear-axle radius is
about 1.35 m. The position may require a separate alignment maneuver with the
actuators; increasing steering alone does not prove a forward-only exit.
The first trial's confirmation does not authorize a second locomotion.

Final deployed-code validation at the new parked position, bridge disabled:
**`20261022_181114` passed 29/29**, with zero scan-age stops and the mission
timer latched DONE at 5.119 s. Fresh parked scans found some forward candidates
(6--15 after initial settling), but no complete curve was validated. No second
locomotion occurred. Another physical trial requires verifying the curve
route or an alignment maneuver if needed, closing the bridge timing concern,
and confirming movement with the owner.

The owner subsequently confirmed that he returned the car to its starting
position after the physical trial. Later parked captures are at the start,
not the stopping point. Within the moving run, front ranges decreased from
about 1.13 to 0.46 m, independently supporting the reported forward motion.

**Additional steering freedom for this low-speed segment.** Confidence zero
previously limited steering to about 10.24 degrees and added 0.15 m of
lookahead. The segment preset now uses `steer_plan_deg: 12.5`,
`adaptive.sat_cmd_frac: 1.0`, `reactive.local_path_lookahead_m: 0.45`, and zero
confidence lookahead additions. Actual estimated confidence is unchanged;
the speed, distance, duration, obstacle margins and safety timeouts are kept.

Merely increasing the allowance did not ensure a turning reference: the
wall-relative planner had priority and chose nearly straight on the new
actual cloud. The same cloud through the existing bicycle-arcs planner gives
about 10.14 degrees left, with 0.8 m checked free. Added
`reactive.use_wall_planner` defaults to true; only the segment preset sets it
false. Wall tracking and lane/collision checks still run. No steering-cost
retuning or wall-origin change was applied. **216 tests pass**, including
zero-confidence full authority, a short first arc followed by an opposite arc,
and obstacle-limited corner planning. Two slow tests are deselected.

The cap-only dry run `20261022_182511` was **28/29**, with an observer command
gap up to 336 ms versus 33.74 ms maximum recorded by the driver. It had no
scan-age stops and a 5.00004 s mission stop. The arcs-profile dry run
`20261022_184924` calculated up to **12.146 degrees left**, commands <=0.25 m/s
and a 5.060 s mission stop. Its additional diagnostic observer crashed on
ROS's dictionary message-info format, so no formal checker result exists for
that attempt; it must not be marked passed. A later dry check uses the correct
mapping API and records source, reception and callback times. Current source
hashes: race_config
`ea44eaecab6574fa5358cbd5d1d4eeb8f9926c5968591f9df197aa659bf37558`, reactive
`29ea8ce057a533fbba76331b9776f820f5408d152eda19a5dfe09ae83cfac288`.

Recorded DDS timings in `20261022_185446` distinguish publication from
delivery: source maximum gap **31.60 ms**, reception **337.32 ms**, with two
delivery pauses in the observer's first 1.6 s. After 2 s the receive maximum
was **31.58 ms**. `race_bringup_check.py` now explicitly spins for
`--settle-s 2` before starting its graded window; JSON records this. It clears
only observer samples, retaining historical producer/bridge fault counters
and the 100 ms threshold. Three on-Pi arithmetic regressions using the actual
capture verify startup isolation, fault retention and rejection of a later
320 ms gap. This does not clear the separate historical bridge concern above.

Final validation **`20261022_190233`** uses the shipping arm gate (`require_arm:
true`), live sensors, the verified dry command topic and **bridge disabled**.
Preflight WAIT passes **29/29**. A finite 9 s logical arm, including ROS
discovery/static-start overhead, exercises the actual mission: DONE at
**5.0048 s**, all subsequent commands zero, no stale-scan/pose/core-busy stops
or sign fault, global driver publication gap <=34.66 ms, confidence still zero.
This particular cloud selected up to 4.403 degrees; the preceding dry run
demonstrated the available 12.5 degrees. Allowed authority is not a fixed
steering command or proof of a complete curve. The validation helper's final
explicit false-arm CLI exceeded its 3 s cleanup limit after the finite arm
publisher exited; the finally block stopped the stack and saved DONE evidence.
The helper cleanup limit/catch was corrected. No second locomotion occurred.

The settled start cloud also isolated a remaining preference for deferred
steering in arcs mode. For this segment only, `reactive.w_smooth` is now
**0.1 instead of 0.3**. Actual recorded LiDAR replay in the deployed **Pi
arm64 image** changes the selected steering from **2.560 to 10.139 degrees**,
with checked free distance **0.7 -> 0.8 m**, confidence zero and permitted
authority 12.5 degrees. Margins and stop rules are unchanged. A tracked fixture
from the real `190233` input prevents recurrence; **53 focused tests pass**
after the final preset change. The complete 216-test run above preceded this
last weight change. The trial runner verifies this parameter at runtime.
This recorded-input result requires a new owner-confirmed physical trial;
it is not a measured new turn or proof of full-curve completion.

**Second owner-authorized trial, `20261022_193341`.** The owner reports that
the wheels turned in the correct direction and the car tried to advance but
lacked force. The bridge reports up to **12.5 degrees applied** (PWM-derived,
not an independent wheel-angle measurement). Mission DONE latched on duration
at **5.0407 s**, with only **0.0100 m accumulated odometry path** and zero
estimated speed; that small path is not evidence of useful physical progress.
Commands stayed forward-only and <=0.25 m/s, with zero stale-LiDAR stops or
steering-sign fault. This is **not a successful driving/curve validation**:
there were **60 core-busy stop ticks and 22 pose-age stop ticks**. Scan-lock
holding peaked at 166.4 ms and the last reactive step was about 122 ms.

The 33/33 preflight passed, but the bridge subsequently recorded **164.14 ms**
maximum callback gap, while the driver maximum publication gap was 51.56 ms.
The new bridge message-info diagnostic distinguishes this from the earlier
~335 ms transport pauses: for the 164 ms event, DDS reception gap was 23.625 ms
and receive-to-callback lag 140.897 ms. Whole-run command timing therefore
still fails the 100 ms requirement. Power logs also record **live `0x50005`**
before arming and during the attempt, and 76.8 C peak. Bits 0 and 2 mean live
undervoltage and throttling according to the
[Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/os.html#get_throttled).
This does not measure the traction battery's state of charge or prove a
single common cause of the control delays and weak launch. The trial helper
now additionally requires three clean host power readings and <75 C immediately
before arm; the ROS image's preflight did not report those host-only readings.
An isolated host-only check after stopping the stack correctly rejected a
fresh live `0x50005` reading, without starting ROS or actuators. The Pi power
condition can therefore recur while the driving stack is stopped as well.

Preparation attempts 191630, 192230, 192440 and 192915 remained disarmed when
command timing failed. Best-effort command subscriptions and asynchronous
publication did not remove the ~335 ms delivery pauses; both experiments
were reverted. The race launcher now defaults `FASTDDS_BUILTIN_TRANSPORTS` to
**UDPv4**, forwarded by Compose; normal Compose retains DEFAULT and reliable
QoS remains unchanged. Installed Fast DDS is 2.14.5. Its upstream SHM
health-check source contains the sleeping check described in
[Fast DDS issue 6535](https://github.com/eProsima/Fast-DDS/issues/6535);
this is a supported explanation for the old pauses, without a captured native
call stack. UDP diagnostic 193209 passed 33/33 and final readback with bridge
maximum 30.08 ms. Trial 193341 also passed preflight and did not record a 335 ms
transport pause, but the later callback delays above remain unresolved.

**Requested motor PWM versus application.** In trial 193341, nonzero motor
requests and bridge application both ranged **18.3--23.25 % of the ESC command
span**. At the actual 20,000,000 ns PWM period, sysfs telemetry reports
**1,591,500--1,616,250 ns high time** (7.9575--8.08125 % duty), versus
1,500,000 ns neutral. Most nonzero samples reached the 23.25 % ceiling. That
percentage is not measured motor power or battery charge. The configured
17 % start threshold (1,585 microseconds) is a prior, not a fresh measurement.
The first successful advance at 175815 used the same maximum 1,616.25 us,
with median nonzero pulse 1,595.485 us versus 1,616.25 us now. Thus there is
no additional hidden PWM clamp, but the bounded ceiling can be insufficient
for today's charge/load and the repeated zero commands also interrupt launch.
The active WT61 firmware publishes IMU CSV only. An unused legacy
`/vehicle/battery_v` parser is not a valid battery reading; no active battery
voltage or charge percentage was found. Battery charge remains unknown.

**Subsequent owner request: raise the ceiling only.** The owner explicitly
deferred all other changes and authorized another bounded trial. Only this
segment's command ceiling changed **0.25 -> 0.40 m/s in command space**, with
the helper's affine bridge ceiling **23.25 -> 28.2 %** and matching runtime
assertions. This raises maximum motor pulse **1,616.25 -> 1,641 us**. Desired
exploration speed, steering, margins, 1 m estimated path / 5 s bounds and
forward-only operation remain as before. Four segment-profile tests pass.
No additional adaptive, command-history or scheduling fix was applied.

Preparations `194948` and `195226` ended before arm because driver status
never appeared while the LiDAR start remained delayed by its supply guard.
Their host logs show respectively **25/71 and 20/71 samples with live power
faults**, so these are not simply old historical flags. Three unloaded host
readings subsequently passed at 53.8 C / `0x50000`, and preparation `195451`
allowed a 100 s post-start wait with the same stack settings. It also did not
reach driver readiness. Those preparation failures did not apply the higher
motor command and therefore did not test whether it overcame launch resistance.

**LiDAR startup correction.** Linux's `rpi_volt` hwmon alarm reflects the
latched undervoltage bit, whereas `vcgencmd get_throttled` bit 0 reports the
current condition; see the
[kernel driver](https://github.com/torvalds/linux/blob/master/drivers/hwmon/raspberrypi-hwmon.c).
In race runs, `SupplyGuard` now reads that current bit from the existing host
`power.csv` logger when its last row is at most 3 s old. Missing, partial or
stale status falls back to hwmon. `power_hold_max_s` is now one cumulative
hold budget across the motor ramp, rather than a new budget at each PWM step.
The initial wait, soft ramp and separate host pre-arm current-power/temperature
check remain. **33 sensor-timing tests pass**, covering current versus history
flags, stale/partial status and the cumulative multi-step budget. Actual
preparation `200501` started the LiDAR in 12.1 s and passed 33/33, then correctly
refused arm on a current `0x50005` power fault. Startup readiness alone does
not establish healthy power.

A temporary 1.5 GHz diagnostic trial `201056` reached arm, but the owner
reported no movement. It requested only 18.3% / 1,591.5 us motor output and
recorded 146 core-busy stops and 15 pose-age stops. Its mission did not latch
DONE. The diagnostic restored the normal 2.4 GHz ceiling; no persistent CPU
cap, planner-grid reduction or command-history correction was applied.

**Owner-replaced batteries, trial `20261022_201510`.** With the same bounded
preset and normal 2.4 GHz CPU ceiling, the owner confirms **advance and turn,
but no completed curve**. Border contact was not explicitly reported for this
attempt. Preflight passed **33/33**; all **97 host power samples** contained
only historical `0x50000` flags, with **no live undervoltage or throttle** and
53.2--73.0 C temperature. This is evidence of improved power in this run,
without a battery voltage/charge measurement or proof of a permanent fix.

The mission latched DONE on duration at **5.0827 s**, with **0.6884 m estimated
accumulated odometry path**, net x/y +0.6105/+0.1294 m and yaw +11.66 degrees.
Forward-only commands peaked at 0.3361 m/s and actual motor PWM at
**1,630.458 us / 26.0917% ESC command span**, below the 28.2% ceiling. Applied
steering reached +12.5 degrees, still a PWM-derived value rather than a wheel
angle sensor. Whole-run driver publication and bridge callback maxima were
**49.73 and 36.93 ms**, below the unchanged 100 ms gate. There were zero
core-busy stops, pose-age stops, stale-scan stops or sign faults.

The incomplete curve was **not solely a five-second limit**: the planner
reported zero feasible candidates and commanded zero from **2.1008 s** onward.
Replay of the recorded scans reproduces the final blocked state: a 0.1 m free
prefix against the 0.3 m at-rest stopping rule. This establishes why the
controller stopped; it does not establish that the physical curve is
impassable or that the LiDAR is faulty. Extending duration alone would retain
the same blocked-path stop. The stack is stopped after this completed attempt.

**Steering geometry remains unvalidated.** At the configured 0.30 m wheelbase,
12.5 degrees implies a nominal minimum rear-axle radius of **1.35 m**; 20 and
25 degrees would imply 0.82 and 0.64 m if physically achievable. Corridor width
alone does not specify the curve radius. The configured 5--10% servo duty
window, 8.2% centre and -1 direction sign correspond nominally to +23.04
degrees left and -12.96 degrees right. The servo intentionally imposes the
smaller symmetric authority, with a further 12.5-degree driver/bridge cap.
Those are configured PWM equivalents, not measured mechanical wheel limits.
Using more left authority would require consistent asymmetric limits through
the servo, bridge, planner and controller, plus validation of actual wheel
travel. No steering mapping or limit was changed in trial 201510, and neither
a complete curve nor moving loop closure has been validated.

**Directional steering follow-up.** The bounded segment now opts into
**22.5 degrees nominal left / 12.5 degrees right**. The configured servo window
remains 5--10% duty at 50 Hz, centred at 8.2%; its angle scale is unchanged.
These limits correspond to about 1,015 us left and 1,987.22 us right, within
the existing 1,000--2,000 us window. They do not establish measured wheel
angles. Other profiles retain the symmetric default.

Per-side limits now pass through the reactive arc search, wall-path curvature
checks, pursuit/control clamp, adaptive command envelope, bridge Twist inverse
and servo clamp. The search keeps its 21x11 count and includes zero curvature
and both directional extrema. Saturation observations are normalized by the
configured side so a larger left range does not redefine the right range.
The vehicle-YAML load preserves explicit per-side profile parameters. Modules
that require one scalar limit, including the global closed-lap planner, retain
the conservative side. Driver and bridge status expose the directional limits;
preflight verifies both against the configured PWM authority.

On the **actual initial scan sequence of trial 201510**, the changed planner
selects 2.2 m of checked free path and 99 feasible candidates at 22.5 degrees,
versus 0.9 m / 44 candidates at 12.5 degrees. It maintains the larger turn
through the first second of recorded poses. This is an open-loop comparison:
the scans and poses still describe the old physical trajectory, so it cannot
validate completing the curve. The old endpoint remains blocked in replay.
The tracked real-cloud regression also selects the larger turn and the
synthetic corner test finds a 2.2 m exit with unchanged footprint/margins.

Validation: **340 local tests pass**, 8 ROS-dependent tests skipped and 5 slow
tests deselected. A separate Pi ROS container with no devices and network
disabled passes **all 29 bridge tests**, including those eight ROS tests using
the test PWM backend. Live **disarmed** diagnostic `203211` passes **33/33**,
reads back 22.5/12.5 <= 23.04/12.96 degrees configured PWM authority, and has a
36.08 ms bridge historical callback maximum. Three host power readings are
clean (`0x50000`, 67.5--68.1 C). No new traction occurred in that diagnostic.
The owner then confirmed the car was at the start with the track clear, and
physical trial **203513** completed. It latched DONE on **1.0026 m estimated
accumulated path after 3.840 s**, with net odometry x/y +0.662/+0.5809 m and
**58.61 degrees left**. Applied steering reached 22.5 degrees (PWM-derived),
motor command peaked at 24.4857% and command speed at 0.2874 m/s. Final motor
command was zero. Whole-run driver/bridge maxima were **38.77/39.29 ms**, with
zero core-busy, pose-age, stale-scan or scan-age stops and no sign fault. Power
remained historical-only `0x50000`, with 73 C peak.

The owner later reported that this turn worked but the car caught a very low
border outside the LiDAR plane. This is **not a contact-free curve validation**;
the larger steering range does not make such low obstacles detectable. On
2026-10-23 the owner said he had moved the obstruction and explicitly requested
the same bounded plan again. That relaunch uses the unchanged preset and
independent preflight/arm checks; its result is recorded separately.

**2026-10-23 initial relaunch preparations.** These remained disarmed. The initial
startup `040017` failed to publish driver status. Retry `040452` passed 33/33,
then the host pre-arm power check rejected **current `0x50005` at 54.9 C**;
59 of 94 captured power rows contained live faults. A temporary 2 GHz CPU
ceiling trial `040826` also passed 33/33 and was rejected by that same power
gate at 56 C. It did not arm. This is a recurring Pi supply problem rather
than evidence that the larger steering command failed. It does not establish
the charge state of a separate traction battery.

A final 1.5 GHz preparation `041121` was interrupted before arming when the
owner asked about changing batteries. It is not a failed motion test or a
completed 1.5 GHz preflight. Cleanup verified the stopped/removed stack,
1,500 us motor neutral and restored 2.4 GHz ceiling. The owner subsequently
reported battery replacements and requested the same trial again.

Further owner battery replacements did not clear the loaded supply fault.
Normal-clock `041521` passed 33/33 and was rejected at 60.9 C. The completed
minimum supported **1.5 GHz** attempt `041827` also passed 33/33 but its final
power readings changed from `0x50000` to `0x50005` at 58.7 C. It did not arm;
the actual 2.4 GHz ceiling was restored afterwards. `042539` again passed
33/33 and failed that power check, with live faults in 44 of 89 power rows.

The owner then confirmed that **one battery feeds both the Pi and traction
motor** and tried another battery. Five stopped readings in `power_recheck_043053`
were clean at 4.801--4.823 V. Full preparation **043237** nevertheless returned
live faults in **49 of 88 host power samples**. Its preflight passed 33/33;
the final power samples were `0x50000`, then `0x50005` at 62 C, so it remained
disarmed. There is no `arm.log`, exploration start or mission completion for
this attempt. Docker is stopped/removed and motor PWM is verified at 1,500 us
neutral. No new locomotion occurred in these relaunch preparations.

The recurring fault identifies unstable power reaching the Pi, including
before traction starts; it does not locate the loss in the battery, conversion,
cable or contacts. Instantaneous ADC voltage samples cannot exclude brief
dips. In particular, `042539/voltage_transition.json` was captured around
cleanup without a per-sample container-state marker and must not be used to
claim a measured loaded voltage drop. Battery charge is not measured. No
power, timing, freshness, clearance or arm gate was removed to force motion.

**Power settling and actual armed attempt `044356`.** The diagnostic runner
now waits, disarmed, for ten consecutive seconds without live Pi power flags
and below 75 C, with a 90 s deadline. Missing readings or polling gaps over
2.5 s reset the clean window. It then runs the original 33-check stationary
preflight and the final three power readings; these gates remain unchanged.
Eight tests cover resets, missing/non-finite readings, deadlines and historical
flags. Historical-only `0x50000` is permitted; current faults are not.

At a temporary 1.5 GHz ceiling, `044356` passed those checks and actually armed.
The owner reported an attempt to move without enough force. Estimated speed
remained zero; ESC input pulses reached **1,614.188 us** (22.8375% command),
interrupted by frequent neutral commands. These pulses measure the ESC order,
not battery or motor-terminal voltage. The driver recorded 72 core-busy stops
and 25 pose-age stops. Its first reactive calculation took 249.5 ms. Mission
progress remained null; the finite arm publisher and cleanup stopped the run.
Live `0x50005` reappeared during the attempt. This does not isolate weak motor
supply from interrupted commands as the cause of the failed launch.

The real-car node now also checks the five-second mission deadline before
taking the perception lock, so a long planner calculation cannot defer its
zero command. The existing busy, arm, pose and scan watchdogs remain active.
Three ROS tests cover expiry with a held lock and the terminal latch.
An optional `reactive.preview_while_waiting` computes paths while disarmed;
its default is false and it never starts mission travel or produces motion
in WAIT. Diagnostic `050245`, at 1.5 GHz with preview enabled and the original
21x11 search, stayed disarmed because the strict preflight measured 99% driver
CPU against its 85% maximum (32/33 checks passed).

At the owner's explicit request, the segment profile is restored to
**preview disabled, 21x11 search**, with the working 22.5/12.5-degree steering
and unchanged 1 m / 5 s / 0.40 m/s forward-only bounds. Attempt **051206** used
a temporary **2 GHz** ceiling, verified by readback. It stayed disarmed: the
90.007 s power-settling window recorded live `0x50005` in **45/91 readings**;
the longest clean interval was 5.046 s, below the required ten seconds. It did
not reach the strict preflight or arm publisher. Cleanup verified no running
containers, motor PWM at 1,500 us neutral and restored 2.4 GHz CPU ceiling.
The restored segment profile remains deployed. No reduced search grid was
deployed for a physical trial.

The owner requested an unchanged repeat, **051622**, again at a temporary
2 GHz ceiling. This attempt passed 33/33, ten seconds of stable power and the
three final readings, then actually drove. It stopped on **1.0764 m estimated
travel after 2.423 s**, with net odometry +0.5402/+0.8159 m and **83.23 degrees
left**. The owner confirmed forward motion almost to the end of the straight,
but **the second right turn was not completed**. The final active steering
command was already 3.374 degrees right, approximately 103 ms before DONE.
The distance bound therefore cut this observation short; the run does not
validate the full sequence or establish that extending the bound alone will
complete the right turn. Border contact was not explicitly reported either way.

All host power samples were historical-only `0x50000`, with 68.1 C peak. There
were no core-busy, pose-age, stale-scan or scan-age stops and no sign fault.
Driver/bridge command gaps peaked at 39.72/33.72 ms. ESC command peaked at
24.6244%, with requested speed 0.29165 m/s; estimated speed reached 0.448 m/s
in exploration and 0.476 m/s at the first DONE sample. The 0.40 m/s profile
bound limits commands, not independently measured physical speed. Final PWM
was 1,500 us neutral, Docker was removed and the CPU ceiling restored to
2.4 GHz. The controller configuration was unchanged for this repeat.

**Full oval profile, requested 2026-10-23.** The owner cleared the whole oval,
located its entry to the right of the current car and reported a lap longer
than 10 m. `race_oval.yaml` preserves the working segment's reactive search,
speeds and margins, enabling one lap with independent 30 m / 120 s bounds.
The forward command ceiling remains 0.40 m/s. Right steering now reaches the
entire existing calibrated PWM window: **12.96 nominal degrees / 2,000 us**;
left remains 22.5 degrees. This does not establish the unmeasured mechanical
stop or widen the servo's 1,000--2,000 us window. The adaptive envelope uses
the conservative directional authority so the old common 12.5-degree value
does not silently cut the requested right range.

Recovery is enabled for at most six attempts per lap, each bounded to 0.25 m
or 1.5 s, with reverse commands capped separately at 0.30 m/s. It checks rear
clearance throughout the manoeuvre, shortens its distance budget for a nearby
observed obstacle and cuts immediately on a rear block or exhausted budget.
An empty accumulated cloud grants no reverse permission. Chassis shadows
still limit rear visibility; an absence of observed obstacles is not a
measurement of all rear space. Forward speed feedback can command coasting
without inadvertently engaging reverse. The previously validated ESC
brake/neutral/reverse sequence remains asynchronous.

`tools/core/arm_until_done.py` supplies a Pi-local, finite arm stream until
DONE, its independent 124 s deadline, or current power/temperature trouble.
Missing/stale host power telemetry also releases arm. It logs driver and
bridge status, sends false on exit and runs under an independent 126 s shell
timeout. The driver retains its mission deadline and freshness/watchdog
guards. The original 33-check preflight and startup power gates remain in
place. Validation: **332 local tests passed**, 10 skipped and 5 slow tests
deselected; an isolated Pi ROS container without devices or network passed
**52 tests**, including full right travel, recovery cuts, the node deadline
and arm power checks. Full-oval physical validation is recorded separately.

First armed oval attempt **181855** advanced and reversed, as confirmed by
the owner, but did not complete the curve. It exhausted its six recoveries
and remained blocked; arm was interrupted after 76.485 s. Estimated path
length was 2.7006 m, with net pose +1.2479/-0.6260 m and -24.38 degrees yaw.
All power readings were `0x0`, peak 64.8 C. Right steering reached its full
12.96-degree PWM equivalent. A **285.156 ms bridge callback gap** occurred at
the first reverse entry: messages arrived 27.424 ms apart, with 258.668 ms
between reception and callback. The driver publication maximum was 51.69 ms.
This localizes the delay to dispatch inside the bridge but not the blocking
operation. Cleanup verified neutral motor, no stack and restored CPU ceiling.

The identifier estimated the reverse deadband at **0.24799 m/s**, almost the
original 0.25 command ceiling. At the owner's request the ceiling is now
**0.30 m/s / 27.9% ESC command**, up from 26.25%; distance/time bounds stay
unchanged. Recovery distance now also counts pose increments when the speed
gate reports zero, so slow reversing cannot disappear from that bound.
The requested left-steer review found six forward-command samples above five
degrees left, peaking at 12.529 degrees, before the first reverse. Initially
the anticipation/inverse chain produced left steering while desired curvature
was still right; later the selected desired path also turned left. The oval
profile now bounds anticipation by the requested curvature magnitude, keeping
its physical turn target from changing sign solely due to lead. Identified
centre trim and obstacle-driven path selection still apply.

The arm observer now uses best-effort depth-one subscriptions, so observation
does not back-pressure reliable control publishers, and releases arm if the
bridge command-gap maximum reaches 100 ms. This change alone does not prove
the earlier dispatch delay resolved. An initial helper-path failure `181616`
stayed disarmed; the runner now stages the helper in its mounted diagnostic
directory and checks that path before preflight.

**General control and recovery corrections, 2026-10-23.** Physical run
`190748` still failed to finish the curve: estimated path 2.7188 m before the
120 s stop, 1,053 core-busy stop ticks, zero pose-age or scan-age stops, and
power `0x0` throughout. The owner observed a right turn interrupted by a
strong left correction, then a late stop facing the outer wall. Requested
curvature already turned left, so this is not solely servo sign or lag lead.
The two first reverses only covered about 0.153 and 0.141 m before timeout.
Changing the wall-distance target alone did not remove the left selections
in recorded-pose replay; it is not proof of a corrected curve.

Predicted-pose reception now uses its own latest-value channel alongside
control, outside the heavy scan callback group. The actual ROS regression
accepts eight new poses while scan processing holds the core lock. A further
handoff lets a pending control tick acquire the lock before another heavy
sensor callback; it keeps the 250 ms busy-stop guard. Holds and fallback zero
commands are also included in the identifier's history, rather than leaving
it to learn from an earlier command which was no longer being sent.
In a 60 s Pi dry comparison `192624`, with the bridge disabled and CPU capped
at 2 GHz, this produced zero busy-stop, pose-age-stop and scan-age-stop ticks,
maximum predicted-pose consumer age 23.74 ms and driver gap 49.91 ms. This
does not establish moving performance or eliminate occasional raw LiDAR loss.

The shared SIM/car `core/recovery.py` evaluates reverse bicycle arcs using the
swept footprint and the forward space available from their endpoints. It
selects straight or curved backing from current geometry and local desired
curvature, with no track template, prescribed turn order or persistent turn
direction rule. A fresh obstacle in the curved swept path aborts recovery.
Forward probes score a maneuver only; normal LiDAR planning runs again before
advancing. Model and visibility limits still apply, including rear chassis
shadows. The rejected oval-specific turn-commitment experiment was removed
before deployment.

At the owner's request, the current hardware trial profile targets 20 cm
beside either body side (0.36 m from the centre for a 0.32 m wide car). Recovery
now permits at most 0.50 m or 3 s, still shortened by observed rear obstacles;
early completion requires a feasible forward path with at least 0.60 m usable
length. Reverse command remains capped at 0.30 m/s, six attempts, and the
whole mission remains one lap / 30 m / 120 s. Hard collision margin, freshness,
power and command-gap guards are unchanged. These are vehicle/test bounds,
not a hard-coded route. No complete moving lap has yet been validated.
Validation of these corrections: 321 local shared-core/profile tests passed
(three ROS modules skipped, five slow tests deselected), and 61 relevant tests
passed in an isolated Pi container without network or devices. The captured
blocked-corner geometry gives more forward room after the 0.50 m probe than
after 0.25 m; this is a fixed-geometry prediction, not a completed physical
maneuver. Thirty evaluations on the Pi at 2.4 GHz averaged 3.62 ms, maximum
4.08 ms. The actual bounded trial still uses the temporary 2 GHz ceiling.

**Moving result and continuous-lap preparation.** In `193423` the owner
reported almost completing the first lap, then stopping on entry to the
second curve; reverse steering was physically appropriate. All six permitted
recoveries had been used, preventing another recovery at that block. The
recorded maneuvers selected straight, half-left or full-left reverse arcs
from geometry; this is not a fixed side. Reverse travel per attempt was
0.148, 0.284, 0.346, 0.390, 0.107 and 0.067 m. Recorded pose heading accumulated
about -197 degrees and total path, including reversing, reached 5.915 m.
These odometry values are not an independently measured lap length.
There were zero busy-stop and pose-age-stop ticks, one scan-age-stop tick,
driver gap 68.98 ms and bridge gap 68.93 ms. Power stayed `0x0`, peak 68.6 C.
Arm was released at 106.49 s after the car remained blocked with no recovery
attempts left; neutral, stack shutdown and CPU restoration were verified.

At the owner's request for continuous laps, the current profile now allows
**two detected laps, 40 m or 300 s**, and **12 recovery attempts per lap**.
Each reverse retains its 0.50 m / 3 s bounds and geometry selection; speed,
servo, clearance and freshness guards remain. Status exposes the used and
allowed recovery counts. The finite arm/deadline chain is correspondingly
304 / 306 / 312 s; manual SIGINT is recognized as an intentional arm exit.
The SLAM builds the map during exploration; after detecting the first return,
the driver performs its existing 3 m overlap, plans and validates a race line,
then switches to map tracking if handover succeeds. Simply starting the second
physical circuit does not prove map tracking is active. Two-lap completion and
the EXPLORE/CLOSING/PLANNING/RACE transition still require physical validation.

**Fresh start on every run.** The owner subsequently reported moving the car
during `194456` and returning it to the initial position. That attempt was
interrupted and cleaned up; it is not valid evidence of inherited software
state. Its saved pre-arm status already showed WAIT, no plan, zero laps,
travel, recovery count and mission distance, with prior steering gain 1.0.

`apex_race_up.sh` now always removes the preceding stack before creating a new
race/dry/replay/nudge run. `race_fresh_start.py` verifies removal and neutral
or disabled PWM, clears only the two APEX packages' bytecode caches, creates
an unused output directory and records `fresh_start.json` with the preceding
run id. Even two starts in the same second cannot reuse an old output folder.
Recorded evidence, configuration, trained weights and saved maps are preserved.
The bounded runner then verifies fresh driver counters/prior, no container
restart and an empty/unset live SLAM `map_file_name` before permitting arm.
Deleting bytecode is not itself an odometry/SLAM reset; the new processes and
these live checks provide that reset. Six isolated tests cover evidence and
weight preservation, directory uniqueness, symlink handling and PWM checks.

**Preserving the working settings and fixing a lateral rear return.** In
`195524` fresh state was verified, but the car stayed stopped: the owner
identified a real border close to the front-left corner. The saved config
comparison against working run `193423` differs only in recovery count and
mission distance/time budgets, plus the separately supplied lap count. A
regression fixture now locks every other driver setting to that working run.
The forward planner continues to treat the real border as an obstacle.

The scalar rear check projected a return at about x=-0.25 m, y=+0.30 m onto
the rear axis, reporting only 0.024 m behind the bumper and prohibiting all
reverse maneuvers. This is over-conservative for a point beside the swept
body. Geometry-selected recovery now uses its checked footprint instead of
that projection, extending the rear footprint by the existing 0.15 m stopping
reserve. The 0.08 m hard clearance remains; the existing 0.02 m escape tolerance
applies individually only to obstacles already inside that hard clearance.
A close front border therefore cannot weaken checks against a different rear
obstacle. No obstacle returns are deleted. On the recorded blocked-start
cloud this admits a checked 0.50 m straight reverse, whose endpoint probes
open about 1.75 m forward; this is an offline prediction, not physical proof.

`195524` also ended automatically at 74.03 s on a 133.79 ms bridge command gap,
while commands were already zero. Source/receive gaps were 133.78/133.75 ms,
dispatch 0.25 ms; this occurrence was upstream of bridge dispatch, unlike
the earlier `181855` gap. Maximum core-control hold was 125.9 ms; its blocking
operation remains unproven. The 100 ms arm-release guard is unchanged.
The runner now stores exact package sources, vehicle/profile YAML and SHA-256
hashes beside each new trial, making later working versions reproducible.

**Moving trial `201912`: final-curve stop diagnosed from the saved observations.**
The owner reports almost completing the first circuit and leaving the car at
its stopping position. The run used all 12 recovery attempts (last start at
87.58 s), issued its last nonzero command at 122.00 s, then ended at the 300.12 s
mission deadline. It recorded no lap crossing or global plan. Final map pose
was 0.917 m from the starting pose with a different heading; gated travel was
9.884 m (the first closure arm threshold is 10 m). The cumulative mission
estimate of 11.697 m includes reversing and small pose variations.

Replaying the final accumulated cloud reproduces the recorded wall distance,
clearance diagnostic and 22 feasible candidates. Every admissible candidate
has only 0.30 m free, equal to the stopping reserve, so the forward command is
effectively zero. The exhausted recovery allowance prevents another backout.
In the same frozen scene, permitting one additional attempt admits a checked
0.50 m reverse arc and predicts a 2.15 m forward continuation at its endpoint;
this is an offline geometric prediction, not a completed physical maneuver.
There were zero core-busy/pose-age stop ticks, one scan-age stop tick and a
53.74 ms maximum driver command gap. The motor was confirmed neutral afterward.

The driver received 213 map callbacks, but this run did not save a final SLAM
occupancy grid before teardown. Its diagnostic figure reconstructs observed
LiDAR points using recorded odometry and must not be presented as that grid.
It confirms boundary observations in the stopping area, not that the full
remaining curve was represented in the original SLAM map. No reset, relaunch
or production controller change was made during this inspection.

**Recovery allowance after successful forward escape.** The next two-lap
profile keeps the same steering, speed, clearance, closure and per-maneuver
bounds, but separates a total per-lap ceiling of 36 recoveries from at most
12 attempts without forward escape. Both 0.75 m net odometry displacement
from the last recovery endpoint and 0.75 m projected forward travel under a
usable forward command are required to replenish the latter allowance.
Measured speed must exceed 0.05 m/s. Reverse motion, rocking, parked pose noise,
SLAM corrections and waiting do not replenish it; the per-lap ceiling remains.
The optional separate limit defaults to disabled for existing profiles.

On the recorded `201912` motion, this accounting replenishes after the first
five attempts and reaches the final curve with seven used out of twelve,
rather than exhausting the old total of twelve. The real final cloud is a
regression fixture; another bounded reverse is admitted with this allowance.
This replay does not establish that a new physical lap will finish. Validation:
342 local tests passed (3 ROS skips, 5 slow deselections), 82 Pi tests passed.
The bounded runner also captures the actual latched SLAM occupancy grid and
graph, when available, after disarming and before teardown, outside moving
control callbacks. Trial `204107` is the first physical run of this correction.

**First physical circuit and the transition gate (`204107`).** The owner
confirmed a complete first circuit. Replaying the logged map poses finds a
valid start-line crossing at 48.163 s, 8.341 m accumulated travel, 0.0328 m
lateral offset and 3.66 degrees heading error. The old 10 m minimum travel
excluded that crossing, so no planning was submitted. The current bounded
profile uses a 5 m arming minimum, preserving the 12 s arming time, return
position/heading checks and 3 m overlap. A recorded-trajectory regression
checks the first crossing and subsequent planning submission.

**Plans requiring maneuvers.** The saved real SLAM grid (`204107`, 92 by 78
cells at 0.05 m) produces a forward candidate exceeding its curvature and
clearance bounds. The shared planner now records `direct_tracking_allowed`
and explicit rejection reasons, with the observed SLAM route stored separately.
A valid forward candidate retains the existing alignment check and pure-pursuit
handover. A rejected candidate selects the explicit `map_guided` phase instead:
the observed route guides the existing live-LiDAR local arc planner through
a forward route window, and bounded geometric recovery remains available.
The rejected candidate is never sent to pure pursuit. `map_guided` is actual
SLAM route guidance, not evidence of successful direct race-line tracking.
Real and simulation share these core changes. Validation after both changes:
348 local tests passed, 3 ROS skips, 5 slow deselections; 88 Pi tests passed.

**Publication evidence.** `tools/pc/plot_race_evidence.py RUN --output BASE`
exports a vector PDF, SVG, 600 dpi PNG, English caption and LaTeX snippet, trace
CSV and source hashes. It uses the actual saved SLAM grid and recorded onboard
trajectory, commands, phases and lap counts; no external ground truth is
implied. The bounded runner retains map/route/plan artifacts for each trial.
An exploration-only run must not be presented as a validated transition or
completed map-tracking run. Trial `205605` starts the physical validation of
the corrected first-lap gate and explicit map-guided execution.

**Continuous handover using the learned plant.** The
`210234` trial recorded a first return at 61.14 s and entered map-guided local
planning at 104.46 s, but did not complete a second lap. The owner reported poor
driving after the return. Logged gains and trim remain identical across the map
handover; the identifier and adaptive shaper were not reset. The motion near
the start line preceded map guidance by approximately 43 s.

Two unit inconsistencies were corrected. Saturation is already stored in
command-curvature units by the identifier, so the adaptive envelope must not
divide it by the learned gain again. With the recorded posterior, that division
unnecessarily reduced the right command limit from 0.767 to 0.537 inverse
meters. Optional `adaptive.plan_with_identified_model` now makes local arcs,
recovery arcs and the global planning snapshot use physical curvatures from the
same learned forward model that the adaptive inverse compensates. Geometry-only
equivalent angles can differ from command angles; the published actuator
commands retain the configured 22.5/12.96 degree nominal limits and PWM window.

The current bounded profile requests a fresh map immediately after the first
valid return (`overlap_m: 0`) while continuing with the same local controller
during asynchronous planning. It does not stop or drive back to station zero.
Guide cost is blended in over one second while the local wall-reference cost
fades out. Route station selection uses heading and smooth odometry continuity,
including backward travel and wrapping across the seam, to avoid jumping to
a nearby branch. Valid direct tracking retains its pose-alignment check; an
invalid forward candidate remains explicitly map-guided with live footprint
checks and bounded recovery. All shared changes are identical in simulation.
Validation: 359 local tests passed (3 ROS skips, 5 slow deselections), 99 Pi tests
passed. Reverse authority is the owner's requested 0.40 m/s
command / 31.2 percent ESC ceiling; each maneuver remains at most 0.50 m / 3 s.

Trial `212358` never armed: available storage was below the unchanged 1024 MB
startup threshold. Verified old logs were compressed, then `212633` started and
ran for 24.77 seconds before the independent arm supervisor stopped it. Its
power log records a live `0x50005` fault at 20:28:27 UTC with temperature 63.7 C;
this was not a storage or thermal stop. The trial remained in exploration with
zero recognized laps, so it provides no new validation of the handover.

Before the next attempt, 71 completed ROS run directories were copied to the
PC, verified by SHA256 and removed from the Pi; 144,722,259 file bytes were
offloaded and 1,386,147,840 bytes were free afterward. The latest run, diagnostic
maps and model weights were retained. A separate comparison verified all 133
selected source, configuration, model and runner files between PC and Pi.
The only differing deployment script and ten missing auxiliary odometry files
were synchronized; the active controller, perception code and weights already
matched. Trial `213545` passed all 33 preflight checks and the stable-power
window, then the arm supervisor stopped after 2.86 seconds: the power log
records another live `0x50005` fault at 20:37:18 UTC, temperature 62.0 C.
It recognized no laps and did not reach map guidance. Estimated travel was
0.046 m; no independent measurement of that movement is claimed. The repeated
loaded input-power fault remains unresolved despite storage cleanup. These
flags do not distinguish the battery from wiring or regulation in the shared
supply. The motor was verified at neutral, the container removed and CPU ceiling
restored to 2.4 GHz; 1,382,379,520 bytes remained free. No power guard was removed.

Two further explicitly requested battery retries did not resolve the input
power fault. Trial `214156` stopped after 2.85 s with a live `0x50005` sample;
the same fault had also appeared before arming during startup. Its logs were
retrieved after connectivity returned. After another battery replacement and
Pi reboot, diagnostic `215650` / ROS run `215652` passed all 33 checks and the
stable-power window, but the arm helper's first power check rejected a new
live fault. Its 0.32 s result includes disarm cleanup; no movement was enabled.
The kernel independently reported repeated undervoltage before arming, with
the motor neutral. These observations establish a Pi input-power problem,
without isolating battery, cable or regulation. Cleanup again verified neutral,
no container and the restored 2.4 GHz ceiling. All 133 selected PC/Pi file hashes
and the run's 27 snapshotted source/configuration files matched the local tree.

Trial `220317` / ROS run `220320` maintained clean power throughout (maximum
recorded temperature 68.1 C), recognized one lap and entered continuous
map-guided execution. The owner reported better, faster driving in the second
lap. The candidate forward line remained invalid: maximum smoothed curvature
3.585/m against 1.396/m, footprint separation 0.09758 m against 0.10 m. This
rejects that candidate; it does not prove that the circuit has no forward-only
solution. The actual map and phase-colored English publication figure are
retained in the trial's `paper/` artifacts and document a partial second lap.

The independent arm helper stopped this attempt at 110.23 s on a 111.502 ms
bridge callback gap. DDS evidence measured source spacing 15.038 ms, source to
receipt 0.097 ms, and receipt to callback 96.828 ms, while the driver's maximum
publication gap was 39.65 ms. This locates the delay at bridge dispatch, without
identifying the exact blocking operation. The affine bridge now receives
commands separately from serialized actuator/status work using two callback
groups, two executor threads and coherent locked input snapshots. The legacy
executor and the existing 100 ms arm gate / 350 ms bridge watchdog remain.
A ROS regression deliberately blocks status publication, confirms command
reception continues, then stops commands and verifies neutral.

The geometry review also found that smoothing repeated reversals produced
nearly coincident reference stations. Uniform arc-length resampling after
smoothing removes that numerical distortion. On the recorded map replay it
reduced reference curvature from hundreds per meter to below 5/m and met the
unchanged map-clearance check, but the candidate still exceeded the turn limit.
Separately, rotating near the start had triggered the radius fallback before
the actual start plane crossing. The fallback now requires exiting past that
plane in the original direction and clears a rejected stale approach. The
recorded attempt waits for its actual line crossing at about 70.18 s instead
of the earlier radius event near 65.44 s. Existing short-lap replay still passes.
All shared fixes are identical in simulation. Validation: 384 local tests,
13 ROS skips, 5 slow deselections; 104 Pi tests passed. Trial `222959` begins
physical validation of these corrections with unchanged margins and authority.

**Completed two-lap trial `222959`.** Both recorded events were line crossings:
lap 1 at 34.209 s (0.0854 m lateral offset, 1.85 degree heading error), lap 2 at
49.053 s (0.1214 m, 2.96 degrees). The second lap took 14.844 s; the driver entered
DONE at 49.920 s and the finite arm helper exited normally at 53.261 s including
initialization/disarm overhead. Planning started at 34.660 s and map-guided
execution at 38.840 s. The run had five reverse starts during reconnaissance
and one during second-lap planning at 38.200 s, continuing into map guidance.
The owner reported this remaining difficulty at the first curve and requested
archival for the paper, deferring further control changes.

Power stayed clean, peak temperature was 69.7 C, bridge command-reception gap
peaked at 78.20 ms, driver publication gap at 42.45 ms, and the bridge reported
no watchdog timeouts. Neutral, container removal and restoration of the 2.4 GHz
CPU ceiling were verified after capture. The optimized forward candidate still
failed curvature (2.177/m versus 1.449/m) and clearance (0.0891 m versus 0.10 m),
so the executed SLAM stage must be described as map-guided local planning.

The local paper archive is
`archive/generated/paper/real_vehicle_20261023_222959/`, with complete run and
diagnostic logs verified against the Pi, actual SLAM grid and graph vertices,
source/configuration/weight snapshots, validation/environment metadata, English
PDF/SVG/600 dpi PNG figures and captions, trace CSVs, reproduction scripts and
SHA256 manifests. `plot_paper_details.py` adds a map/lap view, handover detail and
single-run descriptive lap comparison. The archive documents onboard estimates;
it does not claim external ground truth, collision-free validation or a
controlled ablation. The deferred first-curve reverse is recorded in
`NEXT_STEPS.md`; no additional motion was requested during archival.

## 6. The no-movement ladder

| # | Step | Pass criterion |
|---|---|---|
| U | `pytest real_vehicle/ros2_ws/src/apex_race/test real_vehicle/ros2_ws/src/apex_telemetry/test` | all green: 13/13 parity hashes, the firewall, the 48°→12.5° regression, the curvature round-trip to 1e-9, and identifiability (the affine envelope identifies, the legacy one provably cannot) |
| B | `pytest simulation/ros2_ws/src/apex_fusion_research/test -m "not slow"` | all green |
| H1 | `race_harness_2d.py --vehicle-yaml … --driver-params …` | §4 |
| C | `apex_race_up.sh sensors-only` + `record_real2sim_sensors.py` + `sensor_timing_audit.py --profile APEX_real_wt61` | LiDAR 10/10 as in `lidar_check_static_02.json`; IMU 100 Hz ±5 %; 10 ± 20 % IMU samples per revolution |
| R1 | `replay_slam.py prepare` on that capture | the offline estimator matches the Pi's `estimator.csv` within 1 mm |
| R2 | `replay_slam.py run` on that capture | replay observer receives ≥90 % of input scans; ≥90 % of motion-eligible map updates produced; map→odom correction stays under 5 cm over ≥120 s at rest (see §5l) |
| S | static servo sweep, wheels off the ground, **ESC unplugged** | symmetric authority ≥12.0°; centre within 0.5°; `applied_deg` never exceeds the limit; `right_ratio` measured into `vehicle_real.yaml` |
| P | `vcgencmd get_throttled` / `measure_temp` / governor | no live under-voltage, <75 °C, `performance` |
| I | IMU axis signs, static | at rest `az ≈ +9.8`; nose up 30° → `ax ≈ −4.9`; **yawing left → `gz > 0`**. The WT61PC firmware passes the raw sensor axes through, and a flipped `gz` latches `sign_fault` and kills the adaptation silently |
| D1 | `apex_race_up.sh dry` + `check --duration 180 --expect-phase explore` | TF chain complete; estimator p95 latency <250 ms and compute <40 ms; **command gap p99.9 <100 ms**; no core above 85 %; 0 scan-age stops |
| D2 | `apex_race_up.sh replay` + `check` | `race_events.json` reaches `planning`, `handover`, `t_race`, `t_done`; `scan_age_stops == 0`; the command gap holds **through the planner burst** |

Step **H** -- pushing the car by hand with the ESC unplugged -- is the only way
to measure the odometry's accuracy at the real speed before the car drives
itself. It is movement, and it is announced like any other.

## 7. Go / no-go for the first motion test

U, B, H1 · C 10/10 · R1 <1 mm · R2 <5 cm · S ≥12° symmetric · I all four signs ·
D1 and D2 with 0 scan-age stops, no core above 85 % and the command gap under
100 ms · P with no live under-voltage · `/apex/cmd_vel_track` with exactly one
publisher · ≥1 GB free · and a person holding the arm with physical access to
the battery.

Then, staged: 2 m straight → 5 m, reading the identified `v_gain` off the status
to re-set `speed_pct_per_mps` → a figure of eight for `gain_left/right` and
`trim` → lap 1 → three laps at 0.90 m/s.
