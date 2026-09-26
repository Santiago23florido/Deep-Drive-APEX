# apex_fusion_research

Sensor models and an unaided inertial navigation baseline for **LiDAR–IMU
sensor-fusion research** on the APEX 1/10-scale car in Gazebo.

The package is the first stage of a research line whose goal is a publishable
LiDAR–IMU fusion method for small autonomous vehicles. It provides:

| Component | What it does |
| --- | --- |
| **LiDAR noise model** | Corrupts Gazebo's ideal scan with a physically motivated, reproducible stochastic model (heteroscedastic range noise, incidence-angle effects, calibration errors, angular jitter, dropouts, short and spurious returns, quantization). |
| **IMU model** | Turns Gazebo's ideal inertial signals into *raw* MEMS measurements (white noise, turn-on bias, Gauss–Markov bias instability, bias random walk, scale factor, misalignment, g-sensitivity, quantization, saturation). No orientation is provided, exactly like a physical 6-axis IMU. |
| **Strapdown INS** | Integrates the raw IMU (attitude, velocity, position) after a realistic static alignment. No aiding: the error accumulation of a real IMU becomes visible. |
| **Ground truth + evaluation** | 6-DoF truth from Gazebo, per-sample navigation errors, CSV/metadata recording and publication-quality plots. |
| **SLAM baseline** | Identical `slam_toolbox` instances fed with good and damaged sensors, run headless. Maps, trajectories, the real track and raw measurements are exported to CSV, then scored against the real track (section 9). |
| **Validation tools** | Allan-variance identification, LiDAR-model statistical report, unit tests with analytical references. |
| **Real2sim sensors + closed loop** | Gazebo-native sensors made real by a realism layer (RPLIDAR A2M8 rolling revolutions, LSM6DS3 chip), the learned LiDAR-inertial odometry + slam_toolbox driving a pose-dataset format on its own estimate, with a truth referee and the evaluation (section 13). |

Every model lives in a ROS-independent module configured by a dataclass. Every
dataclass field is automatically exposed as a ROS parameter, loadable from a
YAML preset and changeable at runtime.

---

## 1. Quick start

```bash
# Conda/venv Pythons break ROS 2 Jazzy; the launcher removes them from the environment.
cd ~/AiAtonomousRc
./simulation/tools/sim/apex_fusion_research_up.sh            # Gazebo GUI + RViz, car idle
./simulation/tools/sim/apex_fusion_research_up.sh --arm      # same, the car drives the autonomous tour
```

What you get:

* **Gazebo** with the car on the track.
* **RViz** (fixed frame `world`) showing, in green, the ground-truth path and the
  noisy scan projected with the true pose and, in orange, the pure-INS path and
  the same scan projected with the INS pose. The orange cloud smears and drifts
  away from the walls as inertial errors accumulate.
* A run directory `simulation/data/fusion_research/<timestamp>/` with the CSV
  of navigation errors, metadata and a copy of the configuration used.

After (or during) a run:

```bash
ros2 run apex_fusion_research plot_ins_drift simulation/data/fusion_research/<run>
```

SLAM baseline capture, fully headless (section 9):

```bash
./simulation/tools/sim/apex_fusion_slam_capture.sh
```

Useful options of the launcher (`--help` lists all of them):

| Option | Meaning |
| --- | --- |
| `--imu-config <preset\|file.yaml>` | `consumer_mems` (default), `white_noise_only`, `ideal`, or your own YAML |
| `--lidar-config <preset\|file.yaml>` | `rplidar_like` (default), `ideal`, or your own YAML |
| `--ins-config <preset\|file.yaml>` | `static_coarse` (default) or `truth_init` |
| `--imu-seed N`, `--lidar-seed N` | New random realization without editing the YAML (Monte-Carlo runs) |
| `--arm`, `--arm-delay-s S` | Let the car drive; arming happens after the INS alignment |
| `--headless`, `--no-rviz`, `--no-record` | Batch runs |
| `--run-name`, `--output-root` | Where the run is stored |

Equivalent direct launch:

```bash
ros2 launch apex_fusion_research fusion_research_sim.launch.py imu_config:=consumer_mems imu_seed:=3
```

> **Important:** start driving only after the INS log prints
> `navigation started`. The static alignment needs the car to stand still for
> `alignment.duration_s` (5 s by default) and restarts whenever it moves.

---

## 2. Architecture

```mermaid
flowchart LR
  subgraph Gazebo["Gazebo (ideal sensors, noise = 0)"]
    GI["/apex/sim/imu<br/>ideal IMU"]
    GS["/apex/sim/scan<br/>ideal scan"]
    GP["dynamic_pose/info<br/>model + link poses"]
  end
  GI --> IMU["fusion_imu_sensor<br/>IMU error model"]
  GS --> LID["fusion_lidar_noise<br/>LiDAR noise model"]
  GP --> TRU["fusion_truth<br/>6-DoF truth of IMU frame"]
  IMU -->|"/apex/fusion/imu/data_raw"| INS["fusion_strapdown_ins<br/>alignment + strapdown"]
  TRU -->|"initial pose only"| INS
  INS -->|"/apex/fusion/ins/odom"| MON["fusion_ins_error_monitor<br/>errors, CSV, metadata"]
  TRU --> MON
  IMU -->|"true biases"| MON
  LID -->|"/apex/fusion/scan_noisy"| PRJ["fusion_scan_projector<br/>RViz clouds"]
  TRU --> PRJ
  INS --> PRJ
```

The vehicle keeps driving with the existing APEX control stack. The research
nodes only observe it, so the experiments never change the vehicle behaviour.
The launch file forces all Gazebo sensor noise to zero, so **every error comes
from the models of this package** and is fully known.

### Package layout

```
apex_fusion_research/
├── apex_fusion_research/
│   ├── core/                  # ROS-independent, unit-tested models
│   │   ├── lidar_noise.py     # LidarNoiseConfig, LidarNoiseModel
│   │   ├── imu_error.py       # ImuErrorConfig, TriadErrorConfig, ImuErrorModel
│   │   ├── strapdown.py       # StrapdownIntegrator, static_coarse_alignment, InsConfig
│   │   ├── allan.py           # overlapping Allan deviation + N/B/K fit
│   │   ├── rotation.py        # quaternion utilities (Hamilton, [w,x,y,z])
│   │   ├── slam_odometry.py   # planar SLAM motion prior (heading_only / full_pose)
│   │   ├── occupancy.py       # OccupancyGrid -> points, PGM/YAML export
│   │   ├── map_metrics.py     # SE(2), map similarity, point-to-line ICP, ATE
│   │   ├── lidar_rolling.py   # real2sim: gpu_lidar captures -> rolling A2M8 revolutions
│   │   ├── imu_chip.py        # real2sim: Gazebo 1 kHz IMU -> LSM6DS3 output
│   │   └── config_io.py       # dataclass <-> flat parameters <-> ROS YAML
│   ├── nodes/                 # thin ROS 2 wrappers around core/
│   │   ├── imu_sensor_node.py
│   │   ├── lidar_noise_node.py
│   │   ├── truth_node.py
│   │   ├── strapdown_ins_node.py
│   │   ├── ins_error_monitor_node.py
│   │   ├── scan_projector_node.py
│   │   ├── slam_odometry_node.py      # <tag>/odom -> <tag>/base_link prior
│   │   ├── slam_map_recorder_node.py  # SLAM maps + trajectories + real track to CSV
│   │   ├── real_sensor_node.py        # real2sim: Gazebo sensors -> /apex/imu/data_raw, /lidar/scan_raw
│   │   ├── sim_actuation_node.py      # real2sim: /apex/cmd_vel_track -> dataset ESC / servo -> Gazebo
│   │   ├── track_driver_node.py       # real2sim: pure pursuit + speed plan on the estimated pose
│   │   ├── run_referee_node.py        # real2sim: lap / collision / off-track verdict (truth)
│   │   └── _common.py         # ConfigParameters: dataclass -> ROS parameters
│   └── tools/                 # offline analysis (console scripts)
│       ├── plot_ins_drift.py
│       ├── allan_analysis.py
│       ├── lidar_noise_report.py
│       ├── plot_slam_maps.py  # SLAM maps vs real track, metrics
│       └── wait_for.py        # script helper: INS ready / tour finished
├── config/{imu,lidar,ins,slam}/*.yaml   # parameter presets
├── doc/images/                # figures used in this README
├── launch/fusion_research_sim.launch.py
├── launch/apex_real2sim.launch.py     # real2sim closed loop (section 13)
├── rviz/fusion_research.rviz
└── test/                      # pytest, runs without ROS
```

Design rules:

* **Models are pure NumPy** (`core/`). They can be reused in Monte-Carlo
  scripts, other simulators or a future fusion filter without ROS.
* **Nodes are thin**: they convert messages, call the model and publish.
* **One dataclass per model** is the single source of truth for parameters.
  `ConfigParameters` exposes each field as a ROS parameter
  (`gyro.noise_density`, `alignment.mode`, ...). Changing a parameter at runtime
  rebuilds the model, which for the IMU is equivalent to a power cycle.

---

## 3. LiDAR noise model

`core/lidar_noise.py` corrupts the ideal scan beam by beam. It extends the
beam mixture model of Thrun, Burgard and Fox [1] with a heteroscedastic,
incidence-dependent Gaussian term, systematic calibration errors and angular
jitter, the dominant error sources reported for 2D laser scanners [2, 3].

For beam *i* with nominal azimuth θᵢ and ideal range rᵢ:

1. **Angular jitter.** The beam is fired along θᵢ + δθ, δθ ~ N(0, σ_θ²).
   The true range r\*ᵢ along that direction is interpolated from the ideal scan
   when the neighbouring beams hit the same surface; the published azimuth
   stays θᵢ (encoder error).
2. **Incidence angle** αᵢ between the beam and the surface normal, estimated
   from the neighbouring beams of the ideal scan.
3. **Outcome** (one per beam):
   * **DROPOUT** (published as `+inf`) with probability
     p_drop = p₀ + p_r (r\*/r_max)² + p_graze · 1[α > α_graze]
   * otherwise a mixture of
     * **SHORT** (p_short): unexpected close return, z ~ TruncExp(λ) on [r_min, r\*]
     * **RANDOM** (p_rand): spurious return, z ~ U(r_min, r_max)
     * **HIT** (remaining): z = (1 + s) r\* + b + e,
       e ~ N(0, σ²), σ(r\*, α) = √(σ₀² + (k r\*)²) · (1 + g (1/cos α − 1))

   b ~ N(0, σ_b²) and s ~ N(0, σ_s²) are drawn **once per realization**
   (power-up): range offset and scale calibration errors.
4. **Quantization** to the range resolution and REP-117 conventions
   (`-inf` below `range_min`, `+inf` above `range_max`).

| Parameter | Symbol | Unit | `rplidar_like` |
| --- | --- | --- | --- |
| `range_sigma_const_m` | σ₀ | m | 0.003 |
| `range_sigma_prop` | k | – | 0.005 |
| `incidence_sigma_gain` | g | – | 0.5 |
| `max_incidence_angle_deg` | – | deg | 85 |
| `range_bias_std_m` | σ_b | m | 0.005 |
| `range_scale_std` | σ_s | – | 0.002 |
| `angle_jitter_std_rad` | σ_θ | rad | 0.0035 (0.2°) |
| `dropout_prob_base` | p₀ | – | 0.005 |
| `dropout_prob_range_coeff` | p_r | – | 0.05 |
| `grazing_angle_deg` / `dropout_prob_grazing` | α_graze / p_graze | deg / – | 75 / 0.3 |
| `short_prob`, `short_rate_per_m` | p_short, λ | –, 1/m | 0.002, 1.0 |
| `random_prob` | p_rand | – | 0.001 |
| `range_resolution_m` | – | m | 0.00025 |
| `seed` | – | – | 7 |

The node publishes the empirical outcome fractions and the RMS HIT residual on
`/apex/fusion/lidar/status`, and the realized b, s on the latched topic
`/apex/fusion/lidar/realization`.

---

## 4. IMU error model

`core/imu_error.py` implements the standard inertial sensor error equations
(IEEE Std 952 [4], IEEE Std 1293 [5], Groves [6] §4.4):

```
ω̃ = sat(Q( (I + S_g + M_g) ω + b_g(t) + G_g f + n_g ))
f̃ = sat(Q( (I + S_a + M_a) f + b_a(t)          + n_a ))
```

* ω, f: true angular rate and **specific force** in the sensor frame (Gazebo
  reports +g on z at rest, like a real accelerometer).
* S: diagonal scale-factor errors. M: off-diagonal misalignment and
  non-orthogonality. G_g: gyro g-sensitivity. All three are drawn once per
  realization.
* b(t) = b_on + b_GM(t) + b_RW(t):
  * turn-on bias b_on ~ N(0, σ_on²), constant for one power-up;
  * bias instability as a first-order Gauss–Markov process,
    b_GM[k+1] = e^(−Δt/τ) b_GM[k] + σ_GM √(1 − e^(−2Δt/τ)) w_k;
  * bias random walk (rate / acceleration random walk K),
    b_RW[k+1] = b_RW[k] + K √Δt w_k.
* n: white noise of density N (angle / velocity random walk), per-sample std
  N/√Δt, where Δt comes from the message timestamps.
* Q: quantization to 1 LSB = 2·FS/2^bits. sat: clipping at ±FS.

The model uses four independent random streams (static and dynamic, for gyro
and accelerometer) derived from one seed. Changing gyro parameters therefore
leaves the accelerometer realization untouched. This gives common random
numbers for controlled parameter studies.

| Parameter (`gyro.*` / `accel.*`) | Unit (gyro / accel) | `consumer_mems` gyro | `consumer_mems` accel |
| --- | --- | --- | --- |
| `noise_density` | rad/s/√Hz / m/s²/√Hz | 8.727e-5 (0.005 °/s/√Hz) | 3.923e-3 (400 µg/√Hz) |
| `turn_on_bias_std` | rad/s / m/s² | 8.727e-3 (0.5 °/s) | 0.05 |
| `bias_instability_std` | rad/s / m/s² | 8.727e-5 | 5.0e-4 |
| `bias_correlation_time_s` | s | 100 | 100 |
| `bias_random_walk` | rad/s/√s / m/s²/√s | 1.0e-6 | 1.0e-4 |
| `scale_factor_std` | – | 0.01 | 0.01 |
| `misalignment_std_rad` | rad | 0.007 | 0.007 |
| `g_sensitivity_std` | (rad/s)/(m/s²) | 1.0e-4 | – |
| `full_scale` | rad/s / m/s² | 8.727 (±500 °/s) | 39.227 (±4 g) |
| `adc_bits` | – | 16 | 16 |

White-noise densities are MPU-6050 datasheet values. The other values are
representative of consumer MEMS after basic calibration. **Identify the IMU
that will be mounted on the car with `allan_analysis` and a multi-position
calibration, then write its own preset** before drawing quantitative
conclusions.

Outputs: `/apex/fusion/imu/data_raw` (`sensor_msgs/Imu`, `orientation_covariance[0] = -1`,
white-noise variances on the diagonals), plus the **true** total biases
`/apex/fusion/imu/true_gyro_bias` and `/apex/fusion/imu/true_accel_bias` for
evaluating bias-estimating filters later.

---

## 5. Strapdown INS

`core/strapdown.py` integrates the raw IMU in a local-level, flat,
non-rotating navigation frame (Gazebo world, ENU, g_n = [0, 0, −g]):

```
q_k = q_{k-1} ⊗ Exp(ω̄ Δt)
v_k = v_{k-1} + (R(q_mid) f̄ + g_n) Δt,      q_mid = q_{k-1} ⊗ Exp(ω̄ Δt / 2)
p_k = p_{k-1} + (v_{k-1} + v_k) Δt / 2
```

ω̄ and f̄ are bias-compensated and trapezoidally averaged over the interval
(`integration_scheme: trapezoidal`; `euler` uses the latest sample). Earth rate
and transport rate are neglected, because Gazebo does not simulate them and
the area is a few metres wide.

**Alignment** (`alignment.mode`):

* `static_coarse` (realistic, default): while ground truth confirms the car is
  stationary for `duration_s`, the INS levels itself from the mean specific
  force, estimates the gyro bias as the mean rate and estimates the accelerometer
  bias along gravity from |f̄| − g [6, §5.6]. Horizontal accelerometer bias
  cannot be distinguished from tilt at rest, so it is absorbed into the attitude
  (a fundamental observability limit). Position and heading come from the known
  start pose, since a 6-axis IMU cannot observe heading.
* `truth`: the full initial state is copied from truth and no bias is
  calibrated. This shows the raw effect of every error.

After alignment, ground truth is **never used again**. Expected error growth
for an unaided INS [6, 7]:

| Error source | Position error growth |
| --- | --- |
| Accelerometer white noise (VRW) | ∝ t^1.5 |
| Accelerometer bias b_a | b_a t² / 2 |
| Gyro white noise (ARW), through tilt | ∝ t^2.5 |
| Horizontal gyro bias ε (tilt ε t couples gravity) | g ε t³ / 6 |
| Vertical channel | unstable for any INS (`clamp_vertical_channel` optionally holds it) |

The unit tests check the t² and t³ laws against the integrator.

Service `~/reset` (`std_srvs/Trigger`) restarts the alignment. That's handy for
repeated experiments in one session:
`ros2 service call /fusion_strapdown_ins/reset std_srvs/srv/Trigger`.

---

## 6. Ground truth and evaluation

`fusion_truth` reads `/world/default/dynamic_pose/info`, which runs at about
60 Hz of simulation time. In that stream the model pose is in the world frame
and link poses are relative to the model, so the chassis pose is
T_world_link = T_world_model · T_model_link. The node applies the IMU lever arm
and publishes `/apex/fusion/truth/odom`, containing the pose, the world-frame
velocity from a backward difference, and the body rate.

`fusion_ins_error_monitor` interpolates the truth at every INS timestamp
(linear for position and velocity, SLERP for attitude) and computes position,
velocity and attitude errors. The attitude error is expressed both as Euler
differences and as the rotation vector Log(R_ins R_trueᵀ).

| Topic | Type | Content |
| --- | --- | --- |
| `/apex/fusion/imu/data_raw` | `sensor_msgs/Imu` | raw corrupted IMU |
| `/apex/fusion/imu/true_{gyro,accel}_bias` | `geometry_msgs/Vector3Stamped` | true total biases |
| `/apex/fusion/imu/realization` | `std_msgs/String` (JSON, latched) | config + constant errors drawn |
| `/apex/fusion/scan_noisy` | `sensor_msgs/LaserScan` | corrupted scan |
| `/apex/fusion/lidar/status`, `/realization` | JSON | outcome fractions, residual RMS, b and s |
| `/apex/fusion/truth/odom`, `/path` | `nav_msgs/Odometry`, `Path` | 6-DoF truth of the IMU frame |
| `/apex/fusion/ins/odom`, `/path` | `nav_msgs/Odometry`, `Path` | pure INS solution |
| `/apex/fusion/ins/status`, `/alignment` | JSON | state machine, alignment result (latched) |
| `/apex/fusion/ins/error` | `std_msgs/Float64MultiArray` | t, e_xyz, e_h, e_v, e_rpy (labels in layout) |
| `/apex/fusion/ins/error_summary` | JSON (1 Hz) | INS state and current / maximum errors |
| `/apex/fusion/viz/scan_on_{truth,ins}_pose` | `sensor_msgs/PointCloud2` | scan projected with each pose |
| `/apex/slam/<tag>/map` | `nav_msgs/OccupancyGrid` | map of SLAM instance `<tag>` (`good`, `noisy`, `good_imu`) |
| `/apex/fusion/scan_<tag>` | `sensor_msgs/LaserScan` | scan fed to SLAM `<tag>`, in frame `<tag>/laser` |
| `/apex/fusion/slam_<tag>/odom` | `nav_msgs/Odometry` | motion prior of SLAM `<tag>` |

### Recorded run

```
<run_dir>/
├── ins_error.csv       # 20 Hz: truth, INS, errors, true biases, distance travelled
├── metadata.json       # IMU and LiDAR realizations + configs, INS alignment result
├── launch_args.json
├── config/             # exact copies of the YAML presets used
└── ins_drift.png/.pdf  # written by plot_ins_drift
```

---

## 7. Tuning parameters

There are three equivalent ways to change a parameter:

1. **YAML preset.** Copy a file from `config/<kind>/`, edit it, and pass its path:
   `--imu-config ~/my_imu.yaml`. The nested YAML keys map to the dataclass fields.
2. **Launch arguments** for seeds: `--imu-seed 3 --lidar-seed 11`.
3. **At runtime:**
   ```bash
   ros2 param set /fusion_lidar_noise range_sigma_prop 0.01
   ros2 param set /fusion_imu_sensor gyro.noise_density 2.0e-4   # re-creates the IMU (power cycle)
   ros2 param set /fusion_strapdown_ins alignment.mode truth     # resets the INS
   ```
   `rqt_reconfigure` works as well. Invalid values, such as an unknown alignment
   mode, are rejected.

To add a new error term, add a field to the config dataclass, use it in the
model and add a test. The ROS parameter, YAML loading and metadata recording
pick it up automatically.

---

## 8. Analysis and validation tools

```bash
# Error accumulation figure + table at 10/30/60/120 s
ros2 run apex_fusion_research plot_ins_drift <run_dir> [--tmax 120] [--show]

# Allan deviation: validate the IMU model (synthetic) or identify a real IMU (CSV)
ros2 run apex_fusion_research allan_analysis --config consumer_mems --duration 7200
ros2 run apex_fusion_research allan_analysis --csv static.csv --time-col t \
    --gyro-cols gx gy gz --accel-cols ax ay az

# LiDAR model statistics on an analytic room (sigma(r) law, N(0,1) residuals, outcome rates)
ros2 run apex_fusion_research lidar_noise_report --config rplidar_like --scans 2000
```

Unit tests run without ROS:

```bash
cd simulation/ros2_ws/src/apex_fusion_research
/usr/bin/python3 -m pytest -q test
```

They cover rotation algebra, occupancy-grid export, map metrics and ICP,
the SLAM motion prior and the map evaluation. They also cover the LiDAR outcome probabilities, the σ(r) law,
incidence estimation on a wall, and reproducibility. For the IMU they cover the
white-noise level, constant errors, independent streams, quantization and
saturation, the Gauss–Markov stationary std, and Allan recovery of N and K. For
the INS they cover a perfect-IMU circle, the b·t²/2 and g·ε·t³/6 laws, and
static alignment.

---

## 9. SLAM baseline: good vs damaged sensors

This is the first capture of the errors. It records how a standard 2D SLAM
(`slam_toolbox`, the default ROS 2 library) responds to the damaged sensors,
so later fusion results can be compared against it. Every instance uses the
same slam_toolbox tuning (`config/slam/slam_toolbox_2d.yaml`) and runs in its
own TF tree `<tag>/map -> <tag>/odom -> <tag>/base_link -> <tag>/laser`. The
instances differ only in their inputs:

| Tag | LiDAR | Motion prior (`<tag>/odom -> <tag>/base_link`) | Role |
| --- | --- | --- | --- |
| `good` | ideal Gazebo scan | ground-truth pose (`full_pose`): ideal odometry, scan matching and loop closing off (as in the APEX ideal mapping mode) | reference map ("good sensors") |
| `noisy` | noisy scan (`rplidar_like`) | yaw of the INS on the noisy IMU, zero translation (`heading_only`) | damaged sensors |
| `good_imu` | ideal Gazebo scan | yaw of an INS on the ideal IMU, zero translation | ablation: separates the heading-only method limit from the sensor noise |

With a heading-only prior the translation comes entirely from LiDAR scan
matching. That is the usual low-cost LiDAR + IMU setup without wheel
odometry, and it is the gap a LiDAR–IMU fusion method has to close. The car
itself keeps driving with the APEX recognition-tour controller, unchanged.

### Running it

```bash
./simulation/tools/sim/apex_fusion_slam_capture.sh [--run-name NAME] [--no-ablation] [-- LAUNCHER OPTIONS]
# e.g. another noise realization:  apex_fusion_slam_capture.sh -- --imu-seed 3 --lidar-seed 11
```

The script runs everything headless:

1. It launches Gazebo and the stack with `--slam --record-measurements`.
2. It arms the tour once both INS alignments are done.
3. It waits for the end of the tour.
4. It stops the stack with SIGINT, so every recorder writes its final files.
5. It writes the figures.

It runs `ros2 run apex_fusion_research plot_slam_maps <run_dir>` itself. The
same command regenerates the figure and metrics from any earlier run.

### Output

```
<run_dir>/
├── slam/
│   ├── track_truth_points.csv      # real track walls (world): x_m, y_m
│   ├── truth_trajectory.csv        # true base_link pose: t, x, y, yaw
│   ├── map_<tag>_points.csv        # occupied cells: x_map, y_map, x_world, y_world, occupancy
│   ├── map_<tag>.pgm / .yaml       # map_server format
│   ├── slam_<tag>_trajectory.csv   # t, x_map, y_map, yaw_map, x_world, y_world, yaw_world
│   ├── snapshots/map_<tag>_t<s>.csv  # map every 10 s (map growth)
│   ├── alignment.json              # T_world_map of every SLAM (initial-pose anchor)
│   └── slam_summary.json
├── measurements_noisy/             # raw damaged-sensor measurements (rc_sim_description recorder)
│   ├── lidar_points.csv            # one row per beam: stamp, angle, range, x/y in the sensor frame
│   ├── scan_index.csv, imu_raw.csv, odom_fused.csv (= INS), ground_truth_odom.csv
│   └── track_geometry.csv          # real track walls
├── measurements_ideal/             # same files for the ideal sensors
├── slam_maps.png / .pdf, slam_metrics.json / .csv
└── ins_error.csv, ins_drift.png, metadata.json, config/ ...
```

### Evaluation

- **Anchoring.** Each SLAM map frame is anchored to the world with the true
  pose at its first update only (`alignment.json`). All later drift and
  distortion count as error.
- **Reference.** The reference is the observed track: the real wall points
  within 0.10 m of an ideal LiDAR return projected with the true pose. Walls
  the car never saw do not penalise coverage.
- **Anchored metrics** judge the map in the world:
  - *precision* is the fraction of map cells within 0.10 m of the reference;
  - *coverage* is the fraction of the reference within 0.10 m of the map;
  - *chamfer* is the mean of both nearest-neighbour distances.
- **Shape metrics** repeat the comparison after a rigid point-to-line ICP of
  the map onto the track.
- **ATE** is the RMS position error of the SLAM trajectory against the true
  base_link trajectory.

### Sample result: slam_toolbox baseline response

![slam_toolbox baseline response: good vs damaged sensors](doc/images/slam_baseline_response.png)

Run `slam_run3`: baseline scenario, IMU `consumer_mems` (seed 42), LiDAR `rplidar_like` (seed 7), 7.3 m driven.

| SLAM | precision | coverage | chamfer [m] | shape chamfer [m] | ATE rmse [m] | final error [m] |
| --- | --- | --- | --- | --- | --- | --- |
| good sensors (reference) | 0.99 | 0.85 | 0.043 | 0.043 | 0.00 | 0.00 |
| damaged sensors | 0.44 | 0.53 | 0.470 | 0.248 | 2.61 | 4.35 |
| ideal LiDAR + ideal-IMU heading | 0.40 | 0.49 | 0.478 | 0.229 | 2.56 | 4.28 |

Observations from this first capture:

- **The reference is sound.** The good-sensor map lies on the real walls (chamfer 4 cm, about one cell).
- **The damaged-sensor SLAM compresses the track.** Its trajectory is 3.6 m long against 7.3 m driven, about 50 %, and it ends 4.3 m from the true position. The map is squeezed along the corridor, so the curve appears 1–2 m too early.
- **The method limit dominates the sensor noise.** The ablation, with ideal LiDAR and ideal IMU, is almost as wrong as the damaged sensors. The loss comes from the heading-only prior combined with the degenerate corridor geometry. Sensor noise only adds scatter: the damaged map has more stray cells and a worse shape chamfer, 0.248 m against 0.229 m.
- **Why the translation is lost.** slam_toolbox's correlative scan matcher favours poses whose points fall on already-mapped cells. In a 1.1 m wide corridor, motion along the track is weakly observable, so this biases the estimate towards too little motion.
- **Even a perfect prior is affected.** A preliminary run fed the ground-truth prior with scan matching on (`slam_run2`). Its position still fell 2.8 m behind the true one after 7.2 m, with an ATE of 1.56 m. That is why the reference disables scan matching.

Providing observable translation, through IMU-aided velocity or motion
constraints, is therefore the first target for the LiDAR–IMU fusion work.

### Limitations of this baseline

- **Partial lap.** The recognition tour stops at its 60 s wall-clock timeout
  (`global_timeout_s` in `apex_params.yaml`). At the simulation's real-time
  factor that covers about 7–9 m of the 31.7 m lap. Only that part of the
  track is mapped.
- **Anchoring.** It uses the true initial pose. The shape metrics after ICP
  remove only a rigid offset.
- **Map saving.** `slam_toolbox`'s own `save_map` needs `nav2_map_server`,
  which is not installed. Maps are exported from the map topic instead.
- **Run length.** One run takes about 8 minutes of wall time and about 50 MB of data.

## 10. Reproducibility

* Every stochastic element is seeded (`seed` in each YAML, overridable at
  launch). The same seed and configuration reproduce the same sensor errors.
* Each run stores the exact YAML files, the launch arguments, and the realized
  constant errors (`metadata.json`).
* Timing uses message timestamps (simulation time), not wall-clock time, so
  results do not depend on the real-time factor of the machine.

## 11. Assumptions and limitations

* Gazebo samples the IMU instantaneously at the sensor rate (about 120 Hz, with
  timestamps rounded to the 1 ms physics step). A real IMU low-pass filters and
  decimates internally, so vibration aliasing differs.
* Earth rotation, temperature effects and non-linear scale factor are not modelled.
* The simulated scan is instantaneous, so there is no motion distortion (skew)
  within a revolution yet. That is the next LiDAR model extension, and IMU-based
  de-skewing is a natural fusion contribution.
* Ground truth is limited by Gazebo's pose stream at about 60 Hz. Truth velocity
  is a backward difference.
* Preset values are representative, not identified: see sections 3 and 4.
* The 2D LiDAR is assumed to stay parallel to the ground. Chassis roll and pitch
  are not propagated into the scan geometry.

## 12. Roadmap

1. ~~Identify the real IMU and LiDAR, then create hardware presets.~~ Done: `APEX_real` (section 13).
2. ~~Model LiDAR motion distortion.~~ Done: `core/lidar_rolling.py` (section 13).
3. Loosely coupled error-state Kalman filter: INS prediction plus LiDAR scan
   matching updates, with bias estimation checked against `true_*_bias`.
   Evaluate it against the SLAM baseline of section 9.
4. Complete laps: raise the recognition-tour timeout for the baseline
   captures, so loop closure and full-track coverage enter the comparison.
5. Monte-Carlo campaigns over seeds and parameter sweeps, with NEES/NIS
   consistency and RMSE statistics.

## 13. Real2sim: Gazebo-native sensors and the learned odometry in closed loop

**Goal.** Run the car in Gazebo the way the real car will run, as the sensors
and the software stack of the car see it:
- Gazebo measures with its own sensors, and the realism layer (`core/lidar_rolling.py`, `core/imu_chip.py`) makes the measurement look like the real hardware.
- The learned LiDAR-inertial odometry (`learning/lidar_imu_pose/live_odometry.py`) estimates the pose.
- `slam_toolbox` builds the map on it.
- The car drives a pose-dataset format (track, motion profile, seed, laps) using only that estimate.

The same realism layer produced the training data (`tools/pose_dataset/README.md`, §13).

```
Gazebo (track world, car with suspension, gpu_lidar capture 26 Hz, 1 kHz IMU with Gazebo noise)
  -> real_sensor_node   A2M8 revolution (13 Hz, clockwise from +89 deg, stamp = first sample) on /lidar/scan_raw
                        LSM6DS3 sample (104 Hz, stamp = sampling instant) on /apex/imu/data_raw
                        (published when the device would deliver them, simulation clock)
  -> learned_odometry_node (learning/.venv: torch)
        TF odom_learned -> base_link at every scan stamp, /apex/odometry/learned (sigma -> covariance),
        /apex/odometry/learned_predicted (IMU-propagated, for control), /apex/learned_odometry/scan_deskewed
  -> slam_toolbox (config/slam/slam_toolbox_learned.yaml): /map, TF map -> odom_learned
  -> track_driver_node  T_world_map (start pose) * map->odom * predicted pose -> pure pursuit + speed plan
  -> /apex/cmd_vel_track (Twist, the real car's interface) -> sim_actuation_node (dataset ESC / servo) -> Gazebo
run_referee_node (truth only): lap completed / collision / off-track -> run_result.json, real walls for the map evaluation
```

**What the car knows.**
- **Calibration:** only its nominal calibration (`base_link -> laser` at 0.18/0/0.12 and `-> imu_link` as static TFs). The simulated true mounts differ by the drawn calibration error.
- **Start pose:** by default (`--start-pose true`) the path is laid out from where the car stands, as when it plans in its own map; the truth is used once and never updated. `--start-pose nominal` uses the start line instead, and the seeded placement error (up to 2°) is then unknown to it.
- **Truth:** only the referee and the evaluation see it.

**Frequencies.**
- **Physics:** 1 kHz.
- **Sensors:** IMU at 104 Hz, LiDAR at 13 Hz.
- **Estimator:** one update per revolution, once the IMU covers it. Compute is ~20 ms on the CPU with 4 threads, the default; the GPU is slower for one car.
- **Latency:** the corrected pose is available ~100 ms after its stamp (sweep + compute). The predicted pose follows the IMU.
- **SLAM:** ≤ 5 Hz.
- **Driver:** 50 Hz.
- **Clock:** every node runs on the simulation clock.

```bash
./simulation/tools/sim/apex_real2sim_up.sh --track val_mixed --motion medium --seed 1 --laps 2
# --checkpoint <file>           default: learning/outputs/real2sim_v2/hybrid_submap/best_model.pt (trained on APEX_real)
# --motion slow_variable        real-car speeds
# --sensor APEX_real_compat     A2M8 in the 2 kHz protocol
# --drive truth                 drive on the exact pose (diagnostic: estimator without the closed loop)
# --gui / --no-rviz             Gazebo GUI, no RViz
```

**Outputs.** Each run writes `simulation/data/real2sim/<run>/`:
- **Evaluation:** `run_result.json` (referee), `evaluation.json` / `evaluation.png`.
  - Odometry against the truth at the true revolution starts, with the offline metrics (speed error, RPE, segment drift, σ coverage).
  - SLAM map metrics, latencies, the achieved real-time factor and the received sensor timing.
- **Logs:** `estimator.csv`, `driver.csv`, `truth_scans.csv`, `truth_track.csv`, `imu_raw.csv`, `lidar_scans.csv` (the published revolutions, one row each).
- **Sensor data and map figures:** `learning/.venv/bin/python tools/analysis/plot_real2sim_sensors.py <run_dir>` writes `sensores.png` (IMU and LiDAR as the car received them) and `mapa.png` (the car's occupancy grid, and the same map in the world over the real walls with the true, SLAM and odometry-only trajectories).
- **SLAM:** `slam/` (maps, trajectories) and `slam_metrics.json` (`plot_slam_maps`).

**Validation with the retrained model** (`real2sim_v2/hybrid_submap`, `val_mixed`, a track never seen in training, seed 1, 2 laps, `APEX_real`, CPU).

| | `medium` (fast, 3.3 m/s cruise) | `slow_variable` (0.3–1.5 m/s) |
| --- | --- | --- |
| Referee | both laps completed (103.5 of 103.7 m), no collision | both laps completed (102.8 of 103.3 m), no collision |
| Max lateral error (truth) | 0.29 m | 0.15 m |
| Belief error of the driver, p50 / p95 | 0.13 / 0.25 m | 0.10 / 0.16 m |
| Odometry: speed error, segment drift | 4.36 cm/s, 0.85 % (offline, validation runs of this format: 0.87 %) | 1.15 cm/s, 0.63 % (offline, same: 0.59 %) |
| SLAM map, anchored: precision / coverage / chamfer | 0.65 / 0.84 / 8.9 cm | 0.75 / 0.75 / 6.9 cm |
| SLAM map, shape (after rigid ICP): precision / coverage / chamfer | 0.74 / 0.83 / 8.0 cm | 0.97 / 0.94 / 3.8 cm |
| SLAM ATE (RMSE) | 0.15 m | 0.10 m |
| Estimator latency p50 / p95 (scan stamp → estimate) | 152 / 176 ms | 155 / 178 ms |
| Predicted pose age used by the driver, p95 | 32 ms | 33 ms |
| Real-time factor | 0.46 | 0.51 |
| Timing audit | passed | passed |

- **Live equals offline.** The drift of the live odometry matches the offline evaluation of the same format; the estimator reproduces the batch path exactly (`learning/lidar_imu_pose/tests/test_live_odometry.py`).
- **Anchored against shape.** The anchored map is offset by a rigid 0.2–0.6° and a few centimetres. The car only knows its nominal LiDAR mount, and the drawn mount error of these runs is 0.35–0.77° in yaw; the early heading drift adds to it. The shape of the map is good, with 3.8 cm chamfer at slow speed.
- **Latency.** About 80 ms is the revolution itself plus its publication, and about 10 ms is the IMU that must cover the revolution. The rest is the network and the ICP, 62–64 ms (wall clock) while Gazebo, the sensor layer and the SLAM share the CPU; alone they take ~20 ms. Control is not affected: the driver uses the IMU-propagated pose, compensated for its age.

**Timing audit.** `tools/analysis/sensor_timing_audit.py <dir> --profile APEX_real` checks a recording against the timing the model expects. Simulated runs pass it; today's real-car recordings do not (see `real_vehicle/docs/sensor_timing_requirements.md`, recommendations only).

**Tests.** `test/test_realism_layer.py` checks:
- the chip rate and output noise density;
- the bias labels;
- the rolling emulation against exact per-sample ray casting;
- clockwise order, blind sector and stamps.

`tools/pose_dataset/tests/test_suspension.py` checks the sprung-body response.

## References

1. S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005 (ch. 6).
2. Slamtec, *RPLIDAR A2 Datasheet*.
3. C. Ye, J. Borenstein, "Characterization of a 2-D Laser Scanner for Mobile Robot Obstacle Negotiation", *Proc. IEEE ICRA*, 2002.
4. *IEEE Std 952-1997*, Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros (Annex C: Allan variance).
5. *IEEE Std 1293-2018*, Specification Format Guide and Test Procedure for Linear Single-Axis, Nongyroscopic Accelerometers.
6. P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., Artech House, 2013.
7. D. H. Titterton, J. L. Weston, *Strapdown Inertial Navigation Technology*, 2nd ed., IET, 2004.
8. N. El-Sheimy, H. Hou, X. Niu, "Analysis and Modeling of Inertial Sensors Using Allan Variance", *IEEE Trans. Instrum. Meas.*, 57(1), 2008.
9. InvenSense, *MPU-6000/MPU-6050 Product Specification*, rev. 3.4.
