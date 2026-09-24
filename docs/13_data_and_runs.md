# Data and Runs

## Purpose

Runs make failures reproducible and allow algorithms to be compared against prior versions or simulation ground truth.

## Main Locations

| Path | Content |
| --- | --- |
| `simulation/data/` | Simulated captures, maps, reconstructions, and results. |
| `real_vehicle/data/` | Captures and analyses from physical hardware. |
| `archive/artifacts/` | Historical images and logs. |
| `archive/generated/` | Archived builds, caches, and runtime material. |

## Simulation Data

A run may contain capture metadata, LiDAR and IMU data, estimated odometry, ground truth, planner paths, control commands, maps, reconstruction output, JSON summaries, and plots.

Example:

```text
simulation/data/rc_sim_description/runs/lap_manual_01/
└── offline_reconstruction/
    └── reconstruction_overview.png
```

## Multi-Scenario Pose Dataset

`simulation/tools/generate_pose_dataset.py` runs Gazebo headless and in-process on generated tracks and stores noisy LiDAR/IMU streams with the exact simulator pose in one SQLite file (`simulation/data/multiscenario_pose/`, ignored by Git). Splits are made per trajectory and per whole track; zero-shot variants are kept out of the database. See [the generator README](../simulation/tools/pose_dataset/README.md) and [ZERO_SHOT.md](../simulation/tools/pose_dataset/ZERO_SHOT.md).

```bash
cd simulation
python3 tools/generate_pose_dataset.py --preset full --headless \
  --database data/multiscenario_pose/pose_dataset.sqlite3 --resume
```

## Real-Vehicle Data

`real_vehicle/data/` contains LiDAR, IMU, PWM, curve, and recognition-tour captures. The real vehicle has no native ground truth; comparisons require an external reference or a matched simulation.

## Interpreting Runs

Check the scenario or track, initial pose, parameters, commit, recorded topics, duration, rates, frames, and units before comparing results.

## Data Hygiene

- Do not treat `build/`, `install/`, `log/`, or virtual environments as run data.
- Keep metadata beside each run.
- Separate simulated and physical data.
- Do not rewrite historical captures merely to update old absolute paths.
- Store non-operational artifacts under `archive/`.

## Related Documentation

- [Mapping and Recording](18_mapping_and_recording_pipeline.md)
- [Gazebo Simulation](08_simulation_gazebo.md)
- [Known Limitations](16_known_limitations_and_legacy_parts.md)
