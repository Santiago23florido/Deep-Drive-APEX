# Zero-shot variants (held out of every dataset split)

These variants are defined and reproducible, but **they are not in
`pose_dataset.sqlite3`**, in no split: not in train, validation or test. They
are kept for zero-shot evaluation after training and model selection. They
measure generalization to domains the model and its hyper-parameter search
have never seen.

## Protections

- Every variant carries `held_out: true` in its config file (`tracks.yaml`, `motion.yaml`, `sensors.yaml`).
- The `smoke` and `full` presets have `role: main`, and `build_plan` raises if a held-out track, motion or sensor appears in them.
- A database stores its `role` in `dataset_meta`. A `main` database refuses `zeroshot` runs, and the reverse.
- `campaign_report.json` → `held_out_variants_in_database` lists any held-out variant found in the database. It must be empty.
- `tests/test_tracks_campaign.py::test_main_presets_exclude_held_out_variants` checks the presets.

## Variants

| Name | Kind | What is new | Why it tests generalization |
| --- | --- | --- | --- |
| `zs_real_track_replica` | track | Estimated real-scale replica of the APEX track (`docs/reports/Piste.jpeg`): about 8 × 5.5 m, 1.1 m corridor, 1.0 m corners at the steering limit, tables and people around it. v_max 2.5 m/s. | Sim-to-real proxy. It shares the L-boot topology with the training track `boot_lshape` (×1.5, mirrored) but has a narrower corridor, tighter corners and different surroundings. Dimensions are estimated from the photo, not measured. |
| `zs_warehouse_aisles` | track | A long loop between repetitive, nearly parallel walls with few features. | Degenerate LiDAR geometry: translation along the aisle is weakly observable, so the IMU must carry it. |
| `zs_roundabout_city` | track | Right-left-right roundabout-like sequence, junctions at non-90° angles, plaza curbs. | New curve combinations absent from the urban training track. |
| `ZS_D_dense_270` | sensors | LiDAR 15 Hz, **720 beams over 270°** (rear blind sector), 25 m range; IMU **500 Hz** mounted **rotated 90° in yaw**. | New beam count, field of view and rates, and an IMU axis convention never seen. The model must use the extrinsics. |
| `ZS_E_degraded_offset` | sensors | LiDAR 8 Hz, 240 beams, 10 % missing returns, 5 % lost scans; IMU 50 Hz with very large biases; **15 ms LiDAR–IMU time offset**. | Extreme degradation and an uncompensated time offset. |
| `zs_creep` | motion | 12–22 % of v_max with gentle speed changes. | Speeds far below training: near-static IMU and tiny inter-scan motion. |
| `zs_weave` | motion | ±0.25 m sinusoidal weaving inside the lane at 65 % of v_max. | Lateral excitation and yaw-rate oscillations absent from training laps. |
| `low_friction` | dynamics (variant of `test_unseen` and `oval_asym`) | Floor friction µ = 0.5 instead of 1.2, speeds scaled by 0.65 (≈ √(0.5/1.2)). | Different tyre slip and vehicle dynamics for the same geometry. |

## Generating them (separate file only)

```bash
cd simulation
python3 tools/generate_pose_dataset.py --preset zeroshot --estimate-only
python3 tools/generate_pose_dataset.py --preset zeroshot --headless \
  --database data/multiscenario_pose/zeroshot_eval.sqlite3
```

The zero-shot database uses the same schema and loader. Every run has
`split = 'test'` and the database `role` is `zeroshot`. Load it with
`PoseSequenceDataset(path, "test")`. The `zeroshot` preset contains:

- the three zero-shot tracks with the known sensors and motions (A/B/C × low/medium/variable/stop_and_go, 2 seeds);
- the zero-shot sensors on the validation and test tracks;
- the zero-shot motions on `test_unseen` and `boot_lshape`;
- the low-friction variant.

All generators are seeded, so the file is reproducible. The main campaign
never simulates the zero-shot tracks, and never uses seeds 301–303 of the
sensor, motion and friction variants.

## Suggested protocol

1. Train on `train` only. Select checkpoints and hyper-parameters on `validation` only.
2. Report `test` (unseen track + unseen sensor/speed combinations).
3. Freeze everything, then generate `zeroshot_eval.sqlite3` and report each variant separately. Do not tune on it.
4. For `ZS_D_dense_270` (different beam count) the model must accept variable beam counts or resample using `angle_min`/`angle_increment`. `collate_pose_batch` pads beams and marks the padding in `scan_valid`.
