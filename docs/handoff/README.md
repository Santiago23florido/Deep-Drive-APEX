# Real-vehicle data collection — run sheet and handover spec

The method, the trained model and the deployment are frozen and documented.
What the paper still needs is **physical data**. This page is the complete
instruction set: the commands to execute, what to measure, and what to upload.

**You run and upload. You do not analyse.** No metrics, no figures, no
conclusions, no interpretation of whether a run was "good" — those are produced
on the receiving side from the raw outputs. A failed run uploaded with an honest
`handover.json` is worth more than a successful run without one.

Companion files: [`remaining_results.json`](remaining_results.json) (the same
plan, machine-readable) and [`handover_template.json`](handover_template.json)
(copy it into every run directory and fill it in).

## 1. Frozen configuration — do not change it

| Item | Value |
| --- | --- |
| Runtime commit / tag | `f907cce` / `real-car-calibrated-20261024-220316` |
| Default profile | `tracking` |
| Weights (unchanged) | `real2sim_v7e_standstill/hybrid_submap_v4` — `best_model.pt` `91ce3790…`, `best_model.onnx` `eff4ba0e…` |
| Reference example of a complete run directory | [`20261024_220316`](../../real_vehicle/docs/paper/20261024_220316/) |

Do not edit source, YAML, thresholds or weights. If something had to change to
make the car run at all, do not hide it: record the exact file and reason in
`handover.json` and say so in the pull request.

## 2. One-time setup

```bash
# From the repository root on the PC. Confirm access without touching the car.
./real_vehicle/tools/pc/race_pi.sh --host ensta@raspberrypi status
./real_vehicle/tools/pc/race_pi.sh --host ensta@raspberrypi --dry-run run

# Deploy the frozen runtime and the exact v7e weights (car stopped, no container).
./real_vehicle/tools/core/sync_apex_code_to_pi.sh \
  ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/ \
  --exclude '/docs/' --exclude '/data/'
APEX_PI_HOST=ensta@raspberrypi \
APEX_ODOM_CKPT_DIR=simulation/learning/outputs/real2sim_v7e_standstill/hybrid_submap_v4 \
  ./real_vehicle/tools/core/sync_odom_rt_to_pi.sh
```

Then build the track reference and the camera rig once, as described in §5.
Full command reference: [real-vehicle README](../../real_vehicle/README.md#from-the-pc-ssh-preflight-and-recognition-to-slam-run).

## 3. Runs to collect

Every `run` moves the car. Clear the track, car at the marked start pose.

| # | Purpose | Command | Repetitions | What changes between repetitions |
| --- | --- | --- | --- | --- |
| S1 | Repeatability of the accepted configuration | `./real_vehicle/tools/pc/race_pi.sh --host ensta@raspberrypi run` | **5** | Nothing. Same track, same start pose, fresh battery state logged each time |
| S2 | Controller A/B against the retained profile | `./real_vehicle/tools/pc/race_pi.sh --host ensta@raspberrypi --profile approved run` | **5** | Nothing. Alternate order with S1 (S1, S2, S1, S2, …), not five of one then five of the other |
| S3 | A track the controller was never tuned on | same command as S1 | **3** | Rearrange the poles into a different layout **before** the first run and never touch them again. Photograph and measure the new layout as in §5 |
| S4 | Longest straight that fits the room | same command as S1 | **3** | Layout with the longest possible straight, so the onboard identifier accumulates more valid updates. Measure and photograph this layout too |
| S5 | Stationary reference, car never moves | `./real_vehicle/tools/pc/race_pi.sh --host ensta@raspberrypi preflight` | **2** | None. Leave the car parked and untouched for the whole check; it never arms |
| S6 | Hand-driven capture with varied motion | teleoperated, see below | **3** captures of 2–3 min | Nothing in the setup; the variety is in how you drive it |

**S6 in detail.** The autonomous runs all drive the same oval at nearly constant
speed, which is not enough to characterise the estimator. S6 fills that in, and
it is the lowest-risk session: no mission, no autonomy, you drive by hand and
slowly. The estimator and raw logging run while the autonomous planners stay
off, through `tools/core/apex_manual_control_up.sh` on the Pi, driven from
`real_vehicle/tools/pc/run_windows_gamepad_ros_bridge_pc.sh`. **This path has
not been exercised recently: bring it up once and confirm sensors, estimator and
steering respond before recording. If it does not come up, say so in the pull
request and skip the session — do not modify the scripts to force it.**

Each capture must happen entirely inside the camera's field of view, with the
video running, and must contain: at least 3 complete stops of 5 s or more; one
run along the longest available straight at roughly constant speed; at least one
left and one right turn; and a return to the start marks, with the final pose
measured as in §4.

**Minimum useful delivery** if time or batteries run out: the §5 track reference
+ S1 + S5. Stop there and upload rather than skipping §5 or the per-run
measurements. Partial sessions are fine; undocumented ones are not.

**Not your job, and do not improvise:** switching the learned estimator off,
disabling online adaptation, or disabling the standstill hold would each need a
runtime switch that does not exist today. Those comparisons are made in
simulation and by replaying your recorded sensor data offline, which is why the
raw capture and the track dimensions matter more than any extra on-car variant.
Do not edit `start_apex_pipeline.sh` or the YAML to emulate them.

## 4. Per-run procedure — every run, including failures

1. Mark the start pose on the floor with tape: two marks for the rear axle ends,
   one for the nose. Photograph the marks once per layout.
2. Start the video (§5) **before** launching, and keep it running until the car
   is fully stopped.
3. Launch the command from the table. Do not touch the car during the run.
4. Wait for the line **"RAM evidence copied and SHA256-verified after stop"**.
   An SSH exit on its own is not a finished run.
5. Before moving the car, measure its final pose against the start marks:
   longitudinal offset, lateral offset, heading offset. Photograph it in place.
6. Note anything you observed: contacts, scrapes, stalls, when you intervened,
   battery swaps, poles that shifted. Plain facts, no judgement.
7. Fill `handover.json` from the template, then pull the data (§6).

Keep every attempt — aborted, stopped, crashed, "looked wrong". Deleting a run
destroys the success rate the paper reports.

## 5. Track reference and video — this is what makes accuracy possible

Yes: **the measured pole lengths plus photos are enough** to build a real
external reference, and with one fixed camera they are enough for a full
trajectory reference. Two tiers, both worth collecting:

**Tier A — dimensions and photos (mandatory).** Known track geometry in metres
gives a metric ground-truth map: map scale error, loop-closure error, and true
mean speed per lap from true path length ÷ lap time. Deliver:

- Length of every pole/segment in metres, measured with a tape to ±1 cm.
- A hand sketch of the layout with those lengths and the lane width written on
  it; a phone photo of the sketch is fine.
- **Pole height above the floor, cross-section shape and width/diameter, and
  material or surface finish** (wood, plastic, painted, matte, shiny). State
  whether every pole is identical, and list the exceptions if not.
- 4+ photos of the assembled track from different corners, plus the highest
  viewpoint available, with a tape measure or a known-length object visible in
  at least one of them.
- Floor surface type, and a note of anything that moved mid-session.

These dimensions are used for two things: the metric reference above, and
rebuilding each layout in the simulator so that simulated and physical runs are
compared on the same geometry. Height, cross-section and material matter because
they determine what the LiDAR returns. If a layout cannot be reconstructed from
what you upload, the comparison is lost — when in doubt, measure and photograph
more.

**Tier B — one fixed camera (adds trajectory accuracy, ~1–3 cm).** With the
known pole positions as the calibration target, a static video yields the car's
position in every frame, which is what ATE/RPE needs. Requirements, all of them:

- A phone on a tripod, **immobile and without zoom** for the entire session; the
  whole track in frame. 1080p / 30 fps minimum. If it gets bumped, say so.
- At least 6 known floor points (pole bases or corners) visible in every frame.
- A flat high-contrast marker taped on top of the car — a printed ArUco tag, or
  a thick black cross on white A5. Measure and report: **marker height above the
  floor**, and the marker's offset from the LiDAR centre (forward, lateral).
- The camera's approximate position and height relative to one identified pole
  corner, ±5 cm is enough.
- Time sync: with the camera already recording, film a laptop or phone screen
  running `date -Ins` for ~5 s, then leave the camera untouched. Also note the
  wall-clock launch time of each run.
- One continuous file per run if possible. Do not re-encode or downscale the
  original.

Without Tier B there is no per-timestamp accuracy — a still photo carries no
information about where the car was at a given instant. Tier B is a phone and a
tripod; no motion-capture system is required.

## 6. Pulling the data off the Pi

```bash
RUN=<run_id>        # the timestamp directory the run printed, e.g. 20261130_193045
mkdir -p real_vehicle/docs/paper/$RUN/raw
rsync -avz --checksum \
  ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/ros2_ws/apex_race/$RUN/ \
  real_vehicle/docs/paper/$RUN/raw/
rsync -avz --checksum \
  ensta@raspberrypi:/home/ensta/AiAtonomousRc/APEX/data/diagnostics/20261022_open_segment/ \
  real_vehicle/docs/paper/$RUN/raw/diagnostics/
# Integrity manifest, so the upload can be verified on arrival.
( cd real_vehicle/docs/paper/$RUN && find raw -type f -print0 \
  | sort -z | xargs -0 sha256sum > sha256sum.txt )
```

Nothing is deleted from the Pi until the upload is confirmed.

## 7. What to upload, and how

One directory per run, raw files only — no plots, no derived tables:

```
real_vehicle/docs/paper/<run_id>/
  handover.json        # copied from handover_template.json and filled in
  raw/                 # exactly what came off the Pi, unmodified
  sha256sum.txt        # manifest produced above
  track/               # sketch, photos, pole lengths (once per layout, or a note naming the layout)
  physical/            # start/stop pose measurements and the photos of the marks
```

```bash
git checkout -b data/<run_id>-<session>      # e.g. data/20261130_193045-s1
git add -f real_vehicle/docs/paper/<run_id>  # -f is required: *.csv is ignored repo-wide
git commit -m "Upload raw physical run <run_id> (<session>)"
git push -u origin data/<run_id>-<session>
```

Open a pull request to `main`. In the description, list only facts: which
session, which command, attempts versus completions with stop reasons, anything
that changed, and anything you are unsure about. No conclusions.

**Video does not go into git** — GitHub rejects files over 100 MB. Upload the
originals to a shared folder, and in `handover.json` put the link plus each
file's `sha256sum`. Batches of runs can share one pull request; several small
pull requests are better than one large late one.

## 8. Hard rules

- Retain every attempt, with its stop reason.
- Never weaken a confidence gate, clearance limit, speed cap or the recovery
  behaviour to obtain a cleaner run.
- Never retrain or replace the weights; the hashes above must keep matching.
- Do not rearrange the track mid-session, except where S3 and S4 say to, and
  then only before the first run of that session.
- Report the onboard map and estimated trajectory as onboard estimates. The
  external reference is the measured geometry and the video, nothing else.
- When in doubt, upload more raw data and describe the doubt in `handover.json`.
