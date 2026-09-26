"""The live (message-by-message) odometry reproduces the batch evaluation.

A recorded validation run is replayed through ``LiveOdometry`` as a robot
would receive it (IMU samples and complete revolutions with their stamps) and
compared with the path used to train and evaluate the network (``run_icp`` +
``stream_predict`` over the stored split). Needs the SQLite dataset and the
checkpoint (skipped otherwise); runs on the CPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridsearch_sqlite import DEFAULT_DB  # noqa: E402
from icp_odometry import ICP_VARIANTS, run_icp  # noqa: E402
from live_odometry import MAX_IMU, LidarCalibration, LiveOdometry  # noqa: E402
from sqlite_streams import StreamData, load_split  # noqa: E402
from sqlite_windows import imu_group_delay_s  # noqa: E402
from streaming_model import load_checkpoint  # noqa: E402
from train_streaming import CACHE, OUT, stream_predict  # noqa: E402
from pose_dataset.blobs import decode_mask, decode_ranges  # noqa: E402

RUN = "val_mixed__medium__s1__A_nominal"
CKPT = OUT / "hybrid_submap" / "best_model.pt"
V2_CKPTS = (OUT.parent / "real2sim_v2" / "hybrid_submap" / "best_model.pt", CKPT)  # trained on APEX_real, else v1
V2_DBS = (DEFAULT_DB.with_name("pose_dataset_v2.sqlite3"), DEFAULT_DB.with_name("check_v2.sqlite3"))
# v1 (A_nominal: counter-clockwise from -pi, beam i at i * dt) and v2 (the
# real A2M8: clockwise from +89 deg, per-beam time from the calibration).
CASES = [((DEFAULT_DB,), RUN, (CKPT,)), (V2_DBS, "val_mixed__medium__s1__APEX_real", V2_CKPTS)]


def _database_with(dbs: tuple[Path, ...], run_key: str) -> Path | None:
    """First database that holds ``run_key`` as a completed validation run."""
    for db in dbs:
        if db.exists():
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            hit = conn.execute("SELECT 1 FROM runs WHERE run_key = ? AND split = 'validation' AND status = 'completed'", (run_key,)).fetchone()
            conn.close()
            if hit:
                return db
    return None


def _single_run(data: dict, run_key: str) -> dict:
    i = next(k for k, r in enumerate(data["runs"]) if r["run_key"] == run_key)
    r = data["runs"][i]
    sel = np.arange(r["offset"], r["offset"] + r["n"])
    n_total = data["target"].shape[0]
    out = {k: (v[sel] if torch.is_tensor(v) and v.shape[:1] == (n_total,) else v) for k, v in data.items()}
    for k in ("range_max", "angle_min", "angle_increment", "time_increment_s", "lidar_xy", "lidar_sigma", "beam_time_frac"):
        if k in data:
            out[k] = data[k][i : i + 1]
    out["runs"] = [{**r, "offset": 0}]
    out["run_index"] = torch.zeros(r["n"], dtype=torch.int64)
    return out


def _fast_motion_checkpoint(ckpt: Path, tmp: Path) -> Path:
    """The same weights with the fast-motion options on (submap v3 ICP, gyro
    profile in the network, IMU filter delay): the live and batch paths
    must still agree."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck["config"] = {**ck["config"], "icp_variant": "submap_v3", "sweep_profile": True, "imu_delay_comp": True}
    path = tmp / "fast_motion.pt"
    torch.save(ck, path)
    return path


@pytest.mark.parametrize("fast", [False, True], ids=["historical", "fast_motion_options"])
@pytest.mark.parametrize("dbs,run_key,ckpts", CASES, ids=["v1_A_nominal", "v2_APEX_real"])
def test_live_replay_matches_batch(dbs: tuple[Path, ...], run_key: str, ckpts: tuple[Path, ...], fast: bool, tmp_path: Path):
    db = _database_with(dbs, run_key)
    ckpt = next((c for c in ckpts if c.exists()), None)
    if db is None or ckpt is None:
        pytest.skip("needs a pose dataset with this run and the hybrid checkpoint")
    if fast:
        ckpt = _fast_motion_checkpoint(ckpt, tmp_path)
    dev = torch.device("cpu")
    data = _single_run(load_split(db, CACHE, "validation", imu_delay_comp=fast), run_key)
    sd = StreamData(data, dev)
    # The IMU encoder convolves before masking the padding, so the last
    # samples of an interval see the zeros after it. The training splits pad
    # to 29-31 samples (intervals with a lost scan), i.e. virtually every
    # interval had padding; the live node pads the same way. This run alone
    # would pad to its own longest interval (11) and leave most unpadded.
    sd.max_len = MAX_IMU
    model, ck = load_checkpoint(ckpt, dev)
    sd.attach_icp(run_icp(sd, ICP_VARIANTS[ck["config"]["icp_variant"]], log=lambda m: None))
    batch = stream_predict(model, sd, chunk=64)["pred"].numpy()

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    run_id = conn.execute("SELECT run_id FROM runs WHERE run_key = ?", (run_key,)).fetchone()[0]
    scans = conn.execute("SELECT timestamp_ns, beam_count, ranges_encoding, ranges_blob, valid_mask_blob FROM lidar_scans WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
    imu = conn.execute("SELECT timestamp_ns, ax, ay, az, gx, gy, gz FROM imu_samples WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
    ext, noise = conn.execute("SELECT extrinsic_translation, noise_configuration_json FROM sensor_calibration WHERE run_id = ? AND sensor_name = 'lidar'", (run_id,)).fetchone()
    imu_cfg = json.loads(conn.execute("SELECT noise_configuration_json FROM sensor_calibration WHERE run_id = ? AND sensor_name = 'imu'", (run_id,)).fetchone()[0])
    conn.close()
    prof = json.loads(noise)
    prof["extrinsic"] = {"xyz": json.loads(ext)}
    cal = LidarCalibration.from_profile(prof)
    delay = imu_group_delay_s(imu_cfg)
    live = LiveOdometry(ckpt, cal, device="cpu", imu_wait_s=10.0, imu_group_delay_s=delay)
    lag = int(round(delay * 1e9)) if fast else 0  # a sample covers the revolution once its compensated stamp does

    stamps = [s[0] for s in scans]
    imu_t = np.array([m[0] for m in imu])
    est = {}
    j = 0
    for k, (t, beams, enc, blob, mask) in enumerate(scans):
        r = decode_ranges(blob, enc, beams).astype(np.float32)
        r = np.where(decode_mask(mask, beams), r, np.inf)
        # Exact-end mode: the revolution of scan k lasts until the next stamp,
        # like the stored gyro rate of the training data.
        scan_time = (stamps[k + 1] - t) * 1e-9 if k + 1 < len(stamps) else 1.0 / cal.rate_hz
        end = stamps[k + 1] if k + 1 < len(stamps) else t + int(scan_time * 1e9)
        # Deliver every IMU sample up to the end of this revolution (plus the
        # next one, needed to interpolate the gyro at the boundary).
        while j < len(imu) and (j == 0 or imu_t[j - 1] - lag <= end):
            m = imu[j]
            live.add_imu(m[0], np.array(m[4:7]), np.array(m[1:4]))
            j += 1
        live.add_scan(t, r, scan_time, arrival_ns=end)
        for e in live.process(now_ns=end):
            est[e.stamp_ns] = e
    live_pred = np.array([est[s].delta if s in est else [np.nan] * 3 for s in stamps])
    ok = np.isfinite(live_pred).all(1) & np.isfinite(batch).all(1)
    ok[-1] = False  # the stored last interval reuses its own gyro rate
    assert ok.sum() > 0.95 * (len(stamps) - 1)
    err = np.abs(live_pred[ok] - batch[ok])
    assert err[:, :2].max() < 1e-5, err[:, :2].max()  # metres
    assert err[:, 2].max() < 1e-6, err[:, 2].max()  # radians
