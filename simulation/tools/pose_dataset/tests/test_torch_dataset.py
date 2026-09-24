"""PyTorch reader: shapes, real interval lengths, padding masks, workers, splits."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pose_dataset.db import connect, finalize, init_db, write_run  # noqa: E402
from pose_dataset.torch_dataset import PoseSequenceDataset, collate_pose_batch  # noqa: E402

STEP = 1_000_000  # 1 ms physics grid


def _fake_run(seed: int, beams: int, lidar_hz: float, imu_hz: float, n_scans: int, drop: float):
    rng = np.random.default_rng(seed)
    lid_period = int(round(1000 / lidar_hz))
    imu_period = int(round(1000 / imu_hz))
    start = 500
    scan_idx = start + lid_period * np.arange(n_scans)
    imu_idx = np.arange(start + 1, scan_idx[-1] + 1, imu_period)
    imu_idx = imu_idx[rng.random(len(imu_idx)) > drop]
    gt_idx = np.unique(np.concatenate((scan_idx, scan_idx + 5, imu_idx)))
    t = gt_idx * 1e-3
    yaw = 0.3 * t
    gt = {
        "idx": gt_idx, "t_ns": gt_idx * STEP,
        "xyz": np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t))),
        "quat_xyzw": np.column_stack((np.zeros_like(t), np.zeros_like(t), np.sin(yaw / 2), np.cos(yaw / 2))),
        "rpy": np.column_stack((np.zeros_like(t), np.zeros_like(t), yaw)),
        "v_world": rng.normal(size=(len(t), 3)), "v_body": rng.normal(size=(len(t), 3)), "w_body": rng.normal(size=(len(t), 3)),
        "a_world": rng.normal(size=(len(t), 3)), "body_roll_pitch": np.zeros((len(t), 2)),
    }
    ranges = rng.uniform(0.2, 10.0, (n_scans, beams)).astype(np.float32)
    valid = rng.random((n_scans, beams)) > 0.1
    ranges[~valid] = np.inf
    ext = {"xyz": [0.1, 0, 0.1], "quat_xyzw": [0, 0, 0, 1], "xyz_true": [0.1, 0, 0.1], "quat_true_xyzw": [0, 0, 0, 1]}
    synth = {
        "gt": gt,
        "lidar": {
            "start_idx": scan_idx, "end_idx": scan_idx + 5, "t_report_ns": scan_idx * STEP + 1000,
            "ranges": ranges, "valid": valid, "outcome": np.ones((n_scans, beams), np.uint8),
            "angle_min": -np.pi, "angle_increment": 2 * np.pi / beams, "range_min": 0.1, "range_max": 12.0,
            "beams": beams, "time_increment_ns": 1e5, "extrinsic": ext, "realization": {},
        },
        "imu": {
            "true_idx": imu_idx, "t_report_ns": imu_idx * STEP + 2000,
            "values": rng.normal(size=(len(imu_idx), 6)), "bias_true": np.zeros((len(imu_idx), 6)),
            "extrinsic": ext, "realization": {},
        },
        "profile": {"lidar": {"rate_hz": lidar_hz}, "imu": {"rate_hz": imu_hz}},
    }
    return synth


def _row(key: str, split: str):
    return {
        "run_key": key, "trajectory_key": key, "scenario_name": "t/m", "track_name": "t", "track_family": "t", "split": split,
        "motion_profile": "m", "sensor_profile": "S", "seed": 1, "direction": 1, "simulator_version": "test", "git_commit": "x",
        "generator_version": "test", "start_time": None, "end_time": None, "simulated_duration_s": 1.0, "start_sim_time_ns": 500 * STEP,
        "lap_completed": 1, "laps_target": 1.0, "distance_m": 1.0, "v_ref_mps": 1.0, "status": "completed", "failure_reason": None,
        "qa_passed": 1, "configuration_json": json.dumps({}),
    }


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "tiny.sqlite3"
    conn = connect(path)
    init_db(conn, {"role": "main"})
    write_run(conn, _row("run_train", "train"), _fake_run(1, 360, 10.0, 100.0, 30, 0.15), [], {"passed": True})
    write_run(conn, _row("run_val", "validation"), _fake_run(2, 360, 20.0, 200.0, 25, 0.05), [], {"passed": True})
    res = finalize(conn)
    conn.close()
    assert res["integrity_check"] == "ok" and res["foreign_key_violations"] == 0
    return path


def test_item_shapes_and_real_interval_lengths(db_path):
    ds = PoseSequenceDataset(db_path, "train", seq_len=5, stride=1)
    assert len(ds) == 30 - 5 + 1
    item = ds[3]
    assert item["scans"].shape == (5, 360) and item["scan_valid"].dtype == torch.bool
    assert len(item["imu"]) == 4 and item["imu_len"].shape == (4,)
    assert item["gt_delta"].shape == (4, 3) and item["gt_pose"].shape == (5, 7)
    lengths = item["imu_len"].tolist()
    assert len(set(lengths)) > 1 or ds[0]["imu_len"].tolist() != ds[7]["imu_len"].tolist()  # dropouts -> variable counts
    for k in range(4):
        assert item["imu"][k].shape == (lengths[k], 6)
        assert item["imu_dt"][k].shape == (lengths[k],)
        # Every sample lies strictly after scan k and not after scan k+1.
        t0, t1 = item["scan_t_ns"][k].item(), item["scan_t_ns"][k + 1].item()
        rel = item["imu_t_rel"][k]
        assert torch.all(rel > 0) and torch.all(rel <= (t1 - t0) * 1e-9 + 1e-6)
    assert torch.all(item["scans"][~item["scan_valid"]] == 0.0)


def test_collate_padding_is_masked(db_path):
    ds = PoseSequenceDataset(db_path, "train", seq_len=4, stride=2)
    items = [ds[i] for i in range(4)]
    batch = collate_pose_batch(items)
    b, k, l_max, _ = batch["imu"].shape
    assert (b, k) == (4, 3)
    assert torch.equal(batch["imu_mask"].sum(-1), batch["imu_len"])
    assert l_max == int(batch["imu_len"].max())
    # Padding must be ignored: garbage in the padded slots does not change masked statistics.
    noisy = batch["imu"].clone()
    noisy[~batch["imu_mask"]] = 1e6
    m = batch["imu_mask"].unsqueeze(-1).float()
    mean_a = (batch["imu"] * m).sum(2) / m.sum(2).clamp(min=1)
    mean_b = (noisy * m).sum(2) / m.sum(2).clamp(min=1)
    assert torch.allclose(mean_a, mean_b)
    # A packed GRU gives identical outputs whatever the amount of padding.
    torch.manual_seed(0)
    gru = torch.nn.GRU(6, 8, batch_first=True)
    seqs = noisy.reshape(b * k, l_max, 6)
    lens = batch["imu_len"].reshape(-1).clamp(min=1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(seqs, lens.cpu(), batch_first=True, enforce_sorted=False)
    _, h_batch = gru(packed)
    for i in range(b * k):
        n = int(lens[i])
        _, h_single = gru(seqs[i : i + 1, :n])
        assert torch.allclose(h_batch[0, i], h_single[0, 0], atol=1e-5)


def test_dataloader_workers_never_mix_splits(db_path):
    ds = PoseSequenceDataset(db_path, "validation", seq_len=6, stride=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=3, num_workers=2, collate_fn=collate_pose_batch)
    seen = 0
    for batch in loader:
        assert set(batch["split"]) == {"validation"}
        assert batch["scans"].shape[1:] == (6, 360)
        # Windows never cross runs: scan timestamps strictly increase within each item.
        assert torch.all(batch["scan_t_ns"][:, 1:] > batch["scan_t_ns"][:, :-1])
        seen += batch["scans"].shape[0]
    assert seen == len(ds)
    train = PoseSequenceDataset(db_path, "train", seq_len=6, stride=3)
    assert not set(train.runs) & set(ds.runs)
