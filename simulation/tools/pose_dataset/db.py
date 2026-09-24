"""SQLite storage of the pose dataset (one file per campaign).

Design:

* one row per run in ``runs`` (a run = one trajectory seen by one sensor
  profile); ``trajectory_key`` groups the runs that share the same physical
  lap, and a trajectory never spans two splits;
* sensor rows keep their *reported* timestamps (int ns, simulation clock) and
  point to the ground-truth row of their *true* acquisition instant;
* LiDAR ranges are float32 BLOBs (see ``blobs.py``), never one row per beam;
* the model inputs (``lidar_scans``, ``imu_samples``, ``sensor_calibration``
  nominal extrinsics) are separated from the labels (``ground_truth``,
  ``imu_bias_truth``, ``lidar_beam_truth``, true extrinsics);
* every run is written in a single transaction: a crash or SIGINT leaves no
  half-written run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import numpy as np

from .blobs import encode_mask, encode_outcome, encode_ranges

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id               INTEGER PRIMARY KEY,
    run_key              TEXT NOT NULL UNIQUE,
    trajectory_key       TEXT NOT NULL,
    scenario_name        TEXT NOT NULL,
    track_name           TEXT NOT NULL,
    track_family         TEXT NOT NULL,
    split                TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    motion_profile       TEXT NOT NULL,
    sensor_profile       TEXT NOT NULL,
    seed                 INTEGER NOT NULL,
    direction            INTEGER NOT NULL,
    simulator_version    TEXT,
    git_commit           TEXT,
    generator_version    TEXT,
    start_time           TEXT,
    end_time             TEXT,
    simulated_duration_s REAL,
    start_sim_time_ns    INTEGER,
    lap_completed        INTEGER,
    laps_target          REAL,
    distance_m           REAL,
    v_ref_mps            REAL,
    status               TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'interrupted')),
    failure_reason       TEXT,
    qa_passed            INTEGER,
    configuration_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_split_status ON runs(split, status);
CREATE INDEX IF NOT EXISTS runs_trajectory ON runs(trajectory_key);

CREATE TABLE IF NOT EXISTS ground_truth (
    ground_truth_id  INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    timestamp_ns     INTEGER NOT NULL,
    relative_time_ns INTEGER NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    qx REAL NOT NULL, qy REAL NOT NULL, qz REAL NOT NULL, qw REAL NOT NULL,
    roll REAL NOT NULL, pitch REAL NOT NULL, yaw REAL NOT NULL,
    vx REAL NOT NULL, vy REAL NOT NULL, vz REAL NOT NULL,
    vx_body REAL NOT NULL, vy_body REAL NOT NULL, vz_body REAL NOT NULL,
    wx REAL NOT NULL, wy REAL NOT NULL, wz REAL NOT NULL,
    ax REAL NOT NULL, ay REAL NOT NULL, az REAL NOT NULL,
    body_roll REAL NOT NULL, body_pitch REAL NOT NULL,
    UNIQUE (run_id, timestamp_ns)
);

CREATE TABLE IF NOT EXISTS lidar_scans (
    scan_id             INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seq                 INTEGER NOT NULL,
    timestamp_ns        INTEGER NOT NULL,
    relative_time_ns    INTEGER NOT NULL,
    ground_truth_id     INTEGER NOT NULL REFERENCES ground_truth(ground_truth_id),
    ground_truth_end_id INTEGER NOT NULL REFERENCES ground_truth(ground_truth_id),
    angle_min           REAL NOT NULL,
    angle_increment     REAL NOT NULL,
    range_min           REAL NOT NULL,
    range_max           REAL NOT NULL,
    beam_count          INTEGER NOT NULL,
    time_increment_ns   REAL NOT NULL,
    valid_count         INTEGER NOT NULL,
    ranges_encoding     TEXT NOT NULL,
    ranges_blob         BLOB NOT NULL,
    valid_mask_blob     BLOB NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS lidar_run_time ON lidar_scans(run_id, timestamp_ns);

CREATE TABLE IF NOT EXISTS lidar_beam_truth (
    scan_id      INTEGER PRIMARY KEY REFERENCES lidar_scans(scan_id) ON DELETE CASCADE,
    outcome_blob BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS imu_samples (
    imu_id           INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seq              INTEGER NOT NULL,
    timestamp_ns     INTEGER NOT NULL,
    relative_time_ns INTEGER NOT NULL,
    ground_truth_id  INTEGER NOT NULL REFERENCES ground_truth(ground_truth_id),
    ax REAL NOT NULL, ay REAL NOT NULL, az REAL NOT NULL,
    gx REAL NOT NULL, gy REAL NOT NULL, gz REAL NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS imu_run_time ON imu_samples(run_id, timestamp_ns);

CREATE TABLE IF NOT EXISTS imu_bias_truth (
    imu_id INTEGER PRIMARY KEY REFERENCES imu_samples(imu_id) ON DELETE CASCADE,
    bax REAL NOT NULL, bay REAL NOT NULL, baz REAL NOT NULL,
    bgx REAL NOT NULL, bgy REAL NOT NULL, bgz REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sensor_calibration (
    run_id                     INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sensor_name                TEXT NOT NULL CHECK (sensor_name IN ('lidar', 'imu')),
    extrinsic_translation      TEXT NOT NULL,
    extrinsic_rotation         TEXT NOT NULL,
    nominal_frequency          REAL NOT NULL,
    noise_configuration_json   TEXT NOT NULL,
    true_extrinsic_translation TEXT NOT NULL,
    true_extrinsic_rotation    TEXT NOT NULL,
    realization_json           TEXT NOT NULL,
    PRIMARY KEY (run_id, sensor_name)
);

CREATE TABLE IF NOT EXISTS events (
    event_id         INTEGER PRIMARY KEY,
    run_id           INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    timestamp_ns     INTEGER NOT NULL,
    relative_time_ns INTEGER NOT NULL,
    event_type       TEXT NOT NULL,
    details_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run_time ON events(run_id, timestamp_ns);

CREATE TABLE IF NOT EXISTS run_qa (
    run_id       INTEGER PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    passed       INTEGER NOT NULL,
    metrics_json TEXT NOT NULL
);
"""

BLOB_FORMAT_DOC = {
    "ranges_blob": "float32 little-endian (<f4), beam_count values in beam order; encoding in lidar_scans.ranges_encoding: 'f32le' raw or 'f32le+zlib' zlib-compressed; +inf = no return / beyond range_max, -inf = below range_min, NaN never stored",
    "valid_mask_blob": "numpy.packbits(valid, bitorder='little'); bit i%8 of byte i//8 = beam i valid; ceil(beam_count/8) bytes",
    "outcome_blob": "uint8 per beam: 0 no_truth, 1 hit, 2 dropout, 3 short outlier, 4 random outlier (label only)",
    "timestamps": "integer nanoseconds of simulation time; sensor rows hold the reported (latency + jitter) stamp, ground_truth rows the true physics instant; relative_time_ns = timestamp_ns - runs.start_sim_time_ns",
    "frames": "ground truth = base_link in the world frame (ENU, z up): origin midway between the axles at wheel-contact height, x forward, y left; quaternion (qx, qy, qz, qw); v* world frame, v*_body and w* base_link frame, a* world frame without gravity; body_roll/body_pitch = sprung-body attitude relative to base_link",
    "imu_samples": "specific force (m/s^2) and angular rate (rad/s) in the IMU sensor frame (see sensor_calibration)",
}


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60.0)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=60.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-65536")
    return conn


def init_db(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    conn.executescript(SCHEMA)
    existing = dict(conn.execute("SELECT key, value FROM dataset_meta").fetchall())
    values = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": existing.get("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "blob_format": json.dumps(BLOB_FORMAT_DOC),
        **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in meta.items()},
    }
    if "role" in existing and "role" in values and existing["role"] != values["role"]:
        raise RuntimeError(f"database role is {existing['role']!r}, refusing to add {values['role']!r} runs")
    conn.execute("BEGIN")
    for k, v in values.items():
        conn.execute("INSERT INTO dataset_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (k, v))
    conn.execute("COMMIT")


def run_states(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("SELECT run_id, run_key, status, split, trajectory_key, qa_passed FROM runs").fetchall()
    return {r[1]: {"run_id": r[0], "status": r[2], "split": r[3], "trajectory_key": r[4], "qa_passed": r[5]} for r in rows}


def _next_id(conn: sqlite3.Connection, table: str, column: str) -> int:
    return int(conn.execute(f"SELECT IFNULL(MAX({column}), 0) + 1 FROM {table}").fetchone()[0])


def _chunks(rows: list, size: int = 20000) -> Iterable[list]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def delete_run(conn: sqlite3.Connection, run_key: str) -> None:
    """Remove every row of one run (before retrying it).

    Children are deleted explicitly with foreign keys temporarily off: the
    cascade would otherwise check each deleted ground-truth row against the
    unindexed ``ground_truth_id`` columns of the sensor tables.
    """
    row = conn.execute("SELECT run_id FROM runs WHERE run_key = ?", (run_key,)).fetchone()
    if row is None:
        return
    run_id = int(row[0])
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM imu_bias_truth WHERE imu_id IN (SELECT imu_id FROM imu_samples WHERE run_id = ?)", (run_id,))
        conn.execute("DELETE FROM lidar_beam_truth WHERE scan_id IN (SELECT scan_id FROM lidar_scans WHERE run_id = ?)", (run_id,))
        for table in ("imu_samples", "lidar_scans", "ground_truth", "events", "sensor_calibration", "run_qa", "runs"):
            conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        conn.execute("COMMIT")
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def write_run(conn: sqlite3.Connection, run: dict[str, Any], synth: dict[str, Any] | None, events: list[dict[str, Any]], qa: dict[str, Any] | None, compress_ranges: bool = False) -> int:
    """Insert one run (metadata + data + QA) in a single transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        cols = list(run.keys())
        conn.execute(f"INSERT INTO runs({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [run[c] for c in cols])
        run_id = int(conn.execute("SELECT run_id FROM runs WHERE run_key = ?", (run["run_key"],)).fetchone()[0])
        t0 = int(run["start_sim_time_ns"] or 0)
        if synth is not None:
            gt = synth["gt"]
            gt_first = _next_id(conn, "ground_truth", "ground_truth_id")
            gt_ids = gt_first + np.arange(len(gt["idx"]))
            idx_to_gt = dict(zip(gt["idx"].tolist(), gt_ids.tolist()))
            rows = []
            for i in range(len(gt["idx"])):
                x, y, z = gt["xyz"][i]
                qx, qy, qz, qw = gt["quat_xyzw"][i]
                r, p, yw = gt["rpy"][i]
                rows.append((
                    int(gt_ids[i]), run_id, int(gt["t_ns"][i]), int(gt["t_ns"][i]) - t0, x, y, z, qx, qy, qz, qw, r, p, yw,
                    *gt["v_world"][i], *gt["v_body"][i], *gt["w_body"][i], *gt["a_world"][i], *gt["body_roll_pitch"][i],
                ))
            for chunk in _chunks(rows):
                conn.executemany(f"INSERT INTO ground_truth VALUES ({','.join('?' * 28)})", chunk)

            li = synth["lidar"]
            scan_first = _next_id(conn, "lidar_scans", "scan_id")
            scan_rows, truth_rows = [], []
            for i in range(len(li["start_idx"])):
                sid = scan_first + i
                blob, enc = encode_ranges(li["ranges"][i], compress=compress_ranges)
                scan_rows.append((
                    sid, run_id, i, int(li["t_report_ns"][i]), int(li["t_report_ns"][i]) - t0,
                    idx_to_gt[int(li["start_idx"][i])], idx_to_gt[int(li["end_idx"][i])],
                    float(li["angle_min"]), float(li["angle_increment"]), float(li["range_min"]), float(li["range_max"]),
                    int(li["beams"]), float(li["time_increment_ns"]), int(li["valid"][i].sum()), enc, blob, encode_mask(li["valid"][i]),
                ))
                truth_rows.append((sid, encode_outcome(li["outcome"][i])))
            for chunk in _chunks(scan_rows, 5000):
                conn.executemany(f"INSERT INTO lidar_scans VALUES ({','.join('?' * 17)})", chunk)
            for chunk in _chunks(truth_rows, 5000):
                conn.executemany("INSERT INTO lidar_beam_truth VALUES (?, ?)", chunk)

            imu = synth["imu"]
            imu_first = _next_id(conn, "imu_samples", "imu_id")
            vals = imu["values"]
            bias = imu["bias_true"]
            imu_rows = [
                (imu_first + i, run_id, i, int(imu["t_report_ns"][i]), int(imu["t_report_ns"][i]) - t0, idx_to_gt[int(imu["true_idx"][i])], *vals[i])
                for i in range(len(vals))
            ]
            bias_rows = [(imu_first + i, *bias[i]) for i in range(len(vals))]
            for chunk in _chunks(imu_rows):
                conn.executemany(f"INSERT INTO imu_samples VALUES ({','.join('?' * 12)})", chunk)
            for chunk in _chunks(bias_rows):
                conn.executemany("INSERT INTO imu_bias_truth VALUES (?, ?, ?, ?, ?, ?, ?)", chunk)

            for sensor, stream, freq in (("lidar", li, synth["profile"]["lidar"]["rate_hz"]), ("imu", imu, synth["profile"]["imu"]["rate_hz"])):
                ext = stream["extrinsic"]
                conn.execute(
                    "INSERT INTO sensor_calibration VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, sensor, json.dumps([float(v) for v in ext["xyz"]]), json.dumps(ext["quat_xyzw"]), float(freq),
                        json.dumps(synth["profile"][sensor]), json.dumps([float(v) for v in ext["xyz_true"]]), json.dumps(ext["quat_true_xyzw"]),
                        json.dumps(stream["realization"], default=float),
                    ),
                )
        ev_rows = [(run_id, int(e["t_ns"]), int(e["t_ns"]) - t0, e["type"], json.dumps(e.get("details", {}), default=float)) for e in events]
        conn.executemany("INSERT INTO events(run_id, timestamp_ns, relative_time_ns, event_type, details_json) VALUES (?, ?, ?, ?, ?)", ev_rows)
        if qa is not None:
            conn.execute("INSERT INTO run_qa VALUES (?, ?, ?)", (run_id, int(bool(qa["passed"])), json.dumps(qa, default=float)))
        conn.execute("COMMIT")
        return run_id
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def finalize(conn: sqlite3.Connection) -> dict[str, Any]:
    """Checkpoint the WAL, analyse and run the integrity checks."""
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("ANALYZE")
    integrity = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    return {"integrity_check": integrity[0] if len(integrity) == 1 else integrity, "foreign_key_violations": len(fk)}
