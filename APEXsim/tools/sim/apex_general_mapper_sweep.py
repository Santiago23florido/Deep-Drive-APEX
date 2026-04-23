#!/usr/bin/env python3
"""Run a bounded parameter sweep for the general offline track mapper."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path


def _score_tuple(payload: dict[str, object]) -> tuple[float, float, float]:
    evaluation = payload.get("evaluation", {}) if isinstance(payload, dict) else {}
    if not isinstance(evaluation, dict):
        evaluation = {}
    runtime_s = float(payload.get("processing_elapsed_s", 1.0e9) or 1.0e9)
    return (
        float(evaluation.get("dilated_iou", 0.0) or 0.0),
        -float(evaluation.get("chamfer_distance_m", 1.0e9) or 1.0e9),
        -runtime_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--world-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument(
        "--mapper-script",
        default=str(
            Path(__file__).resolve().parents[2]
            / "ros2_ws"
            / "src"
            / "rc_sim_description"
            / "scripts"
            / "apex_general_track_mapper.py"
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    world_path = Path(args.world_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    mapper_script = Path(args.mapper_script).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combinations = list(
        itertools.product(
            (0.14, 0.16, 0.18),
            (8, 10, 12),
            (120, 140, 160),
        )
    )
    results: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None

    for keyframe_distance_m, submap_keyframes, max_points_per_keyframe in combinations:
        tag = (
            f"d{keyframe_distance_m:.2f}_s{submap_keyframes:d}_p{max_points_per_keyframe:d}"
            .replace(".", "p")
        )
        output_dir = output_root / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python_executable,
            "-u",
            str(mapper_script),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--status-json",
            str(output_dir / "build_status.json"),
            "--evaluation-world",
            str(world_path),
            "--evaluation-json",
            str(output_dir / "mapping_evaluation.json"),
            "--keyframe-distance-m",
            str(keyframe_distance_m),
            "--submap-keyframes",
            str(submap_keyframes),
            "--max-points-per-keyframe",
            str(max_points_per_keyframe),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(args.timeout_s),
            check=False,
        )
        elapsed_s = time.monotonic() - started
        summary_path = output_dir / "mapping_summary.json"
        summary_payload: dict[str, object] = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary_payload = loaded
            except Exception:
                summary_payload = {}
        result_payload = {
            "tag": tag,
            "keyframe_distance_m": keyframe_distance_m,
            "submap_keyframes": submap_keyframes,
            "max_points_per_keyframe": max_points_per_keyframe,
            "return_code": int(completed.returncode),
            "elapsed_wall_s": elapsed_s,
            "summary": summary_payload,
            "stdout_tail": completed.stdout[-8000:],
        }
        results.append(result_payload)
        if completed.returncode == 0 and summary_payload:
            runtime_s = float(summary_payload.get("processing_elapsed_s", 1.0e9) or 1.0e9)
            if runtime_s <= 60.0:
                if best_payload is None or _score_tuple(summary_payload) > _score_tuple(best_payload):
                    best_payload = summary_payload

    sweep_summary = {
        "run_dir": str(run_dir),
        "world_path": str(world_path),
        "output_root": str(output_root),
        "results": results,
        "best_summary": best_payload,
    }
    (output_root / "sweep_summary.json").write_text(
        json.dumps(sweep_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(sweep_summary, indent=2))


if __name__ == "__main__":
    main()
