"""Run Gazebo jobs in parallel, isolated worker processes.

Each trajectory runs in its own Python process (``python -m
pose_dataset.gz_runner job.json``) with its own gz-transport partition, so a
crash or a hang only affects that job, and the pool keeps going.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterator

TOOLS_DIR = Path(__file__).resolve().parents[1]


def worker_env(partition: str) -> dict[str, str]:
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(TOOLS_DIR),
        "GZ_PARTITION": partition,
        "GZ_IP": "127.0.0.1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return env


class JobPool:
    """Minimal subprocess pool with timeouts and cooperative shutdown."""

    def __init__(self, workers: int, python: str | None = None) -> None:
        self.workers = max(1, int(workers))
        self.python = python or sys.executable
        self.running: dict[str, tuple[subprocess.Popen, dict[str, Any], float, Any]] = {}
        self.stop_requested = False

    def terminate_all(self) -> None:
        """Stop every worker. gz-sim traps SIGTERM to pause its loop, so the
        whole process group is killed after a short grace period; partial
        worker outputs are never read (the job is simply redone on resume)."""
        procs = [(proc, log) for proc, _job, _t0, log in self.running.values()]
        for proc, _log in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        deadline = time.time() + 1.5
        for proc, _log in procs:
            try:
                proc.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
        for _proc, log in procs:
            log.close()
        self.running.clear()

    def run(self, jobs: list[dict[str, Any]], timeout_fn: Callable[[dict[str, Any]], float]) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None, str]]:
        """Yield ``(job, summary_or_None, error_text)`` as jobs finish."""
        pending = list(jobs)
        counter = 0
        while (pending or self.running) and not self.stop_requested:
            while pending and len(self.running) < self.workers and not self.stop_requested:
                job = pending.pop(0)
                counter += 1
                job_path = Path(job["out_npz"]).with_suffix(".job.json")
                job_path.parent.mkdir(parents=True, exist_ok=True)
                job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
                log = open(Path(job["out_npz"]).with_suffix(".log"), "w", encoding="utf-8")
                partition = f"pose_ds_{os.getpid()}_{counter}"
                proc = subprocess.Popen(
                    [self.python, "-m", "pose_dataset.gz_runner", str(job_path)],
                    cwd=str(TOOLS_DIR),
                    env=worker_env(partition),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.running[job["trajectory_key"]] = (proc, job, time.time(), log)
            time.sleep(0.2)
            for key, (proc, job, t0, log) in list(self.running.items()):
                rc = proc.poll()
                timed_out = rc is None and time.time() - t0 > timeout_fn(job)
                if rc is None and not timed_out:
                    continue
                if timed_out:
                    proc.kill()
                    proc.wait()
                log.close()
                del self.running[key]
                summary_path = Path(job["out_npz"]).with_suffix(".summary.json")
                if rc == 0 and summary_path.exists():
                    yield job, json.loads(summary_path.read_text(encoding="utf-8")), ""
                else:
                    tail = ""
                    try:
                        tail = Path(job["out_npz"]).with_suffix(".log").read_text(encoding="utf-8")[-1500:]
                    except OSError:
                        pass
                    reason = "wall-clock timeout" if timed_out else f"worker exit code {rc}"
                    yield job, None, f"{reason}: {tail}"
        if self.stop_requested:
            self.terminate_all()
