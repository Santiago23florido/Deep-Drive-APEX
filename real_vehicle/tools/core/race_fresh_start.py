"""Prepare an unused race directory after the previous stack has stopped.

Only runtime bytecode caches and the last-run pointer are reset. Recorded
evidence, trained weights, configuration and serialized maps are preserved;
the normal mapping launch must separately verify it does not restore a map.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def verify_neutral(pwm: Path) -> None:
    if not pwm.exists():  # dry/sensor-only machine, with no registered PWM
        return
    enabled = pwm / 'enable'
    if enabled.exists() and enabled.read_text().strip() == '0':
        return
    period = int((pwm / 'period').read_text())
    duty = int((pwm / 'duty_cycle').read_text())
    if abs(duty - .075 * period) > 100:
        raise RuntimeError('previous stack did not leave the motor at 7.5% neutral')


def prepare_run(root: Path, *, stamp: str | None = None, owner: tuple[int, int] | None = None) -> Path:
    root = root.resolve()
    runs = root / 'ros2_ws/apex_race'
    runs.mkdir(parents=True, exist_ok=True)
    if owner is not None:
        os.chown(runs, *owner)
    previous = runs / '.last_run'
    old_id = previous.read_text().strip() if previous.exists() else None
    removed = []
    for package in ('apex_race', 'apex_telemetry'):
        directory = root / 'ros2_ws/src' / package / package
        if directory.is_symlink():
            continue
        for cache in list(directory.rglob('__pycache__')):
            if cache.is_dir() and not cache.is_symlink():
                shutil.rmtree(cache)
                removed.append(str(cache.relative_to(root)))
    stem = stamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    if not stem or Path(stem).name != stem or stem in ('.', '..'):
        raise ValueError('invalid run id')
    for index in range(1000):
        run = runs / (stem if index == 0 else f'{stem}_{index:02d}')
        try:
            run.mkdir()  # exclusive: never append to an older run, even in the same second
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('cannot allocate an unused run directory')
    if owner is not None:
        os.chown(run, *owner)
    report = {'run_id': run.name, 'previous_run': old_id, 'cache_dirs_removed': removed,
              'previous_artifacts_preserved': True}
    (run / 'fresh_start.json').write_text(json.dumps(report, indent=2) + '\n')
    previous.write_text(run.name + '\n')
    if owner is not None:
        os.chown(previous, *owner)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--container', default='apex_pipeline')
    parser.add_argument('--uid', type=int, required=True)
    parser.add_argument('--gid', type=int, required=True)
    args = parser.parse_args()
    containers = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    if args.container in containers:
        raise RuntimeError('previous stack container still exists; stop it before a fresh start')
    verify_neutral(Path('/sys/class/pwm/pwmchip0/pwm0'))
    run = prepare_run(args.root, owner=(args.uid, args.gid))
    print('[race] fresh start: old stack removed; unused run directory and cleared bytecode caches', file=sys.stderr)
    print(run.name)


if __name__ == '__main__':
    main()
