#!/usr/bin/env python3
"""Run the unchanged bounded driver with trial CSV/JSON output on host tmpfs."""
import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

from race_fresh_start import verify_neutral
from ram_trial_storage import RamTrialStorage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--storage-self-test', action='store_true', help='No vehicle stack or actuators; test storage with an isolated Docker process')
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('Host mount setup requires root')
    names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    if 'apex_pipeline' in names:
        raise RuntimeError('Previous vehicle container must be stopped and removed')
    verify_neutral(Path('/sys/class/pwm/pwmchip0/pwm0'))
    owner = pwd.getpwnam('ensta')
    storage = RamTrialStorage(args.root, (owner.pw_uid, owner.pw_gid))
    result, sampler = None, None
    try:
        storage.mount()
        print('Trial CSV/JSON paths verified on RAM; controller and bounds unchanged', flush=True)
        if args.storage_self_test:
            tag = '.ram_storage_self_test_' + storage.token
            run = storage.targets['runs'] / tag
            diag = storage.targets['diagnostics'] / tag
            run.mkdir(); diag.mkdir()
            (run / 'host.txt').write_text('host RAM write\n')
            code = ("from pathlib import Path; import subprocess; "
                    "p=Path('/work/ros2_ws/apex_race/" + tag + "'); "
                    "assert subprocess.check_output(['stat','-f','-c','%T',str(p)],text=True).strip()=='tmpfs'; "
                    "(p/'docker.txt').write_text('isolated container RAM write\\n'); "
                    "q=Path('/work/data/diagnostics/20261022_open_segment/" + tag + "'); "
                    "assert subprocess.check_output(['stat','-f','-c','%T',str(q)],text=True).strip()=='tmpfs'; "
                    "(q/'docker.txt').write_text('diagnostic RAM write\\n')")
            subprocess.run(['docker', 'run', '--rm', '--network', 'none',
                            '-v', str(args.root / 'ros2_ws') + ':/work/ros2_ws',
                            '-v', str(args.root / 'data') + ':/work/data',
                            '--entrypoint', 'python3', 'docker-apex_pipeline:latest', '-c', code], check=True, timeout=20)
            # No fresh vehicle run was created in this storage-only exercise.
            pointer = storage.targets['runs'] / '.last_run'
            if pointer.exists(): pointer.unlink()
            result = 0
        else:
            # The runner below creates a fresh run and still owns all arming,
            # live-power checks, finite mission guards and final disarming.
            command = ['python3', str(storage.targets['diagnostics'] / 'run_segment_cpu_limit.py'),
                       '--max-khz', '2000000', '--runner', 'run_oval.py']
            result = subprocess.run(command, cwd=args.root, timeout=650).returncode
    finally:
        # Stop before copying to SD even after startup/timeout exceptions.
        running = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
        if 'apex_pipeline' in running:
            subprocess.run([str(args.root / 'tools/core/apex_race_up.sh'), 'down'], cwd=args.root, check=True, timeout=35)
        verify_neutral(Path('/sys/class/pwm/pwmchip0/pwm0'))
        if storage.ram.exists():
            try:
                report = storage.finish()
                print('RAM evidence copied and SHA256-verified after stop: ' + str(report), flush=True)
            except BaseException:
                # Restore the historical directories; leave RAM/staging intact
                # whenever a copy/publication error occurs.
                storage.unmount()
                print('Capture retained for recovery in ' + str(storage.ram), file=sys.stderr, flush=True)
                raise
    return result or 0


if __name__ == '__main__':
    raise SystemExit(main())
