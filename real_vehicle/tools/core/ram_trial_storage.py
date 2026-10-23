"""Keep finite trial output in tmpfs and publish verified copies after stopping.

Only the two log directories are overlaid. Controller sources, configuration,
weights, actuator bounds and deadlines are unaffected. Failed backup/publication
retains the RAM source and any staging files for recovery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(root: Path) -> None:
    for path in root.rglob('*'):
        if path.is_file():
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
    for path in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        sync_directory(path)
    sync_directory(root)


def inventory(root: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise RuntimeError(f'Unexpected symlink in trial output: {path}')
        if path.is_file():
            result[str(path.relative_to(root))] = {
                'size': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def verified_stage(ram: Path, stage: Path) -> dict:
    before = inventory(ram)
    needed = sum(row['size'] for row in before.values())
    if shutil.disk_usage(stage.parent).free < needed + 128 * 2**20:
        raise RuntimeError('Insufficient durable space for the complete RAM capture')
    shutil.copytree(ram, stage)  # exclusive; never merge with an older capture
    if inventory(stage) != before or inventory(ram) != before:
        raise RuntimeError('RAM capture changed or copy verification failed; both copies retained')
    sync_tree(stage)  # all SD writeback happens after motion has stopped
    return {'files': len(before), 'bytes': needed, 'sha256': before}


def publish(stage: Path, runs: Path, diagnostics: Path) -> list[str]:
    moves = []
    allowed = {'runs': {'.last_run'}, 'diagnostics': {'run_oval.py', 'run_segment_cpu_limit.py',
               'save_final_map.py', 'sample_resources.py', '.ram_capture_active.json'}}
    for source, destination in ((stage / 'runs', runs), (stage / 'diagnostics', diagnostics)):
        for path in source.iterdir():
            if path.is_dir():
                moves.append((path, destination / path.name))
            elif path.name not in allowed[source.name]:
                raise RuntimeError(f'Unrecognized output retained in staging: {path}')
    # Precheck every destination before publishing anything. All directories are
    # newly allocated and old trial evidence must never be overwritten.
    for _, target in moves:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Existing evidence preserved: {target}')
    published = []
    for source, target in moves:
        expected = inventory(source)
        if source.stat().st_dev == target.parent.stat().st_dev:
            source.rename(target)
        else:
            shutil.copytree(source, target)
            sync_tree(target)
        if inventory(target) != expected:
            raise RuntimeError(f'Published data verification failed; RAM source retained: {target}')
        published.append(str(target))
        sync_directory(target.parent)
    pointer = stage / 'runs/.last_run'
    if pointer.exists():
        run_id = pointer.read_text().strip()
        if Path(run_id).name != run_id or not (runs / run_id).is_dir():
            raise RuntimeError('Refusing an invalid last-run pointer')
        temp = runs / ('.last_run.' + uuid.uuid4().hex + '.tmp')
        temp.write_text(run_id + '\n')
        with temp.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temp, runs / '.last_run')
        sync_directory(runs)
    return published


class RamTrialStorage:
    def __init__(self, root: Path, owner: tuple[int, int], *, ram_root: Path = Path('/dev/shm')):
        self.root, self.owner = root.resolve(), owner
        self.targets = {'runs': self.root / 'ros2_ws/apex_race',
                        'diagnostics': self.root / 'data/diagnostics/20261022_open_segment'}
        self.token = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.ram = ram_root / ('apex_trial_' + self.token)
        self.stage = self.root / 'data' / ('ram_capture_staging_' + self.token)
        self.mounted = []

    def mount(self):
        if shutil.disk_usage(self.ram.parent).free < 512 * 2**20:
            raise RuntimeError('Less than 512 MiB tmpfs space available')
        for target in self.targets.values():
            if target.is_symlink() or os.path.ismount(target):
                raise RuntimeError(f'Refusing to cover another mount or symlink: {target}')
        self.ram.mkdir()
        os.chown(self.ram, *self.owner)
        for name in self.targets:
            (self.ram / name).mkdir()
            os.chown(self.ram / name, *self.owner)
        for name, target in self.targets.items():
            target.mkdir(parents=True, exist_ok=True)
            destination = self.ram / name
            if name == 'diagnostics':
                for script in ('run_oval.py', 'run_segment_cpu_limit.py', 'save_final_map.py', 'sample_resources.py'):
                    shutil.copy2(target / script, destination / script)
            elif (target / '.last_run').is_file():
                shutil.copy2(target / '.last_run', destination / '.last_run')
                os.chown(destination / '.last_run', *self.owner)
            subprocess.run(['mount', '--bind', str(destination), str(target)], check=True)
            self.mounted.append(target)
            filesystem = subprocess.check_output(['stat', '-f', '-c', '%T', str(target)], text=True).strip()
            if filesystem != 'tmpfs':
                raise RuntimeError(f'Output is not on tmpfs: {target}')
        (self.targets['diagnostics'] / '.ram_capture_active.json').write_text(json.dumps({
            'token': self.token, 'ram_source': str(self.ram), 'storage': 'tmpfs',
            'copy_to_sd_after_neutral_only': True}) + '\n')
        return self

    def unmount(self):
        for target in reversed(self.mounted[:]):
            subprocess.run(['umount', str(target)], check=True)
            self.mounted.remove(target)

    def finish(self):
        # Call only after all writers are stopped and the motor is neutral.
        started = time.monotonic()
        result = verified_stage(self.ram, self.stage)
        self.unmount()
        paths = publish(self.stage, self.targets['runs'], self.targets['diagnostics'])
        result.update(storage='tmpfs_during_trial', verified_after_motor_stop=True,
                      published=paths, copy_and_verify_s=time.monotonic() - started)
        report = self.root / 'data/diagnostics' / ('ram_capture_' + self.token + '.json')
        report.write_text(json.dumps(result, indent=2) + '\n')
        os.chown(report, *self.owner)
        with report.open('rb') as handle:
            os.fsync(handle.fileno())
        sync_directory(report.parent)
        # Evidence is now verified on durable storage. Never remove either
        # source when any earlier operation failed.
        shutil.rmtree(self.ram)
        shutil.rmtree(self.stage)
        return report
