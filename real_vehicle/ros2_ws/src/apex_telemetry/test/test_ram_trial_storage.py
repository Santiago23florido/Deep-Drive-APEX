"""RAM logging must retain evidence through failures and preserve prior trials."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[4] / 'tools/core/ram_trial_storage.py'
spec = importlib.util.spec_from_file_location('ram_trial_storage', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_tree(tmp_path):
    ram, stage, disk = tmp_path / 'ram', tmp_path / 'stage', tmp_path / 'disk'
    for p in (ram / 'runs/new', ram / 'diagnostics/oval_new', disk / 'runs/old', disk / 'diagnostics/oval_old'):
        p.mkdir(parents=True)
    (ram / 'runs/new/driver.csv').write_bytes(b't,v\n0,0\n')
    (ram / 'runs/.last_run').write_text('new\n')
    (ram / 'diagnostics/oval_new/scans.csv').write_bytes(bytes(range(256)) * 100)
    (disk / 'runs/old/data').write_text('preserve')
    (disk / 'runs/.last_run').write_text('old\n')
    return ram, stage, disk


def test_complete_capture_is_verified_and_published_without_modifying_old_runs(tmp_path):
    ram, stage, disk = fixture_tree(tmp_path)
    before = module.inventory(ram)
    result = module.verified_stage(ram, stage)
    assert result['sha256'] == before
    module.publish(stage, disk / 'runs', disk / 'diagnostics')
    assert (disk / 'runs/old/data').read_text() == 'preserve'
    assert (disk / 'runs/.last_run').read_text().strip() == 'new'
    assert (disk / 'diagnostics/oval_new/scans.csv').read_bytes() == (ram / 'diagnostics/oval_new/scans.csv').read_bytes()
    assert module.inventory(ram) == before


def test_destination_collision_preserves_both_captures_and_old_pointer(tmp_path):
    ram, stage, disk = fixture_tree(tmp_path)
    module.verified_stage(ram, stage)
    (disk / 'diagnostics/oval_new').mkdir()
    (disk / 'diagnostics/oval_new/data').write_text('do not overwrite')
    with pytest.raises(FileExistsError): module.publish(stage, disk / 'runs', disk / 'diagnostics')
    assert not (disk / 'runs/new').exists()
    assert (disk / 'runs/.last_run').read_text().strip() == 'old'
    assert (stage / 'runs/new/driver.csv').exists() and (ram / 'runs/new/driver.csv').exists()
    assert (disk / 'diagnostics/oval_new/data').read_text() == 'do not overwrite'


def test_corrupted_staging_copy_is_rejected_and_ram_is_retained(tmp_path, monkeypatch):
    ram, stage, disk = fixture_tree(tmp_path)
    original = module.shutil.copytree
    def corrupt(source, target, *args, **kwargs):
        result = original(source, target, *args, **kwargs)
        if Path(target) == stage: (stage / 'runs/new/driver.csv').write_text('corrupt')
        return result
    monkeypatch.setattr(module.shutil, 'copytree', corrupt)
    with pytest.raises(RuntimeError, match='verification failed'): module.verified_stage(ram, stage)
    assert (ram / 'runs/new/driver.csv').read_bytes() == b't,v\n0,0\n'
    assert (stage / 'runs/new/driver.csv').read_text() == 'corrupt'
    assert (disk / 'runs/.last_run').read_text().strip() == 'old'


def test_unexpected_top_level_output_cannot_be_silently_discarded(tmp_path):
    ram, stage, disk = fixture_tree(tmp_path)
    (ram / 'diagnostics/important.csv').write_text('keep')
    module.verified_stage(ram, stage)
    with pytest.raises(RuntimeError, match='Unrecognized output'): module.publish(stage, disk / 'runs', disk / 'diagnostics')
    assert (stage / 'diagnostics/important.csv').read_text() == 'keep'
    assert not (disk / 'runs/new').exists()


def test_capture_symlink_is_rejected_without_following_external_files(tmp_path):
    ram, stage, disk = fixture_tree(tmp_path)
    (ram / 'runs/new/link').symlink_to(disk / 'runs/old/data')
    with pytest.raises(RuntimeError, match='symlink'): module.verified_stage(ram, stage)
    assert not stage.exists()
    assert (disk / 'runs/old/data').read_text() == 'preserve'
