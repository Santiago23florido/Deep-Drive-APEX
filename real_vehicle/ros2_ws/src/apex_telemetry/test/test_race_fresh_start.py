"""Fresh starts must preserve evidence and weights and never reuse runtime output."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[4] / 'tools/core/race_fresh_start.py'
spec = importlib.util.spec_from_file_location('race_fresh_start', MODULE)
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def test_cache_cleanup_preserves_previous_logs_weights_and_maps(tmp_path):
    old = tmp_path / 'ros2_ws/apex_race/old'
    old.mkdir(parents=True)
    (old / 'driver.csv').write_text('evidence')
    (old.parent / '.last_run').write_text('old\n')
    for name in ('model.pt', 'track.posegraph'):
        (tmp_path / name).write_text('preserve')
    cache = tmp_path / 'ros2_ws/src/apex_race/apex_race/core/__pycache__'
    cache.mkdir(parents=True)
    (cache / 'driver.pyc').write_bytes(b'stale')
    run = fresh.prepare_run(tmp_path, stamp='new')
    assert not cache.exists()
    assert (old / 'driver.csv').read_text() == 'evidence'
    assert all((tmp_path / name).read_text() == 'preserve' for name in ('model.pt', 'track.posegraph'))
    assert json.loads((run / 'fresh_start.json').read_text())['previous_run'] == 'old'
    assert (old.parent / '.last_run').read_text().strip() == run.name


def test_two_starts_in_the_same_second_get_different_directories(tmp_path):
    first = fresh.prepare_run(tmp_path, stamp='20261023_200000')
    (first / 'driver.csv').write_text('first')
    second = fresh.prepare_run(tmp_path, stamp='20261023_200000')
    assert first != second
    assert (first / 'driver.csv').read_text() == 'first'
    assert not (second / 'driver.csv').exists()


def test_cleanup_does_not_follow_a_cache_symlink(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'data').write_text('keep')
    package = tmp_path / 'ros2_ws/src/apex_race/apex_race'
    package.mkdir(parents=True)
    (package / '__pycache__').symlink_to(outside, target_is_directory=True)
    fresh.prepare_run(tmp_path, stamp='new')
    assert (outside / 'data').read_text() == 'keep'


@pytest.mark.parametrize('duty, ok', [(1500000, True), (1700000, False)])
def test_motor_must_be_neutral_before_start(tmp_path, duty, ok):
    (tmp_path / 'period').write_text('20000000')
    (tmp_path / 'duty_cycle').write_text(str(duty))
    if ok:
        fresh.verify_neutral(tmp_path)
    else:
        with pytest.raises(RuntimeError, match='neutral'):
            fresh.verify_neutral(tmp_path)


def test_disabled_pwm_is_safe_before_initial_hardware_setup(tmp_path):
    (tmp_path / 'enable').write_text('0')
    fresh.verify_neutral(tmp_path)
