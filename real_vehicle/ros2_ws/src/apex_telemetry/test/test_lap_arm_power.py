"""An autonomous lap must release arm on missing or live-fault host telemetry."""
import importlib.util
import os
from pathlib import Path
import time

import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location('lap_arm', ROOT / 'tools/core/arm_until_done.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('flags,temp,expected', [
    ('0x0', '60', True), ('0x50000', '60', True),
    ('0x50005', '60', False), ('0x0', '75', False), ('0x0', 'nan', False),
])
def test_current_flags_and_temperature_determine_arm_permission(tmp_path, flags, temp, expected):
    path = tmp_path / 'power.csv'
    path.write_text('utc_iso,lcrit_alarm,throttled,temp_c,freq_khz\nnow,0,'+flags+','+temp+',2000000\n')
    assert MODULE.power_ok(path) is expected


def test_missing_partial_and_stale_power_log_do_not_keep_arm_alive(tmp_path):
    path = tmp_path / 'power.csv'
    assert not MODULE.power_ok(path)
    path.write_text('utc_iso,lcrit_alarm,throttled,temp_c,freq_khz\nnow,0,')
    assert not MODULE.power_ok(path)
    path.write_text('utc_iso,lcrit_alarm,throttled,temp_c,freq_khz\nnow,0,0x0,60,2000000\n')
    os.utime(path, (time.time()-4, time.time()-4))
    assert not MODULE.power_ok(path)
