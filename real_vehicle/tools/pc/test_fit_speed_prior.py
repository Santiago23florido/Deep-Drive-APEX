"""An offline hardware calibration must not fit its replay-validation run."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('fit_speed_prior_tool', Path(__file__).with_name('fit_speed_prior.py'))
FIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIT)


def run(points):
    return dict(run_id='synthetic', source_sha256={},
                segments=[dict(command_median=u, speed_median=v) for u, v in points])


def test_independent_validation_does_not_change_the_fitted_hardware_map():
    training = run([(u, 6.3*u-.08) for u in [.02, .03, .04, .05, .06, .08, .1]])
    a = FIT.fit_prior([training], [run([(.08, .4)])])
    b = FIT.fit_prior([training], [run([(.08, 20.)])])
    assert a['configured_prior'] == b['configured_prior'] == dict(gain=6.3, offset=-.08)
    assert a['normalized_forward_bridge']['esc_zero_pct'] == 15.42
    assert a['normalized_forward_bridge']['pct_per_mps'] == 5.24


def test_one_command_level_cannot_establish_a_gain_and_offset():
    with pytest.raises(ValueError, match='single command'):
        FIT.fit_prior([run([(.09, .5)] * 30)], [])
