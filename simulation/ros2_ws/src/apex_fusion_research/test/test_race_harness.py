"""End to end on the 2D harness: reactive lap 1 on the unseen test track, lap
closure on the car's own map pose, race line planned on its map, one race lap."""

import sys
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[4]


@pytest.mark.slow
def test_unseen_track_lap1_plan_and_race():
    sys.path.insert(0, str(SIM / "tools" / "analysis"))
    import race_harness_2d as harness  # noqa: PLC0415

    from apex_fusion_research.core.race_config import RaceDriverConfig  # noqa: PLC0415

    res = harness.run_one("test_unseen", "medium", 7, 2, RaceDriverConfig(), None, 200.0)
    assert res["status"] == "completed", res["failure_reason"]
    assert len(res["crossings"]) == 2 and all(abs(c["true_progress_error_m"]) < 0.5 for c in res["crossings"])
    assert res["plan"]["lap_path_source"] == "graph"
    assert res["plan_truth"]["corridor_sound_frac"] == 1.0
    assert res["plan_truth"]["line_min_clearance_m"] > 0.05
    assert res["min_true_clearance_m"] > 0.05
    assert res["true_lap_times_s"][1] < res["true_lap_times_s"][0]  # the planned lap is faster than the exploration
