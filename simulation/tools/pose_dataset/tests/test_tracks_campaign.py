import numpy as np
import pytest

from pose_dataset.campaign import build_plan, held_out_names
from pose_dataset.tracks import load_track, load_track_specs, rounded_polygon


def test_rounded_polygon_square():
    path, arcs = rounded_polygon([[0, 0], [10, 0], [10, 10], [0, 10]], [2, 2, 2, 2], ds=0.01)
    assert abs(path.length - (4 * 6 + 2 * np.pi * 2)) < 0.02
    assert all(abs(a["radius"] - 2) < 1e-9 and abs(a["angle_deg"] - 90) < 1e-6 for a in arcs)


@pytest.mark.parametrize("name", sorted(load_track_specs()))
def test_every_track_is_valid(name):
    t = load_track(name)
    assert t.meta["reference_length_m"] > 15
    assert t.meta["min_reference_radius_m"] >= float(t.spec.get("min_path_radius_m", 1.15)) - 1e-6


def test_main_presets_exclude_held_out_variants():
    held = held_out_names()
    assert held["tracks"] and held["motions"] and held["sensors"]
    for preset in ("smoke", "full"):
        plan = build_plan(preset)
        for t in plan["trajectories"]:
            assert t["track"] not in held["tracks"] and t["motion"] not in held["motions"]
            assert all(r["sensor"] not in held["sensors"] for r in t["runs"])


def test_full_splits_are_by_trajectory_and_whole_tracks():
    plan = build_plan("full")
    split_of = {}
    for t in plan["trajectories"]:
        assert split_of.setdefault(t["trajectory_key"], t["split"]) == t["split"]
    tracks = {}
    for t in plan["trajectories"]:
        tracks.setdefault(t["track"], set()).add(t["split"])
    assert tracks["val_mixed"] == {"validation"}
    assert tracks["test_unseen"] == {"test"}
    train_combos = {(r["sensor"], t["motion"]) for t in plan["trajectories"] if t["split"] == "train" for r in t["runs"]}
    assert ("C_fast", "high") not in train_combos and ("B_economic", "stop_and_go") not in train_combos
    test_combos = {(r["sensor"], t["motion"]) for t in plan["trajectories"] if t["split"] == "test" for r in t["runs"]}
    assert ("C_fast", "high") in test_combos
    n = {s: sum(len(t["runs"]) for t in plan["trajectories"] if t["split"] == s) for s in ("train", "validation", "test")}
    total = sum(n.values())
    assert 0.65 < n["train"] / total < 0.75 and 0.12 < n["validation"] / total < 0.18 and 0.12 < n["test"] / total < 0.18
