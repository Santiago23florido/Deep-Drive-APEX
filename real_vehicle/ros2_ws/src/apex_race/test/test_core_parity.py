"""The car's driving logic must be the simulation's, not a relative of it.

The shared modules of ``core/`` and the ROS helpers are copies of the simulation's.
A copy that drifts is the worst failure mode available here: the thing validated
in Gazebo and in the 2D harness would stop being the thing the car runs, and
nothing would say so. So the copy is checked mechanically, two ways:

* byte-identical -- the file, including the comments, which carry the reasoning
  behind nearly every constant in these modules;
* semantically identical -- the unparsed AST, so a failure of the first test
  immediately tells you whether the drift is cosmetic or behavioural.

Only two files of ``core/`` are allowed to differ, and they are the two the
simulation cannot provide: ``sim_paths.py`` (there is no simulation tree on the
car) and ``vehicle_io.py`` (the simulation reads the vehicle description with a
module that also builds the Gazebo model).

The suite is a *pre-deployment* gate: it skips where there is no simulation
tree, which is exactly the situation on the car.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

# Paths, computed here and not imported from `conftest`: pytest keeps one
# module named `conftest` per session, so a suite that also collects the
# simulation's tests would import the wrong one.
PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[3]
SIM_ROOT = REPO_ROOT / "simulation"
SIM_TEST = SIM_ROOT / "ros2_ws" / "src" / "apex_fusion_research" / "test"

SIM_PKG = SIM_ROOT / "ros2_ws" / "src" / "apex_fusion_research" / "apex_fusion_research"
SIM_POSE = SIM_ROOT / "tools" / "pose_dataset"
CAR_PKG = PKG_ROOT / "apex_race"

pytestmark = pytest.mark.skipif(not SIM_PKG.is_dir(), reason="no simulation tree (running on the car)")

# Copied verbatim: (path inside the package, path inside the simulation package).
COPIES = [(f"core/{name}.py", f"core/{name}.py") for name in (
    "config_io", "race_config", "map_metrics", "occupancy", "local_cloud", "lap_closure",
    "pose_correction", "reactive", "race_planner", "plant_id", "adaptive", "race_driver", "recovery",
)] + [("nodes/_common.py", "nodes/_common.py"), ("nodes/estimator_health.py", "nodes/estimator_health.py")]

# Written for the car, and the only files of core/ that may differ.
CAR_ONLY = {"core/sim_paths.py", "core/vehicle_io.py", "core/__init__.py"}

# Vendored object by object, because only part of these two modules may travel.
VENDORED = [
    ("vendor/pose_dataset/tracks.py", SIM_POSE / "tracks.py", ["Path2D", "path_from_points"]),
    ("vendor/pose_dataset/controller.py", SIM_POSE / "controller.py", ["_forward_backward", "PurePursuit"]),
]


def _ast_hash(source: str) -> str:
    return hashlib.sha256(ast.unparse(ast.parse(source)).encode()).hexdigest()


def _top_level(source: str, name: str) -> str:
    """The unparsed source of one top-level class or function."""
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} is not defined at module level")


@pytest.mark.parametrize(("car_rel", "sim_rel"), COPIES, ids=[c for c, _ in COPIES])
def test_copied_modules_are_byte_identical(car_rel: str, sim_rel: str):
    car, sim = (CAR_PKG / car_rel).read_bytes(), (SIM_PKG / sim_rel).read_bytes()
    assert car == sim, (f"{car_rel} drifted from the simulation. Re-copy it:\n"
                        f"  cp {SIM_PKG / sim_rel} {CAR_PKG / car_rel}\n"
                        f"(if the change belongs on the car only, it does not belong in this file)")


@pytest.mark.parametrize(("car_rel", "sim_rel"), COPIES, ids=[c for c, _ in COPIES])
def test_copied_modules_are_semantically_identical(car_rel: str, sim_rel: str):
    car = _ast_hash((CAR_PKG / car_rel).read_text(encoding="utf-8"))
    sim = _ast_hash((SIM_PKG / sim_rel).read_text(encoding="utf-8"))
    assert car == sim, f"{car_rel}: the code itself differs from the simulation, not just the comments"


@pytest.mark.parametrize(("car_rel", "sim_path", "names"), VENDORED, ids=[v[0] for v in VENDORED])
def test_vendored_objects_match_their_originals(car_rel: str, sim_path: Path, names: list[str]):
    car_src = (CAR_PKG / car_rel).read_text(encoding="utf-8")
    sim_src = sim_path.read_text(encoding="utf-8")
    for name in names:
        assert _top_level(car_src, name) == _top_level(sim_src, name), f"{car_rel}: {name} drifted from {sim_path.name}"


def test_every_core_module_is_either_copied_or_declared_car_only():
    """A new file in core/ must be a deliberate decision, not an accident."""
    copied = {c for c, _ in COPIES}
    present = {f"core/{p.name}" for p in (CAR_PKG / "core").glob("*.py")}
    unaccounted = present - copied - CAR_ONLY
    assert not unaccounted, f"core/ holds files that are neither copies nor declared car-only: {sorted(unaccounted)}"


def test_the_two_car_only_modules_really_are_different():
    """If the simulation ever grows the car's version, the copy list must say so."""
    for rel in ("core/sim_paths.py", "core/vehicle_io.py"):
        sim_file = SIM_PKG / rel
        if not sim_file.exists():
            continue
        assert (CAR_PKG / rel).read_bytes() != sim_file.read_bytes(), f"{rel} is identical to the simulation's: move it to COPIES"
