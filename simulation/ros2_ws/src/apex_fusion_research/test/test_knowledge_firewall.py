"""The race driver may only use what the real car has: its odometry model, its
own limits and the noisy sensors (through the estimator and the SLAM). No
ground truth, no track definition, no per-track speed calibration."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "apex_fusion_research"
CAR_SIDE = [
    PKG / "nodes" / "race_driver_node.py",
    PKG / "core" / "race_driver.py",
    PKG / "core" / "race_config.py",
    PKG / "core" / "reactive.py",
    PKG / "core" / "local_cloud.py",
    PKG / "core" / "lap_closure.py",
    PKG / "core" / "pose_correction.py",
    PKG / "core" / "race_planner.py",
]
FORBIDDEN = ("run_setup", "load_track", "speed_calibration", "/apex/sim/", "ground_truth", "truth", "race_eval", "clearance import",
             "_pose_tools", "gz_runner", "tracks_index")


def _code_without_docstrings(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_car_side_code_has_no_truth_or_track_access():
    for path in CAR_SIDE:
        code = _code_without_docstrings(path)
        for word in FORBIDDEN:
            assert word not in code, f"{path.name} uses {word!r}"


def test_race_driver_declares_no_track_parameters():
    tree = ast.parse((PKG / "nodes" / "race_driver_node.py").read_text(encoding="utf-8"))
    declared = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "dp" and n.args and isinstance(n.args[0], ast.Constant)}
    assert declared.isdisjoint({"track", "motion", "seed", "v_ref", "start_pose", "truth_topic", "pose_source"})
    assert "laps" in declared


def test_only_pose_dataset_utilities_are_imported():
    allowed = {"pose_dataset.controller", "pose_dataset.tracks", "pose_dataset.vehicle"}
    for path in CAR_SIDE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pose_dataset"):
                assert node.module in allowed, f"{path.name} imports {node.module}"
                names = {a.name for a in node.names}
                if node.module == "pose_dataset.tracks":
                    assert names <= {"Path2D", "path_from_points"}, f"{path.name}: {names}"
                if node.module == "pose_dataset.vehicle":
                    assert names <= {"load_vehicle_config"}, f"{path.name}: {names}"
