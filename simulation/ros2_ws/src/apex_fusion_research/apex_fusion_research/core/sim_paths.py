"""Location of the ``simulation/`` tree (ROS-independent).

The nodes, the offline tools and the tests share the pose-dataset package
(``simulation/tools/pose_dataset``: tracks, vehicle model, driver) through
``sys.path``; nothing here imports rclpy.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def sim_root() -> Path:
    env = os.environ.get("APEX_SIM_ROOT", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tools" / "pose_dataset").is_dir():
            return parent
    raise RuntimeError("cannot locate simulation/ (set APEX_SIM_ROOT)")


def ensure_tools_path() -> Path:
    """Make ``simulation/tools`` importable (``pose_dataset``); return it."""
    tools = sim_root() / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return tools
