#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "analysis" / "analyze_forward_raw_capture.py"
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
