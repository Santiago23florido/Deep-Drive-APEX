import sys
from pathlib import Path

# Allow running the unit tests without building/sourcing the ROS workspace.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
