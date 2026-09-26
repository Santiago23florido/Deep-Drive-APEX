import sys
from pathlib import Path

# Allow running the unit tests without building/sourcing the ROS workspace.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs of the 2D race harness (~15 s)")
