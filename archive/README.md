# Archive

This directory contains historical, generated, or non-operational material. Nothing here is part of the recommended simulation or real-vehicle startup path.

The versioned publication copy of the current approved real capture is
[run 20261024_005311](../real_vehicle/docs/paper/20261024_005311/README.md), including
six English figure sets and the verified portable data/source/model archive.
Local copies under `generated/paper/` are working archival storage; the committed
paper directory is the downloadable repository reference. See the
[preprint readiness report](../docs/reports/preprint_readiness_20261024.md) for
remaining simulation work. Historical and third-party READMEs below preserve
their original context and language.

## Contents

| Path | Purpose |
| --- | --- |
| `legacy/full_soft/` | Old Python vehicle stack that operates without SLAM. |
| `legacy/Lidar/` | Older distributed LiDAR utilities. |
| `legacy/voiture_system_sim_source/` | Historical simulation source snapshot. |
| `artifacts/` | Images and logs kept for reference. |
| `generated/` | Archived builds, caches, and runtime state. |
| `docs_latex_artifacts/` | Generated LaTeX artifacts. |

Do not add `archive/` to `colcon` base paths or `PYTHONPATH`. Recover individual algorithms into an active workspace through small, reviewed, and tested ports.

See [Known Limitations and Legacy Code](../docs/16_known_limitations_and_legacy_parts.md).
