# Curve Window Detection

## Responsibility

`curve_window_detection.py` turns a static LiDAR snapshot into a curve-entry interpretation.

Its output is not only a binary "curve detected" flag. It also provides:

- the detected candidate side,
- the first corridor geometry,
- the second-corridor axis,
- the entry line center,
- the target point inside the second corridor,
- and a local path skeleton that can be passed to the world-frame planner.

## Inputs

The main input is a 1D LiDAR range array. The module first converts it to forward-left Cartesian coordinates with:

- `scan_ranges_to_forward_left_xy()`

The coordinate convention is:

- `x`: forward
- `y`: left

This convention is used consistently through the perception and planning stack.

## Core Data Structures

The file defines several data classes:

- `CurveWindowDetectionConfig`
- `SideProfile`
- `CurveWindowCandidate`
- `CurveWindowTrajectory`
- `CurveWindowDetectionResult`

This structure is important because the module is explicitly split into:

- raw geometric evidence,
- candidate selection,
- and trajectory generation.

That separation is what allows later modules to preserve the detection logic while changing only the path-shaping logic.

## Algorithm Overview

The pipeline is:

1. Convert LiDAR ranges to forward-left points.
2. Build one lateral profile for the left wall and one for the right wall.
3. Fit a straight baseline on the pre-curve region.
4. Measure deviations from that baseline.
5. Detect candidate curve openings.
6. Score the candidates.
7. Build a local entry trajectory from the winning candidate.

## Side Profiles

`_build_side_profile()` bins the LiDAR cloud in `x` and keeps one median `y` value per bin for each side.

The theoretical reason is simple:

- raw point clouds are noisy,
- walls are approximately continuous,
- and median values per longitudinal bin give a robust corridor estimate.

A line is fitted on the pre-curve region. This line acts as the "expected straight corridor wall". Deviations from it are then interpreted as evidence of a bend or an opening.

## Candidate Detection

`_detect_curve_candidate()` uses several cues:

- baseline deviation,
- same-side gap,
- opposite-side continuation,
- front closure,
- and visible corridor width.

This is a heuristic detector, not an optimizer or a machine learning model.

Its strength is that each condition corresponds to a physical scene interpretation:

- a same-side gap often means the corridor opened,
- opposite-side continuity means the far wall is still visible,
- front closure means the scene is not just open space,
- and sustained baseline deviation means the wall really bent.

## Why the Detection Logic Was Preserved

The main requirement for this work was to keep the window-identification logic intact. That means:

- the candidate search remains the same,
- the scoring remains the same,
- and the detected opening still comes from the same evidence.

The changes were applied only to how the local target and the local path are constructed after the candidate already exists.

## Gap-Only Opening Geometry

The critical branch is:

- `if candidate.gap_only_opening:`

This case appears when the curve is inferred mainly from an opening rather than from a fully visible smooth inner bend.

The current implementation does the following:

1. It keeps the original second-corridor axis guess.
2. It looks for same-side points after the gap.
3. It estimates the outer visible branch from those post-gap points.
4. It lifts the entry line center toward that visible branch.
5. It keeps the target depth inside the second corridor fixed by configuration.
6. It constrains the target so it remains between the opposite wall and the visible outer branch.
7. It builds a Catmull-Rom path through anchor points that approach the entry line and then enter the second corridor.

The key practical effect is this:

- the detector still identifies the same opening,
- but the generated target is no longer allowed to drop into the lower wall region just because the same-side wall disappeared locally.

## Important Parameters

These parameters are defined in `CurveWindowDetectionConfig` and populated from YAML by the planner node:

- `x_bin_m`
- `fit_x_min_m`
- `fit_x_max_m`
- `search_x_min_m`
- `deviation_threshold_m`
- `gap_threshold_m`
- `opposite_continuation_min_m`
- `front_closure_x_window_m`
- `front_closure_min_points`
- `second_corridor_target_depth_m`
- `second_corridor_target_depth_min_m`
- `second_corridor_target_depth_max_m`
- `inner_vertex_clearance_m`
- `curve_apex_width_fraction`

In practice, the most influential ones for geometry are:

- `second_corridor_target_depth_*`
- `inner_vertex_clearance_m`
- `curve_apex_width_fraction`

## Theoretical Notes

### Why median wall profiles work

If a wall is approximately planar or slowly varying in `x`, then the median `y` value in each `x` bin suppresses outliers while preserving corridor structure.

### Why baseline deviation is useful

A curve can be interpreted as a controlled departure from the local straight-wall model. Measuring the deviation from a fitted line is therefore a simple way to estimate "curve-ness".

### Why the second corridor should be tied to visible geometry

In the problematic runs, the old geometry extrapolated through the gap. That is unstable because no actual measurements support that lower target placement. The current implementation instead biases the entry line toward the visible upper branch of the opening.

## Relevant File

- `ros2_ws/src/apex_telemetry/apex_telemetry/perception/curve_window_detection.py`
