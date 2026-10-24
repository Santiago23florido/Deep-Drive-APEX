#!/usr/bin/env python3
"""Fit a nominal forward prior from archived commands/onboard speed, offline.

Whole training runs and replay-validation runs are supplied separately. Segment
medians have equal weight; the fit has no access to ground truth or track shape.
This is a moving-branch approximation, not motor metrology or a closed-loop trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def steady_segments(run: Path) -> dict:
    events = json.loads((run / 'race_events.json').read_text())
    rows = np.atleast_1d(np.genfromtxt(run / 'driver.csv', delimiter=',', names=True, dtype=None, encoding='utf-8'))
    t, u, speed = rows['t'], rows['v_cmd'], rows['speed']
    active = (u > .005) & (t >= events['t_explore']) & (t <= events['t_done'])
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    segments = []
    for a, b in zip(starts, ends):
        keep = active & (t >= t[a] + .8) & (t <= t[b] - .15) & (speed > .05)
        if keep.sum() < 4:
            continue
        segments.append(dict(start_s=float(t[a] - events['t_explore']),
                             end_s=float(t[b] - events['t_explore']), samples=int(keep.sum()),
                             command_median=float(np.median(u[keep])), speed_median=float(np.median(speed[keep]))))
    return dict(run_id=events.get('run_id', run.parent.parent.name), segments=segments,
                source_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                               for name in ['driver.csv', 'race_events.json']})


def fit_prior(training: list[dict], validation: list[dict]) -> dict:
    points = [s for run in training for s in run['segments']]
    x = np.array([s['command_median'] for s in points])
    y = np.array([s['speed_median'] for s in points])
    if len(x) < 6 or np.ptp(x) < .03:
        raise ValueError('A single command level cannot identify both gain and offset')
    result = least_squares(lambda b: b[0] * x + b[1] - y, [6., -.1],
                           bounds=([.5, -.3], [12., .3]), loss='soft_l1', f_scale=.06)
    if not result.success:
        raise ValueError(result.message)
    gain, offset = (float(value) for value in result.x)
    rounded = dict(gain=round(gain, 1), offset=round(offset, 2))
    groups = []
    for role, runs in [('training', training), ('replay_validation', validation)]:
        for run in runs:
            seg = run['segments']
            u = np.array([s['command_median'] for s in seg])
            v = np.array([s['speed_median'] for s in seg])
            if not len(seg):
                raise ValueError('No steady forward segments in supplied run')
            pred = rounded['gain'] * u + rounded['offset']
            groups.append(dict(role=role, **run, nominal_identity_mae_mps=float(np.mean(abs(u-v))),
                               initialized_model_mae_mps=float(np.mean(abs(pred-v)))))
    return dict(scope='Nominal forward moving-branch calibration from onboard estimates; no external ground truth',
                fit=dict(gain=gain, offset=offset, loss='soft_l1', equal_weight_per_segment=True),
                configured_prior=rounded, command_for_reference_0_5_mps=(.5-rounded['offset'])/rounded['gain'],
                normalized_forward_bridge=dict(esc_zero_pct=round(15.-33.*rounded['offset']/rounded['gain'],2),
                                               pct_per_mps=round(33./rounded['gain'],2),max_pct=28.2,
                                               kalman_initial_gain=1.0,kalman_initial_offset=0.0),
                selection=dict(trim_start_s=.8, trim_end_s=.15, min_samples=4, min_onboard_speed_mps=.05),
                runs=groups,
                limitations=['Inputs are closed-loop observations, with unmeasured battery/load changes.',
                             'Replay validation is not a new physical or untouched zero-shot trial.',
                             'Gain and offset retain declared uncertainty; no Kalman confidence is forced.',
                             'The controller reference is used directly; only the forward hardware map and declared speed parameters are calibrated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--validation', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if set(p.resolve() for p in args.train) & set(p.resolve() for p in args.validation):
        parser.error('Training and replay-validation runs must be disjoint')
    report = fit_prior([steady_segments(p) for p in args.train], [steady_segments(p) for p in args.validation])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['fit', 'configured_prior', 'command_for_reference_0_5_mps']}, indent=2))


if __name__ == '__main__':
    main()
