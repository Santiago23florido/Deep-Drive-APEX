"""Reproduce per-lap speed tracking from the two retained local acquisitions.

Run from the repository root. No ground truth, time alignment, model fitting,
hardware connection or actuator command is used by this script.
"""
import csv
import json
from pathlib import Path

import numpy as np


def analyze(root):
    run = root / 'raw/ros_run'
    events = json.loads((run / 'race_events.json').read_text())
    with (run / 'driver.csv').open() as f:
        rows = list(csv.DictReader(f))
    values = {key: np.array([float(r[key]) for r in rows]) for key in
              ('t', 'v_cmd', 'speed', 'recover', 'pose_age_s')}
    t, u, v, recovery = (values[k] for k in ('t', 'v_cmd', 'speed', 'recover'))
    target = np.array([float(r['v_target']) for r in rows]) if 'v_target' in rows[0] else None
    dt = np.r_[np.diff(t), 0.]
    bounds = [events['t_explore']] + [r['t'] for r in events['crossings']]
    stable = np.zeros(len(t), dtype=bool)
    if target is not None:
        for i in range(len(t)):
            j = np.searchsorted(t, t[i]-.8)
            stable[i] = (t[i]-t[j] >= .7 and np.all(target[j:i+1] > .1)
                         and np.ptp(target[j:i+1]) < .025 and np.all(recovery[j:i+1] == 0))
    report = []
    for lap, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        weights = np.maximum(0., np.minimum(t+dt, hi)-np.maximum(t, lo))
        def mean(data, mask):
            keep = mask & (weights > 0.)
            return float(np.average(data[keep], weights=weights[keep])) if keep.any() else None
        forward, reverse = u > .005, u < -.005
        item = dict(lap=lap, duration_s=hi-lo,
                    reverse_starts=sum(lo <= r.get('t', -1) < hi and r['kind'] == 2
                                       for r in events.get('recovery', [])),
                    forward_command_s=float(weights[forward].sum()),
                    reverse_command_s=float(weights[reverse].sum()),
                    zero_command_s=float(weights[~(forward | reverse)].sum()),
                    onboard_speed_forward_moving_mean_mps=mean(v, forward & (v > .05)))
        if target is not None:
            for label, keep in [('forward_requested', target > .005),
                                ('forward_moving', (target > .005) & (v > .05)),
                                ('settled', stable & (v > .05))]:
                item[label] = dict(selected_s=float(weights[keep].sum()),
                                   reference_mean_mps=mean(target, keep), onboard_mean_mps=mean(v, keep),
                                   mae_mps=mean(abs(target-v), keep))
        report.append(item)
    return report


if __name__ == '__main__':
    base = Path('archive/generated/paper')
    result = {run_id: analyze(base / ('real_vehicle_'+run_id))
              for run_id in ('20261024_005311', '20261024_190956')}
    output = base / 'real_vehicle_20261024_190956/tracking_reproduced.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    # Check the independently exported review, including its richer diagnostics.
    retained = json.loads(output.with_name('controller_review.json').read_text())
    for run_id, laps in result.items():
        for actual, saved in zip(laps, retained[run_id]['laps']):
            for key, value in actual.items():
                if isinstance(value, dict):
                    for name, metric in value.items():
                        assert np.isclose(metric, saved[key][name]), (run_id, key, name)
                else:
                    assert np.isclose(value, saved[key]), (run_id, key)
    print('Per-lap tracking metrics reproduced and checked against controller_review.json.')
