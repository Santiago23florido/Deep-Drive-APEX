#!/usr/bin/env python3
"""Report development calibration and a Gazebo run without turning it into a pass.

Consumes saved files only. The onboard/reference comparison measures tracking,
not external odometry accuracy. Simulator truth is used only for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--gazebo-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.fit.read_text())
    run = args.gazebo_run
    result = json.loads((run / 'run_result.json').read_text())
    events = json.loads((run / 'race_events.json').read_text())
    slam = json.loads((run / 'slam_metrics.json').read_text())
    rows = list(csv.DictReader((run / 'driver.csv').open()))
    t, target, measured, command = (np.array([float(row[key]) for row in rows])
                                    for key in ('t', 'v_target', 'speed', 'v_cmd'))
    # Left-held control reference; exclude waits, reverses and speed below 0.05.
    dt = np.r_[np.diff(t), 0.]
    active = (target > .1) & (measured > .05) & (dt > 0.)
    mean = lambda x: float(np.average(x[active], weights=dt[active]))
    metrics = dict(reference_mean_mps=mean(target), onboard_mean_mps=mean(measured),
                   mae_mps=mean(abs(target-measured)), selected_duration_s=float(dt[active].sum()),
                   selection='v_target > 0.1, onboard speed > 0.05, positive time interval',
                   interpretation='Reference tracking against onboard estimates; not odometry accuracy')
    sources = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in
               ('driver.csv', 'run_result.json', 'run_config.json', 'race_events.json', 'slam_metrics.json')}
    summary = dict(hardware_validated=False, neural_retraining=False,
                   control_law='Existing geometric control, inverse plant model and bounded P feedback',
                   forward_bridge=report['normalized_forward_bridge'],
                   replay_validation=[r for r in report['runs'] if r['role'] == 'replay_validation'],
                   gazebo=dict(run=run.name, source_sha256=sources, result=result,
                               recognized_laps=max(int(row['lap_own']) for row in rows),
                               tracking=metrics, slam_metrics=slam,
                               # The launcher's vehicle YAML overrides individual vehicle fields.
                               vehicle_yaml=json.loads((run / 'run_config.json').read_text())['car']['vehicle_yaml'],
                               initial_deadband_command=float(rows[0]['u_dead']),
                               limitation='Full mission failed timeout. Simulator vehicle YAML was used; '
                               'vehicle.min_effective_mps override was superseded by that YAML.'),
                   limitations=['No new physical validation while the car is disconnected.',
                                'Existing physical data informed this calibration; it is not untouched zero-shot evidence.',
                                'The complete Gazebo mission is a failure even though longitudinal tracking is close.',
                                'The approved physical acquisition and its figures remain unchanged.'])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    # A portable copy of the plotted inputs; all rows are retained.
    with (out / 'gazebo_speed_trace.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['time_since_explore_s', 'reference_mps', 'onboard_speed_mps', 'actuator_input', 'recognized_laps'])
        writer.writerows(zip(t-events['t_explore'], target, measured, command, [r['lap_own'] for r in rows]))
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout='constrained')
    ax = axes[0]
    for role, marker, label in [('training', 'o', 'Calibration runs'),
                                ('replay_validation', '^', 'Approved run (replay)')]:
        points = [s for r in report['runs'] if r['role'] == role for s in r['segments']]
        ax.scatter([s['command_median'] for s in points], [s['speed_median'] for s in points],
                   marker=marker, s=23, label=label)
    x = np.linspace(0., .12, 100)
    p = report['configured_prior']
    ax.plot(x, np.maximum(0., p['gain']*x+p['offset']), c='#222222', label='Moving-branch fit')
    ax.set(xlabel='Previous actuator-channel input', ylabel='Onboard speed [m/s]',
           title='(a) Archived real-vehicle observations')
    ax.legend(fontsize=6.5)
    ax = axes[1]
    moving = (t >= events['t_explore']) & (t <= events['t_explore']+65.)
    ax.plot(t[moving]-events['t_explore'], target[moving], c='#222222', lw=1., label='Control reference')
    ax.plot(t[moving]-events['t_explore'], measured[moving], c='#0072b2', lw=.7, label='Onboard estimate')
    ax.set(xlabel='Time since exploration start [s]', ylabel='Speed [m/s]',
           title='(b) Gazebo tracking (mission timed out)')
    ax.legend(fontsize=6.5)
    for ax in axes:
        ax.grid(alpha=.2)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('speed_calibration.'+extension), dpi=350)
    plt.close(fig)
    svg = out / 'speed_calibration.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
