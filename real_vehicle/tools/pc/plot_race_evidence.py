#!/usr/bin/env python3
"""Export English publication figures from a real autonomous run's saved logs.

Uses the actual saved SLAM grid and onboard estimates. It does not reconstruct
an occupancy map, invent ground truth, or count unrecorded controller phases.
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

PHASES = {
    'explore': ('Exploration', '#2171b5'),
    'closing': ('Loop overlap', '#e69f00'),
    'planning': ('Global planning', '#9467bd'),
    'map_guided': ('SLAM-guided local planning', '#009e73'),
    'race': ('Direct path tracking', '#d55e00'),
    'reactive_only': ('Reactive fallback', '#777777'),
    'final': ('Finishing', '#555555'),
    'done': ('Stopped', '#222222'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='output basename without extension')
    args = parser.parse_args()
    source, base = args.run_directory, args.output
    base.parent.mkdir(parents=True, exist_ok=True)
    events = json.loads((source / 'race_events.json').read_text())
    rows = list(csv.DictReader((source / 'driver.csv').open()))
    rows = [r for r in rows if r.get('phase') in PHASES and r.get('recover') is not None]
    if not rows:
        raise ValueError('no recorded execution samples')
    t0 = events.get('t_explore', float(rows[0]['t']))
    t = np.array([float(r['t']) - t0 for r in rows])
    values = {k: np.array([float(r[k]) for r in rows]) for k in
              ('x_map', 'y_map', 'speed', 'v_cmd', 'steer_deg', 'lap_own', 'recover')}
    phase = np.array([r['phase'] for r in rows])
    transitions = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1, len(rows)]
    phases_present = list(dict.fromkeys(phase.tolist()))
    with np.load(source / 'slam_map_final.npz', allow_pickle=False) as data:
        occupancy = data['occupancy'].copy()
        meta = json.loads(str(data['metadata']))
    if abs(meta['origin']['qz']) > 1e-6:
        raise ValueError('rotated occupancy origin requires an explicit map transform')
    extent = [meta['origin']['x'], meta['origin']['x'] + meta['width'] * meta['resolution'],
              meta['origin']['y'], meta['origin']['y'] + meta['height'] * meta['resolution']]
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(7.16, 4.65), layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=(1.25, 1))
    ax_map = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_steer = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_laps = fig.add_subplot(gs[2, 1], sharex=ax_v)
    image = np.where(occupancy < 0, .65, 1 - occupancy.astype(float) / 100)
    ax_map.imshow(image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest', rasterized=True)
    route_file = source / 'map_route.csv'
    if not route_file.exists():
        route_file = source / 'plan/map_route.csv'
    if route_file.exists() and 'plan' in events:
        route = np.genfromtxt(route_file, delimiter=',', names=True)
        ax_map.plot(route['x'], route['y'], '--', lw=.8, c='#292929', label='SLAM route (guide)', zorder=2)
    seen = set()
    for a, b in zip(transitions[:-1], transitions[1:]):
        label, color = PHASES[phase[a]]
        end = min(b + 1, len(rows))
        ax_map.plot(values['x_map'][a:end], values['y_map'][a:end], lw=1.4, c=color,
                    label=label if phase[a] not in seen else None, zorder=3)
        seen.add(phase[a])
        for ax in (ax_v, ax_steer, ax_laps):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=.08, lw=0)
            if a:
                ax.axvline(t[a], color=color, lw=.5, alpha=.6)
    ax_map.scatter(values['x_map'][0], values['y_map'][0], s=35, marker='*', c='#111111', label='Start', zorder=5)
    ax_map.scatter(values['x_map'][-1], values['y_map'][-1], s=20, marker='x', c='#111111', label='Final estimate', zorder=5)
    ax_map.set(xlabel='Map x [m]', ylabel='Map y [m]', title='(a) Onboard SLAM and estimated trajectory', aspect='equal')
    ax_map.legend(loc='lower left', framealpha=.95)
    has_reference = all('v_target' in row for row in rows)
    if has_reference:
        reference = np.array([float(row['v_target']) for row in rows])
        ax_v.plot(t, reference, lw=.8, c='#222222', label='Control reference')
        actuator_axis = ax_v.twinx()
        actuator_axis.plot(t, values['v_cmd'], lw=.5, ls=':', c='#888888', alpha=.65)
        actuator_axis.set_ylabel('Actuator-channel input', color='#666666')
        actuator_axis.tick_params(axis='y', colors='#666666')
    else:
        ax_v.plot(t, values['v_cmd'], lw=.8, c='#222222', label='Actuator-channel command')
    ax_v.plot(t, values['speed'], lw=.7, c='#0072b2', alpha=.85, label='Onboard estimate')
    ax_v.set(ylabel='Speed [m/s]', title='(b) Longitudinal motion')
    ax_v.axhline(0, c='.5', lw=.4)
    ax_v.legend(loc='best', ncol=2)
    ax_steer.plot(t, values['steer_deg'], lw=.7, c='#6f4693')
    ax_steer.set(ylabel='Command [deg]', title='(c) Steering command')
    ax_steer.axhline(0, c='.5', lw=.4)
    ax_laps.step(t, values['lap_own'], where='post', lw=1., c='#222222')
    maximum_lap = int(values['lap_own'].max())
    ax_laps.set(xlabel='Time [s]', ylabel='Lap count', title='(d) Recognized laps',
                yticks=range(maximum_lap + 1), ylim=(-.15, max(1.15, maximum_lap + .3)))
    for ax in (ax_v, ax_steer, ax_laps):
        ax.set_xlim(max(0, t.min()), t.max())
        ax.grid(alpha=.2, lw=.5)
    ax_v.tick_params(labelbottom=False)
    ax_steer.tick_params(labelbottom=False)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(base.with_suffix('.' + suffix), dpi=600)
    plt.close(fig)

    fields = ('time_s', 'phase', 'map_x_m', 'map_y_m', 'estimated_speed_mps', 'actuator_channel_input',
              'steering_command_deg', 'recognized_laps', 'recovery_mode', 'control_reference_mps')
    with base.with_name(base.name + '_trace.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, row in enumerate(rows):
            writer.writerow([t[i], phase[i], *(values[k][i] for k in
                ('x_map', 'y_map', 'speed', 'v_cmd', 'steer_deg', 'lap_own', 'recover')), row.get('v_target', '')])
    run_id = (source / 'run_id.txt').read_text().strip() if (source / 'run_id.txt').exists() else source.name
    caption = (f'Real-vehicle autonomous execution, run {run_id}. '
               '(a) Actual SLAM occupancy grid with the estimated vehicle trajectory colored by controller phase. '
               + ('(b) Actual control reference and onboard speed; dotted right-axis trace is the final actuator-channel input. '
                if has_reference else '(b) Final actuator-channel command and onboard speed; the planner reference was not logged. ')
               + '(c) Steering command in nominal servo-angle units. '
               '(d) Laps recognized by the onboard return detector. Shading marks recorded controller phases. '
               'Map coordinates and motion estimates are onboard estimates, not external ground truth. '
               'A SLAM route, when shown, guides live LiDAR local planning; it is not a claim of direct tracking of an infeasible trajectory. '
               + f'The onboard detector recorded {maximum_lap} completed lap(s) out of {events.get("laps", "unspecified")} requested.')
    base.with_name(base.name + '_caption.txt').write_text(caption + '\n')
    base.with_suffix('.tex').write_text('\\begin{figure*}[t]\n\\centering\n'
        + '\\includegraphics[width=\\textwidth]{' + base.with_suffix('.pdf').name + '}\n'
        + '\\caption{' + caption.replace('_', '\\_') + '}\n'
        + '\\label{fig:real-vehicle-execution}\n\\end{figure*}\n')
    files = ('driver.csv', 'race_events.json', 'slam_map_final.npz', 'source_manifest.json',
             'map_route.csv', 'plan.json', 'plan/map_route.csv', 'plan/plan.json', 'arm_result.json', 'bridge_before.json')
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files if (source / name).exists()}
    metadata = {'run_id': run_id, 'source_directory': str(source.resolve()), 'source_sha256': hashes,
                'recorded_phases': phases_present, 'recognized_laps': maximum_lap,
                'crossings': events.get('crossings', []), 'mission': events.get('mission'),
                'direct_tracking_allowed': events.get('plan', {}).get('direct_tracking_allowed'),
                'external_ground_truth': False, 'control_reference_logged': has_reference, 'caption': caption}
    base.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'figure_pdf': str(base.with_suffix('.pdf')), 'figure_png': str(base.with_suffix('.png')),
                      'recorded_phases': phases_present, 'recognized_laps': maximum_lap}, indent=2))


if __name__ == '__main__':
    main()
