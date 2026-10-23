#!/usr/bin/env python3
"""Additional English figures and metrics from a saved real-vehicle run.

Reads the actual captured SLAM grid and driver logs. No connection to hardware
and no external ground-truth claim. The lap comparison is descriptive, from
one run, rather than a controlled ablation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.run_directory, args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads((source / 'race_events.json').read_text())
    rows = [r for r in csv.DictReader((source / 'driver.csv').open()) if r['phase'] != 'wait']
    t0 = events['t_explore']
    t = np.array([float(r['t']) - t0 for r in rows])
    v = {name: np.array([float(r[name]) for r in rows]) for name in
         ('x_map', 'y_map', 'v_cmd', 'speed', 'steer_deg', 'g_left', 'g_right', 'recover', 'lap_own')}
    phase = np.array([r['phase'] for r in rows])
    crossings = events['crossings']
    cross = np.array([c['t'] - t0 for c in crossings])
    reversals = []
    pending = None
    for event in events.get('recovery', []):
        if 't' in event:
            pending = event
        elif pending is not None and 't_end' in event:
            reversals.append({'start_s': pending['t'] - t0, 'end_s': event['t_end'] - t0,
                              'phase_at_start': pending['phase'], 'distance_m': event.get('travel_m')})
            pending = None
    if pending is not None:
        reversals.append({'start_s': pending['t'] - t0, 'end_s': t[-1],
                          'phase_at_start': pending['phase'], 'distance_m': None})
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name, caption):
        for extension in ('pdf', 'svg', 'png'):
            fig.savefig(out / f'{name}.{extension}', dpi=600)
        (out / f'{name}_caption.txt').write_text(caption + '\n')
        plt.close(fig)

    with np.load(source / 'slam_map_final.npz', allow_pickle=False) as data:
        occ = data['occupancy'].copy()
        meta = json.loads(str(data['metadata']))
    if abs(meta['origin']['qz']) > 1e-6:
        raise ValueError('rotated map origin requires an explicit map transform')
    extent = [meta['origin']['x'], meta['origin']['x'] + meta['width'] * meta['resolution'],
              meta['origin']['y'], meta['origin']['y'] + meta['height'] * meta['resolution']]
    image = np.where(occ < 0, .65, 1 - occ.astype(float) / 100)
    fig, ax = plt.subplots(figsize=(5.3, 4.7), layout='constrained')
    ax.imshow(image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1,
              interpolation='nearest', rasterized=True)
    route_file = source / 'map_route.csv'
    if route_file.exists():
        route = np.genfromtxt(route_file, delimiter=',', names=True)
        ax.plot(route['x'], route['y'], '--', color='.25', lw=.9, label='SLAM route (guide)')
    for i, (a, b, label, color) in enumerate([
            (0., cross[0], 'Lap 1: reconnaissance', '#0072b2'),
            (cross[0], cross[1] if len(cross) > 1 else t[-1],
             'Lap 2: planning and SLAM guidance' + (' (partial)' if len(cross) < 2 else ''), '#009e73')]):
        indices = np.flatnonzero((t >= a) & (t <= b))
        if len(indices):
            indices = np.arange(max(0, indices[0] - 1), min(len(t), indices[-1] + 2))
            ax.plot(v['x_map'][indices], v['y_map'][indices], color=color, lw=1.5, label=label)
    if len(cross) >= 2:
        stop = t >= cross[-1]
        ax.plot(v['x_map'][stop], v['y_map'][stop], color='.4', lw=1., label='Final stopping segment')
    for i, recovery in enumerate(reversals):
        k = int(np.argmin(abs(t - .5 * (recovery['start_s'] + recovery['end_s']))))
        ax.scatter(v['x_map'][k], v['y_map'][k], marker='v', s=26, color='#d55e00',
                   label='Reverse maneuver' if i == 0 else None, zorder=5)
    ax.scatter(v['x_map'][0], v['y_map'][0], marker='*', s=65, color='k', label='Start', zorder=6)
    ax.scatter(v['x_map'][-1], v['y_map'][-1], marker='x', s=35, color='k', label='Final estimate', zorder=6)
    ax.set(xlabel='Map x [m]', ylabel='Map y [m]', aspect='equal',
           title='Real-vehicle SLAM map and estimated trajectory')
    ax.set_xlim(v['x_map'].min() - .65, v['x_map'].max() + .65)
    ax.set_ylim(v['y_map'].min() - .65, v['y_map'].max() + .65)
    ax.legend(loc='lower left', framealpha=.95)
    save(fig, 'slam_map_and_laps',
         'Actual saved SLAM occupancy grid, cropped around the onboard estimated trajectory. '
         'Colors distinguish reconnaissance and the subsequent lap containing planning and SLAM-guided local execution. '
         'Orange markers show recorded reverse maneuvers. The dashed route is a local-planning guide, '
         'not an executed optimized forward race line. No external trajectory ground truth is available. '
         + ('The second lap is incomplete; only one start-line crossing was recorded.' if len(cross) < 2 else ''))

    planning = events.get('planning', {}).get('t_submit', t0) - t0
    guided = events.get('t_map_guided', t0) - t0
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.1), sharex=True, layout='constrained')
    for ax in axes:
        ax.axvspan(cross[0], planning, color='#e69f00', alpha=.1)
        ax.axvspan(planning, guided, color='#9467bd', alpha=.1)
        ax.axvspan(guided, t[-1], color='#009e73', alpha=.08)
        for stamp, color in ((cross[0], '#e69f00'), (planning, '#9467bd'), (guided, '#009e73')):
            ax.axvline(stamp, color=color, lw=.9)
        for recovery in reversals:
            if recovery['start_s'] >= cross[0]:
                ax.axvspan(recovery['start_s'], recovery['end_s'], color='#d55e00', alpha=.15)
        ax.grid(alpha=.2, lw=.5)
    axes[0].plot(t, v['v_cmd'], color='.15', label='Command', lw=1.)
    axes[0].plot(t, v['speed'], color='#0072b2', label='Onboard estimate', lw=.9)
    axes[0].set(ylabel='Speed [m/s]', title='Continuous transition to SLAM-guided local execution')
    speed_legend = axes[0].legend(loc='lower left', ncol=2)
    axes[0].add_artist(speed_legend)
    axes[0].legend(handles=[
        Line2D([], [], color='#e69f00', label='First lap completed'),
        Line2D([], [], color='#9467bd', label='Planning started'),
        Line2D([], [], color='#009e73', label='SLAM guidance active'),
        Patch(facecolor='#d55e00', alpha=.15, label='Reverse maneuver'),
    ], loc='upper right', ncol=2, fontsize=6.5)
    axes[1].plot(t, v['steer_deg'], color='#6f4693', lw=1.)
    axes[1].set(ylabel='Steering [deg]')
    axes[2].plot(t, v['g_left'], color='#0072b2', label='Left-turn gain')
    axes[2].plot(t, v['g_right'], color='#d55e00', label='Right-turn gain', linestyle='--')
    axes[2].set(ylabel='Identified gain [1]', xlabel='Time from reconnaissance start [s]')
    axes[2].legend(loc='best', ncol=2)
    axes[2].set_xlim(max(0., cross[0] - 4.), min(t[-1], guided + 5.))
    save(fig, 'slam_handover_detail',
         'Recorded transition after the first line crossing, with speed, nominal steering command and '
         'identified steering gains. The same learned model is retained through asynchronous planning '
         'and map-guided local execution. Red shading marks the reverse maneuver that started during '
         'planning and continued into SLAM guidance. Estimates are onboard quantities.')

    if len(cross) >= 2:
        durations = [float(cross[0]), float(cross[1] - cross[0])]
        counts = [int(sum(r['start_s'] < cross[0] for r in reversals)),
                  int(sum(cross[0] <= r['start_s'] < cross[1] for r in reversals))]
        labels = ['Lap 1\nReconnaissance', 'Lap 2\nPlanning + SLAM guidance']
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), layout='constrained')
        for ax, values, ylabel in ((axes[0], durations, 'Lap duration [s]'),
                                    (axes[1], counts, 'Reverse maneuvers [count]')):
            bars = ax.bar(labels, values, color=['#0072b2', '#009e73'], width=.55)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(values) * 1.2)
            ax.bar_label(bars, labels=[f'{x:.2f}' if ax is axes[0] else str(x) for x in values], padding=3)
            ax.grid(axis='y', alpha=.2, lw=.5)
            ax.set_axisbelow(True)
        axes[0].set_title('Recorded lap times')
        axes[1].set_title('Recorded recovery maneuvers')
        save(fig, 'single_run_lap_comparison',
             'Descriptive comparison within one real-vehicle trial: elapsed times between onboard '
             'start-line crossings and recorded reverse-maneuver starts. The second lap includes '
             'asynchronous planning followed by SLAM-guided local execution. This is not a controlled '
             'ablation or a statistical performance claim; no external ground truth is implied.')
        run_id = (source / 'run_id.txt').read_text().strip() if (source / 'run_id.txt').exists() else source.name
        metrics = {'run_id': run_id, 'lap_durations_s': durations,
                   'reverse_starts_per_lap': counts, 'first_crossing_s': float(cross[0]),
                   'second_crossing_s': float(cross[1]), 'planning_started_s': planning,
                   'slam_guidance_started_s': guided, 'driver_done_s': float(t[-1]),
                   'reversals': reversals, 'direct_tracking_allowed': events.get('plan', {}).get('direct_tracking_allowed'),
                   'external_ground_truth': False, 'comparison_type': 'descriptive_single_trial'}
        (out / 'paper_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        with (out / 'lap_metrics.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['lap', 'duration_s', 'reverse_maneuver_starts', 'execution'])
            for i in range(2):
                writer.writerow([i + 1, durations[i], counts[i], labels[i].replace('\n', ': ')])
    print(json.dumps({'output_directory': str(out), 'figures': ['slam_map_and_laps', 'slam_handover_detail']
                     + (['single_run_lap_comparison'] if len(cross) >= 2 else [])}, indent=2))


if __name__ == '__main__':
    main()
