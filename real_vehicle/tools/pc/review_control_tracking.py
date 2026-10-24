#!/usr/bin/env python3
"""Review recorded reference tracking without ground truth or fitted time shifts.

Input is a diagnostics directory containing driver.csv, race_events.json and
bridge_before.json. A nominal legacy reference is never relabeled as a calibrated
physical setpoint. All comparisons use the original recorded values.
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


def review(source: Path, output: Path) -> dict:
    events = json.loads((source / 'race_events.json').read_text())
    with (source / 'driver.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not rows or 'v_target' not in rows[0]:
        raise ValueError('Actual control reference was not logged; it cannot be reconstructed honestly.')
    values = {k: np.array([float(r[k]) for r in rows]) for k in
              ('t', 'v_target', 'speed', 'v_cmd', 'recover', 'kap_des', 'kap_meas')}
    t, ref, speed, command, recovery = (values[k] for k in ('t', 'v_target', 'speed', 'v_cmd', 'recover'))
    dt = np.r_[np.diff(t), 0.]
    if (dt < 0.).any():
        raise ValueError('Control timestamps are not monotonic')
    stable = np.zeros(len(t), bool)
    for i in range(len(t)):
        j = np.searchsorted(t, t[i]-.8)
        stable[i] = (t[i]-t[j] >= .7 and np.all(ref[j:i+1] > .005)
                     and np.ptp(ref[j:i+1]) < .025 and np.all(recovery[j:i+1] == 0))
    bridge = json.loads((source / 'bridge_before.json').read_text())
    mapping = bridge['speed_map']
    legacy = abs(mapping['pct_per_mps']-33.) < 1e-6
    cfg = events['config']['adaptive']
    scale = cfg.get('speed_feedback_forward_scale', 1.)
    gain = cfg.get('kp_speed_forward', 0.) or cfg['kp_speed']
    authority = cfg['speed_authority'] * np.maximum(abs(ref), .3*scale)
    saturated = (abs(scale*gain*(ref-speed)) >= authority-.001).astype(float)
    report = dict(run_id=(source / 'run_id.txt').read_text().strip(), ground_truth_available=False,
                  reference_convention='Legacy nominal reference; physical speed mapping is uncalibrated' if legacy
                                       else 'Recorded reference; inspect the declared bridge calibration',
                  bridge_map=mapping, selection=dict(stable_window_s=.8, minimum_window_coverage_s=.7,
                  maximum_reference_range=.025, minimum_moving_speed_mps=.05), laps=[],
                  adaptation=events['adaptation'], direct_tracking_allowed=events.get('plan', {}).get('direct_tracking_allowed'),
                  counters=events.get('counters'), scan_age_stops=events.get('scan_age_stops'),
                  cmd_gap_max_ms=events.get('cmd_gap_max_ms'),
                  source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
                                 ('driver.csv', 'race_events.json', 'bridge_before.json')},
                  interpretation='Numerical reference/onboard residuals, not absolute physical speed accuracy. '
                                 'No measured signal is scaled or time-shifted to match the reference.')
    bounds = [events['t_explore']] + [r['t'] for r in events['crossings']]
    report['recognized_laps'] = len(events['crossings'])
    end = events.get('t_done', float(t[-1]))
    if len(events['crossings']) < events['laps']:
        report['uncompleted_lap_duration_s'] = max(0., end-bounds[-1])
    for lap, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        w = np.maximum(0., np.minimum(t+dt, hi)-np.maximum(t, lo))
        def mean(value, mask):
            keep = (w > 0.) & mask
            return float(np.average(value[keep], weights=w[keep])) if keep.any() else None
        forward, reverse = command > .005, command < -.005
        item = dict(lap=lap, duration_s=hi-lo,
                    reverse_starts=sum(lo <= r.get('t', -1) < hi and r['kind'] == 2
                                       for r in events.get('recovery', [])),
                    forward_command_s=float(w[forward].sum()), reverse_command_s=float(w[reverse].sum()),
                    zero_command_s=float(w[~(forward | reverse)].sum()))
        for label, mask in [('forward_requested', (ref > .005) & forward),
                            ('forward_moving', (ref > .005) & forward & (speed > .05)),
                            ('settled_forward', stable & forward & (speed > .05))]:
            square = mean((speed-ref)**2, mask)
            item[label] = dict(duration_s=float(w[mask].sum()), reference_mean=mean(ref, mask),
                               onboard_speed_mean_mps=mean(speed, mask), numeric_bias=mean(speed-ref, mask),
                               numeric_mae=mean(abs(speed-ref), mask),
                               numeric_rmse=float(np.sqrt(square)) if square is not None else None,
                               feedback_saturated_fraction=mean(saturated, mask))
        kmask = forward & (ref > .005) & (speed > .05) & np.isfinite(values['kap_meas'])
        item['raw_curvature_diagnostic'] = dict(valid_s=float(w[kmask].sum()),
            mae_inv_m=mean(abs(values['kap_des']-values['kap_meas']), kmask),
            scope='Raw contemporaneous diagnostic, not a steering acceptance metric. Includes delay, '
                  'transients and curvature observations that the identifier may reject as too slow.')
        report['laps'].append(item)
    output.mkdir(parents=True, exist_ok=True)
    (output/'control_tracking.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.2), layout='constrained')
    x = t-events['t_explore']
    axes[0].plot(x, ref, color='black', lw=.9,
                 label='Logged nominal reference' if legacy else 'Reference (reverse: legacy gauge)')
    axes[0].plot(x, speed, color='#0072b2', lw=.7, label='Onboard speed estimate')
    axes[0].set(ylabel='Speed / nominal reference' if legacy else 'Speed [m/s] / reverse reference',
                title='Recorded control signals (legacy reference gauge)' if legacy else 'Recorded control signals')
    axes[0].legend(fontsize=7, ncol=2)
    moving = (ref > .005) & (command > .005) & (speed > .05)
    residual = np.where(moving, speed-ref, np.nan)
    axes[1].plot(x, residual, lw=.7, color='#d55e00', label='Forward-motion numerical residual')
    axes[1].scatter(x[stable & moving], residual[stable & moving], s=3, color='#009e73', label='Settled forward samples')
    axes[1].axhline(0., color='black', lw=.5)
    axes[1].set(ylabel='Estimate minus reference' if legacy else 'Forward residual [m/s]')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel('Time since reconnaissance start [s]')
    for ax in axes:
        for crossing in events['crossings']:
            ax.axvline(crossing['t']-events['t_explore'], color='.5', ls='--', lw=.7)
        ax.grid(alpha=.2)
        ax.set_xlim(0., max(end, float(t[-1]))-bounds[0]+.1)
    for extension in ('png', 'pdf'):
        fig.savefig(output/('control_tracking.'+extension), dpi=400)
    plt.close(fig)
    convention = ('The original reference convention is retained; numerical residuals do not establish '
                  'calibrated physical tracking or odometry accuracy. ' if legacy else
                  'Forward references use the declared onboard-derived bridge calibration; reverse '
                  'retains its legacy command gauge. Forward residuals compare the actual reference '
                  'with the unmodified onboard estimate, not independent motion ground truth. ')
    (output/'control_tracking_caption.txt').write_text(
        'Recorded internal references and onboard estimates. Dashed lines mark recognized laps. '
        'Green points meet the stated settled-motion selection. ' + convention +
        'Curvature comparison includes measurement/actuation delay and transients.\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.source, args.output)['laps'], indent=2))
