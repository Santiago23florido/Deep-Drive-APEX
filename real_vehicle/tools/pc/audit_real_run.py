#!/usr/bin/env python3
"""Export real-run timing and odometry consistency evidence without ground truth.

No model fitting, trajectory alignment to SLAM, or accuracy claims are made.
Geometric residuals reuse the estimator's deskewed LiDAR and are correlated
consistency diagnostics, not an independent reference trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n': 0}
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.median(x)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                max=float(x.max()), min=float(x.min()))


def json_safe(value):
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def table(path):
    return np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8')


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def scan_consistency(path, t0, end):
    """Coverage-penalized point-to-line diagnostics for consecutive live scans.

    A 7-neighbor PCA supplies local surface normals, with explicit planarity,
    neighborhood-size and nearest-neighbor gates. A missing correspondence is
    penalized at 0.20 m; this prevents lower overlap from improving the score.
    The identity-pose comparison uses exactly the same points and thresholds.
    """
    results, previous = [], None
    with path.open() as handle:
        reader = csv.reader(handle)
        names = next(reader)
        for row in reader:
            stamp = float(row[0]) * 1e-9
            # Remaining metadata contains phase strings, so parse fields explicitly.
            get = lambda key: float(row[names.index(key)])
            ranges = np.asarray(row[len(names):], dtype=float)
            angles = get('angle_min') + np.arange(len(ranges)) * get('angle_inc')
            valid = np.isfinite(ranges) & (ranges >= .15) & (ranges <= 4.)
            polar = ranges[valid, None] * np.column_stack((np.cos(angles[valid]), np.sin(angles[valid])))
            cloud = polar @ rotation(get('laser_yaw')).T + [get('laser_x'), get('laser_y')]
            # Exclude chassis/wheel returns; same measured 0.46 x 0.32 m footprint.
            cloud = cloud[(abs(cloud[:, 0]) > .26) | (abs(cloud[:, 1]) > .19)]
            pose = np.array([get('x_stamp'), get('y_stamp'), get('yaw_stamp')])
            if previous is not None:
                ps, pp, pc = previous
                dt = stamp - ps
                if .06 <= dt <= .2 and len(cloud) >= 20 and len(pc) >= 20 and t0 <= stamp <= end:
                    tree = cKDTree(pc)
                    distance, neighbors = tree.query(pc, k=7)
                    patches = pc[neighbors]
                    centers = patches.mean(axis=1)
                    delta = patches - centers[:, None, :]
                    cov = np.einsum('nki,nkj->nij', delta, delta) / 7
                    eig, vectors = np.linalg.eigh(cov)
                    planar = (eig[:, 0] / np.maximum(eig[:, 1], 1e-12) < .1) & (eig[:, 1] > 1e-5) & (distance[:, -1] < .25)
                    normal = vectors[:, :, 0]

                    def residual(q):
                        d, idx = tree.query(q)
                        ok = (d < .2) & planar[idx]
                        r = np.full(len(q), .2)
                        r[ok] = np.minimum(.2, abs(np.sum((q[ok] - centers[idx[ok]]) * normal[idx[ok]], axis=1)))
                        return float(r.mean()), float(ok.mean())

                    moved = ((cloud @ rotation(pose[2]).T + pose[:2]) - pp[:2]) @ rotation(pp[2])
                    after, overlap = residual(moved)
                    before, identity_overlap = residual(cloud)
                    dh = np.angle(np.exp(1j * (pose[2] - pp[2])))
                    results.append([stamp - t0, dt, np.linalg.norm(pose[:2] - pp[:2]), np.degrees(dh),
                                    after, before, overlap, identity_overlap])
            if previous is None or stamp > previous[0]:
                previous = stamp, pose, cloud
    return np.asarray(results, dtype=float).reshape(-1, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.run_directory, args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads((source / 'race_events.json').read_text())
    arm_result = (json.loads((source / 'arm_result.json').read_text())
                  if (source / 'arm_result.json').exists() else {})
    stop_reason = arm_result.get('reason', 'external_stop')
    if (source / 'operator_feedback.json').exists():
        feedback = json.loads((source / 'operator_feedback.json').read_text())
        if feedback.get('stop_requested_by_operator'):
            stop_reason = 'operator_stop'
    estimator, driver = table(source / 'estimator.csv'), table(source / 'driver.csv')
    t0 = events['t_explore']
    # Disarmed ticks return before the driver CSV is written. Its final row
    # therefore bounds armed control, including legitimate zero commands.
    # A slow final callback stamps its row at entry; retain the gap report at
    # exit rather than censoring the failure from the timing window.
    final_control_stamp = float(driver['t'][-1])
    gaps = []
    if (source / 'container.log').exists():
        for line in (source / 'container.log').read_text().splitlines():
            match = re.search(r'\[(\d+\.\d+)\].*command delivery gap: (\{.*\})', line)
            if match:
                gaps.append({'stamp_s': float(match[1]), **json.loads(match[2])})
    end = events.get('t_done', max([final_control_stamp] + [g['stamp_s'] for g in gaps
                                   if final_control_stamp <= g['stamp_s'] <= final_control_stamp + .7]))
    et = estimator['stamp_ns'] * 1e-9
    active = (et >= t0) & (et <= end)
    est = estimator[active]
    static = estimator[(et >= t0 - 10.) & (et <= t0 - .2)]
    dactive = driver[(driver['t'] >= t0) & (driver['t'] <= end)]
    report = {'run_id': (source / 'run_id.txt').read_text().strip(),
              'ground_truth_available': False, 'model_weights_updated': False,
              'scope': 'Deployed hybrid odometry pipeline, including ICP and stationary hold; not the neural model in isolation.',
              'active_estimator_samples': len(est), 'analysis_window_duration_s': end - t0,
              'stop_reason': stop_reason,
              'window_end': 'driver_done' if 't_done' in events else 'last_recorded_armed_control_or_its_command_gap_report',
              'command_gap_diagnostics': gaps,
              'estimator': {k: stats(est[k]) for k in ('compute_ms', 'latency_ms', 'dt', 'imu_samples')},
              'control': {k: stats(dactive[k]) for k in ('pose_age_s', 'react_ms')},
              'timing_definitions': {
                  'compute_ms': 'Per-estimate processing: feature preparation, hybrid ICP/network or hold, deskew and state integration; excludes input waiting and output publication.',
                  'latency_ms': 'Scan header stamp to estimator output-handler entry; includes sweep acquisition, waiting and processing; not pure inference or measured actuator latency.',
                  'pose_age_s': 'Age of the propagated pose read by the driver; not the age of the last independent LiDAR correction.',
                  'receipt_age_ms': 'Raw message header stamp to passive recorder callback entry; includes acquisition/driver transport and recorder scheduling.',
                  'react_ms': 'Sensor-callback reactive planning computation, logged at control rate; scan-pair diagnostics below use one record per scan.'}}
    report['estimator']['compute_over_100ms_count'] = int((est['compute_ms'] > 100).sum())
    report['estimator']['update_intervals_above_150ms_count'] = int((est['dt'] > .15).sum())
    report['estimator']['stationary_hold_fraction'] = float(est['standstill_hold'].mean())
    xy = np.column_stack((static['x'], static['y']))
    report['stationary_10s_before_motion'] = {
        'samples': len(static), 'observed_duration_s': float(np.ptp(static['stamp_ns']) * 1e-9),
        'net_displacement_m': float(np.linalg.norm(xy[-1] - xy[0])),
        'max_displacement_from_first_m': float(np.linalg.norm(xy - xy[0], axis=1).max()),
        'accumulated_position_variation_m': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        'yaw_span_deg': float(np.degrees(np.ptp(np.unwrap(static['yaw'])))),
        'stationary_hold_fraction': float(static['standstill_hold'].mean()),
        'note': 'Car confirmed stationary by operator before launch. Values describe the full deployed pipeline including stationary hold.'}
    crossings = events.get('crossings', [])
    bounds = [t0] + [c['t'] for c in crossings]
    recovery = events.get('recovery', [])
    report['laps'] = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        starts = [r for r in recovery if 't' in r and a <= r['t'] < b and r['kind'] == 2]
        report['laps'].append({'lap': i, 'duration_s': b - a, 'reverse_starts': len(starts)})
    if len(crossings) < events['laps']:
        report['uncompleted_lap'] = {
            'lap': len(crossings) + 1, 'observed_duration_s': end - bounds[-1],
            'reverse_starts': sum('t' in r and bounds[-1] <= r['t'] <= end and r['kind'] == 2 for r in recovery)}
    report['guidance'] = {k: events[k] - t0 for k in ('t_route_guidance', 't_map_guided') if k in events}
    report['plan'] = events.get('plan', {})
    report['recoveries'] = [{**r, 't_relative_s': r.get('t', r.get('t_end')) - t0} for r in recovery]

    receipt = list(csv.DictReader((source / 'sensors/sensor_receipt_timing.csv').open()))
    report['raw_sensors'] = {}
    sensor_trace = []
    for sensor, nominal in (('imu', 100.), ('lidar', 10.)):
        rows = [r for r in receipt if r['sensor'] == sensor]
        stamp = np.array([int(r['stamp_ns']) for r in rows], dtype=np.int64)
        rec = np.array([int(r['receipt_ros_ns']) for r in rows], dtype=np.int64)
        age = (rec - stamp) * 1e-6
        sel = (stamp * 1e-9 >= t0) & (stamp * 1e-9 <= end)
        periods = np.diff(stamp[sel]) * 1e-9
        report['raw_sensors'][sensor] = {
            'recorded_messages': len(rows), 'active_messages': int(sel.sum()),
            'active_header_rate_hz': float((sel.sum() - 1) / (np.ptp(stamp[sel]) * 1e-9)),
            'period_ms': stats(periods * 1e3), 'receipt_age_ms': stats(age[sel]),
            'nonpositive_period_count': int((periods <= 0).sum()),
            'gaps_above_1_5_nominal_periods': int((periods > 1.5 / nominal).sum()),
            'receipt_older_than_50ms_count': int((age[sel] > 50).sum()),
            'note': 'Gap counts are timing anomalies, not a measured transport packet-loss count.'}
        sensor_trace.extend(zip([sensor] * int(sel.sum()), stamp[sel] * 1e-9 - t0, age[sel]))
    power = list(csv.DictReader((source / 'power.csv').open()))
    flags = [int(r['throttled'], 16) for r in power]
    report['power'] = {'recorded_samples': len(power), 'live_fault_samples': sum(bool(f & 0xffff) for f in flags),
                       'temperature_c': stats([float(r['temp_c']) for r in power]),
                       'cpu_frequency_khz': stats([float(r['freq_khz']) for r in power]),
                       'note': 'Entire startup-and-execution log; not a motor battery voltage measurement.'}
    preflight = json.loads((source / 'check.json').read_text())
    report['preflight_cpu'] = [c for c in preflight['checks'] if c['name'].startswith('cpu ')]
    statuses = [json.loads(s) for s in (source / 'status.jsonl').read_text().splitlines() if s]
    bridge = [json.loads(s) for s in (source / 'bridge.jsonl').read_text().splitlines() if s]
    report['runtime'] = {'final_counters': statuses[-1].get('counters'),
                         'final_core_holds': statuses[-1].get('core_hold_ms'),
                         'driver_command_max_gap_ms': max(s.get('cmd_gap_max_ms', 0) for s in statuses),
                         'bridge_command_max_gap_ms': max(s.get('cmd_gap_ms_max', 0) for s in bridge),
                         'bridge_timeout_samples': sum(bool(s.get('timed_out')) for s in bridge)}

    resources = []
    if (source / 'host_resources.csv').exists():
        grouped = {}
        for row in csv.DictReader((source / 'host_resources.csv').open()):
            stamp = float(row['unix_s'])
            if t0 <= stamp <= end:
                sample = grouped.setdefault(stamp, {'cpu': {}, 'rss': {}, 'host_cpu': float(row['host_cpu_percent_all_cores']),
                                                    'available_mb': float(row['host_mem_available_mb'])})
                label = row['label']
                sample['cpu'][label] = sample['cpu'].get(label, 0.) + float(row['cpu_percent_one_core'])
                sample['rss'][label] = sample['rss'].get(label, 0.) + float(row['rss_mb'])
        resources = sorted(grouped.items())
        labels = sorted({label for _, sample in resources for label in sample['cpu']})
        report['execution_resources'] = {
            'samples': len(resources), 'sampling_nominal_hz': 1.,
            'cpu_percent_one_core': {label: stats([s['cpu'].get(label, 0.) for _, s in resources]) for label in labels},
            'rss_mb': {label: stats([s['rss'].get(label, 0.) for _, s in resources]) for label in labels},
            'container_cpu_percent_one_core': stats([sum(s['cpu'].values()) for _, s in resources]),
            'host_cpu_percent_all_cores': stats([s['host_cpu'] for _, s in resources]),
            'host_mem_available_mb': stats([s['available_mb'] for _, s in resources]),
            'notes': 'Passive host /proc samples for processes in this container cgroup. 100% CPU equals one core. Other includes launch/planner worker processes. Per-role RSS sums may double-count shared pages; they are not unique physical memory.'}

    consistency = scan_consistency(source / 'driver_inputs.csv', t0, end)
    report['scan_geometry_consistency'] = {
        'scan_pairs': len(consistency), 'odom_compensated_capped_mean_residual_m': stats(consistency[:, 4]),
        'identity_pose_capped_mean_residual_m': stats(consistency[:, 5]),
        'usable_correspondence_fraction': stats(consistency[:, 6]),
        'method': 'Consecutive deskewed scans, 0.15..4.0 m ranges, measured body mask; 7-neighbor PCA normals, planar eigenvalue ratio <0.1, patch radius <0.25 m; unmatched/nonplanar points penalized at 0.20 m. Each point contributes, including failures. Recorded stamp poses transform current scan into previous base frame.',
        'limitation': 'Correlated, scene-dependent self-consistency only; LiDAR deskew and poses both depend on the deployed estimator. Neither residuals nor identity comparison are absolute odometry error or a competitive baseline.'}
    moving = consistency[(consistency[:, 2] > .01) | (abs(consistency[:, 3]) > 1.)]
    report['scan_geometry_consistency']['moving_pairs'] = len(moving)
    report['scan_geometry_consistency']['moving_odom_residual_m'] = stats(moving[:, 4])
    report['scan_geometry_consistency']['moving_identity_residual_m'] = stats(moving[:, 5])
    with (out / 'scan_consistency.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['t_s', 'dt_s', 'estimated_translation_m', 'estimated_yaw_deg', 'odom_residual_m',
                         'identity_residual_m', 'usable_fraction', 'identity_usable_fraction'])
        writer.writerows(consistency)
    with (out / 'sensor_timing.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['sensor', 't_s', 'receipt_age_ms']); writer.writerows(sensor_trace)
    (out / 'real_run_metrics.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 4.7), layout='constrained')
    t = et[active] - t0
    ax[0, 0].plot(t, est['compute_ms'], lw=.7)
    ax[0, 0].axhline(100, c='r', ls='--', lw=.7, label='Nominal scan period')
    ax[0, 0].set(title='(a) Hybrid odometry processing', ylabel='Processing time [ms]')
    ax[0, 0].legend()
    for sensor, color in (('imu', '#009e73'), ('lidar', '#e69f00')):
        trace = np.array([[r[1], r[2]] for r in sensor_trace if r[0] == sensor])
        ax[0, 1].plot(trace[:, 0], trace[:, 1], lw=.5, c=color, alpha=.65, label=sensor.upper() + ' receipt')
    ax[0, 1].plot(t, est['latency_ms'], lw=.7, c='#0072b2', label='Estimate output')
    ax[0, 1].set(title='(b) Ages from sensor header stamps', ylabel='Age [ms]')
    ax[0, 1].legend()
    ax[1, 0].plot(consistency[:, 0], consistency[:, 4] * 100, lw=.7, label='Recorded odometry')
    ax[1, 0].plot(consistency[:, 0], consistency[:, 5] * 100, lw=.5, alpha=.65, label='Identity pose diagnostic')
    ax[1, 0].set(title='(c) Scan consistency; no ground truth', ylabel='Capped geometric residual [cm]')
    ax[1, 0].legend()
    ax[1, 1].plot((static['stamp_ns'] - static['stamp_ns'][0]) * 1e-9,
                   np.linalg.norm(xy - xy[0], axis=1) * 1000, lw=.8)
    ax[1, 1].set(title='(d) Stationary pipeline stability', ylabel='Displacement from first pose [mm]',
                  xlabel='Stationary interval time [s]')
    for item in ax.ravel(): item.grid(alpha=.2)
    for item in (ax[0, 0], ax[0, 1], ax[1, 0]): item.set_xlabel('Time since exploration start [s]')
    for suffix in ('pdf', 'svg', 'png'): fig.savefig(out / ('real_odometry_and_timing.' + suffix), dpi=600)
    plt.close(fig)
    if resources:
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.7), sharex=True, layout='constrained')
        rt = np.array([stamp - t0 for stamp, _ in resources])
        for label in labels:
            axes[0].plot(rt, [s['cpu'].get(label, 0.) for _, s in resources], lw=.7, label=label.capitalize())
            axes[1].plot(rt, [s['rss'].get(label, 0.) for _, s in resources], lw=.7, label=label.capitalize())
        axes[0].plot(rt, [sum(s['cpu'].values()) for _, s in resources], c='k', lw=1., label='Container total')
        axes[0].set(ylabel='CPU [% of one core]', title='(a) Raspberry Pi process CPU usage')
        axes[0].legend(ncol=4)
        axes[1].set(xlabel='Time since exploration start [s]', ylabel='Sum of process RSS [MiB]',
                    title='(b) Resident memory; shared pages may be counted more than once')
        for item in axes: item.grid(alpha=.2)
        for suffix in ('pdf', 'svg', 'png'): fig.savefig(out / ('raspberry_resources.' + suffix), dpi=600)
        plt.close(fig)
    print(json.dumps({k: report[k] for k in ('run_id', 'laps', 'stationary_10s_before_motion', 'estimator', 'raw_sensors')}, indent=2))


if __name__ == '__main__':
    main()
