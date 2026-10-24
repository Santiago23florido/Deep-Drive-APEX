#!/usr/bin/env python3
"""Inspect saved commands and ESC telemetry without commanding the vehicle.

Driver values are held between CSV updates; busy sensor callbacks can publish
the held command without adding a driver row. Durations below therefore use
zero-order holds and clip intervals to the recorded lap boundaries. Bridge
JSON snapshots lack timestamps: their distributions are sample statistics,
never time fractions or precisely aligned actuator measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def distribution(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return None
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    quantiles = [float(values[min(np.searchsorted(cumulative, p * cumulative[-1]), len(values) - 1)])
                 for p in (.5, .95)]
    return dict(n=len(values), mean=float(np.average(values, weights=weights)),
                p50=quantiles[0], p95=quantiles[1], min=float(values[0]), max=float(values[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.run_directory, args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    events = json.loads((source / 'race_events.json').read_text())
    driver = np.genfromtxt(source / 'driver.csv', delimiter=',', names=True, dtype=None, encoding=None)
    bridge = [json.loads(line) for line in (source / 'bridge.jsonl').read_text().splitlines() if line]
    start = events['t_explore']
    end = events.get('t_done', float(driver['t'][-1]))
    stamps = driver['t']
    next_stamps = np.r_[stamps[1:], max(end, stamps[-1])]
    command, estimate = driver['v_cmd'], driver['speed']
    epsilon = .005  # SpeedMap.cmd_epsilon_mps; driver CSV rounded to .001 m/s.
    intervals = [(f'lap_{i}', a, b) for i, (a, b) in enumerate(zip(
        [start] + [x['t'] for x in events['crossings'][:-1]],
        [x['t'] for x in events['crossings']]), 1)]
    intervals.append(('entire_execution', start, end))
    report = {'run_id': (source / 'run_id.txt').read_text().strip(),
              'definitions': {
                  'driver_durations': 'Zero-order hold of rounded driver CSV rows, clipped to lap/execution boundaries; includes held commands during core-busy callbacks.',
                  'forward_reverse_threshold_mps': epsilon,
                  'near_zero_band_mps': .02,
                  'bridge_distributions': 'All saved bridge snapshots, including pre/post-motion samples; no timestamps, not time-weighted. Positive/negative subsets selected separately.',
                  'esc_percent': 'Normalized signed ESC command scale, not duty-cycle percent or measured electrical/mechanical power.',
                  'motor_pulse': 'Snapshot of software PWM readback; reverse handshake/braking/neutral phases may differ from requested ESC command.',
                  'speed_estimate': 'Onboard odometry speed, not external motion truth.'},
              'intervals': [], 'bridge': {}}
    categories = {'forward_command_s': command >= epsilon,
                  'reverse_command_s': command <= -epsilon,
                  'zero_or_deadband_command_s': abs(command) < epsilon}
    for label, a, b in intervals:
        weights = np.maximum(0., np.minimum(next_stamps, b) - np.maximum(stamps, a))
        assert abs(weights.sum() - (b - a)) < .001, 'Driver CSV does not cover requested interval'
        item = {'interval': label, 'duration_s': b - a,
                **{name: float(weights[mask].sum()) for name, mask in categories.items()},
                'near_zero_command_abs_le_0_02_s': float(weights[abs(command) <= .02].sum()),
                'reverse_recovery_mode_s': float(weights[driver['recover'] == 2].sum()),
                'forward_command_mps': distribution(command, weights * (command >= epsilon)),
                'reverse_command_magnitude_mps': distribution(-command, weights * (command <= -epsilon)),
                'onboard_speed_magnitude_during_forward_mps': distribution(abs(estimate), weights * (command >= epsilon)),
                'forward_command_with_estimated_speed_below_0_03_s': float(weights[(command >= epsilon) & (abs(estimate) < .03)].sum()),
                'driver_update_interval_s': distribution(next_stamps - stamps, weights),
                'phase_duration_s': {phase: float(weights[driver['phase'] == phase].sum()) for phase in np.unique(driver['phase']) if weights[driver['phase'] == phase].sum() > 0}}
        report['intervals'].append(item)
    mapping = bridge[0]['speed_map']
    report['bridge']['speed_map'] = mapping
    applied = np.array([r['applied_speed_pct'] for r in bridge])
    pulses = np.array([r['motor_pwm_state']['duty_cycle_ns'] / 1000. for r in bridge])
    report['bridge']['snapshot_count'] = len(bridge)
    report['bridge']['forward_requested_esc_pct'] = distribution(applied[applied > 0])
    report['bridge']['reverse_requested_esc_magnitude_pct'] = distribution(-applied[applied < 0])
    report['bridge']['forward_pwm_pulse_us'] = distribution(pulses[applied > 0])
    report['bridge']['reverse_request_pwm_pulse_us'] = distribution(pulses[applied < 0])
    report['bridge']['forward_request_at_ceiling_snapshots'] = int((applied >= mapping['max_pct'] - 1e-6).sum())
    report['bridge']['forward_request_below_legacy_minimum_fraction'] = float((applied[(applied > 0)] < bridge[0]['min_effective_speed_pct']).mean())
    report['bridge']['minimum_parameter_note'] = 'min_effective_speed_pct is a legacy-profile parameter; affine profile does not impose this minimum. Its presence in status does not establish the current physical motor threshold.'
    report['bridge']['reverse_phase_snapshot_counts'] = {p: sum(r['motor_state']['reverse_phase'] == p for r in bridge) for p in sorted({r['motor_state']['reverse_phase'] for r in bridge})}
    report['adaptation_final'] = {k: events['adaptation'][k] for k in (
        'v_gain', 'v_offset', 'v_gain_estimated', 'v_offset_estimated',
        'longitudinal_applied', 'conf_long', 'dead_mps', 'dead_rev_mps', 'confidence', 'n')
        if k in events['adaptation']}
    report['interpretation'] = [
        'Recorded maneuvers and small forward commands contribute to elapsed time; increasing a ceiling alone does not increase commands below that ceiling.',
        'Affine mapping permits commands below the former legacy minimum. Physical breakaway threshold may vary with battery/load, but is not measured here.',
        'Longitudinal adaptation sample count is reported; do not interpret a single accepted update as a well-identified motor model.',
        'Use timing/resource metrics to inspect computation. Pi supply flags do not measure motor voltage/current, and command telemetry cannot prove battery health.'
    ]
    (out / 'motion_effort_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'lap_command_budget.csv').open('w', newline='') as stream:
        fields = ['interval', 'duration_s', *categories, 'near_zero_command_abs_le_0_02_s', 'reverse_recovery_mode_s']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in report['intervals'])
    print(json.dumps({'intervals': report['intervals'], 'bridge': report['bridge'], 'adaptation_final': report['adaptation_final']}, indent=2))


if __name__ == '__main__':
    main()
