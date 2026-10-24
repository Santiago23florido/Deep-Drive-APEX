"""Passive, finite host /proc capture for one named Docker container."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--container', default='apex_pipeline')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--duration', type=float, default=430.)
args = parser.parse_args()
deadline = time.monotonic() + 90.
while True:
    probe = subprocess.run(['docker', 'inspect', '--format', '{{json .State}} {{.Id}}', args.container],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        state, container = probe.stdout.strip().rsplit(' ', 1)
        if json.loads(state)['Running']:
            break
    if time.monotonic() >= deadline:
        raise SystemExit('Container did not become available; no resource data collected')
    time.sleep(1.)

args.output.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(__file__, args.output.parent / 'resource_probe_source.py')
labels = {'learned_odometry': 'odometry', 'race_driver': 'driver', 'cmd_vel_to_apex': 'bridge',
          'rplidar_publisher': 'lidar', 'nano_accel_serial': 'imu', 'slam_toolbox': 'slam',
          'record_real2sim_sensors': 'recorder'}
hz, page = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGE_SIZE')
previous, last, start, seen = {}, time.monotonic(), time.monotonic(), False
with args.output.open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['unix_s', 'monotonic_s', 'pid', 'label', 'cpu_percent_one_core', 'rss_mb',
                     'host_cpu_percent_all_cores', 'host_mem_available_mb'])
    while time.monotonic() - start < args.duration:
        now, wall = time.monotonic(), time.time()
        cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        total, idle = sum(cpu), cpu[3] + cpu[4]
        host = (0. if 'host' not in previous else
                100 * (1 - (idle - previous['host'][1]) / max(1, total - previous['host'][0])))
        previous['host'] = total, idle
        mem = {line.split()[0].rstrip(':'): int(line.split()[1])
               for line in Path('/proc/meminfo').read_text().splitlines()}
        records = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if container not in (entry / 'cgroup').read_text():
                    continue
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                ticks, key = int(fields[11]) + int(fields[12]), (entry.name, fields[19])
                command = (entry / 'cmdline').read_bytes().replace(bytes([0]), b' ').decode(errors='replace')
                role = next((label for fragment, label in labels.items() if fragment in command), 'other')
                used = (0. if key not in previous else
                        100 * (ticks - previous[key]) / (hz * max(now - last, 1e-6)))
                previous[key] = ticks
                records.append([wall, now, entry.name, role, used, int(fields[21]) * page / 2**20,
                                host, mem['MemAvailable'] / 1024])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if records:
            seen = True
        elif seen:
            break
        writer.writerows(records)
        handle.flush()
        last = now
        time.sleep(max(0., 1 - (time.monotonic() - now)))
print('Resource capture closed: ' + str(args.output), flush=True)
