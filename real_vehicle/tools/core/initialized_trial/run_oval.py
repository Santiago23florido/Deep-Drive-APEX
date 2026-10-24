"""Two laps with a recorded forward bridge calibration and unchanged control law."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess as sp
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--execute', action='store_true', help='activate the bridge and perform the agreed trial')
parser.add_argument('--diagnose-only', action='store_true', help='perform the same preflight, never publish arm=true')
args = parser.parse_args()
if args.execute and args.diagnose_only:
    parser.error('choose either --execute or --diagnose-only')
if not args.execute and not args.diagnose_only:
    parser.exit(message='Prepared: full driver, two continuous laps, at most 40 m estimated / 300 s, target 0.45--0.55 m/s forward; unchanged forward/reverse PWM ceilings.\n')

root = Path.home() / 'AiAtonomousRc/APEX'
os.chdir(root)
sys.path.insert(0, str(root / 'tools/core'))
from power_stability import wait_for_stable_power
from forward_profiles import forward_profile
profile = forward_profile(os.environ.get('APEX_FORWARD_PROFILE', 'initialized'))
out = root / 'data/diagnostics/20261022_open_segment' / ('oval_' + time.strftime('%Y%m%d_%H%M%S'))
out.mkdir()
shutil.copyfile(root / "tools/core/arm_until_done.py", out / "arm_until_done.py")
shutil.copyfile(Path(__file__).with_name('save_final_map.py'), out / 'save_final_map.py')
# Keep the exact sources/config with each trial so a working run is reproducible.
snapshot = out / 'source_snapshot'
sources = sorted((root / 'ros2_ws/src/apex_race/apex_race').rglob('*.py'))
sources += [root / profile['race_yaml'], root / profile['vehicle_yaml'],
            root / 'tools/core/forward_profiles.py',
            root / 'tools/core/initialized_trial/run_oval.py']
manifest = {}
for source in sources:
    relative = source.relative_to(root)
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
(out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(out / 'forward_profile.json').write_text(json.dumps(profile, indent=2) + '\n')
env = dict(os.environ,
           APEX_RACE_PARAMS_FILE='/work/' + profile['race_yaml'],
           APEX_RACE_VEHICLE_YAML='/work/' + profile['vehicle_yaml'],
           APEX_BRIDGE_ESC_ZERO_PCT=str(profile['esc_zero_pct']),
           APEX_BRIDGE_SPEED_PCT_PER_MPS=str(profile['pct_per_mps']),
           APEX_BRIDGE_MAX_SPEED_PCT='28.2',
           APEX_BRIDGE_MAX_STEERING_LEFT_DEG='22.5',
           APEX_BRIDGE_MAX_STEERING_RIGHT_DEG='12.96',
           APEX_BRIDGE_MAX_REVERSE_SPEED_PCT='31.2',
           APEX_RACE_LAPS='2',
           APEX_ODOM_WARMUP_S='25')

def ros(command, timeout=12):
    return sp.check_output(['docker', 'exec', 'apex_pipeline', 'bash', '-lc',
                            'source /opt/ros/jazzy/setup.bash && ' + command],
                           text=True, stderr=sp.STDOUT, timeout=timeout)

def check_power():
    """Require clean live power flags; historical bits alone are permitted."""
    samples = []
    for index in range(3):
        flags = sp.check_output(['vcgencmd', 'get_throttled'], text=True, timeout=3).strip()
        temperature = sp.check_output(['vcgencmd', 'measure_temp'], text=True, timeout=3).strip()
        bits = int(flags.split('=', 1)[1], 16)
        temp_c = float(temperature.split('=', 1)[1].split("'")[0])
        samples.append({'throttled': flags, 'temp_c': temp_c})
        (out / 'power_before.json').write_text(json.dumps(samples, indent=2) + '\n')
        assert bits & 0xffff == 0, 'live Pi power/throttle fault before arm: ' + flags
        assert temp_c < 75.0, 'Pi temperature >=75 C before arm'
        if index < 2:
            time.sleep(1.0)

run = None
capture = None
capture_log = None
resource = None
resource_log = None
try:
    with (out / 'startup.log').open('w') as f:
        sp.run(['./tools/core/apex_race_up.sh', 'armed'], env=env, stdout=f,
               stderr=sp.STDOUT, timeout=130, check=True)
    run = (root / 'ros2_ws/apex_race/.last_run').read_text().strip()
    (out / 'run_id.txt').write_text(run + '\n')
    state = json.loads(sp.check_output(['docker', 'inspect', 'apex_pipeline']))[0]
    current = dict(s.split('=', 1) for s in state['Config']['Env'] if '=' in s)
    assert current['APEX_ENABLE_CMDVEL_ACTUATION_BRIDGE'] == '1'
    assert current['APEX_RACE_REQUIRE_ARM'].lower() == 'true'
    assert current['APEX_BRIDGE_MAX_SPEED_PCT'] == '28.2'
    assert current['APEX_BRIDGE_MAX_STEERING_DEG'] == '12.5'
    assert current['APEX_BRIDGE_MAX_STEERING_LEFT_DEG'] == '22.5'
    assert current['APEX_BRIDGE_MAX_STEERING_RIGHT_DEG'] == '12.96'
    assert current['APEX_BRIDGE_MAX_REVERSE_SPEED_PCT'] == '31.2'
    assert current['APEX_RACE_LAPS'] == '2'
    assert float(current['APEX_BRIDGE_ESC_ZERO_PCT']) == profile['esc_zero_pct']
    assert float(current['APEX_BRIDGE_SPEED_PCT_PER_MPS']) == profile['pct_per_mps']
    initial_text = ros('timeout 5 ros2 topic echo /apex/race_driver/status --once --field data')
    (out / 'startup_status.txt').write_text(initial_text)
    initial = json.loads(next(line for line in initial_text.splitlines() if line.startswith('{')))
    assert initial['phase'] == 'wait' and not initial['armed']
    assert initial['lap_own'] == initial['travel_m'] == initial['recover_count'] == 0
    assert initial['recover_stuck_count'] == 0
    assert initial['mission']['distance_m'] == 0 and not initial['plan_ready']
    assert initial['g_left'] == 1.0 and initial['confidence'] == 0.0
    assert initial['speed_model']['gain'] == 1.0 and initial['speed_model']['offset'] == 0.0
    assert initial['speed_model']['forward_only'] and not initial['speed_model']['longitudinal_applied']
    assert state['RestartCount'] == 0
    assert current['APEX_RACE_RUN_DIR'] == '/work/ros2_ws/apex_race/' + run
    clean_report = root / 'ros2_ws/apex_race' / run / 'fresh_start.json'
    assert clean_report.exists(), 'fresh-start cleanup did not create this run'
    shutil.copyfile(clean_report, out / 'fresh_start.json')
    map_check = (
        "import json,rclpy; from rcl_interfaces.srv import GetParameters; rclpy.init(); "
        "n=rclpy.create_node('fresh_map_check'); c=n.create_client(GetParameters,'/slam_toolbox/get_parameters'); "
        "assert c.wait_for_service(timeout_sec=3); q=GetParameters.Request(); q.names=['map_file_name','mode']; "
        "f=c.call_async(q); rclpy.spin_until_future_complete(n,f,timeout_sec=3); v=f.result().values; "
        "assert v[0].type==0 or (v[0].type==4 and not v[0].string_value), 'SLAM restores a previous map'; "
        "assert v[1].type==0 or v[1].string_value=='mapping'; "
        "print(json.dumps({'map_file_name':v[0].string_value,'mode':v[1].string_value,'restore_disabled':True})); "
        "n.destroy_node(); rclpy.shutdown()"
    )
    import shlex
    (out / 'fresh_map_check.json').write_text(ros('python3 -c ' + shlex.quote(map_check)))
    # Validate parameters read back from the running node, inside the same ROS image.
    checks = (
        "import rclpy; from rcl_interfaces.srv import GetParameters; rclpy.init(); "
        "n=rclpy.create_node('segment_parameter_check'); "
        "c=n.create_client(GetParameters,'/apex_race_driver/get_parameters'); "
        "assert c.wait_for_service(timeout_sec=3); "
        "q=GetParameters.Request(); "
        "q.names=['require_arm','mission.enabled','mission.max_distance_m','mission.max_duration_s',"
        "'mission.max_command_mps','mission.forward_only','adaptive.recover_enabled','reactive.w_min_m',"
        "'vehicle.steer_plan_deg','adaptive.sat_cmd_frac','reactive.local_path_lookahead_m',"
        "'adaptive.lookahead_add_m','adaptive.lookahead_add_s','reactive.use_wall_planner','reactive.w_smooth',"
        "'vehicle.steer_max_left_deg','vehicle.steer_max_right_deg','vehicle.steer_plan_left_deg','vehicle.steer_plan_right_deg',"
        "'reactive.preview_while_waiting','reactive.n_k1','reactive.n_k2',"
        "'mission.max_reverse_command_mps','adaptive.recover_distance_m','adaptive.recover_timeout_s','adaptive.recover_max_per_lap','laps','adaptive.lead_preserve_turn','reactive.d_follow_left_m','reactive.d_follow_m','reactive.c_soft_m','adaptive.recover_exit_free_m','adaptive.recover_max_without_progress','adaptive.recover_progress_reset_m','closure.arm_distance_m','adaptive.plan_with_identified_model','closure.overlap_m','race.guide_handover_s',"
        "'reactive.v_explore_mps','reactive.v_min_mps','reactive.v_blind_mps','race.v_max_mps',"
        "'adaptive.longitudinal_forward_only','adaptive.longitudinal_allow_overspeed',"
        "'adaptive.launch_forward_command_scale','vehicle_yaml','adaptive.speed_feedback_forward_scale','adaptive.kp_speed_forward','adaptive.steering_confidence_gate']; "
        "f=c.call_async(q); rclpy.spin_until_future_complete(n,f,timeout_sec=3); "
        "v=f.result().values; "
        "assert [v[0].bool_value,v[1].bool_value,v[5].bool_value,v[6].bool_value]==[True,True,False,True]; "
        "assert [v[2].double_value,v[3].double_value,v[4].double_value,v[7].double_value]==[40.,300.," + repr(profile['command_cap']) + ",.9]; "
        "assert [x.double_value for x in v[8:13]]==[12.5,1.,.45,0.,0.]; assert not v[13].bool_value; assert v[14].double_value==.1; "
        "assert [x.double_value for x in v[15:19]]==[22.5,12.96,22.5,12.96]; "
        "assert not v[19].bool_value; "
        "assert [v[20].integer_value,v[21].integer_value]==[21,11]; "
        "assert [x.double_value for x in v[22:25]]==[.40,.50,3.0]; "
        "assert [v[25].integer_value,v[26].integer_value]==[36,2]; assert v[27].bool_value; "
        "assert [x.double_value for x in v[28:32]]==[0.,.36,.15,.60]; "
        "assert v[32].integer_value==12 and v[33].double_value==.75; "
        "assert v[34].double_value==5.; "
        "assert v[35].bool_value and v[36].double_value==0. and v[37].double_value==1.; "
        "assert [x.double_value for x in v[38:42]]==" + repr(profile['references']) + "; "
        "assert v[42].bool_value and v[43].bool_value and v[44].double_value==" + repr(profile['forward_scale']) + "; "
        "assert v[45].string_value==" + repr('/work/' + profile['vehicle_yaml']) + "; "
        "assert v[46].double_value==" + repr(profile['forward_scale']) + " and v[47].double_value==.17; "
        "assert v[48].bool_value==" + repr(profile['steering_confidence_gate']) + "; "
        "print('runtime limits verified'); n.destroy_node(); rclpy.shutdown()"
    )
    import shlex
    arm_path = '/work/' + str((out / 'arm_until_done.py').relative_to(root))
    (out / 'arm_helper_check.log').write_text(ros('python3 ' + shlex.quote(arm_path) + ' --help'))
    (out / 'parameter_check.log').write_text(ros('python3 -c ' + shlex.quote(checks)))
    (out / 'after_parameter_status.txt').write_text(ros('timeout 5 ros2 topic echo /apex/race_driver/status --once --field data'))
    ram_marker = Path(__file__).with_name('.ram_capture_active.json')
    if ram_marker.exists():
        storage = json.loads(ram_marker.read_text())
        for label, directory in [('diagnostics', out), ('race', root / 'ros2_ws/apex_race' / run)]:
            host_fs = sp.check_output(['stat', '-f', '-c', '%T', str(directory)], text=True).strip()
            docker_fs = ros('stat -f -c %T ' + shlex.quote('/work/' + str(directory.relative_to(root)))).strip()
            assert host_fs == docker_fs == 'tmpfs', 'trial output is not on RAM: ' + label
            storage[label] = {'host': host_fs, 'container': docker_fs}
        (out / 'ram_storage_preflight.json').write_text(json.dumps(storage, indent=2) + '\n')
    # Passive raw sensors enable independent replay after the car is returned.
    # This recorder never publishes commands and does not backpressure sensors.
    sensors = out / 'sensors'
    sensors.mkdir()
    capture_log = (out / 'sensor_capture.log').open('w')
    sensor_out = '/work/' + str(sensors.relative_to(root))
    sensor_pid = '/work/' + str((out / 'sensor_capture.pid').relative_to(root))
    capture_command = ('source /opt/ros/jazzy/setup.bash && '
                       'echo $$ > ' + shlex.quote(sensor_pid) + ' && exec python3 '
                       '/work/ros2_ws/scripts/capture/record_real2sim_sensors.py '
                       '--best-effort --duration 430 --out ' + shlex.quote(sensor_out))
    capture = sp.Popen(['docker', 'exec', 'apex_pipeline', 'bash', '-lc', capture_command],
                       stdout=capture_log, stderr=sp.STDOUT)
    resource_log = (out / 'resource_probe.log').open('w')
    resource = sp.Popen(['python3', str(Path(__file__).with_name('sample_resources.py')),
                         '--output', str(out / 'host_resources.csv'), '--duration', '430'],
                        stdout=resource_log, stderr=sp.STDOUT)
    print('Waiting for 10 s of clean host power while DISARMED (maximum 90 s)', flush=True)
    with (out / 'power_settling.jsonl').open('w') as settling:
        def log_power_sample(sample):
            settling.write(json.dumps(sample) + '\n')
            settling.flush()
        wait_for_stable_power(on_sample=log_power_sample)
    with (out / 'check.log').open('w') as f:
        sp.run(['./tools/core/apex_race_up.sh', 'check', '--duration', '5', '--stationary',
                '--expect-phase', 'wait', '--cmd-topic', '/apex/cmd_vel_track',
                '--json', '/work/' + str((out / 'check.json').relative_to(root))],
               stdout=f, stderr=sp.STDOUT, timeout=25, check=True)
    status_text = ros('timeout 5 ros2 topic echo /apex/race_driver/status --once --field data')
    status = json.loads(next(s for s in status_text.splitlines() if s.startswith('{')))
    (out / 'before.json').write_text(json.dumps(status, indent=2))
    assert status['phase'] == 'wait' and not status['armed'] and status['estimator_usable']
    assert status['mission']['max_command_mps'] == profile['command_cap']
    assert not status['sign_fault']
    bridge_text = ros('timeout 5 ros2 topic echo /apex/vehicle/drive_bridge_status --once --field data')
    bridge = json.loads(next(s for s in bridge_text.splitlines() if s.startswith('{')))
    (out / 'bridge_before.json').write_text(json.dumps(bridge, indent=2))
    assert bridge['actuation_profile'] == 'affine'
    assert bridge['speed_map']['esc_zero_pct'] == profile['esc_zero_pct']
    assert bridge['speed_map']['pct_per_mps'] == profile['pct_per_mps']
    assert bridge['speed_map']['esc_zero_rev_pct'] == 18.0 and bridge['speed_map']['rev_pct_per_mps'] == 33.0
    assert bridge['max_speed_pct'] == 28.2 and bridge['max_steering_deg'] <= 12.5
    assert bridge['max_steering_left_deg'] == 22.5 and bridge['max_steering_right_deg'] == 12.96
    assert bridge['max_reverse_speed_pct'] == 31.2
    servo = bridge['steering_servo_state']
    assert servo['asymmetric_limits']
    assert bridge['max_steering_left_deg'] <= servo['authority_left_deg']
    assert bridge['max_steering_right_deg'] <= servo['authority_right_deg']
    assert bridge['applied_speed_pct'] == 0.0 and bridge['desired_linear_x_mps'] == 0.0
    assert bridge['cmd_gap_ms_max'] < 100.0, 'bridge developed a command gap after the preflight check'
    check_power()
    assert capture.poll() is None, 'raw sensor recorder exited before arm'
    assert resource.poll() is None, 'resource recorder exited before arm'
    assert (out / 'host_resources.csv').stat().st_size > 200, 'no process resource capture before arm'
    assert (sensors / 'imu_raw.csv').stat().st_size > 100, 'no raw IMU capture before arm'
    assert (sensors / 'lidar_scans.csv').stat().st_size > 100, 'no raw LiDAR capture before arm'
    if args.diagnose_only:
        print('Disarmed diagnostic preflight passed, run ' + run, flush=True)
        raise SystemExit(0)
    print('Starting the agreed bounded autonomous trial, run ' + run, flush=True)
    # Both timeouts live on the Pi; a lost SSH session cannot keep arm alive.
    # Driver mission = 300 s / 40 m; publisher also exits on DONE or bad power.
    command = ('source /opt/ros/jazzy/setup.bash && '
               'timeout --signal=INT --kill-after=1s 306s python3 ' + shlex.quote(arm_path) + ' --max-seconds 304 '
               '--output /work/' + str(out.relative_to(root)) +
               ' --power-file /work/ros2_ws/apex_race/' + run + '/power.csv')
    with (out / 'arm.log').open('w') as f:
        result = sp.run(['docker', 'exec', 'apex_pipeline', 'bash', '-lc', command],
                        stdout=f, stderr=sp.STDOUT, timeout=312)
    assert result.returncode in (0, 3, 124, 130, 137), result.returncode

finally:
    try:
        ros("timeout 3 ros2 topic pub -1 /apex/tracking/arm std_msgs/msg/Bool '{data: false}'", timeout=5)
    except (sp.CalledProcessError, sp.TimeoutExpired):
        pass
    if capture is not None:
        try:
            stop_capture = ("import os,signal; from pathlib import Path; "
                            "p=int(Path(" + repr(sensor_pid) + ").read_text()); "
                            "c=Path('/proc/'+str(p)+'/cmdline').read_bytes().split(bytes([0])); "
                            "assert b'/work/ros2_ws/scripts/capture/record_real2sim_sensors.py' in c; "
                            "assert " + repr(sensor_out.encode()) + " in c; os.kill(p,signal.SIGINT)")
            ros('python3 -c ' + shlex.quote(stop_capture), timeout=5)
            capture.wait(timeout=5)
        except (sp.CalledProcessError, sp.TimeoutExpired, FileNotFoundError) as exc:
            (out / 'sensor_capture_stop_error.txt').write_text(str(exc))
        capture_log.close()
    try:
        (out / 'after.txt').write_text(ros('timeout 3 ros2 topic echo /apex/race_driver/status --once --field data', timeout=5))
    except (sp.CalledProcessError, sp.TimeoutExpired):
        pass
    try:
        command = 'python3 ' + shlex.quote('/work/' + str((out / 'save_final_map.py').relative_to(root)))
        command += ' --output ' + shlex.quote('/work/' + str(out.relative_to(root)))
        (out / 'save_final_map.log').write_text(ros(command, timeout=8))
    except (sp.CalledProcessError, sp.TimeoutExpired) as exc:
        (out / 'save_final_map.log').write_text(str(exc))
    try:
        (out / 'container.log').write_bytes(sp.check_output(['docker', 'logs', 'apex_pipeline'], stderr=sp.STDOUT, timeout=5))
    except (sp.CalledProcessError, sp.TimeoutExpired):
        pass
    sp.run(['./tools/core/apex_race_up.sh', 'down'], check=False, timeout=30)
    if resource is not None:
        try:
            resource.wait(timeout=4)
        except sp.TimeoutExpired:
            resource.terminate()
            resource.wait(timeout=3)
        resource_log.close()
    if run is None:
        last_run = root / 'ros2_ws/apex_race/.last_run'
        if last_run.exists():
            candidate = last_run.read_text().strip()
            # Do not copy another attempt's logs if startup failed before it
            # created a run. The startup log records the actual run identity.
            if ('[race] run ' + candidate + ' ->') in (out / 'startup.log').read_text():
                run = candidate
                (out / 'run_id.txt').write_text(run + '\n')
    if run:
        for name in ['driver.csv', 'driver_inputs.csv', 'estimator.csv', 'race_events.json', 'power.csv',
                     'plan.json', 'raceline.csv', 'map_route.csv', 'corridor.csv', 'lap1_path.csv',
                     'wall_log.csv', 'map_snapshot.pgm', 'map_snapshot.yaml']:
            src = root / 'ros2_ws/apex_race' / run / name
            if not src.exists():
                src = root / 'ros2_ws/apex_race' / run / 'plan' / name
            if src.exists():
                shutil.copyfile(src, out / name)
    print('Stopped; evidence ' + str(out), flush=True)
