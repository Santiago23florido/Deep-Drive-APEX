"""Exercise the PC SSH boundary without a network connection or vehicle."""
from pathlib import Path
import os
import subprocess

import pytest

SCRIPT = Path(__file__).with_name('race_pi.sh')


@pytest.fixture
def ssh_mock(tmp_path):
    executable = tmp_path / 'ssh'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RACE_TEST_ARGS"\ncat > "$RACE_TEST_BODY"\nexit 23\n')
    executable.chmod(0o755)
    args, body = tmp_path / 'args', tmp_path / 'body'
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'],
               RACE_TEST_ARGS=str(args), RACE_TEST_BODY=str(body))
    return env, args, body


@pytest.mark.parametrize('mode', ['status', 'preflight', 'run'])
def test_explicit_mode_is_forwarded_and_ssh_failure_is_preserved(ssh_mock, mode):
    env, args, body = ssh_mock
    result = subprocess.run(['bash', str(SCRIPT), '--host', 'ensta@pi-test', mode],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 23
    assert args.read_text().splitlines()[-6:] == ['ensta@pi-test', 'bash', '-s', '--', mode, 'approved']
    assert 'BatchMode=yes' in args.read_text()
    assert body.read_text().startswith('set -euo pipefail\n')
    assert subprocess.run(['bash', '-n'], input=body.read_text(), text=True).returncode == 0


@pytest.mark.parametrize('arguments', [[], ['run', 'preflight'], ['--host', '-oProxyCommand=bad', 'run'],
                                       ['--host', 'pi;touch /tmp/unwanted', 'run'], ['--profile', 'unknown', 'run']])
def test_missing_or_ambiguous_mode_and_host_injection_never_connect(ssh_mock, arguments):
    env, args, body = ssh_mock
    result = subprocess.run(['bash', str(SCRIPT), *arguments], env=env, capture_output=True)
    assert result.returncode == 2
    assert not args.exists() and not body.exists()


def test_initialized_preview_lists_the_deployment_without_connecting(ssh_mock):
    env, args, body = ssh_mock
    result = subprocess.run(['bash', str(SCRIPT), '--profile', 'initialized', '--dry-run', 'run'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Would deploy and SHA256-verify 10 files' in result.stdout
    assert 'bash -s -- run initialized' in result.stdout
    assert not args.exists() and not body.exists()


def test_preview_prints_the_invocation_without_connecting(ssh_mock):
    env, args, body = ssh_mock
    result = subprocess.run(['bash', str(SCRIPT), '--dry-run', 'preflight'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert 'bash -s -- preflight' in result.stdout
    assert '--diagnose-only' in result.stdout
    assert not args.exists() and not body.exists()
