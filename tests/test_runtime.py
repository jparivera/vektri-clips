"""Lifecycle regressions use isolated state; no production library or RAM bypass."""
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
import urllib.request
import time
from types import SimpleNamespace

import psutil
import pytest

from clipstudio import cli, config, db, runtime, worker


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    directory = tmp_path / 'studio'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    db.migrate()
    monkeypatch.setattr(worker, 'stopping', False)


def memory(monkeypatch, amount):
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=amount))


def test_ram_guard_preserves_queue_attempts_and_reports_live_waiting_worker(monkeypatch):
    memory(monkeypatch, runtime.MIN_AVAILABLE_RAM - 1)
    queued = db.enqueue('test', {})
    monkeypatch.setattr(db, 'claim', lambda: pytest.fail('RAM guard must run before claiming work'))
    heartbeat = runtime.WorkerHeartbeat()
    try:
        assert worker.run_once(heartbeat) == 'waiting_memory'
        row = db.one('SELECT * FROM jobs WHERE id=?', (queued['id'],))
        assert row['status'] == 'queued' and row['attempts'] == 0
        assert '1.5 GiB' in row['stage']
        state = runtime.worker_status()
        assert state['alive'] is True and state['state'] == 'waiting_memory'
        assert state['workers'][0]['job_id'] is None
    finally:
        heartbeat.close()


def test_ram_guard_does_not_mark_empty_queue_as_blocked(monkeypatch):
    memory(monkeypatch, 0)
    assert worker.run_once() == 'idle'


def test_once_memory_deferral_returns_retryable_exit_code(monkeypatch, capsys):
    memory(monkeypatch, 0)
    db.enqueue('test', {})
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    assert worker.main(['--once']) == 75
    assert json.loads(capsys.readouterr().out)['state'] == 'waiting_memory'
    assert runtime.worker_status()['alive'] is False


def test_once_empty_queue_has_successful_idle_result(monkeypatch, capsys):
    memory(monkeypatch, 0)
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    assert worker.main(['--once']) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'idle'


def test_idle_heartbeat_is_alive_and_close_removes_only_own_record():
    first, second = runtime.WorkerHeartbeat(), runtime.WorkerHeartbeat()
    try:
        first.report('idle')
        second.report('busy', 'example-job')
        state = runtime.worker_status()
        assert state['alive'] and state['state'] == 'busy' and len(state['workers']) == 2
        first.close()
        assert second.path.exists() and runtime.worker_status()['state'] == 'busy'
    finally:
        first.close()
        second.close()
    assert runtime.worker_status()['alive'] is False


@pytest.mark.parametrize('change', ['stale', 'pid_reused', 'invalid_json'])
def test_stale_or_invalid_heartbeat_does_not_claim_worker_alive(change):
    heartbeat = runtime.WorkerHeartbeat()
    try:
        heartbeat.report('idle')
        entry = json.loads(heartbeat.path.read_text())
        if change == 'stale':
            entry['heartbeat_at'] = time.time() - runtime.HEARTBEAT_MAX_AGE - 1
        elif change == 'pid_reused':
            entry['process_started_at'] -= 60
        heartbeat.path.write_text('{broken' if change == 'invalid_json' else json.dumps(entry))
        assert runtime.worker_status()['alive'] is False
    finally:
        heartbeat.close()


def test_worker_startup_failure_reports_exit_without_waiting():
    with pytest.raises(RuntimeError, match='during startup.*exit 3'):
        cli.wait_worker_ready(SimpleNamespace(poll=lambda: 3, pid=999999), timeout=1)


def test_worker_startup_readiness_requires_its_own_heartbeat():
    heartbeat = runtime.WorkerHeartbeat()
    try:
        heartbeat.report('idle')
        cli.wait_worker_ready(SimpleNamespace(poll=lambda: None, pid=os.getpid()), timeout=0.5)
        with pytest.raises(RuntimeError, match='did not report readiness'):
            cli.wait_worker_ready(SimpleNamespace(poll=lambda: None, pid=999999), timeout=0.01)
    finally:
        heartbeat.close()


def test_occupied_port_fails_before_database_or_worker_changes(monkeypatch):
    try:
        bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bound.bind(('127.0.0.1', 0))
        bound.listen(1)
    except PermissionError:
        pytest.skip('Host socket permission required for real occupied-port startup regression')
    try:
        monkeypatch.setattr(config, 'PORT', bound.getsockname()[1])
        monkeypatch.setattr(db, 'migrate', lambda: pytest.fail('Occupied port must fail before touching the library'))
        monkeypatch.setattr(cli.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Occupied port must not spawn a worker'))
        with pytest.raises(RuntimeError, match='Cannot start local API'):
            cli.serve()
    finally:
        bound.close()


@pytest.mark.timeout(20)
def test_real_server_startup_manages_worker_and_stops_if_worker_dies():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
    except PermissionError:
        pytest.skip('Host socket permission required for real startup/shutdown regression')
    source = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(source / 'backend'),
           'VEKTRI_DATA_DIR': str(config.DATA), 'VEKTRI_PORT': str(port)}
    process = subprocess.Popen([sys.executable, '-m', 'clipstudio.cli', 'serve'], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    managed_worker = None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert process.poll() is None, process.communicate()[1].decode()
            try:
                with opener.open(f'http://127.0.0.1:{port}/api/v1/health', timeout=0.2) as response:
                    assert json.load(response)['status'] == 'ok'
                break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail('Local API never became ready')
        state = runtime.worker_status()
        assert state['alive'] and state['state'] == 'idle'
        managed_worker = psutil.Process(state['workers'][0]['pid'])
        assert managed_worker.ppid() == process.pid
        managed_worker.kill()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 2
        assert b'Media worker exited unexpectedly' in stderr
        assert runtime.worker_status()['alive'] is False
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)
        if managed_worker is not None:
            try:
                if managed_worker.is_running() and managed_worker.status() != psutil.STATUS_ZOMBIE:
                    managed_worker.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize(('field', 'value'), [
    ('state', None), ('state', []), ('state', 'unknown'),
    ('pid', True), ('pid', '1'), ('pid', 1.5), ('pid', -1),
    ('heartbeat_at', []), ('heartbeat_at', float('nan')),
    ('process_started_at', True), ('process_started_at', float('inf')),
    ('job_id', {}), ('available_ram_bytes', '123'), ('available_ram_bytes', True),
])
def test_malformed_live_heartbeat_is_ignored(field, value):
    heartbeat = runtime.WorkerHeartbeat()
    try:
        heartbeat.report('idle')
        entry = json.loads(heartbeat.path.read_text())
        if field == 'state' and value is None:
            del entry[field]
        else:
            entry[field] = value
        heartbeat.path.write_text(json.dumps(entry))
        assert runtime.worker_status()['alive'] is False
    finally:
        heartbeat.close()


def test_each_heartbeat_samples_current_available_ram(monkeypatch):
    readings = iter([runtime.MIN_AVAILABLE_RAM + 123, runtime.MIN_AVAILABLE_RAM + 456])
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=next(readings)))
    heartbeat = runtime.WorkerHeartbeat()
    try:
        heartbeat.report('busy', 'example-job')
        assert runtime.worker_status()['workers'][0]['available_ram_bytes'] == runtime.MIN_AVAILABLE_RAM + 123
        heartbeat.report('busy', 'example-job')
        assert runtime.worker_status()['workers'][0]['available_ram_bytes'] == runtime.MIN_AVAILABLE_RAM + 456
    finally:
        heartbeat.close()


def test_worker_reports_current_ram_instead_of_reusing_guard_sample(monkeypatch):
    readings = iter([0, 123])
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=next(readings)))
    heartbeat = runtime.WorkerHeartbeat()
    try:
        assert worker.run_once(heartbeat) == 'idle'
        assert runtime.worker_status()['workers'][0]['available_ram_bytes'] == 123
    finally:
        heartbeat.close()


@pytest.mark.timeout(20)
@pytest.mark.parametrize('shutdown_signal', [signal.SIGTERM, signal.SIGINT])
def test_real_shutdown_during_readiness_stops_detached_worker(tmp_path, shutdown_signal):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
    except PermissionError:
        pytest.skip('Host socket permission required for startup-signal lifecycle regression')
    source = Path(__file__).resolve().parents[1]
    marker = tmp_path / 'readiness-worker.pid'
    # Only hold readiness open; the app spawns its actual worker unchanged.
    code = f'''from pathlib import Path
from clipstudio import cli
original = cli.wait_worker_ready
cli.runtime.worker_status = lambda: {{"workers": []}}
def held(child, timeout=10, interrupted=None):
    Path({str(marker)!r}).write_text(str(child.pid))
    return original(child, timeout=15, interrupted=interrupted)
cli.wait_worker_ready = held
cli.serve()
'''
    env = {**os.environ, 'PYTHONPATH': str(source / 'backend'),
           'VEKTRI_DATA_DIR': str(config.DATA), 'VEKTRI_PORT': str(port)}
    process = subprocess.Popen([sys.executable, '-c', code], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    managed_worker = None
    try:
        deadline = time.monotonic() + 8
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None, process.communicate()[1].decode()
            time.sleep(0.02)
        assert marker.exists(), 'Server never reached the deliberately held readiness wait'
        managed_worker = psutil.Process(int(marker.read_text()))
        assert managed_worker.ppid() == process.pid
        os.kill(process.pid, shutdown_signal)
        _, stderr = process.communicate(timeout=8)
        assert process.returncode == 128 + shutdown_signal, stderr.decode()
        try:
            assert not managed_worker.is_running() or managed_worker.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
        assert runtime.worker_status()['alive'] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
        if managed_worker is not None:
            try:
                if managed_worker.is_running() and managed_worker.status() != psutil.STATUS_ZOMBIE:
                    managed_worker.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.timeout(40)
def test_serve_managed_worker_exits_when_its_supervisor_dies_without_cleanup(tmp_path):
    """A closed terminal or SIGKILLed server must not leave a worker holding the library."""
    source = Path(__file__).resolve().parents[1]
    data = tmp_path / 'studio'
    env = {**os.environ, 'PYTHONPATH': str(source / 'backend'), 'VEKTRI_DATA_DIR': str(data)}
    # Stand-in supervisor: spawns the worker exactly as `serve` does, then idles until it is killed
    # without any chance to run cleanup code.
    script = ('import os,subprocess,sys,time\n'
              "child=subprocess.Popen([sys.executable,'-m','clipstudio.worker','--supervisor-pid',str(os.getpid())],start_new_session=True)\n"
              'print(child.pid,flush=True)\n'
              'time.sleep(60)\n')
    supervisor = subprocess.Popen([sys.executable, '-c', script], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, start_new_session=True)
    managed = None
    try:
        managed = psutil.Process(int(supervisor.stdout.readline().decode().strip()))
        heartbeats = data / 'logs' / 'workers'
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(json.loads(p.read_text()).get('pid') == managed.pid for p in heartbeats.glob('*.json')):
                break
            assert managed.is_running(), 'worker exited before reporting readiness'
            time.sleep(0.05)
        else:
            pytest.fail('Managed worker never reported a heartbeat')
        assert managed.ppid() == supervisor.pid
        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=5)
        managed.wait(timeout=15)
        assert not any(json.loads(p.read_text()).get('pid') == managed.pid for p in heartbeats.glob('*.json'))
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
        if managed is not None:
            try:
                if managed.is_running() and managed.status() != psutil.STATUS_ZOMBIE:
                    managed.kill()
            except psutil.NoSuchProcess:
                pass


def test_worker_rejects_a_supervisor_that_is_not_its_parent(monkeypatch, capsys):
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    # PID 1 can legitimately be the parent inside an isolated executor.
    monkeypatch.setattr(worker.os, 'getppid', lambda: 2)
    assert worker.main(['--once', '--supervisor-pid', '1']) == 2
    assert 'not the parent process' in capsys.readouterr().err
