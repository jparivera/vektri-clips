"""Local process readiness and expiring worker heartbeats (no credentials)."""
import json
import math
import os
from pathlib import Path
import time
import uuid

import psutil

from . import config

MIN_AVAILABLE_RAM = 1536 * 1024 * 1024
HEARTBEAT_MAX_AGE = 10.0


def worker_directory():
    return config.DATA / 'logs' / 'workers'


class WorkerHeartbeat:
    def __init__(self):
        directory = worker_directory()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.pid = os.getpid()
        self.created_at = psutil.Process(self.pid).create_time()
        self.path = directory / f'{self.pid}-{uuid.uuid4().hex}.json'
        self.temporary = self.path.with_suffix('.tmp')

    def report(self, state, job_id=None, available_ram=None):
        if available_ram is None:
            available_ram = psutil.virtual_memory().available
        payload = {'pid': self.pid, 'process_started_at': self.created_at,
                   'heartbeat_at': time.time(), 'state': state, 'job_id': job_id,
                   'available_ram_bytes': available_ram,
                   'minimum_available_ram_bytes': MIN_AVAILABLE_RAM}
        self.temporary.write_text(json.dumps(payload, allow_nan=False))
        self.temporary.chmod(0o600)
        self.temporary.replace(self.path)

    def close(self):
        self.path.unlink(missing_ok=True)
        self.temporary.unlink(missing_ok=True)


def worker_status():
    """A worker is alive even while idle or safely waiting for RAM.

    Heartbeats expire, and PID creation times prevent a reused PID from making
    an old crashed worker look healthy. This does not infer life from queued jobs.
    """
    now = time.time()
    workers = []
    for path in worker_directory().glob('*.json'):
        try:
            if path.stat().st_size > 16 * 1024:
                continue
            entry = json.loads(path.read_text())
            if not isinstance(entry, dict):
                continue
            pid, state = entry['pid'], entry['state']
            heartbeat_at, started_at = entry['heartbeat_at'], entry['process_started_at']
            job_id, available = entry.get('job_id'), entry.get('available_ram_bytes')
            if type(pid) is not int or not 0 < pid < 2**31:
                continue
            if not isinstance(state, str) or state not in {'idle', 'busy', 'waiting_memory'}:
                continue
            if any(type(value) not in (int, float) for value in (heartbeat_at, started_at)):
                continue
            heartbeat_at, started_at = float(heartbeat_at), float(started_at)
            if not all(math.isfinite(value) and value > 0 for value in (heartbeat_at, started_at)):
                continue
            if job_id is not None and (not isinstance(job_id, str) or len(job_id) > 160):
                continue
            if available is not None and (type(available) is not int or not 0 <= available < 2**63):
                continue
            age = now - heartbeat_at
            if not 0 <= age <= HEARTBEAT_MAX_AGE:
                continue
            process = psutil.Process(pid)
            if abs(process.create_time() - started_at) > 0.01:
                continue
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            workers.append({'pid': pid, 'state': state, 'heartbeat_at': heartbeat_at,
                            'process_started_at': started_at, 'job_id': job_id,
                            'available_ram_bytes': available,
                            'minimum_available_ram_bytes': MIN_AVAILABLE_RAM,
                            'heartbeat_age_seconds': round(age, 3)})
        except (OSError, ValueError, KeyError, TypeError, OverflowError, psutil.Error):
            continue
    workers.sort(key=lambda item: item['pid'])
    states = {worker['state'] for worker in workers}
    state = ('busy' if 'busy' in states else 'waiting_memory' if 'waiting_memory' in states
             else 'idle' if workers else 'unavailable')
    return {'alive': bool(workers), 'state': state, 'workers': workers,
            'minimum_available_ram_bytes': MIN_AVAILABLE_RAM}
