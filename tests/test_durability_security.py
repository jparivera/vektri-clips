"""Isolated state-machine/security checks. Mock HTTP/DNS tests are control-path tests.

These do not claim live platform import, real remote publishing, or process-kill QA;
those are separate integration evidence. No production data directory is touched.
"""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import sqlite3
import ssl
import stat
import tarfile
import time

from fastapi.testclient import TestClient
import pytest

from clipstudio import config, db, security, service, task
from clipstudio.api import app
from clipstudio.schemas import default_timeline


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    directory = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    config.initialize()
    db.migrate()
    return directory


@pytest.fixture
def client():
    with TestClient(app) as value:
        yield value


def ready_project():
    pid = service.new_project('Durability test', 'source.mp4', {'permission_status': 'unknown'})
    source = config.project_dir(pid) / 'source/original.bin'
    source.write_bytes(b'original source integrity control fixture')
    record = json.loads(service.project_row(pid)['source'])
    record.update(path='source/original.bin', sha256=security.checksum(source), metadata={'duration': 24, 'width': 640, 'height': 360, 'audio': True})
    timeline = default_timeline(24)
    with db.tx() as connection:
        connection.execute("UPDATE projects SET source=?,revision=1,status='ready' WHERE id=?", (db.dumps(record), pid))
        connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (pid, 1, db.dumps(timeline), db.fingerprint(timeline), time.time()))
    return pid, source


def expire(job_id):
    with db.tx() as connection:
        connection.execute('UPDATE jobs SET lease_until=? WHERE id=?', (time.time() - 1, job_id))


def test_competing_claims_are_transactional_and_global_cap_one():
    queued = [db.enqueue('test', {'index': number}) for number in range(4)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda _: db.claim(), range(8)))
    successful = [item for item in attempts if item]
    assert len(successful) == 1
    assert successful[0]['id'] == queued[0]['id']
    assert successful[0]['attempts'] == 1
    assert len(db.all("SELECT * FROM jobs WHERE status='running'")) == 1


def test_idempotency_same_request_and_mismatch():
    original = db.enqueue('render', {'revision': 1}, key='one-action')
    assert db.enqueue('render', {'revision': 1}, key='one-action')['id'] == original['id']
    assert db.enqueue('render', {'revision': 1})['id'] == original['id']
    with pytest.raises(db.Conflict):
        db.enqueue('render', {'revision': 2}, key='one-action')
    with pytest.raises(db.Conflict):
        db.enqueue('analyze', {'revision': 1}, key='one-action')


def test_expired_lease_cannot_finish_before_recovery():
    db.enqueue('test', {})
    owner = db.claim()
    expire(owner['id'])
    with pytest.raises(db.LeaseLost):
        db.check_lease(owner['id'], owner['lease_token'])
    assert db.finish(owner['id'], owner['lease_token'], 'succeeded', {'invalid': True}) is False


def test_stale_recovery_fences_old_worker_and_reuses_attempt_budget():
    db.enqueue('test', {})
    first = db.claim()
    expire(first['id'])
    replacement = db.claim()
    assert replacement['id'] == first['id']
    assert replacement['attempts'] == 2
    assert replacement['lease_token'] != first['lease_token']
    assert db.finish(first['id'], first['lease_token'], 'succeeded') is False
    with pytest.raises(db.LeaseLost):
        db.heartbeat(first['id'], first['lease_token'])
    assert db.finish(replacement['id'], replacement['lease_token'], 'succeeded', {'valid': True})


def test_cancel_running_and_queued_and_retry_limit():
    queued = db.enqueue('first', {})
    assert db.cancel(queued['id'])['status'] == 'cancelled'
    assert db.claim() is None
    db.retry(queued['id'])
    first = db.claim()
    db.cancel(first['id'])
    with pytest.raises(db.Cancelled):
        db.check_lease(first['id'], first['lease_token'])
    db.finish(first['id'], first['lease_token'], 'succeeded', {'should_not_commit': True})
    ended = db.one('SELECT * FROM jobs WHERE id=?', (first['id'],))
    assert ended['status'] == 'cancelled' and ended['result'] is None
    for _ in range(config.MAX_ATTEMPTS - 1):
        db.retry(first['id'])
        owner = db.claim()
        db.finish(owner['id'], owner['lease_token'], 'failed', error='controlled failure')
    with pytest.raises(db.Conflict, match='Retry limit'):
        db.retry(first['id'])


def test_stale_cancelled_job_never_requeues():
    job = db.enqueue('test', {})
    owner = db.claim()
    db.cancel(job['id'])
    expire(job['id'])
    assert db.claim() is None
    assert db.one('SELECT status FROM jobs WHERE id=?', (job['id'],))['status'] == 'cancelled'


def test_cancel_queued_import_updates_project_state(client):
    imported = client.post('/api/v1/imports', files={'file': ('source.mp4', b'accepted bytes', 'video/mp4')}).json()
    response = client.post(f'/api/v1/jobs/{imported["job"]["id"]}/cancel')
    assert response.status_code == 200 and response.json()['status'] == 'cancelled'
    assert service.project_row(imported['project_id'])['status'] == 'cancelled'


def test_successful_stage_reused_after_retry_and_corruption_rebuilt():
    pid, source = ready_project()
    original_hash = security.checksum(source)
    job = db.enqueue('test', {}, pid)
    first = db.claim()
    context = task.StageContext(first['id'], first['lease_token'])
    calls = []
    def stage_work(folder):
        calls.append(True)
        (folder / 'result.json').write_text('{"ok":true}')
        return {'ok': True}
    _, path = context.stage('expensive', {'source': original_hash}, stage_work)
    db.finish(job['id'], first['lease_token'], 'failed', error='later stage failed')
    db.retry(job['id'])
    second = db.claim()
    resumed = task.StageContext(second['id'], second['lease_token'])
    result, reused_path = resumed.stage('expensive', {'source': original_hash}, stage_work)
    assert result == {'ok': True} and path == reused_path and calls == [True]
    assert resumed.reused == ['expensive']
    (config.project_dir(pid) / path / 'result.json').write_text('corrupt')
    resumed.stage('expensive', {'source': original_hash}, stage_work)
    assert len(calls) == 2
    assert security.checksum(source) == original_hash


def test_stage_lost_lease_cannot_publish_artifacts():
    pid, _ = ready_project()
    db.enqueue('test', {}, pid)
    owner = db.claim()
    context = task.StageContext(owner['id'], owner['lease_token'])
    def lose_lease(folder):
        (folder / 'result.json').write_text('uncommitted')
        expire(owner['id'])
        return {'invalid': True}
    with pytest.raises(db.LeaseLost):
        context.stage('test', {}, lose_lease)
    assert not db.all('SELECT * FROM stages')
    assert not list((config.project_dir(pid) / 'artifacts/test').glob('*/complete.json'))


def test_changed_source_blocks_render_before_work():
    pid, source = ready_project()
    job = service.queue_render(pid)
    owner = db.claim()
    source.write_bytes(b'changed externally')
    with pytest.raises(ValueError, match='Source integrity'):
        task.execute(job['id'], owner['lease_token'])
    assert not db.all('SELECT * FROM exports')


@pytest.mark.parametrize('segments', [[{'start': 24, 'end': 25, 'text': 'Outside source'}],
                                     [{'start': 0, 'end': 2}], [{'start': 2, 'end': 1, 'text': 'Reversed'}]])
def test_transcript_correction_segments_must_be_valid_evidence(client, segments):
    pid, _ = ready_project()
    response = client.put(f'/api/v1/projects/{pid}/transcript', json={'revision': 1, 'words': [], 'segments': segments})
    assert response.status_code == 422
    assert service.project_row(pid)['revision'] == 1
    assert service.project_row(pid)['transcript'] is None


def test_asr_concurrent_correction_is_reported_not_silent_success(monkeypatch):
    """Mock inference only to exercise a correction race; real ASR is tested separately."""
    from clipstudio import analysis
    pid, _ = ready_project()
    corrected = {'segments': [{'start': 1, 'end': 2, 'text': 'Operator correction'}],
                 'words': [{'start': 1, 'end': 2, 'word': 'Correction'}], 'corrected': True, 'audio_stream': 0}
    def inference_with_concurrent_correction(source, out, options, check):
        with db.tx() as connection:
            connection.execute('UPDATE projects SET transcript=? WHERE id=?', (db.dumps(corrected), pid))
        return {'segments': [{'start': 1, 'end': 2, 'text': 'Generated'}], 'words': [], 'audio_stream': 0}
    monkeypatch.setattr(analysis, 'transcribe', inference_with_concurrent_correction)
    job = db.enqueue('transcribe', {'model': 'tiny.en', 'audio_stream': 0, 'base_transcript_hash': None}, pid)
    owner = db.claim()
    with pytest.raises(db.Conflict, match='correct|changed|preserv'):
        task.execute(job['id'], owner['lease_token'])
    assert json.loads(service.project_row(pid)['transcript']) == corrected
    assert db.one("SELECT * FROM stages WHERE project_id=? AND name='transcribe'", (pid,)) is not None


def test_semantic_segment_authority_survives_caption_timing_edits(client):
    pid, _ = ready_project()
    original_words = [{'start': 1, 'end': 4, 'word': 'Original caption.'}]
    original_segments = [{'id': 'provider-0', 'start': 1, 'end': 4, 'text': 'Original caption.', 'avg_logprob': -0.2}]
    with db.tx() as connection:
        connection.execute('UPDATE projects SET transcript=? WHERE id=?', (db.dumps({'words': original_words, 'segments': original_segments}), pid))
    route = f'/api/v1/projects/{pid}/transcript'
    corrected_segments = [{'start': 1, 'end': 4, 'text': 'Corrected semantic description.'}]
    corrected = client.put(route, json={'revision': 1, 'words': original_words, 'segments': corrected_segments})
    assert corrected.status_code == 200
    assert corrected.json()['transcript']['segment_text_corrected'] is True
    assert corrected.json()['transcript']['words'] == original_words
    timing_words = [{'start': 1.2, 'end': 4.2, 'word': 'Original caption.'}]
    timed = client.put(route, json={'revision': corrected.json()['revision'], 'words': timing_words, 'segments': corrected_segments})
    assert timed.status_code == 200
    assert timed.json()['transcript']['segment_text_corrected'] is True
    assert timed.json()['transcript']['segments'][0]['text'] == 'Corrected semantic description.'
    new_words = [{'start': 1.2, 'end': 4.2, 'word': 'Explicitly changed caption text.'}]
    word_edit = client.put(route, json={'revision': timed.json()['revision'], 'words': new_words, 'segments': corrected_segments})
    assert word_edit.status_code == 200
    assert word_edit.json()['transcript']['segment_text_corrected'] is False


def test_segment_comparison_ignores_provider_metadata_and_empty_segments_derive_words(client):
    pid, _ = ready_project()
    words = [{'start': 1, 'end': 4, 'word': 'Original caption.'}]
    segments = [{'id': 'provider-0', 'start': 1, 'end': 4, 'text': 'Original caption.', 'avg_logprob': -0.2}]
    with db.tx() as connection:
        connection.execute('UPDATE projects SET transcript=? WHERE id=?', (db.dumps({'words': words, 'segments': segments}), pid))
    route = f'/api/v1/projects/{pid}/transcript'
    metadata_only = client.put(route, json={'revision': 1, 'words': words, 'segments': [{'start': 1, 'end': 4, 'text': 'Original caption.', 'id': 'new-id', 'avg_logprob': -0.7}]})
    assert metadata_only.status_code == 200
    assert metadata_only.json()['transcript']['segment_text_corrected'] is False
    with db.tx() as connection:
        transcript = metadata_only.json()['transcript']
        transcript['segment_text_corrected'] = True
        transcript['segments'][0]['text'] = 'An independent corrected segment.'
        connection.execute('UPDATE projects SET transcript=? WHERE id=?', (db.dumps(transcript), pid))
    cleared = client.put(route, json={'revision': metadata_only.json()['revision'], 'words': words, 'segments': []})
    assert cleared.status_code == 200
    assert cleared.json()['transcript']['segment_text_corrected'] is False
    assert cleared.json()['transcript']['segments'] == [{'start': 1, 'end': 4, 'text': 'Original caption.'}]


def test_full_disk_stage_failure_preserves_source(monkeypatch):
    pid, source = ready_project()
    original = source.read_bytes()
    db.enqueue('test', {}, pid)
    owner = db.claim()
    monkeypatch.setattr(security.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 0})())
    context = task.StageContext(owner['id'], owner['lease_token'])
    with pytest.raises(OSError, match='disk space'):
        context.stage('full-disk', {}, lambda _: pytest.fail('must not run without disk'))
    assert source.read_bytes() == original


@pytest.mark.parametrize('address', ['127.0.0.1', '10.0.0.2', '172.16.2.3', '192.168.0.1', '169.254.169.254', '0.0.0.0', '::1', 'fc00::1', 'fe80::1', '::ffff:127.0.0.1', '100.64.0.1'])
def test_ssrf_nonpublic_dns_results_rejected_control_path(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 80))])
    with pytest.raises(security.ImportRejected):
        security.validate_url('http://public-looking.example/media.mp4')


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://example.com/media', 'http://user:secret@example.com/a', 'http://localhost/a', 'http://test.local/a', 'http://example.com:8000/a', 'http://example.com/\nheader', 'http://[broken/a'])
def test_invalid_source_urls_rejected_without_fetch(url):
    with pytest.raises(security.ImportRejected):
        security.validate_url(url)


class MockResponse:
    """Control-path response; not a real media provider."""
    def __init__(self, body=b'video-bytes', status=200, headers=None):
        self.status = status
        self.headers = {'Content-Type': 'video/mp4', 'Content-Length': str(len(body)), **(headers or {})}
        self.body = io.BytesIO(body)
    def getheader(self, name):
        return self.headers.get(name)
    def read(self, size):
        return self.body.read(size)
    def read1(self, size):
        return self.body.read(size)


def mock_connections(monkeypatch, responses):
    connections = []
    class MockConnection:
        def __init__(self, host, port, address, timeout):
            connections.append({'host': host, 'address': address})
            self.response = responses.pop(0)
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return self.response
        def close(self):
            pass
    monkeypatch.setattr(security, 'PinnedHTTP', MockConnection)
    monkeypatch.setattr(security, 'PinnedHTTPS', MockConnection)
    return connections


def public_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 80))]


def test_dns_rebinding_redirect_is_revalidated_control_path(monkeypatch, tmp_path):
    lookups = 0
    def rebinding(*args, **kwargs):
        nonlocal lookups
        lookups += 1
        return public_dns() if lookups == 1 else [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 80))]
    monkeypatch.setattr(socket, 'getaddrinfo', rebinding)
    connections = mock_connections(monkeypatch, [MockResponse(status=302, headers={'Location': '/changed.mp4'})])
    destination = tmp_path / 'download.bin'
    with pytest.raises(security.ImportRejected, match='blocked'):
        security.fetch_public_media('http://example.com/media', destination)
    assert len(connections) == 1 and connections[0]['address'] == '93.184.216.34'
    assert not destination.exists()


def test_pinned_connection_uses_validated_ip_not_hostname(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(socket, 'create_connection', lambda address, timeout: calls.append(address) or sentinel)
    connection = security.PinnedHTTP('example.com', 80, '93.184.216.34', 10)
    connection.connect()
    assert calls == [('93.184.216.34', 80)] and connection.sock is sentinel


def test_approved_ip_fallback_prefers_ipv4_without_resolving_again(monkeypatch, tmp_path):
    """Mock connection failure validates routing only; not a live download claim."""
    lookups, attempted, closed = [], [], []
    def dual_stack_dns(host, port, **kwargs):
        lookups.append((host, port))
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2606:4700:4700::1111', port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.35', port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', port))]
    monkeypatch.setattr(socket, 'getaddrinfo', dual_stack_dns)
    class FailFirstConnection:
        def __init__(self, host, port, address, timeout):
            self.address = address
            attempted.append((host, port, address))
        def request(self, *args, **kwargs):
            if self.address == '93.184.216.34':
                raise OSError('network unreachable for this approved address')
        def getresponse(self):
            return MockResponse(body=b'approved second address')
        def close(self):
            closed.append(self.address)
    monkeypatch.setattr(security, 'PinnedHTTPS', FailFirstConnection)
    destination = tmp_path / 'source.bin'
    receipt = security.fetch_public_media('https://media.example/clip.mp4', destination)
    assert lookups == [('media.example', 443)]
    assert attempted == [('media.example', 443, '93.184.216.34'), ('media.example', 443, '93.184.216.35')]
    assert closed == ['93.184.216.34', '93.184.216.35']
    assert destination.read_bytes() == b'approved second address' and receipt['redirects'] == 0


def test_fallback_never_uses_private_address_from_mixed_dns(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: public_dns() + [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.0.0.1', 80))])
    connections = mock_connections(monkeypatch, [])
    with pytest.raises(security.ImportRejected, match='blocked'):
        security.fetch_public_media('http://example.com/clip.mp4', tmp_path / 'source.bin')
    assert connections == []


def test_approved_ip_attempts_capped_at_eight(monkeypatch, tmp_path):
    addresses = [f'93.184.216.{number}' for number in range(30, 42)]
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 80)) for address in addresses])
    attempted = []
    class Unreachable:
        def __init__(self, host, port, address, timeout):
            attempted.append(address)
        def request(self, *args, **kwargs):
            raise OSError('controlled unreachable address')
        def close(self):
            pass
    monkeypatch.setattr(security, 'PinnedHTTP', Unreachable)
    with pytest.raises(security.ImportRejected, match='Could not connect'):
        security.fetch_public_media('http://example.com/clip.mp4', tmp_path / 'source.bin')
    assert attempted == addresses[:8]
    assert not (tmp_path / 'source.bin').exists()


@pytest.mark.parametrize('response,message', [(MockResponse(status=403), 'HTTP 403'),
                                             (MockResponse(headers={'Content-Type': 'text/html'}), 'direct media')])
def test_http_or_media_rejection_does_not_retry_another_ip(monkeypatch, tmp_path, response, message):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: public_dns() + [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.35', 80))])
    connections = mock_connections(monkeypatch, [response, MockResponse()])
    with pytest.raises(security.ImportRejected, match=message):
        security.fetch_public_media('http://example.com/clip.mp4', tmp_path / 'source.bin')
    assert len(connections) == 1


def test_https_pinned_address_keeps_hostname_for_tls_verification(monkeypatch):
    context = ssl.create_default_context()
    raw_socket, wrapped_socket = object(), object()
    connected, wrapped = [], []
    monkeypatch.setattr(socket, 'create_connection', lambda address, timeout: connected.append(address) or raw_socket)
    monkeypatch.setattr(context, 'wrap_socket', lambda sock, server_hostname: wrapped.append((sock, server_hostname)) or wrapped_socket)
    monkeypatch.setattr(security.ssl, 'create_default_context', lambda: context)
    connection = security.PinnedHTTPS('media.example', 443, '93.184.216.34', 10)
    connection.connect()
    assert context.check_hostname is True and context.verify_mode == ssl.CERT_REQUIRED
    assert connected == [('93.184.216.34', 443)]
    assert wrapped == [(raw_socket, 'media.example')] and connection.sock is wrapped_socket


@pytest.mark.parametrize('abort', ['cancel', 'deadline'])
def test_download_checks_limits_between_incremental_reads_control_path(monkeypatch, tmp_path, abort):
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    clock, reads = [0.0], []
    monkeypatch.setattr(security, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    class TrickleResponse(MockResponse):
        def __init__(self):
            super().__init__(body=b'abcdefghij')
        def read(self, size):
            raise AssertionError('Buffered read can hide trickled bytes from deadline/cancellation checks')
        def read1(self, size):
            reads.append(size)
            clock[0] += 2
            return self.body.read(1)
    mock_connections(monkeypatch, [TrickleResponse()])
    def check_cancel():
        if abort == 'cancel' and reads:
            raise RuntimeError('controlled cancellation')
    destination = tmp_path / 'source.bin'
    with pytest.raises(RuntimeError if abort == 'cancel' else security.ImportRejected, match='cancellation|time limit'):
        security.fetch_public_media('http://example.com/clip.mp4', destination, check=check_cancel, deadline_seconds=3)
    assert len(reads) == (1 if abort == 'cancel' else 2)
    assert not destination.exists() and not destination.with_suffix('.download').exists()


def test_feedback_policy_reports_only_intervals_that_fit_the_limit(client):
    pid, _ = ready_project()
    with db.tx() as connection:
        for index in range(3):
            db.audit(connection, 'feedback.recorded', {'candidate_id': f'candidate-{index}', 'decision': 'reject',
                     'candidate_snapshot': {'start': 4 + index * 4, 'end': 5 + index * 4}}, pid)
    manual = [{'start': 0, 'end': 1} for _ in range(99)]
    response = client.post(f'/api/v1/projects/{pid}/analyze', json={'mode': 'heuristic', 'transcribe': False, 'use_feedback': True, 'exclusions': manual})
    assert response.status_code == 202
    options = response.json()['params']
    policy = options['feedback_policy']
    assert len(options['exclusions']) == 100
    assert policy['count'] == 1 and policy['omitted_due_to_limit'] == 2
    assert policy['applied_rejected_intervals'] == options['exclusions'][99:]


def test_audit_endpoint_filters_by_project_event_prefix_and_sequence_cursor(client):
    pid, _ = ready_project()
    other, _ = ready_project()
    with db.tx() as connection:
        for index in range(3):
            db.audit(connection, 'job.succeeded', {'job_id': f'job-{index}'}, pid)
        db.audit(connection, 'timeline.saved', {'revision': 2}, pid)
        db.audit(connection, 'job.failed', {'job_id': 'job-other'}, other)
    everything = client.get('/api/v1/audit').json()
    sequence = [row['seq'] for row in everything]
    assert sequence == sorted(sequence, reverse=True) and len(everything) >= 5
    assert all(isinstance(row['detail'], dict) for row in everything)
    jobs = client.get(f'/api/v1/audit?project_id={pid}&event=job.').json()
    assert [row['event'] for row in jobs] == ['job.succeeded'] * 3
    assert all(row['project_id'] == pid for row in jobs)
    page = client.get(f'/api/v1/audit?project_id={pid}&event=job.&limit=2').json()
    assert [row['seq'] for row in page] == [row['seq'] for row in jobs[:2]]
    older = client.get(f"/api/v1/audit?project_id={pid}&event=job.&limit=2&before={page[-1]['seq']}").json()
    assert [row['seq'] for row in older] == [row['seq'] for row in jobs[2:]]
    # LIKE wildcards in the prefix are literal: an underscore matches only an underscore.
    assert client.get('/api/v1/audit?event=job_').json() == []
    assert client.get('/api/v1/audit?event=%').json() == []
    assert [row['event'] for row in client.get('/api/v1/audit?event=job.failed').json()] == ['job.failed']
    assert client.get('/api/v1/audit?limit=0').status_code == 422
    assert client.get('/api/v1/audit?limit=5000').status_code == 422
    assert client.get('/api/v1/audit?before=0').status_code == 422
    assert client.delete('/api/v1/audit').status_code == 405
    assert client.post('/api/v1/audit', json={}).status_code == 405


def test_redirect_count_limit_control_path(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    connections = mock_connections(monkeypatch, [MockResponse(status=302, headers={'Location': '/again'}) for _ in range(6)])
    with pytest.raises(security.ImportRejected, match='redirects'):
        security.fetch_public_media('http://example.com/media', tmp_path / 'source.bin')
    assert len(connections) == 6


@pytest.mark.parametrize('response,message', [(MockResponse(headers={'Content-Type': 'text/html'}), 'direct media'),
                                               (MockResponse(body=b'12345', headers={'Content-Length': '100000'}), 'limit'),
                                               (MockResponse(body=b'12', headers={'Content-Length': '3'}), 'interrupted')])
def test_remote_response_failure_never_installs_partial_control_path(monkeypatch, tmp_path, response, message):
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    mock_connections(monkeypatch, [response])
    destination = tmp_path / 'source.bin'
    with pytest.raises(security.ImportRejected, match=message):
        security.fetch_public_media('http://example.com/media', destination, max_bytes=100)
    assert not destination.exists() and not destination.with_suffix('.download').exists()


def test_artifact_paths_symlinks_and_unregistered_types():
    pid, source = ready_project()
    assert security.authorized_artifact(pid, 'source/original.bin') == source
    for path in ['../studio.sqlite3', '/etc/passwd', 'artifacts/.private/a.json', 'uploads/file.bin']:
        with pytest.raises((ValueError, FileNotFoundError)):
            security.authorized_artifact(pid, path)
    folder = config.project_dir(pid) / 'artifacts'
    (folder / 'outside.json').symlink_to(config.DB)
    with pytest.raises(security.ImportRejected, match='Symlink'):
        security.authorized_artifact(pid, 'artifacts/outside.json')
    (folder / 'script.exe').write_bytes(b'bad')
    with pytest.raises(security.ImportRejected, match='type'):
        security.authorized_artifact(pid, 'artifacts/script.exe')


def test_origin_host_and_cross_site_boundaries(client):
    assert client.get('/api/v1/health', headers={'Host': 'evil.example'}).status_code == 400
    assert client.get('/api/v1/health', headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/v1/uploads', headers={'Sec-Fetch-Site': 'cross-site'}, json={'filename': 'a.mp4', 'size': 10}).status_code == 403
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['X-Frame-Options'] == 'DENY'


def test_resumable_offsets_interrupted_upload_and_duplicate_completion(client):
    created = client.post('/api/v1/uploads', json={'filename': '../clip — café.mp4', 'size': 6}).json()
    path = f'/api/v1/uploads/{created["id"]}'
    assert client.patch(path, headers={'Upload-Offset': '0'}, content=b'abc').status_code == 200
    assert client.post(path + '/complete').status_code == 409
    assert client.get(path).json()['offset'] == 3
    assert client.patch(path, headers={'Upload-Offset': '0'}, content=b'abc').status_code == 409
    assert client.patch(path, headers={'Upload-Offset': '3'}, content=b'defg').status_code == 422
    assert client.patch(path, headers={'Upload-Offset': '3'}, content=b'def').status_code == 200
    first = client.post(path + '/complete')
    second = client.post(path + '/complete')
    assert first.status_code == second.status_code == 202
    assert first.json()['project_id'] == second.json()['project_id']
    assert first.json()['job']['id'] == second.json()['job']['id']
    project = service.get_project(first.json()['project_id'])
    assert project['source']['original_name'] == 'clip — café.mp4'
    assert (config.project_dir(project['id']) / 'source/original.bin').read_bytes() == b'abcdef'


def test_fixture_idempotency_rejects_different_asset(client):
    headers = {'Idempotency-Key': 'fixture-request'}
    first = client.post('/api/v1/fixtures/spoken_interview.mp4', headers=headers)
    assert first.status_code == 202
    repeat = client.post('/api/v1/fixtures/spoken_interview.mp4', headers=headers)
    assert repeat.status_code == 202 and repeat.json()['project_id'] == first.json()['project_id']
    assert client.post('/api/v1/fixtures/action_motion.mp4', headers=headers).status_code == 409


def test_import_key_cannot_alias_an_unrelated_job(client):
    other = db.enqueue('model-download', {'name': 'tiny.en'}, key='collision')
    response = client.post('/api/v1/imports', headers={'Idempotency-Key': 'collision'}, files={'file': ('clip.mp4', b'abc', 'video/mp4')})
    assert response.status_code == 409


def test_multipart_key_checks_actual_bytes_and_metadata(client):
    headers = {'Idempotency-Key': 'actual-byte-identity'}
    first = client.post('/api/v1/imports', headers=headers, files={'file': ('clip.mp4', b'first', 'video/mp4')}, data={'title': 'Same'})
    repeat = client.post('/api/v1/imports', headers=headers, files={'file': ('clip.mp4', b'first', 'video/mp4')}, data={'title': 'Same'})
    assert first.status_code == repeat.status_code == 202
    assert first.json()['project_id'] == repeat.json()['project_id']
    changed = client.post('/api/v1/imports', headers=headers, files={'file': ('clip.mp4', b'second', 'video/mp4')}, data={'title': 'Same'})
    assert changed.status_code == 409
    changed_title = client.post('/api/v1/imports', headers=headers, files={'file': ('clip.mp4', b'first', 'video/mp4')}, data={'title': 'Different'})
    assert changed_title.status_code == 409
    assert len(db.all('SELECT * FROM projects')) == 1
    source = config.project_dir(first.json()['project_id']) / 'source/original.bin'
    assert source.read_bytes() == b'first'


def test_url_import_identity_reuses_only_identical_request(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    first = service.import_url('https://example.com/source.mp4', 'Remote', key='remote-key')
    repeat = service.import_url('https://example.com/source.mp4', 'Remote', key='remote-key')
    assert repeat['project_id'] == first['project_id']
    with pytest.raises(db.Conflict):
        service.import_url('https://example.com/different.mp4', 'Remote', key='remote-key')
    assert len(db.all('SELECT * FROM projects')) == 1


def test_accept_staged_transaction_rollback_keeps_upload_bytes(monkeypatch):
    staged = config.DATA / 'uploads/staged'
    staged.write_bytes(b'recoverable bytes')
    original_enqueue = db.enqueue
    def interrupted(*args, **kwargs):
        raise RuntimeError('process interruption before job commit')
    monkeypatch.setattr(db, 'enqueue', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        service.accept_staged(staged, 'clip.mp4', key='recoverable')
    assert db.all('SELECT * FROM projects') == []
    assert db.all('SELECT * FROM jobs') == []
    assert staged.read_bytes() == b'recoverable bytes'
    monkeypatch.setattr(db, 'enqueue', original_enqueue)
    recovered = service.accept_staged(staged, 'clip.mp4', key='recoverable')
    assert recovered['job']['status'] == 'queued'
    assert len(db.all('SELECT * FROM projects')) == 1


def test_old_finalizing_upload_can_resume(client):
    created = client.post('/api/v1/uploads', json={'filename': 'resume.mp4', 'size': 3}).json()
    path = f'/api/v1/uploads/{created["id"]}'
    client.patch(path, headers={'Upload-Offset': '0'}, content=b'abc')
    with db.tx() as connection:
        connection.execute("UPDATE uploads SET status='finalizing' WHERE id=?", (created['id'],))
    resumed = client.post(path + '/complete')
    assert resumed.status_code == 202
    assert client.get(path).json()['status'] == 'complete'


def test_concurrent_same_key_import_commits_one_project(tmp_path):
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'parallel import control fixture')
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: service.import_path(source, key='parallel-import'), range(4)))
    assert len({item['project_id'] for item in responses}) == 1
    assert len({item['job']['id'] for item in responses}) == 1
    assert len(db.all('SELECT * FROM projects')) == 1


def test_stale_agent_evidence_rejected_after_timeline_edit():
    pid, _ = ready_project()
    packet = service.evidence_packet(pid)
    timeline = service.get_project(pid)['timeline']
    timeline['segments'][0]['end'] = 10
    service.update_timeline(pid, 1, timeline)
    with pytest.raises(db.Conflict, match='Evidence changed'):
        service.apply_agent_decisions(pid, {'packet_fingerprint': packet['packet_fingerprint'], 'agent_name': 'offline test', 'candidates': []})


@pytest.mark.parametrize('evidence,accepted', [
    ({'start': 2, 'end': 4, 'text': 'External reviewer evidence'}, True),
    ({'time': 0}, True),
    ({'start': 24, 'end': 24}, True),
    ({'start': 2, 'end': 25}, False),
    ({'start': 2, 'end': 1}, False),
    ({'start': 2, 'end': True}, False),
    ({'start': 0, 'end': False}, False),
    ({'time': True}, False),
    ({'start': False, 'end': 1}, False),
    ({'start': 2, 'end': '4'}, False),
    ({'start': 2, 'end': None}, False),
    ({'start': 2, 'end': float('nan')}, False),
    ({'start': 2, 'end': float('inf')}, False),
])
def test_agent_evidence_time_and_end_bounds(evidence, accepted):
    pid, _ = ready_project()
    packet = service.evidence_packet(pid)
    candidate = {'id': 'external-proposal', 'start': 0, 'end': 5, 'title': 'Review required',
                 'rationale': 'A timestamp validation test, not a verified semantic claim.',
                 'evidence': [evidence], 'confidence': 0.5, 'uncertainty': 'External proposal',
                 'scores': {}, 'method': 'agent-assisted'}
    decisions = {'packet_fingerprint': packet['packet_fingerprint'], 'agent_name': 'offline test', 'candidates': [candidate]}
    if accepted:
        result = service.apply_agent_decisions(pid, decisions)
        assert result['candidates'][0]['evidence'] == [evidence]
        assert 'human review' in result['limitations'][0]
    else:
        with pytest.raises(ValueError, match='[Ee]vidence'):
            service.apply_agent_decisions(pid, decisions)
        assert service.project_row(pid)['analysis'] is None
        assert db.all("SELECT * FROM audit WHERE event='agent.proposals'") == []


def test_agent_evidence_revalidated_at_commit_race(monkeypatch):
    pid, _ = ready_project()
    packet = service.evidence_packet(pid)
    original_packet = service.evidence_packet
    invoked = False
    def change_after_packet_read(project_id):
        nonlocal invoked
        value = original_packet(project_id)
        if not invoked:
            invoked = True
            timeline = service.get_project(pid)['timeline']
            timeline['segments'][0]['end'] = 10
            service.update_timeline(pid, 1, timeline)
        return value
    monkeypatch.setattr(service, 'evidence_packet', change_after_packet_read)
    with pytest.raises(db.Conflict, match='Evidence changed'):
        service.apply_agent_decisions(pid, {'packet_fingerprint': packet['packet_fingerprint'], 'agent_name': 'offline test', 'candidates': []})


def test_publication_is_draft_and_edit_invalidates_approval(client):
    pid, _ = ready_project()
    eid = db.uid()
    with db.tx() as connection:
        connection.execute('INSERT INTO exports VALUES(?,?,?,?,?,?,?)', (eid, pid, 1, 'asset-hash', 'artifacts/render/fake', '{}', time.time()))
    response = client.post('/api/v1/publications', json={'project_id': pid, 'export_id': eid, 'destination': 'manual',
                           'account': 'manual operator', 'caption': 'A review draft', 'visibility': 'private', 'timezone': 'Australia/Perth', 'permission_status': 'unknown'})
    assert response.status_code == 200, response.text
    publication = response.json()
    assert publication['can_publish'] is False and publication['status'] == 'draft'
    assert client.post(f'/api/v1/publications/{publication["id"]}/publish').status_code == 409
    # Simulate a future adapter's approved record to test invalidation policy.
    with db.tx() as connection:
        connection.execute("UPDATE publications SET approval=?,status='approved' WHERE id=?", ('{"confirmed":true}', publication['id']))
    timeline = service.get_project(pid)['timeline']
    timeline['segments'][0]['end'] = 10
    service.update_timeline(pid, 1, timeline)
    updated = db.one('SELECT * FROM publications WHERE id=?', (publication['id'],))
    assert updated['status'] == 'approval-invalidated' and updated['approval'] is None
    assert updated['remote_id'] is None


def draft_for_approval(client):
    pid, _ = ready_project()
    eid, asset = db.uid(), 'a' * 64
    with db.tx() as connection:
        connection.execute('INSERT INTO exports VALUES(?,?,?,?,?,?,?)', (eid, pid, 1, asset, 'artifacts/render/fake', '{}', time.time()))
    payload = {'project_id': pid, 'export_id': eid, 'destination': 'manual', 'account': 'review operator', 'caption': 'Exact caption',
               'visibility': 'private', 'schedule': '2026-12-04T10:00:00', 'timezone': 'Australia/Perth', 'permission_status': 'unknown'}
    publication = client.post('/api/v1/publications', json=payload).json()
    approval = {'draft': publication['draft'], 'asset_fingerprint': asset, 'rights_acknowledgment': 'I reviewed the supplied permission status; it remains unknown.', 'confirm_manual_handoff': True}
    return pid, publication, approval


@pytest.mark.parametrize('field,value', [('destination', 'youtube'), ('account', 'different account'), ('caption', 'Changed caption'),
                                         ('visibility', 'public'), ('schedule', '2026-12-05T10:00:00'), ('timezone', 'UTC'), ('permission_status', 'owned')])
def test_manual_approval_requires_exact_reviewed_fields(client, field, value):
    _, publication, approval = draft_for_approval(client)
    approval['draft'][field] = value
    response = client.post(f'/api/v1/publications/{publication["id"]}/approve', json=approval)
    assert response.status_code == 409
    assert db.one('SELECT approval FROM publications WHERE id=?', (publication['id'],))['approval'] is None


def test_manual_approval_asset_fingerprint_and_old_revision(client):
    pid, publication, approval = draft_for_approval(client)
    path = f'/api/v1/publications/{publication["id"]}/approve'
    wrong_asset = {**approval, 'asset_fingerprint': 'b' * 64}
    assert client.post(path, json=wrong_asset).status_code == 409
    timeline = service.get_project(pid)['timeline']
    timeline['segments'][0]['end'] = 12
    service.update_timeline(pid, 1, timeline)
    assert client.post(path, json=approval).status_code == 409


def test_manual_approval_valid_then_draft_edit_invalidates(client):
    _, publication, approval = draft_for_approval(client)
    path = f'/api/v1/publications/{publication["id"]}'
    approved = client.post(path + '/approve', json=approval)
    assert approved.status_code == 200
    assert approved.json()['status'] == 'ready-for-manual-upload'
    assert approved.json()['can_publish'] is False
    assert approved.json()['remote_id'] is None
    assert client.post(path + '/publish').status_code == 409
    draft = dict(publication['draft'])
    draft['caption'] = 'Edited after approval'
    edited = client.put(path, json=draft)
    assert edited.status_code == 200 and edited.json()['approval'] is None and edited.json()['status'] == 'draft'


def test_manual_approval_needs_nonblank_acknowledgment(client):
    _, publication, approval = draft_for_approval(client)
    approval['rights_acknowledgment'] = '   '
    assert client.post(f'/api/v1/publications/{publication["id"]}/approve', json=approval).status_code == 422


@pytest.mark.parametrize('field,value', [('timezone', 'Nowhere/Invalid'), ('schedule', 'not a date')])
def test_publication_update_validates_schedule_and_timezone(client, field, value):
    _, publication, _ = draft_for_approval(client)
    draft = dict(publication['draft'])
    draft[field] = value
    assert client.put(f'/api/v1/publications/{publication["id"]}', json=draft).status_code == 422


def write_tar(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, body in entries:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'projects/../../outside', 'logs/unexpected'])
def test_restore_rejects_traversal_and_unexpected_paths(tmp_path, name):
    from clipstudio.backup import restore
    archive, destination = tmp_path / 'hostile.tar.gz', tmp_path / 'restored'
    write_tar(archive, [(name, b'hostile')])
    with pytest.raises(ValueError, match='Unsafe|Unexpected'):
        restore(archive, destination)
    assert not destination.exists() and not destination.with_name('restored.restoring').exists()


@pytest.mark.parametrize('link_type', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_restore_rejects_link_members(tmp_path, link_type):
    from clipstudio.backup import restore
    archive = tmp_path / 'symlink.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        item = tarfile.TarInfo('projects/symlink')
        item.type = link_type
        item.linkname = '/etc/passwd'
        handle.addfile(item)
    with pytest.raises(ValueError, match='Unsafe'):
        restore(archive, tmp_path / 'restored')


def test_restore_rejects_duplicate_and_checksum_mismatch(tmp_path):
    from clipstudio.backup import restore
    duplicate = tmp_path / 'duplicate.tar.gz'
    write_tar(duplicate, [('studio.sqlite3', b'one'), ('studio.sqlite3', b'two')])
    with pytest.raises(ValueError, match='Duplicate'):
        restore(duplicate, tmp_path / 'restore-duplicate')
    wrong_hash = tmp_path / 'bad-hash.tar.gz'
    manifest = {'schema_version': 1, 'database_version': 1, 'files': {'studio.sqlite3': '0' * 64}}
    write_tar(wrong_hash, [('studio.sqlite3', b'fake-database'), ('backup-manifest.json', json.dumps(manifest).encode())])
    with pytest.raises(ValueError, match='checksum mismatch'):
        restore(wrong_hash, tmp_path / 'restore-bad-hash')


def test_full_backup_restore_original_artifact_model_and_database_parity(tmp_path):
    from clipstudio.backup import backup, restore
    pid, source = ready_project()
    artifact = config.project_dir(pid) / 'artifacts/example/result.json'
    artifact.parent.mkdir()
    artifact.write_text('{"saved":"edit decision"}')
    model = config.DATA / 'models/model.bin'
    model.write_bytes(b'local model control fixture')
    hashes = {str(path.relative_to(config.DATA)): security.checksum(path) for path in (source, artifact, model)}
    archive = tmp_path / 'full-backup.tar.gz'
    created = backup(archive)
    assert created['contains_originals'] is True
    destination = tmp_path / 'restored'
    restored = restore(archive, destination)
    assert restored['verified_files'] == created['files']
    assert all(security.checksum(destination / name) == value for name, value in hashes.items())
    with sqlite3.connect(destination / 'studio.sqlite3') as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert connection.execute('SELECT title FROM projects WHERE id=?', (pid,)).fetchone()[0] == 'Durability test'
        assert connection.execute('SELECT revision FROM revisions WHERE project_id=?', (pid,)).fetchone()[0] == 1
    assert (destination / source.relative_to(config.DATA)).stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match='new, nonexistent'):
        restore(archive, destination)
    assert security.checksum(source) == hashes[str(source.relative_to(config.DATA))]


def test_full_backup_dereferences_hardlinked_artifacts_and_stays_private(tmp_path):
    from clipstudio.backup import backup, restore
    pid, source = ready_project()
    artifacts = config.project_dir(pid) / 'artifacts/hardlink-cache'
    artifacts.mkdir()
    rendered = artifacts / 'render.mp4'
    rendered_bytes = b'complete artifact bytes\x00\xff' * 4096
    rendered.write_bytes(rendered_bytes)
    cached = artifacts / 'same-render-cached.mp4'
    cached.hardlink_to(rendered)
    source_copy = artifacts / 'source-copy.bin'
    source_copy.hardlink_to(source)
    assert rendered.stat().st_ino == cached.stat().st_ino
    assert source.stat().st_ino == source_copy.stat().st_ino
    inputs = (source, source_copy, rendered, cached)
    contents = {str(path.relative_to(config.DATA)): path.read_bytes() for path in inputs}
    hashes = {str(path.relative_to(config.DATA)): security.checksum(path) for path in inputs}

    archive_path = tmp_path / 'hardlinked-library.tar.gz'
    result = backup(archive_path)
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
    assert result['sha256'] == security.checksum(archive_path)
    assert not archive_path.with_suffix('.gz.partial').exists()
    with tarfile.open(archive_path, 'r:gz') as archive:
        assert all(member.isfile() and not member.islnk() and not member.issym() for member in archive.getmembers())
        for name, expected in contents.items():
            member = archive.getmember(name)
            assert member.size == len(expected)
            assert archive.extractfile(member).read() == expected

    destination = tmp_path / 'restored-hardlinked-library'
    restored = restore(archive_path, destination)
    assert restored['verified_files'] == result['files']
    for name, expected in contents.items():
        restored_file = destination / name
        assert stat.S_ISREG(restored_file.stat().st_mode) and not restored_file.is_symlink()
        assert restored_file.read_bytes() == expected
        assert security.checksum(restored_file) == hashes[name]
    assert len({(destination / name).stat().st_ino for name in contents}) == len(contents)
    restored_source = destination / source.relative_to(config.DATA)
    assert restored_source.stat().st_mode & 0o222 == 0
    assert all(security.checksum(path) == hashes[str(path.relative_to(config.DATA))] for path in inputs)


def test_backup_refuses_pending_jobs_and_unfinished_uploads(client, tmp_path):
    from clipstudio.backup import backup
    job = db.enqueue('test', {})
    with pytest.raises(ValueError, match='pending jobs|Finish or cancel'):
        backup(tmp_path / 'pending.tar.gz')
    db.cancel(job['id'])
    upload = client.post('/api/v1/uploads', json={'filename': 'partial.mp4', 'size': 10}).json()
    client.patch(f'/api/v1/uploads/{upload["id"]}', headers={'Upload-Offset': '0'}, content=b'abc')
    with pytest.raises(ValueError, match='upload'):
        backup(tmp_path / 'unfinished-upload.tar.gz')
