"""Isolated API/CLI and actual-byte integrity mechanics; no live media/cloud job.

The small file fixtures below have recorded decode flags solely to exercise the
binding checks. These tests do not claim those fixture bytes are decoded video.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import struct
import sys
import time
import urllib.request
import zipfile

from fastapi.testclient import TestClient
import pytest

from clipstudio import cli, config, db, operational_api as operation, service
from clipstudio.api import app
from clipstudio.schemas import default_timeline


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', data)
    monkeypatch.setattr(config, 'DB', data / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(data))
    config.initialize()
    db.migrate()
    with TestClient(app) as client:
        pid = service.new_project('Preflight fixture', 'source.bin')
        root = config.project_dir(pid)
        source = root / 'source/original.bin'
        source.write_bytes(b'Immutable source bytes for preflight integrity tests')
        source_record = json.loads(service.project_row(pid)['source'])
        source_record.update(path='source/original.bin', sha256=digest(source.read_bytes()),
                             metadata={'duration': 10, 'video': True, 'audio': True})
        timeline = default_timeline(10)
        export_id, asset = db.uid(), 'c' * 64
        folder = root / 'artifacts/export-fixture'
        folder.mkdir()
        video = folder / 'video.mp4'
        video.write_bytes(b'Bound bytes for testing recorded decode evidence only')
        manifest = {'project_id': pid, 'revision': 1, 'fingerprint': asset,
                    'source_sha256': source_record['sha256'], 'source_provenance': source_record['provenance'],
                    'timeline': timeline, 'video': 'video.mp4',
                    'verification': {'full_decode': 'passed', 'dimensions': 'passed',
                                     'duration_error_seconds': 0.001, 'duration_tolerance_seconds': 0.15},
                    'files': {'video.mp4': {'sha256': digest(video.read_bytes()), 'bytes': video.stat().st_size}}}
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        with zipfile.ZipFile(folder / 'package.zip', 'w', compression=zipfile.ZIP_DEFLATED) as package:
            package.write(video, 'video.mp4')
            package.write(folder / 'manifest.json', 'manifest.json')
        with db.tx() as connection:
            connection.execute("UPDATE projects SET source=?,revision=1,status='ready' WHERE id=?", (db.dumps(source_record), pid))
            connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (pid, 1, db.dumps(timeline), db.fingerprint(timeline), time.time()))
            connection.execute('INSERT INTO exports VALUES(?,?,?,?,?,?,?)',
                               (export_id, pid, 1, asset, 'artifacts/export-fixture', db.dumps(manifest), time.time()))
        publication = client.post('/api/v1/publications', json={'project_id': pid, 'export_id': export_id,
            'destination': 'manual', 'account': '', 'caption': 'Reviewed fixture credit',
            'visibility': 'private', 'timezone': 'Australia/Perth', 'permission_status': 'unknown'}).json()
        approved = client.post('/api/v1/publications/' + publication['id'] + '/approve', json={
            'draft': publication['draft'], 'asset_fingerprint': asset, 'confirm_manual_handoff': True,
            'rights_acknowledgment': 'I reviewed the recorded permission status; it is unknown.'})
        assert approved.status_code == 200
        yield {'client': client, 'pid': pid, 'publication': publication['id'], 'source': source,
               'source_record': source_record, 'folder': folder, 'video': video,
               'export_id': export_id, 'manifest': manifest, 'data': data}


def scenario(e):
    return {'request': {'project_id': e['pid'], 'source_sha256': e['source_record']['sha256'],
            'provider': 'hypothetical', 'model': 'hypothetical', 'clip_start_ms': 0, 'clip_end_ms': 8_000,
            'transmitted_fields': ['query_text'], 'input_tokens_max': 100, 'output_tokens_max': 100,
            'frame_count_max': 0, 'transfer_bytes_max': 400}, 'price_schedule': None,
            'budget': {'per_job_limit_micro_usd': 1_000, 'total_limit_micro_usd': 10_000,
                       'spent_micro_usd': 0, 'reserved_micro_usd': 0}}


def tree(path):
    return {str(p.relative_to(path)): digest(p.read_bytes()) for p in path.rglob('*') if p.is_file()}


def get_preflight(e):
    return e['client'].get('/api/v1/publications/' + e['publication'] + '/preflight')


def test_actual_api_preflights_leave_database_files_and_publication_unchanged(environment, monkeypatch):
    e = environment

    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must not queue, decode or perform an external network action.')

    monkeypatch.setattr(db, 'enqueue', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    before = tree(e['data'])
    cloud = e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=scenario(e))
    assert cloud.status_code == 200, cloud.text
    c = cloud.json()
    assert c['simulation_only'] and not c['can_execute'] and c['source_integrity_checked']
    assert c['estimated_max_micro_usd'] is None and 'price_schedule_missing' in c['blockers']
    assert 'hypothetical' in c['scenario_notice'] and c['spend_reserved_micro_usd'] == 0
    publication = get_preflight(e)
    assert publication.status_code == 200, publication.text
    p = publication.json()
    assert p['manual_handoff_ready'] and not p['can_publish'] and not p['publishing_attempted']
    assert 'account_scopes_unverified' in p['remote_blockers']
    assert p['current_file_integrity_verified'] is True
    assert tree(e['data']) == before  # Includes absent/present original WAL/SHM names.
    assert db.all('SELECT * FROM jobs') == []


@pytest.mark.parametrize('field', ['source', 'export', 'approval', 'api_key', 'account_scopes'])
def test_cloud_body_rejects_forged_trusted_observations_and_credentials(environment, field):
    e = environment; payload = scenario(e); payload[field] = {'claimed': True}
    response = e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=payload)
    assert response.status_code == 422


def test_cloud_budget_cannot_claim_authoritative_observed_timestamp(environment):
    e = environment; payload = scenario(e); payload['budget']['observed_at_unix'] = int(time.time())
    assert e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=payload).status_code == 422


def test_cloud_checks_url_selection_and_actual_source_hash(environment):
    e = environment; payload = scenario(e); payload['request']['project_id'] = 'f' * 32
    assert e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=payload).status_code == 422
    payload = scenario(e); payload['request']['source_sha256'] = 'f' * 64
    response = e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=payload)
    assert 'selected_source_mismatch' in response.json()['blockers']
    e['source'].write_bytes(b'Changed source')
    response = e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=scenario(e))
    assert response.status_code == 409


@pytest.mark.parametrize('change', ['video', 'package', 'manifest', 'source', 'missing_video', 'symlink_video'])
def test_current_file_tampering_blocks_manual_readiness(environment, tmp_path, change):
    e = environment
    if change == 'video': e['video'].write_bytes(b'Changed bytes')
    if change == 'source': e['source'].write_bytes(b'Changed source')
    if change == 'package': (e['folder'] / 'package.zip').write_bytes(b'Not a zip archive')
    if change == 'manifest': (e['folder'] / 'manifest.json').write_text('{}')
    if change == 'missing_video': e['video'].unlink()
    if change == 'symlink_video':
        outside = tmp_path / 'not-project-data'; outside.write_text('Do not read this target')
        e['video'].unlink(); e['video'].symlink_to(outside)
    result = get_preflight(e)
    assert result.status_code == 200, result.text
    assert not result.json()['manual_handoff_ready'] and result.json()['manual_blockers']
    assert 'Do not read this target' not in result.text
    assert result.json()['current_file_integrity_verified'] is False
    assert 'incomplete or failed' in result.json()['verification_basis']


@pytest.mark.parametrize('change', ['revision', 'approval', 'caption', 'permission', 'asset', 'timeline'])
def test_database_bindings_are_current_and_no_client_query_can_override_them(environment, change):
    e = environment
    with db.tx() as connection:
        if change == 'revision': connection.execute('UPDATE projects SET revision=2 WHERE id=?', (e['pid'],))
        if change == 'approval': connection.execute('UPDATE publications SET approval=NULL WHERE id=?', (e['publication'],))
        if change == 'asset': connection.execute('UPDATE publications SET asset_fingerprint=? WHERE id=?', ('f' * 64, e['publication']))
        if change == 'caption':
            draft = json.loads(db.one('SELECT draft FROM publications WHERE id=?', (e['publication'],))['draft'])
            draft['caption'] = 'Changed after approval'
            connection.execute('UPDATE publications SET draft=? WHERE id=?', (db.dumps(draft), e['publication']))
        if change == 'permission':
            source = copy.deepcopy(e['source_record']); source['provenance']['permission_status'] = 'owned'
            connection.execute('UPDATE projects SET source=? WHERE id=?', (db.dumps(source), e['pid']))
        if change == 'timeline': connection.execute('UPDATE revisions SET timeline=? WHERE project_id=?', ('{}', e['pid']))
    response = e['client'].get('/api/v1/publications/' + e['publication'] + '/preflight',
                              params={'approval': 'true', 'artifact_hashes_passed': 'true'})
    assert response.status_code == 200, response.text
    assert not response.json()['manual_handoff_ready']


def test_source_symlink_refused_before_hashing_target(environment, tmp_path):
    e = environment; outside = tmp_path / 'unrelated'; outside.write_bytes(e['source'].read_bytes())
    e['source'].unlink(); e['source'].symlink_to(outside)
    response = e['client'].post(f'/api/v1/projects/{e["pid"]}/cloud-preflight', json=scenario(e))
    assert response.status_code == 409


def test_duplicate_archive_members_and_wrong_archived_bytes_are_not_verified(environment):
    e = environment
    with zipfile.ZipFile(e['folder'] / 'package.zip', 'w') as package:
        package.writestr('video.mp4', b'Wrong bytes despite intact external file')
        package.write(e['folder'] / 'manifest.json', 'manifest.json')
    result = get_preflight(e).json()
    assert not result['manual_handoff_ready']
    assert 'artifact_size_mismatch' in result['manual_blockers'] or 'package_artifact_hash_changed' in result['manual_blockers']


def test_archive_expansion_is_bounded(environment, monkeypatch):
    e = environment
    monkeypatch.setattr(operation, 'READ_LIMIT', 64 * 1024)
    with zipfile.ZipFile(e['folder'] / 'package.zip', 'w', compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr('video.mp4', b'0' * (128 * 1024))
        package.write(e['folder'] / 'manifest.json', 'manifest.json')
    result = get_preflight(e).json()
    assert not result['manual_handoff_ready']
    assert 'package_expansion_limit_exceeded' in result['manual_blockers']


def test_archive_directory_is_bounded_before_zipfile_allocates_it(environment):
    e = environment; size = operation.ARCHIVE_METADATA_LIMIT + 1
    # Minimal EOCD advertises a directory larger than the allowed metadata read.
    eocd = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, 1, 1, size, 0, 0)
    (e['folder'] / 'package.zip').write_bytes(b'0' * size + eocd)
    result = get_preflight(e).json()
    assert not result['manual_handoff_ready']
    assert 'package_metadata_limit_exceeded' in result['manual_blockers']


def test_concurrent_database_change_invalidates_preflight(environment, monkeypatch):
    e = environment; original = operation.reference_snapshot; calls = []
    def changed():
        snapshot = original(); calls.append(1)
        if len(calls) == 2:
            next(p for p in snapshot['projects'] if p['id'] == e['pid'])['revision'] += 1
        return snapshot
    monkeypatch.setattr(operation, 'reference_snapshot', changed)
    assert get_preflight(e).status_code == 409


def test_metadata_without_actual_export_decode_evidence_remains_blocked(environment):
    e = environment; manifest = copy.deepcopy(e['manifest']); manifest['verification']['full_decode'] = 'unrun'
    with db.tx() as connection:
        connection.execute('UPDATE exports SET manifest=? WHERE id=?', (db.dumps(manifest), e['export_id']))
    (e['folder'] / 'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(e['folder'] / 'package.zip', 'w') as package:
        package.write(e['video'], 'video.mp4'); package.write(e['folder'] / 'manifest.json', 'manifest.json')
    result = get_preflight(e).json()
    assert not result['manual_handoff_ready']
    assert 'export_full_decode_unverified' in result['manual_blockers']


@pytest.mark.parametrize('command', ['cloud-preflight', 'publication-preflight'])
def test_cli_routes_only_to_preflight_and_prints_json(environment, monkeypatch, capsys, tmp_path, command):
    e = environment; calls = []
    def request(method, path, payload=None, key=None):
        calls.append((method, path, payload, key)); return {'simulation_only': True, 'can_execute': False}
    monkeypatch.setattr(cli, 'request', request)
    if command == 'cloud-preflight':
        file = tmp_path / 'scenario.json'; file.write_text(json.dumps(scenario(e)))
        argv = ['clipstudio', command, e['pid'], str(file)]
    else: argv = ['clipstudio', command, e['publication']]
    monkeypatch.setattr(sys, 'argv', argv); cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result['ok'] and result['result']['simulation_only']
    assert len(calls) == 1 and 'preflight' in calls[0][1] and calls[0][3] is None


def test_cli_rejects_unbounded_or_credential_fields_before_any_request(environment, monkeypatch, tmp_path):
    e = environment; file = tmp_path / 'scenario.json'; payload = scenario(e); payload['token'] = 'not-accepted'
    file.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'cloud-preflight', e['pid'], str(file)])
    monkeypatch.setattr(cli, 'request', lambda *args: pytest.fail('Invalid scenario must not be sent'))
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code == 2
    file.write_text(' ' * (64 * 1024 + 1))
    with pytest.raises(ValueError, match='64 KiB'): operation.read_cloud_scenario(file)


def test_cli_refuses_fifo_and_symlink_without_reading_targets(tmp_path):
    fifo = tmp_path / 'scenario.pipe'; os.mkfifo(fifo)
    with pytest.raises(ValueError, match='regular file'): operation.read_cloud_scenario(fifo)
    target = tmp_path / 'target.json'; target.write_text('{}')
    link = tmp_path / 'scenario.json'; link.symlink_to(target)
    with pytest.raises(OSError): operation.read_cloud_scenario(link)


def test_missing_identifiers_are_404(environment):
    e = environment
    assert e['client'].get('/api/v1/publications/' + 'f' * 32 + '/preflight').status_code == 404
    payload = scenario(e); payload['request']['project_id'] = 'f' * 32
    assert e['client'].post('/api/v1/projects/' + 'f' * 32 + '/cloud-preflight', json=payload).status_code == 404
