"""Real isolated filesystem/SQLite regressions; never alter the user library."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import psutil
import pytest

from clipstudio import db, retention

PID = 'a' * 32
SHA = 'b' * 64


@pytest.fixture
def library(tmp_path):
    root = tmp_path / 'data'
    for name in ('projects', 'models', 'uploads', 'logs'):
        (root / name).mkdir(parents=True)
    with sqlite3.connect(root / 'studio.sqlite3') as conn:
        conn.executescript(db.MIGRATION_1 + 'PRAGMA user_version=1;')
        conn.execute('PRAGMA journal_mode=WAL')
        source = {'path': 'source/original.bin', 'sha256': SHA}
        conn.execute('INSERT INTO projects(id,title,genre,status,source,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                     (PID, 'Test', 'action', 'ready', json.dumps(source), 0, 0))
    conn.close()
    source_dir = root / 'projects' / PID / 'source'
    source_dir.mkdir(parents=True)
    (source_dir / 'original.bin').write_bytes(b'original media must survive')
    return root


def sql(root, statement, params=()):
    with sqlite3.connect(root / 'studio.sqlite3') as conn:
        conn.execute(statement, params)
    conn.close()


def update_source(root, **values):
    sql(root, 'UPDATE projects SET source=? WHERE id=?',
        (json.dumps({'path': 'source/original.bin', 'sha256': SHA, **values}), PID))


def stage(root, name='prepare', number=1, files=None):
    fp = f'{number:064x}'
    relative = f'artifacts/{name}/{fp}'
    directory = root / 'projects' / PID / relative
    directory.mkdir(parents=True)
    files = files or {'proxy.mp4': b'x' * 16384, 'metadata.json': b'{}'}
    for filename, data in files.items():
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if name == 'render':
        timeline = {'clips': [{'start': 0, 'end': number}]}
        result = {'source_sha256': SHA, 'timeline': timeline, 'timeline_sha256': db.fingerprint(timeline)}
        sql(root, 'INSERT OR REPLACE INTO revisions VALUES(?,?,?,?,?)',
            (PID, number, json.dumps(timeline), db.fingerprint(timeline), 0))
    else:
        result = {'sha256': SHA}
    receipt = {'fingerprint': fp, 'result': result,
               'checksums': {filename: hashlib.sha256(data).hexdigest() for filename, data in files.items()}}
    (directory / 'complete.json').write_text(json.dumps(receipt))
    sql(root, 'INSERT INTO stages VALUES(?,?,?,?,?,?)', (PID, name, fp, relative, json.dumps(result), 0))
    return directory


def relative(root, path):
    return str(path.relative_to(root))


def paths(result):
    return {item['path'] for item in result['review_candidates']}


def heartbeat(root):
    target = root / 'logs' / 'workers' / 'test.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'pid': os.getpid(), 'process_started_at': psutil.Process().create_time(),
                                  'heartbeat_at': time.time(), 'state': 'idle'}))
    return target


def test_read_only_plan_and_verify_do_not_create_original_wal_or_change_files(library):
    cache = stage(library)
    before = {str(p.relative_to(library)): p.read_bytes() for p in library.rglob('*') if p.is_file()}
    report = retention.plan(library)
    assert report['status'] == 'review_only', report
    assert paths(report) == {relative(library, cache)}
    assert report['estimated_reclaimable_bytes'] > 0
    assert report['read_only'] and report['automatic_deletion_supported'] is False
    assert retention.verify_plan(report, library)['valid']
    after = {str(p.relative_to(library)): p.read_bytes() for p in library.rglob('*') if p.is_file()}
    assert after == before
    assert not hasattr(retention, 'apply_plan')


def test_current_prepared_analysis_export_publication_and_all_assets_are_protected(library):
    prepared = stage(library, number=1)
    analyzed = stage(library, number=2)
    exported = stage(library, 'render', number=3)
    unreferenced = stage(library, number=4)
    update_source(library, prepared_dir=str(prepared.relative_to(library / 'projects' / PID)))
    sql(library, 'UPDATE projects SET analysis=?', (json.dumps({'artifact_dir': str(analyzed.relative_to(library / 'projects' / PID))}),))
    sql(library, 'INSERT INTO exports VALUES(?,?,?,?,?,?,?)', ('export', PID, 3, 'fp', str(exported.relative_to(library / 'projects' / PID)), '{}', 0))
    sql(library, 'INSERT INTO publications(id,project_id,export_id,asset_fingerprint,revision,draft,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)', ('publication', PID, 'export', 'fp', 3, '{}', 0, 0))
    (library / 'models' / 'model.bin').write_bytes(b'model')
    (library / 'uploads' / 'upload.bin').write_bytes(b'upload')
    report = retention.plan(library)
    assert paths(report) == {relative(library, unreferenced)}
    reasons = ' '.join(str(row['reasons']) for row in report['protected_paths'])
    assert 'Immutable original' in reasons and 'Current analysis' in reasons and 'Saved export' in reasons
    assert report['inventory']['models']['logical_bytes'] == 5
    assert report['inventory']['uploads']['logical_bytes'] == 6


@pytest.mark.parametrize('ref_type', ['relative', 'api_url', 'local_file', 'unknown'])
def test_current_evidence_to_old_stage_protects_that_stage(library, ref_type):
    old = stage(library)
    project_relative = str(old.relative_to(library / 'projects' / PID))
    value = {'relative': project_relative + '/proxy.mp4#/frame/1',
             'api_url': '/api/v1/artifacts/' + PID + '/' + project_relative + '/proxy.mp4',
             'local_file': 'proxy.mp4#/frame/1', 'unknown': 'older/evidence.json'}[ref_type]
    analysis = {'artifact_dir': project_relative, 'evidence': [{'ref': value}]} if ref_type == 'local_file' else {'evidence': [{'ref': value}]}
    sql(library, 'UPDATE projects SET analysis=?', (json.dumps(analysis),))
    assert not paths(retention.plan(library))


@pytest.mark.parametrize('modification', ['extra_file', 'missing_file', 'extra_dir', 'wrong_result', 'unsafe_receipt_path'])
def test_receipt_inventory_must_match_exactly(library, modification):
    cache = stage(library)
    if modification == 'extra_file':
        (cache / 'operator-notes.txt').write_text('Keep this')
    elif modification == 'missing_file':
        (cache / 'proxy.mp4').unlink()
    elif modification == 'extra_dir':
        (cache / 'operator-folder').mkdir()
    else:
        receipt = json.loads((cache / 'complete.json').read_text())
        if modification == 'wrong_result':
            receipt['result'] = {'sha256': 'c' * 64}
        else:
            receipt['checksums']['../outside'] = SHA
        (cache / 'complete.json').write_text(json.dumps(receipt))
    report = retention.plan(library)
    assert not paths(report) and report['estimated_reclaimable_bytes'] == 0
    assert report['retained_caches']


def test_render_requires_retained_source_and_revision_recipe(library):
    valid = stage(library, 'render', 1)
    stage(library, 'render', 2)
    sql(library, 'DELETE FROM revisions WHERE revision=2')
    report = retention.plan(library)
    assert paths(report) == {relative(library, valid)}
    (library / 'projects' / PID / 'source' / 'original.bin').unlink()
    assert not paths(retention.plan(library))


def test_unknown_unregistered_stages_and_quarantines_are_retained(library):
    stage(library, 'analyze')
    stage(library, 'package')
    unregistered = stage(library, 'prepare', 2)
    sql(library, 'DELETE FROM stages WHERE fingerprint=?', (f'{2:064x}',))
    (unregistered.parent / ('.invalid-' + 'b' * 64 + '-' + 'd' * 32)).mkdir()
    assert not paths(retention.plan(library))


def test_hardlinks_to_protected_and_external_files_have_zero_savings(library, tmp_path):
    cache = stage(library)
    (library / 'exports').mkdir()
    for file in cache.iterdir():
        os.link(file, library / 'exports' / file.name)
        os.link(file, tmp_path / file.name)
    report = retention.plan(library)
    assert paths(report)
    assert report['estimated_reclaimable_bytes'] == 0
    assert report['review_candidates'][0]['estimated_reclaimable_bytes'] == 0
    assert report['inventory']['total']['logical_bytes'] > report['inventory']['total']['unique_inode_bytes']


def test_shared_candidate_hardlinks_are_counted_once_in_total(library):
    first = stage(library, number=1)
    second = stage(library, number=2)
    target = second / 'proxy.mp4'
    target.unlink()
    os.link(first / 'proxy.mp4', target)
    report = retention.plan(library)
    assert len(paths(report)) == 2
    shared_size = min(target.stat().st_size, target.stat().st_blocks * 512)
    assert report['estimated_reclaimable_bytes'] == sum(row['estimated_reclaimable_bytes'] for row in report['review_candidates']) + shared_size


def test_abandoned_temps_require_age_and_no_active_lease_and_unknown_temp_retained(library):
    parent = library / 'projects' / PID / 'artifacts' / 'render'
    old = parent / ('.' + '1' * 64 + '-' + '2' * 32)
    old.mkdir(parents=True)
    (old / 'partial.mp4').write_bytes(b'partial')
    fresh = parent / ('.' + '3' * 64 + '-' + '4' * 32)
    fresh.mkdir()
    (parent / '.manual-notes').mkdir()
    for item in (old, old / 'partial.mp4'):
        os.utime(item, (time.time() - 90000,) * 2)
    report = retention.plan(library)
    assert paths(report) == {relative(library, old)}
    assert report['review_candidates'][0]['kind'] == 'abandoned_temporary'
    sql(library, "INSERT INTO jobs(id,kind,params,status,request_hash,lease_token,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", ('job', 'test', '{}', 'running', 'h', '2' * 32, 0, 0))
    assert not paths(retention.plan(library))


def test_busy_jobs_protect_projects_and_idle_worker_requires_maintenance(library):
    cache = stage(library)
    heartbeat(library)
    idle_report = retention.plan(library)
    assert paths(idle_report) == {relative(library, cache)}
    assert idle_report['maintenance_required'] and idle_report['worker']['alive']
    sql(library, 'INSERT INTO jobs(id,project_id,kind,params,status,request_hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)', ('job', PID, 'render', '{}', 'queued', 'h', 0, 0))
    busy_report = retention.plan(library)
    assert busy_report['maintenance_required'] and not paths(busy_report)
    assert not retention.verify_plan(idle_report, library)['valid']


def test_incomplete_upload_blocks_future_maintenance_but_report_is_read_only(library):
    stage(library)
    sql(library, 'INSERT INTO uploads(id,filename,size,options,created_at) VALUES(?,?,?,?,?)', ('upload', 'video', 100, '{}', 0))
    report = retention.plan(library)
    assert report['maintenance_required'] and paths(report)
    assert any('upload' in reason.lower() for reason in report['maintenance_reasons'])


@pytest.mark.parametrize('mutation', ['candidate_file', 'candidate_inode', 'original_metadata', 'new_reference', 'extra_candidate_file'])
def test_plan_staleness_covers_candidate_metadata_and_protecting_refs(library, mutation):
    cache = stage(library)
    old = retention.plan(library)
    if mutation == 'candidate_file':
        (cache / 'proxy.mp4').write_bytes(b'changed')
    elif mutation == 'candidate_inode':
        target = cache / 'proxy.mp4'
        content = target.read_bytes()
        target.unlink()
        target.write_bytes(content)
    elif mutation == 'original_metadata':
        (library / 'projects' / PID / 'source' / 'original.bin').write_bytes(b'changed source')
    elif mutation == 'extra_candidate_file':
        (cache / 'notes.txt').write_text('Operator addition')
    else:
        update_source(library, prepared_dir=str(cache.relative_to(library / 'projects' / PID)))
    assert not retention.verify_plan(old, library)['valid']


def test_heartbeat_and_logs_do_not_invalidate_plan(library):
    stage(library)
    beat = heartbeat(library)
    old = retention.plan(library)
    payload = json.loads(beat.read_text())
    payload['heartbeat_at'] = time.time()
    beat.write_text(json.dumps(payload))
    (library / 'logs' / 'worker.log').write_text('new logs')
    assert retention.verify_plan(old, library)['valid']


def test_wal_visible_export_reference_protects_cache(library):
    cache = stage(library)
    conn = sqlite3.connect(library / 'studio.sqlite3')
    try:
        conn.execute('PRAGMA wal_autocheckpoint=0')
        conn.execute('INSERT INTO exports VALUES(?,?,?,?,?,?,?)', ('export', PID, 0, 'fp', str(cache.relative_to(library / 'projects' / PID)), '{}', 0))
        conn.commit()
        assert (library / 'studio.sqlite3-wal').stat().st_size > 0
        report = retention.plan(library)
        assert report['status'] == 'review_only', report
        assert not paths(report)
    finally:
        conn.close()


@pytest.mark.parametrize('kind', ['file', 'directory', 'root', 'receipt_swap'])
def test_symlinks_are_refused_without_following_them(library, tmp_path, monkeypatch, kind):
    cache = stage(library)
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'private.txt'
    secret.write_text('must not be read')
    if kind == 'file':
        (cache / 'link').symlink_to(secret)
    elif kind == 'directory':
        (library / 'alias').symlink_to(outside, target_is_directory=True)
    elif kind == 'root':
        alias = tmp_path / 'alias'
        alias.symlink_to(library, target_is_directory=True)
        library = alias
    else:
        real = retention._read_json
        def swapped(root, rel, expected, budget, *args):
            if rel.endswith('complete.json'):
                marker = library / rel
                marker.unlink()
                marker.symlink_to(secret)
            return real(root, rel, expected, budget, *args)
        monkeypatch.setattr(retention, '_read_json', swapped)
    result = retention.plan(library)
    assert result['status'] == 'blocked' and not paths(result)
    assert secret.read_text() == 'must not be read'


@pytest.mark.parametrize('unsafe', ['../outside', '/tmp/outside', 'artifacts/../source', 'artifacts//render', 'artifacts\\render'])
def test_path_traversal_in_database_references_blocks_plan(library, unsafe):
    stage(library)
    update_source(library, prepared_dir=unsafe)
    result = retention.plan(library)
    assert result['status'] == 'blocked' and result['estimated_reclaimable_bytes'] == 0


def test_ref_changes_during_scan_are_rejected(library, monkeypatch):
    cache = stage(library)
    scan = retention._scan
    def concurrent(*args):
        entries = scan(*args)
        update_source(library, prepared_dir=str(cache.relative_to(library / 'projects' / PID)))
        return entries
    monkeypatch.setattr(retention, '_scan', concurrent)
    result = retention.plan(library)
    assert result['status'] == 'blocked'
    assert any('references changed' in reason for reason in result['blockers'])


def test_entry_depth_database_and_metadata_budgets_fail_closed(library, monkeypatch):
    stage(library)
    assert retention.plan(library, max_entries=1)['status'] == 'blocked'
    assert retention.plan(library, max_depth=1)['status'] == 'blocked'
    monkeypatch.setattr(retention, 'DB_LIMIT', 1)
    assert retention.plan(library)['status'] == 'blocked'


def test_large_model_bytes_are_inventoried_without_reading_or_hashing(library, monkeypatch):
    stage(library)
    model = library / 'models' / 'large.bin'
    with model.open('wb') as out:
        out.truncate(128 * 1024**2)
    real_open = retention._open_fd
    def checked(root, relative, **kwargs):
        assert not relative.startswith('models/'), 'Planner must not open model bytes'
        return real_open(root, relative, **kwargs)
    monkeypatch.setattr(retention, '_open_fd', checked)
    result = retention.plan(library)
    assert result['status'] == 'review_only'
    assert result['inventory']['models']['logical_bytes'] == 128 * 1024**2


def test_plan_tampering_wrong_library_and_blocked_plan_do_not_verify(library, tmp_path):
    stage(library)
    result = retention.plan(library)
    assert not retention.verify_plan(result, tmp_path)['valid']
    result['review_candidates'][0]['path'] = '../outside'
    assert not retention.verify_plan(result, library)['valid']
    assert not retention.verify_plan(retention.plan(library, max_entries=1), library)['valid']
    assert not retention.verify_plan({}, library)['valid']


def test_unreadable_schema_and_missing_database_are_blocked_without_initialization(library, tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert retention.plan(empty)['status'] == 'blocked'
    assert list(empty.iterdir()) == []
    sql(library, 'PRAGMA user_version=2')
    assert retention.plan(library)['status'] == 'blocked'


def test_invalid_bounds_do_not_scan(library):
    for kwargs in ({'max_entries': 50001}, {'max_depth': 25}, {'min_temp_age_seconds': 1}, {'min_temp_age_seconds': float('nan')}):
        with pytest.raises(ValueError):
            retention.plan(library, **kwargs)
