"""Real stage caching/transactions with mocked vision, never model inference."""
import json
import time

import pytest

from clipstudio import analysis, config, db, service, task
from clipstudio.schemas import default_timeline
from clipstudio.security import checksum


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    directory = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    monkeypatch.setattr(analysis, 'transcribe', lambda *args: pytest.fail('Visual search must not invoke ASR'))
    monkeypatch.setattr(analysis, 'model_status', lambda: {'models': [{'id': 'clip-vit-b32', 'revision': 'test-pin-1', 'available': True}]})
    db.migrate()


def ready_project():
    pid = service.new_project('Visual isolation control', 'source.mp4', {'genre': 'speech'})
    path = config.project_dir(pid) / 'source/original.bin'
    path.write_bytes(b'original immutable source control; inference is mocked')
    source = {'path': 'source/original.bin', 'sha256': checksum(path), 'provenance': {},
              'metadata': {'duration': 24, 'audio': True, 'video': True}}
    transcript = {'segments': [{'id': 'original', 'start': 1, 'end': 2, 'text': 'Original text'}],
                  'words': [], 'audio_stream': 0, 'language': 'en', 'corrected': True}
    timeline = default_timeline(24)
    with db.tx() as connection:
        connection.execute("UPDATE projects SET source=?,revision=1,status='ready',transcript=? WHERE id=?",
                           (db.dumps(source), db.dumps(transcript), pid))
        connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (pid, 1, db.dumps(timeline), db.fingerprint(timeline), time.time()))
    return pid, path, transcript


def correct_captions(pid, text='New human caption'):
    with db.tx() as connection:
        row = connection.execute('SELECT * FROM projects WHERE id=?', (pid,)).fetchone()
        transcript = json.loads(row['transcript'])
        transcript['segments'][0]['text'] = text
        transcript['words'] = [{'word': text, 'start': 1, 'end': 2}]
        transcript['corrected'] = True
        timeline = json.loads(connection.execute('SELECT timeline FROM revisions WHERE project_id=? AND revision=?',
                                                (pid, row['revision'])).fetchone()['timeline'])
        timeline['captions']['words'] = transcript['words']
        revision = row['revision'] + 1
        connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (pid, revision, db.dumps(timeline), db.fingerprint(timeline), time.time()))
        connection.execute('UPDATE projects SET transcript=?,analysis=NULL,revision=?,updated_at=? WHERE id=?',
                           (db.dumps(transcript), revision, time.time(), pid))
    return transcript, timeline


def run(pid, **overrides):
    original = json.loads(service.project_row(pid)['transcript'])
    options = {'genre': 'action', 'mode': 'local', 'visual_search': True, 'query': 'a racing car',
               'transcribe': False, 'base_transcript_hash': db.fingerprint(original), **overrides}
    job = db.enqueue('analyze', options, pid)
    owner = db.claim()
    assert owner is not None
    try:
        result = task.execute(job['id'], owner['lease_token'])
    except BaseException:
        db.finish(job['id'], owner['lease_token'], 'failed', error='controlled test failure')
        raise
    db.finish(job['id'], owner['lease_token'], 'succeeded', result)
    return result


def visual_result():
    return {'method': 'visual-control', 'candidates': [{'id': 'visual-one', 'start': 1, 'end': 3,
                                                     'method': 'local_visual_semantic', 'title': 'Visual evidence'}]}


def test_visual_caption_correction_during_stage_commits_without_overwriting_transcript(monkeypatch):
    pid, _, _ = ready_project()
    corrected = {}
    def inspect(_source, _metadata, transcript, _options, out, _check):
        assert transcript is None
        corrected['transcript'], corrected['timeline'] = correct_captions(pid)
        result = visual_result()
        (out / 'analysis.json').write_text(db.dumps(result))
        return result
    monkeypatch.setattr(analysis, 'analyze', inspect)
    result = run(pid)
    saved = service.get_project(pid)
    assert result['analysis']['candidates'][0]['id'] == 'visual-one'
    assert saved['analysis']['method'] == 'visual-control'
    assert saved['revision'] == 2 and saved['timeline'] == corrected['timeline']
    assert saved['transcript'] == corrected['transcript']


def test_visual_stage_reuses_completed_result_after_caption_revision(monkeypatch):
    pid, _, _ = ready_project()
    calls = []
    def inspect(_source, _metadata, transcript, _options, out, _check):
        assert transcript is None
        calls.append(True)
        result = visual_result()
        (out / 'analysis.json').write_text(db.dumps(result))
        return result
    monkeypatch.setattr(analysis, 'analyze', inspect)
    first = run(pid)
    corrected, _ = correct_captions(pid)
    second = run(pid)
    assert len(calls) == 1
    assert second['reused_stages'] == ['analyze']
    assert first['analysis']['artifact_dir'] == second['analysis']['artifact_dir']
    assert json.loads(service.project_row(pid)['transcript']) == corrected


def test_speech_analysis_still_rejects_concurrent_transcript_correction(monkeypatch):
    pid, _, original = ready_project()
    corrected = {}
    def inspect(_source, _metadata, transcript, _options, _out, _check):
        assert transcript == original
        corrected['transcript'], _ = correct_captions(pid)
        return {'method': 'speech-control', 'candidates': []}
    monkeypatch.setattr(analysis, 'analyze', inspect)
    with pytest.raises(db.Conflict, match='Transcript was corrected during analysis'):
        run(pid, visual_search=False, genre='speech')
    assert json.loads(service.project_row(pid)['transcript']) == corrected['transcript']
    assert service.project_row(pid)['analysis'] is None


@pytest.mark.parametrize('change', ['source', 'query', 'model_revision'])
def test_visual_cache_still_invalidates_on_real_inputs(monkeypatch, change):
    pid, path, _ = ready_project()
    calls = []
    def inspect(_source, _metadata, transcript, _options, _out, _check):
        assert transcript is None
        calls.append(True)
        return visual_result()
    monkeypatch.setattr(analysis, 'analyze', inspect)
    first = run(pid)
    options = {}
    if change == 'source':
        path.write_bytes(b'different source control; hash updated as a new source identity')
        record = json.loads(service.project_row(pid)['source'])
        record['sha256'] = checksum(path)
        with db.tx() as connection:
            connection.execute('UPDATE projects SET source=? WHERE id=?', (db.dumps(record), pid))
    elif change == 'query':
        options['query'] = 'a car entering the pit lane'
    else:
        monkeypatch.setattr(analysis, 'model_status', lambda: {'models': [{'id': 'clip-vit-b32', 'revision': 'test-pin-2', 'available': True}]})
    second = run(pid, **options)
    assert len(calls) == 2 and second['reused_stages'] == []
    assert first['analysis']['artifact_dir'] != second['analysis']['artifact_dir']
