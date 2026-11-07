"""Real isolated DB/stage caching; OCR/ASR inference is never invoked."""
import json
import sys
import time

import pytest

from clipstudio import api, analysis, cli, config, db, models, ocr, service, task
from clipstudio.schemas import OCROptions, default_timeline
from clipstudio.security import checksum


@pytest.fixture
def project(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', data)
    monkeypatch.setattr(config, 'DB', data / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(data))
    monkeypatch.setattr(analysis, 'transcribe', lambda *a, **kw: pytest.fail('OCR must not invoke ASR'))
    db.migrate()
    pid = service.new_project('Routing control', 'source.mp4')
    path = config.project_dir(pid) / 'source/original.bin'
    path.write_bytes(b'Controlled media identity; no actual inference')
    source = {'path': 'source/original.bin', 'sha256': checksum(path), 'metadata': {'duration': 20}, 'provenance': {}}
    transcript = {'corrected': True, 'segments': [{'start': 0, 'end': 1, 'text': 'Human caption'}], 'words': []}
    timeline = default_timeline(20)
    with db.tx() as connection:
        connection.execute("UPDATE projects SET status='ready',revision=1,source=?,transcript=? WHERE id=?",
                           (db.dumps(source), db.dumps(transcript), pid))
        connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)',
                           (pid, 1, db.dumps(timeline), db.fingerprint(timeline), time.time()))
    calls = []
    def read(source_path, duration, options, out, check):
        check()
        assert source_path == path and duration == 20
        calls.append(dict(options))
        value = {'model_id': options['ocr_model'], 'configuration': dict(options),
                 'samples': [{'time': 2., 'detections': [{'text': 'raw x', 'model_score': .9}]}]}
        (out / 'ocr.json').write_text(db.dumps(value))
        return value
    monkeypatch.setattr(ocr, 'extract_text', read)
    return pid, path, transcript, calls


def run(pid, **overrides):
    options = OCROptions(**overrides)
    # Exercise the actual API queue function and validated payload, without a server.
    queued = api.queue_ocr(pid, options, None)
    owner = db.claim()
    assert owner['id'] == queued['id']
    try:
        result = task.execute(owner['id'], owner['lease_token'])
        assert db.finish(owner['id'], owner['lease_token'], 'succeeded', result)
        return result
    except BaseException:
        db.finish(owner['id'], owner['lease_token'], 'failed', error='controlled test failure')
        raise


def test_default_api_request_routes_baseline_v4_and_returns_selected_result(project):
    pid, path, transcript, calls = project
    result = run(pid)
    assert calls == [OCROptions().model_dump()]
    assert calls[0]['ocr_mode'] == 'baseline' and calls[0]['ocr_model'] == 'ppocr-v4'
    assert calls[0]['ocr_quality_samples'] == 1
    assert api.latest_ocr(pid)['ocr'] == result['ocr']
    assert result['source_sha256'] == checksum(path)
    assert service.get_project(pid)['transcript'] == transcript


@pytest.mark.parametrize('changed', [
    {'ocr_model': 'ppocr-v5-mobile'}, {'ocr_mode': 'detail'}, {'ocr_quality_samples': 3},
    {'ocr_sample_seconds': 9}, {'ocr_max_samples': 60},
])
def test_each_operator_option_invalidates_cache_and_repeating_it_reuses(project, changed):
    pid, _, transcript, calls = project
    baseline = run(pid)
    modified = run(pid, **changed)
    repeated = run(pid, **changed)
    assert len(calls) == 2
    assert baseline['artifact_dir'] != modified['artifact_dir']
    assert repeated['artifact_dir'] == modified['artifact_dir']
    assert repeated['reused_stages'] == ['ocr']
    assert modified['ocr']['configuration'] == OCROptions(**changed).model_dump()
    assert service.get_project(pid)['transcript'] == transcript


def test_cache_uses_selected_model_revision_not_unselected_default(project, monkeypatch):
    pid, _, _, calls = project
    first = run(pid, ocr_model='ppocr-v5-mobile', ocr_mode='detail')
    monkeypatch.setitem(models.CATALOG, 'ppocr-v4', {**models.CATALOG['ppocr-v4'], 'revision': 'unselected-revision-changed'})
    unselected_changed = run(pid, ocr_model='ppocr-v5-mobile', ocr_mode='detail')
    assert len(calls) == 1 and unselected_changed['reused_stages'] == ['ocr']
    monkeypatch.setitem(models.CATALOG, 'ppocr-v5-mobile', {**models.CATALOG['ppocr-v5-mobile'], 'revision': 'selected-revision-changed'})
    selected_changed = run(pid, ocr_model='ppocr-v5-mobile', ocr_mode='detail')
    assert len(calls) == 2 and selected_changed['reused_stages'] == []
    assert first['artifact_dir'] != selected_changed['artifact_dir']


def test_source_identity_invalidates_cache_and_stale_result_is_hidden(project):
    pid, path, _, calls = project
    first = run(pid, ocr_mode='detail')
    path.write_bytes(b'Different controlled source, deliberately assigned a new identity')
    source = json.loads(service.project_row(pid)['source'])
    source['sha256'] = checksum(path)
    with db.tx() as connection:
        connection.execute('UPDATE projects SET source=? WHERE id=?', (db.dumps(source), pid))
    assert api.latest_ocr(pid) is None
    changed = run(pid, ocr_mode='detail')
    assert len(calls) == 2 and changed['artifact_dir'] != first['artifact_dir']


def test_caption_edit_during_ocr_survives_and_does_not_invalidate_image_cache(project, monkeypatch):
    pid, _, transcript, calls = project
    original = ocr.extract_text
    corrected = {**transcript, 'segments': [{'start': 0, 'end': 1, 'text': 'Concurrent human correction'}]}
    def read(*args):
        with db.tx() as connection:
            connection.execute('UPDATE projects SET transcript=? WHERE id=?', (db.dumps(corrected), pid))
        return original(*args)
    monkeypatch.setattr(ocr, 'extract_text', read)
    first = run(pid, ocr_model='ppocr-v5-mobile', ocr_mode='detail')
    repeated = run(pid, ocr_model='ppocr-v5-mobile', ocr_mode='detail')
    assert len(calls) == 1 and repeated['artifact_dir'] == first['artifact_dir']
    assert service.get_project(pid)['transcript'] == corrected


@pytest.mark.parametrize('payload', [
    {'ocr_model': '/tmp/model.onnx'}, {'ocr_model': 'ppocr-v5'}, {'ocr_mode': 'guess'},
    {'ocr_quality_samples': True}, {'ocr_quality_samples': 0}, {'ocr_quality_samples': 4},
    {'ocr_quality_samples': '2'}, {'ocr_quality_samples': 1.5},
    {'ocr_sample_seconds': float('nan')}, {'ocr_max_samples': 121}, {'source': '/private/file'},
])
def test_schema_rejects_unknown_models_modes_and_unbounded_quality(payload):
    with pytest.raises(ValueError):
        OCROptions.model_validate(payload)


def test_cli_sends_all_explicit_options_and_keeps_asr_parser_independent(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, 'request', lambda *args: calls.append(args) or {'queued': True})
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'ocr', 'project-id', '--model', 'ppocr-v5-mobile',
                        '--mode', 'detail', '--quality-samples', '3', '--sample-seconds', '7', '--max-samples', '40', '--key', 'key'])
    cli.main()
    assert calls == [('POST', '/projects/project-id/ocr', {'ocr_model': 'ppocr-v5-mobile', 'ocr_mode': 'detail',
                     'ocr_quality_samples': 3, 'ocr_sample_seconds': 7., 'ocr_max_samples': 40}, 'key')]
    assert json.loads(capsys.readouterr().out)['ok']
    defaults = cli.parser().parse_args(['ocr', 'project-id'])
    assert (defaults.model, defaults.mode, defaults.quality_samples) == ('ppocr-v4', 'baseline', 1)
    asr = cli.parser().parse_args(['analyze', 'project-id', '--model', 'tiny', '--language', 'pt'])
    assert asr.model == 'tiny' and asr.language == 'pt'
