"""Offline multilingual ASR and settings safety; optional real-footage smoke check."""
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import pytest
from fastapi.testclient import TestClient

from clipstudio import analysis, config, db, models, runtime, service, task
from clipstudio.api import app, TranscribeRequest
from clipstudio.schemas import AnalysisOptions, default_timeline
from clipstudio.security import checksum

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    directory = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    db.migrate()


def ready_project(transcript):
    pid = service.new_project('ASR settings control', 'source.mp4', {'genre': 'speech'})
    source = config.project_dir(pid) / 'source/original.bin'
    source.write_bytes(b'original bytes for isolated settings control; inference is mocked')
    record = {'path': 'source/original.bin', 'sha256': checksum(source), 'provenance': {},
              'metadata': {'duration': 24, 'audio': True, 'audio_stream_count': 1}}
    timeline = default_timeline(24)
    with db.tx() as connection:
        connection.execute("UPDATE projects SET source=?,revision=1,status='ready',transcript=? WHERE id=?",
                           (db.dumps(record), db.dumps(transcript) if transcript else None, pid))
        connection.execute('INSERT INTO revisions VALUES(?,?,?,?,?)', (pid, 1, db.dumps(timeline), db.fingerprint(timeline), time.time()))
    return pid


def transcript(model='tiny.en', language='en', corrected=False):
    return {'provider': 'faster-whisper', 'model': 'Systran/faster-whisper-' + model,
            'language': language, 'configuration': {'requested_model': model, 'requested_language': language},
            'audio_stream': 0, 'corrected': corrected, 'words': [],
            'segments': [{'id': 'one', 'start': 1, 'end': 4, 'text': 'Original reviewed text.'}]}


def execute_job(kind, pid, options, original):
    job = db.enqueue(kind, {**options, 'base_transcript_hash': db.fingerprint(original) if original else None}, pid)
    owner = db.claim()
    return task.execute(job['id'], owner['lease_token'])


def test_default_stays_english():
    for cls in (AnalysisOptions, TranscribeRequest):
        assert cls().model == 'tiny.en' and cls().language == 'en'
        assert cls(model='tiny', language='pt').language == 'pt'
        assert cls(model='tiny', language='auto').language == 'auto'


@pytest.mark.parametrize('payload', [{'model': 'tiny.en', 'language': 'pt'}, {'model': 'tiny.en', 'language': 'auto'},
                                     {'model': 'tiny', 'language': 'xx'}, {'model': 'base', 'language': 'pt'}])
def test_incompatible_model_language_requests_rejected(isolated, payload):
    with TestClient(app) as client:
        assert client.post('/api/v1/projects/' + '0' * 32 + '/transcribe', json=payload).status_code == 422
        assert client.post('/api/v1/projects/' + '0' * 32 + '/analyze', json=payload).status_code == 422
    assert not db.all('SELECT * FROM jobs')


def test_cached_tiny_is_exact_pinned_revision_without_network(monkeypatch):
    monkeypatch.setattr(models.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('Model discovery must not download'))
    try:
        path = models.model_path('tiny', verify=True)
    except models.ModelUnavailable:
        pytest.skip('Complete pinned multilingual tiny model not installed; no download is attempted')
    assert path.name == models.CATALOG['tiny']['revision']
    assert models.CATALOG['tiny']['license'] == 'MIT'


@pytest.mark.parametrize('changes', [{'model': 'tiny', 'language': 'pt'}, {'model': 'tiny', 'language': 'auto'}])
def test_analysis_cannot_silently_replace_corrections_for_new_language(isolated, monkeypatch, changes):
    original = transcript(corrected=True)
    pid = ready_project(original)
    monkeypatch.setattr(analysis, 'transcribe', lambda *args: pytest.fail('Corrected transcript must not be overwritten implicitly'))
    with pytest.raises(db.Conflict, match='corrected.*language'):
        execute_job('analyze', pid, {'genre': 'speech', **changes}, original)
    assert json.loads(service.project_row(pid)['transcript']) == original


def test_language_change_retranscribes_uncorrected_text_and_keys_stage_options(isolated, monkeypatch):
    original = transcript()
    pid = ready_project(original)
    seen = []
    def infer(_source, _out, options, _check):
        seen.append(options)
        return transcript(options['model'], options['language'])
    monkeypatch.setattr(analysis, 'transcribe', infer)
    monkeypatch.setattr(analysis, 'analyze', lambda *args: {'candidates': [], 'method': 'control'})
    result = execute_job('analyze', pid, {'genre': 'speech', 'model': 'tiny', 'language': 'pt'}, original)
    assert seen == [{'model': 'tiny', 'language': 'pt', 'audio_stream': 0}]
    assert json.loads(service.project_row(pid)['transcript'])['language'] == 'pt'
    assert result['analysis']['method'] == 'control'


def test_explicit_transcription_can_replace_current_correction(isolated, monkeypatch):
    original = transcript(corrected=True)
    pid = ready_project(original)
    monkeypatch.setattr(analysis, 'transcribe', lambda _s, _o, options, _c: transcript(options['model'], options['language']))
    result = execute_job('transcribe', pid, {'model': 'tiny', 'language': 'pt'}, original)
    assert result['transcript']['language'] == 'pt'
    assert json.loads(service.project_row(pid)['transcript'])['corrected'] is False


def test_visual_search_does_not_start_irrelevant_transcription(isolated, monkeypatch):
    pid = ready_project(None)
    monkeypatch.setattr(analysis, 'transcribe', lambda *args: pytest.fail('Visual search must not start ASR'))
    monkeypatch.setattr(analysis, 'analyze', lambda *args: {'candidates': [], 'method': 'visual-control'})
    assert execute_job('analyze', pid, {'genre': 'speech', 'visual_search': True}, None)['analysis']['method'] == 'visual-control'


def test_multilingual_no_audio_reports_no_detected_language(tmp_path):
    result = analysis.transcribe(ROOT / 'fixtures/portrait — café 日本語.mp4', tmp_path, {'model': 'tiny', 'language': 'auto'}, lambda: None)
    assert result['status'] == 'no_audio' and result['language'] is None and result['language_segments'] == []


@pytest.mark.timeout(60)
def test_real_cached_tiny_portuguese_and_auto_on_supplied_recording(tmp_path, monkeypatch):
    source = os.environ.get('VEKTRI_TEST_PORTUGUESE_MEDIA')
    if not source or not Path(source).is_file():
        pytest.skip('Set VEKTRI_TEST_PORTUGUESE_MEDIA to a deliberately selected permitted recording; no media download')
    if psutil.virtual_memory().available < runtime.MIN_AVAILABLE_RAM:
        pytest.skip('At least 1.5 GiB available RAM required; no safety guard bypass')
    try:
        models.model_path('tiny', verify=True)
    except models.ModelUnavailable:
        pytest.skip('Pinned multilingual tiny unavailable; no download')
    monkeypatch.setattr(models.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('Offline ASR must not download'))
    sample = tmp_path / 'first-30s.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', source, '-t', '30',
                    '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(sample)], check=True, timeout=20)
    for language in ('pt', 'auto'):
        result = analysis.transcribe(sample, tmp_path / language, {'model': 'tiny', 'language': language}, lambda: None)
        assert result['status'] == 'succeeded' and result['words']
        assert result['model'] == models.CATALOG['tiny']['repository']
        assert result['revision'] == models.CATALOG['tiny']['revision']
        assert result['configuration']['requested_language'] == language
        assert result['language_segments'] and result['language']
        assert all(0 <= word['start'] < word['end'] <= 30 for word in result['words'])
        if language == 'pt':
            assert result['language'] == 'pt'
        # This checks actual offline inference/metadata, not human transcription accuracy.
