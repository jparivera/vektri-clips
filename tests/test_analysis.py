"""Real local-model integration tests plus separately identified control-path tests."""
import io
from copy import deepcopy
import json
from pathlib import Path
import re
import os
import signal
import subprocess
import sys
import time

import pytest

from clipstudio import analysis, models

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'


def check():
    return None


def need_model(name):
    try:
        return models.model_path(name, verify=True)
    except models.ModelUnavailable:
        pytest.skip(f'Real {name} model absent; no simulated model substituted')


@pytest.fixture(scope='module')
def actual_transcript(tmp_path_factory):
    need_model('tiny.en')
    return analysis.transcribe(FIXTURES / 'spoken_interview.mp4', tmp_path_factory.mktemp('real-asr'), {}, check)


def test_real_asr_audio_words_and_timing(actual_transcript):
    result = actual_transcript
    assert result['status'] == 'succeeded'
    text = ' '.join(segment['text'] for segment in result['segments']).lower()
    assert 'original video' in text and 'context' in text
    assert len(result['words']) >= 30
    assert all(0 <= word['start'] < word['end'] <= 24 for word in result['words'])
    assert all(a['end'] <= b['start'] for a, b in zip(result['words'], result['words'][1:]))
    assert abs(result['words'][0]['start'] - 1) < 0.75
    assert abs(result['words'][-1]['end'] - 18.835) < 0.75


def test_real_embedding_semantic_retrieval():
    need_model('minilm')
    model = analysis.MiniLM()
    vectors = model.encode(['How do I preserve source footage?', 'Keep the original video safe and save edits separately.', 'Bananas grow in a warm climate.'], check)
    similarity = vectors[1:] @ vectors[0]
    assert float(similarity[0]) > float(similarity[1]) + 0.15


def test_real_ai_selection_preserves_question_answer(actual_transcript, tmp_path):
    need_model('minilm')
    result = analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, actual_transcript,
                              {'genre': 'speech', 'mode': 'local', 'query': 'Why keep the original video safe?',
                               'min_length': 4, 'max_length': 20, 'target_count': 3}, tmp_path, check)
    assert result['ai_used'] is True
    assert result['speech_search']['scoring_complete'] is True
    assert result['speech_search']['windows_scored'] > 0
    assert result['speech_search']['elapsed_seconds'] >= 0
    assert len(result['candidates']) == 1
    candidate = result['candidates'][0]
    assert candidate['start'] < 1 and candidate['end'] > 18
    assert candidate['scores']['question_answer_pair'] == 1
    assert candidate['method'] == 'local_ai_semantic'
    assert candidate['evidence']


def test_speech_windows_do_not_start_mid_asr_sentence(actual_transcript, tmp_path):
    result = analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, actual_transcript,
                              {'genre': 'speech', 'mode': 'heuristic', 'min_length': 4, 'max_length': 10}, tmp_path, check)
    assert result['candidates']
    assert all(not candidate['title'].lower().startswith('a sentence') for candidate in result['candidates'])
    assert all(candidate['end'] - candidate['start'] <= 10.001 for candidate in result['candidates'])


def test_corrected_segments_ground_real_semantic_query_and_preserve_word_captions(tmp_path):
    need_model('minilm')
    # These are explicit editable test records, not a claimed ASR result.
    safe_text = 'Keep the original video safe and store a separate backup of your source footage.'
    unrelated_text = 'Bananas grow in a warm climate and need plenty of sunshine.'
    transcript = {
        'segment_text_corrected': True,
        'segments': [{'id': 'corrected-0', 'start': 2, 'end': 8, 'text': safe_text},
                     {'id': 'corrected-1', 'start': 12, 'end': 18, 'text': unrelated_text}],
        'words': [{'start': 2, 'end': 8, 'word': unrelated_text},
                  {'start': 12, 'end': 18, 'word': safe_text}],
    }
    original = deepcopy(transcript)
    options = {'genre': 'speech', 'mode': 'local', 'query': 'How should I preserve the original source video?',
               'min_length': 4, 'max_length': 7, 'target_count': 1}
    corrected = analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, transcript, options, tmp_path / 'corrected', check)
    candidate = corrected['candidates'][0]
    assert corrected['ai_used'] is True and candidate['start'] < 3
    assert candidate['evidence'][0]['text'] == safe_text
    assert candidate['evidence'][0]['ref'] == 'corrected-0'
    assert 'original video' in candidate['title']
    assert transcript == original, 'Semantic selection must not rewrite independent caption words/timing'
    # Explicitly switching authority back to words must use the edited word text.
    transcript['segment_text_corrected'] = False
    word_based = analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, transcript, options, tmp_path / 'word-based', check)
    assert word_based['candidates'][0]['start'] > 11


def test_cleared_corrected_segments_do_not_fall_back_to_stale_caption_words():
    candidates, limitations = analysis._speech_candidates(
        {'segment_text_corrected': True, 'segments': [], 'words': [{'start': 1, 'end': 8, 'word': 'Stale caption text.'}]},
        24, 4, 10, '', False, check)
    assert candidates == []


def test_missing_audio_valid_empty_without_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'model_path', lambda *args, **kwargs: pytest.fail('No-audio source must not load a model'))
    result = analysis.transcribe(FIXTURES / 'portrait — café 日本語.mp4', tmp_path, {}, check)
    assert result['status'] == 'no_audio'
    assert result['words'] == result['segments'] == []


def test_nonexistent_audio_stream_rejected_even_without_audio(tmp_path):
    with pytest.raises(ValueError, match='source has 0 audio'):
        analysis.transcribe(FIXTURES / 'portrait — café 日本語.mp4', tmp_path, {'audio_stream': 1}, check)
    with pytest.raises(ValueError, match='source has 1 audio'):
        analysis.transcribe(FIXTURES / 'spoken_interview.mp4', tmp_path, {'audio_stream': 1}, check)


@pytest.mark.parametrize('index', [-1, 0.5, True, '1'])
def test_audio_stream_index_is_explicit_nonnegative_integer(index):
    with pytest.raises(ValueError, match='nonnegative integer'):
        analysis._audio_stream({'audio_stream': index}, 2)


def test_real_second_audio_stream_transcription_and_signal_selection(tmp_path):
    need_model('tiny.en')
    source = tmp_path / 'two-audio-streams.mp4'
    # Audio track 0 is silence; track 1 starts 2 seconds later than the video.
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-threads', '1',
                    '-i', str(FIXTURES / 'spoken_interview.mp4'), '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                    '-itsoffset', '2', '-i', str(FIXTURES / 'spoken_interview.mp4'),
                    '-map', '0:v:0', '-map', '1:a:0', '-map', '2:a:0', '-c:v', 'copy', '-c:a', 'aac', '-threads', '1', '-t', '24', str(source)], check=True)
    result = analysis.transcribe(source, tmp_path / 'asr', {'audio_stream': 1}, check)
    text = ' '.join(item['text'] for item in result['segments']).lower()
    assert 'original video' in text and 'context' in text
    assert result['audio_stream'] == result['configuration']['audio_stream'] == 1
    assert abs(result['words'][0]['start'] - 3) < 0.75, 'ASR must retain the selected track\'s delayed start'
    with pytest.raises(ValueError, match='different audio stream'):
        analysis.analyze(source, {}, result, {'genre': 'speech', 'mode': 'heuristic', 'audio_stream': 0}, tmp_path / 'wrong-transcript', check)
    silent = analysis.analyze(source, {}, None, {'genre': 'action', 'mode': 'heuristic', 'audio_stream': 0}, tmp_path / 'silent', check)
    spoken = analysis.analyze(source, {}, None, {'genre': 'action', 'mode': 'heuristic', 'audio_stream': 1}, tmp_path / 'spoken', check)
    silent_signals = json.loads((tmp_path / 'silent/signals.json').read_text())
    spoken_signals = json.loads((tmp_path / 'spoken/signals.json').read_text())
    assert max(item['audio_rms'] for item in silent_signals['signals']) < 0.0001
    assert max(item['audio_rms'] for item in spoken_signals['signals']) > 0.02
    assert max(item['audio_rms'] for item in spoken_signals['signals'][:4]) < 0.0001
    assert silent['configuration']['audio_stream'] == 0 and spoken['configuration']['audio_stream'] == 1
    with pytest.raises(ValueError, match='source has 2 audio'):
        analysis.analyze(source, {}, None, {'genre': 'action', 'audio_stream': 2}, tmp_path / 'invalid', check)


def test_real_action_evidence_constraints_and_no_semantic_claims(tmp_path):
    result = analysis.analyze(FIXTURES / 'action_motion.mp4', {}, None,
                              {'genre': 'action', 'mode': 'local', 'min_length': 4, 'max_length': 5, 'target_count': 8,
                               'min_separation': 1, 'exclusions': [{'start': 0, 'end': 5}]}, tmp_path, check)
    assert result['candidates']
    assert result['ai_used'] is False
    for candidate in result['candidates']:
        assert candidate['start'] >= 5 and 4 <= candidate['end'] - candidate['start'] <= 5.001
        assert candidate['method'] == 'heuristic_visual_audio'
        assert not any(word in candidate['title'].lower() for word in ('goal', 'crash', 'winner'))
    ordered = sorted(result['candidates'], key=lambda item: item['start'])
    assert all(b['start'] >= a['end'] + 1 for a, b in zip(ordered, ordered[1:]))
    assert len(result['candidates']) < 8
    assert (tmp_path / 'signals.json').exists()


def test_exclusions_not_filled_with_bad_duplicates(actual_transcript, tmp_path):
    result = analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, actual_transcript,
                              {'genre': 'speech', 'mode': 'heuristic', 'min_length': 4, 'max_length': 20,
                               'exclusions': [{'start': 0, 'end': 24}]}, tmp_path, check)
    assert result['status'] == 'no_candidates'
    assert result['candidates'] == []


@pytest.mark.parametrize('mode', ['agent', 'cloud'])
def test_unconfigured_modes_fail_explicitly(tmp_path, mode):
    with pytest.raises((ValueError, models.ModelUnavailable), match='configured|packet'):
        analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, None, {'mode': mode}, tmp_path, check)
    assert not (tmp_path / 'analysis.json').exists()


def test_analysis_cancellation_before_work(tmp_path):
    def cancelled():
        raise RuntimeError('cancel requested')
    with pytest.raises(RuntimeError, match='cancel requested'):
        analysis.transcribe(FIXTURES / 'spoken_interview.mp4', tmp_path, {}, cancelled)
    assert not (tmp_path / 'transcript.json').exists()


def test_lost_model_not_silently_replaced_by_heuristic(actual_transcript, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise models.ModelUnavailable('model lost')
    monkeypatch.setattr(analysis, 'model_path', unavailable)
    with pytest.raises(models.ModelUnavailable, match='model lost'):
        analysis.analyze(FIXTURES / 'spoken_interview.mp4', {}, actual_transcript,
                         {'genre': 'speech', 'mode': 'local', 'max_length': 20}, tmp_path, check)


def test_download_cancellation_atomicity_control_path(tmp_path, monkeypatch):
    # This deliberately mocked HTTP body tests cancellation, not model inference.
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setitem(models.CATALOG, 'test-model', {'repository': 'example/test', 'revision': 'pinned', 'license': 'test',
                                                    'files': {'model': (1_000_000, 'unreachable')}})
    monkeypatch.setattr(models.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'1' * 1_000_000))
    calls = 0
    def cancel_during_stream():
        nonlocal calls
        calls += 1
        if calls > 3:
            raise RuntimeError('cancel requested')
    with pytest.raises(RuntimeError, match='cancel requested'):
        models.download_model('test-model', tmp_path / 'out', cancel_during_stream)
    assert not (tmp_path / 'data/models/test-model/pinned').exists()
    assert not list((tmp_path / 'data/models').glob('.test-model-*'))


def test_download_rejects_checksum_mismatch_control_path(tmp_path, monkeypatch):
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setitem(models.CATALOG, 'test-model', {'repository': 'example/test', 'revision': 'pinned', 'license': 'test', 'files': {'model': (3, 'bad')}})
    monkeypatch.setattr(models.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'abc'))
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        models.download_model('test-model', tmp_path / 'out', check)
    assert not (tmp_path / 'data/models/test-model/pinned').exists()


@pytest.mark.parametrize('options', [{'min_length': float('nan')}, {'target_count': 0}, {'min_length': 10, 'max_length': 5}, {'exclusions': [{'start': -1, 'end': 2}]}])
def test_invalid_analysis_constraints(options):
    with pytest.raises(ValueError):
        analysis._options(options, 24)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux parent-death protection; POSIX process-group cancellation tested by worker integration')
def test_analysis_child_terminated_when_stage_is_sigkilled(tmp_path):
    """Real process lifecycle, no inference mock and no unrelated process signals."""
    import psutil
    pid_file = tmp_path / 'child.pid'
    child_code = 'import os,time;from pathlib import Path;Path(' + repr(str(pid_file)) + ').write_text(str(os.getpid()));time.sleep(30)'
    stage_code = 'from clipstudio.analysis import _run;import sys;_run([sys.executable,"-c",' + repr(child_code) + '],lambda:None)'
    parent = subprocess.Popen([sys.executable, '-c', stage_code], env={**os.environ, 'PYTHONPATH': str(ROOT / 'backend')},
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                raise AssertionError(parent.stderr.read().decode())
            time.sleep(0.02)
        assert pid_file.exists(), 'Controlled child did not start'
        child_pid = int(pid_file.read_text())
        assert os.getpgid(child_pid) == parent.pid, 'Child must inherit its stage process group'
        parent.kill()
        parent.wait(timeout=3)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.02)
        else:
            pytest.fail('Analysis child survived stage SIGKILL')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        if child_pid:
            try:
                process = psutil.Process(child_pid)
                if process.status() != psutil.STATUS_ZOMBIE:
                    process.kill()
            except psutil.NoSuchProcess:
                pass


def long_transcript(count=1700):
    return {'segments': [
        {'id': f'long-{index}', 'start': index * 3, 'end': index * 3 + 2.8,
         'text': 'Bananas grow best in a warm tropical climate.'}
        for index in range(count)
    ]}


def test_full_source_heuristic_ranking_reaches_late_query_and_bounds_shortlist():
    transcript = long_transcript()
    transcript['segments'][-1]['text'] = 'zebramarker appears in the final explanation.'
    original = deepcopy(transcript)
    candidates, limitations = analysis._speech_candidates(transcript, 5100, 2.5, 3, 'zebramarker', False, check)
    selected = analysis._select(candidates, 1, 0, [])
    assert selected[0]['start'] > 5090
    assert selected[0]['evidence'][0]['ref'] == 'long-1699'
    assert len(candidates) <= analysis.SPEECH_GLOBAL_SHORTLIST + 85 * analysis.SPEECH_PER_MINUTE
    assert any('Scored 1700' in note and 'full source' in note for note in limitations)
    assert transcript == original


def test_full_source_real_semantic_ranking_reaches_late_answer_with_bounded_batches(monkeypatch):
    need_model('minilm')
    transcript = long_transcript()
    transcript['segments'][-1]['text'] = 'Preserve your original source video and save a separate backup before editing.'
    batch_sizes = []
    original_encode = analysis.MiniLM.encode

    def observe_encode(self, texts, callback):
        batch_sizes.append(len(texts))
        return original_encode(self, texts, callback)

    monkeypatch.setattr(analysis.MiniLM, 'encode', observe_encode)
    candidates, limitations = analysis._speech_candidates(
        transcript, 5100, 2.5, 3, 'How should I protect the original footage?', True, check)
    selected = analysis._select(candidates, 1, 0, [])
    assert selected[0]['start'] > 5090
    assert selected[0]['method'] == 'local_ai_semantic'
    assert selected[0]['evidence'][0]['ref'] == 'long-1699'
    assert sum(batch_sizes) == 1701, 'Every window plus the query must use real embeddings'
    assert max(batch_sizes) <= analysis.SPEECH_BATCH_SIZE
    assert any('Scored 1700' in note for note in limitations)


def test_exclusions_apply_before_shortlist_and_preserve_eligible_late_material():
    transcript = long_transcript()
    for item in transcript['segments'][:-10]:
        item['text'] = 'specialanswer specialanswer specialanswer.'
    transcript['segments'][-1]['text'] = 'An eligible final specialanswer.'
    excluded = [{'start': 0, 'end': 5070}]
    candidates, limitations = analysis._speech_candidates(
        transcript, 5100, 2.5, 3, 'specialanswer', False, check, excluded)
    assert candidates
    assert all(candidate['start'] >= 5070 for candidate in candidates)
    assert any(candidate['evidence'][-1]['ref'] == 'long-1699' for candidate in candidates)
    assert any('Scored 9' in note for note in limitations), 'Padding overlap must honor exclusions too'


def test_streamed_speech_analysis_checks_cancellation_after_old_window_cutoff():
    transcript = long_transcript()
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls > 7000:
            raise RuntimeError('late cancellation')

    with pytest.raises(RuntimeError, match='late cancellation'):
        analysis._speech_candidates(transcript, 5100, 2.5, 3, '', False, cancel)


def test_real_asr_retains_speech_after_sixty_second_chunk_boundary(tmp_path):
    need_model('tiny.en')
    source = tmp_path / 'long-synthetic-speech.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-stream_loop', '2',
                    '-i', str(FIXTURES / 'spoken_interview.mp4'), '-t', '67', '-c', 'copy', str(source)], check=True)
    result = analysis.transcribe(source, tmp_path / 'asr', {}, check)
    assert result['status'] == 'succeeded'
    assert result['configuration']['chunk_seconds'] == 60
    assert any(word['start'] >= 60 for word in result['words'])
    assert result['words'][-1]['end'] > 64
    assert all(a['end'] <= b['start'] for a, b in zip(result['words'], result['words'][1:]))
    assert not (tmp_path / 'asr/transcript.partial.json').exists()


@pytest.mark.parametrize('duration', [7201, 43200])
def test_analysis_accepts_import_supported_long_source_duration(monkeypatch, duration):
    # Metadata boundary only; this does not claim running a 12-hour media job.
    from types import SimpleNamespace
    probe = {'format': {'duration': str(duration)}, 'streams': [{'codec_type': 'audio'}, {'codec_type': 'video'}]}
    monkeypatch.setattr(analysis.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(probe)))
    assert analysis._metadata(Path('metadata-only.mp4')) == (duration, 1, True)


def test_analysis_rejects_duration_over_import_limit(monkeypatch):
    from types import SimpleNamespace
    probe = {'format': {'duration': '43200.1'}, 'streams': []}
    monkeypatch.setattr(analysis.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(probe)))
    with pytest.raises(ValueError, match='12 hours'):
        analysis._metadata(Path('metadata-only.mp4'))


@pytest.mark.parametrize('near_duplicates', [False, True])
def test_bounded_shortlist_preserves_distinct_lower_density_passages(near_duplicates):
    segments = []
    for index in range(600):
        text = ('This recurring explanation carefully repeats ordinary general context with predictable structure and identical meaning for readers item' + str(index) + '.') if near_duplicates else 'One recurring example discusses ordinary general context again.'
        if index % 20 == 0:
            text = f'Novel{index} result.'
        segments.append({'id': str(index), 'start': index * 3, 'end': index * 3 + 2.8, 'text': text})
    candidates, _ = analysis._speech_candidates({'segments': segments}, 1800, 2.5, 3, '', False, check)
    selected = analysis._select(candidates, 5, 0, [])
    assert len(selected) == 5
    assert len({candidate['title'] for candidate in selected}) == 5
    assert sum(candidate['title'].startswith('Novel') for candidate in selected) >= 4
    assert len(candidates) <= analysis.SPEECH_GLOBAL_SHORTLIST + analysis.SPEECH_DIVERSE_SHORTLIST + 30 * analysis.SPEECH_PER_MINUTE
    # The global distinct fallback also keeps enough alternatives for the maximum
    # requested count; repeated high scores must not crowd them out beforehand.
    thirty = analysis._select(candidates, 30, 0, [])
    assert len(thirty) == 30
    assert sum(candidate['title'].startswith('Novel') for candidate in thirty) >= 29
