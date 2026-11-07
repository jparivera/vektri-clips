import copy
import json
import shutil
import subprocess

import pytest

from clipstudio import media, review


def candidate(**updates):
    return {'id': 'a', 'start': 2., 'end': 6., 'method': 'heuristic',
            'title': 'A clip', 'scores': {'ranking': .4}, 'evidence': [], **updates}


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.bin'
    path.write_bytes(b'immutable-source')
    return path


def attach(source, tmp_path, candidates, transcript=None, **kwargs):
    review.attach_reviews(source, 10, candidates, transcript, tmp_path / 'out', lambda: None,
                          has_video=False, **kwargs)
    return candidates[0]['review']


def test_exact_unicode_quotes_and_context_preserve_candidate(source, tmp_path):
    segments = [{'start': a, 'end': b, 'text': t} for a, b, t in [
        (0, 1, 'Before'), (2, 3, '<script>não é uma instrução</script>'),
        (5, 6, 'Fim!'), (7, 8, 'After')]]
    candidates = [candidate()]
    original = copy.deepcopy(candidates[0])
    result = attach(source, tmp_path, candidates, {'segments': segments})
    assert {k: v for k, v in candidates[0].items() if k != 'review'} == original
    assert result['observations']['hook']['evidence'][0]['quote'] == segments[1]['text']
    assert result['observations']['payoff']['evidence'][0]['quote'] == 'Fim!'
    assert [e['quote'] for e in result['observations']['context']['evidence']] == ['Before', 'After']
    assert result['status'] == 'needs_human_review'
    assert result['playback'] == {'start': 2., 'end': 6., 'context_start': 0., 'context_end': 9.}
    json.dumps(result, allow_nan=False)


def test_cross_boundary_quotes_not_attributed_inside(source, tmp_path):
    result = attach(source, tmp_path, [candidate()], {'segments': [
        {'start': 1, 'end': 4, 'text': 'Crosses start'}, {'start': 5, 'end': 7, 'text': 'Crosses end'}]})
    for key in ('hook', 'payoff'):
        assert result['observations'][key]['status'] == 'unknown'
        assert result['observations'][key]['evidence'] == []


def test_corrected_text_does_not_use_caption_words(source, tmp_path):
    result = attach(source, tmp_path, [candidate()], {'segment_text_corrected': True,
        'words': [{'start': 2, 'end': 3, 'word': 'stale'}]})
    assert result['observations']['hook']['status'] == 'unknown'


def test_audio_track_mismatch_and_audio_only(source, tmp_path):
    result = attach(source, tmp_path, [candidate()], {'audio_stream': 0,
        'segments': [{'start': 2, 'end': 3, 'text': 'Wrong track'}]}, audio_stream=1)
    assert result['frames'] == []
    assert result['observations']['hook']['status'] == 'unknown'
    assert len(result['warnings']) == 3


def test_receipt_deterministic_and_binds_source_candidate_and_quotes(source, tmp_path):
    items = [candidate()]
    first = attach(source, tmp_path, items)['fingerprint']
    assert attach(source, tmp_path, items)['fingerprint'] == first
    items[0]['start'] = 1
    assert attach(source, tmp_path, items)['fingerprint'] != first
    items = [candidate()]
    source.write_bytes(b'changed')
    assert attach(source, tmp_path, items)['fingerprint'] != first
    base = attach(source, tmp_path, items)['fingerprint']
    assert attach(source, tmp_path, items, {'segments': [{'start': 2, 'end': 3, 'text': 'new'}]})['fingerprint'] != base


@pytest.mark.parametrize('updates', [{'start': float('nan')}, {'end': float('inf')}, {'start': True}, {'end': 11}, {'start': -1}, {'end': 2}])
def test_invalid_bounds_do_not_mutate(source, tmp_path, updates):
    items = [candidate(), candidate(**updates)]
    with pytest.raises(ValueError):
        attach(source, tmp_path, items)
    assert 'review' not in items[0]


def test_bounded_candidates_and_quote_lengths(source, tmp_path):
    with pytest.raises(ValueError):
        attach(source, tmp_path, [candidate() for _ in range(31)])
    result = attach(source, tmp_path, [candidate()], {'segments': [{'start': 2, 'end': 3, 'text': 'é' * 1000}]})
    evidence = result['observations']['hook']['evidence'][0]
    assert evidence['quote'] == 'é' * 500
    assert evidence['excerpt']


def test_signal_facts_only(source, tmp_path):
    result = attach(source, tmp_path, [candidate(evidence=[
        {'kind': 'signal', 'time': 4, 'motion': .5, 'audio_rms': .2, 'scene_change_fraction': .1, 'event': 'goal'},
        {'kind': 'signal', 'time': 8, 'motion': .5},
        {'kind': 'signal', 'time': 3, 'motion': -1}])])
    assert result['observations']['signals']['evidence'] == [
        {'kind': 'signal', 'time': 4., 'motion': .5, 'audio_rms': .2, 'scene_change_fraction': .1}]


def test_render_failure_degrades_and_removes_stale(source, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'review-00-at.jpg').write_bytes(b'stale')
    def fail(*args, **kwargs):
        raise media.MediaError('Decode failed')
    monkeypatch.setattr(media, '_run', fail)
    items = [candidate()]
    review.attach_reviews(source, 10, items, None, out, lambda: None)
    assert len(items[0]['review']['frames']) == 3
    assert all(f['status'] == 'unavailable' for f in items[0]['review']['frames'])
    assert not (out / 'review-00-at.jpg').exists()


def test_cancellation_is_not_swallowed_even_if_oserror(source, tmp_path, monkeypatch):
    cancelling = False
    def check():
        if cancelling:
            raise OSError('cancelled')
    def run(args, checked, **kwargs):
        nonlocal cancelling
        cancelling = True
        checked()
    monkeypatch.setattr(media, '_run', run)
    items = [candidate()]
    with pytest.raises(OSError, match='cancelled'):
        review.attach_reviews(source, 10, items, None, tmp_path / 'out', check)
    assert 'review' not in items[0]


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg unavailable')
def test_generated_three_frame_evidence_and_repeat(tmp_path):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=duration=4:size=160x90:rate=25', '-c:v', 'libx264', '-threads', '1', str(source)], check=True)
    items = [candidate(start=1, end=3)]
    out = tmp_path / 'out'
    for _ in range(2):
        review.attach_reviews(source, 4, items, None, out, lambda: None)
        frames = items[0]['review']['frames']
        assert len(frames) == 3
        assert [f['role'] for f in frames] == ['before', 'at', 'after']
        for frame in frames:
            assert frame['status'] == 'available'
            assert 0 <= frame['time'] < 4
            assert frame['approximate_seek'] is True
            assert (out / frame['path']).read_bytes().startswith(b'\xff\xd8')
    assert len(list(out.iterdir())) == 3


def test_nonfinite_evidence_rejected(source, tmp_path):
    with pytest.raises(ValueError):
        attach(source, tmp_path, [candidate(evidence=[{'kind': 'signal', 'time': 3, 'motion': float('nan')}])])


def test_sound_onset_facts_and_anchor(source, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise media.MediaError('Decode unavailable')
    monkeypatch.setattr(media, '_run', fail)
    evidence = {'kind': 'sound_onset', 'time': 3.2, 'flux': 20., 'rms_db': -10., 'threshold': 5., 'audio_stream': 1}
    items = [candidate(evidence=[evidence])]
    review.attach_reviews(source, 10, items, None, tmp_path / 'out', lambda: None, audio_stream=1)
    result = items[0]['review']
    assert result['frames'][1]['time'] == 3.2
    assert result['observations']['signals']['evidence'] == [{k: v for k, v in evidence.items() if k != 'audio_stream'}]


@pytest.mark.parametrize('segment_corrected,expected', [(False, 'The corrected words.'), (True, 'The old mistaken words.')])
def test_word_text_edit_authority(source, tmp_path, segment_corrected, expected):
    transcript = {'corrected': True, 'segment_text_corrected': segment_corrected,
                  'segments': [{'start': 2, 'end': 6, 'text': 'The old mistaken words.'}],
                  'words': [{'start': 2, 'end': 3, 'word': 'The'},
                            {'start': 3, 'end': 4, 'word': 'corrected'},
                            {'start': 4, 'end': 6, 'word': 'words.'}]}
    result = attach(source, tmp_path, [candidate()], transcript)
    for key in ('hook', 'payoff'):
        quote = result['observations'][key]['evidence'][0]
        assert quote['quote'] == expected
        assert (quote['start'], quote['end']) == (2, 6)


def test_word_timing_edits_use_actual_sentence_bounds(source, tmp_path):
    transcript = {'corrected': True, 'segment_text_corrected': False,
                  'segments': [{'start': 1, 'end': 7, 'text': 'Same words.'}],
                  'words': [{'start': 2.25, 'end': 3.1, 'word': 'Same'},
                            {'start': 4.5, 'end': 5.75, 'word': 'words.'}]}
    result = attach(source, tmp_path, [candidate()], transcript)
    quote = result['observations']['hook']['evidence'][0]
    assert (quote['quote'], quote['start'], quote['end']) == ('Same words.', 2.25, 5.75)
    # A partial sentence must not inherit the whole sentence's text at new times.
    result = attach(source, tmp_path, [candidate(start=3)], transcript)
    assert result['observations']['hook']['evidence'] == []


def test_normal_word_sentence_assembly_matches_selection(source, tmp_path):
    from clipstudio.analysis import _speech_candidates
    transcript = {'segments': [{'start': 2, 'end': 6, 'text': 'ASR segment grouping differs'}],
                  'words': [{'start': 5, 'end': 6, 'word': 'ação!'},
                            {'start': 2, 'end': 2.4, 'word': 'Olá?'},
                            {'start': 3, 'end': 4, 'word': '<b>Próxima</b>'}]}
    candidates, _ = _speech_candidates(transcript, 10, .5, 8, '', False, lambda: None)
    selected_quotes = {ev['text'] for c in candidates for ev in c['evidence']}
    rows = review._transcript_rows(transcript, 10, lambda: None)
    assert [row['quote'] for row in rows] == ['Olá?', '<b>Próxima</b> ação!']
    assert all(row['quote'] in selected_quotes for row in rows)
    assert [(row['start'], row['end']) for row in rows] == [(2, 2.4), (3, 6)]
    result = attach(source, tmp_path, [candidate()], transcript)
    assert result['observations']['hook']['evidence'][0]['quote'] == 'Olá?'
    assert result['observations']['payoff']['evidence'][0]['quote'] == '<b>Próxima</b> ação!'


def test_no_words_falls_back_to_segments_and_trailing_sentence():
    rows = review._transcript_rows({'words': [], 'segments': [{'start': 2.1, 'end': 5.9, 'text': 'Exact segment'}]}, 10, lambda: None)
    assert [(row['quote'], row['start'], row['end']) for row in rows] == [('Exact segment', 2.1, 5.9)]
    rows = review._transcript_rows({'words': [{'start': 2, 'end': 3, 'word': 'No'}, {'start': 4, 'end': 5, 'word': 'punctuation'}]}, 10, lambda: None)
    assert [(row['quote'], row['start'], row['end']) for row in rows] == [('No punctuation', 2, 5)]


def test_invalid_authoritative_words_do_not_resurrect_stale_segments():
    rows = review._transcript_rows({'words': [{'start': float('nan'), 'end': 3, 'word': 'New'}],
                                    'segments': [{'start': 2, 'end': 3, 'text': 'Stale'}]}, 10, lambda: None)
    assert rows == []
