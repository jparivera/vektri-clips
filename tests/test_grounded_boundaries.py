"""Source-end rounding regressions with actual fractional-duration media."""
import copy
import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from clipstudio import analysis, review, vision
from clipstudio.schemas import Timeline


@pytest.fixture
def fractional_audio(tmp_path):
    path = tmp_path / 'fractional.wav'
    samples = (.1 * np.sin(2 * np.pi * 1000 * np.arange(96012) / 16000) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    return path


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg unavailable')
@pytest.mark.parametrize('genre', ['speech', 'action'])
def test_actual_fractional_source_rounding_does_not_fail_analysis(fractional_audio, tmp_path, genre):
    duration, _, _ = analysis._metadata(fractional_audio)
    assert duration == pytest.approx(6.00075, abs=.000001)
    transcript = {'segments': [{'start': 0., 'end': 6.,
                               'text': 'This is a complete and useful source sentence.'}]}
    result = analysis.analyze(fractional_audio, {}, transcript,
        {'genre': genre, 'mode': 'heuristic', 'min_length': 5, 'max_length': 30,
         'target_count': 1}, tmp_path / 'analysis', lambda: None)
    assert result['status'] == 'succeeded'
    candidate = result['candidates'][0]
    assert candidate['end'] == 6.001  # Preserve existing selector timing/identity.
    playback = candidate['review']['playback']
    assert playback['end'] == duration
    assert playback['context_end'] == duration
    assert any('rounded' in warning for warning in candidate['review']['warnings'])
    Timeline.model_validate({'segments': [
        {key: candidate[key] for key in ('id', 'start', 'end')}]}).validate_source(duration)
    assert json.loads((tmp_path / 'analysis/analysis.json').read_text()) == result


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg unavailable')
def test_visual_routing_rounded_candidate_has_bounded_real_frames(tmp_path, monkeypatch):
    source = tmp_path / 'fractional.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x90:rate=24', '-frames:v', '145',
                    '-c:v', 'libx264', '-threads', '1', str(source)], check=True)
    duration, _, _ = analysis._metadata(source)
    assert round(duration, 3) > duration
    proposed = {'start': 0., 'end': round(duration, 3), 'title': 'Controlled visual match',
                'rationale': 'Routing control only; no visual model ran in this test.',
                'method': 'local_clip_visual', 'scores': {'ranking': .4},
                'evidence': [{'kind': 'visual', 'time': 3., 'cosine': .4}]}
    monkeypatch.setattr(vision, 'visual_candidates',
                        lambda *args: ([proposed], {'frames_scored': 1}, ['Controlled routing test']))
    result = analysis.analyze(source, {}, None,
        {'genre': 'action', 'visual_search': True, 'query': 'test image', 'target_count': 1},
        tmp_path / 'analysis', lambda: None)
    candidate = result['candidates'][0]
    assert candidate['end'] == round(duration, 3)
    assert candidate['review']['playback']['end'] == duration
    assert len(candidate['review']['frames']) == 3
    for frame in candidate['review']['frames']:
        assert 0 <= frame['time'] < duration
        assert frame['status'] == 'available'
        assert (tmp_path / 'analysis' / frame['path']).read_bytes().startswith(b'\xff\xd8')


@pytest.mark.parametrize('end', [6.00075, 6.001, 6.00175])
def test_review_tolerates_only_existing_source_end_slack(fractional_audio, tmp_path, end):
    candidate = {'id': 'rounded', 'start': 0., 'end': end, 'evidence': []}
    original = copy.deepcopy(candidate)
    review.attach_reviews(fractional_audio, 6.00075, [candidate], None,
                          tmp_path / 'out', lambda: None, has_video=False)
    assert {k: v for k, v in candidate.items() if k != 'review'} == original
    assert candidate['review']['playback']['end'] == 6.00075


@pytest.mark.parametrize('start,end', [(0., 6.001751), (6.00075, 6.001), (6.001, 6.0015)])
def test_genuinely_outside_source_still_fails_before_mutation(fractional_audio, tmp_path, start, end):
    candidates = [{'id': 'valid', 'start': 0., 'end': 6.},
                  {'id': 'outside', 'start': start, 'end': end}]
    with pytest.raises(ValueError, match='outside source bounds'):
        review.attach_reviews(fractional_audio, 6.00075, candidates, None,
                              tmp_path / 'out', lambda: None, has_video=False)
    assert all('review' not in candidate for candidate in candidates)


@pytest.mark.parametrize('segment_corrected', [False, True])
def test_rounded_transcript_end_preserves_exact_supplied_evidence(fractional_audio, tmp_path, segment_corrected):
    candidate = {'id': 'rounded', 'start': 0., 'end': 6.001, 'evidence': []}
    transcript = {'segment_text_corrected': segment_corrected,
                  'segments': [{'start': 0., 'end': 6.001, 'text': 'Corrected segment.'}],
                  'words': [{'start': 0., 'end': 3., 'word': 'Earlier.'},
                            {'start': 3., 'end': 6.001, 'word': 'Ending.'}]}
    review.attach_reviews(fractional_audio, 6.00075, [candidate], transcript,
                          tmp_path / 'out', lambda: None, has_video=False)
    receipt = candidate['review']
    payoff = receipt['observations']['payoff']['evidence'][0]
    assert payoff['end'] == 6.001  # Supplied timestamp, not an invented replacement.
    assert payoff['quote'] == ('Corrected segment.' if segment_corrected else 'Ending.')
    assert receipt['playback']['end'] == 6.00075
