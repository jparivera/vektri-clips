"""Routing/cache regressions; real CLIP inference is tested in test_vision.py."""
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from clipstudio import analysis, task
from clipstudio.schemas import AnalysisOptions


@pytest.mark.parametrize('options', [
    {'visual_search': True},
    {'visual_search': True, 'query': '   '},
    {'visual_search': True, 'query': 'cars', 'mode': 'heuristic'},
])
def test_explicit_visual_search_requires_query_and_local_mode(options):
    with pytest.raises(ValidationError, match='Visual search'):
        AnalysisOptions(**options)


def test_visual_search_does_not_mix_transcript_or_motion_scores(tmp_path, monkeypatch):
    from clipstudio import review
    # Keep this a routing-only control; grounded review has real-media tests.
    monkeypatch.setattr(review, 'attach_reviews', lambda *a, **kw: None)
    monkeypatch.setattr(analysis, '_metadata', lambda _: (20, 1, True))
    monkeypatch.setattr(analysis, '_signals', lambda *args: pytest.fail('CLIP search must not mix signal ranking'))
    monkeypatch.setattr(analysis, '_speech_candidates', lambda *args: pytest.fail('CLIP search must not mix text ranking'))
    candidate = {'start': 4, 'end': 8, 'title': 'Visual match', 'rationale': 'Sampled frames',
                 'method': 'local_clip_visual', 'scores': {'ranking': -.1}, 'evidence': [{'time': 5, 'cosine': -.1}]}
    monkeypatch.setitem(sys.modules, 'clipstudio.vision', SimpleNamespace(
        visual_candidates=lambda *args: ([candidate], {'frames_scored': 10, 'scoring_complete': True}, ['Frame similarity only'])))
    transcript = {'language': 'en', 'audio_stream': 7, 'segments': [{'start': 0, 'end': 8, 'text': 'Unrelated words'}]}
    result = analysis.analyze(tmp_path/'unused.mp4', {}, transcript,
                              {'genre': 'mixed', 'visual_search': True, 'query': 'cars', 'target_count': 1}, tmp_path/'out', lambda: None)
    assert result['provider'] == 'local_onnx_clip' and result['ai_used']
    assert result['candidates'][0]['method'] == 'local_clip_visual'
    assert result['candidates'][0]['scores']['ranking'] == -.1
    assert result['speech_search'] is None
    assert result['visual_search']['frames_scored'] == 10
    assert result['paid_usage'] == 0 and result['data_transmitted'] is False


def test_audio_only_visual_search_fails_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, '_metadata', lambda _: (20, 1, False))
    with pytest.raises(ValueError, match='video stream'):
        analysis.analyze(tmp_path/'audio.wav', {}, None, {'visual_search': True, 'query': 'cars'}, tmp_path/'out', lambda: None)
    assert not (tmp_path/'out/analysis.json').exists()


def test_non_english_speech_does_not_use_english_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, '_metadata', lambda _: (20, 1, True))
    seen = []
    def speech(*args):
        seen.append(args[5])
        return [], []
    monkeypatch.setattr(analysis, '_speech_candidates', speech)
    result = analysis.analyze(tmp_path/'unused.mp4', {},
        {'language': 'pt', 'segments': [{'start': 0, 'end': 8, 'text': 'carros na pista'}]},
        {'genre': 'speech', 'mode': 'local', 'query': 'carros'}, tmp_path/'out', lambda: None)
    assert seen == [False]
    assert not result['ai_used']
    assert any('English-only' in item for item in result['limitations'])


def test_visual_implementation_changes_invalidate_analysis_stage(monkeypatch):
    revision = ['old']
    seen = []
    def checksum(path):
        seen.append(path.name)
        return revision[0] if path.name == 'vision.py' else 'unchanged'
    monkeypatch.setattr(task, 'checksum', checksum)
    before = task.implementation_fingerprint('analyze')
    revision[0] = 'new'
    after = task.implementation_fingerprint('analyze')
    assert 'vision.py' in seen and before != after
