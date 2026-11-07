"""Benchmark controls/metrics only; these tests do not claim real CLIP inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_visual_search as benchmark


def annotation():
    return {'schema_version': 1, 'source_sha256': 'a' * 64,
            'model_scores_observed_before_annotation': False,
            'queries': [{'id': 'frozen', 'query': 'Cars driving on a track',
                         'positive_intervals': [{'start': 12, 'end': 14}],
                         'contrast_intervals': [{'start': 0, 'end': 11}]}]}


def test_frozen_query_and_exact_bytes_are_preserved(tmp_path):
    path = tmp_path / 'annotations.json'
    raw = json.dumps(annotation(), indent=4).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    captured, document, query, actual = benchmark.freeze_annotations(path, 'a' * 64, 'frozen', digest)
    assert captured == raw and actual == digest
    query['query'] = 'changed by caller'
    assert document['queries'][0]['query'] == 'Cars driving on a track'
    assert path.read_bytes() == raw


@pytest.mark.parametrize('change,message', [
    ('source', 'different source'), ('scores', 'not observed'), ('query', 'unique'),
    ('overlap', 'must not overlap'), ('core', 'within their scene'), ('hash', 'pre-frozen'),
])
def test_refuse_invalid_or_post_score_references(tmp_path, change, message):
    value = annotation()
    expected = None
    if change == 'source':
        value['source_sha256'] = 'b' * 64
    elif change == 'scores':
        value['model_scores_observed_before_annotation'] = True
    elif change == 'query':
        value['queries'].append(copy.deepcopy(value['queries'][0]))
    elif change == 'overlap':
        value['queries'][0]['contrast_intervals'][0]['end'] = 13
    elif change == 'core':
        value['queries'][0]['positive_intervals'][0].update(core_start=10, core_end=13)
    elif change == 'hash':
        expected = '0' * 64
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        benchmark.freeze_annotations(path, 'a' * 64, 'frozen', expected)


def test_overlap_separates_unknown_from_explicit_contrasts():
    query = annotation()['queries'][0]
    metric = benchmark.overlap_metrics([{'id': 'first', 'start': 10, 'end': 15}], query)
    top = metric['top_candidate']
    assert top['positive_overlap_seconds'] == 2
    assert top['contrast_overlap_seconds'] == 1
    assert top['unannotated_seconds'] == 2
    assert top['positive_fraction_of_candidate'] == .4
    assert top['positive_reference_fraction_covered'] == 1
    assert metric['positive_intervals_hit'] == 1


def test_overlap_merges_duplicate_labels_and_returned_candidate_coverage():
    query = {'positive_intervals': [{'start': 10, 'end': 20}, {'start': 15, 'end': 25}],
             'contrast_intervals': []}
    result = benchmark.overlap_metrics([{'start': 15, 'end': 20}, {'start': 18, 'end': 25}], query)
    assert result['annotated_positive_seconds'] == 15
    assert result['top_candidate']['positive_overlap_seconds'] == 5
    assert result['all_returned_positive_reference_fraction_covered'] == pytest.approx(10 / 15, abs=1e-6)


def test_empty_candidates_are_reported_without_inventing_a_hit():
    result = benchmark.overlap_metrics([], annotation()['queries'][0])
    assert result['top_candidate'] is None
    assert result['all_returned_positive_reference_fraction_covered'] == 0


def test_optional_safe_core_metric_does_not_change_envelope_metric():
    query = annotation()['queries'][0]
    query['positive_intervals'][0].update(core_start=12.5, core_end=13.5)
    result = benchmark.overlap_metrics([{'start': 12, 'end': 14}], query)
    assert result['top_candidate']['positive_overlap_seconds'] == 2
    assert result['safe_positive_cores']['top_candidate']['positive_overlap_seconds'] == 1


def clip_analysis():
    return {'provider': 'local_onnx_clip', 'ai_used': True, 'model': benchmark.CLIP_REPOSITORY,
            'revision': 'b' * 40, 'visual_search': {'model': benchmark.CLIP_REPOSITORY,
                'revision': 'b' * 40, 'sample_count': 2, 'sampling_complete': True,
                'samples': [{'time': 13, 'cosine_similarity': -.1}, {'time': 5, 'cosine_similarity': -.2}]},
            'candidates': [{'id': 'one', 'start': 10, 'end': 15, 'method': 'local_clip_visual',
                            'evidence': [{'kind': 'visual_frame', 'time': 13, 'cosine_similarity': -.1}]}]}


def test_real_provider_contract_required_and_negative_cosines_preserved():
    result = clip_analysis()
    benchmark.require_clip_analysis(result, 'b' * 40)
    ranked = benchmark.top_frame_metrics(result['visual_search'], annotation()['queries'][0])
    assert ranked['top_frames'][0]['time'] == 13
    assert ranked['top_frames'][0]['cosine_similarity'] == -.1
    assert ranked['counts'] == {'positive': 1, 'contrast': 1, 'unannotated': 0}


@pytest.mark.parametrize('change', ['provider', 'revision', 'ai', 'metadata', 'incomplete', 'nan', 'method'])
def test_reject_fake_or_incomplete_clip_results(change):
    result = clip_analysis()
    if change == 'provider': result['provider'] = 'heuristics'
    elif change == 'revision': result['revision'] = 'c' * 40
    elif change == 'ai': result['ai_used'] = False
    elif change == 'metadata': result['visual_search'] = None
    elif change == 'incomplete': result['visual_search']['sampling_complete'] = False
    elif change == 'nan': result['visual_search']['samples'][0]['cosine_similarity'] = float('nan')
    elif change == 'method': result['candidates'][0]['method'] = 'heuristic_visual_audio'
    with pytest.raises(ValueError):
        benchmark.require_clip_analysis(result, 'b' * 40)


def test_render_uses_top_candidate_exact_timing_without_changing_project():
    project = {'timeline': {'schema_version': 1, 'audio': {'normalize': True},
                           'captions': {'words': [{'start': 1, 'end': 2, 'word': 'old'}]}}}
    original = copy.deepcopy(project)
    candidate = {'id': 'poor-overlap-still-top', 'start': 1.234, 'end': 5.678}
    timeline = benchmark.render_timeline(project, candidate, 'Licensed source')
    assert project == original
    assert [(s['start'], s['end']) for s in timeline['segments']] == [(1.234, 5.678)]
    assert timeline['segments'][0]['crop']['mode'] == 'fit'
    assert timeline['captions']['words'][0]['word'] == 'Licensed source'


def test_poll_transport_failure_cancels_only_owned_job(monkeypatch):
    calls = []
    def stalled(*args):
        raise httpx.ReadTimeout('transport stalled')
    monkeypatch.setattr(benchmark, 'poll', stalled)
    monkeypatch.setattr(benchmark, 'request', lambda client, method, path: calls.append((method, path)) or {'status': 'cancelled'})
    record = {}
    with pytest.raises(httpx.ReadTimeout):
        benchmark.complete_owned_job(None, {'id': 'owned-job'}, record, lambda: None, 5)
    assert calls == [('POST', '/api/v1/jobs/owned-job/cancel')]
    assert record['cancellation_observed_status'] == 'cancelled'


def test_failed_cancellation_is_explicitly_uncertain(monkeypatch):
    monkeypatch.setattr(benchmark, 'poll', lambda *a: (_ for _ in ()).throw(httpx.ReadTimeout('lost')))
    monkeypatch.setattr(benchmark, 'request', lambda *a: (_ for _ in ()).throw(httpx.ConnectError('offline')))
    record = {}
    with pytest.raises(httpx.ReadTimeout):
        benchmark.complete_owned_job(None, {'id': 'owned-job'}, record, lambda: None, 5)
    assert record['id'] == 'owned-job'
    assert record['cancellation_outcome'].startswith('uncertain')


@pytest.mark.parametrize('worker', [
    {'alive': False}, {'alive': True, 'state': 'waiting_memory'},
    {'alive': True, 'state': 'idle', 'workers': [{'available_ram_bytes': 1024}]},
])
def test_worker_readiness_keeps_ram_guard(worker):
    with pytest.raises(RuntimeError):
        benchmark.ensure_ready({'worker_active': False, 'queue_length': 0, 'worker': worker})


def test_accepts_ready_worker_only_when_queue_empty():
    health = {'worker_active': False, 'queue_length': 0, 'worker': {'alive': True, 'state': 'idle',
              'workers': [{'available_ram_bytes': 2 * 1024**3}]}}
    benchmark.ensure_ready(health)
    with pytest.raises(RuntimeError, match='Other jobs'):
        benchmark.ensure_ready({**health, 'queue_length': 1})


def temporal_analysis():
    frames = [{'kind': 'temporal_frame', 'time': 10 + i * .5, 'ref': f'frame-{i}.jpg'} for i in range(8)]
    return {'provider': 'local_onnx_xclip', 'ai_used': True, 'model': benchmark.TEMPORAL_REPOSITORY,
        'revision': 'c' * 40, 'visual_search': {'model': benchmark.TEMPORAL_REPOSITORY, 'revision': 'c' * 40,
            'model_id': 'xclip-b16-k600', 'method': benchmark.TEMPORAL_METHOD, 'learned_temporal_encoder': True,
            'full_classifier': False, 'sampling_complete': True, 'frames_per_window': 8, 'window_count': 1,
            'sample_count': 8, 'windows': [{'start': 10, 'end': 14, 'cosine_similarity': -.3, 'frames': frames}]},
        'candidates': [{'start': 10, 'end': 14, 'method': benchmark.TEMPORAL_METHOD,
                        'scores': {'ranking': -.3}, 'evidence': frames}]}


def test_temporal_contract_and_unmodified_raw_scores():
    analysis = temporal_analysis()
    benchmark.require_temporal_analysis(analysis, 'c' * 40)
    result = benchmark.top_temporal_metrics(analysis['visual_search'], annotation()['queries'][0])
    assert result['unit'] == 'eight-frame temporal windows'
    assert result['top_candidate']['model_scores']['cosine_similarity'] == -.3
    assert result['top_candidate']['positive_overlap_seconds'] == 2


@pytest.mark.parametrize('change', ['provider', 'revision', 'ai', 'incomplete', 'full_classifier', 'reverse', 'outside', 'nan', 'method'])
def test_reject_fake_or_incomplete_temporal_results(change):
    result = temporal_analysis()
    meta = result['visual_search']
    if change == 'provider': result['provider'] = 'local_onnx_clip'
    elif change == 'revision': result['revision'] = 'unverified'
    elif change == 'ai': result['ai_used'] = False
    elif change == 'incomplete': meta['sampling_complete'] = False
    elif change == 'full_classifier': meta['full_classifier'] = True
    elif change == 'reverse': meta['windows'][0]['frames'].reverse()
    elif change == 'outside': meta['windows'][0]['frames'][0]['time'] = 1
    elif change == 'nan': meta['windows'][0]['cosine_similarity'] = float('nan')
    elif change == 'method': result['candidates'][0]['method'] = 'heuristic_visual_audio'
    with pytest.raises(ValueError): benchmark.require_temporal_analysis(result, 'c' * 40)


def test_visual_model_cli_defaults_to_clip_and_selects_temporal(tmp_path):
    source = tmp_path / 'test.webm'; source.write_bytes(b'parse-only fixture')
    annotations = tmp_path / 'annotations.json'; annotations.write_text('{}')
    args = [str(source), '--annotations', str(annotations), '--query-id', 'frozen',
            '--attribution', 'test source', '--output-dir', str(tmp_path / 'out')]
    assert benchmark.parse_args(args).visual_model == 'clip-vit-b32'
    assert benchmark.parse_args(args + ['--visual-model', 'xclip-b16-k600']).visual_model == 'xclip-b16-k600'
    with pytest.raises(SystemExit): benchmark.parse_args(args + ['--visual-model', 'unknown-model'])


def test_temporal_top_candidate_export_keeps_exact_times():
    project = {'timeline': {'audio': {}, 'captions': {}}}
    candidate = {'start': 42.123, 'end': 50.123, 'method': benchmark.TEMPORAL_METHOD}
    timeline = benchmark.render_timeline(project, candidate, 'Public domain source')
    assert [(s['start'], s['end']) for s in timeline['segments']] == [(42.123, 50.123)]
