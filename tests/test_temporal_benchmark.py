"""Pure benchmark regressions. No app, model, network or media-processing calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_temporal_search as benchmark
TEST_DATA = None


@pytest.fixture(autouse=True)
def isolated_fixture_inputs(tmp_path, monkeypatch):
    """Bind invented media bytes to label-shaped fixtures; never decode them."""
    folder = tmp_path / 'fixtures'
    presets = {'schema_version': 1, 'model_scores_observed_before_preset': False, 'presets': {}}
    for name, duration, length in [('basketball', 16.3, 3), ('handwashing', 23.003, 9)]:
        target = folder / name
        target.mkdir(parents=True)
        source_name = 'clean-hands-short.webm' if name == 'handwashing' else 'indiana-legends-lyons.webm'
        source = target / source_name
        source.write_bytes(('Undecoded test fixture for ' + name).encode())
        digest = benchmark.sha256(source)
        if name == 'basketball':
            query_id, absent_id = 'receive_then_shoot', 'absent_dunk'
            phases = [{'start': 8.35, 'end': 9.2}, {'start': 9.2, 'end': 10.5}, {'start': 10.55, 'end': 11.25}]
            contrasts = [{'start': 0, 'end': 7.5}, {'start': 12.5, 'end': 16.3}]
        else:
            query_id, absent_id = 'rinse_then_dry', 'absent_toothbrushing'
            phases = [{'start': 11.5, 'end': 13.15}, {'start': 13.7, 'end': 14.6}, {'start': 14.7, 'end': 20.2}]
            contrasts = [{'start': 1.3, 'end': 11.2}, {'start': 21.15, 'end': 23.003}]
        annotations = {'annotation_version': 1, 'source_sha256': digest, 'duration_seconds': duration,
            'model_scores_observed_before_annotation': False, 'queries': [
                {'id': query_id, 'query': 'First action then second action', 'evaluation_kind': 'temporal_order',
                 'positive_intervals': [{'start': phases[0]['start'], 'end': phases[-1]['end']}],
                 'contrast_intervals': contrasts, 'ordered_phases': phases, 'requires_all_phases_in_order': True},
                {'id': absent_id, 'query': 'A visibly absent event', 'evaluation_kind': 'absent_event',
                 'expected_absent': True, 'positive_intervals': [], 'contrast_intervals': [{'start': 0, 'end': duration}]}]}
        provenance = {'sha256': digest, 'source_page_url': 'https://example.invalid/fixture',
            'direct_media_url': 'https://example.invalid/fixture.webm', 'license': 'Invented test bytes',
            'attribution': 'Local unit test', 'duration_seconds': duration}
        ann_path, prov_path = target / 'pre-score-annotations.json', target / 'source-provenance.json'
        ann_path.write_text(json.dumps(annotations)); prov_path.write_text(json.dumps(provenance))
        presets['presets'][name] = {'source_sha256': digest, 'annotations_sha256': benchmark.sha256(ann_path),
            'provenance_sha256': benchmark.sha256(prov_path), 'query_ids': [query_id, absent_id],
            'options': {'genre': 'action', 'mode': 'local', 'visual_search': True, 'transcribe': False,
                        'min_length': length, 'max_length': length, 'target_count': 3, 'min_separation': 2}}
    (folder / 'matched-presets.json').write_text(json.dumps(presets))
    monkeypatch.setattr(sys.modules[__name__], 'TEST_DATA', folder)


def inputs(name='handwashing', query='rinse_then_dry'):
    folder = TEST_DATA / name
    source = folder / ('clean-hands-short.webm' if name == 'handwashing' else 'indiana-legends-lyons.webm')
    return (source, folder / 'pre-score-annotations.json', folder / 'source-provenance.json',
            TEST_DATA / 'matched-presets.json', name, query)


def argv(output, name='handwashing', query='rinse_then_dry'):
    source, annotations, _, presets, _, _ = inputs(name, query)
    return [str(source), '--annotations', str(annotations), '--preset', name,
            '--presets', str(presets),
            '--query-id', query, '--output-dir', str(output), '--repository-dir', str(ROOT / 'unrelated-repo')]


@pytest.mark.parametrize('name,query', [('basketball', 'receive_then_shoot'), ('handwashing', 'rinse_then_dry'),
                                      ('basketball', 'absent_dunk'), ('handwashing', 'absent_toothbrushing')])
def test_frozen_fixture_sources_and_both_query_types_validate(name, query):
    frozen = benchmark.freeze_inputs(*inputs(name, query))
    assert frozen['query']['id'] == query
    assert frozen['snapshots']['annotations-frozen.json'] == inputs(name, query)[1].read_bytes()


def test_changed_annotation_is_rejected_before_any_api(tmp_path):
    values = list(inputs())
    changed = tmp_path / 'changed.json'
    changed.write_bytes(values[1].read_bytes() + b' ')
    values[1] = changed
    with pytest.raises(ValueError, match='frozen preset hashes'):
        benchmark.freeze_inputs(*values)


def test_eight_second_preset_cannot_cover_nine_second_order_reference(tmp_path):
    values = list(inputs())
    presets = json.loads(values[3].read_text())
    presets['presets']['handwashing']['options'].update(min_length=8, max_length=8)
    changed = tmp_path / 'presets.json'
    changed.write_text(json.dumps(presets))
    values[3] = changed
    with pytest.raises(ValueError, match='too short'):
        benchmark.freeze_inputs(*values)


def test_absent_query_has_no_fake_recall_iou_or_accuracy():
    query = benchmark.freeze_inputs(*inputs('basketball', 'absent_dunk'))['query']
    result = benchmark.overlap_metrics([{'start': 8, 'end': 11, 'scores': {'ranking': -.25}}], query)
    top = result['top_candidate']
    assert result['expected_absent']
    assert top['positive_temporal_iou'] is None
    assert top['positive_fraction_of_candidate'] is None
    assert result['all_returned_positive_reference_fraction_covered'] is None
    assert top['contrast_fraction_of_candidate'] == 1
    assert top['raw_retrieval_scores']['ranking'] == -.25
    assert not any('confidence' in key or 'accuracy' in key for key in top)


def test_phase_coverage_is_not_claimed_temporal_verification():
    query = benchmark.freeze_inputs(*inputs())['query']
    full = {'start': 11.4, 'end': 20.4}
    partial = {'start': 12, 'end': 21}
    result = benchmark.overlap_metrics([full, partial], query)
    assert result['top_candidate']['covers_all_annotated_phases'] is True
    assert result['ranked_candidates'][1]['covers_all_annotated_phases'] is False
    assert result['top_candidate']['temporal_order_semantically_verified'] is False
    assert result['all_returned_positive_reference_fraction_covered'] == 1
    assert len(result['top_candidate']['ordered_phase_coverage']) == 3


def temporal_analysis():
    spec = benchmark.MODELS['temporal']
    frames = [{'kind': 'temporal_frame', 'time': 1 + .3 * i, 'ref': f'frame-{i}.jpg'} for i in range(8)]
    return {'provider': spec['provider'], 'ai_used': True, 'model': spec['repository'], 'revision': spec['revision'],
        'visual_search': {'model': spec['repository'], 'revision': spec['revision'], 'model_id': spec['id'],
            'method': spec['method'], 'learned_temporal_encoder': True, 'full_classifier': False,
            'sampling_complete': True, 'frames_per_window': 8, 'window_count': 1, 'sample_count': 8,
            'windows': [{'start': 1, 'end': 4, 'cosine_similarity': -.2, 'frames': frames}]},
        'candidates': [{'start': 1, 'end': 4, 'method': spec['method'], 'evidence': frames,
                        'scores': {'ranking': -.2, 'temporal_similarity': -.2}}]}


def test_temporal_provenance_and_negative_cosine_are_preserved():
    analysis = temporal_analysis()
    benchmark.require_visual_analysis(analysis, 'temporal')
    query = benchmark.freeze_inputs(*inputs('basketball', 'absent_dunk'))['query']
    result = benchmark.ranked_sample_metrics(analysis, 'temporal', query)
    assert result['top_candidate']['raw_retrieval_scores']['cosine_similarity'] == -.2


@pytest.mark.parametrize('change', ['provider', 'revision', 'full_classifier', 'incomplete', 'reverse', 'outside', 'nan', 'method'])
def test_wrong_or_incomplete_temporal_evidence_is_rejected(change):
    data = temporal_analysis()
    meta = data['visual_search']
    if change == 'provider': data['provider'] = 'heuristics'
    elif change == 'revision': data['revision'] = 'unverified'
    elif change == 'full_classifier': meta['full_classifier'] = True
    elif change == 'incomplete': meta['sampling_complete'] = False
    elif change == 'reverse': meta['windows'][0]['frames'].reverse()
    elif change == 'outside': meta['windows'][0]['frames'][-1]['time'] = 5
    elif change == 'nan': meta['windows'][0]['cosine_similarity'] = float('nan')
    elif change == 'method': data['candidates'][0]['method'] = 'local_clip_visual'
    with pytest.raises(ValueError): benchmark.require_visual_analysis(data, 'temporal')


def test_dry_run_creates_exact_snapshots_and_never_constructs_http_client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Dry run attempted network/API access')
    monkeypatch.setattr(benchmark.httpx, 'Client', forbidden)
    output = tmp_path / 'plan'
    report = benchmark.main(argv(output))
    assert report['status'] == 'planned' and report['jobs'] == []
    assert all(report['original_integrity'].values())
    assert report['matched_options']['min_length'] == report['matched_options']['max_length'] == 9
    assert (output / 'annotations-frozen.json').read_bytes() == inputs()[1].read_bytes()


@pytest.mark.parametrize('url', ['https://127.0.0.1:8741', 'http://evil.example', 'http://user:secret@localhost',
                               'http://localhost/path', 'http://localhost?redirect=example', 'http://localhost:99999'])
def test_non_loopback_or_ambiguous_api_targets_rejected(tmp_path, url):
    with pytest.raises(SystemExit): benchmark.parse_args(argv(tmp_path / 'out') + ['--base-url', url])


def test_outputs_inside_repository_rejected(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv(tmp_path / 'repo' / 'media') + ['--repository-dir', str(tmp_path / 'repo')])


@pytest.mark.parametrize('available_ram', [0, 1024 ** 3])
def test_guard_blocks_below_existing_worker_reserve(available_ram):
    health = {'worker_active': False, 'queue_length': 0, 'worker': {'alive': True,
        'minimum_available_ram_bytes': 1536 * 1024 ** 2, 'workers': [{'available_ram_bytes': available_ram}]}}
    with pytest.raises(RuntimeError, match='safety threshold'): benchmark.ensure_ready(health)


def mock_api(monkeypatch, *, models_available=True, fail_temporal=False):
    """Exercise submission/report orchestration with a fake client, no sockets."""
    frozen = benchmark.freeze_inputs(*inputs())
    state = {'analysis': None, 'submissions': [], 'imports': 0}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'project_id': 'qa-test', 'job': {'id': 'import'}}
    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, path, **kwargs):
            assert path == '/api/v1/imports'
            state['imports'] += 1
            return Response()
    def request(client, method, path, payload=None, key=None):
        if path == '/api/v1/health':
            return {'worker_active': False, 'queue_length': 0, 'worker': {'alive': True,
                'minimum_available_ram_bytes': 1536 * 1024 ** 2, 'workers': [{'available_ram_bytes': 3 * 1024 ** 3}]}}
        if path == '/api/v1/capabilities':
            return {'models': {'models': [{**s, 'available': models_available, 'runtime_available': models_available}
                                         for s in benchmark.MODELS.values()]}}
        if method == 'POST' and path.endswith('/analyze'):
            name = next(n for n, spec in benchmark.MODELS.items() if spec['id'] == payload['visual_model'])
            state['submissions'].append(copy.deepcopy(payload))
            spec = benchmark.MODELS[name]
            if name == 'temporal':
                analysis = temporal_analysis()
                window = analysis['visual_search']['windows'][0]
                window.update(start=11.4, end=20.4)
                for i, frame in enumerate(window['frames']): frame['time'] = 11.4 + i
            else:
                analysis = {'provider': spec['provider'], 'model': spec['repository'], 'revision': spec['revision'],
                    'ai_used': True, 'visual_search': {'model': spec['repository'], 'revision': spec['revision'],
                        'sample_count': 2, 'sampling_complete': True,
                        'samples': [{'time': 12, 'cosine_similarity': -.1}, {'time': 18, 'cosine_similarity': .2}]},
                    'candidates': [{'method': spec['method'], 'evidence': [{'kind': 'visual_frame', 'time': 18}],
                                    'scores': {'ranking': .2}}]}
            analysis['visual_search']['query'] = payload['query']
            analysis['candidates'][0].update(start=11.4, end=20.4)
            state['analysis'] = analysis
            return {'id': name}
        if method == 'GET' and path == '/api/v1/projects/qa-test':
            return {'source': {'sha256': frozen['source_sha256'], 'metadata': {'duration': 23.003}},
                    'analysis': state['analysis']}
        raise AssertionError(f'Unexpected mock API call: {method} {path}')
    def complete(client, job, record, checkpoint, timeout):
        record.update(id=job['id'], status='succeeded')
        if job['id'] == 'temporal' and fail_temporal:
            record['status'] = 'failed'
            raise RuntimeError('Temporal inference blocked: system RAM reserve unavailable')
        return {'id': job['id'], 'status': 'succeeded'}
    monkeypatch.setattr(benchmark.httpx, 'Client', Client)
    monkeypatch.setattr(benchmark, 'request', request)
    monkeypatch.setattr(benchmark, 'complete_owned_job', complete)
    return state


def test_mock_serial_comparison_uses_identical_query_and_options(tmp_path, monkeypatch):
    state = mock_api(monkeypatch)
    result = benchmark.main(argv(tmp_path / 'serial') + ['--execute'])
    assert result['status'] == 'completed' and state['imports'] == 1
    assert len(state['submissions']) == 2
    left, right = state['submissions']
    assert left.pop('visual_model') == 'clip-vit-b32'
    assert right.pop('visual_model') == 'xclip-b16-k600'
    assert left == right and left['transcribe'] is False
    assert left['min_length'] == left['max_length'] == 9
    assert result['methods']['temporal']['candidate_overlap']['top_candidate']['duration_matches_requested_preset']


def test_unavailable_models_never_import_or_submit_jobs(tmp_path, monkeypatch):
    state = mock_api(monkeypatch, models_available=False)
    result = benchmark.main(argv(tmp_path / 'blocked') + ['--execute'])
    assert result['status'] == 'blocked'
    assert state['imports'] == 0 and state['submissions'] == [] and result['jobs'] == []


def test_runtime_guard_failure_preserves_partial_result_without_retry(tmp_path, monkeypatch):
    state = mock_api(monkeypatch, fail_temporal=True)
    output = tmp_path / 'guarded'
    with pytest.raises(RuntimeError, match='RAM reserve'):
        benchmark.main(argv(output) + ['--execute'])
    report = json.loads((output / 'report.json').read_text())
    assert report['methods']['clip']['status'] == 'completed'
    assert report['methods']['temporal']['status'] == 'failed'
    assert len(state['submissions']) == 2
    assert report['pending_submission'] is None
