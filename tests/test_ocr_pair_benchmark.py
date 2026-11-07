"""Pure evaluator tests: no model, network, video decoder, or app jobs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_ocr_pair as benchmark


def frame(status='partial', text='Open door', bounds=None):
    return {'id': 'fixture', 'ground_truth': status,
            'labels': [] if status == 'text_free' else [{'id': 'one', 'text': text, 'box': bounds or [.1, .1, .5, .3]}]}


def detection(text='Open door', bounds=None):
    return {'text': text, 'box_normalized': bounds or [.1, .1, .5, .3], 'model_score': .8}


def metrics(f=None, predictions=None):
    return benchmark.evaluate_frame(f or frame(), {'detections': predictions or []})


@pytest.fixture
def manifest(tmp_path):
    image = tmp_path / 'frame.png'
    Image.new('RGB', (32, 16), 'white').save(image)
    frozen = {**frame(), 'path': 'frame.png', 'width': 32, 'height': 16,
              'sha256': benchmark.file_hash(image), 'provenance': {'kind': 'invented unit-test image'}}
    document = {'schema_version': 1, 'labels_observed_after_ocr': False, 'frames': [frozen]}
    path = tmp_path / 'labels.json'; path.write_text(json.dumps(document))
    return path, document, image


def test_strict_case_and_whitespace_normalization_is_full_label_only():
    strict = metrics(predictions=[detection('  OPEN\t door  ')])['metrics']['raw']['strict']
    assert strict['true_positives'] == 1 and strict['missed_labels'] == 0
    for text in ('Open', 'door', 'Open doorway', 'Please Open door', 'Opendoor'):
        assert metrics(predictions=[detection(text)])['metrics']['raw']['strict']['true_positives'] == 0


def test_missing_word_space_has_separate_compact_result():
    result = metrics(predictions=[detection('Opendoor')])['metrics']['raw']
    assert result['strict']['true_positives'] == 0
    assert result['whitespace_insensitive']['true_positives'] == 1
    assert result['strict']['unknown_predictions'] == 1


def test_correct_text_in_wrong_region_is_not_spatial_true_positive():
    result = metrics(predictions=[detection(bounds=[.6, .6, .9, .9])])
    assert result['metrics']['raw']['strict']['true_positives'] == 0
    assert result['legacy_text_only_diagnostic']['full_labels_present'] == 1


@pytest.mark.parametrize('ground_truth,fp,unknown', [('partial', 0, 2), ('exhaustive', 2, 0), ('text_free', 3, 0)])
def test_exhaustiveness_controls_false_positive_accounting(ground_truth, fp, unknown):
    result = metrics(frame(ground_truth), [detection(), detection('Extra'), detection('Wrong')])['metrics']['raw']['strict']
    assert result['false_positives'] == fp
    assert result['unknown_predictions'] == unknown
    assert result['true_positives'] == (0 if ground_truth == 'text_free' else 1)


def test_raw_duplicates_are_preserved_and_dedup_reported_separately():
    result = metrics(frame('exhaustive'), [detection(), detection(), detection('Different')])
    assert result['raw_accepted_detection_count'] == 3
    assert result['deduplicated_occurrence_count'] == 2
    assert result['dedup_groups'] == [[0, 1], [2]]
    assert result['metrics']['raw']['strict']['false_positives'] == 2
    assert result['metrics']['deduplicated']['strict']['false_positives'] == 1


def test_same_words_at_distant_locations_are_not_deduplicated():
    result = metrics(predictions=[detection(), detection(bounds=[.6, .6, .9, .9])])
    assert result['deduplicated_occurrence_count'] == 2


def test_grouping_never_hides_missing_word_spaces():
    result = metrics(predictions=[detection(), detection('Opendoor')])
    assert result['deduplicated_occurrence_count'] == 2


def test_maximum_matching_does_not_greedily_lose_repeated_label():
    f = frame('exhaustive', text='X', bounds=[0, 0, .5, .5])
    f['labels'].append({'id': 'two', 'text': 'X', 'box': [.2, 0, .7, .5]})
    rows = benchmark.predictions([detection('X', [.1, 0, .6, .5]), detection('X', [0, 0, .3, .5])])
    result = benchmark.match(f, rows)
    assert result['true_positives'] == 2
    assert len({m['prediction_index'] for m in result['matches']}) == 2


def test_one_prediction_cannot_match_two_labels():
    f = frame(); f['labels'].append({'id': 'two', 'text': 'Open door', 'box': [.1, .1, .5, .3]})
    result = metrics(f, [detection()])['metrics']['raw']['strict']
    assert result['true_positives'] == 1 and result['missed_labels'] == 1


def test_quadrilateral_coordinates_are_compared_in_normalized_frame_space():
    value = detection(bounds=[[.1,.1],[.5,.1],[.5,.3],[.1,.3]])
    assert metrics(predictions=[value])['metrics']['raw']['strict']['true_positives'] == 1


@pytest.mark.parametrize('bounds', [[0,0,0,1], [-.1,0,.5,1], [0,0,1.1,1], [0,0,float('nan'),1], [False,0,1,1]])
def test_bad_boxes_fail_closed(bounds):
    with pytest.raises(ValueError):
        metrics(predictions=[detection(bounds=bounds)])


def test_no_new_score_threshold_is_applied():
    value = detection(); value['model_score'] = -123
    assert metrics(predictions=[value])['metrics']['raw']['strict']['true_positives'] == 1


def test_text_free_with_labels_is_rejected():
    f = frame(); f['ground_truth'] = 'text_free'
    with pytest.raises(ValueError, match='Text-free'):
        metrics(f)


def test_frozen_inputs_hash_dimensions_and_pre_score_claim(manifest):
    path, document, _ = manifest
    sha = benchmark.file_hash(path)
    frozen = benchmark.freeze_manifest(path, sha)
    assert frozen['manifest_sha256'] == sha
    assert frozen['manifest_bytes'] == path.read_bytes()
    with pytest.raises(ValueError, match='pre-inference'):
        benchmark.freeze_manifest(path, 'f' * 64)
    document['frames'][0]['width'] = 30; path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='dimensions'):
        benchmark.freeze_manifest(path)
    document['labels_observed_after_ocr'] = True; path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='pre-OCR'):
        benchmark.freeze_manifest(path)


def test_changed_image_and_path_escape_rejected(manifest):
    path, document, image = manifest
    image.write_bytes(b'different bytes')
    with pytest.raises(ValueError, match='Image bytes'):
        benchmark.freeze_manifest(path)
    document['frames'][0]['path'] = '../unrelated.png'; path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='relative'):
        benchmark.freeze_manifest(path)


def test_paired_inference_receives_identical_bytes_and_no_labels(manifest):
    path, _, image = manifest
    frozen = benchmark.freeze_manifest(path)
    calls = []
    def infer(raw, mode):
        calls.append((raw, mode))
        return {'detections': [detection()], 'raw_lines': [{'text':'Open door'}]}
    result = benchmark.paired_run(frozen, infer, {'model_id': 'test-only-no-model'})
    assert calls == [(image.read_bytes(), 'baseline'), (image.read_bytes(), 'detail')]
    assert result['aggregate']['baseline']['raw']['strict']['true_positives'] == 1
    assert result['protocol_sha256'] == benchmark.PROTOCOL_SHA256
    assert result['frames'][0]['modes']['baseline']['resources']['wall_seconds'] >= 0
    assert result['data_transmitted'] is False


def test_dry_run_loads_no_ocr_and_preserves_manifest_bytes(manifest, tmp_path, monkeypatch):
    path, _, _ = manifest
    monkeypatch.setitem(sys.modules, 'clipstudio.ocr', None)
    result = benchmark.main([str(path), '--output-dir', str(tmp_path/'report')])
    assert result['dry_run'] and not result['inference_run']
    assert (tmp_path/'report/manifest-frozen.json').read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match='fresh output'):
        benchmark.main([str(path), '--output-dir', str(tmp_path/'report')])


def test_execute_requires_previously_recorded_hash(manifest, tmp_path):
    with pytest.raises(SystemExit):
        benchmark.main([str(manifest[0]), '--output-dir', str(tmp_path/'out'), '--execute'])
    assert not (tmp_path/'out').exists()


def test_changed_frame_between_freeze_and_inference_blocks_model_call(manifest):
    frozen = benchmark.freeze_manifest(manifest[0]); manifest[2].write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed before inference'):
        benchmark.paired_run(frozen, lambda *args: pytest.fail('must not call inference'), {})


def test_failed_mode_does_not_change_comparison_denominator(manifest):
    frozen = benchmark.freeze_manifest(manifest[0])
    def infer(raw, mode):
        if mode == 'detail':
            raise RuntimeError('invented controlled failure')
        return {'detections': [detection()]}
    result = benchmark.paired_run(frozen, infer, {})
    assert result['frames'][0]['modes']['baseline']['evaluation']['metrics']['raw']['strict']['true_positives'] == 1
    assert result['frames'][0]['modes']['detail']['status'] == 'failed'
    assert result['aggregate_cohort']['scored_frames'] == 0
    assert result['aggregate_cohort']['excluded_frame_ids'] == ['fixture']
    assert result['aggregate']['baseline']['raw']['strict']['annotated_labels'] == 0
    assert result['aggregate']['detail']['raw']['strict']['annotated_labels'] == 0


def test_file_hash_uses_explicit_byte_bound(tmp_path):
    path = tmp_path/'file'; path.write_bytes(b'abcd')
    assert benchmark.file_hash(path, 4) == benchmark.digest(b'abcd')
    with pytest.raises(ValueError, match='byte limit'):
        benchmark.file_hash(path, 3)


def test_frozen_video_control_aliases_do_not_rewrite_labels():
    f = {**frame(), 'source_time_seconds': 2.0}
    doc = {'frames': [f], 'temporal_controls': [{'id':'control', 'sha256':'a'*64, 'frame_ids':['fixture']}]}
    original = copy.deepcopy(doc)
    resolved = benchmark.resolved_controls(doc)
    assert resolved['control']['source_sha256'] == 'a'*64
    assert resolved['control']['frames'][0]['time'] == 2.0
    assert doc == original


def test_model_identity_uses_selected_catalog_and_actual_file_hashes(tmp_path):
    from types import SimpleNamespace
    module = tmp_path/'ocr.py'
    for name in ('ocr.py', 'vision.py', 'models.py', 'runtime.py'):
        (tmp_path/name).write_bytes(name.encode())
    location = tmp_path/'weights'; location.mkdir(); (location/'new.onnx').write_bytes(b'invented')
    catalog = {'ppocr-v5-mobile': {'repository':'test/new', 'revision':'pinned-revision',
                          'files': {'new.onnx': (8, benchmark.digest(b'invented'))}}}
    result = benchmark.selected_model_identity(SimpleNamespace(__file__=str(module)), catalog, 'ppocr-v5-mobile', location)
    assert result['model_id'] == 'ppocr-v5-mobile' and result['model_repository'] == 'test/new'
    assert result['model_file_sha256']['new.onnx'] == benchmark.digest(b'invented')
    assert set(result['pipeline_files_sha256']) == {'ocr.py', 'vision.py', 'models.py', 'runtime.py'}
    (location/'new.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='pinned catalog'):
        benchmark.selected_model_identity(SimpleNamespace(__file__=str(module)), catalog, 'ppocr-v5-mobile', location)


def test_model_choice_dry_run_does_not_load_or_change_default(manifest, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'clipstudio.ocr', None)
    result = benchmark.main([str(manifest[0]), '--ocr-model', 'ppocr-v5-mobile', '--output-dir', str(tmp_path/'v5')])
    assert result['requested_model_id'] == 'ppocr-v5-mobile' and not result['inference_run']
    result = benchmark.main([str(manifest[0]), '--output-dir', str(tmp_path/'default')])
    assert result['requested_model_id'] == 'ppocr-v4'


def test_raw_evidence_streams_to_separate_hash_bound_files(manifest, tmp_path):
    frozen = benchmark.freeze_manifest(manifest[0])
    raw = {'detections': [detection()], 'raw_lines': [{'text': 'unchanged raw'}]}
    result = benchmark.paired_run(frozen, lambda *args: copy.deepcopy(raw), {}, evidence_dir=tmp_path/'evidence')
    for mode in ('baseline', 'detail'):
        record = result['frames'][0]['modes'][mode]
        assert record['raw_result'] is None
        saved = tmp_path/record['raw_evidence']['path']
        assert benchmark.file_hash(saved) == record['raw_evidence']['sha256']
        assert json.loads(saved.read_text()) == raw
    assert result['aggregate_cohort']['scored_frames'] == 1


def test_raw_evidence_cap_does_not_silently_claim_success(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'MAX_RAW_EVIDENCE', 1)
    result = benchmark.paired_run(benchmark.freeze_manifest(manifest[0]), lambda *args: {'detections': [detection()]}, {}, evidence_dir=tmp_path/'limited')
    assert result['aggregate_cohort']['scored_frames'] == 0
    assert result['frames'][0]['modes']['baseline']['failure']['type'] == 'ValueError'
    assert not list((tmp_path/'limited').iterdir())


def test_execute_stub_preserves_guards_and_selected_engine_identity(manifest, tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    module_dir = tmp_path/'module'; module_dir.mkdir()
    for name in ('ocr.py', 'vision.py', 'models.py', 'runtime.py'):
        (module_dir/name).write_text('# invented test module\n')
    weights = tmp_path/'weights'; weights.mkdir(); (weights/'model.onnx').write_bytes(b'not a model')
    calls = []
    def memory(): calls.append('memory')
    def recognizer(check, *, model_id):
        check(); calls.append(model_id)
        return SimpleNamespace(model_id=model_id, close=lambda: calls.append('closed'))
    def recognize(image, engine, check, *, mode):
        check(); calls.append(mode)
        return {'detections': [detection()], 'model_id': engine.model_id}
    ocr = SimpleNamespace(__file__=str(module_dir/'ocr.py'), MAX_WALL_SECONDS=600,
                          _memory_check=memory, _checkpoint=lambda check, deadline: check(),
                          Recognizer=recognizer, recognize_frame=recognize)
    package, models = ModuleType('clipstudio'), ModuleType('clipstudio.models')
    package.ocr = ocr
    models.CATALOG = {'ppocr-v5-mobile': {'repository':'test-only', 'revision':'frozen',
                     'files': {'model.onnx': (11, benchmark.digest(b'not a model'))}}}
    models.model_path = lambda *args, **kwargs: weights
    monkeypatch.setitem(sys.modules, 'clipstudio', package)
    monkeypatch.setitem(sys.modules, 'clipstudio.models', models)
    result = benchmark.main([str(manifest[0]), '--manifest-sha256', benchmark.file_hash(manifest[0]),
                             '--ocr-model', 'ppocr-v5-mobile', '--execute', '--output-dir', str(tmp_path/'stub-run')])
    assert result['identity']['model_id'] == 'ppocr-v5-mobile'
    assert calls.count('memory') >= 5 and calls[-1] == 'closed'
    assert 'baseline' in calls and 'detail' in calls
    assert result['aggregate_cohort']['scored_frames'] == 1
    assert result['overall_wall_budget_seconds'] == 1800


def test_optional_video_labels_only_score_actual_annotated_pts():
    control = {'id': 'stable', 'source_sha256': 'a'*64, 'frames': [{**frame(), 'time': 2.0}]}
    evidence = {'source_sha256': 'a'*64, 'ocr': {'samples': [
        {'time': 2.0, 'detections': [detection()], 'quality_selection': {'selected': 2}},
        {'time': 2.25, 'detections': [detection('unlabeled')] }], 'temporal_agreement': {'matches': 9}}}
    result = benchmark.evaluate_video_control(control, evidence)
    assert result['samples'][0]['evaluation']['metrics']['raw']['strict']['true_positives'] == 1
    assert result['samples'][1]['evaluation'] is None
    assert result['unannotated_sample_count'] == 1
    assert result['temporal_agreement_accuracy_claimed'] is False
    assert result['raw_pipeline_metadata']['temporal_agreement'] == {'matches': 9}
    evidence['source_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='different source'):
        benchmark.evaluate_video_control(control, evidence)


def test_ambiguous_video_timestamp_labels_rejected():
    control = {'id': 'ambiguous', 'source_sha256': 'a'*64, 'frames': [{**frame(), 'time': 2.0}, {**frame(), 'time': 2.0005}]}
    with pytest.raises(ValueError, match='Ambiguous'):
        benchmark.evaluate_video_control(control, {'source_sha256': 'a'*64, 'ocr': {'samples': [{'time': 2, 'detections': []}]}})


def test_aggregate_keeps_known_false_positives_and_unknowns_separate():
    values = []
    for kind in ('partial', 'exhaustive', 'text_free'):
        evaluated = metrics(frame(kind), [detection(), detection('extra')])
        values.append({'modes': {mode: {'evaluation': copy.deepcopy(evaluated)} for mode in benchmark.PROTOCOL['pair']}})
    counts = benchmark.aggregate(values)['baseline']['raw']['strict']
    assert counts == {'annotated_labels': 2, 'predictions': 6, 'true_positives': 2,
                      'missed_labels': 0, 'false_positives': 3, 'unknown_predictions': 1}
