import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import weakref

import numpy as np
from PIL import Image
import psutil
import pytest

from clipstudio import temporal


def noop():
    pass


def test_preprocess_uses_upstream_bilinear_imagenet_center_crop():
    yy, xx = np.indices((227, 401))
    pixels = np.stack((xx % 256, yy % 256, (xx * yy) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(pixels)
    actual = temporal.preprocess_image(image)
    resized = image.resize((int(224 * 401 / 227), 224), Image.Resampling.BILINEAR)
    left = (resized.width - 224) // 2
    expected = (np.asarray(resized.crop((left, 0, left + 224, 224)), dtype=np.float32) / 255 - [.485, .456, .406]) / [.229, .224, .225]
    np.testing.assert_allclose(actual, expected.transpose(2, 0, 1), rtol=1e-5, atol=1e-6)
    assert actual.shape == (3, 224, 224)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


def test_preprocess_converts_grayscale_and_rejects_extreme_aspect():
    assert temporal.preprocess_image(Image.new('L', (224, 224))).shape == (3, 224, 224)
    with pytest.raises(ValueError, match='dimensions'):
        temporal.preprocess_image(Image.new('RGB', (1, 40)))


@pytest.mark.parametrize('value', [np.zeros(512), np.full(512, np.nan), np.full(512, np.inf), np.ones(511)])
def test_invalid_embeddings_rejected(value):
    with pytest.raises(ValueError):
        temporal._unit_vector(value)


def test_long_video_uniform_windows_include_tail_and_report_sparse_coverage():
    windows, plan = temporal.plan_windows(0, 43200, 4, 8, [])
    assert len(windows) == 60
    assert windows[0] == {'start': 0, 'end': 6}
    assert windows[-1] == {'start': 43194, 'end': 43200}
    assert plan['window_cap_reached'] is True
    coverage = temporal._coverage(windows, {'start': 0, 'end': 43200})
    assert coverage['sampled_window_seconds'] == pytest.approx(360)
    assert coverage['sampled_window_coverage_fraction'] == pytest.approx(1 / 120)
    assert coverage['maximum_unsampled_gap_seconds'] > 700


def test_exclusions_reject_full_windows_before_inference():
    windows, plan = temporal.plan_windows(0, 20, 4, 8, [{'start': 2, 'end': 4}])
    assert plan['excluded_windows'] == 2
    assert all(not window['start'] < 4 or not window['end'] > 2 for window in windows)


def test_short_video_does_not_invent_context():
    windows, plan = temporal.plan_windows(10, 12, 4, 8, [])
    assert windows == []
    assert plan['planned_windows'] == 0


@pytest.mark.parametrize('start,end,minimum,maximum', [(0, float('inf'), 4, 8), (0, 43201, 4, 8), (5, 4, 4, 8), (0, 10, 8, 4), (0, 10, 0, 4)])
def test_window_bounds_rejected(start, end, minimum, maximum):
    with pytest.raises(ValueError):
        temporal.plan_windows(start, end, minimum, maximum, [])


def test_frame_times_are_chronological_inside_window():
    times = temporal.window_frame_times({'start': 2, 'end': 8}, {'reported_frame_period_seconds': 1})
    assert len(times) == 8
    assert times[0] == 2
    assert times[-1] == pytest.approx(6.999)
    assert all(a < b < 8 for a, b in zip(times, times[1:]))


def test_encoder_releases_text_before_video_and_loads_video_once(monkeypatch):
    created, text_ref = [], None

    class Session:
        def run(self, outputs, inputs):
            return [np.ones((1, 512), np.float32)]

    def factory(path, inputs, output):
        nonlocal text_ref
        created.append(path.name)
        if path.name.startswith('video'):
            gc.collect()
            assert text_ref() is None
        value = Session()
        if path.name.startswith('text'):
            text_ref = weakref.ref(value)
        return value

    monkeypatch.setattr(temporal, '_session', factory)
    monkeypatch.setattr(temporal, 'tokenize_query', lambda *args: ({}, {}))
    with temporal.TemporalEncoder(Path('/unused'), noop) as encoder:
        encoder.encode_query('A door opens.')
        values = np.zeros((1, 8, 3, 224, 224), np.float32)
        assert encoder.score(values) == pytest.approx(1)
        assert encoder.score(values) == pytest.approx(1)
    assert created == ['text_tower.onnx', 'video_tower.onnx']


def test_session_disables_prepacking_and_checks_fixed_shapes(monkeypatch):
    import onnxruntime as ort
    captured = {}

    def create(path, sess_options, providers):
        captured.update(options=sess_options, providers=providers)
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='pixel_values', type='tensor(float)', shape=[1, 8, 3, 224, 224])],
                               get_outputs=lambda: [SimpleNamespace(name='video_embeds', type='tensor(float)', shape=[1, 512])])
    monkeypatch.setattr(ort, 'InferenceSession', create)
    temporal._session(Path('unused'), [('pixel_values', 'tensor(float)', [1, 8, 3, 224, 224])], 'video_embeds')
    options = captured['options']
    assert options.get_session_config_entry('session.disable_prepacking') == '1'
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert not options.enable_cpu_mem_arena
    assert not options.enable_mem_pattern
    assert captured['providers'] == ['CPUExecutionProvider']


def _mock_pipeline(monkeypatch, tmp_path, scores):
    source = tmp_path / 'source.mp4'
    source.touch()
    monkeypatch.setattr(temporal, '_video_extent', lambda *a: ([], {'start': 0, 'end': 20, 'format_start_time': -0.1, 'reported_frame_period_seconds': .04}))
    monkeypatch.setattr(temporal, 'model_path', lambda *a, **kw: tmp_path)
    monkeypatch.setitem(temporal.CATALOG, temporal.MODEL_ID, {'repository': 'test-fixture', 'revision': 'test-only'})

    def decode(source, desired, target, check, deadline):
        check()
        Image.new('RGB', (224, 224), 'red').save(target)
        return desired - .1
    monkeypatch.setattr(temporal, '_decode_frame', decode)

    class Encoder:
        def __init__(self, *args):
            self.iter_scores = iter(scores)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def encode_query(self, query):
            return {'query_truncated': False}
        def score(self, values):
            assert values.shape == (1, 8, 3, 224, 224)
            return next(self.iter_scores)
    monkeypatch.setattr(temporal, 'TemporalEncoder', Encoder)
    return source


def test_core_preserves_negative_rank_and_actual_pts(monkeypatch, tmp_path):
    source = _mock_pipeline(monkeypatch, tmp_path, [-.9, -.8, -.7, -.6, -.5, -.4])
    candidates, metadata, limitations = temporal._temporal_candidates(source, 20, {'query': 'A door opens.', 'min_length': 4, 'max_length': 8}, tmp_path / 'result', noop)
    assert [candidate['scores']['ranking'] for candidate in candidates] == [-.4, -.5, -.6, -.7, -.8, -.9]
    assert all(candidate['method'] == temporal.METHOD and 'confidence' not in candidate for candidate in candidates)
    assert metadata['sample_count'] == 48
    assert metadata['learned_temporal_encoder'] is True
    assert metadata['full_classifier'] is False
    assert metadata['sampling_complete'] is True
    assert metadata['windows'][0]['frames'][0]['time'] == 0
    assert metadata['sampled_window_coverage_fraction'] == pytest.approx(1)
    assert (tmp_path / 'result/temporal-search.json').is_file()


def test_core_cancellation_before_work(monkeypatch, tmp_path):
    source = _mock_pipeline(monkeypatch, tmp_path, [])
    def cancelled():
        raise RuntimeError('cancelled by operator')
    with pytest.raises(RuntimeError, match='cancelled by operator'):
        temporal._temporal_candidates(source, 20, {'query': 'test'}, tmp_path / 'result', cancelled)
    assert not (tmp_path / 'result').exists()


def test_core_cancellation_after_model_run_is_not_success(monkeypatch, tmp_path):
    source = _mock_pipeline(monkeypatch, tmp_path, [-.4] * 6)
    ran = False
    original = temporal.TemporalEncoder.score
    def score(self, values):
        nonlocal ran
        value = original(self, values)
        ran = True
        return value
    monkeypatch.setattr(temporal.TemporalEncoder, 'score', score)
    def check():
        if ran:
            raise RuntimeError('cancelled after inference')
    with pytest.raises(RuntimeError, match='cancelled after inference'):
        temporal._temporal_candidates(source, 20, {'query': 'test'}, tmp_path / 'result', check)
    assert not (tmp_path / 'result/temporal-search.json').exists()


def test_all_windows_excluded_performs_no_inference(monkeypatch, tmp_path):
    source = _mock_pipeline(monkeypatch, tmp_path, [])
    monkeypatch.setattr(temporal.TemporalEncoder, 'encode_query', lambda *a: pytest.fail('must not load model for excluded intervals'))
    candidates, metadata, _ = temporal._temporal_candidates(source, 20, {'query': 'test', 'exclusions': [{'start': 0, 'end': 20}]}, tmp_path / 'result', noop)
    assert candidates == []
    assert metadata['sample_count'] == 0
    assert metadata['sampled_window_coverage_fraction'] == 0
    assert metadata['sampling_complete'] is True


def test_real_low_fps_short_windows_fail_before_model_loading(monkeypatch, tmp_path):
    source = tmp_path / 'low-fps.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x90:rate=0.2:duration=10', '-an', '-c:v', 'libx264', '-threads', '1', str(source)], check=True)
    monkeypatch.setattr(temporal, 'model_path', lambda *a, **kw: pytest.fail('must reject before model load'))
    with pytest.raises(ValueError, match='too short.*frame cadence'):
        temporal._temporal_candidates(source, 10, {'query': 'test', 'min_length': 1, 'max_length': 1}, tmp_path / 'result', noop)


def test_source_inside_output_and_existing_artifacts_are_preserved(monkeypatch, tmp_path):
    source = _mock_pipeline(monkeypatch, tmp_path, [])
    with pytest.raises(ValueError, match='does not contain the source'):
        temporal._temporal_candidates(source, 20, {'query': 'test'}, tmp_path, noop)
    out = tmp_path / 'result'
    out.mkdir()
    artifact = out / 'temporal-search.json'
    artifact.write_text('preserve')
    with pytest.raises(ValueError, match='already contains artifacts'):
        temporal._temporal_candidates(source, 20, {'query': 'test'}, out, noop)
    assert artifact.read_text() == 'preserve'


def test_public_entry_preserves_source_even_when_ram_blocks_spawn(monkeypatch, tmp_path):
    source = tmp_path / 'temporal-resource-failure.json'
    source.write_text('preserve source')
    monkeypatch.setattr(temporal.psutil, 'virtual_memory', lambda: SimpleNamespace(available=0))
    with pytest.raises(ValueError, match='does not contain the source'):
        temporal.temporal_candidates(source, 20, {'query': 'test'}, tmp_path, noop)
    assert source.read_text() == 'preserve source'


def test_atomic_json_does_not_follow_preexisting_temporary_symlink(tmp_path):
    original = tmp_path / 'original'
    original.write_text('preserve')
    (tmp_path / 'result.json.tmp').symlink_to(original)
    temporal._atomic_json(tmp_path / 'result.json', {'ok': True})
    assert original.read_text() == 'preserve'
    assert json.loads((tmp_path / 'result.json').read_text()) == {'ok': True}


def test_stale_process_identity_is_not_rediscovered(monkeypatch):
    identity = SimpleNamespace(is_running=lambda: False, children=lambda **kw: pytest.fail('must not discover replacement PID descendants'))
    process = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda **kw: 0)
    monkeypatch.setattr(temporal.psutil, 'Process', lambda *a: pytest.fail('must not reconstruct PID identity'))
    assert temporal._tree_rss(process, {}, identity) == 0
    temporal._terminate_owned_tree(process, [], identity)


def test_supervisor_blocks_before_spawn_when_ram_is_low(monkeypatch):
    monkeypatch.setattr(temporal.psutil, 'virtual_memory', lambda: SimpleNamespace(available=temporal.runtime.MIN_AVAILABLE_RAM - 1))
    monkeypatch.setattr(temporal.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='blocked'):
        temporal._run_supervised([], noop, 2, io.StringIO())


@pytest.mark.parametrize('failure', ['rss', 'reserve', 'timeout', 'cancel'])
def test_supervisor_terminates_owned_child_on_each_limit(monkeypatch, tmp_path, failure):
    memory_calls = 0
    def available():
        nonlocal memory_calls
        memory_calls += 1
        return SimpleNamespace(available=temporal.runtime.MIN_AVAILABLE_RAM + (1 if failure != 'reserve' or memory_calls == 1 else -1))
    monkeypatch.setattr(temporal.psutil, 'virtual_memory', available)
    if failure == 'rss':
        monkeypatch.setattr(temporal, '_tree_rss', lambda *args: temporal.MAX_CHILD_RSS + 1)
    children = []
    original = temporal.subprocess.Popen
    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(temporal.subprocess, 'Popen', launch)
    count = 0
    def check():
        nonlocal count
        count += 1
        if failure == 'cancel' and count > 1:
            raise RuntimeError('operator cancelled')
    with (tmp_path / 'log').open('w') as log, pytest.raises(RuntimeError):
        temporal._run_supervised([sys.executable, '-c', 'import time;time.sleep(20)'], check, 0 if failure == 'timeout' else 3, log)
    assert len(children) == 1
    assert children[0].poll() is not None


@pytest.mark.parametrize('crash', [False, True])
def test_supervisor_cleans_owned_descendant_without_signalling_test_group(monkeypatch, tmp_path, crash):
    monkeypatch.setattr(temporal.psutil, 'virtual_memory', lambda: SimpleNamespace(available=temporal.runtime.MIN_AVAILABLE_RAM * 2))
    marker = tmp_path / 'grandchild.pid'
    script = "import subprocess,sys,time,os;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);Path(sys.argv[1]).write_text(str(p.pid));time.sleep(.4);" + ('os._exit(7)' if crash else 'time.sleep(20)')
    def check():
        if not crash and marker.exists() and time.monotonic() - started > .25:
            raise RuntimeError('operator cancelled')
    started = time.monotonic()
    with (tmp_path / 'log').open('w') as log:
        if crash:
            status, _ = temporal._run_supervised([sys.executable, '-c', script, str(marker)], check, 3, log)
            assert status == 7
        else:
            with pytest.raises(RuntimeError, match='cancelled'):
                temporal._run_supervised([sys.executable, '-c', script, str(marker)], check, 3, log)
    pid = int(marker.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_supervised_request_failure_has_no_partial_result(monkeypatch, tmp_path):
    monkeypatch.setattr(temporal.psutil, 'virtual_memory', lambda: SimpleNamespace(available=temporal.runtime.MIN_AVAILABLE_RAM * 2))
    with pytest.raises(ValueError, match='finite local video'):
        temporal.temporal_candidates(tmp_path / 'missing.mp4', 5, {'query': 'test'}, tmp_path / 'result', noop)
    assert not (tmp_path / 'result/temporal-search.json').exists()
    assert not list((tmp_path / 'result').glob('.temporal-stage-*'))


def test_actual_pinned_tokenizer_without_weights():
    from clipstudio import config
    location = config.DATA / 'models' / temporal.MODEL_ID / '649f3c91b59cd24be316dc505e26eacf5cd00801'
    if not (location / 'tokenizer.json').is_file():
        pytest.skip('Optional pinned tokenizer metadata is not installed.')
    inputs, metadata = temporal.tokenize_query(location, 'A person opens a door.')
    assert inputs['input_ids'][0, 0] == 49406
    assert inputs['input_ids'][0, -1] == 49407
    assert inputs['attention_mask'][0, -1] == 0
    assert metadata['query_truncated'] is False
    _, metadata = temporal.tokenize_query(location, 'door ' * 100)
    assert metadata['query_truncated'] is True
