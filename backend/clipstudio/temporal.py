"""Offline X-CLIP temporal embedding retrieval over bounded eight-frame clips.

The exported towers retain learned cross-frame attention and temporal position
embeddings. They omit the full classifier's video-conditioned text prompts.
Cosines are retrieval evidence, never event detections or probabilities.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import psutil

from .models import CATALOG, ModelUnavailable, model_path
from . import runtime
from .vision import _decode_frame, _video_extent, _guard, MAX_DECODE_EDGE

MODEL_ID = 'xclip-b16-k600'
METHOD = 'local_xclip_temporal_retrieval'
MAX_WINDOWS = 60
FRAMES_PER_WINDOW = 8
MAX_SOURCE_SECONDS = 12 * 60 * 60
MAX_WALL_SECONDS = 600
MAX_CHILD_RSS = 1280 * 1024 * 1024
POLL_SECONDS = 0.1
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TemporalResourceLimit(RuntimeError):
    def __init__(self, message, stats):
        super().__init__(message)
        self.stats = stats


def preprocess_image(image):
    """Pinned upstream VideoMAE processor: RGB/bilinear/center224/ImageNet."""
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > 40_000_000 or max(width, height) / min(width, height) > 32:
        raise ValueError('Frame dimensions exceed temporal preprocessing limits.')
    size = (224, int(224 * height / width)) if width < height else (int(224 * width / height), 224)
    resized = image.convert('RGB').resize(size, Image.Resampling.BILINEAR)
    left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
    values = np.asarray(resized.crop((left, top, left + 224, top + 224)), dtype=np.float32) / np.float32(255)
    return np.ascontiguousarray(((values - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def tokenize_query(location, query):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(location / 'tokenizer.json'))
    if (tokenizer.token_to_id('<|startoftext|>'), tokenizer.token_to_id('<|endoftext|>')) != (49406, 49407):
        raise ModelUnavailable('Pinned X-CLIP tokenizer special tokens are incompatible.')
    original_count = len(tokenizer.encode(query).ids)
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=49407, pad_token='<|endoftext|>')
    encoded = tokenizer.encode(query)
    inputs = {name: np.asarray([values], dtype=np.int64) for name, values in
              [('input_ids', encoded.ids), ('attention_mask', encoded.attention_mask)]}
    if any(value.shape != (1, 77) for value in inputs.values()) or encoded.ids[0] != 49406 or encoded.ids[-1] != 49407:
        raise ModelUnavailable('Pinned X-CLIP tokenizer did not produce the expected 77-token input.')
    return inputs, {'tokens_before_truncation': original_count, 'token_limit': 77, 'query_truncated': original_count > 77}


def _unit_vector(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape not in ((1, 512), (512,)) or not np.isfinite(array).all():
        raise ValueError('X-CLIP returned an invalid/non-finite 512-dimensional embedding.')
    vector = array.reshape(512)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('X-CLIP returned a zero or invalid embedding.')
    return vector / norm


def _session(path, inputs, output):
    import onnxruntime as ort
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    settings.enable_cpu_mem_arena = False
    settings.enable_mem_pattern = False
    # Avoid persistent packed copies of this FP32 model's large weight matrices.
    settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    settings.add_session_config_entry('session.disable_prepacking', '1')
    try:
        session = ort.InferenceSession(str(path), sess_options=settings, providers=['CPUExecutionProvider'])
    except Exception as exc:
        raise ModelUnavailable(f'Pinned X-CLIP model is incompatible with the installed CPU runtime: {exc}') from exc
    actual = {item.name: (item.type, item.shape) for item in session.get_inputs()}
    expected = {name: (dtype, shape) for name, dtype, shape in inputs}
    outputs = {item.name: (item.type, item.shape) for item in session.get_outputs()}
    if actual != expected or outputs.get(output) != ('tensor(float)', [1, 512]):
        raise ModelUnavailable('Pinned X-CLIP graph input/output names, dtypes or shapes are incompatible.')
    return session


class TemporalEncoder:
    """One text tower followed by one video tower; run under a memory supervisor."""

    def __init__(self, location, check):
        self.location, self.check = Path(location), check
        self.video = None
        self.query_vector = None

    def encode_query(self, query):
        if self.video is not None:
            raise RuntimeError('Encode the query before loading the temporal video tower.')
        self.check()
        inputs, metadata = tokenize_query(self.location, query)
        session = _session(self.location / 'text_tower.onnx',
                           [('input_ids', 'tensor(int64)', [1, 77]), ('attention_mask', 'tensor(int64)', [1, 77])], 'text_embeds')
        try:
            self.check()
            self.query_vector = _unit_vector(session.run(['text_embeds'], inputs)[0])
            self.check()
            return metadata
        finally:
            del session
            gc.collect()

    def encode_video(self, values):
        self.check()
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (1, 8, 3, 224, 224) or not np.isfinite(array).all():
            raise ValueError('Temporal input must contain eight finite normalized RGB frames.')
        if self.query_vector is None:
            raise RuntimeError('Encode the query before the video.')
        if self.video is None:
            self.video = _session(self.location / 'video_tower.onnx', [('pixel_values', 'tensor(float)', [1, 8, 3, 224, 224])], 'video_embeds')
        self.check()
        vector = _unit_vector(self.video.run(['video_embeds'], {'pixel_values': array})[0])
        self.check()
        return vector

    def score(self, values):
        vector = self.encode_video(values)
        return float(np.clip(np.dot(vector, self.query_vector), -1.0, 1.0))

    def close(self):
        self.video = None
        self.query_vector = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def plan_windows(start, end, minimum, maximum, exclusions):
    """Uniform full-extent placement; fixed midpoint length, no content ranking."""
    if not all(math.isfinite(value) for value in (start, end, minimum, maximum)) or not 0 <= start < end <= MAX_SOURCE_SECONDS:
        raise ValueError('Temporal retrieval supports finite video extents up to 12 hours.')
    if not 0.5 <= minimum <= maximum <= 300:
        raise ValueError('Temporal clip lengths must satisfy 0.5 <= min <= max <= 300 seconds.')
    for item in exclusions:
        if not all(math.isfinite(float(item[key])) for key in ('start', 'end')) or not 0 <= item['start'] < item['end'] <= MAX_SOURCE_SECONDS:
            raise ValueError('Temporal exclusion intervals must be finite source intervals.')
    extent = end - start
    if extent < minimum:
        return [], {'window_length_seconds': extent, 'planned_windows': 0, 'excluded_windows': 0, 'window_cap_reached': False}
    length = min(extent, (minimum + maximum) / 2)
    desired_count = max(1, math.ceil((extent - length) / (length / 2)) + 1)
    count = min(MAX_WINDOWS, desired_count)
    starts = np.linspace(start, end - length, count).tolist() if count > 1 else [start]
    windows = [{'start': at, 'end': min(end, at + length)} for at in starts]
    kept = [window for window in windows if not any(window['start'] < float(item['end']) and window['end'] > float(item['start']) for item in exclusions)]
    return kept, {'window_length_seconds': length, 'planned_windows': count, 'excluded_windows': count - len(kept),
                  'uncapped_window_count': desired_count, 'window_cap_reached': desired_count > MAX_WINDOWS,
                  'nominal_window_start_interval_seconds': (extent - length) / (count - 1) if count > 1 else 0.0}


def window_frame_times(window, extent):
    # Include the beginning and leave a frame-period margin before the end.
    # Requested timestamps are only seeks; returned evidence records decoded PTS.
    margin = max(0.001, float(extent['reported_frame_period_seconds']) + 0.001)
    last = max(window['start'], window['end'] - margin)
    return np.linspace(window['start'], last, FRAMES_PER_WINDOW).tolist()


def _coverage(windows, extent):
    merged = []
    for window in sorted(windows, key=lambda item: item['start']):
        start, end = window['start'], window['end']
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    seconds = sum(end - start for start, end in merged)
    gaps, cursor = [], extent['start']
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start])
        cursor = end
    if cursor < extent['end']:
        gaps.append([cursor, extent['end']])
    return {'sampled_window_intervals': merged, 'sampled_window_seconds': seconds,
            'sampled_window_coverage_fraction': seconds / (extent['end'] - extent['start']),
            'unsampled_intervals': gaps, 'maximum_unsampled_gap_seconds': max((end - start for start, end in gaps), default=0.0)}


def _validate_paths(source, out):
    if source.resolve().is_relative_to(out.resolve()):
        raise ValueError('Temporal output must be a separate directory that does not contain the source.')
    if any(path.exists() or path.is_symlink() for path in (out / 'temporal-frames', out / 'temporal-search.json', out / 'temporal-resource-failure.json')):
        raise ValueError('Temporal output already contains artifacts; use a fresh output directory.')


def _temporal_candidates(source: Path, duration: float, options: dict, out: Path, check):
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS

    def checked():
        check()
        if time.monotonic() >= deadline:
            raise RuntimeError('Temporal retrieval exceeded its 600-second time budget.')

    checked()
    source, out = Path(source), Path(out)
    if not source.is_file() or not math.isfinite(duration) or not 0 < duration <= MAX_SOURCE_SECONDS:
        raise ValueError('Temporal retrieval requires a finite local video of at most 12 hours.')
    _validate_paths(source, out)
    query = options.get('query', '')
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError('Temporal retrieval requires a nonblank query of at most 2,000 characters.')
    query = query.strip()
    exclusions = options.get('exclusions', [])
    if any(not 0 <= float(item['start']) < float(item['end']) <= duration for item in exclusions):
        raise ValueError('Temporal exclusions must be inside the source.')
    _, extent = _video_extent(source, duration, checked, deadline)
    windows, planning = plan_windows(extent['start'], extent['end'], float(options.get('min_length', 4)), float(options.get('max_length', 10)), exclusions)
    frame_period = extent['reported_frame_period_seconds']
    if windows and planning['window_length_seconds'] < 2 * frame_period - 0.000001:
        raise ValueError(f'The chosen temporal window is too short for this video’s reported frame cadence ({frame_period:.3f}s per frame). Increase both clip lengths to at least {2 * frame_period:.2f}s or use footage with a higher frame rate.')
    location = model_path(MODEL_ID, verify=True)
    checked()
    frames_dir = out / 'temporal-frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    candidates, records = [], []
    token_details = {}
    with TemporalEncoder(location, checked) as encoder:
        if windows:
            token_details = encoder.encode_query(query)
        for index, window in enumerate(windows):
            checked()
            values = np.empty((1, FRAMES_PER_WINDOW, 3, 224, 224), dtype=np.float32)
            evidence = []
            for frame_index, desired in enumerate(window_frame_times(window, extent)):
                target = frames_dir / f'window-{index:03d}-frame-{frame_index}.jpg'
                pts = _decode_frame(source, desired, target, checked, deadline)
                if pts is None:
                    raise RuntimeError('Temporal frame sampling returned no frame; search coverage is incomplete.')
                actual = pts - extent['format_start_time']
                if not window['start'] - 0.001 <= actual < window['end']:
                    raise RuntimeError('Temporal decoder returned a frame outside its context window.')
                if evidence and actual < evidence[-1]['time'] - 0.000001:
                    raise RuntimeError('Temporal decoder returned non-monotonic frame timestamps.')
                with Image.open(target) as image:
                    if max(image.size) > MAX_DECODE_EDGE:
                        raise ValueError('Decoder failed to bound temporal frame dimensions.')
                    values[0, frame_index] = preprocess_image(image)
                evidence.append({'kind': 'temporal_frame', 'time': round(actual, 6), 'start': round(actual, 6), 'end': round(actual, 6),
                                 'requested_time': round(desired, 6), 'ref': str(target.relative_to(out))})
            score = encoder.score(values)
            checked()
            repeated = FRAMES_PER_WINDOW - len({item['time'] for item in evidence})
            record = {**window, 'cosine_similarity': score, 'frames': evidence, 'repeated_decoded_frames': repeated}
            records.append(record)
            candidates.append({**window, 'title': 'Temporal match: ' + query[:96], 'method': METHOD,
                               'rationale': 'X-CLIP cross-frame attention and temporal transformer encode these eight chronological source frames; cosine ranks the resulting video embedding against the query.',
                               'scores': {'ranking': score, 'temporal_similarity': score}, 'evidence': evidence,
                               'uncertainty': ['A retrieval score is not a calibrated event probability or proof that the requested event occurred.',
                                               'Eight sampled frames and center cropping can miss brief actions, small subjects and events between windows.',
                                               'The two-tower export omits the full classifier’s video-conditioned text prompts.']})
    checked()
    spec = CATALOG[MODEL_ID]
    metadata = {'model': spec['repository'], 'revision': spec['revision'], 'model_id': MODEL_ID, 'query': query,
                'method': METHOD, 'score': 'raw cosine between learned temporal video and text embeddings',
                'learned_temporal_encoder': True, 'full_classifier': False, **token_details, **planning,
                'source_duration_seconds': duration, 'video_extent': extent, 'window_cap': MAX_WINDOWS,
                'window_count': len(records), 'frames_per_window': FRAMES_PER_WINDOW, 'sample_count': len(records) * FRAMES_PER_WINDOW,
                'sampling_complete': len(records) == len(windows), **_coverage(records, extent),
                'sample_time_basis': 'decoded frame PTS reported by FFmpeg showinfo/copyts, minus format start time',
                'cpu_threads': 2, 'batch_size': 1, 'tower_loading': 'text released before video',
                'frame_decode_max_edge': MAX_DECODE_EDGE, 'wall_time_budget_seconds': MAX_WALL_SECONDS,
                'elapsed_seconds': round(time.monotonic() - started, 3), 'windows': records, 'paid_usage': 0, 'data_transmitted': False}
    limitations = ['Local X-CLIP uses learned temporal attention over eight sampled frames per window; retrieval similarity does not verify events, identities, causes, winners or factual claims.',
                   'This third-party ONNX export omits the original classifier’s video-conditioned text prompt generator.',
                   f'Inspected {len(records)} windows with eight frames each, covering {metadata["sampled_window_coverage_fraction"]:.1%} of video time as sparse windows. Events between frames or windows can be missed.',
                   'The 60-window budget is distributed across the video extent. Long recordings are sampled sparsely; support for 12-hour input is not an exhaustive analysis or throughput claim.',
                   'Frames are reduced to at most 960px, then bilinear-resized and center-cropped to 224px. Small or off-center events may be lost.',
                   'Raw cosine scores are uncalibrated; negative scores and unrelated top results remain possible. Review the original source.']
    if token_details.get('query_truncated'):
        limitations.append('The query exceeded the 77-token text context and was truncated.')
    if any(record['repeated_decoded_frames'] for record in records):
        limitations.append('Sparse source frame rates caused repeated frames inside one or more eight-frame model inputs; see recorded frame timestamps.')
    _atomic_json(out / 'temporal-search.json', metadata)
    return sorted(candidates, key=lambda item: (-item['scores']['ranking'], item['start'], item['end'])), metadata, limitations


def _terminate_owned_tree(process, remembered=(), identity=None):
    """Terminate only our child and its descendants, never the task's group."""
    try:
        children = identity.children(recursive=True) if identity is not None and identity.is_running() else []
    except psutil.NoSuchProcess:
        children = []
    children = list({child.pid: child for child in [*remembered, *children]}.values())
    if process.poll() is None:
        process.terminate()
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    _, alive = psutil.wait_procs(children, timeout=1)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def _tree_rss(process, remembered=None, identity=None):
    try:
        if identity is None or not identity.is_running():
            return 0
        processes = [identity, *identity.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    if remembered is not None:
        remembered.update({child.pid: child for child in processes[1:]})
    for item in processes:
        try:
            total += item.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def _run_supervised(command, check, timeout, output):
    """Bound even native session construction/run; fail without automatic retry."""
    check()
    available = psutil.virtual_memory().available
    if available < runtime.MIN_AVAILABLE_RAM:
        raise TemporalResourceLimit('Temporal inference blocked: less than the required 1.5 GiB system RAM is available.',
                                    {'minimum_system_available_bytes': available, 'minimum_available_ram_bytes': runtime.MIN_AVAILABLE_RAM,
                                     'peak_child_tree_rss_bytes': 0, 'child_tree_rss_limit_bytes': MAX_CHILD_RSS, 'child_started': False})
    started = time.monotonic()
    peak_rss, minimum_available = 0, available
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                               env=env, **_guard())
    try:
        identity = psutil.Process(process.pid)
        identity.create_time()  # Cache PID+creation identity before any poll/reap.
    except psutil.NoSuchProcess:
        identity = None
    remembered = {}
    def stats():
        return {'peak_child_tree_rss_bytes': peak_rss, 'minimum_system_available_bytes': minimum_available,
                'child_tree_rss_limit_bytes': MAX_CHILD_RSS, 'minimum_available_ram_bytes': runtime.MIN_AVAILABLE_RAM,
                'supervisor_poll_seconds': POLL_SECONDS, 'supervised_elapsed_seconds': round(time.monotonic() - started, 3), 'child_started': True}
    try:
        while True:
            check()
            available = psutil.virtual_memory().available
            rss = _tree_rss(process, remembered, identity)
            peak_rss, minimum_available = max(peak_rss, rss), min(minimum_available, available)
            if rss > MAX_CHILD_RSS:
                raise TemporalResourceLimit('Temporal inference stopped: its owned process tree exceeded the 1.25 GiB RAM limit.', stats())
            if available < runtime.MIN_AVAILABLE_RAM:
                raise TemporalResourceLimit('Temporal inference stopped: system available RAM fell below the required 1.5 GiB reserve.', stats())
            if time.monotonic() - started >= timeout:
                raise TemporalResourceLimit(f'Temporal inference exceeded its {timeout:g}-second supervised time budget.', stats())
            if process.poll() is not None:
                break
            time.sleep(POLL_SECONDS)
        check()
        return process.returncode, stats()
    finally:
        # Also collect descendants after crashes or cancellation. On Linux each
        # stage/decode child carries its own parent-death signal as a backstop.
        _terminate_owned_tree(process, remembered.values(), identity)


def _atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.temporal-json-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _supervised_request(request, out, check, timeout):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.temporal-stage-', dir=out) as temporary:
        directory = Path(temporary)
        input_path, result_path = directory / 'input.json', directory / 'result.json'
        _atomic_json(input_path, request)
        with (directory / 'child.log').open('w+') as log:
            try:
                status, stats = _run_supervised([sys.executable, '-m', 'clipstudio.temporal', str(input_path), str(result_path)], check, timeout, log)
            except TemporalResourceLimit as exc:
                _atomic_json(out / 'temporal-resource-failure.json', {'error': str(exc), **exc.stats})
                raise
            check()
            if not result_path.is_file() or result_path.stat().st_size > 4_000_000:
                raise RuntimeError(f'Temporal subprocess exited with status {status} without a bounded result.')
            result = json.loads(result_path.read_text())
        if status or result.get('error'):
            message = result.get('error', f'Temporal subprocess exited with status {status}.')
            error = ModelUnavailable if result.get('error_type') == 'ModelUnavailable' else ValueError if result.get('error_type') == 'ValueError' else RuntimeError
            raise error(message)
        return result['value'], stats


def temporal_candidates(source: Path, duration: float, options: dict, out: Path, check):
    """Production entry point: all model loading/inference stays supervised."""
    _validate_paths(Path(source), Path(out))
    result, stats = _supervised_request({'source': str(Path(source).resolve()), 'duration': duration,
                                        'options': options, 'out': str(Path(out).resolve())}, out, check, MAX_WALL_SECONDS)
    candidates, metadata, limitations = result
    metadata.update(stats)
    metadata['budget_enforcement'] = 'owned child-tree RSS and system RAM reserve checked every 100ms; supervisor enforces wall deadline including native model calls'
    check()
    _atomic_json(Path(out) / 'temporal-search.json', metadata)
    return candidates, metadata, limitations


def _trial(location, query):
    """A shape/finite-output trial, not a semantic-quality benchmark."""
    started = time.monotonic()
    checks = lambda: None
    with TemporalEncoder(location, checks) as encoder:
        metadata = encoder.encode_query(query)
        text_done = time.monotonic()
        yy, xx = np.indices((224, 224))
        values = np.stack([preprocess_image(Image.fromarray(np.stack(((xx + frame * 19) % 256, (yy + frame * 7) % 256,
                             (xx + yy + frame * 13) % 256), axis=-1).astype(np.uint8))) for frame in range(8)])[None]
        vector = encoder.encode_video(values)
        forward_done = time.monotonic()
        reversed_vector = encoder.encode_video(values[:, ::-1].copy())
        return {'trial_only': True, 'semantic_accuracy_tested': False, 'text_seconds': text_done - started,
                'video_load_and_first_inference_seconds': forward_done - text_done,
                'second_video_inference_seconds': time.monotonic() - forward_done,
                'forward_vector': vector.tolist(), 'reversed_vector': reversed_vector.tolist(),
                'order_difference_l2': float(np.linalg.norm(vector - reversed_vector)),
                'query_cosine': float(np.dot(vector, encoder.query_vector)), **metadata}


def guarded_trial(location, out, query='A person opens a door.'):
    """Initial 90-second CPU feasibility test with the production RAM limits."""
    return _supervised_request({'trial_location': str(Path(location).resolve()), 'query': query}, out, lambda: None, 90)


def _main():
    request = json.loads(Path(sys.argv[1]).read_text())
    target = Path(sys.argv[2])
    try:
        if 'trial_location' in request:
            value = _trial(Path(request['trial_location']), request['query'])
        else:
            value = _temporal_candidates(Path(request['source']), request['duration'], request['options'], Path(request['out']), lambda: None)
        _atomic_json(target, {'value': value})
        return 0
    except Exception as exc:
        _atomic_json(target, {'error': str(exc), 'error_type': type(exc).__name__})
        return 1


if __name__ == '__main__':
    raise SystemExit(_main())
