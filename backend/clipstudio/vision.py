"""Offline CLIP image/text retrieval over bounded, full-source frame samples.

Scores measure visual similarity to the operator's text. They are neither event
recognition nor calibrated probabilities; no motion or audio score is mixed in.
"""
from __future__ import annotations

import bisect
import ctypes
import gc
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from .models import CATALOG, ModelUnavailable, model_path

MODEL_ID = 'clip-vit-b32'
MAX_SAMPLES = 900
DEFAULT_SAMPLE_SECONDS = 2.0
MAX_SOURCE_SECONDS = 12 * 60 * 60
MAX_WALL_SECONDS = 600
FRAME_TIMEOUT_SECONDS = 20
MAX_DECODE_EDGE = 960
SOURCE_FLAGS = ['-protocol_whitelist', 'file,pipe', '-format_whitelist',
                'mov,matroska,webm,avi,wav,mp3,ogg,flac,aac,asf,mpeg,mpegts,aiff']
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_LIBC = ctypes.CDLL(None) if sys.platform.startswith('linux') else None


def _die_with_stage(parent):
    _LIBC.prctl(1, signal.SIGKILL)
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)


def _guard():
    from functools import partial
    return {'preexec_fn': partial(_die_with_stage, os.getpid())} if _LIBC is not None else {}


def sample_times(duration):
    if not math.isfinite(duration) or not 0 < duration <= MAX_SOURCE_SECONDS:
        raise ValueError('Visual retrieval supports finite sources up to 12 hours.')
    count = min(MAX_SAMPLES, max(1, math.ceil(duration / DEFAULT_SAMPLE_SECONDS) + 1))
    # Keep the last seek inside the file. Decode has one explicit, reported
    # near-tail fallback for a final timestamp with no decodable video frame.
    tail = max(0.0, duration - min(0.25, duration / 2))
    return np.linspace(0.0, tail, count).tolist() if count > 1 else [duration / 2]


def preprocess_image(image):
    """Pinned CLIP: RGB, shortest edge 224/bicubic, center crop, rescale/normalize."""
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > 40_000_000 or max(width, height) / min(width, height) > 32:
        raise ValueError('Frame dimensions/aspect ratio exceed the bounded visual preprocessing limits.')
    rgb = image.convert('RGB')
    if width < height:
        size = (224, int(224 * height / width))
    else:
        size = (int(224 * width / height), 224)
    resized = rgb.resize(size, Image.Resampling.BICUBIC)
    left, top = (size[0] - 224) // 2, (size[1] - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    values = np.asarray(cropped, dtype=np.float32) / np.float32(255)
    return np.ascontiguousarray(((values - MEAN) / STD).transpose(2, 0, 1)[None], dtype=np.float32)


def tokenize_query(location, query):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(location / 'tokenizer.json'))
    bos = tokenizer.token_to_id('<|startoftext|>')
    eos = tokenizer.token_to_id('<|endoftext|>')
    if (bos, eos) != (49406, 49407):
        raise ModelUnavailable('Pinned CLIP tokenizer special tokens are incompatible.')
    original_count = len(tokenizer.encode(query).ids)
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=eos, pad_token='<|endoftext|>')
    encoded = tokenizer.encode(query)
    ids = np.asarray([encoded.ids], dtype=np.int64)
    if ids.shape != (1, 77) or ids[0, 0] != bos or ids[0, -1] != eos:
        raise ModelUnavailable('Pinned CLIP tokenizer did not produce the expected 77-token input.')
    return ids, {'tokens_before_truncation': original_count, 'token_limit': 77, 'query_truncated': original_count > 77}


def _session(path, input_name, output_name):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    except Exception as exc:
        raise ModelUnavailable(f'Pinned CLIP model is incompatible with the installed CPU runtime: {exc}') from exc
    if {entry.name for entry in session.get_inputs()} != {input_name} or output_name not in {entry.name for entry in session.get_outputs()}:
        raise ModelUnavailable('Pinned CLIP graph input/output names are incompatible.')
    return session


def _unit_vector(value):
    array = np.asarray(value, dtype=np.float32)
    if array.shape not in ((1, 512), (512,)) or not np.isfinite(array).all():
        raise ValueError('CLIP returned an invalid/non-finite 512-dimensional embedding.')
    vector = array.reshape(512)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('CLIP returned a zero or invalid embedding.')
    return vector / norm


def _query_embedding(location, query, check):
    check()
    ids, details = tokenize_query(location, query)
    session = _session(location / 'onnx/text_model_int8.onnx', 'input_ids', 'text_embeds')
    try:
        check()
        vector = _unit_vector(session.run(['text_embeds'], {'input_ids': ids})[0])
        check()
        return vector, details
    finally:
        del session
        gc.collect()


def _duration_tag_seconds(value):
    # FFmpeg's Matroska/WebM DURATION tag records the stream end timestamp,
    # including a delayed stream start; it is not always the stream's length.
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'(\d{1,6}):([0-5]\d):([0-5]\d(?:\.\d{1,9})?)', value.strip())
    if not match:
        return None
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def _video_extent(source, duration, check, deadline):
    check()
    remaining = min(FRAME_TIMEOUT_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise RuntimeError('Visual retrieval exceeded its 600-second time budget.')
    result = subprocess.run(['ffprobe', '-v', 'error', *SOURCE_FLAGS, '-select_streams', 'v:0',
                             '-show_entries', 'stream=duration,start_time,avg_frame_rate,width,height:stream_tags=DURATION:format=start_time',
                             '-of', 'json', str(source)], stdin=subprocess.DEVNULL, capture_output=True,
                            timeout=remaining, check=True, **_guard())
    check()
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise ValueError('Visual retrieval requires a video stream.')
    stream = data['streams'][0]
    if int(stream.get('width', 0)) * int(stream.get('height', 0)) > 40_000_000:
        raise ValueError('Video frame dimensions exceed the 40-megapixel decode limit.')
    def finite_number(value, default):
        try:
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default
    origin = finite_number(data.get('format', {}).get('start_time'), 0.0)
    first = max(0.0, finite_number(stream.get('start_time'), origin) - origin)
    length = finite_number(stream.get('duration'), 0.0)
    tag_end = _duration_tag_seconds(next((value for key, value in stream.get('tags', {}).items() if key.upper() == 'DURATION'), None))
    if length > 0:
        last = min(duration, first + length)
        duration_basis = 'stream_duration'
    elif tag_end is not None and first < tag_end - origin <= duration + 0.001:
        last = min(duration, tag_end - origin)
        duration_basis = 'stream_duration_tag_end_timestamp'
    else:
        last = duration
        duration_basis = 'container_duration_fallback'
    if not 0 <= first < last <= duration:
        raise ValueError('Video stream has no finite extent inside the source.')
    try:
        fps = float(Fraction(stream.get('avg_frame_rate', '0/1')))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    frame_period = 1 / fps if math.isfinite(fps) and fps > 0 else 0.25
    count = min(MAX_SAMPLES, max(1, math.ceil((last - first) / DEFAULT_SAMPLE_SECONDS) + 1))
    tail = max(first, last - max(0.25, frame_period + 0.001))
    times = np.linspace(first, tail, count).tolist() if tail > first and count > 1 else [first]
    return times, {'start': first, 'end': last, 'format_start_time': origin, 'reported_frame_period_seconds': frame_period, 'duration_basis': duration_basis}


def _decode_frame(source, timestamp, target, check, deadline):
    check()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError('Visual retrieval exceeded its 600-second time budget.')
    started = time.monotonic()
    target.unlink(missing_ok=True)
    # FFmpeg autorotation is enabled by default. Bounding the decoded thumbnail
    # before PIL limits memory; this extra 960px reduction is disclosed below.
    args = ['ffmpeg', '-hide_banner', '-v', 'info', '-nostdin', '-y', '-threads', '1',
            '-ss', str(timestamp), '-copyts', *SOURCE_FLAGS, '-i', str(source), '-map', '0:v:0',
            '-frames:v', '1', '-an', '-sn', '-dn', '-filter_threads', '1',
            '-vf', f'showinfo,scale=w=min({MAX_DECODE_EDGE}\\,iw):h=min({MAX_DECODE_EDGE}\\,ih):force_original_aspect_ratio=decrease',
            '-q:v', '2', '-threads', '1', '-fps_mode', 'passthrough', '-update', '1', str(target)]
    with tempfile.TemporaryFile() as error:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error, **_guard())
        try:
            while process.poll() is None:
                check()
                if time.monotonic() >= deadline or time.monotonic() - started >= FRAME_TIMEOUT_SECONDS:
                    raise RuntimeError('Visual frame decode exceeded its bounded time budget.')
                time.sleep(0.02)
            check()
            error.seek(0)
            diagnostic = error.read(65536).decode('utf-8', 'replace')
            if process.returncode:
                raise RuntimeError('Visual frame decode failed: ' + diagnostic[-2000:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    if not target.is_file() or target.stat().st_size == 0:
        return None
    if target.stat().st_size > 4_000_000:
        raise ValueError('Decoded frame exceeded its bounded JPEG size.')
    match = re.search(r'n:\s*0\s+pts:\s*[-0-9]+\s+pts_time:([-+0-9.eE]+)', diagnostic)
    if not match or not math.isfinite(float(match.group(1))):
        raise RuntimeError('Decoder did not report a finite source timestamp for its frame.')
    return float(match.group(1))


def _window_candidates(samples, duration, minimum, maximum, query, exclusions, check, source_start=0.0):
    """Mean cosine of a continuous piecewise-linear signal, never motion energy."""
    if not samples:
        return []
    times = [float(item['time']) for item in samples]
    scores = [float(item['cosine_similarity']) for item in samples]
    if any(not math.isfinite(v) for v in times + scores) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Visual sample times/scores must be finite and strictly ordered.')
    if times[0] < source_start or times[-1] > duration or any(not -1.000001 <= score <= 1.000001 for score in scores):
        raise ValueError('Visual samples lie outside source/similarity bounds.')
    x, y = list(times), list(scores)
    if x[0] > source_start:
        x.insert(0, source_start); y.insert(0, y[0])
    if x[-1] < duration:
        x.append(duration); y.append(y[-1])
    prefix = [0.0]
    for index in range(len(x) - 1):
        prefix.append(prefix[-1] + (x[index + 1] - x[index]) * (y[index] + y[index + 1]) / 2)

    def integral(at):
        index = min(len(x) - 2, max(0, bisect.bisect_right(x, at) - 1))
        delta = at - x[index]
        slope = (y[index + 1] - y[index]) / (x[index + 1] - x[index])
        return prefix[index] + y[index] * delta + slope * delta * delta / 2

    lengths = sorted({minimum, (minimum + maximum) / 2, min(maximum, duration)})
    candidates, seen = [], set()
    for sample_time in times:
        check()
        for length in lengths:
            if length < minimum or length > duration - source_start:
                continue
            start = max(source_start, min(sample_time - length / 2, duration - length))
            end = start + length
            identity = (round(start, 6), round(end, 6))
            if identity in seen:
                continue
            seen.add(identity)
            if any(max(start, item['start']) < min(end, item['end']) for item in exclusions):
                continue
            mean = max(-1.0, min(1.0, (integral(end) - integral(start)) / length))
            first = max(0, bisect.bisect_left(times, start) - 1)
            last = min(len(times), bisect.bisect_right(times, end) + 1)
            evidence = [{'kind': 'visual_frame', 'time': frame['time'], 'start': frame['time'], 'end': frame['time'],
                         'cosine_similarity': frame['cosine_similarity'], 'ref': frame['frame_ref']} for frame in samples[first:last]]
            candidates.append({'start': round(start, 3), 'end': round(end, 3), 'title': 'Visual match: ' + query[:96],
                               'rationale': 'Mean CLIP image/text cosine over this context window, interpolated between the recorded source-frame samples.',
                               'method': 'local_clip_visual', 'scores': {'ranking': round(mean, 6), 'visual_similarity': round(mean, 6)},
                               'evidence': evidence, 'uncertainty': ['Similarity is not a probability or proof that the requested event occurred.',
                                   'Sparse still frames cannot establish action order, overtakes, crashes, winners, identity or missing context.']})
    check()
    return sorted(candidates, key=lambda item: (-item['scores']['ranking'], item['start'], item['end']))


def visual_candidates(source: Path, duration: float, options: dict, out: Path, check):
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    check()
    query = options.get('query', '')
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError('Visual retrieval requires a nonblank query of at most 2,000 characters.')
    query = query.strip()
    minimum = float(options.get('min_length', 4))
    maximum = float(options.get('max_length', 10))
    if not all(math.isfinite(value) for value in (minimum, maximum)) or not 0.5 <= minimum <= maximum <= 300:
        raise ValueError('Visual clip lengths must satisfy 0.5 <= min <= max <= 300 seconds.')
    exclusions = options.get('exclusions', [])
    if any(not all(math.isfinite(float(item[key])) for key in ('start', 'end')) or not 0 <= item['start'] < item['end'] <= duration for item in exclusions):
        raise ValueError('Visual exclusion intervals must be finite and inside the source.')
    source = Path(source)
    if not source.is_file():
        raise ValueError('Visual retrieval requires a local media file.')
    sample_times(duration)  # Validate the shared source-duration safety bound.
    requested_times, extent = _video_extent(source, duration, check, deadline)
    location = model_path(MODEL_ID, verify=True)
    check()
    if time.monotonic() >= deadline:
        raise RuntimeError('Visual retrieval exceeded its 600-second time budget during model verification.')
    query_vector, token_details = _query_embedding(location, query, check)
    check()
    if time.monotonic() >= deadline:
        raise RuntimeError('Visual retrieval exceeded its 600-second time budget during query encoding.')
    # Text tower was released before this vision tower is loaded.
    session = _session(location / 'onnx/vision_model_uint8.onnx', 'pixel_values', 'image_embeds')
    out.mkdir(parents=True, exist_ok=True)
    frames = out / 'visual-frames'
    frames.mkdir(exist_ok=True)
    samples, fallbacks, duplicate_samples = [], 0, 0
    try:
        for index, desired in enumerate(requested_times):
            check()
            target = frames / f'frame-{index:04d}.jpg'
            actual = desired
            timestamp = _decode_frame(source, actual, target, check, deadline)
            if timestamp is None:
                # A/V tails and sparse source frame rates can have no frame after
                # the nominal seek. One bounded earlier seek is recorded exactly.
                actual = max(0.0, desired - 1.0)
                if samples and actual <= samples[-1]['time']:
                    raise RuntimeError('No decodable frame at the requested source tail; sample coverage is incomplete.')
                timestamp = _decode_frame(source, actual, target, check, deadline)
                if timestamp is None:
                    raise RuntimeError('No decodable frame near the requested sample; sample coverage is incomplete.')
                fallbacks += 1
            actual = timestamp - extent['format_start_time']
            if not extent['start'] - 0.001 <= actual <= extent['end'] + 0.001:
                raise RuntimeError('Decoded frame timestamp lies outside the video stream extent.')
            actual = max(extent['start'], min(extent['end'], actual))
            if samples and actual <= samples[-1]['time'] + 1e-6:
                if actual < samples[-1]['time'] - 1e-6:
                    raise RuntimeError('Decoder returned non-monotonic source frame timestamps.')
                target.unlink(missing_ok=True)
                duplicate_samples += 1
                continue
            with Image.open(target) as image:
                if max(image.size) > MAX_DECODE_EDGE:
                    raise ValueError('Decoder failed to bound the visual thumbnail dimensions.')
                values = preprocess_image(image)
            check()
            vector = _unit_vector(session.run(['image_embeds'], {'pixel_values': values})[0])
            cosine = float(np.clip(np.dot(vector, query_vector), -1.0, 1.0))
            check()
            if time.monotonic() >= deadline:
                raise RuntimeError('Visual retrieval exceeded its 600-second time budget.')
            samples.append({'time': round(actual, 6), 'requested_time': round(desired, 6),
                            'cosine_similarity': round(cosine, 8), 'frame_ref': str(target.relative_to(out))})
    finally:
        del session
        gc.collect()
    candidates = _window_candidates(samples, extent['end'], minimum, maximum, query, exclusions, check, extent['start'])
    check()
    gaps = [b['time'] - a['time'] for a, b in zip(samples, samples[1:])]
    spec = CATALOG[MODEL_ID]
    metadata = {'model': spec['repository'], 'revision': spec['revision'], 'model_id': MODEL_ID,
                'query': query, **token_details, 'score': 'raw image/text cosine; context-window mean of piecewise-linear interpolation',
                'sample_count': len(samples), 'sample_cap': MAX_SAMPLES, 'source_duration_seconds': duration,
                'nominal_interval_seconds': max(gaps, default=extent['end'] - extent['start']), 'sampling_complete': len(samples) + duplicate_samples == len(requested_times),
                'sample_requests': len(requested_times), 'duplicate_frame_samples_skipped': duplicate_samples, 'video_extent': extent,
                'first_sample_seconds': samples[0]['time'], 'last_sample_seconds': samples[-1]['time'],
                'tail_unsampled_seconds': duration - samples[-1]['time'], 'tail_fallbacks': fallbacks,
                'sample_time_basis': 'decoded first-frame PTS reported by FFmpeg showinfo/copyts, minus format start time',
                'candidate_windows': len(candidates), 'cpu_threads': 2, 'batch_size': 1,
                'tower_loading': 'text released before vision', 'frame_decode_max_edge': MAX_DECODE_EDGE,
                'wall_time_budget_seconds': MAX_WALL_SECONDS, 'budget_enforcement': 'cooperative between model calls; decoder timeout and worker process cancellation', 'elapsed_seconds': round(time.monotonic() - started, 3),
                'samples': samples, 'paid_usage': 0, 'data_transmitted': False}
    limitations = ['Local CLIP ranks still-frame similarity to the query; it does not verify race events, action order, identity, winners or factual claims.',
                   f'Inspected {len(samples)} frames across the source, maximum gap {metadata["nominal_interval_seconds"]:.2f}s. Brief events between samples may be missed.',
                   'Scores are raw cosine similarities, not calibrated probabilities. Low or negative scores remain possible; review the source context.',
                   'Frames are reduced to at most 960px, then CLIP center-crops 224px; small or off-center subjects, text, negation and fine spatial relations can be missed.',
                   'Candidate scores average a linear interpolation between sampled frames; unsampled motion is not observed.']
    if extent['start'] > 0 or extent['end'] < duration:
        limitations.append(f'Visual retrieval covers the video stream from {extent["start"]:.3f}s to {extent["end"]:.3f}s; source regions without video are not assigned visual scores.')
    if token_details['query_truncated']:
        limitations.append('The query exceeded CLIP’s 77-token context and was truncated; shorten it for faithful retrieval.')
    (out / 'visual-search.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return candidates, metadata, limitations
