#!/usr/bin/env python3
"""Bounded local Sherpa diarization trial; no downloads, accounts or identities.

Run with the app Python (psutil); the model child uses the separately installed
Sherpa Python. Input must already be PCM16 mono 16-kHz WAV, at most 120 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import wave

RUNTIME_VERSION = '2.0.1'
RESERVE_BYTES = 1536 * 1024**2
MAX_RSS_BYTES = 1280 * 1024**2
MAX_SECONDS = 120.0
MAX_WAV_BYTES = 5 * 1024**2
MAX_JSON_BYTES = 2 * 1024**2
MAX_LOG_BYTES = 2 * 1024**2
MAX_TURNS = 10000
TIMESTAMP_TOLERANCE_SECONDS = .05
POLL_SECONDS = .05
MODEL_IDS = ('sherpa-pyannote-segmentation-3.0', 'sherpa-nemo-titanet-small')
HEX = re.compile(r'^[a-f0-9]{64}$')


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path, limit):
    """Open the final component without following links and reject size races."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
            raise ValueError('Input must be a bounded regular nonsymlink file.')
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        if len(raw) > limit or len(raw) != before.st_size or identity(before) != identity(after):
            raise ValueError('Input changed while being read or exceeded its byte cap.')
    return raw


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def file_digest(path, expected_size):
    if type(expected_size) is not int or not 0 < expected_size <= 64 * 1024**2:
        raise ValueError('Model byte size is outside the bounded manifest contract.')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    value, total = hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError('Model is not a regular file of its exact pinned size.')
        while block := handle.read(1024**2):
            value.update(block); total += len(block)
            if total > expected_size:
                raise ValueError('Model grew while hashing.')
        if total != expected_size or identity(before) != identity(os.fstat(handle.fileno())):
            raise ValueError('Model changed while hashing.')
    return value.hexdigest()


def wav_info(raw):
    if len(raw) > MAX_WAV_BYTES or len(raw) < 44 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('Expected a bounded RIFF PCM WAV file.')
    if int.from_bytes(raw[4:8], 'little') + 8 != len(raw):
        raise ValueError('RIFF size does not match the exact file length.')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as audio:
            frames = audio.getnframes()
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, 'NONE'):
                raise ValueError('Input must be PCM16, mono, 16000 Hz; convert a bounded derivative first.')
            if not 0 < frames <= int(16000 * MAX_SECONDS):
                raise ValueError('Input must contain more than zero and at most 120 seconds of audio.')
            pcm = audio.readframes(frames + 1)
            if len(pcm) != frames * 2:
                raise ValueError('WAV audio payload is truncated or inconsistent.')
    except (wave.Error, EOFError) as exc:
        raise ValueError('Malformed PCM WAV input.') from exc
    return {'sample_rate': 16000, 'channels': 1, 'sample_width_bytes': 2,
            'frames': frames, 'duration_seconds': frames / 16000,
            'bytes': len(raw), 'sha256': digest(raw)}, pcm


def load_plan(manifest_path):
    raw = regular_bytes(manifest_path, 256 * 1024)
    doc = json.loads(raw)
    if doc.get('runtime') != 'sherpa-onnx' or doc.get('version') != RUNTIME_VERSION:
        raise ValueError('Manifest must pin Sherpa-ONNX 2.0.1.')
    python = Path(doc['runtime_directory']) / 'bin/python'
    if not python.is_file():
        raise ValueError('The separately installed Sherpa Python is unavailable.')
    models = []
    for model_id in MODEL_IDS:
        matches = [item for item in doc.get('models', []) if item.get('id') == model_id]
        if len(matches) != 1:
            raise ValueError('Manifest must contain one exact segmentation and embedding model.')
        item = matches[0]
        if not HEX.fullmatch(str(item.get('sha256', ''))) or Path(item['filename']).name != item['filename']:
            raise ValueError('Model manifest requires a SHA-256 and plain filename.')
        path = Path(item['target_directory']) / item['filename']
        if file_digest(path, item['bytes']) != item['sha256']:
            raise ValueError('Model does not match its pinned SHA-256.')
        models.append({key: item[key] for key in ('id', 'repository', 'revision', 'filename', 'bytes', 'sha256', 'license')}
                      | {'path': str(path.resolve())})
    return {'python': str(python.absolute()), 'manifest_sha256': digest(raw), 'models': models}


def options(speakers=-1, threshold=.5):
    if type(speakers) is not int or speakers not in {-1, *range(1, 9)}:
        raise ValueError('Speakers must be -1 (unknown) or an explicit known count from 1 to 8.')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Clustering threshold must be finite and strictly between zero and one.')
    return {'provider': 'cpu', 'num_threads': 2, 'num_clusters': speakers,
            'speaker_count_mode': 'unknown' if speakers == -1 else 'explicit_known_count',
            'cluster_threshold': float(threshold), 'threshold_used': speakers == -1,
            'window_shift_ratio': .1, 'min_duration_on': .3, 'min_duration_off': .5}


def write_new(path, value):
    raw = encode(value)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('JSON evidence exceeds the bounded output limit.')
    with Path(path).open('xb') as handle:
        handle.write(raw + b'\n'); handle.flush(); os.fsync(handle.fileno())


def validate_turns(rows, duration):
    if not isinstance(rows, list) or len(rows) > MAX_TURNS:
        raise ValueError('Diarization turn output exceeds its bound.')
    result = []
    previous = -1.0
    for row in rows:
        start, end, speaker = row['start'], row['end'], row['speaker_index']
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end)):
            raise ValueError('Turn times must be finite numbers.')
        if not 0 <= start < end <= duration + TIMESTAMP_TOLERANCE_SECONDS or start < previous:
            raise ValueError('Turn times are outside the source or not sorted.')
        if type(speaker) is not int or not 0 <= speaker < 32:
            raise ValueError('Speaker indices must be bounded anonymous integers.')
        result.append({'start': start, 'end': end, 'speaker_index': speaker,
                       'speaker': f'speaker_{speaker:02d}'})
        previous = start
    return result


def worker(request_path):
    """Internal child. The app-Python parent supplies native-call supervision."""
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    request_path = Path(request_path)
    request = json.loads(regular_bytes(request_path, 128 * 1024))
    raw = regular_bytes(request_path.parent / 'input.wav', MAX_WAV_BYTES)
    source, pcm = wav_info(raw)
    if source != request['source']:
        raise ValueError('Child input differs from the frozen source.')
    models = request['models']
    for item in models:
        if file_digest(item['path'], item['bytes']) != item['sha256']:
            raise ValueError('Model changed before inference.')
    import numpy as np
    import sherpa_onnx as sherpa
    if sherpa.__version__ != RUNTIME_VERSION:
        raise ValueError('Installed Sherpa version differs from the pinned runtime.')
    cfg = request['config']
    if cfg != options(cfg['num_clusters'], cfg['cluster_threshold']):
        raise ValueError('Child configuration differs from the bounded CPU contract.')
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=models[0]['path'], window_shift_ratio=.1),
            num_threads=2, provider='cpu'),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=models[1]['path'],
                                                        num_threads=2, provider='cpu'),
        clustering=sherpa.FastClusteringConfig(num_clusters=cfg['num_clusters'],
                                              threshold=cfg['cluster_threshold']),
        min_duration_on=.3, min_duration_off=.5)
    if not config.validate():
        raise ValueError('Sherpa rejected the pinned model/configuration pair.')
    started = time.monotonic()
    pipeline = sherpa.OfflineSpeakerDiarization(config)
    if pipeline.sample_rate != 16000:
        raise ValueError('Model pipeline requires a different sample rate.')
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / np.float32(32768)
    result = pipeline.process(samples).sort_by_start_time()
    rows = [{'start': float(row.start), 'end': float(row.end), 'speaker_index': int(row.speaker)} for row in result]
    turns = validate_turns(rows, source['duration_seconds'])
    version = sherpa.onnxruntime_version
    native_version = version() if callable(version) else str(version)
    write_new(request_path.parent / 'child-result.json', {
        'source_sha256': source['sha256'], 'config': cfg, 'models': models,
        'turns': turns, 'speaker_count': len({row['speaker_index'] for row in turns}),
        'runtime': {'sherpa_onnx': sherpa.__version__, 'native_onnxruntime': native_version,
                    'numpy': np.__version__, 'python': sys.version, 'executable': sys.executable},
        'inference_seconds': time.monotonic() - started,
        'identity_inference': False, 'data_transmitted': False, 'paid_usage': 0})


class GuardFailure(RuntimeError):
    def __init__(self, reason, stats):
        super().__init__(reason)
        self.stats = stats


def supervise(command, out, *, check=lambda: None, timeout=MAX_SECONDS):
    """Keep the leader unreaped until group cleanup; never signal another group."""
    import psutil
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('The standalone supervisor must run in the main thread for signal-safe cleanup.')
    if not 0 < timeout <= MAX_SECONDS:
        raise ValueError('Supervisor timeout cannot exceed 120 seconds.')
    stats = {'peak_process_tree_rss_bytes': 0, 'minimum_system_available_bytes': None,
             'poll_seconds': POLL_SECONDS, 'ram_reserve_bytes': RESERVE_BYTES,
             'rss_limit_bytes': MAX_RSS_BYTES, 'deadline_seconds': timeout,
             'supervision': 'polled process tree, including native calls; not an instantaneous OS allocation ceiling'}
    started = time.monotonic()
    available = psutil.virtual_memory().available
    stats['minimum_system_available_bytes'] = available
    if available < RESERVE_BYTES:
        raise GuardFailure('System available RAM is below the preserved 1.5 GiB reserve.', stats)
    check()
    log_path = Path(out) / 'worker.log'
    env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'NUMEXPR_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': ''}
    owner = None
    tracked = {}
    process = None
    def same(candidate):
        if candidate.pid not in tracked:
            return False
        try:
            return candidate.is_running() and candidate.create_time() == tracked[candidate.pid][1]
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            return False
    def signal_owned(sig):
        # Popen has already established a new session, but psutil identity setup
        # can fail. This exact child remains unreaped, so its PID cannot be reused.
        if process is not None and process.returncode is None:
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        for candidate, _ in reversed(list(tracked.values())):
            if process is not None and candidate.pid == process.pid:
                continue
            if same(candidate):
                try:
                    candidate.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass
    try:
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, env=env)
            owner = psutil.Process(process.pid)
            tracked[owner.pid] = owner, owner.create_time()
            stats['child_pid'] = owner.pid
            while True:
                check()
                if not same(owner):
                    raise GuardFailure('Owned child identity disappeared before supervised completion.', stats)
                for candidate in [owner, *owner.children(recursive=True)]:
                    if candidate.pid not in tracked:
                        try:
                            tracked[candidate.pid] = candidate, candidate.create_time()
                        except psutil.NoSuchProcess:
                            pass
                rss = 0
                for candidate, _ in tracked.values():
                    if same(candidate):
                        try:
                            rss += candidate.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                available = psutil.virtual_memory().available
                stats['peak_process_tree_rss_bytes'] = max(stats['peak_process_tree_rss_bytes'], rss)
                stats['minimum_system_available_bytes'] = min(stats['minimum_system_available_bytes'], available)
                if rss > MAX_RSS_BYTES:
                    raise GuardFailure('Owned child tree exceeded 1.25 GiB RSS.', stats)
                if available < RESERVE_BYTES:
                    raise GuardFailure('System available RAM fell below the preserved 1.5 GiB reserve.', stats)
                if time.monotonic() - started >= timeout:
                    raise GuardFailure('Diarization exceeded its supervised wall deadline.', stats)
                if log_path.stat().st_size > MAX_LOG_BYTES:
                    raise GuardFailure('Diarization exceeded the bounded native log size.', stats)
                if owner.status() == psutil.STATUS_ZOMBIE:
                    break
                time.sleep(POLL_SECONDS)
    except BaseException as exc:
        # Preserve the sampled resource evidence even for operator cancellation
        # or a psutil observation failure; the finally block adds cleanup status.
        exc.supervisor_stats = stats
        raise
    finally:
        # Popen.poll()/wait() intentionally occurs only after owned-group cleanup,
        # so an exited leader's unreaped PID cannot be reused by an unrelated job.
        if process is not None:
            # A second Ctrl-C/termination signal must not interrupt cleanup and
            # leave native inference running after the supervisor exits.
            previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                signal_owned(signal.SIGTERM)
                time.sleep(.1)
                signal_owned(signal.SIGKILL)
                try:
                    stats['exit_code'] = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    stats['cleanup_timeout'] = True
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        stats['wall_seconds'] = time.monotonic() - started
    if stats.get('exit_code') != 0:
        raise GuardFailure(f'Sherpa child exited with code {stats.get("exit_code")}; see bounded worker.log.', stats)
    return stats


def run_trial(source, manifest, out, *, speakers=-1, threshold=.5, check=lambda: None):
    source, out = Path(source).absolute(), Path(out).absolute()
    cfg = options(speakers, threshold)
    raw = regular_bytes(source, MAX_WAV_BYTES)
    source_info, _ = wav_info(raw)
    plan = load_plan(manifest)
    if out.exists() or out.is_symlink():
        raise ValueError('Use a fresh output directory; existing output is never overwritten.')
    if source.resolve().is_relative_to(out.resolve()):
        raise ValueError('Output must not contain the selected source.')
    out.mkdir(parents=True, exist_ok=False)
    request = {'schema_version': 1, 'source': source_info, 'config': cfg, 'models': plan['models']}
    with (out / 'input.wav').open('xb') as handle:
        handle.write(raw)
    write_new(out / 'request.json', request)
    report = {'schema_version': 1, 'status': 'failed', 'source': source_info,
              'source_path': str(source), 'manifest_sha256': plan['manifest_sha256'],
              'runner_sha256': digest(regular_bytes(__file__, 256 * 1024)),
              'config': cfg, 'models': plan['models'], 'identity_inference': False,
              'data_transmitted': False, 'paid_usage': 0, 'resources': None,
              'timestamp_boundary_tolerance_seconds': TIMESTAMP_TOLERANCE_SECONDS,
              'limitations': ['Anonymous speaker indices are within-recording clusters, not identities.',
                  'Native turn timestamps are preserved, with at most 50 ms boundary quantization tolerance.',
                  'An executed pipeline alone does not establish diarization accuracy or correct overlap attribution.']}
    try:
        report['resources'] = supervise([plan['python'], '-I', str(Path(__file__).resolve()),
                                        '--child', str(out / 'request.json')], out, check=check)
        child = json.loads(regular_bytes(out / 'child-result.json', MAX_JSON_BYTES))
        if child.get('source_sha256') != source_info['sha256'] or child.get('models') != plan['models'] or child.get('config') != cfg:
            raise ValueError('Child result does not match its frozen input, models and configuration.')
        child['turns'] = validate_turns(child['turns'], source_info['duration_seconds'])
        report['result'] = child
        report['status'] = 'succeeded'
    except BaseException as exc:
        report['resources'] = getattr(exc, 'supervisor_stats', getattr(exc, 'stats', report['resources']))
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
        raise
    finally:
        try:
            report['source_unchanged'] = digest(regular_bytes(source, MAX_WAV_BYTES)) == source_info['sha256']
        except (OSError, ValueError):
            report['source_unchanged'] = False
        if not report['source_unchanged']:
            report['status'] = 'failed'
            report['source_integrity_failure'] = 'Original source differs from its pre-run bytes.'
        report['models_unchanged'] = True
        for item in plan['models']:
            try:
                valid = file_digest(item['path'], item['bytes']) == item['sha256']
            except (OSError, ValueError):
                valid = False
            report['models_unchanged'] = report['models_unchanged'] and valid
        if not report['models_unchanged']:
            report['status'] = 'failed'
            report['model_integrity_failure'] = 'Pinned model bytes changed during the trial.'
        write_new(out / 'report.json', report)
    if not report['source_unchanged']:
        raise ValueError('Original source changed during the trial; evidence is invalid.')
    if not report['models_unchanged']:
        raise ValueError('Pinned models changed during the trial; evidence is invalid.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--speakers', type=int, default=-1, help='Explicit known count1–8; -1 means unknown (default).')
    parser.add_argument('--cluster-threshold', type=float, default=.5)
    parser.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        worker(args.child)
        return
    if not all((args.source, args.manifest, args.output_dir)):
        parser.error('--source, --manifest and --output-dir are required')
    def cancelled(signum, frame):
        raise KeyboardInterrupt('Operator cancelled the standalone diarization trial.')
    old = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = run_trial(args.source, args.manifest, args.output_dir,
                           speakers=args.speakers, threshold=args.cluster_threshold)
        print(json.dumps({'report': str(args.output_dir / 'report.json'),
                          'status': result['status'], 'source_unchanged': result['source_unchanged']}))
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
