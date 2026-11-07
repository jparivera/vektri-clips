#!/usr/bin/env python3
"""Bind a successful standalone Sherpa report to frozen DER input; no inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

LIMIT = 2 * 1024**2
TOLERANCE = .05
SHA256 = re.compile(r'^[0-9a-f]{64}$')
REVISION = re.compile(r'^[0-9a-f]{40}$')
REPOSITORY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
MODEL_FILENAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,247}\.onnx$')
MODEL_IDS = ('sherpa-pyannote-segmentation-3.0', 'sherpa-nemo-titanet-small')
MODEL_LICENSES = ('MIT', 'Apache-2.0')


def bounded_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
            raise ValueError('Input must be a bounded regular nonsymlink JSON file.')
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read(LIMIT + 1)
        after = os.fstat(fd)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if len(raw) > LIMIT or len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in keys):
            raise ValueError('Input exceeded its bound or changed while reading.')
    finally:
        os.close(fd)
    def bad_constant(value):
        raise ValueError('Nonfinite JSON is unsupported.')
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('Nonfinite JSON is unsupported.')
        return parsed
    doc = json.loads(raw, parse_constant=bad_constant, parse_float=finite_float)
    if not isinstance(doc, dict):
        raise ValueError('Expected a JSON object.')
    return doc, hashlib.sha256(raw).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha256(value, label):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f'{label} requires an exact lowercase SHA-256.')


def object_field(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be an object.')
    return value


def validate_config(cfg):
    """Match the standalone runner's fixed CPU options, including JSON types."""
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be an object.')
    count, threshold = cfg.get('num_clusters'), cfg.get('cluster_threshold')
    if type(count) is not int or count not in {-1, *range(1, 9)}:
        raise ValueError('Cluster count must be -1 or a known count from 1 to 8.')
    if not number(threshold) or not 0 < threshold < 1:
        raise ValueError('Clustering threshold must be finite and strictly between zero and one.')
    expected = {'provider': 'cpu', 'num_threads': 2, 'num_clusters': count,
                'speaker_count_mode': 'unknown' if count == -1 else 'explicit_known_count',
                'cluster_threshold': threshold, 'threshold_used': count == -1,
                'window_shift_ratio': .1, 'min_duration_on': .3, 'min_duration_off': .5}
    if (set(cfg) != set(expected) or cfg != expected
            or type(cfg.get('num_threads')) is not int
            or type(cfg.get('threshold_used')) is not bool
            or any(not number(cfg.get(key)) for key in ('window_shift_ratio', 'min_duration_on', 'min_duration_off'))):
        raise ValueError('Configuration differs from the bounded CPU/two-thread runner contract.')


def validate_models(models):
    """Check recorded pin syntax only; do not open or rehash installed model files."""
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError('Both pinned pipeline models must be recorded.')
    for item, model_id, license_id in zip(models, MODEL_IDS, MODEL_LICENSES):
        if not isinstance(item, dict) or item.get('id') != model_id:
            raise ValueError('Expected the ordered segmentation and embedding model IDs.')
        if type(item.get('bytes')) is not int or not 0 < item['bytes'] <= 64 * 1024**2:
            raise ValueError('Declared model size must be a positive integer of at most 64 MiB.')
        sha256(item.get('sha256'), 'Model pin')
        revision, repository, filename, path = (item.get(key) for key in ('revision', 'repository', 'filename', 'path'))
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ValueError('Model revision requires a lowercase 40-hex commit pin.')
        if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
            raise ValueError('Model repository must be a nonempty owner/name identifier.')
        if not isinstance(filename, str) or not MODEL_FILENAME.fullmatch(filename):
            raise ValueError('Model filename must be a plain ONNX filename.')
        if (not isinstance(path, str) or not path or len(path) > 4096 or '\x00' in path
                or not Path(path).is_absolute() or '..' in Path(path).parts or Path(path).name != filename):
            raise ValueError('Recorded model path must be absolute and match its plain filename.')
        if item.get('license') != license_id:
            raise ValueError('Model license declaration differs from the expected MIT/Apache-2.0 pair.')


def convert(report, gold, report_sha256, gold_sha256):
    if not isinstance(report, dict) or not isinstance(gold, dict):
        raise ValueError('Report and gold must be objects.')
    for value in (report_sha256, gold_sha256):
        sha256(value, 'Evidence')
    for key in ('runner_sha256', 'manifest_sha256'):
        sha256(report.get(key), key)
    if (report.get('status') != 'succeeded' or report.get('source_unchanged') is not True
            or report.get('models_unchanged') is not True):
        raise ValueError('Only completed reports with preserved source and models are scoreable.')
    resources = object_field(report, 'resources')
    if type(resources.get('exit_code')) is not int or resources['exit_code'] != 0:
        raise ValueError('Native child exit code must be the integer zero.')
    source, audio, result = object_field(report, 'source'), object_field(gold, 'audio'), object_field(report, 'result')
    source_hash = source.get('sha256')
    sha256(source_hash, 'Report source')
    if source_hash != audio.get('sha256') or result.get('source_sha256') != source_hash:
        raise ValueError('Report, child and frozen gold refer to different source bytes.')
    duration = gold.get('duration_seconds')
    if (not number(duration) or not 0 < duration <= 120
            or not number(source.get('duration_seconds')) or source['duration_seconds'] != duration):
        raise ValueError('Report and frozen gold durations must match within the source limit.')
    geometry = ('sample_rate', 'channels', 'sample_width_bytes')
    if any(type(source.get(key)) is not int for key in geometry) or tuple(source[key] for key in geometry) != (16000, 1, 2):
        raise ValueError('Report must describe mono PCM16 at 16 kHz.')
    if type(source.get('frames')) is not int or not 0 < source['frames'] <= 1920000 or source['frames'] / 16000 != duration:
        raise ValueError('Source sample count is inconsistent with its duration.')
    if any(type(doc.get('schema_version')) is not int or doc['schema_version'] != 1 for doc in (gold, report)):
        raise ValueError('Unsupported evidence schema version.')
    validate_config(report.get('config'))
    validate_config(result.get('config'))
    validate_models(report.get('models'))
    validate_models(result.get('models'))
    if result.get('config') != report.get('config') or result.get('models') != report.get('models'):
        raise ValueError('Child configuration/models differ from the parent report.')
    if object_field(result, 'runtime').get('sherpa_onnx') != '2.0.1':
        raise ValueError('Expected the pinned Sherpa runtime.')
    if report.get('timestamp_boundary_tolerance_seconds') != TOLERANCE:
        raise ValueError('Native timestamp tolerance differs from the frozen conversion policy.')
    turns = result.get('turns')
    if not isinstance(turns, list) or len(turns) > 10000:
        raise ValueError('Turn list is absent or exceeds its limit.')
    segments, adjustments, indices = [], [], set()
    previous = -1
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError('Turn must be an object.')
        start, end, speaker_index = (turn.get(key) for key in ('start', 'end', 'speaker_index'))
        if (not number(start) or not number(end) or not 0 <= start < end <= duration + TOLERANCE
                or start < previous):
            raise ValueError('Turn timestamps violate the frozen native boundary contract.')
        if type(speaker_index) is not int or not 0 <= speaker_index < 32:
            raise ValueError('Speaker index must be a bounded anonymous integer.')
        speaker = f'speaker_{speaker_index:02d}'
        if turn.get('speaker') != speaker:
            raise ValueError('Anonymous label differs from its native index.')
        indices.add(speaker_index)
        previous = start
        if start >= duration:
            adjustments.append({'turn_index': index, 'action': 'excluded_outside_audio', 'raw': turn})
            continue
        if end > duration:
            adjustments.append({'turn_index': index, 'action': 'end_intersected_with_audio',
                                'raw_start': start, 'raw_end': end, 'scored_end': duration})
        segments.append({'start': start, 'end': min(end, duration), 'speaker': speaker})
    if type(result.get('speaker_count')) is not int or result['speaker_count'] != len(indices):
        raise ValueError('Native speaker count differs from its turn evidence.')
    return {'schema_version': 1, 'source_sha256': source_hash, 'segments': segments,
            'runner_report_sha256': report_sha256, 'gold_sha256': gold_sha256,
            'runner_sha256': report['runner_sha256'], 'manifest_sha256': report['manifest_sha256'],
            'runtime': result['runtime'], 'config': report['config'], 'models': report['models'],
            'raw_turn_count': len(turns), 'raw_speaker_count': len(indices),
            'scored_speaker_count': len({row['speaker'] for row in segments}),
            'conversion_policy': {'version': 1,
                'rule': 'Intersect native turns with physical audio [0,duration]; retain original report.',
                'allowed_native_endpoint_tolerance_seconds': TOLERANCE,
                'adjustments': adjustments, 'gold_modified': False, 'labels_remapped': False},
            'limitations': ['The separate scorer verifies source file and PCM hashes.',
                'Model/runner/manifest pin syntax and report consistency are checked; installed artifacts are not rehashed or authenticated by this converter.',
                'Recorded model license declarations are checked for the expected pair, not independently reviewed as legal permission.',
                'The scorer refuses more than six distinct output speakers; never collapse labels to fit.',
                'Constructed utterance ownership is not independent speech-activity ground truth.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--gold-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report, report_hash = bounded_json(args.report)
    gold, gold_hash = bounded_json(args.gold)
    if gold_hash != args.gold_sha256:
        raise ValueError('Frozen gold SHA-256 mismatch.')
    prediction = convert(report, gold, report_hash, gold_hash)
    prediction['adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(prediction, indent=2, allow_nan=False).encode() + b'\n'
    if len(encoded) > LIMIT:
        raise ValueError('Prediction exceeds its output bound.')
    with args.output.open('xb') as handle:
        handle.write(encoded)
    print(json.dumps({'prediction': str(args.output), 'segments': len(prediction['segments']),
                      'adjustments': len(prediction['conversion_policy']['adjustments'])}))


if __name__ == '__main__':
    main()
