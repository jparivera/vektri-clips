#!/usr/bin/env python3
"""Frozen, paired OCR evaluation. Dry-run by default; no network or downloads.

Inference receives only identical image bytes and a mode, never labels. All
recognized strings are untrusted data. `unseen` means not previously OCR-scored,
not absent from a model's training corpus. Optional recorded video evidence is
reported separately and is scored only at explicitly annotated timestamps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import threading
import time

PROTOCOL = {
    'schema_version': 1,
    'pair': ['baseline', 'detail'], 'fixed_frame_quality_samples': 1,
    'strict_text': 'Unicode casefold, collapse whitespace, exact full string',
    'compact_text': 'Unicode casefold, remove whitespace, exact full string; separate diagnostic',
    'spatial': 'axis-aligned bounding-box IoU in normalized input coordinates',
    'match_iou_minimum': .30,
    'deduplication': 'connected components of accepted detections with identical strict text and IoU >= .50; a component matches a label when any member matches spatially',
    'dedup_iou_minimum': .50,
    'assignment': 'deterministic maximum-cardinality one-to-one matching, higher IoU edges visited first',
    'partial_ground_truth': 'unmatched predictions are UNKNOWN, never false positives',
    'exhaustive_ground_truth': 'unmatched predictions are false positives',
    'text_free_ground_truth': 'full frame explicitly reviewed text-free; every accepted prediction is a false positive',
    'video_annotation_tolerance_seconds': .001,
    'score_interpretation': 'raw model scores are not calibrated probabilities; no score tuning or new threshold',
}
MAX_JSON = 16 * 1024**2
MAX_IMAGE = 64 * 1024**2
MAX_FRAMES = 64
MAX_LABELS = 256
MAX_DETECTIONS = 2048
MAX_RAW_EVIDENCE = 128 * 1024**2
HEX = re.compile(r'^[a-f0-9]{64}$')
CAUTION = ('Descriptive counts on these frozen annotations only. Partial annotations do not '
           'support a false-positive rate. Raw and grouped results are distinct. Model scores '
           'are not probabilities. No inference of model training-set independence is made.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path, limit=512 * 1024**2):
    value, total = hashlib.sha256(), 0
    with Path(path).open('rb') as handle:
        while data := handle.read(1024**2):
            total += len(data)
            if total > limit:
                raise ValueError('Hash input exceeds its byte limit.')
            value.update(data)
    return value.hexdigest()


PROTOCOL_SHA256 = digest(canonical(PROTOCOL))


def read_bytes(path, limit):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Selected input is missing, not a regular file, or exceeds its byte limit.')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Input grew beyond its byte limit.')
    return raw


def normalize(text, compact=False):
    if not isinstance(text, str) or len(text) > 10000:
        raise ValueError('Text must be a bounded string.')
    parts = text.casefold().split()
    return ''.join(parts) if compact else ' '.join(parts)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def box(value):
    if not isinstance(value, list):
        raise ValueError('A normalized box must be an array.')
    if len(value) == 4 and all(finite(v) for v in value):
        x0, y0, x1, y1 = value
    elif len(value) == 4 and all(isinstance(p, list) and len(p) == 2 and all(finite(v) for v in p) for p in value):
        x0, x1 = min(p[0] for p in value), max(p[0] for p in value)
        y0, y1 = min(p[1] for p in value), max(p[1] for p in value)
    else:
        raise ValueError('Expected normalized XYXY or four normalized corner points.')
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError('Boxes must be nonempty and inside normalized frame bounds.')
    return [float(x0), float(y0), float(x1), float(y1)]


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union else 0


def validate_frame(frame, *, require_image=True):
    if not isinstance(frame, dict) or not isinstance(frame.get('id'), str) or not frame['id']:
        raise ValueError('Each frame needs a unique nonempty id.')
    if frame.get('ground_truth') not in ('partial', 'exhaustive', 'text_free'):
        raise ValueError('Every frame must declare partial, exhaustive, or text_free ground truth.')
    labels = frame.get('labels')
    if not isinstance(labels, list) or len(labels) > MAX_LABELS:
        raise ValueError('Expected bounded labels array.')
    ids = set()
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get('id'), str) or not label['id'] or label['id'] in ids:
            raise ValueError('Label ids must be unique within each frame.')
        ids.add(label['id'])
        if not normalize(label.get('text')):
            raise ValueError('Labels need nonempty literal text.')
        box(label.get('box'))
    if frame['ground_truth'] == 'text_free' and labels:
        raise ValueError('Text-free controls cannot contain positive labels.')
    if require_image:
        if not HEX.fullmatch(str(frame.get('sha256', ''))):
            raise ValueError('Each frame needs its exact SHA-256.')
        if any(type(frame.get(k)) is not int or not 4 <= frame[k] <= 16384 for k in ('width', 'height')) or frame['width'] * frame['height'] > 16_000_000:
            raise ValueError('Frame dimensions exceed bounded OCR input limits.')
        if not isinstance(frame.get('provenance'), dict) or not frame['provenance']:
            raise ValueError('Frame provenance must be explicit.')
    return frame


def freeze_manifest(path, expected_hash=None):
    path = Path(path).resolve(strict=True)
    raw = read_bytes(path, MAX_JSON)
    sha = digest(raw)
    if expected_hash is not None and expected_hash != sha:
        raise ValueError('Manifest differs from the pre-inference frozen hash.')
    doc = json.loads(raw)
    if doc.get('schema_version') != 1 or doc.get('labels_observed_after_ocr') is not False:
        raise ValueError('Require schema_version=1 and explicitly pre-OCR labels.')
    frames = doc.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError('Expected 1–64 frozen frames.')
    seen, selected = set(), []
    for frame in frames:
        validate_frame(frame)
        if frame['id'] in seen:
            raise ValueError('Frame ids must be unique.')
        seen.add(frame['id'])
        relative = Path(frame.get('path', ''))
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            raise ValueError('Frame paths must be relative to the selected manifest.')
        image = (path.parent / relative).resolve(strict=True)
        if not image.is_relative_to(path.parent):
            raise ValueError('Frame path escapes its selected manifest directory.')
        data = read_bytes(image, MAX_IMAGE)
        if digest(data) != frame['sha256']:
            raise ValueError('Image bytes differ from the frozen manifest.')
        from PIL import Image
        with Image.open(io.BytesIO(data)) as header:
            if header.size != (frame['width'], frame['height']) or getattr(header, 'n_frames', 1) != 1:
                raise ValueError('Frozen dimensions must match one static image frame.')
        selected.append({'frame': copy.deepcopy(frame), 'path': image})
    return {'manifest': doc, 'manifest_bytes': raw, 'manifest_sha256': sha, 'frames': selected}


def predictions(detections):
    if not isinstance(detections, list) or len(detections) > MAX_DETECTIONS:
        raise ValueError('Expected bounded accepted-detection array.')
    result = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict) or not normalize(detection.get('text')):
            raise ValueError('Accepted detections need nonempty literal text.')
        result.append({'text': detection['text'], 'boxes': [box(detection.get('box_normalized'))], 'members': [index]})
    return result


def group_duplicates(rows):
    """Explicit evaluator grouping; never edits the pipeline's raw evidence."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    text_indices = {}
    for i, row in enumerate(rows):
        same_text = text_indices.setdefault(normalize(row['text']), [])
        for j in same_text:
            if iou(row['boxes'][0], rows[j]['boxes'][0]) >= PROTOCOL['dedup_iou_minimum']:
                parents[root(i)] = root(j)
        same_text.append(i)
    groups = {}
    for i, row in enumerate(rows):
        group = groups.setdefault(root(i), {'text': row['text'], 'boxes': [], 'members': []})
        group['boxes'].extend(row['boxes']); group['members'].extend(row['members'])
    return list(groups.values())


def match(frame, rows, *, compact=False, spatial=True):
    labels = frame['labels']
    edges = []
    text_indices = {}
    for pi, prediction in enumerate(rows):
        text_indices.setdefault(normalize(prediction['text'], compact), []).append(pi)
    for label in labels:
        candidates = []
        label_box = box(label['box'])
        for pi in text_indices.get(normalize(label['text'], compact), []):
            prediction = rows[pi]
            overlap = max(iou(label_box, pb) for pb in prediction['boxes'])
            if not spatial or overlap >= PROTOCOL['match_iou_minimum']:
                candidates.append((pi, overlap))
        edges.append(sorted(candidates, key=lambda pair: (-pair[1], pair[0])))
    assigned = {}
    def augment(li, seen):
        for pi, _ in edges[li]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in assigned or augment(assigned[pi], seen):
                assigned[pi] = li
                return True
        return False
    for li in range(len(labels)):
        augment(li, set())
    matched_labels = set(assigned.values())
    unmatched = [i for i in range(len(rows)) if i not in assigned]
    exhaustive = frame['ground_truth'] in ('exhaustive', 'text_free')
    return {'annotated_labels': len(labels), 'predictions': len(rows), 'true_positives': len(assigned),
            'missed_labels': len(labels) - len(assigned),
            'false_positives': len(unmatched) if exhaustive else 0,
            'unknown_predictions': 0 if exhaustive else len(unmatched),
            'false_positive_assessment': 'exhaustive' if exhaustive else 'unavailable_partial_ground_truth',
            'matches': [{'label_id': labels[li]['id'], 'prediction_index': pi, 'raw_prediction_indices': rows[pi]['members'],
                         'box_iou': max(iou(box(labels[li]['box']), pb) for pb in rows[pi]['boxes'])} for pi, li in sorted(assigned.items())],
            'missed_label_ids': [label['id'] for li, label in enumerate(labels) if li not in matched_labels],
            'unmatched_prediction_indices': unmatched}


def evaluate_frame(frame, result):
    validate_frame(frame, require_image=False)
    raw = predictions(result.get('detections'))
    dedup = group_duplicates(raw)
    metrics = {}
    for name, rows in (('raw', raw), ('deduplicated', dedup)):
        metrics[name] = {'strict': match(frame, rows), 'whitespace_insensitive': match(frame, rows, compact=True)}
    text_only = match(frame, raw, spatial=False)
    return {'frame_id': frame['id'], 'ground_truth': frame['ground_truth'],
            'raw_accepted_detection_count': len(raw), 'deduplicated_occurrence_count': len(dedup),
            'raw_model_line_count': len(result['raw_lines']) if isinstance(result.get('raw_lines'), list) else None,
            'dedup_groups': [row['members'] for row in dedup], 'metrics': metrics,
            'legacy_text_only_diagnostic': {'full_labels_present': text_only['true_positives'], 'missed_labels': text_only['missed_labels'],
                'notice': 'Exact strict full-label matching without spatial checks; reported separately, not a spatial true-positive score.'}}


def aggregate(evaluations):
    fields = ('annotated_labels', 'predictions', 'true_positives', 'missed_labels', 'false_positives', 'unknown_predictions')
    result = {}
    # A failed mode must not make the other mode appear better on a different
    # denominator. Both aggregate columns use only the same complete pairs.
    paired = [frame for frame in evaluations if all(frame['modes'][mode].get('evaluation') is not None for mode in PROTOCOL['pair'])]
    for mode in PROTOCOL['pair']:
        selected = [frame['modes'][mode]['evaluation'] for frame in paired]
        result[mode] = {kind: {text_mode: {key: sum(e['metrics'][kind][text_mode][key] for e in selected) for key in fields}
                              for text_mode in ('strict', 'whitespace_insensitive')} for kind in ('raw', 'deduplicated')}
        result[mode]['legacy_text_only_diagnostic'] = {key: sum(e['legacy_text_only_diagnostic'][key] for e in selected)
                                                       for key in ('full_labels_present', 'missed_labels')}
    return result


class Resources:
    """50ms process-RSS sampling, explicitly not a guaranteed native allocation peak."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event(); self.peak = 0
        self.started = time.perf_counter(); self.cpu = time.process_time()
        def observe():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.process.memory_info().rss)
                self.stop.wait(.05)
        self.thread = threading.Thread(target=observe, daemon=True); self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=1)
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.result = {'wall_seconds': time.perf_counter()-self.started, 'cpu_seconds': time.process_time()-self.cpu,
                       'sampled_process_peak_rss_bytes': self.peak, 'rss_sample_interval_seconds': .05,
                       'rss_scope': 'whole benchmark process; shared model sessions included, native transient peaks may be missed'}


def paired_run(frozen, infer, identity, *, evidence_dir=None):
    """infer(image_bytes, mode) receives no annotation text or spatial labels."""
    records, evidence_bytes = [], 0
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir()
    for frame_index, selected in enumerate(frozen['frames']):
        frame = selected['frame']; raw = read_bytes(selected['path'], MAX_IMAGE)
        if digest(raw) != frame['sha256']:
            raise ValueError('Frozen frame changed before inference.')
        record = {'id': frame['id'], 'image_sha256': frame['sha256'], 'modes': {}}
        for mode in PROTOCOL['pair']:
            result, evaluation, failure, raw_evidence = None, None, None, None
            resources = None
            try:
                with Resources() as resources:
                    result = infer(raw, mode)
                if evidence_dir is not None:
                    encoded = canonical(result)
                    if len(encoded) > MAX_JSON or evidence_bytes + len(encoded) > MAX_RAW_EVIDENCE:
                        raise ValueError('Raw benchmark evidence exceeds its bounded storage budget.')
                    name = f'{frame_index:03d}-{mode}.json'
                    with (evidence_dir / name).open('xb') as handle:
                        handle.write(encoded)
                    evidence_bytes += len(encoded)
                    raw_evidence = {'path': f'{evidence_dir.name}/{name}', 'sha256': digest(encoded), 'bytes': len(encoded)}
                evaluation = evaluate_frame(frame, result)
            except Exception as exc:
                failure = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
            record['modes'][mode] = {'status': 'failed' if failure else 'succeeded', 'failure': failure,
                'raw_result': result if evidence_dir is None else None, 'raw_evidence': raw_evidence,
                'resources': getattr(resources, 'result', None), 'evaluation': evaluation}
        records.append(record)
    return {'schema_version': 1, 'kind': 'paired_fixed_frame_ocr', 'caution': CAUTION,
            'manifest_sha256': frozen['manifest_sha256'], 'protocol': PROTOCOL, 'protocol_sha256': PROTOCOL_SHA256,
            'identity': identity, 'frames': records, 'aggregate': aggregate(records),
            'aggregate_cohort': {'rule': 'only frames successfully evaluated in BOTH modes',
                'attempted_frames': len(records), 'scored_frames': sum(all(r['modes'][m]['evaluation'] is not None for m in PROTOCOL['pair']) for r in records),
                'excluded_frame_ids': [r['id'] for r in records if any(r['modes'][m]['evaluation'] is None for m in PROTOCOL['pair'])]},
            'quality_sampling': 'disabled: exact same frozen image bytes for both modes',
            'raw_evidence_storage_bytes': evidence_bytes, 'raw_evidence_storage_limit_bytes': MAX_RAW_EVIDENCE,
            'temporal_agreement_accuracy_claimed': False, 'data_transmitted': False, 'paid_usage': 0}


def evaluate_video_control(control, evidence):
    """Score only exact annotated actual PTS; keep all other samples unknown."""
    if evidence.get('source_sha256') != control.get('source_sha256') or not HEX.fullmatch(str(control.get('source_sha256', ''))):
        raise ValueError('Recorded video evidence is bound to a different source.')
    payload = evidence.get('ocr', evidence)
    rows = []
    for sample in payload.get('samples', []):
        timestamp = sample.get('time')
        if not finite(timestamp):
            raise ValueError('Video samples require actual finite PTS timestamps.')
        matching = [label for label in control.get('frames', []) if finite(label.get('time')) and abs(label['time']-timestamp) <= PROTOCOL['video_annotation_tolerance_seconds']]
        if len(matching) > 1:
            raise ValueError('Ambiguous timestamp ground truth.')
        rows.append({'actual_time': timestamp, 'ground_truth_available': bool(matching),
                     'evaluation': evaluate_frame(matching[0], sample) if matching else None,
                     'accepted_prediction_count': len(sample.get('detections', [])),
                     'quality_selection': sample.get('quality_selection')})
    return {'control_id': control['id'], 'samples': rows,
            'unannotated_sample_count': sum(not row['ground_truth_available'] for row in rows),
            'raw_pipeline_metadata': {key: value for key, value in payload.items() if key != 'samples'},
            'temporal_agreement_accuracy_claimed': False,
            'notice': 'Only actual selected timestamps with frozen labels are scored. Agreement is repeated OCR evidence, not truth.'}


def resolved_controls(manifest):
    """Accept frozen inline labels or references without rewriting manifest bytes."""
    frames = {frame['id']: frame for frame in manifest['frames']}
    controls = {}
    for original in manifest.get('temporal_controls', []):
        control = copy.deepcopy(original)
        if not isinstance(control.get('id'), str) or control['id'] in controls:
            raise ValueError('Temporal controls need unique ids.')
        control['source_sha256'] = control.get('source_sha256', control.get('sha256'))
        if not HEX.fullmatch(str(control.get('source_sha256', ''))):
            raise ValueError('Temporal controls need frozen source SHA-256.')
        if 'frame_ids' in control:
            control['frames'] = []
            for identifier in control['frame_ids']:
                if identifier not in frames:
                    raise ValueError('Temporal frame reference is missing from frozen labels.')
                frame = copy.deepcopy(frames[identifier])
                frame['time'] = frame.get('source_time_seconds')
                if not finite(frame['time']) or frame['time'] < 0:
                    raise ValueError('Temporal reference needs actual source_time_seconds.')
                control['frames'].append(frame)
        controls[control['id']] = control
    return controls


def selected_model_identity(ocr, catalog, model_id, location):
    if model_id not in ('ppocr-v4', 'ppocr-v5-mobile') or model_id not in catalog:
        raise ValueError('Selected OCR model is not present in the reviewed installed catalog.')
    spec = catalog[model_id]
    hashes = {name: file_hash(location / name) for name in spec['files']}
    if any(hashes[name] != expected[1] for name, expected in spec['files'].items()):
        raise ValueError('Model bytes differ from the pinned catalog.')
    pipeline_files = {name: file_hash(Path(ocr.__file__).parent / name, MAX_JSON)
                      for name in ('ocr.py', 'vision.py', 'models.py', 'runtime.py')}
    return {'model_id': model_id, 'model_repository': spec['repository'], 'model_revision': spec['revision'],
            'model_file_sha256': hashes, 'pipeline_files_sha256': pipeline_files,
            'pipeline_sha256': digest(canonical(pipeline_files))}


def save_new(path, value):
    with Path(path).open('xb') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode() + b'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--ocr-model', choices=('ppocr-v4', 'ppocr-v5-mobile'), default='ppocr-v4', help='Explicit installed model selection; default remains V4')
    parser.add_argument('--max-wall-seconds', type=int, default=1800, help='Overall cooperative execution deadline, 60–7200 seconds; default 1800')
    parser.add_argument('--video-results', type=Path, help='Optional recorded source-bound results keyed by frozen temporal control id; never runs video inference')
    args = parser.parse_args(argv)
    if args.execute and not args.manifest_sha256:
        parser.error('--execute requires the manifest SHA-256 recorded before inference')
    if not 60 <= args.max_wall_seconds <= 7200:
        parser.error('--max-wall-seconds must be between 60 and 7200')
    frozen = freeze_manifest(args.manifest, args.manifest_sha256)
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory; benchmark artifacts are never overwritten.')
    if any(selected['path'].is_relative_to(output) for selected in frozen['frames']):
        raise ValueError('Output must not contain selected inputs.')
    output.mkdir(parents=True)
    with (output / 'manifest-frozen.json').open('xb') as handle:
        handle.write(frozen['manifest_bytes'])
    save_new(output / 'protocol.json', PROTOCOL)
    runner_hash = digest(read_bytes(Path(__file__), MAX_JSON))
    result = {'dry_run': not args.execute, 'manifest_sha256': frozen['manifest_sha256'],
              'protocol_sha256': PROTOCOL_SHA256, 'runner_sha256': runner_hash, 'frame_count': len(frozen['frames']),
              'caution': CAUTION, 'inference_run': False, 'requested_model_id': args.ocr_model}
    if args.execute:
        from PIL import Image
        from clipstudio import ocr
        from clipstudio.models import model_path, CATALOG
        from importlib.metadata import version, PackageNotFoundError
        execution_deadline = time.monotonic() + args.max_wall_seconds
        def execution_check():
            if time.monotonic() >= execution_deadline:
                raise RuntimeError('Paired benchmark exceeded its overall cooperative wall-time budget.')
            ocr._memory_check()
        location = model_path(args.ocr_model, verify=True)
        identity = selected_model_identity(ocr, CATALOG, args.ocr_model, location)
        identity['runner_sha256'] = runner_hash
        identity['runtime_versions'] = {'python': sys.version}
        for package in ('onnxruntime', 'numpy', 'Pillow', 'opencv-python', 'opencv-python-headless', 'psutil'):
            try:
                identity['runtime_versions'][package] = version(package)
            except PackageNotFoundError:
                pass
        started = time.perf_counter()
        engine = ocr.Recognizer(execution_check, model_id=args.ocr_model)
        identity['model_load_wall_seconds'] = time.perf_counter() - started
        try:
            if getattr(engine, 'model_id', None) != args.ocr_model:
                raise ValueError('Loaded OCR engine identity differs from selected pinned model.')
            def infer(raw, mode):
                deadline = min(execution_deadline, time.monotonic() + ocr.MAX_WALL_SECONDS)
                check = lambda: ocr._checkpoint(execution_check, deadline)
                check()
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    return ocr.recognize_frame(image, engine, check, mode=mode)
            result = paired_run(frozen, infer, identity, evidence_dir=output/'raw-evidence')
            result['overall_wall_budget_seconds'] = args.max_wall_seconds
            result['resource_enforcement'] = 'cooperative checks before/after native calls; not a hard ORT interrupt; whole-process sampled RSS reported'
            result['inference_run'] = True
        finally:
            engine.close()
    if args.video_results:
        recorded_bytes = read_bytes(args.video_results, MAX_JSON)
        recorded = json.loads(recorded_bytes)
        controls = resolved_controls(frozen['manifest'])
        if not isinstance(recorded, dict) or not set(recorded) <= controls.keys():
            raise ValueError('Video result ids must be declared in the frozen manifest.')
        result['video_controls'] = [evaluate_video_control(controls[key], evidence) for key, evidence in recorded.items()]
        result['video_results_sha256'] = digest(recorded_bytes)
    save_new(output / 'report.json', result)
    print(json.dumps({'report': str(output / 'report.json'), 'manifest_sha256': frozen['manifest_sha256'],
                      'protocol_sha256': PROTOCOL_SHA256, 'inference_run': result['inference_run']}))
    return result


if __name__ == '__main__':
    main()
