#!/usr/bin/env python3
"""Compare local CLIP/X-CLIP retrieval using immutable, pre-score source labels.

Dry-run by default: validates and snapshots the selected source, provenance,
annotations and matched duration preset without contacting the app. --execute
deliberately imports one new QA project and runs the selected installed models
serially through a running loopback app. Never downloads models, renders media,
connects accounts, publishes, changes thresholds or bypasses resource guards.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

import httpx

from benchmark_delivery import request, save_json, sha256
from benchmark_visual_search import (complete_owned_job, ensure_ready, interval,
    intersection_seconds, merged_intervals, require_clip_analysis, top_frame_metrics)

MODELS = {
    'clip': {'id': 'clip-vit-b32', 'repository': 'Xenova/clip-vit-base-patch32',
             'revision': 'd15189d7028b43f1d3e65039190477f6af591c2a',
             'provider': 'local_onnx_clip', 'method': 'local_clip_visual'},
    'temporal': {'id': 'xclip-b16-k600', 'repository': 'imbcmdth/xclip-onnx',
                 'revision': '649f3c91b59cd24be316dc505e26eacf5cd00801',
                 'provider': 'local_onnx_xclip', 'method': 'local_xclip_temporal_retrieval'},
}
CAUTION = ('Descriptive retrieval overlap against these frozen assistant annotations only. '
           'Cosine is an uncalibrated similarity, not confidence, an event probability, '
           'hallucination accuracy or proof of event order. Unannotated time is unknown. '
           'Complete phase coverage is necessary evidence coverage, not semantic verification.')


def read_json_snapshot(path):
    raw = path.read_bytes()
    return raw, json.loads(raw), hashlib.sha256(raw).hexdigest()


def freeze_inputs(source, annotations, provenance, preset_file, preset_id, query_id):
    """Validate the original labels without adding a field or rewriting their bytes."""
    source_hash = sha256(source)
    araw, doc, ahash = read_json_snapshot(annotations)
    praw, prov, phash = read_json_snapshot(provenance)
    sraw, presets, shash = read_json_snapshot(preset_file)
    if presets.get('schema_version') != 1 or presets.get('model_scores_observed_before_preset') is not False:
        raise ValueError('Require a pre-score matched preset manifest, schema_version=1.')
    preset = copy.deepcopy(presets.get('presets', {}).get(preset_id))
    if not isinstance(preset, dict):
        raise ValueError('Unknown frozen preset.')
    if (source_hash != preset.get('source_sha256') or ahash != preset.get('annotations_sha256')
            or phash != preset.get('provenance_sha256')):
        raise ValueError('Source, annotations or provenance differ from the frozen preset hashes.')
    if doc.get('schema_version', doc.get('annotation_version')) != 1:
        raise ValueError('Unsupported annotation schema.')
    if doc.get('source_sha256') != source_hash or prov.get('sha256') != source_hash:
        raise ValueError('Labels/provenance are bound to a different media source.')
    if doc.get('model_scores_observed_before_annotation') is not False:
        raise ValueError('Require explicitly pre-score annotations.')
    if not prov.get('source_page_url') or not prov.get('direct_media_url') or not prov.get('license'):
        raise ValueError('Source provenance must retain the source, download URL and license.')
    queries = doc.get('queries', [])
    if not isinstance(queries, list):
        raise ValueError('Annotation queries must be an array.')
    matches = [q for q in queries if isinstance(q, dict) and q.get('id') == query_id]
    if len(matches) != 1 or query_id not in preset.get('query_ids', []):
        raise ValueError('Select a unique query present in both frozen labels and preset.')
    query = copy.deepcopy(matches[0])
    if not isinstance(query.get('query'), str) or not 1 <= len(query['query'].strip()) <= 2000:
        raise ValueError('Frozen query must contain 1–2000 characters.')
    positives, contrasts = query.get('positive_intervals'), query.get('contrast_intervals')
    if not isinstance(positives, list) or not isinstance(contrasts, list):
        raise ValueError('Require positive_intervals and contrast_intervals arrays.')
    absent = query.get('expected_absent') is True
    if (not positives and not absent) or (absent and positives):
        raise ValueError('Only explicitly absent queries may have empty positives.')
    duration = float(doc.get('duration_seconds', prov.get('duration_seconds', 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Source duration is missing or invalid.')
    for item in positives + contrasts + query.get('ordered_phases', []):
        if interval(item)[1] > duration + .05:
            raise ValueError('Annotation interval exceeds source duration.')
    if intersection_seconds(merged_intervals(positives), merged_intervals(contrasts)) > 1e-8:
        raise ValueError('Positive and contrast labels overlap.')
    phases = query.get('ordered_phases', [])
    if query.get('evaluation_kind') == 'temporal_order':
        if len(phases) < 2 or query.get('requires_all_phases_in_order') is not True:
            raise ValueError('Temporal-order labels need at least two ordered phases.')
        if any(interval(a)[1] > interval(b)[0] for a, b in zip(phases, phases[1:])):
            raise ValueError('Ordered annotation phases overlap or are reversed.')
    options = preset.get('options', {})
    length = options.get('min_length')
    if (isinstance(length, bool) or not isinstance(length, (int, float))
            or not math.isfinite(length) or not 1 <= length <= 300
            or options.get('max_length') != length):
        raise ValueError('Presets must specify a finite fixed matched duration.')
    if options.get('transcribe') is not False or options.get('mode') != 'local' or options.get('visual_search') is not True:
        raise ValueError('Preset must use local visual retrieval with transcription disabled.')
    if 'visual_model' in options or 'query' in options:
        raise ValueError('Model/query are selected separately; they must not override frozen labels.')
    if phases and length + 1e-6 < phases[-1]['end'] - phases[0]['start']:
        raise ValueError('Fixed window is too short to contain the annotated ordered sequence.')
    return {'source_sha256': source_hash, 'annotations': doc, 'annotations_sha256': ahash,
            'provenance': prov, 'provenance_sha256': phash, 'presets_sha256': shash,
            'preset': preset, 'query': query,
            'snapshots': {'annotations-frozen.json': araw, 'provenance-frozen.json': praw,
                          'presets-frozen.json': sraw}}


def overlap_metrics(candidates, query):
    positives = merged_intervals(query['positive_intervals'])
    contrasts = merged_intervals(query['contrast_intervals'])
    total = sum(b - a for a, b in positives)
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        start, end = interval(candidate)
        length = end - start
        positive = intersection_seconds([(start, end)], positives)
        contrast = intersection_seconds([(start, end)], contrasts)
        phases = []
        for phase in query.get('ordered_phases', []):
            a, b = interval(phase)
            covered = intersection_seconds([(start, end)], [(a, b)])
            phases.append({'label': phase.get('label'), 'start': a, 'end': b,
                           'overlap_seconds': round(covered, 6),
                           'reference_fraction_covered': round(covered / (b - a), 6)})
        rows.append({'rank': rank, 'candidate_id': candidate.get('id'), 'start': start, 'end': end,
            'duration_seconds': length, 'method': candidate.get('method'),
            'raw_retrieval_scores': candidate.get('scores'),
            'positive_overlap_seconds': round(positive, 6),
            'positive_fraction_of_candidate': round(positive / length, 6) if total else None,
            'positive_reference_fraction_covered': round(positive / total, 6) if total else None,
            'positive_temporal_iou': round(positive / (length + total - positive), 6) if total else None,
            'contrast_overlap_seconds': round(contrast, 6),
            'contrast_fraction_of_candidate': round(contrast / length, 6),
            'unannotated_seconds': round(max(0, length - positive - contrast), 6),
            'ordered_phase_coverage': phases,
            'covers_all_annotated_phases': (all(item['reference_fraction_covered'] >= 1 - 1e-6 for item in phases)
                                          if phases else None),
            'temporal_order_semantically_verified': False})
    union = merged_intervals(candidates)
    return {'caveat': CAUTION, 'evaluation_kind': query.get('evaluation_kind'),
            'expected_absent': query.get('expected_absent') is True,
            'candidate_count': len(candidates), 'annotated_positive_seconds': total,
            'annotated_contrast_seconds': sum(b - a for a, b in contrasts),
            'top_candidate': rows[0] if rows else None, 'ranked_candidates': rows,
            'all_returned_positive_reference_fraction_covered': (
                round(intersection_seconds(union, positives) / total, 6) if total else None),
            'absent_query_interpretation': ('An absent query can still return ranked retrieval results. '
                'Returned count and raw similarities are descriptive; no presence threshold or accuracy is inferred.'
                if query.get('expected_absent') is True else None)}


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def require_visual_analysis(analysis, name):
    spec = MODELS[name]
    if (analysis.get('provider') != spec['provider'] or analysis.get('ai_used') is not True
            or analysis.get('model') != spec['repository'] or analysis.get('revision') != spec['revision']):
        raise ValueError('Returned provider/model/revision differs from the requested pinned visual model.')
    if name == 'clip':
        require_clip_analysis(analysis, spec['revision'])
        return
    metadata = analysis.get('visual_search') or {}
    windows = metadata.get('windows', [])
    if (metadata.get('model') != spec['repository'] or metadata.get('revision') != spec['revision']
            or metadata.get('model_id') != spec['id'] or metadata.get('method') != spec['method']
            or metadata.get('learned_temporal_encoder') is not True or metadata.get('full_classifier') is not False
            or metadata.get('sampling_complete') is not True or metadata.get('frames_per_window') != 8
            or not isinstance(windows, list) or not windows or len(windows) > 60
            or metadata.get('window_count') != len(windows) or metadata.get('sample_count') != 8 * len(windows)):
        raise ValueError('Temporal provenance or complete eight-frame window sampling evidence is missing.')
    for window in windows:
        start, end = interval(window)
        frames = window.get('frames', [])
        if not finite(window.get('cosine_similarity')) or not -1 <= window['cosine_similarity'] <= 1 or len(frames) != 8:
            raise ValueError('Temporal window requires a raw finite cosine and eight source frames.')
        previous = -1
        for frame in frames:
            at = frame.get('time')
            if (frame.get('kind') != 'temporal_frame' or not finite(at) or not start - .001 <= at < end
                    or at < previous or not frame.get('ref')):
                raise ValueError('Temporal frame evidence must retain ordered actual timestamps and references.')
            previous = at
    for candidate in analysis.get('candidates', []):
        if (candidate.get('method') != spec['method'] or not candidate.get('evidence')
                or not any(e.get('kind') == 'temporal_frame' for e in candidate['evidence'])):
            raise ValueError('Temporal candidates must retain their actual temporal-model evidence.')


def ranked_sample_metrics(analysis, name, query):
    metadata = analysis['visual_search']
    if name == 'clip':
        return {'unit': 'individual sampled frames', **top_frame_metrics(metadata, query)}
    windows = sorted(metadata['windows'], key=lambda w: (-w['cosine_similarity'], w['start'], w['end']))[:10]
    candidates = [{**w, 'method': MODELS[name]['method'], 'scores': {'cosine_similarity': w['cosine_similarity']}}
                  for w in windows]
    return {'unit': 'eight-frame temporal windows', **overlap_metrics(candidates, query)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--presets', type=Path, default=Path(__file__).with_name('matched-presets.json'))
    parser.add_argument('--preset', choices=['basketball', 'handwashing'], required=True)
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--visual-model', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repository-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--base-url', default='http://127.0.0.1:8741')
    parser.add_argument('--job-timeout', type=float, default=600)
    parser.add_argument('--execute', action='store_true', help='Run installed local models; otherwise only freeze a plan.')
    args = parser.parse_args(argv)
    for name in ('source', 'annotations', 'presets'):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    args.provenance = (args.provenance or args.annotations.with_name('source-provenance.json')).expanduser().resolve(strict=True)
    args.repository_dir = args.repository_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir == args.repository_dir or args.repository_dir in args.output_dir.parents:
        parser.error('Choose an output directory outside the application repository.')
    if any(not p.is_file() for p in (args.source, args.annotations, args.provenance, args.presets)):
        parser.error('Source, annotations, provenance and presets must be deliberately selected local files.')
    address = urlparse(args.base_url)
    if (address.scheme != 'http' or address.hostname not in {'localhost', '127.0.0.1'}
            or address.username or address.password or address.path not in {'', '/'} or address.query or address.fragment):
        parser.error('Use only a deliberately running loopback HTTP API.')
    try:
        port = address.port
    except ValueError:
        parser.error('Invalid loopback port.')
    if port is not None and not 1 <= port <= 65535:
        parser.error('Invalid loopback port.')
    if not math.isfinite(args.job_timeout) or not 1 <= args.job_timeout <= 3600:
        parser.error('Timeout must be 1–3600 seconds.')
    if len(set(args.visual_model)) != len(args.visual_model):
        parser.error('Do not repeat a model in one comparison.')
    # Canonical serial order avoids accidental argument-order changes between runs.
    args.visual_model = [name for name in MODELS if name in args.visual_model]
    return args


def main(argv=None):
    args = parse_args(argv)
    frozen = freeze_inputs(args.source, args.annotations, args.provenance, args.presets, args.preset, args.query_id)
    query, preset = frozen['query'], frozen['preset']
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in frozen['snapshots'].items():
        (output / name).write_bytes(raw)
    run_id, started = uuid.uuid4().hex, time.monotonic()
    options = {**preset['options'], 'query': query['query']}
    report = {'schema_version': 1, 'run_id': run_id, 'status': 'planned',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_name': args.source.name, 'source_sha256': frozen['source_sha256'],
        'annotations_sha256': frozen['annotations_sha256'], 'provenance_sha256': frozen['provenance_sha256'],
        'presets_sha256': frozen['presets_sha256'], 'source_provenance': frozen['provenance'],
        'preset_id': args.preset, 'preset': preset, 'query_id': args.query_id, 'frozen_query': query,
        'matched_options': options, 'requested_models': {name: MODELS[name] for name in args.visual_model},
        'methods': {}, 'jobs': [], 'paid_usage_usd': 0, 'publishing_attempted': False,
        'limitations': [CAUTION, 'Models have different encoders and sampling plans; matched source/query/window lengths do not make frame budgets identical.',
            'CLIP compares individual frames; X-CLIP uses eight frames and omits the full classifier text-prompt generator.',
            'The benchmark does not calibrate an absent-event rejection threshold or verify a depicted event occurred.',
            'No reversed source is created here. Any later reversed video must be labeled as a synthetic perturbation.',
            'RAM safeguards are preserved. Unavailable or blocked models are reported without a fallback or automatic retry.']}

    def checkpoint():
        report['elapsed_wall_seconds'] = round(time.monotonic() - started, 3)
        save_json(output / 'report.json', report)

    def complete(client, job, name):
        record = {'benchmark_stage': name}
        report['jobs'].append(record)
        result = complete_owned_job(client, job, record, checkpoint, args.job_timeout)
        record['benchmark_stage'] = name
        save_json(output / (name + '-job-response.json'), result)
        return result, record

    def mark_submission(name, path):
        key = f'temporal-benchmark:{run_id}:{name}'
        report['pending_submission'] = {'operation': name, 'path': path, 'idempotency_key': key,
            'outcome': 'pending; response not yet observed'}
        checkpoint()
        return key

    active_method = None
    checkpoint()
    try:
        if not args.execute:
            report['execution'] = 'Dry run only: no API request, media import or inference.'
            return report
        report['status'] = 'running'
        with httpx.Client(base_url=args.base_url.rstrip('/'), timeout=60, trust_env=False, follow_redirects=False) as client:
            ensure_ready(request(client, 'GET', '/api/v1/health'))
            caps = request(client, 'GET', '/api/v1/capabilities')
            save_json(output / 'capabilities-before.json', caps)
            available = []
            for name in args.visual_model:
                spec = MODELS[name]
                model = next((m for m in caps.get('models', {}).get('models', []) if m.get('id') == spec['id']), None)
                if (not model or not model.get('available') or not model.get('runtime_available')
                        or model.get('repository') != spec['repository'] or model.get('revision') != spec['revision']):
                    report['methods'][name] = {'status': 'blocked', 'reason': 'Pinned installed model/runtime unavailable or mismatched; no download or substitution.', 'observed_model': model}
                else:
                    report['methods'][name] = {'status': 'ready', 'installed_model': model}
                    available.append(name)
            checkpoint()
            if not available:
                report['status'] = 'blocked'
                return report
            key = mark_submission('import', '/api/v1/imports')
            with args.source.open('rb') as media:
                response = client.post('/api/v1/imports', files={'file': (args.source.name, media,
                    mimetypes.guess_type(args.source.name)[0] or 'application/octet-stream')},
                    data={'title': f'Temporal comparison QA · {args.query_id} · {run_id[:8]}',
                          'genre': options['genre'], 'attribution': frozen['provenance']['attribution'],
                          'permission_status': 'licensed'}, headers={'Idempotency-Key': key})
            response.raise_for_status()
            imported = response.json()
            report['pending_submission'] = None
            save_json(output / 'import-response.json', imported)
            pid = report['project_id'] = imported['project_id']
            complete(client, imported['job'], 'import')
            project = request(client, 'GET', '/api/v1/projects/' + pid)
            if project['source']['sha256'] != frozen['source_sha256']:
                raise ValueError('Imported source differs from the frozen source hash.')
            report['source_metadata'] = project['source']['metadata']
            for name in available:
                active_method = name
                report['methods'][name]['status'] = 'running'
                # Recheck before every model. Do not lower either app or runtime reserve.
                ensure_ready(request(client, 'GET', '/api/v1/health'))
                method_started = time.monotonic()
                payload = {**options, 'visual_model': MODELS[name]['id']}
                save_json(output / (name + '-request.json'), payload)
                path = f'/api/v1/projects/{pid}/analyze'
                key = mark_submission(name, path)
                job = request(client, 'POST', path, payload, key)
                report['pending_submission'] = None
                save_json(output / (name + '-submission-response.json'), job)
                _, metrics = complete(client, job, name)
                project = request(client, 'GET', '/api/v1/projects/' + pid)
                analysis = project['analysis']
                save_json(output / (name + '-analysis-response.json'), analysis)
                require_visual_analysis(analysis, name)
                if analysis['visual_search'].get('query') != query['query'].strip():
                    raise ValueError('Visual metadata query differs from the frozen query.')
                candidate_metrics = overlap_metrics(analysis.get('candidates', []), query)
                for row in candidate_metrics['ranked_candidates']:
                    row['duration_matches_requested_preset'] = abs(row['duration_seconds'] - options['min_length']) <= .05
                report['methods'][name].update({'status': 'completed', 'request': payload,
                    'client_elapsed_seconds': round(time.monotonic() - method_started, 3),
                    'job_metrics': copy.deepcopy(metrics), 'provider': analysis['provider'],
                    'model': analysis['model'], 'revision': analysis['revision'],
                    'sampling_metadata': analysis['visual_search'], 'model_limitations': analysis.get('limitations'),
                    'candidate_overlap': candidate_metrics,
                    'ranked_sample_overlap': ranked_sample_metrics(analysis, name, query)})
                active_method = None
                checkpoint()
            report['comparison'] = {'source_query_and_requested_window_presets_matched': True,
                'score_dependent_tuning_performed': False, 'acceptance_threshold_applied': False,
                'completed_models': [name for name, value in report['methods'].items() if value['status'] == 'completed'],
                'interpretation': 'Compare labeled coverage and contrast overlap descriptively; cosine scales are not calibrated across models.'}
            report['status'] = 'completed' if all(v['status'] == 'completed' for v in report['methods'].values()) else 'partial'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)[:1500]
        if active_method:
            report['methods'][active_method].update(status=report['status'], error=report['error'])
        if report.get('pending_submission'):
            report['pending_submission']['outcome'] = 'uncertain; inspect recorded job/project/idempotency key before any retry'
        raise
    finally:
        originals = [(args.source, 'source_sha256'), (args.annotations, 'annotations_sha256'),
                     (args.provenance, 'provenance_sha256'), (args.presets, 'presets_sha256')]
        report['original_integrity'] = {name: sha256(path) == frozen[name] for path, name in originals}
        if not all(report['original_integrity'].values()):
            report['status'] = 'failed'
            report['integrity_error'] = 'An original source, label, provenance or preset changed during the run.'
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}, indent=2))
    return report


if __name__ == '__main__':
    result = main()
    if result['status'] in {'failed', 'interrupted', 'blocked', 'partial'}:
        raise SystemExit(2)
