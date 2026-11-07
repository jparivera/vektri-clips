#!/usr/bin/env python3
"""Compare heuristics and a selected local visual model against frozen annotations.

Imports one new QA project and renders the first returned visual candidate without
manual timing changes. Defaults to CLIP; --visual-model xclip-b16-k600 selects the
guarded temporal encoder. Requires a running loopback app and installed weights.
Never downloads models/footage, contacts accounts, spends or posts.
All source footage, raw responses and outputs remain outside the repository.
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
import re
import time
from urllib.parse import urlparse
import uuid

import httpx

from benchmark_delivery import download, poll, request, save_json, sha256, verify_output

CLIP_REPOSITORY = 'Xenova/clip-vit-base-patch32'
TEMPORAL_REPOSITORY = 'imbcmdth/xclip-onnx'
TEMPORAL_METHOD = 'local_xclip_temporal_retrieval'
CAUTION = ('Annotations are incomplete, coarse visual references made before model scores; '
           'unannotated time is unknown, not a negative. Overlap is not a human editorial '
           'rating, event recognition accuracy, or proof that a depicted action occurred.')


def interval(item):
    if not isinstance(item, dict):
        raise ValueError('Intervals must be objects.')
    values = (item.get('start'), item.get('end'))
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError('Intervals require finite numeric start/end values.')
    start, end = map(float, values)
    if start < 0 or end <= start:
        raise ValueError('Intervals require 0 <= start < end.')
    return start, end


def merged_intervals(items):
    result = []
    for start, end in sorted(interval(item) for item in items):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection_seconds(left, right):
    """Both inputs are already merged; no overlapping labels are double-counted."""
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def freeze_annotations(path, source_hash, query_id, expected_hash=None):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash.lower():
        raise ValueError('Annotation SHA-256 differs from the pre-frozen expected hash.')
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise ValueError('Unsupported annotation schema_version.')
    if document.get('source_sha256') != source_hash:
        raise ValueError('Annotations are bound to a different source SHA-256.')
    if document.get('model_scores_observed_before_annotation') is not False:
        raise ValueError('Annotations must explicitly state model scores were not observed beforehand.')
    queries = document.get('queries', [])
    matches = [q for q in queries if isinstance(q, dict) and q.get('id') == query_id]
    if len(matches) != 1:
        raise ValueError('Select exactly one existing, unique annotation query ID.')
    selected = copy.deepcopy(matches[0])
    query = selected.get('query')
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError('The frozen query must contain 1–2000 characters.')
    positives = selected.get('positive_intervals')
    contrasts = selected.get('contrast_intervals')
    if not isinstance(positives, list) or not positives or not isinstance(contrasts, list):
        raise ValueError('Provide positive_intervals and contrast_intervals arrays.')
    if intersection_seconds(merged_intervals(positives), merged_intervals(contrasts)) > 1e-8:
        raise ValueError('Positive and contrast annotations must not overlap.')
    for item in positives:
        if 'core_start' in item or 'core_end' in item:
            a, b = interval({'start': item.get('core_start'), 'end': item.get('core_end')})
            if not item['start'] <= a < b <= item['end']:
                raise ValueError('Positive core intervals must remain within their scene envelope.')
    return raw, document, selected, digest


def overlap_metrics(candidates, query):
    positives = merged_intervals(query['positive_intervals'])
    contrasts = merged_intervals(query['contrast_intervals'])
    positive_total = sum(end - start for start, end in positives)
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        start, end = interval(candidate)
        length = end - start
        positive = intersection_seconds([(start, end)], positives)
        contrast = intersection_seconds([(start, end)], contrasts)
        rows.append({'rank': rank, 'candidate_id': candidate.get('id'), 'start': start, 'end': end,
                     'positive_overlap_seconds': round(positive, 6),
                     'positive_fraction_of_candidate': round(positive / length, 6),
                     'positive_reference_fraction_covered': round(positive / positive_total, 6),
                     'positive_temporal_iou': round(positive / (length + positive_total - positive), 6),
                     'contrast_overlap_seconds': round(contrast, 6),
                     'contrast_fraction_of_candidate': round(contrast / length, 6),
                     'unannotated_seconds': round(max(0.0, length - positive - contrast), 6),
                     'method': candidate.get('method'), 'model_scores': candidate.get('scores')})
    selected_union = merged_intervals(candidates)
    coverage = intersection_seconds(selected_union, positives)
    result = {'annotation_caveat': CAUTION, 'candidate_count': len(candidates),
              'annotated_positive_seconds': round(positive_total, 6),
              'annotated_contrast_seconds': round(sum(b - a for a, b in contrasts), 6),
              'top_candidate': rows[0] if rows else None, 'ranked_candidates': rows,
              'all_returned_positive_reference_fraction_covered': round(coverage / positive_total, 6),
              'positive_intervals_hit': sum(intersection_seconds(selected_union, [(a, b)]) > 0 for a, b in positives),
              'positive_interval_count': len(positives)}
    if any('core_start' in item for item in query['positive_intervals']):
        cores = [{'start': item['core_start'], 'end': item['core_end']}
                 for item in query['positive_intervals'] if 'core_start' in item]
        result['safe_positive_cores'] = overlap_metrics(candidates, {**query, 'positive_intervals': cores})
    return result


def ensure_ready(health):
    if health.get('worker_active') or health.get('queue_length'):
        raise RuntimeError('Other jobs are running or queued; no existing job was changed.')
    worker = health.get('worker', {})
    if not worker.get('alive'):
        raise RuntimeError('A live worker is required before importing benchmark media.')
    minimum = worker.get('minimum_available_ram_bytes', 1536 * 1024 * 1024)
    if worker.get('state') == 'waiting_memory' or not any(
            (entry.get('available_ram_bytes') or 0) >= minimum for entry in worker.get('workers', [])):
        raise RuntimeError('Worker is below its available-RAM safety threshold; no import was submitted.')


def require_clip_analysis(analysis, expected_revision):
    if (analysis.get('provider') != 'local_onnx_clip' or analysis.get('ai_used') is not True
            or analysis.get('model') != CLIP_REPOSITORY or analysis.get('revision') != expected_revision):
        raise ValueError('Visual search did not return the verified installed local CLIP provider/model/revision.')
    metadata = analysis.get('visual_search')
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('CLIP sampling/coverage metadata is missing.')
    samples = metadata.get('samples', [])
    if (metadata.get('model') != CLIP_REPOSITORY or metadata.get('revision') != expected_revision
            or metadata.get('sample_count') != len(samples) or not samples
            or metadata.get('sampling_complete') is not True):
        raise ValueError('CLIP model provenance or completed frame-sampling evidence is missing.')
    for item in samples:
        if any(isinstance(item.get(key), bool) or not isinstance(item.get(key), (int, float))
               or not math.isfinite(item[key]) for key in ('time', 'cosine_similarity')):
            raise ValueError('CLIP frame timestamps and cosine scores must be finite.')
    for candidate in analysis.get('candidates', []):
        if (candidate.get('method') != 'local_clip_visual' or not candidate.get('evidence')
                or not any(item.get('kind') == 'visual_frame' for item in candidate['evidence'])):
            raise ValueError('CLIP candidates must retain actual visual-model evidence.')


def require_temporal_analysis(analysis, expected_revision):
    if (analysis.get('provider') != 'local_onnx_xclip' or analysis.get('ai_used') is not True
            or analysis.get('model') != TEMPORAL_REPOSITORY or analysis.get('revision') != expected_revision):
        raise ValueError('Temporal search did not return the installed local X-CLIP provider/model/revision.')
    metadata = analysis.get('visual_search') or {}
    windows = metadata.get('windows', [])
    if (metadata.get('model') != TEMPORAL_REPOSITORY or metadata.get('revision') != expected_revision
            or metadata.get('model_id') != 'xclip-b16-k600' or metadata.get('method') != TEMPORAL_METHOD
            or metadata.get('learned_temporal_encoder') is not True or metadata.get('full_classifier') is not False
            or metadata.get('sampling_complete') is not True or metadata.get('frames_per_window') != 8
            or not isinstance(windows, list) or not windows or len(windows) > 60
            or metadata.get('window_count') != len(windows) or metadata.get('sample_count') != 8 * len(windows)):
        raise ValueError('X-CLIP provenance or completed eight-frame window sampling evidence is missing.')
    for window in windows:
        start, end = interval(window)
        score = window.get('cosine_similarity')
        frames = window.get('frames', [])
        if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not -1 <= score <= 1 or len(frames) != 8):
            raise ValueError('X-CLIP windows require a finite raw cosine and eight frames.')
        previous = -1
        for frame in frames:
            at = frame.get('time')
            if (isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at)
                    or not start - .001 <= at < end or at < previous
                    or frame.get('kind') != 'temporal_frame' or not frame.get('ref')):
                raise ValueError('X-CLIP frame evidence requires ordered actual timestamps and references.')
            previous = at
    for candidate in analysis.get('candidates', []):
        if (candidate.get('method') != TEMPORAL_METHOD or not candidate.get('evidence')
                or not any(e.get('kind') == 'temporal_frame' for e in candidate['evidence'])):
            raise ValueError('X-CLIP candidates must retain actual temporal-model evidence.')


def top_temporal_metrics(metadata, query, count=10):
    windows = sorted(metadata.get('windows', []), key=lambda w: (-w['cosine_similarity'], w['start'], w['end']))[:count]
    rows = [{**window, 'method': TEMPORAL_METHOD, 'scores': {'cosine_similarity': window['cosine_similarity']}}
            for window in windows]
    return {'unit': 'eight-frame temporal windows', **overlap_metrics(rows, query)}


def top_frame_metrics(metadata, query, count=10):
    """Rank actual sampled-frame scores; preserve negatives and original values."""
    positives, contrasts = merged_intervals(query['positive_intervals']), merged_intervals(query['contrast_intervals'])
    result = []
    for rank, frame in enumerate(sorted(metadata.get('samples', []), key=lambda item: item['cosine_similarity'], reverse=True)[:count], 1):
        at = frame['time']
        classification = ('positive' if any(a <= at < b for a, b in positives)
                          else 'contrast' if any(a <= at < b for a, b in contrasts) else 'unannotated')
        result.append({'rank': rank, 'time': at, 'cosine_similarity': frame['cosine_similarity'],
                       'annotation': classification, 'frame_ref': frame.get('frame_ref')})
    return {'annotation_caveat': CAUTION, 'top_frames': result,
            'counts': {label: sum(item['annotation'] == label for item in result)
                       for label in ('positive', 'contrast', 'unannotated')}}


def render_timeline(project, candidate, credit_label):
    """Use the top returned interval exactly; annotations cannot adjust the edit."""
    start, end = interval(candidate)
    timeline = copy.deepcopy(project['timeline'])
    timeline['segments'] = [{'id': 'top-clip-result', 'start': start, 'end': end,
                             'crop': {'mode': 'fit', 'x': .5, 'y': .5, 'zoom': 1, 'second_x': .75, 'keyframes': []},
                             'gain_db': 0}]
    timeline['output'] = {'width': 360, 'height': 640, 'fps': 24, 'crf': 24}
    timeline['audio'].update(normalize=False, fade=0, stream_index=0)
    caption_end = min(end, start + 3)
    timeline['captions'].update(enabled=True, burn_in=True, highlight=False, font_size=20, margin=24,
                                words=[{'start': start, 'end': caption_end, 'word': credit_label}])
    return timeline


def complete_owned_job(client, job, record, checkpoint, timeout):
    record['id'] = job['id']
    try:
        return poll(client, job, record, checkpoint, timeout)
    except BaseException:
        record.setdefault('id', job['id'])
        if not record.get('benchmark_cancel_requested') and record.get('status') not in {'succeeded', 'failed', 'cancelled'}:
            try:
                cancelled = request(client, 'POST', '/api/v1/jobs/' + job['id'] + '/cancel')
                record['benchmark_cancel_requested'] = True
                record['cancellation_observed_status'] = cancelled.get('status')
            except Exception as error:
                record['cancellation_outcome'] = 'uncertain; inspect this job before retrying'
                record['cancellation_error'] = str(error)[:500]
        checkpoint()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--annotations-sha256', help='Optional expected hash recorded before scoring')
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--attribution', required=True)
    parser.add_argument('--permission-status', choices=['unknown', 'permission-recorded', 'owned', 'licensed'], default='unknown')
    parser.add_argument('--credit-label', default='Source credit in export manifest')
    parser.add_argument('--base-url', default='http://127.0.0.1:8741')
    parser.add_argument('--job-timeout', type=float, default=600)
    parser.add_argument('--min-length', type=float, default=4)
    parser.add_argument('--max-length', type=float, default=8)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--visual-model', choices=['clip-vit-b32', 'xclip-b16-k600'], default='clip-vit-b32')
    args = parser.parse_args(argv)
    args.source = args.source.expanduser().resolve(strict=True)
    args.annotations = args.annotations.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if args.output_dir == repo or repo in args.output_dir.parents:
        parser.error('Choose an output directory outside the source repository.')
    if not args.source.is_file() or not args.annotations.is_file():
        parser.error('Source and annotations must be deliberately selected local files.')
    address = urlparse(args.base_url)
    if (address.scheme != 'http' or address.hostname not in {'127.0.0.1', 'localhost'}
            or address.username or address.password or address.path not in {'', '/'} or address.query or address.fragment):
        parser.error('Use only the deliberately running loopback HTTP API.')
    try:
        port = address.port
    except ValueError:
        parser.error('Invalid loopback port.')
    if port is not None and not 1 <= port <= 65535:
        parser.error('Invalid loopback port.')
    if not math.isfinite(args.job_timeout) or not 1 <= args.job_timeout <= 3600:
        parser.error('--job-timeout must be 1–3600 seconds.')
    if not all(math.isfinite(x) for x in (args.min_length, args.max_length)) or not 1 <= args.min_length <= args.max_length <= 300:
        parser.error('Require 1 <= min-length <= max-length <= 300.')
    if not 1 <= args.count <= 30 or not 1 <= len(args.credit_label) <= 80 or not 1 <= len(args.attribution) <= 2000:
        parser.error('Count must be 1–30, credit-label 1–80 and attribution 1–2000 characters.')
    if args.annotations_sha256 and not re.fullmatch('[a-fA-F0-9]{64}', args.annotations_sha256):
        parser.error('--annotations-sha256 must be a SHA-256 hex digest.')
    return args


def main(argv=None):
    args = parse_args(argv)
    visual_name = 'clip' if args.visual_model == 'clip-vit-b32' else 'temporal'
    visual_repository = CLIP_REPOSITORY if visual_name == 'clip' else TEMPORAL_REPOSITORY
    source_hash = sha256(args.source)
    raw_annotations, annotations, query, annotation_hash = freeze_annotations(
        args.annotations, source_hash, args.query_id, args.annotations_sha256)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'annotations-frozen.json').write_bytes(raw_annotations)
    run_id, started = uuid.uuid4().hex, time.monotonic()
    options = {'genre': 'action', 'query': query['query'], 'transcribe': False,
               'min_length': args.min_length, 'max_length': args.max_length,
               'target_count': args.count, 'min_separation': 2}
    report = {'schema_version': 1, 'run_id': run_id, 'status': 'running',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_name': args.source.name, 'source_sha256': source_hash,
              'annotations_sha256': annotation_hash, 'query_id': args.query_id, 'frozen_query': query,
              'annotation_method': annotations.get('method'), 'annotation_boundary_precision': annotations.get('boundary_precision'),
              'shared_options': options, 'visual_model': args.visual_model,
              'attribution': args.attribution, 'permission_status': args.permission_status,
              'methods': {}, 'jobs': [], 'paid_usage_usd': 0, 'publishing_attempted': False,
              'limitations': [CAUTION, 'The heuristic baseline cannot resolve the text query; identical query is recorded for comparison.',
                             'CLIP compares individual sampled frames with text; it does not establish motion, outcomes or identities.',
                             'X-CLIP ranks eight-frame temporal embeddings; the two-tower export omits video-conditioned text prompts and does not verify events or order.',
                             'RAM fields are separate job-process and child-process maxima, not simultaneous total memory.',
                             'Source credit is an authored overlay, not speech transcription.']}

    def checkpoint():
        report['elapsed_wall_seconds'] = round(time.monotonic() - started, 3)
        save_json(output / 'report.json', report)

    def complete(client, job, name):
        record = {'benchmark_stage': name}
        report['jobs'].append(record)
        result = complete_owned_job(client, job, record, checkpoint, args.job_timeout)
        record['benchmark_stage'] = name
        save_json(output / (name + '-job-response.json'), result)
        checkpoint()
        return result, record

    def submit(client, path, payload, name):
        key = f'visual:{run_id}:{name}'
        report['pending_submission'] = {'operation': name, 'path': path, 'idempotency_key': key,
                                        'outcome': 'pending; no successful response observed yet'}
        checkpoint()
        job = request(client, 'POST', path, payload, key)
        save_json(output / (name + '-submission-response.json'), job)
        report['pending_submission'] = None
        return job

    checkpoint()
    try:
        with httpx.Client(base_url=args.base_url.rstrip('/'), timeout=60, trust_env=False, follow_redirects=False) as client:
            health = request(client, 'GET', '/api/v1/health')
            ensure_ready(health)
            capabilities = request(client, 'GET', '/api/v1/capabilities')
            save_json(output / 'capabilities-before.json', capabilities)
            model = next((m for m in capabilities.get('models', {}).get('models', []) if m.get('repository') == visual_repository), None)
            if not model or not model.get('available') or not model.get('runtime_available'):
                raise RuntimeError(f'Installed local {visual_name} model/runtime is unavailable; this benchmark will not download it.')
            report['installed_' + visual_name + '_model'] = model
            report['pending_submission'] = {'operation': 'import', 'path': '/api/v1/imports',
                'idempotency_key': 'visual:' + run_id + ':import', 'outcome': 'pending; no successful response observed yet'}
            checkpoint()
            with args.source.open('rb') as media:
                response = client.post('/api/v1/imports', files={'file': (args.source.name, media,
                    mimetypes.guess_type(args.source.name)[0] or 'application/octet-stream')},
                    data={'title': f'Visual comparison QA · {args.query_id} · {run_id[:8]}', 'genre': 'action',
                          'attribution': args.attribution, 'permission_status': args.permission_status},
                    headers={'Idempotency-Key': 'visual:' + run_id + ':import'})
            response.raise_for_status()
            imported = response.json()
            report['pending_submission'] = None
            save_json(output / 'import-response.json', imported)
            pid = report['project_id'] = imported['project_id']
            complete(client, imported['job'], 'import')
            project = request(client, 'GET', '/api/v1/projects/' + pid)
            if project['source']['sha256'] != source_hash:
                raise ValueError('Imported source hash does not match the frozen source.')
            duration = project['source']['metadata']['duration']
            if any(interval(item)[1] > duration + .05 for kind in ('positive_intervals', 'contrast_intervals') for item in query[kind]):
                raise ValueError('Frozen annotation interval exceeds the imported source duration.')
            report['source_metadata'] = project['source']['metadata']
            for name, visual in [('heuristic', False), (visual_name, True)]:
                ensure_ready(request(client, 'GET', '/api/v1/health'))
                method_started = time.monotonic()
                payload = {**options, 'visual_search': visual, 'mode': 'local' if visual else 'heuristic'}
                if visual:
                    payload['visual_model'] = args.visual_model
                save_json(output / (name + '-request.json'), payload)
                job = submit(client, f'/api/v1/projects/{pid}/analyze', payload, name)
                result, metrics = complete(client, job, name)
                project = request(client, 'GET', '/api/v1/projects/' + pid)
                analysis = project['analysis']
                save_json(output / (name + '-analysis-response.json'), analysis)
                save_json(output / (name + '-project-response.json'), project)
                report['methods'][name] = {'request': payload, 'job_metrics': copy.deepcopy(metrics),
                    'client_elapsed_seconds': round(time.monotonic() - method_started, 3),
                    'analysis_provider': analysis.get('provider'), 'analysis_model': analysis.get('model'),
                    'analysis_revision': analysis.get('revision'), 'ai_used': analysis.get('ai_used'),
                    'visual_search': analysis.get('visual_search'), 'analysis_limitations': analysis.get('limitations'),
                    'overlap': overlap_metrics(analysis.get('candidates', []), query)}
                if visual:
                    if visual_name == 'clip':
                        require_clip_analysis(analysis, model['revision'])
                        report['methods'][name]['frame_overlap'] = top_frame_metrics(analysis['visual_search'], query)
                    else:
                        require_temporal_analysis(analysis, model['revision'])
                        report['methods'][name]['window_overlap'] = top_temporal_metrics(analysis['visual_search'], query)
                    if analysis['visual_search'].get('query') != query['query'].strip():
                        raise ValueError('Visual metadata query differs from the frozen query.')
                elif analysis.get('ai_used') or any(c.get('method') != 'heuristic_visual_audio' for c in analysis.get('candidates', [])):
                    raise ValueError('Baseline did not remain purely heuristic visual/audio analysis.')
                checkpoint()
            top = {name: result['overlap']['top_candidate'] for name, result in report['methods'].items()}
            report['comparison'] = {
                'query_unchanged_between_methods': True,
                'annotation_caveat': CAUTION,
                'top_candidate_positive_fraction_delta_' + visual_name + '_minus_heuristic': (
                    round(top[visual_name]['positive_fraction_of_candidate'] - top['heuristic']['positive_fraction_of_candidate'], 6)
                    if top[visual_name] is not None and top['heuristic'] is not None else None),
                'top_candidate_contrast_fraction_delta_' + visual_name + '_minus_heuristic': (
                    round(top[visual_name]['contrast_fraction_of_candidate'] - top['heuristic']['contrast_fraction_of_candidate'], 6)
                    if top[visual_name] is not None and top['heuristic'] is not None else None),
                'interpretation': 'Descriptive comparison of this query only; no score-dependent retuning or acceptance threshold.'}
            candidates = project['analysis'].get('candidates', [])
            if not candidates:
                raise ValueError(f'{visual_name} returned no candidate; no manual or heuristic interval will be substituted.')
            chosen = copy.deepcopy(candidates[0])
            report['rendered_candidate'] = chosen
            report['render_selection_rule'] = f'First returned {visual_name} candidate; timing and ordering unchanged, regardless of annotation overlap.'
            timeline = render_timeline(project, chosen, args.credit_label)
            saved = request(client, 'PUT', f'/api/v1/projects/{pid}/timeline', {'revision': project['revision'], 'timeline': timeline})
            report['revision'], report['timeline'] = saved['revision'], timeline
            rendered, _ = complete(client, submit(client, f'/api/v1/projects/{pid}/render',
                {'revision': saved['revision']}, 'render'), 'render')
            export = rendered['result']['export']
            report['export_id'], report['render_verification'] = export['id'], export['manifest']['verification']
            video_path = output / f'top-{visual_name}-candidate.mp4'
            package_path = output / f'top-{visual_name}-candidate.zip'
            report['video'] = download(client, export['video_url'], video_path)
            report['package'] = download(client, export['package_url'], package_path)
            report['independent_verification'] = verify_output(video_path, package_path,
                export['manifest'], chosen['end'] - chosen['start'])
            report['preview_url'] = args.base_url.rstrip('/') + '/?project=' + pid
            report['status'] = 'completed'
            report['semantic_acceptance'] = 'unrated; inspect both frozen-reference overlap results, including misses'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)[:1500]
        if report.get('pending_submission'):
            report['pending_submission']['outcome'] = 'uncertain; inspect the local project/job queue and recorded idempotency key before retrying'
        raise
    finally:
        try:
            report['source_unchanged'] = sha256(args.source) == source_hash
            report['annotations_unchanged'] = sha256(args.annotations) == annotation_hash
            if not report['source_unchanged'] or not report['annotations_unchanged']:
                report['status'] = 'failed'
                report['integrity_error'] = 'Original source or annotation file changed during the benchmark.'
        except OSError as error:
            report['status'], report['integrity_error'] = 'failed', str(error)[:500]
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
    if report['status'] != 'completed':
        raise RuntimeError(report.get('integrity_error', report.get('error', 'Benchmark did not complete.')))
    print(json.dumps({'status': report['status'], 'query_id': args.query_id, 'project_id': report['project_id'],
                      'export_id': report['export_id'], 'report': str(output / 'report.json'),
                      'elapsed_wall_seconds': report['elapsed_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
