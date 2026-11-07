#!/usr/bin/env python3
"""Run a deliberate local-file import, analysis, edit and verified export.

Requires an already running local app. Never downloads footage/models, connects
accounts or publishes. Media and report stay in an explicitly chosen output
directory outside this repository. Real footage is not bundled with the code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import mimetypes
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

import httpx

from benchmark_delivery import download, request, save_json, sha256, verify_output, poll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--attribution', required=True)
    parser.add_argument('--credit-label', default='Source credit in export manifest', help='Short authored overlay, not a transcript')
    parser.add_argument('--permission-status', choices=['unknown', 'permission-recorded', 'owned', 'licensed'], default='unknown')
    parser.add_argument('--genre', choices=['speech', 'tutorial', 'action', 'mixed'], default='action')
    parser.add_argument('--transcribe', action='store_true', help='Explicitly enable installed English tiny.en ASR only for suitable English speech')
    parser.add_argument('--base-url', default='http://127.0.0.1:8741')
    parser.add_argument('--job-timeout', type=float, default=600)
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    if output == root or root in output.parents:
        parser.error('Choose an output directory outside the source repository; third-party media must stay local and uncommitted.')
    address = urlparse(args.base_url)
    if address.scheme != 'http' or address.hostname not in {'127.0.0.1', 'localhost'} or address.username or address.password or address.path not in {'', '/'} or address.query or address.fragment:
        parser.error('Use only the deliberately running loopback HTTP API.')
    if not math.isfinite(args.job_timeout) or not 1 <= args.job_timeout <= 3600:
        parser.error('--job-timeout must be 1–3600 seconds.')
    if not source.is_file():
        parser.error('source must be an intentionally selected local file.')
    if args.genre == 'action' and args.transcribe:
        parser.error('Use mixed, speech or tutorial for English transcription; action analysis does not run ASR.')
    if not 1 <= len(args.credit_label) <= 80:
        parser.error('--credit-label must be 1–80 characters.')
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    started = time.monotonic()
    report = {'schema_version': 1, 'run_id': run_id, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'running', 'source_name': source.name, 'source_sha256': sha256(source),
              'attribution': args.attribution, 'permission_status': args.permission_status,
              'authored_credit_label': args.credit_label,
              'genre': args.genre, 'transcription_requested': args.transcribe, 'jobs': [],
              'paid_usage_usd': 0, 'publishing_attempted': False,
              'limitations': ['Action analysis is motion/audio/scene heuristics, not recognition of race events.',
                              'Caption in this benchmark is an explicitly authored source-credit label, not an ASR transcript.',
                              'Mechanical checks do not establish human editorial quality or real-world lip sync.']}

    def checkpoint():
        report['elapsed_wall_seconds'] = round(time.monotonic() - started, 3)
        save_json(output / 'report.json', report)

    def complete(client, job):
        record = {}
        report['jobs'].append(record)
        try:
            return poll(client, job, record, checkpoint, args.job_timeout)
        except BaseException:
            # A transport failure is not proof that accepted work stopped.
            # Cancel only this known benchmark job, never the whole queue.
            record.setdefault('id', job['id'])
            if not record.get('benchmark_cancel_requested') and record.get('status') not in {'succeeded', 'failed', 'cancelled'}:
                try:
                    cancelled = request(client, 'POST', '/api/v1/jobs/' + job['id'] + '/cancel')
                    record['benchmark_cancel_requested'] = True
                    record['cancellation_observed_status'] = cancelled.get('status')
                except Exception as error:
                    record['cancellation_outcome'] = 'uncertain; inspect the recorded job before retrying'
                    record['cancellation_error'] = str(error)[:500]
            checkpoint()
            raise

    checkpoint()
    try:
        with httpx.Client(base_url=args.base_url.rstrip('/'), timeout=60, trust_env=False, follow_redirects=False) as client:
            health = request(client, 'GET', '/api/v1/health')
            if health.get('worker_active') or health.get('queue_length'):
                raise RuntimeError('Other jobs are running or queued; no existing job was changed.')
            worker = health.get('worker', {})
            if not worker.get('alive'):
                raise RuntimeError('Worker readiness is unavailable. Start or restart the updated local app before importing test media.')
            minimum = worker.get('minimum_available_ram_bytes', 1536 * 1024 * 1024)
            if worker.get('state') == 'waiting_memory' or not any((item.get('available_ram_bytes') or 0) >= minimum for item in worker.get('workers', [])):
                raise RuntimeError('The worker needs at least 1.5 GiB available RAM. No import was submitted and the guard was not bypassed.')
            with source.open('rb') as media:
                response = client.post('/api/v1/imports',
                    files={'file': (source.name, media, mimetypes.guess_type(source.name)[0] or 'application/octet-stream')},
                    data={'title': 'Real footage acceptance · ' + source.stem, 'genre': args.genre,
                          'attribution': args.attribution, 'permission_status': args.permission_status},
                    headers={'Idempotency-Key': 'real:' + run_id + ':import'})
            response.raise_for_status()
            imported = response.json()
            pid = report['project_id'] = imported['project_id']
            complete(client, imported['job'])
            project = request(client, 'GET', '/api/v1/projects/' + pid)
            if project['source']['sha256'] != report['source_sha256']:
                raise ValueError('Imported source hash mismatch.')
            report['source_metadata'] = project['source']['metadata']
            options = {'genre': args.genre, 'mode': 'local' if args.transcribe else 'heuristic',
                       'transcribe': args.transcribe, 'min_length': 6, 'max_length': 12, 'target_count': 4, 'min_separation': 2}
            complete(client, request(client, 'POST', f'/api/v1/projects/{pid}/analyze', options, 'real:' + run_id + ':analyze'))
            project = request(client, 'GET', '/api/v1/projects/' + pid)
            analysis = project['analysis']
            report['analysis'] = analysis
            candidates = analysis.get('candidates', [])
            if not candidates:
                raise ValueError('Analysis returned no candidates; no manual clip was substituted for a successful analysis.')
            chosen = candidates[:2]
            for candidate in chosen:
                request(client, 'POST', f'/api/v1/projects/{pid}/feedback', {'candidate_id': candidate['id'], 'decision': 'accept',
                        'note': 'Accepted for mechanical real-footage test; not a human editorial rating.'})
            timeline = copy.deepcopy(project['timeline'])
            timeline['segments'] = [dict(id=f'real-{index}', start=c['start'], end=c['end'],
                crop={'mode': 'fit', 'x': .5, 'y': .5, 'zoom': 1, 'second_x': .75, 'keyframes': []}, gain_db=-2)
                for index, c in enumerate(reversed(chosen))]
            timeline['output'] = {'width': 360, 'height': 640, 'fps': 24, 'crf': 24}
            timeline['audio'].update(normalize=True, fade=.12, stream_index=0)
            first = timeline['segments'][0]
            timeline['captions'].update(enabled=True, burn_in=True, highlight=False, font_size=20, margin=24,
                words=[{'start': first['start'] + .2, 'end': min(first['end'], first['start'] + 3),
                        'word': args.credit_label}])
            saved = request(client, 'PUT', f'/api/v1/projects/{pid}/timeline', {'revision': project['revision'], 'timeline': timeline})
            report['revision'] = saved['revision']
            report['timeline'] = timeline
            rendered = complete(client, request(client, 'POST', f'/api/v1/projects/{pid}/render', {'revision': saved['revision']}, 'real:' + run_id + ':render'))
            export = rendered['result']['export']
            report['export_id'] = export['id']
            report['render_verification'] = export['manifest']['verification']
            report['video'] = download(client, export['video_url'], output / 'clip.mp4')
            report['package'] = download(client, export['package_url'], output / 'package.zip')
            expected = sum(segment['end'] - segment['start'] for segment in timeline['segments'])
            report['independent_verification'] = verify_output(output / 'clip.mp4', output / 'package.zip', export['manifest'], expected)
            report['source_unchanged'] = sha256(source) == report['source_sha256']
            if not report['source_unchanged']:
                raise ValueError('The supplied source changed during the benchmark.')
            report['preview_url'] = args.base_url.rstrip('/') + '/?project=' + pid
            report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)[:1500]
        checkpoint()
        raise
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print(json.dumps({'status': report['status'], 'project_id': report['project_id'], 'export_id': report['export_id'],
                      'report': str(output / 'report.json'), 'elapsed_wall_seconds': report['elapsed_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
