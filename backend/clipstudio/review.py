"""Deterministic review evidence, not an editorial or semantic quality model.

Only the final shortlist is annotated. Existing agent packet fingerprints remain
mandatory for external proposals; this fingerprint is an evidence receipt only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from . import media

MAX_CANDIDATES = 30
MAX_QUOTE = 500
MAX_DURATION = 12 * 60 * 60
SOURCE_END_TOLERANCE = .001  # Match Timeline.validate_source and rounded candidates.


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Review timestamps must be finite numbers.')
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _transcript_rows(transcript, duration, check):
    """Match speech selection's word/segment authority and sentence assembly.

    Word sentences use their actual first/last word timestamps, never invented
    token alignments. Quotes are substrings of the canonical sentence text used
    by _speech_candidates. Explicit segment-text corrections take precedence.
    """
    transcript = transcript or {}
    rows = []
    if transcript.get('words') and not transcript.get('segment_text_corrected'):
        words = []
        for item in transcript['words']:
            check()
            try:
                start, end = _number(item.get('start')), _number(item.get('end'))
            except (ValueError, TypeError):
                return []
            if not (0 <= start < min(end, duration) and end <= duration + SOURCE_END_TOLERANCE
                    and isinstance(item.get('word'), str) and item['word'].strip()):
                # Do not join speech across malformed evidence or revive stale
                # segment text when authoritative word evidence is unusable.
                return []
            words.append(item)
        segments, pending = [], []
        for word in sorted(words, key=lambda item: item['start']):
            check()
            pending.append(word)
            if word['word'].rstrip().endswith(('.', '!', '?')):
                segments.append({'id': f'words-{len(segments)}', 'start': pending[0]['start'],
                                 'end': word['end'], 'text': ' '.join(item['word'] for item in pending)})
                pending = []
        if pending:
            segments.append({'id': f'words-{len(segments)}', 'start': pending[0]['start'],
                             'end': pending[-1]['end'], 'text': ' '.join(item['word'] for item in pending)})
    else:
        segments = transcript.get('segments') or []
    for item in segments:
        check()
        try:
            start, end = _number(item.get('start')), _number(item.get('end'))
        except (ValueError, TypeError):
            continue
        text = item.get('text')
        if (0 <= start < min(end, duration) and end <= duration + SOURCE_END_TOLERANCE
                and isinstance(text, str) and text.strip()):
            rows.append({'kind': 'transcript', 'start': start, 'end': end,
                         'quote': text[:MAX_QUOTE], 'excerpt': len(text) > MAX_QUOTE,
                         'ref': str(item.get('id', ''))[:120]})
    return sorted(rows, key=lambda row: (row['start'], row['end']))


def _observation(summary, evidence):
    return {'status': 'observed' if evidence else 'unknown', 'summary': summary, 'evidence': evidence}


def attach_reviews(source: Path, duration: float, candidates: list[dict], transcript: dict | None,
                   out: Path, check, has_video: bool = True, audio_stream: int = 0) -> None:
    """Append bounded review receipts and up to three source JPEGs per candidate.

    Run inside the existing isolated analysis stage. Never edits timing, ranking,
    or selection. Invalid candidate inputs fail before any candidate is mutated.
    FFmpeg failures degrade to explicit missing evidence, but cancellation always
    propagates. Repeated calls replace deterministic paths, not append artifacts.
    """
    duration = _number(duration)
    if not 0 < duration <= MAX_DURATION or len(candidates) > MAX_CANDIDATES:
        raise ValueError('Review supports at most 30 candidates and 12 hours of source.')
    if isinstance(audio_stream, bool) or not isinstance(audio_stream, int) or audio_stream < 0:
        raise ValueError('Audio stream must be a nonnegative audio-track index.')
    for candidate in candidates:
        start, end = _number(candidate['start']), _number(candidate['end'])
        if not 0 <= start < min(end, duration) or end > duration + SOURCE_END_TOLERANCE:
            raise ValueError('Review candidate is outside source bounds.')
        _digest({k: v for k, v in candidate.items() if k != 'review'})
    check()
    if not candidates:
        return
    digest = hashlib.sha256()
    with Path(source).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            check()
            digest.update(block)
    source_sha = digest.hexdigest()
    rows = _transcript_rows(transcript, duration, check)
    mismatch = bool(transcript and transcript.get('audio_stream', 0) != audio_stream)
    if mismatch:
        rows = []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates):
        check()
        # Existing selectors round to milliseconds; tolerate the same source-end
        # slack as the timeline without seeking beyond the actual source.
        start, end = float(candidate['start']), min(float(candidate['end']), duration)
        # Keep exact supplied transcript timestamps and candidate containment;
        # only playback/frame seeks use the source-capped end below.
        inside = [row for row in rows if start <= row['start'] and row['end'] <= candidate['end']]
        before = [row for row in rows if row['end'] <= start]
        after = [row for row in rows if row['start'] >= end]
        context = ([max(before, key=lambda row: row['end'])] if before else []) + (after[:1] if after else [])
        signals = []
        for item in candidate.get('evidence', []):
            if item.get('kind') not in ('signal', 'sound_onset'):
                continue
            try:
                time = _number(item.get('time'))
                keys = ('flux', 'rms_db', 'threshold') if item['kind'] == 'sound_onset' else ('audio_rms', 'motion', 'scene_change_fraction')
                facts = {key: _number(item[key]) for key in keys if key in item}
            except (ValueError, TypeError):
                continue
            valid = (all(0 <= v <= 1 for v in facts.values()) if item['kind'] == 'signal' else
                     set(facts) == {'flux', 'rms_db', 'threshold'} and
                     0 <= facts['flux'] <= 1e12 and 0 <= facts['threshold'] <= 1e12 and
                     -300 <= facts['rms_db'] <= 0 and item.get('audio_stream', audio_stream) == audio_stream)
            if start <= time <= end and facts and valid:
                signals.append({'kind': item['kind'], 'time': time, **facts})
            if len(signals) == 3:
                break
        anchor = signals[0]['time'] if signals else (start + end) / 2
        playback = {'start': start, 'end': end, 'context_start': max(0., start - 3),
                    'context_end': min(duration, end + 3)}
        observations = {
            'hook': _observation('First fully contained transcript excerpt; hook effectiveness is unknown.' if inside else 'No fully contained transcript excerpt; hook effectiveness is unknown.', inside[:1]),
            'context': _observation('Nearest preceding/following transcript excerpts; whether this clip stands alone needs human review.' if context else 'No preceding/following transcript excerpt available; context sufficiency is unknown.', context),
            'payoff': _observation('Last fully contained transcript excerpt; a resolved payoff is not established.' if inside else 'No fully contained transcript excerpt; a resolved payoff is not established.', inside[-1:]),
            'signals': _observation('Recorded signal measurements only; no event identity or significance inferred.' if signals else 'No supported signal measurements in this candidate.', signals),
        }
        warnings = []
        if candidate['end'] > duration:
            warnings.append('Candidate end is rounded to milliseconds; review playback is capped at the actual source duration.')
        if mismatch:
            warnings.append('Saved transcript belongs to another audio track; no transcript evidence used.')
        if audio_stream != 0:
            warnings.append('Source proxy plays the first audio track; selected analysis track differs. Review selected-track audio in the export.')
        receipt = {'source_sha256': source_sha, 'duration': duration, 'audio_stream': audio_stream,
                   'candidate': {k: v for k, v in candidate.items() if k != 'review'},
                   'observations': observations, 'playback': playback}
        review = {'schema_version': 1, 'method': 'deterministic_evidence_review', 'status': 'needs_human_review',
                  'fingerprint': _digest(receipt), 'source_sha256': source_sha,
                  'observations': observations, 'playback': playback, 'frames': [], 'warnings': warnings,
                  'limitations': ['Hook, context and payoff quality are unknown; human review is required.',
                                  'Quotes are exact transcript substrings with supplied source-time bounds, not verified speech.',
                                  'Three approximate source seeks cannot verify a complete event. Play the full clip and surrounding context.',
                                  'This receipt does not replace the agent evidence packet fingerprint.']}
        if not has_video:
            warnings.append('Audio-only source: video frame evidence is unavailable.')
        else:
            # Before/after are outside the clip where possible, not merely
            # adjacent frames at its midpoint. At is a signal anchor or midpoint.
            times = [('before', max(0., start - 1)), ('at', anchor), ('after', min(duration - min(.05, duration / 2), end + 1))]
            for role, time in times:
                check()
                name = f'review-{index:02d}-{role}.jpg'
                target = out / name
                temporary = out / f'review-{index:02d}-{role}.partial.jpg'
                frame = {'role': role, 'time': time, 'approximate_seek': True,
                         'status': 'unavailable', 'path': None}
                cancelled = False

                def checked():
                    nonlocal cancelled
                    try:
                        check()
                    except BaseException:
                        cancelled = True
                        raise

                try:
                    # Remove any old successful render so a failed retry cannot
                    # advertise stale evidence for changed source or boundaries.
                    target.unlink(missing_ok=True)
                    media._run(['-ss', f'{time:.6f}', *media._input(Path(source)), '-map', '0:V:0',
                                '-frames:v', '1', '-an', '-vf',
                                'scale=480:270:force_original_aspect_ratio=decrease', '-q:v', '3',
                                str(temporary.resolve())], checked, timeout=30)
                    checked()
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        raise media.MediaError('Decoder produced no frame.')
                    os.replace(temporary, target)
                    frame.update(status='available', path=name)
                except (media.MediaError, OSError):
                    if cancelled:
                        raise
                    warnings.append(f'{role.capitalize()} frame unavailable: source frame rendering failed. Play the source to review.')
                finally:
                    temporary.unlink(missing_ok=True)
                review['frames'].append(frame)
        check()
        candidate['review'] = review
