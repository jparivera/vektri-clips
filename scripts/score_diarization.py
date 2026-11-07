#!/usr/bin/env python3
"""Independent exact-interval DER diagnostic; no model or audio inference.

This deliberately documents its own scoring conventions instead of claiming
bit-for-bit equivalence with NIST md-eval or any third-party implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import wave

MAX_SPEAKERS = 6
MAX_SEGMENTS = 10000
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_AUDIO_BYTES = 128 * 1024 * 1024
SHA256 = re.compile(r'^[0-9a-f]{64}$')


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_segments(segments, duration, label):
    if not isinstance(segments, list) or len(segments) > MAX_SEGMENTS:
        raise ValueError(f'{label} segments must be a list of at most {MAX_SEGMENTS}.')
    cleaned = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f'{label} segment {index} must be an object.')
        start, end, speaker = (segment.get(k) for k in ('start', 'end', 'speaker'))
        if not finite_number(start) or not finite_number(end) or not 0 <= start < end <= duration:
            raise ValueError(f'{label} segment {index} requires finite 0 <= start < end <= duration.')
        if not isinstance(speaker, str) or not speaker.strip() or len(speaker) > 128:
            raise ValueError(f'{label} segment {index} requires a nonempty speaker string, at most 128 characters.')
        cleaned.append({'start': float(start), 'end': float(end), 'speaker': speaker})
    if len({s['speaker'] for s in cleaned}) > MAX_SPEAKERS:
        raise ValueError(f'{label} has more than {MAX_SPEAKERS} speakers; bounded exhaustive mapping refused.')
    return cleaned


def _regions(reference, hypothesis, duration, collar, exclude_overlap):
    """Event sweep; same-speaker concurrent segments count as one active speaker."""
    events = defaultdict(lambda: {'reference': Counter(), 'hypothesis': Counter(), 'collar': 0})
    events[0.0]; events[duration]
    for kind, segments in (('reference', reference), ('hypothesis', hypothesis)):
        for segment in segments:
            events[segment['start']][kind][segment['speaker']] += 1
            events[segment['end']][kind][segment['speaker']] -= 1
            if kind == 'reference' and collar:
                # ALL input reference starts/ends, including same-speaker internal boundaries.
                for boundary in (segment['start'], segment['end']):
                    left, right = max(0., boundary-collar), min(duration, boundary+collar)
                    if right > left:
                        events[left]['collar'] += 1
                        events[right]['collar'] -= 1
    counts = {'reference': Counter(), 'hypothesis': Counter()}
    collar_count = 0
    points = sorted(events)
    scored = []
    totals = {'duration_seconds': duration, 'scored_wall_seconds': 0.,
              'collar_excluded_seconds': 0., 'reference_overlap_excluded_seconds': 0.,
              'reference_overlap_total_seconds': 0., 'reference_total_speaker_seconds': 0.}
    for left, right in zip(points, points[1:]):
        for kind in counts:
            counts[kind].update(events[left][kind])
            for speaker in list(counts[kind]):
                if counts[kind][speaker] == 0:
                    del counts[kind][speaker]
                elif counts[kind][speaker] < 0:
                    raise AssertionError('Invalid internal event sweep count.')
        collar_count += events[left]['collar']
        ref, hyp = frozenset(counts['reference']), frozenset(counts['hypothesis'])
        elapsed = right-left
        totals['reference_total_speaker_seconds'] += elapsed*len(ref)
        if len(ref) > 1:
            totals['reference_overlap_total_seconds'] += elapsed
        if collar_count:
            totals['collar_excluded_seconds'] += elapsed
        elif exclude_overlap and len(ref) > 1:
            totals['reference_overlap_excluded_seconds'] += elapsed
        else:
            scored.append((elapsed, ref, hyp))
            totals['scored_wall_seconds'] += elapsed
    return scored, totals


def _optimal_mapping(regions, all_hypothesis_speakers):
    """Maximize correctly coactive speaker-seconds, injectively, <=6! mappings."""
    refs = sorted(set().union(*(ref for _, ref, _ in regions))) if regions else []
    hyps = sorted(set().union(*(hyp for _, _, hyp in regions))) if regions else []
    coactivity = defaultdict(float)
    for seconds, active_ref, active_hyp in regions:
        for ref in active_ref:
            for hyp in active_hyp:
                coactivity[hyp, ref] += seconds
    candidates = []
    if len(hyps) <= len(refs):
        candidates = ({hyp: ref for hyp, ref in zip(hyps, permutation)}
                      for permutation in itertools.permutations(refs, len(hyps)))
    else:
        candidates = ({hyp: ref for ref, hyp in zip(refs, permutation)}
                      for permutation in itertools.permutations(hyps, len(refs)))
    best, best_score, examined = {}, -1., 0
    for candidate in candidates:
        examined += 1
        correct = sum(coactivity[hyp, ref] for hyp, ref in candidate.items())
        # Sorted permutations make exact ties deterministic; no tolerance changes optimum.
        if correct > best_score:
            best, best_score = candidate, correct
    return {hyp: best.get(hyp) for hyp in sorted(all_hypothesis_speakers)}, examined


def score_condition(reference, hypothesis, duration, collar_seconds=0., exclude_reference_overlap=False):
    if not finite_number(duration) or not 0 < duration <= 86400:
        raise ValueError('Duration must be finite, positive and at most 86400 seconds.')
    if not finite_number(collar_seconds) or not 0 <= collar_seconds <= 1:
        raise ValueError('Collar must be finite and between 0 and 1 seconds.')
    if type(exclude_reference_overlap) is not bool:
        raise ValueError('Overlap exclusion must be boolean.')
    reference = validate_segments(reference, duration, 'Reference')
    hypothesis = validate_segments(hypothesis, duration, 'Hypothesis')
    regions, totals = _regions(reference, hypothesis, float(duration), float(collar_seconds), exclude_reference_overlap)
    mapping, examined = _optimal_mapping(regions, {s['speaker'] for s in hypothesis})
    ref_seconds = missed = false_alarm = confusion = correct_seconds = hyp_seconds = 0.
    for seconds, ref, hyp in regions:
        correct = sum(mapping.get(speaker) in ref for speaker in hyp if mapping.get(speaker) is not None)
        ref_seconds += seconds*len(ref)
        hyp_seconds += seconds*len(hyp)
        missed += seconds*max(len(ref)-len(hyp), 0)
        false_alarm += seconds*max(len(hyp)-len(ref), 0)
        confusion += seconds*(min(len(ref), len(hyp))-correct)
        correct_seconds += seconds*correct
    error = missed+false_alarm+confusion
    return {**totals, 'collar_seconds_each_side': collar_seconds,
            'reference_overlap': 'excluded' if exclude_reference_overlap else 'included',
            'reference_speaker_seconds': ref_seconds, 'hypothesis_speaker_seconds': hyp_seconds,
            'correct_speaker_seconds': correct_seconds, 'missed_speech_seconds': missed,
            'false_alarm_seconds': false_alarm, 'speaker_confusion_seconds': confusion,
            'error_speaker_seconds': error, 'der': error/ref_seconds if ref_seconds else None,
            'der_percent': 100*error/ref_seconds if ref_seconds else None,
            'denominator_zero': ref_seconds == 0, 'hypothesis_to_reference_mapping': mapping,
            'mapping_candidates_examined': examined, 'scored_atomic_interval_count': len(regions)}


def evaluate(reference, hypothesis, duration):
    return [score_condition(reference, hypothesis, duration, collar, exclude)
            for collar in (0., .25) for exclude in (False, True)]


def file_hash(path, limit):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'Input is missing or exceeds its size limit: {path.name}')
    digest = hashlib.sha256()
    total = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            total += len(chunk)
            if total > limit:
                raise ValueError('Input grew beyond its bounded size limit.')
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    file_hash(path, MAX_JSON_BYTES)
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant {value} is not accepted.')
    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError('JSON root must be an object.')
    return value


def verify_audio(gold_path, gold):
    audio = gold.get('audio')
    if not isinstance(audio, dict):
        raise ValueError('Gold requires audio path, file SHA-256 and decoded PCM SHA-256.')
    name = audio.get('path')
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Gold audio path must be a contained relative file path.')
    source = (gold_path.parent/name).resolve()
    if not source.is_relative_to(gold_path.parent.resolve()):
        raise ValueError('Gold audio symlink escapes the manifest directory.')
    for key in ('sha256', 'pcm_sha256'):
        if not SHA256.fullmatch(str(audio.get(key, ''))):
            raise ValueError(f'Gold audio requires a lowercase {key}.')
    source_hash = file_hash(source, MAX_AUDIO_BYTES)
    if source_hash != audio['sha256']:
        raise ValueError('Audio file SHA-256 differs from frozen gold.')
    digest = hashlib.sha256()
    with wave.open(str(source), 'rb') as handle:
        if handle.getcomptype() != 'NONE':
            raise ValueError('Audio must be uncompressed PCM WAV.')
        metadata = {'sample_rate': handle.getframerate(), 'channels': handle.getnchannels(),
                    'sample_width_bytes': handle.getsampwidth(), 'sample_frames': handle.getnframes()}
        expected_bytes = metadata['sample_frames']*metadata['channels']*metadata['sample_width_bytes']
        consumed = 0
        while chunk := handle.readframes(65536):
            consumed += len(chunk)
            digest.update(chunk)
        if consumed != expected_bytes:
            raise ValueError('PCM WAV is truncated.')
    if digest.hexdigest() != audio['pcm_sha256']:
        raise ValueError('PCM SHA-256 differs from frozen gold.')
    duration = metadata['sample_frames']/metadata['sample_rate']
    if not finite_number(gold.get('duration_seconds')) or abs(duration-gold['duration_seconds']) > 1/metadata['sample_rate'] + 1e-10:
        raise ValueError('Gold duration differs from PCM sample count by more than one frame.')
    return source, {'path': name, 'sha256': source_hash, 'pcm_sha256': digest.hexdigest(),
                    'pcm_hash_definition': 'WAV data frames concatenated, interleaved channel bytes, excluding RIFF headers',
                    'duration_seconds': duration, **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--gold-sha256', required=True)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if not SHA256.fullmatch(args.gold_sha256) or file_hash(args.gold, MAX_JSON_BYTES) != args.gold_sha256:
        raise ValueError('Gold SHA-256 does not match the required frozen hash.')
    prediction_hash = file_hash(args.prediction, MAX_JSON_BYTES)
    gold, prediction = load_json(args.gold), load_json(args.prediction)
    if gold.get('schema_version') != 1:
        raise ValueError('Gold schema_version must be 1.')
    source, audio = verify_audio(args.gold, gold)
    if prediction.get('source_sha256') not in (None, audio['sha256']):
        raise ValueError('Prediction declares a different audio source.')
    conditions = evaluate(gold.get('segments'), prediction.get('segments'), gold['duration_seconds'])
    report = {'schema_version': 1, 'kind': 'independent_continuous_interval_der_diagnostic',
              'gold_sha256': args.gold_sha256, 'prediction_sha256': prediction_hash,
              'scorer_sha256': file_hash(Path(__file__), MAX_JSON_BYTES), 'audio': audio,
              'prediction_source_binding': 'declared_verified' if prediction.get('source_sha256') else 'not_declared',
              'reference_type': gold.get('reference_type', 'unspecified; no human-VAD reference claim'),
              'reference_limitations': gold.get('limitations', []),
              'interpretation': 'Errors are conditional on the supplied reference intervals and type; not a natural-conversation or human-VAD accuracy claim.',
              'conventions': {'timeline': '[0,duration), speech segments [start,end); interval sweep, no frame quantization',
                'collar': 'remove collar seconds on EACH side of every input reference start/end; union masks; clipped to recording',
                'overlap': 'excluded condition removes times with >1 distinct reference speakers AFTER collar masking and BEFORE speaker mapping',
                'same_speaker_overlap': 'concurrent same-label input segments count once as an active speaker',
                'mapping': 'independently optimize injective mapping per condition by exhaustive permutations, <=6 speakers per input; unmatched hypotheses map to null',
                'error': 'miss=max(Nref-Nhyp,0); FA=max(Nhyp-Nref,0); confusion=min(Nref,Nhyp)-correct_mapped; integrate speaker-seconds',
                'denominator': 'reference speaker-seconds within scoring mask; DER is null if zero, can exceed1',
                'equivalence': 'explicit independent convention; no claim of bit-for-bit md-eval/pyannote equivalence'},
              'conditions': conditions, 'model_inference_performed': False, 'gold_unchanged': True}
    if file_hash(args.gold, MAX_JSON_BYTES) != args.gold_sha256 or file_hash(args.prediction, MAX_JSON_BYTES) != prediction_hash or file_hash(source, MAX_AUDIO_BYTES) != audio['sha256']:
        raise ValueError('An input changed during scoring; report was not written.')
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'report': str(args.output.resolve()), 'gold_sha256': args.gold_sha256,
                      'conditions': [{'collar': c['collar_seconds_each_side'], 'overlap': c['reference_overlap'], 'der': c['der']} for c in conditions]}))
    return report


if __name__ == '__main__':
    main()
