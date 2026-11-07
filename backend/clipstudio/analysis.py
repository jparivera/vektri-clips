"""Evidence-grounded local ASR, semantic retrieval and explicitly heuristic motion cues."""
from __future__ import annotations

import gc
import hashlib
import heapq
import json
import math
import os
import ctypes
from functools import partial
from collections.abc import Sequence
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import wave
from typing import Protocol

from .models import CATALOG, ModelUnavailable, download_model, model_path, model_status

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
MAX_ANALYSIS_SECONDS = 12 * 60 * 60
SOURCE_FLAGS = ['-protocol_whitelist', 'file,pipe', '-format_whitelist',
                'mov,matroska,webm,avi,wav,mp3,ogg,flac,aac,asf,mpeg,mpegts,aiff']
_LIBC = ctypes.CDLL(None) if sys.platform.startswith('linux') else None


def _die_with_stage(expected_parent):
    """Minimal Linux pre-exec hook: a hard-killed stage cannot orphan decoders."""
    _LIBC.prctl(1, signal.SIGKILL)
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), signal.SIGKILL)


def _child_guard():
    # Keep the stage's process group for supervisor cancellation on all POSIX
    # platforms; Linux additionally covers abrupt parent death with SIGKILL.
    return {'preexec_fn': partial(_die_with_stage, os.getpid())} if _LIBC is not None else {}


class TranscriptionProvider(Protocol):
    def transcribe(self, source: Path, out: Path, options: dict, check) -> dict: ...


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str], check): ...


class VisionProvider(Protocol):
    def inspect(self, frames: list[dict], options: dict, check) -> dict: ...


class SemanticSelectionProvider(Protocol):
    def analyze(self, source: Path, metadata: dict, transcript: dict | None, options: dict, out: Path, check) -> dict: ...


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _run(args, check, timeout=1800):
    check()
    with tempfile.TemporaryFile() as error:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error, **_child_guard())
        start = time.monotonic()
        try:
            while process.poll() is None:
                check()
                if time.monotonic() - start > timeout:
                    raise RuntimeError('Media analysis subprocess exceeded its time limit.')
                time.sleep(0.1)
            if process.returncode:
                error.seek(0)
                raise RuntimeError(error.read(4000).decode('utf-8', 'replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _metadata(source):
    result = subprocess.run(['ffprobe', '-v', 'error', *SOURCE_FLAGS, '-show_streams', '-show_format', '-of', 'json', str(source)], stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30, **_child_guard())
    value = json.loads(result.stdout)
    duration = float(value.get('format', {}).get('duration', 0))
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_ANALYSIS_SECONDS:
        raise ValueError('Analysis supports finite source durations up to 12 hours.')
    return duration, sum(stream['codec_type'] == 'audio' for stream in value['streams']), any(stream['codec_type'] == 'video' for stream in value['streams'])


def _audio_stream(options, count):
    index = options.get('audio_stream', 0)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('Audio stream must be a nonnegative integer audio-track index.')
    if index >= count and not (count == 0 and index == 0):
        raise ValueError(f'Audio stream {index} is unavailable; source has {count} audio stream(s), indexed from zero.')
    return index


def transcribe(source: Path, out: Path, options: dict, check) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    check()
    duration, audio, _ = _metadata(source)
    audio_stream = _audio_stream(options, audio)
    name = options.get('model', 'tiny.en')
    language = options.get('language', 'en')
    if name not in ('tiny.en', 'tiny'):
        raise ValueError('Supported transcription models are tiny.en and tiny.')
    if language not in ('en', 'pt', 'auto') or (name == 'tiny.en' and language != 'en'):
        raise ValueError('Select tiny for Portuguese or language detection; tiny.en supports English only.')
    spec = CATALOG[name]
    result = {'segments': [], 'words': [], 'provider': 'faster-whisper', 'model': spec['repository'],
              'revision': spec['revision'], 'alignment': 'model-estimated word timestamps',
              'status': 'no_audio' if not audio else 'running', 'language': None, 'language_segments': [],
              'configuration': {'requested_model': name, 'requested_language': language, 'compute_type': 'int8', 'cpu_threads': 2, 'beam_size': 3, 'chunk_seconds': 60, 'word_timestamps': True, 'audio_stream': audio_stream},
              'audio_stream': audio_stream,
              'limitations': [('English-only tiny model; review recognition and word timings.' if name == 'tiny.en' else 'Small multilingual model; language detection and transcription may be inaccurate. Review recognition and word timings.'), 'No speaker diarization.',
                              'Independent 60-second chunks bound audio memory and can lose context at chunk boundaries.']}
    if not audio:
        result['alignment'] = 'unavailable: source has no audio'
        _write(out / 'transcript.json', result)
        return result
    location = model_path(name, verify=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ModelUnavailable('faster-whisper runtime is not installed. Run the pinned local setup; manual editing/export remain available.') from exc
    started = time.monotonic()
    model = WhisperModel(str(location), device='cpu', compute_type='int8', cpu_threads=2, num_workers=1, local_files_only=True)
    try:
        for offset in range(0, math.ceil(duration), 60):
            check()
            with tempfile.TemporaryDirectory(prefix='vektri-asr-') as directory:
                audio_path = Path(directory) / 'chunk.wav'
                _run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-threads', '1', '-ss', str(offset), *SOURCE_FLAGS, '-i', str(source),
                      '-t', str(min(60, duration - offset)), '-map', f'0:a:{audio_stream}', '-vn', '-af', 'aresample=16000:async=1:first_pts=0',
                      '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(audio_path)], check)
                segments, info = model.transcribe(str(audio_path), language=None if language == 'auto' else language, beam_size=3, word_timestamps=True, vad_filter=True,
                                                 condition_on_previous_text=False, temperature=0, no_speech_threshold=0.6)
                result['language_segments'].append({'start': offset, 'end': min(duration, offset + 60),
                                                    'language': info.language, 'probability': round(float(info.language_probability), 4)})
                languages = {item['language'] for item in result['language_segments']}
                result['language'] = next(iter(languages)) if len(languages) == 1 else 'mixed'
                result['language_probability'] = min(item['probability'] for item in result['language_segments'])
                for segment in segments:
                    check()
                    if not segment.text.strip() or segment.no_speech_prob > 0.8:
                        continue
                    start, end = max(0.0, offset + segment.start), min(duration, offset + segment.end)
                    if end <= start:
                        continue
                    result['segments'].append({'id': f'seg-{len(result["segments"])}', 'start': round(start, 3), 'end': round(end, 3),
                                               'text': segment.text.strip(), 'avg_logprob': round(segment.avg_logprob, 4),
                                               'no_speech_probability': round(segment.no_speech_prob, 4)})
                    for word in segment.words or []:
                        begin, finish = max(start, offset + word.start), min(end, offset + word.end)
                        begin = round(max(begin, result['words'][-1]['end'] if result['words'] else 0), 3)
                        finish = round(finish, 3)
                        if finish > begin and word.word.strip():
                            result['words'].append({'start': begin, 'end': finish, 'word': word.word.strip(),
                                                    'probability': round(word.probability, 4)})
            _write(out / 'transcript.partial.json', result)
    finally:
        del model
        gc.collect()
    result['status'] = 'succeeded' if result['segments'] else 'no_speech_detected'
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    result['limitations'].append(f'Uses selected audio stream {audio_stream} (zero-based audio-track index); no automatic speaker selection. Language is ' + ('detected independently per chunk.' if language == 'auto' else 'explicitly selected.'))
    _write(out / 'transcript.json', result)
    (out / 'transcript.partial.json').unlink(missing_ok=True)
    return result


class MiniLM:
    """Official pinned int8 ONNX with attention-mask mean pooling and L2 normalization."""
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        location = model_path('minilm', verify=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(location / 'onnx/model_quint8_avx2.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(location / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')

    def encode(self, texts, check):
        import numpy as np
        output = []
        for offset in range(0, len(texts), 8):
            check()
            encoded = self.tokenizer.encode_batch(texts[offset:offset + 8])
            ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            types = np.asarray([item.type_ids for item in encoded], dtype=np.int64)
            inputs = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types}
            wanted = {item.name for item in self.session.get_inputs()}
            hidden = self.session.run(None, {key: value for key, value in inputs.items() if key in wanted})[0]
            expanded = mask[:, :, None]
            pooled = (hidden * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1)
            normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            output.extend(normalized)
        return np.asarray(output, dtype=np.float32)


def _options(options, duration):
    minimum = float(options.get('min_length', 4))
    maximum = float(options.get('max_length', 10))
    separation = float(options.get('min_separation', 1))
    count = int(options.get('target_count', 5))
    if not all(math.isfinite(value) for value in [minimum, maximum, separation]) or not 0.5 <= minimum <= maximum <= 300 or not 0 <= separation <= 300 or not 1 <= count <= 30:
        raise ValueError('Clip lengths must satisfy 0.5 ≤ min ≤ max ≤ 300 seconds; separation 0–300; count 1–30.')
    exclusions = options.get('exclusions', [])
    for item in exclusions:
        if not all(math.isfinite(float(item[k])) for k in ('start', 'end')) or not 0 <= item['start'] < item['end'] <= duration:
            raise ValueError('Exclusion intervals must be finite and inside the source.')
    return minimum, maximum, separation, count, exclusions


def _overlap(a, b):
    return max(0, min(a['end'], b['end']) - max(a['start'], b['start']))


def _select(candidates, count, separation, exclusions):
    selected = []
    for candidate in sorted(candidates, key=lambda x: x['scores']['ranking'], reverse=True):
        if any(_overlap(candidate, exclusion) > 0 for exclusion in exclusions):
            continue
        if any(not (candidate['end'] + separation <= other['start'] or other['end'] + separation <= candidate['start']) for other in selected):
            continue
        words = set(re.findall(r'\w+', ' '.join(item.get('text', '') for item in candidate['evidence']).lower()))
        duplicate = False
        if len(words) >= 5:
            for other in selected:
                existing = set(re.findall(r'\w+', ' '.join(item.get('text', '') for item in other['evidence']).lower()))
                if existing and len(words & existing) / len(words | existing) > 0.82:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


# Bound ranking memory by shortlist budgets and the supported source duration.
# Every eligible sentence window is scored; only retained review items are capped.
SPEECH_BATCH_SIZE = 8
SPEECH_GLOBAL_SHORTLIST = 512
SPEECH_DIVERSE_SHORTLIST = 128
SPEECH_PER_MINUTE = 8


class _TranscriptEvidence(Sequence):
    """A shared view avoids copying long evidence ranges for every shortlist item."""
    def __init__(self, segments, first, stop):
        self.segments, self.first, self.stop = segments, first, stop

    def __len__(self):
        return self.stop - self.first

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = self.segments[self.first + index]
        return {'kind': 'transcript', 'start': item['start'], 'end': item['end'],
                'text': item['text'], 'ref': item.get('id')}


def _speech_windows(segments, duration, minimum, maximum, exclusions, check, stats):
    for index, segment in enumerate(segments):
        check()
        source_start = float(segment['start'])
        text = ''
        # Index iteration, rather than segments[index:], also avoids quadratic
        # copying of the remaining transcript for each possible start boundary.
        for last_index in range(index, len(segments)):
            check()
            last = segments[last_index]
            source_end = float(last['end'])
            if source_end - source_start > maximum:
                break
            text += (' ' if text else '') + last['text'].strip()
            padding = min(0.15, (maximum - (source_end - source_start)) / 2)
            start = max(0, source_start - padding)
            finish = min(duration, source_end + padding)
            if finish - start >= minimum:
                window = {'start': start, 'end': finish, 'text': text,
                          'first': index, 'stop': last_index + 1}
                # Rejections must be applied before bounded shortlist retention;
                # otherwise excluded high scores could hide eligible late clips.
                if any(_overlap(window, excluded) > 0 for excluded in exclusions):
                    stats['windows_excluded'] += 1
                else:
                    yield window


def _speech_candidates(transcript, duration, minimum, maximum, query, local, check, exclusions=(), statistics=None):
    # Corrected segment text is authoritative for semantic evidence when the
    # operator has edited it independently of caption words. Caption words and
    # their real timestamps remain untouched. Otherwise use word punctuation to
    # avoid ASR segments that split sentences; never invent word alignments.
    segments = []
    segment_text_corrected = bool(transcript.get('segment_text_corrected'))
    if transcript.get('words') and not segment_text_corrected:
        pending = []
        for word in sorted(transcript['words'], key=lambda item: item['start']):
            pending.append(word)
            if word['word'].rstrip().endswith(('.', '!', '?')):
                segments.append({'id': f'words-{len(segments)}', 'start': pending[0]['start'], 'end': word['end'],
                                 'text': ' '.join(item['word'] for item in pending)})
                pending = []
        if pending:
            segments.append({'id': f'words-{len(segments)}', 'start': pending[0]['start'], 'end': pending[-1]['end'],
                             'text': ' '.join(item['word'] for item in pending)})
    else:
        segments = sorted(transcript.get('segments', []), key=lambda item: item['start'])
    started = time.monotonic()
    stats = statistics if statistics is not None else {}
    stats.update(source_duration_seconds=duration, transcript_segments=len(segments), windows_scored=0,
                 windows_excluded=0, scoring_complete=False, embedding_batch_size=SPEECH_BATCH_SIZE if local else 0,
                 global_shortlist_limit=SPEECH_GLOBAL_SHORTLIST, diverse_shortlist_limit=SPEECH_DIVERSE_SHORTLIST,
                 per_minute_shortlist_limit=SPEECH_PER_MINUTE,
                 search_scope='Every fitting, non-excluded sentence window in all supplied transcript segments')
    model = None
    query_vector = None
    global_best, diverse_best, time_buckets = [], [], {}
    scored = 0
    terms = set(query.lower().split())

    def keep(heap, entry, limit):
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    def keep_diverse(heap, entry, words, limit):
        # Use the final selector's lexical near-duplicate rule before bounded
        # retention, so repeated high-density windows do not occupy every slot.
        # Token sets are temporary: retained evidence remains a shared view.
        duplicates = []
        if len(words) >= 5:
            for index, previous in enumerate(heap):
                existing = set(re.findall(r'\w+', ' '.join(item.get('text', '') for item in previous[2]['evidence']).lower()))
                if existing and len(words & existing) / len(words | existing) > 0.82:
                    if entry[:2] <= previous[:2]:
                        return
                    duplicates.append(index)
        if duplicates:
            heap[:] = [previous for index, previous in enumerate(heap) if index not in duplicates]
            heapq.heapify(heap)
        keep(heap, entry, limit)

    def score(batch):
        nonlocal model, query_vector, scored
        check()
        if local:
            if model is None:
                model = MiniLM()
                prompt = query.strip() or 'A clear, useful, self-contained explanation with enough context and a meaningful result.'
                query_vector = model.encode([prompt], check)[0]
            vectors = model.encode([window['text'] for window in batch], check)
            relevance = (vectors @ query_vector).tolist()
        else:
            relevance = [len(terms.intersection(window['text'].lower().split())) / max(1, len(terms)) for window in batch]
        for window, similarity in zip(batch, relevance):
            check()
            text = window['text']
            complete = float(text.rstrip().endswith(('.', '!', '?')))
            question_answer = float('?' in text and not text.rstrip().endswith('?'))
            density = min(1, len(text.split()) / max(1, (window['end'] - window['start']) * 2))
            similarity = max(-1.0, min(1.0, similarity))
            ranking = 0.58 * ((similarity + 1) / 2 if local else similarity) + 0.2 * complete + 0.12 * density + 0.1 * question_answer
            source_text = text[:110] + ('…' if len(text) > 110 else '')
            uncertainty = ['Hook, context and payoff require human review; scores are ranking aids.']
            if text.endswith('?'):
                uncertainty.append('Ends in a question; answer may be outside this interval.')
            if window['first'] > 0 and text.lower().startswith(('it ', 'this ', 'that ', 'because ', 'and ', 'next ', 'finally ')):
                uncertainty.append('May depend on earlier context or a tutorial prerequisite.')
            candidate = {'start': round(window['start'], 3), 'end': round(window['end'], 3), 'title': source_text,
                         'rationale': 'Semantic similarity to the request plus sentence completeness and speech density.' if local else 'Transcript boundaries, completeness and lexical matching only; no semantic model ran.',
                         'evidence': _TranscriptEvidence(segments, window['first'], window['stop']),
                         'confidence': round(min(0.85, 0.45 + 0.2 * complete + 0.15 * density), 3), 'uncertainty': uncertainty,
                         'scores': {'ranking': round(ranking, 4), 'semantic_similarity': round(similarity, 4) if local else None,
                                    'sentence_completeness': complete, 'speech_density': round(density, 4), 'question_answer_pair': question_answer},
                         'method': 'local_ai_semantic' if local else 'heuristic_transcript'}
            # Earlier equal scores retain deterministic ordering. The temporal
            # shortlist preserves alternatives outside one high-scoring region.
            entry = (candidate['scores']['ranking'], -scored, candidate)
            words = set(re.findall(r'\w+', text.lower()))
            keep(global_best, entry, SPEECH_GLOBAL_SHORTLIST)
            keep_diverse(diverse_best, entry, words, SPEECH_DIVERSE_SHORTLIST)
            keep_diverse(time_buckets.setdefault(int(window['start'] // 60), []), entry, words, SPEECH_PER_MINUTE)
            scored += 1
            stats['windows_scored'] = scored

    try:
        batch = []
        for window in _speech_windows(segments, duration, minimum, maximum, exclusions, check, stats):
            batch.append(window)
            if len(batch) == SPEECH_BATCH_SIZE:
                score(batch)
                batch = []
        if batch:
            score(batch)
    finally:
        if model is not None:
            del model
            gc.collect()
    stats.update(scoring_complete=True, elapsed_seconds=round(time.monotonic() - started, 3),
                 time_buckets=len(time_buckets), shortlist_size=0)
    if not scored:
        return [], ['No complete, non-excluded transcript window fits the requested length; widen the range or trim manually.']
    retained = {entry[1]: entry[2] for entry in global_best + diverse_best}
    for heap in time_buckets.values():
        retained.update((entry[1], entry[2]) for entry in heap)
    candidates = [retained[index] for index in sorted(retained, reverse=True)]
    stats['shortlist_size'] = len(candidates)
    limitations = ['Text embeddings compare meaning; they do not judge truth, virality or audience response.', 'Context beyond the requested maximum length may be omitted.',
                   f'Scored {scored} eligible transcript windows across the full source; retained {len(candidates)} for review using the best {SPEECH_GLOBAL_SHORTLIST} overall, {SPEECH_DIVERSE_SHORTLIST} distinct alternatives and {SPEECH_PER_MINUTE} distinct alternatives per minute. Final constraints and diversity may reduce the returned count.']
    if local:
        limitations.append('Each window embedding reads at most 256 model tokens; long passages may need shorter clip lengths or manual review.')
    if segment_text_corrected:
        limitations.append('Semantic evidence uses operator-corrected segment text and segment boundaries. Word captions and alignment are preserved separately; review them for consistency.')
    return candidates, limitations


def _signals(source, duration, audio, video, out, check, audio_stream=0):
    import numpy as np
    count = math.ceil(duration * 2)
    energy = np.zeros(count)
    motion = np.zeros(count)
    scene = np.zeros(count)
    if audio:
        audio_path = out / 'analysis-audio.wav'
        try:
            _run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-threads', '1', *SOURCE_FLAGS, '-i', str(source), '-map', f'0:a:{audio_stream}',
                  '-vn', '-af', 'aresample=8000:async=1:first_pts=0', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_s16le', str(audio_path)], check)
            with wave.open(str(audio_path), 'rb') as reader:
                for index in range(count):
                    check()
                    values = np.frombuffer(reader.readframes(4000), dtype='<i2').astype(np.float32) / 32768
                    if len(values):
                        energy[index] = float(np.sqrt(np.mean(values * values)))
        finally:
            audio_path.unlink(missing_ok=True)
    if video:
        args = ['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-threads', '1', '-filter_threads', '1', *SOURCE_FLAGS, '-i', str(source), '-vf', 'fps=2,scale=160:90', '-pix_fmt', 'gray', '-f', 'rawvideo', '-an', 'pipe:1']
        with tempfile.TemporaryFile() as error:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error, **_child_guard())
            previous = None
            try:
                for index in range(count):
                    check()
                    data = process.stdout.read(160 * 90)
                    if len(data) < 160 * 90:
                        break
                    current = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                    if previous is not None:
                        differences = np.abs(current - previous)
                        motion[index] = np.mean(differences) / 255
                        scene[index] = float(np.mean(differences > 24))
                    previous = current
                process.stdout.close()
                process.wait(timeout=30)
                if process.returncode:
                    error.seek(0)
                    raise RuntimeError(error.read(2000).decode('utf-8', 'replace'))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    rows = [{'time': index / 2, 'audio_rms': round(float(energy[index]), 6), 'motion': round(float(motion[index]), 6), 'scene_change_fraction': round(float(scene[index]), 6)} for index in range(count)]
    _write(out / 'signals.json', {'sample_seconds': 0.5, 'frame_size': [160, 90], 'signals': rows, 'audio_stream': audio_stream,
                                 'method': 'deterministic signal analysis; no semantic event recognition'})
    return rows


def _action_candidates(rows, duration, minimum, maximum):
    if not rows:
        return []
    max_energy = max(row['audio_rms'] for row in rows) or 1
    max_motion = max(row['motion'] for row in rows) or 1
    candidates = []
    length = min(maximum, max(minimum, min(8, maximum)))
    for index, row in enumerate(rows):
        # A quiet static source does not receive manufactured highlights.
        if row['motion'] < 0.003 and row['audio_rms'] < 0.025 and row['scene_change_fraction'] < 0.1:
            continue
        moving = min(1, row['motion'] / max_motion)
        energy = min(1, row['audio_rms'] / max_energy)
        cut = min(1, row['scene_change_fraction'] / 0.5)
        # Audio alone is deliberately weak; a loud cut isn't proof of an event.
        ranking = 0.55 * moving + 0.25 * energy + 0.2 * cut
        start = max(0, min(duration - length, row['time'] - length * 0.45))
        end = min(duration, start + length)
        if end - start < minimum:
            continue
        label = 'Visible change' if moving >= 0.3 else 'Audio energy change'
        if cut >= 0.6:
            label = 'Possible scene transition'
        candidates.append({'start': round(start, 3), 'end': round(end, 3), 'title': f'{label} near {row["time"]:.1f}s',
                           'rationale': 'Sampled visual difference, scene-change fraction and audio energy with surrounding context; no event identity inferred.',
                           'evidence': [{'kind': 'signal', **row, 'ref': f'signals.json#/signals/{index}'}],
                           'confidence': round(min(0.65, 0.25 + ranking * 0.4), 3),
                           'uncertainty': ['Heuristic-only; the system has not identified a goal, crash, winner, person or meaningful action.', 'Motion and sound can be camera movement, edits, overlays or noise.'],
                           'scores': {'ranking': round(ranking, 4), 'motion': round(moving, 4), 'audio_energy': round(energy, 4), 'scene_change': round(cut, 4)},
                           'method': 'heuristic_visual_audio'})
    return candidates


def analyze(source: Path, metadata: dict, transcript: dict | None, options: dict, out: Path, check) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    check()
    started = time.monotonic()
    duration, audio, video = _metadata(source)
    audio_stream = _audio_stream(options, audio)
    minimum, maximum, separation, count, exclusions = _options(options, duration)
    genre = options.get('genre', 'speech')
    mode = options.get('mode', 'local')
    if genre not in ('speech', 'tutorial', 'action', 'mixed'):
        raise ValueError('Unknown genre; choose speech, tutorial, action or mixed.')
    visual_requested = options.get('visual_search', False)
    impact_requested = options.get('impact_detection', False)
    if impact_requested and (visual_requested or genre not in ('action', 'mixed') or mode not in ('local', 'heuristic')):
        raise ValueError('Impact detection requires action/mixed local or heuristic mode, without visual search.')
    if visual_requested and (mode != 'local' or not str(options.get('query', '')).strip()):
        raise ValueError('Visual search requires local mode and a description in English of visible content.')
    if visual_requested and not video:
        raise ValueError('Visual search requires a video stream; use transcript search for audio-only sources.')
    if not visual_requested and genre in ('speech', 'tutorial', 'mixed') and transcript and transcript.get('segments') and transcript.get('audio_stream', 0) != audio_stream:
        raise ValueError('The saved transcript belongs to a different audio stream. Explicitly transcribe the selected track before analyzing; saved corrections were preserved.')
    if mode not in ('local', 'heuristic', 'cloud', 'agent'):
        raise ValueError('Unknown analysis mode.')
    if mode == 'cloud':
        raise ModelUnavailable('Cloud analysis is unconfigured. No data was transmitted or money spent. Use local analysis or the auditable agent evidence workflow.')
    if mode == 'agent':
        raise ValueError('Agent mode uses the project evidence packet and schema-validated agent-decisions endpoint; no agent runtime is configured.')
    candidates, limitations = [], []
    speech_search, visual_search, impact_search = None, None, None
    semantic_ran = False
    if visual_requested:
        visual_model = options.get('visual_model', 'clip-vit-b32')
        if visual_model == 'xclip-b16-k600':
            from .temporal import temporal_candidates as retrieve_visual
        elif visual_model == 'clip-vit-b32':
            from .vision import visual_candidates as retrieve_visual
        else:
            raise ValueError('Unknown visual model.')
        visual, visual_search, notes = retrieve_visual(source, duration, {**options, 'min_length': minimum, 'max_length': maximum}, out, check)
        candidates.extend(visual)
        limitations.extend(notes)
        semantic_ran = bool(visual_search.get('window_count')) if visual_model == 'xclip-b16-k600' else True
    elif genre in ('speech', 'tutorial', 'mixed') and transcript and transcript.get('segments'):
        semantic_ran = mode == 'local' and transcript.get('language', 'en') == 'en'
        if mode == 'local' and not semantic_ran:
            limitations.append('MiniLM is English-only in this installation. Non-English transcript candidates use lexical/timing matching, not multilingual semantic inference.')
        speech_search = {}
        speech, notes = _speech_candidates(transcript, duration, minimum, maximum, str(options.get('query', ''))[:2000], semantic_ran, check, exclusions, speech_search)
        candidates.extend(speech)
        limitations.extend(notes)
        if not speech:
            semantic_ran = False
    elif genre in ('speech', 'tutorial'):
        limitations.append('No transcript available. Transcribe audio first or mark events manually; no speech suggestions were fabricated.')
    if not visual_requested and genre in ('action', 'mixed'):
        if impact_requested:
            from .impacts import impact_candidates, _configuration
            _configuration(options)
            if audio:
                impacts, impact_search, notes = impact_candidates(source, duration, options, out, check, audio_stream)
                candidates.extend(impacts)
                limitations.extend(notes)
            else:
                impact_search = {'status': 'no_audio', 'method': 'spectral_flux_onset', 'ai_used': False}
                limitations.append('No audio stream: no sound onsets were fabricated. Use motion analysis or mark an event manually.')
            limitations.append('Impact clip boundaries use lead-in/follow-through, not transcript minimum/maximum lengths. Final clip count, exclusion and separation constraints still apply.')
        else:
            rows = _signals(source, duration, audio, video, out, check, audio_stream)
            candidates.extend(_action_candidates(rows, duration, minimum, maximum))
            limitations.append('Action evidence here uses motion/audio/scene heuristics. Enable experimental visual search with an English description for local image-text retrieval; temporal event understanding remains unvalidated.')
        if options.get('query'):
            limitations.append('The text request applies only to transcript semantic retrieval. Sound and motion heuristics cannot resolve named events.')
    selected = _select(candidates, count, separation, exclusions)
    for candidate in selected:
        # Materialize only the bounded final result, not every ranked window.
        candidate['evidence'] = list(candidate['evidence'])
        identity = json.dumps([candidate['start'], candidate['end'], candidate['method'], candidate['title']], ensure_ascii=False)
        candidate['id'] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    from .review import attach_reviews
    attach_reviews(source, duration, selected, transcript, out, check, has_video=video, audio_stream=audio_stream)
    if len(selected) < count:
        limitations.append(f'Returned {len(selected)} of {count} requested clips after length, exclusion and separation constraints; no padding with poor duplicates.')
    selected_model = options.get('visual_model', 'clip-vit-b32') if visual_requested else 'minilm'
    provider = ('local_onnx_xclip' if selected_model == 'xclip-b16-k600' else 'local_onnx_clip') if visual_requested else 'local_onnx_minilm+heuristics' if semantic_ran else 'heuristics'
    result = {'candidates': selected, 'provider': provider,
              'model': CATALOG[selected_model]['repository'] if semantic_ran else None,
              'revision': CATALOG[selected_model]['revision'] if semantic_ran else None,
              'mode': mode, 'genre': genre, 'status': 'succeeded' if selected else 'no_candidates', 'ai_used': semantic_ran,
              'configuration': {**options, 'minimum_length': minimum, 'maximum_length': maximum, 'cpu_threads': 2, 'audio_stream': audio_stream},
              'limitations': list(dict.fromkeys(limitations)), 'elapsed_seconds': round(time.monotonic() - started, 3),
              'speech_search': speech_search, 'visual_search': visual_search, 'impact_search': impact_search,
              'paid_usage': 0, 'data_transmitted': False}
    check()
    _write(out / 'analysis.json', result)
    return result
