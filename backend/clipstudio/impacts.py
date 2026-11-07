"""Bounded, local sound-onset suggestions. This module does not recognize events.

Independent implementation: 25 ms Hann spectra, 10 ms hops, positive band flux.
Flux is the sum of positive magnitudes normalized by Hann-window sum, not an
upstream detector's raw FFT units. A trailing noise floor supplements the user's
absolute threshold. First 100 ms are baseline warm-up, not proposed events.
"""
from __future__ import annotations

from collections import deque
import heapq
import math
from pathlib import Path
import tempfile
import wave

SAMPLE_RATE = 16000
WINDOW = 400
HOP = 160
CHUNK_SECONDS = 30
LIMITATIONS = [
    'Sound onsets only; no event, person, action, or off-camera source is identified. Review the video and surrounding audio.',
    'Recording gain, music, wind, edits and microphone noise affect detection. Thresholds are normalized spectral-flux units, not semantic confidence.',
    'The first 100 ms establish a noise baseline. Quiet, gradual or tightly spaced sounds may be missed.',
]


def _configuration(options):
    defaults = {'impact_low_hz': 1000., 'impact_high_hz': 6000.,
                'impact_threshold': .04, 'impact_min_gap': .6,
                'impact_lead': .4, 'impact_follow': .5}
    bounds = {'impact_low_hz': (0, 7999), 'impact_high_hz': (1, 8000),
              'impact_threshold': (0.000001, 1000), 'impact_min_gap': (.01, 300),
              'impact_lead': (0, 300), 'impact_follow': (0, 300)}
    config = {}
    for key, default in defaults.items():
        value = options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number.')
        low, high = bounds[key]
        if not low <= value <= high:
            raise ValueError(f'{key} must be between {low} and {high}.')
        config[key] = float(value)
    if config['impact_low_hz'] >= config['impact_high_hz']:
        raise ValueError('Impact frequency band must have low < high, within the 8 kHz Nyquist limit.')
    # FFT bins are 40 Hz apart. Reject an empty band instead of silently abstaining.
    if math.ceil(config['impact_low_hz'] / 40) > math.floor(config['impact_high_hz'] / 40):
        raise ValueError('Impact frequency band must contain at least one 40 Hz FFT bin.')
    if config['impact_lead'] + config['impact_follow'] < .04:
        raise ValueError('Impact lead and follow must total at least 0.04 seconds.')
    limit = options.get('impact_max_candidates', 120)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError('impact_max_candidates must be an integer between 1 and 500.')
    config['impact_max_candidates'] = limit
    exclusions = options.get('exclusions', [])
    if not isinstance(exclusions, (list, tuple)) or len(exclusions) > 100:
        raise ValueError('Impact exclusions must be a list of at most 100 intervals.')
    config['exclusions'] = []
    for interval in exclusions:
        if not isinstance(interval, dict):
            raise ValueError('Impact exclusions require start and end times.')
        start, end = interval.get('start'), interval.get('end')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
            raise ValueError('Impact exclusion times must be finite numbers.')
        if not 0 <= start < end <= 86400:
            raise ValueError('Invalid impact exclusion interval.')
        config['exclusions'].append({'start': start, 'end': end})
    return config


def detect_onsets(blocks, options=None, check=lambda: None):
    """Analyze an iterable of bounded mono PCM float blocks, preserving hop state.

    Caller owns block-size bounds. Production feeds <=30s blocks. This function
    retains one block, a <25ms tail, one spectrum, 1s noise history and a bounded
    strongest-peak heap. Nonfinite samples fail closed rather than poisoning ranks.
    """
    import numpy as np
    config = _configuration(options or {})
    check()
    frequencies = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    band = (frequencies >= config['impact_low_hz']) & (frequencies <= config['impact_high_hz'])
    window = np.hanning(WINDOW)
    normalization = window.sum()
    tail = np.empty(0, dtype=np.float32)
    previous = None
    history = deque(maxlen=100)
    left, middle = None, None
    pending = None
    retained = []
    frame_number = 0
    total_peaks = 0

    def retain(peak):
        nonlocal total_peaks
        total_peaks += 1
        entry = (peak['flux'], -peak['time'], peak)
        if len(retained) < config['impact_max_candidates']:
            heapq.heappush(retained, entry)
        elif entry[:2] > retained[0][:2]:
            heapq.heapreplace(retained, entry)

    def consider(peak):
        nonlocal pending
        # Exclude before spacing and top-K retention: discarded loud sounds must
        # not suppress usable neighbouring sounds or fill the bounded shortlist.
        start = max(0., peak['time'] - config['impact_lead'])
        end = peak['time'] + config['impact_follow']
        if any(start < excluded['end'] and end > excluded['start'] for excluded in config['exclusions']):
            return
        if pending is None:
            pending = peak
        elif peak['time'] - pending['time'] >= config['impact_min_gap'] - 1e-9:
            retain(pending)
            pending = peak
        elif peak['flux'] > pending['flux']:
            pending = peak

    for block in blocks:
        check()
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 1 or not np.isfinite(block).all():
            raise ValueError('Impact PCM must be finite mono samples.')
        values = np.concatenate((tail, block))
        position = 0
        while position + WINDOW <= len(values):
            if frame_number % 100 == 0:
                check()
            frame = values[position:position + WINDOW]
            magnitude = np.abs(np.fft.rfft(frame * window))[band] / normalization
            flux = float(np.maximum(magnitude - previous, 0).sum()) if previous is not None else 0.
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            floor = max(config['impact_threshold'], 3 * (sum(history) / len(history) if history else 0.))
            row = {'time': (frame_number * HOP + WINDOW / 2) / SAMPLE_RATE,
                   'flux': flux, 'rms_db': 20 * math.log10(max(rms, 1e-12)),
                   'threshold': floor,
                   'eligible': frame_number >= 10 and rms >= .001 and flux >= floor}
            if middle is not None and left is not None:
                if middle['eligible'] and middle['flux'] > left['flux'] and middle['flux'] >= row['flux']:
                    consider({key: value for key, value in middle.items() if key != 'eligible'})
            left, middle = middle, row
            previous = magnitude
            history.append(flux)
            frame_number += 1
            position += HOP
        tail = values[position:].copy()
    # Last full frame may peak at EOF; it has no right neighbour to disprove it.
    if middle and left and middle['eligible'] and middle['flux'] > left['flux']:
        consider({key: value for key, value in middle.items() if key != 'eligible'})
    if pending is not None:
        retain(pending)
    check()
    peaks = sorted((entry[2] for entry in retained), key=lambda row: row['time'])
    return peaks, {'configuration': config, 'frame_count': frame_number,
                   'peak_count_before_limit': total_peaks, 'retained_peak_count': len(peaks),
                   'sample_rate': SAMPLE_RATE, 'window_seconds': .025, 'hop_seconds': .01,
                   'noise_floor': 'max(user threshold, 3x trailing 1s mean flux)',
                   'time_reference': 'source time, centre of spectral window',
                   'method': 'spectral_flux_onset', 'ai_used': False}


def impact_candidates(source: Path, duration: float, options: dict, out: Path, check, audio_stream=0):
    """Return (candidates, metadata, limitations), writing bounded impacts.json.

    Call only when the source has audio. Uses the same selected audio-track ordinal,
    restricted protocols, subprocess cancellation and cleanup as normal analysis.
    """
    import numpy as np
    from .analysis import SOURCE_FLAGS, _run, _write
    config = _configuration(options)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 43200:
        raise ValueError('Impact analysis supports finite source durations up to 12 hours.')
    if isinstance(audio_stream, bool) or not isinstance(audio_stream, int) or not 0 <= audio_stream <= 31:
        raise ValueError('Impact audio stream must be an integer audio-track index from 0 to 31.')
    check()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def blocks():
        # Each seek is sample-grid aligned. Decode overlap so each chunk contributes
        # exactly its source-time sample range; pad missing/late audio with silence.
        with tempfile.TemporaryDirectory(prefix='vektri-impacts-', dir=out) as directory:
            audio_path = Path(directory) / 'chunk.wav'
            total_samples = round(duration * SAMPLE_RATE)
            chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
            for start in range(0, total_samples, chunk_samples):
                check()
                size = min(chunk_samples, total_samples - start)
                preroll = min(start, SAMPLE_RATE)
                _run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-threads', '1',
                      '-ss', str((start - preroll) / SAMPLE_RATE), *SOURCE_FLAGS, '-i', str(source),
                      '-t', str((size + preroll) / SAMPLE_RATE), '-map', f'0:a:{audio_stream}', '-vn',
                      '-af', 'aresample=16000:async=1:first_pts=0', '-ac', '1', '-ar', '16000',
                      '-c:a', 'pcm_s16le', str(audio_path)], check, timeout=180)
                check()
                with wave.open(str(audio_path), 'rb') as reader:
                    if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != SAMPLE_RATE:
                        raise ValueError('Unexpected impact decoder PCM format.')
                    reader.readframes(preroll)
                    samples = np.frombuffer(reader.readframes(size), dtype='<i2').astype(np.float32) / 32768
                if len(samples) < size:
                    samples = np.pad(samples, (0, size - len(samples)))
                yield samples

    stream = blocks()
    try:
        peaks, metadata = detect_onsets(stream, config, check)
    finally:
        stream.close()
    metadata.update({'audio_stream': audio_stream, 'duration': duration, 'chunk_seconds': CHUNK_SECONDS,
                     'status': 'succeeded' if peaks else 'no_candidates', 'peaks': peaks})
    candidates = []
    for index, peak in enumerate(peaks):
        start = max(0., peak['time'] - config['impact_lead'])
        end = min(duration, peak['time'] + config['impact_follow'])
        if end - start < .04 - 1e-9:
            continue
        candidates.append({'start': start, 'end': end,
                           'title': f'Sound onset near {peak["time"]:.2f}s',
                           'rationale': 'A local rise in positive spectral flux in the selected frequency band. Review the video before treating it as a meaningful moment.',
                           'evidence': [{'kind': 'sound_onset', **peak, 'ref': f'impacts.json#/peaks/{index}', 'audio_stream': audio_stream}],
                           'confidence': .35, 'uncertainty': LIMITATIONS[0],
                           'scores': {'ranking': min(.65, .25 + .1 * peak['flux'] / peak['threshold']), 'spectral_flux': peak['flux']},
                           'method': 'spectral_flux_onset'})
    metadata['status'] = 'succeeded' if candidates else 'no_candidates'
    check()
    _write(out / 'impacts.json', metadata)
    return candidates, metadata, list(LIMITATIONS)
