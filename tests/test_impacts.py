"""Deterministic synthetic coverage; not event-recognition quality claims."""
import json
from pathlib import Path
import shutil
import wave

import numpy as np
import pytest

from clipstudio.impacts import SAMPLE_RATE, detect_onsets, impact_candidates


def signal(seconds=3, hits=((1., 1.), (2., .3))):
    values = np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)
    for time, amplitude in hits:
        values[round(time * SAMPLE_RATE)] = amplitude
    return values


def test_silence_too_short_and_empty_abstain():
    for values in [np.zeros(48000), np.zeros(399), np.ones(100), np.empty(0)]:
        peaks, meta = detect_onsets([values])
        assert peaks == []
        json.dumps(meta, allow_nan=False)


def test_impulses_source_times_gain_and_no_nan():
    for gain in [1., .5, .2]:
        peaks, meta = detect_onsets([signal(hits=((1., 1.), (2., 1.))) * gain])
        assert len(peaks) == 2
        assert [row['time'] for row in peaks] == pytest.approx([1, 2], abs=.015)
        json.dumps([peaks, meta], allow_nan=False)


def test_stationary_noise_abstains_and_hits_survive_background():
    rng = np.random.default_rng(100)
    noise = rng.normal(0, .001, 48000)
    assert detect_onsets([noise])[0] == []
    assert len(detect_onsets([noise + signal()])[0]) == 2
    # Adaptive baseline also abstains on louder stationary noise.
    assert detect_onsets([noise * 50])[0] == []


def test_chunk_edges_preserve_spectrum_hops_and_peaks():
    values = signal(hits=((.99, 1.), (2.01, .5)))
    expected = detect_onsets([values])[0]
    for width in [1, 159, 160, 399, 400, 1001, 16000]:
        blocks = (values[start:start + width] for start in range(0, len(values), width))
        assert detect_onsets(blocks)[0] == expected


def test_minimum_spacing_and_bounded_retention():
    values = signal(seconds=12, hits=tuple((t / 10, 1.) for t in range(10, 115, 2)))
    peaks, meta = detect_onsets([values], {'impact_min_gap': .6, 'impact_max_candidates': 4})
    assert len(peaks) == 4
    assert meta['peak_count_before_limit'] > len(peaks)
    assert all(b['time'] - a['time'] >= .6 - 1e-9 for a, b in zip(peaks, peaks[1:]))


@pytest.mark.parametrize('option,value', [
    ('impact_low_hz', -1), ('impact_high_hz', 8001), ('impact_threshold', 0),
    ('impact_threshold', float('nan')), ('impact_min_gap', float('inf')),
    ('impact_min_gap', -.1), ('impact_lead', -1), ('impact_follow', True),
    ('impact_max_candidates', 501), ('impact_max_candidates', 1.5),
])
def test_reject_invalid_configuration(option, value):
    with pytest.raises(ValueError):
        detect_onsets([], {option: value})


@pytest.mark.parametrize('options', [
    {'impact_low_hz': 6000, 'impact_high_hz': 1000},
    {'impact_low_hz': 1001, 'impact_high_hz': 1002},
    {'impact_lead': 0, 'impact_follow': 0},
])
def test_reject_invalid_combination(options):
    with pytest.raises(ValueError):
        detect_onsets([], options)


def test_band_rejects_out_of_band_tone_onset():
    x = np.zeros(48000)
    t = np.arange(3200) / SAMPLE_RATE
    x[16000:19200] = .5 * np.sin(2 * np.pi * 400 * t) * np.hanning(len(t))
    assert detect_onsets([x], {'impact_low_hz': 2000, 'impact_high_hz': 6000})[0] == []
    assert detect_onsets([x], {'impact_low_hz': 200, 'impact_high_hz': 800, 'impact_threshold': .005})[0]


def test_nonfinite_pcm_rejected():
    with pytest.raises(ValueError):
        detect_onsets([np.array([0., float('nan')])])


def test_cancel_checked_within_large_block():
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == 5:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        detect_onsets([signal(seconds=30)], check=check)
    assert checks == 5


def write_wav(path, values):
    with wave.open(str(path), 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes((values * 32767).astype('<i2').tobytes())


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is not installed')
def test_ffmpeg_chunk_boundary_and_cleanup(tmp_path):
    source = tmp_path / 'source.wav'
    values = signal(32, hits=((29.99, .8), (30.7, .8)))
    write_wav(source, values)
    out = tmp_path / 'analysis'
    candidates, metadata, notes = impact_candidates(source, 32, {}, out, lambda: None)
    assert [p['time'] for p in metadata['peaks']] == pytest.approx([29.99, 30.7], abs=.015)
    assert len(candidates) == 2
    assert all(0 <= c['start'] < c['end'] <= 32 for c in candidates)
    assert all(c['title'].startswith('Sound onset') for c in candidates)
    assert metadata['audio_stream'] == 0
    assert notes
    assert list(out.iterdir()) == [out / 'impacts.json']


def test_decode_cleanup_on_failure_and_selected_track(tmp_path, monkeypatch):
    from clipstudio import analysis
    calls = []
    def run(args, check, timeout):
        calls.append(args)
        Path(args[-1]).write_bytes(b'partial')
        raise RuntimeError('cancelled decoder')
    monkeypatch.setattr(analysis, '_run', run)
    with pytest.raises(RuntimeError, match='cancelled decoder'):
        impact_candidates(tmp_path / 'source.mov', 2, {}, tmp_path / 'out', lambda: None, audio_stream=2)
    assert calls[0][calls[0].index('-map') + 1] == '0:a:2'
    assert '-protocol_whitelist' in calls[0]
    assert list((tmp_path / 'out').iterdir()) == []


def test_exclusions_precede_bounded_retention_and_spacing():
    values = signal(4, hits=((1., .2), (2., 1.), (3., 1.)))
    options = {'impact_max_candidates': 1, 'impact_lead': .1, 'impact_follow': .1,
               'exclusions': [{'start': 1.8, 'end': 3.5}]}
    peaks, _ = detect_onsets([values], options)
    assert len(peaks) == 1
    assert peaks[0]['time'] == pytest.approx(1, abs=.015)
    values = signal(3, hits=((1., 1.), (1.3, .2)))
    options.update({'impact_lead': .02, 'impact_follow': .02,
                    'exclusions': [{'start': .99, 'end': 1.1}]})
    peaks, _ = detect_onsets([values], options)
    assert len(peaks) == 1
    assert peaks[0]['time'] == pytest.approx(1.3, abs=.015)


def test_reject_under_one_video_frame_padding():
    with pytest.raises(ValueError, match='0.04'):
        detect_onsets([], {'impact_lead': .01, 'impact_follow': .01})


def test_skip_clipped_too_short_candidate(tmp_path, monkeypatch):
    import clipstudio.impacts as impacts
    def detector(blocks, options, check):
        return [{'time': .995, 'flux': .2, 'rms_db': -20, 'threshold': .04}], {}
    monkeypatch.setattr(impacts, 'detect_onsets', detector)
    candidates, meta, _ = impact_candidates(tmp_path / 'unused.wav', 1,
        {'impact_lead': 0, 'impact_follow': .04}, tmp_path / 'out', lambda: None)
    assert candidates == []
    assert meta['status'] == 'no_candidates'
