"""CLIP controls/scoring tests plus explicitly real, optional local-model inference."""
from pathlib import Path
import json
import math
import subprocess

import numpy as np
from PIL import Image
import pytest

from clipstudio import models, vision

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'


def noop():
    pass


def installed_model():
    try:
        return models.model_path('clip-vit-b32', verify=True)
    except models.ModelUnavailable:
        pytest.skip('Pinned CLIP weights absent; no download or mocked model substitutes for real inference')


def test_preprocessor_rgb_center_crop_and_pinned_normalization():
    # No resize is needed: blue margins must be absent from the central crop.
    image = Image.new('RGBA', (448, 224), (0, 0, 255, 255))
    image.paste((255, 0, 0, 255), (112, 0, 336, 224))
    result = vision.preprocess_image(image)
    assert result.shape == (1, 3, 224, 224) and result.dtype == np.float32
    expected = (np.array([1, 0, 0], dtype=np.float32) - vision.MEAN) / vision.STD
    assert np.allclose(result, expected[None, :, None, None], atol=1e-6)
    assert result.flags.c_contiguous and np.isfinite(result).all()


def test_preprocessor_resizes_without_changing_solid_color_and_bounds_extreme_geometry():
    result = vision.preprocess_image(Image.new('RGB', (320, 480), (64, 128, 192)))
    expected = (np.array([64, 128, 192], dtype=np.float32) / 255 - vision.MEAN) / vision.STD
    assert np.allclose(result, expected[None, :, None, None], atol=1e-6)
    with pytest.raises(ValueError, match='aspect ratio'):
        vision.preprocess_image(Image.new('RGB', (1, 1000)))


def test_pinned_tokenizer_special_tokens_padding_and_truncation():
    location = installed_model()
    ids, info = vision.tokenize_query(location, 'a photo of a cat')
    assert ids[0, :7].tolist() == [49406, 320, 1125, 539, 320, 2368, 49407]
    assert np.all(ids[0, 6:] == 49407)
    assert ids.dtype == np.int64 and ids.shape == (1, 77)
    assert info['tokens_before_truncation'] == 7 and info['query_truncated'] is False
    long_ids, info = vision.tokenize_query(location, 'an image of a vehicle ' * 100)
    assert info['query_truncated'] is True
    assert long_ids[0, 0] == 49406 and long_ids[0, -1] == 49407


@pytest.mark.parametrize('duration', [0.1, 1, 24, 1800, 43200])
def test_sampling_covers_full_source_with_bounded_cap_and_tail(duration):
    times = vision.sample_times(duration)
    assert 1 <= len(times) <= 900
    assert times[0] == 0
    assert times[-1] <= duration and duration - times[-1] <= 0.25
    assert all(0 <= first < second < duration for first, second in zip(times, times[1:]))
    if duration == 43200:
        assert len(times) == 900
        assert len({int(at / 3600) for at in times}) == 12


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), 43200.1])
def test_bad_source_durations_rejected(duration):
    with pytest.raises(ValueError):
        vision.sample_times(duration)


@pytest.mark.parametrize('value', [np.zeros(512), np.full(512, np.nan), np.ones(768), np.full(512, np.inf)])
def test_invalid_embeddings_never_produce_fake_scores(value):
    with pytest.raises(ValueError):
        vision._unit_vector(value)


def samples(times, scores):
    return [{'time': at, 'cosine_similarity': score, 'frame_ref': f'visual-frames/frame-{index:04d}.jpg'}
            for index, (at, score) in enumerate(zip(times, scores))]


def test_negative_cosines_keep_monotonic_ranking_and_context_means():
    values = samples([0, 5, 10], [-0.8, -0.3, -0.6])
    result = vision._window_candidates(values, 10, 2, 4, 'requested scene', [], noop)
    assert result[0]['start'] == 4 and result[0]['end'] == 6
    assert result[0]['scores']['ranking'] == pytest.approx(-0.34, abs=1e-6)
    assert all(-1 <= row['scores']['ranking'] < 0 for row in result)
    assert all('confidence' not in row for row in result)
    assert all(row['method'] == 'local_clip_visual' for row in result)
    assert all('motion' not in row['scores'] for row in result)
    assert result[0]['evidence'][1]['cosine_similarity'] == -0.3


def test_exclusions_filter_full_context_window_not_only_sample_center():
    values = samples([0, 5, 10], [-0.8, 0.9, -0.6])
    result = vision._window_candidates(values, 10, 2, 4, 'scene', [{'start': 3.9, 'end': 4.1}], noop)
    assert result
    assert all(row['end'] <= 3.9 or row['start'] >= 4.1 for row in result)
    assert not any(row['start'] == 4 and row['end'] == 6 for row in result)


def test_context_candidates_reach_last_sample_and_have_finite_bound():
    times = vision.sample_times(43200)
    values = samples(times, [-0.8] * (len(times) - 1) + [0.9])
    result = vision._window_candidates(values, 43200, 4, 10, 'scene', [], noop)
    assert result[0]['end'] == 43200
    assert len(result) <= 3 * vision.MAX_SAMPLES
    assert result[0]['evidence'][-1]['time'] == times[-1]


@pytest.mark.parametrize('values', [samples([0, 1], [0, float('nan')]), samples([0, 0], [0, 0.5]), samples([0, 1], [0, 2])])
def test_invalid_sample_sequences_rejected(values):
    with pytest.raises(ValueError):
        vision._window_candidates(values, 2, 0.5, 1, 'scene', [], noop)


def test_scorer_checks_cancellation_after_final_window():
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('cancel after final window')
    with pytest.raises(RuntimeError, match='cancel after final window'):
        vision._window_candidates(samples([0, 2], [0.1, 0.2]), 2, 0.5, 1, 'scene', [], cancel)


class PendingDecoder:
    def __init__(self, *args, **kwargs):
        self.killed = False
        self.waited = False
        self.returncode = None
    def poll(self):
        return None
    def kill(self):
        self.killed = True
    def wait(self, timeout):
        self.waited = True
        return 0


def test_decode_cancellation_terminates_child_and_removes_stale_frame(tmp_path, monkeypatch):
    process = PendingDecoder()
    monkeypatch.setattr(vision.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(vision.time, 'sleep', lambda _: None)
    target = tmp_path / 'frame.jpg'
    target.write_bytes(b'stale previous output')
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('cancel decode')
    with pytest.raises(RuntimeError, match='cancel decode'):
        vision._decode_frame(Path('source.mp4'), 0, target, cancel, vision.time.monotonic() + 30)
    assert process.killed and process.waited
    assert not target.exists()


def test_decode_timeout_terminates_child(tmp_path, monkeypatch):
    process = PendingDecoder()
    monkeypatch.setattr(vision.subprocess, 'Popen', lambda *args, **kwargs: process)
    clock = iter(range(100))
    monkeypatch.setattr(vision.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(vision, 'FRAME_TIMEOUT_SECONDS', 0.5)
    with pytest.raises(RuntimeError, match='time budget'):
        vision._decode_frame(Path('source.mp4'), 0, tmp_path / 'frame.jpg', noop, 100)
    assert process.killed and process.waited


def test_missing_model_fails_explicitly_without_decode(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise models.ModelUnavailable('Pinned CLIP model missing')
    monkeypatch.setattr(vision, 'model_path', missing)
    monkeypatch.setattr(vision, '_decode_frame', lambda *args: pytest.fail('Missing model must not start decoder'))
    with pytest.raises(models.ModelUnavailable, match='missing'):
        vision.visual_candidates(FIXTURES / 'action_motion.mp4', 12, {'query': 'a vehicle'}, tmp_path, noop)


def test_real_clip_inference_and_frame_artifact_pipeline(tmp_path):
    installed_model()
    import psutil
    from clipstudio import runtime
    if psutil.virtual_memory().available < runtime.MIN_AVAILABLE_RAM:
        pytest.skip('Real CLIP pipeline requires the unchanged 1.5 GiB available-RAM guard')
    candidates, metadata, limitations = vision.visual_candidates(
        FIXTURES / 'spoken_interview.mp4', 24, {'query': 'an illustration of a person on a dark screen', 'min_length': 3, 'max_length': 6}, tmp_path, noop)
    assert candidates and metadata['sampling_complete'] is True
    assert metadata['sample_count'] == len(vision.sample_times(24))
    assert metadata['last_sample_seconds'] > 23
    assert metadata['model'] == 'Xenova/clip-vit-base-patch32'
    assert metadata['paid_usage'] == 0 and metadata['data_transmitted'] is False
    assert metadata['tower_loading'] == 'text released before vision'
    assert all(math.isfinite(item['cosine_similarity']) and -1 <= item['cosine_similarity'] <= 1 for item in metadata['samples'])
    assert all((tmp_path / item['frame_ref']).is_file() for item in metadata['samples'])
    assert all(row['method'] == 'local_clip_visual' for row in candidates)
    assert json.loads((tmp_path / 'visual-search.json').read_text()) == metadata
    assert limitations


def test_real_sparse_video_extent_and_decoded_pts_exclude_audio_only_tail(tmp_path):
    installed_model()
    import psutil
    from clipstudio import runtime
    if psutil.virtual_memory().available < runtime.MIN_AVAILABLE_RAM:
        pytest.skip('Real CLIP pipeline requires the unchanged 1.5 GiB available-RAM guard')
    source = tmp_path / 'video-shorter-than-audio.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y',
                    '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=1:duration=4',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6', '-c:v', 'libx264',
                    '-threads', '1', '-c:a', 'aac', '-t', '6', str(source)], check=True)
    candidates, metadata, _ = vision.visual_candidates(
        source, 6, {'query': 'a colorful test pattern', 'min_length': 1, 'max_length': 2}, tmp_path / 'vision', noop)
    assert metadata['video_extent']['end'] == pytest.approx(4)
    assert metadata['sample_count'] == 3
    assert [item['time'] for item in metadata['samples']] == [0, 2, 3]
    assert all(row['end'] <= 4 for row in candidates)
    assert metadata['sampling_complete'] is True
    assert 'decoded first-frame PTS' in metadata['sample_time_basis']


def test_real_very_low_fps_sampling_has_no_duplicate_frame_evidence(tmp_path):
    installed_model()
    import psutil
    from clipstudio import runtime
    if psutil.virtual_memory().available < runtime.MIN_AVAILABLE_RAM:
        pytest.skip('Real CLIP pipeline requires the unchanged 1.5 GiB available-RAM guard')
    source = tmp_path / 'very-low-fps.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y', '-f', 'lavfi',
                    '-i', 'testsrc2=size=160x90:rate=1/5:duration=10', '-c:v', 'libx264',
                    '-threads', '1', '-pix_fmt', 'yuv420p', str(source)], check=True)
    candidates, metadata, _ = vision.visual_candidates(
        source, 10, {'query': 'a colorful test pattern', 'min_length': 1, 'max_length': 2}, tmp_path / 'vision', noop)
    assert [item['time'] for item in metadata['samples']] == [0, 5]
    assert metadata['duplicate_frame_samples_skipped'] == 4
    assert metadata['sampling_complete'] is True
    assert candidates


@pytest.mark.parametrize('extension,video_codec,audio_codec,offset', [
    ('webm', 'libvpx', 'libopus', 0),
    ('mkv', 'libx264', 'aac', 0),
    ('mkv', 'libx264', 'aac', 2),
])
def test_real_tagged_video_extent_uses_end_timestamp_not_container_duration_or_length(tmp_path, extension, video_codec, audio_codec, offset):
    # Actual demux/seek/decode regression; no model is needed to verify coverage.
    import time
    source = tmp_path / ('tagged-tail.' + extension)
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-nostdin', '-y',
                    '-itsoffset', str(offset), '-f', 'lavfi', '-i', 'testsrc2=size=160x90:rate=1:duration=4',
                    '-f', 'lavfi', '-i', 'anullsrc=r=8000:cl=mono', '-t', str(offset + 6),
                    '-c:v', video_codec, '-threads', '1', '-c:a', audio_codec, str(source)], check=True)
    data = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(source)],
                                    capture_output=True, check=True).stdout)
    duration = float(data['format']['duration'])
    times, extent = vision._video_extent(source, duration, noop, time.monotonic() + 20)
    origin = extent['format_start_time']
    assert extent['duration_basis'] == 'stream_duration_tag_end_timestamp'
    assert extent['start'] == pytest.approx(offset - origin)
    assert extent['end'] == pytest.approx(offset + 4 - origin)
    assert extent['end'] < duration - 1
    decoded = [vision._decode_frame(source, at, tmp_path / f'frame-{index}.jpg', noop, time.monotonic() + 20)
               for index, at in enumerate(times)]
    assert decoded == [offset, offset + 2, offset + 3]
    assert all(timestamp - origin < extent['end'] for timestamp in decoded)


@pytest.mark.parametrize('value', ['00:60:00', '00:00:60', 'nan', '1e4', '-01:00:00', '00:00:04junk', None, 4])
def test_untrusted_duration_tags_require_valid_timestamp(value):
    assert vision._duration_tag_seconds(value) is None
