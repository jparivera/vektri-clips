"""Bounded OCR regressions with fake model outputs and real local decode only.

No inference or benchmark labels: these fixtures test geometry, provenance,
resource checks, evidence limits, and preservation of untrusted raw readings.
"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from clipstudio import ocr


def noop(*args, **kwargs):
    pass


@pytest.fixture
def resource_safe(monkeypatch):
    # Controlled unit outputs only; real inference/decoding safety is not disabled.
    monkeypatch.setattr(ocr, '_detail_resource_check', noop)


@pytest.mark.parametrize('size', [(960, 540), (1920, 1080), (3840, 2160), (2160, 3840), (4096, 2160), (2974, 2974), (4096, 128)])
def test_tiles_cover_every_pixel_with_bounded_overlap(size):
    width, height = size
    coverage = np.zeros((height, width), np.uint8)
    windows = ocr.tile_windows(width, height)
    assert 1 <= len(windows) <= 20
    for x, y, w, h in windows:
        assert 4 <= w <= 960 and 4 <= h <= 960
        assert x >= 0 and y >= 0 and x + w <= width and y + h <= height
        coverage[y:y + h, x:x + w] += 1
    assert coverage.min() >= 1


def test_frame_resize_records_source_geometry_before_bounded_processing(resource_safe):
    source = Image.new('RGB', (5000, 3000), (255, 0, 0))
    frame, metadata = ocr._detail_bgr(source)
    assert max(frame.shape[:2]) <= ocr.MAX_DETAIL_EDGE
    assert frame.shape[0] * frame.shape[1] <= ocr.MAX_DETAIL_PIXELS
    assert metadata['source_frame_width'] == 5000
    assert metadata['input_scale_xy'] == [frame.shape[1] / 5000, frame.shape[0] / 3000]
    assert frame[0, 0].tolist() == [0, 0, 255]


def test_invalid_detail_options_and_preallocation_dimensions():
    for value in ('wrong', [], True):
        with pytest.raises(ValueError):
            ocr._detail_options({'ocr_mode': value})
    for value in (0, 4, True, 1.5):
        with pytest.raises(ValueError):
            ocr._detail_options({'ocr_quality_samples': value})
    with pytest.raises(ValueError):
        ocr._bounded_size(10000, 10000)
    with pytest.raises(ValueError):
        ocr.tile_windows(5000, 2000)


def test_model_specific_dictionary_hash_supports_verified_multicodepoint_tokens(monkeypatch):
    model_id = 'ppocr-v5-mobile'
    encoded = 'x\ny\n🇩🇪'
    monkeypatch.setitem(ocr.MODEL_SPECS, model_id, {'detector': 'det.onnx', 'recognizer': 'rec.onnx',
                         'alphabet_count': 3, 'maximum_token_codepoints': 2,
                         'alphabet_sha256': hashlib.sha256(encoded.encode()).hexdigest()})
    monkeypatch.setitem(ocr.CATALOG, model_id, {'files': {'det.onnx': (), 'rec.onnx': ()}})
    alphabet = ocr._alphabet(encoded, model_id)
    assert alphabet == ['', 'x', 'y', '🇩🇪', ' ']
    prediction = np.zeros((1, 1, 5), np.float32)
    prediction[0, 0, 3] = .8
    assert ocr._decode_ctc(prediction, alphabet)[0] == '🇩🇪'
    with pytest.raises(ocr.ModelUnavailable, match='hash'):
        ocr._alphabet('y\nx\n🇩🇪', model_id)
    with pytest.raises(ocr.ModelUnavailable, match='dictionary'):
        ocr._alphabet('x\nx\n🇩🇪', model_id)
    with pytest.raises(ocr.ModelUnavailable, match='catalog'):
        ocr._model_spec('/tmp/arbitrary-model.onnx')


def test_resource_guard_preserves_ram_reserve_and_checks_owned_children(monkeypatch):
    monkeypatch.setattr(ocr.psutil, 'virtual_memory', lambda: SimpleNamespace(available=ocr.MIN_AVAILABLE_RAM - 1))
    with pytest.raises(RuntimeError, match='1.5 GiB'):
        ocr._detail_resource_check(1000)
    monkeypatch.setattr(ocr.psutil, 'virtual_memory', lambda: SimpleNamespace(available=ocr.MIN_AVAILABLE_RAM + ocr.DETAIL_RESERVE_BYTES))
    with pytest.raises(RuntimeError, match='reserve'):
        ocr._detail_resource_check(1000)
    monkeypatch.setattr(ocr.psutil, 'virtual_memory', lambda: SimpleNamespace(available=10 * 1024**3))
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=ocr.MAX_DETAIL_RSS_BYTES))
    parent = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1), children=lambda **kw: [child])
    monkeypatch.setattr(ocr.psutil, 'Process', lambda: parent)
    with pytest.raises(RuntimeError, match='RSS'):
        ocr._detail_resource_check()


def engine(tokens=(1, 0, 2), confidence=.9):
    calls = {'detector': 0, 'recognizer': 0}
    def detect(*args):
        calls['detector'] += 1
        return [np.zeros((1, 1, 32, 32), np.float32)]
    def recognize(*args):
        calls['recognizer'] += 1
        values = np.zeros((1, len(tokens), 3), np.float32)
        for i, token in enumerate(tokens):
            values[0, i, token] = confidence
        return [values]
    session = lambda call: SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='x')], run=call)
    return SimpleNamespace(detector=session(detect), recognizer=session(recognize), characters=['', 'x', 'y']), calls


def test_unrotated_fallback_preserves_narrow_upright_glyph_pixels():
    image = np.zeros((60, 40, 3), np.uint8)
    image[10:30, 14:22] = [255, 0, 0]
    image[30:50, 14:22] = [0, 255, 0]
    box = np.array([[14, 10], [21, 10], [21, 49], [14, 49]], np.float32)
    with pytest.raises(ocr.UnsupportedTextOrientation):
        ocr._crop(image, box)
    crop = ocr._axis_crop(image, box)
    assert crop.shape[:2] == (40, 8)
    assert crop[:20].mean(axis=(0, 1)).tolist() == [255, 0, 0]
    assert crop[20:].mean(axis=(0, 1)).tolist() == [0, 255, 0]


def test_detail_keeps_steep_raw_reading_without_orientation_claim(resource_safe, monkeypatch):
    box = np.array([[14, 10], [21, 10], [21, 49], [14, 49]], np.float32)
    monkeypatch.setattr(ocr, '_boxes', lambda *args: ([(box, .8)], False))
    recognizer, calls = engine()
    result = ocr.recognize_frame(np.zeros((60, 40, 3), np.uint8), recognizer, mode='detail')
    assert result['raw_lines'][0]['text'] == 'xy'
    assert result['raw_lines'][0]['crop_method'] == 'axis_aligned_unrotated_orientation_unknown'
    assert result['orientation_established'] is False
    assert result['axis_aligned_fallback_lines'] == 1 and calls['recognizer'] == 1


def test_tile_coordinates_and_raw_low_scores_are_preserved(resource_safe, monkeypatch):
    box = np.array([[10, 10], [50, 10], [50, 20], [10, 20]], np.float32)
    monkeypatch.setattr(ocr, '_boxes', lambda *args: ([(box, .8)], False))
    recognizer, _ = engine(confidence=.4)
    result = ocr.recognize_frame(np.zeros((1080, 1920, 3), np.uint8), recognizer, mode='detail')
    assert result['detections'] == [] and len(result['raw_lines']) == result['planned_pass_count']
    global_line = result['raw_lines'][0]
    assert global_line['box'][0] == [20., 20.]
    last_line = result['raw_lines'][-1]
    x, y, _, _ = last_line['tile_region_xywh']
    assert last_line['box'][0] == [x + 10., y + 10.]
    assert last_line['text'] == 'xy' and last_line['accepted'] is False
    assert result['passes_complete']


def test_frame_line_cap_stops_extra_native_inference_and_marks_unread_tiles(resource_safe, monkeypatch):
    box = np.array([[10, 10], [50, 10], [50, 20], [10, 20]], np.float32)
    monkeypatch.setattr(ocr, '_boxes', lambda *args: ([(box, .8)] * 50, False))
    recognizer, calls = engine()
    result = ocr.recognize_frame(np.zeros((1080, 1920, 3), np.uint8), recognizer, mode='detail', max_raw_lines=7)
    assert len(result['raw_lines']) == 7 and calls['recognizer'] == 7
    assert calls['detector'] == 1
    assert result['boxes_truncated'] and not result['passes_complete']
    assert result['processed_pass_count'] < result['planned_pass_count']


def test_duplicate_links_never_drop_raw_lines_and_temporal_links_do_not_correct_text():
    box = [[0., 0.], [.2, 0.], [.2, .1], [0., .1]]
    lines = [{'line_id': str(i), 'text': text, 'box_normalized': box, 'accepted': True} for i, text in enumerate(['x y', 'Ｘ  Ｙ', 'single'])]
    groups = ocr._overlap_groups(lines, noop)
    assert len(groups) == 2 and groups[0]['line_ids'] == ['0', '1']
    assert [line['text'] for line in lines] == ['x y', 'Ｘ  Ｙ', 'single']
    samples = [{'time': 0, 'overlap_groups': groups}, {'time': 5, 'overlap_groups': [groups[0]]}]
    agreement = ocr._temporal_agreement(samples, noop)
    assert len(agreement['links']) == 1 and agreement['links'][0]['from_line_ids'] == ['0', '1']
    assert lines[2]['text'] == 'single'  # no one-frame observation is removed


def test_sharpness_proxy_distinguishes_blur_without_inspecting_text():
    pattern = np.indices((200, 200)).sum(axis=0) // 8 % 2 * 255
    sharp = np.repeat(pattern[:, :, None], 3, axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 4)
    assert ocr._quality_score(Image.fromarray(sharp)) > ocr._quality_score(Image.fromarray(blurred))
    assert ocr.quality_times(5., 0., 10., 3) == [5., 4.75, 5.25]
    assert ocr.quality_times(0., 0., 10., 3) == [0., .25]


def fake_job(tmp_path, monkeypatch):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'never change source')
    out = tmp_path / 'out'
    extent = {'start': 0., 'end': 12., 'format_start_time': 0., 'reported_frame_period_seconds': .04}
    monkeypatch.setattr(ocr, '_memory_check', noop)
    monkeypatch.setattr(ocr, '_detail_resource_check', noop)
    monkeypatch.setattr(ocr, '_video_extent', lambda *args: ([], extent))
    monkeypatch.setattr(ocr, '_source_geometry', lambda *args: {'encoded_width': 64, 'encoded_height': 32, 'ffmpeg_autorotation': True})
    monkeypatch.setattr(ocr, 'Recognizer', lambda check: SimpleNamespace(close=noop))
    calls = []
    def decode(source, requested, target, check, deadline):
        check()
        calls.append(requested)
        Image.new('RGB', (64, 32), 'white').save(target)
        return requested
    monkeypatch.setattr(ocr, '_decode_detail_frame', decode)
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    def recognize(*args, **kwargs):
        line = {'text': 'raw untrusted x', 'line_id': 'tile-000/line-000', 'pass_id': 'tile-000',
                'box_normalized': box, 'accepted': True}
        return {'detections': [line], 'raw_lines': [line], 'overlap_groups': [], 'frame_width': 64, 'frame_height': 32}
    monkeypatch.setattr(ocr, 'recognize_frame', recognize)
    return source, out, calls


def test_quality_selection_uses_actual_frame_pts_and_reports_every_candidate(tmp_path, monkeypatch):
    source, out, calls = fake_job(tmp_path, monkeypatch)
    scores = iter([1., 4., 1., 3., 9., 1., 3., 9., 1., 3.])
    monkeypatch.setattr(ocr, '_quality_score', lambda image: next(scores))
    result = ocr.extract_text(source, 12, {'ocr_mode': 'detail', 'ocr_quality_samples': 3}, out, noop)
    assert result['configuration']['ocr_quality_samples'] == 3
    assert result['samples'][0]['time'] == .25
    assert result['samples'][0]['raw_lines'][0]['time'] == .25
    assert len(result['samples'][0]['quality_selection']['candidates']) == 2
    assert result['samples'][0]['frame_ref'].endswith('quality-1.jpg')
    assert len(calls) > result['sample_count']
    assert result['sampling_complete'] and source.read_bytes() == b'never change source'


@pytest.mark.parametrize('budget', ['raw_lines', 'raw_text', 'serialized'])
def test_job_output_budgets_report_partial_without_oversized_evidence(tmp_path, monkeypatch, budget):
    source, out, calls = fake_job(tmp_path, monkeypatch)
    if budget == 'raw_lines':
        monkeypatch.setattr(ocr, 'MAX_JOB_RAW_LINES', 1)
    elif budget == 'raw_text':
        monkeypatch.setattr(ocr, 'MAX_JOB_RAW_TEXT_BYTES', 1)
    else:
        monkeypatch.setattr(ocr, 'SAMPLE_EVIDENCE_BUDGET', 1)
    result = ocr.extract_text(source, 12, {'ocr_mode': 'detail'}, out, noop)
    assert result['raw_readings_truncated'] and not result['sampling_complete']
    assert result['budget_truncation'] and result['sample_count'] <= 1
    assert len(calls) == 1
    assert (out / 'ocr.json').stat().st_size <= ocr.MAX_EVIDENCE_BYTES
    assert json.loads((out / 'ocr.json').read_text()) == result


def test_output_symlink_and_existing_operator_files_are_rejected_before_loading(tmp_path, monkeypatch):
    source, out, _ = fake_job(tmp_path, monkeypatch)
    monkeypatch.setattr(ocr, 'Recognizer', lambda *args: pytest.fail('unsafe output must be rejected first'))
    out.mkdir()
    (out / 'ocr.json').symlink_to(tmp_path / 'not-yet-created')
    with pytest.raises(ValueError, match='already exists'):
        ocr.extract_text(source, 12, {}, out, noop)
    (out / 'ocr.json').unlink()
    alias = tmp_path / 'alias'
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        ocr.extract_text(source, 12, {}, alias, noop)


def test_detail_decoder_preserves_small_text_resolution_and_timestamp_without_ocr(tmp_path):
    source = tmp_path / 'geometry.mp4'
    # A plain color geometry fixture contains no OCR text and runs no model.
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=white:s=1920x1080:r=2:d=1',
                    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(source)], check=True, timeout=20)
    target = tmp_path / 'frame.jpg'
    actual = ocr._decode_detail_frame(source, .1, target, noop, time.monotonic() + 30)
    with Image.open(target) as frame:
        assert frame.size == (1920, 1080)
    assert actual == .5
    with pytest.raises(ValueError, match='replacement'):
        ocr._decode_detail_frame(source, .1, target, noop, time.monotonic() + 30)


@pytest.mark.parametrize('stop', ['cancel', 'timeout'])
def test_decoder_cancellation_and_timeout_reap_only_own_child(tmp_path, monkeypatch, stop):
    real_popen = subprocess.Popen
    processes = []
    def own_child(*args, **kwargs):
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(5)'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(child)
        return child
    monkeypatch.setattr(ocr.subprocess, 'Popen', own_child)
    if stop == 'timeout':
        monkeypatch.setattr(ocr, 'FRAME_TIMEOUT_SECONDS', .01)
    calls = []
    def check():
        calls.append(1)
        if stop == 'cancel' and len(calls) > 1:
            raise RuntimeError('cancelled by test')
    with pytest.raises(RuntimeError, match='cancelled|time budget'):
        ocr._decode_detail_frame(tmp_path / 'unused.mp4', 0, tmp_path / 'out.jpg', check, time.monotonic() + 5)
    assert len(processes) == 1 and processes[0].poll() is not None
