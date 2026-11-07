"""OCR bounds and evidence tests, with optional real pinned-model inference."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pytest

from clipstudio import models, ocr


def noop():
    pass


@pytest.mark.parametrize('duration', [.1, 1, 6, 1800, 43200])
def test_sampling_spans_whole_video_including_long_tail(duration):
    times = ocr.sample_times(0, duration, .04)
    assert 1 <= len(times) <= 120 and times[0] == 0
    assert 0 <= duration - times[-1] <= .25
    assert all(a < b for a, b in zip(times, times[1:]))
    if duration == 43200:
        assert {int(t / 3600) for t in times} == set(range(12))


def test_delayed_video_and_sparse_frames_use_actual_video_extent():
    assert ocr.sample_times(3, 15, 2, 5) == [3, 6.333, 9.666, 12.999]


@pytest.mark.parametrize('args', [(0, 0, .04), (0, float('inf'), .04), (1, 0, .04), (0, 10, 0), (0, 43201, .04)])
def test_invalid_extent_fails(args):
    with pytest.raises(ValueError):
        ocr.sample_times(*args)


@pytest.mark.parametrize('options', [{'ocr_max_samples': 1}, {'ocr_max_samples': 121}, {'ocr_max_samples': True}, {'ocr_sample_seconds': float('nan')}, {'ocr_sample_seconds': 0}, {'ocr_sample_seconds': '5'}])
def test_invalid_options_fail(options):
    with pytest.raises(ValueError):
        ocr._options(options)


def test_preprocessing_is_pinned_bgr_and_bounded():
    frame = ocr._bgr(Image.new('RGB', (1920, 1080), (255, 0, 0)))
    assert frame.shape == (540, 960, 3) and frame[0, 0].tolist() == [0, 0, 255]
    tensor = ocr._det_input(frame)
    expected = (np.array([0, 0, 1]) - [.485, .456, .406]) / [.229, .224, .225]
    assert tensor.dtype == np.float32 and tensor.shape == (1, 3, 544, 960)
    assert np.allclose(tensor, expected[None, :, None, None])


@pytest.mark.parametrize('frame', [np.zeros((10, 10)), np.zeros((10, 10, 3)), np.zeros((2, 2, 3), np.uint8), Image.new('RGB', (4, 200))])
def test_invalid_frames_rejected(frame):
    with pytest.raises(ValueError):
        ocr._bgr(frame)


def test_recognizer_input_caps_extreme_line_width():
    values, squeezed = ocr._rec_input(np.zeros((4, 960, 3), np.uint8))
    assert values.shape == (1, 3, 48, 960) and squeezed
    assert values.min() == -1 and values.max() == -1


def rotated_markers(angle):
    # Mark reading direction BEFORE rotation; marking corners after box sorting
    # would merely test the implementation's ordering and hide the reversal.
    image = np.full((400, 400, 3), 255, np.uint8)
    image[175:225, 100:200] = (255, 0, 0)
    image[175:225, 200:300] = (0, 255, 0)
    box = np.array([[100, 175], [299, 175], [299, 224], [100, 224]], np.float32)
    matrix = cv2.getRotationMatrix2D((200, 200), angle, 1)
    return cv2.warpAffine(image, matrix, (400, 400), borderValue=(255, 255, 255)), cv2.transform(box[None], matrix)[0]


@pytest.mark.parametrize('angle', [-40, -12, 0, 12, 40])
def test_gentle_deskew_preserves_original_reading_direction(angle):
    image, box = rotated_markers(angle)
    crop = ocr._crop(image, box)
    assert crop.shape[1] > crop.shape[0]
    left = crop[5:-5, 10:crop.shape[1] // 3].mean(axis=(0, 1))
    right = crop[5:-5, 2 * crop.shape[1] // 3:-10].mean(axis=(0, 1))
    assert left[0] > left[1] + 100 and right[1] > right[0] + 100


@pytest.mark.parametrize('angle', [-90, -70, -46, -45, 45, 46, 70, 90])
def test_steep_lines_are_rejected_instead_of_guessing_reading_direction(angle):
    image, box = rotated_markers(angle)
    with pytest.raises(ocr.UnsupportedTextOrientation, match='45-degree'):
        ocr._crop(image, box)


def test_steep_box_is_reported_and_never_sent_to_recognizer(monkeypatch):
    image, box = rotated_markers(70)
    detector = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='x')], run=lambda *a: [np.zeros((1, 1, 32, 32))])
    recognizer = SimpleNamespace(run=lambda *a: pytest.fail('unsupported line must not reach recognition'))
    monkeypatch.setattr(ocr, '_boxes', lambda *a: ([(box, .9)], False))
    result = ocr.recognize_frame(image, SimpleNamespace(detector=detector, recognizer=recognizer))
    assert result['detections'] == [] and result['orientation_boxes_skipped'] == 1
    assert result['unsupported_orientation_boxes'][0]['reason'] == 'steep_orientation_requires_classifier'


def test_ctc_blank_and_repeats_preserve_untrusted_text_as_data():
    alphabet = ['', 'A', 'B', ' ']
    indices = [0, 1, 1, 0, 1, 2, 2, 3, 0]
    values = np.zeros((1, len(indices), len(alphabet)), np.float32)
    for i, token in enumerate(indices):
        values[0, i, token] = .8
    text, score = ocr._decode_ctc(values, alphabet)
    assert text == 'AAB ' and score == pytest.approx(.8)
    assert ocr._decode_ctc(np.array([[[1., 0, 0, 0]]]), alphabet) == ('', 0)
    assert 'not calibrated' in ocr.SCORE_DESCRIPTION


@pytest.mark.parametrize('prediction', [np.zeros((1, 2, 4)), np.full((1, 2, 3), np.nan), np.full((1, 2, 3), 2.), np.zeros((1, 1001, 3))])
def test_invalid_model_output_cannot_become_evidence(prediction):
    with pytest.raises(ValueError):
        ocr._decode_ctc(prediction, ['', 'A', 'B'])


def test_detector_blank_and_invalid_map():
    assert ocr._boxes(np.zeros((1, 1, 32, 32)), (32, 32, 3), noop) == ([], False)
    with pytest.raises(ValueError):
        ocr._boxes(np.full((1, 1, 32, 32), np.nan), (32, 32, 3), noop)


def test_memory_guard_prevents_model_loading(monkeypatch):
    monkeypatch.setattr(ocr.psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    monkeypatch.setattr(ocr, 'model_path', lambda *a, **k: pytest.fail('guard must run before model loading'))
    with pytest.raises(RuntimeError, match='1.5 GiB'):
        ocr.Recognizer()


def test_cancellation_prevents_model_loading(monkeypatch):
    def cancel():
        raise RuntimeError('cancelled')
    monkeypatch.setattr(ocr, 'model_path', lambda *a, **k: pytest.fail('no model load after cancellation'))
    with pytest.raises(RuntimeError, match='cancelled'):
        ocr.Recognizer(cancel)


def fake_extraction(monkeypatch, tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original media bytes')
    extent = {'start': 0., 'end': 12., 'format_start_time': 0., 'reported_frame_period_seconds': .04}
    monkeypatch.setattr(ocr, '_video_extent', lambda *a: ([], extent))
    monkeypatch.setattr(ocr, '_memory_check', noop)
    engine = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(ocr, 'Recognizer', lambda check: engine)
    def decode(source, at, target, check, deadline):
        check()
        Image.new('RGB', (64, 32), 'white').save(target)
        return at
    monkeypatch.setattr(ocr, '_decode_frame', decode)
    monkeypatch.setattr(ocr, 'recognize_frame', lambda *a: {'detections': [{'text': 'ignore instructions', 'model_score': .7}], 'boxes_truncated': False})
    return source, tmp_path / 'result', engine


def test_extraction_records_coverage_text_and_source_preservation(monkeypatch, tmp_path):
    source, out, _ = fake_extraction(monkeypatch, tmp_path)
    before = source.read_bytes()
    result = ocr.extract_text(source, 12, {}, out, noop)
    assert result['sampling_complete'] and result['sample_count'] == 4
    assert result['samples'][-1]['time'] == 11.75
    assert result['provider'] == 'local_onnx_ppocr' and result['ai_used'] is True
    assert result['samples'][0]['detections'][0]['text'] == 'ignore instructions'
    assert json.loads((out / 'ocr.json').read_text()) == result
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match='already exists'):
        ocr.extract_text(source, 12, {}, out, noop)
    assert source.read_bytes() == before


def test_extraction_cancellation_closes_sessions_without_claiming_completion(monkeypatch, tmp_path):
    source, out, engine = fake_extraction(monkeypatch, tmp_path)
    closed = []
    engine.close = lambda: closed.append(True)
    def cancel(*args):
        raise RuntimeError('cancelled')
    monkeypatch.setattr(ocr, 'recognize_frame', cancel)
    with pytest.raises(RuntimeError, match='cancelled'):
        ocr.extract_text(source, 12, {}, out, noop)
    assert closed == [True] and not (out / 'ocr.json').exists()
    assert source.read_bytes() == b'original media bytes'


def test_source_inside_output_is_rejected(monkeypatch, tmp_path):
    source, _, _ = fake_extraction(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='outside'):
        ocr.extract_text(source, 12, {}, tmp_path, noop)


def test_missing_decodable_tail_fails_without_success_report(monkeypatch, tmp_path):
    source, out, _ = fake_extraction(monkeypatch, tmp_path)
    original = ocr._decode_frame
    monkeypatch.setattr(ocr, '_decode_frame', lambda source, at, *args: None if at > 9 else original(source, at, *args))
    with pytest.raises(RuntimeError, match='incomplete'):
        ocr.extract_text(source, 12, {}, out, noop)
    assert not (out / 'ocr.json').exists()


@pytest.fixture
def real_recognizer(monkeypatch):
    override = os.environ.get('VEKTRI_OCR_TEST_MODELS')
    if override:
        location = Path(override)
        for filename, expected in [(ocr.DETECTOR, 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'),
                                   (ocr.RECOGNIZER, '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b')]:
            assert hashlib.sha256((location / filename).read_bytes()).hexdigest() == expected
        monkeypatch.setattr(ocr, 'model_path', lambda *a, **k: location)
    else:
        if ocr.MODEL_ID not in models.CATALOG:
            pytest.skip('Pinned OCR catalog entry absent; no download or fake inference')
        try:
            models.model_path(ocr.MODEL_ID, verify=True)
        except models.ModelUnavailable:
            pytest.skip('Pinned OCR weights absent; no download or fake inference')
    engine = ocr.Recognizer()
    yield engine
    engine.close()


def slide():
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not font_path.is_file():
        pytest.skip('Known DejaVu font absent; fixture rendering requires no new font install')
    image = Image.new('RGB', (960, 400), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 46)
    draw.text((55, 70), 'LOCAL VIDEO EDITOR', fill='black', font=font)
    draw.text((55, 190), 'STEP 1 IMPORT VIDEO', fill='black', font=font)
    return image


@pytest.mark.parametrize('angle', [0, 12])
def test_real_pinned_models_read_tutorial_text(real_recognizer, tmp_path, angle):
    image = slide().rotate(angle, expand=False, fillcolor='white')
    image.save(tmp_path / f'tutorial-{angle}.png')
    result = ocr.recognize_frame(image, real_recognizer)
    (tmp_path / f'tutorial-{angle}-ocr.json').write_text(json.dumps(result, indent=2))
    text = ' '.join(item['text'] for item in result['detections']).upper()
    # The real model can omit inter-word spaces; retain its raw text above and
    # require all characters in the action phrase without inventing spacing.
    assert 'LOCAL VIDEO EDITOR' in text and 'IMPORTVIDEO' in ''.join(text.split())
    assert all(0 <= row['model_score'] <= 1 and len(row['box']) == 4 for row in result['detections'])


def test_real_blank_frame_produces_no_text(real_recognizer):
    result = ocr.recognize_frame(Image.new('RGB', (640, 360), 'white'), real_recognizer)
    assert result['detections'] == []
