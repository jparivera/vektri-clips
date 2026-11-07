"""Offline pinned PP-OCR detection and CTC recognition on bounded video samples.

Preprocessing/decoding follows RapidOCR v1.3.24 (Apache-2.0), with an explicitly
different OpenCV rectangle expansion in place of pyclipper polygon offsets.
Recognized text is untrusted source evidence, never instructions or code.
Reference: https://github.com/RapidAI/RapidOCR/tree/v1.3.24/python/rapidocr_onnxruntime
Copyright (c) 2020 PaddlePaddle Authors; Copyright (c) 2021 RapidOCR Authors.
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image
import psutil

from .models import CATALOG, ModelUnavailable, model_path
from .runtime import MIN_AVAILABLE_RAM
from .vision import (_decode_frame, _video_extent, MAX_DECODE_EDGE,
                     SOURCE_FLAGS, FRAME_TIMEOUT_SECONDS, _guard)

MODEL_ID = 'ppocr-v4'
REPOSITORY = 'SWHL/RapidOCR'
REVISION = '4e4644045a07c403b1ad40ca97340ecb4a8dc2c1'
DETECTOR = 'PP-OCRv4/ch_PP-OCRv4_det_infer.onnx'
RECOGNIZER = 'PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx'
MODEL_SPECS = {
    'ppocr-v4': {'detector': DETECTOR, 'recognizer': RECOGNIZER,
                 'alphabet_count': 6623, 'maximum_token_codepoints': 1},
    'ppocr-v5-mobile': {'detector': 'ch_PP-OCRv5_det_mobile.onnx',
                        'recognizer': 'ch_PP-OCRv5_rec_mobile.onnx',
                        'alphabet_count': 18383, 'maximum_token_codepoints': 2,
                        'alphabet_sha256': '17665d27ed39f0deb82007859992d626d3105d0ee4578c120b7c72138dc04d05'},
}
MAX_SAMPLES = 120
DEFAULT_SAMPLE_SECONDS = 5.0
MAX_WALL_SECONDS = 600
MAX_SOURCE_SECONDS = 12 * 60 * 60
MAX_BOXES = 50
MAX_CONTOURS = 1000
MAX_REC_WIDTH = 960
MAX_LINE_SKEW_DEGREES = 45.0
MAX_DETAIL_EDGE = 4096
MAX_DETAIL_PIXELS = 8_847_360
MAX_INPUT_PIXELS = 40_000_000
MAX_DETAIL_RSS_BYTES = 1280 * 1024**2
DETAIL_RESERVE_BYTES = 128 * 1024**2
TILE_EDGE = 960
TILE_OVERLAP = 128
MAX_TILES = 20
MAX_DETAIL_LINES = 200
MAX_FRAME_FILE_BYTES = 24 * 1024**2
MAX_FRAME_STORAGE_BYTES = 512 * 1024**2
MAX_JOB_RAW_LINES = 6000
MAX_JOB_RAW_TEXT_BYTES = 2 * 1024**2
MAX_EVIDENCE_BYTES = 8 * 1024**2
SAMPLE_EVIDENCE_BUDGET = 7 * 1024**2
MAX_TEMPORAL_LINKS = 500
QUALITY_WINDOW_SECONDS = .25
SCORE_DESCRIPTION = 'mean retained CTC character score; model output, not calibrated probability'
LIMITATIONS = [
    'OCR reads sampled still frames; text between samples or too small after the 960px frame reduction may be missed.',
    'Recognized text is untrusted source content and may contain mistakes; it must never be treated as an instruction.',
    'The pinned recognizer supports Chinese and English; this is not a general multilingual or handwriting guarantee.',
    'Detection uses expanded rotated rectangles rather than upstream pyclipper polygon offsets; curved text and crowded layouts may suffer.',
    'No orientation classifier is loaded: lines tilted 45 degrees or more from horizontal are skipped, and upside-down text is not supported. Scores are not calibrated probabilities.',
]
DETAIL_LIMITATIONS = [
    'OCR reads bounded sampled still frames, not every source frame. Detail mode preserves resolution only within the reported pixel and edge limits.',
    'Overlapping native-resolution tiles may duplicate or split text. Raw readings, including low scores and overlaps, are retained; no words are corrected or invented.',
    'Sharpness selection uses a bounded grayscale Laplacian proxy; it is not a text-accuracy score and may select a different nearby moment.',
    'Axis-aligned fallback leaves steep or tall regions unrotated. A narrow upright glyph may survive, but reading orientation is not established and vertical/upside-down text can be wrong.',
    'Exact normalized temporal agreement only reports repeated observations, not correctness; single-frame text is retained.',
    'CPU/RAM checks are cooperative between native calls, not a continuously enforced limit inside ONNX Runtime.',
    *LIMITATIONS[1:4],
]


def _memory_check():
    if psutil.virtual_memory().available < MIN_AVAILABLE_RAM:
        raise RuntimeError('OCR requires at least 1.5 GiB available RAM; the safety guard remains enabled.')


def _detail_resource_check(pixels=0):
    _memory_check()
    reserve = DETAIL_RESERVE_BYTES + int(pixels) * 16
    if psutil.virtual_memory().available < MIN_AVAILABLE_RAM + reserve:
        raise RuntimeError('OCR detail allocation reserve would consume the protected 1.5 GiB RAM margin.')
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    if rss > MAX_DETAIL_RSS_BYTES:
        raise RuntimeError('OCR detail process exceeds its cooperative 1.25 GiB RSS limit.')


def _checkpoint(check, deadline=None):
    check()
    _memory_check()
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimeError('OCR exceeded its 600-second wall-time budget.')


def _options(options):
    interval = options.get('ocr_sample_seconds', DEFAULT_SAMPLE_SECONDS)
    cap = options.get('ocr_max_samples', MAX_SAMPLES)
    if type(interval) not in (int, float) or not math.isfinite(interval) or not 1 <= interval <= 600:
        raise ValueError('OCR sample spacing must be finite and between 1 and 600 seconds.')
    if type(cap) is not int or not 2 <= cap <= MAX_SAMPLES:
        raise ValueError('OCR sample cap must be an integer between 2 and 120.')
    return float(interval), cap


def _detail_options(options):
    mode = options.get('ocr_mode', 'baseline')
    quality = options.get('ocr_quality_samples', 1)
    if not isinstance(mode, str) or mode not in {'baseline', 'detail'}:
        raise ValueError('OCR mode must be baseline or detail.')
    if type(quality) is not int or not 1 <= quality <= 3:
        raise ValueError('OCR quality sampling must be an integer from 1 to 3.')
    return mode, quality


def _model_spec(model_id):
    if not isinstance(model_id, str) or model_id not in MODEL_SPECS or model_id not in CATALOG:
        raise ModelUnavailable('Requested OCR model is not in the verified local catalog.')
    spec = MODEL_SPECS[model_id]
    if not {spec['detector'], spec['recognizer']} <= CATALOG[model_id]['files'].keys():
        raise ModelUnavailable('OCR model catalog files do not match the verified engine specification.')
    return spec


def _alphabet(encoded, model_id):
    spec = _model_spec(model_id)
    if not isinstance(encoded, str) or not 1 < len(encoded) < 100_000:
        raise ModelUnavailable('Pinned OCR recognizer lacks its required embedded character dictionary.')
    alphabet = encoded.splitlines()
    if (len(alphabet) != spec['alphabet_count'] or len(set(alphabet)) != len(alphabet) or
            any(not 1 <= len(token) <= spec['maximum_token_codepoints'] or
                len(token.encode('utf-8')) > 8 or token == ' ' or
                any(ord(char) < 32 or ord(char) == 127 for char in token) for token in alphabet)):
        raise ModelUnavailable('Pinned OCR character dictionary is invalid.')
    if spec.get('alphabet_sha256') and hashlib.sha256(encoded.encode('utf-8')).hexdigest() != spec['alphabet_sha256']:
        raise ModelUnavailable('Pinned OCR character dictionary hash does not match the verified model.')
    return [''] + alphabet + [' ']


def _bounded_size(width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < 4 or width * height > MAX_INPUT_PIXELS or max(width, height) / min(width, height) > 32:
        raise ValueError('OCR source dimensions exceed bounded decoding limits.')
    factor = min(1., MAX_DETAIL_EDGE / max(width, height), math.sqrt(MAX_DETAIL_PIXELS / (width * height)))
    return max(4, math.floor(width * factor)), max(4, math.floor(height * factor))


def tile_windows(width, height):
    """Cover the bounded frame exactly; border overlap may exceed 128 pixels."""
    if width * height > MAX_DETAIL_PIXELS or max(width, height) > MAX_DETAIL_EDGE or min(width, height) < 4:
        raise ValueError('OCR tile input exceeds the detail frame budget.')
    def axis(length):
        if length <= TILE_EDGE:
            return [0]
        values = list(range(0, length - TILE_EDGE + 1, TILE_EDGE - TILE_OVERLAP))
        if values[-1] != length - TILE_EDGE:
            values.append(length - TILE_EDGE)
        return values
    windows = [(x, y, min(TILE_EDGE, width - x), min(TILE_EDGE, height - y)) for y in axis(height) for x in axis(width)]
    if len(windows) > MAX_TILES:
        raise ValueError('OCR geometry exceeds the tile count budget.')
    return windows


def _detail_bgr(image):
    if isinstance(image, Image.Image):
        width, height = image.size
        size = _bounded_size(width, height)
        _detail_resource_check(width * height)
        resized = image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)
        frame = np.ascontiguousarray(np.asarray(resized.convert('RGB'))[:, :, ::-1])
    else:
        frame = np.asarray(image)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('OCR array frames must be uint8 BGR with three channels.')
        height, width = frame.shape[:2]
        size = _bounded_size(width, height)
        _detail_resource_check(width * height)
        frame = np.ascontiguousarray(frame if size == (width, height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    return frame, {'source_frame_width': width, 'source_frame_height': height,
                   'input_scale_xy': [frame.shape[1] / width, frame.shape[0] / height]}


def sample_times(first, last, period, interval=DEFAULT_SAMPLE_SECONDS, cap=MAX_SAMPLES):
    if not all(math.isfinite(x) for x in (first, last, period)) or not 0 <= first < last <= MAX_SOURCE_SECONDS or period <= 0:
        raise ValueError('OCR needs a finite video extent of at most 12 hours.')
    interval, cap = _options({'ocr_sample_seconds': interval, 'ocr_max_samples': cap})
    tail = max(first, last - max(0.25, period + 0.001))
    count = min(cap, max(1, math.ceil((last - first) / interval) + 1))
    return np.linspace(first, tail, count).tolist() if tail > first else [first]


def _bgr(image):
    if isinstance(image, Image.Image):
        width, height = image.size
        if width < 4 or height < 4 or width * height > 16_000_000 or max(width, height) / min(width, height) > 32:
            raise ValueError('OCR frame dimensions exceed bounded preprocessing limits.')
        array = np.asarray(image.convert('RGB'))[:, :, ::-1]
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('OCR array frames must be uint8 BGR with three channels.')
        height, width = array.shape[:2]
        if width < 4 or height < 4 or width * height > 16_000_000 or max(width, height) / min(width, height) > 32:
            raise ValueError('OCR frame dimensions exceed bounded preprocessing limits.')
    scale = min(1.0, MAX_DECODE_EDGE / max(width, height))
    if scale < 1:
        array = cv2.resize(array, (max(4, round(width * scale)), max(4, round(height * scale))))
    return np.ascontiguousarray(array)


def _det_input(frame):
    height, width = frame.shape[:2]
    # Limit the largest edge. Upstream's minimum-edge resize can allocate very
    # wide tensors for tutorials with unusually shaped screen captures.
    new_h, new_w = max(32, round(height / 32) * 32), max(32, round(width / 32) * 32)
    resized = cv2.resize(frame, (new_w, new_h)).astype(np.float32) / np.float32(255)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(((resized - mean) / std).transpose(2, 0, 1)[None])


def _order_box(box):
    # Order around the center, then rotate so the upper-left edge comes first.
    points = np.asarray(box, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def _boxes(prediction, shape, check):
    values = np.asarray(prediction)
    if values.ndim != 4 or values.shape[:2] != (1, 1) or values.shape[2] > 960 or values.shape[3] > 960 or not np.isfinite(values).all():
        raise ValueError('OCR detector returned an invalid bounded probability map.')
    scores = values[0, 0]
    if scores.size == 0 or scores.min() < -1e-5 or scores.max() > 1.00001:
        raise ValueError('OCR detector returned values outside its expected score range.')
    bitmap = cv2.dilate((scores > .3).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Process largest regions first if a noisy map reaches the contour cap.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    source_h, source_w = shape[:2]
    height, width = scores.shape
    found = []
    for contour in contours[:MAX_CONTOURS]:
        check()
        center, (rw, rh), angle = cv2.minAreaRect(contour)
        if min(rw, rh) < 3:
            continue
        original = cv2.boxPoints((center, (rw, rh), angle))
        x1, y1 = np.maximum(0, np.floor(original.min(axis=0)).astype(int))
        x2, y2 = np.minimum([width - 1, height - 1], np.ceil(original.max(axis=0)).astype(int))
        mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
        cv2.fillPoly(mask, [np.round(original - [x1, y1]).astype(np.int32)], 1)
        score = float(cv2.mean(scores[y1:y2 + 1, x1:x2 + 1], mask)[0])
        if score < .5:
            continue
        # Analytic rotated-rectangle expansion using DB area/perimeter distance.
        # This deliberately does not claim pyclipper's rounded polygon parity.
        distance = rw * rh * 1.6 / (2 * (rw + rh))
        box = cv2.boxPoints((center, (rw + 2 * distance, rh + 2 * distance), angle))
        box *= np.array([source_w / width, source_h / height], np.float32)
        box[:, 0] = np.clip(box[:, 0], 0, source_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, source_h - 1)
        found.append((_order_box(box), score))
    found.sort(key=lambda item: item[1], reverse=True)
    limited = found[:MAX_BOXES]
    limited.sort(key=lambda item: (float(item[0][:, 1].min()), float(item[0][:, 0].min())))
    return limited, len(contours) > MAX_CONTOURS or len(found) > MAX_BOXES


class UnsupportedTextOrientation(ValueError):
    pass


def _line_skew(box):
    """Major-axis skew modulo 180 degrees; this does not identify reading order."""
    points = _order_box(box)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    return (math.degrees(math.atan2(float(edge[1]), float(edge[0]))) + 90) % 180 - 90


def _crop(frame, box):
    box = _order_box(box)
    if abs(_line_skew(box)) >= MAX_LINE_SKEW_DEGREES - 1e-4:
        # A rectangle cannot distinguish clockwise vertical text from an
        # upside-down line. Rotating every tall crop guessed wrong for +70/90
        # degree source rotations; retain an explicit unsupported-box record.
        raise UnsupportedTextOrientation('Text line reaches the unsupported 45-degree skew boundary; an orientation classifier is required.')
    width = max(4, round(float(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))))
    height = max(4, round(float(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(box, target)
    result = cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(result)


def _axis_crop(frame, box):
    """Keep raster axes unchanged; no guess about text orientation or order."""
    points = np.asarray(box, np.float32)
    left, top = np.maximum(0, np.floor(points.min(axis=0)).astype(int))
    right, bottom = np.minimum([frame.shape[1], frame.shape[0]], np.ceil(points.max(axis=0) + 1).astype(int))
    if right <= left or bottom <= top:
        raise ValueError('OCR crop has no pixels inside its frame.')
    return np.ascontiguousarray(frame[top:bottom, left:right])


def _text_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _box_iou(first, second):
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    lo_a, hi_a, lo_b, hi_b = a.min(axis=0), a.max(axis=0), b.min(axis=0), b.max(axis=0)
    intersection = float(np.prod(np.maximum(0, np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b))))
    union = float(np.prod(hi_a - lo_a) + np.prod(hi_b - lo_b)) - intersection
    return intersection / union if union > 0 else 0.


def _overlap_groups(lines, check):
    groups = []
    for line in lines:
        check()
        if not line['accepted']:
            continue
        key = _text_key(line['text'])
        matched = next((group for group in groups if group['normalized_text'] == key and
                        _box_iou(group['box_normalized'], line['box_normalized']) >= .5), None)
        if matched is None:
            matched = {'id': f'overlap-{len(groups):04d}', 'normalized_text': key,
                       'box_normalized': line['box_normalized'], 'line_ids': []}
            groups.append(matched)
        matched['line_ids'].append(line['line_id'])
        line['overlap_group'] = matched['id']
    return groups


def _temporal_agreement(samples, check):
    """Link exact normalized text in neighboring sampled frames; never rewrite it."""
    matches = []
    for previous, current in zip(samples, samples[1:]):
        check()
        # Group duplicate tile reads before linking observations; preserve all ids.
        previous_groups = previous.get('overlap_groups', [])
        current_groups = current.get('overlap_groups', [])
        for group in current_groups:
            check()
            for before in previous_groups:
                if group['normalized_text'] == before['normalized_text'] and _box_iou(group['box_normalized'], before['box_normalized']) >= .5:
                    matches.append({'normalized_text': group['normalized_text'],
                                    'from_sample_time': previous['time'], 'to_sample_time': current['time'],
                                    'from_line_ids': before['line_ids'], 'to_line_ids': group['line_ids'],
                                    'method': 'adjacent sampled frames, exact normalized text and axis-aligned box IoU >= 0.5'})
                    if len(matches) >= MAX_TEMPORAL_LINKS:
                        return {'links': matches, 'truncated': True, 'limit': MAX_TEMPORAL_LINKS}
                    break
    return {'links': matches, 'truncated': False, 'limit': MAX_TEMPORAL_LINKS}


def _quality_score(image):
    """A bounded texture/sharpness proxy, never a text correctness estimate."""
    thumbnail = image.copy()
    thumbnail.thumbnail((960, 960), Image.Resampling.BILINEAR)
    gray = np.asarray(thumbnail.convert('L'))
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def quality_times(desired, start, end, count):
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError('OCR quality sampling must be an integer from 1 to 3.')
    offsets = [0.] if count == 1 else [0., -QUALITY_WINDOW_SECONDS] if count == 2 else [0., -QUALITY_WINDOW_SECONDS, QUALITY_WINDOW_SECONDS]
    return list(dict.fromkeys(max(start, min(end, desired + offset)) for offset in offsets))


def _rec_input(crop):
    h, w = crop.shape[:2]
    scaled = max(1, math.ceil(48 * w / h))
    width = min(MAX_REC_WIDTH, max(320, scaled))
    resized_width = min(width, scaled)
    resized = cv2.resize(crop, (resized_width, 48)).astype(np.float32).transpose(2, 0, 1) / np.float32(127.5) - np.float32(1)
    tensor = np.zeros((1, 3, 48, width), np.float32)
    tensor[0, :, :, :resized_width] = resized
    return tensor, scaled > MAX_REC_WIDTH


def _decode_ctc(prediction, characters):
    values = np.asarray(prediction)
    if values.ndim != 3 or values.shape[0] != 1 or not 0 < values.shape[1] <= 1000 or values.shape[2] != len(characters):
        raise ValueError('OCR recognizer output and embedded character dictionary are incompatible.')
    if not np.isfinite(values).all() or values.min() < -1e-5 or values.max() > 1.00001:
        raise ValueError('OCR recognizer returned invalid/non-finite character scores.')
    indices = values[0].argmax(axis=1)
    maxima = values[0].max(axis=1)
    selected = (indices != 0)
    selected[1:] &= indices[1:] != indices[:-1]
    retained = indices[selected]
    return ''.join(characters[int(i)] for i in retained), float(maxima[selected].mean()) if retained.size else 0.0


class Recognizer:
    """Two bounded CPU sessions; loading never triggers a download."""
    def __init__(self, check=lambda: None, *, model_id=MODEL_ID):
        spec = _model_spec(model_id)
        self.model_id = model_id
        self.model_metadata = {key: CATALOG[model_id][key] for key in ('repository', 'revision', 'license')}
        check()
        _memory_check()
        location = model_path(model_id, verify=True)
        import onnxruntime as ort
        config = ort.SessionOptions()
        config.intra_op_num_threads = 2
        config.inter_op_num_threads = 1
        config.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        config.enable_cpu_mem_arena = False
        config.enable_mem_pattern = False
        self.detector = self.recognizer = None
        try:
            check()
            self.detector = ort.InferenceSession(str(location / spec['detector']), sess_options=config, providers=['CPUExecutionProvider'])
            check()
            _memory_check()
            self.recognizer = ort.InferenceSession(str(location / spec['recognizer']), sess_options=config, providers=['CPUExecutionProvider'])
            for session in (self.detector, self.recognizer):
                if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1 or session.get_inputs()[0].type != 'tensor(float)':
                    raise ModelUnavailable('Pinned OCR graph inputs/outputs are incompatible.')
            encoded = self.recognizer.get_modelmeta().custom_metadata_map.get('character')
            self.characters = _alphabet(encoded, model_id)
            if model_id == 'ppocr-v5-mobile':
                rec_input, rec_output = self.recognizer.get_inputs()[0], self.recognizer.get_outputs()[0]
                det_input, det_output = self.detector.get_inputs()[0], self.detector.get_outputs()[0]
                if (len(rec_input.shape) != 4 or rec_input.shape[1:3] != [3, 48] or
                        len(rec_output.shape) != 3 or rec_output.shape[-1] != len(self.characters) or
                        len(det_input.shape) != 4 or det_input.shape[1] != 3 or
                        len(det_output.shape) != 4 or det_output.shape[1] != 1):
                    raise ModelUnavailable('Pinned OCRv5 graph dimensions do not match verified metadata.')
            check()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.detector = self.recognizer = None
        gc.collect()


def _recognize_pass(frame, recognizer, check, *, fallback=False, line_budget=MAX_BOXES):
    check()
    tensor = _det_input(frame)
    check()
    session = recognizer.detector
    prediction = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    check()
    boxes, truncated = _boxes(prediction, frame.shape, check)
    detections, raw, resized_lines, unsupported = [], [], 0, []
    truncated = truncated or len(boxes) > line_budget
    for box, detection_score in boxes[:line_budget]:
        check()
        crop_method = 'perspective_deskew_under_45_degrees'
        try:
            crop = _crop(frame, box)
        except UnsupportedTextOrientation:
            if not fallback:
                unsupported.append({'box': np.round(box, 3).tolist(), 'line_skew_degrees': round(_line_skew(box), 3),
                                    'reason': 'steep_orientation_requires_classifier'})
                continue
            crop = _axis_crop(frame, box)
            crop_method = 'axis_aligned_unrotated_orientation_unknown'
        values, squeezed = _rec_input(crop)
        resized_lines += int(squeezed)
        session = recognizer.recognizer
        prediction = session.run(None, {session.get_inputs()[0].name: values})[0]
        check()
        text, score = _decode_ctc(prediction, recognizer.characters)
        row = {'text': text, 'model_score': round(score, 6), 'detection_model_score': round(detection_score, 6),
               'box': np.round(box, 3).tolist(), 'box_normalized': np.round(box / [frame.shape[1], frame.shape[0]], 6).tolist(),
               'crop_method': crop_method, 'line_skew_degrees': round(_line_skew(box), 3),
               'accepted': bool(text.strip()) and score >= .5}
        raw.append(row)
        if row['accepted']:
            detections.append(row)
    return {'detections': detections, 'frame_width': frame.shape[1], 'frame_height': frame.shape[0],
            'coordinate_space': 'decoded/reduced frame pixels', 'boxes_truncated': truncated,
            'raw_lines': raw,
            'wide_lines_squeezed': resized_lines, 'orientation_boxes_skipped': len(unsupported),
            'unsupported_orientation_boxes': unsupported, 'score_description': SCORE_DESCRIPTION}


def recognize_frame(image, recognizer, check=lambda: None, *, mode='baseline', max_raw_lines=MAX_DETAIL_LINES):
    """Same input for paired tests: RGB PIL or uint8 BGR. No text correction."""
    if mode not in {'baseline', 'detail'}:
        raise ValueError('OCR mode must be baseline or detail.')
    if type(max_raw_lines) is not int or not 1 <= max_raw_lines <= MAX_DETAIL_LINES:
        raise ValueError('OCR frame raw-line budget must be from 1 to 200.')
    check()
    if mode == 'baseline':
        result = _recognize_pass(_bgr(image), recognizer, check)
        for index, line in enumerate(result['raw_lines']):
            line.update(line_id=f'global/line-{index:03d}', pass_id='global',
                        tile_region_xywh=[0, 0, result['frame_width'], result['frame_height']])
        result['mode'] = mode
        result['model_id'] = getattr(recognizer, 'model_id', MODEL_ID)
        return result
    frame_deadline = time.monotonic() + MAX_WALL_SECONDS
    frame, input_metadata = _detail_bgr(image)
    height, width = frame.shape[:2]
    windows = tile_windows(width, height)
    specifications = [('global', (0, 0, width, height))]
    if max(width, height) > MAX_DECODE_EDGE:
        specifications.extend((f'tile-{index:03d}', window) for index, window in enumerate(windows))
    raw, passes, unsupported = [], [], []
    squeezed = 0
    truncated = False
    def bounded_check():
        _checkpoint(check, frame_deadline)
        _detail_resource_check(width * height)
    for pass_id, (x, y, w, h) in specifications:
        bounded_check()
        remaining = max_raw_lines - len(raw)
        if remaining <= 0:
            truncated = True
            break
        view = _bgr(frame) if pass_id == 'global' else np.ascontiguousarray(frame[y:y + h, x:x + w])
        evidence = _recognize_pass(view, recognizer, bounded_check, fallback=True, line_budget=min(MAX_BOXES, remaining))
        sx, sy = w / view.shape[1], h / view.shape[0]
        provenance = {'pass_id': pass_id, 'region_xywh': [x, y, w, h],
                      'processed_width': view.shape[1], 'processed_height': view.shape[0],
                      'scale_to_frame_xy': [sx, sy], 'raw_line_count': len(evidence['raw_lines']),
                      'boxes_truncated': evidence['boxes_truncated']}
        passes.append(provenance)
        for index, line in enumerate(evidence['raw_lines']):
            box = np.asarray(line['box']) * [sx, sy] + [x, y]
            raw.append({**line, 'pass_box': line['box'], 'box': np.round(box, 3).tolist(),
                        'box_normalized': np.round(box / [width, height], 6).tolist(),
                        'line_id': f'{pass_id}/line-{index:03d}', 'pass_id': pass_id,
                        'tile_region_xywh': [x, y, w, h]})
        squeezed += evidence['wide_lines_squeezed']
        truncated |= evidence['boxes_truncated']
    groups = _overlap_groups(raw, bounded_check)
    return {'mode': mode, 'model_id': getattr(recognizer, 'model_id', MODEL_ID),
            'detections': [line for line in raw if line['accepted']], 'raw_lines': raw,
            'frame_width': width, 'frame_height': height, **input_metadata,
            'coordinate_space': 'bounded decoded frame pixels; every tile box mapped to this frame',
            'passes': passes, 'planned_pass_count': len(specifications), 'processed_pass_count': len(passes),
            'passes_complete': len(passes) == len(specifications), 'boxes_truncated': truncated,
            'wide_lines_squeezed': squeezed, 'orientation_boxes_skipped': 0,
            'unsupported_orientation_boxes': unsupported, 'orientation_established': False,
            'axis_aligned_fallback_lines': sum(line['crop_method'].startswith('axis_aligned') for line in raw),
            'overlap_groups': groups, 'accepted_overlap_group_count': len(groups),
            'text_normalization_for_links': 'Unicode NFKC, casefold, collapse whitespace; raw text unchanged',
            'score_description': SCORE_DESCRIPTION,
            'resource_limits': {'frame_pixels': MAX_DETAIL_PIXELS, 'frame_edge': MAX_DETAIL_EDGE,
                                'tiles': MAX_TILES, 'lines_per_frame': MAX_DETAIL_LINES,
                                'rss_bytes': MAX_DETAIL_RSS_BYTES, 'minimum_available_ram_bytes': MIN_AVAILABLE_RAM,
                                'reserve_bytes': DETAIL_RESERVE_BYTES + width * height * 16,
                                'enforcement': 'cooperative checks between allocations/native calls; no continuous ORT supervision'}}


def _source_geometry(source, check, deadline):
    check()
    remaining = min(FRAME_TIMEOUT_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise RuntimeError('OCR exceeded its 600-second wall-time budget.')
    probe = subprocess.run(['ffprobe', '-v', 'error', *SOURCE_FLAGS, '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height', '-of', 'json', str(source)],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=remaining, check=True, **_guard())
    check()
    stream = json.loads(probe.stdout)['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    _bounded_size(width, height)
    return {'encoded_width': width, 'encoded_height': height, 'ffmpeg_autorotation': True}


def _decode_detail_frame(source, timestamp, target, check, deadline):
    """Decode one bounded higher-resolution JPEG with PTS and process timeout."""
    check()
    started = time.monotonic()
    if started >= deadline:
        raise RuntimeError('OCR exceeded its 600-second wall-time budget.')
    if target.exists() or target.is_symlink():
        raise ValueError('OCR decoder output already exists; refusing replacement.')
    factor = f'min(1,min({MAX_DETAIL_EDGE}/max(iw,ih),sqrt({MAX_DETAIL_PIXELS}/(iw*ih))))'
    filters = f"showinfo,scale=w='max(4,trunc(iw*{factor}))':h='max(4,trunc(ih*{factor}))'"
    args = ['ffmpeg', '-hide_banner', '-v', 'info', '-nostdin', '-n', '-threads', '1',
            '-ss', str(timestamp), '-copyts', *SOURCE_FLAGS, '-i', str(source), '-map', '0:v:0',
            '-frames:v', '1', '-an', '-sn', '-dn', '-filter_threads', '1', '-vf', filters,
            '-q:v', '2', '-threads', '1', '-fps_mode', 'passthrough', '-update', '1', str(target)]
    with tempfile.TemporaryFile() as error:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error, **_guard())
        try:
            while process.poll() is None:
                check()
                if time.monotonic() >= deadline or time.monotonic() - started >= FRAME_TIMEOUT_SECONDS:
                    raise RuntimeError('OCR frame decode exceeded its bounded time budget.')
                time.sleep(.02)
            check()
            error.seek(0)
            diagnostic = error.read(65536).decode('utf-8', 'replace')
            if process.returncode:
                raise RuntimeError('OCR frame decode failed: ' + diagnostic[-2000:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    if not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return None
    if target.stat().st_size > MAX_FRAME_FILE_BYTES:
        raise ValueError('OCR decoded frame exceeded its JPEG storage budget.')
    with Image.open(target) as frame:
        if frame.width * frame.height > MAX_DETAIL_PIXELS or max(frame.size) > MAX_DETAIL_EDGE:
            raise ValueError('OCR decoder exceeded its bounded detail dimensions.')
    match = re.search(r'n:\s*0\s+pts:\s*[-0-9]+\s+pts_time:([-+0-9.eE]+)', diagnostic)
    if not match or not math.isfinite(float(match[1])):
        raise RuntimeError('OCR decoder did not report a finite frame timestamp.')
    return float(match[1])


def _paths(source, out):
    source, out = Path(source).absolute(), Path(out).absolute()
    for path in (source, out):
        if any(component.is_symlink() for component in (path, *path.parents)):
            raise ValueError('OCR source/output paths must not contain symlinks.')
    source, out = source.resolve(), out.resolve()
    if not source.is_file() or source == out or out in source.parents:
        raise ValueError('OCR requires a local source outside its dedicated output directory.')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('OCR output already exists; use a fresh empty job output directory.')
    return source, out


def extract_text(source, duration, options, out, check):
    """Save bounded timestamped readings; never mutate source/transcript state."""
    started = time.monotonic()
    deadline = started + MAX_WALL_SECONDS
    interval, cap = _options(options)
    mode, quality_count = _detail_options(options)
    model_id = options.get('ocr_model', MODEL_ID)
    _model_spec(model_id)
    model_metadata = CATALOG[model_id]
    source_pixels = 0
    def checkpoint():
        _checkpoint(check, deadline)
        if mode == 'detail':
            _detail_resource_check(source_pixels)
    checkpoint()
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= MAX_SOURCE_SECONDS:
        raise ValueError('OCR supports finite video durations of at most 12 hours.')
    source, out = _paths(source, out)
    _, extent = _video_extent(source, duration, checkpoint, deadline)
    geometry = _source_geometry(source, checkpoint, deadline) if mode == 'detail' else None
    if geometry:
        source_pixels = geometry['encoded_width'] * geometry['encoded_height']
    checkpoint()
    times = sample_times(extent['start'], extent['end'], extent['reported_frame_period_seconds'], interval, cap)
    frames = out / 'ocr-frames'
    engine = Recognizer(checkpoint) if model_id == MODEL_ID else Recognizer(checkpoint, model_id=model_id)
    samples, duplicates, fallbacks = [], 0, 0
    raw_count = text_bytes = sample_bytes = stored_bytes = 0
    truncation = []
    decoder = _decode_detail_frame if mode == 'detail' else _decode_frame
    try:
        frames.mkdir(parents=True)
        for index, desired in enumerate(times):
            checkpoint()
            if raw_count >= MAX_JOB_RAW_LINES:
                truncation.append({'reason': 'job_raw_line_limit', 'unprocessed_sample_requests': len(times) - index})
                break
            candidates = []
            for candidate_index, requested in enumerate(quality_times(desired, extent['start'], times[-1], quality_count)):
                checkpoint()
                target = frames / (f'frame-{index:04d}.jpg' if quality_count == 1 else f'frame-{index:04d}-quality-{candidate_index}.jpg')
                timestamp = decoder(source, requested, target, checkpoint, deadline)
                if timestamp is None:
                    candidates.append({'requested_time': round(requested, 6), 'decoded': False})
                    continue
                actual = timestamp - extent['format_start_time']
                if not extent['start'] - .001 <= actual <= extent['end'] + .001:
                    raise RuntimeError('OCR decoded a frame outside the video extent.')
                actual = max(extent['start'], min(extent['end'], actual))
                stored_bytes += target.stat().st_size
                if stored_bytes > MAX_FRAME_STORAGE_BYTES:
                    raise RuntimeError('OCR frame evidence exceeded its 512 MiB storage budget; reduce sampling.')
                with Image.open(target) as image:
                    limit = MAX_DETAIL_EDGE if mode == 'detail' else MAX_DECODE_EDGE
                    if max(image.size) > limit or image.width * image.height > MAX_DETAIL_PIXELS:
                        raise ValueError('OCR decoder exceeded its bounded frame dimensions.')
                    score = _quality_score(image) if quality_count > 1 else None
                    size = list(image.size)
                candidates.append({'requested_time': round(requested, 6), 'time': round(actual, 6),
                                   'decoded': True, 'frame_ref': str(target.relative_to(out)),
                                   'sharpness_proxy': score, 'frame_size': size})
            available = [item for item in candidates if item['decoded']]
            if not available:
                earlier = max(extent['start'], desired - max(1., 2 * extent['reported_frame_period_seconds']))
                if samples and earlier <= samples[-1]['time']:
                    raise RuntimeError('OCR could not decode the final sample; coverage is incomplete.')
                target = frames / f'frame-{index:04d}-fallback.jpg'
                timestamp = decoder(source, earlier, target, checkpoint, deadline)
                fallbacks += 1
                if timestamp is None:
                    raise RuntimeError('OCR could not decode its sample; coverage is incomplete.')
                actual = timestamp - extent['format_start_time']
                if not extent['start'] - .001 <= actual <= extent['end'] + .001:
                    raise RuntimeError('OCR decoded a fallback outside the video extent.')
                stored_bytes += target.stat().st_size
                if stored_bytes > MAX_FRAME_STORAGE_BYTES:
                    raise RuntimeError('OCR frame evidence exceeded its 512 MiB storage budget; reduce sampling.')
                available = [{'requested_time': round(earlier, 6), 'time': round(actual, 6), 'decoded': True,
                              'frame_ref': str(target.relative_to(out)), 'sharpness_proxy': None, 'fallback': True}]
                candidates.extend(available)
            selected = max(available, key=lambda item: item['sharpness_proxy'] or 0.)
            actual = selected['time']
            if samples and actual <= samples[-1]['time'] + 1e-6:
                if actual < samples[-1]['time'] - 1e-6:
                    raise RuntimeError('OCR decoded non-monotonic selected frame timestamps.')
                duplicates += 1
                continue
            target = out / selected['frame_ref']
            with Image.open(target) as image:
                if mode == 'baseline':
                    evidence = recognize_frame(image, engine, checkpoint)
                else:
                    evidence = recognize_frame(image, engine, checkpoint, mode='detail',
                                               max_raw_lines=min(MAX_DETAIL_LINES, MAX_JOB_RAW_LINES - raw_count))
            for line in evidence.get('raw_lines', []):
                line['time'] = actual
                line['frame_ref'] = selected['frame_ref']
            sample = {'time': actual, 'requested_time': round(desired, 6),
                      'frame_ref': selected['frame_ref'], **evidence,
                      'quality_selection': {'requested_candidates': quality_count, 'candidates': candidates,
                                            'selected_frame_ref': selected['frame_ref'],
                                            'method': 'maximum bounded grayscale Laplacian variance; no OCR/text score' if quality_count > 1 else 'single requested frame'}}
            if geometry:
                area_ratio = evidence['frame_width'] * evidence['frame_height'] / source_pixels
                sample['decode_geometry'] = {'encoded_size': [geometry['encoded_width'], geometry['encoded_height']],
                                             'decoded_size': [evidence['frame_width'], evidence['frame_height']],
                                             'pixel_area_ratio': area_ratio, 'linear_scale_from_area': math.sqrt(area_ratio),
                                             'resolution_reduced': area_ratio < 1.,
                                             'axis_note': 'FFmpeg autorotation may swap encoded axes; boxes use the saved decoded frame'}
            raw = evidence.get('raw_lines', evidence['detections'])
            new_text_bytes = sum(len(line['text'].encode('utf-8')) for line in raw)
            new_bytes = len(json.dumps(sample, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'))
            if raw_count + len(raw) > MAX_JOB_RAW_LINES or text_bytes + new_text_bytes > MAX_JOB_RAW_TEXT_BYTES or sample_bytes + new_bytes > SAMPLE_EVIDENCE_BUDGET:
                truncation.append({'reason': 'job_evidence_limit', 'omitted_sample_time': actual,
                                   'omitted_raw_lines': len(raw), 'frame_ref': selected['frame_ref'],
                                   'unprocessed_sample_requests': len(times) - index - 1})
                break
            raw_count += len(raw)
            text_bytes += new_text_bytes
            sample_bytes += new_bytes
            samples.append(sample)
    finally:
        engine.close()
    checkpoint()
    gaps = [b['time'] - a['time'] for a, b in zip(samples, samples[1:])]
    result = {'provider': 'local_onnx_ppocr', 'ai_used': True, 'model_id': model_id, 'model': model_metadata['repository'],
              'revision': model_metadata['revision'], 'license': model_metadata['license'], 'recognizer_dictionary': 'embedded model metadata',
              'model_files': {name: {'bytes': identity[0], 'sha256': identity[1]} for name, identity in model_metadata['files'].items()},
              'model_integrity': model_metadata.get('integrity_pin', 'Catalog-pinned file sizes and SHA-256, verified before session loading'),
              'configuration': {'ocr_mode': mode, 'ocr_quality_samples': quality_count, 'ocr_sample_seconds': interval,
                                'ocr_max_samples': cap, 'ocr_model': model_id},
              'sample_count': len(samples), 'sample_requests': len(times), 'sample_cap': cap,
              'requested_interval_seconds': interval, 'maximum_sample_gap_seconds': max(gaps, default=0),
              'sampling_complete': not truncation and len(samples) + duplicates == len(times),
              'duplicate_frame_samples_skipped': duplicates, 'video_extent': extent,
              'source_duration_seconds': duration, 'tail_fallbacks': fallbacks,
              'first_sample_seconds': samples[0]['time'] if samples else None,
              'last_sample_seconds': samples[-1]['time'] if samples else None,
              'tail_unsampled_seconds': duration - samples[-1]['time'] if samples else duration,
              'sample_time_basis': 'decoded frame PTS reported by FFmpeg, minus format start time',
              'text_count': sum(len(sample['detections']) for sample in samples), 'samples': samples,
              'raw_line_count': raw_count, 'raw_text_bytes': text_bytes,
              'raw_readings_truncated': bool(truncation) or any(sample.get('boxes_truncated') for sample in samples),
              'budget_truncation': truncation,
              'evidence_limits': {'raw_lines_per_frame': MAX_DETAIL_LINES if mode == 'detail' else MAX_BOXES,
                                  'raw_lines_per_job': MAX_JOB_RAW_LINES, 'raw_text_bytes': MAX_JOB_RAW_TEXT_BYTES,
                                  'serialized_evidence_bytes': MAX_EVIDENCE_BYTES, 'frame_storage_bytes': MAX_FRAME_STORAGE_BYTES},
              'temporal_agreement': _temporal_agreement(samples, checkpoint) if mode == 'detail' else {'links': [], 'truncated': False, 'method': 'not computed in baseline mode'},
              'score_description': SCORE_DESCRIPTION, 'cpu_threads': 2, 'batch_size': 1,
              'frame_decode_max_edge': MAX_DETAIL_EDGE if mode == 'detail' else MAX_DECODE_EDGE,
              'source_geometry': geometry, 'max_text_boxes_per_pass': MAX_BOXES,
              'max_text_boxes_per_frame': MAX_DETAIL_LINES if mode == 'detail' else MAX_BOXES,
              'recognizer_max_width': MAX_REC_WIDTH, 'wall_time_budget_seconds': MAX_WALL_SECONDS,
              'max_supported_line_skew_degrees': MAX_LINE_SKEW_DEGREES,
              'max_supported_line_skew_exclusive': True, 'orientation_established': False,
              'orientation_boxes_skipped': sum(sample.get('orientation_boxes_skipped', 0) for sample in samples),
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'budget_enforcement': 'cooperative between model calls; decoder timeout and polled process-tree RSS; worker process cancellation' if mode == 'detail' else 'cooperative between model calls; decoder timeout and worker process cancellation',
              'postprocessing': 'OpenCV expanded rotated rectangles, not upstream polygon unclip',
              'limitations': DETAIL_LIMITATIONS if mode == 'detail' else LIMITATIONS, 'paid_usage': 0, 'data_transmitted': False}
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
    if len(encoded) > MAX_EVIDENCE_BYTES:
        raise RuntimeError('OCR report exceeds its 8 MiB serialized evidence budget; no complete report was written.')
    with (out / 'ocr.json').open('xb') as target:
        target.write(encoded)
    return result
