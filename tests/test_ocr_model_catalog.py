"""Catalog-owned OCR download origins and install integrity; no network/inference."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipstudio import models


V5 = 'ppocr-v5-mobile'
REVISION = 'bd941dae09da793e76c6ba15c7aaf40f8424232a'
HOST = 'https://www.modelscope.cn'
DETECTOR = 'ch_PP-OCRv5_det_mobile.onnx'
RECOGNIZER = 'ch_PP-OCRv5_rec_mobile.onnx'
FILES = {
    DETECTOR: (4819576, '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae'),
    RECOGNIZER: (16631306, '5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5'),
}
URLS = {
    name: f'{HOST}/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/{kind}/{name}'
    for name, kind in ((DETECTOR, 'det'), (RECOGNIZER, 'rec'))
}


@pytest.fixture(autouse=True)
def isolated_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf-cache'))
    monkeypatch.setattr(models.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10**9))
    monkeypatch.setattr(models.urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('Unexpected network request'))


@pytest.fixture
def tiny_v5(monkeypatch):
    """Retain the real fixed URLs, replacing only weight payload sizes/hashes."""
    blobs = {DETECTOR: b'controlled detector', RECOGNIZER: b'controlled recognizer'}
    spec = copy.deepcopy(models.CATALOG[V5])
    spec['files'] = {name: (len(blob), hashlib.sha256(blob).hexdigest()) for name, blob in blobs.items()}
    monkeypatch.setitem(models.CATALOG, V5, spec)
    return blobs


def destination():
    return models.model_root() / V5 / REVISION


def assert_no_staging():
    assert not list(models.model_root().glob(f'.{V5}-*'))


def test_v5_exact_published_catalog_and_default_ocr():
    spec = models.CATALOG[V5]
    assert spec['repository'] == 'RapidAI/RapidOCR'
    assert spec['revision'] == REVISION
    assert spec['download_version_path'] == 'v3.10.0'
    assert spec['license'] == 'Apache-2.0'
    assert spec['runtime'] == 'onnxruntime'
    assert spec['files'] == FILES
    assert spec['file_urls'] == URLS
    assert sum(size for size, _ in spec['files'].values()) == 21450882
    # Adding a candidate must not silently select it for existing OCR projects.
    status = models.model_status()
    assert status['ocr']['model'] == 'ppocr-v4'
    entry = next(value for value in status['models'] if value['id'] == V5)
    assert entry['download_bytes'] == 21450882
    assert entry['disk_required_bytes'] == 42901764
    assert entry['automatic_download'] is False
    assert entry['status'] == 'model_missing'


@pytest.mark.parametrize('url', [
    'https://untrusted.invalid/model.onnx',
    'https://www.modelscope.cn.untrusted.invalid/model.onnx',
    'http://www.modelscope.cn/model.onnx',
    'https://user@www.modelscope.cn/model.onnx',
    'https://user:password@www.modelscope.cn/model.onnx',
    'https://@www.modelscope.cn/model.onnx',
    'https://www.modelscope.cn/model.onnx?download=1',
    'https://www.modelscope.cn/model.onnx#weights',
    'https://www.modelscope.cn:8443/model.onnx',
    'https://www.modelscope.cn:invalid/model.onnx',
])
def test_unapproved_catalog_origin_rejected_before_request(url, tiny_v5, tmp_path, monkeypatch):
    models.CATALOG[V5]['file_urls'][DETECTOR] = url
    with pytest.raises(ValueError):
        models.download_model(V5, tmp_path / 'receipt', lambda: None)
    assert not destination().exists()
    assert_no_staging()


def test_unknown_model_rejected_before_network_or_output(tmp_path):
    receipt = tmp_path / 'receipt'
    with pytest.raises(ValueError, match='Unknown model'):
        models.download_model('https://untrusted.invalid/weights.onnx', receipt, lambda: None)
    assert not receipt.exists()
    assert not models.model_root().exists()


def test_fixed_v5_origins_publish_only_after_all_files_verified(tiny_v5, tmp_path, monkeypatch):
    requests, installs = [], []
    real_replace = models.os.replace

    def fetch(request, timeout):
        requests.append(request.full_url)
        assert timeout == 30
        assert not destination().exists()
        return io.BytesIO(tiny_v5[request.full_url.rsplit('/', 1)[-1]])

    def replace(source, target):
        source, target = Path(source), Path(target)
        assert target == destination()
        assert not target.exists()
        assert {name: (source / name).read_bytes() for name in tiny_v5} == tiny_v5
        provenance = json.loads((source / 'provenance.json').read_text())
        assert provenance['verified_sha256'] is True
        assert provenance['file_urls'] == URLS
        installs.append(target)
        real_replace(source, target)

    monkeypatch.setattr(models.urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(models.os, 'replace', replace)
    receipt = models.download_model(V5, tmp_path / 'receipt', lambda: None)
    assert requests == list(URLS.values())
    assert installs == [destination()]
    assert receipt['sha256_verified'] is True
    assert receipt['downloaded_bytes'] == sum(map(len, tiny_v5.values()))
    assert models.model_path(V5, verify=True) == destination()
    assert_no_staging()
    # A verified cache hit must not request the weights a second time.
    monkeypatch.setattr(models.urllib.request, 'urlopen',
                        lambda *args, **kwargs: pytest.fail('Verified cache should be reused'))
    cached = models.download_model(V5, tmp_path / 'cached-receipt', lambda: None)
    assert cached['status'] == 'already_available'
    assert cached['downloaded_bytes'] == 0


@pytest.mark.parametrize('failure', ['truncated', 'oversized', 'checksum', 'cancelled'])
def test_failed_second_file_preserves_existing_install(tiny_v5, tmp_path, monkeypatch, failure):
    # A damaged earlier installation remains useful diagnostic evidence.
    destination().mkdir(parents=True)
    sentinel = destination() / 'existing-damaged-file'
    sentinel.write_bytes(b'preserve this existing data')
    requests = []
    cancelled = False

    class CancelDuringBody(io.BytesIO):
        def read(self, size=-1):
            nonlocal cancelled
            payload = super().read(size)
            cancelled = True
            return payload

    def fetch(request, timeout):
        requests.append(request.full_url)
        name = request.full_url.rsplit('/', 1)[-1]
        payload = tiny_v5[name]
        if name == RECOGNIZER:
            if failure == 'truncated':
                payload = payload[:-1]
            elif failure == 'oversized':
                payload += b'!'
            elif failure == 'checksum':
                payload = b'!' * len(payload)
            else:
                return CancelDuringBody(payload)
        return io.BytesIO(payload)

    def check():
        if cancelled:
            raise RuntimeError('cancel requested')

    monkeypatch.setattr(models.urllib.request, 'urlopen', fetch)
    expected = {'oversized': 'exceeded', 'cancelled': 'cancel requested'}.get(failure, 'size/SHA-256 mismatch')
    with pytest.raises(RuntimeError, match=expected):
        models.download_model(V5, tmp_path / 'receipt', check)
    assert requests == list(URLS.values())
    assert sentinel.read_bytes() == b'preserve this existing data'
    assert list(destination().iterdir()) == [sentinel]
    assert not list(destination().parent.glob(f'{REVISION}.invalid-*'))
    assert not (tmp_path / 'receipt/model-download.json').exists()
    assert_no_staging()


def test_same_size_corrupted_cache_is_repaired_and_preserved(tiny_v5, tmp_path, monkeypatch):
    destination().mkdir(parents=True)
    for name, data in tiny_v5.items():
        (destination() / name).write_bytes(data)
    corrupt = b'!' * len(tiny_v5[RECOGNIZER])
    (destination() / RECOGNIZER).write_bytes(corrupt)
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        # The damaged installation is retained until both new files verify.
        assert (destination() / RECOGNIZER).read_bytes() == corrupt
        return io.BytesIO(tiny_v5[request.full_url.rsplit('/', 1)[-1]])

    monkeypatch.setattr(models.urllib.request, 'urlopen', fetch)
    receipt = models.download_model(V5, tmp_path / 'receipt', lambda: None)
    assert receipt['status'] == 'available'
    assert requests == list(URLS.values())
    assert models.model_path(V5, verify=True) == destination()
    backups = list(destination().parent.glob(f'{REVISION}.invalid-*'))
    assert len(backups) == 1
    assert (backups[0] / RECOGNIZER).read_bytes() == corrupt
    assert (destination() / RECOGNIZER).read_bytes() == tiny_v5[RECOGNIZER]
    assert_no_staging()


@pytest.mark.parametrize('missing_module,expected_runtime', [('cv2', False), ('onnxruntime', False), (None, True)])
def test_v5_status_requires_opencv_and_ort(tiny_v5, monkeypatch, missing_module, expected_runtime):
    destination().mkdir(parents=True)
    for name, data in tiny_v5.items():
        (destination() / name).write_bytes(data)
    monkeypatch.setattr(models.importlib.util, 'find_spec',
                        lambda name: None if name == missing_module else object())
    status = models.model_status()
    entry = next(value for value in status['models'] if value['id'] == V5)
    assert entry['available'] is True
    assert entry['runtime_available'] is expected_runtime
    assert entry['status'] == ('ready' if expected_runtime else 'runtime_missing')
    assert status['ocr']['model'] == 'ppocr-v4'
