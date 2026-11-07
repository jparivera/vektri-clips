"""Small pinned local-model catalog. Downloads occur only via explicit jobs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CATALOG = {
    'tiny.en': {
        'repository': 'Systran/faster-whisper-tiny.en', 'revision': '0d3d19a32d3338f10357c0889762bd8d64bbdeba',
        'license': 'MIT', 'capability': 'English transcription and estimated word alignment', 'runtime': 'faster_whisper',
        'files': {
            'config.json': (2317, '14b1b421a90349bc551b881461426b561a874049cb9e4c4864f2ca384f6a7cc5'),
            'model.bin': (75537502, '1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a'),
            'tokenizer.json': (2128466, '929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df'),
            'vocabulary.txt': (422309, 'ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf'),
        },
    },
    'minilm': {
        'repository': 'sentence-transformers/all-MiniLM-L6-v2', 'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        'license': 'Apache-2.0', 'capability': 'English semantic text similarity; not generative vision or event recognition', 'runtime': 'onnxruntime',
        'files': {
            'onnx/model_quint8_avx2.onnx': (23046789, 'b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21'),
            'tokenizer.json': (466247, 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
        },
    },
    'tiny': {
        'repository': 'Systran/faster-whisper-tiny',
        'revision': 'd90ca5fe260221311c53c58e660288d3deb8d356',
        'license': 'MIT',
        'capability': 'Multilingual transcription; English, Portuguese or detected language; review accuracy',
        'runtime': 'faster_whisper',
        'files': {
            'config.json': (2249, 'a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8'),
            'model.bin': (75538270, 'dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1'),
            'tokenizer.json': (2203239, 'fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab'),
            'vocabulary.txt': (459861, '34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913'),
        },
    },
    'clip-vit-b32': {
        'repository': 'Xenova/clip-vit-base-patch32',
        'revision': 'd15189d7028b43f1d3e65039190477f6af591c2a',
        'license': 'MIT',
        'capability': 'English image-text retrieval on sampled frames; not temporal event recognition or calibrated confidence',
        'runtime': 'onnxruntime',
        'files': {
            'onnx/text_model_int8.onnx': (64070791, '18845f2ccc35223bb7fec403383a131154b11ac0918df25cf51986df5efd3a21'),
            'onnx/vision_model_uint8.onnx': (88648915, 'b95448754a6ae56964ec80c570c80bb9863787ee85f51b664abed8c0e5f22a7a'),
            'tokenizer.json': (2224119, 'f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd'),
            'preprocessor_config.json': (520, '6f638fb9401a6d6296feff533ee7efe657b787c49f954f82f5906b36ef2a1b1f'),
            'tokenizer_config.json': (775, '60ba2912bc6344c94bc16bbdec27fa1209409167b6f2fdf3cfe9e65462ea3967'),
            'special_tokens_map.json': (472, 'c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac'),
            'config.json': (4524, '493ef57ff783e42d1530c91b53469b7fdf8db8a9c1408e86998fcb7899a4f495'),
        },
    },
    'xclip-b16-k600': {
        'repository': 'imbcmdth/xclip-onnx',
        'revision': '649f3c91b59cd24be316dc505e26eacf5cd00801',
        'license': 'MIT',
        'capability': 'Experimental learned eight-frame temporal embedding retrieval; third-party export omits full classifier prompt generator',
        'runtime': 'onnxruntime',
        'files': {
            'video_tower.onnx': (499458926, '87a87b51ae52efa549e6e48f3a3b41f4fe0e5b91e154afcda5d9ac8dd1b975b4'),
            'text_tower.onnx': (253850982, 'c14df0fdda1ba530330e278b8fabeb0bfd9a58c8caa10dfdee38b6ea197d0afc'),
            'config.json': (8897, '43d49d757dc8a3ef3dfd242d7ebe0c2effcfa0404f9343864e50c45245f99d3a'),
            'preprocessor_config.json': (310, '3376c13ec67e5505734b3d5b61d51b7f753cb1896d9661171c25ad35afcb69b6'),
            'tokenizer_config.json': (927, '1636bf45a35613aa5bc18520bca07d58550d3b272b6f8a12fc17363f52b2a930'),
            'special_tokens_map.json': (472, 'c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac'),
            'tokenizer.json': (2224208, 'f3dba9c5f2100fc09edb20622a18083f1fbf5041e4c0af371959bd7502428d44'),
        },
        'file_sources': {
            'config.json': ('microsoft/xclip-base-patch16-kinetics-600', 'e4921c41fc296102aae210d43d4127c5e3e51928', 'config.json'),
            'preprocessor_config.json': ('microsoft/xclip-base-patch16-kinetics-600', 'e4921c41fc296102aae210d43d4127c5e3e51928', 'preprocessor_config.json'),
            'tokenizer_config.json': ('microsoft/xclip-base-patch16-kinetics-600', 'e4921c41fc296102aae210d43d4127c5e3e51928', 'tokenizer_config.json'),
            'special_tokens_map.json': ('microsoft/xclip-base-patch16-kinetics-600', 'e4921c41fc296102aae210d43d4127c5e3e51928', 'special_tokens_map.json'),
            'tokenizer.json': ('microsoft/xclip-base-patch16-kinetics-600', 'e4921c41fc296102aae210d43d4127c5e3e51928', 'tokenizer.json'),
        },
    },
    'ppocr-v4': {
        'repository': 'SWHL/RapidOCR',
        'revision': '4e4644045a07c403b1ad40ca97340ecb4a8dc2c1',
        'license': 'Apache-2.0',
        'capability': 'Local Chinese/English text detection and recognition; review small, rotated and stylized text',
        'runtime': 'onnxruntime',
        'files': {
            'PP-OCRv4/ch_PP-OCRv4_det_infer.onnx': (4745517, 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'),
            'PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx': (10857958, '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b'),
        },
    },
    'ppocr-v5-mobile': {'repository': 'RapidAI/RapidOCR',
     'revision': 'bd941dae09da793e76c6ba15c7aaf40f8424232a',
     'license': 'Apache-2.0',
     'capability': 'Experimental Chinese/English mobile OCRv5 conversion; compare quality before selecting',
     'runtime': 'onnxruntime',
     'download_version_path': 'v3.10.0',
     'integrity_pin': 'Expected size and SHA-256; hosting version path is not an immutable commit',
     'files': {'ch_PP-OCRv5_det_mobile.onnx': (4819576,
                                               '4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae'),
               'ch_PP-OCRv5_rec_mobile.onnx': (16631306,
                                               '5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5')},
     'file_urls': {'ch_PP-OCRv5_det_mobile.onnx': 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx',
                   'ch_PP-OCRv5_rec_mobile.onnx': 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx'}},
}


class ModelUnavailable(RuntimeError):
    pass


def model_root():
    return Path(os.environ.get('VEKTRI_DATA_DIR', str(ROOT / 'data'))) / 'models'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def model_path(name, verify=False):
    if name not in CATALOG:
        raise ValueError('Unknown model; supported models: ' + ', '.join(CATALOG))
    spec = CATALOG[name]
    paths = [model_root() / name / spec['revision']]
    if name in ('tiny.en', 'tiny'):
        paths.append(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub' / ('models--' + spec['repository'].replace('/', '--')) / 'snapshots' / spec['revision'])
    for base in paths:
        if all((base / file).is_file() and (base / file).stat().st_size == size for file, (size, _) in spec['files'].items()):
            if verify and any(digest(base / file) != expected for file, (_, expected) in spec['files'].items()):
                continue
            return base
    raise ModelUnavailable(f'{name} is unavailable or incomplete. Use the explicit model download action ({sum(v[0] for v in spec["files"].values()):,} bytes); manual editing/export remain available.')


def model_status():
    models = []
    for name, spec in CATALOG.items():
        try:
            path = model_path(name)
            available = True
        except ModelUnavailable:
            path, available = None, False
        runtime = importlib.util.find_spec(spec['runtime']) is not None
        if name in ('minilm', 'clip-vit-b32', 'xclip-b16-k600'):
            runtime = runtime and importlib.util.find_spec('tokenizers') is not None
        if name in ('clip-vit-b32', 'xclip-b16-k600'):
            runtime = runtime and importlib.util.find_spec('PIL') is not None
        if name in ('ppocr-v4', 'ppocr-v5-mobile'):
            runtime = runtime and importlib.util.find_spec('cv2') is not None
        models.append({'id': name, 'name': name, 'available': available, 'runtime_available': runtime,
                       'status': 'ready' if available and runtime else 'model_missing' if not available else 'runtime_missing',
                       'download_bytes': sum(v[0] for v in spec['files'].values()),
                       'disk_required_bytes': 2 * sum(v[0] for v in spec['files'].values()),
                       'path': str(path) if path else None, 'revision': spec['revision'], 'repository': spec['repository'],
                       'license': spec['license'], 'capability': spec['capability'], 'automatic_download': False})
    return {'models': models, 'local_only': True, 'cloud': {'status': 'unconfigured', 'paid_usage': 0},
            'diarization': 'unavailable',
            'ocr': {'status': next(m['status'] for m in models if m['id'] == 'ppocr-v4'), 'model': 'ppocr-v4',
                    'capability': 'Sampled upright Chinese/English text; review omissions and transcription errors', 'automatic_download': False},
            'temporal_vision': {'status': next(m['status'] for m in models if m['id'] == 'xclip-b16-k600'),
                    'model': 'xclip-b16-k600', 'capability': 'Experimental learned temporal embedding retrieval; event correctness unvalidated',
                    'maximum_tree_rss_bytes': 1342177280, 'minimum_available_ram_bytes': 1610612736, 'automatic_download': False},
            'semantic_vision': {
                'status': next(m['status'] for m in models if m['id'] == 'clip-vit-b32'),
                'capability': 'English image-text retrieval over sampled frames; not temporal event understanding',
                'model': 'clip-vit-b32', 'automatic_download': False}, 'model_directory': str(model_root())}


def download_model(name: str, out: Path, check) -> dict:
    if name not in CATALOG:
        raise ValueError('Unknown model; supported models: ' + ', '.join(CATALOG))
    check()
    spec = CATALOG[name]
    out.mkdir(parents=True, exist_ok=True)
    try:
        existing = model_path(name, verify=True)
        return {'name': name, 'status': 'already_available', 'path': str(existing), 'revision': spec['revision'], 'downloaded_bytes': 0}
    except ModelUnavailable:
        pass
    root = model_root()
    root.mkdir(parents=True, exist_ok=True)
    total = sum(size for size, _ in spec['files'].values())
    if shutil.disk_usage(root).free < total * 2 + 50_000_000:
        raise RuntimeError(f'Insufficient free disk: allow at least {total * 2 + 50_000_000:,} bytes for verified model installation.')
    destination = root / name / spec['revision']
    completed = 0
    with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=root) as staging_name:
        staging = Path(staging_name)
        for file, (expected_size, expected_hash) in spec['files'].items():
            check()
            target = staging / file
            target.parent.mkdir(parents=True, exist_ok=True)
            origin_repo, origin_revision, origin_file = spec.get('file_sources', {}).get(file, (spec['repository'], spec['revision'], file))
            # Per-file origins are immutable catalog entries, never request parameters.
            url = spec.get('file_urls', {}).get(file) or f'https://huggingface.co/{origin_repo}/resolve/{origin_revision}/{origin_file}'
            # Only checked-in catalog URLs are accepted; callers choose a model ID.
            from urllib.parse import urlsplit
            origin = urlsplit(url)
            if origin.scheme != 'https' or origin.hostname not in {'huggingface.co', 'www.modelscope.cn'} or origin.port not in (None, 443) or origin.username is not None or origin.password is not None or origin.query or origin.fragment:
                raise ValueError('Model catalog URL is not an approved pinned model origin.')
            request = urllib.request.Request(url, headers={'User-Agent': 'VektriClipStudio-local-model-downloader/1'})
            size = 0
            sha = hashlib.sha256()
            # URLs are fixed catalog entries, never media/user URLs. HF's signed CDN redirect is required.
            with urllib.request.urlopen(request, timeout=30) as response, target.open('wb') as sink:
                while data := response.read(256 * 1024):
                    check()
                    size += len(data)
                    if size > expected_size:
                        raise RuntimeError('Model response exceeded the pinned file size.')
                    sink.write(data)
                    sha.update(data)
                    progress = {'name': name, 'file': file, 'downloaded_bytes': completed + size, 'total_bytes': total}
                    (out / 'download-progress.json').write_text(json.dumps(progress))
            if size != expected_size or sha.hexdigest() != expected_hash:
                raise RuntimeError(f'Pinned size/SHA-256 mismatch for {file}; model was not installed.')
            completed += size
        check()
        (staging / 'provenance.json').write_text(json.dumps({'name': name, **spec, 'verified_sha256': True}, indent=2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Preserve a damaged pre-existing installation for diagnosis.
            backup = destination.with_name(destination.name + '.invalid-' + next(tempfile._get_candidate_names()))
            destination.rename(backup)
        os.replace(staging, destination)
    result = {'name': name, 'status': 'available', 'path': str(destination), 'revision': spec['revision'],
              'downloaded_bytes': completed, 'sha256_verified': True, 'license': spec['license'], 'provider': spec['repository']}
    (out / 'model-download.json').write_text(json.dumps(result, indent=2))
    return result
