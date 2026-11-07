"""Model downloads preserve pinned per-file origins and SHA verification."""
import hashlib
import io

import pytest
from clipstudio import models


def test_verified_auxiliary_files_use_their_pinned_upstream(tmp_path, monkeypatch):
    blobs={'weights.onnx':b'controlled model test bytes','tokenizer.json':b'{"model":"test"}'}
    spec={'repository':'test/export','revision':'a'*40,'runtime':'onnxruntime','license':'MIT',
          'files':{name:(len(data),hashlib.sha256(data).hexdigest()) for name,data in blobs.items()},
          'file_sources':{'tokenizer.json':('test/upstream','b'*40,'tokenizer.json')}}
    monkeypatch.setitem(models.CATALOG,'controlled-test',spec)
    monkeypatch.setenv('VEKTRI_DATA_DIR',str(tmp_path/'data'))
    urls=[]
    def fetch(request,timeout):
        urls.append(request.full_url)
        return io.BytesIO(blobs[request.full_url.rsplit('/',1)[-1]])
    monkeypatch.setattr(models.urllib.request,'urlopen',fetch)
    receipt=models.download_model('controlled-test',tmp_path/'receipt',lambda:None)
    assert receipt['sha256_verified']
    assert urls==[f'https://huggingface.co/test/export/resolve/{"a"*40}/weights.onnx',f'https://huggingface.co/test/upstream/resolve/{"b"*40}/tokenizer.json']
    assert models.model_path('controlled-test',verify=True).joinpath('tokenizer.json').read_bytes()==blobs['tokenizer.json']
    # Exact length alone must not make corrupted local weights reusable.
    models.model_path('controlled-test').joinpath('weights.onnx').write_bytes(b'!'*len(blobs['weights.onnx']))
    with pytest.raises(models.ModelUnavailable):models.model_path('controlled-test',verify=True)
