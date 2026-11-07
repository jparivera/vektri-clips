"""Actual parser/middleware checks; chunk framing is also exercised at ASGI level.

Installed TestClient/httpx coalesces generator bodies internally. Its generator
tests prove omitted Content-Length handling, while the direct ASGI cases prove
incremental reads, bounded forwarding, and partial multipart resource cleanup.
"""
import asyncio

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.testclient import TestClient
import pytest
import starlette.formparsers

from clipstudio import config, db
from clipstudio.api import app
from clipstudio.limits import RequestBodyLimitMiddleware, RequestBodyTooLarge


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    directory = tmp_path / 'data'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    config.initialize()
    db.migrate()


def request_scope(path='/body', content_type='application/json', **changes):
    scope = {
        'type': 'http', 'http_version': '1.1', 'asgi': {'version': '3.0'},
        'method': 'POST', 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
        'root_path': '', 'query_string': b'', 'server': ('testserver', 80),
        'client': ('127.0.0.1', 12345),
        'headers': [(b'host', b'testserver'), (b'content-type', content_type.encode())],
    }
    scope.update(changes)
    return scope


def run_asgi(application, scope, chunks, *, before_receive=None):
    """Provide genuine separate ASGI frames without prejoining the test body."""
    messages = []
    received = []
    iterator = iter(chunks)

    async def receive():
        await asyncio.sleep(0)
        if before_receive:
            before_receive(len(received))
        try:
            chunk, more_body = next(iterator)
        except StopIteration:
            return {'type': 'http.disconnect'}
        message = {'type': 'http.request', 'body': chunk, 'more_body': more_body}
        received.append(message)
        return message

    async def send(message):
        messages.append(message)

    asyncio.run(application(scope, receive, send))
    return messages, received


def test_actual_api_rejects_json_generator_without_content_length():
    def body():
        yield b'{"url":"'
        for _ in range(9):
            yield b'x' * (1024**2)
        yield b'"}'

    with TestClient(app) as client:
        response = client.post('/api/v1/imports/url', content=body(), headers={'Content-Type': 'application/json'})
    assert 'content-length' not in response.request.headers
    assert response.status_code == 413
    assert response.headers['connection'] == 'close'
    assert db.all('SELECT * FROM projects') == []


@pytest.mark.parametrize('content_type', ['application/json', 'application/problem+json', 'text/plain', 'multipart/form-data; boundary=x'])
def test_non_media_route_cannot_expand_limit_by_content_type(content_type):
    mini = FastAPI()

    @mini.post('/body')
    async def body(request: Request):
        return {'size': len(await request.body())}

    mini.add_middleware(RequestBodyLimitMiddleware, json_limit=16, media_limit=1024)
    with TestClient(mini) as client:
        response = client.post('/body', content=(part for part in [b'a' * 9, b'b' * 8]), headers={'Content-Type': content_type})
    assert response.status_code == 413
    assert 'content-length' not in response.request.headers


def test_incremental_limit_forwards_original_frames_and_does_not_drain():
    forwarded = []

    async def consumer(scope, receive, send):
        while True:
            message = await receive()
            forwarded.append(message)

    def chunks():
        yield b'a' * 8, True
        yield b'b' * 8, True
        yield b'c', True
        raise AssertionError('Oversized body was drained or prebuffered')

    messages, received = run_asgi(RequestBodyLimitMiddleware(consumer, json_limit=16), request_scope(), chunks())
    assert len(forwarded) == 2 and len(received) == 3
    assert all(first is second for first, second in zip(forwarded, received))
    assert sum(len(message['body']) for message in forwarded) == 16
    assert messages[0]['type'] == 'http.response.start' and messages[0]['status'] == 413


def test_exact_limit_and_real_disconnect_are_forwarded_unchanged():
    observed = []

    async def consumer(scope, receive, send):
        observed.extend([await receive(), await receive()])
        await send({'type': 'http.response.start', 'status': 204, 'headers': []})
        await send({'type': 'http.response.body', 'body': b''})

    messages, received = run_asgi(RequestBodyLimitMiddleware(consumer, json_limit=16), request_scope(), [(b'x' * 16, False)])
    assert observed[0] is received[0]
    assert observed[1] == {'type': 'http.disconnect'}
    assert messages[0]['status'] == 204


def test_false_small_content_length_does_not_bypass_count():
    async def consumer(scope, receive, send):
        await receive()

    scope = request_scope()
    scope['headers'].append((b'content-length', b'1'))
    messages, _ = run_asgi(RequestBodyLimitMiddleware(consumer, json_limit=16), scope, [(b'x' * 17, False)])
    assert messages[0]['status'] == 413


@pytest.mark.parametrize('declared', [b'17', b'-1', b'not-a-number'])
def test_invalid_or_oversized_declaration_rejected_before_reading(declared):
    async def consumer(scope, receive, send):
        pytest.fail('Rejected request must not enter parser')

    def chunks():
        raise AssertionError('Rejected declaration must not read a body')
        yield b'', False

    scope = request_scope()
    scope['headers'].append((b'content-length', declared))
    messages, received = run_asgi(RequestBodyLimitMiddleware(consumer, json_limit=16), scope, chunks())
    assert messages[0]['status'] == 413 and received == []


def test_partial_multipart_spool_closes_on_chunked_overflow(monkeypatch):
    opened = []
    original = starlette.formparsers.SpooledTemporaryFile

    def tracked_spool(*args, **kwargs):
        spool = original(*args, **kwargs)
        opened.append(spool)
        return spool

    monkeypatch.setattr(starlette.formparsers, 'SpooledTemporaryFile', tracked_spool)
    # Force an actual disk rollover on the first frame to exercise both cleanup
    # and the absence of writes from the rejected second frame.
    monkeypatch.setattr(starlette.formparsers.MultiPartParser, 'spool_max_size', 4)
    mini = FastAPI()

    @mini.post('/api/v1/imports')
    async def upload(file: UploadFile = File()):
        pytest.fail('Oversized multipart must never enter the endpoint')

    mini.add_middleware(RequestBodyLimitMiddleware, json_limit=16, media_limit=256)
    prefix = b'--test\r\nContent-Disposition: form-data; name="file"; filename="video.mp4"\r\nContent-Type: video/mp4\r\n\r\n'
    first = prefix + b'a' * 32

    def before_receive(number):
        if number == 1:
            assert len(opened) == 1 and not opened[0].closed
            assert opened[0]._rolled
            assert opened[0].tell() == 32

    def chunks():
        yield first, True
        yield b'b' * 256, True
        raise AssertionError('Multipart overflow must not drain the rest')

    messages, received = run_asgi(mini, request_scope('/api/v1/imports', 'multipart/form-data; boundary=test'), chunks(), before_receive=before_receive)
    assert messages[0]['status'] == 413 and len(received) == 2
    assert len(opened) == 1 and opened[0].closed


def test_actual_chunked_multipart_upload_and_json_still_work():
    body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="sample.mp4"\r\nContent-Type: video/mp4\r\n\r\naccepted local bytes\r\n--test--\r\n'
    with TestClient(app) as client:
        response = client.post('/api/v1/imports', content=(body[offset:offset + 13] for offset in range(0, len(body), 13)), headers={'Content-Type': 'multipart/form-data; boundary=test'})
        assert 'content-length' not in response.request.headers
        assert response.status_code == 202, response.text
        pid = response.json()['project_id']
        patched = client.patch(f'/api/v1/projects/{pid}', json={'title': 'Normal JSON survives'})
        assert patched.status_code == 200, patched.text
        assert patched.json()['title'] == 'Normal JSON survives'


def test_actual_chunked_multipart_aggregate_limit_rejects_before_acceptance(monkeypatch):
    monkeypatch.setattr(config, 'MAX_UPLOAD', 64)
    prefix = b'--test\r\nContent-Disposition: form-data; name="file"; filename="sample.mp4"\r\n\r\n'
    with TestClient(app) as client:
        response = client.post('/api/v1/imports', content=(part for part in [prefix, b'x' * (1024**2 + 65), b'\r\n--test--\r\n']), headers={'Content-Type': 'multipart/form-data; boundary=test'})
    assert 'content-length' not in response.request.headers
    assert response.status_code == 413
    assert db.all('SELECT * FROM projects') == []
    assert list((config.DATA / 'uploads').iterdir()) == []


def test_media_allowance_does_not_expand_raw_resumable_chunks():
    async def consumer(scope, receive, send):
        await receive()

    scope = request_scope('/api/v1/uploads/test', 'application/octet-stream', method='PATCH')
    messages, _ = run_asgi(RequestBodyLimitMiddleware(consumer, json_limit=16, media_limit=1024), scope, [(b'x' * 17, False)])
    assert messages[0]['status'] == 413


def test_no_second_response_after_streaming_response_began():
    messages = []

    async def consumer(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await receive()

    async def receive():
        return {'type': 'http.request', 'body': b'x' * 17, 'more_body': True}

    async def send(message):
        messages.append(message)

    with pytest.raises(RequestBodyTooLarge):
        asyncio.run(RequestBodyLimitMiddleware(consumer, json_limit=16)(request_scope(), receive, send))
    assert len(messages) == 1 and messages[0]['status'] == 200


@pytest.mark.parametrize('scope', [{'type': 'lifespan'}, {'type': 'websocket'}])
def test_non_http_scopes_pass_through(scope):
    observed = []

    async def consumer(actual_scope, receive, send):
        observed.append(actual_scope)

    async def unused():
        pytest.fail('Non-HTTP middleware must not consume messages')

    asyncio.run(RequestBodyLimitMiddleware(consumer)(scope, unused, unused))
    assert observed == [scope]
