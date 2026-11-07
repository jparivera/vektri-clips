"""Protocol and boundary checks; no agent, remote service or production data used."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from clipstudio import mcp_bridge as mcp

PROJECT = 'a' * 32
FINGERPRINT = 'b' * 64


def request(method, params=None, request_id=1):
    return {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params or {}}


def initialize(bridge):
    result = bridge.handle(request('initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                                                  'clientInfo': {'name': 'test-client', 'version': '1'}}))
    assert result['result']['protocolVersion'] == mcp.PROTOCOL_VERSION
    assert bridge.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


def decisions():
    return {'packet_fingerprint': FINGERPRINT, 'agent_name': 'test review', 'candidates': [
        {'id': 'proposal-1', 'start': 10, 'end': 18, 'title': 'Test proposal', 'rationale': 'Review fixture evidence.',
         'evidence': [{'type': 'transcript', 'start': 10, 'end': 18, 'note': 'Source evidence.'}],
         'confidence': .5, 'uncertainty': 'Requires human review', 'scores': {}, 'method': 'review'}]}


def test_lifecycle_readonly_catalog_and_protocol_negotiation():
    bridge = mcp.Bridge(request=lambda *a: pytest.fail('Initialization must not reach the app'))
    assert bridge.handle(request('tools/list'))['error']['code'] == -32002
    initialized = bridge.handle(request('initialize', {'protocolVersion': 'future-version', 'capabilities': {},
                                                       'clientInfo': {'name': 'client', 'version': '1'}}, 2))
    assert initialized['result']['protocolVersion'] == '2025-11-25'
    assert initialized['result']['capabilities'] == {'tools': {'listChanged': False}}
    assert bridge.handle(request('tools/list', request_id=3))['error']['code'] == -32002
    bridge.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
    tools = bridge.handle(request('tools/list', request_id=4))['result']['tools']
    assert {t['name'] for t in tools} == {'capabilities', 'projects', 'project', 'jobs', 'evidence'}
    assert all(t['annotations']['readOnlyHint'] and not t['annotations']['openWorldHint'] for t in tools)
    assert bridge.handle(request('ping', request_id=5))['result'] == {}


@pytest.mark.parametrize('name,args,expected', [
    ('capabilities', {}, ('GET', '/api/v1/capabilities')),
    ('projects', {'query': 'local & safe', 'limit': 2, 'offset': 10}, ('GET', '/api/v1/projects?q=local+%26+safe&limit=2&offset=10')),
    ('project', {'project_id': PROJECT}, ('GET', '/api/v1/projects/' + PROJECT)),
    ('evidence', {'project_id': PROJECT}, ('GET', '/api/v1/projects/' + PROJECT + '/evidence')),
    ('jobs', {}, ('GET', '/api/v1/jobs')),
])
def test_read_routes_are_fixed_and_results_are_structured(name, args, expected):
    calls = []
    bridge = mcp.Bridge(request=lambda *a: calls.append(a) or {'sample': 'untrusted\nsource'})
    initialize(bridge)
    result = bridge.handle(request('tools/call', {'name': name, 'arguments': args}, 2))['result']
    assert calls == [expected]
    assert result['isError'] is False
    assert json.loads(result['content'][0]['text']) == result['structuredContent']


def test_proposal_is_explicit_exact_one_post_and_duplicate_id_cannot_resubmit():
    calls = []
    bridge = mcp.Bridge(allow_proposals=True, request=lambda *a: calls.append(a) or {'method': 'agent-assisted'})
    initialize(bridge)
    tool = next(t for t in bridge.tool_list() if t['name'] == 'submit_decisions')
    assert tool['annotations']['readOnlyHint'] is False
    assert tool['annotations']['destructiveHint'] is True
    assert 'AgentDecisions' in tool['inputSchema']['$defs']
    message = request('tools/call', {'name': 'submit_decisions', 'arguments': {'project_id': PROJECT, 'decisions': decisions()}}, 2)
    assert bridge.handle(message)['result']['isError'] is False
    assert calls[0][:2] == ('POST', '/api/v1/projects/' + PROJECT + '/agent-decisions')
    assert calls[0][2]['packet_fingerprint'] == FINGERPRINT
    assert bridge.handle(message)['error']['code'] == -32600
    assert len(calls) == 1


@pytest.mark.parametrize('name,args', [
    ('publish', {}), ('render', {}), ('import', {}), ('shell', {'command': 'pwd'}),
    ('submit_decisions', {'project_id': PROJECT, 'decisions': decisions()}),
    ('project', {'project_id': '../secrets'}), ('project', {'project_id': 'http://example.com'}),
    ('project', {'project_id': PROJECT, 'url': 'https://example.com'}),
    ('projects', {'limit': True}), ('projects', {'limit': '10'}), ('projects', {'limit': 101}),
    ('jobs', {'path': '/tmp'}), ('jobs', None), ([], {}),
])
def test_invalid_or_disallowed_tools_never_make_network_requests(name, args):
    bridge = mcp.Bridge(request=lambda *a: pytest.fail('No transport for invalid input'))
    initialize(bridge)
    result = bridge.handle(request('tools/call', {'name': name, 'arguments': args}, 2))
    assert result['error']['code'] == -32602


def test_invalid_proposal_is_rejected_locally_without_echoing_private_input():
    bridge = mcp.Bridge(allow_proposals=True, request=lambda *a: pytest.fail('Invalid proposal cannot reach API'))
    initialize(bridge)
    bad = decisions()
    bad['packet_fingerprint'] = 'PRIVATE-INVALID-INPUT'
    result = bridge.handle(request('tools/call', {'name': 'submit_decisions', 'arguments': {'project_id': PROJECT, 'decisions': bad}}, 2))
    assert result['error']['code'] == -32602
    assert 'PRIVATE-INVALID-INPUT' not in json.dumps(result)


def test_stale_proposal_and_unavailable_app_are_tool_errors_without_retry():
    calls = []
    def stale(*args):
        calls.append(args)
        raise mcp.LocalAPIError('Local API HTTP 409: Evidence changed. Request a fresh packet.')
    bridge = mcp.Bridge(allow_proposals=True, request=stale)
    initialize(bridge)
    result = bridge.handle(request('tools/call', {'name': 'submit_decisions', 'arguments': {'project_id': PROJECT, 'decisions': decisions()}}, 2))
    assert result['result']['isError'] is True
    assert '409' in result['result']['content'][0]['text']
    assert len(calls) == 1


def test_notifications_and_late_cancellations_never_dispatch_or_reply():
    bridge = mcp.Bridge(request=lambda *a: pytest.fail('Notifications do not call tools'))
    initialize(bridge)
    for message in [
        {'jsonrpc': '2.0', 'method': 'notifications/cancelled', 'params': {'requestId': 1}},
        {'jsonrpc': '2.0', 'method': 'notifications/cancelled', 'params': {'requestId': 'unknown'}},
        {'jsonrpc': '2.0', 'method': 'notifications/cancelled', 'params': 'malformed'},
        {'jsonrpc': '2.0', 'method': 'tools/call', 'params': {'name': 'submit_decisions', 'arguments': {}}},
    ]:
        assert bridge.handle(message) is None


@pytest.mark.parametrize('message', [[], {'jsonrpc': '2.0', 'id': None, 'method': 'ping'},
    {'jsonrpc': '2.0', 'id': True, 'method': 'ping'}, {'jsonrpc': '2.0', 'id': 1.5, 'method': 'ping'},
    {'jsonrpc': '1.0', 'id': 1, 'method': 'ping'}, {'jsonrpc': '2.0', 'id': 1, 'method': 'ping', 'params': []}])
def test_invalid_jsonrpc(message):
    result = mcp.Bridge().handle(message)
    assert result['error']['code'] == -32600
    assert 'id' in result
    if not isinstance(message, dict) or type(message.get('id')) not in (str, int):
        assert result['id'] is None
    else:
        assert result['id'] == message['id']


def test_session_id_tracking_is_bounded(monkeypatch):
    monkeypatch.setattr(mcp, 'MAX_REQUESTS', 1)
    bridge = mcp.Bridge()
    assert bridge.handle(request('ping'))['result'] == {}
    assert bridge.handle(request('ping', request_id=2))['error']['code'] == -32000


class Response:
    def __init__(self, status, body):
        self.status, self.body = status, body
    def read(self, limit):
        return self.body[:limit]


def mock_connection(monkeypatch, *, status=200, body=b'{}', failure=None):
    calls = []
    class Connection:
        def __init__(self, host, port, timeout):
            calls.append(('connect', host, port, timeout))
        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            if failure:
                raise failure
        def getresponse(self):
            return Response(status, body)
        def close(self):
            calls.append(('close',))
    monkeypatch.setattr(mcp.http.client, 'HTTPConnection', Connection)
    return calls


def test_transport_is_loopback_without_proxy_redirect_or_credentials(monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'https://credential-stealer.invalid')
    calls = mock_connection(monkeypatch, status=302, body=b'{}')
    with pytest.raises(mcp.LocalAPIError, match='redirect rejected'):
        mcp.local_request('GET', '/api/v1/projects')
    assert calls[0] == ('connect', '127.0.0.1', mcp.config.PORT, 10)
    assert len([c for c in calls if c[0] == 'GET']) == 1
    assert calls[-1] == ('close',)


def test_transport_bounded_response_and_strict_json(monkeypatch):
    monkeypatch.setattr(mcp, 'MAX_RESPONSE_BYTES', 8)
    mock_connection(monkeypatch, body=b'x' * 9)
    with pytest.raises(mcp.LocalAPIError, match='limit'):
        mcp.local_request('GET', '/api/v1/projects')
    mock_connection(monkeypatch, body=b'NaN')
    with pytest.raises(mcp.LocalAPIError, match='invalid JSON'):
        mcp.local_request('GET', '/api/v1/projects')


def test_write_connection_error_reports_uncertain_outcome_without_retry(monkeypatch):
    calls = mock_connection(monkeypatch, failure=TimeoutError('not exposed'))
    with pytest.raises(mcp.LocalAPIError, match='outcome is uncertain'):
        mcp.local_request('POST', '/api/v1/projects/' + PROJECT + '/agent-decisions', decisions())
    assert len([c for c in calls if c[0] == 'POST']) == 1


def test_read_connection_error_is_actionable(monkeypatch):
    mock_connection(monkeypatch, failure=ConnectionRefusedError())
    with pytest.raises(mcp.LocalAPIError, match='Start Clip Studio locally'):
        mcp.local_request('GET', '/api/v1/capabilities')


def test_stdio_limits_and_recovers_after_parse_error(monkeypatch):
    source = io.BytesIO(b'not json\n' + json.dumps(request('ping')).encode() + b'\n')
    destination = io.BytesIO()
    mcp.run(stdin=source, stdout=destination)
    messages = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert messages[0]['error']['code'] == -32700
    assert 'id' in messages[0] and messages[0]['id'] is None
    assert messages[1]['result'] == {}
    monkeypatch.setattr(mcp, 'MAX_MESSAGE_BYTES', 8)
    destination = io.BytesIO()
    mcp.run(stdin=io.BytesIO(b'x' * 100 + b'\n'), stdout=destination)
    assert len(destination.getvalue().splitlines()) == 1
    oversized = json.loads(destination.getvalue())
    assert oversized['error']['code'] == -32600
    assert 'id' in oversized and oversized['id'] is None


def test_real_stdio_subprocess_handshake_tool_catalog_and_clean_eof(tmp_path):
    root = Path(__file__).resolve().parents[1]
    messages = [request('initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                                      'clientInfo': {'name': 'subprocess-test', 'version': '1'}}),
                {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                request('tools/list', request_id=2), request('ping', request_id=3)]
    result = subprocess.run([sys.executable, '-m', 'clipstudio.mcp_bridge'],
        input=''.join(json.dumps(m) + '\n' for m in messages), text=True,
        capture_output=True, timeout=10,
        env={**os.environ, 'PYTHONPATH': str(root/'backend'), 'PYTHONDONTWRITEBYTECODE': '1',
             'VEKTRI_DATA_DIR': str(tmp_path/'untouched-data')})
    assert result.returncode == 0 and result.stderr == ''
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r['id'] for r in responses] == [1, 2, 3]
    assert len(responses[1]['result']['tools']) == 5
    assert not (tmp_path/'untouched-data').exists()


def test_actual_api_proposals_are_fingerprinted_and_leave_timeline_jobs_unchanged(tmp_path, monkeypatch):
    """Real FastAPI/service/SQLite validation with isolated metadata, no media job.

    Like the existing API tests, this needs a host that permits TestClient's
    cross-thread loopback primitives; socket-restricted sandboxes can stall it.
    """
    from fastapi.testclient import TestClient
    from clipstudio import config, db, service
    from clipstudio.api import app
    data = tmp_path/'studio'
    monkeypatch.setattr(config, 'DATA', data)
    monkeypatch.setattr(config, 'DB', data/'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(data))
    db.migrate()
    pid = service.new_project('Isolated bridge metadata fixture', 'fixture.mp4', {'permission_status': 'synthetic'})
    source = {'sha256': 'c'*64, 'metadata': {'duration': 24}}
    with db.tx() as connection:
        connection.execute("UPDATE projects SET source=?,status='ready' WHERE id=?", (db.dumps(source), pid))
    before = dict(service.project_row(pid))
    with TestClient(app) as client:
        def through_api(method, path, payload=None):
            response = client.request(method, path, json=payload)
            if response.is_error:
                raise mcp.LocalAPIError(f'HTTP {response.status_code}: {response.json()["detail"]}')
            return response.json()
        bridge = mcp.Bridge(allow_proposals=True, request=through_api)
        initialize(bridge)
        packet = bridge.handle(request('tools/call', {'name': 'evidence', 'arguments': {'project_id': pid}}, 2))['result']['structuredContent']['data']
        proposed = decisions()
        proposed['packet_fingerprint'] = packet['packet_fingerprint']
        call = {'name': 'submit_decisions', 'arguments': {'project_id': pid, 'decisions': proposed}}
        accepted = bridge.handle(request('tools/call', call, 3))
        assert accepted['result']['isError'] is False
        after = dict(service.project_row(pid))
        assert after['revision'] == before['revision']
        assert after['source'] == before['source']
        assert json.loads(after['analysis'])['candidates'][0]['id'] == 'proposal-1'
        for table in ('jobs', 'exports', 'publications', 'revisions'):
            assert db.one('SELECT count(*) n FROM '+table)['n'] == 0
        stale = bridge.handle(request('tools/call', call, 4))
        assert stale['result']['isError'] is True
        assert '409' in stale['result']['content'][0]['text']
        assert dict(service.project_row(pid))['analysis'] == after['analysis']
