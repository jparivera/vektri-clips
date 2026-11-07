"""Read-only API/CLI exposure stays confined to the configured library."""
import json
import sys

from fastapi.testclient import TestClient
import pytest

from clipstudio import api, cli, config, db


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / 'library'
    monkeypatch.setattr(config, 'DATA', root)
    monkeypatch.setattr(config, 'DB', root / 'studio.sqlite3')
    db.migrate()
    return root


def test_api_report_verify_and_no_delete_routes(library):
    with TestClient(api.app) as client:
        response = client.get('/api/v1/maintenance/storage-plan?data_dir=/outside')
        assert response.status_code == 200
        plan = response.json()
        assert plan['data_directory'] == str(library)
        assert plan['read_only'] and not plan['automatic_deletion_supported']
        verified = client.post('/api/v1/maintenance/verify-plan', json=plan)
        assert verified.status_code == 200 and verified.json()['valid']
        plan['data_directory'] = '/outside'
        rejected = client.post('/api/v1/maintenance/verify-plan', json=plan)
        assert rejected.status_code == 200 and not rejected.json()['valid']
        assert client.post('/api/v1/maintenance/apply-plan', json={}).status_code in (404, 405)
        assert client.delete('/api/v1/maintenance/storage-plan').status_code == 405


def test_cli_plan_and_verify_work_with_api_stopped(library, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'request', lambda *args, **kwargs: pytest.fail('Read-only local planning must not require the server'))
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'plan-storage'])
    cli.main()
    output = json.loads(capsys.readouterr().out)
    assert output['ok'] and output['result']['data_directory'] == str(library)
    saved = tmp_path / 'plan.json'
    saved.write_text(json.dumps(output))
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'verify-storage-plan', str(saved)])
    cli.main()
    assert json.loads(capsys.readouterr().out)['result']['valid']


def test_cli_rejects_oversized_plan_without_reading_it(library, tmp_path, monkeypatch, capsys):
    saved = tmp_path / 'oversized.json'
    with saved.open('wb') as out:
        out.truncate(8 * 1024**2 + 1)
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'verify-storage-plan', str(saved)])
    with pytest.raises(SystemExit, match='2'):
        cli.main()
    assert 'at most 8 MiB' in capsys.readouterr().err


def test_cli_missing_library_does_not_initialize_it(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'does-not-exist'
    monkeypatch.setattr(config, 'DATA', root)
    monkeypatch.setattr(sys, 'argv', ['clipstudio', 'plan-storage'])
    cli.main()
    assert json.loads(capsys.readouterr().out)['result']['status'] == 'blocked'
    assert not root.exists()


def test_api_storage_summary_is_read_only_and_counts_hardlinks_once(library):
    (library / 'uploads').mkdir(parents=True, exist_ok=True)
    original = library / 'uploads' / 'staged.bin'
    original.write_bytes(b'x' * 4096)
    (library / 'projects').mkdir(exist_ok=True)
    import os
    os.link(original, library / 'projects' / 'linked.bin')
    (library / 'dangling').symlink_to('/nonexistent/target')
    with TestClient(api.app) as client:
        before = sorted(p.name for p in library.iterdir())
        response = client.get('/api/v1/maintenance/storage')
        assert response.status_code == 200
        summary = response.json()
        assert summary['data_directory'] == str(library)
        assert set(summary['directories']) == {'projects', 'models', 'uploads', 'logs'}
        assert summary['disk']['total'] >= summary['disk']['free'] > 0
        assert summary['database_bytes'] > 0
        assert summary['symlinks_skipped'] == 1 and summary['complete'] is True
        # The hardlinked bytes are attributed once, to whichever directory is walked first.
        assert summary['directories']['projects'] + summary['directories']['uploads'] == 4096
        assert sorted(p.name for p in library.iterdir()) == before
        assert client.delete('/api/v1/maintenance/storage').status_code == 405
        assert client.post('/api/v1/maintenance/storage').status_code == 405
