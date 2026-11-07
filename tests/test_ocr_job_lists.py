"""OCR list projection uses isolated SQLite data; no model/media jobs are run."""
import json
import time

from fastapi.testclient import TestClient
import pytest

from clipstudio import api, config, db, service


@pytest.fixture
def environment(tmp_path, monkeypatch):
    directory=tmp_path/'data'
    monkeypatch.setattr(config,'DATA',directory)
    monkeypatch.setattr(config,'DB',directory/'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR',str(directory))
    with TestClient(api.app) as client:
        pid=service.new_project('List projection fixture','source.mp4')
        source=json.loads(service.project_row(pid)['source'])
        source['sha256']='a'*64
        with db.tx() as connection:
            connection.execute('UPDATE projects SET source=? WHERE id=?',(db.dumps(source),pid))
        yield client,pid,source


def insert_job(pid,kind,result,*,ordinal=0,status='succeeded',params=None,raw=False):
    jid=db.uid();now=time.time()+ordinal
    with db.tx() as connection:
        connection.execute('''INSERT INTO jobs
            (id,project_id,kind,params,status,stage,request_hash,result,created_at,
             updated_at,started_at,finished_at,lease_token)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (jid,pid,kind,db.dumps(params or {}),status,status,'fixture-'+jid,
             result if raw else db.dumps(result) if result is not None else None,
             now,now,now-2,now,'never-expose-lease-token'))
    return jid


def test_large_ocr_results_are_omitted_before_list_rows_reach_python(environment):
    client,pid,source=environment
    evidence={'source_sha256':source['sha256'],'ocr':{'samples':[{'text':'x'*(2*1024**2)}]}}
    ids=[insert_job(pid,'ocr',evidence,ordinal=i) for i in range(4)]
    # The DB helper's raw rows, before JSON decoding or service projection, are
    # already small. This catches SELECT * followed by stripping in Python.
    raw_rows=db.list_job_rows(pid,30)
    assert len(raw_rows)==4
    assert all(row['result'] is None and row['result_omitted']==1 for row in raw_rows)
    assert len(db.dumps(raw_rows).encode())<10_000
    response=client.get('/api/v1/jobs')
    assert response.status_code==200 and len(response.content)<10_000
    for job in response.json():
        assert job['result'] is None and job['result_omitted'] is True
        assert job['detail_url']==f'/api/v1/jobs/{job["id"]}'
        assert 'lease_token' not in job
    project=client.get(f'/api/v1/projects/{pid}').json()
    assert len(db.dumps(project['jobs']).encode())<10_000
    assert project['ocr']==evidence  # Existing project OCR panel contract.
    assert client.get(f'/api/v1/jobs/{ids[-1]}').json()['result']==evidence
    assert client.get(f'/api/v1/projects/{pid}/ocr').json()==evidence
    assert json.loads(db.one('SELECT result FROM jobs WHERE id=?',(ids[-1],))['result'])==evidence


@pytest.mark.parametrize('kind',['render','track','analyze','transcribe','import','import-url','model-download','future-kind'])
def test_non_ocr_job_results_remain_identical_to_detail(environment,kind):
    client,pid,source=environment
    result={'export':{'revision':7,'video_url':'/api/v1/artifacts/example/video.mp4'},
            'keyframes':[{'time':1,'x':.2}],'nested':{'keep':'all values'}}
    jid=insert_job(pid,kind,result,params={'revision':7})
    full=client.get(f'/api/v1/jobs/{jid}').json()
    global_row=client.get('/api/v1/jobs').json()[0]
    project_row=client.get(f'/api/v1/projects/{pid}').json()['jobs'][0]
    assert global_row==full==project_row
    assert global_row['result']==result
    assert 'result_omitted' not in global_row and 'detail_url' not in global_row


def test_job_list_limits_project_filter_and_order_are_preserved(environment):
    client,pid,source=environment
    other=service.new_project('Other project','other.mp4')
    with db.tx() as connection:
        for index in range(205):
            owner=pid if index<100 else other
            connection.execute('''INSERT INTO jobs
                (id,project_id,kind,params,request_hash,result,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)''',
                (f'{index:032x}',owner,'ocr','{}',f'fixture-{index}',
                 db.dumps({'large':'x'*32768}),index,index))
    rows=client.get('/api/v1/jobs').json()
    assert len(rows)==200 and rows[0]['id']==f'{204:032x}' and rows[-1]['id']==f'{5:032x}'
    assert len(db.dumps(rows).encode())<200_000
    project_rows=client.get(f'/api/v1/projects/{pid}').json()['jobs']
    assert len(project_rows)==30
    assert all(row['project_id']==pid for row in project_rows)
    assert project_rows[0]['id']==f'{99:032x}' and project_rows[-1]['id']==f'{70:032x}'


@pytest.mark.parametrize('status',['queued','running','failed','cancelled'])
def test_ocr_without_a_result_is_not_marked_as_omitted(environment,status):
    client,pid,source=environment
    jid=insert_job(pid,'ocr',None,status=status)
    row=client.get('/api/v1/jobs').json()[0]
    assert row['result'] is None and row['result_omitted'] is False
    assert row['detail_url']==f'/api/v1/jobs/{jid}' and row['status']==status


def test_ocr_projection_does_not_parse_result_json(environment):
    client,pid,source=environment
    insert_job(pid,'ocr','invalid JSON that lists must not parse',status='failed',raw=True)
    response=client.get('/api/v1/jobs')
    assert response.status_code==200
    assert response.json()[0]['result_omitted'] is True
    assert response.json()[0]['result'] is None


def test_detail_omission_flags_are_list_only(environment):
    client,pid,source=environment
    jid=insert_job(pid,'ocr',{'source_sha256':source['sha256'],'ocr':{'samples':[]}})
    detailed=client.get(f'/api/v1/jobs/{jid}').json()
    assert 'result_omitted' not in detailed and 'detail_url' not in detailed
    assert detailed['result']['ocr']['samples']==[]
    assert client.get('/api/v1/jobs/'+('f'*32)).status_code==404
