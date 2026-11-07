"""Real subprocess recovery and stage reuse, separate from control-path mocks."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil
import pytest
from clipstudio import config,db,service,runtime
from clipstudio.security import checksum

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def studio(tmp_path,monkeypatch):
    if psutil.virtual_memory().available < runtime.MIN_AVAILABLE_RAM:
        pytest.skip('Requires at least 1.5 GiB available RAM; production safety guard is not bypassed')
    data=tmp_path/'studio';monkeypatch.setattr(config,'DATA',data);monkeypatch.setattr(config,'DB',data/'studio.sqlite3')
    db.migrate()
    env={**os.environ,'VEKTRI_DATA_DIR':str(data),'PYTHONPATH':str(ROOT/'backend'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    return env

def worker(env):
    return subprocess.Popen([sys.executable,'-m','clipstudio.worker','--once'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

def run(env):
    p=worker(env)
    output=p.communicate(timeout=90)[0]
    if p.returncode == 75:
        pytest.skip('Worker deferred queued work because available RAM fell below 1.5 GiB; guard preserved')
    assert p.returncode==0,output.decode()

def wait_for(fn,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=fn()
        if value:return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting for condition')

@pytest.mark.timeout(120)
def test_real_worker_import_render_cached_revision_and_integrity(studio):
    source=ROOT/'fixtures'/'action_motion.mp4';sha=checksum(source)
    result=service.import_path(source,'Worker recovery fixture',{'genre':'action'},'worker:import')
    run(studio)
    p=service.get_project(result['project_id']);assert p['status']=='ready'
    timeline=p['timeline'];timeline['output']={'width':360,'height':640,'fps':24,'crf':25};timeline['segments'][0]['start']=2;timeline['segments'][0]['end']=5
    changed=service.update_timeline(p['id'],p['revision'],timeline)
    first=service.queue_render(p['id'],changed['revision'],'worker:first');run(studio)
    job=db.decode_job(db.one('SELECT * FROM jobs WHERE id=?',(first['id'],)))
    assert job['status']=='succeeded',job['error']
    second=service.queue_render(p['id'],changed['revision'],'worker:second');run(studio)
    job2=db.decode_job(db.one('SELECT * FROM jobs WHERE id=?',(second['id'],)))
    assert job2['status']=='succeeded',job2['error']
    assert set(job2['result']['reused_stages'])=={'render','package'}
    assert len(db.all('SELECT * FROM exports'))==1
    assert checksum(source)==sha
    assert checksum(config.project_dir(p['id'])/'source'/'original.bin')==sha
    # Different revision with identical pixels reuses the render but binds a new package.
    other=json.loads(json.dumps(timeline));other['audio']['fade']=.1
    rev3=service.update_timeline(p['id'],changed['revision'],other)['revision']
    rev4=service.update_timeline(p['id'],rev3,timeline)['revision']
    third=service.queue_render(p['id'],rev4,'worker:revision');run(studio)
    j3=db.decode_job(db.one('SELECT * FROM jobs WHERE id=?',(third['id'],)))
    assert j3['status']=='succeeded',j3['error']
    assert 'render' in j3['result']['reused_stages']
    assert j3['result']['export']['revision']==rev4
    assert len(db.all('SELECT * FROM exports'))==2
    # External mutation is rejected before another render starts.
    original=config.project_dir(p['id'])/'source'/'original.bin';original.chmod(0o644)
    with original.open('ab') as f:f.write(b'corruption')
    broken=service.queue_render(p['id'],rev4,'worker:mutated');run(studio)
    failure=db.one('SELECT * FROM jobs WHERE id=?',(broken['id'],))
    assert failure['status']=='failed' and 'integrity' in failure['error']

@pytest.mark.timeout(120)
def test_real_supervisor_sigkill_stale_lease_restart(studio):
    result=service.import_path(ROOT/'fixtures'/'action_motion.mp4','Kill/restart fixture',{'genre':'action'})
    run(studio)
    p=service.get_project(result['project_id']);timeline=p['timeline']
    timeline['output']={'width':1080,'height':1920,'fps':30,'crf':22}
    timeline['segments']=[{'id':f's{i}','start':0,'end':10,'crop':{'mode':'fit'},'gain_db':0} for i in range(8)]
    rev=service.update_timeline(p['id'],p['revision'],timeline)['revision'];job=service.queue_render(p['id'],rev,'worker:kill')
    process=worker(studio)
    try:
        def started_or_resource_blocked():
            if process.poll() == 75:
                pytest.skip('Worker deferred render because available RAM fell below 1.5 GiB; guard preserved')
            if process.poll() is not None:
                raise AssertionError(process.stdout.read().decode())
            return db.one("SELECT * FROM jobs WHERE id=? AND status='running' AND stage='render'",(job['id'],))
        wait_for(started_or_resource_blocked)
        children=psutil.Process(process.pid).children(recursive=True)
        assert children
        os.kill(process.pid,signal.SIGKILL);process.wait(timeout=5)
        def all_ended():
            for child in children:
                try:
                    if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:return False
                except psutil.NoSuchProcess:pass
            return True
        wait_for(all_ended,10)
        # Exercise actual lease expiry, not an artificial successful HTTP response.
        wait_for(lambda:db.one('SELECT lease_until FROM jobs WHERE id=?',(job['id'],))['lease_until']<time.time(),25)
        run(studio)
        finished=db.decode_job(db.one('SELECT * FROM jobs WHERE id=?',(job['id'],)))
        assert finished['status']=='succeeded',finished['error']
        assert finished['attempts']==2
        assert len(db.all('SELECT * FROM exports WHERE project_id=?',(p['id'],)))==1
        assert any(a['event']=='job.recovered' for a in db.all('SELECT * FROM audit WHERE project_id=?',(p['id'],)))
    finally:
        if process.poll() is None:process.kill();process.wait()
