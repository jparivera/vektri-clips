"""One conservative supervisor; transactional global cap survives extra workers."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import psutil
from . import config,db,runtime

stopping=False
supervisor_pid=None
def stop(*args):
    global stopping
    stopping=True

def supervisor_lost():
    """A serve-managed worker must not outlive its supervisor.

    When `clipstudio serve` dies without running its cleanup (closed terminal,
    SIGKILL, crash), the worker is re-parented; it then stops like a SIGTERM so a
    running job's lease expires and a restarted supervisor can reclaim it.
    """
    return supervisor_pid is not None and os.getppid()!=supervisor_pid

def terminate_tree(process):
    try:
        parent=psutil.Process(process.pid);children=parent.children(recursive=True)
        for child in reversed(children):
            try:child.terminate()
            except psutil.Error:pass
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            for child in reversed(children):
                try:child.kill()
                except psutil.Error:pass
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=3)
    except psutil.NoSuchProcess:pass

def run_once(heartbeat=None):
    available = psutil.virtual_memory().available
    def report(state, job_id=None):
        if heartbeat is not None:
            heartbeat.report(state, job_id)
    if available < runtime.MIN_AVAILABLE_RAM:
        with db.tx() as c:
            # Recover stale/cancelled work without claiming it or spending an attempt.
            db.recover(c, time.time())
            queued = c.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone()
            if queued:
                c.execute("UPDATE jobs SET stage='waiting for at least 1.5 GiB available RAM' WHERE status='queued'")
        report('waiting_memory' if queued else 'idle')
        return 'waiting_memory' if queued else 'idle'
    job=db.claim()
    if not job:
        report('idle')
        return 'idle'
    report('busy', job['id'])
    jid,token=job['id'],job['lease_token']
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}
    log=config.DATA/'logs'/(jid+'.log')
    with log.open('a') as output:
        process=subprocess.Popen([sys.executable,'-m','clipstudio.task',jid,token],env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        last=0;cancelled=False;lost=False
        while process.poll() is None:
            if supervisor_lost():stop()
            if stopping:
                terminate_tree(process);break
            try:
                if time.monotonic()-last>2:
                    db.heartbeat(jid,token);report('busy',jid);last=time.monotonic()
                else:db.check_lease(jid,token)
            except db.Cancelled:
                cancelled=True;terminate_tree(process);break
            except db.LeaseLost:
                lost=True;terminate_tree(process);break
            time.sleep(.25)
        if cancelled:db.finish(jid,token,'cancelled',error='Cancelled by operator; originals and completed stages preserved.')
        elif stopping:
            # Expire now; a restarted supervisor reclaims, retaining successful stages.
            with db.tx() as c:c.execute('UPDATE jobs SET lease_until=? WHERE id=? AND lease_token=? AND status=\'running\'',(time.time()-1,jid,token))
        elif not lost:
            row=db.one('SELECT status FROM jobs WHERE id=?',(jid,))
            if row and row['status']=='running':
                with db.tx() as c:c.execute('UPDATE jobs SET lease_until=? WHERE id=? AND lease_token=?',(time.time()-1,jid,token))
    report('idle')
    return 'worked'

def main(argv=None):
    global stopping, supervisor_pid
    stopping = False
    parser = argparse.ArgumentParser(description='One local worker; low-memory jobs stay queued.')
    parser.add_argument('--once', action='store_true', help='Try one job; exit 75 if queued work is RAM-blocked')
    parser.add_argument('--supervisor-pid', type=int, default=None,
                        help='Stop when this process is no longer the parent (set by `clipstudio serve`)')
    args = parser.parse_args(argv)
    supervisor_pid = args.supervisor_pid
    if supervisor_pid is not None:
        if os.getppid() != supervisor_pid:
            print(json.dumps({'error': 'The declared supervisor is not the parent process'}), file=sys.stderr)
            return 2
        try:
            # Linux PR_SET_PDEATHSIG: a dead supervisor thread delivers SIGTERM; the ppid poll covers the rest.
            import ctypes
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        except (OSError, AttributeError):
            pass
    db.migrate();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    heartbeat = runtime.WorkerHeartbeat()
    try:
        heartbeat.report('idle', available_ram=psutil.virtual_memory().available)
        while not stopping:
            if supervisor_lost():
                stop()
                break
            state = run_once(heartbeat)
            if args.once:
                print(json.dumps({'state': state, 'minimum_available_ram_bytes': runtime.MIN_AVAILABLE_RAM}))
                return 75 if state == 'waiting_memory' else 0
            if state != 'worked':
                time.sleep(1)
        return 0
    finally:
        heartbeat.close()

if __name__=='__main__':raise SystemExit(main())
