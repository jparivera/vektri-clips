"""SQLite migrations, transactional leases and append-only audit history."""
import contextlib
import hashlib
import json
import sqlite3
import time
import uuid
from . import config

def dumps(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def fingerprint(obj): return hashlib.sha256(dumps(obj).encode()).hexdigest()
def uid(): return uuid.uuid4().hex

MIGRATION_1 = '''
CREATE TABLE IF NOT EXISTS projects (
 id TEXT PRIMARY KEY,title TEXT NOT NULL,genre TEXT NOT NULL,tags TEXT NOT NULL DEFAULT '[]',
 status TEXT NOT NULL,source TEXT NOT NULL DEFAULT '{}',revision INTEGER NOT NULL DEFAULT 0,
 transcript TEXT,analysis TEXT,created_at REAL NOT NULL,updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
 project_id TEXT NOT NULL REFERENCES projects(id), revision INTEGER NOT NULL,timeline TEXT NOT NULL,
 fingerprint TEXT NOT NULL,created_at REAL NOT NULL, PRIMARY KEY(project_id,revision)
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY,project_id TEXT REFERENCES projects(id),kind TEXT NOT NULL,params TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'queued',stage TEXT NOT NULL DEFAULT 'queued',idempotency_key TEXT UNIQUE,
 request_hash TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,max_attempts INTEGER NOT NULL DEFAULT 3,
 lease_token TEXT,lease_until REAL,heartbeat_at REAL,cancel_requested INTEGER NOT NULL DEFAULT 0,
 error TEXT,result TEXT,created_at REAL NOT NULL,updated_at REAL NOT NULL,started_at REAL,finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status,created_at);
CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id,created_at);
CREATE TABLE IF NOT EXISTS stages (
 project_id TEXT NOT NULL,name TEXT NOT NULL,fingerprint TEXT NOT NULL,artifact_dir TEXT NOT NULL,
 result TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(project_id,name,fingerprint)
);
CREATE TABLE IF NOT EXISTS exports (
 id TEXT PRIMARY KEY,project_id TEXT NOT NULL,revision INTEGER NOT NULL,fingerprint TEXT NOT NULL,
 artifact_dir TEXT NOT NULL,manifest TEXT NOT NULL,created_at REAL NOT NULL,
 UNIQUE(project_id,fingerprint)
);
CREATE TABLE IF NOT EXISTS feedback (
 id TEXT PRIMARY KEY,project_id TEXT NOT NULL,candidate_id TEXT,decision TEXT NOT NULL,note TEXT NOT NULL,created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS presets (id TEXT PRIMARY KEY,name TEXT NOT NULL,timeline TEXT NOT NULL,created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS publications (
 id TEXT PRIMARY KEY,project_id TEXT NOT NULL,export_id TEXT NOT NULL,asset_fingerprint TEXT NOT NULL,
 revision INTEGER NOT NULL,draft TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'draft',approval TEXT,
 remote_id TEXT,operation_key TEXT UNIQUE,created_at REAL NOT NULL,updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
 seq INTEGER PRIMARY KEY AUTOINCREMENT,project_id TEXT,event TEXT NOT NULL,detail TEXT NOT NULL,created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
 id TEXT PRIMARY KEY,filename TEXT NOT NULL,size INTEGER NOT NULL,offset INTEGER NOT NULL DEFAULT 0,
 options TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'uploading',project_id TEXT,created_at REAL NOT NULL
);
'''

def connect():
    config.initialize()
    c=sqlite3.connect(config.DB,timeout=15,isolation_level=None)
    c.row_factory=sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON')
    c.execute('PRAGMA busy_timeout=15000')
    return c

def migrate():
    with contextlib.closing(connect()) as c:
        c.execute('PRAGMA journal_mode=WAL')
        version=c.execute('PRAGMA user_version').fetchone()[0]
        if version>1: raise RuntimeError('Database is newer than this application; do not downgrade')
        if version<1:
            c.executescript('BEGIN IMMEDIATE;'+MIGRATION_1+'PRAGMA user_version=1;COMMIT;')

@contextlib.contextmanager
def tx():
    with contextlib.closing(connect()) as c:
        c.execute('BEGIN IMMEDIATE')
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise

def one(sql,args=()):
    with contextlib.closing(connect()) as c:
        r=c.execute(sql,args).fetchone()
        return dict(r) if r else None

def all(sql,args=()):
    with contextlib.closing(connect()) as c:
        return [dict(r) for r in c.execute(sql,args).fetchall()]

def list_job_rows(project_id=None,limit=200):
    """Project OCR evidence out in SQLite before rows enter Python memory.

    Detail reads still select the stored result. Keep all other job kinds' result
    values intact because render/tracking consumers use them in project views.
    """
    columns='''id,project_id,kind,params,status,stage,idempotency_key,request_hash,
        attempts,max_attempts,lease_until,heartbeat_at,cancel_requested,error,
        created_at,updated_at,started_at,finished_at,
        CASE WHEN kind='ocr' THEN NULL ELSE result END AS result,
        CASE WHEN kind='ocr' AND result IS NOT NULL THEN 1 ELSE 0 END AS result_omitted'''
    where=' WHERE project_id=?' if project_id is not None else ''
    args=(project_id,) if project_id is not None else ()
    return all(f'SELECT {columns} FROM jobs{where} ORDER BY created_at DESC LIMIT ?',
               (*args,max(0,min(int(limit),200))))

def audit(c,event,detail=None,project_id=None):
    c.execute('INSERT INTO audit(project_id,event,detail,created_at) VALUES(?,?,?,?)',
              (project_id,event,dumps(detail or {}),time.time()))

def decode_job(row):
    if not row:return None
    r=dict(row)
    for k in ('params','result'):
        r[k]=json.loads(r[k]) if r.get(k) else None
    r['elapsed_seconds']=max(0,(r.get('finished_at') or time.time())-(r.get('started_at') or time.time()))
    r.pop('lease_token',None)
    return r

class Conflict(ValueError): pass
class Cancelled(RuntimeError): pass
class LeaseLost(RuntimeError): pass

def enqueue(kind,params,project_id=None,key=None,connection=None):
    now=time.time(); h=fingerprint({'kind':kind,'project':project_id,'params':params})
    if key and (len(key)>160 or len(key)<1): raise ValueError('Idempotency key must be 1–160 characters')
    with (contextlib.nullcontext(connection) if connection is not None else tx()) as c:
        if key:
            old=c.execute('SELECT * FROM jobs WHERE idempotency_key=?',(key,)).fetchone()
            if old:
                if old['request_hash']!=h: raise Conflict('Idempotency key already belongs to a different request')
                return decode_job(old)
        # Double-clicks with identical active inputs share the existing work.
        old=c.execute("SELECT * FROM jobs WHERE request_hash=? AND status IN ('queued','running')",(h,)).fetchone()
        if old:return decode_job(old)
        jid=uid()
        c.execute('INSERT INTO jobs(id,project_id,kind,params,idempotency_key,request_hash,created_at,updated_at,max_attempts) VALUES(?,?,?,?,?,?,?,?,?)',
                  (jid,project_id,kind,dumps(params),key,h,now,now,config.MAX_ATTEMPTS))
        audit(c,'job.queued',{'job_id':jid,'kind':kind},project_id)
        return decode_job(c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone())

def recover(c,now):
    stale=c.execute("SELECT * FROM jobs WHERE status='running' AND lease_until<?",(now,)).fetchall()
    for j in stale:
        status='cancelled' if j['cancel_requested'] else ('failed' if j['attempts']>=j['max_attempts'] else 'queued')
        message='Worker interrupted; completed stages will be reused.' if status=='queued' else 'Worker lease expired.'
        c.execute('UPDATE jobs SET status=?,lease_token=NULL,lease_until=NULL,error=?,updated_at=?,finished_at=? WHERE id=?',
                  (status,message,now,now if status!='queued' else None,j['id']))
        audit(c,'job.recovered',{'job_id':j['id'],'status':status},j['project_id'])

def claim():
    now=time.time()
    with tx() as c:
        recover(c,now)
        # Global cap holds even if a second worker is accidentally started.
        if c.execute("SELECT 1 FROM jobs WHERE status='running' AND lease_until>=?",(now,)).fetchone():return None
        j=c.execute("SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 AND attempts<max_attempts ORDER BY created_at LIMIT 1").fetchone()
        if not j:return None
        token=uid()
        c.execute("UPDATE jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,heartbeat_at=?,updated_at=?,started_at=COALESCE(started_at,?),finished_at=NULL,error=NULL WHERE id=?",
                  (token,now+config.LEASE_SECONDS,now,now,now,j['id']))
        if j['kind'] in ('import','import-url'):
            c.execute("UPDATE projects SET status='importing',updated_at=? WHERE id=?",(now,j['project_id']))
        result=dict(c.execute('SELECT * FROM jobs WHERE id=?',(j['id'],)).fetchone())
        audit(c,'job.claimed',{'job_id':j['id'],'attempt':result['attempts']},j['project_id'])
        return result

def check_lease(job_id,token,c=None):
    row=(c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone() if c else one('SELECT * FROM jobs WHERE id=?',(job_id,)))
    if not row or row['lease_token']!=token or row['status']!='running' or row['lease_until']<time.time():
        raise LeaseLost('Worker no longer owns this job')
    if row['cancel_requested']:raise Cancelled('Cancelled by operator')
    return row

def heartbeat(job_id,token):
    now=time.time()
    with tx() as c:
        check_lease(job_id,token,c)
        c.execute('UPDATE jobs SET heartbeat_at=?,lease_until=?,updated_at=? WHERE id=? AND lease_token=?',
                  (now,now+config.LEASE_SECONDS,now,job_id,token))

def stage(job_id,token,name):
    with tx() as c:
        check_lease(job_id,token,c)
        c.execute('UPDATE jobs SET stage=?,updated_at=? WHERE id=?',(name,time.time(),job_id))

def finish(job_id,token,status,result=None,error=None):
    if status not in ('succeeded','failed','cancelled'):raise ValueError('Invalid terminal job state')
    with tx() as c:
        j=c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not j or j['lease_token']!=token or j['status']!='running' or j['lease_until']<time.time():return False
        if j['cancel_requested']:status='cancelled'; result=None
        now=time.time()
        c.execute('UPDATE jobs SET status=?,stage=?,result=?,error=?,updated_at=?,finished_at=?,lease_token=NULL,lease_until=NULL WHERE id=?',
                  (status,status,dumps(result) if result is not None else None,error,now,now,job_id))
        audit(c,'job.'+status,{'job_id':job_id,'error':error},j['project_id'])
        if status in ('failed','cancelled') and j['kind'] in ('import','import-url'):
            c.execute('UPDATE projects SET status=?,updated_at=? WHERE id=?',(status,now,j['project_id']))
        return True

def cancel(job_id):
    with tx() as c:
        j=c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not j:raise KeyError('Job not found')
        if j['status'] in ('queued','running'):
            now=time.time()
            c.execute("UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=?,finished_at=CASE WHEN status='queued' THEN ? ELSE finished_at END WHERE id=?",(now,now,job_id))
            if j['status']=='queued' and j['kind'] in ('import','import-url'):
                c.execute("UPDATE projects SET status='cancelled',updated_at=? WHERE id=?",(now,j['project_id']))
            audit(c,'job.cancel_requested',{'job_id':job_id},j['project_id'])
    return decode_job(one('SELECT * FROM jobs WHERE id=?',(job_id,)))

def retry(job_id):
    with tx() as c:
        j=c.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not j:raise KeyError('Job not found')
        if j['status'] not in ('failed','cancelled'):raise Conflict('Only failed or cancelled jobs can be retried')
        if j['attempts']>=j['max_attempts']:raise Conflict('Retry limit reached; inspect the failure before submitting a new job')
        c.execute("UPDATE jobs SET status='queued',stage='queued',cancel_requested=0,finished_at=NULL,error=NULL,updated_at=? WHERE id=?",(time.time(),job_id))
        audit(c,'job.retry',{'job_id':job_id},j['project_id'])
    return decode_job(one('SELECT * FROM jobs WHERE id=?',(job_id,)))
