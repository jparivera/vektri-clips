import asyncio
import json
import os
import time
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated,Literal
from urllib.parse import urlsplit
from fastapi import FastAPI,HTTPException,Request,UploadFile,File,Form,Header,Body,Query
from fastapi.responses import FileResponse,JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ValidationError,field_validator
from . import config,db,service
from .schemas import (TimelineUpdate,ProjectPatch,AnalysisOptions,AgentDecisions,PublicationDraft,Timeline,Word,Strict,TranscriptionOptions,OCROptions)
from .security import authorized_artifact,clean_name,free_space,ImportRejected
from .limits import RequestBodyLimitMiddleware
from .operational_api import CloudPreflightScenario, cloud_preflight_for_project, publication_preflight

@asynccontextmanager
async def lifespan(app):
    db.migrate();yield

app=FastAPI(title='Vektri Clip Studio',version='1.0.0',description='Local single-operator media API. Accepted jobs are asynchronous; poll their status. No publishing adapter is configured.',lifespan=lifespan,docs_url=None,redoc_url=None)
app.add_middleware(RequestBodyLimitMiddleware)

@app.middleware('http')
async def local_boundary(request:Request,call_next):
    host=request.headers.get('host','').lower()
    allowed={f'127.0.0.1:{config.PORT}',f'localhost:{config.PORT}','127.0.0.1','localhost','testserver'}
    if host not in allowed:return JSONResponse({'detail':'Only the configured loopback host is allowed'},400)
    origin=request.headers.get('origin')
    if origin and origin not in {f'http://127.0.0.1:{config.PORT}',f'http://localhost:{config.PORT}'}:
        return JSONResponse({'detail':'Cross-origin access is disabled'},403)
    if request.headers.get('sec-fetch-site')=='cross-site':return JSONResponse({'detail':'Cross-site access is disabled'},403)
    size=request.headers.get('content-length')
    if size:
        try:too_large=int(size)>config.MAX_UPLOAD+1024**2 or int(size)<0
        except ValueError:too_large=True
        if too_large:return JSONResponse({'detail':'Request exceeds the local upload limit'},413)
        if request.headers.get('content-type','').startswith('application/json') and int(size)>8*1024**2:
            return JSONResponse({'detail':'JSON request exceeds 8 MiB; use media upload for files'},413)
    try:response=await call_next(request)
    except Exception:raise
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers['Cache-Control']='no-store'
    return response

@app.exception_handler(db.Conflict)
async def conflict_handler(request,exc):return JSONResponse({'detail':str(exc)},409)
@app.exception_handler(KeyError)
async def missing_handler(request,exc):return JSONResponse({'detail':str(exc).strip("'")},404)
@app.exception_handler(ValueError)
async def value_handler(request,exc):return JSONResponse({'detail':str(exc)[:1000]},422)
@app.exception_handler(OSError)
async def disk_handler(request,exc):return JSONResponse({'detail':str(exc)[:1000]},507 if getattr(exc,'errno',None)==28 or 'disk space' in str(exc) else 400)

prefix='/api/v1'
Key=Annotated[str|None,Header(alias='Idempotency-Key')]

@app.get(prefix+'/health')
def health():
    from .runtime import worker_status
    return {'status':'ok','api_version':1,'database_version':1,'data_directory':str(config.DATA),
            'worker_active':bool(db.one("SELECT 1 FROM jobs WHERE status='running' AND lease_until>?",(time.time(),))),
            'worker':worker_status(),
            'queue_length':db.one("SELECT count(*) n FROM jobs WHERE status='queued'")['n']}

@app.get(prefix+'/capabilities')
def capabilities():
    from .analysis import model_status
    return {'api_version':1,'local_only':True,'models':model_status(),
            'modes':{'local':'available','heuristic':'available','agent':'offline schema exchange','cloud':'blocked: configure a reviewed provider and explicit budget; no paid provider is wired'},
            'publishing':{'manual':True,'youtube':False,'instagram':False,'tiktok':False},
            'imports':{'file':True,'direct_public_url':True,'platform_downloads':False,'max_upload_bytes':config.MAX_UPLOAD,'max_url_bytes':config.MAX_URL},
            'limits':{'heavy_workers':1,'cpu_threads':2,'render_batch':20},
            'diarization':'unavailable','active_speaker_association':'unavailable','ocr':model_status()['ocr'],
            'rights_notice':'Recording permission status does not establish legal rights. Import, private editing, export and publishing are separate.'}

@app.get(prefix+'/projects')
def projects(q:str='',limit:int=100,offset:int=0):return service.list_projects(q,limit,offset)

@app.get(prefix+'/maintenance/storage-plan')
def storage_plan():
    """Bounded advisory storage report; never deletes or moves library files."""
    from .retention import plan
    return plan(config.DATA)

@app.get(prefix+'/maintenance/storage')
def storage_summary():
    """Read-only size report for the library; creates nothing and follows no symlinks."""
    return service.storage_summary()

@app.post(prefix+'/maintenance/verify-plan')
def verify_storage_plan(payload:dict=Body(...)):
    """Read-only refresh against this configured library, not an apply endpoint."""
    from .retention import verify_plan
    return verify_plan(payload, config.DATA)

@app.get(prefix+'/projects/{pid}')
def project(pid:str):return service.get_project(pid)
@app.post(prefix+'/projects/{pid}/cloud-preflight')
def cloud_preflight(pid:str,payload:CloudPreflightScenario):
    return cloud_preflight_for_project(pid,payload)
@app.patch(prefix+'/projects/{pid}')
def patch_project(pid:str,payload:ProjectPatch):return service.patch_project(pid,payload.model_dump(exclude_none=True))
@app.put(prefix+'/projects/{pid}/timeline')
def update_timeline(pid:str,payload:TimelineUpdate):return service.update_timeline(pid,payload.revision,payload.timeline.model_dump())
@app.get(prefix+'/projects/{pid}/revisions')
def revisions(pid:str):
    service.project_row(pid)
    return [{**r,'timeline':json.loads(r['timeline'])} for r in db.all('SELECT * FROM revisions WHERE project_id=? ORDER BY revision DESC LIMIT 200',(pid,))]

@app.post(prefix+'/imports',status_code=202)
async def upload(file:Annotated[UploadFile,File()],title:Annotated[str,Form()]='',genre:Annotated[str,Form()]='speech',attribution:Annotated[str,Form()]='',permission_status:Annotated[str,Form()]='unknown',key:Key=None):
    if file.size is not None and file.size>config.MAX_UPLOAD:raise HTTPException(413,'File exceeds local import size limit')
    config.initialize()
    temporary=config.DATA/'uploads'/('http-'+db.uid())
    try:
        count=0;digest=hashlib.sha256()
        with temporary.open('xb') as f:
            while chunk:=await file.read(1024*1024):
                count+=len(chunk)
                if count>config.MAX_UPLOAD:raise HTTPException(413,'Import size limit exceeded')
                free_space(temporary.parent,len(chunk));f.write(chunk);digest.update(chunk)
            f.flush();os.fsync(f.fileno())
        if count==0:raise ValueError('The selected file is empty')
        return service.accept_staged(temporary,file.filename or 'media',title,{'genre':genre,'attribution':attribution,'permission_status':permission_status},key,content_identity='sha256:'+digest.hexdigest())
    finally:
        temporary.unlink(missing_ok=True)
        await file.close()

class URLImport(Strict):
    url:str=Field(max_length=2048)
    title:str=Field(default='',max_length=200)
    genre:Literal['speech','tutorial','action','mixed']='speech'
    attribution:str=Field(default='',max_length=2000)
    permission_status:Literal['unknown','permission-recorded','owned','licensed']='unknown'

@app.post(prefix+'/imports/url',status_code=202)
def url_import(payload:URLImport,key:Key=None):return service.import_url(payload.url,payload.title,payload.model_dump(),key)

def fixture_catalog():
    root=config.ROOT/'fixtures'
    return [{'name':p.name,'title':p.stem.replace('_',' ').title(),'size':p.stat().st_size,'synthetic':True}
            for p in sorted(root.glob('*.mp4')) if p.name!='corrupt-input.mp4']
@app.get(prefix+'/fixtures')
def fixtures():return fixture_catalog()
@app.post(prefix+'/fixtures/{name}',status_code=202)
def fixture_import(name:str,key:Key=None):
    if name not in {f['name'] for f in fixture_catalog()}:raise KeyError('Fixture not found; run scripts/generate_fixtures.py first')
    genre='tutorial' if 'tutorial' in name else ('action' if 'action' in name else ('mixed' if 'mixed' in name or 'portrait' in name else 'speech'))
    return service.import_path(config.ROOT/'fixtures'/name,'Synthetic · '+Path(name).stem.replace('_',' '),{'genre':genre,'permission_status':'synthetic','attribution':'Original deterministic Vektri test fixture; Flite synthesized speech where present. Not real-world genre evidence.','synthetic':True},key)

class UploadStart(Strict):
    filename:str=Field(max_length=180)
    size:int=Field(gt=0,le=config.MAX_UPLOAD)
    title:str=Field(default='',max_length=200)
    genre:Literal['speech','tutorial','action','mixed']='speech'
    attribution:str=Field(default='',max_length=2000)
    permission_status:Literal['unknown','permission-recorded','owned','licensed']='unknown'
@app.post(prefix+'/uploads',status_code=201)
def start_upload(payload:UploadStart):
    free_space(config.DATA,payload.size);uid=db.uid()
    (config.DATA/'uploads'/uid).touch(mode=0o600)
    with db.tx() as c:c.execute('INSERT INTO uploads(id,filename,size,options,created_at) VALUES(?,?,?,?,?)',(uid,clean_name(payload.filename),payload.size,db.dumps(payload.model_dump()),time.time()))
    return {'id':uid,'offset':0,'size':payload.size,'chunk_limit':8*1024**2}
@app.get(prefix+'/uploads/{uid}')
def upload_status(uid:str):
    row=db.one('SELECT id,filename,size,offset,status,project_id FROM uploads WHERE id=?',(uid,))
    if not row:raise KeyError('Upload not found')
    return row
@app.patch(prefix+'/uploads/{uid}')
async def chunk_upload(uid:str,request:Request,upload_offset:Annotated[int,Header(alias='Upload-Offset')]):
    body=bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body)>8*1024**2:raise HTTPException(413,'Maximum chunk is 8 MiB')
    if not body:raise ValueError('Empty upload chunk')
    with db.tx() as c:
        r=c.execute('SELECT * FROM uploads WHERE id=?',(uid,)).fetchone()
        if not r:raise KeyError('Upload not found')
        if r['status']!='uploading':raise db.Conflict('Upload is no longer writable')
        if upload_offset!=r['offset']:raise db.Conflict(f'Expected upload offset {r["offset"]}')
        if upload_offset+len(body)>r['size']:raise ValueError('Chunk exceeds declared file length')
        path=config.DATA/'uploads'/uid;free_space(path.parent,len(body))
        with path.open('r+b') as f:
            f.seek(upload_offset);f.write(body);f.truncate();f.flush();os.fsync(f.fileno())
        c.execute('UPDATE uploads SET offset=? WHERE id=?',(upload_offset+len(body),uid))
    return upload_status(uid)
@app.post(prefix+'/uploads/{uid}/complete',status_code=202)
def complete_upload(uid:str):
    # Serialize reservation; repeated complete returns the existing import job.
    with db.tx() as c:
        r=c.execute('SELECT * FROM uploads WHERE id=?',(uid,)).fetchone()
        if not r:raise KeyError('Upload not found')
        r=dict(r)
        if r['status']=='complete':
            j=c.execute('SELECT * FROM jobs WHERE idempotency_key=?',('upload:'+uid,)).fetchone()
            return {'project_id':r['project_id'],'job':db.decode_job(j)}
        if r['offset']!=r['size']:raise db.Conflict('Upload is incomplete; resume from the stored offset')
        opts=json.loads(r['options']);staged=config.DATA/'uploads'/uid
        if not staged.is_file() or staged.stat().st_size!=r['size']:raise ValueError('Upload bytes do not match the stored size; resume or start a fresh upload')
        result=service.accept_staged(staged,r['filename'],opts['title'] or r['filename'],opts,'upload:'+uid,connection=c,content_identity='resumable:'+uid)
        c.execute("UPDATE uploads SET status='complete',project_id=? WHERE id=?",(result['project_id'],uid))
        return result

def ready_project(pid):
    p=service.project_row(pid)
    if p['status']!='ready':raise db.Conflict('Wait for source preparation to finish')
    return p
@app.post(prefix+'/projects/{pid}/analyze',status_code=202)
def analyze(pid:str,payload:AnalysisOptions,key:Key=None):
    p=ready_project(pid)
    if payload.mode=='cloud':raise HTTPException(409,'Cloud analysis is not configured. No data was sent and no money was spent. Use local mode or an evidence packet.')
    if payload.mode=='agent':raise HTTPException(409,'Agent mode uses GET /evidence and POST /agent-decisions. No external-agent connection is configured.')
    options=payload.model_dump()
    if payload.use_feedback:
        latest={}
        for row in db.all("SELECT detail FROM audit WHERE project_id=? AND event='feedback.recorded' ORDER BY seq DESC LIMIT 1000",(pid,)):
            detail=json.loads(row['detail']);cid=detail.get('candidate_id')
            if cid not in latest:latest[cid]=detail
        current=json.loads(p['analysis']) if p['analysis'] else {}
        current_by_id={c['id']:c for c in current.get('candidates',[])}
        rejected=[]
        for cid,f in latest.items():
            candidate=f.get('candidate_snapshot') or current_by_id.get(cid)
            if f['decision']=='reject' and candidate:rejected.append({'start':candidate['start'],'end':candidate['end']})
        applied=rejected[:max(0,100-len(options['exclusions']))]
        options['exclusions']=options['exclusions']+applied
        options['feedback_policy']={'applied_rejected_intervals':applied,'count':len(applied),'omitted_due_to_limit':len(rejected)-len(applied),'effect':'Operator opted in to exclude previously rejected intervals; no model training.'}
    return db.enqueue('analyze',{**options,'base_transcript_hash':db.fingerprint(json.loads(p['transcript'])) if p['transcript'] else None},pid,key)

@app.post(prefix+'/projects/{pid}/ocr',status_code=202)
def queue_ocr(pid:str,payload:OCROptions=Body(default=OCROptions()),key:Key=None):
    ready_project(pid)
    return db.enqueue('ocr',payload.model_dump(),pid,key)

@app.get(prefix+'/projects/{pid}/ocr')
def latest_ocr(pid:str):
    return service.ocr_result(pid)

class TranscribeRequest(TranscriptionOptions):
    pass
@app.post(prefix+'/projects/{pid}/transcribe',status_code=202)
def transcribe(pid:str,payload:TranscribeRequest=Body(default=TranscribeRequest()),key:Key=None):
    p=ready_project(pid)
    return db.enqueue('transcribe',{**payload.model_dump(),'base_transcript_hash':db.fingerprint(json.loads(p['transcript'])) if p['transcript'] else None},pid,key)

class TranscriptSegment(Strict):
    start:float=Field(ge=0,le=86400)
    end:float=Field(gt=0,le=86400)
    text:str=Field(min_length=1,max_length=10000)
    id:str|None=Field(default=None,max_length=100)
    avg_logprob:float|None=None
    no_speech_probability:float|None=Field(default=None,ge=0,le=1)
    @field_validator('text')
    @classmethod
    def valid_text(cls,v):
        if not v.strip() or '\x00' in v:raise ValueError('Transcript text must be nonblank')
        return v
class TranscriptUpdate(Strict):
    words:list[Word]=Field(max_length=100000)
    segments:list[TranscriptSegment]=Field(default_factory=list,max_length=20000)
    revision:int=Field(ge=1)
@app.put(prefix+'/projects/{pid}/transcript')
def transcript_update(pid:str,payload:TranscriptUpdate):
    p=service.get_project(pid)
    if p['revision']!=payload.revision:raise db.Conflict('Project changed; reload before correcting captions')
    timeline=p['timeline'];timeline['captions']['words']=[w.model_dump() for w in payload.words]
    Timeline.model_validate(timeline).validate_source(p['source']['metadata']['duration'])
    if any(s.end<=s.start or s.end>p['source']['metadata']['duration']+.001 for s in payload.segments):raise ValueError('Transcript segment times must be ordered and within source duration')
    if any(a.end>b.start+.001 for a,b in zip(payload.segments,payload.segments[1:])):raise ValueError('Transcript segments must be ordered and non-overlapping')
    segments=[s.model_dump(exclude_none=True) for s in payload.segments]
    previous=p['transcript'] or {}
    canonical=lambda items:[(s.get('start'),s.get('end'),s.get('text','').strip()) for s in items]
    segment_changed=bool(segments) and canonical(segments)!=canonical(previous.get('segments',[]))
    word_text_changed=[w.word for w in payload.words]!=[w.get('word') for w in previous.get('words',[])]
    # Segment corrections ground semantic retrieval without inventing word alignment.
    # Timing-only word edits must not restore stale ASR text as semantic evidence.
    use_segments=segment_changed or (bool(previous.get('segment_text_corrected')) and not word_text_changed and bool(segments))
    data={**(p['transcript'] or {}),'words':timeline['captions']['words'],'segments':segments or [{'start':w.start,'end':w.end,'text':w.word} for w in payload.words],
          'corrected':True,'segment_text_corrected':use_segments,'correction_time':time.time(),'alignment':'user-corrected'}
    # Revision update and transcript must be one transaction (service method nests no transactions here).
    with db.tx() as c:
        current=c.execute('SELECT revision FROM projects WHERE id=?',(pid,)).fetchone()
        if current['revision']!=payload.revision:raise db.Conflict('Project changed; reload before correcting captions')
        rev=payload.revision+1;now=time.time();h=db.fingerprint(timeline)
        c.execute('INSERT INTO revisions VALUES(?,?,?,?,?)',(pid,rev,db.dumps(timeline),h,now))
        c.execute('UPDATE projects SET transcript=?,analysis=NULL,revision=?,updated_at=? WHERE id=?',(db.dumps(data),rev,now,pid))
        c.execute("UPDATE publications SET status='approval-invalidated',approval=NULL,updated_at=? WHERE project_id=? AND approval IS NOT NULL",(now,pid))
        db.audit(c,'transcript.corrected',{'word_count':len(payload.words),'revision':rev},pid)
    return {'transcript':data,'revision':rev,'timeline':timeline}

class RenderRequest(Strict):revision:int|None=Field(default=None,ge=1)
@app.post(prefix+'/projects/{pid}/render',status_code=202)
def render(pid:str,payload:RenderRequest=Body(default=RenderRequest()),key:Key=None):return service.queue_render(pid,payload.revision,key)

class TrackRequest(Strict):
    mode:Literal['face','object']='face'
    start:float=Field(default=0,ge=0)
    end:float|None=Field(default=None,gt=0)
    x:float=Field(default=.3,ge=0,le=1)
    y:float=Field(default=.3,ge=0,le=1)
    width:float=Field(default=.2,gt=0,le=1)
    height:float=Field(default=.3,gt=0,le=1)
@app.post(prefix+'/projects/{pid}/track',status_code=202)
def track(pid:str,payload:TrackRequest,key:Key=None):
    p=ready_project(pid);d=json.loads(p['source'])['metadata']['duration']
    opts=payload.model_dump();opts['end']=opts['end'] or d
    if opts['end']>d or opts['end']<=opts['start'] or payload.x+payload.width>1 or payload.y+payload.height>1:raise ValueError('Tracking bounds are outside the source')
    return db.enqueue('track',opts,pid,key)

class Feedback(Strict):
    candidate_id:str=Field(max_length=80)
    decision:Literal['accept','reject']
    note:str=Field(default='',max_length=2000)
@app.post(prefix+'/projects/{pid}/feedback')
def feedback(pid:str,payload:Feedback):
    p=service.project_row(pid)
    analysis=json.loads(p['analysis']) if p['analysis'] else {}
    candidate=next((c for c in analysis.get('candidates',[]) if c['id']==payload.candidate_id),None)
    with db.tx() as c:
        c.execute('INSERT INTO feedback VALUES(?,?,?,?,?,?)',(db.uid(),pid,payload.candidate_id,payload.decision,payload.note,time.time()))
        db.audit(c,'feedback.recorded',{**payload.model_dump(),'candidate_snapshot':candidate},pid)
    return {'saved':True,'effect':'Recorded for editorial review; no model training or hidden preference change.'}

@app.get(prefix+'/projects/{pid}/candidates')
def candidates(pid:str):
    p=service.project_row(pid)
    return (json.loads(p['analysis']) if p['analysis'] else {}).get('candidates',[])

class Event(Strict):
    time:float=Field(ge=0)
    before:float=Field(default=3,ge=0,le=300)
    after:float=Field(default=5,ge=0,le=300)
    title:str=Field(default='Marked moment',min_length=1,max_length=200)
@app.post(prefix+'/projects/{pid}/event')
def event(pid:str,payload:Event):
    p=service.get_project(pid);duration=p['source']['metadata']['duration']
    if payload.time>duration:raise ValueError('Event is outside source')
    a=max(0,payload.time-payload.before);b=min(duration,payload.time+payload.after)
    if b-a<.04:raise ValueError('Event needs a nonempty context window')
    candidate={'id':db.uid(),'start':a,'end':b,'title':payload.title,'rationale':'Operator marked an event and chose the context window.','evidence':[{'time':payload.time,'type':'manual-event'}],
               'confidence':1,'uncertainty':'Human label; not an AI event claim.','scores':{},'method':'manual'}
    analysis=p['analysis'] or {'candidates':[],'method':'manual'};analysis['candidates'].append(candidate)
    with db.tx() as c:
        c.execute('UPDATE projects SET analysis=?,updated_at=? WHERE id=?',(db.dumps(analysis),time.time(),pid));db.audit(c,'event.marked',candidate,pid)
    return candidate

class SilenceRequest(Strict):
    start:float=Field(ge=0)
    end:float=Field(gt=0)
    gap:float=Field(default=1.0,ge=.3,le=10)
    fillers:bool=False
@app.post(prefix+'/projects/{pid}/silence-plan')
def silence_plan(pid:str,payload:SilenceRequest):
    from .media import silence_plan as plan
    p=service.get_project(pid)
    if not p['transcript'] or not p['transcript'].get('words'):raise db.Conflict('Word timing is required; transcribe first or edit the timeline manually')
    if payload.end<=payload.start or payload.end>p['source']['metadata']['duration']:raise ValueError('Invalid source interval')
    return plan(p['transcript']['words'],payload.start,payload.end,payload.gap,payload.fillers)

@app.get(prefix+'/projects/{pid}/evidence')
def evidence(pid:str):return service.evidence_packet(pid)
@app.post(prefix+'/projects/{pid}/agent-decisions')
def agent_decisions(pid:str,payload:AgentDecisions):return service.apply_agent_decisions(pid,payload.model_dump())

@app.get(prefix+'/jobs')
def jobs():return service.list_jobs()
@app.get(prefix+'/jobs/{jid}')
def job(jid:str):
    j=db.decode_job(db.one('SELECT * FROM jobs WHERE id=?',(jid,)))
    if not j:raise KeyError('Job not found')
    return j
@app.post(prefix+'/jobs/{jid}/cancel')
def cancel(jid:str):return db.cancel(jid)
@app.post(prefix+'/jobs/{jid}/retry')
def retry(jid:str):return db.retry(jid)

@app.post(prefix+'/models/{name}/download',status_code=202)
def download_model(name:str,key:Key=None):
    from .analysis import model_status
    if name not in {m['id'] for m in model_status()['models']}:raise ValueError('Unknown model')
    return db.enqueue('model-download',{'name':name},None,key)

@app.get(prefix+'/artifacts/{pid}/{path:path}')
def artifact(pid:str,path:str):
    project=service.project_row(pid)
    try:p=authorized_artifact(pid,path)
    except FileNotFoundError:raise HTTPException(404,'Artifact not found') from None
    if p.name=='original.bin':
        original=json.loads(project['source']).get('original_name','original-media')
        return FileResponse(p,media_type='application/octet-stream',filename=clean_name(original),content_disposition_type='attachment')
    # MP4 range responses are provided by Starlette FileResponse.
    return FileResponse(p,filename=p.name if p.suffix in ('.zip','.srt','.vtt') else None,content_disposition_type='attachment' if p.suffix in ('.zip','.srt','.vtt') else 'inline')

@app.get(prefix+'/exports')
def exports():return [service.export_view(e) for e in db.all('SELECT * FROM exports ORDER BY created_at DESC LIMIT 200')]
@app.get(prefix+'/exports/{eid}/download')
def export_download(eid:str):
    e=db.one('SELECT * FROM exports WHERE id=?',(eid,))
    if not e:raise KeyError('Export not found')
    return FileResponse(authorized_artifact(e['project_id'],e['artifact_dir']+'/package.zip'),filename='clip-'+eid[:8]+'.zip')

class Preset(Strict):
    name:str=Field(min_length=1,max_length=100)
    timeline:Timeline
@app.get(prefix+'/presets')
def presets():return [{**p,'timeline':json.loads(p['timeline'])} for p in db.all('SELECT * FROM presets ORDER BY created_at DESC')]
@app.post(prefix+'/presets')
def preset(payload:Preset):
    pid=db.uid()
    # Presets retain styling/layout only; never silently import another source's captions/timing.
    t=payload.timeline.model_dump();t['captions']['words']=[]
    with db.tx() as c:
        c.execute('INSERT INTO presets VALUES(?,?,?,?)',(pid,payload.name,db.dumps(t),time.time()));db.audit(c,'preset.saved',{'id':pid,'name':payload.name})
    return {'id':pid,'name':payload.name,'timeline':t}

class BatchItem(Strict):project_id:str;revision:int|None=None
class Batch(Strict):
    items:list[BatchItem]=Field(min_length=1,max_length=20)
    idempotency_key:str=Field(min_length=1,max_length=100)
@app.post(prefix+'/batch/render',status_code=202)
def batch(payload:Batch):
    result=[]
    for i,item in enumerate(payload.items):
        try:result.append({'project_id':item.project_id,'job':service.queue_render(item.project_id,item.revision,payload.idempotency_key+':'+str(i))})
        except (ValueError,KeyError) as e:result.append({'project_id':item.project_id,'error':str(e)})
    return {'items':result}

@app.get(prefix+'/publications')
def publications():return [service.publication_view(p) for p in db.all('SELECT * FROM publications ORDER BY created_at DESC LIMIT 200')]
@app.get(prefix+'/publications/{pid}/preflight')
def publication_readiness(pid:str):return publication_preflight(pid)
@app.post(prefix+'/publications')
def publication(payload:PublicationDraft):return service.create_publication(payload.model_dump())
@app.put(prefix+'/publications/{pid}')
def update_publication(pid:str,payload:PublicationDraft):
    with db.tx() as c:
        p=c.execute('SELECT * FROM publications WHERE id=?',(pid,)).fetchone()
        if not p:raise KeyError('Publication draft not found')
        if p['project_id']!=payload.project_id or p['export_id']!=payload.export_id:raise ValueError('Create a new draft to change its project or asset')
        c.execute("UPDATE publications SET draft=?,status='draft',approval=NULL,updated_at=? WHERE id=?",(db.dumps(payload.model_dump()),time.time(),pid))
        db.audit(c,'publication.edited',{'id':pid,'approval_invalidated':True},p['project_id'])
    return service.publication_view(db.one('SELECT * FROM publications WHERE id=?',(pid,)))

class Approval(Strict):
    draft:PublicationDraft
    asset_fingerprint:str=Field(pattern=r'^[a-f0-9]{64}$')
    rights_acknowledgment:str=Field(min_length=1,max_length=1000)
    confirm_manual_handoff:Literal[True]
    @field_validator('rights_acknowledgment')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Record a meaningful rights acknowledgment before manual handoff')
        return value.strip()
@app.post(prefix+'/publications/{pid}/approve')
def approve_publication(pid:str,payload:Approval):
    with db.tx() as c:
        p=c.execute('SELECT * FROM publications WHERE id=?',(pid,)).fetchone()
        if not p:raise KeyError('Publication draft not found')
        project=c.execute('SELECT revision FROM projects WHERE id=?',(p['project_id'],)).fetchone()
        if project['revision']!=p['revision']:raise db.Conflict('Edits changed after this export. Render the current revision and prepare a new draft.')
        if payload.asset_fingerprint!=p['asset_fingerprint'] or db.dumps(payload.draft.model_dump())!=p['draft']:
            raise db.Conflict('The draft or asset changed. Review the exact current destination, account, caption, visibility, schedule and time zone.')
        approval={**payload.model_dump(),'confirmed_at':time.time(),'scope':'manual handoff only; no remote upload','rights_notice':'Acknowledgment records operator review and does not establish legal rights.'}
        c.execute("UPDATE publications SET approval=?,status='ready-for-manual-upload',updated_at=? WHERE id=?",(db.dumps(approval),time.time(),pid))
        db.audit(c,'publication.manual_approval',{'id':pid,'asset_fingerprint':p['asset_fingerprint']},p['project_id'])
    return service.publication_view(db.one('SELECT * FROM publications WHERE id=?',(pid,)))
@app.post(prefix+'/publications/{pid}/publish')
def publish(pid:str):raise HTTPException(409,'Posting is unavailable: no authorized account or supported publishing adapter is connected. Download the export package for manual upload.')

@app.get(prefix+'/audit')
def audit(project_id:str|None=None,event:str|None=Query(default=None,max_length=80),limit:int=Query(default=300,ge=1,le=1000),before:int|None=Query(default=None,ge=1)):
    """Append-only history, newest first. `event` matches a dotted prefix such as `job.`; `before` pages by sequence number."""
    where=[];args=[]
    if project_id:where.append('project_id=?');args.append(project_id)
    if event:where.append("event LIKE ? ESCAPE '\\'");args.append(event.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%')
    if before is not None:where.append('seq<?');args.append(before)
    rows=db.all('SELECT * FROM audit '+('WHERE '+' AND '.join(where)+' ' if where else '')+'ORDER BY seq DESC LIMIT ?',(*args,limit))
    return [{**r,'detail':json.loads(r['detail'])} for r in rows]

@app.get('/docs',include_in_schema=False,response_class=HTMLResponse)
def api_reference():
    # Offline documentation: no CDN script/font requests and no permissive CSP exception.
    from html import escape
    spec=app.openapi();rows=[]
    for path,methods in spec['paths'].items():
        for method,operation in methods.items():
            rows.append('<details><summary><b>'+escape(method.upper())+'</b> '+escape(path)+' — '+escape(operation.get('summary',''))+'</summary><pre>'+escape(json.dumps(operation,indent=2))+'</pre></details>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Vektri · Local API v1</title><style>body{background:#101416;color:#dce5e8;font:15px system-ui;margin:40px auto;max-width:1050px;padding:0 20px}a,b{color:#d3ec78}details{border-bottom:1px solid #334047;padding:16px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}h1{font-size:32px}</style><h1>Vektri Clip Studio · Local API v1</h1><p>Single operator, loopback only. Accepted jobs run asynchronously; poll their status. No platform posting or paid provider is connected.</p><p><a href="/">Back to studio</a> · <a href="/openapi.json">Download OpenAPI JSON</a></p>'+''.join(rows)+'<details><summary>JSON schemas</summary><pre>'+escape(json.dumps(spec.get('components',{}),indent=2))+'</pre></details></html>'

frontend=config.ROOT/'frontend'/'dist'
if (frontend/'assets').is_dir():app.mount('/assets',StaticFiles(directory=frontend/'assets'),name='assets')
@app.get('/{path:path}',include_in_schema=False)
def frontend_page(path:str):
    if path.startswith(('api/','assets/')):raise HTTPException(404,'Route not found')
    index=frontend/'index.html'
    if index.is_file():return FileResponse(index)
    return JSONResponse({'message':'Build the frontend with cd frontend && npm ci && npm run build. API is available at /docs.'})
