"""Versioned structured CLI; local imports use the same service as the API."""
import argparse
import json
import os
import signal
import subprocess
import socket
import threading
import time
import sys
import urllib.request
import urllib.error
from pathlib import Path
from . import config,db,service,runtime

def request(method,path,payload=None,key=None):
    data=json.dumps(payload,allow_nan=False).encode() if payload is not None else None
    headers={'Content-Type':'application/json'}
    if key:headers['Idempotency-Key']=key
    req=urllib.request.Request(f'http://127.0.0.1:{config.PORT}/api/v1'+path,data=data,headers=headers,method=method)
    try:
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    except urllib.error.HTTPError as e:
        detail=json.load(e).get('detail',str(e));raise ValueError(detail) from None
    except urllib.error.URLError:raise ValueError('Local API is not running. Start with ./scripts/start.sh or clipstudio serve.') from None

def parser():
    p=argparse.ArgumentParser(prog='clipstudio',description='Vektri Clip Studio v1 local API/CLI. JSON output. No accounts or publishing configured.')
    s=p.add_subparsers(dest='command',required=True)
    s.add_parser('serve',help='Run loopback API and one durable worker; Ctrl-C stops both')
    s.add_parser('worker',help='Run only the durable worker')
    mcp=s.add_parser('mcp',help='Local MCP evidence bridge; no external agent is connected')
    mcp.add_argument('--allow-proposals',action='store_true',help='Permit validated candidate proposals only; never render or publish')
    s.add_parser('capabilities',help='Inspect local models and real integration status')
    s.add_parser('projects',help='List searchable projects').add_argument('--query',default='')
    s.add_parser('project',help='Inspect a project').add_argument('id')
    a=s.add_parser('cloud-preflight',help='Simulate a declared cloud cost scenario; never sends media or reserves/spends money')
    a.add_argument('id');a.add_argument('file',help='Explicit hypothetical price/budget/transfer scenario JSON')
    s.add_parser('publication-preflight',help='Read current export/manual approval checks; never posts').add_argument('id')
    i=s.add_parser('import',help='Copy an intentionally supplied local file; original remains untouched')
    i.add_argument('path');i.add_argument('--title');i.add_argument('--genre',choices=['speech','tutorial','action','mixed'],default='speech');i.add_argument('--attribution',default='');i.add_argument('--permission-status',choices=['unknown','permission-recorded','owned','licensed'],default='unknown');i.add_argument('--key')
    i=s.add_parser('import-url',help='Queue a bounded public direct media download; platform pages unsupported')
    i.add_argument('url');i.add_argument('--title',default='');i.add_argument('--key')
    a=s.add_parser('analyze',help='Local model analysis or explicitly heuristic-only analysis')
    a.add_argument('id');a.add_argument('--genre',choices=['speech','tutorial','action','mixed'],default='speech');a.add_argument('--mode',choices=['local','heuristic'],default='local');a.add_argument('--query',default='');a.add_argument('--min-length',type=float,default=5);a.add_argument('--max-length',type=float,default=30);a.add_argument('--count',type=int,default=5);a.add_argument('--separation',type=float,default=2);a.add_argument('--no-transcribe',action='store_true');a.add_argument('--key')
    a.add_argument('--audio-stream',type=int,default=0);a.add_argument('--use-feedback',action='store_true')
    a.add_argument('--visual-search',action='store_true',help='Experimental local visual retrieval; requires an English query')
    a.add_argument('--visual-model',choices=['clip-vit-b32','xclip-b16-k600'],default='clip-vit-b32')
    a.add_argument('--impact-detection',action='store_true',help='Sound onset suggestions for action/mixed footage; review the actual event')
    a.add_argument('--impact-low-hz',type=float,default=1000);a.add_argument('--impact-high-hz',type=float,default=6000)
    a.add_argument('--impact-threshold',type=float,default=.04);a.add_argument('--impact-min-gap',type=float,default=.6)
    a.add_argument('--impact-lead',type=float,default=.4);a.add_argument('--impact-follow',type=float,default=.5)
    o=s.add_parser('ocr',help='Queue offline sampled on-screen text extraction');o.add_argument('id');o.add_argument('--sample-seconds',type=float,default=5);o.add_argument('--max-samples',type=int,default=120);o.add_argument('--key')
    o.add_argument('--model',choices=['ppocr-v4','ppocr-v5-mobile'],default='ppocr-v4');o.add_argument('--mode',choices=['baseline','detail'],default='baseline');o.add_argument('--quality-samples',type=int,choices=[1,2,3],default=1)
    s.add_parser('ocr-result',help='Read latest completed OCR evidence').add_argument('id')
    a.add_argument('--model',choices=['tiny.en','tiny'],default='tiny.en');a.add_argument('--language',choices=['en','pt','auto'],default='en')
    t=s.add_parser('transcribe',help='Queue genuine offline Whisper ASR');t.add_argument('id');t.add_argument('--key');t.add_argument('--audio-stream',type=int,default=0)
    t.add_argument('--model',choices=['tiny.en','tiny'],default='tiny.en');t.add_argument('--language',choices=['en','pt','auto'],default='en')
    t=s.add_parser('timeline',help='Save versioned timeline JSON with optimistic revision');t.add_argument('id');t.add_argument('file');t.add_argument('--revision',required=True,type=int)
    r=s.add_parser('render',help='Queue exact timeline revision for verified export');r.add_argument('id');r.add_argument('--revision',type=int);r.add_argument('--key')
    s.add_parser('exports',help='List completed verified exports')
    s.add_parser('jobs',help='Inspect durable job queue')
    for name in ('job','cancel','retry','evidence','candidates'):
        s.add_parser(name).add_argument('id')
    a=s.add_parser('agent-submit',help='Validate timestamped proposals bound to an evidence packet');a.add_argument('id');a.add_argument('file')
    a=s.add_parser('publication-draft',help='Prepare a manual-upload draft; never posts');a.add_argument('file')
    a=s.add_parser('batch-render',help='Bounded <=20 per-item render requests from JSON');a.add_argument('file')
    a=s.add_parser('download-model',help='Explicit cancelable pinned model download; inspect capabilities for bytes first');a.add_argument('name');a.add_argument('--key')
    a=s.add_parser('backup',help='Full verified backup, including originals; stop app and finish jobs first');a.add_argument('path')
    a=s.add_parser('restore',help='Restore safely into a new directory only');a.add_argument('archive');a.add_argument('destination')
    s.add_parser('storage',help='Inspect storage without deleting anything')
    s.add_parser('plan-storage',help='Bounded read-only retention report; no files are deleted or moved')
    a=s.add_parser('verify-storage-plan',help='Refresh a saved storage plan read-only; never authorizes deletion');a.add_argument('file')
    return p

def wait_worker_ready(child, timeout=10, interrupted=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if interrupted is not None:
            interrupted()
        code = child.poll()
        if code is not None:
            raise RuntimeError(f'Worker exited during startup (exit {code}); inspect the startup error above.')
        if any(item['pid'] == child.pid for item in runtime.worker_status()['workers']):
            return
        time.sleep(0.05)
    raise RuntimeError('Worker did not report readiness within 10 seconds; check storage and process permissions.')


def stop_worker(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            from .worker import terminate_tree
            terminate_tree(child)


def serve():
    """Bind before spawning work; stop the API if its managed worker dies."""
    import uvicorn
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run the local server on the main thread so shutdown signals can be handled safely.')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    child = None
    monitor = None
    finished = threading.Event()
    failure = []
    server = None
    shutdown_signal = None
    previous_handlers = {}
    def request_shutdown(signum, frame):
        # Record first, rather than raising inside Popen before its child is assigned.
        nonlocal shutdown_signal
        shutdown_signal = signum
        if server is not None:
            server.should_exit = True
    def check_shutdown():
        if shutdown_signal is not None:
            raise SystemExit(128 + shutdown_signal)
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, request_shutdown)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', config.PORT))
            listener.listen(128)
        except OSError as exc:
            raise RuntimeError(f'Cannot start local API at 127.0.0.1:{config.PORT}: {exc}. '
                               'If another instance is running, use it or stop it before restarting.') from exc
        db.migrate()
        check_shutdown()
        child = subprocess.Popen([sys.executable, '-m', 'clipstudio.worker', '--supervisor-pid', str(os.getpid())],
                                 start_new_session=True)
        wait_worker_ready(child, interrupted=check_shutdown)
        check_shutdown()
        server = uvicorn.Server(uvicorn.Config('clipstudio.api:app', host='127.0.0.1', port=config.PORT,
                                              log_level='warning', access_log=False))
        def watch_worker():
            announced = False
            while not finished.wait(0.25):
                if server.started and not announced:
                    print(f'Vektri Clip Studio ready: http://127.0.0.1:{config.PORT}', file=sys.stderr, flush=True)
                    announced = True
                code = child.poll()
                if code is not None:
                    failure.append(f'Media worker exited unexpectedly (exit {code}); API stopped. Restart the app to recover queued work.')
                    server.should_exit = True
                    return
        monitor = threading.Thread(target=watch_worker, name='clipstudio-worker-watch', daemon=True)
        monitor.start()
        server.run(sockets=[listener])
        check_shutdown()
        if failure:
            raise RuntimeError(failure[0])
        if not server.started:
            raise RuntimeError('Local API failed during startup; inspect the server error above.')
    finally:
        finished.set()
        if monitor is not None:
            monitor.join(timeout=1)
        try:
            if child is not None:
                stop_worker(child)
        finally:
            listener.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    args=parser().parse_args();cmd=args.command
    try:
        if cmd=='mcp':
            from .mcp_bridge import run
            run(allow_proposals=args.allow_proposals);return
        if cmd=='serve':
            serve()
            return
        if cmd=='worker':
            from .worker import main as worker_main
            raise SystemExit(worker_main([]))
        if cmd=='import':
            db.migrate();result=service.import_path(args.path,args.title,{'genre':args.genre,'attribution':args.attribution,'permission_status':args.permission_status},args.key)
        elif cmd=='import-url':result=request('POST','/imports/url',{'url':args.url,'title':args.title},args.key)
        elif cmd=='projects':
            from urllib.parse import quote
            result=request('GET','/projects?q='+quote(args.query))
        elif cmd=='project':result=request('GET','/projects/'+args.id)
        elif cmd=='cloud-preflight':
            from .operational_api import read_cloud_scenario
            config.project_dir(args.id)
            result=request('POST',f'/projects/{args.id}/cloud-preflight',read_cloud_scenario(args.file))
        elif cmd=='publication-preflight':
            config.project_dir(args.id)
            result=request('GET',f'/publications/{args.id}/preflight')
        elif cmd=='capabilities':result=request('GET','/capabilities')
        elif cmd=='analyze':result=request('POST',f'/projects/{args.id}/analyze',{'genre':args.genre,'mode':args.mode,'query':args.query,'min_length':args.min_length,'max_length':args.max_length,'target_count':args.count,'min_separation':args.separation,'transcribe':not args.no_transcribe,'model':args.model,'language':args.language,'audio_stream':args.audio_stream,'use_feedback':args.use_feedback,'visual_search':args.visual_search,'visual_model':args.visual_model,**{name:getattr(args,name) for name in ('impact_detection','impact_low_hz','impact_high_hz','impact_threshold','impact_min_gap','impact_lead','impact_follow')}},args.key)
        elif cmd=='ocr':result=request('POST',f'/projects/{args.id}/ocr',{'ocr_sample_seconds':args.sample_seconds,'ocr_max_samples':args.max_samples,'ocr_mode':args.mode,'ocr_quality_samples':args.quality_samples,'ocr_model':args.model},args.key)
        elif cmd=='ocr-result':result=request('GET',f'/projects/{args.id}/ocr')
        elif cmd=='transcribe':result=request('POST',f'/projects/{args.id}/transcribe',{'model':args.model,'language':args.language,'audio_stream':args.audio_stream},args.key)
        elif cmd=='timeline':result=request('PUT',f'/projects/{args.id}/timeline',{'revision':args.revision,'timeline':json.loads(Path(args.file).read_text())})
        elif cmd=='render':result=request('POST',f'/projects/{args.id}/render',{'revision':args.revision},args.key)
        elif cmd=='jobs':result=request('GET','/jobs')
        elif cmd=='job':result=request('GET','/jobs/'+args.id)
        elif cmd in ('cancel','retry'):result=request('POST',f'/jobs/{args.id}/{cmd}',{})
        elif cmd=='exports':result=request('GET','/exports')
        elif cmd=='evidence':result=request('GET',f'/projects/{args.id}/evidence')
        elif cmd=='candidates':result=request('GET',f'/projects/{args.id}/candidates')
        elif cmd=='agent-submit':result=request('POST',f'/projects/{args.id}/agent-decisions',json.loads(Path(args.file).read_text()))
        elif cmd=='publication-draft':result=request('POST','/publications',json.loads(Path(args.file).read_text()))
        elif cmd=='batch-render':result=request('POST','/batch/render',json.loads(Path(args.file).read_text()))
        elif cmd=='download-model':result=request('POST','/models/'+args.name+'/download',{},args.key)
        elif cmd=='backup':
            from .backup import backup
            result=backup(args.path)
        elif cmd=='restore':
            from .backup import restore
            result=restore(args.archive,args.destination)
        elif cmd=='storage':result=service.storage_summary()
        elif cmd=='plan-storage':
            from .retention import plan
            result=plan(config.DATA)
        elif cmd=='verify-storage-plan':
            import stat
            from .retention import verify_plan
            descriptor=os.open(args.file,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(descriptor,'rb') as source:
                metadata=os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size>8*1024**2:
                    raise ValueError('Saved storage plan must be a regular JSON file of at most 8 MiB')
                payload=json.loads(source.read(8*1024**2+1))
            # Accept either the raw API report or this CLI's structured output.
            result=verify_plan(payload.get('result',payload) if isinstance(payload,dict) else payload,config.DATA)
        print(json.dumps({'api_version':1,'ok':True,'result':result},ensure_ascii=False,indent=2,allow_nan=False))
    except (ValueError,KeyError,OSError,RuntimeError) as e:
        print(json.dumps({'api_version':1,'ok':False,'error':str(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)

if __name__=='__main__':main()
