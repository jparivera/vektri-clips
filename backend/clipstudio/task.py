"""Isolated job execution with immutable stage artifacts and fenced commits."""
import contextlib
import json
import os
import resource
import shutil
import signal
import sys
import time
from pathlib import Path
from . import config,db,service
from .schemas import default_timeline,Timeline
from .security import checksum,free_space,fetch_public_media

PIPELINE_VERSION='1.0.0'

def implementation_fingerprint(stage):
    groups={'prepare':['media.py','vendor/chopify.py'],'render':['media.py','vendor/chopify.py','schemas.py'],
            'track':['media.py','vendor/chopify.py'],'transcribe':['analysis.py','models.py'],
            'analyze':['analysis.py','models.py','vision.py','temporal.py','review.py','impacts.py','media.py','schemas.py'],
            'ocr':['ocr.py','vision.py','models.py'],'download':['security.py'],'package':['task.py']}
    root=Path(__file__).parent
    return db.fingerprint({f:checksum(root/f) for f in groups.get(stage,['task.py'])})

class StageContext:
    def __init__(self,job_id,token):
        self.job_id=job_id;self.token=token
        j=db.check_lease(job_id,token);self.pid=j['project_id']
        self.root=config.project_dir(self.pid) if self.pid else config.DATA
        self.reused=[]
    def check(self):db.check_lease(self.job_id,self.token)
    def stage(self,name,inputs,fn):
        self.check();db.stage(self.job_id,self.token,name)
        fp=db.fingerprint({'version':PIPELINE_VERSION,'implementation':implementation_fingerprint(name),'stage':name,'inputs':inputs})
        parent=self.root/'artifacts'/name;parent.mkdir(parents=True,exist_ok=True)
        final=parent/fp
        marker=final/'complete.json'
        if marker.is_file():
            try:
                stored=json.loads(marker.read_text())
                if stored['fingerprint']==fp and all((final/n).is_file() and checksum(final/n)==h for n,h in stored['checksums'].items()):
                    self.check();self.reused.append(name)
                    self._register(name,fp,final,stored['result'])
                    return stored['result'],str(final.relative_to(self.root))
            except (OSError,ValueError,KeyError):pass
        free_space(parent)
        tmp=parent/('.'+fp+'-'+self.token)
        # Only a temp directory bound to this claim is eligible for replacement.
        if tmp.exists():shutil.rmtree(tmp)
        tmp.mkdir(mode=0o700)
        try:
            started=time.monotonic();result=fn(tmp)
            self.check()
            files={str(p.relative_to(tmp)):checksum(p) for p in tmp.rglob('*') if p.is_file() and not p.is_symlink()}
            receipt={'fingerprint':fp,'result':result,'checksums':files,'elapsed_seconds':round(time.monotonic()-started,3),'pipeline_version':PIPELINE_VERSION}
            with (tmp/'complete.json').open('w') as f:
                f.write(db.dumps(receipt));f.flush();os.fsync(f.fileno())
            with db.tx() as c:
                db.check_lease(self.job_id,self.token,c)
                if final.exists():
                    # A corrupt prior result is quarantined, never overwrites a verified source.
                    os.replace(final,parent/('.invalid-'+fp+'-'+self.token))
                os.replace(tmp,final)
                fd=os.open(parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                self._register(name,fp,final,result,c)
            return result,str(final.relative_to(self.root))
        except BaseException:
            if tmp.exists():shutil.rmtree(tmp)
            raise
    def _register(self,name,fp,folder,result,c=None):
        args=(self.pid or 'models',name,fp,str(folder.relative_to(self.root)),db.dumps(result),time.time())
        if c is not None:c.execute('INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?,?)',args)
        else:
            with db.tx() as c:
                db.check_lease(self.job_id,self.token,c)
                c.execute('INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?,?)',args)

def set_transcript(ctx,transcript,base_hash):
    """Never overwrite a correction saved while an ASR job was running."""
    with db.tx() as c:
        db.check_lease(ctx.job_id,ctx.token,c)
        p=c.execute('SELECT * FROM projects WHERE id=?',(ctx.pid,)).fetchone()
        current=json.loads(p['transcript']) if p['transcript'] else None
        if (db.fingerprint(current) if current else None)!=base_hash:
            return False
        now=time.time()
        c.execute('UPDATE projects SET transcript=?,analysis=NULL,updated_at=? WHERE id=?',(db.dumps(transcript),now,ctx.pid))
        rev=p['revision'];row=c.execute('SELECT timeline FROM revisions WHERE project_id=? AND revision=?',(ctx.pid,rev)).fetchone()
        timeline=json.loads(row['timeline'])
        # Refresh untouched generated words, preserving manually edited timeline captions.
        old_words=[{k:w[k] for k in ('word','start','end')} for w in (current or {}).get('words',[])]
        can_seed=not timeline['captions']['words'] or timeline['captions']['words']==old_words
        if can_seed and transcript.get('words'):
            words=[];previous=0
            for word in transcript['words']:
                a=max(previous,float(word['start']));b=float(word['end'])
                if b>a:
                    words.append({'word':word['word'],'start':a,'end':b});previous=b
            timeline['captions']['words']=words
            Timeline.model_validate(timeline).validate_source(json.loads(p['source'])['metadata']['duration'])
            c.execute('INSERT INTO revisions VALUES(?,?,?,?,?)',(ctx.pid,rev+1,db.dumps(timeline),db.fingerprint(timeline),now))
            c.execute('UPDATE projects SET revision=? WHERE id=?',(rev+1,ctx.pid))
            c.execute("UPDATE publications SET status='approval-invalidated',approval=NULL,updated_at=? WHERE project_id=? AND approval IS NOT NULL",(now,ctx.pid))
        db.audit(c,'transcript.generated',{'provider':transcript.get('provider'),'model':transcript.get('model'),'word_count':len(transcript.get('words',[]))},ctx.pid)
        return True

def execute(job_id,token):
    from . import media
    job=db.check_lease(job_id,token);kind=job['kind'];params=json.loads(job['params'])
    ctx=StageContext(job_id,token)
    if kind=='model-download':
        from .analysis import download_model
        db.stage(job_id,token,'downloading model')
        result=download_model(params['name'],config.DATA/'models',ctx.check)
        return {'model':result}
    p=service.project_row(ctx.pid);source=json.loads(p['source'])
    if kind=='import-url':
        from .adapters import PublicMediaAdapter
        receipt,download_dir=ctx.stage('download',{'url':params['url']},lambda out:PublicMediaAdapter().fetch(params['url'],out/'source.bin',ctx.check))
        dest=ctx.root/'source'/'original.bin'
        if not dest.exists():
            temp=dest.with_suffix('.partial')
            shutil.copyfile(ctx.root/download_dir/'source.bin',temp);ctx.check();os.replace(temp,dest)
        source['path']='source/original.bin';source['download']=receipt;source['size']=dest.stat().st_size
        with db.tx() as c:
            db.check_lease(job_id,token,c)
            c.execute('UPDATE projects SET source=? WHERE id=?',(db.dumps(source),ctx.pid))
    path=ctx.root/source['path']
    if kind in ('import','import-url'):
        sha=checksum(path);ctx.check()
        meta,folder=ctx.stage('prepare',{'sha256':sha},lambda out:media.prepare(path,out,ctx.check))
        source.update(sha256=sha,metadata=meta,prepared_dir=folder)
        path.chmod(0o444)
        with db.tx() as c:
            db.check_lease(job_id,token,c)
            row=c.execute('SELECT revision FROM projects WHERE id=?',(ctx.pid,)).fetchone()
            current_source=json.loads(c.execute('SELECT source FROM projects WHERE id=?',(ctx.pid,)).fetchone()['source'])
            source['provenance']=current_source['provenance']
            if row['revision']==0:
                timeline=default_timeline(meta['duration'])
                c.execute('INSERT INTO revisions VALUES(?,?,?,?,?)',(ctx.pid,1,db.dumps(timeline),db.fingerprint(timeline),time.time()))
            c.execute("UPDATE projects SET source=?,status='ready',revision=MAX(revision,1),updated_at=? WHERE id=?",(db.dumps(source),time.time(),ctx.pid))
            db.audit(c,'source.prepared',{'sha256':sha,'metadata':meta},ctx.pid)
        return {'project_id':ctx.pid,'metadata':meta,'reused_stages':ctx.reused}
    if p['status']!='ready':raise ValueError('Source preparation has not completed')
    sha=source['sha256']
    # Detect accidental external source mutation before expensive work.
    if checksum(path)!=sha:raise ValueError('Source integrity check failed. Restore the immutable source from backup before continuing.')
    if kind in ('transcribe','analyze'):
        from .analysis import transcribe,analyze,model_status
        transcript=json.loads(p['transcript']) if p['transcript'] else None
        requested_stream=params.get('audio_stream',0)
        stream_changed=bool(transcript and transcript.get('audio_stream',0)!=requested_stream)
        requested_model=params.get('model','tiny.en');requested_language=params.get('language','en')
        prior_config=(transcript or {}).get('configuration',{})
        prior_model=prior_config.get('requested_model',(transcript or {}).get('model','tiny.en').rsplit('/',1)[-1].removeprefix('faster-whisper-'))
        prior_language=prior_config.get('requested_language',(transcript or {}).get('language') or 'en')
        settings_changed=bool(transcript and (stream_changed or prior_model!=requested_model or prior_language!=requested_language))
        wants_asr=kind=='transcribe' or (params.get('transcribe',True) and not params.get('visual_search',False) and params.get('genre',p['genre']) in ('speech','tutorial','mixed'))
        if kind=='analyze' and wants_asr and settings_changed and transcript.get('corrected'):
            raise db.Conflict('The saved corrected transcript belongs to a different audio track, model or language setting. Explicitly transcribe with the selected settings first; saved timeline revisions retain your earlier corrections.')
        do_asr=kind=='transcribe' or (wants_asr and (not transcript or settings_changed))
        if do_asr:
            options={'model':requested_model,'language':requested_language,'audio_stream':requested_stream}
            model=next((m for m in model_status()['models'] if m['id']==options['model']),{})
            transcript,folder=ctx.stage('transcribe',{'sha256':sha,'options':options,'revision':model.get('revision')},lambda out:transcribe(path,out,options,ctx.check))
            applied=set_transcript(ctx,transcript,params.get('base_transcript_hash'))
            if not applied:
                if kind=='transcribe':raise db.Conflict('Transcript changed during transcription. Your correction was preserved and the new ASR artifact retained, but it was not applied. Review before retrying.')
                current=service.project_row(ctx.pid)
                transcript=json.loads(current['transcript']) if current['transcript'] else transcript
        if kind=='transcribe':return {'transcript':transcript,'reused_stages':ctx.reused}
        visual_only=bool(params.get('visual_search',False))
        # Visual inference depends on source frames, settings and model pins, not
        # caption revisions. Do not pass an unused transcript into that stage.
        analysis_transcript=None if visual_only else transcript
        transcript_fp=None if visual_only else db.fingerprint(transcript)
        options={k:v for k,v in params.items() if k!='base_transcript_hash'}
        stage_inputs={'sha256':sha,'options':options,'models':[{k:m.get(k) for k in ('id','revision','available')} for m in model_status()['models']]}
        if not visual_only:stage_inputs['transcript']=transcript_fp
        result,folder=ctx.stage('analyze',stage_inputs,lambda out:analyze(path,source['metadata'],analysis_transcript,options,out,ctx.check))
        result['artifact_dir']=folder
        if params.get('feedback_policy'):result['preference_effect']=params['feedback_policy']
        with db.tx() as c:
            db.check_lease(job_id,token,c)
            if not visual_only:
                current=c.execute('SELECT transcript FROM projects WHERE id=?',(ctx.pid,)).fetchone()
                actual=json.loads(current['transcript']) if current['transcript'] else None
                if db.fingerprint(actual)!=transcript_fp:raise db.Conflict('Transcript was corrected during analysis. Completed artifacts were retained; analyze again using the correction.')
            prior=c.execute('SELECT analysis FROM projects WHERE id=?',(ctx.pid,)).fetchone()['analysis']
            manual=[x for x in (json.loads(prior).get('candidates',[]) if prior else []) if x.get('method')=='manual']
            result['candidates'].extend(manual)
            c.execute('UPDATE projects SET analysis=?,updated_at=? WHERE id=?',(db.dumps(result),time.time(),ctx.pid))
            db.audit(c,'analysis.completed',{'candidate_count':len(result.get('candidates',[])),'method':result.get('method'),'artifact_dir':folder},ctx.pid)
        return {'analysis':result,'reused_stages':ctx.reused}
    if kind=='ocr':
        from .ocr import extract_text
        from .models import CATALOG
        result,folder=ctx.stage('ocr',{'sha256':sha,'options':params,'revision':CATALOG[params.get('ocr_model','ppocr-v4')]['revision']},
                                lambda out:extract_text(path,source['metadata']['duration'],params,out,ctx.check))
        return {'ocr':result,'source_sha256':sha,'artifact_dir':folder,'reused_stages':ctx.reused}
    if kind=='track':
        result,folder=ctx.stage('track',{'sha256':sha,'options':params},lambda out:media.track(path,params,out,ctx.check))
        return {'tracking':result,'artifact_dir':folder,'reused_stages':ctx.reused}
    if kind=='render':
        timeline=Timeline.model_validate(params['timeline']).validate_source(source['metadata']['duration']).model_dump()
        rendered,render_folder=ctx.stage('render',{'sha256':sha,'timeline':timeline},lambda out:media.render(path,timeline,out,ctx.check))
        render_fp=db.fingerprint({'sha256':sha,'timeline':timeline,'revision':params['revision'],'provenance':source['provenance'],
                                  'pipeline':PIPELINE_VERSION,'media_implementation':implementation_fingerprint('render'),'package_implementation':implementation_fingerprint('package')})
        def package_to(out):
            manifest=json.loads(db.dumps(rendered))
            for name in ('video.mp4','clean.mp4','captions.srt','captions.vtt','captions.ass','thumbnail.jpg'):
                if (ctx.root/render_folder/name).is_file():os.link(ctx.root/render_folder/name,out/name)
            manifest['project_id']=ctx.pid;manifest['revision']=params['revision'];manifest['source_provenance']=source['provenance'];manifest['source_sha256']=sha
            manifest['timeline']=timeline;manifest['fingerprint']=render_fp;manifest['pipeline_version']=PIPELINE_VERSION
            (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            # Rebuild the package after adding complete reproducibility/provenance metadata.
            import zipfile
            with zipfile.ZipFile(out/'package.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for name in ('video.mp4','clean.mp4','captions.srt','captions.vtt','captions.ass','thumbnail.jpg','manifest.json'):
                    if (out/name).is_file():z.write(out/name,name)
            with zipfile.ZipFile(out/'package.zip') as z:
                if z.testzip():raise ValueError('Export package CRC validation failed')
            return manifest
        manifest,folder=ctx.stage('package',{'render_folder':render_folder,'revision':params['revision'],'fingerprint':render_fp},package_to)
        with db.tx() as c:
            db.check_lease(job_id,token,c)
            old=c.execute('SELECT * FROM exports WHERE project_id=? AND fingerprint=?',(ctx.pid,render_fp)).fetchone()
            eid=old['id'] if old else db.uid()
            if not old:c.execute('INSERT INTO exports VALUES(?,?,?,?,?,?,?)',(eid,ctx.pid,params['revision'],render_fp,folder,db.dumps(manifest),time.time()))
            db.audit(c,'export.verified',{'export_id':eid,'revision':params['revision'],'fingerprint':render_fp,'reused_stages':ctx.reused},ctx.pid)
        return {'export':service.export_view(db.one('SELECT * FROM exports WHERE id=?',(eid,))),'reused_stages':ctx.reused}
    raise ValueError('Unsupported job kind')

def main():
    # Child is terminated if its supervisor dies on Linux. FFmpeg additionally has its own cancellation guard.
    if sys.platform.startswith('linux'):
        import ctypes
        parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent:raise SystemExit(1)
    jid,token=sys.argv[1:3];started=time.monotonic()
    try:
        result=execute(jid,token)
        result['elapsed_seconds']=round(time.monotonic()-started,3)
        result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['child_peak_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        db.finish(jid,token,'succeeded',result)
    except db.Cancelled as e:db.finish(jid,token,'cancelled',error=str(e))
    except db.LeaseLost:pass
    except BaseException as e:
        # Error text is operational only; no provider keys or process environment is logged.
        message=f'{type(e).__name__}: {str(e)[:1500]}'
        db.finish(jid,token,'failed',error=message)
        print(message,file=sys.stderr)
        raise SystemExit(1)

if __name__=='__main__':main()
