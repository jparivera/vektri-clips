"""Bounded file/API-contract and fake-child tests. No model is imported or run."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import wave

import psutil
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_diarization.py'
spec = importlib.util.spec_from_file_location('diarization_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def wav(seconds=.05, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, 'wb') as audio:
        audio.setnchannels(channels); audio.setsampwidth(width); audio.setframerate(rate)
        audio.writeframes(b'\0' * int(seconds * rate) * width * channels)
    return out.getvalue()


@pytest.mark.parametrize('values', [{'rate':8000}, {'channels':2}, {'width':1}, {'seconds':0}, {'seconds':120.01}])
def test_invalid_audio_geometry_or_duration_is_rejected(values):
    with pytest.raises(ValueError):
        runner.wav_info(wav(**values))


def test_wave_source_hash_and_pcm_are_exact():
    raw = wav(120)
    info, pcm = runner.wav_info(raw)
    assert info['frames'] == 1920000 and info['duration_seconds'] == 120
    assert info['sha256'] == runner.digest(raw) and len(pcm) == 3840000


@pytest.mark.parametrize('mutation', ['truncated', 'riff_size', 'too_big', 'bad_magic', 'declared_long'])
def test_malformed_or_unbounded_wave_is_rejected_before_model_loading(mutation):
    raw = bytearray(wav())
    if mutation == 'truncated':
        raw = raw[:-10]; raw[4:8] = (len(raw)-8).to_bytes(4,'little')
    elif mutation == 'riff_size': raw[4:8] = (123).to_bytes(4,'little')
    elif mutation == 'too_big': raw.extend(b'x' * runner.MAX_WAV_BYTES)
    elif mutation == 'bad_magic': raw[:4] = b'NOPE'
    elif mutation == 'declared_long': raw[40:44] = (16000*2*121).to_bytes(4,'little')
    with pytest.raises(ValueError): runner.wav_info(bytes(raw))


def test_symlink_fifo_and_byte_cap_do_not_read_unbounded_data(tmp_path):
    original = tmp_path/'source.wav'; original.write_bytes(wav())
    linked = tmp_path/'linked.wav'; linked.symlink_to(original)
    fifo = tmp_path/'fifo'; os.mkfifo(fifo)
    with pytest.raises(OSError): runner.regular_bytes(linked, runner.MAX_WAV_BYTES)
    with pytest.raises(ValueError): runner.regular_bytes(fifo, runner.MAX_WAV_BYTES)
    with pytest.raises(ValueError): runner.regular_bytes(original, 10)


@pytest.mark.parametrize('speaker,threshold', [(0,.5),(9,.5),(True,.5),(-2,.5),(-1,float('nan')),(-1,0),(-1,1),(-1,True)])
def test_invalid_clustering_options_rejected(speaker, threshold):
    with pytest.raises(ValueError): runner.options(speaker,threshold)


def test_known_count_is_explicit_and_unknown_uses_threshold():
    assert runner.options()['speaker_count_mode'] == 'unknown'
    assert runner.options()['threshold_used'] is True
    assert runner.options(2)['speaker_count_mode'] == 'explicit_known_count'
    assert runner.options(2)['threshold_used'] is False
    assert runner.options(2)['num_threads'] == 2 and runner.options(2)['provider'] == 'cpu'


@pytest.mark.parametrize('rows', [[{'start':0,'end':float('nan'),'speaker_index':0}],
    [{'start':-.01,'end':.5,'speaker_index':0}], [{'start':0,'end':2,'speaker_index':0}],
    [{'start':0,'end':.5,'speaker_index':-1}], [{'start':0,'end':.5,'speaker_index':True}],
    [{'start':.5,'end':.7,'speaker_index':0},{'start':.1,'end':.3,'speaker_index':1}]])
def test_invalid_turn_evidence_fails_closed(rows):
    with pytest.raises(ValueError): runner.validate_turns(rows,1)


def test_overlap_is_preserved_and_labels_stay_anonymous():
    rows=runner.validate_turns([{'start':0,'end':.7,'speaker_index':0},
                               {'start':.2,'end':.8,'speaker_index':1}],1)
    assert [r['speaker'] for r in rows]==['speaker_00','speaker_01']
    assert rows[1]['start'] < rows[0]['end']


@pytest.fixture
def planned(tmp_path):
    source=tmp_path/'source.wav'; source.write_bytes(wav())
    runtime=tmp_path/'runtime'; (runtime/'bin').mkdir(parents=True)
    (runtime/'bin/python').symlink_to(sys.executable)
    models=[]
    for index,mid in enumerate(runner.MODEL_IDS):
        directory=tmp_path/f'model{index}'; directory.mkdir()
        raw=b'invented non-model bytes '+str(index).encode()
        filename='fixture.onnx'; (directory/filename).write_bytes(raw)
        models.append({'id':mid,'repository':'fixture','revision':'fixture','filename':filename,
            'bytes':len(raw),'sha256':runner.digest(raw),'license':'fixture','target_directory':str(directory)})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'runtime':'sherpa-onnx','version':'2.0.1','runtime_directory':str(runtime),'models':models}))
    return source,manifest,tmp_path/'out'


def fake_result(out, **overrides):
    request=json.loads((out/'request.json').read_text())
    result={'source_sha256':request['source']['sha256'],'models':request['models'],
            'config':request['config'],'turns':[], 'runtime':{'sherpa_onnx':'2.0.1'}} | overrides
    runner.write_new(out/'child-result.json',result)


def test_successful_result_preserves_original_and_full_config(planned,monkeypatch):
    source,manifest,out=planned; before=source.read_bytes()
    def fake(command, output, **kwargs):
        assert '-I' in command
        fake_result(output)
        return {'exit_code':0,'peak_process_tree_rss_bytes':123}
    monkeypatch.setattr(runner,'supervise',fake)
    report=runner.run_trial(source,manifest,out,speakers=2)
    assert report['status']=='succeeded' and report['source_unchanged']
    assert source.read_bytes()==before==(out/'input.wav').read_bytes()
    assert report['config']['num_clusters']==2 and report['result']['turns']==[]
    assert json.loads((out/'report.json').read_text())==report


@pytest.mark.parametrize('failure',['source_mismatch','bad_json','missing_result','source_changed','runtime_failure'])
def test_error_evidence_is_recorded_without_success_claim(planned,monkeypatch,failure):
    source,manifest,out=planned
    def fake(command, output, **kwargs):
        if failure=='runtime_failure': raise runner.GuardFailure('controlled fake failure',{'exit_code':3})
        if failure=='missing_result': return {'exit_code':0}
        if failure=='bad_json': (output/'child-result.json').write_text('{broken')
        elif failure=='source_mismatch': fake_result(output,source_sha256='f'*64)
        else:
            fake_result(output); source.write_bytes(wav(.06))
        return {'exit_code':0}
    monkeypatch.setattr(runner,'supervise',fake)
    with pytest.raises((ValueError,OSError,runner.GuardFailure)):
        runner.run_trial(source,manifest,out)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='failed'
    assert report['source_unchanged'] is (failure!='source_changed')


def test_existing_output_is_preserved(planned):
    source,manifest,out=planned; out.mkdir(); (out/'keep').write_text('preserve')
    with pytest.raises(ValueError,match='fresh output'):
        runner.run_trial(source,manifest,out)
    assert (out/'keep').read_text()=='preserve'


def test_model_hash_mismatch_fails_before_creating_output(planned):
    source,manifest,out=planned
    doc=json.loads(manifest.read_text()); doc['models'][0]['sha256']='0'*64
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='SHA-256'): runner.run_trial(source,manifest,out)
    assert not out.exists()


def test_models_changed_during_run_invalidate_evidence(planned,monkeypatch):
    source,manifest,out=planned
    def fake(command, output, **kwargs):
        fake_result(output)
        request=json.loads((output/'request.json').read_text())
        path=Path(request['models'][0]['path'])
        path.write_bytes(b'x'*path.stat().st_size)
        return {'exit_code':0}
    monkeypatch.setattr(runner,'supervise',fake)
    with pytest.raises(ValueError,match='models changed'):
        runner.run_trial(source,manifest,out)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='failed' and not report['models_unchanged']
    assert report['source_unchanged']


@pytest.fixture
def enough_fake_memory(monkeypatch):
    # Only fake-child tests use controlled memory observations. No model runs.
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=runner.RESERVE_BYTES+1024**3))


def test_memory_guard_prevents_child_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=runner.RESERVE_BYTES-1))
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not start child'))
    with pytest.raises(runner.GuardFailure,match='reserve'):
        runner.supervise([sys.executable,'-c','pass'],tmp_path)


def assert_no_live_child(pid):
    try:
        process=psutil.Process(pid)
        assert not process.is_running() or process.status()==psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.timeout(10)
def test_fake_child_success_and_resource_observations(tmp_path,enough_fake_memory):
    report=runner.supervise([sys.executable,'-c','print("controlled fake child")'],tmp_path,timeout=2)
    assert report['exit_code']==0
    assert report['minimum_system_available_bytes']>=runner.RESERVE_BYTES
    assert (tmp_path/'worker.log').read_text().strip()=='controlled fake child'
    assert_no_live_child(report['child_pid'])


@pytest.mark.timeout(10)
@pytest.mark.parametrize('stop',['timeout','rss','cancel','available_ram'])
def test_fake_child_is_killed_for_each_supervisor_guard(tmp_path,enough_fake_memory,monkeypatch,stop):
    if stop=='rss':monkeypatch.setattr(runner,'MAX_RSS_BYTES',1)
    if stop=='available_ram':
        values=iter([runner.RESERVE_BYTES+1024**3,runner.RESERVE_BYTES-1])
        monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=next(values)))
    calls=[0]
    def check():
        calls[0]+=1
        if stop=='cancel' and calls[0]>1:raise KeyboardInterrupt('controlled cancellation')
    with pytest.raises((runner.GuardFailure,KeyboardInterrupt)) as caught:
        runner.supervise([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,
                         timeout=.15 if stop=='timeout' else 2,check=check)
    if isinstance(caught.value,runner.GuardFailure):assert_no_live_child(caught.value.stats['child_pid'])


@pytest.mark.timeout(10)
def test_crashing_leader_does_not_leave_its_owned_child_running(tmp_path,enough_fake_memory):
    pidfile=tmp_path/'child.pid'
    code=('import subprocess,sys,os;from pathlib import Path;'
          'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
          f'Path({str(pidfile)!r}).write_text(str(p.pid));os._exit(3)')
    with pytest.raises(runner.GuardFailure,match='code 3'):
        runner.supervise([sys.executable,'-c',code],tmp_path,timeout=2)
    assert_no_live_child(int(pidfile.read_text()))


@pytest.mark.timeout(10)
@pytest.mark.parametrize('failure_stage',['constructor','creation_time'])
def test_ownership_setup_failure_still_kills_exact_unreaped_child(tmp_path,enough_fake_memory,monkeypatch,failure_stage):
    spawned=[]
    original_popen=runner.subprocess.Popen
    original_process=psutil.Process
    def tracked_popen(*args,**kwargs):
        child=original_popen(*args,**kwargs); spawned.append(child); return child
    def failing_process(pid):
        if failure_stage=='constructor':raise psutil.AccessDenied(pid)
        def fail():raise psutil.AccessDenied(pid)
        return SimpleNamespace(pid=pid,create_time=fail,is_running=fail)
    monkeypatch.setattr(runner.subprocess,'Popen',tracked_popen)
    monkeypatch.setattr(psutil,'Process',failing_process)
    with pytest.raises(psutil.AccessDenied):
        runner.supervise([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,timeout=2)
    monkeypatch.setattr(psutil,'Process',original_process)
    assert len(spawned)==1 and spawned[0].returncode is not None
    assert_no_live_child(spawned[0].pid)


@pytest.mark.timeout(10)
def test_second_cancellation_signal_cannot_interrupt_cleanup(tmp_path,enough_fake_memory,monkeypatch):
    original_sleep=time.sleep
    old=signal.getsignal(signal.SIGINT)
    def interrupt_cleanup(seconds):
        if seconds==.1:os.kill(os.getpid(),signal.SIGINT)
        original_sleep(seconds)
    monkeypatch.setattr(runner.time,'sleep',interrupt_cleanup)
    report=runner.supervise([sys.executable,'-c','pass'],tmp_path,timeout=2)
    assert report['exit_code']==0 and signal.getsignal(signal.SIGINT)==old
    assert_no_live_child(report['child_pid'])


def test_new_json_does_not_overwrite_or_write_nonfinite_values(tmp_path):
    target=tmp_path/'record.json'; runner.write_new(target,{'ok':True})
    with pytest.raises(FileExistsError):runner.write_new(target,{'ok':False})
    with pytest.raises(ValueError):runner.write_new(tmp_path/'nan.json',{'invalid':float('nan')})
    assert json.loads(target.read_text())=={'ok':True}
