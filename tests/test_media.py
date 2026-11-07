"""Actual media mechanics and failure paths using the shared synthetic corpus."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import pytest

from clipstudio import media

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def noop():
    pass


def timeline(mode="fit", burned=False):
    return {"schema_version": 1, "segments": [
        {"id": "a", "start": 1, "end": 2.5, "crop": {"mode":mode,"x":.3,"y":.5,"zoom":1,"second_x":.75,"keyframes":[]},"gain_db":-2},
        {"id": "b", "start": 7, "end": 8.5, "crop": {"mode":mode,"x":.7,"y":.5,"zoom":1,"second_x":.25,"keyframes":[]},"gain_db":0}],
        "output": {"width":180,"height":320,"fps":24,"crf":27},
        "captions": {"enabled":True,"burn_in":burned,"font_size":18,"margin":20,"alignment":2,"style":"clean","highlight":True,
                     "words":[{"word":"Synthetic","start":1.2,"end":1.7},{"word":"caption.","start":1.8,"end":2.3},
                              {"word":"Reordered","start":7.1,"end":7.6},{"word":"timing.","start":7.7,"end":8.4}]},
        "audio":{"normalize":False,"fade":0}}


def test_probe_preserves_stream_metadata_and_unicode():
    info=media.probe(FIXTURES / "portrait — café 日本語.mp4")
    assert (info["width"],info["height"])==(360,640)
    assert info["audio"] is False
    assert info["fps"]==24
    assert info["time_base"] and info["streams"]


def test_corrupt_and_playlist_inputs_rejected(tmp_path):
    bad=tmp_path/"corrupt.mp4"
    bad.write_text("#EXTM3U\nhttp://169.254.169.254/latest/meta-data")
    with pytest.raises(media.MediaError,match="container"):
        media.probe(bad)


def test_prepare_actual_proxy_waveform_and_no_audio(tmp_path):
    source=FIXTURES / "spoken_interview.mp4"
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    result=media.prepare(source,tmp_path/"prepare",noop)
    assert result["sha256"]==original
    assert media.probe(tmp_path/"prepare/proxy.mp4")["audio"]
    waveform=json.loads((tmp_path/"prepare/waveform.json").read_text())
    assert 0<len(waveform["bins"])<=2048 and max(waveform["bins"])>0
    assert (tmp_path/"prepare/thumbnail.jpg").stat().st_size>100
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original
    silent=media.prepare(FIXTURES/"portrait — café 日本語.mp4",tmp_path/"silent",noop)
    assert silent["audio"] is False
    assert set(json.loads((tmp_path/"silent/waveform.json").read_text())["bins"])=={0.0}


@pytest.mark.parametrize("source,mode,burned",[
    ("spoken_interview.mp4","fit",True),
    ("tutorial_screen.mp4","split",False),
    ("action_motion.mp4","crop",False),
    ("mixed_nonverbal.mp4","fit",False),
    ("portrait — café 日本語.mp4","crop",False),
])
def test_real_render_all_genres_and_layouts(tmp_path,source,mode,burned):
    source=FIXTURES/source
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    edits=timeline(mode,burned)
    if mode=="crop":
        edits["segments"][0]["crop"]["keyframes"]=[{"time":1,"x":.2,"y":.4},{"time":2.5,"x":.8,"y":.6}]
    result=media.render(source,edits,tmp_path,noop)
    assert result["verification"]["full_decode"]=="passed"
    assert result["duration"]==pytest.approx(3,abs=.15)
    assert result["source_sha256"]==before==hashlib.sha256(source.read_bytes()).hexdigest()
    assert (result["width"],result["height"])==(180,320)
    assert result["verification"]["audio_present"] is True
    with zipfile.ZipFile(tmp_path/"package.zip") as package:
        assert package.testzip() is None
        assert set(package.namelist()) >= {"video.mp4","captions.srt","captions.vtt","manifest.json","thumbnail.jpg"}
        assert ("clean.mp4" in package.namelist())==burned
    srt=(tmp_path/"captions.srt").read_text()
    assert "00:00:01,600 --> 00:00:02,900" in srt
    assert "Synthetic caption." in srt
    assert not (tmp_path/"parts").exists()


def test_caption_mapping_reorder_repeat_and_cut_words():
    words=[{"word":"hello","start":1,"end":2},{"word":"world","start":5,"end":6}]
    actual=media.map_caption_words(words,[{"start":5,"end":7},{"start":1.5,"end":2.5},{"start":5,"end":6}])
    assert [(w["word"],w["start"],w["end"]) for w in actual]==[("world",0,1),("hello",2,2.5),("world",3,4)]


def test_actual_audio_only_export_and_normalization(tmp_path):
    source=tmp_path/"original audio.wav"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","sine=frequency=440:duration=4","-c:a","pcm_s16le",str(source)],check=True)
    metadata=media.prepare(source,tmp_path/"proxy",noop)
    assert metadata["video"] is False and metadata["audio"] is True
    edits=timeline()
    edits["segments"]=edits["segments"][:1]
    edits["audio"]={"normalize":True,"fade":.1}
    result=media.render(source,edits,tmp_path/"render",noop)
    assert result["duration"]==pytest.approx(1.5,abs=.15)
    assert result["audio"]["normalize_requested"]
    assert result["verification"]["sample_peak_dbfs"]<0


def test_cancel_kills_encoder_and_no_final_success(tmp_path,monkeypatch):
    calls=0
    children=[]
    original=media.subprocess.Popen
    def record(*args,**kwargs):
        child=original(*args,**kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(media.subprocess,"Popen",record)
    class Cancelled(Exception):
        pass
    def cancelled():
        nonlocal calls
        calls+=1
        if calls>=4:
            raise Cancelled("operator cancelled")
    edits=timeline()
    edits["segments"]=edits["segments"][:1]
    edits["segments"][0]["end"]=20
    edits["output"]={"width":720,"height":1280,"fps":30,"crf":22}
    with pytest.raises(Cancelled):
        media.render(FIXTURES/"action_motion.mp4",edits,tmp_path,cancelled)
    assert children and all(c.poll() is not None for c in children)
    assert not (tmp_path/"video.mp4").exists()
    assert not (tmp_path/"manifest.json").exists()


def test_full_disk_rejected_before_render(tmp_path,monkeypatch):
    monkeypatch.setattr(media.shutil,"disk_usage",lambda path:shutil._ntuple_diskusage(1000,999,1))
    with pytest.raises(media.MediaError,match="free disk"):
        media.render(FIXTURES/"action_motion.mp4",timeline(),tmp_path,noop)
    assert not (tmp_path/"video.mp4").exists()


def test_caption_safe_text_unicode_and_long_word_warning(tmp_path):
    words=[{"word":r"{\pos(0,0)} café 日本語 "+"x"*70,"start":0,"end":1,"segment":0}]
    warnings=media._subtitles(words,{"font_size":18,"margin":20,"highlight":True},180,320,tmp_path)
    assert warnings
    ass=(tmp_path/"captions.ass").read_text()
    assert r"\pos" not in ass
    assert "café 日本語" in ass


def test_silence_plan_does_not_change_source_and_is_optional():
    words=[{"word":"hello","start":1,"end":1.5},{"word":"um","start":2,"end":2.2},{"word":"world","start":4,"end":5}]
    unchanged=copy.deepcopy(words)
    result=media.silence_plan(words,0,6,gap=1,fillers=False)
    assert words==unchanged
    assert result["reversible"] and result["duration"]<6
    assert any(w["word"]=="um" for w in result["words"])


def test_assisted_tracking_honest_fallback(tmp_path):
    result=media.track(FIXTURES/"action_motion.mp4",{"mode":"face","start":0,"end":3},tmp_path,noop)
    assert result["status"] in {"fallback","assisted"}
    assert result["samples"] and result["keyframes"]
    assert "Not audio diarization" in result["limitations"][0]
    assert all(0<=k["x"]<=1 and 0<=k["y"]<=1 for k in result["keyframes"])


def test_trimmed_keyframe_path_preserves_original_interpolation():
    expression=media._axis_expression([{"time":0,"x":0},{"time":10,"x":1}],"x",2,4,.5)
    assert "0.20000000" in expression
    assert "0.60000000" in expression


@pytest.mark.skipif(not sys.platform.startswith("linux"),reason="Linux parent-death protection")
def test_killed_stage_cannot_leave_ffmpeg_running(tmp_path):
    code="from clipstudio.media import _run; _run(['-re','-f','lavfi','-i','testsrc2=size=144x256:rate=12','-t','60','-f','null','-'],lambda:None)"
    child=subprocess.Popen([sys.executable,"-c",code],start_new_session=True)
    ffmpeg_pid=None
    try:
        until=time.monotonic()+5
        while time.monotonic()<until:
            children=Path(f"/proc/{child.pid}/task/{child.pid}/children").read_text().split()
            if children:
                ffmpeg_pid=int(children[0])
                break
            time.sleep(.03)
        assert ffmpeg_pid is not None
        child.kill()  # Deliberately only the stage PID, exercising parent-death.
        child.wait(timeout=3)
        until=time.monotonic()+3
        while time.monotonic()<until:
            stat=Path(f"/proc/{ffmpeg_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2]=="Z":
                break
            time.sleep(.03)
        else:
            pytest.fail("FFmpeg remained running after the stage was killed")
    finally:
        try:
            os.killpg(child.pid,9)
        except ProcessLookupError:
            pass
        child.wait(timeout=3)


def test_output_audio_flash_sync_across_reordered_cuts(tmp_path):
    import cv2
    import numpy as np
    edits=timeline()
    edits["output"]={"width":320,"height":180,"fps":24,"crf":24}
    edits["captions"]["enabled"]=False
    edits["segments"][0].update(start=5,end=7,gain_db=0)
    edits["segments"][1].update(start=1,end=3,gain_db=0)
    result=media.render(FIXTURES/"action_motion.mp4",edits,tmp_path,noop)
    cap=cv2.VideoCapture(str(tmp_path/"video.mp4"))
    flashes=[]
    index=0
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        if frame[-12:,-40:].mean()>200:
            if not flashes or index/24-flashes[-1]>.15:
                flashes.append(index/24)
        index+=1
    cap.release()
    pcm=subprocess.check_output(["ffmpeg","-v","error","-i",str(tmp_path/"video.mp4"),"-vn","-ac","1","-ar","16000","-f","f32le","-"])
    wave=np.frombuffer(pcm,dtype=np.float32)
    levels=[float(np.sqrt(np.mean(wave[i:i+160]**2))) for i in range(0,len(wave),160)]
    onsets=[i*.01 for i,value in enumerate(levels) if value>.12 and (i==0 or levels[i-1]<=.12)]
    assert len(flashes)==len(onsets)==2
    assert max(abs(a-b) for a,b in zip(flashes,onsets))<=.065
    assert flashes==pytest.approx([1,3],abs=1/24)


def test_chosen_object_tracking_mechanics_against_known_boxes(tmp_path):
    truth=json.loads((FIXTURES/"action_motion.truth.json").read_text())
    result=media.track(FIXTURES/"action_motion.mp4",{"mode":"object","start":0,"end":3,
                       "x":20/640,"y":157/360,"width":60/640,"height":60/360},tmp_path,noop)
    errors=[]
    for sample in result["samples"]:
        frame=truth["frames"][round(sample["time"]*24)]
        x,y,w,h=frame["subjects"][0]["box"]
        errors.append(abs(sample["x"]-(x+w/2)/640)*640)
    assert result["confidence"]>=.7
    assert max(errors)<=16
    assert "template" in result["method"]


def test_rotation_vfr_multiaudio_metadata_and_render(tmp_path):
    base=tmp_path/"multi.mp4"
    subprocess.run(["ffmpeg","-v","error","-i",str(FIXTURES/"action_motion.mp4"),
                    "-f","lavfi","-i","sine=frequency=880:duration=3","-t","3",
                    "-map","0:v:0","-map","0:a:0","-map","1:a:0","-c:v","copy","-c:a","aac",str(base)],check=True)
    rotated=tmp_path/"rotated.mp4"
    subprocess.run(["ffmpeg","-v","error","-display_rotation:v:0","90","-i",str(base),"-map","0","-c","copy",str(rotated)],check=True)
    meta=media.probe(rotated)
    assert abs(meta["rotation"])==90
    assert (meta["width"],meta["height"])==(360,640)
    assert meta["audio_stream_count"]==2
    prepared=media.prepare(rotated,tmp_path/"prepared",noop)
    assert prepared["warnings"] and media.probe(tmp_path/"prepared/proxy.mp4")["width"]<media.probe(tmp_path/"prepared/proxy.mp4")["height"]
    vfr=tmp_path/"vfr.mkv"
    subprocess.run(["ffmpeg","-v","error","-i",str(base),"-vf","select='if(lt(t,1),not(mod(n,2)),1)'",
                    "-fps_mode","vfr","-c:v","libx264","-threads","2","-c:a","aac",str(vfr)],check=True)
    edits=timeline()
    edits["segments"]=edits["segments"][:1]
    result=media.render(vfr,edits,tmp_path/"render",noop)
    assert result["duration"]==pytest.approx(1.5,abs=.15)
    assert media.probe(tmp_path/"render/video.mp4")["fps"]==24


def test_selected_audio_stream_is_real_and_invalid_index_rejected(tmp_path):
    import numpy as np
    source=tmp_path/"two-tracks.mp4"
    subprocess.run(["ffmpeg","-v","error","-i",str(FIXTURES/"action_motion.mp4"),
                    "-f","lavfi","-i","sine=frequency=880:duration=3","-t","3",
                    "-map","0:v:0","-map","0:a:0","-map","1:a:0","-c:v","copy","-c:a","aac",str(source)],check=True)
    edits=timeline()
    edits["segments"]=edits["segments"][:1]
    edits["captions"]["enabled"]=False
    edits["audio"]["stream_index"]=1
    result=media.render(source,edits,tmp_path/"selected",noop)
    assert result["audio"]["selected_audio_ordinal"]==1
    assert result["audio"]["source_stream_index"]==2
    pcm=subprocess.check_output(["ffmpeg","-v","error","-i",str(tmp_path/"selected/video.mp4"),"-vn","-ac","1","-ar","8000","-f","f32le","-"])
    wave=np.frombuffer(pcm,dtype=np.float32)[800:8800]
    frequency=np.fft.rfftfreq(len(wave),1/8000)[np.abs(np.fft.rfft(wave)).argmax()]
    assert frequency==pytest.approx(880,abs=2)
    edits["audio"]["stream_index"]=2
    with pytest.raises(media.MediaError,match="unavailable"):
        media.render(source,edits,tmp_path/"invalid",noop)
    with pytest.raises(media.MediaError,match="unavailable"):
        media.render(FIXTURES/"portrait — café 日本語.mp4",edits,tmp_path/"silent-invalid",noop)


def test_caption_typography_and_color_are_present_in_real_render(tmp_path):
    import cv2
    import numpy as np
    edits=timeline(burned=True)
    edits["segments"]=edits["segments"][:1]
    edits["output"]={"width":320,"height":180,"fps":24,"crf":18}
    edits["captions"].update(font_family="DejaVu Serif",color="#00FFFF",highlight_color="#FF00FF",stroke_width=1,
                             font_size=24,words=[{"word":"café résumé","start":1.2,"end":2.3}])
    result=media.render(FIXTURES/"spoken_interview.mp4",edits,tmp_path,noop)
    assert result["captions"]["font"]=="system DejaVu Serif"
    ass=(tmp_path/"captions.ass").read_text()
    assert "DejaVu Serif" in ass and "&H00FFFF00" in ass and "&HFF00FF&" in ass
    cap=cv2.VideoCapture(str(tmp_path/"video.mp4"));cap.set(cv2.CAP_PROP_POS_MSEC,500)
    ok,frame=cap.read();cap.release()
    assert ok
    # Magenta active words are actually encoded, not only listed in metadata.
    region=frame[90:]
    assert np.count_nonzero((region[:,:,0]>140)&(region[:,:,2]>140)&(region[:,:,1]<120))>20
    with pytest.raises(media.MediaError,match="colors"):
        media._subtitles([],{"color":"red;movie=/etc/passwd"},320,180,tmp_path)
    with pytest.raises(media.MediaError,match="font"):
        media._subtitles([],{"font_family":"arbitrary"},320,180,tmp_path)


def test_selected_delayed_audio_keeps_source_timestamps(tmp_path):
    import numpy as np
    source=tmp_path/"delayed-track.mp4"
    subprocess.run(["ffmpeg","-v","error","-i",str(FIXTURES/"action_motion.mp4"),
                    "-itsoffset","2","-f","lavfi","-i","sine=frequency=880:duration=0.4", "-t","4",
                    "-map","0:v:0","-map","0:a:0","-map","1:a:0","-c:v","copy","-c:a","aac",str(source)],check=True)
    edits=timeline()
    edits["segments"]=edits["segments"][:1]
    edits["segments"][0].update(start=1,end=4,gain_db=0)
    edits["captions"]["enabled"]=False
    edits["audio"]["stream_index"]=1
    media.render(source,edits,tmp_path/"render",noop)
    pcm=subprocess.check_output(["ffmpeg","-v","error","-i",str(tmp_path/"render/video.mp4"),"-vn","-ac","1","-ar","8000","-f","f32le","-"])
    wave=np.frombuffer(pcm,dtype=np.float32)
    onset=np.flatnonzero(np.abs(wave)>.03)[0]/8000
    assert onset==pytest.approx(1,abs=.065), "Delayed selected audio must stay aligned with the source timeline"
    assert np.max(np.abs(wave[:6000]))<.001


def test_one_hundred_fractional_cuts_do_not_accumulate_audio_or_caption_drift(tmp_path):
    import cv2
    import numpy as np
    edits=timeline()
    edits["segments"]=[{"id":f"cut-{i}","start":1.95,"end":2.12,"crop":{"mode":"fit"},"gain_db":0} for i in range(100)]
    edits["output"]={"width":320,"height":180,"fps":24,"crf":25}
    edits["captions"].update(enabled=True,burn_in=False,words=[{"word":"Marker","start":2,"end":2.1}])
    result=media.render(FIXTURES/"action_motion.mp4",edits,tmp_path,noop)
    assert result["duration"]==pytest.approx(17,abs=.05)
    assert media.probe(tmp_path/"video.mp4")["fps"]==pytest.approx(24,abs=.001)
    pcm=subprocess.check_output(["ffmpeg","-v","error","-i",str(tmp_path/"video.mp4"),"-vn","-ac","1","-ar","16000","-f","f32le","-"])
    wave=np.frombuffer(pcm,dtype=np.float32)
    levels=[float(np.sqrt(np.mean(wave[i:i+160]**2))) for i in range(0,len(wave),160)]
    onsets=[i*.01 for i,value in enumerate(levels) if value>.12 and (i==0 or levels[i-1]<=.12)]
    cap=cv2.VideoCapture(str(tmp_path/"video.mp4"));flashes=[];index=0;previous=False
    while True:
        ok,frame=cap.read()
        if not ok:break
        bright=frame[-12:,-40:].mean()>200
        if bright and not previous:flashes.append(index/24)
        previous=bright;index+=1
    cap.release()
    expected=[i*.17+.05 for i in range(100)]
    assert len(onsets)==len(flashes)==100
    assert max(abs(a-b) for a,b in zip(onsets,expected))<=.02
    assert max(abs(a-b) for a,b in zip(flashes,expected))<=.065
    assert max(abs(a-b) for a,b in zip(flashes,onsets))<=.065
    # Sidecars retain the same exact output timeline after the hundredth cut.
    assert "00:00:16,880 --> 00:00:16,980" in (tmp_path/"captions.srt").read_text()
