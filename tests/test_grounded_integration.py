"""Real FFmpeg integration on generated media; no model or network required."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
import pytest

from clipstudio import analysis, models, task
from clipstudio.schemas import AnalysisOptions, Candidate, Timeline

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg unavailable')


@pytest.fixture(autouse=True)
def no_model_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Heuristic integration must not access models')
    monkeypatch.setattr(analysis, 'model_path', forbidden)
    monkeypatch.setattr(models, 'model_path', forbidden)
    monkeypatch.setattr(analysis, 'MiniLM', forbidden)


@pytest.fixture(scope='module')
def generated_media(tmp_path_factory):
    root = tmp_path_factory.mktemp('grounded-integration')
    video = root / 'video-only.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=duration=6:size=160x90:rate=25', '-an',
                    '-c:v', 'libx264', '-threads', '1', str(video)], check=True)
    tracks = []
    for index, times in enumerate(((1., 4.), (2., 5.))):
        pcm = np.zeros(6 * 16000, dtype='<i2')
        for time in times:
            pcm[round(time * 16000)] = 30000
        track = root / f'track-{index}.wav'
        with wave.open(str(track), 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm.tobytes())
        tracks.append(track)
    source = root / 'two-tracks.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video),
                    '-i', str(tracks[0]), '-i', str(tracks[1]),
                    '-map', '0:v:0', '-map', '1:a:0', '-map', '2:a:0',
                    '-c:v', 'copy', '-c:a', 'pcm_s16le', str(source)], check=True)
    return source, video


def options(**updates):
    return AnalysisOptions.model_validate({
        'genre': 'action', 'mode': 'heuristic', 'impact_detection': True,
        'target_count': 5, 'min_separation': 0, 'transcribe': False,
        **updates,
    }).model_dump()


def accepted_timeline(candidates, duration=6):
    return Timeline.model_validate({'segments': [
        {key: candidate[key] for key in ('id', 'start', 'end')}
        for candidate in candidates]}).validate_source(duration)


@pytest.mark.parametrize('track,expected', [(0, [1., 4.]), (1, [2., 5.])])
def test_real_analysis_review_frames_source_and_selected_track(generated_media, tmp_path, track, expected):
    source, _ = generated_media
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = analysis.analyze(source, {}, None, options(audio_stream=track), tmp_path, lambda: None)
    assert result == json.loads((tmp_path / 'analysis.json').read_text())
    assert result['status'] == 'succeeded'
    assert result['provider'] == 'heuristics'
    assert result['ai_used'] is False
    assert result['data_transmitted'] is False
    assert result['paid_usage'] == 0
    impacts = json.loads((tmp_path / 'impacts.json').read_text())
    assert impacts == result['impact_search']
    assert impacts['audio_stream'] == track
    assert [p['time'] for p in impacts['peaks']] == pytest.approx(expected, abs=.015)
    candidates = result['candidates']
    assert len(candidates) == 2
    timeline = accepted_timeline(candidates)
    for candidate, segment in zip(candidates, timeline.segments):
        # External-agent Candidate intentionally excludes the trusted review receipt.
        Candidate.model_validate({key: value for key, value in candidate.items() if key != 'review'})
        evidence = candidate['evidence'][0]
        peak_index = int(evidence['ref'].rsplit('/', 1)[1])
        assert evidence['time'] == impacts['peaks'][peak_index]['time']
        assert evidence['audio_stream'] == track
        assert (segment.start, segment.end) == (candidate['start'], candidate['end'])
        receipt = candidate['review']
        assert receipt['source_sha256'] == source_sha
        assert len(receipt['fingerprint']) == 64
        assert receipt['status'] == 'needs_human_review'
        assert receipt['playback']['start'] == segment.start
        assert receipt['playback']['end'] == segment.end
        assert receipt['observations']['hook']['status'] == 'unknown'
        assert receipt['observations']['signals']['evidence'][0]['time'] == evidence['time']
        assert bool(receipt['warnings']) == bool(track)
        assert len(receipt['frames']) == 3
        assert receipt['frames'][1]['time'] == evidence['time']
        for frame in receipt['frames']:
            assert frame['status'] == 'available'
            assert Path(frame['path']).name == frame['path']
            assert (tmp_path / frame['path']).read_bytes().startswith(b'\xff\xd8')
            assert 0 <= frame['time'] < 6
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert not list(tmp_path.glob('vektri-impacts-*'))
    assert not list(tmp_path.glob('*.partial*'))


def test_no_audio_abstains_without_fabricated_review(generated_media, tmp_path):
    _, video = generated_media
    result = analysis.analyze(video, {}, None, options(), tmp_path, lambda: None)
    assert result['status'] == 'no_candidates'
    assert result['candidates'] == []
    assert result['impact_search']['status'] == 'no_audio'
    assert list(tmp_path.iterdir()) == [tmp_path / 'analysis.json']


def test_cancellation_between_detection_and_review_never_saves_success(generated_media, tmp_path):
    source, _ = generated_media
    class Cancelled(RuntimeError):
        pass
    def check():
        if (tmp_path / 'impacts.json').exists():
            raise Cancelled('cancelled after detector')
    with pytest.raises(Cancelled, match='cancelled after detector'):
        analysis.analyze(source, {}, None, options(), tmp_path, check)
    assert not (tmp_path / 'analysis.json').exists()
    assert not list(tmp_path.glob('review-*'))
    assert not list(tmp_path.glob('vektri-impacts-*'))


def test_minimum_padding_is_accepted_by_timeline(generated_media, tmp_path):
    source, _ = generated_media
    result = analysis.analyze(source, {}, None,
                              options(impact_lead=.02, impact_follow=.02), tmp_path, lambda: None)
    assert len(result['candidates']) == 2
    accepted_timeline(result['candidates'])


@pytest.mark.parametrize('dependency', ['review.py', 'impacts.py', 'media.py', 'schemas.py'])
def test_analysis_fingerprint_includes_grounding_dependencies(monkeypatch, dependency):
    original = task.checksum
    baseline = task.implementation_fingerprint('analyze')
    def altered(path):
        return 'changed-test-content' if path.name == dependency else original(path)
    monkeypatch.setattr(task, 'checksum', altered)
    assert task.implementation_fingerprint('analyze') != baseline


def test_stage_receipt_hashes_frames_and_rebuilds_tampering(generated_media, tmp_path, monkeypatch):
    from clipstudio import config, db, security, service
    directory = tmp_path / 'isolated-data'
    monkeypatch.setattr(config, 'DATA', directory)
    monkeypatch.setattr(config, 'DB', directory / 'studio.sqlite3')
    monkeypatch.setenv('VEKTRI_DATA_DIR', str(directory))
    config.initialize()
    db.migrate()
    pid = service.new_project('Generated grounding test', 'synthetic.mkv',
                              {'permission_status': 'synthetic'})
    db.enqueue('test', {}, pid)
    owner = db.claim()
    context = task.StageContext(owner['id'], owner['lease_token'])
    source, _ = generated_media
    calls = []
    def work(out):
        calls.append(True)
        return analysis.analyze(source, {}, None, options(target_count=1), out, context.check)
    inputs = {'source': security.checksum(source), 'options': options(target_count=1)}
    result, relative = context.stage('analyze', inputs, work)
    folder = config.project_dir(pid) / relative
    marker = json.loads((folder / 'complete.json').read_text())
    assert marker['result'] == result
    assert len(marker['checksums']) == 5  # analysis, impacts and three JPEGs
    for name, checksum in marker['checksums'].items():
        assert checksum == security.checksum(folder / name)
    assert context.stage('analyze', inputs, work)[0] == result
    assert len(calls) == 1
    frame = folder / result['candidates'][0]['review']['frames'][1]['path']
    original_frame_sha = security.checksum(frame)
    frame.write_bytes(b'tampered review frame')
    context.stage('analyze', inputs, work)
    assert len(calls) == 2
    assert security.checksum(frame) == original_frame_sha
    assert security.checksum(source) == inputs['source']
