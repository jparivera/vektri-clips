"""Bounded, cancellable media work. Called only from private worker processes.

Originals are read-only. All intermediate and final files belong to the supplied
stage directory. Parent worker commits that directory only after success.
"""
from __future__ import annotations

import array
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

from .vendor.chopify import ass_escape, crop_for, plan_cuts, remap_words, smooth_track, srt_time

Check = Callable[[], None]
THREADS = 2
FORMAT_ALLOWLIST = "mov,matroska,webm,avi,wav,mp3,ogg,flac,aac,asf,mpeg,mpegts,aiff"
_LIBC = ctypes.CDLL(None) if sys.platform.startswith("linux") else None


def _die_with_stage(expected_parent: int) -> None:
    # The API never calls this module in its request process. Each media stage
    # runs in a disposable worker child. Linux parent-death protection also
    # covers SIGKILL/crashes where Python finally/check cannot execute.
    if _LIBC is not None:
        _LIBC.prctl(1, signal.SIGKILL)
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGKILL)


class MediaError(RuntimeError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict | list) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _input(path: Path) -> list[str]:
    # Containers cannot open network resources or arbitrary playlist formats.
    return ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMAT_ALLOWLIST,
            "-threads", str(THREADS), "-i", str(path.resolve())]


def _run(args: list[str], check: Check, *, timeout: float = 3600, cwd: Path | None = None,
         loglevel: str = "error") -> str:
    check()
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", loglevel,
               "-threads", str(THREADS), "-filter_threads", "1", "-filter_complex_threads", "1", *args]
    with tempfile.TemporaryFile() as error:
        # Inherit the stage process group. The durable worker can then kill the
        # complete stage (including FFmpeg) after a lease loss or hard crash.
        # FFmpeg itself spawns threads, not child processes; a local check
        # exception can terminate this process without killing its parent.
        stage_pid = os.getpid()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=error, cwd=cwd,
                                 preexec_fn=(lambda: _die_with_stage(stage_pid)) if _LIBC is not None else None)
        started = time.monotonic()
        try:
            while child.poll() is None:
                check()
                if time.monotonic() - started > timeout:
                    raise MediaError(f"Media operation exceeded {timeout:.0f} seconds; retry with a smaller source or output.")
                time.sleep(0.08)
            error.seek(0, os.SEEK_END)
            size = error.tell()
            error.seek(max(0, size - 12000))
            message = error.read().decode("utf-8", "replace")
            if child.returncode:
                raise MediaError(f"FFmpeg failed ({child.returncode}): {message[-5000:]}")
            check()
            return message
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
            raise


def _fraction(value: str | None) -> float | None:
    try:
        a, b = (value or "").split("/")
        answer = float(a) / float(b)
        return answer if math.isfinite(answer) and answer > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise MediaError("Source is missing. Restore the original or import it again.")
    try:
        stage_pid = os.getpid()
        result = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                                 "-format_whitelist", FORMAT_ALLOWLIST, "-show_format", "-show_streams",
                                 "-of", "json", str(path.resolve())], capture_output=True, text=True,
                                timeout=30, check=True,
                                preexec_fn=(lambda: _die_with_stage(stage_pid)) if _LIBC is not None else None)
        raw = json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        raise MediaError("Could not decode this container. Use a playable local video/audio file; playlists and network containers are unsupported.") from exc
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    try:
        duration = float(raw.get("format", {}).get("duration") or max(float(s.get("duration") or 0) for s in streams))
    except (ValueError, TypeError):
        duration = 0
    if not math.isfinite(duration) or duration <= 0 or not (video or audio):
        raise MediaError("Source has no supported timed audio/video stream or reports an invalid duration.")
    if duration > 12 * 3600:
        raise MediaError("Source exceeds the 12-hour local safety limit. Split it before import.")
    rotation = 0.0
    if video:
        rotation = float(video.get("tags", {}).get("rotate", 0) or 0)
        for side in video.get("side_data_list", []):
            if "rotation" in side:
                rotation = float(side["rotation"])
    width, height = (int(video.get("width", 0)), int(video.get("height", 0))) if video else (0, 0)
    if width * height > 40_000_000:
        raise MediaError("Video frames exceed the 40-megapixel decode limit; create a smaller local copy first.")
    display_width, display_height = (height, width) if int(abs(rotation)) % 180 == 90 else (width, height)
    return {"duration": duration, "width": display_width, "height": display_height,
            "encoded_width": width, "encoded_height": height, "rotation": rotation,
            "fps": _fraction(video.get("avg_frame_rate")) if video else None,
            "r_frame_rate": video.get("r_frame_rate") if video else None,
            "time_base": video.get("time_base") if video else None,
            "audio": bool(audio), "video": bool(video), "size": path.stat().st_size,
            "format": raw.get("format", {}).get("format_name"), "streams": streams,
            "selected_video_stream": video.get("index") if video else None,
            "audio_stream_count": len(audio), "selected_audio_stream": audio[0].get("index") if audio else None,
            "normalization": "FFmpeg autorotation; explicit constant frame-rate proxy/export; proxy uses first audio track, export uses timeline selection"}


def _video_encode_args(crf: int = 23) -> list[str]:
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-threads", str(THREADS),
            "-pix_fmt", "yuv420p"]


def _encode_args(crf: int = 23) -> list[str]:
    return [*_video_encode_args(crf), "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart"]


def _thumb(video: Path, out: Path, check: Check, duration: float) -> None:
    temp = out.with_name(out.stem + ".partial.jpg")
    _run(["-ss", str(min(duration / 3, 3)), *_input(video), "-frames:v", "1",
          "-vf", "scale=480:-2", "-q:v", "3", str(temp)], check, timeout=120)
    os.replace(temp, out)


def _waveform(source: Path, out: Path, metadata: dict, check: Check) -> dict:
    bins = min(2048, max(64, math.ceil(metadata["duration"] * 8)))
    if not metadata["audio"]:
        return {"duration": metadata["duration"], "bins": [0.0] * bins, "audio": False, "sample_rate": 1000}
    raw = out / "waveform.partial.f32"
    try:
        _run([*_input(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "1000", "-f", "f32le", str(raw)],
             check, timeout=max(120, metadata["duration"] * 3))
        count = raw.stat().st_size // 4
        span = max(1, math.ceil(count / bins))
        peaks = []
        with raw.open("rb") as handle:
            while block := handle.read(span * 4):
                check()
                values = array.array("f")
                values.frombytes(block)
                peaks.append(round(min(1.0, max((abs(v) for v in values), default=0)), 5))
        return {"duration": metadata["duration"], "bins": peaks, "audio": True, "sample_rate": 1000,
                "measurement": "downsampled peak envelope for navigation; not a true-peak clipping measurement"}
    finally:
        raw.unlink(missing_ok=True)


def prepare(source: Path, out: Path, check: Check) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    metadata = probe(source)
    metadata["sha256"] = _hash(source)
    check()
    partial = out / "proxy.partial.mp4"
    if metadata["video"]:
        args = [*_input(source), "-map", f"0:{metadata['selected_video_stream']}", "-map", "0:a:0?", "-vf",
                "scale=w='min(960,iw)':h='min(540,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps=24,setsar=1"]
    else:
        args = [*_input(source), "-f", "lavfi", "-i", "color=c=0x15191f:s=960x540:r=24",
                "-map", "1:v:0", "-map", "0:a:0", "-t", str(metadata["duration"])]
    _run([*args, *_encode_args(27), str(partial)], check, timeout=max(300, metadata["duration"] * 12))
    os.replace(partial, out / "proxy.mp4")
    _thumb(out / "proxy.mp4", out / "thumbnail.jpg", check, metadata["duration"])
    _write_json(out / "waveform.json", _waveform(source, out, metadata, check))
    metadata["proxy"] = {"path": "proxy.mp4", "width": probe(out / "proxy.mp4")["width"], "fps": 24}
    metadata["warnings"] = (["Multiple audio streams detected; the proxy preview uses the first audio track. Export uses the timeline audio selection (first track by default)."] if metadata["audio_stream_count"] > 1 else [])
    _write_json(out / "metadata.json", metadata)
    return metadata


def _axis_expression(keyframes: list[dict], key: str, start: float, duration: float, fallback: float) -> str:
    original = sorted([(float(k["time"]), max(0.0,min(1.0,float(k[key])))) for k in keyframes])
    if not original:
        return f"{fallback:.8f}"
    def at(t):
        if t <= original[0][0]:
            return original[0][1]
        for (a,x),(b,y) in zip(original,original[1:]):
            if t < b:
                return x+(y-x)*(t-a)/(b-a)
        return original[-1][1]
    # Trimming keeps the original path at both new boundaries; it does not
    # squeeze all remaining motion into the shorter segment.
    points = [(0.0,at(start)), *[(t-start,x) for t,x in original if start<t<start+duration],
              (duration,at(start+duration))]
    expression = f"{points[-1][1]:.8f}"
    for (a, x), (b, y) in reversed(list(zip(points, points[1:]))):
        if b - a <= 0.000001:
            continue
        interpolation = f"({x:.8f}+({y-x:.8f})*(t-{a:.8f})/{b-a:.8f})"
        expression = f"if(lt(t,{b:.8f}),{interpolation},{expression})"
    return f"if(lt(t,{points[0][0]:.8f}),{points[0][1]:.8f},{expression})"


def _visual_filter(metadata: dict, segment: dict, width: int, height: int, fps: int, label: str) -> str:
    sw, sh = (metadata["width"], metadata["height"]) if metadata["video"] else (width, height)
    crop = segment.get("crop", {})
    mode = crop.get("mode", "fit")
    duration = segment["end"] - segment["start"]
    end = f",fps={fps},setsar=1,format=yuv420p[v]"
    prefix = f"{label}setpts=PTS-STARTPTS"
    if mode == "fit":
        return (prefix + f",scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x101318" + end)
    if mode == "split":
        panel = (height // 4) * 2
        cw, ch, _, _ = crop_for(sw, sh, width, panel)
        first = float(crop.get("x", 0.25))
        second = float(crop.get("second_x", 0.75))
        px = [max(0, min(sw-cw, round(sw*x-cw/2))) for x in (first, second)]
        py = max(0, min(sh-ch, round(sh*float(crop.get("y", 0.5))-ch/2)))
        return (prefix + f",split=2[a][b];[a]crop={cw}:{ch}:{px[0]}:{py},scale={width}:{panel}[top];"
                f"[b]crop={cw}:{ch}:{px[1]}:{py},scale={width}:{panel}[bottom];"
                f"[top][bottom]vstack=inputs=2,pad={width}:{height}:0:0:color=0x101318" + end)
    cw, ch, _, _ = crop_for(sw, sh, width, height)
    zoom = float(crop.get("zoom", 1))
    cw, ch = max(2, int(cw / zoom) // 2 * 2), max(2, int(ch / zoom) // 2 * 2)
    x = _axis_expression(crop.get("keyframes", []), "x", segment["start"], duration, float(crop.get("x", .5)))
    y = _axis_expression(crop.get("keyframes", []), "y", segment["start"], duration, float(crop.get("y", .5)))
    return (prefix + f",crop={cw}:{ch}:x='max(0,min({sw-cw},{sw}*({x})-{cw}/2))':"
            f"y='max(0,min({sh-ch},{sh}*({y})-{ch}/2))',scale={width}:{height}" + end)


def map_caption_words(words: list[dict], segments: list[dict]) -> list[dict]:
    """Map measured source timings through ordered edits; never infer alignment."""
    mapped = []
    offset = 0.0
    for index, segment in enumerate(segments):
        a, b = float(segment["start"]), float(segment["end"])
        for word in words:
            start, end = max(a, float(word["start"])), min(b, float(word["end"]))
            if end <= start:
                continue
            mapped.append({"word": str(word.get("word", word.get("text", ""))),
                           "start": offset + start - a, "end": offset + end - a,
                           "source_start": float(word["start"]), "source_end": float(word["end"]), "segment": index})
        offset += b - a
    return sorted(mapped, key=lambda w: (w["start"], w["end"]))


def _caption_groups(words: list[dict], width: int, height: int) -> list[list[dict]]:
    limit = 28 if height > width else 46
    groups, current = [], []
    for word in words:
        if current and (word["segment"] != current[-1]["segment"] or word["start"]-current[-1]["end"] > .65
                        or len(current) >= 7 or len(" ".join(w["word"] for w in current + [word])) > limit * 2):
            groups.append(current)
            current = []
        current.append(word)
        if re.search(r"[.!?]$", word["word"]):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    hour, cs = divmod(cs, 360000)
    minute, cs = divmod(cs, 6000)
    second, cs = divmod(cs, 100)
    return f"{hour}:{minute:02}:{second:02}.{cs:02}"


def _caption_color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise MediaError("Caption colors must be #RRGGBB values.")
    return value[5:7] + value[3:5] + value[1:3]


def _caption_text(group: list[dict], width: int, height: int, active: int | None = None,
                  color: str = "FFFFFF", highlight: str = "78ECD3") -> str:
    limit = 28 if height > width else 46
    result, line = [], 0
    for index, word in enumerate(group):
        text = ass_escape(word["word"].replace("\n", " ").replace("\r", " "))
        separator = " " if result else ""
        if line and line + len(text) + 1 > limit:
            separator, line = r"\N", 0
        if active == index:
            text = "{\\c&H" + highlight + "&}" + text + "{\\c&H" + color + "&}"
        result.append(separator + text)
        line += len(word["word"]) + 1
    return "".join(result)


def _subtitles(words: list[dict], config: dict, width: int, height: int, out: Path) -> list[str]:
    groups = _caption_groups(words, width, height)
    srt, vtt = [], ["WEBVTT\n"]
    size = int(config.get("font_size", 42))
    margin = min(int(config.get("margin", 100)), height // 3)
    alignment = int(config.get("alignment", 2))
    background = bool(config.get("background", False)) or config.get("style") == "boxed"
    font = config.get("font_family", "DejaVu Sans")
    if font not in {"DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"}:
        raise MediaError("Caption font must be an installed supported DejaVu family.")
    color = _caption_color(config.get("color", "#FFFFFF"))
    highlight = _caption_color(config.get("highlight_color", "#D3EC78"))
    stroke = float(config.get("stroke_width", 2))
    if not math.isfinite(stroke) or not 0 <= stroke <= 8:
        raise MediaError("Caption stroke width must be between 0 and 8 pixels.")
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0", "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           f"Style: Default,{font},{size},&H00{color},&H00{highlight},&H00101010,&H80000000,-1,0,0,0,100,100,0,0,{3 if background else 1},{stroke},0,{alignment},{max(16,width//16)},{max(16,width//16)},{margin},1", "", "[Events]",
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for index, group in enumerate(groups, 1):
        a, b = group[0]["start"], group[-1]["end"]
        text = " ".join(w["word"].replace("\n", " ").replace("\r", " ") for w in group)
        srt.append(f"{index}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n")
        vtt.append(f"{srt_time(a).replace(',', '.')} --> {srt_time(b).replace(',', '.')}\n{text}\n")
        if config.get("highlight"):
            cursor = a
            for word_index, word in enumerate(group):
                if word["start"] > cursor:
                    ass.append(f"Dialogue: 0,{_ass_time(cursor)},{_ass_time(word['start'])},Default,,0,0,0,,{_caption_text(group,width,height,None,color,highlight)}")
                ass.append(f"Dialogue: 0,{_ass_time(word['start'])},{_ass_time(word['end'])},Default,,0,0,0,,{_caption_text(group,width,height,word_index,color,highlight)}")
                cursor = word["end"]
        else:
            ass.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Default,,0,0,0,,{_caption_text(group,width,height,None,color,highlight)}")
    (out / "captions.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "captions.vtt").write_text("\n".join(vtt), encoding="utf-8")
    (out / "captions.ass").write_text("\n".join(ass)+"\n", encoding="utf-8")
    return (["An individual caption word exceeds the safe line width; review Unicode/font shaping and long words in the render."]
            if any(len(w["word"]) > (28 if height > width else 46) for w in words) else [])


def silence_plan(words: list[dict], start: float, end: float, gap: float = .8, fillers: bool = False) -> dict:
    """Opt-in suggestion only; caller must commit it as an explicit timeline edit."""
    keeps = plan_cuts(words, start, end, gap=gap, fillers=fillers, silence=True)
    mapped, duration = remap_words(words, start, keeps)
    return {"segments": [{"start": start+a, "end": start+b} for a, b in keeps], "words": mapped,
            "duration": duration, "method": "heuristic pauses; review before accepting", "reversible": True}


def render(source: Path, timeline: dict, out: Path, check: Check) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    metadata = probe(source)
    segments = timeline["segments"]
    output = timeline["output"]
    width, height, fps = int(output["width"]), int(output["height"]), int(output.get("fps", 30))
    total = sum(s["end"] - s["start"] for s in segments)
    if not segments or total <= 0 or total > 7200 or len(segments) > 100:
        raise MediaError("Timeline must contain 1–100 valid segments and at most 2 hours of output.")
    for segment in segments:
        if not (0 <= segment["start"] < segment["end"] <= metadata["duration"] + .05):
            raise MediaError("An edit boundary is outside the immutable source duration.")
    if width % 2 or height % 2 or not (16 <= width <= 1920 and 16 <= height <= 1920):
        raise MediaError("Output dimensions must be even and no larger than 1920 pixels.")
    # Intermediates use 48 kHz stereo signed-16 PCM: no per-cut AAC priming.
    if shutil.disk_usage(out).free < max(100 * 1024**2, total * (width * height * .15 + 48000 * 2 * 2)):
        raise MediaError("Insufficient free disk for this export. Free space or choose lower output dimensions.")
    audio = timeline.get("audio", {})
    captions = timeline.get("captions", {})
    audio_ordinal = audio.get("stream_index", 0)
    if isinstance(audio_ordinal, bool) or not isinstance(audio_ordinal, int) or audio_ordinal < 0:
        raise MediaError("Selected audio stream must be a zero-based audio track index.")
    if audio_ordinal >= metadata["audio_stream_count"] and (metadata["audio"] or audio_ordinal != 0):
        raise MediaError(f"Selected audio track {audio_ordinal} is unavailable; source has {metadata['audio_stream_count']} audio tracks.")
    work = out / "parts"
    work.mkdir(exist_ok=True)
    started = time.monotonic()
    for index, segment in enumerate(segments):
        check()
        duration = segment["end"] - segment["start"]
        args = ["-ss", str(segment["start"]), *_input(source)]
        video_label, audio_label = f"[0:{metadata['selected_video_stream']}]", f"[0:a:{audio_ordinal}]"
        if not metadata["video"]:
            args += ["-f", "lavfi", "-i", f"color=c=0x15191f:s={width}x{height}:r={fps}"]
            video_label = "[1:v:0]"
        if not metadata["audio"]:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            audio_label = "[1:a:0]"
        graph = _visual_filter(metadata, segment, width, height, fps, video_label)
        af = f"{audio_label}aresample=48000:first_pts=0,apad,atrim=duration={duration:.8f},asetpts=PTS-STARTPTS,volume={float(segment.get('gain_db',0)):.4f}dB"
        fade = min(float(audio.get("fade", 0)), duration / 2)
        if fade > 0:
            af += f",afade=t=in:d={fade:.5f},afade=t=out:st={duration-fade:.5f}:d={fade:.5f}"
        args += ["-filter_complex", graph + ";" + af + "[a]", "-map", "[v]", "-map", "[a]",
                 "-t", f"{duration:.8f}", *_video_encode_args(int(output.get("crf", 22))),
                 "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(work / f"segment-{index:04}.mkv")]
        _run(args, check, timeout=max(180, duration * 40))
    listing = work / "concat.txt"
    listing.write_text("".join(f"file 'segment-{i:04}.mkv'\nduration {s['end']-s['start']:.8f}\n" for i,s in enumerate(segments)), encoding="utf-8")
    assembled = out / "assembled.mp4"
    # Concat starts each input at the requested timeline boundary, independent
    # of rounded container/frame durations. Apply output frame cadence once
    # globally and encode continuous audio once, rather than joining AAC padding.
    _run(["-f", "concat", "-safe", "1", "-i", str(listing.resolve()),
          "-vf", f"setpts=PTS-STARTPTS,fps={fps}",
          "-af", f"aresample=48000:async=1:first_pts=0,apad,atrim=duration={total:.8f}", "-t", f"{total:.8f}",
          *_encode_args(int(output.get("crf",22))), str(assembled)], check, timeout=max(180,total*40))
    clean = out / "clean.mp4"
    if audio.get("normalize"):
        _run([*_input(assembled), "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(clean)], check, timeout=max(180,total*10))
        assembled.unlink()
    else:
        os.replace(assembled, clean)
    words = map_caption_words(captions.get("words", []), segments) if captions.get("enabled", True) else []
    warnings = _subtitles(words, captions, width, height, out)
    partial = out / "video.partial.mp4"
    if captions.get("burn_in") and words:
        _run([*_input(clean), "-vf", "ass=captions.ass", *_video_encode_args(int(output.get("crf",22))),
              "-c:a", "copy", "-movflags", "+faststart", str(partial.resolve())], check, cwd=out, timeout=max(180,total*40))
    else:
        os.replace(clean, partial)
        if captions.get("burn_in") and not words:
            warnings.append("Caption burn-in requested but no timed words are present; clean video exported.")
    info = probe(partial)
    duration_error = abs(info["duration"]-total)
    tolerance = .15
    if (info["width"], info["height"]) != (width,height) or duration_error > tolerance or abs((info["fps"] or 0)-fps) > .001:
        raise MediaError(f"Export verification failed: dimensions/duration mismatch ({duration_error:.3f}s error; tolerance {tolerance:.3f}s).")
    # The full output is decoded. volumedetect measures decoded sample peak,
    # not true peak; this distinction is explicit in the manifest.
    audio_log = _run([*_input(partial), "-af", "volumedetect", "-f", "null", "-"], check,
                     timeout=max(180,total*12), loglevel="info")
    peak_match = re.search(r"max_volume:\s*(-?[0-9.]+) dB", audio_log)
    peak = float(peak_match.group(1)) if peak_match else None
    if peak is not None and peak >= 0:
        warnings.append("Decoded audio reaches 0 dBFS; reduce gain and review possible clipping.")
    os.replace(partial, out / "video.mp4")
    _thumb(out / "video.mp4", out / "thumbnail.jpg", check, info["duration"])
    manifest = {"schema_version": 1, "created_at": time.time(), "source_sha256": _hash(source),
                "timeline_sha256": hashlib.sha256(json.dumps(timeline,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
                "timeline": timeline, "duration": info["duration"], "expected_duration": total,
                "width": width, "height": height, "fps": fps, "video": "video.mp4", "package": "package.zip", "thumbnail": "thumbnail.jpg",
                "verification": {"full_decode": "passed", "dimensions": "passed", "duration_error_seconds": round(duration_error,6),
                                 "duration_tolerance_seconds": tolerance, "audio_present": info["audio"],
                                 "sample_peak_dbfs": peak, "true_peak_measured": False,
                                 "audio_sync": "timestamp-normalized per segment; no independent perceptual sync rating"},
                "captions": {"words": len(words), "alignment": "supplied source word timestamps" if words else "none", "burned": bool(captions.get("burn_in") and words),
                             "font": "system " + captions.get("font_family", "DejaVu Sans"), "color": captions.get("color", "#FFFFFF"),
                             "highlight_color": captions.get("highlight_color", "#D3EC78"), "stroke_width": captions.get("stroke_width", 2),
                             "language_limitations": "Unicode shaping and font coverage require review; no claim all languages tested"},
                "audio": {"missing_source_audio": not metadata["audio"], "normalize_requested": bool(audio.get("normalize")), "normalization_target_lufs": -14 if audio.get("normalize") else None},
                "timing": "PCM intermediates, explicit source-edit concat durations, global constant frame cadence; no per-cut AAC padding",
                "runtime_seconds": round(time.monotonic()-started,4), "warnings": warnings, "files": {}}
    audio_streams = [s for s in metadata["streams"] if s.get("codec_type") == "audio"]
    manifest["audio"].update(selected_audio_ordinal=audio_ordinal,
                             source_stream_index=audio_streams[audio_ordinal]["index"] if audio_streams else None,
                             preview_audio="Source proxy uses first audio track; selected track and processing are authoritative in the export.")
    for name in ["video.mp4", "clean.mp4", "captions.srt", "captions.vtt", "captions.ass", "thumbnail.jpg"]:
        path = out / name
        if path.exists():
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": _hash(path)}
    _write_json(out / "manifest.json", manifest)
    check()
    with zipfile.ZipFile(out / "package.partial.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in [*manifest["files"], "manifest.json"]:
            check()
            archive.write(out / name, name)
    with zipfile.ZipFile(out / "package.partial.zip") as archive:
        if archive.testzip() is not None:
            raise MediaError("Export package failed ZIP integrity verification.")
    os.replace(out / "package.partial.zip", out / "package.zip")
    shutil.rmtree(work)
    return manifest


def track(source: Path, options: dict, out: Path, check: Check) -> dict:
    """Assisted face detector or user-selected template tracking, never diarization."""
    import cv2
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    metadata = probe(source)
    if not metadata["video"]:
        raise MediaError("Assisted framing needs a video stream. Audio-only projects use a plain background.")
    start = max(0.0, float(options.get("start", 0)))
    end = min(metadata["duration"], float(options.get("end", metadata["duration"])))
    if end <= start or end-start > 600:
        raise MediaError("Track a selected interval of at most 10 minutes at a time.")
    mode = options.get("mode", "face")
    if mode not in {"face", "object"}:
        raise MediaError("Tracking mode must be face or object.")
    proxy = out / "tracking.mp4"
    _run(["-ss", str(start), *_input(source), "-map", f"0:{metadata['selected_video_stream']}", "-t", str(end-start), "-vf", "fps=3,scale=640:-2", "-an",
          "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-threads", "2", str(proxy)], check, timeout=max(180,(end-start)*8))
    cap = cv2.VideoCapture(str(proxy))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if mode == "face" else None
    x, y = float(options.get("x", .5)), float(options.get("y", .5))
    box_width, box_height = float(options.get("width", .2)), float(options.get("height", .3))
    if mode == "object":
        # The request describes a normalized top-left rectangle; generated
        # timeline keyframes use normalized subject centers.
        x, y = x + box_width/2, y + box_height/2
    template, previous, samples, index = None, None, [], 0
    try:
        while True:
            check()
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cut = previous is not None and float(np.mean(cv2.absdiff(gray,previous))) > 45
            confidence = 0.0
            found = False
            if mode == "face":
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(24,24))
                if len(faces):
                    bx, by, bw, bh = min(faces, key=lambda b: ((b[0]+b[2]/2)/w-x)**2 + ((b[1]+b[3]/2)/h-y)**2)
                    x, y = (bx+bw/2)/w, (by+bh/2)/h
                    confidence, found = .65, True
            else:
                if template is None and index == 0:
                    bw,bh = max(8,int(box_width*w)),max(8,int(box_height*h))
                    bx,by = max(0,min(w-bw,int(x*w-bw/2))),max(0,min(h-bh,int(y*h-bh/2)))
                    template = gray[by:by+bh,bx:bx+bw].copy()
                    confidence,found = 1.0,True
                elif cut:
                    # Template identity after a cut is unknown: hold framing.
                    template = None
                elif template is not None and float(template.std()) > 1:
                    response = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
                    _, confidence, _, location = cv2.minMaxLoc(response)
                    if confidence >= .65:
                        x=(location[0]+template.shape[1]/2)/w
                        y=(location[1]+template.shape[0]/2)/h
                        found=True
            samples.append({"time": round(start+index/3,4), "x": round(x,6), "y": round(y,6),
                            "confidence": round(max(0,float(confidence)),4), "detected": found, "scene_cut": bool(cut)})
            previous=gray
            index+=1
    finally:
        cap.release()
        proxy.unlink(missing_ok=True)
    # Smooth separately around cuts, preserving resets. Source smoothing uses
    # geometry-safe bounds; a 0.1 normalized crop width retains a 5% safe edge.
    keyframes=[]
    group=[]
    for sample in samples + [None]:
        if sample is None or sample["scene_cut"]:
            if group:
                any_detected=any(s["detected"] for s in group)
                xs=smooth_track([(s["time"],s["x"] if s["detected"] or not any_detected else None) for s in group],1.0,.1)
                ys=smooth_track([(s["time"],s["y"] if s["detected"] or not any_detected else None) for s in group],1.0,.1)
                keyframes.extend({"time":t,"x":round(x,6),"y":round(ys[i][1],6)} for i,(t,x) in enumerate(xs))
            group=[]
        if sample is not None:
            group.append(sample)
    detected=sum(s["detected"] for s in samples)
    result={"schema_version":1,"mode":mode,"method":"OpenCV Haar face detector" if mode=="face" else "user-selected normalized template correlation",
            "status":"assisted" if detected else "fallback", "keyframes":keyframes, "samples":samples,
            "confidence":round(detected/max(1,len(samples)),4), "confidence_kind":"fraction of sampled frames with a matched detection; not calibrated semantic confidence",
            "fallback":"hold/center framing when detection is absent; manually review and lock framing if wrong",
            "request_coordinates":"object ROI: normalized top-left x/y plus width/height; keyframes: normalized centers",
            "limitations":["Not audio diarization or audiovisual active-speaker association.","Object matching is visual template tracking, not semantic recognition.","Scene-cut template loss requires choosing the object again."]}
    _write_json(out / "tracking.json",result)
    return result
