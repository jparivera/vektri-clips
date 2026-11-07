#!/usr/bin/env python3
"""Serial synthetic delivery benchmark against an already running local API.

Run deliberately with: .venv/bin/python scripts/benchmark_delivery.py
Creates four named projects; never downloads models, posts media, or clears data.
Failure preserves partial evidence and exits nonzero. This is not a human quality
or real-world genre evaluation. Use the repository's locked Python environment.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from urllib.parse import quote, urlparse
import uuid
import zipfile

import httpx
import psutil


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = [
    ("spoken_interview.mp4", "speech", "Find a complete explanation of keeping context around a highlight."),
    ("tutorial_screen.mp4", "tutorial", "Find the clearest complete explanation of the demonstrated process."),
    ("action_motion.mp4", "action", ""),
    ("mixed_nonverbal.mp4", "mixed", ""),
]
FORMATS = "mov,matroska,webm,avi,wav,mp3,ogg,flac,aac,asf,mpeg,mpegts,aiff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit(event: str, **details) -> None:
    print(json.dumps({"event": event, **details}, ensure_ascii=False), flush=True)


def local_artifact(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/api/v1/artifacts/") or "?" in path or "#" in path:
        raise ValueError("API returned an unexpected artifact URL; no external request was made")
    return path


def download(client: httpx.Client, url: str, destination: Path | None = None) -> dict:
    temporary = destination.with_suffix(destination.suffix + ".partial") if destination else None
    digest, size = hashlib.sha256(), 0
    handle = temporary.open("wb") if temporary else None
    try:
        with client.stream("GET", local_artifact(url), timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 256 * 1024**2:
                    raise ValueError("Synthetic benchmark artifact exceeded the 256 MiB download limit")
                digest.update(chunk)
                if handle:
                    handle.write(chunk)
        if handle:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, destination)
        return {"sha256": digest.hexdigest(), "bytes": size}
    finally:
        if handle and not handle.closed:
            handle.close()
        if temporary:
            temporary.unlink(missing_ok=True)


def request(client: httpx.Client, method: str, path: str, payload=None, key=None):
    response = client.request(method, path, json=payload, headers={"Idempotency-Key": key} if key else None)
    if response.is_error:
        raise RuntimeError(f"{method} {path}: HTTP {response.status_code}: {response.text[:1200]}")
    return response.json()


def job_metrics(job: dict, observed: float) -> dict:
    result = job.get("result") or {}
    created, started = job.get("created_at"), job.get("started_at")
    return {
        "id": job["id"], "kind": job["kind"], "status": job["status"], "stage": job.get("stage"),
        "attempts": job.get("attempts"), "error": job.get("error"),
        "job_elapsed_seconds": job.get("elapsed_seconds"),
        "queue_wait_seconds": round(started - created, 4) if started and created else None,
        "client_observed_seconds": round(observed, 4),
        "runtime_self": {"elapsed_seconds": result.get("elapsed_seconds"), "peak_rss_kib": result.get("peak_rss_kib")},
        "runtime_children": {"peak_rss_kib": result.get("child_peak_rss_kib")},
        "reused_stages": result.get("reused_stages", []),
    }


def poll(client: httpx.Client, job: dict, record: dict, checkpoint, timeout: float) -> dict:
    started, last_state = time.monotonic(), None
    try:
        while True:
            current = request(client, "GET", "/api/v1/jobs/" + job["id"])
            record.clear()
            record.update(job_metrics(current, time.monotonic() - started))
            state = (current["status"], current.get("stage"))
            if state != last_state:
                emit("job", **record)
                checkpoint()
                last_state = state
            if current["status"] == "succeeded":
                return current
            if current["status"] in {"failed", "cancelled"}:
                raise RuntimeError(f"Job {current['id']} {current['status']}: {current.get('error')}")
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Job {current['id']} exceeded the {timeout:g}s benchmark wait limit")
            time.sleep(0.5)
    except (TimeoutError, KeyboardInterrupt):
        # Only the job created by this benchmark is cancelled; other work is untouched.
        try:
            request(client, "POST", "/api/v1/jobs/" + job["id"] + "/cancel")
            record["benchmark_cancel_requested"] = True
        except Exception as error:
            record["cancellation_error"] = str(error)[:500]
        checkpoint()
        raise


def verify_output(video: Path, package: Path, manifest: dict, expected_duration: float) -> dict:
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                            "-show_format", "-show_streams", "-of", "json", str(video)],
                           capture_output=True, text=True, check=True, timeout=30)
    media = json.loads(probe.stdout)
    video_stream = next(stream for stream in media["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in media["streams"] if stream["codec_type"] == "audio")
    duration = float(media["format"]["duration"])
    tolerance = float(manifest["verification"]["duration_tolerance_seconds"])
    if (video_stream["width"], video_stream["height"]) != (360, 640):
        raise ValueError("Downloaded output dimensions differ from the saved timeline")
    if abs(duration - expected_duration) > tolerance:
        raise ValueError("Downloaded output duration exceeds its measured export tolerance")
    if video_stream["codec_name"] != "h264" or audio_stream["codec_name"] != "aac":
        raise ValueError("Output does not contain the requested H.264/AAC streams")
    decoded = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-threads", "2",
                              "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                              "-i", str(video), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=120)
    if decoded.returncode or decoded.stderr.strip():
        raise ValueError("Downloaded video failed full decode: " + decoded.stderr[-1500:])
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise ValueError("Downloaded package CRC verification failed")
        packaged_manifest = json.loads(archive.read("manifest.json"))
        if packaged_manifest != manifest:
            raise ValueError("Package manifest differs from the registered export manifest")
        for name, facts in manifest["files"].items():
            if name not in archive.namelist():
                raise ValueError(f"Manifest artifact {name} is missing from the package")
            with archive.open(name) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != facts["sha256"]:
                raise ValueError(f"Packaged artifact hash mismatch: {name}")
        names = archive.namelist()
    if sha256(video) != manifest["files"]["video.mp4"]["sha256"]:
        raise ValueError("Downloaded video hash differs from its manifest")
    return {"independent_full_decode": "passed", "package_crc": "passed", "package_artifact_hashes": "passed",
            "package_files": names, "duration_seconds": duration, "expected_duration_seconds": expected_duration,
            "duration_error_seconds": round(abs(duration - expected_duration), 6), "duration_tolerance_seconds": tolerance,
            "width": video_stream["width"], "height": video_stream["height"],
            "streams": [{key: stream.get(key) for key in ("index", "codec_type", "codec_name", "duration", "start_time", "time_base", "avg_frame_rate", "sample_rate", "channels")}
                        for stream in media["streams"]], "perceptual_sync": "unrated; see separate synthetic marker regression"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8741")
    parser.add_argument("--job-timeout", type=float, default=600)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence")
    args = parser.parse_args()
    address = urlparse(args.base_url)
    if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"} or address.username or address.password:
        parser.error("This benchmark only operates an intentionally running loopback HTTP API")
    if not math.isfinite(args.job_timeout) or not 1 <= args.job_timeout <= 3600:
        parser.error("--job-timeout must be 1–3600 seconds")
    evidence = args.evidence_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = evidence / "delivery-benchmark-runs" / run_id
    run_dir.mkdir(parents=True)
    report_path = evidence / "delivery-benchmark.json"
    if report_path.exists():
        shutil.copy2(report_path, run_dir / "previous-report.json")
    report = {"schema_version": 1, "run_id": run_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "api": args.base_url, "serial": True, "cases": [],
              "scope": "Deterministic synthetic mechanics and genuine local inference; no real-world or human quality claim.",
              "limitations": ["Action and nonverbal selection uses honest visual/audio heuristics, not semantic event understanding.",
                              "Self and child ru_maxrss are Linux KiB high-water marks, not summed concurrent process-tree RAM.",
                              "Models may be cached on disk; each heavy job runs in a separate worker subprocess.",
                              "No model downloads, paid providers, external agent calls or publishing operations are performed."],
              "subjective_quality": {"hook_context_payoff": "unrun", "real_world_genres": "unrun", "human_caption_readability": "unrun"},
              "hardware": {"platform": platform.platform(), "cpu_logical": psutil.cpu_count(),
                           "ram_bytes": psutil.virtual_memory().total, "available_ram_at_start_bytes": psutil.virtual_memory().available,
                           "free_disk_at_start_bytes": shutil.disk_usage(evidence).free}}
    started = time.monotonic()

    def checkpoint():
        report["elapsed_wall_seconds"] = round(time.monotonic() - started, 4)
        save_json(report_path, report)
        save_json(run_dir / "report.json", report)

    checkpoint()
    try:
        with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=30, follow_redirects=False, trust_env=False) as client:
            health = request(client, "GET", "/api/v1/health")
            if health.get("worker_active") or health.get("queue_length"):
                raise RuntimeError("Other jobs are active or queued. Run this serial benchmark after they finish; none were changed.")
            capabilities = request(client, "GET", "/api/v1/capabilities")
            model_state = capabilities["models"]["models"]
            report["models_at_start"] = [{k: model.get(k) for k in ("id", "status", "available", "runtime_available", "repository", "revision", "license")}
                                         for model in model_state]
            for required in ("tiny.en", "minilm"):
                if not any(model["id"] == required and model.get("available") and model.get("runtime_available") for model in model_state):
                    raise RuntimeError(f"Required local model {required} is not ready; benchmark did not download or simulate it")
            catalog = {item["name"] for item in request(client, "GET", "/api/v1/fixtures")}
            fixture_manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
            expected = {item["filename"]: item["sha256"] for item in fixture_manifest["fixtures"]}
            report["fixture_manifest_sha256"] = sha256(ROOT / "fixtures/manifest.json")
            for filename, genre, query in FIXTURES:
                case = {"fixture": filename, "genre": genre, "synthetic": True, "status": "running", "jobs": []}
                report["cases"].append(case)
                checkpoint()
                if filename not in catalog:
                    raise RuntimeError(f"Fixture {filename} is absent from the live API catalog")
                before = sha256(ROOT / "fixtures" / filename)
                if before != expected[filename]:
                    raise ValueError(f"Fixture {filename} differs from its provenance manifest")
                prefix = f"delivery:{run_id}:{genre}:"
                accepted = request(client, "POST", "/api/v1/fixtures/" + quote(filename, safe=""), key=prefix + "import")
                pid = case["project_id"] = accepted["project_id"]
                job_record = {}; case["jobs"].append(job_record)
                poll(client, accepted["job"], job_record, checkpoint, args.job_timeout)
                request(client, "PATCH", f"/api/v1/projects/{pid}", {"title": "Delivery benchmark · " + genre,
                        "tags": ["synthetic", "delivery-benchmark", run_id], "genre": genre})
                project = request(client, "GET", f"/api/v1/projects/{pid}")
                source = project["source"]
                imported = download(client, source["original_url"])
                if source["sha256"] != before or imported["sha256"] != before:
                    raise ValueError("Imported original differs from the synthetic source")
                case["source"] = {"sha256": before, "imported_sha256": imported["sha256"], "bytes": imported["bytes"],
                                  "duration_seconds": source["metadata"]["duration"], "provenance": source["provenance"]}
                # Nonverbal fixtures do not need invented speech or fabricated captions.
                options = {"mode": "local", "genre": genre, "query": query, "min_length": 3, "max_length": 10,
                           "target_count": 3, "min_separation": 1, "transcribe": genre in {"speech", "tutorial"},
                           "audio_stream": 0, "use_feedback": False}
                queued = request(client, "POST", f"/api/v1/projects/{pid}/analyze", options, prefix + "analyze")
                job_record = {}; case["jobs"].append(job_record)
                analyzed = poll(client, queued, job_record, checkpoint, args.job_timeout)
                project = request(client, "GET", f"/api/v1/projects/{pid}")
                analysis = project["analysis"]
                transcript = project.get("transcript") or {}
                candidates = analysis.get("candidates", [])
                case["analysis"] = {key: analysis.get(key) for key in ("status", "mode", "provider", "model", "revision", "ai_used", "elapsed_seconds", "paid_usage", "data_transmitted", "limitations")}
                case["analysis"].update(request=options, candidate_count=len(candidates), methods=sorted({c["method"] for c in candidates}), candidates=candidates)
                case["transcript"] = {key: transcript.get(key) for key in ("status", "provider", "model", "revision", "alignment", "configuration", "limitations")}
                case["transcript"]["word_count"] = len(transcript.get("words", []))
                if genre in {"speech", "tutorial"} and (not analysis.get("ai_used") or not transcript.get("words")):
                    raise ValueError("Spoken fixture did not produce real local ASR and semantic model results")
                chosen = max(candidates, key=lambda item: item.get("scores", {}).get("ranking", 0)) if candidates else None
                interval = {"start": chosen["start"], "end": chosen["end"]} if chosen else {"start": 2, "end": min(6, source["metadata"]["duration"])}
                case["selection"] = {"method": "highest returned ranking" if chosen else "manual fallback because analysis returned no candidates",
                                     "candidate_id": chosen["id"] if chosen else None, **interval, "human_reviewed": False}
                timeline = copy.deepcopy(project["timeline"])
                timeline["segments"] = [{"id": "delivery-clip", **interval, "crop": {"mode": "fit", "x": 0.5, "y": 0.5, "zoom": 1,
                                         "second_x": 0.75, "keyframes": []}, "gain_db": 0}]
                timeline["output"] = {"width": 360, "height": 640, "fps": 24, "crf": 22}
                # The saved timeline already contains validated editable words;
                # provider probabilities belong to transcript provenance only.
                timeline["captions"].update(enabled=True, burn_in=bool(timeline["captions"]["words"]),
                                            font_family="DejaVu Sans", font_size=26, margin=48, alignment=2, style="clean",
                                            background=False, highlight=True, color="#FFFFFF", highlight_color="#D3EC78", stroke_width=2)
                timeline["audio"] = {"stream_index": 0, "normalize": False, "fade": 0}
                saved = request(client, "PUT", f"/api/v1/projects/{pid}/timeline", {"revision": project["revision"], "timeline": timeline})
                case["revision"] = saved["revision"]
                queued = request(client, "POST", f"/api/v1/projects/{pid}/render", {"revision": saved["revision"]}, prefix + "render")
                job_record = {}; case["jobs"].append(job_record)
                rendered = poll(client, queued, job_record, checkpoint, args.job_timeout)
                export = rendered["result"]["export"]
                manifest = export["manifest"]
                case_dir = run_dir / genre; case_dir.mkdir()
                video, package = case_dir / "video.mp4", case_dir / "package.zip"
                video_facts = download(client, export["video_url"], video)
                package_facts = download(client, export["package_url"], package)
                verified = verify_output(video, package, manifest, interval["end"] - interval["start"])
                after = download(client, source["original_url"])["sha256"]
                if after != before or sha256(ROOT / "fixtures" / filename) != before or manifest["source_sha256"] != before:
                    raise ValueError("Source integrity changed during analysis/render")
                case["source"]["integrity_after_render"] = "passed"
                case["export"] = {"id": export["id"], "video_url": export["video_url"], "package_url": export["package_url"],
                                  "local_video": str(video.relative_to(evidence)), "local_package": str(package.relative_to(evidence)),
                                  "video": video_facts, "package": package_facts, "verification": verified,
                                  "manifest": manifest}
                if genre == "speech":
                    for source_path, final_name in ((video, "sample-clip.mp4"), (package, "sample-clip.zip")):
                        temporary = evidence / (final_name + ".partial")
                        shutil.copyfile(source_path, temporary)
                        os.replace(temporary, evidence / final_name)
                    report["sample"] = {"project_id": pid, "export_id": export["id"], "video": "sample-clip.mp4", "package": "sample-clip.zip",
                                        "video_sha256": video_facts["sha256"], "package_sha256": package_facts["sha256"], "synthetic": True}
                case["status"] = "passed"
                checkpoint()
                emit("case_passed", fixture=filename, project_id=pid, candidate_count=len(candidates), ai_used=analysis.get("ai_used"), duration=verified["duration_seconds"])
            report["status"] = "passed"
            report["paid_usage"] = sum(case["analysis"]["paid_usage"] for case in report["cases"])
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint()
            emit("benchmark_passed", report=str(report_path), cases=len(report["cases"]), elapsed_seconds=report["elapsed_wall_seconds"])
            return 0
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
        if report["cases"] and report["cases"][-1]["status"] == "running":
            report["cases"][-1]["status"] = report["status"]
        report["unrun_fixtures"] = [name for name, _, _ in FIXTURES if name not in {case["fixture"] for case in report["cases"]}]
        checkpoint()
        emit("benchmark_failed", report=str(report_path), error=report["error"])
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
