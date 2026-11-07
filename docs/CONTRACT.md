# Internal v1 contract (implementation target)

Single-source projects. Source seconds are floating-point, finite and bounded. Output is a concatenation of ordered source segments. Edits are immutable numbered revisions; API uses optimistic revision checks. Caption word times and crop keyframes are SOURCE times; export maps them through segments. No invented word alignment.

Timeline JSON:
```
{"schema_version":1,"segments":[{"id":"s1","start":0,"end":10,"crop":{"mode":"fit","x":0.5,"y":0.5,"zoom":1,"keyframes":[],"second_x":0.75},"gain_db":0}],"output":{"width":720,"height":1280,"fps":30,"crf":22},"captions":{"enabled":true,"burn_in":false,"style":"clean","font_size":42,"margin":100,"alignment":2,"background":false,"highlight":false,"words":[]},"audio":{"normalize":false,"fade":0}}
```
Crop modes fit/crop/split. keyframes [{time,x,y}] normalized center. Fonts are installed DejaVu Sans, Serif or Sans Mono; colors, stroke and highlight colors are configurable. `audio.stream_index` selects the zero-based audio ordinal for export, and analysis/transcription `audio_stream` selects its evidence track. Supported output ratios 9:16,1:1,4:5,16:9. All numeric ranges server-validated. Width/height even <=1920; output total <=2h; segment count <=100. Text never interpreted as instructions/code.

Media service Python functions in media.py:
- `probe(path: Path) -> dict` duration,width,height,fps,streams,rotation,size, audio bool, raw ffprobe subset.
- `prepare(source: Path, out: Path, check: Callable[[],None]) -> dict` makes proxy.mp4, thumbnail.jpg, waveform.json (bounded bins), metadata.json; returns metadata. Called only in durable subprocess, out is private stage work dir.
- `render(source: Path, timeline: dict, out: Path, check) -> dict` outputs video.mp4, clean.mp4 (if burned selected), captions.srt/vtt, thumbnail.jpg, manifest.json, package.zip. Streams verified/decode. Returns manifest. Source immutable. `check()` raises cancellation/lease error. Use FFmpeg arrays, bounded threads, timeout and cancel polling, clean subprocess groups.
- `track(source: Path, options: dict, out: Path, check) -> dict` optional assisted tracking result keyframes/confidence with honest fallback.

Analysis service Python functions in analysis.py:
- `transcribe(source: Path, out: Path, options: dict, check) -> dict` writes transcript.json with {segments:[{start,end,text}], words:[{start,end,word}], provider,model,revision,alignment,status,limitations}. missing audio -> empty valid result, not fabricated success. Offline model only; explicit download separate.
- `analyze(source: Path, metadata: dict, transcript: dict|None, options: dict, out: Path, check) -> dict` writes analysis.json {candidates:[{id,start,end,title,rationale,evidence:[...],confidence,uncertainty,scores,method}], provider/model/revision,limitations}. Options genre speech/tutorial/action/mixed, mode local/heuristic/agent/cloud, query, min_length,max_length,target_count,min_separation, exclusions [{start,end}]. Local semantic model evidence + transparent heuristics; no forced count. Default lengths adapt to 20s fixtures via requested 4..10.
- `model_status() -> dict` cached availability, pins,size, capabilities. Never implicitly download.
- `download_model(name: str, out: Path, check) -> dict` explicit job. Small bounded catalog, progress/cancellation, license provenance.

Root owns config.py, schemas.py, db.py, service.py, api.py, worker.py, task.py, cli.py; agents MUST NOT edit those.

API prefix /api/v1. Same-origin UI; default port 8741, loopback only.
GET /health /capabilities /projects?q= /projects/{id} /jobs /jobs/{id} /presets /publications
POST /imports multipart file,title,genre,attribution,permission_status -> {project_id,job}; POST /imports/url JSON {url,title,genre,...} -> queued URL download. POST /fixtures/{name} for named catalog fixture import. GET /fixtures lists entries.
GET project -> {id,title,tags,genre,status,source:{... metadata,sha256,original_name,provenance},timeline,revision,transcript,analysis,exports:[...],jobs:[...]}
PUT /projects/{id}/timeline JSON {revision,timeline} -> {revision,timeline}; optimistic conflict 409.
PATCH /projects/{id} {title,tags,genre,permission_status,attribution}
POST /projects/{id}/analyze JSON analysis options + transcribe bool -> job
POST /projects/{id}/transcribe JSON {model:"tiny.en"} -> job
PUT /projects/{id}/transcript {words,segments,revision} -> persisted correction; corrections are not overwritten automatically.
POST /projects/{id}/render JSON {revision} -> job; Idempotency-Key supported.
POST /projects/{id}/track JSON {mode:"face"|"object",start,end,x,y,width,height} -> job
POST /projects/{id}/feedback JSON {candidate_id,decision:"accept"|"reject",note}
POST /projects/{id}/event {time,before,after,title} -> manual candidate
GET /projects/{id}/evidence -> timestamped agent packet. POST /projects/{id}/agent-decisions -> schema-validated proposals bound to packet fingerprint (no execution).
POST /jobs/{id}/cancel /jobs/{id}/retry
GET /artifacts/{project_id}/{artifact_path:path} only registered project directory, no traversal/symlinks. Source original accessible only its own project.
POST /presets {name,timeline}; GET /presets
POST /batch/render {items:[{project_id,revision}],idempotency_key} -> per-item results, <=20
POST /publications {project_id,export_id,destination,account,caption,visibility,schedule,timezone,permission_status} -> draft only. Manual handoff; posting unavailable without a configured supported adapter.
GET /audit?project_id=; GET /exports/{id}/download via registered artifact package.
```
job = {id,project_id,kind,status,stage,created_at,updated_at,started_at,elapsed_seconds,attempts,max_attempts,error,result,cancel_requested}
```
Stage status is honest queued/running/succeeded/failed/cancelled. No fake percent. Artifact hrefs included by API. Frontend can poll every 2s while active. Modal errors actionable and retained. API docs /docs + /openapi.json.
