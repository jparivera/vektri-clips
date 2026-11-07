# Storage, recovery and local operation

Default data directory: `data/` inside the checkout, or the explicit `VEKTRI_DATA_DIR`. Keep this directory private. The SQLite database stores projects, timeline revisions, durable jobs/stages, exports, presets, feedback, publication drafts/approvals, upload offsets and audit events. `projects/<id>/source/original.bin` is the copied original; `artifacts/` separates prepared media, transcripts, analysis, OCR evidence, tracking, encoded renders and version-bound packages. `models/` stores explicitly installed models. The original's checksum is rechecked before expensive processing. External original files are never altered.

The audit table is append-only. The **Audit trail** page shows it newest first with an event-prefix filter, a project filter, a one-line summary per entry, the full JSON record on demand and "Show older entries" paging; the same view is `GET /api/v1/audit?event=job.&project_id=ID&limit=100&before=SEQ`. There is no edit or delete operation for audit rows in the API or the UI.

Database migration 1 is in `backend/clipstudio/db.py`. Startup applies supported migrations transactionally; later unknown schema versions fail rather than downgrade. Back up before application updates. Source code and dependency locks belong in Git; media, model weights, database and secrets do not. Keep recordings, generated frames, results, model weights, runtime state and credentials out of Git.

## Run and stop

```bash
cd /path/to/vektri-clips
./scripts/start.sh
```

Ctrl-C stops the foreground API and worker. Default URL: `http://127.0.0.1:8741`. If you configure your own service manager, ensure it uses the same private `VEKTRI_DATA_DIR` for both components. Only one active heavy job is allowed per database.

## Recovery design

`BEGIN IMMEDIATE` serializes claims. A random lease token fences stage and final commits; heartbeats extend a 20-second lease. Expired work becomes recoverable with bounded retries (default three attempts). Linux parent-death handling and process-group cleanup stop orphaned decoders/renderers, and the serve-managed worker itself exits when its supervising `clipstudio serve` process is gone (PR_SET_PDEATHSIG plus a parent-PID check). Cancellation terminates running child work and leaves source media intact.

Stages write private temporary directories, then atomically replace verified output directories. Cache fingerprints include source SHA, relevant edit/config values, model revision and implementation-file hashes. Completed reusable stages avoid unnecessary ASR/encoding after recovery. Encoding and packaging are separate: identical pixels can be reused while a new revision gets its own provenance manifest. At-least-once execution can repeat interrupted work; no exactly-once claim is made. Never edit database rows manually to force a successful status.

The worker waits for at least 1.5 GiB available RAM before claiming heavy work. This is a start guard, not a guaranteed memory ceiling. One heavy job, bounded inference threads and proxy/frame sampling keep the baseline conservative. Whisper uses int8, CLIP has int8/uint8 towers, while X-CLIP uses FP32 CPU inference; do not assume every model is int8. The first guarded X-CLIP run used about 686 MiB peak RSS, which is a measured run rather than a guaranteed ceiling. OCR is bounded to 120 frames and checks the RAM guard between frames. Close other memory-heavy applications if jobs wait. No GPU acceleration was certified. Long human recordings and large libraries have not been stress tested.

## Backup, restore and retention

Finish/cancel queued jobs and finish resumable uploads, then stop the service. `clipstudio storage` reports usage without deleting files. A backup refuses active/pending work and unfinished uploads rather than silently omit them.

```bash
.venv/bin/clipstudio backup /safe/location/vektri-backup.tar.gz
.venv/bin/clipstudio restore /safe/location/vektri-backup.tar.gz /new/empty-library
VEKTRI_DATA_DIR=/new/empty-library ./scripts/start.sh
```

Backups include SQLite metadata, originals, artifacts and models stored inside the data directory, with checksums. An externally reused Hugging Face cache is not copied; on another machine explicitly install the pinned model if absent. Restore requires a new destination, rejects traversal, links, duplicate/unexpected archive entries and insufficient space, then verifies file hashes and database integrity. A failed restore does not overwrite an existing library. No backup encryption is built in; use encrypted storage where appropriate. Test restore before relying on a backup.

Retention planning is read-only. `plan-storage` inventories protected paths, retained caches and advisory review candidates; `verify-storage-plan` refreshes their metadata and protecting database references. It preserves source/model paths, current and referenced revisions/exports/publications, active jobs/uploads and other protected artifacts. Hardlinks are counted conservatively, and symlinks/special files or changing references block unsafe conclusions.

```bash
.venv/bin/clipstudio plan-storage > storage-plan.json
.venv/bin/clipstudio verify-storage-plan storage-plan.json
```

The endpoints are GET `/api/v1/maintenance/storage` (size summary), GET `/api/v1/maintenance/storage-plan` and POST `/api/v1/maintenance/verify-plan`; the **Storage** page in the UI shows the same summary, plan status, blockers, candidates and verification result without any delete control. A symlink or special file inside the data directory (for example a runtime virtual environment under `data/runtimes`) blocks the plan by design; the page reports the exact path. A valid verification is an observation, not a lock or authorization to delete later. Plans do not rehash every media/model file; reported savings are estimates. A live worker, even idle, is reported as requiring maintenance shutdown for any future cleanup. There is no apply/delete/move/archive operation and no automatic age-based retention. A supported per-project deletion/cache-pruning UI is not implemented. Do not manually remove source files while the app runs. Low-disk failure preserves originals and reports a failed stage; free space, inspect the job and retry deliberately.

## Nonexecuting operational preflights

`clipstudio cloud-preflight PROJECT_ID scenario.json` hashes the selected current source and checks an operator-declared plan, price schedule and budget scenario. These inputs are assumptions, not authoritative provider prices or a spending ledger. Missing/stale pricing produces blockers or an unknown estimate. No data leaves the computer, no budget is reserved, and `can_execute` remains false.

`clipstudio publication-preflight PUBLICATION_ID` reads the stored draft, current source/export/revision and exact recorded manual approval. It checks current file hashes and package CRC against the existing verified decode record; it does not rerun a decoder or accept fabricated observations from a request body. `manual_handoff_ready` can be true, but `can_publish` always remains false. There are no account connections, posting calls or schedule execution. See [API request contracts](API.md).

## Optional container operation

Native setup is the tested delivery. A container build/run has not been validated for this release. A local optional Compose recipe is provided in `container/`; it uses host networking so the application's existing loopback bind remains private on Linux. It is not a public deployment and is not a redistribution clearance for native codec binaries. Review `container/README.md` before use. Container operation on Docker Desktop/macOS/Windows is unrun.
