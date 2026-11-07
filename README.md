# Vektri Clips

**A local video workspace, from source footage to reviewed clips.**

Import a video, inspect what actually happened, shape a timeline, and export a verified file. Your footage is processed on your computer. No paid AI key is needed for the local workflow.

This is an early **source release**, not a hosted service. Run it on a trusted Linux computer and keep it bound to localhost. Public source availability does not grant an open-source license; see [Rights and notices](RIGHTS.md).

## What it does

- **Import and preserve.** Local files, resumable browser uploads, or direct public media URLs. Originals remain separate from edits, with checksums and recorded provenance.
- **Find moments.** Optional local transcription, transcript search, visual retrieval and OCR. Models download only after an explicit action.
- **Review before accepting.** Timestamped evidence, surrounding context and before/at/after frames. Hook, context and payoff remain human judgments where evidence cannot establish them. Optional impact-sound detection is a heuristic, not event recognition.
- **Edit precisely.** Source-time trims, multi-clip cuts, undo/redo, revision history, crop keyframes, captions and audio settings.
- **Export and verify.** An accurate rendered preview, MP4, subtitles, thumbnail and provenance package, with decoding and integrity checks.
- **Keep control.** Local storage inspection, backup/restore and a manual publication queue. The app does not post to social accounts or run paid cloud jobs.

## Quick start

Requirements: Linux, Python 3.11 managed by [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22.12+ and npm, FFmpeg/ffprobe with libx264, libass and Flite, and DejaVu system fonts. Dependencies are pinned. Keep at least 1.5 GiB of memory available before heavy jobs and enough disk for source, proxy and exported copies.

```bash
git clone https://github.com/jparivera/vektri-clips.git
cd vektri-clips
./scripts/setup.sh
./scripts/start.sh
```

Open **http://127.0.0.1:8741**. Ctrl-C stops the API and worker. Setup installs locked dependencies and creates clearly labeled synthetic practice footage. It does not download AI models or create external accounts.

To choose a private media directory:

```bash
VEKTRI_DATA_DIR=/path/to/private/storage ./scripts/start.sh
```

Environment variables must be passed explicitly; `.env` files are not loaded automatically. See [.env.example](.env.example). Do not expose the service to your LAN or a public reverse proxy: remote authentication is not implemented.

## Make your first clip

1. Import media you are permitted to use, or choose a generated synthetic fixture. Record its source and permission status.
2. Inspect the source and select the analysis you need. Manual editing works without AI models.
3. Review a candidate's actual timestamps, quote, context and frames. Accept only after watching it.
4. Edit the timeline and use **Render accurate preview** to check the saved revision, selected audio and burned captions.
5. Export, play the saved result, and review any publication draft manually. A rights checkbox records your judgment; it does not establish rights to imported media.

## Development and checks

```bash
uv sync --locked --extra dev
(cd frontend && npm ci && npm run build)
uv run --locked python scripts/generate_fixtures.py
uv run --locked python scripts/generate_fixtures.py --verify
uv run --locked python -m pytest -q
(cd frontend && node --experimental-strip-types --test ../tests/*.test.ts)
```

If fixtures already exist, use `--verify` instead of regenerating. `--force` replaces only the named synthetic fixtures. Browser checks require a running app, an installed Chromium browser, and any models named by the selected integration test:

```bash
(cd frontend && VEKTRI_BROWSER_EXECUTABLE=/path/to/chromium npm run test:browser)
```

See [Verification](docs/VERIFICATION.md) for this release's actual results and unrun checks, [frontend development](frontend/README.md), and [contributing](CONTRIBUTING.md). Test-generated media, local evidence, installed dependencies and model files stay out of Git.

## Architecture and reference

React/TypeScript interface → FastAPI service → SQLite job queue → isolated Python/FFmpeg workers. One heavy job runs at a time by default. Local models are optional and their assets are not included in this repository.

- [Architecture](docs/ARCHITECTURE.md) and [API guide](docs/API.md)
- [Grounded review](docs/GROUNDED-REVIEW.md), [OCR](docs/OCR.md), and [local agent bridge](docs/AGENT-BRIDGE.md)
- [Storage, backup and recovery](docs/OPERATIONS.md) and [troubleshooting](docs/TROUBLESHOOTING.md)
- [Dependency and model notices](docs/DEPENDENCIES.md) and [synthetic fixtures](docs/fixtures.md)

## Limits worth knowing

This release does not establish human editorial quality, reliable sports/action understanding, speaker diarization, broad language accuracy, or long-video performance. Visual retrieval, temporal retrieval, OCR and tracking require review. Lightweight browser preview is approximate; rendered output is authoritative.

Browser/Safari/mobile behavior, real-footage impact precision/recall and native/container redistribution need additional validation. Social publishing and paid-cloud execution are not implemented. No recordings, private histories, runtime libraries, model weights or prebuilt binaries are included.
