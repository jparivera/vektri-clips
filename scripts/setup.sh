#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg through your OS package manager first.'; exit 1; }
command -v uv >/dev/null || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.'; exit 1; }
uv sync --locked --extra dev
(cd frontend && npm ci && npm run build)
if [ -f fixtures/spoken_interview.mp4 ]; then
  uv run --locked python scripts/generate_fixtures.py --verify
else
  uv run --locked python scripts/generate_fixtures.py
fi
printf '%s\n' 'Ready. Run ./scripts/start.sh and open http://127.0.0.1:8741. Model downloads are explicit in the app.'
