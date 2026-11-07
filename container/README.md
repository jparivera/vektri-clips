# Optional Linux container recipe — unrun

Native source installation is the supported path. This container recipe has not been built or exercised for this release. Official Python 3.11.15 and Node 26.7.0 image tags/digests were checked against Docker Hub on 2026-11-03. Python/npm dependencies use the same locks. Debian's FFmpeg/font packages are resolved during build, not bit-for-bit pinned; inspect their exact versions and notices before any image distribution. No prebuilt image is provided; binary redistribution needs a separate license review.

On Linux with Docker Engine and Compose already installed:

```bash
cd /path/to/vektri-clips
# Stop the native app first to release port 8741.
docker compose -f container/compose.yaml build
docker compose -f container/compose.yaml up
# Ctrl-C stops it. Keep the data volume:
docker compose -f container/compose.yaml down
```

The recipe uses [Docker host networking](https://docs.docker.com/engine/network/drivers/host/) so the application retains its `127.0.0.1:8741` bind. It does not add a public bind or port mapping. Host networking shares the network namespace; it is not network isolation. No host directories are mounted. Media/model storage goes into a dedicated named volume, separate from the native library. Never run `down -v` unless intentionally deleting that volume after backup. Models are not bundled; explicit in-app downloads are still required. Default container user is UID 1000 and has no added capabilities.

Build-time generated fixtures can differ bytewise with Debian's FFmpeg version; run the full fixture/mechanics/AI/browser suite before trusting this as equivalent to the native build. The native benchmark is not evidence of container performance. Docker Desktop and non-Linux environments require separate networking and codec validation. The locked installation pattern follows the official [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
