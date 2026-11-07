# Optional Linux recipe; image manifests checked 2026-11-03. Runtime untested here.
FROM node:26.7.0-bookworm-slim@sha256:4db36457f406501e6f608802e5da617e5fbd0e80b75901b6a09de1ae5a667d32 AS ui
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY backend/ backend/
RUN uv sync --locked --no-dev --no-editable
COPY scripts/ scripts/
COPY fixtures/ fixtures/
COPY --from=ui /build/frontend/dist/ frontend/dist/
RUN .venv/bin/python scripts/generate_fixtures.py && useradd --uid 1000 --create-home studio && mkdir /data && chown studio:studio /data
ENV VEKTRI_APP_DIR=/app VEKTRI_DATA_DIR=/data PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER studio
CMD ["/app/.venv/bin/clipstudio", "serve"]
