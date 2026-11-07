# Vektri browser application

The React/TypeScript application uses the same `/api/v1` services as the CLI. It does not simulate imports, model output, job completion, renders or publication. See the [project setup](../README.md) for the API/worker, local models and FFmpeg requirements.

```bash
cd frontend
npm ci --ignore-scripts
npm run build
```

The local API serves `dist/` at `http://127.0.0.1:8741`. For frontend development with the API already running:

```bash
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite development server binds to loopback on port 5173 and exits if that port is occupied. The proxy targets the API on port 8741 and rewrites only the exact `127.0.0.1:5173` or `localhost:5173` development origins. Foreign origins and cross-site requests remain subject to the API's rejection checks; unexpected incoming proxy hosts are rejected. Keep these explicit ports aligned if changing the development configuration. This uses Vite's documented [proxy configuration](https://vite.dev/config/server-options#server-proxy), without broad CORS or origin rewriting. No external fonts, UI assets, analytics or CDN scripts are used. Dependency versions and integrity are pinned in `package-lock.json`; see [licenses](LICENSES.md).

The main flow is Library → Source → Analysis and candidates → Editor → Export → manual Publishing queue. File uploads use resumable byte offsets; pausing retains an upload ID on this browser. Selecting the same file and import details resumes it. Imported sources remain immutable. Edits autosave as optimistic revisions; a conflict is an explicit error, not a silent overwrite.

Editor controls support source in/out, split/delete/reorder, undo/redo, keyboard stepping, source and output clocks, waveform, crop keyframes, fit/crop/fixed split layouts, reviewed tracking results, transcript corrections, caption word timing/style/font/colors, audio track/gain/normalization/fades, presets and export. Pause cuts require an explicit review and apply step. Their full source crop paths survive trimming. Preview uses source audio; export is authoritative for the chosen audio track, gain, normalization and fades. Browser seeking and font shaping are not pixel/frame-perfect guarantees for every source.

Local semantic AI, manual events, agent proposals and heuristic-only suggestions are distinguished. Candidate evidence preserves the actual method identifiers. Rejections and acceptance persist, and using past rejections as ranking exclusions is opt-in, not model training. No audio diarization, active-speaker association or semantic sports-event recognition is claimed. Agent mode exchanges validated files; no external agent is contacted.

Publishing records an exact destination/account, export version, caption, visibility, schedule/time zone and rights review. Manual-handoff approval never posts. Later edits invalidate approval. No social API/account is configured.

## Browser regression checks

With the API and worker running on port 8741, generated fixtures available, and the explicitly downloaded tiny.en/MiniLM models ready:

```bash
npm run test:browser
```

Tests use a configured `VEKTRI_BROWSER_EXECUTABLE`, otherwise `/usr/bin/google-chrome` if available, otherwise Playwright's installed Chromium. On a fresh machine without a browser, explicitly install the test browser with `npx playwright install chromium` (a separate browser download), or point the environment variable at your existing compatible browser. Tests create clearly labeled local synthetic projects and exports; they do not publish or clean originals.

The two serial integration tests exercise real upload/preparation, persisted timeline math, captions, renders/ZIP/playback, local model proposals, manual events, explicit handoff approval, feedback readback, transcript correction, pause cuts/undo, split preview and assisted object tracking. They also inspect 1440px desktop, 1024px tablet and 393px phone layouts, reduced motion, horizontal overflow, and browser errors. Generated evidence stays local. See the [release verification](../docs/VERIFICATION.md) for passed, skipped and unrun checks.
