# Dependencies, models, assets and codec notices

Audited 2026-11-03 against the actual project installation, not repository license badges. Python direct versions are pinned in `pyproject.toml`; full resolution and wheel hashes are in `uv.lock`. Frontend direct versions are pinned and full resolution/integrities are in `frontend/package-lock.json`. Installation does not require API accounts. These dependencies are installed locally; the source repository does not contain `.venv`, `node_modules`, system FFmpeg, or downloaded model binaries.

## Incorporated upstream code

The foundation copies Chopify pure timing/geometry primitives: `crop_for`, `smooth_track`, `ass_escape`, `plan_cuts`, `remap_words`, `_shift`, `srt_time`. Pin: **`0d13b4a8ac8b91fd309cd54ffe8431bd3b13c5b6`**, MIT copyright 2026 Mehbul Islam. This is reviewed **unreleased main**, distinct from stable v1.3.0. The unchanged function bodies and extracted constants live in `backend/clipstudio/vendor/chopify.py`; the copyright/license and exact extraction/update notes are adjacent in `LICENSE.chopify` and `NOTICE.md`. No AutoClip, BridgeClip, OpenCut, proprietary product code, logos or branded templates are incorporated. Research checkouts remain ignored local evidence. The later `clipstudio.ocr` module follows Apache-2.0 RapidOCR/PaddleOCR preprocessing and CTC conventions, with bounded inference and an explicitly different OpenCV rectangle expansion instead of pyclipper polygon offsets. Upstream attribution, modification notes and Apache text are retained in `licenses/ocr/`. The RapidOCR package, pyclipper and shapely were not installed.

## Actual Python dependency inventory

The machine-readable [inventory](licenses/python-inventory.json) includes declared metadata, package URLs, and paths/SHA-256 for retained license documents from the installed distributions. The final lock comparison is in [lock-verification.json](licenses/lock-verification.json): all 46 installed distributions match their inventory and locked versions. The 48-entry Python lock also contains Windows-only colorama 0.4.6 and pyreadline3 3.5.6, which are not installed or inspected on this Linux machine. Refresh their notices before a Windows distribution. License text copies are in `licenses/python/`; CTranslate2/tokenizers notices omitted from wheel metadata were obtained from exact upstream version tags and retained under `licenses/upstream/`. This table records package-level declarations; native libraries require the separate audit below.

| Distribution | Installed version | Declared license / important extra terms |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.16.0 | MIT |
| av | 18.1.0 | BSD-3-Clause |
| certifi | 2026.10.23 | MPL-2.0 |
| click | 8.5.0 | BSD-3-Clause |
| coloredlogs | 15.0.1 | MIT |
| ctranslate2 | 4.8.5 | MIT plus bundled libgomp runtime |
| fastapi | 0.162.0 | MIT |
| faster-whisper | 1.2.1 | MIT |
| filelock | 4.2.7 | MIT |
| flatbuffers | 25.12.19 | Apache-2.0 |
| fsspec | 2026.9.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.8.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface_hub | 1.33.0 | Apache-2.0 |
| humanfriendly | 10.0 | MIT |
| idna | 3.20 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| mpmath | 1.3.0 | BSD-3-Clause |
| numpy | 2.2.6 | BSD-3-Clause plus bundled native notices |
| onnxruntime | 1.23.2 | MIT plus third-party notices |
| opencv-python-headless | 4.12.0.88 | Apache 2.0 |
| opentelemetry-api | 1.45.2 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.3.0 | MIT-CMU |
| pluggy | 1.7.0 | MIT |
| protobuf | 7.36.2 | BSD-3-Clause |
| psutil | 7.2.2 | BSD-3-Clause |
| pydantic | 2.14.2 | MIT |
| pydantic_core | 2.50.2 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pytest | 9.2.0 | MIT |
| pytest-timeout | 2.4.0 | MIT |
| python-multipart | 0.0.33 | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| sympy | 1.14.0 | BSD-3-Clause |
| tokenizers | 0.22.2 | Apache-2.0 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.54.0 | BSD-3-Clause |
| vektri-clip-studio | 0.1.0 | not declared (project code) |

## Actual JavaScript dependency inventory

The [JavaScript inventory](licenses/javascript-inventory.json) records all 73 lock entries, including optional platform packages not installed here; 31 packages are installed on this Linux machine. The table below reads installed `package.json` license fields (the generated lock omits those fields). Retained texts are under `licenses/javascript/`. Uninstalled platform-specific optional variants have not had their binary payloads inspected; a build for another OS needs a refreshed inventory.

| Package | Installed version | License | Use |
|---|---|---|---|
| @axe-core/playwright | 4.14.0 | MPL-2.0 | build/test |
| @oxc-project/types | 0.156.0 | MIT | build/test |
| @playwright/test | 1.64.1 | Apache-2.0 | build/test |
| @rolldown/binding-linux-x64-gnu | 1.2.16 | MIT | build/test |
| @rolldown/binding-linux-x64-musl | 1.2.16 | MIT | build/test |
| @rolldown/pluginutils | 1.0.1 | MIT | build/test |
| @types/react | 19.2.14 | MIT | build/test |
| @types/react-dom | 19.2.3 | MIT | build/test |
| @typescript/typescript-linux-x64 | 7.0.2 | Apache-2.0 | build/test |
| axe-core | 4.14.0 | MPL-2.0 | build/test |
| csstype | 3.2.3 | MIT | build/test |
| detect-libc | 2.1.2 | Apache-2.0 | build/test |
| fdir | 6.5.0 | MIT | build/test |
| lightningcss | 1.33.0 | MPL-2.0 | build/test |
| lightningcss-linux-x64-gnu | 1.33.0 | MPL-2.0 | build/test |
| lightningcss-linux-x64-musl | 1.33.0 | MPL-2.0 | build/test |
| nanoid | 3.3.20 | MIT | build/test |
| picocolors | 1.1.1 | ISC | build/test |
| picomatch | 4.0.7 | MIT | build/test |
| playwright | 1.64.1 | Apache-2.0 | build/test |
| playwright-core | 1.64.1 | Apache-2.0 | build/test |
| postcss | 8.5.32 | MIT | build/test |
| prettier | 3.9.10 | MIT plus retained bundled third-party notices | build/test |
| react | 19.3.1 | MIT | runtime |
| react-dom | 19.3.1 | MIT | runtime |
| rolldown | 1.2.16 | MIT | build/test |
| scheduler | 0.28.0 | MIT | runtime |
| source-map-js | 1.2.2 | BSD-3-Clause | build/test |
| tinyglobby | 0.2.17 | MIT | build/test |
| typescript | 7.0.2 | Apache-2.0 | build/test |
| vite | 8.3.5 | MIT | build/test |

MIT, ISC, BSD and Apache components require retention of their notices/licenses; Apache components also require preservation of applicable NOTICE files and marking modifications. MPL-2.0 appears in certifi, tqdm (dual/mixed declaration), build-time Lightning CSS, and the test-only axe-core accessibility engine with its Playwright adapter (added 2026-11-04; they run only inside the browser test suite and are not part of the production bundle). Preserve their source/notice obligations for redistributed covered files and modifications; do not relabel those files as this project's own license. PSF and MIT-CMU notices are also retained. Bundled development-tool code is separate from npm lock package boundaries: Prettier retains notices for 250 bundled dependencies (MIT, ISC, BSD, Apache and BlueOak-1.0.0; minimatch 10.2.5 requires its license text or license link to accompany copies). Playwright's bundled Babel tooling includes caniuse-lite 1.0.30001810 under CC-BY-4.0; retain its attribution/license and indicate modifications when redistributing that covered data. The exact nested Playwright `.LICENSE` sidecars, WebP/Emscripten notices, Rolldown third-party notice and TypeScript vscode-jsonrpc license are retained in the inventory. These development tools are not copied into exported videos or the production browser bundle. No third-party license grants rights to user-imported media.

## Model weights and trained assets are separate

| Asset | Exact revision / bytes | Verified terms and limitations |
|---|---|---|
| `Systran/faster-whisper-tiny.en` | `0d3d19a32d3338f10357c0889762bd8d64bbdeba`; 78,090,594 total catalog bytes | Model card declares MIT and identifies conversion from OpenAI Whisper tiny.en. Card and upstream Whisper MIT notice retained in `licenses/upstream/`. The model API catalogs SHA-256 for each file. English ASR with estimated timing, not diarization. |
| `Systran/faster-whisper-tiny` | `d90ca5fe260221311c53c58e660288d3deb8d356`; 78,203,619 total catalog bytes | MIT conversion of OpenAI Whisper tiny. Exact cached files are verified against the pinned upstream manifest. Optional multilingual ASR with en/pt/auto controls; not a guarantee of recognition accuracy or diarization. |
| `Xenova/clip-vit-base-patch32` ONNX | `d15189d7028b43f1d3e65039190477f6af591c2a`; 154,950,116 total catalog bytes | MIT model/conversion terms. Uses pinned int8 text plus uint8 vision graphs and tokenizer/preprocessor metadata. Xenova conversion, not an OpenAI-authored ONNX export. Still-image English retrieval with model-card limits; no temporal event or identity recognition claim. Retain upstream notices on redistribution. |
| `sentence-transformers/all-MiniLM-L6-v2` ONNX quantized AVX2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; 23,513,036 total catalog bytes | Pinned model card declares Apache-2.0. Card and Apache text retained. The catalog pins the ONNX and tokenizer SHA-256. Text similarity, not a generative LLM or semantic sports/vision understanding. |
| `imbcmdth/xclip-onnx` FP32 towers plus Microsoft metadata | Towers `649f3c91b59cd24be316dc505e26eacf5cd00801`; metadata `e4921c41fc296102aae210d43d4127c5e3e51928`; 755,544,722 total catalog bytes | Catalog MIT. Third-party ONNX export with separately pinned upstream `microsoft/xclip-base-patch16-kinetics-600` tokenizer/config. Existing ORT CPU FP32, not int8. Export omits the full classifier prompt generator; genuine temporal embeddings do not establish action-order quality. |
| `SWHL/RapidOCR` PP-OCRv4 detector/recognizer | `4e4644045a07c403b1ad40ca97340ecb4a8dc2c1`; 15,603,475 total catalog bytes | Apache-2.0. Two neural ONNX graphs, embedded recognizer dictionary; Chinese/English visible-text evidence with documented misses/skew limits. License/NOTICE retained in `licenses/ocr/`; no model weights in Git. |
| Existing faster-whisper Silero VAD v6 asset | Installed `faster-whisper` 1.2.1; 1,245,151 bytes; SHA256 `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2` | Silero MIT; speech/non-speech detection only. This existing asset does not supply speaker embeddings or diarization. |
| OpenCV frontal-face Haar cascade | Installed OpenCV 4.12.0 `haarcascade_frontalface_default.xml` | The trained XML header carries its own Intel Open Source Computer Vision license, with source/binary notice retention and no endorsement condition. Header retained at `licenses/haarcascade-frontalface-NOTICE.txt`; it is not assumed Apache merely because current OpenCV code is Apache. This detector can miss faces and is not active-speaker recognition. |

This capability round installed no new software packages or runtime upgrades. OCR reuses ONNX Runtime, OpenCV, NumPy and Pillow; X-CLIP reuses ONNX Runtime, NumPy, Pillow and tokenizers. Optional model assets remain separate from dependency installation.

Model downloads use the explicit catalog action, bounded size/disk checks, temporary installation, SHA-256 validation and cancellation checks. These files are not copied into Git or the export ZIP. They can be installed without paid API credentials. Do not infer that model licensing establishes rights to training data, imported footage, persons' likenesses, or publication destinations.

Official model sources inspected: [tiny.en pinned model card](https://huggingface.co/Systran/faster-whisper-tiny.en/blob/0d3d19a32d3338f10357c0889762bd8d64bbdeba/README.md), [MiniLM pinned model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md). Original retrieved document hashes/URLs: `licenses/upstream/sources.json`. Added sources: [multilingual tiny immutable tree](https://huggingface.co/Systran/faster-whisper-tiny/tree/d90ca5fe260221311c53c58e660288d3deb8d356), [Xenova CLIP conversion pin](https://huggingface.co/Xenova/clip-vit-base-patch32/commit/d15189d7028b43f1d3e65039190477f6af591c2a), [OpenAI CLIP MIT notice](https://github.com/openai/CLIP/blob/main/LICENSE) and [CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md). The model card calls for task-specific evaluation and limits supported language use to English; pinned quantization/runtime compatibility does not establish product quality.

Additional exact model sources: [PP-OCRv4 upload manifest](https://huggingface.co/SWHL/RapidOCR/commit/4e4644045a07c403b1ad40ca97340ecb4a8dc2c1), [X-CLIP ONNX conversion](https://huggingface.co/imbcmdth/xclip-onnx/tree/649f3c91b59cd24be316dc505e26eacf5cd00801), [Microsoft X-CLIP metadata](https://huggingface.co/microsoft/xclip-base-patch16-kinetics-600/tree/e4921c41fc296102aae210d43d4127c5e3e51928) and [Silero MIT license](https://github.com/snakers4/silero-vad/blob/master/LICENSE). Speaker diarization is not an integrated application capability.

## System and wheel-native codecs

The actual command-line FFmpeg is **8.0.1-3ubuntu2**, configured with `--enable-gpl --enable-libx264 --enable-libass --enable-libflite` among other libraries. `ffmpeg -L` reports **GPL version 2 or later**. Its complete build/configuration/license output is retained in [system-ffmpeg.txt](licenses/system-ffmpeg.txt). This application invokes the user-installed executable through argument arrays; it does not copy the executable into the repository or export packages.

Python wheels also contain native binaries, independently of the system executable:

- PyAV 18.1.0 declares BSD-3-Clause for its Python/binding code. Its installed bundled FFmpeg reports **LGPL version 3 or later** through `avcodec_license()`. Its configuration also lists x264/x265 and its wheel has those shared libraries. This deserves upstream build/source and codec-library license verification before redistributing that wheel inside a product image; the package-level BSD declaration is insufficient.
- OpenCV's installed bundled FFmpeg reports **LGPL version 2.1 or later**; the OpenCV wheel includes FFmpeg, OpenSSL, OpenBLAS, libpng and other third-party terms in `LICENSE-3RD-PARTY.txt`, retained in this audit.
- NumPy's wheel includes OpenBLAS, libgfortran and libquadmath. Its retained license contains the relevant native notices and runtime exceptions. CTranslate2 contains libgomp; its own MIT text is not a substitute for the GNU runtime exception terms.
- ONNX Runtime supplies a large `ThirdPartyNotices.txt`, retained unchanged. Pillow also carries image-codec dependencies covered by its own installed license bundle.

Actual native-library names, reported FFmpeg licenses and configure strings are retained in [native-ffmpeg-builds.json](licenses/native-ffmpeg-builds.json). No conclusion that all installed native binaries are permissively redistributable is made. **Local source installation is the supported delivery here; public bundled-binary/container redistribution remains unreviewed** until corresponding source/build instructions, runtime exceptions, and applicable codec/patent obligations have been accounted for. This does not block private local editing/export. The [official FFmpeg license guidance](https://ffmpeg.org/legal.html) explains why optional GPL components change a build's terms; it does not grant a general codec patent license.

## Fonts, images, fixtures and brand

The frontend uses system fonts. Rendered captions use installed **DejaVu Sans, Serif or Sans Mono**; deterministic fixture labels use DejaVu Sans. No AutoClip Inter/Anton TTFs or reference-product assets are copied. DejaVu's Bitstream Vera/DejaVu notices are retained in `fixtures/NOTICE-dejavu.txt`; preserve them if font files are ever bundled. No promise is made that this font covers every language.

The deterministic regression fixtures are generated locally from project-authored shapes, scripts and known timestamps. Third-party evaluation footage and derived frames/clips are not included in this repository. The spoken tracks use installed Flite 2.2 synthesis, clearly labeled synthetic rather than consented human recordings; its notice is retained in `fixtures/NOTICE-flite.txt`. `fixtures/manifest.json` records generator/tool versions, source hash, per-file synthetic status and fixture provenance. Synthetic tests measure mechanics and real ASR execution on synthesized audio, not real-world editorial quality.

## APIs and deployment boundary

No paid/cloud model or social publishing account is configured, and no subscription browser is used as an API. The manual publishing queue creates local drafts only. Cloud/publication preflights are nonexecuting local checks: declared scenario prices/budgets are not a provider quote or spend ledger, and manual-handoff readiness is not authorization or capability to post. A future supported adapter requires its own current provider terms, authorization, scopes/review/quota, transfer disclosure and cost budget. Imported source provenance and recorded permission status are operator records, not legal-rights verification.

No project-wide open-source license is provided for the original application code. Third-party terms remain applicable to their respective portions. Refresh this inventory for the target lock/platform before redistribution, retain the corresponding notices, and do not treat a one-platform review as blanket legal clearance. See [Rights and notices](../RIGHTS.md).

## Optional PP-OCRv5 mobile candidate

The explicit `ppocr-v5-mobile` catalog entry uses Apache-2.0 RapidAI ONNX conversions of PaddlePaddle PP-OCRv5 mobile detection and recognition weights. The 21,450,882-byte pair is optional; V4 remains default. The existing ONNX Runtime/OpenCV/NumPy stack runs it without new packages. Exact hashes, official/source-catalog revisions, conversion provenance and upstream differences are in [OCR model identities](OCR.md) and the [retained notice](licenses/ocr/NOTICE.txt). Weights and inference media are excluded from this repository. Local benchmark results do not establish upstream parity or general OCR accuracy.
