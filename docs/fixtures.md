# Synthetic benchmark fixtures

These original procedural fixtures are **synthetic mechanics tests**. They contain no imported video, human interview, sporting event, private information, proprietary interface, or cloned voice. Geometric subjects are not faces. A successful crop test on a square is not evidence of reliable face tracking or sports understanding.

Run from the repository root:

```bash
python3 scripts/generate_fixtures.py --verify
# Explicitly regenerate the named fixtures (does not touch other filenames):
python3 scripts/generate_fixtures.py --force
```

The generator uses the installed FFmpeg 8.0.1 (`libx264`, `libflite`), Pillow 12.3.0, Flite 2.2 and DejaVu Sans 2.37. It requires no download, API credential, GPU, or network. It encodes one file at a time with one video/filter thread. Frames are streamed through a pipe instead of retained in memory; temporary staging files are removed on exit. MP4s use H.264/yuv420p and AAC mono audio where applicable.

| File | Seconds | Dimensions | Purpose |
| --- | ---: | --- | --- |
| `spoken_interview.mp4` | 24 | 640×360 | Original scripted question and answer; actual synthetic speech from two Flite voices. |
| `tutorial_screen.mp4` | 24 | 640×360 | Original mock interface and three prerequisite → edit → result utterances. |
| `action_motion.mp4` | 22 | 640×360 | Known moving-square trajectory, scene cut at 11 s, five audiovisual sync markers. |
| `mixed_nonverbal.mp4` | 20 | 640×360 | One spoken introduction, then motion/audio with cuts at 5, 10, 15 s. |
| `portrait — café 日本語.mp4` | 18 | 360×640 | Unicode filename, moving subject, no audio stream. |
| `corrupt-input.mp4` | — | — | Deliberately invalid media; import must report a failure. |

All videos have 24 fps. `fixtures/manifest.json` records SHA-256, exact byte sizes, dimensions, duration, dependencies, provenance and measured creation resources. Each `<id>.truth.json` includes every frame's time, scene number, subject bounding box, flash state and the speech script. Boxes use `[x, y, width, height]` in source pixels. Their right/bottom coordinates follow the geometric drawing intent; encoded antialiased edges may spread slightly.

Utterance start/end times come from the actual synthesized PCM sample lengths, not guessed word lengths. No word alignment or ASR output is fabricated. Speech synthesis is deterministic test input; it must still be decoded and passed to the real ASR provider. Do not use these truth scripts as app transcripts or claim that reading them tests ASR. `spoken_interview` has a question from 1.000 to 4.779063 s and an answer from 8.500 to 18.835 s, which tests context retention across a pause. The deliberate gap is not grounds to delete the question.

The 1 kHz audio markers last 100 ms. Their matching white video flashes last two frames (83.333 ms); both start at the listed timestamp. The action fixture markers occur at 2, 6, 10, 14, and 18 s. AAC may introduce encoding-frame/priming effects; assess decoded signals and presentation timestamps, not compressed packet positions. Target sync tolerance: one source frame plus one AAC frame (about 106 ms at this sample rate), with any tighter measured result reported separately.

## Verification and limits

The generator verifies SHA-256 integrity, dimensions, duration within one source frame, audio presence and full FFmpeg decoding. Its measured timing and encoder peak RSS are recorded locally; if peak RSS is unavailable it is explicitly null, never estimated. Fixed scripts and metadata support repeatable generation with the same dependencies, but hashes may differ across codec, font and library versions.

Synthetic mechanics are not a substitute for real human voices, editorial review, long-source performance, variable-frame-rate sources or broad language/font testing.

## Rights and dependency provenance

The scripts, UI drawing and speech sentences were written for this project. No project-wide license is provided; see [Rights and notices](../RIGHTS.md). These definitions do not grant rights over user-imported footage. No ownership attestation is required to run the tests.

Speech is produced by the system Flite package, voices `kal16` and `rms`. Package notices are copied unchanged to `fixtures/NOTICE-flite.txt`; they describe permissive Carnegie Mellon/Edinburgh/Cepstral licenses as well as licenses of build/support files. The generator links to no Flite source and bundles no engine or voice-model binaries. No separate restriction on generated speech was found in the inspected package notices; this is an audit of the installed package, not an endorsement by any voice contributor. Never describe these voices as the voice of a real participant.

DejaVu Sans renders the fixture labels. `fixtures/NOTICE-dejavu.txt` preserves its installed Bitstream Vera/DejaVu notices. No font file is bundled. DejaVu additions are public domain; the underlying font software has notice/renaming conditions. Raster output is distinct from redistribution of the font software. The Unicode path exercises filename handling; it does not establish full Japanese typography support.

The installed FFmpeg build enables GPL components including libx264. Running the system executable and distributing generated media are distinct from bundling FFmpeg binaries. Any future application bundle must satisfy that build's software obligations and assess codec patents independently. This fixture generator does not redistribute FFmpeg. Pillow uses its permissive HPND license; installed version and font hash are in the manifest.

Primary references examined: [FFmpeg flite filter](https://ffmpeg.org/ffmpeg-filters.html#flite), [Flite upstream](https://github.com/festvox/flite), [DejaVu license](https://dejavu-fonts.github.io/License.html), and installed Debian package copyright records. The official online Flite manual URL was unavailable during this audit; installed package notices were available and are preserved.

## User-import benchmark for real-world quality (unrun)

Use a local source you are permitted to inspect privately; record its source URL/path, checksum, attribution, permission status (including unknown), and import date. Import does not authorize publication. Keep originals immutable. Choose at least one 2–5 minute sample per genre and an appropriate spoken sample with consent for the intended use. Do not submit footage to cloud providers without approving that data transfer.

1. Before analysis, annotate several complete question/answer or prerequisite/result spans, scene boundaries and important subjects independently of model output.
2. Run the same recorded model, preset, clip length/count and exclusions on each source. Record elapsed time, peak RAM, disk and actual paid usage.
3. Measure ASR word error rate against a reviewed transcript; boundary error and caption alignment against independent timestamps; crop retention/jitter against reviewed subject boxes; decode, dimensions and audio sync against source references.
4. Review diversity and factuality. Count unsupported event/person claims. Rate hook, context, payoff and on-screen legibility manually on a documented 1–5 rubric, retaining the reviewer identity and notes. Do not replace these judgments with a ranking score.
5. Compare with a simple evenly spaced/energy baseline. Save acceptance/rejection and corrected boundaries/captions/crops explicitly. Report real-world semantic evaluation as unrun until these steps actually happen.
