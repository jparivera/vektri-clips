# Release verification

This source release has been checked in an isolated Linux cloud workspace. Results below distinguish executable tests from editorial or visual acceptance.

## Passed

- Locked Python and npm dependency installation.
- Production frontend build and strict TypeScript unused-code checks.
- 47 TypeScript unit tests, with zero failures.
- 73 focused backend tests covering grounded review, sound onsets, actual FFmpeg decoding, multi-audio sources, duration boundaries, cache integrity and fixture-generator portability.
- Five original synthetic media fixtures generated, hash checked and fully decoded. Dimensions, duration, audio presence and deliberate corrupt-input rejection passed.
- Playwright discovered all 14 browser tests successfully. Collection is not execution.
- Independent source-content disclosure and third-party-notice review. The published tree excludes private history, internal reports, recordings, generated media, runtime state, model weights and installed dependencies.

## Broad backend run

Final broad run: **896 passed, 20 skipped, 1 explicitly deselected, 94 subtests passed**, with one warning, in 107.08 seconds. Command: `python -m pytest -q --disable-warnings -k 'not test_killed_stage_cannot_leave_ffmpeg_running'`.

The first unfiltered run exposed two failures: unavailable sandbox process inspection, and a unit test that incorrectly assumed PID 1 could never be its parent. The latter now controls the parent PID explicitly and passed in the final run; no runtime supervision check was weakened.

## Blocked and unrun

- The local Chromium process could not start because this cloud prohibits its socket creation. Five selected browser tests failed before any UI assertions. The managed browser could not reach the local app. No local browser appearance, accessibility, responsive-layout or playback pass is claimed.
- The Linux parent-death test `test_killed_stage_cannot_leave_ffmpeg_running` cannot inspect the child-process list in this sandbox. It is explicitly excluded from the final broad run, not counted as passed.
- Model-dependent tests skip when verified assets are absent. No model downloads or simulated model quality were substituted.
- Real-footage editorial quality, impact precision/recall, Safari/mobile behavior, long-source peak memory and container operation remain unverified.

The GitHub workflow separately attempts the backend suite and controlled browser lifecycle tests. Those browser tests mock API/media state; FFmpeg tests cover actual decoding separately. A workflow definition is not proof that a run passed. Inspect the exact commit's Actions result.

## Fixes in this release

- Grounded candidate review with source-bound quotes, context and frames, plus optional explicitly heuristic impact-sound suggestions.
- Portable synthetic-fixture timing without requiring GNU `time`; unavailable RSS is represented as null rather than invented.
- Supervisor unit test no longer assumes PID 1 can never be its real parent.
- Neutral local-workspace UI, public setup documentation and preserved third-party notices, with no invented project-wide license.
