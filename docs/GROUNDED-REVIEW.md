# Grounded candidate review and sound onsets

This feature keeps the existing shortlist, source-time timeline, immutable source, leased analysis job and proposal packet. It introduces no external model, paid API, new persistent store, automatic acceptance, rendering or publishing.

## Review a suggestion

Run analysis, then choose **Review in context** on a candidate. The review shows:

- The existing selection reason and method, not an invented editorial judgment.
- **Hook**: the first fully contained transcript excerpt. A quotation does not establish a strong hook.
- **Context**: nearest preceding/following transcript excerpts, with their original times. Whether the cut stands alone remains unknown.
- **Payoff**: the last fully contained transcript excerpt. A resolved payoff is not established automatically.
- **Signals**: actual motion/audio measurements or spectral-flux onset values, when available.
- Three JPEG samples before, at and after the candidate. Seeks are approximate. Watch full motion and surrounding context before accepting an event.

Quotes are exact substrings of the same canonical transcript sentences used for candidate selection, bounded to 500 characters and visibly marked **Excerpt, truncated** when shortened. Explicit segment-text corrections take precedence. Otherwise word text and its original timestamps remain authoritative, including user word edits. A cross-boundary transcript sentence is not assigned invented subsegment timing. Missing transcript, mismatched selected audio track, unavailable image, audio-only media and legacy suggestions remain explicit unknowns.

Playback is opt-in, starts at the selected interval and pauses at its end. Existing candidate/transcript times rounded to milliseconds retain the timeline's 1 ms source-end tolerance; review playback and frame seeks are capped at the actual source duration, without rewriting the supplied evidence times or candidate identity. Larger overruns and intervals starting outside the source are rejected. Switching between the candidate and surrounding context pauses and seeks without autoplay. Closing or replacing a candidate/source stops the player. The source proxy uses the first audio track; the review warns if analysis selected another track. Use a rendered preview to judge that track and final audio processing.

Each new review has a source SHA-256 and a fingerprint binding the candidate, observations, selected audio track and playback bounds. The analysis stage registers generated JPEGs with the existing checksummed artifact system. A changed source/evidence receipt invalidates an open review. These receipts do not replace the existing evidence packet fingerprint required for external agent proposals. Legacy/manual/external candidates remain usable with clearly labelled unstructured review until new analysis is run.

**Open this cut in editor** remains an explicit operator decision and records acceptance for editing. Nothing is automatically approved.

## Optional impact sounds

Choose action or mixed footage, turn off visual search and enable **Impact sounds** in local or heuristic mode. This replaces the coarse action signal shortlist with sound-onset candidates. Mixed footage can still include transcript retrieval, so its transcription/model settings continue to apply.

Settings:

| Option | Default | Meaning |
| --- | --- | --- |
| `impact_detection` | false | Explicit opt-in |
| `impact_low_hz` / `impact_high_hz` | 1000 / 6000 | Frequency band, within 0–8000 Hz and containing at least one 40 Hz FFT bin |
| `impact_threshold` | 0.04 | Minimum normalized positive spectral flux, not probability or an upstream detector's threshold units |
| `impact_min_gap` | 0.6 s | Minimum distance between retained sound onsets |
| `impact_lead` / `impact_follow` | 0.4 / 0.5 s | Source-time context around each sound; combined duration must be at least 40 ms |

Lead/follow controls define impact clip boundaries, independently of transcript minimum/maximum length. Requested count, exclusion ranges, feedback exclusions and final clip separation still apply. Set clip separation to zero for closer montage cuts. Intervals clipped below 40 ms at the source edge are omitted. Silence and no-audio sources produce no fabricated impacts.

Example:

```sh
clipstudio analyze PROJECT_ID --genre action --mode heuristic --no-transcribe \
  --impact-detection --impact-lead 0.4 --impact-follow 0.5 --separation 0
```

The independent detector reads 16 kHz mono audio from the selected track in bounded 30-second decode blocks, preserving overlap and spectral state. It uses 25 ms Hann windows, 10 ms hops and positive band spectral flux. Its threshold is the maximum of the configured floor and three times the trailing one-second mean. The first 100 ms establish a baseline. Exclusions precede spacing and bounded strongest-peak retention. At most 120 peaks are retained in the application, before final selection of at most 30 clips. Temporary audio is cleaned up, cancellation propagates, and source-time measurements are saved in `impacts.json`.

Music, footsteps, microphone noise, cuts and off-camera sounds can all trigger suggestions. Quiet/gradual sounds, the first 100 ms and tightly spaced sounds may be missed. A detected sound does not prove an impact, goal, crash, winner or completed action. Thresholds remain recording-dependent. Real-world precision/recall and long-source peak-memory measurements have not been established by synthetic tests.

## Verification and limits

Focused tests cover exact quotes, Unicode/HTML-like source text, source hashing, invalid/nonfinite bounds, missing frames, cancellation, repeated generation, multi-track audio, real FFmpeg onset timing across 30-second boundaries, gain/noise, exclusions and bounded retention. Integration tests use generated video/audio, never unapproved real footage.

Run the normal backend suite, UI build and frontend unit tests. The added `browser-grounded-review.spec.ts` separately simulates media state for UI lifecycle checks; it does not claim video decode verification. Actual frame decoding is tested through FFmpeg.

The cloud build's Chromium launch was blocked by `socket() EPERM`, so browser assertions, screenshots, accessibility and responsive visual acceptance remain unrun. The existing Linux parent-death process inspection test also could not inspect `/proc/<pid>/task/<pid>/children` in this sandbox. Model-dependent tests skip when verified local models are absent; no model downloads were initiated. See [release verification](VERIFICATION.md) for exact run counts.

This is an evidence/review layer, not a newly validated semantic editorial model. Human genre-quality review, real-world onset accuracy, Safari/phone behavior and long-video memory benchmarks remain future work.
