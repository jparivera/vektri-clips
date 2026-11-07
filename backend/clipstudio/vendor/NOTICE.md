# Chopify component notice

The adjacent Python primitives are extracted from Chopify main commit `0d13b4a8ac8b91fd309cd54ffe8431bd3b13c5b6`, copyright 2026 Mehbul Islam, MIT licensed. That commit contains unreleased changes; it is not advertised as stable v1.3.0.

Extracted unchanged function bodies: `render_clips.py::{crop_for,smooth_track,ass_escape}`, `tighten.py::{plan_cuts,remap_words,_shift}`, `captions.py::srt_time`. Their standard-library imports and numeric/token constants are retained. No downloader, dashboard, model, font, branded caption template, or application asset is included. New adapters call these functions with validated internal data.

The new media worker uses `crop_for` for aspect-constrained crop geometry, `smooth_track` to stabilize assisted horizontal tracking, `srt_time` for real timestamp subtitles, `ass_escape` for safe caption text, and the keep-range planner/remapper for opt-in silence edits. The planner is heuristic and remains reversible.

Update method: fetch proposed upstream hash in an isolated checkout; review extracted-function and license diffs; run `tests/test_media.py` plus shared synthetic fixture benchmark; deliberately update hash and notice. Do not auto-update from upstream main.
