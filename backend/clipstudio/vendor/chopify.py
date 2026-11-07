"""Pinned, adapted pure primitives from Chopify.

Upstream: https://github.com/mehbul/chopify
Commit: 0d13b4a8ac8b91fd309cd54ffe8431bd3b13c5b6 (main, unreleased)
MIT Copyright (c) 2026 Mehbul Islam. See LICENSE.chopify and NOTICE.md.
Only pure geometry, smoothing and caption/timing planning are incorporated.
"""
import re

EMA_ALPHA = 0.20
JUMP_FRAC = 0.22
MARGIN_FRAC = 0.18
FILLER_TOKENS = {"um", "uh", "uhh", "umm", "erm", "er", "hmm", "mmm", "mm", "ah", "eh"}
FILLER_SEQS = {("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")}
DEFAULT_GAP = 0.6
GUARD = 0.15
MIN_KEEP = 0.30

def crop_for(W, H, tw, th):
    """Largest centred crop of the WxH source matching the target aspect."""
    tar = tw / float(th)
    src = W / float(H)
    if abs(src - tar) < 1e-3:
        cw, ch = W, H
    elif src > tar:                      # source wider -> crop width
        ch = H
        cw = int(round(H * tar))
    else:                                # source taller -> crop height
        cw = W
        ch = int(round(W / tar))
    cw -= cw % 2
    ch -= ch % 2
    x0 = max(0, (W - cw) // 2)
    y0 = max(0, (H - ch) // 2)
    return cw, ch, x0, y0

def smooth_track(track, W, crop_w):
    half = crop_w / 2.0
    max_off = half - crop_w * MARGIN_FRAC
    jump = W * JUMP_FRAC
    known = [c for _, c in track if c is not None]
    base = sorted(known)[len(known) // 2] if known else W / 2.0
    filled = []
    last = base
    for t, c in track:
        if c is None:
            c = last
        last = c
        filled.append((t, c))
    if not filled:
        return [(0.0, min(max(base, half), W - half))]
    out = []
    c = filled[0][1]
    prev_face = filled[0][1]
    for t, face in filled:
        if abs(face - prev_face) > jump:
            c = face
        else:
            c += EMA_ALPHA * (face - c)
        if face - c > max_off:
            c = face - max_off
        elif c - face > max_off:
            c = face + max_off
        prev_face = face
        out.append((t, min(max(c, half), W - half)))
    return out

def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")

def plan_cuts(words, clip_start, clip_end, gap=DEFAULT_GAP,
              fillers=True, silence=True):
    """Return keep-ranges [(a, b)] in *clip-relative* seconds.

    words: the transcript word list (absolute video time).
    fillers: cut filler words.  silence: cut pauses longer than `gap`.
    """
    sub = [w for w in words if w["end"] > clip_start and w["start"] < clip_end]
    if not sub:
        return [(0.0, clip_end - clip_start)]
    dur = clip_end - clip_start

    drop = []                                    # absolute-time ranges to cut
    if fillers:
        toks = [re.sub(r"[^\w']", "", w["word"].lower()) for w in sub]
        for i, w in enumerate(sub):
            if toks[i] in FILLER_TOKENS:
                drop.append((w["start"], w["end"]))
            elif i + 1 < len(sub) and (toks[i], toks[i + 1]) in FILLER_SEQS:
                drop.append((w["start"], sub[i + 1]["end"]))
    if silence:
        for a, b in zip(sub, sub[1:]):
            if b["start"] - a["end"] > gap:
                drop.append((a["end"], b["start"]))

    drop = [(max(a, clip_start + GUARD), min(b, clip_end - GUARD))
            for a, b in sorted(drop)]
    drop = [(a, b) for a, b in drop if b - a > 0.08]

    keeps, cur = [], clip_start                  # subtract drops -> keep ranges
    for a, b in drop:
        if a > cur:
            keeps.append((cur, a))
        cur = max(cur, b)
    if cur < clip_end:
        keeps.append((cur, clip_end))

    keeps = [(round(a - clip_start, 3), round(b - clip_start, 3))
             for a, b in keeps if b - a >= MIN_KEEP]
    return keeps or [(0.0, dur)]

def remap_words(words, clip_start, keeps):
    """Re-time words into the tightened timeline; returns (words, new_duration)."""
    out = []
    for w in words:
        shifted = _shift(w, clip_start, keeps)
        if shifted:
            out.append(shifted)
    total = sum(b - a for a, b in keeps)
    return out, round(total, 3)

def _shift(w, clip_start, keeps):
    """Map a word's absolute times onto the concatenated (tightened) timeline."""
    offset = 0.0
    s = t = None
    for a, b in keeps:
        a_abs, b_abs = clip_start + a, clip_start + b
        ws, we = max(w["start"], a_abs), min(w["end"], b_abs)
        if we > ws:                              # word overlaps this keep range
            if s is None:
                s = offset + (ws - a_abs)
            t = offset + (we - a_abs)
        elif s is not None:
            break                                # word fully past
        offset += b - a
    if s is None:
        return None
    # words that straddle a cut keep their kept portion; pad tiny words
    e = t if t and t > s else s + 0.12
    return {"word": w["word"], "start": round(s, 3), "end": round(e, 3)}

def srt_time(seconds):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

