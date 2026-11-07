import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { api, post, list, message, seconds, when } from "./api";
import type {
  Crop,
  Project,
  Timeline,
  Segment,
  Preset,
  Word,
  Job,
  Revision,
} from "./types";
import { Icon, Badge, Field, NumberField, ErrorBox, Empty, Modal } from "./ui";
import { video } from "./App";
import { renderPreviewState } from "./renderPreview";

const clone = <T,>(v: T): T => structuredClone(v);
function duration(t: Timeline) {
  return t.segments.reduce((v, s) => v + s.end - s.start, 0);
}
/** Ruler ticks at a readable interval for the given duration. */
function ticks(total: number) {
  if (!(total > 0)) return [0];
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const step = steps.find((s) => total / s <= 8) ?? 3600;
  const out: number[] = [];
  for (let t = 0; t < total * 0.88; t += step) out.push(+t.toFixed(3));
  return out;
}
const FRAME_SNAP = (fps: number) => 1 / (fps || 30);
export function Editor({
  project: p,
  onChange,
  onError,
  onRendered,
}: {
  project: Project;
  onChange: () => Promise<void>;
  onError: (e: string) => void;
  onRendered: () => void;
}) {
  const [draft, setDraft] = useState<Timeline>(() => clone(p.timeline)),
    [revision, setRevision] = useState(p.revision),
    [selected, setSelected] = useState(0),
    [clock, setClock] = useState(0),
    [playing, setPlaying] = useState(false),
    [sequence, setSequence] = useState(false),
    [panel, setPanel] = useState("framing"),
    [saveStatus, setSaveStatus] = useState("Saved"),
    [saveError, setSaveError] = useState(""),
    [history, setHistory] = useState<Timeline[]>([]),
    [future, setFuture] = useState<Timeline[]>([]),
    [waveform, setWaveform] = useState<number[]>([]),
    [presets, setPresets] = useState<Preset[]>([]),
    [presetName, setPresetName] = useState(""),
    [presetOpen, setPresetOpen] = useState(false),
    [shortcuts, setShortcuts] = useState(false),
    [rendering, setRendering] = useState(false),
    [renderPreviewOpen, setRenderPreviewOpen] = useState(false),
    [previewRevision, setPreviewRevision] = useState<number | null>(null),
    [previewJob, setPreviewJob] = useState<Job | null>(null),
    [previewError, setPreviewError] = useState(""),
    [transcriptOpen, setTranscriptOpen] = useState(false),
    [trackOpen, setTrackOpen] = useState(false),
    [pausePlan, setPausePlan] = useState<{
      segments: { start: number; end: number }[];
      duration: number;
      method: string;
      segmentId: string;
    } | null>(null),
    [pauseGap, setPauseGap] = useState(1),
    [pauseBusy, setPauseBusy] = useState(false),
    [trackMode, setTrackMode] = useState("face"),
    [roi, setRoi] = useState({ x: 0.4, y: 0.3, width: 0.2, height: 0.3 }),
    [historyOpen, setHistoryOpen] = useState(false),
    [revisions, setRevisions] = useState<Revision[] | null>(null),
    [revisionsError, setRevisionsError] = useState(""),
    [drop, setDrop] = useState<{ from: number; to: number; x: number } | null>(
      null,
    );
  const player = useRef<HTMLVideoElement>(null),
    second = useRef<HTMLVideoElement>(null),
    draftRef = useRef(draft),
    revisionRef = useRef(revision),
    saved = useRef(JSON.stringify(p.timeline)),
    savePromise = useRef<Promise<number> | null>(null),
    mounted = useRef(true),
    dragSerial = useRef(0),
    dragged = useRef(false),
    lastSeekAt = useRef(0);
  const latestTracking = p.jobs?.find(
    (j) => j.kind === "track" && j.status === "succeeded",
  )?.result?.tracking as
    | {
        status: string;
        method: string;
        confidence: number;
        confidence_kind: string;
        fallback: string;
        keyframes: { time: number; x: number; y: number }[];
        limitations: string[];
      }
    | undefined;
  const trackingRunning = p.jobs?.some(
    (j) => j.kind === "track" && ["queued", "running"].includes(j.status),
  );
  const segment = draft?.segments?.[selected] ?? draft?.segments?.[0],
    sourceDuration = Number(
      p.source?.duration ?? p.source?.metadata?.duration ?? 0,
    ),
    sourceWidth = Number(p.source?.width ?? p.source?.metadata?.width ?? 640),
    sourceHeight = Number(
      p.source?.height ?? p.source?.metadata?.height ?? 360,
    ),
    total = draft ? duration(draft) : 0;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const save = useCallback(async (): Promise<number> => {
    if (savePromise.current) {
      await savePromise.current;
      if (JSON.stringify(draftRef.current) !== saved.current) return save();
      return revisionRef.current;
    }
    const snapshot = clone(draftRef.current),
      serialized = JSON.stringify(snapshot);
    if (serialized === saved.current) return revisionRef.current;
    setSaveStatus("Saving…");
    const task = (async () => {
      try {
        const result = await api<{ revision: number; timeline: Timeline }>(
          "/projects/" + p.id + "/timeline",
          {
            method: "PUT",
            body: JSON.stringify({
              revision: revisionRef.current,
              timeline: snapshot,
            }),
          },
        );
        revisionRef.current = result.revision;
        saved.current = serialized;
        if (mounted.current) {
          setRevision(result.revision);
          setSaveStatus(
            JSON.stringify(draftRef.current) === serialized
              ? "Saved"
              : "Unsaved changes",
          );
          setSaveError("");
        }
        return result.revision;
      } catch (e) {
        if (mounted.current) {
          setSaveStatus("Not saved");
          setSaveError(message(e));
        }
        throw e;
      } finally {
        savePromise.current = null;
      }
    })();
    savePromise.current = task;
    return task;
  }, [p.id]);
  useEffect(
    () => () => {
      if (
        draftRef.current &&
        JSON.stringify(draftRef.current) !== saved.current
      )
        void save().catch(() => {});
    },
    [save],
  );
  useEffect(() => {
    draftRef.current = draft;
    if (!draft || JSON.stringify(draft) === saved.current) {
      // Undo back to the saved state: nothing to write, so say so.
      if (draft && !savePromise.current) setSaveStatus("Saved");
      return;
    }
    setSaveStatus("Unsaved changes");
    const timer = setTimeout(() => {
      void save().catch(() => {});
    }, 1000);
    return () => clearTimeout(timer);
  }, [draft, save]);
  useEffect(() => {
    if (
      p.revision > revisionRef.current &&
      JSON.stringify(draftRef.current) === saved.current &&
      !savePromise.current
    ) {
      const next = clone(p.timeline);
      draftRef.current = next;
      saved.current = JSON.stringify(next);
      revisionRef.current = p.revision;
      setDraft(next);
      setRevision(p.revision);
      setSelected(0);
    }
  }, [p.revision, p.timeline]);
  useEffect(() => {
    const before = (e: BeforeUnloadEvent) => {
      if (JSON.stringify(draftRef.current) !== saved.current) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", before);
    return () => window.removeEventListener("beforeunload", before);
  }, []);
  useEffect(() => {
    let stopped = false;
    const url = p.source?.waveform_url;
    if (url)
      fetch(url)
        .then((r) => (r.ok ? r.json() : null))
        .then((r) => {
          if (stopped || !r) return;
          const values = Array.isArray(r)
            ? r
            : Array.isArray(r.peaks)
              ? r.peaks
              : Array.isArray(r.samples)
                ? r.samples
                : Array.isArray(r.bins)
                  ? r.bins
                  : [];
          setWaveform(
            values.map((v: unknown) =>
              typeof v === "number"
                ? Math.abs(v)
                : Array.isArray(v)
                  ? Math.max(...v.map(Number))
                  : Number((v as { peak?: number })?.peak ?? 0),
            ),
          );
        })
        .catch(() => {});
    api<unknown>("/presets")
      .then((r) => {
        if (!stopped) setPresets(list<Preset>(r, "presets"));
      })
      .catch(() => {});
    return () => {
      stopped = true;
    };
  }, [p.id, p.source?.waveform_url]);
  // Continuous controls (sliders, typed numbers, color pickers) pass a key so a
  // single drag becomes one undo step instead of dozens.
  const lastEdit = useRef<{ key: string; at: number } | null>(null);
  const edit = useCallback((fn: (t: Timeline) => void, coalesce?: string) => {
    const previous = clone(draftRef.current),
      next = clone(previous);
    fn(next);
    draftRef.current = next;
    const now = Date.now();
    const merge =
      coalesce !== undefined &&
      lastEdit.current?.key === coalesce &&
      now - lastEdit.current.at < 1200;
    if (!merge) setHistory((h) => [...h.slice(-59), previous]);
    lastEdit.current =
      coalesce === undefined ? null : { key: coalesce, at: now };
    setFuture([]);
    setDraft(next);
    setSaveError("");
  }, []);
  const undo = useCallback(
    () =>
      setHistory((h) => {
        if (!h.length) return h;
        const next = h[h.length - 1];
        setFuture((f) => [clone(draftRef.current), ...f].slice(0, 60));
        draftRef.current = next;
        lastEdit.current = null;
        setDraft(next);
        return h.slice(0, -1);
      }),
    [],
  );
  const redo = useCallback(
    () =>
      setFuture((f) => {
        if (!f.length) return f;
        setHistory((h) => [...h, clone(draftRef.current)].slice(-60));
        draftRef.current = f[0];
        lastEdit.current = null;
        setDraft(f[0]);
        return f.slice(1);
      }),
    [],
  );
  const seek = useCallback(
    (time: number) => {
      if (player.current) {
        player.current.currentTime = Math.max(
          0,
          Math.min(sourceDuration, time),
        );
        setClock(player.current.currentTime);
        if (second.current)
          second.current.currentTime = player.current.currentTime;
      }
    },
    [sourceDuration],
  );
  const toggle = useCallback(() => {
    if (!player.current) return;
    if (player.current.paused)
      void player.current.play().catch((e) => onError(message(e)));
    else player.current.pause();
  }, [onError]);
  const split = useCallback(() => {
    const t = draftRef.current,
      s = t.segments[Math.min(selected, t.segments.length - 1)];
    if (!s || clock <= s.start + 0.04 || clock >= s.end - 0.04) {
      onError(
        "Place the playhead at least 40 ms inside the selected segment to split.",
      );
      return;
    }
    edit((n) => {
      const a = n.segments[selected],
        b = { ...clone(a), id: crypto.randomUUID(), start: clock };
      a.end = clock;
      n.segments.splice(selected + 1, 0, b);
    });
  }, [clock, selected, edit, onError]);
  const updateSegment = useCallback(
    (fn: (s: Segment) => void, coalesce?: string) =>
      edit(
        (t) => fn(t.segments[Math.min(selected, t.segments.length - 1)]),
        coalesce,
      ),
    [edit, selected],
  );
  const selectSegment = (index: number) => {
    setSelected(index);
    seek(draft.segments[index].start);
  };
  const remove = useCallback(() => {
    if (draftRef.current.segments.length <= 1) {
      onError("Keep at least one segment in the edit.");
      return;
    }
    edit((t) => t.segments.splice(selected, 1));
    setSelected(Math.max(0, selected - 1));
  }, [edit, selected, onError]);
  const markIn = useCallback(() => {
    updateSegment((s) => {
      s.start = Math.max(0, Math.min(clock, s.end - 0.04));
    });
  }, [updateSegment, clock]);
  const markOut = useCallback(() => {
    updateSegment((s) => {
      s.end = Math.min(
        sourceDuration || Infinity,
        Math.max(clock, s.start + 0.04),
      );
    });
  }, [updateSegment, clock, sourceDuration]);
  useEffect(() => {
    const handle = (e: KeyboardEvent) => {
      if (
        (e.target as HTMLElement)?.closest(
          "input,textarea,select,[contenteditable],dialog",
        )
      )
        return;
      const frame = FRAME_SNAP(draftRef.current.output.fps);
      const current = draftRef.current.segments[selected];
      if (e.key === " ") {
        e.preventDefault();
        toggle();
      } else if (
        e.altKey &&
        (e.key === "ArrowLeft" || e.key === "ArrowRight")
      ) {
        e.preventDefault();
        reorder(e.key === "ArrowLeft" ? -1 : 1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        seek((player.current?.currentTime ?? 0) + (e.shiftKey ? 1 : frame));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        seek((player.current?.currentTime ?? 0) - (e.shiftKey ? 1 : frame));
      } else if (e.key === "Home" && current) {
        e.preventDefault();
        seek(current.start);
      } else if (e.key === "End" && current) {
        e.preventDefault();
        seek(current.end);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save().catch(() => {});
      } else if (e.ctrlKey || e.metaKey || e.altKey) {
        return;
      } else if (e.key.toLowerCase() === "s") split();
      else if (e.key.toLowerCase() === "i") markIn();
      else if (e.key.toLowerCase() === "o") markOut();
      else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        remove();
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [
    toggle,
    seek,
    undo,
    redo,
    save,
    split,
    markIn,
    markOut,
    remove,
    reorder,
    selected,
  ]);
  const reorderTo = useCallback(
    (from: number, to: number) => {
      const count = draftRef.current.segments.length;
      if (from === to || from < 0 || to < 0 || from >= count || to >= count)
        return;
      edit((t) => {
        const [s] = t.segments.splice(from, 1);
        t.segments.splice(to, 0, s);
      });
      setSelected(to);
    },
    [edit],
  );
  function reorder(direction: number) {
    reorderTo(selected, selected + direction);
  }
  const clamp = (v: number, lo: number, hi: number) =>
    Math.min(hi, Math.max(lo, v));
  const throttledSeek = (time: number) => {
    const now = performance.now();
    if (now - lastSeekAt.current < 80) return;
    lastSeekAt.current = now;
    seek(time);
  };
  /** Drag a clip edge in either lane; pointer capture keeps the drag alive. */
  const startTrim = (
    e: PointerEvent<HTMLElement>,
    index: number,
    side: "start" | "end",
    laneSelector: string,
    laneSeconds: number,
  ) => {
    if (e.button !== 0 || !(laneSeconds > 0)) return;
    e.preventDefault();
    e.stopPropagation();
    const handle = e.currentTarget;
    const lane = handle.closest(laneSelector) as HTMLElement | null;
    const perPixel = laneSeconds / (lane?.getBoundingClientRect().width || 1);
    const seg = draftRef.current.segments[index];
    if (!seg) return;
    const origin = { x: e.clientX, start: seg.start, end: seg.end };
    const key = `trim:${seg.id}:${side}:${++dragSerial.current}`;
    setSelected(index);
    handle.setPointerCapture(e.pointerId);
    const move = (ev: globalThis.PointerEvent) => {
      const delta = (ev.clientX - origin.x) * perPixel;
      edit((t) => {
        const s = t.segments[index];
        if (!s) return;
        if (side === "start")
          s.start = +clamp(origin.start + delta, 0, origin.end - 0.04).toFixed(
            2,
          );
        else
          s.end = +clamp(
            origin.end + delta,
            origin.start + 0.04,
            sourceDuration || origin.end + delta,
          ).toFixed(2);
      }, key);
      const s = draftRef.current.segments[index];
      if (s) throttledSeek(side === "start" ? s.start : s.end);
    };
    const finish = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", finish);
      handle.removeEventListener("pointercancel", finish);
      const s = draftRef.current.segments[index];
      if (s) seek(side === "start" ? s.start : s.end);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
  };
  /** Drag a clip to a new position; a click without movement still selects. */
  const startReorder = (e: PointerEvent<HTMLButtonElement>, index: number) => {
    if (e.button !== 0 || (e.target as HTMLElement).closest(".trim-handle"))
      return;
    const clip = e.currentTarget;
    const track = clip.closest(".clip-track") as HTMLElement | null;
    if (!track || draftRef.current.segments.length < 2) return;
    const startX = e.clientX;
    let moved = false,
      to = index;
    clip.setPointerCapture(e.pointerId);
    const move = (ev: globalThis.PointerEvent) => {
      if (!moved && Math.abs(ev.clientX - startX) < 6) return;
      moved = true;
      const others = [
        ...track.querySelectorAll<HTMLElement>(".timeline-clip"),
      ].filter((_, i) => i !== index);
      let position = 0;
      for (const other of others) {
        const r = other.getBoundingClientRect();
        if (ev.clientX > r.left + r.width / 2) position++;
      }
      to = position;
      const rect = track.getBoundingClientRect();
      const marker =
        position < others.length
          ? others[position].getBoundingClientRect().left
          : (others[others.length - 1]?.getBoundingClientRect().right ??
            rect.left);
      setDrop({ from: index, to, x: marker - rect.left });
    };
    const finish = () => {
      clip.removeEventListener("pointermove", move);
      clip.removeEventListener("pointerup", finish);
      clip.removeEventListener("pointercancel", finish);
      setDrop(null);
      if (moved) {
        dragged.current = true;
        reorderTo(index, to);
      }
    };
    clip.addEventListener("pointermove", move);
    clip.addEventListener("pointerup", finish);
    clip.addEventListener("pointercancel", finish);
  };
  const clipKey = (e: ReactKeyboardEvent<HTMLButtonElement>, i: number) => {
    if (e.altKey || (e.key !== "ArrowRight" && e.key !== "ArrowLeft")) return;
    e.preventDefault();
    e.stopPropagation();
    const next = clamp(
      i + (e.key === "ArrowRight" ? 1 : -1),
      0,
      draft.segments.length - 1,
    );
    selectSegment(next);
    e.currentTarget.parentElement
      ?.querySelectorAll<HTMLElement>(".timeline-clip")
      [next]?.focus();
  };
  /** Loading an older revision is itself a new, undoable edit; nothing is deleted. */
  const restoreRevision = (r: Revision) => {
    edit((t) => {
      const snapshot = clone(r.timeline);
      t.segments = snapshot.segments;
      t.output = snapshot.output;
      t.captions = snapshot.captions;
      t.audio = snapshot.audio;
    });
    setSelected(0);
    setHistoryOpen(false);
    const first = r.timeline.segments[0];
    if (first) seek(first.start);
  };
  useEffect(() => {
    if (!historyOpen) return;
    let stopped = false;
    setRevisions(null);
    setRevisionsError("");
    api<unknown>("/projects/" + p.id + "/revisions")
      .then((r) => {
        if (!stopped) setRevisions(list<Revision>(r, "revisions"));
      })
      .catch((e) => {
        if (!stopped) setRevisionsError(message(e));
      });
    return () => {
      stopped = true;
    };
  }, [historyOpen, p.id]);
  const outputTime =
    draft?.segments
      ?.slice(0, selected)
      .reduce((v, s) => v + s.end - s.start, 0) +
    Math.max(0, Math.min(segment?.end ?? 0, clock) - (segment?.start ?? 0));
  const seekFromSourceLane = (e: PointerEvent<HTMLDivElement>) => {
    if (!sourceDuration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(
      1,
      Math.max(0, (e.clientX - rect.left) / rect.width),
    );
    const time = ratio * sourceDuration;
    const index = draft.segments.findIndex(
      (s) => time >= s.start && time <= s.end,
    );
    if (index >= 0 && index !== selected) setSelected(index);
    seek(time);
  };
  const onTime = () => {
    const el = player.current;
    if (!el) return;
    setClock(el.currentTime);
    if (
      second.current &&
      Math.abs(second.current.currentTime - el.currentTime) > 0.09
    )
      second.current.currentTime = el.currentTime;
    if (sequence && playing && el.currentTime >= segment.end - 0.025) {
      if (selected < draft.segments.length - 1) {
        setSelected(selected + 1);
        el.currentTime = draft.segments[selected + 1].start;
      } else {
        el.pause();
        el.currentTime = segment.end;
      }
    }
  };
  const cropAt = useMemo(() => {
    const c = segment?.crop;
    if (!c) return { x: 0.5, y: 0.5 };
    const frames = c.keyframes ?? [];
    if (!frames.length) return { x: c.x, y: c.y };
    if (clock <= frames[0].time) return frames[0];
    for (let i = 1; i < frames.length; i++) {
      if (clock <= frames[i].time) {
        const a = frames[i - 1],
          b = frames[i],
          r = (clock - a.time) / (b.time - a.time);
        return { x: a.x + (b.x - a.x) * r, y: a.y + (b.y - a.y) * r };
      }
    }
    return frames[frames.length - 1];
  }, [segment?.crop, clock]);
  const videoStyle = (split = false, lower = false): CSSProperties => {
    if (segment?.crop.mode === "fit")
      return { objectFit: "contain", width: "100%", height: "100%" };
    const outputHeight = split
        ? Math.floor(draft.output.height / 4) * 2
        : draft.output.height,
      ratio = draft.output.width / outputHeight;
    let cw = sourceWidth,
      ch = sourceWidth / ratio;
    if (ch > sourceHeight) {
      ch = sourceHeight;
      cw = ch * ratio;
    }
    cw = Math.round(cw);
    cw -= cw % 2;
    ch = Math.round(ch);
    ch -= ch % 2;
    cw = Math.max(2, Math.floor(cw / (split ? 1 : segment.crop.zoom) / 2) * 2);
    ch = Math.max(2, Math.floor(ch / (split ? 1 : segment.crop.zoom) / 2) * 2);
    const x = Math.max(
        0,
        Math.min(
          sourceWidth - cw,
          sourceWidth *
            (lower
              ? segment.crop.second_x
              : split
                ? segment.crop.x
                : cropAt.x) -
            cw / 2,
        ),
      ),
      y = Math.max(
        0,
        Math.min(
          sourceHeight - ch,
          sourceHeight * (split ? segment.crop.y : cropAt.y) - ch / 2,
        ),
      );
    return {
      position: "absolute",
      maxWidth: "none",
      width: `${(sourceWidth / cw) * 100}%`,
      height: `${(sourceHeight / ch) * 100}%`,
      left: `${(-x / cw) * 100}%`,
      top: `${(-y / ch) * 100}%`,
      objectFit: "fill",
    };
  };
  const wordGroup = useMemo(() => {
    const words = (draft?.captions.words ?? [])
      .filter((w) => w.start < segment.end && w.end > segment.start)
      .map((w) => ({
        ...w,
        start: Math.max(w.start, segment.start),
        end: Math.min(w.end, segment.end),
      }));
    const groups: Word[][] = [],
      limit = draft.output.height > draft.output.width ? 28 : 46;
    let group: Word[] = [];
    for (const w of words) {
      if (
        group.length &&
        (w.start - group[group.length - 1].end > 0.65 ||
          group.length >= 7 ||
          [...group, w].map((x) => x.word).join(" ").length > limit * 2)
      ) {
        groups.push(group);
        group = [];
      }
      group.push(w);
      if (/[.!?]$/.test(w.word)) {
        groups.push(group);
        group = [];
      }
    }
    if (group.length) groups.push(group);
    return (
      groups.find((g) => clock >= g[0].start && clock < g[g.length - 1].end) ??
      []
    );
  }, [
    draft?.captions.words,
    draft.output.width,
    draft.output.height,
    segment?.start,
    segment?.end,
    clock,
  ]);

  const unsavedPreviewChanges = JSON.stringify(draft) !== saved.current;
  const renderedPreview = renderPreviewState(
    p,
    previewRevision,
    revision,
    unsavedPreviewChanges,
    previewJob,
  );
  const currentPreview = renderPreviewState(
    p,
    revision,
    revision,
    unsavedPreviewChanges,
  );
  const showRenderedPreview = async () => {
    player.current?.pause();
    second.current?.pause();
    setRenderPreviewOpen(true);
    setPreviewRevision(null);
    setPreviewJob(null);
    setPreviewError("");
    setRendering(true);
    try {
      const rev = await save();
      if (!mounted.current) return;
      setPreviewRevision(rev);
      const existing = renderPreviewState(p, rev, rev, false);
      if (existing.exported) return;
      if (existing.job) {
        setPreviewJob(existing.job);
        return;
      }
      const job = await post<Job>(
        "/projects/" + p.id + "/render",
        { revision: rev },
        true,
      );
      if (mounted.current) setPreviewJob(job);
      await onChange();
    } catch (e) {
      if (mounted.current) setPreviewError(message(e));
    } finally {
      if (mounted.current) setRendering(false);
    }
  };
  const runRender = async () => {
    setRendering(true);
    try {
      const rev = await save();
      await post("/projects/" + p.id + "/render", { revision: rev }, true);
      await onChange();
      onRendered();
    } catch (e) {
      onError(message(e));
    } finally {
      setRendering(false);
    }
  };
  if (!draft || !segment)
    return (
      <Empty title="Preparing the source…">
        The editor becomes available after metadata and preview preparation
        completes.
      </Empty>
    );
  return (
    <>
      <div className="editor-toolbar">
        <div className="row">
          <Badge tone="accent">Revision {revision}</Badge>
          <span
            className={`save-state ${saveError ? "bad" : ""}`}
            role="status"
          >
            <span className="status-dot" />
            {saveStatus}
          </span>
        </div>
        <div className="row">
          <button
            className="icon-btn"
            disabled={!history.length}
            onClick={undo}
            aria-label="Undo"
            title="Undo (Ctrl+Z)"
          >
            <Icon name="undo" />
          </button>
          <button
            className="icon-btn"
            disabled={!future.length}
            onClick={redo}
            aria-label="Redo"
            title="Redo (Ctrl+Shift+Z)"
          >
            <Icon name="redo" />
          </button>
          <span className="separator" />
          <button
            className="button secondary small"
            onClick={() => setHistoryOpen(true)}
          >
            <Icon name="history" size={15} />
            History
          </button>
          <button
            className="button secondary small"
            onClick={() => setPresetOpen(true)}
          >
            Presets
          </button>
          <button
            className="button primary small"
            onClick={() => void runRender()}
            disabled={rendering}
          >
            <Icon name="download" size={16} />
            {rendering ? "Queuing…" : "Export edit"}
          </button>
        </div>
      </div>
      {saveError && (
        <div className="editor-save-error">
          <ErrorBox error={"Edit is not saved: " + saveError} />
          <button
            className="text-button"
            onClick={() => void save().catch(() => {})}
          >
            Retry save
          </button>
          <button
            className="text-button"
            onClick={() => {
              const fresh = clone(p.timeline);
              saved.current = JSON.stringify(fresh);
              draftRef.current = fresh;
              revisionRef.current = p.revision;
              setDraft(fresh);
              setRevision(p.revision);
              setSaveError("");
              setSaveStatus("Saved");
              setHistory([]);
              setFuture([]);
            }}
          >
            Discard local changes and reload revision {p.revision}
          </button>
        </div>
      )}
      <div className="editor-workspace">
        <aside className="edit-cuts" aria-label="Cuts">
          <div className="section-kicker">
            YOUR CUT <span>{draft.segments.length} segments</span>
          </div>
          <div className="cut-list">
            {draft.segments.map((s, i) => (
              <button
                className={`cut-item ${i === selected ? "selected" : ""}`}
                key={s.id}
                onClick={() => selectSegment(i)}
              >
                <span className="cut-index">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <strong>
                    {seconds(s.start)} → {seconds(s.end)}
                  </strong>
                  <span>
                    {(s.end - s.start).toFixed(2)}s · {s.crop.mode}
                  </span>
                </div>
                <Icon name="chevron" size={14} />
              </button>
            ))}
          </div>
          <button
            className="button secondary small full"
            onClick={() =>
              edit((t) => {
                const start = Math.min(
                  clock,
                  Math.max(0, sourceDuration - 0.1),
                );
                t.segments.push({
                  id: crypto.randomUUID(),
                  start,
                  end: Math.min(sourceDuration, start + 5),
                  crop: clone(segment.crop),
                  gain_db: 0,
                });
              })
            }
          >
            <Icon name="plus" size={15} />
            Add from playhead
          </button>
          <div className="cut-context">
            <Icon name="file" size={18} />
            <p>
              Every segment refers to the original source. Reorder your edit
              without changing the media.
            </p>
          </div>
          <button
            className="button secondary small full"
            onClick={() => setTranscriptOpen(true)}
          >
            Transcript & corrections
          </button>
        </aside>
        <section className="preview-panel">
          <div className="section-kicker">
            <span>EDIT PREVIEW</span>
            <span>
              {draft.output.width} × {draft.output.height}
            </span>
          </div>
          <div className="preview-stage">
            <div
              className={
                "output-frame " +
                (segment.crop.mode === "split" ? "split-frame" : "")
              }
              style={{
                aspectRatio: `${draft.output.width}/${draft.output.height}`,
              }}
            >
              <div
                className="video-viewport"
                style={
                  segment.crop.mode === "split"
                    ? {
                        height: `${((Math.floor(draft.output.height / 4) * 2) / draft.output.height) * 100}%`,
                        bottom: "auto",
                      }
                    : undefined
                }
              >
                <video
                  ref={player}
                  src={video(p)}
                  preload="metadata"
                  poster={p.source?.thumbnail_url}
                  style={videoStyle(segment.crop.mode === "split")}
                  onTimeUpdate={onTime}
                  onPlay={() => {
                    setPlaying(true);
                    if (second.current)
                      void second.current.play().catch(() => {});
                  }}
                  onPause={() => {
                    setPlaying(false);
                    second.current?.pause();
                  }}
                  onLoadedMetadata={() => seek(segment.start)}
                  playsInline
                  aria-label="Edited video preview"
                />
              </div>
              {segment.crop.mode === "split" && (
                <div
                  className="video-viewport"
                  style={{
                    top: `${((Math.floor(draft.output.height / 4) * 2) / draft.output.height) * 100}%`,
                    height: `${((Math.floor(draft.output.height / 4) * 2) / draft.output.height) * 100}%`,
                    bottom: "auto",
                  }}
                >
                  <video
                    ref={second}
                    src={video(p)}
                    preload="metadata"
                    muted
                    playsInline
                    style={videoStyle(true, true)}
                    aria-label="Second split panel"
                  />
                </div>
              )}
              <div className="safe-zone" aria-hidden="true" />
              {draft.captions.enabled && wordGroup.length > 0 && (
                <div
                  className={
                    "caption-preview " +
                    (draft.captions.background ||
                    draft.captions.style === "boxed"
                      ? "boxed"
                      : "")
                  }
                  style={{
                    fontSize: `${(draft.captions.font_size / draft.output.height) * 100}cqh`,
                    fontFamily: draft.captions.font_family ?? "DejaVu Sans",
                    color: draft.captions.color ?? "#FFFFFF",
                    WebkitTextStroke: `${((draft.captions.stroke_width ?? 2) / draft.output.height) * 100}cqh black`,
                    paintOrder: "stroke fill",
                    textShadow: "none",
                    bottom:
                      draft.captions.alignment < 7
                        ? `${(Math.min(draft.captions.margin, draft.output.height / 3) / draft.output.height) * 100}%`
                        : undefined,
                    top:
                      draft.captions.alignment >= 7
                        ? `${(Math.min(draft.captions.margin, draft.output.height / 3) / draft.output.height) * 100}%`
                        : undefined,
                    textAlign: [1, 7].includes(draft.captions.alignment)
                      ? "left"
                      : [3, 9].includes(draft.captions.alignment)
                        ? "right"
                        : "center",
                  }}
                >
                  {wordGroup.map((w, i) => (
                    <span
                      key={i}
                      style={
                        draft.captions.highlight &&
                        clock >= w.start &&
                        clock < w.end
                          ? {
                              color:
                                draft.captions.highlight_color ?? "#D3EC78",
                            }
                          : undefined
                      }
                      className={
                        draft.captions.highlight &&
                        clock >= w.start &&
                        clock < w.end
                          ? "active-word"
                          : ""
                      }
                    >
                      {w.word}{" "}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
          <div className="transport">
            <div className="row">
              <button
                className="icon-btn"
                onClick={() => seek(clock - 1 / draft.output.fps)}
                aria-label="Previous frame"
                title="Previous frame (←)"
              >
                <Icon name="chevron-left" />
              </button>
              <button
                className="play-control"
                onClick={toggle}
                aria-label={playing ? "Pause preview" : "Play preview"}
              >
                <Icon name={playing ? "pause" : "play"} size={19} />
              </button>
              <button
                className="icon-btn"
                onClick={() => seek(clock + 1 / draft.output.fps)}
                aria-label="Next frame"
                title="Next frame (→)"
              >
                <Icon name="chevron" />
              </button>
            </div>
            <div className="clocks">
              <span>
                SOURCE <strong>{seconds(clock)}</strong>
              </span>
              <span>
                OUTPUT{" "}
                <strong>
                  {seconds(outputTime)} / {seconds(total)}
                </strong>
              </span>
            </div>
            <label className="check-label tiny">
              <input
                type="checkbox"
                checked={sequence}
                onChange={(e) => {
                  setSequence(e.target.checked);
                  if (e.target.checked) seek(segment.start);
                }}
              />
              Play edit sequence
            </label>
          </div>
          <input
            className="source-scrubber"
            type="range"
            min={0}
            max={sourceDuration || 1}
            step={1 / draft.output.fps}
            value={Math.min(clock, sourceDuration)}
            onChange={(e) => seek(+e.target.value)}
            aria-label="Scrub source video"
          />
          <div className="preview-note">
            Edit preview uses source audio and approximate caption typography.
            Review a rendered preview for the selected audio track, effects,
            final framing and captions.
          </div>
          <div className="row">
            <button
              className="button secondary small"
              onClick={() => void showRenderedPreview()}
              disabled={rendering}
            >
              <Icon name="play" size={15} />
              {!unsavedPreviewChanges && currentPreview.exported
                ? "View rendered preview"
                : !unsavedPreviewChanges && currentPreview.job
                  ? "View render progress"
                  : "Render accurate preview"}
            </button>
            <span className="tiny subtle">
              Reuses a finished render of the same saved revision.
            </span>
          </div>
        </section>
        <aside className="inspector" aria-label="Inspector">
          <div className="inspector-tabs">
            {[
              ["framing", "Framing"],
              ["captions", "Captions"],
              ["audio", "Audio"],
            ].map(([id, label]) => (
              <button
                key={id}
                className={panel === id ? "active" : ""}
                onClick={() => setPanel(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="inspector-content">
            {panel === "framing" ? (
              <>
                <div className="section-kicker">CANVAS</div>
                <div className="ratio-grid">
                  {[
                    [9, 16],
                    [1, 1],
                    [4, 5],
                    [16, 9],
                  ].map(([w, h]) => (
                    <button
                      key={w + ":" + h}
                      className={
                        Math.abs(
                          draft.output.width / draft.output.height - w / h,
                        ) < 0.01
                          ? "active"
                          : ""
                      }
                      onClick={() =>
                        edit((t) => {
                          const unit =
                            w === 16 ? 40 : w === 9 ? 40 : w === 4 ? 90 : 360;
                          t.output.width = w * unit;
                          t.output.height = h * unit;
                        })
                      }
                    >
                      <span style={{ aspectRatio: `${w}/${h}` }} />
                      {w}:{h}
                    </button>
                  ))}
                </div>
                <div className="form-grid">
                  <NumberField
                    label="Width"
                    value={draft.output.width}
                    min={144}
                    max={1920}
                    step={2}
                    decimals={0}
                    validate={(n) => Number.isInteger(n) && n % 2 === 0}
                    onCommit={(n) =>
                      edit((t) => {
                        t.output.width = n;
                      }, "output.width")
                    }
                  />
                  <NumberField
                    label="Height"
                    value={draft.output.height}
                    min={144}
                    max={1920}
                    step={2}
                    decimals={0}
                    validate={(n) => Number.isInteger(n) && n % 2 === 0}
                    onCommit={(n) =>
                      edit((t) => {
                        t.output.height = n;
                      }, "output.height")
                    }
                  />
                </div>
                <p className="tiny subtle">
                  Even dimensions from 144 to 1920 pixels.
                </p>
                <Field label="Layout">
                  <select
                    value={segment.crop.mode}
                    onChange={(e) =>
                      updateSegment((s) => {
                        s.crop.mode = e.target.value as Crop["mode"];
                      })
                    }
                  >
                    <option value="fit">Fit / letterbox</option>
                    <option value="crop">Fill / manual crop</option>
                    <option value="split">Split screen / fixed crops</option>
                  </select>
                </Field>
                {segment.crop.mode !== "fit" && (
                  <>
                    <Range
                      label="Horizontal position"
                      value={segment.crop.x}
                      min={0}
                      max={1}
                      step={0.01}
                      onChange={(n) =>
                        updateSegment((s) => {
                          s.crop.x = n;
                        }, "crop.x")
                      }
                    />
                    <Range
                      label="Vertical position"
                      value={segment.crop.y}
                      min={0}
                      max={1}
                      step={0.01}
                      onChange={(n) =>
                        updateSegment((s) => {
                          s.crop.y = n;
                        }, "crop.y")
                      }
                    />
                    {segment.crop.mode !== "split" && (
                      <Range
                        label="Zoom"
                        value={segment.crop.zoom}
                        min={1}
                        max={3}
                        step={0.01}
                        onChange={(n) =>
                          updateSegment((s) => {
                            s.crop.zoom = n;
                          }, "crop.zoom")
                        }
                      />
                    )}
                    {segment.crop.mode === "split" && (
                      <Range
                        label="Second panel horizontal position"
                        value={segment.crop.second_x}
                        min={0}
                        max={1}
                        step={0.01}
                        onChange={(n) =>
                          updateSegment((s) => {
                            s.crop.second_x = n;
                          }, "crop.second_x")
                        }
                      />
                    )}
                    {segment.crop.mode !== "split" && (
                      <div className="row wrap">
                        <button
                          className="button secondary small"
                          onClick={() =>
                            updateSegment((s) => {
                              s.crop.keyframes = [
                                ...s.crop.keyframes.filter(
                                  (k) => Math.abs(k.time - clock) > 0.001,
                                ),
                                {
                                  time: +clock.toFixed(4),
                                  x: s.crop.x,
                                  y: s.crop.y,
                                },
                              ].sort((a, b) => a.time - b.time);
                            })
                          }
                        >
                          Add crop keyframe at {seconds(clock)}
                        </button>
                        {segment.crop.keyframes.length > 0 && (
                          <button
                            className="text-button"
                            onClick={() =>
                              updateSegment((s) => {
                                s.crop.keyframes = [];
                              })
                            }
                          >
                            Clear keyframes
                          </button>
                        )}
                      </div>
                    )}
                    {segment.crop.mode !== "split" &&
                      segment.crop.keyframes.length > 0 && (
                        <div className="keyframes">
                          {segment.crop.keyframes.map((k, i) => (
                            <div key={k.time}>
                              <button
                                onClick={() => seek(k.time)}
                                className="text-button"
                              >
                                ◆ {seconds(k.time)} · {k.x.toFixed(2)},{" "}
                                {k.y.toFixed(2)}
                              </button>
                              <button
                                className="icon-btn"
                                aria-label={"Remove keyframe " + (i + 1)}
                                onClick={() =>
                                  updateSegment((s) => {
                                    s.crop.keyframes.splice(i, 1);
                                  })
                                }
                              >
                                <Icon name="close" size={13} />
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    <button
                      className="button secondary small full"
                      onClick={() => setTrackOpen(true)}
                    >
                      <Icon name="eye" size={15} />
                      {trackingRunning
                        ? "Tracking in progress…"
                        : latestTracking
                          ? "Review tracking result"
                          : "Assisted tracking"}
                    </button>
                  </>
                )}
                <div className="inspector-divider" />
                <div className="section-kicker">EXPORT QUALITY</div>
                <Field label="Frame rate">
                  <select
                    value={draft.output.fps}
                    onChange={(e) =>
                      edit((t) => {
                        t.output.fps = +e.target.value;
                      })
                    }
                  >
                    {[24, 25, 30, 60].map((n) => (
                      <option key={n} value={n}>
                        {n} fps
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="H.264 quality">
                  <select
                    value={draft.output.crf}
                    onChange={(e) =>
                      edit((t) => {
                        t.output.crf = +e.target.value;
                      })
                    }
                  >
                    <option value={28}>Compact · CRF 28</option>
                    <option value={22}>Balanced · CRF 22</option>
                    <option value={18}>High quality · CRF 18</option>
                  </select>
                </Field>
              </>
            ) : panel === "audio" ? (
              <>
                <div className="section-kicker">SELECTED SEGMENT</div>
                <Field
                  label="Export audio track"
                  hint="Edit preview uses the first track. Render an accurate preview to hear the selected track."
                >
                  <select
                    value={draft.audio.stream_index ?? 0}
                    onChange={(e) =>
                      edit((t) => {
                        t.audio.stream_index = +e.target.value;
                      })
                    }
                  >
                    {Array.from(
                      {
                        length: Math.max(
                          1,
                          Number(
                            p.source?.audio_stream_count ??
                              p.source?.metadata?.audio_stream_count ??
                              1,
                          ),
                        ),
                      },
                      (_, i) => (
                        <option key={i} value={i}>
                          Track {i + 1}
                        </option>
                      ),
                    )}
                  </select>
                </Field>
                <Range
                  label="Audio gain (dB)"
                  value={segment.gain_db}
                  min={-40}
                  max={12}
                  step={0.5}
                  onChange={(n) =>
                    updateSegment((s) => {
                      s.gain_db = n;
                    }, "gain")
                  }
                />
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={draft.audio.normalize}
                    onChange={(e) =>
                      edit((t) => {
                        t.audio.normalize = e.target.checked;
                      })
                    }
                  />
                  Normalize output loudness
                </label>
                <Range
                  label="Fade duration (seconds)"
                  value={draft.audio.fade}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(n) =>
                    edit((t) => {
                      t.audio.fade = n;
                    }, "audio.fade")
                  }
                />
                <div className="inspector-divider" />
                <div className="section-kicker">PAUSE CUTS</div>
                <p className="info-note">
                  Audio follows the same source segments as the video. Review
                  pauses before cutting to preserve breaths and context.
                </p>
                <NumberField
                  label="Minimum pause (seconds)"
                  value={pauseGap}
                  min={0.3}
                  max={10}
                  step={0.1}
                  decimals={2}
                  onCommit={setPauseGap}
                />
                <button
                  className="button secondary small full"
                  disabled={pauseBusy || !p.transcript?.words?.length}
                  onClick={async () => {
                    setPauseBusy(true);
                    try {
                      const result = await post<{
                        segments: { start: number; end: number }[];
                        duration: number;
                        method: string;
                      }>("/projects/" + p.id + "/silence-plan", {
                        start: segment.start,
                        end: segment.end,
                        gap: pauseGap,
                        fillers: false,
                      });
                      setPausePlan({ ...result, segmentId: segment.id });
                    } catch (e) {
                      onError(message(e));
                    } finally {
                      setPauseBusy(false);
                    }
                  }}
                >
                  {pauseBusy ? "Inspecting pauses…" : "Suggest pause cuts"}
                </button>
                {!p.transcript?.words?.length && (
                  <p className="tiny subtle">
                    Transcribe first to inspect timestamped speech pauses.
                  </p>
                )}
              </>
            ) : (
              <>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={draft.captions.enabled}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.enabled = e.target.checked;
                      })
                    }
                  />
                  Enable captions
                </label>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={draft.captions.burn_in}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.burn_in = e.target.checked;
                      })
                    }
                  />
                  Burn captions into video
                </label>
                <Field label="Caption template">
                  <select
                    value={draft.captions.style}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.style = e.target.value;
                        t.captions.highlight = e.target.value === "accent";
                        t.captions.background = e.target.value === "boxed";
                      })
                    }
                  >
                    <option value="clean">Clean</option>
                    <option value="boxed">Quiet box</option>
                    <option value="accent">Word accent</option>
                  </select>
                </Field>
                <Field label="Caption font">
                  <select
                    value={draft.captions.font_family ?? "DejaVu Sans"}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.font_family = e.target.value as NonNullable<
                          Timeline["captions"]["font_family"]
                        >;
                      })
                    }
                  >
                    {["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"].map(
                      (n) => (
                        <option key={n}>{n}</option>
                      ),
                    )}
                  </select>
                </Field>
                <div className="form-grid">
                  <Field label="Text color">
                    <input
                      type="color"
                      value={draft.captions.color ?? "#FFFFFF"}
                      onChange={(e) =>
                        edit((t) => {
                          t.captions.color = e.target.value;
                        }, "captions.color")
                      }
                    />
                  </Field>
                  <Field label="Highlight color">
                    <input
                      type="color"
                      value={draft.captions.highlight_color ?? "#D3EC78"}
                      onChange={(e) =>
                        edit((t) => {
                          t.captions.highlight_color = e.target.value;
                        }, "captions.highlight_color")
                      }
                    />
                  </Field>
                </div>
                <Range
                  label="Text stroke (output pixels)"
                  value={draft.captions.stroke_width ?? 2}
                  min={0}
                  max={8}
                  step={0.5}
                  onChange={(n) =>
                    edit((t) => {
                      t.captions.stroke_width = n;
                    }, "captions.stroke")
                  }
                />
                <Range
                  label="Font size (output pixels)"
                  value={draft.captions.font_size}
                  min={12}
                  max={100}
                  step={1}
                  onChange={(n) =>
                    edit((t) => {
                      t.captions.font_size = n;
                    }, "captions.size")
                  }
                />
                <Range
                  label="Safe margin (output pixels)"
                  value={draft.captions.margin}
                  min={10}
                  max={400}
                  step={1}
                  onChange={(n) =>
                    edit((t) => {
                      t.captions.margin = n;
                    }, "captions.margin")
                  }
                />
                <Field label="Alignment">
                  <select
                    value={draft.captions.alignment}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.alignment = +e.target.value;
                      })
                    }
                  >
                    <option value={2}>Bottom center</option>
                    <option value={1}>Bottom left</option>
                    <option value={3}>Bottom right</option>
                    <option value={8}>Top center</option>
                    <option value={7}>Top left</option>
                    <option value={9}>Top right</option>
                  </select>
                </Field>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={draft.captions.background}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.background = e.target.checked;
                      })
                    }
                  />
                  Caption background
                </label>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={draft.captions.highlight}
                    disabled={!draft.captions.words.length}
                    onChange={(e) =>
                      edit((t) => {
                        t.captions.highlight = e.target.checked;
                      })
                    }
                  />
                  Highlight timestamped words
                </label>
                <p className="tiny subtle">
                  {p.transcript?.alignment ?? "No alignment yet"}. Word timing
                  must come from the provider or your edits.
                </p>
                {!draft.captions.words.length && p.transcript?.words?.length ? (
                  <button
                    className="button secondary small"
                    onClick={() =>
                      edit((t) => {
                        t.captions.words = p.transcript!.words.map(
                          ({ start, end, word }) => ({ start, end, word }),
                        );
                      })
                    }
                  >
                    Use transcript word timings
                  </button>
                ) : null}
                <button
                  className="button secondary small full"
                  onClick={() => setTranscriptOpen(true)}
                >
                  Edit words & source timings
                </button>
                <p className="tiny subtle">
                  DejaVu fonts · SRT and VTT sidecars included. Language and
                  font coverage vary; inspect your rendered captions.
                </p>
              </>
            )}
          </div>
        </aside>
      </div>
      <section className="timeline-panel">
        <div className="timeline-toolbar">
          <div className="row">
            <strong>Timeline</strong>
            <Badge>{seconds(total)}</Badge>
            <span className="subtle tiny">OUTPUT TIME</span>
          </div>
          <div className="row">
            <button className="button secondary small" onClick={split}>
              <Icon name="cut" size={15} />
              Split <kbd>S</kbd>
            </button>
            <button
              className="icon-btn"
              onClick={() => reorder(-1)}
              disabled={selected === 0}
              aria-label="Move segment earlier"
              title="Move segment earlier"
            >
              <Icon name="back" size={17} />
            </button>
            <button
              className="icon-btn"
              onClick={() => reorder(1)}
              disabled={selected === draft.segments.length - 1}
              aria-label="Move segment later"
              title="Move segment later"
            >
              <Icon name="arrow" size={17} />
            </button>
            <button
              className="icon-btn"
              onClick={remove}
              disabled={draft.segments.length === 1}
              aria-label="Delete segment"
              title="Delete segment (Delete)"
            >
              <Icon name="trash" size={16} />
            </button>
            <button
              className="icon-btn"
              onClick={() => setShortcuts(true)}
              aria-label="Keyboard shortcuts"
              title="Keyboard shortcuts"
            >
              <Icon name="keyboard" />
            </button>
          </div>
        </div>
        <div className="timeline-ruler" aria-hidden="true">
          {ticks(total).map((t) => (
            <span
              key={t}
              className="tick"
              style={{ left: `${total ? (t / total) * 100 : 0}%` }}
            >
              {seconds(t)}
            </span>
          ))}
          <span className="tick end">{seconds(total)}</span>
          {total > 0 && (
            <i
              className="playhead"
              style={{
                left: `${(Math.min(outputTime, total) / total) * 100}%`,
              }}
            />
          )}
        </div>
        <div className="timeline-lane">
          <span className="track-label" title="Output clips">
            <Icon name="film" size={16} />
          </span>
          <div
            className="clip-track"
            role="group"
            aria-label="Output clips. Drag a clip to reorder it, drag its edges to trim, or use the arrow keys to move between clips."
          >
            {draft.segments.map((s, i) => (
              <button
                key={s.id}
                className={
                  "timeline-clip " +
                  (i === selected ? "selected " : "") +
                  (drop?.from === i ? "dragging" : "")
                }
                style={{ flexGrow: s.end - s.start, flexBasis: 0 }}
                onPointerDown={(e) => startReorder(e, i)}
                onClick={() => {
                  if (dragged.current) {
                    dragged.current = false;
                    return;
                  }
                  selectSegment(i);
                }}
                onKeyDown={(e) => clipKey(e, i)}
                aria-pressed={i === selected}
                title={`${seconds(s.start)}–${seconds(s.end)} source · drag to reorder, drag edges to trim`}
              >
                <i
                  className="trim-handle start"
                  aria-hidden="true"
                  onPointerDown={(e) =>
                    startTrim(e, i, "start", ".clip-track", total)
                  }
                />
                <span>{String(i + 1).padStart(2, "0")}</span>
                <strong>{(s.end - s.start).toFixed(1)}s</strong>
                <span className="sr-only">
                  {`, segment ${i + 1}, source ${seconds(s.start)} to ${seconds(s.end)}`}
                </span>
                <i
                  className="trim-handle end"
                  aria-hidden="true"
                  onPointerDown={(e) =>
                    startTrim(e, i, "end", ".clip-track", total)
                  }
                />
              </button>
            ))}
            {drop && (
              <i
                className="drop-marker"
                style={{ left: drop.x }}
                aria-hidden="true"
              />
            )}
            {total > 0 && (
              <i
                className="playhead"
                style={{
                  left: `${(Math.min(outputTime, total) / total) * 100}%`,
                }}
              />
            )}
          </div>
        </div>
        <div className="timeline-lane waveform-lane">
          <span className="track-label" title="Source audio">
            <Icon name="volume" size={16} />
          </span>
          <div
            className="waveform-wrap"
            onPointerDown={seekFromSourceLane}
            role="presentation"
          >
            {waveform.length ? (
              <svg
                viewBox={`0 0 ${waveform.length} 44`}
                preserveAspectRatio="none"
                aria-label="Source audio waveform"
              >
                <path
                  d={waveform
                    .map(
                      (v, i) =>
                        `M${i},${22 - Math.min(1, v) * 20}v${Math.min(1, v) * 40}`,
                    )
                    .join(" ")}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth=".8"
                />
              </svg>
            ) : (
              <span className="tiny subtle lane-note">
                {p.source?.audio === false
                  ? "No audio stream"
                  : "Waveform unavailable"}
              </span>
            )}
            {sourceDuration > 0 &&
              draft.segments.map((s, i) => (
                <span
                  key={s.id}
                  className={
                    "source-range " + (i === selected ? "selected" : "")
                  }
                  style={{
                    left: `${(s.start / sourceDuration) * 100}%`,
                    width: `${((s.end - s.start) / sourceDuration) * 100}%`,
                  }}
                >
                  <i
                    className="trim-handle start"
                    aria-hidden="true"
                    onPointerDown={(e) =>
                      startTrim(e, i, "start", ".waveform-wrap", sourceDuration)
                    }
                  />
                  <i
                    className="trim-handle end"
                    aria-hidden="true"
                    onPointerDown={(e) =>
                      startTrim(e, i, "end", ".waveform-wrap", sourceDuration)
                    }
                  />
                </span>
              ))}
            {sourceDuration > 0 && (
              <i
                className="playhead"
                style={{
                  left: `${(Math.min(clock, sourceDuration) / sourceDuration) * 100}%`,
                }}
              />
            )}
            <small>
              Source · {seconds(sourceDuration)} · click to seek, drag range
              edges to trim
            </small>
          </div>
        </div>
        <div className="boundary-controls">
          <span className="tiny subtle">
            SEGMENT {selected + 1} · SOURCE TIMES
          </span>
          <NumberField
            label="In, seconds"
            value={segment.start}
            min={0}
            max={segment.end - 0.04}
            step={0.01}
            onCommit={(n) =>
              updateSegment((s) => {
                s.start = n;
              }, "segment.start")
            }
          />
          <button
            className="text-button"
            onClick={markIn}
            title="Set in at playhead (I)"
          >
            Set in at playhead
          </button>
          <NumberField
            label="Out, seconds"
            value={segment.end}
            min={segment.start + 0.04}
            max={sourceDuration || undefined}
            step={0.01}
            onCommit={(n) =>
              updateSegment((s) => {
                s.end = n;
              }, "segment.end")
            }
          />
          <button
            className="text-button"
            onClick={markOut}
            title="Set out at playhead (O)"
          >
            Set out at playhead
          </button>
          <span className="timeline-hint">Autosaved · non-destructive</span>
        </div>
      </section>
      {renderPreviewOpen && (
        <Modal
          title="Rendered preview"
          onClose={() => setRenderPreviewOpen(false)}
        >
          <div className="modal-body stack">
            <div role="status" aria-live="polite">
              {previewRevision === null
                ? rendering
                  ? "Saving the current edit…"
                  : "The edit could not be saved for preview."
                : `Preview of saved revision ${previewRevision}.`}
              {renderedPreview.stale && (
                <p className="warning">
                  This render is from an earlier edit. Your current changes are
                  not included. Render the latest edit before reviewing it.
                </p>
              )}
            </div>
            {previewError && <ErrorBox error={previewError} />}
            {renderedPreview.exported ? (
              <>
                <div
                  className="export-preview"
                  style={{ height: "min(55vh, 580px)" }}
                >
                  <video
                    key={renderedPreview.exported.id}
                    src={renderedPreview.exported.video_url}
                    poster={renderedPreview.exported.thumbnail_url}
                    controls
                    playsInline
                    preload="metadata"
                    aria-label={`Rendered video preview revision ${previewRevision}`}
                    onError={() =>
                      setPreviewError(
                        "The rendered file could not be loaded. Check the export in the Exports tab.",
                      )
                    }
                  />
                </div>
                <p className="subtle">
                  This is the verified export: selected audio track, gain,
                  normalization, fades, framing and any burned-in captions.
                  Sidecar-only captions are supplied in the download package.
                </p>
                <a
                  className="button secondary small"
                  href={
                    String(
                      renderedPreview.exported.package_url ??
                        renderedPreview.exported.download_url ??
                        "",
                    ) ||
                    "/api/v1/exports/" +
                      renderedPreview.exported.id +
                      "/download"
                  }
                  download
                >
                  <Icon name="download" size={15} /> Download this revision
                </a>
              </>
            ) : previewRevision !== null && !previewError ? (
              <div role="status" aria-live="polite">
                {renderedPreview.job?.status === "failed" ? (
                  <ErrorBox
                    error={
                      "Render failed: " + message(renderedPreview.job.error)
                    }
                  />
                ) : renderedPreview.job?.status === "cancelled" ? (
                  <p>
                    Render cancelled. Your source and saved edit are preserved.
                  </p>
                ) : (
                  <p>
                    {renderedPreview.job?.status === "succeeded"
                      ? "Render finished. Loading the verified export…"
                      : renderedPreview.job?.stage ||
                        "Render queued. Waiting for the local worker…"}{" "}
                    You can close this preview while the job runs.
                  </p>
                )}
              </div>
            ) : null}
            {(renderedPreview.stale ||
              previewError ||
              ["failed", "cancelled"].includes(
                renderedPreview.job?.status ?? "",
              )) && (
              <button
                className="button primary small"
                disabled={rendering}
                onClick={() => void showRenderedPreview()}
              >
                {rendering ? "Queuing…" : "Render latest edit"}
              </button>
            )}
          </div>
        </Modal>
      )}
      {historyOpen && (
        <Modal title="Revision history" onClose={() => setHistoryOpen(false)}>
          <div className="modal-body stack">
            <p className="subtle">
              Every autosave is a numbered revision. Restoring loads an older
              edit into the editor as a new revision; nothing is deleted and the
              restore itself can be undone.
            </p>
            {revisionsError && <ErrorBox error={revisionsError} />}
            {revisions === null && !revisionsError ? (
              <p className="subtle">Loading revisions…</p>
            ) : (
              <div className="revision-list">
                {(revisions ?? []).map((r) => {
                  const t = r.timeline,
                    isCurrent = r.revision === revision,
                    rendered = p.exports?.some(
                      (e) => e.revision === r.revision && e.video_url,
                    );
                  return (
                    <div
                      className={`revision-row ${isCurrent ? "current" : ""}`}
                      key={r.revision}
                      data-revision={r.revision}
                    >
                      <div className="revision-number">r{r.revision}</div>
                      <div className="revision-info">
                        <strong>{when(r.created_at) || "Unknown time"}</strong>
                        <span>
                          {t.segments.length}{" "}
                          {t.segments.length === 1 ? "segment" : "segments"} ·{" "}
                          {seconds(duration(t))} · {t.output.width} ×{" "}
                          {t.output.height} · captions{" "}
                          {t.captions.enabled
                            ? t.captions.burn_in
                              ? "burned in"
                              : "sidecar"
                            : "off"}
                        </span>
                        <span className="mono tiny">
                          {t.segments
                            .slice(0, 4)
                            .map((s) => `${seconds(s.start)}–${seconds(s.end)}`)
                            .join("  ")}
                          {t.segments.length > 4 ? "  …" : ""}
                        </span>
                      </div>
                      <div className="row revision-actions">
                        {isCurrent && <Badge tone="accent">Current</Badge>}
                        {rendered && <Badge tone="success">Rendered</Badge>}
                        {!isCurrent && (
                          <button
                            className="button secondary small"
                            onClick={() => restoreRevision(r)}
                          >
                            Restore
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
                {revisions?.length === 0 && (
                  <p className="subtle">No revisions recorded yet.</p>
                )}
              </div>
            )}
          </div>
        </Modal>
      )}
      {presetOpen && (
        <Modal
          title="Reusable edit presets"
          onClose={() => setPresetOpen(false)}
        >
          <div className="modal-body stack">
            <p className="subtle">
              Apply canvas, framing style, captions, and audio settings without
              replacing your source intervals.
            </p>
            {presets.map((pr) => (
              <button
                className="preset-row"
                key={pr.id}
                onClick={() => {
                  edit((t) => {
                    t.output = clone(pr.timeline.output);
                    t.audio = clone(pr.timeline.audio);
                    t.captions = {
                      ...clone(pr.timeline.captions),
                      words: t.captions.words,
                    };
                    for (const s of t.segments)
                      s.crop = {
                        ...clone(pr.timeline.segments[0].crop),
                        keyframes: [],
                      };
                  });
                  setPresetOpen(false);
                }}
              >
                {pr.name}
                <Icon name="arrow" size={16} />
              </button>
            ))}
            <Field label="Save current settings as">
              <input
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                placeholder="My portrait caption style"
              />
            </Field>
            <button
              className="button primary"
              disabled={!presetName.trim()}
              onClick={async () => {
                try {
                  await post("/presets", { name: presetName, timeline: draft });
                  setPresets(list<Preset>(await api("/presets"), "presets"));
                  setPresetName("");
                } catch (e) {
                  onError(message(e));
                }
              }}
            >
              Save preset
            </button>
          </div>
        </Modal>
      )}
      {shortcuts && (
        <Modal title="Keyboard shortcuts" onClose={() => setShortcuts(false)}>
          <div className="modal-body shortcuts">
            <p>
              <kbd>Space</kbd> Play / pause
            </p>
            <p>
              <kbd>←</kbd>
              <kbd>→</kbd> Step one output frame interval
            </p>
            <p>
              <kbd>Shift + ←</kbd>
              <kbd>Shift + →</kbd> Step one second
            </p>
            <p>
              <kbd>Home</kbd>
              <kbd>End</kbd> Jump to the selected segment’s in / out point
            </p>
            <p>
              <kbd>I</kbd>
              <kbd>O</kbd> Set in / out at the playhead
            </p>
            <p>
              <kbd>S</kbd> Split selected segment at playhead
            </p>
            <p>
              <kbd>Delete</kbd> Remove selected segment
            </p>
            <p>
              <kbd>Alt + ←</kbd>
              <kbd>Alt + →</kbd> Move the selected segment earlier / later
            </p>
            <p>
              <kbd>←</kbd>
              <kbd>→</kbd> on a timeline clip: select the previous / next clip
            </p>
            <p>
              <kbd>Ctrl / ⌘ + Z</kbd> Undo
            </p>
            <p>
              <kbd>Ctrl / ⌘ + Shift + Z</kbd> Redo
            </p>
            <p>
              <kbd>Ctrl / ⌘ + S</kbd> Save now
            </p>
            <p className="tiny subtle">
              Browser seeking is time-based; exact decoded frame positioning can
              vary with source codecs.
            </p>
          </div>
        </Modal>
      )}
      {pausePlan && (
        <Modal title="Review pause cuts" onClose={() => setPausePlan(null)}>
          <div className="modal-body stack">
            <p>
              {pausePlan.method}. This would replace the selected segment with{" "}
              {pausePlan.segments.length} intervals, keeping{" "}
              {pausePlan.duration.toFixed(2)} seconds.
            </p>
            <p className="info-note">
              Original media remains intact. Listen to each boundary; timing
              gaps can include meaningful silence, music or breaths. Undo
              restores the previous edit.
            </p>
            <div className="pause-intervals">
              {pausePlan.segments.map((s, i) => (
                <button
                  className="button secondary small"
                  key={i}
                  onClick={() => seek(s.start)}
                >
                  {seconds(s.start)} — {seconds(s.end)}
                </button>
              ))}
            </div>
            <button
              className="button primary"
              disabled={
                !pausePlan.segments.length ||
                draft.segments.length - 1 + pausePlan.segments.length > 100
              }
              onClick={() => {
                edit((t) => {
                  const index = t.segments.findIndex(
                    (s) => s.id === pausePlan.segmentId,
                  );
                  if (index < 0) return;
                  const original = t.segments[index];
                  t.segments.splice(
                    index,
                    1,
                    ...pausePlan.segments.map((s) => ({
                      ...clone(original),
                      id: crypto.randomUUID(),
                      start: s.start,
                      end: s.end,
                      crop: {
                        ...clone(original.crop),
                        keyframes: clone(original.crop.keyframes),
                      },
                    })),
                  );
                });
                setPausePlan(null);
              }}
            >
              Apply reviewed pause cuts
            </button>
          </div>
        </Modal>
      )}
      {transcriptOpen && (
        <TranscriptEditor
          project={p}
          revision={revision}
          onCorrection={(r) => {
            draftRef.current = r.timeline;
            saved.current = JSON.stringify(r.timeline);
            revisionRef.current = r.revision;
            setDraft(r.timeline);
            setRevision(r.revision);
            setHistory([]);
            setFuture([]);
            setSaveStatus("Saved");
          }}
          words={draft.captions.words}
          onClose={() => setTranscriptOpen(false)}
          onSeek={seek}
          onWords={(words) =>
            edit((t) => {
              t.captions.words = words;
            })
          }
          onChange={onChange}
          onError={onError}
        />
      )}
      {trackOpen && (
        <Modal
          title="Assisted subject tracking"
          onClose={() => setTrackOpen(false)}
        >
          <div className="modal-body stack">
            <p className="info-note">
              Tracking suggests crop positions. It does not identify a person or
              determine who is speaking. Review the result and keep fit framing
              when context matters.
            </p>
            {latestTracking && (
              <section className="tracking-result">
                <div className="row between">
                  <h3>Latest result</h3>
                  <Badge
                    tone={
                      latestTracking.status === "assisted" ? "success" : "muted"
                    }
                  >
                    {latestTracking.status}
                  </Badge>
                </div>
                <p className="tiny subtle">
                  {latestTracking.method} · detections in{" "}
                  {Math.round(latestTracking.confidence * 100)}% of sampled
                  frames. {latestTracking.confidence_kind}
                </p>
                <p className="tiny subtle">{latestTracking.fallback}</p>
                <button
                  className="button secondary full"
                  disabled={latestTracking.status !== "assisted"}
                  onClick={() => {
                    const keyframes = clone(latestTracking.keyframes);
                    if (!keyframes.length || keyframes.length > 2000) {
                      onError(
                        "Choose a segment covered by this tracking result with no more than 2,000 keyframes. Shorten the tracking interval if needed.",
                      );
                      return;
                    }
                    updateSegment((s) => {
                      s.crop.mode = "crop";
                      s.crop.keyframes = keyframes;
                      s.crop.x = keyframes[0].x;
                      s.crop.y = keyframes[0].y;
                    });
                    setTrackOpen(false);
                  }}
                >
                  Apply reviewed keyframes to this segment
                </button>
                <details>
                  <summary>Inspect tracking evidence</summary>
                  <pre>{JSON.stringify(latestTracking, null, 2)}</pre>
                </details>
              </section>
            )}
            <Field label="Track method">
              <select
                value={trackMode}
                onChange={(e) => setTrackMode(e.target.value)}
              >
                <option value="face">Detected face</option>
                <option value="object">Chosen object region</option>
              </select>
            </Field>
            {trackMode === "object" && (
              <div className="form-grid">
                {(["x", "y", "width", "height"] as const).map((k) => (
                  <NumberField
                    key={k}
                    label={
                      "Region " +
                      (k === "x" ? "left" : k === "y" ? "top" : k) +
                      " (0–1)"
                    }
                    value={roi[k]}
                    min={0}
                    max={1}
                    step={0.01}
                    onCommit={(n) => setRoi((v) => ({ ...v, [k]: n }))}
                  />
                ))}
              </div>
            )}
            <p>
              Source interval {seconds(segment.start)} — {seconds(segment.end)}
            </p>
            <button
              className="button primary"
              onClick={async () => {
                try {
                  await save();
                  await post("/projects/" + p.id + "/track", {
                    mode: trackMode,
                    start: segment.start,
                    end: segment.end,
                    ...roi,
                  });
                  setTrackOpen(false);
                  await onChange();
                } catch (e) {
                  onError(message(e));
                }
              }}
            >
              Analyze tracking
            </button>
            <p className="tiny subtle">
              Inspect the tracking job result before applying its keyframes. No
              framing changes are assumed from a queued job.
            </p>
          </div>
        </Modal>
      )}
    </>
  );
}
function Range({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="range-field">
      <span>
        {label}
        <output>{Number(value).toFixed(step < 1 ? 2 : 0)}</output>
      </span>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
      />
    </label>
  );
}
function TranscriptEditor({
  project: p,
  revision,
  onCorrection,
  words,
  onWords,
  onSeek,
  onClose,
  onChange,
  onError,
}: {
  project: Project;
  revision: number;
  onCorrection: (r: { revision: number; timeline: Timeline }) => void;
  words: Word[];
  onWords: (w: Word[]) => void;
  onSeek: (n: number) => void;
  onClose: () => void;
  onChange: () => Promise<void>;
  onError: (s: string) => void;
}) {
  const [local, setLocal] = useState(clone(words)),
    [segments, setSegments] = useState(clone(p.transcript?.segments ?? [])),
    [filter, setFilter] = useState(""),
    [start, setStart] = useState(0),
    [end, setEnd] = useState(1),
    [word, setWord] = useState(""),
    [error, setError] = useState("");
  const filtered = local
    .map((w, i) => ({ ...w, i }))
    .filter((w) => w.word.toLowerCase().includes(filter.toLowerCase()));
  const valid = () => {
    for (let i = 0; i < local.length; i++) {
      const w = local[i];
      if (
        !w.word.trim() ||
        w.start < 0 ||
        w.end <= w.start ||
        (i > 0 && local[i - 1].end > w.start + 0.001)
      )
        throw new Error(
          "Words must have text and ordered, non-overlapping source timings.",
        );
    }
    return [...local];
  };
  return (
    <Modal title="Transcript & caption timing" onClose={onClose}>
      <div className="modal-body stack">
        <p className="info-note">
          All times are source seconds. Corrections are saved separately from
          model output. Word alignment is{" "}
          {p.transcript?.alignment ??
            "unavailable until transcription or manual timing"}
          .
        </p>
        {p.transcript?.segment_text_corrected && (
          <p className="info-note">
            Segment corrections affect semantic search. Word captions retain
            their separate text and timing; edit those explicitly below.
          </p>
        )}
        <input
          aria-label="Search caption words"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Find a caption word…"
        />
        {error && <ErrorBox error={error} />}
        <div className="word-editor">
          <div className="word-row word-heading">
            <span>Caption text</span>
            <span>Start</span>
            <span>End</span>
            <span />
          </div>
          {filtered.slice(0, 150).map((w) => (
            <div className="word-row" key={w.i}>
              <input
                aria-label={"Caption word " + (w.i + 1)}
                value={w.word}
                onChange={(e) =>
                  setLocal((v) =>
                    v.map((x, i) =>
                      i === w.i ? { ...x, word: e.target.value } : x,
                    ),
                  )
                }
              />
              <input
                aria-label={"Word " + (w.i + 1) + " source start"}
                type="number"
                min={0}
                step={0.01}
                value={w.start}
                onChange={(e) =>
                  setLocal((v) =>
                    v.map((x, i) =>
                      i === w.i ? { ...x, start: +e.target.value } : x,
                    ),
                  )
                }
              />
              <input
                aria-label={"Word " + (w.i + 1) + " source end"}
                type="number"
                min={0}
                step={0.01}
                value={w.end}
                onChange={(e) =>
                  setLocal((v) =>
                    v.map((x, i) =>
                      i === w.i ? { ...x, end: +e.target.value } : x,
                    ),
                  )
                }
              />
              <button
                className="icon-btn"
                aria-label={"Remove word " + (w.i + 1)}
                onClick={() => setLocal((v) => v.filter((_, i) => i !== w.i))}
              >
                <Icon name="close" size={14} />
              </button>
            </div>
          ))}
        </div>
        {filtered.length > 150 && (
          <p className="tiny subtle">
            Showing first 150 matches. Search to edit a specific word in a long
            transcript.
          </p>
        )}
        <div className="word-row">
          <input
            aria-label="New caption word"
            placeholder="Add a word"
            value={word}
            onChange={(e) => setWord(e.target.value)}
          />
          <input
            aria-label="New word start"
            type="number"
            min={0}
            step={0.01}
            value={start}
            onChange={(e) => setStart(+e.target.value)}
          />
          <input
            aria-label="New word end"
            type="number"
            min={0}
            step={0.01}
            value={end}
            onChange={(e) => setEnd(+e.target.value)}
          />
          <button
            className="icon-btn"
            disabled={!word.trim()}
            aria-label="Add caption word"
            onClick={() => {
              setLocal((v) =>
                [...v, { word, start, end }].sort((a, b) => a.start - b.start),
              );
              setWord("");
            }}
          >
            <Icon name="plus" />
          </button>
        </div>
        <button
          className="button primary"
          onClick={() => {
            try {
              onWords(valid());
              onClose();
            } catch (e) {
              setError(message(e));
            }
          }}
        >
          Apply caption words to edit
        </button>
        <details>
          <summary>Transcript segments ({segments.length})</summary>
          <p className="tiny subtle">
            Editing segment text changes semantic search evidence. Caption words
            and their alignment remain separate; no new word timing is invented.
          </p>
          <div className="transcript-segments">
            {segments.slice(0, 100).map((s, i) => (
              <div key={i}>
                <button
                  className="text-button mono"
                  onClick={() => onSeek(s.start)}
                >
                  {seconds(s.start)} — {seconds(s.end)}
                </button>
                <textarea
                  aria-label={"Transcript segment " + (i + 1)}
                  value={s.text}
                  onChange={(e) =>
                    setSegments((v) =>
                      v.map((x, j) =>
                        j === i ? { ...x, text: e.target.value } : x,
                      ),
                    )
                  }
                />
              </div>
            ))}
          </div>
          <button
            className="button secondary"
            onClick={async () => {
              try {
                const corrected = await api<{
                  revision: number;
                  timeline: Timeline;
                }>("/projects/" + p.id + "/transcript", {
                  method: "PUT",
                  body: JSON.stringify({ words: valid(), segments, revision }),
                });
                onCorrection(corrected);
                await onChange();
                onClose();
              } catch (e) {
                setError(message(e));
                onError(message(e));
              }
            }}
          >
            Save transcript corrections
          </button>
        </details>
      </div>
    </Modal>
  );
}
