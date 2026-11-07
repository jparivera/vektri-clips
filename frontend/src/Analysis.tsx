import { useEffect, useState } from "react";
import { api, post, message, seconds } from "./api";
import type { Project, Candidate, Timeline } from "./types";
import { planCut, type CutOrder } from "./cutPlan";
import {
  eligibleCandidatePicks,
  resolveCandidateDecisions,
} from "./candidateReview";
import { Badge, Empty, Field, Icon, Modal, ErrorBox, NumberField } from "./ui";
import { Jobs } from "./App";
import { OCRPanel } from "./OCRPanel";
import { GroundedReview, reviewIdentity } from "./GroundedReview";
import { initialSpeechSettings } from "./analysisSettings";

export function Analysis({
  project: p,
  onChange,
  onEdit,
  onError,
}: {
  project: Project;
  onChange: () => Promise<void>;
  onEdit: () => void;
  onError: (s: string) => void;
}) {
  const savedSpeech = initialSpeechSettings(p.transcript);
  const [genre, setGenre] = useState(p.genre || "mixed"),
    [mode, setMode] = useState("local"),
    [query, setQuery] = useState(p.analysis?.configuration?.query ?? ""),
    [visualModel, setVisualModel] = useState(
      p.analysis?.configuration?.visual_model ?? "clip-vit-b32",
    ),
    [visualSearch, setVisualSearch] = useState(
      p.analysis?.configuration?.impact_detection
        ? false
        : (p.analysis?.configuration?.visual_search ?? p.genre === "action"),
    ),
    [impactDetection, setImpactDetection] = useState(
      p.analysis?.configuration?.impact_detection ?? false,
    ),
    [impactLow, setImpactLow] = useState(
      p.analysis?.configuration?.impact_low_hz ?? 1000,
    ),
    [impactHigh, setImpactHigh] = useState(
      p.analysis?.configuration?.impact_high_hz ?? 6000,
    ),
    [impactThreshold, setImpactThreshold] = useState(
      p.analysis?.configuration?.impact_threshold ?? 0.04,
    ),
    [impactGap, setImpactGap] = useState(
      p.analysis?.configuration?.impact_min_gap ?? 0.6,
    ),
    [impactLead, setImpactLead] = useState(
      p.analysis?.configuration?.impact_lead ?? 0.4,
    ),
    [impactFollow, setImpactFollow] = useState(
      p.analysis?.configuration?.impact_follow ?? 0.5,
    ),
    [speechLanguage, setSpeechLanguage] = useState<string>(
      savedSpeech.language,
    ),
    [speechModel, setSpeechModel] = useState<string>(savedSpeech.model),
    [min, setMin] = useState(4),
    [max, setMax] = useState(10),
    [count, setCount] = useState(5),
    [gap, setGap] = useState(2),
    [transcribe, setTranscribe] = useState(true),
    [audioStream, setAudioStream] = useState(savedSpeech.audioStream),
    [useFeedback, setUseFeedback] = useState(false),
    [busy, setBusy] = useState(false),
    [preview, setPreview] = useState<{ id: string; identity: string } | null>(
      null,
    ),
    [agent, setAgent] = useState(false),
    [packet, setPacket] = useState(""),
    [decisions, setDecisions] = useState(""),
    [agentError, setAgentError] = useState(""),
    [exclusions, setExclusions] = useState(""),
    [eventOpen, setEventOpen] = useState(false),
    [eventTime, setEventTime] = useState(5),
    [before, setBefore] = useState(2),
    [after, setAfter] = useState(4),
    [eventTitle, setEventTitle] = useState("Marked moment"),
    [picked, setPicked] = useState<string[]>([]),
    [order, setOrder] = useState<CutOrder>("source");
  const candidates = resolveCandidateDecisions(
      p.analysis?.candidates ?? [],
      p.feedback ?? [],
    ),
    running = p.jobs?.some(
      (j) =>
        ["analyze", "transcribe"].includes(j.kind) &&
        ["queued", "running"].includes(j.status),
    );
  useEffect(() => {
    const reviewed = resolveCandidateDecisions(
      p.analysis?.candidates ?? [],
      p.feedback ?? [],
    );
    setPicked((previous) => {
      const next = eligibleCandidatePicks(previous, reviewed);
      return next.length === previous.length ? previous : next;
    });
  }, [p.id, p.analysis?.candidates, p.feedback]);
  const impactEnabled =
    impactDetection &&
    ["action", "mixed"].includes(genre) &&
    ["local", "heuristic"].includes(mode);
  const impactBandEmpty =
    Math.ceil(impactLow / 40) > Math.floor(impactHigh / 40);
  const activePreview = candidates.find(
    (c) => c.id === preview?.id && reviewIdentity(p, c) === preview.identity,
  );
  useEffect(() => {
    if (preview && !activePreview) setPreview(null);
  }, [preview, activePreview]);
  const invoke = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      await onChange();
      return true;
    } catch (e) {
      onError(message(e));
      return false;
    } finally {
      setBusy(false);
    }
  };
  const analyze = () => {
    let excluded: { start: number; end: number }[] = [];
    try {
      excluded = exclusions.trim()
        ? exclusions.split(",").map((s) => {
            const [start, end] = s.split("-").map(Number);
            if (
              !Number.isFinite(start) ||
              !Number.isFinite(end) ||
              end <= start
            )
              throw Error(
                "Use exclusion ranges like 2-5, 12-18 (source seconds).",
              );
            return { start, end };
          })
        : [];
    } catch (e) {
      onError(message(e));
      return;
    }
    void invoke(() =>
      post(
        "/projects/" + p.id + "/analyze",
        {
          genre,
          mode,
          query,
          min_length: min,
          max_length: max,
          target_count: count,
          min_separation: gap,
          exclusions: excluded,
          visual_search: mode === "local" && visualSearch,
          visual_model: visualModel,
          impact_detection: impactEnabled,
          impact_low_hz: impactLow,
          impact_high_hz: impactHigh,
          impact_threshold: impactThreshold,
          impact_min_gap: impactGap,
          impact_lead: impactLead,
          impact_follow: impactFollow,
          transcribe: transcribe && !(mode === "local" && visualSearch),
          model: speechModel,
          language: speechLanguage,
          audio_stream: audioStream,
          use_feedback: useFeedback,
        },
        true,
      ),
    );
  };
  const segmentFor = (start: number, end: number) => ({
    id: crypto.randomUUID(),
    start,
    end,
    crop: {
      mode: (genre === "action" ? "fit" : "crop") as "fit" | "crop",
      x: 0.5,
      y: 0.5,
      zoom: 1,
      second_x: 0.75,
      keyframes: [] as { time: number; x: number; y: number }[],
    },
    gain_db: 0,
  });
  const withCaptions = (next: Timeline) => {
    if (p.transcript?.words?.length)
      next.captions.words = p.transcript.words.map(({ start, end, word }) => ({
        start,
        end,
        word,
      }));
    return next;
  };
  const accept = async (c: Candidate) => {
    if (!p.timeline) {
      onError("Wait for source preparation before creating an edit.");
      return;
    }
    const next = withCaptions(structuredClone(p.timeline));
    next.segments = [segmentFor(c.start, c.end)];
    await invoke(async () => {
      await api("/projects/" + p.id + "/timeline", {
        method: "PUT",
        body: JSON.stringify({ revision: p.revision, timeline: next }),
      });
      await post("/projects/" + p.id + "/feedback", {
        candidate_id: c.id,
        decision: "accept",
        note: "Opened in editor",
      });
      onEdit();
    });
  };
  const selectable = candidates.filter((c) => c.decision !== "reject");
  const pickedCandidates = picked
    .map((id) => selectable.find((c) => c.id === id))
    .filter((c): c is Candidate => !!c);
  const pickedIds = new Set(pickedCandidates.map((c) => c.id));
  const plan = planCut(pickedCandidates, order);
  const togglePick = (id: string) =>
    setPicked((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  const buildCut = async () => {
    if (!p.timeline) {
      onError("Wait for source preparation before creating an edit.");
      return;
    }
    if (!plan.ranges.length) return;
    const next = withCaptions(structuredClone(p.timeline));
    next.segments = plan.ranges.map((r) => segmentFor(r.start, r.end));
    const count = plan.ranges.length;
    await invoke(async () => {
      await api("/projects/" + p.id + "/timeline", {
        method: "PUT",
        body: JSON.stringify({ revision: p.revision, timeline: next }),
      });
      for (const c of pickedCandidates)
        await post("/projects/" + p.id + "/feedback", {
          candidate_id: c.id,
          decision: "accept",
          note: `Included in a ${count}-clip cut`,
        });
      setPicked([]);
      onEdit();
    });
  };
  const startAgent = async () => {
    try {
      const r = await api("/projects/" + p.id + "/evidence");
      setPacket(JSON.stringify(r, null, 2));
      setAgent(true);
    } catch (e) {
      onError(message(e));
    }
  };
  return (
    <div
      className={
        "analysis-layout" + (candidates.length ? " has-candidates" : "")
      }
    >
      <aside className="analysis-controls" aria-label="Analysis settings">
        <div className="section-kicker">DISCOVERY SETTINGS</div>
        <h2>What are we looking for?</h2>
        <p className="subtle">
          Give every suggestion a direction. You stay in control of the cut.
        </p>
        <Field label="Footage type">
          <select
            value={genre}
            onChange={(e) => {
              setGenre(e.target.value);
              setVisualSearch(e.target.value === "action");
              setImpactDetection(false);
            }}
          >
            <option value="speech">Podcast / interview</option>
            <option value="tutorial">Tutorial / screen recording</option>
            <option value="action">Action / nonverbal</option>
            <option value="mixed">Mixed footage</option>
          </select>
        </Field>
        <Field label="Analysis mode">
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="local">Local AI + evidence signals</option>
            <option value="heuristic">Heuristic signals only</option>
            <option value="agent">Agent evidence exchange</option>
            <option value="cloud" disabled>
              Cloud / BYOK · not configured
            </option>
          </select>
        </Field>
        <p className="tiny subtle">
          {mode === "local"
            ? visualSearch
              ? visualModel === "xclip-b16-k600"
                ? "Experimental X-CLIP compares ordered eight-frame windows with your description. It can miss short actions and does not verify event order, outcomes or absence. Review every suggestion."
                : "Experimental CLIP search matches sampled frames to your English description. It does not establish an overtake, crash, winner or other event. Review each suggested interval."
              : "English transcript search uses semantic similarity. Other speech uses lexical matching; nonverbal proposals use motion, scene and audio signals."
            : mode === "heuristic"
              ? "No semantic model inference. Suggestions use scene, motion, audio and timing signals."
              : "Download the evidence packet and import validated editorial proposals. No external agent is contacted."}
        </p>
        {mode === "local" && (
          <label className="check-label">
            <input
              type="checkbox"
              checked={visualSearch}
              onChange={(e) => {
                setVisualSearch(e.target.checked);
                if (e.target.checked) setImpactDetection(false);
              }}
            />
            Search visible content (experimental)
          </label>
        )}
        {mode === "local" && visualSearch && (
          <Field label="Visual search model">
            <select
              value={visualModel}
              onChange={(e) => setVisualModel(e.target.value)}
            >
              <option value="clip-vit-b32">CLIP · sampled frames</option>
              <option value="xclip-b16-k600">
                X-CLIP · ordered frame windows
              </option>
            </select>
          </Field>
        )}
        {["local", "heuristic"].includes(mode) &&
          ["action", "mixed"].includes(genre) && (
            <div className="stack">
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={impactDetection}
                  onChange={(e) => {
                    setImpactDetection(e.target.checked);
                    if (e.target.checked) setVisualSearch(false);
                  }}
                />
                Impact sounds (optional)
              </label>
              {impactEnabled && (
                <>
                  <p className="tiny subtle">
                    Find short bursts in a chosen frequency band on the selected
                    analysis audio track. A sound is a lead to inspect, not
                    proof of a hit, crash or outcome. Visual search is off.
                  </p>
                  <details>
                    <summary>Impact sound settings</summary>
                    <div className="form-grid">
                      <NumberField
                        label="Low frequency, Hz"
                        value={impactLow}
                        min={0}
                        max={7999}
                        step={100}
                        decimals={0}
                        onCommit={setImpactLow}
                      />
                      <NumberField
                        label="High frequency, Hz"
                        value={impactHigh}
                        min={1}
                        max={8000}
                        step={100}
                        decimals={0}
                        onCommit={setImpactHigh}
                      />
                      <NumberField
                        label="Impact threshold"
                        value={impactThreshold}
                        min={0.000001}
                        max={1000}
                        step={0.01}
                        decimals={6}
                        onCommit={setImpactThreshold}
                      />
                      <NumberField
                        label="Impact gap, seconds"
                        value={impactGap}
                        min={0.01}
                        max={300}
                        step={0.1}
                        decimals={2}
                        onCommit={setImpactGap}
                      />
                      <NumberField
                        label="Impact lead, seconds"
                        value={impactLead}
                        min={0}
                        max={300}
                        step={0.1}
                        decimals={2}
                        onCommit={setImpactLead}
                      />
                      <NumberField
                        label="Impact follow, seconds"
                        value={impactFollow}
                        min={0}
                        max={300}
                        step={0.1}
                        decimals={2}
                        onCommit={setImpactFollow}
                      />
                    </div>
                    <p className="tiny subtle">
                      Lower thresholds find quieter bursts and more false
                      alarms. The frequency band filters which sounds count.
                      Lead and follow set the interval around an impact; they do
                      not change speech-length controls. Gap separates detected
                      bursts.
                    </p>
                  </details>
                  {impactBandEmpty && (
                    <p className="warn-text">
                      Widen the frequency band to include a 40 Hz measurement
                      step.
                    </p>
                  )}
                  {impactLead + impactFollow < 0.04 && (
                    <p className="warn-text">
                      Impact lead and follow must total at least 0.04 seconds.
                    </p>
                  )}
                  {impactHigh <= impactLow && (
                    <p className="warn-text">
                      High frequency must exceed low frequency.
                    </p>
                  )}
                </>
              )}
            </div>
          )}
        <Field label="Editorial direction">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            placeholder={
              mode === "local" && visualSearch
                ? "Describe visible content in English, e.g. racing cars driving on a race track"
                : "Find the clearest explanation of…"
            }
          />
        </Field>
        {mode === "local" && visualSearch && (
          <p className="tiny subtle">
            {visualModel === "xclip-b16-k600"
              ? "Requires the local X-CLIP model from AI & models. Samples eight frames in at most 60 windows distributed across the video."
              : "Requires the local CLIP model from AI & models. Samples up to 900 frames distributed across the video."}{" "}
            Brief events and gaps between samples can be missed. Results use
            visual relevance only. No transcription runs in this mode.
          </p>
        )}
        <div className="form-grid">
          <NumberField
            label="Minimum seconds"
            value={min}
            min={1}
            max={300}
            step={1}
            decimals={1}
            onCommit={setMin}
          />
          <NumberField
            label="Maximum seconds"
            value={max}
            min={1}
            max={300}
            step={1}
            decimals={1}
            onCommit={setMax}
          />
          <NumberField
            label="Target candidates"
            value={count}
            min={1}
            max={30}
            step={1}
            decimals={0}
            validate={Number.isInteger}
            onCommit={setCount}
          />
          <NumberField
            label="Separation, seconds"
            value={gap}
            min={0}
            max={300}
            step={1}
            decimals={1}
            onCommit={setGap}
          />
        </div>
        {max < min && (
          <p className="tiny warn-text">
            Maximum length must be at least the minimum.
          </p>
        )}
        <Field
          label="Exclude source intervals"
          hint="Optional, in source seconds"
        >
          <input
            value={exclusions}
            onChange={(e) => setExclusions(e.target.value)}
            placeholder="2-5, 12-18"
          />
        </Field>
        <label className="check-label">
          <input
            type="checkbox"
            checked={transcribe}
            disabled={(mode === "local" && visualSearch) || genre === "action"}
            onChange={(e) => setTranscribe(e.target.checked)}
          />
          Transcribe speech when available
        </label>
        {transcribe &&
          !(mode === "local" && visualSearch) &&
          genre !== "action" && (
            <div className="form-grid stacked">
              <Field label="Speech model">
                <select
                  value={speechModel}
                  onChange={(e) => {
                    setSpeechModel(e.target.value);
                    if (e.target.value === "tiny.en") setSpeechLanguage("en");
                  }}
                >
                  <option value="tiny.en">English tiny.en</option>
                  <option value="tiny">Multilingual tiny</option>
                </select>
              </Field>
              <Field label="Speech language">
                <select
                  value={speechLanguage}
                  onChange={(e) => {
                    setSpeechLanguage(e.target.value);
                    if (e.target.value !== "en") setSpeechModel("tiny");
                  }}
                >
                  <option value="en">English</option>
                  <option value="pt">Portuguese</option>
                  <option value="auto">Detect language</option>
                </select>
              </Field>
            </div>
          )}
        {Number(
          p.source?.audio_stream_count ??
            p.source?.metadata?.audio_stream_count ??
            0,
        ) > 1 && (
          <Field label="Transcription audio track">
            <select
              value={audioStream}
              onChange={(e) => setAudioStream(+e.target.value)}
            >
              {Array.from(
                {
                  length: Number(
                    p.source?.audio_stream_count ??
                      p.source?.metadata?.audio_stream_count,
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
        )}
        <label className="check-label">
          <input
            type="checkbox"
            checked={useFeedback}
            onChange={(e) => setUseFeedback(e.target.checked)}
          />
          Exclude previously rejected intervals
        </label>
        {useFeedback && (
          <p className="tiny subtle">
            Saved rejections become visible ranking exclusions. This does not
            train a model.
          </p>
        )}
        <button
          className="button primary full"
          disabled={
            busy ||
            running ||
            max < min ||
            (impactEnabled &&
              (impactHigh <= impactLow ||
                impactBandEmpty ||
                impactLead + impactFollow < 0.04)) ||
            (mode === "local" && visualSearch && !query.trim())
          }
          onClick={mode === "agent" ? () => void startAgent() : analyze}
        >
          <Icon name="spark" />
          {busy
            ? "Queuing…"
            : running
              ? "Analysis in progress"
              : mode === "agent"
                ? "Prepare evidence packet"
                : "Find candidate moments"}
        </button>
        <button
          className="button secondary full"
          onClick={() => setEventOpen(true)}
        >
          <Icon name="plus" />
          Mark a moment manually
        </button>
        <p className="tiny subtle">
          The target count is a preference, not a quota. Weak and overlapping
          candidates may be omitted.
        </p>
      </aside>
      <section className="candidates">
        <div className="candidates-heading">
          <div>
            <div className="eyebrow">REVIEW, REFINE, MAKE THE CUT</div>
            <h2>
              Candidate moments{" "}
              <span className="subtle">
                {candidates.length
                  ? String(candidates.length).padStart(2, "0")
                  : ""}
              </span>
            </h2>
          </div>
          {p.analysis && (
            <Badge tone="accent">
              {candidates.some(
                (c) => c.method === "local_xclip_temporal_retrieval",
              )
                ? "Experimental temporal retrieval"
                : candidates.some((c) => c.method === "local_clip_visual")
                  ? "Experimental visual search"
                  : candidates.some((c) => c.method === "local_ai_semantic")
                    ? "Local AI + signals"
                    : candidates.every((c) => c.method === "manual")
                      ? "Manual selection"
                      : "Evidence signals"}
            </Badge>
          )}
        </div>
        <Jobs
          compact
          jobs={
            p.jobs?.filter((j) => ["analyze", "transcribe"].includes(j.kind)) ??
            []
          }
          onAction={async (fn) => {
            await invoke(fn);
          }}
        />
        <OCRPanel project={p} invoke={invoke} busy={busy} />
        {p.analysis?.limitations?.length ? (
          <div className="info-note">
            <Icon name="info" size={16} />
            <p>{p.analysis.limitations.join(" ")}</p>
          </div>
        ) : null}
        {!candidates.length ? (
          <Empty icon="spark" title="There’s a moment in there.">
            Run an analysis or mark an event yourself. Suggestions will include
            source timestamps, evidence, and uncertainty so you can judge them
            in context.
          </Empty>
        ) : (
          <>
            <div
              className={
                "cut-builder" + (pickedCandidates.length ? " active" : "")
              }
              role="region"
              aria-label="Cut builder"
            >
              <div>
                <strong>
                  {pickedCandidates.length
                    ? `${pickedCandidates.length} selected · ${plan.ranges.length} clip${plan.ranges.length === 1 ? "" : "s"} · ${plan.duration.toFixed(1)} s`
                    : "Build one cut from several candidates"}
                </strong>
                <small>
                  {!pickedCandidates.length
                    ? "Tick candidates below, choose their order, then open them together in the editor. Rejected candidates cannot be included."
                    : order === "source"
                      ? plan.overlaps
                        ? `${plan.overlaps} overlapping range${plan.overlaps === 1 ? "" : "s"} merged so footage is not repeated. Opens in the editor as a new revision; the current timeline stays in History.`
                        : "Clips follow source time. Opens in the editor as a new revision; the current timeline stays in History."
                      : plan.overlaps
                        ? `Clips keep the order you picked; ${plan.overlaps} overlap${plan.overlaps === 1 ? "s" : ""} in source time, so footage repeats.`
                        : "Clips keep the order you picked. Opens in the editor as a new revision; the current timeline stays in History."}
                </small>
              </div>
              <div className="row">
                <select
                  aria-label="Clip order"
                  value={order}
                  onChange={(e) => setOrder(e.target.value as CutOrder)}
                >
                  <option value="source">Source order</option>
                  <option value="selection">Selection order</option>
                </select>
                <button
                  className="text-button"
                  onClick={() => setPicked(selectable.map((c) => c.id))}
                  disabled={
                    !selectable.length ||
                    pickedCandidates.length === selectable.length
                  }
                >
                  Select all
                </button>
                <button
                  className="text-button"
                  onClick={() => setPicked([])}
                  disabled={!picked.length}
                >
                  Clear
                </button>
                <button
                  className="button primary small"
                  disabled={!plan.ranges.length || busy}
                  onClick={() => void buildCut()}
                >
                  <Icon name="layers" size={15} />
                  Build cut
                  {pickedCandidates.length
                    ? ` from ${pickedCandidates.length}`
                    : ""}
                </button>
              </div>
            </div>
            <div className="candidate-list">
              {candidates.map((c, i) => (
                <CandidateCard
                  key={c.id}
                  candidate={c}
                  index={i}
                  picked={pickedIds.has(c.id)}
                  onPick={() => togglePick(c.id)}
                  onPreview={() =>
                    setPreview({ id: c.id, identity: reviewIdentity(p, c) })
                  }
                  onAccept={() => void accept(c)}
                  onReject={async () => {
                    const ok = await invoke(() =>
                      post("/projects/" + p.id + "/feedback", {
                        candidate_id: c.id,
                        decision: "reject",
                        note: "Rejected in candidate review",
                      }),
                    );
                    if (ok)
                      setPicked((prev) => prev.filter((id) => id !== c.id));
                    return ok;
                  }}
                />
              ))}
            </div>
          </>
        )}
      </section>
      {activePreview && (
        <GroundedReview
          key={preview!.identity}
          project={p}
          candidate={activePreview}
          onClose={() => setPreview(null)}
          onAccept={() => {
            void accept(activePreview);
            setPreview(null);
          }}
        />
      )}
      {agent && (
        <Modal title="Agent evidence exchange" onClose={() => setAgent(false)}>
          <div className="modal-body stack">
            <p className="info-note">
              External agents are not connected. Evidence is untrusted source
              content; proposals are schema-validated and tied to this packet’s
              fingerprint.
            </p>
            <a
              className="button secondary"
              href={
                "data:application/json;charset=utf-8," +
                encodeURIComponent(packet)
              }
              download={`evidence-${p.id}.json`}
            >
              <Icon name="download" />
              Download timestamped evidence
            </a>
            <details>
              <summary>Inspect evidence packet</summary>
              <pre>{packet}</pre>
            </details>
            <Field label="Import editorial decisions JSON">
              <textarea
                rows={8}
                value={decisions}
                onChange={(e) => setDecisions(e.target.value)}
                placeholder='{"schema_version":1,"packet_fingerprint":"…","agent_name":"…","candidates":[]}'
              />
            </Field>
            {agentError && <ErrorBox error={agentError} />}
            <button
              className="button primary"
              onClick={async () => {
                try {
                  const parsed = JSON.parse(decisions);
                  await post("/projects/" + p.id + "/agent-decisions", parsed);
                  await onChange();
                  setAgent(false);
                } catch (e) {
                  setAgentError(message(e));
                }
              }}
            >
              Validate and import proposals
            </button>
          </div>
        </Modal>
      )}
      {eventOpen && (
        <Modal title="Mark a source moment" onClose={() => setEventOpen(false)}>
          <form
            className="modal-body stack"
            onSubmit={(e) => {
              e.preventDefault();
              void invoke(async () => {
                await post("/projects/" + p.id + "/event", {
                  time: eventTime,
                  before,
                  after,
                  title: eventTitle,
                });
                setEventOpen(false);
              });
            }}
          >
            <Field label="Factual title">
              <input
                value={eventTitle}
                onChange={(e) => setEventTitle(e.target.value)}
                required
              />
            </Field>
            <NumberField
              label="Source time, seconds"
              value={eventTime}
              min={0}
              max={Number(
                p.source?.duration ?? p.source?.metadata?.duration ?? 86400,
              )}
              step={0.1}
              decimals={2}
              onCommit={setEventTime}
            />
            <div className="form-grid">
              <NumberField
                label="Seconds before"
                value={before}
                min={0}
                max={300}
                step={0.1}
                decimals={2}
                onCommit={setBefore}
              />
              <NumberField
                label="Seconds after"
                value={after}
                min={0.1}
                max={300}
                step={0.1}
                decimals={2}
                onCommit={setAfter}
              />
            </div>
            <p className="tiny subtle">
              Window {seconds(Math.max(0, eventTime - before))} —{" "}
              {seconds(eventTime + after)} in source time.
            </p>
            <button className="button primary" type="submit">
              Add manual candidate
            </button>
          </form>
        </Modal>
      )}
    </div>
  );
}
function CandidateCard({
  candidate: c,
  index,
  picked,
  onPick,
  onPreview,
  onAccept,
  onReject,
}: {
  candidate: Candidate;
  index: number;
  picked: boolean;
  onPick: () => void;
  onPreview: () => void;
  onAccept: () => void;
  onReject: () => Promise<boolean>;
}) {
  const [decision, setDecision] = useState(
    c.decision === "reject"
      ? "rejected"
      : c.decision === "accept"
        ? "accepted"
        : "",
  );
  useEffect(
    () =>
      setDecision(
        c.decision === "reject"
          ? "rejected"
          : c.decision === "accept"
            ? "accepted"
            : "",
      ),
    [c.decision],
  );
  return (
    <article
      data-method={c.method}
      className={
        "candidate-card " +
        (decision === "rejected" ? "rejected " : "") +
        (picked ? "picked" : "")
      }
    >
      <label className="candidate-pick">
        <input
          type="checkbox"
          checked={picked}
          disabled={decision === "rejected"}
          onChange={onPick}
          aria-label={`Include candidate ${index + 1} in the cut`}
        />
      </label>
      <div className="candidate-number">
        {String(index + 1).padStart(2, "0")}
      </div>
      <div className="candidate-copy">
        <div className="row">
          <Badge tone={c.method?.includes("heuristic") ? "muted" : "accent"}>
            {c.method === "local_xclip_temporal_retrieval"
              ? "X-CLIP temporal similarity"
              : c.method === "local_clip_visual"
                ? "CLIP visual similarity"
                : c.method === "local_ai_semantic"
                  ? "Local semantic AI"
                  : c.method === "manual"
                    ? "Manual event"
                    : c.method?.includes("heuristic")
                      ? "Heuristic signals"
                      : c.method?.includes("agent")
                        ? "Agent suggestion"
                        : "Suggestion"}
          </Badge>
          <span className="mono tiny subtle">
            {seconds(c.start)} — {seconds(c.end)} ·{" "}
            {(c.end - c.start).toFixed(1)}s
          </span>
        </div>
        <h3>{c.title}</h3>
        {decision === "accepted" && <Badge>Accepted for editing</Badge>}
        <p>{c.rationale}</p>
        <div className="candidate-caution">
          {Array.isArray(c.uncertainty)
            ? c.uncertainty.join(" ")
            : c.uncertainty}
        </div>
        <div className="candidate-actions">
          <button className="button secondary small" onClick={onPreview}>
            <Icon name="play" size={15} />
            Context
          </button>
          <button className="button primary small" onClick={onAccept}>
            Edit this clip <Icon name="arrow" size={15} />
          </button>
          <button
            className="text-button"
            onClick={async () => {
              if (await onReject()) setDecision("rejected");
            }}
          >
            {decision === "rejected" ? "Rejected" : "Reject"}
          </button>
          <details>
            <summary>Evidence & scores</summary>
            <pre>
              {JSON.stringify(
                {
                  method: c.method,
                  evidence: c.evidence,
                  confidence: c.confidence,
                  component_scores: c.scores,
                },
                null,
                2,
              )}
            </pre>
            <p className="tiny subtle">
              Relevance scores are not calibrated probabilities, event proof, or
              a prediction of virality or revenue.
            </p>
          </details>
        </div>
      </div>
    </article>
  );
}
