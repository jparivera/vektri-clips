import { useEffect, useRef, useState } from "react";
import type { Candidate, CandidateReview, Project } from "./types";
import { Badge, Modal } from "./ui";
import { video } from "./App";
import { reviewPlaybackRange, reviewExcerptLabel } from "./groundedReviewHelpers";

// Decision-only refreshes do not interrupt playback. Evidence/source changes do.
export function reviewIdentity(p: Project, c: Candidate) {
  const { decision: _decision, ...evidence } = c;
  return JSON.stringify([
    p.id,
    p.source?.sha256,
    video(p),
    p.analysis?.artifact_dir,
    p.artifact_base,
    p.analysis?.configuration,
    evidence,
  ]);
}
const time = (value: number) => `${value.toFixed(3)} s`;
function ReviewFrame({
  frame,
  base,
}: {
  frame: CandidateReview["frames"][number];
  base?: string;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <figure className="review-frame">
      {frame.status === "available" && frame.path && base && !failed ? (
        <img
          src={`${base}/${frame.path}`}
          alt={`${frame.role} sample at source ${time(frame.time)}`}
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="review-frame-missing">Frame unavailable</div>
      )}
      <figcaption>
        <strong>{frame.role}</strong> · {time(frame.time)}
        {frame.approximate_seek ? " · approximate seek" : ""}
      </figcaption>
    </figure>
  );
}
export function GroundedReview({
  project: p,
  candidate: c,
  onClose,
  onAccept,
}: {
  project: Project;
  candidate: Candidate;
  onClose: () => void;
  onAccept: () => void;
}) {
  const media = useRef<HTMLVideoElement>(null);
  const [context, setContext] = useState(false);
  const [mediaError, setMediaError] = useState(false);
  const review = c.review;
  const stale =
    !!review && !!p.source?.sha256 && review.source_sha256 !== p.source.sha256;
  const usable = stale ? undefined : review;
  const duration = Number(
    p.source?.duration ?? p.source?.metadata?.duration ?? c.end + 2,
  );
  const range = reviewPlaybackRange(c, duration, context, usable);
  const src = video(p);
  const audioOnly =
    p.source?.video === false || p.source?.metadata?.video === false;
  const track =
    p.analysis?.configuration?.audio_stream ?? p.transcript?.audio_stream;
  const base =
    p.artifact_base && p.analysis?.artifact_dir
      ? `${p.artifact_base.replace(/\/$/, "")}/${p.analysis.artifact_dir.replace(/^\/|\/$/g, "")}`
      : undefined;
  const seekStart = (player: HTMLVideoElement) => {
    player.pause();
    if (player.readyState >= 1) player.currentTime = range.start;
  };
  useEffect(() => {
    const player = media.current;
    if (player) seekStart(player);
  }, [context]);
  useEffect(() => {
    const player = media.current;
    return () => {
      if (player) {
        player.pause();
        player.removeAttribute("src");
        player.load();
      }
    };
  }, []);
  return (
    <Modal title="Review in context" onClose={onClose} wide>
      <div className="modal-body stack grounded-review">
        <div className="row between">
          <h3>{c.title}</h3>
          <Badge>Needs human review</Badge>
        </div>
        <p>{c.rationale}</p>
        <div className="row" role="group" aria-label="Playback interval">
          <button
            className="button secondary small"
            aria-pressed={!context}
            onClick={() => {
              if (media.current) media.current.pause();
              setContext(false);
            }}
          >
            Candidate interval
          </button>
          <button
            className="button secondary small"
            aria-pressed={context}
            onClick={() => {
              if (media.current) media.current.pause();
              setContext(true);
            }}
          >
            Surrounding context
          </button>
        </div>
        <p className="mono tiny">
          Source {time(range.start)} – {time(range.end)} · candidate{" "}
          {time(c.start)} – {time(c.end)}
        </p>
        {src && !mediaError ? (
          <video
            ref={media}
            className={
              "context-video" + (audioOnly ? " audio-only-review" : "")
            }
            src={src}
            controls
            preload="metadata"
            aria-label={
              audioOnly
                ? "Candidate audio preview"
                : "Candidate context preview"
            }
            onLoadedMetadata={(e) => seekStart(e.currentTarget)}
            onError={() => setMediaError(true)}
            onPlay={(e) => {
              const player = e.currentTarget;
              if (
                player.currentTime < range.start ||
                player.currentTime >= range.end
              )
                player.currentTime = range.start;
            }}
            onTimeUpdate={(e) => {
              const player = e.currentTarget;
              if (!player.paused && player.currentTime >= range.end) {
                player.pause();
                player.currentTime = range.end;
              }
            }}
          />
        ) : (
          <p role="status" className="info-note">
            Source preview unavailable. Do not accept this as a verified motion
            review; inspect the original source before editing.
          </p>
        )}
        <p className="tiny subtle">
          Press Play to inspect motion and sound. Playback pauses at the
          selected interval’s end; switching intervals pauses and seeks without
          autoplay. Browser seeking is approximate.
        </p>
        <p className="info-note">
          Source preview plays the first audio track.{" "}
          {track === undefined
            ? "The analysis audio track was not recorded. "
            : `Analysis used track ${track + 1}; ${track !== 0 ? "these tracks differ. " : ""}`}
          Verify the selected track in a rendered preview before judging final
          audio.
        </p>
        {audioOnly && (
          <p className="info-note">
            Audio-only source: visual evidence is unavailable.
          </p>
        )}
        {stale && (
          <p role="alert" className="warn-text">
            Review evidence belongs to a different source. Re-run analysis
            before relying on it.
          </p>
        )}
        {!usable && (
          <p className="info-note">
            {stale ? "No current" : "Legacy candidate: no structured"} evidence
            review is available. Hook, context and payoff are unknown until you
            inspect the source.
          </p>
        )}
        <div className="review-observations">
          {(["hook", "context", "payoff", "signals"] as const).map((key) => {
            if (key === "signals" && !usable?.observations.signals) return null;
            const observation = usable?.observations[key];
            return (
              <section key={key} aria-label={`${key} evidence`}>
                <h4>
                  {key}{" "}
                  <Badge>
                    {observation?.status === "observed"
                      ? "Observed evidence"
                      : "Unknown"}
                  </Badge>
                </h4>
                <p>
                  {observation?.summary ??
                    "Not established by the available evidence."}
                </p>
                {observation?.evidence?.map((item, index) => (
                  <div className="review-evidence" key={index}>
                    <span className="mono tiny">
                      {typeof item.start === "number"
                        ? `${time(item.start)}${typeof item.end === "number" ? ` – ${time(item.end)}` : ""}`
                        : typeof item.time === "number"
                          ? time(item.time)
                          : "Time unavailable"}
                    </span>
                    <p>
                      {typeof item.quote === "string"
                        ? item.quote
                        : typeof item.excerpt === "string"
                          ? item.excerpt
                          : JSON.stringify(item)}
                    </p>
                    {reviewExcerptLabel(item.excerpt) && (
                      <span className="tiny warn-text">
                        {reviewExcerptLabel(item.excerpt)}
                      </span>
                    )}
                  </div>
                ))}
              </section>
            );
          })}
        </div>
        <p className="tiny subtle">
          Observed means a quoted passage or measured signal exists, not that
          the hook works or the event is proven. Editorial completeness remains
          your decision.
        </p>
        <div className="review-frames">
          {usable?.frames.map((frame, index) => (
            <ReviewFrame
              key={`${frame.path}-${index}`}
              frame={frame}
              base={base}
            />
          ))}
        </div>
        <p className="tiny subtle">
          Before / at / after samples locate evidence. Still frames cannot prove
          motion, event order, a hit, a crash or an outcome. Watch the interval
          and surrounding context.
        </p>
        {[
          ...(usable?.warnings ?? []),
          ...(usable?.limitations ?? []),
          ...(Array.isArray(c.uncertainty)
            ? c.uncertainty
            : c.uncertainty
              ? [c.uncertainty]
              : []),
        ].map((warning, index) => (
          <p key={index} className="tiny warn-text">
            {warning}
          </p>
        ))}
        <button className="button primary" onClick={onAccept}>
          Open this cut in editor
        </button>
        <p className="tiny subtle">
          Opening the editor records your acceptance for editing. Nothing is
          automatically accepted, rendered or published.
        </p>
      </div>
    </Modal>
  );
}
