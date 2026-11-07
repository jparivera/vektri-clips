import type { Candidate, CandidateReview } from "./types";

export function reviewPlaybackRange(
  candidate: Pick<Candidate, "start" | "end">,
  duration: number,
  context: boolean,
  review?: Pick<CandidateReview, "playback">,
): { start: number; end: number } {
  const start = context
    ? (review?.playback.context_start ?? candidate.start - 2)
    : (review?.playback.start ?? candidate.start);
  const end = context
    ? (review?.playback.context_end ?? candidate.end + 2)
    : (review?.playback.end ?? candidate.end);
  const limit = Math.max(0, duration);
  const boundedStart = Math.min(limit, Math.max(0, start));
  return {
    start: boundedStart,
    end: Math.max(boundedStart, Math.min(limit, Math.max(0, end))),
  };
}

export function reviewExcerptLabel(excerpt: unknown): string | null {
  return excerpt === true ? "Excerpt, truncated" : null;
}
