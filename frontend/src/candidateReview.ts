import type { Candidate, Project } from "./types";

/** Feedback is newest-first, as returned by the project API. The same resolved
 * decision drives both candidate cards and inclusion in a multi-clip cut. */
export function resolveCandidateDecisions(
  candidates: readonly Candidate[],
  feedback: NonNullable<Project["feedback"]>,
): Candidate[] {
  const latest = new Map<string, "accept" | "reject">();
  for (const row of feedback) {
    if (!latest.has(row.candidate_id))
      latest.set(row.candidate_id, row.decision);
  }
  return candidates.map((candidate) => ({
    ...candidate,
    decision: latest.get(candidate.id) ?? candidate.decision,
  }));
}

/** Forget picks that became rejected or disappeared; later acceptance must not
 * silently reselect a candidate the operator can no longer see as selected. */
export function eligibleCandidatePicks(
  picked: readonly string[],
  candidates: readonly Candidate[],
): string[] {
  const eligible = new Set(
    candidates.filter((c) => c.decision !== "reject").map((c) => c.id),
  );
  return picked.filter((id) => eligible.has(id));
}
