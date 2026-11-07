/** Event-name prefixes the audit page can filter by; the API matches them literally. */
export const AUDIT_GROUPS: readonly [string, string][] = [
  ["", "All events"],
  ["project.", "Projects"],
  ["source.", "Source preparation"],
  ["timeline.", "Timeline saves"],
  ["job.", "Jobs"],
  ["analysis.", "Analyses"],
  ["transcript.", "Transcripts"],
  ["event.", "Marked events"],
  ["feedback.", "Candidate decisions"],
  ["agent.", "Agent proposals"],
  ["export.", "Exports"],
  ["publication.", "Publishing"],
  ["preset.", "Presets"],
];
export function auditTone(event: string) {
  return event.endsWith(".failed")
    ? "danger"
    : event.endsWith(".succeeded") || event === "export.verified"
      ? "success"
      : event.includes("cancel") || event === "job.recovered"
        ? "warn"
        : event === "publication.manual_approval" || event === "timeline.saved"
          ? "accent"
          : "muted";
}
export const shortId = (value: unknown) =>
  typeof value === "string" && value.length > 12
    ? value.slice(0, 8) + "…"
    : String(value);
/** Identifiers and hashes are shortened; titles, names and messages are shown whole (up to 80 characters). */
const IDENTIFIER = /(^|_)(id|sha256|fingerprint|hash|key)$/;
export const compact = (key: string, value: unknown) =>
  IDENTIFIER.test(key)
    ? shortId(value)
    : typeof value === "string" && value.length > 80
      ? value.slice(0, 77) + "…"
      : String(value);
/** One line of the facts an operator looks for first; the full record stays in the expandable JSON. */
export function auditSummary(
  event: string,
  detail: Record<string, unknown>,
  clock: (secondsValue: number) => string = (n) => n.toFixed(1) + " s",
): string {
  const parts: string[] = [];
  const scalar = (key: string, label = key) => {
    const value = detail[key];
    if (value !== undefined && value !== null && typeof value !== "object")
      parts.push(`${label} ${compact(key, value)}`);
  };
  if (event.startsWith("job.")) {
    scalar("kind");
    scalar("job_id", "job");
    scalar("status");
    scalar("attempt");
    if (typeof detail.error === "string" && detail.error)
      parts.push(detail.error.slice(0, 160));
  } else if (event === "timeline.saved") scalar("revision");
  else if (event === "feedback.recorded") {
    scalar("decision");
    scalar("candidate_id", "candidate");
    if (typeof detail.note === "string" && detail.note)
      parts.push(`“${detail.note.slice(0, 80)}”`);
  } else if (event === "project.created") scalar("original_name", "file");
  else if (event === "project.updated")
    parts.push(
      Object.keys(detail).length
        ? "changed " + Object.keys(detail).join(", ")
        : "no fields",
    );
  else if (event === "event.marked") {
    scalar("title");
    if (typeof detail.start === "number" && typeof detail.end === "number")
      parts.push(`${clock(detail.start)} – ${clock(detail.end)}`);
  } else if (event.startsWith("publication.")) {
    scalar("destination");
    scalar("id", "publication");
    scalar("export_id", "export");
    if (detail.approval_invalidated) parts.push("approval invalidated");
  } else if (event === "agent.proposals") {
    scalar("agent_name", "agent");
    scalar("count", "proposals");
  } else if (event === "transcript.corrected") {
    scalar("word_count", "words");
    scalar("revision");
  } else if (event === "preset.saved") scalar("name");
  else
    for (const [key, value] of Object.entries(detail)) {
      if (parts.length >= 3) break;
      if (value !== null && typeof value !== "object")
        parts.push(`${key} ${compact(key, value)}`);
    }
  return parts.join(" · ");
}
