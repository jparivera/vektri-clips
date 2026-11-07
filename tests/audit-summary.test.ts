import assert from "node:assert/strict";
import test from "node:test";
import {
  AUDIT_GROUPS,
  auditSummary,
  auditTone,
  shortId,
} from "../frontend/src/audit.ts";

test("job events summarise kind, shortened id, status and the error text", () => {
  assert.equal(
    auditSummary("job.failed", {
      job_id: "0123456789abcdef",
      kind: "render",
      error: "ffmpeg exited 1",
      status: "failed",
    }),
    "kind render · job 01234567… · status failed · ffmpeg exited 1",
  );
  assert.equal(
    auditSummary("job.queued", { job_id: "short", kind: "analyze" }),
    "kind analyze · job short",
  );
});

test("editorial events read as decisions, revisions and windows", () => {
  assert.equal(
    auditSummary("timeline.saved", { revision: 4, fingerprint: "abc" }),
    "revision 4",
  );
  assert.equal(
    auditSummary("feedback.recorded", {
      decision: "accept",
      candidate_id: "cand-1",
      note: "Included in a 2-clip cut",
      candidate_snapshot: { start: 1 },
    }),
    "decision accept · candidate cand-1 · “Included in a 2-clip cut”",
  );
  assert.equal(
    auditSummary("event.marked", {
      title: "Opening line",
      start: 2,
      end: 5,
      evidence: [],
    }),
    "title Opening line · 2.0 s – 5.0 s",
  );
  assert.equal(
    auditSummary(
      "event.marked",
      { title: "Opening line", start: 2, end: 5 },
      (n) => `t=${n}`,
    ),
    "title Opening line · t=2 – t=5",
  );
  assert.equal(
    auditSummary("project.updated", { genre: "speech", tags: ["a"] }),
    "changed genre, tags",
  );
  assert.equal(
    auditSummary("publication.edited", {
      id: "pub-1",
      approval_invalidated: true,
    }),
    "publication pub-1 · approval invalidated",
  );
});

test("identifiers and hashes are shortened while titles and file names stay whole", () => {
  assert.equal(
    auditSummary("event.marked", {
      title: "A deliberately long candidate title for review",
      start: 1,
      end: 2,
    }),
    "title A deliberately long candidate title for review · 1.0 s – 2.0 s",
  );
  assert.equal(
    auditSummary("project.created", {
      original_name: "interview-with-a-very-long-file-name.mp4",
      provenance: {},
    }),
    "file interview-with-a-very-long-file-name.mp4",
  );
  assert.equal(
    auditSummary("source.prepared", {
      sha256: "0123456789abcdef0123456789abcdef",
    }),
    "sha256 01234567…",
  );
  assert.equal(
    auditSummary("agent.proposals", {
      agent_name: "Example agent",
      packet_fingerprint: "fedcba9876543210ff",
      count: 3,
    }),
    "agent Example agent · proposals 3",
  );
});

test("unknown events fall back to the first three scalar fields and empty details to an empty line", () => {
  assert.equal(
    auditSummary("source.prepared", {
      width: 1920,
      height: 1080,
      fps: 24,
      nested: { x: 1 },
      codec: "h264",
    }),
    "width 1920 · height 1080 · fps 24",
  );
  assert.equal(auditSummary("analysis.completed", {}), "");
});

test("tones highlight failures, successes, cancellations and approvals", () => {
  assert.equal(auditTone("job.failed"), "danger");
  assert.equal(auditTone("job.succeeded"), "success");
  assert.equal(auditTone("export.verified"), "success");
  assert.equal(auditTone("job.cancel_requested"), "warn");
  assert.equal(auditTone("job.recovered"), "warn");
  assert.equal(auditTone("publication.manual_approval"), "accent");
  assert.equal(auditTone("project.created"), "muted");
});

test("filter groups are literal dotted prefixes and ids shorten only when long", () => {
  assert.ok(
    AUDIT_GROUPS.every(([prefix]) => prefix === "" || prefix.endsWith(".")),
  );
  assert.equal(shortId("0123456789abcdef"), "01234567…");
  assert.equal(shortId("abc"), "abc");
  assert.equal(shortId(7), "7");
});
