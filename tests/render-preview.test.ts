import assert from "node:assert/strict";
import test from "node:test";
import { renderPreviewState } from "../frontend/src/renderPreview.ts";
import type { Export, Job } from "../frontend/src/types.ts";

const projectId = "preview-project";
const exported = (
  revision: number,
  overrides: Partial<Export> = {},
): Export => ({
  id: `export-${revision}`,
  project_id: projectId,
  revision,
  video_url: `/api/v1/artifacts/${projectId}/revision-${revision}/video.mp4`,
  ...overrides,
});
const job = (
  revision: number,
  status = "queued",
  overrides: Partial<Job> = {},
): Job => ({
  id: `job-${revision}`,
  project_id: projectId,
  kind: "render",
  params: { revision },
  status,
  stage: status,
  created_at: "",
  updated_at: "",
  attempts: 0,
  max_attempts: 3,
  ...overrides,
});

test("only a playable export of the requested project and revision is selected", () => {
  const wrongProject = exported(3, { project_id: "other-project" });
  const noVideo = exported(3, { video_url: undefined });
  const matching = exported(3);
  const result = renderPreviewState(
    {
      id: projectId,
      exports: [exported(4), wrongProject, noVideo, matching],
      jobs: [],
    },
    3,
    3,
    false,
  );
  assert.equal(result.exported, matching);
  assert.equal(result.stale, false);
});

test("an older finished export cannot replace the revision currently rendering", () => {
  const pending = job(4, "running", {
    stage: "waiting for at least 1.5 GiB available RAM",
  });
  const result = renderPreviewState(
    { id: projectId, exports: [exported(3)], jobs: [pending] },
    4,
    4,
    false,
  );
  assert.equal(result.exported, undefined);
  assert.equal(result.job, pending);
  assert.equal(result.stale, false);
});

test("unsaved edits and newer saved revisions both invalidate the current-preview claim", () => {
  const project = { id: projectId, exports: [exported(3)], jobs: [] };
  assert.equal(renderPreviewState(project, 3, 3, true).stale, true);
  assert.equal(renderPreviewState(project, 3, 4, false).stale, true);
  assert.equal(renderPreviewState(project, 3, 3, false).stale, false);
});

test("failed or cancelled jobs replace the initial POST response during refresh", () => {
  for (const status of ["failed", "cancelled"]) {
    const initial = job(3);
    const refreshed = job(3, status, { error: "Media validation failed" });
    const result = renderPreviewState(
      { id: projectId, exports: [], jobs: [refreshed] },
      3,
      3,
      false,
      initial,
    );
    assert.equal(result.job, refreshed);
    assert.equal(result.exported, undefined);
  }
});

test("pending POST response bridges refresh delay but never leaks a different job", () => {
  const project = {
    id: projectId,
    exports: [],
    jobs: [job(8), job(3, "running", { project_id: "other-project" })],
  };
  const initial = job(3);
  assert.equal(renderPreviewState(project, 3, 3, false, initial).job, initial);
  assert.equal(renderPreviewState(project, 3, 3, false, job(2)).job, undefined);
  assert.equal(
    renderPreviewState(project, null, 3, true, initial).job,
    undefined,
  );
});
