import type { Export, Job, Project } from "./types";

/** An editor preview must never substitute a render from another revision. */
export function renderPreviewState(
  project: Pick<Project, "id" | "exports" | "jobs">,
  targetRevision: number | null,
  currentRevision: number,
  hasUnsavedChanges: boolean,
  requestedJob?: Job | null,
): { exported?: Export; job?: Job; stale: boolean } {
  const stale =
    targetRevision !== null &&
    (targetRevision !== currentRevision || hasUnsavedChanges);
  if (targetRevision === null) return { stale };
  const exported = project.exports?.find(
    (item) =>
      item.revision === targetRevision &&
      Boolean(item.video_url) &&
      (!item.project_id || item.project_id === project.id),
  );
  const matches = (job: Job) =>
    job.kind === "render" &&
    job.project_id === project.id &&
    job.params?.revision === targetRevision;
  // Prefer the refreshed job over the POST response, which may still say queued.
  const job =
    (requestedJob &&
      project.jobs?.find(
        (item) => item.id === requestedJob.id && matches(item),
      )) ||
    project.jobs?.find(
      (item) => matches(item) && ["queued", "running"].includes(item.status),
    ) ||
    (requestedJob && matches(requestedJob) ? requestedJob : undefined);
  return { exported, job, stale };
}
