type BatchResult = {
  items: { project_id: string; job?: { id: string }; error?: string }[];
};

/** Only confirmed jobs from this submission may remove a current selection. */
export async function submitSelectedBatch(
  submittedIds: string[],
  submit: () => Promise<BatchResult>,
  onQueued: (ids: ReadonlySet<string>) => void,
) {
  const submitted = new Set(submittedIds);
  // A rejected HTTP request must leave selection untouched.
  const response = await submit();
  const items = Array.isArray(response.items) ? response.items : [];
  const queuedIds = new Set(
    items
      .filter(
        (item) =>
          submitted.has(item.project_id) &&
          !item.error &&
          typeof item.job?.id === "string" &&
          item.job.id.length > 0,
      )
      .map((item) => item.project_id),
  );
  if (queuedIds.size) onQueued(queuedIds);
  return {
    queuedIds,
    failedIds: [...submitted].filter((id) => !queuedIds.has(id)),
    errors: items
      .filter((item) => submitted.has(item.project_id) && item.error)
      .map((item) => item.error!),
  };
}

export function retainUnqueuedSelection(
  current: ReadonlySet<string>,
  queued: ReadonlySet<string>,
) {
  return new Set([...current].filter((id) => !queued.has(id)));
}
