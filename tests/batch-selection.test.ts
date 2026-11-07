import assert from "node:assert/strict";
import test from "node:test";
import {
  submitSelectedBatch,
  retainUnqueuedSelection,
} from "../frontend/src/batchSelection.ts";

test("HTTP failure preserves selection without claiming any queued item", async () => {
  let selected = new Set(["a", "b"]);
  await assert.rejects(
    submitSelectedBatch(
      ["a", "b"],
      async () => {
        throw new Error("HTTP 409");
      },
      (queued) => {
        selected = retainUnqueuedSelection(selected, queued);
      },
    ),
    /HTTP 409/,
  );
  assert.deepEqual([...selected], ["a", "b"]);
});

test("partial success retains failed and missing projects", async () => {
  let selected = new Set(["a", "b", "missing"]);
  const result = await submitSelectedBatch(
    [...selected],
    async () => ({
      items: [
        { project_id: "a", job: { id: "job-a" } },
        { project_id: "b", error: "Project not ready" },
      ],
    }),
    (queued) => {
      selected = retainUnqueuedSelection(selected, queued);
    },
  );
  assert.deepEqual([...selected], ["b", "missing"]);
  assert.deepEqual([...result.queuedIds], ["a"]);
  assert.deepEqual(result.failedIds, ["b", "missing"]);
  assert.deepEqual(result.errors, ["Project not ready"]);
});

test("selection changes during an outstanding submission are preserved", async () => {
  let selected = new Set(["a", "b"]);
  let resolve!: (value: {
    items: { project_id: string; job: { id: string } }[];
  }) => void;
  const response = new Promise<{
    items: { project_id: string; job: { id: string } }[];
  }>((yes) => {
    resolve = yes;
  });
  const pending = submitSelectedBatch(
    [...selected],
    () => response,
    (queued) => {
      selected = retainUnqueuedSelection(selected, queued);
    },
  );
  selected = new Set(["a", "new-selection"]); // b was deselected while request ran
  resolve({
    items: [
      { project_id: "a", job: { id: "job-a" } },
      { project_id: "b", job: { id: "job-b" } },
    ],
  });
  await pending;
  assert.deepEqual([...selected], ["new-selection"]);
});

test("unsubmitted IDs and items without confirmed job IDs never clear selection", async () => {
  let selected = new Set(["a", "b", "not-submitted"]);
  const result = await submitSelectedBatch(
    ["a", "b"],
    async () => ({
      items: [
        { project_id: "a", job: { id: "" } },
        {
          project_id: "b",
          job: { id: "job-b" },
          error: "Conflicting response",
        },
        { project_id: "not-submitted", job: { id: "unexpected" } },
      ],
    }),
    (queued) => {
      selected = retainUnqueuedSelection(selected, queued);
    },
  );
  assert.deepEqual([...selected], ["a", "b", "not-submitted"]);
  assert.equal(result.queuedIds.size, 0);
});

test("all confirmed submitted jobs clear selection exactly once", async () => {
  let selected = new Set(["a", "b"]),
    applications = 0;
  const result = await submitSelectedBatch(
    [...selected],
    async () => ({
      items: [
        { project_id: "a", job: { id: "job-a" } },
        { project_id: "b", job: { id: "job-b" } },
      ],
    }),
    (queued) => {
      applications++;
      selected = retainUnqueuedSelection(selected, queued);
    },
  );
  assert.equal(selected.size, 0);
  assert.equal(applications, 1);
  assert.deepEqual(result.failedIds, []);
});
