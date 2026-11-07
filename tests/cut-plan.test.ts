import assert from "node:assert/strict";
import test from "node:test";
import { planCut } from "../frontend/src/cutPlan.ts";

const c = (id: string, start: number, end: number) => ({ id, start, end });

test("source order sorts picks by time and merges overlapping or touching ranges", () => {
  const plan = planCut(
    [
      c("c", 15, 17),
      c("a", 2, 5),
      c("b", 4.5, 8),
      c("d", 8.04, 9),
      c("e", 10, 11),
    ],
    "source",
  );
  assert.deepEqual(plan.ranges, [
    { start: 2, end: 9, ids: ["a", "b", "d"] },
    { start: 10, end: 11, ids: ["e"] },
    { start: 15, end: 17, ids: ["c"] },
  ]);
  assert.equal(plan.overlaps, 2);
  assert.equal(plan.duration, 7 + 1 + 2);
});

test("source order keeps separate ranges apart when the gap exceeds 50 ms", () => {
  const plan = planCut([c("a", 2, 5), c("b", 5.06, 7)], "source");
  assert.equal(plan.ranges.length, 2);
  assert.equal(plan.overlaps, 0);
  assert.equal(plan.duration.toFixed(2), "4.94");
});

test("selection order keeps the picked sequence and only counts overlaps", () => {
  const plan = planCut(
    [c("c", 15, 17), c("a", 2, 5), c("b", 4.5, 8)],
    "selection",
  );
  assert.deepEqual(
    plan.ranges.map((r) => r.ids),
    [["c"], ["a"], ["b"]],
  );
  assert.equal(plan.overlaps, 1);
  assert.equal(plan.duration, 2 + 3 + 3.5);
});

test("an empty selection plans nothing", () => {
  assert.deepEqual(planCut([], "source"), {
    ranges: [],
    overlaps: 0,
    duration: 0,
  });
});
