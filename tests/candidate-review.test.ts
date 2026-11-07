import assert from "node:assert/strict";
import test from "node:test";
import {
  eligibleCandidatePicks,
  resolveCandidateDecisions,
} from "../frontend/src/candidateReview.ts";
import { planCut } from "../frontend/src/cutPlan.ts";

const candidate = (id: string, decision?: string) => ({
  id,
  start: 2,
  end: 5,
  title: id,
  rationale: "Operator review",
  decision,
});
const feedback = (candidate_id: string, decision: "accept" | "reject") => ({
  candidate_id,
  decision,
});

test("reject then accept makes the same accepted candidate available to the cut", () => {
  const reviewed = resolveCandidateDecisions(
    [candidate("a")],
    [feedback("a", "accept"), feedback("a", "reject")],
  );
  assert.equal(reviewed[0].decision, "accept");
  const plan = planCut(
    reviewed.filter((c) => c.decision !== "reject"),
    "source",
  );
  assert.deepEqual(plan.ranges, [{ start: 2, end: 5, ids: ["a"] }]);
});

test("a newer rejection excludes a previously accepted candidate", () => {
  const reviewed = resolveCandidateDecisions(
    [candidate("a", "accept")],
    [feedback("a", "reject"), feedback("a", "accept")],
  );
  assert.equal(reviewed[0].decision, "reject");
  assert.equal(
    planCut(
      reviewed.filter((c) => c.decision !== "reject"),
      "source",
    ).ranges.length,
    0,
  );
});

test("latest decisions are resolved independently for every candidate", () => {
  const rows = resolveCandidateDecisions(
    [candidate("a"), candidate("b"), candidate("c")],
    [
      feedback("b", "reject"),
      feedback("a", "accept"),
      feedback("a", "reject"),
      feedback("b", "accept"),
      feedback("missing", "reject"),
    ],
  );
  assert.deepEqual(
    rows.map((c) => [c.id, c.decision]),
    [
      ["a", "accept"],
      ["b", "reject"],
      ["c", undefined],
    ],
  );
});

test("stored candidate decisions remain effective when no newer feedback is available", () => {
  const rows = resolveCandidateDecisions(
    [candidate("a", "reject"), candidate("b", "accept")],
    [],
  );
  assert.deepEqual(
    rows.filter((c) => c.decision !== "reject").map((c) => c.id),
    ["b"],
  );
  assert.equal(
    resolveCandidateDecisions(
      [candidate("a", "reject")],
      [feedback("a", "accept")],
    )[0].decision,
    "accept",
  );
});

test("review resolution preserves source order and never changes inputs", () => {
  const candidates = [candidate("b"), candidate("a")];
  const history = [feedback("a", "accept"), feedback("a", "reject")];
  const original = structuredClone({ candidates, history });
  assert.deepEqual(
    resolveCandidateDecisions(candidates, history).map((c) => c.id),
    ["b", "a"],
  );
  assert.deepEqual({ candidates, history }, original);
});

test("external rejection or removal forgets the pick even after later acceptance or reappearance", () => {
  for (const changed of [[candidate("a", "reject")], []]) {
    const picked = eligibleCandidatePicks(["a"], changed);
    assert.deepEqual(picked, []);
    assert.deepEqual(
      eligibleCandidatePicks(picked, [candidate("a", "accept")]),
      [],
    );
  }
  assert.deepEqual(
    eligibleCandidatePicks(["b", "a"], [candidate("a"), candidate("b")]),
    ["b", "a"],
  );
});
