import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";
import {
  reviewPlaybackRange,
  reviewExcerptLabel,
} from "../frontend/src/groundedReviewHelpers.ts";

test("frontend module stems stay distinct on case-insensitive filesystems", () => {
  const modules = readdirSync(new URL("../frontend/src/", import.meta.url))
    .filter((name) => /\.tsx?$/.test(name));
  const seen = new Map<string, string>();
  for (const name of modules) {
    const stem = name.replace(/\.tsx?$/, "").toLowerCase();
    assert.equal(seen.has(stem), false,
      `${name} conflicts with ${seen.get(stem)} during case-insensitive module resolution`);
    seen.set(stem, name);
  }
});

const candidate = Object.freeze({ start: 4, end: 6 });
const review = Object.freeze({
  playback: Object.freeze({
    start: 4.001,
    end: 5.999,
    context_start: 2,
    context_end: 8,
  }),
});

test("candidate playback uses precise reviewed source times, not rounded display boundaries", () => {
  assert.deepEqual(reviewPlaybackRange(candidate, 20, false, review), {
    start: 4.001,
    end: 5.999,
  });
});

test("context playback uses the separate reviewed context interval", () => {
  assert.deepEqual(reviewPlaybackRange(candidate, 20, true, review), {
    start: 2,
    end: 8,
  });
});

test("candidate and context playback stay inside source boundaries", () => {
  const outside = {
    playback: { start: -1, end: 21, context_start: -3, context_end: 24 },
  };
  for (const context of [false, true]) {
    assert.deepEqual(reviewPlaybackRange(candidate, 20, context, outside), {
      start: 0,
      end: 20,
    });
  }
});

test("legacy candidate rounded past the source end clamps without adding duration", () => {
  assert.deepEqual(
    reviewPlaybackRange({ start: 1, end: 5.001 }, 5.0005, false),
    {
      start: 1,
      end: 5.0005,
    },
  );
});

test("legacy context adds two seconds only where source footage exists", () => {
  assert.deepEqual(reviewPlaybackRange({ start: 1, end: 5 }, 6, true), {
    start: 0,
    end: 6,
  });
  assert.deepEqual(reviewPlaybackRange(candidate, 20, true), {
    start: 2,
    end: 8,
  });
});

test("absent usable review after stale-source filtering uses candidate fallback", () => {
  const usableReview = undefined;
  assert.deepEqual(reviewPlaybackRange(candidate, 20, false, usableReview), {
    start: 4,
    end: 6,
  });
  assert.deepEqual(reviewPlaybackRange(candidate, 7, true, usableReview), {
    start: 2,
    end: 7,
  });
});

test("out-of-source and inverted ranges never return a reversed interval", () => {
  assert.deepEqual(reviewPlaybackRange({ start: 22, end: 24 }, 20, false), {
    start: 20,
    end: 20,
  });
  assert.deepEqual(reviewPlaybackRange({ start: 6, end: 4 }, 20, false), {
    start: 6,
    end: 6,
  });
});

test("only the explicit boolean excerpt marker adds a truncation label", () => {
  assert.equal(reviewExcerptLabel(true), "Excerpt, truncated");
  for (const value of [
    false,
    "true",
    "quoted excerpt text",
    "",
    undefined,
    null,
    1,
    0,
    {},
  ]) {
    assert.equal(reviewExcerptLabel(value), null);
  }
});

test("range and label helpers leave source evidence and exact quote text unchanged", () => {
  const item = Object.freeze({
    quote: "  “Listen… now.”\nNext line.  ",
    excerpt: true,
  });
  const original = structuredClone({ candidate, review, item });
  reviewPlaybackRange(candidate, 20, false, review);
  reviewPlaybackRange(candidate, 20, true, review);
  reviewExcerptLabel(item.excerpt);
  assert.deepEqual({ candidate, review, item }, original);
});
