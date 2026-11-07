import { test, expect } from "../frontend/node_modules/@playwright/test";

function fixture() {
  const unknown = {
    status: "unknown",
    summary: "Editorial completeness is not established.",
    evidence: [],
  };
  return {
    id: "grounded",
    title: "Grounded evidence fixture",
    genre: "mixed",
    status: "ready",
    revision: 1,
    tags: [],
    jobs: [],
    exports: [],
    feedback: [],
    artifact_base: "/review-artifacts",
    source: {
      sha256: "a".repeat(64),
      preview_url: "/test-preview.mp4",
      metadata: {
        duration: 20,
        width: 640,
        height: 360,
        video: true,
        audio: true,
        audio_stream_count: 2,
      },
    },
    timeline: {
      schema_version: 1,
      segments: [],
      output: { width: 720, height: 1280, fps: 24, crf: 22 },
      captions: { enabled: false, words: [] },
      audio: { normalize: false, fade: 0, stream_index: 0 },
    },
    analysis: {
      artifact_dir: "analysis-fixture",
      configuration: { audio_stream: 1 },
      candidates: [
        {
          id: "candidate",
          start: 4,
          end: 6,
          title: "Measured burst",
          rationale:
            "Audio signal around source 5 seconds; inspect the footage.",
          method: "heuristic",
          review: {
            schema_version: 1,
            method: "deterministic_evidence_review",
            status: "needs_human_review",
            fingerprint: "b".repeat(64),
            source_sha256: "a".repeat(64),
            playback: { start: 4, end: 6, context_start: 2, context_end: 8 },
            observations: {
              hook: {
                status: "observed",
                summary: "Timed passage is present.",
                evidence: [
                  {
                    kind: "transcript",
                    start: 4,
                    end: 6,
                    quote: "Listen to this sound.",
                    excerpt: true,
                  },
                ],
              },
              context: unknown,
              payoff: unknown,
            },
            frames: [
              {
                role: "before",
                time: 2,
                approximate_seek: true,
                status: "available",
                path: "before.jpg",
              },
              {
                role: "at",
                time: 5,
                approximate_seek: true,
                status: "unavailable",
                path: null,
              },
              {
                role: "after",
                time: 8,
                approximate_seek: true,
                status: "unavailable",
                path: null,
              },
            ],
            warnings: ["Signal does not establish a collision."],
            limitations: ["Frames are sampled evidence only."],
          },
        },
      ],
    },
  };
}

test("grounded review is explicit, bounded, repeatable and never auto-accepts", async ({
  page,
}, testInfo) => {
  const project = fixture();
  const writes: unknown[] = [];
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  // Media state is simulated here. Actual decoding and artifact extraction are backend fixture tests.
  await page.addInitScript(() => {
    const positions = new WeakMap<HTMLMediaElement, number>();
    const playing = new WeakSet<HTMLMediaElement>();
    Object.defineProperty(HTMLMediaElement.prototype, "readyState", {
      get: () => 1,
    });
    Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
      get() {
        return positions.get(this) ?? 0;
      },
      set(v) {
        positions.set(this, v);
      },
    });
    Object.defineProperty(HTMLMediaElement.prototype, "paused", {
      get() {
        return !playing.has(this);
      },
    });
    HTMLMediaElement.prototype.pause = function () {
      playing.delete(this);
    };
    HTMLMediaElement.prototype.play = function () {
      playing.add(this);
      this.dispatchEvent(new Event("play"));
      return Promise.resolve();
    };
    HTMLMediaElement.prototype.load = function () {};
  });
  await page.route("**/*", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/")) {
      if (request.method() !== "GET") {
        writes.push(request.postDataJSON());
        return route.fulfill({ json: { ok: true } });
      }
      const data =
        path === "/api/v1/projects"
          ? [project]
          : path === "/api/v1/projects/grounded"
            ? project
            : path.endsWith("/health")
              ? { status: "ok", worker_active: false }
              : [];
      return route.fulfill({ json: data });
    }
    if (path === "/test-preview.mp4")
      return route.fulfill({ contentType: "video/mp4", body: "" });
    if (path.startsWith("/review-artifacts/"))
      return route.fulfill({ status: 404, body: "Missing fixture frame" });
    if (
      request.method() === "GET" &&
      new URL(request.url()).origin === "http://127.0.0.1:8741"
    )
      return route.continue();
    return route.abort();
  });
  await page.goto("/?project=grounded&tab=analysis");
  await page.getByRole("button", { name: "Context", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("Needs human review");
  await page.screenshot({
    path: testInfo.outputPath("grounded-review-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 393, height: 852 });
  expect(await dialog.evaluate((e) => e.scrollWidth <= e.clientWidth)).toBe(
    true,
  );
  await page.screenshot({
    path: testInfo.outputPath("grounded-review-mobile.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(dialog).toContainText(
    "Analysis used track 2; these tracks differ",
  );
  await expect(dialog).toContainText("Listen to this sound.");
  await expect(dialog).toContainText("Excerpt, truncated");
  await expect(dialog.getByLabel("payoff evidence")).toContainText("Unknown");
  await expect(dialog).toContainText("Still frames cannot prove motion");
  await expect(dialog.locator(".review-frame-missing")).toHaveCount(3);
  // Empty media fixture may fire an error; restore the media path in a new open with errors suppressed.
  await page.getByRole("button", { name: "Close dialog" }).click();
  await page.evaluate(() =>
    document.addEventListener(
      "error",
      (e) => {
        if (e.target instanceof HTMLVideoElement) e.stopImmediatePropagation();
      },
      true,
    ),
  );
  await page.getByRole("button", { name: "Context", exact: true }).click();
  const player = dialog.locator("video");
  await expect(player).toHaveCount(1);
  expect(await player.evaluate((e: HTMLVideoElement) => e.paused)).toBe(true);
  expect(await player.getAttribute("autoplay")).toBeNull();
  await player.evaluate((e: HTMLVideoElement) => e.play());
  await page.getByRole("button", { name: "Surrounding context" }).click();
  expect(
    await player.evaluate((e: HTMLVideoElement) => [e.paused, e.currentTime]),
  ).toEqual([true, 2]);
  await page.getByRole("button", { name: "Candidate interval" }).click();
  expect(
    await player.evaluate((e: HTMLVideoElement) => [e.paused, e.currentTime]),
  ).toEqual([true, 4]);
  await player.evaluate(async (e: HTMLVideoElement) => {
    await e.play();
    e.currentTime = 7;
    e.dispatchEvent(new Event("timeupdate"));
  });
  expect(
    await player.evaluate((e: HTMLVideoElement) => [e.paused, e.currentTime]),
  ).toEqual([true, 6]);
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await page.getByRole("button", { name: "Context", exact: true }).click();
  expect(
    await dialog.locator("video").evaluate((e: HTMLVideoElement) => e.paused),
  ).toBe(true);
  await page.keyboard.press("Escape");
  // Rounded selector bounds may exceed the source by <1 ms. Playback must
  // obey the review's actual source end, without altering the candidate cut.
  project.source.metadata.duration = 6.00075;
  project.analysis.candidates[0].end = 6.001;
  project.analysis.candidates[0].review.playback.end = 6.00075;
  project.analysis.candidates[0].review.playback.context_end = 6.00075;
  await page.evaluate(() =>
    document.dispatchEvent(new Event("visibilitychange")),
  );
  await expect(page.locator(".candidate-card")).toContainText("Measured burst");
  await page.getByRole("button", { name: "Context", exact: true }).click();
  await dialog.locator("video").evaluate(async (e: HTMLVideoElement) => {
    await e.play();
    e.currentTime = 6.001;
    e.dispatchEvent(new Event("timeupdate"));
  });
  expect(
    await dialog.locator("video").evaluate((e: HTMLVideoElement) => [e.paused, e.currentTime]),
  ).toEqual([true, 6.00075]);
  await page.keyboard.press("Escape");
  // A newer candidate invalidates the open evidence immediately on refresh.
  await page.getByRole("button", { name: "Context", exact: true }).click();
  project.analysis.candidates[0].start = 4.5;
  await page.evaluate(() =>
    document.dispatchEvent(new Event("visibilitychange")),
  );
  await expect(dialog).toHaveCount(0);
  // Legacy and audio-only inputs retain honest unknowns and no visual claims.
  delete (project.analysis.candidates[0] as { review?: unknown }).review;
  project.source.metadata.video = false;
  await page.evaluate(() =>
    document.dispatchEvent(new Event("visibilitychange")),
  );
  await expect(page.locator(".candidate-card")).toContainText("Measured burst");
  await page.getByRole("button", { name: "Context", exact: true }).click();
  await expect(dialog).toContainText("Legacy candidate");
  await expect(dialog).toContainText("Audio-only source");
  await expect(dialog.getByLabel("hook evidence")).toContainText("Unknown");
  await page.keyboard.press("Escape");
  expect(writes).toEqual([]);
  await page.getByLabel("Impact sounds (optional)").check();
  await expect(
    page.getByLabel("Search visible content (experimental)"),
  ).not.toBeChecked();
  await page.getByText("Impact sound settings", { exact: true }).click();
  await expect(page.getByLabel("Impact threshold")).toHaveValue("0.04");
  await page.getByLabel("Search visible content (experimental)").check();
  await expect(page.getByLabel("Impact sounds (optional)")).not.toBeChecked();
  await page.getByLabel("Impact sounds (optional)").check();
  await page.getByRole("button", { name: "Find candidate moments" }).click();
  expect(writes).toHaveLength(1);
  expect(writes[0]).toMatchObject({
    impact_detection: true,
    visual_search: false,
    impact_lead: 0.4,
    impact_follow: 0.5,
    impact_threshold: 0.04,
  });
  expect(errors).toEqual([]);
});
