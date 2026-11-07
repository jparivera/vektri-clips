import { test, expect } from "../frontend/node_modules/@playwright/test";
import fs from "node:fs";
import path from "node:path";
const evidence = path.resolve(process.env.VEKTRI_TEST_EVIDENCE_DIR || path.resolve(__dirname, "../frontend/evidence"));
fs.mkdirSync(evidence, { recursive: true });

test("actual transcript correction, reversible pause edits, split preview and assisted tracking review", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Import media", exact: true }).click();
  await page
    .getByRole("button", { name: "Synthetic samples", exact: true })
    .click();
  await page.getByRole("button", { name: /Spoken Interview/ }).click();
  const id = await expect
    .poll(async () => new URL(page.url()).searchParams.get("project"))
    .not.toBeNull()
    .then(() => new URL(page.url()).searchParams.get("project")!);
  const get = async () => (await request.get("/api/v1/projects/" + id)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  const queue = await request.post("/api/v1/projects/" + id + "/transcribe", {
    data: {},
  });
  expect(queue.ok()).toBeTruthy();
  await expect
    .poll(
      async () => {
        const p = await get();
        return p.jobs.find((j: { kind: string }) => j.kind === "transcribe")
          ?.status;
      },
      { timeout: 60000 },
    )
    .toBe("succeeded");
  const original = await get();
  expect(original.transcript.words.length).toBeGreaterThan(10);
  const timeline = structuredClone(original.timeline);
  timeline.segments[0].start = 0;
  timeline.segments[0].end = 22;
  timeline.segments[0].crop.mode = "crop";
  timeline.segments[0].crop.keyframes = [
    { time: 0, x: 0.3, y: 0.5 },
    { time: 24, x: 0.7, y: 0.5 },
  ];
  expect(
    (
      await request.put("/api/v1/projects/" + id + "/timeline", {
        data: { revision: original.revision, timeline },
      })
    ).ok(),
  ).toBeTruthy();
  await page.reload();
  await page.getByRole("tab", { name: "03 Editor" }).click();
  await page.getByRole("button", { name: "Transcript & corrections" }).click();
  await page.getByText(/Transcript segments \(/).click();
  await page
    .getByLabel("Transcript segment 1", { exact: true })
    .fill("A corrected question about preserving the original video.");
  await page
    .getByRole("button", { name: "Save transcript corrections", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect
    .poll(async () => (await get()).transcript.segment_text_corrected)
    .toBe(true);
  const corrected = await get();
  expect(corrected.transcript.words).toEqual(
    original.transcript.words.map(
      ({ start, end, word }: { start: number; end: number; word: string }) => ({
        start,
        end,
        word,
      }),
    ),
  );
  expect(corrected.transcript.segments[0].text).toBe(
    "A corrected question about preserving the original video.",
  );
  await page.getByRole("button", { name: "Audio", exact: true }).click();
  await page
    .getByRole("button", { name: "Suggest pause cuts", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Review pause cuts" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Apply reviewed pause cuts" }).click();
  await expect
    .poll(async () => (await get()).timeline.segments.length)
    .toBeGreaterThan(1);
  const cut = await get();
  for (const segment of cut.timeline.segments)
    expect(segment.crop.keyframes).toEqual(timeline.segments[0].crop.keyframes);
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect(page.locator(".cut-item")).toHaveCount(1);
  await expect.poll(async () => (await get()).timeline.segments.length).toBe(1);
  await page.getByRole("button", { name: "Framing", exact: true }).click();
  await page.getByLabel("Layout", { exact: true }).selectOption("split");
  await expect(page.locator(".video-viewport video")).toHaveCount(2);
  expect(await page.getByRole("slider", { name: "Zoom" }).count()).toBe(0);
  await page.getByRole("button", { name: "Play preview", exact: true }).click();
  await expect
    .poll(() =>
      page
        .locator(".video-viewport video")
        .first()
        .evaluate((v) => (v as HTMLVideoElement).currentTime),
    )
    .toBeGreaterThan(0.4);
  await page
    .getByRole("button", { name: "Pause preview", exact: true })
    .click();
  const dimensions = await page.locator(".video-viewport").evaluateAll((v) =>
    v.map((e) => ({
      height: e.getBoundingClientRect().height,
      top: e.getBoundingClientRect().top,
    })),
  );
  expect(Math.abs(dimensions[0].height - dimensions[1].height)).toBeLessThan(1);
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "editor-tablet.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByLabel("Layout", { exact: true }).selectOption("crop");
  await page
    .getByRole("button", { name: "Assisted tracking", exact: true })
    .click();
  await page.getByLabel("Track method").selectOption("object");
  await page.getByLabel("Region left (0–1)").fill(".1");
  await page.getByLabel("Region top (0–1)").fill(".35");
  await page.getByLabel("Region width (0–1)").fill(".3");
  await page.getByLabel("Region height (0–1)").fill(".5");
  await page.getByRole("button", { name: "Analyze tracking" }).click();
  await expect
    .poll(
      async () =>
        (await get()).jobs.find((j: { kind: string }) => j.kind === "track")
          ?.status,
      { timeout: 60000 },
    )
    .toBe("succeeded");
  await page
    .getByRole("button", { name: "Review tracking result", exact: true })
    .click();
  await expect(
    page
      .locator("p")
      .filter({ hasText: /user-selected normalized template correlation/ }),
  ).toBeVisible();
  const tracked = await get();
  const tracking = tracked.jobs.find(
    (j: { kind: string }) => j.kind === "track",
  ).result.tracking;
  if (tracking.status === "assisted") {
    await page
      .getByRole("button", { name: "Apply reviewed keyframes to this segment" })
      .click();
    await expect
      .poll(
        async () => (await get()).timeline.segments[0].crop.keyframes.length,
      )
      .toBe(tracking.keyframes.length);
  } else {
    await expect(
      page.getByRole("button", {
        name: "Apply reviewed keyframes to this segment",
      }),
    ).toBeDisabled();
    await page.getByRole("button", { name: "Close dialog" }).click();
  }
  await page.setViewportSize({ width: 393, height: 852 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "editor-mobile.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  fs.writeFileSync(
    path.join(evidence, "browser-editor-summary.json"),
    JSON.stringify(
      {
        project_id: id,
        actual_asr_words: original.transcript.words.length,
        segment_correction_kept_word_alignment: true,
        pause_cuts: cut.timeline.segments.length,
        crop_keyframes_preserved: true,
        pause_undo: true,
        split_preview_dimensions: dimensions,
        tracking_status: tracking.status,
        tracking_method: tracking.method,
        tracking_keyframes: tracking.keyframes.length,
        viewports: [
          { width: 1024, height: 768 },
          { width: 393, height: 852 },
        ],
        console_errors: errors,
      },
      null,
      2,
    ),
  );
  expect(errors).toEqual([]);
});
