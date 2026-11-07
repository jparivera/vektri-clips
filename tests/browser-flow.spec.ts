import { test, expect } from "../frontend/node_modules/@playwright/test";
import path from "node:path";
import fs from "node:fs";

const root = path.resolve(__dirname, "..");
const evidence = path.resolve(process.env.VEKTRI_TEST_EVIDENCE_DIR || path.join(root, "frontend/evidence"));
fs.mkdirSync(evidence, { recursive: true });
test("real local source → persisted edit → render → playback and manual handoff", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Your library." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Import media", exact: true }).click();
  await page
    .locator("input[type=file]")
    .setInputFiles(path.join(root, "fixtures/spoken_interview.mp4"));
  const title = "QA · Browser edit " + Date.now();
  await page.getByLabel("Project title").fill(title);
  await page
    .getByLabel("Source attribution")
    .fill("Original synthetic Flite test fixture, browser regression.");
  await page.getByLabel("Footage type").selectOption("speech");
  await page.getByRole("button", { name: "Import into workspace" }).click();
  await expect(
    page.getByRole("heading", { name: title, level: 1 }),
  ).toBeVisible();
  const id = new URL(page.url()).searchParams.get("project")!;
  await expect
    .poll(async () => {
      const p = await (await request.get("/api/v1/projects/" + id)).json();
      return p.status;
    })
    .toBe("ready");
  await page.getByRole("tab", { name: "03 Editor" }).click();
  await page.getByLabel("In, seconds", { exact: true }).fill("2");
  await page.getByLabel("Out, seconds", { exact: true }).fill("8");
  await page.getByRole("button", { name: "9:16", exact: true }).click();
  await page.getByLabel("Layout", { exact: true }).selectOption("crop");
  await page.getByRole("slider", { name: "Horizontal position" }).focus();
  await page.keyboard.press("Home");
  for (let i = 0; i < 32; i++) await page.keyboard.press("ArrowRight");
  await expect
    .poll(async () => {
      const p = await (await request.get("/api/v1/projects/" + id)).json();
      return [
        p.timeline.segments[0].start,
        p.timeline.segments[0].end,
        p.timeline.output.width,
      ];
    })
    .toEqual([2, 8, 360]);
  await page.getByRole("button", { name: "Play preview", exact: true }).click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .first()
        .evaluate((v) => (v as HTMLVideoElement).currentTime),
    )
    .toBeGreaterThan(2.1);
  await page
    .getByRole("button", { name: "Pause preview", exact: true })
    .click();
  await page
    .locator("video")
    .first()
    .evaluate((v) => {
      (v as HTMLVideoElement).currentTime = 5;
    });
  await expect
    .poll(() => page.locator(".clocks").textContent())
    .toContain("00:05.0");
  await page.getByRole("button", { name: "Split S", exact: true }).click();
  await expect(page.locator(".cut-item")).toHaveCount(2);
  await page.locator(".cut-item").nth(1).click();
  await page.getByRole("button", { name: "Move segment earlier" }).click();
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await page.getByRole("button", { name: "Redo", exact: true }).click();
  await expect
    .poll(async () => {
      const p = await (await request.get("/api/v1/projects/" + id)).json();
      return p.timeline.segments.map((s: { start: number; end: number }) => [
        s.start,
        s.end,
      ]);
    })
    .toEqual([
      [5, 8],
      [2, 5],
    ]);
  await page.reload();
  await page.getByRole("tab", { name: "03 Editor" }).click();
  await expect(page.locator(".cut-item")).toHaveCount(2);
  await expect(page.getByLabel("In, seconds", { exact: true })).toHaveValue(
    "5",
  );
  await page.getByRole("button", { name: "Captions", exact: true }).click();
  await page
    .getByRole("button", { name: "Edit words & source timings" })
    .click();
  await page.getByLabel("New caption word", { exact: true }).fill("Synthetic");
  await page.getByLabel("New word start", { exact: true }).fill("5");
  await page.getByLabel("New word end", { exact: true }).fill("5.8");
  await page.getByRole("button", { name: "Add caption word" }).click();
  await page.getByLabel("New caption word", { exact: true }).fill("sample.");
  await page.getByLabel("New word start", { exact: true }).fill("5.8");
  await page.getByLabel("New word end", { exact: true }).fill("6.5");
  await page.getByRole("button", { name: "Add caption word" }).click();
  await page
    .getByRole("button", { name: "Apply caption words to edit" })
    .click();
  await page.getByLabel("Burn captions into video").check();
  await page.getByLabel("Highlight timestamped words").check();
  await page.getByLabel("Caption font").selectOption("DejaVu Serif");
  await page.getByLabel("Text color", { exact: true }).fill("#fff8e8");
  await page.getByLabel("Highlight color", { exact: true }).fill("#d3ec78");
  await page.getByRole("button", { name: "Audio", exact: true }).click();
  await page.getByLabel("Normalize output loudness").check();
  await page.getByRole("button", { name: "Framing", exact: true }).click();
  await page.getByRole("button", { name: "Presets", exact: true }).click();
  await page
    .getByLabel("Save current settings as")
    .fill("Browser portrait test");
  await page.getByRole("button", { name: "Save preset", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Browser portrait test" }).last(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.getByRole("button", { name: "Play preview", exact: true }).click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .first()
        .evaluate((v) => (v as HTMLVideoElement).currentTime),
    )
    .toBeGreaterThan(5.2);
  await page
    .getByRole("button", { name: "Pause preview", exact: true })
    .click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(page.getByText("Saved", { exact: true })).toBeVisible();
  await page.screenshot({
    path: path.join(evidence, "editor-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Export edit", exact: true }).click();
  await expect(page.getByText("Make it a real file.")).toBeVisible();
  await expect
    .poll(
      async () => {
        const p = await (await request.get("/api/v1/projects/" + id)).json();
        const failed = p.jobs.find(
          (j: { kind: string; status: string }) =>
            j.kind === "render" && j.status === "failed",
        );
        if (failed) throw Error(JSON.stringify(failed));
        return p.exports.length;
      },
      { timeout: 90000 },
    )
    .toBeGreaterThan(0);
  await expect(
    page.getByRole("link", { name: "Download package" }),
  ).toBeVisible();
  const data = await (await request.get("/api/v1/projects/" + id)).json();
  const exported = data.exports[0];
  expect(exported.manifest.width).toBe(360);
  expect(exported.manifest.height).toBe(640);
  expect(Math.abs(exported.manifest.duration - 6)).toBeLessThan(0.15);
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download package" }).click();
  const downloaded = await download;
  await downloaded.saveAs(path.join(evidence, "browser-export.zip"));
  expect(
    fs.statSync(path.join(evidence, "browser-export.zip")).size,
  ).toBeGreaterThan(5000);
  await page
    .locator("video")
    .first()
    .evaluate((v) => {
      (v as HTMLVideoElement).muted = true;
      return (v as HTMLVideoElement).play();
    });
  await expect
    .poll(() =>
      page
        .locator("video")
        .first()
        .evaluate((v) => (v as HTMLVideoElement).currentTime),
    )
    .toBeGreaterThan(0.2);
  await page
    .locator("video")
    .first()
    .evaluate((v) => (v as HTMLVideoElement).pause());
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "export-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Prepare handoff" }).click();
  await page.getByLabel("Intended account").fill("Manual test destination");
  await page
    .getByLabel("Caption", { exact: true })
    .fill("Synthetic test clip. Manual review only.");
  await page.getByRole("button", { name: "Save handoff draft" }).click();
  await expect(
    page.getByText("Manual handoff draft saved. Nothing was published."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Publishing", exact: true }).click();
  const handoff = page
    .locator(".publication-card")
    .filter({ hasText: exported.id });
  await expect(handoff).toContainText("Manual test destination");
  await handoff.getByRole("button", { name: "Review exact handoff" }).click();
  await page
    .getByLabel("Rights review acknowledgment")
    .fill(
      "Synthetic test fixture generated for this local review; no remote posting requested.",
    );
  await page
    .getByLabel(
      "I confirm this exact asset, destination, account, caption, visibility, schedule and time zone for manual handoff.",
    )
    .check();
  await page
    .getByRole("button", { name: "Approve manual handoff only" })
    .click();
  await expect(handoff).toContainText("ready-for-manual-upload");
  await page.goto("/?project=" + id);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await page.getByRole("button", { name: "Mark a moment manually" }).click();
  await page.getByLabel("Factual title").fill("Known source event");
  await page.getByRole("button", { name: "Add manual candidate" }).click();
  await expect(
    page.getByRole("heading", { name: "Known source event" }),
  ).toBeVisible();
  await page.getByLabel("Analysis mode").selectOption("local");
  await page
    .getByLabel("Editorial direction")
    .fill("Why keep the original video?");
  await page.getByRole("button", { name: "Find candidate moments" }).click();
  await expect
    .poll(
      async () => {
        const p = await (await request.get("/api/v1/projects/" + id)).json();
        const job = p.jobs.find((j: { kind: string }) => j.kind === "analyze");
        return job?.status;
      },
      { timeout: 90000 },
    )
    .toBe("succeeded");
  await expect(page.locator(".candidate-card").first()).toHaveAttribute(
    "data-method",
    "local_ai_semantic",
  );
  await expect(
    page.getByRole("button", { name: "Find candidate moments" }),
  ).toBeEnabled();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "candidates-desktop.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Context", exact: true })
    .first()
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator(".context-video")
        .evaluate((v) => (v as HTMLVideoElement).readyState),
    )
    .toBeGreaterThanOrEqual(2);
  await expect
    .poll(() =>
      page
        .locator(".context-video")
        .evaluate((v) => (v as HTMLVideoElement).currentTime),
    )
    .toBeGreaterThan(0.2);
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.setViewportSize({ width: 393, height: 852 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "review-mobile.png"),
    fullPage: false,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Back to library" }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: path.join(evidence, "library-mobile.png"),
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "AI & models", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "AI & models." }),
  ).toBeVisible();
  await expect(page.getByText("tiny.en", { exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/?project=" + id);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await page
    .locator(".candidate-card")
    .last()
    .getByRole("button", { name: "Reject", exact: true })
    .click();
  await expect(
    page
      .locator(".candidate-card")
      .last()
      .getByRole("button", { name: "Rejected", exact: true }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await expect(
    page
      .locator(".candidate-card")
      .last()
      .getByRole("button", { name: "Rejected", exact: true }),
  ).toBeVisible();
  const proposed = await (await request.get("/api/v1/projects/" + id)).json();
  const firstCandidate = proposed.analysis.candidates[0];
  await page
    .locator(".candidate-card")
    .first()
    .getByRole("button", { name: "Edit this clip" })
    .click();
  await expect(page.getByLabel("In, seconds", { exact: true })).toHaveValue(
    String(firstCandidate.start),
  );
  await expect
    .poll(async () => {
      const p = await (await request.get("/api/v1/projects/" + id)).json();
      return p.timeline.segments.map((s: { start: number; end: number }) => [
        s.start,
        s.end,
      ]);
    })
    .toEqual([[firstCandidate.start, firstCandidate.end]]);
  fs.writeFileSync(
    path.join(evidence, "browser-flow-summary.json"),
    JSON.stringify(
      {
        project_id: id,
        export_id: exported.id,
        revision: exported.revision,
        manifest: exported.manifest,
        console_errors: errors,
        checks: [
          "upload",
          "source preparation",
          "manual trim",
          "crop",
          "split",
          "reorder",
          "undo redo",
          "autosave reload readback",
          "manual caption timings",
          "burned captions",
          "audio normalization",
          "preset save",
          "real render",
          "ZIP download",
          "browser output playback",
          "manual publication draft",
          "explicit manual handoff approval",
          "caption font and colors",
          "manual event",
          "local AI proposals",
          "context playback",
          "393px overflow",
          "reduced motion",
          "model status",
          "rejection reload readback",
          "candidate acceptance persisted edit",
        ],
      },
      null,
      2,
    ),
  );
  expect(errors).toEqual([]);
});
