import { test, expect } from "../frontend/node_modules/@playwright/test";

test("temporal selection and V5 detail OCR run real local jobs and preserve corrected captions", async ({
  page,
  request,
}) => {
  test.setTimeout(300000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const imported = await request.post("/api/v1/fixtures/tutorial_screen.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id;
  const get = async () => (await request.get(`/api/v1/projects/${id}`)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  const prepared = await get();
  // A nonempty operator correction makes accidental caption/ASR mutation visible.
  const corrected = await request.put(`/api/v1/projects/${id}/transcript`, {
    data: {
      revision: prepared.revision,
      words: [
        { start: 0.5, end: 1.0, word: "Keep" },
        { start: 1.1, end: 1.8, word: "this correction." },
      ],
      segments: [{ start: 0.5, end: 1.8, text: "Keep this correction." }],
    },
  });
  expect(corrected.ok()).toBeTruthy();
  const original = await get();
  expect(original.timeline.captions.words).toHaveLength(2);
  expect(original.transcript.corrected).toBe(true);
  await page.goto(`/?project=${id}`);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await page.getByText("Read on-screen text (OCR)", { exact: true }).click();
  await page
    .getByLabel("OCR model", { exact: true })
    .selectOption("ppocr-v5-mobile");
  await page.getByLabel("OCR detail", { exact: true }).selectOption("detail");
  await page
    .getByLabel("Nearby frames to compare for sharpness")
    .selectOption("3");
  await page.getByLabel("OCR sample spacing (seconds)").fill("6");
  const queuedResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === `/api/v1/projects/${id}/ocr` &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Extract on-screen text", exact: true })
    .click();
  const response = await queuedResponse;
  expect(response.status()).toBe(202);
  expect(response.request().postDataJSON()).toEqual({
    ocr_sample_seconds: 6,
    ocr_max_samples: 120,
    ocr_model: "ppocr-v5-mobile",
    ocr_mode: "detail",
    ocr_quality_samples: 3,
  });
  const queued = await response.json();
  const listedJob = async () => {
    const listing = await request.get("/api/v1/jobs");
    expect(listing.ok()).toBeTruthy();
    return (await listing.json()).find(
      (j: { id: string }) => j.id === queued.id,
    );
  };
  await expect
    .poll(async () => (await listedJob())?.status, { timeout: 150000 })
    .toMatch(/^(succeeded|failed|cancelled)$/);
  const listed = await listedJob();
  expect(listed.status, JSON.stringify(listed.error)).toBe("succeeded");
  expect(listed.result).toBeNull();
  expect(listed.result_omitted).toBe(true);
  expect(listed.detail_url).toBe(`/api/v1/jobs/${queued.id}`);
  const detailResponse = await request.get(listed.detail_url);
  expect(detailResponse.ok()).toBeTruthy();
  const detail = await detailResponse.json();
  const text = await get();
  expect(detail.result).toEqual(text.ocr);
  expect(
    await (await request.get(`/api/v1/projects/${id}/ocr`)).json(),
  ).toEqual(text.ocr);
  const embeddedJob = text.jobs.find((j: { id: string }) => j.id === queued.id);
  expect(embeddedJob.result).toBeNull();
  expect(embeddedJob.result_omitted).toBe(true);
  expect(embeddedJob.detail_url).toBe(listed.detail_url);
  expect(text.ocr.source_sha256).toBe(original.source.sha256);
  expect(text.ocr.ocr.provider).toBe("local_onnx_ppocr");
  expect(text.ocr.ocr.ai_used).toBe(true);
  expect(text.ocr.ocr.model_id).toBe("ppocr-v5-mobile");
  expect(text.ocr.ocr.configuration).toEqual({
    ocr_sample_seconds: 6,
    ocr_max_samples: 120,
    ocr_model: "ppocr-v5-mobile",
    ocr_mode: "detail",
    ocr_quality_samples: 3,
  });
  expect(text.ocr.ocr.sample_count).toBeGreaterThan(0);
  expect(text.ocr.ocr.text_count).toBeGreaterThan(0);
  expect(text.ocr.ocr.raw_line_count).toBeGreaterThanOrEqual(
    text.ocr.ocr.text_count,
  );
  for (const sample of text.ocr.ocr.samples) {
    expect(sample.model_id).toBe("ppocr-v5-mobile");
    expect(sample.mode).toBe("detail");
    expect(sample.raw_lines).toBeInstanceOf(Array);
    expect(sample.passes.length).toBeGreaterThan(0);
    expect(sample.quality_selection.requested_candidates).toBe(3);
    const selected = sample.quality_selection.candidates.find(
      (candidate: { frame_ref?: string }) =>
        candidate.frame_ref === sample.frame_ref,
    );
    expect(selected.decoded).toBe(true);
    expect(selected.time).toBe(sample.time);
    expect(sample.quality_selection.selected_frame_ref).toBe(sample.frame_ref);
  }
  expect(text.timeline).toEqual(original.timeline);
  expect(text.transcript).toEqual(original.transcript);
  expect(text.revision).toBe(original.revision);
  expect(text.jobs.some((j: { kind: string }) => j.kind === "transcribe")).toBe(
    false,
  );
  await expect(
    page.getByRole("link", { name: "Download full OCR evidence" }),
  ).toBeVisible();
  await expect(
    page.getByText(
      /Result: ppocr-v5-mobile, detail; up to 3 nearby frame\(s\) compared/,
    ),
  ).toBeVisible();

  await page.getByLabel("Search visible content (experimental)").check();
  await page.getByLabel("Visual search model").selectOption("xclip-b16-k600");
  await page
    .getByLabel("Editorial direction")
    .fill("A computer screen showing a tutorial");
  await page.getByLabel("Target candidates").fill("1");
  await page
    .getByRole("button", { name: "Find candidate moments", exact: true })
    .click();
  await expect
    .poll(
      async () =>
        (await get()).jobs.find((j: { kind: string }) => j.kind === "analyze")
          ?.status,
      { timeout: 90000 },
    )
    .toBe("succeeded");
  const found = await get();
  expect(found.analysis.provider).toBe("local_onnx_xclip");
  expect(found.analysis.candidates[0].method).toBe(
    "local_xclip_temporal_retrieval",
  );
  expect(found.analysis.data_transmitted).toBe(false);
  expect(found.analysis.paid_usage).toBe(0);
  expect(found.timeline).toEqual(original.timeline);
  expect(found.transcript).toEqual(original.transcript);
  expect(found.revision).toBe(original.revision);
  expect(
    found.jobs.some((j: { kind: string }) => j.kind === "transcribe"),
  ).toBe(false);
  await expect(
    page.getByText("Experimental temporal retrieval", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await expect(page.getByLabel("Visual search model")).toHaveValue(
    "xclip-b16-k600",
  );
  await page.getByText("Read on-screen text (OCR)", { exact: true }).click();
  await expect(
    page.getByText(
      /Result: ppocr-v5-mobile, detail; up to 3 nearby frame\(s\) compared/,
    ),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("partial OCR evidence shows saved configuration, actual frame links and escaped bounded readings without inference", async ({
  page,
}) => {
  const id = "ocr-partial-ui-fixture";
  const errors: string[] = [];
  const writes: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const hostileText =
    '<img src=x onerror="window.ocrInjected=true"> & literal OCR evidence';
  const artifact = "artifacts/ocr/partial-fixture";
  const detections = Array.from({ length: 55 }, (_, index) => ({
    text: index === 0 ? hostileText : `Visible reading ${index}`,
    model_score: 0.75,
  }));
  const project = {
    id,
    title: "Partial OCR UI fixture",
    genre: "tutorial",
    status: "ready",
    revision: 2,
    tags: [],
    exports: [],
    feedback: [],
    source: {
      sha256: "a".repeat(64),
      metadata: {
        duration: 5,
        width: 1920,
        height: 1080,
        audio: false,
        video: true,
      },
    },
    timeline: null,
    transcript: null,
    analysis: null,
    jobs: [
      {
        id: "fixture-ocr-job",
        kind: "ocr",
        status: "succeeded",
        stage: "ocr",
        result: null,
        result_omitted: true,
        detail_url: "/api/v1/jobs/fixture-ocr-job",
        attempts: 1,
        max_attempts: 3,
      },
    ],
    ocr: {
      source_sha256: "a".repeat(64),
      artifact_dir: artifact,
      ocr: {
        model_id: "ppocr-v5-mobile",
        configuration: {
          ocr_model: "ppocr-v5-mobile",
          ocr_mode: "detail",
          ocr_quality_samples: 3,
        },
        sampling_complete: false,
        raw_readings_truncated: true,
        sample_count: 1,
        text_count: 55,
        maximum_sample_gap_seconds: 0,
        limitations: [
          "Fixture: the remaining samples were omitted at an evidence limit.",
        ],
        samples: [
          {
            time: 1.75,
            requested_time: 2,
            frame_ref: "ocr-frames/selected-neighbor.jpg",
            detections,
          },
        ],
      },
    },
  };
  await page.route("**/api/v1/**", async (route) => {
    const incoming = route.request();
    const path = new URL(incoming.url()).pathname;
    if (incoming.method() !== "GET") {
      writes.push(`${incoming.method()} ${path}`);
      await route.fulfill({
        status: 405,
        json: { detail: "Read-only browser fixture" },
      });
      return;
    }
    const body =
      path === `/api/v1/projects/${id}`
        ? project
        : path === "/api/v1/projects"
          ? [project]
          : path === "/api/v1/jobs"
            ? project.jobs
            : [];
    await route.fulfill({ json: body });
  });
  await page.goto(`/?project=${id}`);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await page.getByText("Read on-screen text (OCR)", { exact: true }).click();
  const panel = page
    .locator("details")
    .filter({
      has: page.locator("summary", { hasText: "Read on-screen text (OCR)" }),
    });
  await expect(panel.getByRole("status")).toHaveText(
    "Partial evidence: sampling or text-region limits were reached. Review the full report for omitted work.",
  );
  await expect(
    panel.getByText(
      /Result: ppocr-v5-mobile, detail; up to 3 nearby frame\(s\) compared/,
    ),
  ).toBeVisible();
  await expect(
    panel.locator("li").filter({ hasText: hostileText }),
  ).toBeVisible();
  await expect(panel.locator("li")).toHaveCount(50);
  await expect(
    panel.locator("li").filter({ hasText: "Visible reading 49" }),
  ).toBeVisible();
  await expect(
    panel.locator("li").filter({ hasText: "Visible reading 50" }),
  ).toHaveCount(0);
  await expect(
    panel.getByText(
      "Showing the first 50 regions; download the evidence for all results.",
    ),
  ).toBeVisible();
  await expect(panel.locator("li img, li script")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => (window as Window & { ocrInjected?: boolean }).ocrInjected,
    ),
  ).toBeUndefined();
  const firstFrame = panel.locator("li").first().getByRole("link");
  await expect(firstFrame).toHaveText("00:01.7");
  await expect(firstFrame).toHaveAttribute(
    "href",
    `/api/v1/artifacts/${id}/${artifact}/ocr-frames/selected-neighbor.jpg`,
  );
  await expect(
    panel.getByRole("link", { name: "Download full OCR evidence" }),
  ).toHaveAttribute("href", `/api/v1/artifacts/${id}/${artifact}/ocr.json`);
  // Editing controls cannot relabel the already completed result.
  await panel.getByLabel("OCR model", { exact: true }).selectOption("ppocr-v4");
  await panel
    .getByLabel("OCR detail", { exact: true })
    .selectOption("baseline");
  await panel
    .getByLabel("Nearby frames to compare for sharpness")
    .selectOption("1");
  await expect(
    panel.getByText(
      /Result: ppocr-v5-mobile, detail; up to 3 nearby frame\(s\) compared/,
    ),
  ).toBeVisible();
  expect(writes).toEqual([]);
  expect(errors).toEqual([]);
});
