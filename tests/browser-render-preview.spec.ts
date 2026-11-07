import { test, expect } from "../frontend/node_modules/@playwright/test";

test("accurate editor preview uses verified output, reuses its revision, and marks later edits stale", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const imported = await request.post("/api/v1/fixtures/spoken_interview.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id;
  const get = async () => (await request.get("/api/v1/projects/" + id)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  const original = await get();
  const timeline = structuredClone(original.timeline);
  timeline.segments[0].start = 1;
  timeline.segments[0].end = 3;
  timeline.segments[0].gain_db = -12;
  timeline.audio.fade = 0.1;
  timeline.output = { width: 180, height: 320, fps: 24, crf: 27 };
  expect(
    (
      await request.put("/api/v1/projects/" + id + "/timeline", {
        data: { revision: original.revision, timeline },
      })
    ).ok(),
  ).toBeTruthy();
  await page.goto("/?project=" + id);
  await page.getByRole("tab", { name: "03 Editor" }).click();
  await page
    .getByRole("button", { name: "Render accurate preview", exact: true })
    .click();
  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "Rendered preview" }),
  ).toBeVisible();
  const current = await get();
  const revision = current.revision;
  const rendered = dialog.getByLabel(
    "Rendered video preview revision " + revision,
    { exact: true },
  );
  await expect(rendered).toBeVisible({ timeout: 60000 });
  await expect
    .poll(() =>
      rendered.evaluate((element) => (element as HTMLVideoElement).readyState),
    )
    .toBeGreaterThanOrEqual(1);
  expect(
    await rendered.evaluate(
      (element) => (element as HTMLVideoElement).duration,
    ),
  ).toBeCloseTo(2, 0);
  const completed = await get();
  const exported = completed.exports.find(
    (item: { revision: number }) => item.revision === revision,
  );
  await expect(rendered).toHaveAttribute("src", exported.video_url);
  await expect(
    dialog.getByRole("link", { name: "Download this revision" }),
  ).toHaveAttribute("href", exported.package_url);
  const renderCount = completed.jobs.filter(
    (item: { kind: string }) => item.kind === "render",
  ).length;
  await dialog.getByRole("button", { name: "Close dialog" }).click();
  await page
    .getByRole("button", { name: "View rendered preview", exact: true })
    .click();
  await expect(rendered).toBeVisible();
  expect(
    (await get()).jobs.filter(
      (item: { kind: string }) => item.kind === "render",
    ),
  ).toHaveLength(renderCount);
  // Simulate a second editor saving a change while this exact-revision player is open.
  const changed = structuredClone(completed.timeline);
  changed.segments[0].gain_db = -6;
  expect(
    (
      await request.put("/api/v1/projects/" + id + "/timeline", {
        data: { revision, timeline: changed },
      })
    ).ok(),
  ).toBeTruthy();
  await expect(
    dialog.getByText(/This render is from an earlier edit/),
  ).toBeVisible();
  await expect(rendered).toHaveAttribute("src", exported.video_url);
  await dialog
    .getByRole("button", { name: "Render latest edit", exact: true })
    .click();
  const latest = dialog.getByLabel(
    "Rendered video preview revision " + (revision + 1),
    { exact: true },
  );
  await expect(latest).toBeVisible({ timeout: 60000 });
  await expect(
    dialog.getByText(/This render is from an earlier edit/),
  ).toHaveCount(0);
  expect(errors).toEqual([]);
});
