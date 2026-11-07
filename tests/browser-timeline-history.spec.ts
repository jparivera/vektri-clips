import { test, expect } from "../frontend/node_modules/@playwright/test";

test("drag trimming, drag reordering, keyboard clip navigation, revision restore and the read-only storage page", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  const imported = await request.post("/api/v1/fixtures/spoken_interview.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id;
  const get = async () => (await request.get("/api/v1/projects/" + id)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  const original = await get();
  const timeline = structuredClone(original.timeline);
  const base = timeline.segments[0];
  timeline.segments = [
    { ...structuredClone(base), id: "first", start: 2, end: 6 },
    { ...structuredClone(base), id: "second", start: 8, end: 12 },
  ];
  const saved = await request.put("/api/v1/projects/" + id + "/timeline", {
    data: { revision: original.revision, timeline },
  });
  expect(saved.ok()).toBeTruthy();
  const twoSegmentRevision = (await saved.json()).revision;
  const segments = async () =>
    (await get()).timeline.segments.map(
      (s: { id: string; start: number; end: number }) => [s.id, s.start, s.end],
    );

  await page.goto("/?project=" + id + "&tab=editor");
  const clips = page.locator(".timeline-clip");
  await expect(clips).toHaveCount(2);
  // Raw mouse coordinates only hit elements inside the viewport.
  await page.locator(".timeline-panel").scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);

  // Drag the first clip's right edge left by a quarter of the track: 8 s of
  // output over the track width means about 2 s, so the out point lands near 4.
  const track = await page.locator(".clip-track").boundingBox();
  const handle = await clips.first().locator(".trim-handle.end").boundingBox();
  expect(track && handle).toBeTruthy();
  const hx = handle!.x + handle!.width / 2,
    hy = handle!.y + handle!.height / 2;
  await page.mouse.move(hx, hy);
  await page.mouse.down();
  await page.mouse.move(hx - track!.width * 0.25, hy, { steps: 10 });
  await page.mouse.up();
  await expect
    .poll(async () => (await segments())[0][2], { timeout: 10000 })
    .toBeLessThan(4.6);
  const trimmed = await segments();
  expect(trimmed[0][2]).toBeGreaterThan(3.4);
  expect(trimmed[0][1]).toBe(2);
  expect(trimmed[1]).toEqual(["second", 8, 12]);
  // One drag is one undo step: a single undo restores the 6 s out point.
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await expect.poll(async () => (await segments())[0][2]).toBe(6);

  // Drag the second clip to the front. Clicking Undo scrolled to the toolbar,
  // so bring the timeline back into the viewport before measuring.
  await page.locator(".timeline-panel").scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);
  const secondBox = await clips.nth(1).boundingBox();
  const firstBox = await clips.first().boundingBox();
  await page.mouse.move(
    secondBox!.x + secondBox!.width / 2,
    secondBox!.y + secondBox!.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(firstBox!.x + 8, firstBox!.y + firstBox!.height / 2, {
    steps: 12,
  });
  await page.mouse.up();
  await expect
    .poll(async () => (await segments()).map((s: unknown[]) => s[0]))
    .toEqual(["second", "first"]);
  await expect(clips.first()).toHaveClass(/selected/);

  // Arrow keys move between clips without seeking frames.
  await clips.first().focus();
  await page.keyboard.press("ArrowRight");
  await expect(clips.nth(1)).toHaveClass(/selected/);
  await expect(clips.nth(1)).toBeFocused();
  await expect(page.locator(".cut-item").nth(1)).toHaveClass(/selected/);

  // Revision history lists every autosave and restores an older edit as a new revision.
  await page.getByRole("button", { name: "History", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(
    dialog.getByRole("heading", { name: "Revision history" }),
  ).toBeVisible();
  const rows = dialog.locator(".revision-row");
  await expect.poll(() => rows.count()).toBeGreaterThanOrEqual(3);
  await expect(rows.first()).toHaveClass(/current/);
  const target = dialog.locator(
    `.revision-row[data-revision="${twoSegmentRevision}"]`,
  );
  await expect(target).toContainText("2 segments");
  await target.getByRole("button", { name: "Restore", exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect
    .poll(async () => await segments(), { timeout: 10000 })
    .toEqual([
      ["first", 2, 6],
      ["second", 8, 12],
    ]);
  const restored = await get();
  expect(restored.revision).toBeGreaterThan(twoSegmentRevision + 1);
  expect(
    (await (await request.get("/api/v1/projects/" + id + "/revisions")).json())
      .length,
  ).toBe(restored.revision);

  // Storage page: read-only inventory plus the advisory plan; no write requests.
  const writes: string[] = [];
  page.on("request", (r) => {
    if (r.method() !== "GET" && new URL(r.url()).pathname.startsWith("/api/"))
      writes.push(r.method() + " " + new URL(r.url()).pathname);
  });
  await page.getByRole("button", { name: "Storage", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Storage." })).toBeVisible();
  await expect(page.locator(".storage-table").first()).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Retention plan" }),
  ).toBeVisible();
  await expect
    .poll(() => page.locator(".plan-toolbar .badge").textContent(), {
      timeout: 30000,
    })
    .not.toBe("Planning…");
  await expect(
    page.getByRole("link", { name: "Download plan JSON" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Verify this plan" }).click();
  await expect(page.locator(".verify-result")).toBeVisible({ timeout: 30000 });
  expect(writes).toEqual(["POST /api/v1/maintenance/verify-plan"]);
  const summary = await (
    await request.get("/api/v1/maintenance/storage")
  ).json();
  expect(summary.read_only ?? true).toBeTruthy();
  expect(summary.directories.projects).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
