import { test, expect } from "../frontend/node_modules/@playwright/test";

type Range = { start: number; end: number };
type FeedbackRow = { candidate_id: string; decision: string };

test("multi-candidate cut building, rejected candidates excluded, audit trail filters and the appearance switch", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  const writes: string[] = [];
  page.on("request", (r) => {
    if (r.method() !== "GET" && r.url().includes("/api/"))
      writes.push(r.method() + " " + new URL(r.url()).pathname);
  });

  const imported = await request.post("/api/v1/fixtures/spoken_interview.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id as string;
  const get = async () => (await request.get("/api/v1/projects/" + id)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  const ranges = async () =>
    (await get()).timeline.segments.map((s: Range) => [s.start, s.end]);

  // Operator-marked events give deterministic candidates without running an analysis.
  const mark = async (
    time: number,
    before: number,
    after: number,
    title: string,
  ) => {
    const r = await request.post(`/api/v1/projects/${id}/event`, {
      data: { time, before, after, title },
    });
    expect(r.ok()).toBeTruthy();
    return (await r.json()) as { id: string } & Range;
  };
  const first = await mark(3, 1, 2, "Opening line"); // 2–5 s
  const second = await mark(10, 2, 2, "Middle answer"); // 8–12 s
  const third = await mark(16, 1, 1, "Closing remark"); // 15–17 s
  const rejected = await request.post(`/api/v1/projects/${id}/feedback`, {
    data: { candidate_id: second.id, decision: "reject", note: "QA rejection" },
  });
  expect(rejected.ok()).toBeTruthy();

  await page.goto(`/?project=${id}&tab=analysis`);
  const cards = page.locator(".candidate-card");
  await expect(cards).toHaveCount(3);
  const builder = page.getByRole("region", { name: "Cut builder" });
  await expect(builder).toContainText("Build one cut from several candidates");
  await expect(
    page.getByLabel("Include candidate 2 in the cut"),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Build cut", exact: true }),
  ).toBeDisabled();

  // Pick the last candidate first, then the first one.
  await page.getByLabel("Include candidate 3 in the cut").check();
  await page.getByLabel("Include candidate 1 in the cut").check();
  await expect(builder).toContainText("2 selected · 2 clips · 5.0 s");
  await expect(cards.nth(0)).toHaveClass(/picked/);
  await expect(cards.nth(1)).not.toHaveClass(/picked/);
  await expect(cards.nth(2)).toHaveClass(/picked/);
  await page.getByLabel("Clip order").selectOption("selection");
  await expect(builder).toContainText("Clips keep the order you picked");
  await page.getByLabel("Clip order").selectOption("source");
  await expect(builder).toContainText("Clips follow source time");
  await page.getByRole("button", { name: "Build cut from 2" }).click();
  await expect(page).toHaveURL(/tab=editor/);
  await expect(page.locator(".timeline-clip")).toHaveCount(2);
  const built = await get();
  expect(built.timeline.segments.map((s: Range) => [s.start, s.end])).toEqual([
    [2, 5],
    [15, 17],
  ]);
  const accepted = (built.feedback as FeedbackRow[])
    .filter((f) => f.decision === "accept")
    .map((f) => f.candidate_id)
    .sort();
  expect(accepted).toEqual([first.id, third.id].sort());
  expect(
    (built.feedback as FeedbackRow[]).find((f) => f.candidate_id === second.id)
      ?.decision,
  ).toBe("reject");

  // Back in the review, the picks are cleared and the acceptances are shown; a
  // selection-order build keeps the picked sequence.
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await expect(
    page.locator(".candidate-card .badge", { hasText: "Accepted for editing" }),
  ).toHaveCount(2);
  await expect(builder).toContainText("Build one cut from several candidates");
  await page.getByLabel("Include candidate 3 in the cut").check();
  await page.getByLabel("Include candidate 1 in the cut").check();
  await page.getByLabel("Clip order").selectOption("selection");
  await page.getByRole("button", { name: "Build cut from 2" }).click();
  await expect(page).toHaveURL(/tab=editor/);
  await expect.poll(ranges).toEqual([
    [15, 17],
    [2, 5],
  ]);
  const revision = (await get()).revision;
  expect(revision).toBeGreaterThanOrEqual(3);

  // Audit trail page: read-only, filterable by project and event prefix.
  const projectTitle = built.title as string;
  await page.goto("/?page=audit");
  await expect(
    page.getByRole("heading", { name: "Audit trail." }),
  ).toBeVisible();
  const rows = page.locator(".audit-table tbody tr");
  await expect(rows.first()).toBeVisible();
  await page.getByLabel("Filter audit by project").selectOption(id);
  await expect.poll(() => rows.count()).toBeGreaterThan(3);
  const owners = await rows.locator("td:nth-child(3)").allInnerTexts();
  expect(owners.length).toBeGreaterThan(3);
  expect(owners.every((t) => t.trim() === projectTitle)).toBeTruthy();
  await page.getByLabel("Filter audit by event type").selectOption("feedback.");
  // One API rejection plus two accepted candidates per build.
  await expect.poll(() => rows.count()).toBe(5);
  const events = await rows.locator("td.event").allInnerTexts();
  expect(events.every((e) => e.trim() === "feedback.recorded")).toBeTruthy();
  await expect(rows.first().locator("td.details")).toContainText(
    "decision accept · candidate",
  );
  await expect(rows.first().locator("td.details")).toContainText(
    "Included in a 2-clip cut",
  );
  await page.getByLabel("Filter audit by event type").selectOption("timeline.");
  await expect.poll(() => rows.count()).toBeGreaterThanOrEqual(2);
  await expect(rows.first().locator("td.details")).toContainText(
    `revision ${revision}`,
  );
  await rows.first().locator("td.details summary").click();
  await expect(rows.first().locator("td.details pre")).toContainText(
    "fingerprint",
  );
  await page.getByLabel("Filter audit by event type").selectOption("");
  await rows.first().getByRole("button", { name: projectTitle }).click();
  await expect(page).toHaveURL(new RegExp("project=" + id));
  await expect(
    page.getByRole("heading", { name: projectTitle, level: 1 }),
  ).toBeVisible();

  // Appearance: the switch writes <html data-theme>, persists across reloads,
  // updates the browser chrome colour and "System" clears the override.
  await page.goto("/?page=audit");
  const html = page.locator("html");
  const background = () =>
    page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const dark = await background();
  await page.getByRole("radio", { name: "Light" }).click();
  await expect(html).toHaveAttribute("data-theme", "light");
  const light = await background();
  expect(light).not.toBe(dark);
  expect(
    await page.evaluate(() =>
      document
        .querySelector('meta[name="theme-color"]')
        ?.getAttribute("content"),
    ),
  ).toBe(light);
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", "light");
  expect(await background()).toBe(light);
  await expect(page.getByRole("radio", { name: "Light" })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await page.getByRole("radio", { name: "System" }).click();
  await expect(html).not.toHaveAttribute("data-theme");
  expect(await background()).toBe(dark);
  expect(
    await page.evaluate(() => localStorage.getItem("vektri.theme")),
  ).toBeNull();

  // The audit page and the appearance switch issued no writes; the builds did.
  expect(
    writes.filter((w) => !w.endsWith("/timeline") && !w.endsWith("/feedback")),
  ).toEqual([]);
  expect(writes.filter((w) => w.startsWith("PUT ")).length).toBe(2);
  expect(errors).toEqual([]);
});
