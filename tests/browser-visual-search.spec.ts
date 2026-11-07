import { test, expect } from "../frontend/node_modules/@playwright/test";

test("visual search uses real CLIP and exposes separate speech language controls", async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const capabilities = await (await request.get("/api/v1/capabilities")).json();
  expect(
    capabilities.models.models.find(
      (m: { id: string }) => m.id === "clip-vit-b32",
    ).status,
  ).toBe("ready");
  const imported = await request.post("/api/v1/fixtures/action_motion.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id;
  const get = async () => (await request.get("/api/v1/projects/" + id)).json();
  await expect.poll(async () => (await get()).status).toBe("ready");
  await page.goto("/?project=" + id);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  const visual = page.getByLabel(
    "Search visible content (experimental)",
  );
  await expect(visual).toBeChecked();
  const find = page.getByRole("button", {
    name: "Find candidate moments",
    exact: true,
  });
  await expect(find).toBeDisabled();
  await page
    .getByLabel("Editorial direction")
    .fill("A colorful object moving across a dark background");
  await page.getByLabel("Target candidates").fill("1");
  const queued = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().endsWith("/analyze"),
  );
  await find.click();
  const submitted = (await queued).postDataJSON();
  expect(submitted.visual_search).toBe(true);
  expect(submitted.transcribe).toBe(false);
  await expect
    .poll(
      async () =>
        (await get()).jobs.find((j: { kind: string }) => j.kind === "analyze")
          ?.status,
      { timeout: 90000 },
    )
    .toBe("succeeded");
  const completed = await get();
  expect(completed.analysis.provider).toBe("local_onnx_clip");
  expect(completed.analysis.ai_used).toBe(true);
  expect(completed.analysis.visual_search.sampling_complete).toBe(true);
  expect(completed.analysis.candidates[0].method).toBe("local_clip_visual");
  expect(
    completed.jobs.some((j: { kind: string }) => j.kind === "transcribe"),
  ).toBe(false);
  await expect(
    page.getByText("CLIP visual similarity", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Footage type").selectOption("speech");
  await expect(visual).not.toBeChecked();
  await page.getByLabel("Speech language").selectOption("pt");
  await expect(page.getByLabel("Speech language")).toHaveValue("pt");
  expect(errors).toEqual([]);
});
