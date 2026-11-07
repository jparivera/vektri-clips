import { test as base, expect, type Request, type Page } from "../frontend/node_modules/@playwright/test";

// All API traffic is fulfilled in-memory; this suite must never use
// the APIRequestContext fixture or send mutations to the real application.
type Reply = { status?: number; body: unknown };
type Call = { method: string; path: string; query: string; body: unknown };
type Handler = (call: Call, url: URL) => Reply | Promise<Reply>;
const response = (body: unknown, status = 200): Reply => ({ status, body });
const clone = <T,>(value: T): T => structuredClone(value);

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

class HeldReply {
  readonly incoming = deferred<Request>();
  readonly result = deferred<Reply>();
  request?: Request;
  released = false;
  async handle(request: Request) {
    this.request = request;
    this.incoming.resolve(request);
    return this.result.promise;
  }
  release(reply: Reply) {
    if (!this.released) {
      this.released = true;
      this.result.resolve(reply);
    }
  }
}

function makeProject(id: string, title: string) {
  return {
    id, title, genre: "speech", status: "ready", revision: 1,
    tags: [], created_at: "2026-11-04T00:00:00Z", updated_at: "2026-11-04T00:00:00Z",
    source: {
      sha256: "a".repeat(64), original_name: "in-memory-fixture.mp4",
      metadata: { duration: 20, width: 1280, height: 720, fps: 24, audio: false, video: true },
      provenance: { permission_status: "synthetic", attribution: "In-memory UI regression fixture" },
    },
    timeline: {
      schema_version: 1,
      segments: [{ id: "initial", start: 0, end: 20, gain_db: 0,
        crop: { mode: "fit", x: .5, y: .5, zoom: 1, second_x: .75, keyframes: [] } }],
      output: { width: 720, height: 1280, fps: 24, crf: 22 },
      captions: { enabled: false, burn_in: false, style: "clean", font_size: 42,
        margin: 100, alignment: 2, background: false, highlight: false, words: [] },
      audio: { normalize: false, fade: 0, stream_index: 0 },
    },
    transcript: null, ocr: null, jobs: [], exports: [],
    feedback: [] as { candidate_id: string; decision: string; note: string }[],
    analysis: {
      provider: "manual", limitations: [],
      candidates: [
        { id: "candidate-first", start: 2, end: 5, title: "First moment", rationale: "Manual fixture", method: "manual", uncertainty: [] },
        { id: "candidate-second", start: 8, end: 11, title: "Second moment", rationale: "Manual fixture", method: "manual", uncertainty: [] },
      ],
    },
  };
}

type Mock = {
  projects: ReturnType<typeof makeProject>[];
  calls: Call[];
  on(method: string, path: string, handler: Handler): void;
  hold(method: string, path: string, predicate?: (url: URL) => boolean): HeldReply;
};

const test = base.extend<{ mock: Mock }>({
  mock: async ({ context, page, baseURL }, use, testInfo) => {
    const origin = new URL(baseURL || "http://127.0.0.1:8741").origin;
    const projects = [makeProject("ui-race-project-one", "UI race source one"), makeProject("ui-race-project-two", "UI race source two")];
    const calls: Call[] = [];
    const unexpected: string[] = [];
    const pageErrors: string[] = [];
    const apiRequested = new Set<Request>();
    const apiIntercepted = new Set<Request>();
    const handlers = new Map<string, Handler>();
    const holds: { method: string; path: string; predicate: (url: URL) => boolean; gate: HeldReply }[] = [];
    context.on("request", (request) => {
      if (new URL(request.url()).pathname.startsWith("/api/")) apiRequested.add(request);
    });
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const on = (method: string, path: string, handler: Handler) => handlers.set(`${method} ${path}`, handler);
    on("GET", "/api/v1/projects", () => response(clone(projects)));
    on("GET", "/api/v1/jobs", () => response([]));
    on("GET", "/api/v1/exports", () => response([]));
    on("GET", "/api/v1/presets", () => response([]));
    on("GET", "/api/v1/health", () => response({ status: "ok", queue_length: 0, worker_active: false,
      worker: { alive: true, state: "idle", workers: [] } }));
    for (const project of projects) {
      on("GET", `/api/v1/projects/${project.id}`, () => response(clone(project)));
      on("GET", `/api/v1/projects/${project.id}/revisions`, () => response([]));
    }
    await context.route("**/*", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      if (!url.pathname.startsWith("/api/")) {
        // Only the existing local static application is fetched. No media,
        // arbitrary document, external host, or non-GET request is allowed out.
        if (url.origin === origin && request.method() === "GET" &&
            (url.pathname === "/" || url.pathname.startsWith("/assets/") ||
             ["/favicon.svg", "/favicon.ico", "/third-party-notices.txt"].includes(url.pathname))) {
          await route.continue();
        } else {
          unexpected.push(`Blocked non-static request: ${request.method()} ${url.href}`);
          await route.abort("blockedbyclient");
        }
        return;
      }
      apiIntercepted.add(request);
      const call: Call = { method: request.method(), path: url.pathname, query: url.search,
        body: request.postData() ? request.postDataJSON() : null };
      calls.push(call);
      const held = holds.find((h) => !h.gate.request && h.method === call.method && h.path === call.path && h.predicate(url));
      const handler = handlers.get(`${call.method} ${call.path}`);
      let reply: Reply;
      if (held) reply = await held.gate.handle(request);
      else if (handler) reply = await handler(call, url);
      else {
        unexpected.push(`Unmocked API request: ${call.method} ${url.pathname}${url.search}`);
        reply = response({ detail: "Blocked by the isolated UI regression harness" }, 599);
      }
      // API requests always terminate here. Do not replace with continue/fetch/fallback.
      await route.fulfill({ status: reply.status ?? 200, json: reply.body });
    });
    try {
      await use({ projects, calls, on, hold(method, path, predicate = () => true) {
        const gate = new HeldReply();
        holds.push({ method, path, predicate, gate });
        return gate;
      } });
    } finally {
      for (const { gate } of holds) gate.release(response({ detail: "Fixture teardown" }, 503));
      await page.close();
      await context.unrouteAll({ behavior: "wait" });
      await testInfo.attach("mocked-network.json", { body: Buffer.from(JSON.stringify({ calls, unexpected,
        apiRequested: apiRequested.size, apiIntercepted: apiIntercepted.size }, null, 2)), contentType: "application/json" });
      expect(unexpected, "No request may escape the API/static allowlist").toEqual([]);
      expect([...apiRequested].filter((request) => !apiIntercepted.has(request)), "Every API request must be intercepted").toEqual([]);
      expect(pageErrors).toEqual([]);
    }
  },
});

// Wait for the exact held response to finish, then allow React to paint it.
// This uses response completion and animation frames, never a timing sleep.
async function releaseAndPaint(page: Page, gate: HeldReply, reply: Reply) {
  const request = await gate.incoming.promise;
  const delivered = page.waitForResponse((r: { request(): Request }) => r.request() === request);
  gate.release(reply);
  await (await delivered).finished();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

test("latest acceptance makes a previously rejected candidate usable in the cut builder", async ({ page, mock }) => {
  const project = mock.projects[0];
  const timelineWrites: unknown[] = [];
  mock.on("PUT", `/api/v1/projects/${project.id}/timeline`, (call) => {
    const body = call.body as { timeline: typeof project.timeline };
    project.timeline = clone(body.timeline);
    project.revision += 1;
    timelineWrites.push(clone(body.timeline));
    return response({ revision: project.revision, timeline: clone(project.timeline) });
  });
  mock.on("POST", `/api/v1/projects/${project.id}/feedback`, (call) => {
    // Backend project feedback is newest-first; history intentionally retains rejection.
    project.feedback.unshift(clone(call.body as (typeof project.feedback)[number]));
    return response({ ok: true });
  });
  await page.goto(`/?project=${project.id}&tab=analysis`);
  const first = page.locator(".candidate-card").first();
  await first.getByRole("button", { name: "Reject", exact: true }).click();
  await expect(page.getByLabel("Include candidate 1 in the cut")).toBeDisabled();
  await first.getByRole("button", { name: "Edit this clip" }).click();
  await expect(page).toHaveURL(/tab=editor/);
  await page.getByRole("tab", { name: "02 Analysis & candidates" }).click();
  await expect(first).toContainText("Accepted for editing");
  await expect(page.getByLabel("Include candidate 1 in the cut")).toBeEnabled();
  await page.getByLabel("Include candidate 1 in the cut").check();
  await page.getByLabel("Include candidate 2 in the cut").check();
  await expect(page.getByRole("region", { name: "Cut builder" })).toContainText("2 selected · 2 clips · 6.0 s");
  await page.getByRole("button", { name: "Build cut from 2", exact: true }).click();
  await expect(page).toHaveURL(/tab=editor/);
  expect(project.timeline.segments.map(({ start, end }) => [start, end])).toEqual([[2, 5], [8, 11]]);
  expect(timelineWrites).toHaveLength(2);
  expect(project.feedback.some((f) => f.candidate_id === "candidate-first" && f.decision === "reject")).toBe(true);
});

function auditRow(seq: number, event: string) {
  return { seq, event, project_id: null, created_at: 1793754000, detail: { revision: seq } };
}

test("an older audit page cannot contaminate the current filter", async ({ page, mock }) => {
  mock.on("GET", "/api/v1/audit", (_call, url) => response(url.searchParams.get("event") === "timeline."
    ? [auditRow(2000, "timeline.saved")]
    : Array.from({ length: 100 }, (_, index) => auditRow(1000 - index, "job.succeeded"))));
  const oldPage = mock.hold("GET", "/api/v1/audit", (url) => url.searchParams.has("before") && !url.searchParams.has("event"));
  await page.goto("/?page=audit");
  await expect(page.locator(".audit-table tbody tr")).toHaveCount(100);
  await page.getByRole("button", { name: "Show older entries" }).click();
  await oldPage.incoming.promise;
  await page.getByLabel("Filter audit by event type").selectOption("timeline.");
  await expect(page.locator(".audit-table tbody tr")).toHaveCount(1);
  await expect(page.locator(".audit-table tbody tr")).toHaveAttribute("data-seq", "2000");
  await releaseAndPaint(page, oldPage, response([auditRow(900, "job.failed")]));
  await expect(page.getByLabel("Filter audit by event type")).toHaveValue("timeline.");
  await expect(page.locator(".audit-table tbody tr")).toHaveCount(1);
  await expect(page.locator(".audit-table td.event")).toHaveText("timeline.saved");
});

function storagePlan(fingerprint: string) {
  return {
    schema_version: 1, data_directory: "in-memory-ui-library", generated_at: 1793754000,
    read_only: true, automatic_deletion_supported: false, status: "review_only",
    maintenance_required: false, maintenance_reasons: [], blockers: [], settings: {},
    inventory: {}, inventory_complete: true, protected_paths: [], retained_caches: [],
    review_candidates: [], estimated_reclaimable_bytes: 0,
    snapshot_fingerprint: fingerprint, limitations: ["Synthetic UI fixture; no filesystem scanned"],
    plan_fingerprint: fingerprint,
  };
}
function verification(fingerprint: string, meaning: string) {
  return { valid: true, read_only: true, automatic_deletion_supported: false,
    reasons: [], current_fingerprint: fingerprint, maintenance_required: false,
    maintenance_reasons: [], meaning };
}

test("verification for an old storage plan is discarded after refresh", async ({ page, mock }) => {
  const planA = storagePlan("a".repeat(64)), planB = storagePlan("b".repeat(64));
  let plansRead = 0;
  mock.on("GET", "/api/v1/maintenance/storage", () => response({
    data_directory: "in-memory-ui-library", generated_at: 1793754000,
    disk: { total: 100000, used: 10000, free: 90000 },
    directories: { projects: 4000, models: 0, uploads: 0, logs: 0 }, other_directories: {},
    database_bytes: 1000, file_count: 2, symlinks_skipped: 0, complete: true,
    policy: "No automatic deletion.",
  }));
  mock.on("GET", "/api/v1/maintenance/storage-plan", () => response(++plansRead === 1 ? planA : planB));
  const verifyA = mock.hold("POST", "/api/v1/maintenance/verify-plan");
  mock.on("POST", "/api/v1/maintenance/verify-plan", (call) => {
    expect((call.body as typeof planB).plan_fingerprint).toBe(planB.plan_fingerprint);
    return response(verification(planB.snapshot_fingerprint, "Fresh verification for plan B"));
  });
  await page.goto("/?page=storage");
  await page.getByRole("button", { name: "Verify this plan", exact: true }).click();
  const oldRequest = await verifyA.incoming.promise;
  expect(oldRequest.postDataJSON().plan_fingerprint).toBe(planA.plan_fingerprint);
  await page.getByRole("button", { name: "Refresh plan", exact: true }).click();
  const download = page.getByRole("link", { name: "Download plan JSON" });
  await expect.poll(async () => JSON.parse(decodeURIComponent((await download.getAttribute("href"))!.split(",")[1])).plan_fingerprint).toBe(planB.plan_fingerprint);
  await releaseAndPaint(page, verifyA, response(verification(planA.snapshot_fingerprint, "STALE verification for plan A")));
  await expect(page.locator(".verify-result")).toHaveCount(0);
  await page.getByRole("button", { name: "Verify this plan", exact: true }).click();
  await expect(page.locator(".verify-result")).toContainText("Fresh verification for plan B");
  await expect(page.locator(".verify-result")).not.toContainText("STALE");
});

test("a failed batch render preserves the selected projects for retry", async ({ page, mock }) => {
  mock.on("POST", "/api/v1/batch/render", (call) => {
    expect((call.body as { items: unknown[] }).items).toHaveLength(2);
    return response({ detail: "Synthetic batch rejection; no jobs were created" }, 409);
  });
  await page.goto("/");
  for (const project of mock.projects)
    await page.getByLabel(`Select ${project.title} for batch render`).check();
  await page.getByRole("button", { name: "Render 2 selected", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Synthetic batch rejection; no jobs were created");
  for (const project of mock.projects)
    await expect(page.getByLabel(`Select ${project.title} for batch render`)).toBeChecked();
  await expect(page.getByRole("button", { name: "Render 2 selected", exact: true })).toBeVisible();
  expect(mock.calls.filter((call) => call.method === "POST" && call.path === "/api/v1/batch/render")).toHaveLength(1);
});
