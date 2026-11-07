import { test, expect } from "../frontend/node_modules/@playwright/test";
import AxeBuilder from "../frontend/node_modules/@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(__dirname, "..");
const evidence = path.resolve(
  process.env.VEKTRI_TEST_EVIDENCE_DIR || path.join(root, "frontend/evidence"),
);
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"];

test("axe reports no WCAG 2.1 AA or best-practice violations on any page or dialog in either colour scheme", async ({
  page,
  request,
}) => {
  test.setTimeout(360_000); // 13 views × 2 colour schemes, each a full axe run
  const imported = await request.post("/api/v1/fixtures/spoken_interview.mp4");
  expect(imported.ok()).toBeTruthy();
  const id = (await imported.json()).project_id as string;
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/v1/projects/" + id)).json()).status,
    )
    .toBe("ready");
  const marked = await request.post(`/api/v1/projects/${id}/event`, {
    data: { time: 3, before: 1, after: 2, title: "Opening line" },
  });
  expect(marked.ok()).toBeTruthy();

  const views: [string, string, (() => Promise<void>)?][] = [
    ["library", "/"],
    ["exports", "/?page=exports"],
    ["jobs", "/?page=jobs"],
    ["publishing", "/?page=publishing"],
    ["models", "/?page=models"],
    ["storage", "/?page=storage"],
    ["audit", "/?page=audit"],
    ["source", `/?project=${id}`],
    ["analysis", `/?project=${id}&tab=analysis`],
    ["editor", `/?project=${id}&tab=editor`],
    ["export", `/?project=${id}&tab=export`],
    [
      "import-dialog",
      "/",
      async () => {
        await page
          .getByRole("button", { name: "Import media", exact: true })
          .click();
        await expect(
          page.getByRole("dialog", { name: "Bring a source into your studio" }),
        ).toBeVisible();
      },
    ],
    [
      "history-dialog",
      `/?project=${id}&tab=editor`,
      async () => {
        await page
          .getByRole("button", { name: "History", exact: true })
          .click();
        await expect(
          page.getByRole("dialog", { name: "Revision history" }),
        ).toBeVisible();
        await expect(page.locator(".revision-row").first()).toBeVisible();
      },
    ],
  ];
  const results: {
    view: string;
    scheme: string;
    url: string;
    passes: number;
    incomplete: string[];
    violations: {
      id: string;
      impact: string | null | undefined;
      help: string;
      nodes: number;
      targets: string[];
      samples: { html: string; data: unknown }[];
    }[];
  }[] = [];
  for (const scheme of ["dark", "light"] as const) {
    await page.emulateMedia({ colorScheme: scheme });
    for (const [view, url, prepare] of views) {
      await page.goto(url);
      await expect(page.locator("h1").first()).toBeVisible();
      await page.waitForTimeout(400);
      if (prepare) await prepare();
      const axe = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      results.push({
        view,
        scheme,
        url,
        passes: axe.passes.length,
        incomplete: [...new Set(axe.incomplete.map((r) => r.id))],
        violations: axe.violations.map((v) => ({
          id: v.id,
          impact: v.impact,
          help: v.help,
          nodes: v.nodes.length,
          targets: v.nodes.slice(0, 4).map((n) => n.target.join(" ")),
          samples: v.nodes.slice(0, 2).map((n) => ({
            html: n.html.slice(0, 200),
            data: n.any[0]?.data ?? null,
          })),
        })),
      });
    }
  }
  fs.mkdirSync(evidence, { recursive: true });
  const axeVersion = JSON.parse(
    fs.readFileSync(
      path.join(root, "frontend/node_modules/axe-core/package.json"),
      "utf8",
    ),
  ).version as string;
  fs.writeFileSync(
    path.join(evidence, "accessibility-summary.json"),
    JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        axe_core: axeVersion,
        tags: TAGS,
        views: views.length,
        schemes: ["dark", "light"],
        total_violations: results.reduce((n, r) => n + r.violations.length, 0),
        results,
      },
      null,
      2,
    ) + "\n",
  );
  const violations = results.flatMap((r) =>
    r.violations.map(
      (v) =>
        `${r.view}/${r.scheme}: ${v.id} [${v.impact}] ×${v.nodes} — ${v.targets.join(" | ")}`,
    ),
  );
  expect(violations).toEqual([]);
});
