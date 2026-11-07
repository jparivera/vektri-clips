import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";
import path from "node:path";
const evidenceDir = process.env.VEKTRI_TEST_EVIDENCE_DIR || "evidence";
const executablePath =
  process.env.VEKTRI_BROWSER_EXECUTABLE ||
  (existsSync("/usr/bin/google-chrome") ? "/usr/bin/google-chrome" : undefined);
export default defineConfig({
  testDir: "../tests",
  testMatch: "browser*.spec.ts",
  workers: 1,
  timeout: 120000,
  expect: { timeout: 15000 },
  use: {
    actionTimeout: 15000,
    navigationTimeout: 20000,
    baseURL: "http://127.0.0.1:8741",
    viewport: { width: 1440, height: 1000 },
    // The app follows prefers-color-scheme; existing flows and screenshots stay dark.
    colorScheme: "dark",
    launchOptions: {
      executablePath,
      args: ["--no-sandbox"],
    },
    trace: "retain-on-failure",
  },
  reporter: [
    ["list"],
    ["json", { outputFile: path.join(evidenceDir, "browser-results.json") }],
  ],
  outputDir: path.join(evidenceDir, "test-artifacts"),
});
