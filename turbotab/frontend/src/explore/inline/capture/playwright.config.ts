/// <reference types="node" />
/**
 * Review captures for /lab/explore/inline (screenshots, the motion frame strip, visible word
 * counts) plus the keyboard and reduced-motion checks. Needs no API server:
 *
 *   npx playwright test --config src/explore/inline/capture/playwright.config.ts
 *
 * Outputs land in docs/turbotab-next/m1/explore/inline/.
 */
import { defineConfig, devices } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "../../../..");
const PORT = Number(process.env.E2E_PORT ?? 5474);

export default defineConfig({
  testDir: HERE,
  testMatch: /.*\.e2e\.ts$/,
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: `npx vite --port ${PORT} --strictPort --host 127.0.0.1`,
    cwd: FRONTEND,
    url: `http://127.0.0.1:${PORT}/lab/explore/inline`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
