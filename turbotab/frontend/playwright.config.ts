import { defineConfig, devices } from "@playwright/test";

// Default: start the mock dev server (MSW) on E2E_PORT and drive it.
// Against a real server: E2E_BASE_URL=http://127.0.0.1:5173 npm run test:e2e
const PORT = Number(process.env.E2E_PORT ?? 5391);
const external = process.env.E2E_BASE_URL;
const baseURL = external ?? `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: external
    ? undefined
    : {
        command: `npm run dev:mock -- --port ${PORT} --strictPort --host 127.0.0.1`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
      },
});
