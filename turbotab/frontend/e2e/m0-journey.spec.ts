/**
 * What remains of the M0 journey (BLUEPRINT §8, tier C). The project-screen journeys it
 * drove — the M0 Record and the Rows / Columns / Results pipeline panel — were retired in
 * M1 part 2 (M1_CONTRACT §10: the panel's jobs moved to the banner and the stage); the
 * Record's journey is e2e/m1-record.spec.ts. The motion lab is unchanged and checked here.
 *
 *   npm run test:e2e                                      the MSW mock (Playwright starts Vite)
 *   E2E_BASE_URL=http://127.0.0.1:8792 npm run test:e2e   a running TurboTab server
 *
 * Screenshots land in docs/turbotab-next/m0/screens/<prefix>-*.png for review.
 */
import { mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const SCREENS = resolve(REPO, "docs/turbotab-next/m0/screens");
const MAX_SCREEN_BYTES = 300 * 1024;

let prefix = process.env.E2E_SCREEN_PREFIX ?? "mock";

/** Ask the backend what it is, and pick the screenshot prefix to match. */
async function backend(page: Page) {
  const health = await page.evaluate(async () => {
    const res = await fetch("/api/health");
    return (await res.json()) as { version: string };
  });
  prefix = process.env.E2E_SCREEN_PREFIX ?? (health.version.endsWith("-mock") ? "mock" : "real");
}

async function shoot(page: Page, name: string) {
  mkdirSync(SCREENS, { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(400); // let settle/arrive finish
  const path = resolve(SCREENS, `${prefix}-${name}.png`);
  await page.screenshot({ path, animations: "disabled" });
  expect(statSync(path).size, `${name} screenshot size`).toBeLessThan(MAX_SCREEN_BYTES);
}

test("the motion lab", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/lab");
  await expect(page.getByRole("heading", { name: "Motion lab" })).toBeVisible();
  await backend(page);
  await page.getByRole("button", { name: "Change an upstream answer" }).click();
  await expect(page.getByTestId("lab-veil-0")).toHaveAttribute("data-veil", "stale");
  await expect(page.getByTestId("lab-veil-3")).toHaveAttribute("data-veil", "fresh", {
    timeout: 5_000,
  });
  await page.getByTestId("reduced-motion").click();
  await expect(page.getByTestId("reduced-motion")).toHaveAttribute("aria-checked", "true");
  await shoot(page, "lab-light");
});
