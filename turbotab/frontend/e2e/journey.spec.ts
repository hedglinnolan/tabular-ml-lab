/**
 * The M0 journey (BLUEPRINT §8, tier C): open a table, answer lens -> outcome ->
 * purpose, see the decision sentences and the pipeline panel, then change the
 * outcome and watch the downstream sections go stale and come back fresh.
 * Screenshots land in docs/turbotab-next/m0/screens/ for review.
 */
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const SCREENS = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../docs/turbotab-next/m0/screens",
);
const PREFIX = process.env.E2E_SCREEN_PREFIX ?? "mock";

async function shoot(page: Page, name: string) {
  mkdirSync(SCREENS, { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(400); // let settle/arrive finish
  await page.screenshot({
    path: resolve(SCREENS, `${PREFIX}-${name}.png`),
    animations: "disabled",
  });
}

async function idle(page: Page) {
  await expect(page.getByLabel("Work in progress")).toHaveCount(0, { timeout: 15_000 });
}

test("open -> lens -> outcome -> purpose, then a changed outcome propagates", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  await shoot(page, "start-light");

  // Open by path through the file browser.
  await page.getByRole("button", { name: "data folder" }).click();
  await page.getByRole("button", { name: "Open dietary_recalls.csv" }).click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);

  // Ingest finishes: the Rows node counts the loaded table.
  await expect(page.getByTestId("rows-n")).toHaveAttribute("data-value", "600");
  await expect(page.getByTestId("rows-n")).toHaveText("600");

  // Columns are grouped by type, with counts.
  await expect(page.getByTestId("group-numeric")).toContainText("12");
  await expect(page.getByTestId("group-integer")).toContainText("2");
  await expect(page.getByTestId("group-categorical")).toContainText("2");
  await expect(page.getByTestId("group-datetime")).toContainText("1");

  // Lens: hints are shown beside the options, never pre-selected.
  await expect(page.getByText("because there is a total-energy column")).toBeVisible();
  await expect(page.getByTestId("lens-dietary")).toHaveAttribute("aria-pressed", "false");
  await page.getByTestId("lens-dietary").click();
  await page.getByTestId("lens-clinical").click();
  await page.getByRole("button", { name: "Record these 2 lenses" }).click();
  const lens = page.getByTestId("decision-lens");
  await expect(lens).toContainText("The table was read through the dietary and clinical lenses.");

  // Outcome: search the column picker, choose with the keyboard, record.
  const search = page.getByRole("combobox");
  await search.fill("hba1c");
  await expect(page.getByTestId("picker-count")).toHaveText("1 of 17");
  await search.press("Enter");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText(
    "hba1c was chosen as the outcome.",
  );

  // Task: detected at high confidence, so it is stated rather than asked (and not green).
  await expect(page.getByTestId("skip-task")).toContainText("Not asked");
  await expect(page.getByTestId("skip-task")).toContainText("regression");

  // Purpose.
  await page.getByTestId("purpose-prediction").click();
  await expect(page.getByTestId("decision-purpose")).toContainText(
    "The analysis was declared for prediction",
  );
  await expect(page.locator('[data-block="decision"]')).toHaveCount(3);

  // The pipeline panel: the outcome is marked in Columns.
  await expect(page.getByTestId("columns-target")).toContainText("hba1c");
  await expect(page.getByTestId("columns-target")).toContainText("regression");
  await expect(page.locator('[data-testid="group-numeric"] [data-target]')).toHaveText("hba1c");

  // Findings: bounded to five with a counted expander.
  await expect(page.getByTestId("findings-more")).toHaveText(/^\d+ more findings?$/);
  await expect(page.getByTestId("veil-findings")).toHaveAttribute("data-veil", "fresh");
  await idle(page);
  await shoot(page, "project-light");

  await page.emulateMedia({ colorScheme: "dark" });
  await shoot(page, "project-dark");
  await page.emulateMedia({ colorScheme: "light" });

  // Change the outcome: downstream goes stale, then fresh.
  await page.getByRole("button", { name: "Change the outcome" }).click();
  await page.getByRole("combobox").fill("bmi");
  await page.getByRole("combobox").press("Enter");
  await page.getByTestId("record-target").click();

  const columns = page.getByTestId("veil-columns");
  await expect(columns).toHaveAttribute("data-veil", /stale|recomputing/);
  await expect(page.getByTestId("veil-findings")).toHaveAttribute("data-veil", /stale|recomputing/);
  await expect(page.getByTestId("veil-task")).toHaveAttribute("data-veil", /stale|recomputing/);
  await page.waitForTimeout(200);
  await page.screenshot({
    path: resolve(SCREENS, `${PREFIX}-project-stale-light.png`),
    animations: "disabled",
  });

  await expect(columns).toHaveAttribute("data-veil", "fresh", { timeout: 15_000 });
  await expect(page.getByTestId("veil-findings")).toHaveAttribute("data-veil", "fresh");
  await expect(page.getByTestId("decision-target")).toContainText("bmi was chosen as the outcome.");
  await expect(page.getByTestId("columns-target")).toContainText("bmi");
  await expect(page.locator('[data-testid="group-numeric"] [data-target]')).toHaveText("bmi");
  // The earlier answer stays in the record, marked superseded.
  const history = page.getByLabel("Earlier answers");
  await expect(history).toContainText("hba1c was chosen as the outcome.");
  await expect(history).toContainText("superseded");
  // A finding that reads the outcome was recomputed for the new one.
  await expect(page.getByTestId("finding-energy_adjustment")).toContainText("bmi");
});

test("a wide table and the motion lab", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/");
  await expect(page.getByRole("link", { name: /genomics counts wide/ })).toBeVisible();
  await shoot(page, "start-dark");
  await page.getByRole("link", { name: /genomics counts wide/ }).click();
  await expect(page.getByTestId("rows-n")).toHaveAttribute("data-value", "60");
  await expect(page.getByTestId("group-integer")).toContainText("1,998");
  await page.getByRole("combobox").fill("ENSG0000010");
  await expect(page.getByTestId("picker-count")).toHaveText(/of 2,000$/);
  await idle(page);
  await shoot(page, "genomics-dark");

  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/lab");
  await expect(page.getByRole("heading", { name: "Motion lab" })).toBeVisible();
  await page.getByRole("button", { name: "Change an upstream answer" }).click();
  await expect(page.getByTestId("lab-veil-0")).toHaveAttribute("data-veil", "stale");
  await expect(page.getByTestId("lab-veil-3")).toHaveAttribute("data-veil", "fresh", {
    timeout: 5_000,
  });
  await page.getByTestId("reduced-motion").click();
  await expect(page.getByTestId("reduced-motion")).toHaveAttribute("aria-checked", "true");
  await shoot(page, "lab-light");
});
