/**
 * The M0 journey (BLUEPRINT §8, tier C). One spec, two backends:
 *
 *   npm run test:e2e                                      the MSW mock (Playwright starts Vite)
 *   E2E_BASE_URL=http://127.0.0.1:8792 npm run test:e2e   a running TurboTab server
 *
 * Which backend answers is read from GET /api/health (the mock's version ends in
 * "-mock"), and the fixtures and the screenshot prefix (mock-*, real-*; override with
 * E2E_SCREEN_PREFIX) follow: the real server opens the repository's sample data by
 * path through the file browser; the mock opens its pretend files.
 *
 * Open a table -> lens -> outcome -> purpose; the decision sentences settle and the
 * pipeline panel fills; change the outcome and watch stale -> fresh. Then open wide
 * tables and check the table preview and the outcome picker stay responsive.
 * Screenshots land in docs/turbotab-next/m0/screens/<prefix>-*.png for review.
 */
import { existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../..");
const SCREENS = resolve(REPO, "docs/turbotab-next/m0/screens");
const MAX_SCREEN_BYTES = 300 * 1024;

interface Wide {
  dir: string;
  file: string;
  rows: number;
  cols: number;
  /** A search the outcome picker should narrow to a strict subset. */
  search: string;
  shot?: string;
}

interface Fixtures {
  dir: string;
  dietary: string;
  /** Hint text the profile gives the dietary lens (a substring). */
  dietaryHint: RegExp;
  wide: Wide[];
}

const SAMPLE = resolve(REPO, "turbotab/sample_data");
const REAL: Fixtures = {
  dir: SAMPLE,
  dietary: "dietary_recalls.csv",
  dietaryHint: /energy|intake|kcal|diet/i,
  wide: [
    // A local-only NHANES extract (not in git): long rather than wide; skipped when absent.
    {
      dir: REPO,
      file: "_tt_tmp_nhanes.csv",
      rows: 21_849,
      cols: 29,
      search: "fat",
      shot: "nhanes",
    },
    {
      dir: SAMPLE,
      file: "genomics_expression.csv",
      rows: 60,
      cols: 500,
      search: "gene_01",
      shot: "genomics",
    },
  ].filter((w) => existsSync(resolve(w.dir, w.file))),
};
const MOCK: Fixtures = {
  dir: "/Users/researcher/data",
  dietary: "dietary_recalls.csv",
  dietaryHint: /total-energy column/,
  wide: [
    {
      dir: "/Users/researcher/data",
      file: "genomics_counts_wide.csv",
      rows: 60,
      cols: 2_000,
      search: "ENSG0000010",
      shot: "genomics",
    },
  ],
};

let prefix = process.env.E2E_SCREEN_PREFIX ?? "mock";

/**
 * Ask the backend what it is, and pick fixtures and the screenshot prefix to match.
 * Call it once the page has rendered: the app renders only after the mock worker (if
 * any) is running, and a request made before that would reach Vite, not the mock.
 */
async function fixturesFor(page: Page): Promise<Fixtures> {
  const health = await page.evaluate(async () => {
    const res = await fetch("/api/health");
    return (await res.json()) as { version: string };
  });
  const mock = health.version.endsWith("-mock");
  prefix = process.env.E2E_SCREEN_PREFIX ?? (mock ? "mock" : "real");
  return mock ? MOCK : REAL;
}

async function shoot(page: Page, name: string) {
  mkdirSync(SCREENS, { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(400); // let settle/arrive finish
  const path = resolve(SCREENS, `${prefix}-${name}.png`);
  await page.screenshot({ path, animations: "disabled" });
  expect(statSync(path).size, `${name} screenshot size`).toBeLessThan(MAX_SCREEN_BYTES);
}

/** Wait until no job chip is showing. */
async function idle(page: Page, timeout = 30_000) {
  await expect(page.getByLabel("Work in progress")).toHaveCount(0, { timeout });
}

/** Walk the file browser from the disk's top to `dir`, one folder button at a time. */
async function browseTo(page: Page, dir: string) {
  const crumbs = page.getByRole("navigation", { name: "Folder" });
  await expect(crumbs).toBeVisible();
  await crumbs.getByRole("button", { name: "/", exact: true }).click();
  await expect(crumbs.locator('[aria-current="location"]')).toHaveCount(0);
  const entries = page.getByTestId("fs-entries");
  for (const part of dir.split("/").filter(Boolean)) {
    await entries.getByRole("button", { name: `${part} folder`, exact: true }).click();
    await expect(crumbs.locator('[aria-current="location"]')).toHaveText(part);
  }
}

async function openByBrowsing(page: Page, dir: string, file: string, shot?: string) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  await browseTo(page, dir);
  // Photographed inside the fixture folder, not the home folder the browser opens on.
  if (shot) await shoot(page, shot);
  await page.getByRole("button", { name: `Open ${file}`, exact: true }).click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
}

/** The count a Columns group shows, as a number. */
async function groupCounts(page: Page): Promise<number> {
  const groups = page.locator('[aria-label="Columns by type"] > li');
  const n = await groups.count();
  let total = 0;
  for (let i = 0; i < n; i++) {
    const text = await groups.nth(i).locator("div").first().innerText();
    total += Number(text.replace(/[^\d]/g, ""));
  }
  return total;
}

/** Record every `data-veil` value the given sections take from now on. */
async function watchVeils(page: Page, ids: string[]) {
  await page.evaluate((ids) => {
    const w = window as unknown as { __veils: Record<string, string[]> };
    w.__veils = Object.fromEntries(ids.map((id) => [id, []]));
    const note = () => {
      for (const id of ids) {
        const v = document.querySelector(`[data-testid="${id}"]`)?.getAttribute("data-veil");
        const seen = w.__veils[id]!;
        if (v && seen[seen.length - 1] !== v) seen.push(v);
      }
    };
    note();
    new MutationObserver(note).observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["data-veil"],
    });
  }, ids);
}

async function veilHistory(page: Page, id: string): Promise<string[]> {
  return page.evaluate(
    (id) => (window as unknown as { __veils: Record<string, string[]> }).__veils[id] ?? [],
    id,
  );
}

test("open -> lens -> outcome -> purpose, then a changed outcome propagates", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  const fx = await fixturesFor(page);

  // Open by path through the file browser.
  await openByBrowsing(page, fx.dir, fx.dietary, "start-light");

  // Ingest finishes: the Rows node counts the loaded table.
  await expect(page.getByTestId("rows-n")).toHaveAttribute("data-value", "600", {
    timeout: 30_000,
  });
  await expect(page.getByTestId("rows-n")).toHaveText("600");

  // Columns are grouped by type, and the groups account for all 17 columns.
  await expect(page.getByTestId("group-numeric")).toBeVisible();
  await expect.poll(() => groupCounts(page)).toBe(17);

  // Lens: hints are shown beside the options, never pre-selected.
  const dietary = page.getByTestId("lens-dietary");
  await expect(dietary).toHaveAttribute("aria-pressed", "false");
  await expect(dietary.locator("xpath=..")).toContainText("suggested", { timeout: 30_000 });
  await expect(dietary.locator("xpath=..")).toContainText(fx.dietaryHint);
  await dietary.click();
  await page.getByRole("button", { name: "Record this lens" }).click();
  const lens = page.getByTestId("decision-lens");
  await expect(lens).toContainText("The table was read through the dietary lens.");

  // Outcome: search the column picker, choose with the keyboard, record.
  const search = page.getByRole("combobox");
  await search.fill("hba1c");
  await expect(page.getByTestId("picker-count")).toHaveText("1 of 17");
  await search.press("Enter");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText(
    "hba1c was chosen as the outcome.",
  );

  // Task: hba1c is numeric with many values. If detection is confident it is stated,
  // not asked; otherwise it is asked, and regression is the answer.
  const skip = page.getByTestId("skip-task");
  const ask = page.getByTestId("question-task");
  await expect(skip.or(ask)).toBeVisible({ timeout: 30_000 });
  const asked = await ask.isVisible();
  if (asked) {
    await page.getByTestId("task-regression").click();
    await expect(page.getByTestId("decision-task")).toContainText(
      "hba1c was modeled as a regression task.",
    );
  } else {
    await expect(skip).toContainText("Not asked");
    await expect(skip).toContainText("regression");
  }

  // Purpose.
  await page.getByTestId("purpose-prediction").click();
  await expect(page.getByTestId("decision-purpose")).toContainText(
    "The analysis was declared for prediction",
  );
  await expect(page.locator('[data-block="decision"]')).toHaveCount(asked ? 4 : 3);

  // The pipeline panel: the outcome is marked in Columns.
  await expect(page.getByTestId("columns-target")).toContainText("hba1c");
  await expect(page.getByTestId("columns-target")).toContainText("regression", {
    timeout: 30_000,
  });
  await expect(page.locator("[data-target]")).toHaveText("hba1c");

  // Findings render, bounded to five with a counted expander when there are more.
  const findings = page.getByTestId("findings");
  await expect(page.getByTestId("veil-findings")).toHaveAttribute("data-veil", "fresh", {
    timeout: 60_000,
  });
  await expect(findings.locator('[data-testid^="finding-"]').first()).toBeVisible();
  expect(await findings.locator('li[data-testid^="finding-"]').count()).toBeLessThanOrEqual(5);
  await idle(page);
  await shoot(page, "project-light");

  await page.emulateMedia({ colorScheme: "dark" });
  await shoot(page, "project-dark");
  await page.emulateMedia({ colorScheme: "light" });

  // Change the outcome: downstream goes stale, then fresh. A real server can recompute
  // in tens of milliseconds, faster than a polling assertion samples, so every veil state
  // the sections pass through is recorded as it happens. For the review screenshot, stage
  // results are held back for a moment (the statuses still arrive live over SSE) so the
  // veiled state stays on screen long enough to photograph.
  const hold = "**/api/projects/*/stages/**";
  await page.route(hold, async (route) => {
    await new Promise((r) => setTimeout(r, 1_500));
    await route.continue();
  });
  await watchVeils(page, ["veil-columns", "veil-findings"]);
  await page.getByRole("button", { name: "Change the outcome" }).click();
  await page.getByRole("combobox").fill("bmi");
  await page.getByRole("combobox").press("Enter");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("veil-findings")).not.toHaveAttribute("data-veil", "fresh");
  await page.waitForTimeout(400); // the propagate sweep
  await page.screenshot({
    path: resolve(SCREENS, `${prefix}-project-stale-light.png`),
    animations: "disabled",
  });
  await page.unroute(hold);

  for (const id of ["veil-columns", "veil-findings"]) {
    await expect
      .poll(() => veilHistory(page, id), {
        timeout: 60_000,
        message: `${id} went stale, then fresh`,
      })
      .toEqual(expect.arrayContaining([expect.stringMatching(/stale|recomputing/)]));
    await expect(page.getByTestId(id)).toHaveAttribute("data-veil", "fresh", { timeout: 60_000 });
  }
  await expect(page.getByTestId("decision-target")).toContainText("bmi was chosen as the outcome.");
  // A task answer belongs to the outcome it was given for: bmi's task is detected or asked
  // anew, and an answer given for hba1c stays in the history, marked for another outcome.
  const taskSlot = page.locator('[data-slot="task"]');
  await expect(taskSlot.locator("[data-block]").first()).toContainText("bmi", { timeout: 30_000 });
  if (asked) {
    await expect(taskSlot.getByLabel("Earlier answers")).toContainText(
      "hba1c was modeled as a regression task.",
    );
    await expect(taskSlot.getByLabel("Earlier answers")).toContainText("another outcome");
  }
  await expect(page.getByTestId("columns-target")).toContainText("bmi");
  await expect(page.locator("[data-target]")).toHaveText("bmi");
  // The earlier answer stays in the record, marked superseded.
  const history = page.getByLabel("Earlier answers");
  await expect(history).toContainText("hba1c was chosen as the outcome.");
  await expect(history).toContainText("superseded");
  await idle(page);
});

test("large and wide tables keep the table preview and the outcome picker responsive", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  const fx = await fixturesFor(page);
  test.skip(fx.wide.length === 0, "no wide fixture on this machine");

  for (const [i, w] of fx.wide.entries()) {
    await openByBrowsing(page, w.dir, w.file, i === 0 ? "start-dark" : undefined);
    await expect(page.getByTestId("rows-n")).toHaveAttribute("data-value", String(w.rows), {
      timeout: 60_000,
    });
    await expect.poll(() => groupCounts(page), { timeout: 30_000 }).toBe(w.cols);

    // The table preview: cells arrive, and a far scroll fetches a new window promptly.
    const grid = page.getByRole("grid", { name: "Working data preview" });
    await expect(grid.getByRole("gridcell").first()).not.toBeEmpty({ timeout: 30_000 });
    const t0 = Date.now();
    await grid.evaluate((el) => {
      el.scrollTop = el.scrollHeight;
      el.scrollLeft = el.scrollWidth;
    });
    const lastRow = grid.locator(`[aria-rowindex="${w.rows + 1}"]`);
    await expect(lastRow).toBeVisible();
    await expect(lastRow.getByRole("gridcell").last()).not.toBeEmpty();
    expect(Date.now() - t0, "far scroll to a filled window").toBeLessThan(5_000);

    // The outcome picker (it appears once a lens is recorded) filters as you type.
    await page.getByTestId(`lens-${w.shot === "genomics" ? "genomics" : "dietary"}`).click();
    await page.getByRole("button", { name: "Record this lens" }).click();
    const picker = page.getByRole("combobox");
    await expect(picker).toBeVisible();
    await expect(page.getByTestId("picker-count")).toHaveText(
      `${w.cols.toLocaleString("en-US")} columns`,
    );
    const t1 = Date.now();
    await picker.fill(w.search);
    await expect(page.getByTestId("picker-count")).toHaveText(
      new RegExp(` of ${w.cols.toLocaleString("en-US")}$`),
    );
    expect(Date.now() - t1, "picker filter").toBeLessThan(2_000);
    await expect(page.getByRole("option").first()).toBeVisible();
    await picker.press("PageDown");
    await picker.press("Enter");
    await expect(page.getByTestId("record-target")).toBeEnabled();
    if (w.shot) await shoot(page, `${w.shot}-dark`);
  }
});

test("the motion lab", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/lab");
  await expect(page.getByRole("heading", { name: "Motion lab" })).toBeVisible();
  await fixturesFor(page);
  await page.getByRole("button", { name: "Change an upstream answer" }).click();
  await expect(page.getByTestId("lab-veil-0")).toHaveAttribute("data-veil", "stale");
  await expect(page.getByTestId("lab-veil-3")).toHaveAttribute("data-veil", "fresh", {
    timeout: 5_000,
  });
  await page.getByTestId("reduced-motion").click();
  await expect(page.getByTestId("reduced-motion")).toHaveAttribute("aria-checked", "true");
  await shoot(page, "lab-light");
});
