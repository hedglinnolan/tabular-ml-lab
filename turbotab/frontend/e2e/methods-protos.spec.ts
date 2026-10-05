/**
 * The three living-methods prototypes (BLUEPRINT §11.4) on their one shared scenario
 * (src/explore/methods-shared/SCENARIO.md), in mock mode with no server:
 *
 *   npm run test:e2e -- methods-protos
 *
 * Each is walked by clicks alone, from its first draft (after its reset) to the locked Table 2 and
 * "Which of my decisions mattered?", and reset again; then the numbers each shows are compared:
 * the three must print the same Table 2 and the same declared alternatives, and those must be the
 * engine's (the scenario's Model 2 estimate). The chooser at /lab/methods opens each.
 */
import { expect, test, type Page } from "@playwright/test";
import { walker as paper } from "./methods-protos/document";
import { walker as map } from "./methods-protos/map";
import { walker as questlog } from "./methods-protos/questlog";
import type { Walker } from "./methods-protos/walker";

const WALKERS: Walker[] = [paper, questlog, map];

type Shown = Record<string, { estimate: string; ci: string }>;
const table2: Record<string, Shown> = {};
const mattered: Record<string, Shown> = {};

test.describe.configure({ mode: "serial" });

async function read(page: Page, testid: string, row: string, est: string, ci: string): Promise<Shown> {
  const box = page.getByTestId(testid).first();
  await expect(box).toBeVisible();
  const rows = box.locator(`[${row}]`);
  await expect(rows.first()).toBeVisible();
  const out: Shown = {};
  for (const r of await rows.all()) {
    const key = (await r.getAttribute(row))!;
    const estimate = (await r.getAttribute(est))!;
    // The number the row carries for the check is the number it prints.
    await expect(r).toContainText(estimate);
    out[key] = { estimate, ci: (await r.getAttribute(ci))! };
  }
  return out;
}

async function start(page: Page, w: Walker) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(w.path);
  const reset = page.getByTestId("proto-reset");
  await expect(reset).toBeVisible();
  await reset.click();
  await expect(page.getByTestId("table2")).toHaveCount(0);
}

for (const w of WALKERS) {
  test(`${w.name}: walked by clicks from the first draft to Table 2 and what mattered`, async ({ page }) => {
    await start(page, w);
    await w.toTable2(page);
    table2[w.name] = await read(page, "table2", "data-t2-row", "data-t2-estimate", "data-t2-ci");
    await w.toMattered(page);
    mattered[w.name] = await read(page, "mattered", "data-mattered-row", "data-estimate", "data-ci");
    // Reset returns to the first draft: no estimate is on screen.
    await page.getByTestId("proto-reset").click();
    await expect(page.getByTestId("table2")).toHaveCount(0);
    await expect(page.getByTestId("mattered")).toHaveCount(0);
  });
}

test("the three prototypes show the same Table 2 and the same alternatives", () => {
  const [first, ...rest] = WALKERS.map((w) => w.name);
  expect(Object.keys(table2)).toHaveLength(WALKERS.length);
  const t2 = table2[first!]!;
  expect(Object.keys(t2)).toEqual(["crude", "model_1", "model_2", "model_3"]);
  // The scenario's primary estimate, as the engine served it (SCENARIO.md).
  expect(t2.model_2).toEqual({ estimate: "−0.0199", ci: "−0.0327 to −0.00718" });
  for (const name of rest) {
    expect(table2[name], `${name}'s Table 2`).toEqual(t2);
    expect(mattered[name], `${name}'s declared alternatives`).toEqual(mattered[first!]);
  }
  expect(Object.keys(mattered[first!]!).length).toBeGreaterThanOrEqual(6);
});

test("the chooser opens each prototype", async ({ page }) => {
  await page.goto("/lab/methods");
  for (const [i, w] of WALKERS.entries()) {
    await page.goto("/lab/methods");
    const card = page.getByTestId("proto-card").nth(i);
    await expect(card).toBeVisible();
    await card.getByRole("link", { name: /open/i }).click();
    await expect(page).toHaveURL(new RegExp(`${w.path}$`));
    await expect(page.getByTestId("proto-reset")).toBeVisible();
  }
});
