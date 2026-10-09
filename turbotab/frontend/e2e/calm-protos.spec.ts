/**
 * The four calm structures (src/explore/calm-*) on the one kit and the one scenario, in the static
 * entry (calm.html, hash routes; no server, no mock worker):
 *
 *   npm run test:e2e -- calm-protos
 *
 * Each structure is walked by clicks alone, from its first draft (after its reset) to the locked
 * Table 2 and "Which of my decisions mattered?", and reset again; then the four must print the same
 * Table 2, the same interval note (Model 2's degrees of freedom) and the same alternatives, and
 * those must be the engine's (SCENARIO.md). The kit demo (#/kit) is checked in light and dark at
 * 1440 and 390 px: no sideways scroll, every layout and the canvas at rest drawn, nothing fetched from /api/.
 */
import { expect, test, type Page } from "@playwright/test";
import { walker as map } from "./calm-protos/map";
import { walker as paper } from "./calm-protos/paper";
import { walker as qa } from "./calm-protos/qa";
import { walker as quest } from "./calm-protos/quest";
import { watchApi } from "./calm-protos/scenario";
import type { Walker } from "./calm-protos/walker";

const WALKERS: Walker[] = [qa, paper, quest, map];

type Shown = Record<string, { estimate: string; ci: string }>;
const table2: Record<string, Shown> = {};
const mattered: Record<string, Shown> = {};
const footnote: Record<string, string> = {};

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
    const note = await page.getByTestId("t2-inference").first().innerText();
    footnote[w.name] = /\bt\(([\d,]+)\)/.exec(note)?.[1] ?? note;
    await w.toMattered(page);
    mattered[w.name] = await read(page, "mattered", "data-mattered-row", "data-estimate", "data-ci");
    // Reset returns to the first draft: no estimate is on screen.
    await page.getByTestId("proto-reset").click();
    await expect(page.getByTestId("table2")).toHaveCount(0);
    await expect(page.getByTestId("mattered")).toHaveCount(0);
  });
}

test("the four structures show the same Table 2 and the same alternatives", () => {
  const [first, ...rest] = WALKERS.map((w) => w.name);
  expect(Object.keys(table2)).toHaveLength(WALKERS.length);
  const t2 = table2[first!]!;
  expect(Object.keys(t2)).toEqual(["crude", "model_1", "model_2", "model_3"]);
  // The scenario's primary estimate, as the engine served it (SCENARIO.md).
  expect(t2.model_2).toEqual({ estimate: "−0.0199", ci: "−0.0327 to −0.00718" });
  expect(footnote[first!], `${first}'s interval note`).toBe("21,830");
  for (const name of rest) {
    expect(table2[name], `${name}'s Table 2`).toEqual(t2);
    expect(footnote[name], `${name}'s interval note`).toBe(footnote[first!]);
    expect(mattered[name], `${name}'s declared alternatives`).toEqual(mattered[first!]);
  }
  expect(Object.keys(mattered[first!]!).length).toBeGreaterThanOrEqual(6);
});

test("the chooser opens each built structure and says which are not built yet", async ({ page }) => {
  for (const w of WALKERS) {
    await page.goto("/calm.html#/");
    const id = w.path.split("#/")[1]!;
    const card = page.getByTestId(`structure-${id}`);
    if ((await card.getAttribute("aria-disabled")) === "true") {
      // not built: no link, says so; its route still opens the shared reference walk
      await expect(card).toContainText("Not built yet");
      await page.goto(w.path);
    } else {
      await card.click();
      await expect(page).toHaveURL(new RegExp(`#/${id}$`));
    }
    await expect(page.getByTestId("proto-reset")).toBeVisible();
  }
});

for (const theme of ["light", "dark"] as const) {
  for (const width of [1440, 390]) {
    test(`the kit demo in ${theme} at ${width} px: every part and layout, no sideways scroll, nothing fetched`, async ({ page }) => {
      const api = watchApi(page);
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ colorScheme: theme, reducedMotion: "reduce" });
      await page.goto("/calm.html#/kit");
      await expect(page.getByTestId("kit")).toBeVisible();
      for (const layout of ["focus", "strip", "flow", "routing", "angles", "none", "refused", "rest"])
        await expect(page.getByTestId(`demo-${layout}`).first(), layout).toBeVisible();
      await expect(page.getByTestId("table2")).toBeVisible();
      await expect(page.getByTestId("mattered")).toBeVisible();
      const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      expect(bg, "the page's ground").toBe(theme === "dark" ? "rgb(20, 17, 14)" : "rgb(249, 245, 240)");
      expect(await page.evaluate(() => document.documentElement.scrollWidth), "the page's width").toBeLessThanOrEqual(width);
      expect(api, "requests to /api/").toEqual([]);
    });
  }
}
