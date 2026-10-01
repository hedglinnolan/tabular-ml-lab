/**
 * The stage (M1 part 2, M1_CONTRACT §10–§13), against the mock's NHANES project at /lab/stage.
 *
 *   npm run test:e2e -- m1-stage        (Playwright starts the mock dev server)
 *
 * Previews an energy-adjustment option and watches the residual storyboard play on the flip;
 * steps through options with the flip held; saves a figure as SVG and PNG; opens a finding's
 * evidence and the banner's segments; adds the refit band to the substitution curves; changes
 * the energy method after fitting and watches the Results re-flow. Writes, for review:
 *   docs/turbotab-next/m1/screens/stage-flip-<ms>ms.png   the residual flip at 0 … 900 ms
 *   docs/turbotab-next/m1/screens/stage-<scene>-<theme>.png   preview, evidence, Results
 */
import { mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCREENS = resolve(HERE, "../../../docs/turbotab-next/m1/screens");
const MAX_SCREEN_BYTES = 300 * 1024;

interface StageHook {
  manual: (v: boolean) => void;
  advance: (ms: number) => void;
  state: () => { pos: number; target: number; last: number; paused: boolean };
}
declare global {
  interface Window {
    __turbotabStage?: StageHook;
  }
}

async function shot(page: Page, name: string, target?: ReturnType<Page["locator"]>) {
  mkdirSync(SCREENS, { recursive: true });
  const path = resolve(SCREENS, `${name}.png`);
  const box = target ? await target.boundingBox() : null;
  if (box) {
    // A little margin, so nothing flush with the stage's edge is cut by the crop.
    const m = 10;
    await page.screenshot({ path, clip: { x: box.x - m, y: box.y - m, width: box.width + 2 * m, height: box.height + 2 * m } });
  } else await page.screenshot({ path });
  expect(statSync(path).size, `${name}.png stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
}

async function open(page: Page, theme: "light" | "dark") {
  await page.emulateMedia({ colorScheme: theme, reducedMotion: "no-preference" });
  await page.goto("/lab/stage");
  await expect(page.getByTestId("stage")).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
}

/** Wait until the player has landed (not moving) on a side. */
async function landed(page: Page, side: "now" | "with") {
  const player = page.getByTestId("player");
  await expect(player).toHaveAttribute("data-side", side);
  await expect(player).not.toHaveAttribute("data-moving", "true");
}

async function previewOption(page: Page, question: string, option: string) {
  await page.locator(`[data-question=${question}]`).click();
  await page.locator(`[data-option=${option}]`).focus();
}

test.describe.configure({ mode: "serial" });

test("a preview plays its storyboard on the flip, holds the flip across options, and saves real states", async ({
  page,
}) => {
  await open(page, "light");
  await previewOption(page, "energy_adjustment", "energy-0");
  await expect(page.getByTestId("stage-title")).toHaveText("Willett residual model");
  await expect(page.getByTestId("stage-pill")).toHaveText("Preview");
  // The first preview arrives at "your data now" and plays forward on its own.
  await landed(page, "with");
  await expect(page.locator("[aria-label^='Step ']")).toHaveCount(4);
  await expect(page.getByTestId("readout")).toContainText("0.84");
  await expect(page.getByTestId("readout")).toContainText("0.00");
  await expect(page.getByTestId("r-badge")).toContainText("0.00");

  // Space flips back, and the storyboard plays in reverse.
  await page.keyboard.press("Space");
  await landed(page, "now");
  await expect(page.getByTestId("r-badge")).toContainText("0.84");
  await page.keyboard.press("Space");
  await landed(page, "with");

  // Arrow keys move between options; the flip holds its side and nothing replays.
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("stage-title")).toHaveText("Nutrient density alone");
  const s = await page.evaluate(() => window.__turbotabStage!.state());
  expect(s).toMatchObject({ pos: s.last, target: s.last });
  await expect(page.getByTestId("player")).toHaveAttribute("data-side", "with");
  await expect(page.getByTestId("r-badge")).toContainText("0.16");
  await page.keyboard.press("ArrowUp");
  await expect(page.getByTestId("stage-title")).toHaveText("Willett residual model");
  await landed(page, "with");

  // A step dot pauses on that step; its label names it.
  await page.locator("[aria-label^='Step 2 of 4']").click();
  await expect(page.getByTestId("step-label")).toHaveText(/Fit each nutrient on energy/);
  await expect(page.getByTestId("r-badge")).toContainText("0.84");

  // Save the primary plot: the before/after pair as SVG, then the current step as PNG (2×).
  const primary = page.locator("[data-primary]");
  await primary.getByTestId("save-button").click();
  await page.getByRole("radio", { name: "Before and after" }).check();
  const [svg] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "SVG" }).click()]);
  expect(svg.suggestedFilename()).toBe("turbotab-fat-total-against-kcal-before-and-after-pair.svg");
  const text = readFileSync(await svg.path(), "utf-8");
  expect(text).toContain("Preview, not recorded: Willett residual model.");
  expect(text).toContain("a  Your data now");
  expect(text).toContain("b  With this choice");
  await expect(primary.getByRole("status")).toContainText("saved turbotab-fat-total");
  await primary.getByTestId("save-button").click();
  await page.getByRole("radio", { name: /This step: Fit each nutrient on energy/ }).check();
  const [png] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PNG 2×" }).click()]);
  expect(png.suggestedFilename()).toMatch(/-step\.png$/);
  const bytes = readFileSync(await png.path());
  expect(bytes.subarray(1, 4).toString()).toBe("PNG");
  // 2×: the PNG is twice the SVG figure's 720-px width.
  expect(bytes.readUInt32BE(16)).toBe(1440);

  // A refused option says why and offers the way out, which previews.
  await page.locator("[data-option=energy-sugar]").focus();
  await expect(page.getByTestId("refusal")).toContainText("Not available");
  await page.getByRole("button", { name: /Partition protein, carb, fat_total instead/ }).click();
  await expect(page.getByTestId("stage-title")).toHaveText("Partition protein, carb, fat_total instead");
  await landed(page, "with");
});

test("the residual flip, frame by frame (0 → 900 ms)", async ({ page }) => {
  await open(page, "light");
  await previewOption(page, "energy_adjustment", "energy-0");
  await landed(page, "with");
  // Step the player's clock by hand so each frame is exactly its time into the flip.
  await page.evaluate(() => window.__turbotabStage!.manual(true));
  await page.keyboard.press("Space");
  await page.evaluate(() => window.__turbotabStage!.advance(2000));
  await page.waitForTimeout(400);
  await page.keyboard.press("Space");
  let t = 0;
  const positions: number[] = [];
  for (const ms of [0, 150, 300, 450, 600, 900]) {
    await page.evaluate((d) => window.__turbotabStage!.advance(d), ms - t);
    t = ms;
    positions.push((await page.evaluate(() => window.__turbotabStage!.state())).pos);
    await page.waitForTimeout(350); // let the lineage and row flow (their own 300 ms) settle
    await shot(page, `stage-flip-${String(ms).padStart(3, "0")}ms`, page.getByTestId("stage"));
  }
  expect(positions[0]).toBe(0);
  expect(positions.at(-1)).toBe(3);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  await page.evaluate(() => window.__turbotabStage!.manual(false));
});

for (const theme of ["light", "dark"] as const) {
  test(`preview, evidence and Results screens (${theme})`, async ({ page }) => {
    await open(page, theme);
    // Nothing focused, fitted: the Results.
    await expect(page.getByTestId("results")).toBeVisible();
    const comparison = page.getByTestId("model-comparison");
    await expect(comparison.locator("[data-family]")).toHaveCount(3);
    await expect(comparison).toContainText("Predicts worse than the outcome's average");
    await expect(page.getByTestId("substitution-curves")).toBeVisible();
    await shot(page, `stage-results-${theme}`);

    await previewOption(page, "energy_adjustment", "energy-0");
    await landed(page, "with");
    await page.waitForTimeout(350);
    await shot(page, `stage-preview-${theme}`);

    await previewOption(page, "exclusions", "ex-0");
    await landed(page, "with");
    await page.waitForTimeout(350);
    await shot(page, `stage-preview-exclusions-${theme}`);

    await page.locator('[data-finding="pack::dietary::implausible_intake"]').click();
    await expect(page.getByTestId("stage-pill")).toHaveText("Evidence");
    await expect(page.getByTestId("stage")).toContainText("Not drawn:");
    await page.waitForTimeout(400);
    await shot(page, `stage-evidence-${theme}`);
  });
}

test("the banner's segments, the band, and the Results re-flowing after a changed answer", async ({ page }) => {
  await open(page, "light");
  await page.locator("[data-segment=rows]").click();
  await expect(page.getByTestId("stage")).toHaveAttribute("data-scene", "rows");
  await expect(page.locator("[data-step=complete_cases]")).toContainText("2,864");
  await page.locator("[data-segment=columns]").click();
  await expect(page.getByTestId("stage")).toHaveAttribute("data-scene", "columns");
  await page.waitForTimeout(400);
  await shot(page, "stage-columns-light");
  await page.locator("[data-segment=models]").click();
  await expect(page.getByTestId("shelf")).toContainText("Boosted trees");

  // The refit band: asked for, computed with progress, drawn.
  await page.locator("[data-segment=result]").click();
  const band = page.getByTestId("add-band");
  await expect(band).toHaveText(/Add an uncertainty band \(about 40 s\)/);
  await band.click();
  await expect(page.getByText(/Refitting each family 40 times/)).toBeVisible();
  await expect(page.getByText(/Bands: 95% intervals from 40 refits/)).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("substitution-curves").scrollIntoViewIfNeeded();
  await shot(page, "stage-results-band-light");

  // Another pair from the navigator records set_substitution.
  await page.getByRole("button", { name: "Move energy from carb to fat_total" }).click();
  await expect(page.getByTestId("results")).toContainText("Recorded: carb → fat_total.");

  // Change the energy method after fitting: the Results veil, recompute, and return.
  await page.locator('[data-answer="Energy: density"]').click();
  await expect(page.locator("[data-veil=recomputing]").first()).toBeVisible();
  await expect(page.locator("[data-veil=recomputing]")).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByTestId("results")).toContainText("divided by");
});
