/**
 * M1 part 2, the whole journey (M1_CONTRACT §9 and §15) through the Record, the banner and the
 * stage together. Against the real server with the NHANES export:
 *
 *   TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d) venv/bin/python -m turbotab.server --port 8812
 *   E2E_BASE_URL=http://127.0.0.1:8812 npx playwright test m1-journey
 *
 * or against the mock (Playwright starts `npm run dev:mock`; its NHANES-shaped table):
 *
 *   npx playwright test m1-journey
 *
 * The journey: open the export → lens dietary → outcome glucose → (task not asked) → prediction →
 * roles confirmed (SEQN an identifier) → exclusions (every screen previewed with the arrow keys;
 * Willett's sex-specific cut-offs recorded) → missing values (each previewed; the mostly-blank
 * columns left out, then complete cases) → a 20% holdout → energy adjustment (every method
 * previewed; the residual storyboard flipped and captured frame by frame, and saved as a
 * before/after pair) → residual recorded → all three families → the fit → the Results → a
 * fat_total → carb substitution → the refit band → a finding's evidence → then the energy method
 * changed to density after fitting, and the banner, the stage and the Results re-flow.
 *
 * Screens (1440 × 900, each < 300 KB): docs/turbotab-next/m1/screens/real-*.png against the real
 * server (mock-*.png under test-results/ against the mock); the exported figures as saved-*.
 */
import { copyFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const REAL = !!process.env.E2E_BASE_URL;
const NHANES = process.env.E2E_NHANES ?? "/Users/nhedglin/tabular-ml-lab/_tt_tmp_nhanes.csv";
const SCREENS = REAL
  ? resolve(HERE, "../../../docs/turbotab-next/m1/screens")
  : resolve(HERE, "../test-results/m1-journey-screens");
const PREFIX = REAL ? "real" : "mock";
const MAX_SCREEN_BYTES = 300 * 1024;

interface StageHook {
  manual: (v: boolean) => void;
  advance: (ms: number) => void;
  state: () => { pos: number; target: number; last: number };
}
/** The stage's review hook (on under `__turbotabReview`): steps the player's clock by hand. */
type Review = { __turbotabStage: StageHook; __turbotabReview?: boolean };

const stage = (page: Page) => page.getByTestId("stage");
const recordCol = (page: Page) => page.locator('main[aria-label="The record"]');
/** Move the pointer off everything previewable, so no hover lingers in a screenshot. */
const park = (page: Page) => page.mouse.move(1430, 890);

const measurements: Record<string, unknown> = {};
const previewMs: { kind: string; ms: number }[] = [];
const shots: string[] = [];

async function settle(page: Page, ms = 400) {
  await page.waitForTimeout(ms);
}

async function shot(page: Page, name: string, opts: { clip?: Locator; themes?: ("light" | "dark")[] } = {}) {
  mkdirSync(SCREENS, { recursive: true });
  for (const theme of opts.themes ?? ["light"]) {
    await page.emulateMedia({ colorScheme: theme });
    await settle(page, 250);
    const path = resolve(SCREENS, `${PREFIX}-${name}${opts.themes ? `-${theme}` : ""}.png`);
    const box = opts.clip ? await opts.clip.boundingBox() : null;
    await page.screenshot(box ? { path, clip: box } : { path });
    expect(statSync(path).size, `${name} (${theme}) stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
    shots.push(path);
  }
  await page.emulateMedia({ colorScheme: "light" });
}

/** The stage shows the focused option's own answer (not the last one's) and has settled. */
async function previewed(page: Page, title: string | RegExp) {
  await expect(page.getByTestId("stage-title")).toHaveText(title, { timeout: 15_000 });
  await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
}

/** The player has landed on a side and stopped moving. */
async function landed(page: Page, side: "now" | "with") {
  const player = page.getByTestId("player");
  await expect(player).toHaveAttribute("data-side", side, { timeout: 10_000 });
  await expect(player).not.toHaveAttribute("data-moving", "true", { timeout: 10_000 });
}

async function openNhanes(page: Page) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  if (REAL) {
    await page.getByPlaceholder("/path/to/table.csv").fill(NHANES);
    await page.getByRole("button", { name: "Open", exact: true }).click();
  } else {
    await page.getByRole("link", { name: /nhanes diet glucose/ }).first().click();
  }
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 60_000 });
}

/** The design's concerns under the lineage, then a changed earlier answer dates a sentence's counts. */
async function concernsAndDatedCounts(page: Page) {
  await page.getByTestId("banner-columns").click();
  await expect(stage(page).locator("[data-card=lineage]").getByTestId("design-warnings")).toContainText(
    "parts of fat_total",
  );
  await park(page);
  await settle(page, 400);
  await shot(page, "columns-concerns");
  await page.getByTestId("banner-columns").click();

  // An earlier answer changes: a sentence whose counts predate it says so.
  await page.getByRole("button", { name: "Change the exclusions" }).click();
  await page.getByTestId("option-none").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-exclusions")).toContainText("No rows were excluded");
  await expect(page.getByTestId("decision-missing-note")).toContainText("counts were made before");
  await page.getByTestId("decision-missing").scrollIntoViewIfNeeded();
  await park(page);
  await settle(page, 400);
  await shot(page, "record-counts-dated");
}

/** Seconds until `done` holds, polled. */
async function timed(done: () => Promise<void>): Promise<number> {
  const t0 = Date.now();
  await done();
  return Math.round((Date.now() - t0) / 100) / 10;
}

test("the M1 journey: every question previewed on the stage, fitted, then re-flowed", async ({ page }) => {
  test.setTimeout(REAL ? 600_000 : 240_000);
  await page.addInitScript(() => {
    (window as unknown as Review).__turbotabReview = true;
  });
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "no-preference" });
  const problems: string[] = [];
  page.on("console", (m) => {
    // The browser logs every non-2xx response; a refusal (409) is an answer the stage shows.
    if (m.type() === "error" && !m.text().includes("status of 409")) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    // A refusal (409) is an answer the stage shows, not a failure.
    if (r.status() >= 400 && r.status() !== 409) problems.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });
  page.on("requestfinished", (req) => {
    if (req.method() !== "POST" || !req.url().endsWith("/preview")) return;
    const t = req.timing();
    let kind = "?";
    try {
      kind = (JSON.parse(req.postData() ?? "{}") as { kind?: string }).kind ?? "?";
    } catch {
      /* not JSON */
    }
    if (t.responseEnd > 0) previewMs.push({ kind, ms: Math.round(t.responseEnd - t.requestStart) });
  });

  // ── open ────────────────────────────────────────────────────────────────────
  measurements.open_s = await timed(() => openNhanes(page));
  await expect(page.getByTestId("banner-rows")).toContainText("21,849");
  await expect(page.getByTestId("banner-rows")).toHaveAttribute("data-now", "true");

  // ── the lens: keyboard focus previews, Space chooses (a multi-select), Enter records ───────
  await page.getByTestId("option-dietary").focus();
  await previewed(page, "Dietary intake");
  await expect(stage(page)).toHaveAttribute("data-focus", "option");
  // The lens shows what it unlocks on this table (review: the stage was empty for lens, outcome
  // and purpose): the columns as it reads them, and the questions it adds.
  if (REAL) {
    await expect(stage(page).locator("[data-card=lineage]")).toContainText("kcal");
    await expect(page.getByTestId("stage-note")).toContainText("energy adjustment");
  }
  // The shown option of a multi-select is not a chosen one, and the stage records what Enter would.
  await expect(page.getByTestId("option-dietary")).toHaveAttribute("aria-selected", "false");
  await expect(page.getByTestId("stage-record")).toHaveText("Record the dietary intake lens");
  await park(page);
  await shot(page, "q-lens", { themes: ["light", "dark"] });
  await page.keyboard.press("Space");
  await expect(page.getByTestId("option-dietary")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-lens")).toContainText("dietary");

  // ── the outcome ─────────────────────────────────────────────────────────────
  await page.getByRole("combobox").fill("glucose");
  await page.getByRole("combobox").press("Enter");
  await previewed(page, /glucose/);
  if (REAL) await expect(stage(page).locator("[data-card=row_flow]")).toContainText("recorded");
  await shot(page, "q-target");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText("glucose");

  // ── the task is detected with high confidence: not asked, and said so ───────────────────
  await expect(page.getByTestId("skip-task")).toContainText("Not asked", { timeout: 30_000 });
  await expect(page.getByTestId("skip-task")).toContainText("regression");

  // ── the purpose ─────────────────────────────────────────────────────────────
  await page.getByTestId("option-prediction").focus();
  await previewed(page, "Prediction");
  if (REAL) {
    await expect(page.getByTestId("stage-note")).toContainText("never saw");
    await expect(stage(page).locator("[data-card=rows]").last()).toBeVisible(); // your data now stays in view
  }
  await shot(page, "q-purpose");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-purpose")).toContainText("prediction");

  // ── the roles: confirm the proposal; SEQN is an identifier, nested nutrients and flags shown ─
  const roles = page.getByTestId("question-roles");
  await expect(roles).toBeVisible({ timeout: 60_000 });
  await expect(roles.locator('[data-role="identifier"]')).toContainText("SEQN");
  await expect(page.getByTestId("role-chip-sugar")).toContainText("⊂ carb");
  await expect(page.getByTestId("role-chip-fat_sat")).toContainText("⊂ fat_total");
  await expect(page.getByTestId("role-chip-imputed_bmi")).toContainText("→ bmi");
  await page.getByTestId("record-roles").focus();
  await previewed(page, "These roles");
  await expect(stage(page).locator("[data-card=lineage]").first()).toContainText("SEQN");
  await park(page);
  await roles.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await shot(page, "q-roles", { themes: ["light", "dark"] });
  await page.getByTestId("record-roles").click();
  await expect(page.getByTestId("decision-roles")).toBeVisible();

  // ── exclusions: every screen previewed with the arrow keys; Willett's sex-specific recorded ─
  const exclusions = page.getByTestId("question-exclusions");
  await expect(exclusions).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("banner-rows")).toHaveAttribute("data-now", "true");
  // Every count names its denominator: the screens' rows among those with the outcome recorded.
  if (REAL) await expect(page.getByTestId("option-willett_by_sex")).toContainText(/−[\d,]+ of 21,849/);
  await page.getByTestId("option-none").focus();
  await previewed(page, "Keep every row");
  for (const [key, name] of [
    ["willett_by_sex", "q-exclusions-willett"],
    ["sex_neutral_500_5000", "q-exclusions-500-5000"],
    ["sex_neutral_500_3500", "q-exclusions-500-3500"],
  ] as const) {
    await page.keyboard.press("ArrowDown");
    await expect(page.getByTestId(`option-${key}`)).toBeFocused();
    await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
    await landed(page, "with");
    await expect(stage(page).locator("[data-card=row_flow]")).toBeVisible();
    await park(page);
    await exclusions.evaluate((el) => el.scrollIntoView({ block: "start" }));
    await shot(page, name, key === "willett_by_sex" ? { themes: ["light", "dark"] } : {});
  }
  await page.getByTestId("option-willett_by_sex").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-exclusions")).toContainText("Willett");
  await expect(page.getByTestId("banner-rows-1")).toHaveText(/\d/, { timeout: 30_000 });

  // ── missing values: the mostly-blank yes/no columns are offered to be left out first ───────
  const missing = page.getByTestId("question-missing");
  await expect(missing).toBeVisible({ timeout: 60_000 });
  const leaveOut = page.getByTestId("option-leave_out");
  await expect(leaveOut).toContainText("meds_chol");
  await expect(leaveOut).toContainText("meds_hbp");
  await leaveOut.focus();
  await previewed(page, /Leave out/);
  await landed(page, "with");
  await park(page);
  await missing.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await shot(page, "q-missing-leave-out", { themes: ["light", "dark"] });
  await page.keyboard.press("ArrowDown");
  await previewed(page, "Complete cases");
  await landed(page, "with");
  await shot(page, "q-missing-complete-cases");
  await page.keyboard.press("ArrowDown");
  await previewed(page, "Impute");
  await landed(page, "with");
  // Imputing a "not asked" column says what it would write, with the lever beside it.
  const caution = page.getByTestId("stage-caution");
  if (REAL) {
    await expect(caution).toContainText("meds_hbp");
    await expect(caution).toContainText("True");
    await expect(caution).toContainText("not asked");
  }
  await shot(page, "q-missing-impute");
  if (REAL) {
    await caution.getByRole("button", { name: "Leave them out first" }).click();
    await previewed(page, /Leave out meds_chol and meds_hbp/); // the same option the Record offers
  }
  await leaveOut.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-missing")).toContainText("meds_chol");

  // ── the split ───────────────────────────────────────────────────────────────
  const split = page.getByTestId("question-split");
  await expect(split).toBeVisible({ timeout: 60_000 });
  await page.getByTestId("option-0.2").focus();
  await previewed(page, /20%/);
  await landed(page, "with");
  await park(page);
  await split.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await shot(page, "q-split");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-split")).toContainText("20%");
  await expect(page.getByTestId("banner-train")).toBeVisible({ timeout: 30_000 });

  // ── energy adjustment: every method previewed with the arrow keys ───────────────────────
  const energy = page.getByTestId("question-energy_adjustment");
  await expect(energy).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("option-residual")).toContainText("usual");
  await expect(page.getByTestId("option-residual")).toHaveAttribute("aria-selected", "false");
  if (REAL) {
    // Every adjusted nutrient is named — sugar too, carbohydrate at 4 kcal/g.
    await expect(energy).toContainText("protein, sugar, carb, fat_total, fat_sat, fat_mon, and fat_poly are adjusted together");
    // A total beside its parts would count fat's energy twice: partition is refused, with the reason.
    await expect(page.getByTestId("option-partition")).toContainText("not applicable");
    await expect(page.getByTestId("option-partition")).toContainText("parts of");
  }
  const methods = await energy
    .getByRole("option")
    .evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  measurements.energy_methods = methods;
  await page.getByTestId(`option-${methods[0]}`).focus();
  const methodMs: Record<string, number> = {};
  for (let i = 0; i < methods.length; i++) {
    const m = methods[i]!;
    if (i > 0) await page.keyboard.press("ArrowDown");
    await expect(page.getByTestId(`option-${m}`)).toBeFocused();
    const t0 = Date.now();
    await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
    await expect(stage(page)).toHaveAttribute("data-group", /set_energy_adjustment/);
    methodMs[m] = Date.now() - t0;
    if (REAL && m === "partition") {
      await expect(page.getByTestId("refusal")).toContainText("count");
      await expect(page.getByTestId("refusal-exit").first()).toHaveText("Partition protein, carb and fat_total");
    }
    await settle(page, 700);
    await park(page);
    await energy.evaluate((el) => el.scrollIntoView({ block: "start" }));
    await shot(page, `q-energy-${m.replace(/_/g, "-")}`, m === "residual" ? { themes: ["light", "dark"] } : {});
  }
  measurements.energy_preview_on_screen_ms = methodMs;

  // ── the residual storyboard on the flip, frame by frame ──────────────────────────────────
  await page.getByTestId("option-residual").focus();
  await previewed(page, "Residual method");
  await landed(page, "with");
  // The basis names the rows the view was computed on: a sample of the training rows.
  if (REAL) await expect(stage(page)).toContainText(/Values on a sample of 5,000 of the [\d,]+ training rows/);
  await expect(page.locator("[aria-label^='Step ']")).toHaveCount(4);
  await expect(page.getByTestId("readout")).toContainText("0.00");
  await park(page);
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.manual(true));
  await page.keyboard.press("Space"); // back to your data now, instantly under the manual clock
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.advance(2000));
  await settle(page, 400);
  await page.keyboard.press("Space"); // and forward: the storyboard plays
  let t = 0;
  const positions: number[] = [];
  for (const ms of [0, 150, 300, 450, 600, 900]) {
    await page.evaluate((d) => (window as unknown as Review).__turbotabStage.advance(d), ms - t);
    t = ms;
    positions.push((await page.evaluate(() => (window as unknown as Review).__turbotabStage.state())).pos);
    await settle(page, 350); // the lineage and the row flow ease over their own 300 ms
    await shot(page, `flip-residual-${String(ms).padStart(3, "0")}ms`);
  }
  measurements.flip_positions = positions;
  expect(positions[0]).toBe(0);
  expect(positions.at(-1)).toBe(3);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.manual(false));
  await landed(page, "with");
  // A step dot pauses on that step and names it.
  await page.locator("[aria-label^='Step 2 of 4']").click();
  await expect(page.getByTestId("step-label")).toHaveText(/Fit/);
  await page.locator("[aria-label^='Step 4 of 4']").click();
  await landed(page, "with");

  // ── save the primary plot as a before/after pair, SVG and PNG ───────────────────────────
  const primary = stage(page).locator("[data-primary]");
  await primary.getByTestId("save-button").click();
  await page.getByRole("radio", { name: "Before and after" }).check();
  await shot(page, "save-pair");
  const [svg] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "SVG" }).click()]);
  const svgText = readFileSync(await svg.path(), "utf-8");
  expect(svgText).toContain("Preview, not recorded:");
  expect(svgText).toContain("Your data now");
  expect(svgText).toContain("With this choice");
  await primary.getByTestId("save-button").click();
  await page.getByRole("radio", { name: "Before and after" }).check();
  const [png] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PNG 2×" }).click()]);
  const pngBytes = readFileSync(await png.path());
  expect(pngBytes.subarray(1, 4).toString()).toBe("PNG");
  if (REAL) {
    copyFileSync(await svg.path(), resolve(SCREENS, "saved-residual-pair.svg"));
    copyFileSync(await png.path(), resolve(SCREENS, "saved-residual-pair.png"));
  }
  measurements.saved = { svg: svg.suggestedFilename(), png: png.suggestedFilename(), png_width: pngBytes.readUInt32BE(16) };

  // A term's card opens in full, inside the window, and covers no option (review: it was clipped
  // and sat on the Residual option).
  const term = energy.locator("[data-term]").first();
  if (await term.count()) {
    await term.hover();
    const tip = page.getByRole("tooltip").filter({ visible: true });
    await expect(tip).toHaveCount(1);
    const tipBox = (await tip.boundingBox())!;
    expect(tipBox.x).toBeGreaterThanOrEqual(0);
    expect(tipBox.x + tipBox.width).toBeLessThanOrEqual(1440);
    for (const opt of await energy.getByRole("option").all()) {
      const o = (await opt.boundingBox())!;
      const overlaps =
        tipBox.x < o.x + o.width && o.x < tipBox.x + tipBox.width && tipBox.y < o.y + o.height && o.y < tipBox.y + tipBox.height;
      expect(overlaps, "the term card covers an option").toBe(false);
    }
    await park(page);
  }

  // Record the residual method from the Record (Enter on its option).
  await page.getByTestId("option-residual").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-energy_adjustment")).toContainText("residual");

  // ── the model families: all three, with the keyboard ─────────────────────────────────────
  const models = page.getByTestId("question-models");
  await expect(models).toBeVisible({ timeout: 60_000 });
  const families = await models.getByRole("option").evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  expect(families).toHaveLength(3);
  await page.getByTestId(`option-${families[0]}`).focus();
  // Never the previous question's preview as a placeholder (review: "PREVIEW Residual method"
  // flashed for ~250 ms): sample the stage's title while the first family's preview loads.
  const titles: string[] = [];
  for (let i = 0; i < 20; i++) {
    titles.push((await page.getByTestId("stage-title").textContent().catch(() => "")) ?? "");
    await page.waitForTimeout(25);
  }
  expect(titles.filter((x) => /Residual/.test(x)), "no foreign preview while loading").toEqual([]);
  await previewed(page, /./);
  // Nothing chosen yet: the stage's button (and Enter) fit the shown family.
  await expect(page.getByTestId("stage-record")).toHaveText(/^Fit the /);
  await park(page);
  await models.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await shot(page, "q-models");
  for (let i = 0; i < families.length; i++) {
    if (i > 0) await page.keyboard.press("ArrowDown");
    await page.keyboard.press("Space");
  }
  await expect(page.getByTestId("record-models")).toHaveText("Fit these 3 families");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-models")).toContainText("Three model families");

  // ── the fit, then the Results (the stage's live view once fitted) ───────────────────────
  measurements.fit_s = await timed(() =>
    expect(page.getByTestId("banner-result-value")).toBeVisible({ timeout: 300_000 }),
  );
  await park(page);
  await expect(page.getByTestId("results")).toBeVisible({ timeout: 30_000 });
  const comparison = page.getByTestId("model-comparison");
  await expect(comparison.locator("[data-family]")).toHaveCount(3);
  await expect(comparison).toContainText("baseline");
  measurements.result = await page.getByTestId("banner-result").innerText().catch(() => null);
  // Findings learn they were answered: the energy finding no longer pushes its lever.
  await expect(page.getByTestId("findings-settled")).toContainText("answered in the record");
  await expect(page.getByTestId("finding-pack::dietary::energy_adjustment")).toHaveCount(0);
  await settle(page, 600);
  await shot(page, "results", { themes: ["light", "dark"] });

  // ── the substitution: fat_total → carb from the stage's matrix ──────────────────────────
  const substitution = page.getByTestId("question-substitution");
  await expect(substitution).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Move energy from fat_total to carb" }).click();
  measurements.substitution_s = await timed(() =>
    expect(page.getByTestId("substitution-curves")).toBeVisible({ timeout: 120_000 }),
  );
  await expect(page.getByTestId("decision-substitution")).toContainText("fat_total");
  const curves = page.getByTestId("substitution-curves");
  await expect(stage(page)).toContainText("per 100 kcal at k = 100");
  // §12.5: moving fat_total moves its parts in proportion, and the Results say so.
  if (REAL) await expect(page.getByTestId("carried")).toContainText("fat_sat");
  await curves.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await settle(page, 500);
  await shot(page, "curves");

  // ── the refit band: offered with its measured time, computed with progress, drawn ─────────
  const band = page.getByTestId("add-band");
  const bandLabel = (await band.textContent()) ?? "";
  measurements.band_offer = bandLabel;
  expect(bandLabel).toMatch(/Add an uncertainty band \(about .+\)/);
  await band.click();
  await expect(page.getByText(/Refitting each family \d+ times/)).toBeVisible({ timeout: 15_000 });
  // Stop it: the curve veils as stopped with a way to run it again outside the veil, the
  // Record says so beside the answer, and the band can be asked for again (review blocker).
  if (REAL) {
    await page.getByRole("button", { name: "Stop", exact: true }).click();
    await expect(page.getByTestId("retry-substitution").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("sentence-stopped-substitution")).toBeVisible();
    await park(page);
    await shot(page, "band-stopped");
    await page.getByTestId("add-band").click();
    await expect(page.getByText(/Refitting each family \d+ times/)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("sentence-stopped-substitution")).toHaveCount(0);
  }
  measurements.band_s = await timed(() =>
    expect(page.getByText(/bands: 95% intervals from \d+ refits/i)).toBeVisible({ timeout: 300_000 }),
  );
  await expect(page.getByTestId("decision-substitution")).toContainText("refits");
  await page.getByTestId("substitution-curves").evaluate((el) => el.scrollIntoView({ block: "start" }));
  await settle(page, 500);
  await shot(page, "band", { themes: ["light", "dark"] });

  // ── a finding's evidence: focusing its card puts the evidence on the stage ───────────────
  // The energy finding is settled by the energy answer: its evidence is a press away there.
  await page.getByTestId("findings-settled-toggle").click();
  const preferred = page.getByTestId("settled-pack::dietary::energy_adjustment");
  const card = (await preferred.count()) ? preferred : page.getByTestId("finding-cards").locator(":scope > li").first();
  await card.scrollIntoViewIfNeeded();
  await card.focus();
  await expect(stage(page)).toHaveAttribute("data-focus", "finding");
  await expect(page.getByTestId("stage-pill")).toHaveText("Evidence", { timeout: 15_000 });
  await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
  // Each r names its basis: the finding's every row, the evidence's training sample.
  if (REAL) {
    await expect(page.getByTestId("stage-title")).toContainText("across all 21,849 rows");
    await expect(stage(page)).toContainText(/Values on a sample of 5,000 of the [\d,]+ training rows/);
  }
  await park(page);
  await settle(page, 500);
  await shot(page, "evidence", { themes: ["light", "dark"] });
  await page.keyboard.press("Escape");
  await expect(stage(page)).toHaveAttribute("data-focus", "live");

  // ── change the energy method after fitting: the banner, the stage and the Results re-flow ──
  const banner = page.getByTestId("banner");
  await recordCol(page).evaluate((el) => (el.scrollTop = 0));
  await park(page);
  await expect(page.getByTestId("banner-columns")).toContainText("residual");
  await shot(page, "banner-before", { clip: banner });
  await shot(page, "reflow-before");
  await page.getByRole("button", { name: "Change the energy adjustment" }).click();
  await page.getByTestId("option-density").focus();
  await previewed(page, "Density alone");
  await landed(page, "with");
  // "Your data now" is the recorded residual result (r 0.00), not the raw nutrient.
  if (REAL) {
    await expect(page.getByTestId("readout")).toContainText("0.00");
    await expect(stage(page)).toContainText("Recorded now");
  }
  await park(page);
  await shot(page, "q-energy-change-density");
  // Every veil change from here on, timed: the order the change propagates in.
  await page.evaluate(() => {
    const w = window as unknown as { __veils: { t: number; at: string; veil: string | null }[] };
    w.__veils = [];
    const t0 = performance.now();
    new MutationObserver((muts) => {
      for (const m of muts) {
        const el = m.target as HTMLElement;
        const at =
          el.closest("[data-segment]")?.getAttribute("data-segment") ??
          el.closest("[data-testid]")?.getAttribute("data-testid") ??
          el.getAttribute("aria-label") ??
          "?";
        w.__veils.push({ t: Math.round(performance.now() - t0), at, veil: el.getAttribute("data-veil") });
      }
    }).observe(document.body, { subtree: true, attributes: true, attributeFilter: ["data-veil"] });
  });
  const t0 = Date.now();
  await page.getByTestId("option-density").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-energy_adjustment")).toContainText("density");
  await park(page);
  // Three frames while the change propagates, at real speed (no throttling): a quarter second
  // after the record (the veils are in), then about one and two seconds in (the refit lands).
  const frames: { at_ms: number; veils: Record<string, string | null> }[] = [];
  const veilsNow = () =>
    page.evaluate(() => {
      const out: Record<string, string | null> = {};
      for (const seg of Array.from(document.querySelectorAll("[data-segment]"))) {
        out[seg.getAttribute("data-segment")!] = seg.querySelector("[data-veil]")?.getAttribute("data-veil") ?? null;
      }
      return out;
    });
  for (const [i, wait] of [[1, 250], [2, 650], [3, 900]] as const) {
    if (wait) await page.waitForTimeout(wait);
    frames.push({ at_ms: Date.now() - t0, veils: await veilsNow() });
    const path = resolve(SCREENS, `${PREFIX}-banner-reflow-${i}.png`);
    await page.screenshot({ path, clip: (await page.getByTestId("banner").boundingBox())! });
    shots.push(path);
    if (i === 2) {
      const during = resolve(SCREENS, `${PREFIX}-reflow-during.png`);
      await page.screenshot({ path: during });
      shots.push(during);
    }
  }
  measurements.reflow_frames = frames;
  const columnsVeil = page.getByTestId("banner-columns").locator("[data-veil]").first();
  const resultVeil = page.getByTestId("banner-result").locator("[data-veil]").first();
  await expect(columnsVeil).toHaveAttribute("data-veil", "fresh", { timeout: 120_000 });
  await expect(resultVeil).toHaveAttribute("data-veil", "fresh", { timeout: 120_000 });
  await expect(page.getByTestId("banner-columns")).toContainText("density");
  measurements.reflow_banner_s = Math.round((Date.now() - t0) / 100) / 10;
  await expect(page.locator("[data-veil=recomputing]")).toHaveCount(0, { timeout: 300_000 });
  measurements.reflow_all_s = Math.round((Date.now() - t0) / 100) / 10;
  const log = await page.evaluate(
    () => (window as unknown as { __veils: { t: number; at: string; veil: string | null }[] }).__veils,
  );
  // The order the veils lifted in: the columns before the result, the result before the curves.
  const firstFresh = (at: string) => log.find((e) => e.at === at && e.veil === "fresh")?.t ?? null;
  const firstVeiled = (at: string) => log.find((e) => e.at === at && e.veil !== "fresh")?.t ?? null;
  measurements.reflow_veils = {
    columns: { veiled_ms: firstVeiled("columns"), fresh_ms: firstFresh("columns") },
    models: { veiled_ms: firstVeiled("models"), fresh_ms: firstFresh("models") },
    result: { veiled_ms: firstVeiled("result"), fresh_ms: firstFresh("result") },
    changes: log.length,
  };
  expect(firstVeiled("result"), "the result segment veils while the fit recomputes").not.toBeNull();
  await expect(page.getByTestId("results")).toBeVisible();
  await expect(page.locator('[data-slot="energy_adjustment"]').getByLabel("Earlier answers")).toContainText("residual");
  await park(page);
  await settle(page, 600);
  await shot(page, "banner-after", { clip: banner });
  await shot(page, "reflow-after", { themes: ["light", "dark"] });
  await page.getByTestId("substitution-curves").evaluate((el) => el.scrollIntoView({ block: "start" }));
  await settle(page, 400);
  await shot(page, "reflow-after-curves");

  // ── the design's concerns are stated under the lineage (review: computed, never shown) ─────
  if (REAL) await concernsAndDatedCounts(page);


  // ── what the run measured ──────────────────────────────────────────────────────────────
  const sorted = previewMs.map((p) => p.ms).sort((a, b) => a - b);
  measurements.previews = {
    n: sorted.length,
    p50_ms: sorted[Math.floor(0.5 * (sorted.length - 1))] ?? null,
    p95_ms: sorted[Math.floor(0.95 * (sorted.length - 1))] ?? null,
    max_ms: sorted.at(-1) ?? null,
  };
  measurements.problems = problems;
  measurements.screens = shots.length;
  mkdirSync(resolve(HERE, "../test-results"), { recursive: true });
  writeFileSync(resolve(HERE, `../test-results/m1-journey-${PREFIX}.json`), JSON.stringify(measurements, null, 1));
  console.log(JSON.stringify(measurements, null, 1));
  expect(problems, "no console errors and no failed requests").toEqual([]);
});
