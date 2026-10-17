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
 * the exclusion preview saved as a before/after pair; Willett's sex-specific cut-offs recorded) →
 * missing values (each previewed; the mostly-blank columns left out, then complete cases) → a 20%
 * holdout → (no energy question: under prediction no coefficient is read as an effect, so the
 * Router says it does not apply) → the three linear families → the fit → Fit pressed on the
 * Results (SIZING P0.8: the fit is served withheld until then, and a fit held for the press
 * computes only after it) → the Results → a finding's evidence → then the split changed after
 * fitting, and the banner, the stage and the Results re-flow; then an earlier answer changed
 * again, dating the counts of the sentence after it. No substitution is asked either: it compares
 * two imagined diets.
 *
 * Screens (1440 × 900, each < 300 KB): docs/turbotab-next/m1/screens/real-*.png against the real
 * server (mock-*.png under test-results/ against the mock); the exported figures as saved-*.
 */
import { copyFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { confirmAsk } from "./ask";
import { toClassic } from "./classic";
import { ANY_STAGE, FIT_CAP, FIT_QUIET, fitOnceComputed, visibleWhileComputing, waitWhileComputing } from "./fit";
import { nhanesExport } from "./nhanes";

const HERE = dirname(fileURLToPath(import.meta.url));
const REAL = !!process.env.E2E_BASE_URL;
const NHANES = nhanesExport();
const SCREENS = REAL
  ? resolve(HERE, "../../../docs/turbotab-next/m1/screens")
  : resolve(HERE, "../test-results/m1-journey-screens");
const PREFIX = REAL ? "real" : "mock";
// Willett's sex-specific screen: the server's key, and the M1 mock's older one.
const WILLETT = REAL ? "willett_2013_by_sex" : "willett_by_sex";
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
const oversize: string[] = [];

/**
 * A guarantee the real server's stage breaks today (W1-C, reported for the shell): the M1 mock
 * asserts it inline, as before; on the real server the journey measures it and the known-defect
 * test after it holds it, so it stays visible in every report instead of passing as an
 * annotation: a screen over 300 KB is a fixme that names it. (The two the energy question's
 * preview showed, a term's card covering an option and the distribution card's Save over the
 * primary's, are not measured: prediction does not ask that question.)
 */
const defects: { oversize?: string[] } = {};
const KNOWN = "known defect on the real server's stage (W1-C open issue; the mock asserts it inline)";

// The known-defect tests read what the journey measured, so they run after it, in one worker.
test.describe.configure({ mode: "serial" });

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
    // On the mock a review screen over 300 KB fails the journey, as it always did; on the real
    // server it is listed, and the known-defect test after the journey names it.
    const bytes = statSync(path).size;
    if (!REAL) expect(bytes, `${name} (${theme}) stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
    if (bytes >= MAX_SCREEN_BYTES) {
      oversize.push(`${name} (${theme}): ${Math.round(bytes / 1024)} KB`);
      test.info().annotations.push({ type: "screen over 300 KB", description: oversize.at(-1) });
    }
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
  await toClassic(page);
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
  // "the eligibility" on the server, "the exclusions" in the M1 mock.
  await page.getByRole("button", { name: /^Change the (eligibility|exclusions)$/ }).click({ timeout: 30_000 });
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
  // Against the real server the steps allow, at most: the fit 900 s (a fit held for the press
  // computes only after it; fitOnceComputed's cap), the re-flow 900 s (the refit is a whole fit,
  // waited for while it runs), and 300 s for everything else: 2,100 s.
  test.setTimeout(REAL ? 2_100_000 : 240_000);
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
    await expect(page.getByTestId("stage-note")).toContainText("scores on unseen rows");
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
  if (REAL) await expect(page.getByTestId(`option-${WILLETT}`)).toContainText(/−[\d,]+ of 21,849/);
  // The screens read kcal as one day's intake: the ledger's ask card says so before screening by it.
  await confirmAsk(page.locator('[data-slot="exclusions"]'));
  await page.getByTestId("option-none").focus();
  await previewed(page, "Keep every row");
  // Every screen down to the last one shot, in the order the question lists them.
  const shots_: Record<string, string> = {
    [WILLETT]: "q-exclusions-willett",
    sex_neutral_500_5000: "q-exclusions-500-5000",
    sex_neutral_500_3500: "q-exclusions-500-3500",
  };
  const listed = await exclusions.getByRole("option").evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  const screens = listed.slice(listed.indexOf("none") + 1);
  const lastShot = Math.max(...Object.keys(shots_).map((k) => screens.indexOf(k)));
  expect(lastShot, "the screens to shoot are offered").toBeGreaterThanOrEqual(0);
  for (const key of screens.slice(0, lastShot + 1)) {
    await page.keyboard.press("ArrowDown");
    await expect(page.getByTestId(`option-${key}`)).toBeFocused();
    await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
    await landed(page, "with");
    await expect(stage(page).locator("[data-card=row_flow]")).toBeVisible();
    const name = shots_[key];
    if (!name) continue;
    await park(page);
    await exclusions.evaluate((el) => el.scrollIntoView({ block: "start" }));
    await shot(page, name, key === WILLETT ? { themes: ["light", "dark"] } : {});
  }

  // ── save the primary plot as a before/after pair, SVG and PNG ───────────────────────────
  // (The energy storyboard it was saved from is not asked under prediction; the exclusion
  // screens' row flow has the same two sides.)
  const primary = stage(page).locator("[data-primary]");
  const primarySave = primary.getByTestId("save-button");
  // What a pointer press at the primary's Save would land on, as Playwright's own hit check sees it
  // (a trial click: scrolled into view, every actionability check, no press): null when the Save
  // itself, else what intercepts it. On the energy question's preview the distribution card's own
  // Save sat over the primary's (a W1-C open issue); on this one nothing covers it, on either
  // backend, which this holds.
  const saveCoveredBy = await primarySave.click({ trial: true, timeout: 5_000 }).then(
    () => null,
    (e: Error) =>
      e.message
        .split("\n")
        .find((line) => line.includes("intercepts pointer events"))
        ?.trim() ?? `not pressable: ${e.message.split("\n")[0]}`,
  );
  measurements.save_covered_by = saveCoveredBy;
  expect(saveCoveredBy, "nothing covers the primary's Save").toBeNull();
  const openSave = async () => primarySave.click();
  await openSave();
  await page.getByRole("radio", { name: "Before and after" }).check();
  await shot(page, "save-pair");
  const [svg] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "SVG" }).click()]);
  const svgText = readFileSync(await svg.path(), "utf-8");
  expect(svgText).toContain("Preview, not recorded:");
  expect(svgText).toContain("Your data now");
  expect(svgText).toContain("With this choice");
  await openSave();
  await page.getByRole("radio", { name: "Before and after" }).check();
  const [png] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "PNG 2×" }).click()]);
  const pngBytes = readFileSync(await png.path());
  expect(pngBytes.subarray(1, 4).toString()).toBe("PNG");
  if (REAL) {
    copyFileSync(await svg.path(), resolve(SCREENS, "saved-exclusions-pair.svg"));
    copyFileSync(await png.path(), resolve(SCREENS, "saved-exclusions-pair.png"));
  }
  measurements.saved = { svg: svg.suggestedFilename(), png: png.suggestedFilename(), png_width: pngBytes.readUInt32BE(16) };
  await park(page);

  await page.getByTestId(`option-${WILLETT}`).focus();
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
  // Imputing a "not asked" column offers its lever on the option itself: its blanks as their own
  // Missing level (the server's option; the M1 mock still says so on the stage).
  if (REAL) {
    const impute = missing.getByRole("option").filter({ hasText: "Fill in each fold" });
    await expect(impute.getByRole("group", { name: "How imputation treats blanks" })).toContainText("meds_hbp");
    await expect(impute).toContainText("Missing");
  }
  await shot(page, "q-missing-impute");
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

  // ── energy adjustment is not asked under prediction ─────────────────────────────────────────
  // No coefficient is read as an effect there, so there is no energy model to choose (BEATS_PREDICT
  // ruling 5, the engine's W1-P11): the Router says so in place, with its reason, and the models
  // are the next question.
  const models = page.getByTestId("question-models");
  await expect(models).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("question-energy_adjustment")).toHaveCount(0);
  const notAsked = page.locator('[data-testid="na-energy_adjustment"], [data-testid="na-group"][data-keys~="energy_adjustment"]').first();
  await expect(notAsked).toContainText("Energy adjustment");
  await expect(notAsked).toContainText("not applicable");
  const why = notAsked.getByRole("button", { name: "why?" });
  if (await why.count()) await why.click();
  await expect(notAsked).toContainText("Under prediction no coefficient is read as an effect");

  // ── the model families: the shelf's first three, with the keyboard ──────────────────────────
  // The roles proposed below high confidence set the predictors only once confirmed: the ledger's
  // ask card lists them on this question, and they are confirmed as shown first.
  const familyOptions = models.getByRole("listbox", { name: /Model families/ }).locator(':scope > [role="option"]');
  await expect(familyOptions.first()).toBeVisible({ timeout: 60_000 });
  await confirmAsk(page.locator('[data-slot="models"]'));
  await expect(familyOptions.first()).toBeVisible({ timeout: 60_000 });
  const shelf = await familyOptions.evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  expect(shelf.length, "the shelf offers at least three families").toBeGreaterThanOrEqual(3);
  // The M1 mock's whole shelf; on the server the three linear families (the shelf now ranks a dozen,
  // and the re-flow refits each chosen family: boosted trees there take most of the journey's
  // time), in the shelf's order.
  const LIGHT = ["linear", "ridge", "elastic_net"];
  const families = REAL ? shelf.filter((k) => LIGHT.includes(k)) : shelf.slice(0, 3);
  expect(families, "three families to fit").toHaveLength(3);
  await page.getByTestId(`option-${families[0]}`).focus();
  // Never the previous question's preview as a placeholder (review: "PREVIEW Residual method", the
  // energy answer's, flashed for ~250 ms): sample the stage's title while the first family's
  // preview loads, and it is not the split's preview, the answer just before it.
  const titles: string[] = [];
  for (let i = 0; i < 20; i++) {
    titles.push((await page.getByTestId("stage-title").textContent().catch(() => "")) ?? "");
    await page.waitForTimeout(25);
  }
  expect(titles.filter((x) => /20%/.test(x)), "no foreign preview while loading").toEqual([]);
  await previewed(page, /./);
  // Nothing chosen yet: the stage's button (and Enter) fit the shown family.
  await expect(page.getByTestId("stage-record")).toHaveText(/^Fit the /);
  await park(page);
  await models.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await shot(page, "q-models");
  // Down the list with the arrow keys, Space on each chosen family.
  let at = shelf.indexOf(families[0]!);
  for (const key of families) {
    for (; at < shelf.indexOf(key); at++) await page.keyboard.press("ArrowDown");
    await expect(page.getByTestId(`option-${key}`)).toBeFocused();
    await page.keyboard.press("Space");
  }
  await expect(page.getByTestId("record-models")).toHaveText("Fit these 3 families");
  // Enter records only while no earlier answer is still being recorded: the card's button is
  // disabled and its list busy until the server's response to the confirmations just pressed has
  // come back, and a key pressed meanwhile is dropped (record/Options.tsx). The server records a
  // confirmation, and the card shows it, a beat before that response arrives, which on a busy CI
  // server is long enough to press into. So the press waits for the button, as a person's does,
  // for as long as any stage computes.
  await visibleWhileComputing(page, page.getByTestId("record-models").and(page.locator(":enabled")), [ANY_STAGE], {
    quiet: 60_000,
    cap: 300_000,
    label: "the families button is enabled",
  });
  await page.keyboard.press("Enter");
  // The Record shows the answer once the server's response arrives. A busy server answers late, so
  // the wait goes on while any stage computes (up to the cap), and ends once none has for a minute.
  await visibleWhileComputing(page, page.getByTestId("decision-models"), [ANY_STAGE], {
    quiet: 60_000,
    cap: 300_000,
    label: "the families are recorded",
  });
  await expect(page.getByTestId("decision-models")).toContainText("Three model families");

  // ── the fit, Fit pressed, then the Results (the stage's live view once fitted) ───────────
  // The wait for the result after the press lasts as long as the fit job runs (fitOnceComputed),
  // so none is kept here.
  measurements.fit_s = await timed(async () => {
    measurements.fit_pressed = await fitOnceComputed(page);
    await expect(page.getByTestId("banner-result-value")).toBeVisible();
  });
  await park(page);
  await expect(page.getByTestId("results")).toBeVisible({ timeout: 30_000 });
  const comparison = page.getByTestId("model-comparison");
  await expect(comparison.locator("[data-family]")).toHaveCount(3);
  await expect(comparison).toContainText("baseline");
  measurements.result = await page.getByTestId("banner-result").innerText().catch(() => null);
  // Findings learn they were answered. The dietary energy noticing does not fire under prediction
  // (no coefficient is read as an effect), so it is not among them.
  await expect(page.getByTestId("findings-settled")).toContainText("answered in the record");
  await expect(page.getByTestId("finding-pack::dietary::energy_adjustment")).toHaveCount(0);
  await expect(page.getByTestId("settled-pack::dietary::energy_adjustment")).toHaveCount(0);
  await settle(page, 600);
  await shot(page, "results", { themes: ["light", "dark"] });

  // ── a finding's evidence: focusing its card puts the evidence on the stage ───────────────
  // A finding an answer settled keeps its evidence a press away.
  await page.getByTestId("findings-settled-toggle").click();
  const settledCard = page.getByTestId("findings-settled").locator('[data-testid^="settled-"]').first();
  const card = (await settledCard.count()) ? settledCard : page.getByTestId("finding-cards").locator(":scope > li").first();
  await card.scrollIntoViewIfNeeded();
  await card.focus();
  await expect(stage(page)).toHaveAttribute("data-focus", "finding");
  await expect(page.getByTestId("stage-pill")).toHaveText("Evidence", { timeout: 15_000 });
  await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout: 15_000 });
  await park(page);
  await settle(page, 500);
  await shot(page, "evidence", { themes: ["light", "dark"] });
  await page.keyboard.press("Escape");
  await expect(stage(page)).toHaveAttribute("data-focus", "live");

  // ── change the split after fitting: the banner, the stage and the Results re-flow ──────────
  // (The energy method it was once is not asked under prediction.) A 30% holdout leaves fewer
  // training rows, so the cohort's flow, the models and the result all recompute.
  const banner = page.getByTestId("banner");
  await recordCol(page).evaluate((el) => (el.scrollTop = 0));
  await park(page);
  await expect(page.getByTestId("banner-train")).toBeVisible();
  const trainBefore = await page.getByTestId("banner-train").innerText();
  await shot(page, "banner-before", { clip: banner });
  await shot(page, "reflow-before");
  await page.getByRole("button", { name: "Change the seal" }).click();
  await page.getByTestId("option-0.3").focus();
  await previewed(page, /30%/);
  await landed(page, "with");
  await park(page);
  await shot(page, "q-split-change-30");
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
  await page.getByTestId("option-0.3").focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-split")).toContainText("30%");
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
  // The re-flow is waited for while the fit job runs, not for a flat time: the refit is a whole fit.
  const resultVeils = page.getByTestId("banner-result").locator("[data-veil]");
  const reflowed = await waitWhileComputing(
    page,
    async () =>
      (await resultVeils.evaluateAll((els) => els.length > 0 && els.every((e) => e.getAttribute("data-veil") === "fresh"))) &&
      (await page.locator("[data-veil=recomputing]").count()) === 0,
    ["fit"],
    { quiet: FIT_QUIET, cap: FIT_CAP },
  );
  expect(reflowed.done, `the banner, the stage and the Results are fresh again (waited ${reflowed.waited} s; ${reflowed.seen})`).toBe(true);
  measurements.reflow_all_s = Math.round((Date.now() - t0) / 100) / 10;
  await expect(page.getByTestId("banner-train")).not.toHaveText(trainBefore);
  const log = await page.evaluate(
    () => (window as unknown as { __veils: { t: number; at: string; veil: string | null }[] }).__veils,
  );
  // The order the veils lifted in: the cohort and the models before the result.
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
  await expect(page.locator('[data-slot="split"]').getByLabel("Earlier answers")).toContainText("20%");
  await park(page);
  await settle(page, 600);
  await shot(page, "banner-after", { clip: banner });
  await shot(page, "reflow-after", { themes: ["light", "dark"] });

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
  measurements.screens_over_300_kb = oversize;
  defects.oversize = oversize;
  mkdirSync(resolve(HERE, "../test-results"), { recursive: true });
  writeFileSync(resolve(HERE, `../test-results/m1-journey-${PREFIX}.json`), JSON.stringify(measurements, null, 1));
  console.log(JSON.stringify(measurements, null, 1));
  expect(problems, "no console errors and no failed requests").toEqual([]);
});

// ── the known defects on the real server, kept visible ────────────────────────────────────────

test("every review screen stays under 300 KB", () => {
  test.skip(!REAL || defects.oversize === undefined, "the mock journey asserts it inline");
  const over = defects.oversize ?? [];
  test.fixme(over.length > 0, `${KNOWN}: over 300 KB: ${over.join("; ")}`);
  expect(defects.oversize).toEqual([]);
});
