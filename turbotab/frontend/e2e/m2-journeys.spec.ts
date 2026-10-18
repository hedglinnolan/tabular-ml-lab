/**
 * M2's acceptance (M2_CONTRACT §9, §14): every lens fixture of §7 and the real NHANES export,
 * through the opening sequence in a real browser against the real server, with the stage
 * previewing each structural choice and the seal's basis named on every fixture.
 *
 *   TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d) venv/bin/python -m turbotab.server --port 8842
 *   E2E_BASE_URL=http://127.0.0.1:8842 npx playwright test m2-journeys
 *
 * The journeys:
 *   dietary       the grain asked (participant_id repeats) → repeats stated → one row per person →
 *                 the mean, recommended, previewed as the reshape (flip strip) → … → the fit → the
 *                 seal opened once
 *   clinical      an impossible value's repair previewed, then applied → time points stated → one
 *                 row per visit → temporal yes → a chronological, grouped seal
 *   metabolomics  the feature-major copy: orientation asked first, the turn previewed (flip strip),
 *                 then diagnosis on the turned table
 *   genomics      the wide path: the roles searched, the grain asked (a sample is not a person),
 *                 elastic net first on the shelf with its measured cost
 *   survey        sentinel codes set to missing (previewed, then applied) → the event level
 *   NHANES        SAS zeros repaired → the grain stated from SEQN → meds_hbp kept as a Missing
 *                 level → the fit → the seal opened once → a later change marked post-seal
 *
 * Screens (1440 × 900, each < 300 KB): docs/turbotab-next/m2/screens/real-*.png, and the frame
 * strips real-reshape-flip-<ms>ms.png and real-orientation-flip-<ms>ms.png. Timings are written to
 * docs/turbotab-next/m2/journeys-real.json (test-results/ is cleared by every Playwright run).
 */
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { onMock } from "./backend";
import { confirmAsk, confirmStageAsk } from "./ask";
import { toClassic } from "./classic";
import { ANY_STAGE, FIT_CAP, FIT_QUIET, attributeWhileComputing, cheapestFamily, fitOnceComputed, openSealCard, shelfEstimates, visibleWhileComputing } from "./fit";
import { nhanesExport } from "./nhanes";
import { confirmWaitingRoles, expectDesignHolds } from "./roles";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../..");
const SAMPLES = resolve(ROOT, "turbotab/sample_data");
const NHANES = nhanesExport();
const SCREENS = resolve(ROOT, "docs/turbotab-next/m2/screens");
const RESULTS = resolve(HERE, "../test-results");
const MAX_SCREEN_BYTES = 300 * 1024;
const FRAMES = [0, 150, 300, 450, 600, 900];

/** The rest of a journey, apart from the waits for a fit: a journey that waits for a fit gets
 *  this and that wait's cap (``FIT_CAP``) for each, so the wait that goes on while the fit job
 *  runs is not cut short by the test's own budget. */
const REST_OF_JOURNEY = 600_000;

interface StageHook {
  manual: (v: boolean) => void;
  advance: (ms: number) => void;
  state: () => { pos: number; target: number; last: number; paused?: boolean };
}
type Review = { __turbotabStage: StageHook; __turbotabReview?: boolean };

const stage = (page: Page) => page.getByTestId("stage");
/** Move the pointer off everything previewable, so no hover lingers in a screenshot. */
const park = (page: Page) => page.mouse.move(1430, 890);

const timings: Record<string, Record<string, unknown>> = {};
const shots: string[] = [];

/** One screen (light, or light and dark), the Record scrolled to `at` first. */
async function shot(page: Page, name: string, opts: { at?: Locator; themes?: ("light" | "dark")[]; clip?: Locator } = {}) {
  mkdirSync(SCREENS, { recursive: true });
  if (opts.at) await opts.at.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await park(page);
  for (const theme of opts.themes ?? ["light"]) {
    await page.emulateMedia({ colorScheme: theme });
    await page.waitForTimeout(300);
    const path = resolve(SCREENS, `real-${name}${opts.themes ? `-${theme}` : ""}.png`);
    const box = opts.clip ? await opts.clip.boundingBox() : null;
    await page.screenshot(box ? { path, clip: box } : { path });
    expect(statSync(path).size, `${name} (${theme}) stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
    shots.push(path);
  }
  await page.emulateMedia({ colorScheme: "light" });
}

/** Seconds until `done` holds. */
async function timed(done: () => Promise<unknown>): Promise<number> {
  const t0 = Date.now();
  await done();
  return Math.round((Date.now() - t0) / 100) / 10;
}

/** The stage shows the focused option's own preview and has settled. */
async function previewed(page: Page, title?: string | RegExp, timeout = 20_000) {
  if (title) await expect(page.getByTestId("stage-title")).toHaveText(title, { timeout });
  await expect(page.getByTestId("stage-loading")).not.toHaveAttribute("data-on", /.*/, { timeout });
}

/** The player has landed on a side and stopped moving. */
async function landed(page: Page, side: "now" | "with") {
  const player = page.getByTestId("player");
  await expect(player).toHaveAttribute("data-side", side, { timeout: 10_000 });
  await expect(player).not.toHaveAttribute("data-moving", "true", { timeout: 10_000 });
}

/** Step the player's clock by hand from "your data now" and save each frame of the flip. */
async function frameStrip(page: Page, prefix: string): Promise<number[]> {
  const hook = () => (window as unknown as Review).__turbotabStage;
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.manual(true));
  await page.getByTestId("flip-now").click();
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.advance(3000));
  await page.waitForTimeout(400);
  await page.getByTestId("flip-with").click();
  let t = 0;
  const positions: number[] = [];
  for (const ms of FRAMES) {
    await page.evaluate((d) => (window as unknown as Review).__turbotabStage.advance(d), ms - t);
    t = ms;
    positions.push((await page.evaluate(() => (window as unknown as Review).__turbotabStage.state())).pos);
    await page.waitForTimeout(350); // the views ease over their own 300 ms
    const path = resolve(SCREENS, `real-${prefix}-${String(ms).padStart(3, "0")}ms.png`);
    const box = (await stage(page).boundingBox())!;
    await page.screenshot({ path, clip: box });
    expect(statSync(path).size, `${prefix} ${ms} ms stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
    shots.push(path);
  }
  void hook;
  await page.evaluate(() => (window as unknown as Review).__turbotabStage.manual(false));
  return positions;
}

/** Console errors and failed requests (a 409 is an answer the app shows, not a failure). */
function watch(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("status of 409")) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.status() !== 409) problems.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });
  return problems;
}

/** Preview round-trips (request start → response end), by decision kind. */
function previewTimes(page: Page): { kind: string; ms: number }[] {
  const out: { kind: string; ms: number }[] = [];
  page.on("requestfinished", (req) => {
    if (req.method() !== "POST" || !req.url().endsWith("/preview")) return;
    const t = req.timing();
    let kind = "?";
    try {
      kind = (JSON.parse(req.postData() ?? "{}") as { kind?: string }).kind ?? "?";
    } catch {
      /* not JSON */
    }
    if (t.responseEnd > 0) out.push({ kind, ms: Math.round(t.responseEnd - t.requestStart) });
  });
  return out;
}

function summarize(ms: { ms: number }[]) {
  const sorted = ms.map((p) => p.ms).sort((a, b) => a - b);
  return {
    n: sorted.length,
    p50_ms: sorted[Math.floor(0.5 * (sorted.length - 1))] ?? null,
    p95_ms: sorted[Math.floor(0.95 * (sorted.length - 1))] ?? null,
    max_ms: sorted.at(-1) ?? null,
  };
}

async function start(page: Page, journey: string) {
  await page.addInitScript(() => {
    (window as unknown as Review).__turbotabReview = true;
  });
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "no-preference" });
  await page.goto("/");
  test.skip(await onMock(page), "the M2 journeys run against the real server");
  timings[journey] = {};
  return { problems: watch(page), previews: previewTimes(page), t0: Date.now() };
}

async function openPath(page: Page, path: string) {
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  await page.getByPlaceholder("/path/to/table.csv").fill(path);
  await page.getByRole("button", { name: "Open", exact: true }).click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
  await toClassic(page);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 60_000 });
}

async function chooseLens(page: Page, lens: string) {
  await page.getByTestId(`option-${lens}`).focus();
  // A lens preview reads every column: on the feature-major metabolomics copy it takes about
  // 45 s on two workers (reported as a W1 follow-up), so it is given two minutes.
  await previewed(page, undefined, 120_000);
  await page.keyboard.press("Space");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-lens")).toBeVisible({ timeout: 30_000 });
}

async function chooseTarget(page: Page, column: string) {
  const box = page.getByRole("combobox");
  await expect(box).toBeVisible({ timeout: 30_000 });
  await box.fill(column);
  await box.press("Enter");
  await previewed(page);
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText(column, { timeout: 30_000 });
}

/** Focus an option of a question (the stage previews it), then record it with Enter. A refusal at
 *  the control with ways forward takes its first one (as a person pressing it would: below a
 *  detection limit, the censoring-aware fill). */
async function answer(page: Page, question: string, option: string, opts: { title?: string | RegExp } = {}) {
  await expect(page.getByTestId(`question-${question}`)).toBeVisible({ timeout: 60_000 });
  const o = page.getByTestId(`question-${question}`).getByTestId(`option-${option}`);
  await o.focus();
  await previewed(page, opts.title);
  await page.keyboard.press("Enter");
  const recorded = page.getByTestId(`decision-${question}`);
  const took = await recorded.waitFor({ state: "visible", timeout: 10_000 }).then(
    () => true,
    () => false,
  );
  if (!took) {
    const exit = page.locator(`[data-slot="${question}"] [role="alert"] button`, { hasText: / instead$/ }).first();
    if (await exit.isVisible().catch(() => false)) await exit.click();
  }
  await expect(recorded).toBeVisible({ timeout: 60_000 });
}

/** The roles as proposed. */
async function confirmRoles(page: Page) {
  await expect(page.getByTestId("question-roles")).toBeVisible({ timeout: 60_000 });
  await page.getByTestId("record-roles").click();
  await expect(page.getByTestId("decision-roles")).toBeVisible({ timeout: 30_000 });
}

/** The light families: the linear ones, which a journey can afford to fit and refit. */
const LIGHT = ["linear", "ridge", "elastic_net"];

/** Fit the families (the first `n` on the shelf; "cheapest", the light family the shelf estimates
 *  cheapest to fit, read when it is chosen, since confirming the readings moves the estimates;
 *  else the three linear ones, else its first three: the shelf now offers a dozen, some estimated
 *  at most of an hour), press Fit once the fit has computed (SIZING P0.8: no score is served
 *  before it), and wait for the Results. `onShelf` checks the shelf once the readings the fit
 *  reads are confirmed (the estimates and the order can move with them), before a family is
 *  chosen. */
async function fit(
  page: Page,
  pick?: number | "cheapest",
  onShelf?: (options: Locator) => Promise<void>,
): Promise<number> {
  // The shelf ranks the families again after every reading confirmed (from seconds to nearly two
  // minutes on the 500-column genomics table, 55 s and 108 s on a throttled machine), and the
  // options wait for it: each wait lasts while it computes.
  const models = page.getByTestId("question-models");
  await visibleWhileComputing(page, models, ["shelf"]);
  // The shelf's own options (the ask card's selects hold options too); the roles proposed below
  // high confidence are confirmed first, on the ledger's ask card, since the fit reads them.
  const options = models.getByRole("listbox", { name: /Model families/ }).locator(':scope > [role="option"]');
  await visibleWhileComputing(page, options.first(), ["shelf"]);
  // Every journey here is a prediction: no coefficient is read as an effect, so the energy model
  // is not asked, and the models come right after the split.
  await expect(page.getByTestId("question-energy_adjustment")).toHaveCount(0);
  await confirmAsk(page.locator('[data-slot="models"]'));
  await visibleWhileComputing(page, options.first(), ["shelf"]);
  await onShelf?.(options);
  const families = await options.evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  // The first `n`; the cheapest light family by the shelf's estimate (read each time they are
  // chosen: a refusal's way forward changes the table, and the estimates with it); else the three
  // linear families where the shelf has them (the later steps refit each chosen family), else its
  // first three.
  const light = families.filter((k) => LIGHT.includes(k));
  const chosenFamilies = async (): Promise<string[]> => {
    if (pick === "cheapest") return [(await cheapestFamily(page, LIGHT)) ?? families[0]!];
    if (pick !== undefined) return families.slice(0, pick);
    return light.length === 3 ? light : families.slice(0, 3);
  };
  // Down the list with the arrow keys, Space on each chosen family, Enter to fit them.
  const choose = async () => {
    await visibleWhileComputing(page, options.first(), ["shelf"]);
    const chosen = await chosenFamilies();
    if (pick === "cheapest") console.log("fit:", chosen.join(), "of", JSON.stringify(await shelfEstimates(page)));
    await page.getByTestId(`option-${chosen[0]}`).focus();
    let at = families.indexOf(chosen[0]!);
    for (const key of chosen) {
      for (; at < families.indexOf(key); at++) await page.keyboard.press("ArrowDown");
      await expect(page.getByTestId(`option-${key}`)).toBeFocused();
      if ((await page.getByTestId(`option-${key}`).getAttribute("aria-selected")) !== "true") {
        await page.keyboard.press("Space");
      }
    }
    await page.keyboard.press("Enter");
  };
  await choose();
  const recorded = page.getByTestId("decision-models");
  for (let tries = 0; tries < 3; tries++) {
    const took = await recorded
      .waitFor({ state: "visible", timeout: 15_000 })
      .then(() => true)
      .catch(() => false);
    if (took) break;
    // Refused: for a reading the fit reads that the card asked about meanwhile (confirmed), or
    // with ways forward at the control (raw counts: the first normalization it offers); then the
    // families are chosen again.
    const exit = page.locator('[data-slot="models"] [role="alert"] button', { hasText: / instead$/ }).first();
    if (await exit.isVisible().catch(() => false)) await exit.click();
    else await confirmAsk(page.locator('[data-slot="models"]'));
    await page.waitForTimeout(1_500);
    if (await recorded.isVisible().catch(() => false)) break;
    await choose();
  }
  await expect(recorded).toBeVisible({ timeout: 30_000 });
  // Fit is pressed on the Results; the wait for the result after it lasts as long as the fit job
  // runs (fitOnceComputed), so none is kept here.
  return timed(async () => {
    await fitOnceComputed(page);
    await expect(page.getByTestId("banner-result-value")).toBeVisible();
  });
}

/** The seal question: its basis is named on the card and on the stage, then 20% is held out. */
async function seal(page: Page, basis: string, name: string, themes?: ("light" | "dark")[]) {
  const q = page.getByTestId("question-split");
  await expect(q).toBeVisible({ timeout: 60_000 });
  await expect(q.getByTestId("seal-basis")).toHaveAttribute("data-basis", basis);
  await q.getByTestId("option-0.2").focus();
  await previewed(page, /20%/);
  await landed(page, "with");
  await expect(stage(page).locator("[data-view=seal_fork]")).toBeVisible();
  await shot(page, name, { at: q, themes });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-split")).toBeVisible({ timeout: 30_000 });
  // The live seal waits for the split to compute: on the 500-column genomics table, beside the
  // shelf's ranking, that took over 30 s on a busy server (2 workers).
  await expect(page.getByTestId("live-seal")).toBeVisible({ timeout: 120_000 });
}

/**
 * Open the seal: the Router's last step, a CONSEQUENCE card under the Results. Until it is pressed
 * no held-out score is on screen; after, each family's held-out score is, once.
 */
async function openSeal(page: Page, journey: string): Promise<number> {
  // The Results are drawn sealed, and the Record names the opening: the waits for both go on while
  // the app computes, not for a flat time.
  const results = page.getByTestId("results");
  await attributeWhileComputing(page, results, "data-seal-phase", "sealed", [ANY_STAGE], { label: "the Results are sealed" });
  const cmp = page.getByTestId("model-comparison");
  await expect(cmp.getByTestId("held-out-cell-sealed").first()).toBeVisible();
  await expect(cmp).not.toContainText(/held out [−\d]/);
  // The Record names the last step and goes to the one card that opens the seal.
  const step = page.getByTestId("open-seal-step");
  await visibleWhileComputing(page, step, [ANY_STAGE], { label: "the Record names the opening" });
  await step.scrollIntoViewIfNeeded();
  await step.getByTestId("go-open-seal").click();
  const card = page.getByTestId("open-seal-card");
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("It happens once.");
  await expect(card.getByTestId("open-seal")).toBeFocused();
  await expect(card).toBeInViewport();
  await shot(page, `${journey}-open-seal`, { themes: ["light", "dark"] });
  const s = await timed(async () => {
    // The opening waits for the triage of open noticings, settled as the refusal offers; with
    // several families the final model is then named on cross-validation, the card's first way
    // forward (e2e/fit.ts openSealCard).
    await openSealCard(page);
    // The opening is a record no stage reads (none of the graph's stages reads the seal slot), so
    // nothing computes after it: the sentence and the opened scores are an answer already served,
    // and these two waits are for a render, not for work that a computing stage would bound.
    await expect(page.getByTestId("decision-open_seal")).toBeVisible({ timeout: 30_000 });
    await expect(results).toHaveAttribute("data-seal-phase", "opened", { timeout: 60_000 });
  });
  await expect(page.getByTestId("open-seal-card")).toHaveCount(0);
  await expect(page.getByTestId("seal-opened")).toContainText("The seal was opened once");
  await expect(cmp.locator("text=/^held out [−\\d]/").first()).toBeVisible();
  // The seal opens once: its sentence offers no change.
  await expect(page.getByTestId("decision-open_seal").getByRole("button", { name: /change/i })).toHaveCount(0);
  await page.getByTestId("seal-opened").scrollIntoViewIfNeeded();
  await shot(page, `${journey}-seal-opened`);
  return s;
}

function finish(journey: string, run: { problems: string[]; previews: { kind: string; ms: number }[]; t0: number }) {
  timings[journey]!.total_s = Math.round((Date.now() - run.t0) / 100) / 10;
  timings[journey]!.previews = summarize(run.previews);
  timings[journey]!.problems = run.problems;
  const path = resolve(ROOT, "docs/turbotab-next/m2/journeys-real.json");
  let prev: Record<string, unknown> = {};
  try {
    prev = JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
  } catch {
    /* first journey */
  }
  writeFileSync(path, JSON.stringify({ ...prev, [journey]: timings[journey] }, null, 1));
  console.log(journey, JSON.stringify(timings[journey]));
  expect(run.problems, "no console errors and no failed requests").toEqual([]);
}

/**
 * What repeats, read from the data: stated when the reading is sure, else asked with the reading
 * marked beside its option ("read from the data, medium"), which is recorded. Either way it says
 * ``words``.
 */
async function whatRepeats(page: Page, words: RegExp) {
  const asked = page.getByTestId("question-repeat_kind");
  const stated = page.getByTestId("skip-repeat_kind");
  await expect(asked.or(stated)).toBeVisible({ timeout: 60_000 });
  if (await stated.isVisible()) {
    await expect(stated).toContainText(words);
    return;
  }
  const read = asked.getByRole("option").filter({ hasText: "read from the data" });
  await expect(read).toHaveCount(1);
  await expect(read).toContainText(words);
  await read.focus();
  await previewed(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-repeat_kind")).toBeVisible({ timeout: 30_000 });
}

/** Whether a question opens within `ms` (it is asked only where the data leave it open). */
async function asked(page: Page, key: string, ms = 20_000): Promise<boolean> {
  return page
    .getByTestId(`question-${key}`)
    .waitFor({ state: "visible", timeout: ms })
    .then(() => true)
    .catch(() => false);
}

/** A yes/no outcome's follow-up, when asked (WP17): the same for everyone, the yes/no outcome
 *  standing, as these fixtures' authors state. */
async function sameFollowUp(page: Page) {
  if (!(await asked(page, "follow_up"))) return;
  const slot = page.locator('[data-slot="follow_up"]');
  await slot.getByTestId("option-same").click();
  const record = slot.getByTestId("record-same");
  if (await record.count()) await record.click();
  await expect(page.getByTestId("decision-follow_up")).toBeVisible({ timeout: 30_000 });
}

/** Which way round an assay table is, when asked: one row per sample, read from the shape. */
async function samplesAsRows(page: Page) {
  if (!(await asked(page, "orientation"))) return;
  await answer(page, "orientation", "sample_major");
}

// Each journey opens its own table on the one server: one failing does not stop the next.
test.describe.configure({ mode: "default" });

// ── dietary ──────────────────────────────────────────────────────────────────────────────────

test("dietary recalls: the grain asked, repeats stated, one row per person, the mean, the seal opened once", async ({
  page,
}) => {
  test.setTimeout(REST_OF_JOURNEY + FIT_CAP); // the rest, and one fit (held for the press: computed only after it)
  const run = await start(page, "dietary");
  const T = timings.dietary!;
  T.open_s = await timed(() => openPath(page, resolve(SAMPLES, "dietary_recalls.csv")));
  await chooseLens(page, "dietary");
  await chooseTarget(page, "hba1c");
  await answer(page, "purpose", "prediction");

  // The grain is asked: participant_id repeats, so it is suggested beside its option, never chosen.
  const grain = page.getByTestId("question-grain");
  await expect(grain).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("option-repeated")).toHaveAttribute("aria-selected", "false");
  await expect(page.getByTestId("grain-id-participant_id")).toBeVisible();
  await page.getByTestId("option-repeated").focus();
  await previewed(page);
  await shot(page, "dietary-grain", { at: grain, themes: ["light", "dark"] });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-grain")).toContainText("participant_id");

  // What repeats: repeated measurements, read from the data.
  await whatRepeats(page, /repeat/i);

  // One row per person.
  await answer(page, "unit", "unit");

  // The aggregation: the mean recommended with its reason; the reshape plays on the flip.
  const agg = page.getByTestId("question-aggregation");
  await expect(agg).toBeVisible({ timeout: 60_000 });
  const mean = page.getByTestId("option-mean");
  await expect(mean).toContainText("recommended");
  await expect(mean).toHaveAttribute("aria-selected", "false");
  // Averaging reads whole-number columns as amounts: the ledger asks before it combines them.
  await confirmAsk(page.locator('[data-slot="aggregation"]'));
  await mean.focus();
  await previewed(page);
  if (await confirmStageAsk(page)) {
    await mean.focus();
    await previewed(page);
  }
  await landed(page, "with");
  const reshape = stage(page).locator("[data-view=reshape_table]");
  await expect(reshape).toBeVisible();
  await expect(page.getByTestId("readout")).toContainText("600");
  await expect(page.getByTestId("readout")).toContainText("300");
  await expect(reshape).toContainText("from rows");
  await shot(page, "dietary-aggregation", { at: agg, themes: ["light", "dark"] });
  T.reshape_positions = await frameStrip(page, "reshape-flip");
  // Another method morphs straight to its result.
  await mean.focus();
  await page.keyboard.press("ArrowDown");
  await previewed(page);
  await mean.focus();
  await previewed(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-aggregation")).toContainText("300");

  // Temporal does not apply to repeats of one measurement.
  await confirmRoles(page);
  await answer(page, "exclusions", "none");
  await expect(page.getByTestId("question-missing")).toBeVisible({ timeout: 60_000 });
  const firstMissing = await page
    .getByTestId("question-missing")
    .getByRole("option")
    .first()
    .getAttribute("data-key");
  await answer(page, "missing", firstMissing!);
  await seal(page, "grouped", "dietary-seal", ["light", "dark"]);
  T.fit_s = await fit(page);
  await expect(page.getByTestId("held-out-sealed")).toBeVisible({ timeout: 30_000 });
  await shot(page, "dietary-results-sealed");
  T.open_seal_s = await openSeal(page, "dietary");
  finish("dietary", run);
});

// ── clinical ─────────────────────────────────────────────────────────────────────────────────

test("clinical longitudinal: an impossible value repaired, time points, temporal, a chronological seal", async ({ page }) => {
  test.setTimeout(REST_OF_JOURNEY + FIT_CAP); // the rest, and one fit (held for the press: computed only after it)
  const run = await start(page, "clinical");
  const T = timings.clinical!;
  T.open_s = await timed(() => openPath(page, resolve(SAMPLES, "clinical_longitudinal.csv")));
  await chooseLens(page, "clinical");

  // The impossible values: previewed on the stage (the changed cells, the column's spread with
  // its band marked), then applied.
  const repair = page.getByTestId("repair-pack::clinical::impossible_vs_extreme");
  await expect(repair).toBeVisible({ timeout: 60_000 });
  const setMissing = repair.locator('[role="option"]').first();
  await setMissing.focus();
  await previewed(page);
  await landed(page, "with");
  await expect(stage(page).locator("[data-card=table_focus]")).toBeVisible();
  await expect(stage(page).locator("[data-card=distribution]")).toBeVisible();
  await shot(page, "clinical-repair-preview", { at: page.getByTestId("repairs"), themes: ["light", "dark"] });
  await page.keyboard.press("Enter");
  const settled = page.getByTestId("repair-settled-pack::clinical::impossible_vs_extreme");
  await expect(settled).toBeVisible({ timeout: 30_000 });
  await expect(settled).toContainText("missing");

  await chooseTarget(page, "progressed");
  // A binary outcome: which level is the event is asked, never guessed.
  const event = page.getByTestId("question-event");
  await expect(event).toBeVisible({ timeout: 60_000 });
  await expect(event.getByRole("option", { selected: true })).toHaveCount(0);
  await answer(page, "event", "1");
  await sameFollowUp(page);
  await answer(page, "purpose", "prediction");

  await answer(page, "grain", "repeated");
  await expect(page.getByTestId("decision-grain")).toContainText("subject_id");
  // Time points, read from the visit dates' spacing.
  await whatRepeats(page, /time points|days apart|schedule/i);
  await answer(page, "unit", "row");
  // Later visits predicted from earlier ones: the seal is drawn by time, grouped too.
  const temporal = page.getByTestId("question-temporal");
  await expect(temporal).toBeVisible({ timeout: 60_000 });
  await temporal.getByTestId("option-true").focus();
  await previewed(page);
  // A chronological split needs the column that says when each row was taken: the preview asks
  // for it, and its first way forward (the visit dates) is previewed in its place, then recorded.
  const orderBy = stage(page).getByRole("button", { name: "Order by visit_date" });
  if (await orderBy.isVisible().catch(() => false)) {
    await orderBy.click();
    await previewed(page, /Order by/);
    await landed(page, "with");
    await shot(page, "clinical-temporal", { at: temporal });
    await page.getByTestId("stage-record").click();
  } else {
    await landed(page, "with");
    await shot(page, "clinical-temporal", { at: temporal });
    await page.keyboard.press("Enter");
  }
  await expect(page.getByTestId("decision-temporal")).toBeVisible({ timeout: 30_000 });

  await confirmRoles(page);
  await answer(page, "exclusions", "none");
  const firstMissing = await page.getByTestId("question-missing").getByRole("option").first().getAttribute("data-key");
  await answer(page, "missing", firstMissing!);
  const q = page.getByTestId("question-split");
  await expect(q).toBeVisible({ timeout: 60_000 });
  await expect(q.getByTestId("seal-basis")).toContainText(/chronolog|time|latest/i);
  await seal(page, "grouped", "clinical-seal", ["light", "dark"]);
  await expect(stage(page).getByTestId("live-seal")).toContainText(/chronolog|time|latest|later/i);
  T.fit_s = await fit(page, 1);
  await expect(page.getByTestId("held-out-sealed")).toBeVisible({ timeout: 30_000 });
  finish("clinical", run);
});

// ── metabolomics ─────────────────────────────────────────────────────────────────────────────

/** The feature-major copy (as the prototype's capture made it): numeric columns, turned. */
function featureMajorCopy(): string {
  const lines = readFileSync(resolve(SAMPLES, "metabolomics_untargeted.csv"), "utf-8").trim().split(/\r?\n/);
  const header = lines[0]!.split(",");
  const rows = lines.slice(1).map((l) => l.split(","));
  const id = header.indexOf("sample_id");
  const numeric = header
    .map((name, j) => ({ name, j }))
    .filter(({ j }) => j !== id && rows.every((r) => r[j] === "" || Number.isFinite(Number(r[j]))));
  const out = [["feature_id", ...rows.map((r) => r[id]!)].join(",")];
  for (const { name, j } of numeric) out.push([name, ...rows.map((r) => r[j]!)].join(","));
  mkdirSync(RESULTS, { recursive: true });
  const path = resolve(RESULTS, "metabolomics_feature_major.csv");
  writeFileSync(path, out.join("\n") + "\n");
  return path;
}

test("metabolomics exported features-in-rows: orientation first, the turn, then diagnosis on the turned table", async ({
  page,
}) => {
  test.setTimeout(REST_OF_JOURNEY); // no fit is waited for
  const run = await start(page, "metabolomics");
  const T = timings.metabolomics!;
  const path = featureMajorCopy();
  T.open_s = await timed(() => openPath(page, path));
  await chooseLens(page, "metabolomics");

  const orientation = page.getByTestId("question-orientation");
  await expect(orientation).toBeVisible({ timeout: 60_000 });
  // Read from the shape, beside its option; never chosen for the user.
  await expect(page.getByTestId("option-feature_major")).toHaveAttribute("aria-selected", "false");
  // The outcome and the findings wait: every check would read across the wrong axis.
  await expect(page.getByTestId("target-withheld")).toBeVisible();
  await expect(page.getByTestId("question-target")).toHaveCount(0);
  await expect(page.getByTestId("findings-await-orientation")).toBeVisible();

  await page.getByTestId("option-feature_major").focus();
  await previewed(page);
  await landed(page, "with");
  const turn = stage(page).locator("[data-view=turn_table]");
  await expect(turn).toBeVisible();
  await expect(page.getByTestId("readout")).toContainText("396");
  await expect(page.getByTestId("readout")).toContainText("80");
  await shot(page, "metabolomics-orientation", { at: orientation, themes: ["light", "dark"] });
  T.orientation_positions = await frameStrip(page, "orientation-flip");
  await page.getByTestId("option-feature_major").focus();
  await previewed(page);
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-orientation")).toBeVisible({ timeout: 30_000 });

  // The turned table: a row per sample, a column per feature; the findings read it that way.
  await expect(page.getByTestId("findings-await-orientation")).toHaveCount(0, { timeout: 60_000 });
  await expect(page.getByTestId("banner-rows")).toContainText("80", { timeout: 60_000 });
  await expect(page.getByTestId("finding-pack::metabolomics::no_run_order")).toHaveCount(0);
  await chooseTarget(page, "responder");
  await answer(page, "event", "1");
  await sameFollowUp(page);
  await answer(page, "purpose", "prediction");
  await answer(page, "grain", "one_row_per_unit");
  await confirmRoles(page);
  await answer(page, "exclusions", "none");
  // Most features have blanks (left-censored): dropping incomplete rows would leave none, so the
  // blanks are imputed (in each fold).
  await answer(page, "missing", "impute");
  await seal(page, "one_row_per_unit", "metabolomics-seal");
  finish("metabolomics", run);
});

// ── genomics ─────────────────────────────────────────────────────────────────────────────────

test("genomics expression: the wide path, the roles searched, elastic net first with its cost", async ({ page }) => {
  test.setTimeout(REST_OF_JOURNEY + FIT_CAP); // the rest, and one fit (held for the press: computed only after it)
  const run = await start(page, "genomics");
  const T = timings.genomics!;
  T.open_s = await timed(() => openPath(page, resolve(SAMPLES, "genomics_expression.csv")));
  await chooseLens(page, "genomics");
  await samplesAsRows(page);
  await chooseTarget(page, "condition");
  await expect(page.getByTestId("question-event")).toBeVisible({ timeout: 60_000 });
  await answer(page, "event", "case");
  await sameFollowUp(page);
  await answer(page, "purpose", "prediction");
  // `sample_id` names a sample, not a person: the grain is asked, never stated from it.
  await expect(page.getByTestId("question-grain")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("skip-grain")).toHaveCount(0);
  await answer(page, "grain", "one_row_per_unit");

  const roles = page.getByTestId("question-roles");
  await expect(roles).toBeVisible({ timeout: 60_000 });
  const search = page.getByTestId("roles-search");
  await expect(search).toBeVisible();
  const count = page.getByTestId("roles-search-count");
  const all = ((await count.textContent()) ?? "").match(/[\d,]+/)?.[0];
  expect(all, "the roles question counts the table's columns").toBeTruthy();
  T.roles_search_ms = await (async () => {
    const t0 = Date.now();
    await search.fill("gene_04");
    await expect(count).toHaveText(new RegExp(`^\\d+ of ${all}$`));
    return Date.now() - t0;
  })();
  await search.fill("gene_0417");
  await expect(count).toHaveText(`1 of ${all}`);
  await expect(page.getByTestId("role-chip-gene_0417")).toBeVisible();
  await shot(page, "genomics-roles-search", { at: roles, themes: ["light", "dark"] });
  await search.fill("");
  await page.getByTestId("record-roles").click();
  await expect(page.getByTestId("decision-roles")).toBeVisible({ timeout: 30_000 });

  await answer(page, "exclusions", "none");
  const firstMissing = await page.getByTestId("question-missing").getByRole("option").first().getAttribute("data-key");
  await answer(page, "missing", firstMissing!);
  await seal(page, "one_row_per_unit", "genomics-seal");

  // Elastic net first on the shelf; a fit long enough to weigh says so before it is chosen.
  const models = page.getByTestId("question-models");
  await visibleWhileComputing(page, models, ["shelf"]);
  await expect(
    models.getByRole("listbox", { name: /Model families/ }).locator(':scope > [role="option"]').first(),
  ).toHaveAttribute("data-key", "elastic_net", { timeout: 60_000 });
  await shot(page, "genomics-models", { at: models });
  // The genes the roles stage proposed alike wait for their confirmation before the families: one
  // line on the ask card and one press for all of them (ruling R-g), after which the server holds
  // no role waiting (e2e/roles.ts).
  const confirmedRoles = await confirmWaitingRoles(page, page.locator('[data-slot="models"]'));
  T.roles_confirmed = confirmedRoles.length;
  // The elastic net, first on this shelf, is estimated at minutes once the 497 genes are confirmed
  // as predictors (the card says so, in the server's words, before a family is chosen), so the
  // journey fits the cheapest light family by the shelf's estimate instead of waiting out the
  // elastic net's fit (the plan's ruling 11).
  T.fit_s = await fit(page, "cheapest", async (options) => {
    await expect(options.first()).toHaveAttribute("data-key", "elastic_net");
    await expect(models.getByTestId("cost-elastic_net")).toContainText(/about \d+ (second|minute|hour)s?/);
  });
  // The model holds the genes confirmed (the server's design, its count of each role).
  T.roles_confirmed_in_design = await expectDesignHolds(page, confirmedRoles);
  await expect(page.getByTestId("held-out-sealed")).toBeVisible({ timeout: 30_000 });
  finish("genomics", run);
});

// ── survey ───────────────────────────────────────────────────────────────────────────────────

test("survey sentinels: codes set to missing, previewed then applied, and the event level asked", async ({ page }) => {
  test.setTimeout(REST_OF_JOURNEY); // no fit is waited for
  const run = await start(page, "survey");
  const T = timings.survey!;
  T.open_s = await timed(() => openPath(page, resolve(SAMPLES, "survey_sentinels.csv")));
  await chooseLens(page, "survey");

  // The codebook's 7/8/9 in a 1–5 item: set to missing, previewed on the item's own rows first.
  const repairs = page.getByTestId("repairs");
  await expect(repairs).toBeVisible({ timeout: 60_000 });
  const card = repairs.locator('[data-testid^="repair-"][data-testid*="sentinel"]').first();
  await expect(card).toBeVisible();
  const option = card.locator('[role="option"]').first();
  await option.focus();
  await previewed(page);
  await landed(page, "with");
  await expect(stage(page).locator("[data-card=table_focus]")).toBeVisible();
  await expect(stage(page).locator("[data-card=distribution]")).toBeVisible();
  await shot(page, "survey-sentinel-preview", { at: repairs, themes: ["light", "dark"] });
  const id = (await card.getAttribute("data-testid"))!.replace(/^repair-/, "");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId(`repair-settled-${id}`)).toBeVisible({ timeout: 30_000 });

  await chooseTarget(page, "sought_support");
  const event = page.getByTestId("question-event");
  await expect(event).toBeVisible({ timeout: 60_000 });
  await expect(event.getByRole("option", { selected: true })).toHaveCount(0);
  const level = await event.getByRole("option").first().getAttribute("data-key");
  await event.getByTestId(`option-${level}`).focus();
  await previewed(page);
  await shot(page, "survey-event", { at: event });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-event")).toBeVisible({ timeout: 30_000 });
  await answer(page, "purpose", "prediction");
  // respondent_id is unique on every row: the grain is stated.
  await expect(page.getByTestId("skip-grain")).toContainText("respondent_id", { timeout: 60_000 });
  await confirmRoles(page);
  await answer(page, "exclusions", "none");
  const firstMissing = await page.getByTestId("question-missing").getByRole("option").first().getAttribute("data-key");
  await answer(page, "missing", firstMissing!);
  await seal(page, "one_row_per_unit", "survey-seal");
  finish("survey", run);
});

// ── NHANES ───────────────────────────────────────────────────────────────────────────────────

test("NHANES: SAS zeros repaired, the grain stated, meds_hbp kept as a Missing level, the seal opened once", async ({
  page,
}) => {
  // The rest of the journey, and two waits for a fit, each up to its cap while the fit job runs:
  // the fit (held for the press, it computes only after it: fitOnceComputed) and the refit after
  // the post-seal change (a whole fit; it took longer than the first one when this was measured).
  test.setTimeout(REST_OF_JOURNEY + 2 * FIT_CAP);
  const run = await start(page, "nhanes");
  const T = timings.nhanes!;
  T.open_s = await timed(() => openPath(page, NHANES));
  await expect(page.getByTestId("banner-rows")).toContainText("21,849");
  await chooseLens(page, "dietary");

  // SAS transport zeros (5.4e-79) → 0, previewed on the cells it changes, then applied.
  const zeros = page.getByTestId("repair-sas_zeros");
  await expect(zeros).toBeVisible({ timeout: 60_000 });
  await zeros.locator('[role="option"]').first().focus();
  await previewed(page);
  await landed(page, "with");
  await expect(stage(page).locator("[data-card=table_focus]")).toBeVisible();
  await shot(page, "nhanes-sas-zeros", { at: page.getByTestId("repairs"), themes: ["light", "dark"] });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("repair-settled-sas_zeros")).toBeVisible({ timeout: 30_000 });

  await chooseTarget(page, "glucose");
  await expect(page.getByTestId("skip-task")).toBeVisible({ timeout: 60_000 });
  await answer(page, "purpose", "prediction");

  // The grain, stated: SEQN is unique on every row. Said once, with "Ask me anyway".
  const grain = page.getByTestId("skip-grain");
  await expect(grain).toBeVisible({ timeout: 60_000 });
  await expect(grain).toContainText("Not asked: every SEQN appears once, so each person is one row.");
  await expect(grain.getByText("Not asked:")).toHaveCount(1);
  // Once the roles question has arrived (and the Record has settled on it), look back at the skip.
  await expect(page.getByTestId("question-roles")).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(700);
  await shot(page, "nhanes-grain-stated", { at: page.getByTestId("decision-target"), themes: ["light", "dark"] });

  await confirmRoles(page);
  // Eligibility withholds the outcome's distribution (constitution §04).
  const eligibility = page.getByTestId("question-exclusions");
  await expect(eligibility).toBeVisible({ timeout: 60_000 });
  await expect(eligibility).not.toContainText("glucose");
  // The screens read kcal as one day's intake: the ledger's ask card says so before screening by it.
  await confirmAsk(page.locator('[data-slot="exclusions"]'));
  await eligibility.getByTestId("option-willett_2013_by_sex").focus();
  await previewed(page);
  await landed(page, "with");
  await expect(stage(page).locator("[data-card=distribution]")).not.toContainText("glucose");
  await shot(page, "nhanes-eligibility", { at: eligibility });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-exclusions")).toContainText("Willett", { timeout: 30_000 });

  // meds_hbp's blanks mean "not asked": kept as their own Missing level, recommended with its reason.
  const level = page.getByTestId("option-missing_level");
  await expect(level).toBeVisible({ timeout: 60_000 });
  await expect(level).toContainText("meds_hbp");
  await expect(level).toHaveAttribute("aria-selected", "false");
  await level.focus();
  await previewed(page);
  await landed(page, "with");
  await shot(page, "nhanes-missing-level", { at: page.getByTestId("question-missing"), themes: ["light", "dark"] });
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-missing")).toContainText("Missing", { timeout: 30_000 });

  await seal(page, "one_row_per_unit", "nhanes-seal", ["light", "dark"]);
  T.fit_s = await fit(page);
  await shot(page, "nhanes-results-sealed");
  T.open_seal_s = await openSeal(page, "nhanes");

  // A later change still refits, and is marked post-seal in the Record and the Results. Under
  // prediction no coefficient is read as an effect, so there is no energy model to change: the
  // eligibility rule is the earlier answer a researcher might revisit after seeing the scores.
  await page.getByRole("button", { name: /^Change the (eligibility|exclusions)$/ }).click();
  // "change" takes the user to the reopened question's heading; the options follow it.
  await expect(page.getByTestId("question-exclusions").locator("h2")).toBeFocused();
  const keepEveryRow = page.getByTestId("option-none");
  await keepEveryRow.focus();
  await previewed(page, "Keep every row");
  await expect(keepEveryRow, "the option keeps focus while its preview loads").toBeFocused();
  await page.keyboard.press("Enter");
  const exclusions = page.getByTestId("decision-exclusions");
  // (A post-seal sentence opens with when it was made: "After the held-out rows were opened, no …".)
  await expect(exclusions).toContainText(/no rows were excluded/i, { timeout: 30_000 });
  await expect(exclusions.getByTestId("post-seal-mark")).toBeVisible({ timeout: 30_000 });
  await page.keyboard.press("Escape");
  // The refit's wait lasts while the fit job runs (it is not a flat one).
  T.post_seal_refit_s = await timed(() =>
    visibleWhileComputing(page, page.locator('[data-testid="results"][data-seal-phase="post_seal"]'), ["fit"], {
      quiet: FIT_QUIET,
      cap: FIT_CAP,
      label: "the refit is post-seal",
    }),
  );
  await expect(page.getByTestId("post-seal")).toContainText("eligibility rules");
  await page.getByTestId("post-seal").scrollIntoViewIfNeeded();
  await shot(page, "nhanes-post-seal", { themes: ["light", "dark"] });
  finish("nhanes", run);
});
