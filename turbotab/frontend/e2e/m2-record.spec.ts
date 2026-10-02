/**
 * M2 part 2, the Record (M2_CONTRACT §10), against the mock (src/mocks/m2-record.ts layered on
 * m1-record.ts). Four journeys, one per thing the Record learned in M2:
 *
 *   NHANES       repairs before the outcome (preview, defer, apply, change) → the grain stated
 *                ("Not asked: every SEQN appears once"; Ask me anyway) → eligibility with the
 *                held finding resurfacing pre-checked, and no outcome distribution → missing values
 *                by mechanism (blanks as a `Missing` level, recommended) → the seal with its basis
 *                and the held-out-size consequence → the fit → open the seal once → a later change
 *                marked post-seal
 *   dietary      the grain asked (participant_id repeats) → repeats stated → the unit → the
 *                aggregation menu, mean recommended with its reason
 *   metabolomics a table exported features-in-rows: orientation asked, the outcome withheld until
 *                it is answered, then the turned table's columns offered
 *   genomics     2,000 columns: the roles question searched, not scrolled
 *
 * Screenshots: docs/turbotab-next/m2/screens/record-*.png (1440 × 900, each < 300 KB).
 */
import { mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCREENS = resolve(HERE, "../../../docs/turbotab-next/m2/screens");
const MAX_SCREEN_BYTES = 300 * 1024;

const stage = (page: Page) => page.getByTestId("stage");

/** Move the pointer to a neutral place so no hover preview lingers in a screenshot. */
const park = (page: Page) => page.mouse.move(1430, 890);

async function shoot(page: Page, name: string, at?: Locator) {
  mkdirSync(SCREENS, { recursive: true });
  if (at) await at.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await page.waitForTimeout(450); // let settle and arrive finish
  const path = resolve(SCREENS, `record-${name}.png`);
  await page.screenshot({ path, animations: "disabled" });
  expect(statSync(path).size, `${name} screenshot size`).toBeLessThan(MAX_SCREEN_BYTES);
}

async function both(page: Page, name: string, at?: Locator) {
  await page.emulateMedia({ colorScheme: "light" });
  await shoot(page, `${name}-light`, at);
  await page.emulateMedia({ colorScheme: "dark" });
  await shoot(page, `${name}-dark`, at);
  await page.emulateMedia({ colorScheme: "light" });
}

async function open(page: Page, name: RegExp) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  const health = await page.evaluate(
    async () => (await (await fetch("/api/health")).json()) as { version: string },
  );
  test.skip(!health.version.endsWith("-mock"), "these journeys run on the mock's tables");
  await page.getByRole("link", { name }).first().click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
}

test("NHANES: repairs, the grain stated, eligibility, missing by mechanism, and the seal opened once", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.emulateMedia({ colorScheme: "light" });
  await open(page, /nhanes diet glucose/);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 20_000 });
  await page.getByTestId("option-dietary").click();
  await page.getByTestId("option-clinical").click();
  await page.getByTestId("record-lens").click();

  // ── repairs, before the outcome (OPENING_SEQUENCE §01) ──────────────────────
  const repairs = page.getByTestId("repairs");
  await expect(repairs).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("repairs-open-count")).toHaveText("2 to decide");
  // The repairs sit above the outcome question in the flow.
  const repairsBox = (await repairs.boundingBox())!;
  const targetBox = (await page.getByTestId("question-target").boundingBox())!;
  expect(repairsBox.y).toBeLessThan(targetBox.y);

  const kcal = page.getByTestId("repair-pack::clinical::impossible_vs_extreme");
  await expect(kcal).toContainText("impossible values outside");
  // Focusing an option previews it on the stage; nothing is recorded.
  const setMissing = kcal.locator('[role="option"]').first();
  await setMissing.focus();
  await expect(stage(page)).toHaveAttribute("data-focus", "option");
  await page.keyboard.press("ArrowDown");
  await expect(kcal.locator('[role="option"]').nth(1)).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(stage(page)).toHaveAttribute("data-focus", "live");
  await setMissing.hover();
  await both(page, "repairs", repairs);
  await park(page);

  // Defer: held for the question it targets, settled here as a sentence.
  await kcal.getByTestId("repair-defer").click();
  const held = page.getByTestId("repair-settled-pack::clinical::impossible_vs_extreme");
  await expect(held).toBeVisible();
  await expect(held).toContainText(/eligibility/i);
  await expect(page.getByTestId("repairs-open-count")).toHaveText("1 to decide");

  // Apply: the binary text column's level, recorded with the server's sentence.
  const gender = page.getByTestId("repair-binary_text__gender");
  await gender.locator('[role="option"]').first().click();
  const genderSettled = page.getByTestId("repair-settled-binary_text__gender");
  await expect(genderSettled).toContainText("female");
  await expect(page.getByTestId("repairs-open-count")).toHaveCount(0);
  // "change" reopens it; nothing changes until a choice is made, and keeping closes it again.
  await genderSettled.getByRole("button", { name: /change/i }).click();
  await expect(page.getByTestId("repair-binary_text__gender")).toContainText("reopened");
  await page.getByTestId("repair-binary_text__gender").getByTestId("repair-keep").click();
  await expect(genderSettled).toBeVisible();

  // ── the outcome, the task stated, the purpose ──────────────────────────────
  await page.getByRole("combobox").fill("glucose");
  await page.getByRole("combobox").press("Enter");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText("glucose was chosen");
  await expect(page.getByTestId("skip-task")).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("option-prediction").click({ timeout: 15_000 });

  // ── the grain, stated: a recognized identifier is unique on every row ──────
  const grain = page.getByTestId("skip-grain");
  await expect(grain).toBeVisible({ timeout: 15_000 });
  await expect(grain).toContainText(
    "Not asked: every SEQN appears once, so each person is one row.",
  );
  await expect(grain.getByText("Not asked:")).toHaveCount(1); // said once, never twice
  // "Ask me anyway" reopens it as a question; nothing is pre-selected.
  await grain.getByRole("button", { name: "Ask me anyway" }).click();
  const grainQ = page.getByTestId("question-grain");
  await expect(grainQ).toBeVisible();
  await expect(page.getByTestId("option-one_row_per_unit")).toHaveAttribute(
    "aria-selected",
    "false",
  );
  await expect(page.getByTestId("option-unknown")).toBeVisible();
  await page.getByTestId("keep").click();
  await expect(page.getByTestId("skip-grain")).toBeVisible();
  await park(page);
  await both(page, "grain-stated", page.getByTestId("decision-purpose"));

  // ── the roles ───────────────────────────────────────────────────────────────
  await expect(page.getByTestId("question-roles")).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("record-roles").click();

  // ── eligibility: the held finding resurfaces, pre-checked and attributed ──
  const eligibility = page.getByTestId("question-exclusions");
  await expect(eligibility).toBeVisible({ timeout: 15_000 });
  const resurfaced = eligibility.getByTestId("resurfaced");
  await expect(resurfaced).toContainText(/You set this aside at #\d+/);
  await expect(
    resurfaced.getByTestId("held-check-pack::clinical::impossible_vs_extreme"),
  ).toBeChecked();
  // The outcome's distribution is withheld (constitution §04): never named on the card.
  await expect(eligibility).not.toContainText("glucose");
  await expect(eligibility.locator("svg")).toHaveCount(0);
  await park(page);
  await both(page, "eligibility", eligibility);
  await page.getByTestId("option-none").click();
  await expect(page.getByTestId("decision-exclusions")).toContainText("No rows were excluded");
  // Recording the question applied the checked repair: the held finding settles into its sentence.
  await expect(held).toContainText("set to missing", { timeout: 10_000 });

  // ── missing values by mechanism ───────────────────────────────────────────
  const level = page.getByTestId("option-missing_level");
  await expect(level).toBeVisible({ timeout: 15_000 });
  await expect(level).toContainText("recommended");
  await expect(level).toContainText("Missing");
  await expect(level).toContainText("not asked");
  await expect(level).toHaveAttribute("aria-selected", "false");
  await level.hover();
  await both(page, "missing", page.getByTestId("question-missing"));
  await level.click();
  await expect(page.getByTestId("decision-missing")).toContainText("Missing");

  // ── the seal: its basis, and what a holdout that size can measure ─────────
  const seal = page.getByTestId("question-split");
  await expect(seal).toBeVisible({ timeout: 15_000 });
  const basis = seal.getByTestId("seal-basis");
  // Grain stated one row per person: the seal says so (never a bare lock).
  await expect(basis).toHaveAttribute("data-basis", "one_row_per_unit");
  await expect(basis).toContainText("The seal");
  const twenty = page.getByTestId("option-0.2");
  await expect(twenty).toContainText(/±|known to about|measure/);
  await twenty.hover();
  await both(page, "seal", seal);
  await twenty.click();
  await expect(page.getByTestId("decision-split")).toBeVisible();

  // ── energy, the models, the fit ───────────────────────────────────────────
  await expect(page.getByTestId("question-energy_adjustment")).toBeVisible({ timeout: 15_000 });
  await page.getByTestId("option-residual").click();
  await expect(page.getByTestId("question-models")).toBeVisible({ timeout: 15_000 });
  // Honest cost: each family's measured fit time is on its option before anything is fit.
  await expect(page.getByTestId("cost-elastic_net")).toHaveText(/^about \d+ s$/);
  await page.getByTestId("option-elastic_net").focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Space");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-models")).toBeVisible();
  // The substitution, from the stage's matrix; opening the seal comes after it.
  await expect(page.getByTestId("question-substitution")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("next")).toContainText("Opening the seal");
  await page.getByRole("button", { name: "Move energy from fat_total to carb" }).click();
  await expect(page.getByTestId("decision-substitution")).toContainText("fat_total", {
    timeout: 30_000,
  });

  // ── open the seal: the Router's last step, a CONSEQUENCE card ─────────────
  const card = page.getByTestId("open-seal-card");
  await expect(card).toBeVisible({ timeout: 30_000 });
  await expect(card).toContainText("It happens once.");
  await expect(card).toContainText("held-out rows");
  // Until it is pressed, no held-out score is on screen.
  await expect(page.getByTestId("open-seal")).toBeEnabled({ timeout: 20_000 });
  await park(page);
  await both(page, "open-seal", card);
  await page.getByTestId("open-seal").click();
  const opened = page.getByTestId("decision-open_seal");
  await expect(opened).toBeVisible({ timeout: 15_000 });
  // The seal opens once: its sentence has no "change".
  await expect(opened.getByRole("button", { name: /change/i })).toHaveCount(0);
  await expect(page.getByTestId("open-seal-card")).toHaveCount(0);

  // A later change still recomputes, and is marked post-seal.
  await page.getByRole("button", { name: "Change the energy adjustment" }).click();
  await page.getByTestId("option-density").click();
  const energy = page.getByTestId("decision-energy_adjustment");
  await expect(energy).toContainText("density");
  await expect(energy.getByTestId("post-seal-mark")).toBeVisible();
  await park(page);
  await both(page, "post-seal", page.getByTestId("decision-models"));
});

test("dietary recalls: the grain asked, repeats stated, the unit, and the aggregation recommended", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await page.emulateMedia({ colorScheme: "light" });
  await open(page, /dietary recalls/);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 20_000 });
  await page.getByTestId("option-dietary").click();
  await page.getByTestId("record-lens").click();

  // A repair dismissed: nothing changes, and the record keeps that it was seen.
  const sex = page.getByTestId("repair-binary_text__sex");
  await expect(sex).toBeVisible({ timeout: 15_000 });
  await sex.getByTestId("repair-dismiss").click();
  await expect(page.getByTestId("repair-settled-binary_text__sex")).toBeVisible();

  await page.getByRole("combobox").fill("hba1c");
  await page.getByRole("combobox").press("Enter");
  await page.getByTestId("record-target").click();
  await page.getByTestId("option-prediction").click({ timeout: 15_000 });

  // The grain is asked: participant_id repeats, so it is suggested beside its option, never chosen.
  const grain = page.getByTestId("question-grain");
  await expect(grain).toBeVisible({ timeout: 15_000 });
  await expect(grain).toContainText("participant_id has 300 values across 600 rows");
  await expect(page.getByTestId("option-repeated")).toContainText("suggested");
  await expect(page.getByTestId("option-repeated")).toHaveAttribute("aria-selected", "false");
  await expect(page.getByTestId("grain-id-participant_id")).toHaveAttribute("aria-pressed", "true");
  // Saying "one row each" would contradict the data: the coach says what it would cost.
  await expect(page.getByTestId("grain-contradiction")).toContainText("both sides of the seal");
  await park(page);
  await both(page, "grain-asked", grain);
  await page.getByTestId("option-repeated").click();
  await expect(page.getByTestId("decision-grain")).toContainText("participant_id");

  // What repeats is stated from the dates, with "Ask me anyway".
  const repeats = page.getByTestId("skip-repeat_kind");
  await expect(repeats).toBeVisible({ timeout: 15_000 });
  await expect(repeats).toContainText("Not asked:");
  await expect(repeats).toContainText("repeated measurements");

  // The unit: no default; the rows each answer leads to, side by side.
  const unit = page.getByTestId("question-unit");
  await expect(unit).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("option-unit")).toContainText("300 rows");
  await expect(page.getByTestId("option-row")).toContainText("600 rows");
  await expect(page.getByTestId("option-unit")).toHaveAttribute("aria-selected", "false");
  await page.getByTestId("option-unit").click();
  await expect(page.getByTestId("decision-unit")).toBeVisible();

  // The aggregation: replicates, so the mean is recommended with its reason, first.
  const agg = page.getByTestId("question-aggregation");
  await expect(agg).toBeVisible({ timeout: 15_000 });
  const mean = page.getByTestId("option-mean");
  await expect(mean).toContainText("recommended");
  await expect(mean).toHaveAttribute("aria-selected", "false");
  const first = agg.locator('[role="option"]').first();
  await expect(first).toHaveAttribute("data-testid", "option-mean");
  await park(page);
  await both(page, "aggregation", agg);
  await mean.click();
  await expect(page.getByTestId("decision-aggregation")).toBeVisible();
  await expect(page.getByTestId("decision-aggregation")).toContainText("600 rows became 300");
  // Temporal prediction does not apply to repeats of one measurement: said in place.
  await expect(page.getByTestId("na-temporal")).toContainText("repeats of one measurement");
  await expect(page.getByTestId("question-roles")).toBeVisible({ timeout: 15_000 });
});

test("metabolomics exported features-in-rows: orientation first, the outcome withheld", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  await open(page, /metabolomics features in rows/);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 20_000 });
  await page.getByTestId("option-metabolomics").click();
  await page.getByTestId("record-lens").click();

  const orientation = page.getByTestId("question-orientation");
  await expect(orientation).toBeVisible({ timeout: 15_000 });
  // Read from the shape, beside its option; never chosen for the user.
  await expect(page.getByTestId("option-feature_major")).toContainText("read from the shape");
  await expect(page.getByTestId("option-feature_major")).toHaveAttribute("aria-selected", "false");
  // The outcome waits: turned around, the columns are samples.
  await expect(page.getByTestId("target-withheld")).toBeVisible();
  await expect(page.getByTestId("question-target")).toHaveCount(0);
  // Findings wait too: every check would read across the wrong axis.
  await expect(page.getByTestId("findings-await-orientation")).toBeVisible();
  await park(page);
  await both(page, "orientation", orientation);
  await page.getByTestId("option-feature_major").click();
  await expect(page.getByTestId("decision-orientation")).toBeVisible();
  // The turned table's columns: one per feature, a row per sample.
  const target = page.getByTestId("question-target");
  await expect(target).toBeVisible({ timeout: 15_000 });
  await expect(target).toContainText("responder");
  await expect(page.getByTestId("findings-await-orientation")).toHaveCount(0);
});

test("a wide table's roles are searched, not scrolled", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await open(page, /genomics counts wide/);
  // The seeded genomics project already answered its lens; the outcome is next.
  await expect(page.getByTestId("decision-lens")).toBeVisible({ timeout: 20_000 });
  await page.getByRole("combobox").fill("condition");
  await page.getByRole("combobox").press("Enter");
  await page.getByTestId("record-target").click();
  // A binary outcome: which level is the event is asked, never guessed.
  const event = page.getByTestId("question-event");
  await expect(event).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("option-case")).toHaveAttribute("aria-selected", "false");
  await expect(page.getByTestId("option-case")).toContainText("30 rows");
  await page.getByTestId("option-case").click();
  await expect(page.getByTestId("decision-event")).toContainText("case");
  await page.getByTestId("option-prediction").click({ timeout: 20_000 });
  await expect(page.getByTestId("skip-grain")).toContainText("sample_id", { timeout: 15_000 });

  const roles = page.getByTestId("question-roles");
  await expect(roles).toBeVisible({ timeout: 20_000 });
  const search = page.getByTestId("roles-search");
  await expect(search).toBeVisible();
  await expect(page.getByTestId("roles-search-count")).toHaveText("1,999 columns");
  const t0 = Date.now();
  await search.fill("ENSG0000010");
  await expect(page.getByTestId("roles-search-count")).toHaveText(/^\d[\d,]* of 1,999$/);
  expect(Date.now() - t0, "roles search").toBeLessThan(2_000);
  await search.fill("ENSG00000100017");
  await expect(page.getByTestId("roles-search-count")).toHaveText("1 of 1,999");
  await expect(page.getByTestId("role-chip-ENSG00000100017")).toBeVisible();
  await park(page);
  await both(page, "roles-search", roles);
  await search.fill("no such gene");
  await expect(roles).toContainText("No column name holds those words.");
});
