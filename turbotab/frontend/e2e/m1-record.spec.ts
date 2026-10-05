/**
 * M1 part 2, the Record and the banner (M1_CONTRACT §10, §14 "record"), against the mock
 * (src/mocks/m1-record.ts, an NHANES-shaped table). One journey through every M1 question:
 *
 *   the lens → the outcome → (the task, not asked) → the purpose → the roles → exclusions →
 *   missing values → the split → energy adjustment (a refusal and its exit) → the models →
 *   the fit → the substitution, with the banner tracking where you are; then a finding's
 *   lever and a changed answer that veils and re-flows the banner.
 *
 * Along the way it checks the M0 review minors: focus moves to the arriving question and the
 * recorded sentence is announced; FACT and CHOICE wear different silhouettes; the findings
 * count sits inside the stale veil. Screenshots: docs/turbotab-next/m1/screens/record-*.png.
 */
import { mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCREENS = resolve(HERE, "../../../docs/turbotab-next/m1/screens");
const MAX_SCREEN_BYTES = 300 * 1024;

const record = (page: Page) => page.locator('main[aria-label="The record"]');
const stage = (page: Page) => page.getByTestId("stage");

interface ShotOptions {
  /** Scroll the Record to its top. */
  top?: boolean;
  /** Scroll the Record so this block starts at the top of its column. */
  at?: Locator;
}

async function shoot(page: Page, name: string, opts: ShotOptions = {}) {
  mkdirSync(SCREENS, { recursive: true });
  if (opts.top) await record(page).evaluate((el) => (el.scrollTop = 0));
  if (opts.at) await opts.at.evaluate((el) => el.scrollIntoView({ block: "start" }));
  await page.waitForTimeout(450); // let settle and arrive finish
  const path = resolve(SCREENS, `record-${name}.png`);
  await page.screenshot({ path, animations: "disabled" });
  expect(statSync(path).size, `${name} screenshot size`).toBeLessThan(MAX_SCREEN_BYTES);
}

async function both(page: Page, name: string, opts: ShotOptions = {}) {
  await page.emulateMedia({ colorScheme: "light" });
  await shoot(page, `${name}-light`, opts);
  await page.emulateMedia({ colorScheme: "dark" });
  await shoot(page, `${name}-dark`, opts);
  await page.emulateMedia({ colorScheme: "light" });
}

/** Move the pointer to a neutral place so no hover preview lingers in a screenshot. */
const park = (page: Page) => page.mouse.move(1430, 890);

async function openNhanes(page: Page) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  const health = await page.evaluate(
    async () => (await (await fetch("/api/health")).json()) as { version: string },
  );
  test.skip(
    !health.version.endsWith("-mock"),
    "this journey runs on the mock's NHANES-shaped table",
  );
  await page
    .getByRole("link", { name: /nhanes diet glucose/ })
    .first()
    .click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
  await expect(page.getByTestId("question-lens")).toBeVisible({ timeout: 20_000 });
}

test("the record follows the Router from the lens to the substitution", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await openNhanes(page);

  // The banner is the map of where you are: the lens acts on the rows.
  const rows = page.getByTestId("banner-rows");
  await expect(rows).toHaveAttribute("data-now", "true");
  await expect(rows).toContainText("21,849");
  await expect(page.getByTestId("banner-result-value")).toHaveCount(0);

  // FACT: flat, the teal marker on the question being asked.
  const lens = page.getByTestId("question-lens");
  await expect(lens).toHaveAttribute("data-grammar", "fact");
  await expect(lens).toHaveAttribute("data-now", "true");
  await expect(page.getByTestId("option-dietary")).toContainText("suggested");

  // Keyboard: focusing an option previews it on the stage; arrows move; Escape goes live.
  await page.getByTestId("option-dietary").focus();
  await expect(stage(page)).toHaveAttribute("data-focus", "option");
  await expect(page.getByTestId("stage-title")).toContainText("Dietary intake");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("option-clinical")).toBeFocused();
  await expect(page.getByTestId("stage-title")).toContainText("Clinical measurements");
  await page.keyboard.press("Escape");
  await expect(stage(page)).toHaveAttribute("data-focus", "live");
  await page.keyboard.press("ArrowUp");
  await page.keyboard.press("Space");
  await expect(page.getByTestId("option-dietary")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Enter");

  // Settle: the sentence is the server's, announced; focus moves to the arriving question.
  await expect(page.getByTestId("decision-lens")).toContainText(
    "The table was read through the dietary lens.",
  );
  await expect(page.getByTestId("announce")).toHaveText(
    "Recorded: The table was read through the dietary lens.",
  );
  const target = page.getByTestId("question-target");
  await expect(target.getByRole("heading")).toBeFocused();
  await expect(stage(page)).toHaveAttribute("data-focus", "live");

  // The outcome.
  await page.getByRole("combobox").fill("glucose");
  await page.getByRole("combobox").press("Enter");
  await expect(stage(page)).toHaveAttribute("data-focus", "option");
  await page.getByTestId("record-target").click();
  await expect(page.getByTestId("decision-target")).toContainText(
    "glucose was chosen as the outcome; it is measured on all 21,849 rows.",
  );

  // The task is detected with high confidence: not asked, said so, never green.
  await expect(page.getByTestId("skip-task")).toContainText("Not asked", { timeout: 15_000 });
  await expect(page.getByTestId("skip-task")).toContainText("regression");
  // "Ask me anyway" opens it; detection is a tag beside an option, never a selection.
  await page.getByTestId("skip-task").getByRole("button", { name: "Ask me anyway" }).click();
  await expect(page.getByTestId("question-task")).toBeVisible();
  await expect(page.getByTestId("option-regression")).toContainText("detected");
  await expect(page.getByTestId("option-regression")).toHaveAttribute("aria-selected", "false");
  await page.getByTestId("keep").click();
  await expect(page.getByTestId("skip-task")).toBeVisible();

  // The purpose.
  await page.getByTestId("option-prediction").click();
  await expect(page.getByTestId("decision-purpose")).toContainText("declared for prediction");

  // The roles: a grouped confirmation of the proposal, nested nutrients and flags shown.
  const roles = page.getByTestId("question-roles");
  await expect(roles).toBeVisible({ timeout: 15_000 });
  await expect(roles.getByRole("heading")).toBeFocused();
  await expect(page.getByTestId("banner-columns")).toHaveAttribute("data-now", "true");
  await expect(page.getByTestId("role-chip-sugar")).toContainText("⊂ carb");
  await expect(page.getByTestId("role-chip-fat_sat")).toContainText("⊂ fat_total");
  await expect(page.getByTestId("role-chip-imputed_bmi")).toContainText("→ bmi");
  await page.getByTestId("role-chip-meds_hbp").click();
  await expect(page.getByTestId("role-menu")).toContainText("Medication use");
  await page.getByTestId("role-excluded").hover();
  await expect(page.getByTestId("stage-title")).toContainText("meds_hbp as excluded");
  await page.getByTestId("role-chip-meds_hbp").click(); // closes the menu; nothing changed
  await park(page);
  await both(page, "roles", { at: roles });
  await page.getByTestId("record-roles").click();
  await expect(page.getByTestId("decision-roles")).toContainText(
    "Column roles were set for 28 columns",
  );

  // Exclusions: CHOICE, a bordered card; each screen counted on this table.
  const exclusions = page.getByTestId("question-exclusions");
  await expect(exclusions).toBeVisible({ timeout: 15_000 });
  await expect(exclusions).toHaveAttribute("data-grammar", "choice");
  await expect(page.getByTestId("option-sex_neutral_500_5000")).toContainText("−501 of 21,849");
  await page.getByTestId("option-sex_neutral_500_5000").click();
  await expect(page.getByTestId("decision-exclusions")).toContainText(
    "501 rows with kcal outside 500–5000 were excluded",
  );

  // Missing values: the mostly-blank yes/no columns can be left out first.
  const leaveOut = page.getByTestId("option-leave_out");
  await expect(leaveOut).toBeVisible({ timeout: 15_000 });
  await expect(leaveOut).toContainText("meds_chol");
  await expect(leaveOut).toContainText("meds_hbp");
  await page.getByTestId("option-complete_case").click();
  await expect(page.getByTestId("decision-missing")).toContainText("2,943 of 21,348 rows remain");

  // The split. The banner's flow reads the cohort and the split.
  await page.getByTestId("option-0.2").click();
  await expect(page.getByTestId("decision-split")).toContainText("A random 20% of the 2,943 rows");
  await expect(page.getByTestId("banner-rows-1")).toHaveText("21,348", { timeout: 15_000 });
  await expect(page.getByTestId("banner-rows-2")).toHaveText("2,943");
  await expect(page.getByTestId("banner-train")).toBeVisible({ timeout: 15_000 });

  // Energy adjustment: the usual method tagged, never pre-selected; partition does not apply
  // here, and pressing it answers at the option with the server's exits.
  const energy = page.getByTestId("question-energy_adjustment");
  await expect(energy).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("option-residual")).toContainText("usual");
  await expect(page.getByTestId("option-residual")).toHaveAttribute("aria-selected", "false");
  await expect(page.getByTestId("option-partition")).toHaveAttribute("aria-disabled", "true");
  await energy.getByTestId("why-energy_adjustment").click();
  await expect(energy.getByTestId("why-panel-energy_adjustment")).toContainText("estimand");
  await energy.getByTestId("drawer-energy_adjustment").click();
  await expect(page.getByTestId("concept-drawer")).toContainText("Fit inside the folds");
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("concept-drawer")).toHaveCount(0);
  await energy.getByTestId("why-energy_adjustment").click();
  await page.getByTestId("option-residual").hover();
  await expect(page.getByTestId("stage-title")).toContainText("Residual method");
  await both(page, "energy", { at: energy });
  // aria-disabled, yet still on the shelf: a press (here, Enter) is answered, never ignored.
  await page.getByTestId("option-partition").focus();
  await page.keyboard.press("Enter");
  const refusal = energy.getByTestId("refusal");
  await expect(refusal).toContainText("cannot run on these columns");
  await park(page);
  await shoot(page, "refusal-light");
  await refusal.getByRole("button", { name: /Willett residual model(, total energy kept)? instead/ }).click();
  await expect(page.getByTestId("decision-energy_adjustment")).toContainText(
    "Energy was adjusted by the residual method",
  );

  // Model families: chosen with the keyboard, none pre-selected, concerns stated.
  const models = page.getByTestId("question-models");
  await expect(models).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("banner-models")).toHaveAttribute("data-now", "true");
  await page.getByTestId("option-elastic_net").focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Space");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Space");
  await expect(page.getByTestId("record-models")).toHaveText("Fit these 3 families");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("decision-models")).toContainText(
    "Three model families were chosen",
  );

  // The fit: the banner's result arrives; the substitution opens on the stage's matrix.
  await expect(page.getByTestId("banner-result-value")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("banner-models")).toContainText("elastic net");
  await expect(page.getByTestId("banner-columns-to")).toBeVisible();
  await expect(page.getByTestId("banner-columns")).toContainText("residual");
  const substitution = page.getByTestId("question-substitution");
  await expect(substitution).toBeVisible({ timeout: 20_000 });
  await expect(substitution).toContainText("matrix on the stage");
  await expect(page.getByTestId("banner-result-value")).toBeVisible();
  await park(page);
  await both(page, "journey", { top: true });
  await both(page, "substitution");

  // Findings: at most three pushed, drawn from the ones still open; a finding whose question
  // has been answered is settled — folded into one green line, its evidence still a press away
  // (review: the cards kept pushing levers for settled questions). The count badge lives inside
  // the stale veil.
  const findings = page.getByTestId("findings");
  await findings.scrollIntoViewIfNeeded();
  await expect(page.getByTestId("veil-findings").getByTestId("findings-count")).toBeVisible();
  const cards = page.getByTestId("finding-cards").locator(":scope > li");
  expect(await cards.count()).toBeLessThanOrEqual(3);
  const settled = page.getByTestId("findings-settled");
  await expect(settled).toContainText(/\d+ answered in the record — /);
  await page.getByTestId("findings-settled-toggle").click();
  const energyFinding = settled
    .locator('[data-testid^="settled-"]')
    .filter({ hasText: /Answered by #\d+: Energy was adjusted/ })
    .first();
  await expect(energyFinding).toBeVisible();
  await expect(settled.getByTestId("lever")).toHaveCount(0);
  await energyFinding.focus();
  await expect(stage(page)).toHaveAttribute("data-focus", "finding");
  await park(page);
  await both(page, "findings");
  await page.getByTestId("findings-settled-toggle").click();

  // Change an earlier answer: the banner's later segments veil in order, then re-flow.
  await page.route("**/api/projects/*/stages/**", async (route) => {
    await new Promise((r) => setTimeout(r, 900));
    await route.continue();
  });
  await page.getByRole("button", { name: "Change the energy adjustment" }).click();
  await page.getByTestId("option-density").click();
  await expect(page.getByTestId("decision-energy_adjustment")).toContainText(
    "nutrient density model",
  );
  const columnsVeil = page.getByTestId("banner-columns").locator("[data-veil]");
  await expect(columnsVeil).not.toHaveAttribute("data-veil", "fresh");
  await park(page);
  await page.waitForTimeout(300);
  await page.screenshot({
    path: resolve(SCREENS, "record-stale-light.png"),
    animations: "disabled",
  });
  await page.unroute("**/api/projects/*/stages/**");
  await expect(columnsVeil).toHaveAttribute("data-veil", "fresh", { timeout: 20_000 });
  await expect(
    page.getByTestId("banner-result-value").locator("xpath=ancestor::*[@data-veil][1]"),
  ).toHaveAttribute("data-veil", "fresh", { timeout: 20_000 });
  await expect(page.getByTestId("banner-columns")).toContainText("density");
  // The earlier answer stays in the record, superseded.
  await expect(
    page.locator('[data-slot="energy_adjustment"]').getByLabel("Earlier answers"),
  ).toContainText("residual method");
});

test("a wide table keeps the outcome picker and the roles question responsive", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  const health = await page.evaluate(
    async () => (await (await fetch("/api/health")).json()) as { version: string },
  );
  test.skip(!health.version.endsWith("-mock"), "the wide mock table");
  await page
    .getByRole("link", { name: /genomics counts wide/ })
    .first()
    .click();
  await expect(page).toHaveURL(/\/p\/[^/]+$/);
  // The seeded genomics project already answered its lens; the outcome is next.
  await expect(page.getByTestId("decision-lens")).toBeVisible({ timeout: 20_000 });
  const picker = page.getByRole("combobox");
  await expect(picker).toBeVisible({ timeout: 20_000 });
  const t0 = Date.now();
  await picker.fill("ENSG0000010");
  await expect(page.getByTestId("picker-count")).toHaveText(/ of 2,000$/);
  expect(Date.now() - t0, "picker filter").toBeLessThan(2_000);
  await picker.fill("condition");
  await picker.press("Enter");
  await page.getByTestId("record-target").click();
  // M2: a binary outcome asks which level is the event, never guessed.
  await page.getByTestId("option-case").click({ timeout: 20_000 });
  await page.getByTestId("option-prediction").click({ timeout: 20_000 });
  // M2: `sample_id` names a sample, not a person, so the grain is asked.
  await page.getByTestId("option-one_row_per_unit").click({ timeout: 20_000 });
  // 1,998 exposures: the group shows a few chips and counts the rest.
  const roles = page.getByTestId("question-roles");
  await expect(roles).toBeVisible({ timeout: 20_000 });
  await expect(roles.locator('[data-role="exposure"] li')).toHaveCount(15);
  await expect(roles).toContainText("1,984 more");
  // Energy adjustment does not apply to a count matrix: the Router says so in place.
  await expect(page.getByTestId("next")).toContainText("not applicable");
});
