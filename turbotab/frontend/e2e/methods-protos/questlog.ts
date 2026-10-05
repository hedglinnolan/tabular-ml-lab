/**
 * Prototype B, the quest log: walked by clicks from the first draft. Each objective's own card
 * records the shared scenario's answer and the quest log moves to the next one: the roles, the
 * unit of `kcal`, every row kept with two screens beside it, complete cases, cross-validation only,
 * three role readings one at a time and then the unlocked block, the exposure and its effect, the
 * adjustment groups, the energy phrase, Model 1, the code readings and the model family, and the
 * fit that locks the plan. Nothing is fetched: the walk issues no request to /api/.
 */
import { expect, type Locator, type Page } from "@playwright/test";
import type { Walker } from "./walker";

const requests = new WeakMap<Page, string[]>();

function card(page: Page): Locator {
  return page.getByTestId("objective");
}

async function record(page: Page) {
  await card(page).getByTestId("record").click();
}

/** The section the open objective belongs to, as the card's header names it. */
async function at(page: Page, section: string | RegExp) {
  await expect(page.getByTestId("center").getByText(section).first()).toBeVisible();
}

/** One of the unguessed covariates' answers: its three questions, then its record control. */
async function answerAsked(page: Page, columns: string, answers: [string, string, string]) {
  const row = card(page).locator(`[data-testid="adjust-asked"][data-columns="${columns}"]`);
  const groups = row.getByRole("radiogroup");
  for (const [i, a] of answers.entries()) await groups.nth(i).locator(`[data-answer="${a}"]`).click();
  await row.getByTestId("record-asked").click();
}

export const walker: Walker = {
  name: "B · the quest log",
  path: "/lab/methods-questlog",
  toTable2: async (page) => {
    const seen: string[] = [];
    requests.set(page, seen);
    page.on("request", (r) => {
      if (new URL(r.url()).pathname.startsWith("/api/")) seen.push(r.url());
    });
    const c = card(page);

    // Data sources: the roles as proposed.
    await at(page, "Data sources and measurement");
    await c.getByRole("button", { name: "Record these roles" }).click();

    // Column readings: the screens need kcal's unit.
    await expect(c.getByTestId("reading")).toHaveCount(1);
    await c.getByTestId("confirm-single").click();

    // Participants: every row, with two screens reported beside it.
    await at(page, "Participants");
    await c.getByTestId("primary-none").click();
    await c.getByTestId("screen-willett_2013_by_sex").click();
    await c.getByTestId("screen-nhs_hpfs_by_sex").click();
    await record(page);

    // Statistical methods: complete cases, then cross-validation only.
    await at(page, "Statistical methods");
    await c.getByTestId("option-complete_case").click();
    await record(page);
    await c.getByTestId("option-0").click();
    await record(page);

    // Column readings: three singles in the card's order, then the unlocked block.
    for (const column of ["bp_di", "bp_sys", "cycle_begin_year"]) {
      await expect(c.locator(`[data-testid="reading"][data-column="${column}"] [data-testid="confirm-single"]`)).toBeVisible();
      await c.getByTestId("confirm-single").click();
    }
    await expect(c.getByTestId("unlock")).toBeVisible();
    await c.getByTestId("block-confirm").click();

    // Variables: the exposure and its effect.
    await at(page, "Variables");
    await expect(c.getByTestId("teach-first")).toBeVisible();
    await c.getByTestId("exposure-sugar").click();
    await c.getByTestId("effect-total").click();
    await c.getByTestId("contrast-substitution").click();
    await record(page);

    // The adjustment set: the three guessed groups, then the unguessed covariates.
    for (let i = 0; i < 3; i++) await c.getByTestId("confirm-group").first().click();
    await answerAsked(page, "cycle_begin_year", ["yes", "yes", "no"]);
    await answerAsked(page, "bp_sys,bp_di,hdl,triglycerides,meds_hbp,meds_chol", ["no", "yes", "yes"]);

    // Quantitative variables: the energy phrase (hovering an option plays it on the canvas).
    await at(page, "Quantitative variables");
    await c.getByTestId("option-residual").hover();
    await expect(page.getByTestId("stage")).toContainText(/residual/i);
    await c.getByTestId("option-standard").click();
    await record(page);

    // Statistical methods: Model 1, the code readings, the family.
    for (const col of ["age", "gender", "kcal"]) await c.getByTestId(`model1-${col}`).click();
    await record(page);
    await c.getByTestId("block-confirm").click();
    await c.getByTestId("option-linear").click();
    await record(page);

    // The fit locks the plan.
    await expect(page.getByTestId("table2")).toHaveCount(0);
    await c.getByRole("button", { name: "Fit and lock the plan" }).click();
    await expect(page.getByTestId("table2")).toBeVisible();
  },
  toMattered: async (page) => {
    await page.getByTestId("show-mattered").click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    // The walk drew everything from the bundled capture: no request reached /api/.
    expect(requests.get(page) ?? []).toEqual([]);
  },
};
