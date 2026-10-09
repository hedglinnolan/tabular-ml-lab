/**
 * Prototype A, the paper (/lab/methods-document): the walk by clicks from the first draft to the
 * locked Table 2, then to "Which of my decisions mattered?". Each step opens the next slot with the
 * toolbar's "Next" (the slot the walk records next is always first) and presses that slot's own
 * controls with the shared scenario's answers (src/explore/methods-shared/SCENARIO.md). On the way
 * it hovers options so the canvas plays their previews, it checks at every slot that the paper's
 * missing-data sentence names the rows the banner shows, and it checks that the page never asks the
 * API anything: the prototype runs from its fixture alone.
 */
import { expect, type Page } from "@playwright/test";
import type { Walker } from "./walker";

/** The scenario's answers for the covariates the pack does not guess: [cause of the exposure,
 *  cause of the outcome, after the exposure] (SCENARIO.md, from turbotab/core/tests/truths.py). */
const UNGUESSED: Record<string, [string, string, string]> = {
  cycle_begin_year: ["yes", "yes", "no"],
  bp_sys: ["no", "yes", "yes"],
  bp_di: ["no", "yes", "yes"],
  hdl: ["no", "yes", "yes"],
  triglycerides: ["no", "yes", "yes"],
  meds_hbp: ["no", "yes", "yes"],
  meds_chol: ["no", "yes", "yes"],
};

const asked = new WeakMap<Page, string[]>();

/** Every request the page makes to the API from here on (none is expected). */
function watch(page: Page): string[] {
  let seen = asked.get(page);
  if (!seen) {
    const list: string[] = [];
    page.on("request", (r) => {
      if (new URL(r.url()).pathname.startsWith("/api/")) list.push(`${r.method()} ${r.url()}`);
    });
    asked.set(page, list);
    seen = list;
  }
  return seen;
}

/** Once the paper's "Missing data." sentence is recorded, it names the rows the banner ends on:
 *  the engine re-renders that sentence when a later answer adds predictors with gaps (the block of
 *  readings takes complete cases from 21,849 rows to 2,996; the adjustment set gives them back). */
async function rowsAgree(page: Page) {
  const para = page.locator("#para-missing");
  if ((await para.getAttribute("data-tier")) !== "recorded") return;
  const rows = (
    await page.getByTestId("banner").locator('[data-testid^="banner-rows-"]').last().innerText()
  ).trim();
  await expect(
    para,
    `the missing-data sentence at ${await page.getByTestId("proto-step").innerText()}`,
  ).toContainText(rows);
}

/** Open the slot the walk records next and wait for its control; the paper so far agrees with the
 *  banner. */
async function next(page: Page, control: string) {
  await page.getByTestId("next-slot").click();
  await expect(page.getByTestId(control)).toBeVisible();
  await rowsAgree(page);
}

async function press(page: Page, testid: string) {
  const b = page.getByTestId(testid);
  await expect(b).toBeEnabled();
  await b.click();
}

/** The canvas is playing a preview the server drew (the production stage, its bar naming the
 *  option, with the flip between the data now and with this choice). */
async function canvasPlays(page: Page, label: RegExp) {
  const canvas = page.getByRole("complementary", { name: "The canvas" });
  await expect(canvas.getByTestId("stage")).toContainText(label);
  await expect(canvas.getByRole("group", { name: "Show the data" })).toBeVisible();
}

export const walker: Walker = {
  name: "A · the paper",
  path: "/lab/methods-document",

  toTable2: async (page) => {
    const api = watch(page);
    const step = page.getByTestId("proto-step");

    // The roles, as proposed.
    await expect(step).toHaveText(/moment 1 of/);
    await next(page, "record-roles");
    await press(page, "record-roles");

    // The screens ask kcal's unit: a reading in the Data section, one press.
    await next(page, "unit-confirm");
    await press(page, "unit-confirm");

    // The rows: hover a screen to watch the canvas, keep every row, both sex-specific screens beside.
    await next(page, "record-exclusions");
    await page.getByTestId("alt-willett_2013_by_sex").hover();
    await canvasPlays(page, /Willett 2013, by sex/);
    await press(page, "alt-keep_every_row");
    await press(page, "beside-willett_2013_by_sex");
    await press(page, "beside-nhs_hpfs_by_sex");
    await press(page, "record-exclusions");

    await next(page, "record-missing");
    await press(page, "alt-complete_case");
    await press(page, "record-missing");

    await next(page, "record-split");
    await press(page, "choice-0");
    await press(page, "record-split");

    // Three readings one at a time unlock the block, which settles the rest.
    await next(page, "confirm-bp_di");
    await press(page, "confirm-bp_di");
    await press(page, "confirm-bp_sys");
    await press(page, "confirm-cycle_begin_year");
    await expect(page.getByTestId("block-unlock")).toBeVisible();
    await press(page, "block-confirm");

    // The question: sugar, its total effect, as a substitution (taught in full here, first met).
    await next(page, "record-estimand");
    await press(page, "exposure-sugar");
    await press(page, "effect-total");
    await press(page, "contrast-substitution");
    await expect(page.getByTestId("concept-full")).toBeVisible();
    await press(page, "record-estimand");

    // The adjustment set: confirm the pack's three guesses, answer the seven it does not guess.
    await next(page, "record-adjustment");
    for (const g of ["demographic", "dietary", "body"]) await press(page, `adj-confirm-${g}`);
    for (const [column, answers] of Object.entries(UNGUESSED))
      for (const [i, a] of answers.entries()) await press(page, `adj-${column}-${i + 1}-${a}`);
    await press(page, "record-adjustment");

    // The energy model: hover an alternative to watch it play, then record the one ranked first.
    await next(page, "record-energy_adjustment");
    await page.getByTestId("alt-residual").hover();
    await canvasPlays(page, /residual model/);
    await press(page, "alt-standard");
    await press(page, "record-energy_adjustment");

    // Model 1: the pack's guess, declared before any estimate.
    await next(page, "record-sequence");
    await press(page, "model1-guess");
    await press(page, "record-sequence");

    // The fit's card: age and cycle_begin_year, amounts or codes, one block.
    await next(page, "block-confirm");
    await press(page, "block-confirm");

    await next(page, "record-models");
    await press(page, "choice-linear");
    await press(page, "record-models");

    // Showing the estimates is the one act left, and it locks the plan.
    await expect(page.getByTestId("table2")).toHaveCount(0);
    await next(page, "lock-plan");
    await press(page, "lock-plan");
    await expect(page.getByTestId("table2")).toBeVisible();
    await rowsAgree(page);
    await expect(step).toHaveText(/moment 17 of 17/);
    expect(api, "requests to the API during the walk").toEqual([]);
  },

  toMattered: async (page) => {
    const api = watch(page);
    await expect(page.getByTestId("mattered")).toHaveCount(0);
    await page.getByTestId("spec-teaser").click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    expect(api, "requests to the API during the walk").toEqual([]);
  },
};
