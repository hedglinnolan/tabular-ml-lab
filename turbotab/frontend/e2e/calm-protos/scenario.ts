/**
 * The scenario's answers (src/explore/methods-shared/SCENARIO.md), as the kit's walk asks them:
 * step id → option id, in order. The card walk below clicks them; a structure's walker may reach
 * the same answers its own way.
 */
import { expect, type Locator, type Page } from "@playwright/test";

export const SCENARIO: [string, string][] = [
  ["unit", "kcal_1"],
  ["exclusions", "none"],
  ["sensitivity", "both"],
  ["missing", "complete_case"],
  ["single:bp_di", "covariate"],
  ["single:bp_sys", "covariate"],
  ["single:cycle_begin_year", "covariate"],
  ["block", "confirm"],
  ["exposure", "sugar"],
  ["effect", "total"],
  ["contrast", "substitution"],
  ["adjust:demographic", "confounder"],
  ["adjust:dietary", "confounder"],
  ["adjust:body", "timing_unknown"],
  ["adjust:unguessed:0", "confounder"],
  ["adjust:unguessed:1", "mediator"],
  ["energy", "standard"],
  ["model1", "guess"],
  ["codes", "confirm"],
  ["lock", "lock"],
];

/** The visible Continue: the card's on wide screens, the footer's on narrow ones. */
export function proceed(page: Page): Locator {
  return page.locator('[data-testid="continue"]:visible, [data-testid="continue-footer"]:visible').first();
}

/** Every /api/ request the page issues (there must be none: the structures run on the fixture). */
export function watchApi(page: Page): string[] {
  const urls: string[] = [];
  page.on("request", (req) => {
    if (new URL(req.url()).pathname.startsWith("/api/")) urls.push(req.url());
  });
  return urls;
}

/** The kit's reference walk: each question on the card, its scenario option, Continue. */
export async function cardWalk(page: Page): Promise<void> {
  for (const [step, option] of SCENARIO) {
    await expect(page.getByTestId("card")).toHaveAttribute("data-step", step);
    await page.getByTestId(`opt-${option}`).click();
    await proceed(page).click();
  }
  await expect(page.getByTestId("table2")).toBeVisible();
}

export async function cardToMattered(page: Page): Promise<void> {
  await proceed(page).click();
  await expect(page.getByTestId("mattered")).toBeVisible();
}
