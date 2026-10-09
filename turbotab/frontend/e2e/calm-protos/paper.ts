/**
 * The paper (calm.html#/paper): its walk from the first draft to the locked Table 2 and what
 * mattered. The manuscript is the interface: each question opens in place, in the section of the
 * paper where its sentence will stand, and Continue at its foot records it and opens the next.
 * Once on the way, a recorded sentence is clicked to reopen its question where it stands. No
 * estimate is on screen before the lock, and no request reaches /api/.
 */
import { expect, type Page } from "@playwright/test";
import { SCENARIO, proceed, watchApi } from "./scenario";
import type { Walker } from "./walker";

const seen = new WeakMap<object, string[]>();

/** The section of the paper each question's sentence stands in (the kit's fixture). */
const SECTION: Record<string, string> = {
  unit: "measurement",
  exclusions: "participants",
  sensitivity: "participants",
  missing: "statistics",
  "single:bp_di": "measurement",
  "single:bp_sys": "measurement",
  "single:cycle_begin_year": "measurement",
  block: "measurement",
  exposure: "variables",
  effect: "variables",
  contrast: "variables",
  "adjust:demographic": "variables",
  "adjust:dietary": "variables",
  "adjust:body": "variables",
  "adjust:unguessed:0": "variables",
  "adjust:unguessed:1": "variables",
  energy: "statistics",
  model1: "statistics",
  codes: "measurement",
  lock: "statistics",
};

/** The open question, where it stands in the paper. */
function slot(page: Page, section: string) {
  return page.getByTestId("manuscript").locator(`[data-section="${section}"]`).getByTestId("card");
}

export const walker: Walker = {
  name: "The paper",
  path: "/calm.html#/paper",
  toTable2: async (page) => {
    seen.set(page, watchApi(page));
    await expect(page.getByTestId("canvas")).toBeVisible();
    for (const [step, option] of SCENARIO) {
      const card = slot(page, SECTION[step]!);
      await expect(card).toHaveAttribute("data-step", step);
      // The leash: no estimate on screen before the plan is locked.
      await expect(page.getByTestId("table2")).toHaveCount(0);
      await card.getByTestId(`opt-${option}`).click();
      await proceed(page).click();
      if (step === "exclusions") {
        // The recorded sentence is the way back: clicking it reopens its question in place, with
        // the answer still chosen; Continue returns to the next open question.
        const sentence = page.getByTestId("ms-exclusions").getByRole("button");
        await expect(sentence).toContainText("No rows were excluded");
        await sentence.click();
        const again = slot(page, "participants");
        await expect(again).toHaveAttribute("data-step", "exclusions");
        await expect(again.getByTestId("opt-none").locator("input")).toBeChecked();
        await proceed(page).click();
      }
    }
    // The results card opens beneath the paper's results paragraph, Table 2 on the canvas.
    await expect(slot(page, "results")).toHaveAttribute("data-step", "table2");
    await expect(page.getByTestId("ms-results:table2")).not.toContainText("Waits for an earlier answer");
    await expect(page.getByTestId("table2")).toBeVisible();
  },
  toMattered: async (page) => {
    await proceed(page).click();
    await expect(slot(page, "results")).toHaveAttribute("data-step", "mattered");
    await expect(page.getByTestId("mattered")).toBeVisible();
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
