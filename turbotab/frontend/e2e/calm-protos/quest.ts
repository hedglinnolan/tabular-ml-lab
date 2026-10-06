/**
 * The quest log (calm.html#/quest): its walk from the first draft to the locked Table 2 and what
 * mattered. Each objective the card holds is answered and continued; the objective line's counts
 * fill as it goes. Once Participants is done, the walk does what the quest log is for: it opens
 * the section's objectives from the line, reopens a recorded one on the card, and comes back to
 * the open slot with "Next objective". No request reaches /api/ on the way.
 */
import { expect } from "@playwright/test";
import { SCENARIO, proceed, watchApi } from "./scenario";
import type { Walker } from "./walker";

const seen = new WeakMap<object, string[]>();

export const walker: Walker = {
  name: "The quest log",
  path: "/calm.html#/quest",
  toTable2: async (page) => {
    seen.set(page, watchApi(page));
    const card = page.getByTestId("card");
    await expect(page.getByTestId("count-participants")).toHaveText("0 of 2");
    for (const [step, option] of SCENARIO) {
      await expect(card).toHaveAttribute("data-step", step);
      await page.getByTestId(`opt-${option}`).click();
      await proceed(page).click();
      if (step === "sensitivity") {
        await expect(page.getByTestId("count-participants")).toHaveText("2 of 2");
        await expect(page.getByTestId("next-objective")).toHaveCount(0);
        await page.getByTestId("section-participants").click();
        await page.getByTestId("objective-exclusions").click();
        await expect(card).toHaveAttribute("data-step", "exclusions");
        await expect(page.getByTestId("opt-none").locator("input")).toBeChecked();
        await page.getByTestId("next-objective").click();
        await expect(page.getByTestId("next-objective")).toHaveCount(0);
      }
    }
    await expect(page.getByTestId("table2")).toBeVisible();
    for (const id of ["participants", "variables", "measurement", "statistics"]) {
      const n = await page.getByTestId(`count-${id}`).innerText();
      const [done, total] = n.split(" of ");
      expect(done, `${id}: every objective recorded`).toBe(total);
    }
  },
  toMattered: async (page) => {
    await proceed(page).click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
