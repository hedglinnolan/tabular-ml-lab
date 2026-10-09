/**
 * The questions (calm.html#/qa): a newcomer's walk from the first draft to the locked Table 2 and
 * what mattered. Each question opens on the card; the person points at the scenario's option (the
 * canvas previews it), chooses it and continues. On the way no estimate is on screen before the
 * lock (the leash), the page never scrolls sideways, and no request reaches /api/. After the lock
 * the manuscript rail opens over the card column with the record and closes again.
 */
import { expect, type Page } from "@playwright/test";
import { SCENARIO, proceed, watchApi } from "./scenario";
import type { Walker } from "./walker";

const seen = new WeakMap<object, string[]>();

async function noSidewaysScroll(page: Page) {
  const [scroll, width] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(scroll, "the page's width").toBeLessThanOrEqual(width);
}

export const walker: Walker = {
  name: "The questions",
  path: "/calm.html#/qa",
  toTable2: async (page) => {
    seen.set(page, watchApi(page));
    const card = page.getByTestId("card");
    for (const [step, option] of SCENARIO) {
      await expect(card).toHaveAttribute("data-step", step);
      await expect(page.getByTestId("canvas")).toBeVisible();
      await expect(page.getByTestId("table2"), "an estimate before the lock").toHaveCount(0);
      const opt = page.getByTestId(`opt-${option}`);
      await opt.hover();
      await opt.click();
      await proceed(page).click();
    }
    await expect(page.getByTestId("table2")).toBeVisible();
    await noSidewaysScroll(page);
  },
  toMattered: async (page) => {
    const rail = page.getByTestId("manuscript-rail");
    if (await rail.isVisible()) {
      await rail.click();
      await expect(page.getByTestId("manuscript-overlay")).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.getByTestId("manuscript-overlay")).toHaveCount(0);
    }
    await proceed(page).click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    await noSidewaysScroll(page);
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
