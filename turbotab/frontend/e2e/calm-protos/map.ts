/**
 * The map (calm.html#/map): its walk from the first draft to the locked Table 2 and what mattered.
 * The band above the card is the analysis drawn as a lineage, at most 120 px tall: each decision a
 * node, open while asked, solid once stated. The walk answers each card and continues, and reads
 * the map as it goes: the node on the card is the current one, and each answer turns its node
 * solid. Midway it goes back by the map (the eligibility node opens its card on the recorded
 * answer) and returns by the open node; after the lock it opens "Which of my decisions mattered?"
 * from the map's Results region. No request reaches /api/ on the way.
 */
import { expect, type Page } from "@playwright/test";
import { SCENARIO, proceed, watchApi } from "./scenario";
import type { Walker } from "./walker";

const seen = new WeakMap<object, string[]>();

const node = (page: Page, id: string) => page.getByTestId(`map-node-${id}`);

/** The decision on the card is the map's current node. */
async function onCard(page: Page, step: string) {
  await expect(page.getByTestId("card")).toHaveAttribute("data-step", step);
  await expect(node(page, step)).toHaveAttribute("aria-current", "step");
}

export const walker: Walker = {
  name: "The map",
  path: "/calm.html#/map",
  toTable2: async (page) => {
    seen.set(page, watchApi(page));
    const band = await page.getByTestId("map").boundingBox();
    expect(band?.height ?? 999, "the map's band").toBeLessThanOrEqual(120);
    // The first draft: the first decision is asked, the rest wait, and no result can open.
    await expect(node(page, "unit")).toHaveAttribute("data-status", "asked");
    await expect(node(page, "exclusions")).toBeDisabled();
    await expect(node(page, "table2")).toBeDisabled();
    for (const [i, [step, option]] of SCENARIO.entries()) {
      await onCard(page, step);
      await page.getByTestId(`opt-${option}`).click();
      await proceed(page).click();
      await expect(node(page, step)).toHaveAttribute("data-status", "stated");
      if (step === "missing") {
        // Back by the map: a stated node opens its card on the recorded answer …
        await node(page, "exclusions").click();
        await onCard(page, "exclusions");
        await expect(page.getByTestId("opt-none").locator("input")).toBeChecked();
        // … and the open (asked) node brings the walk back to where it was.
        const next = SCENARIO[i + 1]![0];
        await expect(node(page, next)).toHaveAttribute("data-status", "asked");
        await node(page, next).click();
      }
    }
    await expect(page.getByTestId("table2")).toBeVisible();
    await expect(node(page, "table2")).toHaveAttribute("aria-current", "step");
  },
  toMattered: async (page) => {
    await node(page, "mattered").click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    await expect(node(page, "mattered")).toHaveAttribute("aria-current", "step");
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
