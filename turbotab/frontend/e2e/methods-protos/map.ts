/**
 * Prototype C, the map (/lab/methods-map): the shared scenario walked by clicks alone, from the
 * first draft to the locked Table 2 and "Which of my decisions mattered?". "Next asked" goes to
 * each asked node in turn (a ring on the map), its card opens, and the scenario's answer is
 * recorded there; the map's own nodes open the same cards. No request reaches /api/ on the way:
 * the prototype runs on its captured fixture alone.
 */
import { expect, type Page } from "@playwright/test";
import type { Walker } from "./walker";

/** Every /api/ request the page issues while it is walked. */
const seen = new WeakMap<Page, string[]>();

function watch(page: Page): void {
  const urls: string[] = [];
  seen.set(page, urls);
  page.on("request", (req) => {
    if (new URL(req.url()).pathname.startsWith("/api/")) urls.push(req.url());
  });
}

/** "Next asked: <node> →" in the header, and that node's card. */
async function next(page: Page, node: string): Promise<void> {
  await page.getByTestId("next").click();
  await expect(page.getByTestId(`card-${node}`)).toBeVisible();
}

export const walker: Walker = {
  name: "C · the map",
  path: "/lab/methods-map",
  toTable2: async (page) => {
    watch(page);
    const t = (id: string) => page.getByTestId(id);
    await expect(t("intro")).toBeVisible();
    await expect(t("node-readings")).toHaveAttribute("data-tier", "asked");

    // Readings: kcal's unit, three role readings one at a time, the block they unlock, the fit's codes.
    await next(page, "readings");
    await t("confirm-unit").click();
    for (const col of ["bp_di", "bp_sys", "cycle_begin_year"]) await t(`confirm-role:${col}`).click();
    await t("confirm-block").click();
    await t("confirm-codes").click();
    await expect(t("node-readings")).toHaveAttribute("data-tier", "recorded");

    // Exclusions: hovering an option plays it on the canvas; every row is kept, both screens beside.
    await next(page, "exclusions");
    await t("opt-willett_2013_by_sex").hover();
    await expect(t("stage")).toHaveAttribute("data-scene", "preview");
    await t("opt-none").click();
    await t("beside-willett_2013_by_sex").check();
    await t("beside-nhs_hpfs_by_sex").check();

    // Missing data: complete cases, opened from its node on the map.
    await t("node-missing").click();
    await expect(t("card-missing")).toBeVisible();
    await t("opt-complete_case").click();
    await expect(t("node-missing")).toHaveAttribute("data-tier", "recorded");

    // The exposure: sugar's total effect, as a substitution.
    await next(page, "exposure");
    await t("contrast-substitution").click();
    await t("record-exposure").click();

    // The adjustment set: the three guessed groups confirmed, the seven unguessed answered.
    await next(page, "adjustment");
    for (const g of ["demographic", "dietary", "body"]) await t(`record-adj-${g}`).click();
    await t("adj-truth").click();
    await t("record-adj-unguessed").click();

    // Model 1: age, gender, kcal.
    await next(page, "model1");
    await t("model1-guess").click();

    // The lock: the plan is recorded with its SHA-256 as the first estimate is shown.
    await expect(t("next")).toHaveText(/lock the plan/);
    await next(page, "lock");
    await t("lock").click();
    await expect(t("table2")).toBeVisible();
  },
  toMattered: async (page) => {
    await page.getByTestId("tab-matter").click();
    await expect(page.getByTestId("mattered")).toBeVisible();
    expect(seen.get(page) ?? [], "requests to /api/ during the walk").toEqual([]);
  },
};
