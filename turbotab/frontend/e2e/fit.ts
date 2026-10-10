import { expect, test, type Page } from "@playwright/test";

/**
 * Fit, pressed as the person presses it (SIZING P0.8; calm/FOUNDATION §7). Until it is, no score or
 * estimate is served: the Results show one button, and the seal waits for it. A fit estimated
 * over about 2 minutes is held for it in the scheduler (RECIPES §4.4), so nothing is computed and
 * there are no Results to carry the button yet: today's app offers Fit nowhere else, so a journey
 * whose fit is held stops there, skipped with that reason (the blocker the W1 shell removes),
 * instead of waiting out its timeout or pressing what the person cannot.
 */
export const HELD_WITHOUT_FIT =
  "the fit is held for Fit (its estimate exceeds about 2 minutes), and the app offers Fit only on a computed fit's Results: nothing to press yet (W1 blocker, the shell's Fit)";

type FitState = "pending" | "held" | "withheld" | "served";

/** Where the fit stands, read from the API as the stage reads it. */
export async function fitState(page: Page, pid: string): Promise<FitState> {
  return page.evaluate(async (p) => {
    const view = (await (await fetch(`/api/projects/${p}`)).json()) as {
      stages?: Record<string, { held?: string | null }>;
    };
    const res = await fetch(`/api/projects/${p}/stages/fit`);
    const body = res.ok ? ((await res.json()) as { artifact?: { withheld_for?: string | null } | null }) : {};
    if (!body.artifact) return view.stages?.fit?.held === "fit" ? "held" : "pending";
    return body.artifact.withheld_for === "fit" ? "withheld" : "served";
  }, pid);
}

/** The Results' Fit, pressed: the Results are shown on the stage (the banner's result segment,
 *  when the stage shows something else), Fit is pressed, and the stage goes back to what it
 *  showed. Waits for the fit to compute first; a held fit with no Fit to press skips the test. */
export async function pressFit(page: Page, timeout = 300_000): Promise<void> {
  const press = page.getByTestId("press-fit");
  const segment = page.getByTestId("banner-result");
  const shown = (await press.isVisible().catch(() => false)) || (await segment.getAttribute("aria-pressed")) === "true";
  if (!shown) await segment.click();
  const held = (await fitState(page, pidOf(page))) === "held";
  const offered = await press
    .waitFor({ state: "visible", timeout: held ? 15_000 : timeout })
    .then(() => true)
    .catch(() => false);
  test.skip(held && !offered, HELD_WITHOUT_FIT);
  expect(offered, "the Results offer Fit").toBe(true);
  await press.click();
  await expect(page.getByTestId("fit-press"), "Fit was taken").toHaveCount(0, { timeout });
  if (!shown && (await segment.getAttribute("aria-pressed")) === "true") await segment.click();
}

/** Whether the Router has reached Fit: the fit is served withheld until it is pressed, or is held
 *  for it. */
export async function fitReached(page: Page, pid: string): Promise<boolean> {
  const state = await fitState(page, pid);
  return state === "withheld" || state === "held";
}

/**
 * Once the fit has computed (or is held for Fit), Fit pressed if the fit waits for it (the real
 * server; the mock serves it whole, so nothing is pressed there). Returns whether Fit was pressed.
 */
export async function fitOnceComputed(page: Page, timeout = 300_000): Promise<boolean> {
  const pid = pidOf(page);
  await expect
    .poll(async () => fitState(page, pid), { message: "the fit computes", timeout, intervals: [500] })
    .not.toBe("pending");
  if ((await fitState(page, pid)) === "served") return false;
  await pressFit(page, timeout);
  return true;
}

/** The project's id from the page's URL (/p/:pid). */
export function pidOf(page: Page): string {
  return decodeURIComponent(/\/p\/([^/]+)$/.exec(page.url())![1]!);
}
