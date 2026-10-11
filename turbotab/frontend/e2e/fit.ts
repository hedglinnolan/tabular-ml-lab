import { expect, test, type Page } from "@playwright/test";

/**
 * Fit, pressed as the person presses it (SIZING P0.8; calm/FOUNDATION §7). Until it is, no score or
 * estimate is served: the Results show one button, and the seal waits for it. A fit estimated
 * over about 2 minutes is held for it in the scheduler (RECIPES §4.4), so nothing is computed and
 * there are no Results to carry the button yet: today's app offers Fit nowhere else, so a journey
 * whose fit is held stops there, skipped with that reason (the blocker the W1 shell removes),
 * instead of waiting out its timeout or pressing what the person cannot.
 *
 * **The skip expires** once the app renders a held fit (W1-B reads `StageStatus.held`): then a held
 * journey presses that Fit, or fails with HELD_RENDERED when it is not the Fit this helper presses;
 * and the fast tier fails (test_w1_real_server_e2e_job.py) until this skip is retired.
 */
export const HELD_WITHOUT_FIT =
  "the fit is held for Fit (its estimate exceeds about 2 minutes), and the app offers Fit only on a computed fit's Results: nothing to press yet (W1 blocker, the shell's Fit)";

/** Why a held fit fails instead of skipping: the app renders the held state, but not as the Fit
 *  this helper presses. */
export const HELD_RENDERED =
  "the app now renders a held fit (W1-B's held Fit, or the shell's), so HELD_WITHOUT_FIT has expired: press that Fit here (e2e/fit.ts pressFit) and retire the skip";

/**
 * What on the page shows a held fit, other than the Fit this helper presses: W1-B's held Fit
 * (`fit-held`), or any visible button named Fit, alone or with its estimate ("Fit (about 4 min)";
 * never the models question's "Fit these 3 families"). Empty while the app renders nothing for a
 * held fit (today's B2).
 */
export async function heldRendered(page: Page): Promise<string[]> {
  const found: string[] = [];
  if ((await page.getByTestId("fit-held").count()) > 0) found.push('data-testid="fit-held"');
  for (const button of await page.getByRole("button", { name: /^Fit(\s*\(.*\))?$/ }).all()) {
    if (await button.isVisible()) found.push(`the button "${(await button.textContent())?.trim()}"`);
  }
  return found;
}

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
  if (held && !offered) {
    // The skip expires once the app renders a held fit (W1-B's Fit, or the shell's): from then on
    // a held journey presses that Fit, and one the helper cannot find fails here instead of
    // skipping in silence.
    const rendered = await heldRendered(page);
    expect(rendered, HELD_RENDERED).toEqual([]);
    test.skip(true, HELD_WITHOUT_FIT);
  }
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
