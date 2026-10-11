import { expect, type Page } from "@playwright/test";

/**
 * Fit, pressed as the person presses it (SIZING P0.8; calm/FOUNDATION §7). Until it is, no score or
 * estimate is served: the Results show one button, and the seal waits for it. A fit estimated
 * over about 2 minutes is held for it in the scheduler (RECIPES §4.4), so nothing is computed yet;
 * the stage then offers the same Fit (W1-B's FitHeld, `press-fit` inside `fit-held`), and a held
 * journey presses it as it presses a computed fit's.
 */

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
 *  showed. Waits for the fit to compute, or for its held Fit. */
export async function pressFit(page: Page, timeout = 300_000): Promise<void> {
  const press = page.getByTestId("press-fit");
  const segment = page.getByTestId("banner-result");
  const shown = (await press.isVisible().catch(() => false)) || (await segment.getAttribute("aria-pressed")) === "true";
  if (!shown) await segment.click();
  // A held fit is offered on the stage itself (FitHeld), a computed one on its Results: either
  // way the one Fit carries this test id.
  const offered = await press
    .waitFor({ state: "visible", timeout })
    .then(() => true)
    .catch(() => false);
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

/**
 * The seal opened from its card as a person settles what it waits for. While the card shows a
 * refusal with a way forward (the final model named on cross-validation, the triage of open
 * noticings kept as recommended, an "act on it" noticing recorded as a limitation), the first is
 * taken; a way forward refused in turn shows its own, taken next. Once nothing is refused, "open
 * the seal" is pressed again. Returns the labels taken, in order.
 */
export async function openSealCard(page: Page, rounds = 16): Promise<string[]> {
  const card = page.getByTestId("open-seal-card");
  const refused = card.getByTestId("open-seal-refused");
  const taken: string[] = [];
  // What the last press left: the card gone (opened), a refusal, or neither (a way forward
  // recorded, the opening still to press).
  const outcome = async () =>
    Promise.race([
      refused.waitFor({ state: "visible", timeout: 30_000 }).then(() => "refused" as const),
      card.waitFor({ state: "detached", timeout: 30_000 }).then(() => "opened" as const),
    ]).catch(() => "neither" as const);
  await card.getByTestId("open-seal").click();
  for (let i = 0; i < rounds; i++) {
    const now = await outcome();
    if (now === "opened") return taken;
    if (now === "neither") {
      await card.getByTestId("open-seal").click();
      continue;
    }
    const exit = card.getByTestId("open-seal-exit").first();
    if (!(await exit.count())) return taken;
    taken.push(((await exit.textContent()) ?? "").trim());
    // Pressing a way forward clears the refusal at once; its answer says what follows (accepted:
    // the refusal stays away; refused: a new refusal; opened: the card goes).
    const answered = page.waitForResponse(
      (r) => r.request().method() === "POST" && /\/decisions$/.test(new URL(r.url()).pathname),
      { timeout: 60_000 },
    );
    await exit.click();
    await answered;
    await expect(card.getByTestId("open-seal"), "the way forward was answered").toBeEnabled({ timeout: 30_000 }).catch(() => {});
    if (!(await card.count())) return taken;
    const again = await refused.waitFor({ state: "visible", timeout: 2_000 }).then(() => true, () => false);
    if (!again && (await card.count())) await card.getByTestId("open-seal").click();
  }
  return taken;
}
