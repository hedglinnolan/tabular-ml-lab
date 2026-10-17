import { expect, type Locator, type Page } from "@playwright/test";

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
 * server; the mock serves it whole, so nothing is pressed there), and the result shown on the
 * banner. Returns whether Fit was pressed.
 *
 * The waits are bounded by what is computing, not by a flat time: a fit goes on being waited for
 * while the fit job runs (up to 15 minutes), and fails once none has run for ``timeout``. A fit
 * held for the press (its measured estimates sum past two minutes, which a slower runner reaches
 * with the families a journey chose) computes only after the press, so the wait for the banner
 * after it covers the whole fit.
 */
export async function fitOnceComputed(page: Page, timeout = 300_000): Promise<boolean> {
  const pid = pidOf(page);
  const cap = Math.max(timeout, 900_000);
  const start = Date.now();
  let busyAt = start;
  while ((await fitState(page, pid)) === "pending") {
    if ((await stagesComputing(page, ["fit"])).computing) busyAt = Date.now();
    if (Date.now() - busyAt >= timeout || Date.now() - start >= cap) break;
    await page.waitForTimeout(500);
  }
  expect(await fitState(page, pid), "the fit computes").not.toBe("pending");
  if ((await fitState(page, pid)) === "served") return false;
  await pressFit(page, timeout);
  // The banner shows the result once the fit is served (today's screen; the shell has no banner).
  if (await page.getByTestId("banner-result").count()) {
    await visibleWhileComputing(page, page.getByTestId("banner-result-value"), ["fit"], {
      quiet: Math.min(timeout, 120_000),
      cap,
      label: "the fit's result is shown",
    });
  }
  return true;
}

/** The project's id from the page's URL (/p/:pid, or today's screen at /p/:pid/classic). */
export function pidOf(page: Page): string {
  return decodeURIComponent(/\/p\/([^/]+)(?:\/classic)?$/.exec(page.url())![1]!);
}

/** Every stage, for :func:`stagesComputing`: the page shows whatever is computing. */
export const ANY_STAGE = "*";

/**
 * Which of the stages the page shows as queued or running: the work chip in the header that the
 * app draws for each (``job-<stage>``, from the statuses the server pushes; every stage named
 * here is a heavy one, so it has a job), or any chip for ``ANY_STAGE``. It reads the page and asks
 * the server nothing, which matters on a wide table, where the project's GET recomputes the
 * models card.
 */
export async function stagesComputing(
  page: Page,
  names: readonly string[],
): Promise<{ computing: boolean; seen: string }> {
  const working: string[] = [];
  for (const name of names) {
    const chips = name === ANY_STAGE ? page.locator('[data-testid^="job-"]') : page.getByTestId(`job-${name}`);
    const n = await chips.count();
    if (n) working.push(name === ANY_STAGE ? `${n} stage${n === 1 ? "" : "s"}` : name);
  }
  return { computing: working.length > 0, seen: working.length ? `${working.join(", ")} computing` : "none computing" };
}

/**
 * ``target`` visible, waiting while any of the ``stages`` it waits on is computing (up to ``cap``),
 * not for a flat time: the wait ends once none has computed for ``quiet`` and ``target`` is still
 * not there. ``label`` names the target in the failure.
 */
export async function visibleWhileComputing(
  page: Page,
  target: Locator,
  stages: readonly string[],
  { quiet = 60_000, cap = 600_000, label = "shown" }: { quiet?: number; cap?: number; label?: string } = {},
): Promise<void> {
  const start = Date.now();
  let busyAt = start;
  let seen: string; // set by the first look, before any exit
  for (;;) {
    if (await target.isVisible().catch(() => false)) return;
    const look = await stagesComputing(page, stages);
    seen = look.seen;
    if (look.computing) busyAt = Date.now();
    if (Date.now() - busyAt >= quiet || Date.now() - start >= cap) break; // after the look
    await page.waitForTimeout(Math.min(2_000, quiet / 3));
  }
  const waited = Math.round((Date.now() - start) / 1_000);
  await expect(target, `${label} (waited ${waited} s; ${seen})`).toBeVisible();
}

/** A family of the shelf with the fit time the server measured for it (null: none was). */
export interface ShelfEstimate {
  key: string;
  seconds: number | null;
}

/**
 * The shelf's families in its order, each with its measured fit time, as the models card reads
 * them (``estimate_seconds``). The estimates move as readings are confirmed (confirming 497 genes
 * raises the elastic net's from seconds to minutes), so a journey reads them when it chooses.
 */
export async function shelfEstimates(page: Page, pid: string = pidOf(page)): Promise<ShelfEstimate[]> {
  return page.evaluate(async (p) => {
    const res = await fetch(`/api/projects/${p}/stages/shelf`);
    const body = res.ok
      ? ((await res.json()) as {
          artifact?: { families?: { key: string; rank: number; estimate_seconds?: number | null }[] } | null;
        })
      : {};
    return [...(body.artifact?.families ?? [])]
      .sort((a, b) => a.rank - b.rank)
      .map((f) => ({ key: f.key, seconds: f.estimate_seconds ?? null }));
  }, pid);
}

/**
 * The family among ``among`` that the shelf estimates cheapest to fit: the smallest measured fit
 * time, the shelf's order breaking a tie. A family with no measured time says nothing about its
 * cost, so it is chosen only when none of them has one (then the first on the shelf). Null when
 * the shelf offers none of ``among``.
 */
export function cheapestOf(shelf: readonly ShelfEstimate[], among: readonly string[]): string | null {
  const offered = shelf.filter((f) => among.includes(f.key));
  const measured = offered.filter((f) => f.seconds !== null);
  const pool = measured.length ? measured : offered;
  let best: ShelfEstimate | null = null;
  for (const f of pool) if (best === null || (f.seconds ?? 0) < (best.seconds ?? 0)) best = f;
  return best?.key ?? null;
}

/** The cheapest of ``among`` by the shelf's estimate now (:func:`cheapestOf`). */
export async function cheapestFamily(page: Page, among: readonly string[]): Promise<string | null> {
  return cheapestOf(await shelfEstimates(page), among);
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
