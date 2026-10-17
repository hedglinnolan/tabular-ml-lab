import { expect, test, type Locator, type Page } from "@playwright/test";

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
 *  showed. Waits for Fit to be offered, then for the press to be taken (the Fit gone from the
 *  Results, for as long as the fit job runs) or refused (the Results say why, and this fails with
 *  it). */
export async function pressFit(page: Page, timeout = 300_000): Promise<{ waited: number; firstBusy: number | null }> {
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
  const section = page.getByTestId("fit-press");
  const refusal = section.getByRole("alert");
  // A fit that was already computed shows its results at once. A held one starts to compute, and a
  // screen that still shows an older withheld fit keeps the Fit until the new one arrives: the
  // wait lasts while the fit job runs, and ends once none has for the quiet time.
  const answered = await waitWhileComputing(
    page,
    async () => (await section.count()) === 0 || (await refusal.count()) > 0,
    ["fit"],
    { quiet: Math.min(timeout, FIT_QUIET), cap: Math.max(timeout, FIT_CAP) },
  );
  if (await refusal.count()) throw new Error(`Fit was refused: ${(await refusal.first().innerText()).trim()}`);
  expect(answered.done, `Fit was taken (waited ${answered.waited} s; ${answered.seen})`).toBe(true);
  if (!shown && (await segment.getAttribute("aria-pressed")) === "true") await segment.click();
  return { waited: answered.waited, firstBusy: answered.firstBusy };
}

/** Whether the Router has reached Fit: the fit is served withheld until it is pressed, or is held
 *  for it. */
export async function fitReached(page: Page, pid: string): Promise<boolean> {
  const state = await fitState(page, pid);
  return state === "withheld" || state === "held";
}

/** How long the fit job may go unseen before a wait for its result gives up, and the longest the
 *  wait lasts (a fit held for the press, then computed on a slow runner). */
export const FIT_QUIET = 120_000;
export const FIT_CAP = 900_000;

/**
 * Once the fit has computed (or is held for Fit), Fit pressed if the fit waits for it (the real
 * server, and the mock under Predict, which withholds the fit until then), and the result shown on
 * the banner. Returns whether Fit was pressed.
 *
 * The waits are bounded by what is computing, not by a flat time: a fit goes on being waited for
 * while the fit job runs (up to ``FIT_CAP``, 15 minutes), and fails once none has run for
 * ``timeout``. The test's own budget (``test.setTimeout``) has to hold that wait as well, or the
 * budget ends it first: a journey adds ``FIT_CAP`` to the rest of its time for each fit it waits for. A fit
 * held for the press (its measured estimates sum past two minutes, which a slower runner reaches
 * with the families a journey chose) computes only after the press, so the wait for the result
 * after it covers the whole fit: it ends when the result shows, or when the fit job has not been
 * seen for two minutes (``FIT_QUIET``) since the press or its last sighting. Every caller gets
 * that wait, and none keeps a flat one after it.
 */
export async function fitOnceComputed(page: Page, timeout = 300_000): Promise<boolean> {
  const pid = pidOf(page);
  const cap = Math.max(timeout, FIT_CAP);
  const start = Date.now();
  let busyAt = start;
  while ((await fitState(page, pid)) === "pending") {
    if ((await stagesComputing(page, ["fit"])).computing) busyAt = Date.now();
    if (Date.now() - busyAt >= timeout || Date.now() - start >= cap) break;
    await page.waitForTimeout(500);
  }
  const before = await fitState(page, pid);
  expect(before, "the fit computes").not.toBe("pending");
  const pressed = before !== "served";
  const taken = pressed ? await pressFit(page, timeout) : null;
  // The banner shows the result once the fit is served (today's screen; the shell has no banner).
  if (await page.getByTestId("banner-result").count()) {
    const shown = await visibleWhileComputing(page, page.getByTestId("banner-result-value"), ["fit"], {
      quiet: Math.min(timeout, FIT_QUIET),
      cap,
      label: "the fit's result is shown",
    });
    // Reported with the run (the report's annotations): whether the press found the fit held, how
    // long after it the fit job was first seen, and how long the result took.
    const after = taken?.waited ?? 0; // seconds from the click to the Fit going
    const first = taken?.firstBusy ?? (shown.firstBusy === null ? null : after + shown.firstBusy);
    try {
      test.info().annotations.push({
        type: "fit",
        description: `${before === "held" ? "held for the press" : before === "withheld" ? "computed, withheld until the press" : "served"}; after the press the fit job was first seen ${first === null ? "never" : `${first} s in`}, the result ${after + shown.waited} s in`,
      });
    } catch {
      /* not inside a test */
    }
  }
  return pressed;
}

/** The project's id from the page's URL (/p/:pid, or today's screen at /p/:pid/classic). */
export function pidOf(page: Page): string {
  return decodeURIComponent(/\/p\/([^/]+)(?:\/classic)?$/.exec(page.url())![1]!);
}

/** Every stage, for :func:`stagesComputing`: the page shows whatever is computing. */
export const ANY_STAGE = "*";

/**
 * The stages the server says are queued or running, from the project's GET (on tables that are not
 * wide: a journey on a wide table walks today's screen). The quest shell draws the work chips too
 * (QuestScreen's header), but a chip is drawn only for a status that carries a job, and a light
 * stage, run in the server's thread pool and not as a job (graph.py ``_start``), reads running with
 * none; a journey that waits on what is computing asks the server, which names every stage that is.
 */
async function busyOnServer(page: Page): Promise<string[]> {
  const pid = /\/p\/([^/]+)$/.exec(new URL(page.url()).pathname)?.[1];
  if (!pid) return [];
  return page.evaluate(async (p) => {
    const res = await fetch(`/api/projects/${p}`);
    if (!res.ok) return [];
    const view = (await res.json()) as { stages?: Record<string, { status?: string }> };
    return Object.entries(view.stages ?? {})
      .filter(([, s]) => s.status === "queued" || s.status === "running")
      .map(([name]) => name);
  }, decodeURIComponent(pid));
}

/**
 * Which of the stages are queued or running. On today's screen (/p/:pid/classic) that is the work
 * chip in the header that the app draws for each (``job-<stage>``, from the statuses the server
 * pushes; every stage named here is a heavy one, so it has a job), or any chip for ``ANY_STAGE``:
 * the page is read and the server asked nothing, which matters on a wide table, where the
 * project's GET recomputes the models card. On the quest shell (/p/:pid) the server's statuses are
 * read instead (:func:`busyOnServer`): the shell draws the same chips, but a light stage computing
 * has none, and ``ANY_STAGE`` there means any stage, not any chip.
 */
export async function stagesComputing(
  page: Page,
  names: readonly string[],
): Promise<{ computing: boolean; seen: string }> {
  const working: string[] = [];
  if (/\/classic$/.test(new URL(page.url()).pathname)) {
    for (const name of names) {
      const chips = name === ANY_STAGE ? page.locator('[data-testid^="job-"]') : page.getByTestId(`job-${name}`);
      const n = await chips.count();
      if (n) working.push(name === ANY_STAGE ? `${n} stage${n === 1 ? "" : "s"}` : name);
    }
  } else {
    const busy = await busyOnServer(page);
    for (const name of names) {
      if (name === ANY_STAGE) {
        if (busy.length) working.push(`${busy.length} stage${busy.length === 1 ? "" : "s"}`);
      } else if (busy.includes(name)) working.push(name);
    }
  }
  return { computing: working.length > 0, seen: working.length ? `${working.join(", ")} computing` : "none computing" };
}

/**
 * Waits for ``done`` to hold, not for a flat time: while any of the ``stages`` it waits on is
 * computing the wait goes on (up to ``cap``), and it ends once none has computed for ``quiet`` and
 * ``done`` still does not hold. Returns whether it held, how long the wait was (seconds), when a
 * stage was first seen computing (seconds into the wait; null if none was) and what the last look
 * saw, for the failure message of a caller that expects it to hold.
 */
export async function waitWhileComputing(
  page: Page,
  done: () => Promise<boolean>,
  stages: readonly string[],
  { quiet = 60_000, cap = 600_000 }: { quiet?: number; cap?: number } = {},
): Promise<{ done: boolean; waited: number; firstBusy: number | null; seen: string }> {
  const start = Date.now();
  let busyAt = start;
  let firstBusy: number | null = null;
  let seen = "no look taken";
  for (;;) {
    const waited = () => Math.round((Date.now() - start) / 1_000);
    if (await done().catch(() => false)) return { done: true, waited: waited(), firstBusy, seen };
    const look = await stagesComputing(page, stages);
    seen = look.seen;
    if (look.computing) {
      busyAt = Date.now();
      firstBusy ??= waited();
    }
    if (Date.now() - busyAt >= quiet || Date.now() - start >= cap) return { done: false, waited: waited(), firstBusy, seen }; // after the look
    await page.waitForTimeout(Math.min(2_000, quiet / 3));
  }
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
): Promise<{ waited: number; firstBusy: number | null }> {
  const shown = await waitWhileComputing(page, () => target.isVisible(), stages, { quiet, cap });
  if (!shown.done) await expect(target, `${label} (waited ${shown.waited} s; ${shown.seen})`).toBeVisible();
  return { waited: shown.waited, firstBusy: shown.firstBusy };
}

/**
 * ``target``'s attribute ``name`` equal to ``value``, waiting while any of the ``stages`` is
 * computing (up to ``cap``): the counterpart of :func:`visibleWhileComputing` for a state the page
 * marks. The wait ends once none has computed for ``quiet`` and the mark is still not there.
 */
export async function attributeWhileComputing(
  page: Page,
  target: Locator,
  name: string,
  value: string,
  stages: readonly string[],
  { quiet = 60_000, cap = 600_000, label = `${name} is ${value}` }: { quiet?: number; cap?: number; label?: string } = {},
): Promise<void> {
  const shown = await waitWhileComputing(page, async () => (await target.getAttribute(name, { timeout: 1_000 })) === value, stages, { quiet, cap });
  if (!shown.done) await expect(target, `${label} (waited ${shown.waited} s; ${shown.seen})`).toHaveAttribute(name, value, { timeout: 1_000 });
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
 *
 * A way forward is a decision the stages read (a noticing's disposition, the selection), so what
 * it changes may recompute before the card moves on: the waits after a press go on while the app
 * computes (up to the cap) and end once nothing has for ``quiet``, not after a flat time. The
 * answer to a press (the decisions POST) is one round trip, which no computing bounds.
 */
export async function openSealCard(page: Page, rounds = 16, { quiet = 30_000 }: { quiet?: number } = {}): Promise<string[]> {
  const card = page.getByTestId("open-seal-card");
  const refused = card.getByTestId("open-seal-refused");
  const taken: string[] = [];
  // What the last press left: the card gone (opened), a refusal, or neither (a way forward
  // recorded, the opening still to press).
  const outcome = async () => {
    // Read once: the page can change between two looks, and a refusal seen and then gone is not
    // an opening.
    let seen = null as "refused" | "opened" | null;
    const left = await waitWhileComputing(
      page,
      async () => {
        seen = (await refused.isVisible()) ? "refused" : (await card.count()) === 0 ? "opened" : null;
        return seen !== null;
      },
      [ANY_STAGE],
      { quiet, cap: 600_000 },
    );
    return left.done && seen !== null ? seen : ("neither" as const);
  };
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
    // The opening can be pressed again once what the answer changed has recomputed.
    await waitWhileComputing(page, async () => (await card.count()) === 0 || (await card.getByTestId("open-seal").isEnabled()), [ANY_STAGE], { quiet, cap: 600_000 });
    if (!(await card.count())) return taken;
    const again = await refused.waitFor({ state: "visible", timeout: 2_000 }).then(() => true, () => false);
    if (!again && (await card.count())) await card.getByTestId("open-seal").click();
  }
  return taken;
}
