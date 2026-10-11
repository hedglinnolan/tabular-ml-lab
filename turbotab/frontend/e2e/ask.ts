import { expect, type Locator, type Page } from "@playwright/test";

/**
 * A preview the stage holds back until readings it reads are confirmed ("Not available: … Tell me
 * about these columns"): its "Confirm each of the N as shown" taken. True when there was one.
 */
export async function confirmStageAsk(page: Page): Promise<boolean> {
  const confirm = page
    .getByTestId("stage")
    .getByRole("button", { name: /^Confirm (each of the [\d,]+|it) as shown$/ })
    .first();
  if (!(await confirm.isVisible().catch(() => false))) return false;
  await confirm.click();
  await expect(confirm).toHaveCount(0, { timeout: 30_000 });
  return true;
}

// The ways forward a refused pair offers that these fixtures' authors know to take: a part read
// as part of its total, and a nutrient's unit in grams (NHANES reports every macronutrient in g).
const PAIR_EXITS = /^\S+ is (part of \S+|in grams) instead$/;

/**
 * A substitution pair refused until the readings it moves are confirmed (a part moves with its
 * total, the values agreeing but not proving it; a nutrient's unit, which a name never says): each
 * such way forward taken in turn, as a person who knows the table does (NHANES: sugar within carb,
 * the fatty acids within fat_total, every nutrient in grams), then the pair chosen again
 * (``choose``: "Recorded. Choose the pair again to draw its curve."), until ``done`` shows.
 * Returns "refused" when the pair is refused with no such way forward (a share of energy moves in
 * percentage points, not kcal), so a caller can choose another pair.
 */
export async function confirmParts(
  page: Page,
  done: Locator,
  choose: () => Promise<void>,
  timeout = 180_000,
): Promise<"drawn" | "refused"> {
  const end = Date.now() + timeout;
  let took = false;
  let bare = 0;
  while (Date.now() < end) {
    if (await done.isVisible().catch(() => false)) return "drawn";
    const exit = page.getByRole("alert").getByRole("button", { name: PAIR_EXITS }).first();
    if (await exit.isVisible().catch(() => false)) {
      await exit.click();
      took = true;
      bare = 0;
      await page.waitForTimeout(1_500);
      continue;
    }
    if (took && (await page.getByText("Choose the pair again").isVisible().catch(() => false))) {
      took = false;
      await choose();
      await page.waitForTimeout(1_500);
      continue;
    }
    const refused = page.getByRole("alert").filter({ hasText: "Not recorded" });
    bare = (await refused.isVisible().catch(() => false)) ? bare + 1 : 0;
    if (bare >= 6) return "refused"; // three seconds refused, with nothing to confirm
    await page.waitForTimeout(500);
  }
  await expect(done).toBeVisible();
  return "drawn";
}

/**
 * The substitution's first pair the Results draw: each pair of the matrix in its order, its
 * readings confirmed (:func:`confirmParts`), the next one tried when a pair is refused outright.
 * Returns the pair's button name.
 */
export async function firstDrawnPair(page: Page, done: Locator, tries = 12): Promise<string> {
  const buttons = page.getByRole("button", { name: /^Move energy from / });
  await expect(buttons.first()).toBeVisible({ timeout: 60_000 });
  const names = await buttons.evaluateAll((els) =>
    els.map((e) => e.getAttribute("aria-label") ?? (e.textContent ?? "").trim()),
  );
  // A share of energy (a *_pct_* column) moves in percentage points, not kcal, and is refused: the
  // pairs of amounts first, in the matrix's order.
  const share = (name: string) => (/_pct_/.test(name) ? 1 : 0);
  names.sort((a, b) => share(a) - share(b));
  for (const name of names.slice(0, tries)) {
    const move = () => page.getByRole("button", { name, exact: true }).click();
    await move();
    await page.waitForTimeout(1_500);
    if ((await confirmParts(page, done, move)) === "drawn") return name;
  }
  throw new Error(`none of the first ${tries} pairs was drawn`);
}

/**
 * The ledger's ask card on a question (BLUEPRINT §14.2), when it asks: answered as a person who
 * knows the table does, until the card is gone. Each round takes "Confirm every line as shown"
 * when the card offers it (it does once one line was confirmed by hand), else the first line's
 * Confirm, else the consumer's own way forward (an energy column's unit and days), and waits for
 * the record to answer (the card changes or goes).
 */
export async function confirmAsk(slot: Locator): Promise<void> {
  const card = slot.getByTestId("ask-card");
  const words = async () => ((await card.count()) ? ((await card.first().textContent()) ?? "") : null);
  for (let round = 0; round < 40; round++) {
    let before = await words();
    if (before === null) {
      // Gone, or only between two renders (a confirmed line recomputes the card): settled once
      // it stays away.
      await slot.page().waitForTimeout(5_000);
      before = await words();
      if (before === null) return;
    }
    // The press the card offers now; while it recomputes it may offer none for a moment.
    const pick = async (): Promise<Locator | null> => {
      const all = card.getByTestId("ask-confirm-all");
      if (await all.count()) return all.first();
      const one = card.getByTestId("ask-confirm");
      if (await one.count()) return one.first();
      const exits = card.getByTestId("ask-exits").getByRole("button");
      return (await exits.count()) ? exits.first() : null;
    };
    let press: Locator | null = null;
    await expect
      .poll(
        async () => {
          if ((await words()) === null) return "gone";
          press = await pick();
          return press ? "ready" : "waiting";
        },
        { message: "the ask card offers a press", timeout: 30_000 },
      )
      .not.toBe("waiting");
    if (press === null) continue; // gone meanwhile: settled once it stays away
    before = await words();
    await (press as Locator).click({ timeout: 15_000 });
    await expect.poll(words, { message: "the ask card answers the press", timeout: 30_000 }).not.toBe(before);
  }
  await expect(card, "the ask card is answered").toHaveCount(0);
}
