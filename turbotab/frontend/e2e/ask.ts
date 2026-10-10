import { expect, type Locator, type Page } from "@playwright/test";

/**
 * A substitution pair refused until the nesting it moves is confirmed (a part moves with its
 * total; the values agree but do not prove it): each "<part> is part of <total> instead" way
 * forward taken in turn, as a person who knows the table does (NHANES: sugar within carb, the
 * fatty acids within fat_total), then the pair chosen again (``choose``: "Recorded. Choose the pair
 * again to draw its curve."), until ``done`` shows.
 */
export async function confirmParts(
  page: Page,
  done: Locator,
  choose: () => Promise<void>,
  timeout = 180_000,
): Promise<void> {
  const end = Date.now() + timeout;
  let took = false;
  while (Date.now() < end) {
    if (await done.isVisible().catch(() => false)) return;
    const exit = page
      .getByRole("alert")
      .getByRole("button", { name: /^\S+ is part of \S+ instead$/ })
      .first();
    if (await exit.isVisible().catch(() => false)) {
      await exit.click();
      took = true;
      await page.waitForTimeout(1_500);
      continue;
    }
    if (took && (await page.getByText("Choose the pair again").isVisible().catch(() => false))) {
      took = false;
      await choose();
    }
    await page.waitForTimeout(500);
  }
  await expect(done).toBeVisible();
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
      await slot.page().waitForTimeout(2_000);
      before = await words();
      if (before === null) return;
    }
    const all = card.getByTestId("ask-confirm-all");
    const one = card.getByTestId("ask-confirm");
    if (await all.count()) await all.click();
    else if (await one.count()) await one.first().click();
    else await card.getByTestId("ask-exits").getByRole("button").first().click();
    await expect.poll(words, { message: "the ask card answers the press", timeout: 30_000 }).not.toBe(before);
  }
  await expect(card, "the ask card is answered").toHaveCount(0);
}
