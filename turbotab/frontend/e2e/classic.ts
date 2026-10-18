import { expect, type Page } from "@playwright/test";

/**
 * Today's screen for the project the page just opened. /p/:pid is the quest shell, and today's
 * screen, which these specs drive, is at /p/:pid/classic until W2 closes (W1_PART2_PLAN ruling 8).
 * The shell's header links there; following the link navigates in the app, so a dev:mock project,
 * which lives in the page's memory, is the same project on the other side.
 */
export async function toClassic(page: Page, timeout = 60_000): Promise<void> {
  await page.getByTestId("to-classic").first().click({ timeout });
  await expect(page).toHaveURL(/\/p\/[^/]+\/classic$/, { timeout });
}
