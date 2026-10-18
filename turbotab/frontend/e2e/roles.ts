/**
 * The roles proposed below high confidence, confirmed as a researcher who knows the table does, and
 * checked against what the server serves (W1 part 2's U3; BLUEPRINT §14 rule 2, ruling R-g): each
 * waiting role line of the quest log is pressed once, a group's line (the columns the roles stage
 * proposed with one role, one confidence and one reason) by its one press for all its columns; then
 * the server holds no role waiting, and once fitted, the served design's predictors include every
 * column confirmed in a predictor role. Nothing here is counted on the journey's own account: every
 * column and count is the server's (GET /quest, GET /stages/design).
 */
import { expect, type Locator, type Page } from "@playwright/test";
import type { Lineage } from "../src/api/m1-stage-types";
import type { QuestLine, QuestLog } from "../src/api/schema";
import { confirmAsk } from "./ask";
import { ANY_STAGE, pidOf, waitWhileComputing } from "./fit";

/** The predictor roles (core/readings.py PREDICTOR_ROLES): a column confirmed in one enters the model. */
const PREDICTOR_ROLES = ["exposure", "covariate", "energy"];

async function served<T>(page: Page, path: string): Promise<T> {
  return page.evaluate(async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} answered ${res.status}`);
    return res.json();
  }, path) as Promise<T>;
}

/** Whether a quest line is a role line (core/quest.py `_role_lines`: a column's own, or a group's). */
export const isRoleLineOf = (line: Pick<QuestLine, "source" | "id">) => line.source === "reading" && line.id === "decision:confirm_role";

/** Every column the served quest log's open role lines hold, with its role: a group line's each. */
export async function waitingRoles(page: Page, pid: string = pidOf(page)): Promise<{ column: string; role: string }[]> {
  const log = await served<QuestLog>(page, `/api/projects/${encodeURIComponent(pid)}/quest`);
  return log.stages
    .flatMap((s) => s.lines)
    .filter((l) => isRoleLineOf(l) && l.status === "open")
    .flatMap((l) => l.items.filter((i) => i.kind === "role").map((i) => ({ column: i.column, role: i.value })));
}

/**
 * On today's screen (Classic), the roles that wait confirmed on the question's ask card, which lists
 * each judgment of the roles stage as one line with one press (ask.py `with_proposals`: a group's
 * line confirms all its columns), before the families are chosen; then the server holds none
 * waiting. Returns the columns confirmed, as the server listed them before.
 */
export async function confirmWaitingRoles(page: Page, slot: Locator): Promise<{ column: string; role: string }[]> {
  const pid = pidOf(page);
  const before = await waitingRoles(page, pid);
  const card = slot.getByTestId("ask-card");
  const bounds = { quiet: 60_000, cap: 600_000 };
  // The card leaves while the stages its answers reshape compute (a group of 485 genes confirmed
  // ranks the shelf again), and comes back with the lines still waiting: it is answered again for as
  // long as the server lists a role waiting, each time once the card is back or none waits.
  for (let round = 0; round < 10 && (await waitingRoles(page, pid)).length; round++) {
    await confirmAsk(slot);
    await waitWhileComputing(page, async () => (await waitingRoles(page, pid)).length === 0 || (await card.isVisible()), [ANY_STAGE], bounds);
  }
  const none = await waitWhileComputing(page, async () => (await waitingRoles(page, pid)).length === 0, [ANY_STAGE], bounds);
  expect(await waitingRoles(page, pid), `no role waits on the server (waited ${none.waited} s; ${none.seen})`).toEqual([]);
  return before;
}

/**
 * The served design's raw predictors include every column confirmed in a predictor role that the
 * design does not itself leave out by another of the project's answers (its `left_out`: the
 * missing-values answer's dropped columns, an unusable column's repair): each named by its own
 * node, or, where the design draws a group of like columns as one node, counted by it (its
 * `count`), so at least as many of each role as were confirmed. Returns how many it holds.
 */
export async function expectDesignHolds(page: Page, confirmed: readonly { column: string; role: string }[]): Promise<number> {
  const pid = pidOf(page);
  const body = await served<{ artifact: { lineage: Lineage; left_out: string[] } | null }>(
    page,
    `/api/projects/${encodeURIComponent(pid)}/stages/design`,
  );
  expect(body.artifact, "the design is served").not.toBeNull();
  const raw = body.artifact!.lineage.nodes.filter((n) => n.lane === "raw");
  const named = new Set(raw.filter((n) => (n.count ?? 1) === 1).map((n) => n.column));
  const out = new Set(body.artifact!.left_out);
  const kept = confirmed.filter((c) => !out.has(c.column));
  for (const role of PREDICTOR_ROLES) {
    const mine = kept.filter((c) => c.role === role);
    if (!mine.length) continue;
    const held = raw.filter((n) => n.role === role).reduce((k, n) => k + (n.count ?? 1), 0);
    expect(held, `the design holds the ${mine.length} columns confirmed as ${role}`).toBeGreaterThanOrEqual(mine.length);
    const grouped = raw.some((n) => n.role === role && (n.count ?? 1) > 1);
    if (!grouped) for (const c of mine) expect(named.has(c.column), `the design holds ${c.column}`).toBe(true);
  }
  return kept.filter((c) => PREDICTOR_ROLES.includes(c.role)).length;
}
