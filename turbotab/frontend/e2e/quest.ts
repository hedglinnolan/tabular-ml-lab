import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";
import type { FitArtifact } from "../src/api/m1-stage-types";
import type { ExplainArtifact } from "../src/api/m3-types";
import type { QuestLine, QuestLog, QuestStage, Triage } from "../src/api/schema";
import { confirmAsk, enabledNow } from "./ask";
import { ANY_STAGE, FIT_CAP, FIT_QUIET, stagesComputing, visibleWhileComputing, waitWhileComputing } from "./fit";

/**
 * What the quest journeys share (plans/W1_PART2_PLAN.md §2 "W1-L"; WALKABLE_PLAN §5 W1 acceptance 1
 * and 4): the project opened through the upload zone, the shell's quest line and its progress bar
 * checked against GET /quest, a card's line answered as a person answers it, and the printed
 * scores checked against the fit artifact the server serves. Nothing here counts, scores or rounds
 * on its own account: an expected value is a served field, read from the API in the page.
 */

// ── what the server serves ───────────────────────────────────────────────────

/** A GET against the project's API, read in the page as the app reads it. */
export async function getJson<T>(page: Page, pid: string, path = ""): Promise<T> {
  return page.evaluate(
    async ([p, rest]) => {
      const res = await fetch(`/api/projects/${p}${rest}`);
      if (!res.ok) throw new Error(`GET ${rest || "/"} answered ${res.status}`);
      return res.json();
    },
    [pid, path] as const,
  ) as Promise<T>;
}

/**
 * Waits for the server to answer a press, as long as the app is working on it (ruling 11 of
 * plans/W1_PART2_PLAN.md: a journey's waits are bounded by what is computing, not by a flat time):
 * ``done`` is looked at while any stage computes (up to 15 minutes), and the wait ends once none has
 * for a minute. Fails naming ``what`` and what the last look saw.
 */
async function settled(page: Page, what: string, done: () => Promise<boolean>): Promise<void> {
  const r = await waitWhileComputing(page, done, [ANY_STAGE], { quiet: 60_000, cap: 900_000 });
  expect(r.done, `${what} (waited ${r.waited} s; ${r.seen})`).toBe(true);
}

export const getQuest = (page: Page, pid: string) => getJson<QuestLog>(page, pid, "/quest");
export const getTriage = (page: Page, pid: string) => getJson<Triage>(page, pid, "/triage");

/** The fit artifact as served now (held-out scores only once the rows are opened). */
export async function getFit(page: Page, pid: string): Promise<FitArtifact> {
  const body = await getJson<{ artifact: FitArtifact | null }>(page, pid, "/stages/fit");
  if (!body.artifact) throw new Error("the fit is not served");
  return body.artifact;
}

/** The stage a quest line sits in, by stage order then the line's own order. */
export function linesOf(log: QuestLog): { stage: QuestStage; line: QuestLine }[] {
  return log.stages.flatMap((stage) => [...stage.lines].sort((a, b) => a.order - b.order).map((line) => ({ stage, line })));
}

export const lineId = (line: Pick<QuestLine, "source" | "key">) => `${line.source}:${line.key}`;

/** A one-line account of where the analysis stands, for a failure message. */
export function describe(log: QuestLog): string {
  return log.stages
    .map((s) => {
      const open = s.lines.filter((l) => l.status === "open" || l.status === "waiting").map((l) => `${l.key}=${l.status}`);
      return `${s.key}${s.reached ? "" : " (not reached)"} ${s.progress ? `${s.progress.answered}/${s.progress.required}` : "-"}${open.length ? ` [${open.join(", ")}]` : ""}`;
    })
    .join("; ");
}

// ── opening a project ────────────────────────────────────────────────────────

/** The table uploaded through the Start screen's drop zone (its file chooser, as a person's file
 *  pick), and the shell open on the project it made. Returns the project's id. */
export async function uploadThroughZone(page: Page, path: string): Promise<string> {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Open a table to begin." })).toBeVisible();
  await expect(page.getByTestId("dropzone")).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles(path);
  await expect(page).toHaveURL(/\/p\/[^/]+$/, { timeout: 120_000 });
  await expect(page.getByTestId("quest-bar")).toBeVisible({ timeout: 60_000 });
  return decodeURIComponent(/\/p\/([^/]+)$/.exec(page.url())![1]!);
}

// ── the quest line and the progress bar equal GET /quest ─────────────────────

interface SegmentRead {
  key: string;
  reached: string | null;
  complete: string | null;
  answered: string | null;
  required: string | null;
  reopened: string | null;
}

/** What the bar's seven segments carry, as the page marks them. */
async function readSegments(page: Page): Promise<SegmentRead[]> {
  return page.evaluate(() =>
    Array.from(document.querySelectorAll('[data-testid="progress-bar"] > li')).map((el) => ({
      key: (el.getAttribute("data-testid") ?? "").replace(/^segment-/, ""),
      reached: el.getAttribute("data-reached"),
      complete: el.getAttribute("data-complete"),
      answered: el.getAttribute("data-answered"),
      required: el.getAttribute("data-required"),
      reopened: el.getAttribute("data-reopened"),
    })),
  );
}

/** What GET /quest says each segment is: empty when not reached (never "0 of N"), else answered of
 *  required, done when complete, dropped back while a reopening stands. */
function wantedSegment(s: QuestStage): SegmentRead {
  const on = s.reached && s.progress;
  return {
    key: s.key,
    reached: s.reached ? "true" : null,
    complete: s.reached && s.progress?.complete ? "true" : null,
    answered: on ? String(s.progress!.answered) : null,
    required: on ? String(s.progress!.required) : null,
    reopened: s.reopened.length ? "true" : null,
  };
}

/** The panel's lines as "key:status", sorted: a multiset, so the panel's own order is not asserted. */
function wantedPanel(s: QuestStage): string[] {
  return s.lines.map((l) => `${l.key}:${l.status}`).sort();
}

async function readPanel(page: Page, stage: string): Promise<string[]> {
  return page.evaluate((key) => {
    const panel = document.querySelector(`[data-testid="panel-${key}"]`);
    if (!panel) return ["(no panel)"];
    return Array.from(panel.querySelectorAll('[data-testid^="panel-line-"]'))
      .map((el) => `${(el.getAttribute("data-testid") ?? "").replace(/^panel-line-/, "")}:${el.getAttribute("data-status")}`)
      .sort();
  }, stage);
}

/** Where the page's marks and GET /quest differ, or "" where they agree. */
function differences(read: SegmentRead[], log: QuestLog): string {
  const out: string[] = [];
  const want = log.stages.map(wantedSegment);
  if (read.length !== want.length) out.push(`the bar has ${read.length} segments, /quest has ${want.length} stages`);
  for (const w of want) {
    const got = read.find((r) => r.key === w.key);
    if (!got) {
      out.push(`${w.key}: no segment`);
      continue;
    }
    for (const field of ["reached", "complete", "answered", "required", "reopened"] as const) {
      if (got[field] !== w[field]) out.push(`${w.key}.${field}: page ${got[field]}, /quest ${w[field]}`);
    }
  }
  return out.join("; ");
}

/** Where the bar's segments and GET /quest, read now, differ ("" where they agree). */
export async function barDifferences(page: Page, pid: string): Promise<string> {
  return differences(await readSegments(page), await getQuest(page, pid));
}

/**
 * The quest line against GET /quest, at this step: each bar segment's reached, complete, answered,
 * required and reopened marks, then, pointing at each stage in turn, the hover panel's line
 * statuses. The page is allowed the moment the server's event takes to reach it: a mismatch is
 * compared again (up to ``within``) against the quest read afresh each time, so a screen that
 * stays stale fails and one a beat behind does not.
 */
export async function expectBarEqualsQuest(page: Page, pid: string, step: string, within = 20_000): Promise<void> {
  let log = await getQuest(page, pid);
  if (differences(await readSegments(page), log)) {
    await expect
      .poll(async () => differences(await readSegments(page), await getQuest(page, pid)), {
        message: `${step}: each bar segment equals GET /quest`,
        timeout: within,
        intervals: [200, 400, 800],
      })
      .toBe("");
    log = await getQuest(page, pid);
  }
  for (const stage of log.stages) {
    await page.getByTestId(`stage-${stage.key}`).hover({ timeout: 30_000 });
    await expect(page.getByTestId(`panel-${stage.key}`), `${step}: pointing at ${stage.name} opens its panel`).toBeVisible({ timeout: 10_000 });
    if (JSON.stringify(await readPanel(page, stage.key)) === JSON.stringify(wantedPanel(stage))) continue;
    await expect
      .poll(
        async () => {
          const now = (await getQuest(page, pid)).stages.find((s) => s.key === stage.key)!;
          const got = await readPanel(page, stage.key);
          const want = wantedPanel(now);
          return JSON.stringify(got) === JSON.stringify(want) ? "" : `page ${JSON.stringify(got)} vs /quest ${JSON.stringify(want)}`;
        },
        { message: `${step}: ${stage.name}'s panel statuses equal GET /quest`, timeout: within, intervals: [200, 400, 800] },
      )
      .toBe("");
  }
  await page.mouse.move(1430, 890);
}

/** The stage the card shows (the quest line's current step). */
export async function shownStage(page: Page): Promise<string | null> {
  const current = page.locator('[data-testid^="stage-"][aria-current="step"]');
  const id = (await current.count()) ? await current.first().getAttribute("data-testid") : null;
  return id ? id.replace(/^stage-/, "") : null;
}

/** A stage opened by its name in the quest line, as a person clicks it. */
export async function openStage(page: Page, key: string): Promise<void> {
  if ((await shownStage(page)) !== key) await page.getByTestId(`stage-${key}`).click();
  await expect(page.getByTestId(`stage-${key}`)).toHaveAttribute("aria-current", "step");
  await expect(page.getByTestId(`card-${key}`)).toBeVisible();
}

// ── answering a line through its card ────────────────────────────────────────

/** The line's `li` on the card of the stage shown. */
export const lineOn = (page: Page, line: QuestLine): Locator => page.locator(`li[data-line="${lineId(line)}"]`);

/** The line expanded to its card: its "Answer" or "Change" pressed when it is collapsed. */
export async function expandLine(page: Page, line: QuestLine): Promise<Locator> {
  const li = lineOn(page, line);
  await visibleWhileComputing(page, li, [ANY_STAGE], { label: `${lineId(line)} is listed on the card` });
  if ((await li.getAttribute("data-expanded")) === null) await li.getByTestId(`line-${line.key}-toggle`).click();
  await expect(li).toHaveAttribute("data-expanded", "true");
  return li;
}

/** The stage's preview of the option pointed at has settled: where the tapestry is the stage, its
 *  loading mark is off (the Results exhibits have none). */
export async function previewSettled(page: Page): Promise<void> {
  const loading = page.getByTestId("stage-loading");
  if (await loading.count()) {
    await settled(page, "the stage's preview has settled", () => loading.evaluateAll((els) => els.every((e) => !e.hasAttribute("data-on"))));
  }
}

/** Nothing the page sent is still being recorded (an Enter pressed while it is, is not sent). */
export async function notPending(list: Locator): Promise<void> {
  await settled(list.page(), "nothing is being recorded", () => list.evaluateAll((els) => els.every((e) => e.getAttribute("aria-busy") !== "true")));
}

/** An option of the card's list chosen as the keyboard chooses it: focused (the stage previews it),
 *  then Enter. A multi-select takes ``space`` first. */
export async function pickOption(page: Page, scope: Locator, testId: string, opts: { space?: boolean } = {}): Promise<void> {
  const option = scope.getByTestId(testId);
  await visibleWhileComputing(page, option, [ANY_STAGE], { label: `${testId} is offered` });
  await option.focus();
  await previewSettled(page);
  await notPending(option.locator("xpath=ancestor::*[@role='listbox'][1]"));
  if (opts.space) await page.keyboard.press("Space");
  await page.keyboard.press("Enter");
}

/** The ledger's ask card on a question, answered as a person who knows the table does: "Confirm
 *  every line as shown" once it is offered, else the first line's Confirm, until the card is gone. */
export async function confirmAskCard(_page: Page, scope: Locator): Promise<number> {
  return confirmAsk(scope);
}

/**
 * A noticing answered on its own card: a repair's first option, as the card lists it (the engine's
 * default order), focused and recorded with Enter; a noticing with no repair is kept as it is
 * ("Keep as is": seen and set aside, recorded as dismissed).
 */
export async function answerFinding(page: Page, li: Locator, line: QuestLine): Promise<void> {
  const card = li.locator('[data-testid^="repair-options-"]');
  const keep = li.getByTestId(`keep-as-is-${line.key}`);
  await visibleWhileComputing(page, card.or(keep).first(), [ANY_STAGE], { label: `${line.key}'s card offers an answer` });
  if (await keep.count()) return void (await keep.click());
  const options = card.locator('[role="option"]');
  await options.first().focus();
  await previewSettled(page);
  await notPending(card);
  await page.keyboard.press("Enter");
}

/**
 * The model families question, answered as a person answers it: the ledger's ask card confirmed
 * (the fit reads the roles it proposed below high confidence), the shelf's options waited for while
 * it ranks again (up to the cap, while it computes), then each family chosen with the keyboard
 * (focus, Space) and the choice recorded with Enter, which is pressed only once nothing is being
 * recorded (an Enter pressed while a decision is in flight is not sent).
 */
export async function chooseFamilies(page: Page, li: Locator, families: readonly string[]): Promise<void> {
  const options = li.getByRole("listbox", { name: /Model families/ }).locator(':scope > [role="option"]');
  await visibleWhileComputing(page, options.first(), ["shelf"], { label: "the shelf's options are listed" });
  await confirmAskCard(page, li);
  await visibleWhileComputing(page, options.first(), ["shelf"], { label: "the shelf's options are listed" });
  const shelf = await options.evaluateAll((els) => els.map((e) => e.getAttribute("data-key") ?? ""));
  for (const f of families) expect(shelf, `the shelf offers ${f}`).toContain(f);
  for (const f of families) {
    await li.getByTestId(`option-${f}`).focus();
    await page.keyboard.press("Space");
    await expect(li.getByTestId(`option-${f}`)).toHaveAttribute("aria-selected", "true");
  }
  const record = li.getByTestId("record-models");
  // The shell's card records the choice; Fit is a separate press at the flowchart's end.
  await expect(record).toHaveText(`Choose these ${families.length} families`);
  await settled(page, "the families button is enabled", () => enabledNow(record));
  await notPending(li.getByTestId("options-models"));
  await page.keyboard.press("Enter");
}

/** What a journey does for a line: answers it on its card, or leaves it for a card further on. */
export interface Answers {
  /** The line left for a later card (the triage): not answered by the walk. */
  deferred: (line: QuestLine) => boolean;
  /** Lines the shell lists as "Not available yet": no card records them. */
  unavailable: readonly string[];
  /** The answer to a line, on its card. */
  answer: (page: Page, li: Locator, line: QuestLine) => Promise<void>;
}

/** The line's status once the server has recorded its answer. */
async function answered(page: Page, pid: string, line: QuestLine): Promise<boolean> {
  const now = linesOf(await getQuest(page, pid)).find((x) => lineId(x.line) === lineId(line));
  return !now || now.line.status !== "open";
}

/** The open lines a card can answer and the journey does not defer: by stage order, a stage's
 *  lines by the engine's order. */
function openLines(log: QuestLog, a: Answers): { stage: QuestStage; line: QuestLine }[] {
  return linesOf(log).filter(
    ({ stage, line }) => stage.reached && line.status === "open" && !a.unavailable.includes(line.key) && !a.deferred(line),
  );
}

/** The sweep a stage waits on, taken with the one button at the card's foot. True when it was. */
async function confirmSweepOf(page: Page, pid: string, stage: QuestStage): Promise<boolean> {
  const sweep = stage.sweep;
  if (!sweep || sweep.answered || sweep.lines === 0) return false;
  await openStage(page, stage.key);
  const primary = page.getByTestId("primary");
  if (!(await primary.count()) || (await primary.isDisabled()) || (await primary.innerText()) !== sweep.action) return false;
  await primary.click();
  await settled(page, `${stage.name}'s sweep is confirmed`, async () => (await getQuest(page, pid)).stages.find((s) => s.key === stage.key)?.sweep?.answered === true);
  return true;
}

/**
 * A card's own control, ready as a person finds it: shown and enabled. The control may be there
 * only when what it reads has been computed (the outcome's picker lists the columns once the file
 * is read, the roles' button waits for the proposals), so the wait goes on while the app computes
 * (up to 15 minutes) and ends once nothing has for a minute, not after a flat thirty seconds. It
 * waits for this control and no other stage: a card is answered while the rest of the analysis
 * computes, as a person answers it.
 */
export async function controlReady(page: Page, control: Locator, what: string): Promise<void> {
  await settled(page, `${what} is ready`, async () => (await control.isVisible()) && (await control.isEnabled()));
}

/**
 * The journey's walk through the quest line: answer, on its card, each open line the journey does
 * not defer (the stage opened by its name in the quest line, the line expanded as its "Answer"
 * does), confirm each stage's sweep with its one button once its lines are answered, and check the
 * bar against GET /quest after every move. It goes on until ``done`` holds of the quest, waiting
 * while a stage computes (up to ``quiet`` with none computing, never past ``cap``), and fails,
 * naming where the analysis stands, when nothing is left to answer and ``done`` does not hold.
 */
export async function walk(
  page: Page,
  pid: string,
  a: Answers,
  done: (log: QuestLog) => boolean,
  { quiet = 90_000, cap = 900_000 }: { quiet?: number; cap?: number } = {},
): Promise<string[]> {
  const taken: string[] = [];
  const start = Date.now();
  let busyAt = start;
  for (let moves = 0; moves < 400; moves++) {
    const log = await getQuest(page, pid);
    if (done(log)) return taken;
    let moved = false;
    for (const next of openLines(log, a)) {
      await openStage(page, next.stage.key);
      const li = lineOn(page, next.line);
      await visibleWhileComputing(page, li, [ANY_STAGE], { label: `${lineId(next.line)} is listed on the card` });
      // Under Predict the draw of the held-out rows waits for its stage's sweep (the card says so):
      // it is answered once the sweep is.
      if ((await li.getAttribute("data-held")) !== null) continue;
      // A line waits again while what it reads recomputes: it is answered when it is open again.
      if ((await li.getAttribute("data-status")) !== "open") continue;
      if ((await li.getAttribute("data-unavailable")) !== null) throw new Error(`${lineId(next.line)} has no card to answer it`);
      await expandLine(page, next.line);
      await a.answer(page, li, next.line);
      await settled(page, `${lineId(next.line)} is answered through its card`, () => answered(page, pid, next.line));
      taken.push(lineId(next.line));
      await expectBarEqualsQuest(page, pid, `after ${lineId(next.line)}`);
      busyAt = Date.now();
      moved = true;
      break;
    }
    if (moved) continue;
    let swept = false;
    for (const stage of log.stages) {
      if (!stage.reached) continue;
      if (await confirmSweepOf(page, pid, stage)) {
        taken.push(`sweep:${stage.key}`);
        await expectBarEqualsQuest(page, pid, `after ${stage.name}'s sweep`);
        swept = true;
        break;
      }
    }
    if (swept) {
      busyAt = Date.now();
      continue;
    }
    const look = await stagesComputing(page, [ANY_STAGE]);
    if (look.computing) busyAt = Date.now();
    if (Date.now() - busyAt >= quiet || Date.now() - start >= cap) {
      throw new Error(`nothing is left to answer and the walk is not done (${look.seen}): ${describe(log)}`);
    }
    await page.waitForTimeout(2_000);
  }
  throw new Error("the walk made 400 moves without finishing");
}

// ── the printed scores equal the served fields ───────────────────────────────

/** Every number in a printed cell, in order: a real minus sign read as one, thousands separators
 *  dropped. "0.104 (0.0867 to 0.122)" gives [0.104, 0.0867, 0.122]. */
export function printedNumbers(text: string): number[] {
  const plain = text.replaceAll("−", "-").replaceAll(",", "");
  return [...plain.matchAll(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)].map((m) => Number(m[0]));
}

/**
 * Whether a printed number is a served field as the paper prints it: to three significant digits
 * (views/format.ts), a value of a thousand or more to the whole number. The tolerance is half a
 * unit of the last digit printed, taken from the served value alone, so a number the page
 * computed itself, or took from another field, would fall outside it.
 */
export function printsServed(printed: number, served: number): boolean {
  if (served === 0) return printed === 0;
  const a = Math.abs(served);
  const half = a >= 1000 ? 0.5 : 0.5 * 10 ** (Math.floor(Math.log10(a)) - 2);
  return Math.abs(printed - served) <= half * (1 + 1e-9);
}

/** A printed cell's numbers against the fields they print, or what differs (empty: they agree). */
export function misprinted(cell: string, served: readonly (number | null | undefined)[], what: string): string {
  const want = served.filter((v): v is number => typeof v === "number");
  const got = printedNumbers(cell);
  if (got.length !== want.length) return `${what}: "${cell}" prints ${got.length} numbers, ${want.length} are served`;
  const wrong = got.map((g, i) => (printsServed(g, want[i]!) ? "" : `${g} for served ${want[i]}`)).filter(Boolean);
  return wrong.length ? `${what}: "${cell}": ${wrong.join(", ")}` : "";
}

/** A table's cells by row key and header label: each row's `td`s read against the header's `th`s,
 *  and each row's own label (its `th scope=row`). */
export async function readTable(
  table: Locator,
): Promise<{ header: string[]; rows: Record<string, string[]>; labels: Record<string, string> }> {
  return table.evaluate((el) => {
    const header = Array.from(el.querySelectorAll("thead th")).slice(1).map((th) => (th.firstChild?.textContent ?? "").trim());
    const rows: Record<string, string[]> = {};
    const labels: Record<string, string> = {};
    for (const tr of Array.from(el.querySelectorAll("tbody tr[data-row]"))) {
      const key = tr.getAttribute("data-row") ?? "";
      rows[key] = Array.from(tr.querySelectorAll("td")).map((td) => (td.textContent ?? "").trim());
      labels[key] = (tr.querySelector("th")?.textContent ?? "").trim();
    }
    return { header, rows, labels };
  });
}

const VERDICT_WORDS = { better: "Better", no_better: "Not shown to be better", worse: "Worse" } as const;

/** The metrics a set of served scores holds (the keys of ``scores`` on any family), each one a
 *  column of the table drawn from them: a metric served with no label could have no column, and
 *  one with a label whose header the table lacks has none. Returns what is missing. */
function missingColumns(fit: FitArtifact, header: readonly string[], scores: (m: FitArtifact["models"][number]) => object | null | undefined, what: string): string[] {
  const served = [...new Set(fit.models.flatMap((m) => Object.keys(scores(m) ?? {})))];
  return served.flatMap((key) => {
    const label = fit.metric_labels[key];
    if (label === undefined) return [`${key} is served on the ${what} scores with no label, so it can have no column`];
    return header.includes(label) ? [] : [`the ${what} table has no column for ${label}, which is served`];
  });
}

/** A verdict against the baseline as a cell prints it: the verdict's words, and where a gain is
 *  served "a gain in <metric> of <estimate (interval)>", the numbers its served fields. */
function verdictDifferences(
  cell: string,
  v: { verdict: keyof typeof VERDICT_WORDS; metric: string; gain: number | null; ci_low: number | null; ci_high: number | null },
  labels: Record<string, string>,
  who: string,
): string {
  const word = VERDICT_WORDS[v.verdict];
  if (v.gain === null) return cell === word ? "" : `${who} verdict: "${cell}" is not the served "${word}"`;
  const head = `${word}: a gain in ${labels[v.metric] ?? v.metric} of `;
  if (!cell.startsWith(head)) return `${who} verdict: "${cell}" does not begin "${head}"`;
  return misprinted(cell.slice(head.length), [v.gain, v.ci_low, v.ci_high], `${who} gain`);
}

/**
 * The held-out exhibit against the fit artifact the server serves, once the rows are opened: for
 * each family, each metric's score with its interval; the no-predictor baseline's scores on the
 * same rows; each family's verdict against it, its gain and the gain's interval. Every expected
 * number is read from the served artifact; the page's text is only parsed. Returns what differs.
 */
export async function heldOutDifferences(page: Page, fit: FitArtifact): Promise<string[]> {
  const table = page.getByTestId("results-held-out").locator("table");
  if (!(await table.count())) return ["the held-out table is not drawn"];
  const { header, rows } = await readTable(table);
  const keyOf = (label: string) => Object.entries(fit.metric_labels).find(([, l]) => l === label)?.[0];
  // Every metric the opened rows are scored on is a column; the loop below reads only those the
  // header lists, so a column the page left out would otherwise go unchecked.
  const out: string[] = missingColumns({ ...fit, models: fit.models.filter((m) => m.holdout) }, header, (m) => m.holdout, "held-out");
  // A verdict column is drawn when, and only when, the server serves a verdict for a family.
  const verdictServed = fit.models.some((m) => m.holdout && m.holdout_detail?.versus_baseline);
  if (header.some((h) => h.startsWith("Against")) !== verdictServed) out.push(`the verdict column is ${verdictServed ? "missing" : "drawn though no verdict is served"}`);
  for (const m of fit.models) {
    if (!m.holdout) continue;
    const cells = rows[m.family];
    if (!cells) {
      out.push(`${m.family}: no row`);
      continue;
    }
    header.forEach((label, i) => {
      const cell = cells[i] ?? "";
      const key = keyOf(label);
      if (key) {
        const iv = m.holdout_detail?.intervals?.[key];
        const served = iv ? [iv.estimate, iv.ci_low, iv.ci_high] : [m.holdout?.[key]];
        out.push(misprinted(cell, served, `${m.family} ${label}`));
      } else if (label.startsWith("Against")) {
        const v = m.holdout_detail?.versus_baseline;
        out.push(v ? verdictDifferences(cell, v, fit.metric_labels, m.family) : cell ? `${m.family}: a verdict "${cell}" is printed though none is served` : "");
      }
    });
  }
  const base = fit.models.find((m) => m.holdout_detail?.baseline)?.holdout_detail?.baseline;
  if (!base) out.push("no held-out baseline is served");
  else {
    const cells = rows["baseline"];
    if (!cells) out.push("the baseline has no row");
    else
      header.forEach((label, i) => {
        const key = keyOf(label);
        // The baseline is scored on the metrics it is served for; the others print nothing.
        if (key) out.push(misprinted(cells[i] ?? "", base.scores[key] === undefined ? [] : [base.scores[key]], `baseline ${label}`));
      });
  }
  return out.filter(Boolean);
}

/**
 * The calibration paragraph under the final model's picture against the served calibration
 * (calibration-in-the-large and the slope, with their intervals; what was predicted and observed
 * on average). Where the engine flags a concern the paragraph is its words, so those are looked for
 * instead. Returns what differs.
 */
export async function calibrationDifferences(page: Page, fit: FitArtifact): Promise<string[]> {
  const final = fit.models.find((m) => m.family === fit.final_model);
  const cal = final?.holdout_detail?.calibration;
  if (!cal) return ["no held-out calibration is served for the final model"];
  const line = page.getByTestId("results-calibration").getByTestId("calibration-line");
  if (!(await line.count())) return ["the calibration paragraph is not drawn"];
  const text = ((await line.innerText()) ?? "").replaceAll("−", "-").replace(/(\d),(?=\d{3}\b)/g, "$1");
  if (cal.concern) {
    const said = cal.concern.replaceAll("`", "");
    return text.includes(said.replaceAll("−", "-").replace(/(\d),(?=\d{3}\b)/g, "$1")) ? [] : [`the paragraph does not hold the engine's concern: "${said}"`];
  }
  const out: string[] = [];
  const risk = fit.task === "binary";
  const avg = /Predicted (-?[\d.]+)(%?) on average against (-?[\d.]+)%? observed/.exec(text);
  if (!avg) out.push(`the paragraph does not say what was predicted on average and observed: "${text}"`);
  else if (risk) {
    // A risk is said as a percentage with one decimal.
    for (const [got, served, what] of [[avg[1], cal.expected, "predicted"], [avg[3], cal.observed, "observed"]] as const) {
      if (Math.abs(Number(got) - served * 100) > 0.05 + 1e-9) out.push(`${what} on average: printed ${got}%, served ${served * 100}%`);
    }
  } else {
    if (!printsServed(Number(avg[1]), cal.expected)) out.push(`predicted on average: printed ${avg[1]}, served ${cal.expected}`);
    if (!printsServed(Number(avg[3]), cal.observed)) out.push(`observed on average: printed ${avg[3]}, served ${cal.observed}`);
  }
  const intercept = /calibration intercept (-?[\d.e+-]+)(?:, [\d.]+% interval (-?[\d.e+-]+) to (-?[\d.e+-]+))?/.exec(text);
  if (cal.intercept.estimate !== null) {
    if (!intercept) out.push("the calibration intercept is not printed");
    else {
      const served = [cal.intercept.estimate, cal.intercept.ci_low, cal.intercept.ci_high];
      out.push(misprinted(intercept.slice(1).filter((x) => x !== undefined).join(" "), served, "calibration intercept"));
    }
  }
  if (cal.slope.estimate !== null) {
    const slope = /Outcomes moved (-?[\d.e+-]+) times as far as the predictions did \(calibration slope(?:, [\d.]+% interval (-?[\d.e+-]+) to (-?[\d.e+-]+))?/.exec(text);
    if (!slope) out.push("the calibration slope is not printed");
    else {
      const served = [cal.slope.estimate, cal.slope.ci_low, cal.slope.ci_high];
      out.push(misprinted(slope.slice(1).filter((x) => x !== undefined).join(" "), served, "calibration slope"));
    }
  }
  return out.filter(Boolean);
}

/** The cross-validated comparison against the served artifact: each family's score on each metric
 *  with its interval, and the no-predictor baseline's. Returns what differs. */
export async function cvDifferences(page: Page, fit: FitArtifact): Promise<string[]> {
  const table = page.getByTestId("results-cv").locator("table");
  if (!(await table.count())) return ["the cross-validated table is not drawn"];
  const { header, rows, labels } = await readTable(table);
  const keyOf = (label: string) => Object.entries(fit.metric_labels).find(([, l]) => l === label)?.[0];
  // Every metric some family is scored on is a column; the loops below read only those the header
  // lists, so a column the page left out would otherwise go unchecked.
  const out: string[] = missingColumns(fit, header, (m) => m.cv, "cross-validated");
  // A verdict column is drawn when, and only when, the server serves a verdict for a family.
  const verdictServed = fit.models.some((m) => m.versus_baseline);
  if (header.some((h) => h.startsWith("Against")) !== verdictServed) out.push(`the verdict column is ${verdictServed ? "missing" : "drawn though no verdict is served"}`);
  for (const m of fit.models) {
    const cells = rows[m.family];
    if (!cells) {
      out.push(`${m.family}: no row`);
      continue;
    }
    header.forEach((label, i) => {
      const cell = cells[i] ?? "";
      const key = keyOf(label);
      if (key) {
        // A metric the family was not scored on prints nothing.
        const s = m.cv[key];
        out.push(misprinted(cell, s ? [s.estimate ?? s.mean, s.ci_low, s.ci_high] : [], `${m.family} ${label} (cross-validated)`));
      } else if (label.startsWith("Against")) {
        const v = m.versus_baseline;
        out.push(v ? verdictDifferences(cell, v, fit.metric_labels, `${m.family} (cross-validated)`) : cell ? `${m.family}: a verdict "${cell}" is printed though none is served` : "");
      }
    });
  }
  // The no-predictor baseline is scored on one metric; the other columns print nothing for it.
  const base = fit.models.find((m) => m.baseline.value !== null)?.baseline;
  if (base) {
    const cells = rows["baseline"];
    if (!cells) out.push("the baseline has no row (cross-validated)");
    else
      header.forEach((label, i) => {
        const key = keyOf(label);
        if (key) out.push(misprinted(cells[i] ?? "", key === base.metric ? [base.value] : [], `baseline ${label} (cross-validated)`));
      });
  }
  // BBC-CV: the row for the best family corrected for having been chosen, drawn when, and only
  // when, the server serves the correction: its metric's estimate with its interval, the other
  // metrics' from ``extras``, and nothing for a metric the correction was not made on.
  const sel = fit.selection;
  const corrected = rows["corrected"];
  if (!sel) {
    if (corrected) out.push("a BBC-CV row is drawn though no selection correction is served");
  } else if (!corrected) {
    out.push("the BBC-CV row is not drawn though the selection correction is served");
  } else {
    const best = fit.models.find((m) => m.family === sel.best)?.label ?? sel.best;
    if (!(labels["corrected"] ?? "").includes(best)) out.push(`the BBC-CV row is labeled "${labels["corrected"]}", not for ${best}`);
    header.forEach((label, i) => {
      const key = keyOf(label);
      if (!key) return;
      const e = sel.extras[key];
      const served =
        key === sel.metric
          ? [sel.corrected, sel.corrected_low, sel.corrected_high]
          : e && e["corrected"] !== undefined
            ? [e["corrected"], e["corrected_low"], e["corrected_high"]]
            : [];
      out.push(misprinted(corrected[i] ?? "", served, `${best} corrected for choosing it (BBC-CV), ${label}`));
    });
  }
  return out.filter(Boolean);
}

// ── Fit, at the analysis flowchart's end ─────────────────────────────────────

/**
 * What the shell shows of Fit, read in one pass of the page's DOM so that nothing can change
 * between the reads: the stage shown, whether the Fit position is drawn and marked running, its
 * progress and Cancel, and the refusal it shows, if any.
 */
async function readFitPosition(page: Page): Promise<{ stage: string | null; running: boolean; progress: boolean; cancel: boolean; refusal: string | null }> {
  return page.evaluate(() => {
    const current = document.querySelector('[data-testid^="stage-"][aria-current="step"]')?.getAttribute("data-testid") ?? null;
    const position = document.querySelector('[data-testid="fit-position"]');
    return {
      stage: current === null ? null : current.replace(/^stage-/, ""),
      running: position?.hasAttribute("data-running") ?? false,
      progress: document.querySelector('[data-testid="fit-progress"]') !== null,
      cancel: document.querySelector('[data-testid="fit-cancel"]') !== null,
      refusal: position?.querySelector('[role="alert"]')?.textContent?.trim() ?? null,
    };
  });
}

/**
 * Fit pressed where the shell offers it (Models, once its lines are answered), and Results awaited.
 * Before the press the position shows the engine's own reason and no Cancel; while the fit job runs
 * it shows its progress and a Cancel (W1_PART2_PLAN ruling 3), and the page must say so whenever
 * the server serves the job as running. Returns whether the press found the job still running (a
 * short fit may have computed already, which the server then serves at once).
 *
 * The fit is held for the press or it is not, whichever the runner's speed makes of its estimate
 * (fit_press.HOLD_SECONDS), and both are walked as the engine serves them. Before the press the
 * position reads the lock's reason in the engine's words; a held fit's reason says how long it is
 * expected to take, and the button reads the same, and no hold is claimed otherwise. A held fit
 * computes only once it is pressed, so the wait for Results covers the whole fit: it goes on
 * while the fit job runs (up to ``cap``) and gives up once none has for ``quiet``, or at once if
 * the press is refused or the fit fails.
 *
 * What the server serves is older than the page by the time it is read: the shell opens Results the
 * moment the fit is computed, and Results has no Fit position. A reading that says the job runs is
 * therefore not held against a page that has moved on. The page must show the job (the position
 * marked running) or Results; while Models shows it, its progress and Cancel are read in the same
 * pass as the mark, and Models that still does not show a job the server serves as running
 * ``shown`` ms after the reading fails.
 */
export async function pressFit(
  page: Page,
  pid: string,
  { quiet = FIT_QUIET, cap = FIT_CAP, shown = 15_000 }: { quiet?: number; cap?: number; shown?: number } = {},
): Promise<boolean> {
  await openStage(page, "models");
  const position = page.getByTestId("fit-position");
  await expect(position).toBeVisible();
  const before = await getQuest(page, pid);
  expect(before.fit?.pressed, "Fit is not pressed yet").toBe(false);
  await expect(position).toContainText(before.fit!.reason!);
  if (before.fit!.held) {
    // A fit held for the press says how long it is expected to take, in the engine's own words,
    // and computes only once Fit is pressed (a slower runner reaches the hold with the same families).
    const said = /expected to take ([^.]+)\./.exec(before.fit!.reason ?? "")?.[1];
    expect(said, "the lock's reason says how long a held fit takes").toBeDefined();
    await expect(position).toHaveAttribute("data-held", "true");
    await expect(page.getByTestId("fit")).toHaveText(`Fit · ${said}`);
  } else {
    await expect(position).not.toHaveAttribute("data-held", "true");
    await expect(page.getByTestId("fit")).toHaveText("Fit");
  }
  await expect(page.getByTestId("fit-cancel"), "no Cancel while nothing runs").toHaveCount(0);
  await page.getByTestId("fit").click();
  let sawRunning = false;
  const start = Date.now();
  let busyAt = start;
  let firstSeen: number | null = null;
  const secondsIn = () => Math.round((Date.now() - start) / 1_000);
  for (;;) {
    const look = await readFitPosition(page);
    if (look.stage === "results") {
      // Reported with the run (the report's annotations): whether the press found the fit held, how
      // long after it the fit job was first seen, and how long Results took.
      test.info().annotations.push({
        type: "fit press",
        description: `${before.fit!.held ? "held for the press" : "not held"}; after the press the fit job was first seen ${firstSeen === null ? "never" : `${firstSeen} s in`}, Results ${secondsIn()} s in`,
      });
      return sawRunning;
    }
    if (look.refusal !== null) throw new Error(`Fit was refused: ${look.refusal}`);
    const view = await getJson<{ stages: Record<string, { status: string; error?: string | null }> }>(page, pid);
    const fit = view.stages.fit;
    let movedOn = false;
    if (["running", "queued"].includes(fit?.status ?? "")) {
      firstSeen ??= secondsIn();
      // The reading may be older than the page by the time it arrives, and a fit that finished
      // meanwhile has opened Results, which has no position to look at: the page shows the job,
      // or Results.
      const until = Date.now() + shown;
      for (;;) {
        const now = await readFitPosition(page);
        if (now.stage === "results") {
          movedOn = true;
          break;
        }
        if (now.running) {
          // One pass read the mark, the progress and the Cancel, so a fit that finishes after it
          // cannot have taken any of them away between the reads.
          expect(now.progress && now.cancel, `Models shows the fit running without its progress and Cancel (progress ${now.progress}, Cancel ${now.cancel})`).toBe(true);
          sawRunning = true;
          break;
        }
        expect(Date.now() < until, `the server serves the fit as ${fit!.status} but Models does not show it (the position is not marked running ${shown / 1_000} s after the reading)`).toBe(true);
        await page.waitForTimeout(250);
      }
      busyAt = Date.now();
    } else if (fit?.status === "error") {
      throw new Error(`the fit failed: ${fit.error ?? "no reason was given"}`);
    }
    if (movedOn) continue;
    const waited = Math.round((Date.now() - start) / 1_000);
    expect(Date.now() - start < cap, `the fit computes and Results opens (waited ${waited} s, to the ${cap / 1_000} s cap; fit ${fit?.status})`).toBe(true);
    expect(Date.now() - busyAt < quiet, `the fit computes and Results opens (waited ${waited} s; the fit job was not seen for ${quiet / 1_000} s; fit ${fit?.status})`).toBe(true);
    await page.waitForTimeout(1_000);
  }
}

// ── Results: the triage, the final model and the opening ─────────────────────

/**
 * Results from the pressed Fit to the opened held-out rows, through its cards: the shrinkage
 * question answered (the opening waits for it, and says so), the noticings left open confirmed
 * with the triage's one button (the opening waits for them too), the opening pressed and refused
 * until the final model is named, the family the first exit names recorded and the rows opened.
 * ``after`` runs at each step (the bar is checked against GET /quest there). Returns the fit as
 * served before the opening (its held-out scores sealed), the family named, and the fit after it.
 */
export async function openHeldOutRows(
  page: Page,
  pid: string,
  families: readonly string[],
  after: (step: string) => Promise<void>,
): Promise<{ sealed: FitArtifact; family: FitArtifact["models"][number]; opened: FitArtifact; triaged: number; refusals: string[] }> {
  const cards = page.getByTestId("results-cards");
  await expect(cards).toBeVisible();
  const sealed = await getFit(page, pid);
  expect(sealed.models.map((m) => m.holdout), "no held-out score is served before the opening").toEqual(families.map(() => null));
  await expect(page.getByTestId("results-tapestry")).toHaveAttribute("data-seal-phase", "sealed");
  await expect(page.getByTestId("results-held-out")).toContainText("sealed");
  await expect.poll(() => cvDifferences(page, sealed), { message: "the cross-validated scores equal the served fields", timeout: 30_000 }).toEqual([]);

  // The shrinkage question comes before the opening, and the press says so until it is answered.
  const open = page.getByTestId("results-open-press");
  await expect(open).toBeDisabled();
  await expect(page.getByTestId("results-open-hint")).toContainText("shrinkage question");
  await pickOption(page, cards, "option-method:none");
  await settled(page, "the shrinkage question is answered", async () => linesOf(await getQuest(page, pid)).find((x) => x.line.key === "set_updating")?.line.status === "answered");
  await after("after the shrinkage question");

  // The triage: each noticing left open, with the engine's recommended disposition.
  const waiting = linesOf(await getQuest(page, pid)).filter((x) => x.line.source === "finding" && x.line.status === "open");
  const triage = await getTriage(page, pid);
  expect(triage.items.length, "the noticings left open are the triage's items").toBe(waiting.length);
  expect(triage.answered).toBe(false);
  const card = page.getByTestId("results-triage");
  if (triage.items.length === 0) {
    await expect(card, "no noticing is open, so no triage is drawn").toHaveCount(0);
  } else {
    await expect(card).toBeVisible();
    await expect(card.locator("[data-item]")).toHaveCount(triage.items.length);
    for (const item of triage.items) await expect(card.locator(`[data-item="${item.id}"]`)).toContainText(item.label);
    const n = triage.items.length;
    await expect(open, "the opening waits for the triage").toBeDisabled();
    await expect(page.getByTestId("results-open-hint")).toContainText(`first confirm the ${n === 1 ? "noticing" : `${n} noticings`} above`);
    await expect(page.getByTestId("triage-confirm")).toHaveText(`Confirm all ${n}`);
    await page.getByTestId("triage-confirm").click();
    await settled(page, "the triage is confirmed", async () => (await getTriage(page, pid)).answered);
    await expect(page.getByTestId("triage-confirm")).toHaveCount(0);
    for (const item of (await getTriage(page, pid)).items) expect(item.recorded, `${item.id} has its disposition recorded`).toBe(item.recommended);
    await after("after the triage");
  }

  // The opening: refused until the final model is named, and while a noticing the triage marked
  // "act on it" is neither acted on nor recorded as a limitation; each refusal's first way through
  // is taken (each is a button that records, then retries), until the rows open once.
  const triaged = await getTriage(page, pid);
  const acting = triaged.items.filter((i) => i.recorded === "act_on_it");
  await settled(page, "the opening is enabled", () => enabledNow(open));
  await open.click();
  const refusals: string[] = [];
  let family: FitArtifact["models"][number] | undefined;
  const refused = page.getByTestId("results-open-refused");
  const rowsOpened = page.getByTestId("results-opened");
  for (let step = 0; step < families.length + triaged.items.length + 2; step++) {
    // Either the rows opened, or the engine refused again and says why: the answer is waited for
    // while the app computes (the opening reads the fit it opens).
    await settled(page, "the opening is answered", async () => (await rowsOpened.count()) > 0 || (await refused.count()) > 0);
    if (await rowsOpened.count()) break;
    const code = (await refused.getAttribute("data-code")) ?? "";
    const message = await refused.innerText();
    refusals.push(code);
    const exits = page.getByTestId("results-open-exit");
    expect(await exits.count(), `the refusal ${code} offers a way through`).toBeGreaterThan(0);
    if (code === "final_model_needed") {
      await expect(exits, "each family fitted is offered as the final model").toHaveCount(families.length);
      const text = (await exits.first().innerText()).trim();
      family = sealed.models.find((m) => text.startsWith(`Open with ${m.label} as the final model`));
      expect(family, `the first exit ("${text}") names a fitted family`).toBeDefined();
    } else {
      // A noticing marked "act on it" and not yet acted on: counted from the served triage, with a
      // way through (record it as a limitation) for each, and its own words naming how many.
      expect(code, "the only other refusal on the way is the open noticings'").toBe("triage_open");
      const left = acting.length - refusals.filter((c) => c === "triage_open").length + 1;
      expect(message, "the refusal counts the noticings marked act on it").toContain(`${left} marked "act on it"`);
      expect(await exits.count(), "each of them has its own exit").toBe(left);
    }
    await exits.first().click();
    // The exit records its answer and retries the opening: wait for that (the refusal gone, a new
    // one, or the rows opened), then hold the bar and the hover panels to GET /quest here too, since
    // each exit changes what the quest log says before the rows open.
    const shown = () => refused.evaluateAll((els) => (els[0] ? (els[0] as HTMLElement).innerText : null));
    await settled(page, `the exit of the refusal ${code} is answered`, async () => (await rowsOpened.count()) > 0 || (await shown()) !== message);
    await after(`after the exit of the refusal ${code}`);
  }
  await visibleWhileComputing(page, rowsOpened, [ANY_STAGE], { label: "the held-out rows are opened" });
  expect(refusals[0], "the first refusal asks for the final model").toBe("final_model_needed");
  expect(family, "a final model was named").toBeDefined();
  const opened = await getFit(page, pid);
  expect(opened.final_model, "the family named is the final model").toBe(family!.family);
  expect(opened.holdout_sealed).toBe(false);
  await after("after the opening");
  return { sealed, family: family!, opened, triaged: triage.items.length, refusals };
}

/**
 * The explanations asked on their card (the curves as accumulated local effects), and the SHAP
 * importance each explained family is drawn with against the served artifact: its rows, each input's
 * rank and its mean absolute SHAP value.
 */
export async function askExplanations(page: Page, pid: string): Promise<void> {
  await pickOption(page, page.getByTestId("results-cards"), "option-curves:ale");
  await settled(page, "the explanations are asked", async () => linesOf(await getQuest(page, pid)).find((x) => x.line.key === "set_explain")?.line.status === "answered");
  const box = page.getByTestId("results-explain");
  // The explanations compute after the question: waited for while the explain stage runs.
  await visibleWhileComputing(page, box.locator('[data-testid="results-importance"]').first(), ["explain"], {
    quiet: 120_000,
    cap: 900_000,
    label: "the explanations are drawn",
  });
  const explain = (await getJson<{ artifact: ExplainArtifact }>(page, pid, "/stages/explain")).artifact;
  for (const f of explain.families.filter((x) => x.explained)) {
    const family = box.locator(`[data-testid="results-importance"][data-family="${f.family}"]`);
    await expect(family.locator("tr[data-row]")).toHaveCount(f.importance.length);
    for (const row of f.importance) {
      const cells = await family.locator(`tr[data-row="${row.input}"] td`).allInnerTexts();
      expect(misprinted(cells[1] ?? "", [row.mean_abs], `${f.family} ${row.input} importance`)).toBe("");
      expect(Number(cells[0]), `${row.input}'s rank`).toBe(row.rank);
    }
  }
}

/**
 * The two comparisons shown to fail where they should, before a journey relies on them: a bar
 * segment the page marks wrongly (its answered count tampered with in the page, then put back), and
 * a served score one unit off the printed one. Each must be caught, and silent once put right.
 */
export async function comparisonsCatchWhatDiffers(page: Page, pid: string, opened: FitArtifact | null): Promise<void> {
  const segment = page.getByTestId("segment-data");
  const was = await segment.getAttribute("data-answered");
  await segment.evaluate((el) => el.setAttribute("data-answered", "999"));
  expect(await barDifferences(page, pid), "a segment marked wrongly is caught").toContain("data.answered");
  await segment.evaluate((el, v) => (v === null ? el.removeAttribute("data-answered") : el.setAttribute("data-answered", v)), was);
  expect(await barDifferences(page, pid), "the segment put right is not").toBe("");
  if (!opened) return;
  const wrong = structuredClone(opened);
  const model = wrong.models.find((m) => m.holdout_detail && Object.keys(m.holdout_detail.intervals ?? {}).length)!;
  const key = Object.keys(model.holdout_detail!.intervals)[0]!;
  const iv = model.holdout_detail!.intervals[key]!;
  iv.estimate = (iv.estimate ?? 0) + 1;
  expect((await heldOutDifferences(page, wrong)).join("; "), "a score one unit off the printed one is caught").toContain(`${model.family}`);
}

// ── the bundle ───────────────────────────────────────────────────────────────

/** The bundle downloaded from the Write-up's one button, its folder made here and removed by
 *  ``cleanup``; ``read`` gives a file of the zip by its path inside it. */
export async function downloadBundle(page: Page): Promise<{ name: string; files: string[]; read: (inside: string) => string; cleanup: () => void }> {
  const dir = mkdtempSync(join(tmpdir(), "quest-bundle-"));
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 180_000 }), page.getByTestId("primary").click()]);
  const zip = join(dir, download.suggestedFilename());
  await download.saveAs(zip);
  return {
    name: download.suggestedFilename(),
    files: execFileSync("unzip", ["-Z1", zip], { encoding: "utf-8" }).split("\n").filter(Boolean),
    read: (inside) => execFileSync("unzip", ["-p", zip, inside], { encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 }),
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
  };
}
