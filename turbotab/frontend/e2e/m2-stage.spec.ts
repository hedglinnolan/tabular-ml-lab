/**
 * The M2 stage (M2_CONTRACT §11), against the mock's M2 scenarios at /lab/stage/m2 — each a mock
 * project replaying the real server's answers on a sample fixture (src/mocks/m2-stage-fixture.json).
 *
 *   npm run test:e2e -- m2-stage        (Playwright starts the mock dev server)
 *
 * The reshape and the orientation turn play their storyboards on the flip (frame strips); the seal
 * draws its basis in all four states and never draws an undetermined one as a clean lock; the
 * Results hold the held-out scores sealed, open them once, and mark a later change post-seal; the
 * coach's notes stay inside their views; repairs, lens, outcome and purpose preview on the stage.
 * Every screen is audited against the purpose registry (BLUEPRINT §11.2). Writes, for review:
 *   docs/turbotab-next/m2/screens/stage-reshape-flip-<ms>ms.png      the reshape flip, 0 … 900 ms
 *   docs/turbotab-next/m2/screens/stage-orientation-flip-<ms>ms.png  the turn, 0 … 900 ms
 *   docs/turbotab-next/m2/screens/stage-<scene>-<theme>.png
 */
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { COACH_ANCHOR_PURPOSES, STAGE_PURPOSES, VIEW_PURPOSES } from "../src/components/stage/purposes";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCREENS = resolve(HERE, "../../../docs/turbotab-next/m2/screens");
const MAX_SCREEN_BYTES = 300 * 1024;
const FRAMES = [0, 150, 300, 450, 600, 900];

interface StageHook {
  manual: (v: boolean) => void;
  advance: (ms: number) => void;
  state: () => { pos: number; target: number; last: number; paused: boolean };
}
declare global {
  interface Window {
    __turbotabStage?: StageHook;
  }
}

async function shot(page: Page, name: string, target?: Locator) {
  mkdirSync(SCREENS, { recursive: true });
  const path = resolve(SCREENS, `${name}.png`);
  const box = target ? await target.boundingBox() : null;
  if (box) {
    const m = 10;
    await page.screenshot({ path, clip: { x: box.x - m, y: box.y - m, width: box.width + 2 * m, height: box.height + 2 * m } });
  } else await page.screenshot({ path });
  expect(statSync(path).size, `${name}.png stays under 300 KB`).toBeLessThan(MAX_SCREEN_BYTES);
}

async function open(page: Page, project: string, theme: "light" | "dark" = "light") {
  await page.emulateMedia({ colorScheme: theme, reducedMotion: "no-preference" });
  await page.goto(`/lab/stage/m2?p=${project}`);
  await expect(page.getByTestId("stage")).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
}

async function landed(page: Page, side: "now" | "with") {
  const player = page.getByTestId("player");
  await expect(player).toHaveAttribute("data-side", side);
  await expect(player).not.toHaveAttribute("data-moving", "true");
}

async function previewOption(page: Page, question: string | null, nth: number) {
  if (question) await page.locator(`[data-question=${question}]`).click();
  await page.locator("[data-option]").nth(nth).focus();
}

/**
 * Every view, composed picture, coach note and stage element on screen has a declared purpose
 * (the registry the vitest gate checks, here on the real screens of the journey).
 */
async function audit(page: Page) {
  const found = await page.evaluate(() => ({
    views: Array.from(document.querySelectorAll("[data-view]")).map((e) => e.getAttribute("data-view")!),
    elements: Array.from(document.querySelectorAll("[data-purpose]")).map((e) => e.getAttribute("data-purpose")!),
    anchors: Array.from(document.querySelectorAll("[data-anchor]")).map((e) => e.getAttribute("data-anchor")!),
  }));
  for (const v of found.views) expect((VIEW_PURPOSES as Record<string, unknown>)[v], `data-view="${v}"`).toBeDefined();
  for (const p of found.elements) expect(STAGE_PURPOSES[p], `data-purpose="${p}"`).toBeDefined();
  for (const a of found.anchors) expect((COACH_ANCHOR_PURPOSES as Record<string, unknown>)[a], `coach anchor "${a}"`).toBeDefined();
}

/** Step the player's clock by hand from "your data now" and save each frame of the flip. */
async function frameStrip(page: Page, prefix: string, settle = 120): Promise<number[]> {
  await page.evaluate(() => window.__turbotabStage!.manual(true));
  await page.getByTestId("flip-now").click();
  await page.evaluate(() => window.__turbotabStage!.advance(3000));
  await page.waitForTimeout(400);
  await page.getByTestId("flip-with").click();
  let t = 0;
  const positions: number[] = [];
  for (const ms of FRAMES) {
    await page.evaluate((d) => window.__turbotabStage!.advance(d), ms - t);
    t = ms;
    positions.push((await page.evaluate(() => window.__turbotabStage!.state())).pos);
    await page.waitForTimeout(settle);
    await shot(page, `${prefix}-${String(ms).padStart(3, "0")}ms`, page.getByTestId("stage"));
  }
  await page.evaluate(() => window.__turbotabStage!.manual(false));
  return positions;
}

test.describe.configure({ mode: "serial" });

test("the reshape plays gather → combine → settle on the flip, and the row map is shown", async ({ page }) => {
  await open(page, "m2-reshape");
  await previewOption(page, null, 0); // average each person's recalls
  await expect(page.getByTestId("stage-title")).toHaveText("Average each person's recalls");
  await landed(page, "with");
  const table = page.locator("[data-view=reshape_table]");
  await expect(table).toHaveAttribute("data-state", "3");
  await expect(table).toHaveAttribute("data-kind", "combine");
  await expect(page.locator("[aria-label^='Step ']")).toHaveCount(4);
  await expect(page.getByTestId("readout")).toContainText("600");
  await expect(page.getByTestId("readout")).toContainText("300");
  // The row map: each settled row says which rows of the file it stands for.
  await expect(table.locator("[data-row='0']")).toContainText("0+1");
  await expect(table).toContainText("from rows");
  // The row flow says the rows were folded in, not dropped.
  await expect(page.locator("[data-step=combined]")).toContainText("300 folded in");
  // The coach points at the picture: at most two notes, inside the card.
  const card = page.locator("[data-primary]");
  const notes = card.getByTestId("coach-note");
  await expect(notes.first()).toBeVisible();
  expect(await notes.count()).toBeLessThanOrEqual(2);
  const cb = (await card.boundingBox())!;
  for (const n of await notes.all()) {
    const b = (await n.boundingBox())!;
    expect(b.x).toBeGreaterThanOrEqual(cb.x);
    expect(b.x + b.width).toBeLessThanOrEqual(cb.x + cb.width + 0.5);
  }
  await audit(page);
  await page.waitForTimeout(350);
  await shot(page, "stage-reshape-mean-light");

  // Each state of the storyboard, from its dot: gathered, then combined.
  await page.locator("[aria-label^='Step 2 of 4']").click();
  await expect(table).toHaveAttribute("data-state", "1");
  await expect(page.getByTestId("step-label")).toHaveText(/Each participant_id's records/);
  await expect(table.locator("[data-row='1']")).toContainText("1");
  await page.locator("[aria-label^='Step 4 of 4']").click();
  await landed(page, "with");

  // Another method morphs straight to its result: the flip holds, the storyboard does not replay.
  await page.locator("[data-option]").nth(0).focus(); // back to the options (the dot took focus)
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  await expect(page.getByTestId("stage-title")).toHaveText("Keep each person's last recall");
  const st = await page.evaluate(() => window.__turbotabStage!.state());
  expect(st).toMatchObject({ pos: st.last, target: st.last });
  await expect(table).toHaveAttribute("data-kind", "keep");
  // The kept record is that record (row 1, the last recall), never its unit's first row.
  await expect(table.locator("[data-row='1']")).toHaveCSS("opacity", "1");
  await expect(table.locator("[data-row='0']")).toHaveCSS("opacity", "0");
  await page.waitForTimeout(350);
  await shot(page, "stage-reshape-last-light");
});

test("the reshape flip, frame by frame (0 → 900 ms)", async ({ page }) => {
  await open(page, "m2-reshape");
  await previewOption(page, null, 0);
  await landed(page, "with");
  const positions = await frameStrip(page, "stage-reshape-flip");
  expect(positions[0]).toBe(0);
  expect(positions.at(-1)).toBe(3);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  writeFileSync(resolve(SCREENS, "stage-frames.json"), JSON.stringify({ reshape: { ms: FRAMES, positions } }, null, 1));
});

test("the orientation turn, and its flip frame by frame", async ({ page }) => {
  await open(page, "m2-orientation");
  await previewOption(page, null, 0); // rows are features: turn the table
  await landed(page, "with");
  const turn = page.locator("[data-view=turn_table]");
  await expect(turn).toHaveAttribute("data-state", "3");
  const readout = page.getByTestId("readout");
  await expect(readout).toContainText("396");
  await expect(readout).toContainText("80");
  await expect(readout).toContainText("397");
  await audit(page);
  await page.waitForTimeout(300);
  await shot(page, "stage-orientation-light");
  const positions = await frameStrip(page, "stage-orientation-flip");
  expect(positions[0]).toBe(0);
  expect(positions.at(-1)).toBe(3);
  const prev = JSON.parse((await import("node:fs")).readFileSync(resolve(SCREENS, "stage-frames.json"), "utf-8"));
  writeFileSync(
    resolve(SCREENS, "stage-frames.json"),
    JSON.stringify({ ...prev, orientation: { ms: FRAMES, positions } }, null, 1),
  );
  // Keeping the table as supplied changes nothing, and says so.
  await page.locator("[data-option]").nth(1).focus();
  await expect(page.getByTestId("stage-title")).toHaveText("Rows are samples: keep it");
});

const BASES = [
  { project: "m2-seal-grouped", basis: "grouped", glyph: "closed", both: "0 units on both sides", kicker: "Would seal" },
  { project: "m2-seal-chronological", basis: "grouped", glyph: "closed", both: "0 units on both sides", kicker: "Would seal" },
  { project: "m2-seal-abandoned", basis: "abandoned", glyph: "abandoned", both: "units on both sides", kicker: "Sealed by row · exploratory" },
  { project: "m2-seal-undetermined", basis: "undetermined", glyph: "undetermined", both: "on both sides: not known", kicker: "Sealed by row · exploratory" },
];

for (const b of BASES) {
  test(`the seal states its basis: ${b.project.replace("m2-seal-", "")}`, async ({ page }) => {
    await open(page, b.project);
    await previewOption(page, "split", 1); // hold out 20%
    await landed(page, "with");
    const fork = page.locator("[data-view=seal_fork]");
    await expect(fork).toHaveAttribute("data-basis", b.basis);
    await expect(fork.locator("[data-seal]")).toHaveAttribute("data-seal", b.glyph);
    await expect(page.getByTestId("seal-basis")).toContainText(b.kicker);
    await expect(page.getByTestId("seal-basis")).toContainText(b.both);
    if (b.glyph !== "closed") {
      // Never a clean lock: amber, and the frame's own label says exploratory.
      await expect(fork.locator("[data-seal]")).toHaveAttribute("data-tone", "warn");
      await expect(page.getByTestId("stage-caution")).toBeVisible();
    }
    if (b.basis === "undetermined") await expect(page.getByTestId("seal-evidence")).toContainText("participant_id");
    if (b.project.endsWith("chronological")) await expect(page.getByTestId("seal-basis")).toContainText("chronological");
    await audit(page);
    await page.waitForTimeout(300);
    await shot(page, `stage-seal-${b.project.replace("m2-seal-", "")}-light`);

    // Recording draws the seal; the live flow names its basis with the recorded glyph.
    await page.keyboard.press("Enter");
    const live = page.getByTestId("live-seal");
    await expect(live).toBeVisible();
    await expect(live.locator("[data-seal]")).toHaveAttribute("data-seal", b.glyph);
    if (b.glyph === "closed") await expect(live.locator("[data-seal]")).toHaveAttribute("data-tone", "ok");
    else await expect(live).toContainText("exploratory");
  });
}

test("the seal fork, frame by frame on the grouped draw", async ({ page }) => {
  await open(page, "m2-seal-abandoned");
  await previewOption(page, "split", 1);
  await landed(page, "with");
  const positions = await frameStrip(page, "stage-seal-flip", 60);
  expect(positions.at(-1)).toBe(4);
});

test("the Results keep held-out scores sealed, open them once, and mark a later change", async ({ page }) => {
  await open(page, "m2-results");
  const results = page.getByTestId("results");
  await expect(results).toHaveAttribute("data-seal-phase", "sealed");
  const cmp = page.getByTestId("model-comparison");
  await expect(cmp.locator("[data-family]")).toHaveCount(3);
  await expect(page.getByTestId("held-out-sealed")).toBeVisible();
  await expect(cmp.getByTestId("held-out-cell-sealed")).toHaveCount(3);
  // No held-out number anywhere before the seal is opened: not in a cell, a tooltip or the basis.
  await expect(cmp).not.toContainText("held out −");
  await expect(cmp).not.toContainText(/held out \d/);
  await expect(page.locator("circle[class*=hollow]")).toHaveCount(0);
  await expect(cmp).toContainText("held-out rows sealed");
  const card = page.getByTestId("open-seal-card");
  await expect(card).toContainText("Opened once");
  await expect(card).toContainText("It happens once.");
  await audit(page);
  await shot(page, "stage-results-sealed-light");
  await card.scrollIntoViewIfNeeded();
  await shot(page, "stage-results-open-card-light", card);

  // Open it, once: the card settles into a recorded line, and the scores arrive.
  await page.getByTestId("open-seal").click();
  await expect(page.getByTestId("seal-opened")).toContainText("The seal was opened once");
  await expect(results).toHaveAttribute("data-seal-phase", "opened");
  await expect(page.getByTestId("open-seal-card")).toHaveCount(0);
  // One score per family in its cell (the hover tip repeats it, and may: the seal is open).
  await expect(cmp.locator("text=/^held out [−\\d]/")).toHaveCount(3);
  await expect(page.getByTestId("held-out-opened")).toContainText("opened once");
  await page.getByTestId("stage").locator("[class*=scene]").first().evaluate((el) => el.scrollTo(0, 0));
  await page.waitForTimeout(400);
  await audit(page);
  await shot(page, "stage-results-opened-light");

  // A change after the opening still refits, and the Results say so.
  await previewOption(page, "energy_adjustment", 1); // nutrient density
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("post-seal")).toBeVisible();
  await expect(page.getByTestId("post-seal")).toContainText("The energy adjustment (#15) changed");
  await expect(results).toHaveAttribute("data-seal-phase", "post_seal");
  await expect(page.getByTestId("seal-opened")).toHaveAttribute("data-post", "true");
  await page.getByTestId("stage").locator("[class*=scene]").first().evaluate((el) => el.scrollTo(0, 0));
  await page.waitForTimeout(400);
  await audit(page);
  await shot(page, "stage-results-post-seal-light");
});

test("the coach annotates every view kind it is given, inside the view, in amber", async ({ page }) => {
  await open(page, "m2-seal-grouped");
  await previewOption(page, "exclusions", 1); // sex-neutral 500–5,000 kcal
  await landed(page, "with");
  const dist = page.locator("[data-card=distribution]");
  await expect(dist.getByTestId("coach-note")).toHaveCount(2);
  await expect(dist.locator("[data-anchor-mark=bracket]")).toHaveCount(2);
  await expect(page.locator("[data-step^='exclusion'] [data-testid=coach-note]")).toHaveCount(1);
  // The note is the coach's color, never the accent or the stop.
  const color = await dist.getByTestId("coach-note").first().evaluate((el) => getComputedStyle(el).color);
  const warn = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--warn").trim());
  expect(color.replace(/\s/g, "")).toBe(hexToRgb(warn));
  for (const card of await page.locator("[data-card]").all()) {
    const cb = (await card.boundingBox())!;
    for (const n of await card.getByTestId("coach-note").all()) {
      const b = (await n.boundingBox())!;
      expect(b.x).toBeGreaterThanOrEqual(cb.x - 0.5);
      expect(b.x + b.width).toBeLessThanOrEqual(cb.x + cb.width + 0.5);
    }
  }
  await audit(page);
  await page.waitForTimeout(300);
  await shot(page, "stage-coach-exclusions-light");

  await open(page, "m2-results");
  await previewOption(page, "energy_adjustment", 0); // residual
  await landed(page, "with");
  await expect(page.locator("[data-card=relationship] [data-testid=coach-note]")).toHaveCount(2);
  await audit(page);
  await page.waitForTimeout(300);
  await shot(page, "stage-coach-energy-light");
});

test("repairs, the lens, the outcome and the purpose preview on the stage", async ({ page }) => {
  await open(page, "m2-repairs");
  await previewOption(page, null, 0);
  await landed(page, "with");
  await expect(page.locator("[data-card=table_focus]")).toBeVisible();
  const dist = page.locator("[data-card=distribution]");
  await expect(dist).toBeVisible();
  await expect(dist).toContainText("9"); // the sentinel code, marked on its axis
  await audit(page);
  await page.waitForTimeout(300);
  await shot(page, "stage-repair-preview-light");
  const finding = page.locator("[data-finding]").first();
  await finding.click();
  await expect(page.getByTestId("stage-pill")).toHaveText("Evidence");

  await open(page, "m2-opening");
  await previewOption(page, "lens", 0);
  await expect(page.getByTestId("stage-note")).toContainText("findings on this table");
  await expect(page.getByTestId("stage")).not.toContainText("Nothing");
  await expect(page.locator("[data-view=lineage]")).toBeVisible();
  await audit(page);
  await page.waitForTimeout(300);
  await shot(page, "stage-lens-light");
  await previewOption(page, "target", 0);
  await expect(page.locator("[data-view=distribution]")).toBeVisible();
  await expect(page.getByTestId("stage")).not.toContainText("Nothing");
  await audit(page);
  await previewOption(page, "purpose", 1);
  await expect(page.getByTestId("stage-note")).toContainText("inference");
  await expect(page.getByTestId("stage")).not.toContainText("Nothing");
});

for (const theme of ["dark"] as const) {
  test(`the M2 stage in ${theme}`, async ({ page }) => {
    await open(page, "m2-reshape", theme);
    await previewOption(page, null, 0);
    await landed(page, "with");
    await page.waitForTimeout(350);
    await shot(page, `stage-reshape-mean-${theme}`);
    await open(page, "m2-seal-undetermined", theme);
    await previewOption(page, "split", 1);
    await landed(page, "with");
    await page.waitForTimeout(300);
    await shot(page, `stage-seal-undetermined-${theme}`);
    await open(page, "m2-results", theme);
    await expect(page.getByTestId("open-seal-card")).toBeVisible();
    await shot(page, `stage-results-sealed-${theme}`);
    await open(page, "m2-seal-grouped", theme);
    await previewOption(page, "exclusions", 1);
    await landed(page, "with");
    await page.waitForTimeout(300);
    await shot(page, `stage-coach-exclusions-${theme}`);
  });
}

test("reduced motion: the flip lands in one frame", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.goto("/lab/stage/m2?p=m2-reshape");
  await previewOption(page, null, 0);
  await landed(page, "with");
  await page.getByTestId("flip-now").click();
  const s = await page.evaluate(() => window.__turbotabStage!.state());
  expect(s.pos).toBe(0);
  await expect(page.locator("[data-view=reshape_table]")).toHaveAttribute("data-state", "0");
});

function hexToRgb(hex: string): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1]!, 16);
  return `rgb(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255})`;
}
