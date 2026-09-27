/// <reference types="node" />
/**
 * /lab/explore/inline — review captures and the two behaviors the brief requires:
 *
 *   1. keyboard: options are reachable and previewable with the arrow keys; Enter records;
 *      a refused option is focusable, says why, and cannot be recorded;
 *   2. prefers-reduced-motion: the morph is instant (the dots are at their target the moment
 *      the key is pressed), and without it they are not;
 *   3. screenshots at 1440 × 900, light and dark, each under 300 KB;
 *   4. the frame strip of S1 moving from residual to density + energy, on a controlled clock;
 *   5. the visible word count of every captured state, counted in the page.
 *
 *   npx playwright test --config src/explore/inline/capture/playwright.config.ts
 */
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../../../..");
const OUT = resolve(REPO, "docs/turbotab-next/m1/explore/inline");
const FRAMES = resolve(OUT, "frames");
const MAX_BYTES = 300 * 1024;
const URL = (s: string) => `/lab/explore/inline?s=${s}`;

type Theme = "light" | "dark";
type Step = { focus: string } | { key: string } | { click: string };

interface Shot {
  name: string;
  label: string;
  scenario: string;
  steps: Step[];
  themes: Theme[];
}

const BOTH: Theme[] = ["light", "dark"];

const SHOTS: Shot[] = [
  { name: "s1-idle", label: "S1 energy · idle", scenario: "energy", steps: [], themes: BOTH },
  {
    name: "s1-residual",
    label: "S1 energy · previewing residual",
    scenario: "energy",
    steps: [{ focus: "residual" }],
    themes: BOTH,
  },
  {
    name: "s1-density",
    label: "S1 energy · previewing density + energy",
    scenario: "energy",
    steps: [{ focus: "residual" }, { key: "ArrowRight" }],
    themes: BOTH,
  },
  {
    name: "s1-compare",
    label: "S1 energy · residual pinned, density + energy in view",
    scenario: "energy",
    steps: [{ focus: "residual" }, { key: "p" }, { key: "ArrowRight" }],
    themes: ["light"],
  },
  {
    name: "s1-refused",
    label: "S1 energy · the refused partition method",
    scenario: "energy",
    steps: [{ focus: "partition" }],
    themes: ["light"],
  },
  {
    name: "s2-idle",
    label: "S2 exclusions · idle",
    scenario: "exclusions",
    steps: [],
    themes: ["light"],
  },
  {
    name: "s2-sex-specific",
    label: "S2 exclusions · previewing the sex-specific rule",
    scenario: "exclusions",
    steps: [{ focus: "sex_specific" }],
    themes: BOTH,
  },
  {
    name: "s3-findings",
    label: "S3 findings · default",
    scenario: "findings",
    steps: [],
    themes: BOTH,
  },
  {
    name: "s3-paged",
    label: "S3 findings · a same-kind group paged open",
    scenario: "findings",
    steps: [{ click: "button[aria-expanded]" }],
    themes: ["light"],
  },
  { name: "s4-idle", label: "S4 wide · idle", scenario: "wide", steps: [], themes: ["light"] },
  {
    name: "s4-wide",
    label: "S4 wide · previewing log2(x + 1) on 495 columns",
    scenario: "wide",
    steps: [{ focus: "log2" }],
    themes: BOTH,
  },
];

async function settle(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(650); // the morph is 300 ms; number tweens 350 ms
}

async function run(page: Page, steps: Step[]) {
  for (const st of steps) {
    if ("focus" in st) await page.locator(`[data-option="${st.focus}"]`).focus();
    else if ("key" in st) await page.keyboard.press(st.key);
    else await page.locator(st.click).first().click();
    await page.waitForTimeout(80);
  }
}

interface Count {
  total: number;
  /** §03: serif — the app speaking (questions, consequences, claims). */
  voice: number;
  /** sans — controls and labels. */
  action: number;
  /** mono — data: column names, values, counts, axis ticks. */
  data: number;
  /** Header, scenario tabs and the motion toggle: lab chrome, not the design. */
  chrome: number;
}

/** Words a reader can see: text nodes that render inside the viewport, with a letter or digit. */
async function visibleWords(page: Page): Promise<Count> {
  return page.evaluate(() => {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const c = { total: 0, voice: 0, action: 0, data: 0, chrome: 0 };
    const chrome = [...document.querySelectorAll("header, [role=tablist]")];
    const toggle = [...document.querySelectorAll("label")].filter((l) =>
      l.textContent?.includes("Reduce motion"),
    );
    while (walker.nextNode()) {
      const node = walker.currentNode as Text;
      const text = node.textContent ?? "";
      if (!text.trim()) continue;
      const el = node.parentElement;
      if (!el || el.closest("title, script, style, .visually-hidden")) continue;
      // A closed <details> lays out nothing but its summary.
      const det = el.closest("details");
      if (det && !det.open && !el.closest("summary")) continue;
      let op = 1;
      for (let e: Element | null = el; e; e = e.parentElement) {
        const cs = getComputedStyle(e);
        if (cs.visibility === "hidden" || cs.display === "none") op = 0;
        op *= Number(cs.opacity);
      }
      if (op < 0.1) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      const seen = [...range.getClientRects()].some(
        (r) =>
          r.width > 1 && r.height > 1 && r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw,
      );
      if (!seen) continue;
      const k = text.split(/\s+/).filter((t) => /[\p{L}\p{N}]/u.test(t)).length;
      c.total += k;
      if ([...chrome, ...toggle].some((h) => h.contains(el))) {
        c.chrome += k;
        continue;
      }
      const family = getComputedStyle(el).fontFamily.toLowerCase();
      if (/mono|consolas/.test(family)) c.data += k;
      else if (/charter|georgia|serif/.test(family) && !/sans-serif/.test(family)) c.voice += k;
      else c.action += k;
    }
    return c;
  });
}

test.describe.configure({ mode: "serial" });

test("keyboard previews every option; reduced motion makes the morph instant", async ({
  browser,
}) => {
  for (const reduced of [false, true]) {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      reducedMotion: reduced ? "reduce" : "no-preference",
    });
    const page = await ctx.newPage();
    await page.goto(URL("energy"));
    await page.locator('[role="radiogroup"]').waitFor();

    // Tab reaches the strip (one stop), and focusing it previews the first option.
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press("Tab");
      if (await page.evaluate(() => document.activeElement?.getAttribute("role") === "radio"))
        break;
    }
    await expect(page.locator('[data-option="residual"]')).toHaveAttribute("aria-checked", "true");
    await expect(page.getByRole("button", { name: "Use the residual method" })).toBeEnabled();
    await page.waitForTimeout(700);

    const dot = page.locator('[data-dots="stage"] circle').nth(7);
    const before = await dot.getAttribute("cy");
    await page.keyboard.press("ArrowRight");
    const soon = await dot.getAttribute("cy");
    await expect(page.locator('[data-option="density_multivariate"]')).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await page.waitForTimeout(700);
    const after = await dot.getAttribute("cy");
    expect(after).not.toBe(before);
    if (reduced) expect(soon).toBe(after);
    else expect(soon).not.toBe(after);

    // Every option is reachable; the refused one is focusable, explains itself, records nothing.
    await page.keyboard.press("End");
    const partition = page.locator('[data-option="partition"]');
    await expect(partition).toBeFocused();
    await expect(partition).toHaveAttribute("aria-disabled", "true");
    await expect(page.locator("[data-stage]").getByText("has no energy factor")).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(page.locator('[data-testid="q-energy"]')).toBeVisible();

    // Home, then Enter records; "change" reopens the question.
    await page.keyboard.press("Home");
    await page.keyboard.press("Enter");
    await expect(page.getByText("Energy was adjusted by the residual method")).toBeVisible();
    await page.getByRole("button", { name: "Change the energy adjustment" }).click();
    await expect(page.locator('[data-testid="q-energy"]')).toBeVisible();
    await ctx.close();
  }
});

test("screenshots and visible word counts", async ({ browser }) => {
  mkdirSync(OUT, { recursive: true });
  const counts: Record<string, Count> = {};
  for (const theme of BOTH) {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: theme,
      reducedMotion: "no-preference",
    });
    const page = await ctx.newPage();
    for (const shot of SHOTS) {
      if (!shot.themes.includes(theme)) continue;
      await page.goto(URL(shot.scenario));
      await page.locator('[role="tabpanel"]').waitFor();
      await page.mouse.move(1430, 890);
      await run(page, shot.steps);
      await settle(page);
      const file = resolve(OUT, `${shot.name}-${theme}.png`);
      await page.screenshot({ path: file });
      expect(statSync(file).size, `${shot.name}-${theme}.png size`).toBeLessThan(MAX_BYTES);
      if (theme === "light") counts[shot.label] = await visibleWords(page);
    }
    await ctx.close();
  }
  writeFileSync(resolve(OUT, "word-counts.json"), `${JSON.stringify(counts, null, 2)}\n`);
});

test("frame strip: residual → density + energy", async ({ browser }) => {
  mkdirSync(FRAMES, { recursive: true });
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    colorScheme: "light",
    reducedMotion: "no-preference",
  });
  const page = await ctx.newPage();
  // Time is ours: requestAnimationFrame and performance.now run on a paused fake clock (Motion's
  // JS animations: the dots, the numbers), and every Web Animation (CSS and Motion's WAAPI
  // opacity fades) is paused and set to the same elapsed time at each frame.
  await page.clock.install();
  await page.goto(URL("energy"));
  await page.locator('[role="radiogroup"]').waitFor();
  await page.evaluate(() => document.fonts.ready);
  const t0 = await page.evaluate(() => Date.now());
  await page.clock.pauseAt(t0 + 500);
  await page.clock.runFor(1000);
  await page.locator('[data-option="residual"]').focus();
  await page.clock.runFor(1500);
  await page.waitForTimeout(500);

  const q = await page.locator('[data-testid="q-energy"]').boundingBox();
  const panel = await page.getByRole("complementary", { name: "Pipeline" }).boundingBox();
  const strip = await page.locator('[role="radiogroup"]').boundingBox();
  const stage = await page.locator("[data-stage]").boundingBox();
  if (!q || !panel || !strip || !stage) throw new Error("layout not found");
  const clip = {
    x: q.x,
    y: strip.y - 8,
    width: panel.x + panel.width - q.x,
    height: stage.y + stage.height - strip.y + 16,
  };

  // Only the animations this press starts are stepped; finished ones keep their end state.
  await page.evaluate(() => {
    (window as unknown as { __before: Set<Animation> }).__before = new Set(
      document.getAnimations(),
    );
  });
  await page.keyboard.press("ArrowRight");
  const times = [0, 60, 120, 180, 260, 400];
  const shots: { at: number; b64: string }[] = [];
  let now = 0;
  for (const at of times) {
    if (at > now) await page.clock.runFor(at - now);
    now = at;
    await page.evaluate((ms) => {
      const before = (window as unknown as { __before: Set<Animation> }).__before;
      for (const a of document.getAnimations()) {
        if (before.has(a)) continue;
        a.pause();
        a.currentTime = ms;
      }
    }, at);
    const file = resolve(FRAMES, `frame-${String(at).padStart(3, "0")}ms.png`);
    const buf = await page.screenshot({ path: file, clip });
    expect(statSync(file).size).toBeLessThan(MAX_BYTES);
    shots.push({ at, b64: buf.toString("base64") });
  }

  // Compose the strip: six frames, two across, at a bit under half size.
  const sheet = await ctx.newPage();
  const w = Math.round(clip.width * 0.46);
  await sheet.setViewportSize({ width: w * 2 + 36, height: 900 });
  await sheet.setContent(`<!doctype html><html><body style="margin:0;background:#f7f8f6;
    font:600 12px Inter,system-ui,sans-serif;color:#1c2b29">
    <div style="display:grid;grid-template-columns:repeat(2,${w}px);gap:12px;padding:12px">
    ${shots
      .map(
        (sh) => `<figure style="margin:0"><figcaption style="margin:0 0 4px">${sh.at} ms after
        → (residual → density + energy)</figcaption><img style="width:${w}px;display:block;
        border:1px solid #dce3e0;border-radius:6px" src="data:image/png;base64,${sh.b64}"></figure>`,
      )
      .join("")}</div></body></html>`);
  const stripFile = resolve(OUT, "frame-strip.png");
  await sheet.locator("div").first().screenshot({ path: stripFile });
  expect(statSync(stripFile).size).toBeLessThan(MAX_BYTES);
  await ctx.close();
});
