/**
 * Review captures for the consequence-stage prototype (not part of the app bundle).
 *
 *   npx vite --port 5417 --strictPort --host 127.0.0.1      # in turbotab/frontend
 *   node src/explore/stage/capture.mjs [base-url] [--only name]
 *
 * Writes docs/turbotab-next/m1/explore/stage/: screenshots at 1440x900 in light and dark,
 * the residual → density frame strip, and words.json (visible words per scenario, counted
 * in the page: every text node whose box is inside the viewport and not clipped away).
 */
import { chromium } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, "../../../../../docs/turbotab-next/m1/explore/stage");
const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5417";
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
mkdirSync(resolve(OUT, "frames"), { recursive: true });

/** Visible words: text nodes inside the viewport, not hidden, not clipped by a scroller. */
function countVisibleWords() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const seen = [];
  let total = 0;
  const voice = { app: 0, action: 0, data: 0 };
  const visible = (el) => {
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden" || Number(cs.opacity) < 0.05) return false;
    }
    return true;
  };
  const clippedOut = (el, rect) => {
    for (let e = el.parentElement; e && e !== document.body; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (/(auto|scroll|hidden|clip)/.test(cs.overflow + cs.overflowX + cs.overflowY)) {
        const r = e.getBoundingClientRect();
        if (rect.right <= r.left || rect.left >= r.right || rect.bottom <= r.top || rect.top >= r.bottom) return true;
      }
    }
    return false;
  };
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n.textContent ?? "";
    if (!text.trim()) continue;
    const el = n.parentElement;
    if (!el || !visible(el)) continue;
    const range = document.createRange();
    range.selectNodeContents(n);
    const rect = range.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) continue;
    if (rect.right <= 0 || rect.left >= vw || rect.bottom <= 0 || rect.top >= vh) continue;
    if (clippedOut(el, rect)) continue;
    const words = text.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w));
    total += words.length;
    const family = getComputedStyle(el).fontFamily;
    if (/Mono|monospace/i.test(family)) voice.data += words.length;
    else if (/Charter|Georgia|serif/i.test(family) && !/sans-serif/i.test(family)) voice.app += words.length;
    else voice.action += words.length;
    seen.push(text.trim());
  }
  return { total, voice, sample: seen };
}

async function open(page, scenario) {
  await page.goto(`${BASE}/lab/explore/stage?s=${scenario}`);
  await page.waitForSelector("[data-testid=stage]");
  await page.waitForTimeout(250);
}

async function previewOption(page, key) {
  await page.locator(`[role=option][data-key="${key}"]`).focus();
  await page.waitForTimeout(700);
}

async function shoot(page, name) {
  await page.mouse.move(2, 890);
  const path = resolve(OUT, `${name}.png`);
  await page.screenshot({ path, type: "png" });
  return path;
}

const shots = [
  { name: "s1-idle", scenario: "energy", run: async () => {} },
  { name: "s1-residual", scenario: "energy", run: (p) => previewOption(p, "residual") },
  { name: "s1-density", scenario: "energy", run: (p) => previewOption(p, "density") },
  { name: "s1-partition", scenario: "energy", run: (p) => previewOption(p, "partition") },
  { name: "s2-sex-specific", scenario: "exclusions", run: (p) => previewOption(p, "sex_specific") },
  {
    name: "s3-findings",
    scenario: "findings",
    run: async (p) => {
      await p.locator('[data-card="energy"]').focus();
      await p.waitForTimeout(700);
    },
  },
  {
    name: "s3-paged",
    scenario: "findings",
    run: async (p) => {
      await p.locator('[data-card="flags"]').focus();
      await p.keyboard.press("ArrowRight");
      await p.keyboard.press("ArrowRight");
      await p.waitForTimeout(700);
    },
  },
  { name: "s3-idle", scenario: "findings", run: async () => {} },
  { name: "s4-wide", scenario: "transform", run: (p) => previewOption(p, "log") },
];

const browser = await chromium.launch();
const words = {};
try {
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
      colorScheme: theme,
    });
    const page = await ctx.newPage();
    for (const shot of shots) {
      if (only && !shot.name.startsWith(only)) continue;
      await open(page, shot.scenario);
      await shot.run(page);
      await shoot(page, `${shot.name}-${theme}`);
      if (theme === "light") {
        const w = await page.evaluate(countVisibleWords);
        words[shot.name] = { total: w.total, ...w.voice };
      }
    }
    await ctx.close();
  }

  // The frame strip: residual → density on one key press. The page plays its stage
  // transitions 10x slower (?slow=10, a review flag), so a frame taken at 600 ms real time
  // shows the morph at 60 ms. Each frame's true offset is measured and written down.
  if (!only || only === "frames") {
    const SLOW = 10;
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
    const page = await ctx.newPage();
    await page.goto(`${BASE}/lab/explore/stage?s=energy&slow=${SLOW}`);
    await page.waitForSelector("[data-testid=stage]");
    await page.mouse.move(2, 890);
    await page.locator('[role=option][data-key="residual"]').focus();
    await page.waitForSelector('[data-card="results"]', { state: "detached" });
    await page.waitForTimeout(400 * SLOW);
    const stage = page.locator("[data-testid=stage]");
    const box = await stage.boundingBox();
    const taken = [];
    await page.keyboard.press("ArrowDown");
    const t0 = Date.now();
    for (const at of [0, 60, 120, 180, 260, 400]) {
      const wait = at * SLOW - (Date.now() - t0);
      if (wait > 0) await page.waitForTimeout(wait);
      const real = Date.now() - t0;
      const file = `frames/residual-to-density-${String(at).padStart(3, "0")}ms.png`;
      await page.screenshot({ path: resolve(OUT, file), clip: box });
      const done = Date.now() - t0;
      taken.push({ at_ms: at, file, captured_ms: +((real + done) / 2 / SLOW).toFixed(1) });
    }
    writeFileSync(
      resolve(OUT, "frames/frames.json"),
      JSON.stringify({ slow: SLOW, frames: taken }, null, 2) + "\n",
    );

    // The strip: the six frames side by side, labeled with their time after the press.
    const cells = taken
      .map((f) => {
        const b64 = readFileSync(resolve(OUT, f.file)).toString("base64");
        return `<figure><img src="data:image/png;base64,${b64}"><figcaption>${f.at_ms} ms</figcaption></figure>`;
      })
      .join("");
    const strip = await ctx.newPage();
    await strip.setViewportSize({ width: 6 * 330 + 7 * 12, height: 400 });
    await strip.setContent(`<!doctype html><style>
      body{margin:0;padding:12px;display:flex;gap:12px;background:#f7f8f6;font:600 13px ui-monospace,monospace;color:#1c2b29}
      figure{margin:0;width:330px} img{width:330px;display:block;border:1px solid #dce3e0;border-radius:8px}
      figcaption{margin-top:6px;text-align:center}</style>${cells}`);
    const h = await strip.evaluate(() => document.body.scrollHeight);
    await strip.setViewportSize({ width: 6 * 330 + 7 * 12, height: h });
    await strip.screenshot({ path: resolve(OUT, "frames/strip.png") });

    // Reduced motion: the same press lands at once.
    const rctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: "light",
      reducedMotion: "reduce",
    });
    const rp = await rctx.newPage();
    await rp.goto(`${BASE}/lab/explore/stage?s=energy`);
    await rp.waitForSelector("[data-testid=stage]");
    await rp.locator('[role=option][data-key="residual"]').focus();
    await rp.waitForTimeout(600);
    const pathNow = () =>
      rp.evaluate(() => document.querySelectorAll('[data-view="relationship"] path')[1]?.getAttribute("d"));
    await rp.keyboard.press("ArrowDown");
    const at0 = await rp.evaluate(
      () =>
        new Promise((r) =>
          requestAnimationFrame(() =>
            r(document.querySelectorAll('[data-view="relationship"] path')[1]?.getAttribute("d")),
          ),
        ),
    );
    await rp.waitForTimeout(800);
    const settled = await pathNow();
    await rp.screenshot({ path: resolve(OUT, "frames/reduced-motion-one-frame.png") });
    writeFileSync(
      resolve(OUT, "frames/reduced-motion.json"),
      JSON.stringify({ instant: at0 === settled }, null, 2) + "\n",
    );
    console.log("reduced motion instant:", at0 === settled);
    await rctx.close();
    await ctx.close();
  }
} finally {
  await browser.close();
}

if (!only) {
  writeFileSync(
    resolve(OUT, "words.json"),
    JSON.stringify(
      {
        viewport: "1440x900",
        method:
          "Playwright, in the page: every text node whose box is inside the viewport, visible, and not clipped by a scrolling ancestor; a word is a whitespace-separated token containing a letter or digit. Split by voice from the computed font: app = serif (the app speaking), action = sans (controls, labels), data = mono (column names, values, ticks).",
        counts: words,
      },
      null,
      2,
    ) + "\n",
  );
}
console.log(JSON.stringify(words, null, 2));
