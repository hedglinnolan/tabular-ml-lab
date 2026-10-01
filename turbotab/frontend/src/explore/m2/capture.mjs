/**
 * Review captures for the M2 design prototype (not part of the app bundle).
 *
 *   npx vite --port 5433 --strictPort --host 127.0.0.1      # in turbotab/frontend
 *   node src/explore/m2/capture.mjs [base-url] [--only name]
 *
 * Writes docs/turbotab-next/m2/explore/: screenshots at 1440x900 in light and dark, frame strips of
 * the reshape flip and the orientation flip (and the stacked gather at 20,000 columns) at 0, 150,
 * 300, 450, 600 and 900 ms, and words.json (visible words per screen, counted in the page).
 *
 * Frames step the production player's clock by hand (`setManual` / `advance`, its review hook), so
 * each frame's player position is exact; value rolls and cell tints, which run on wall time
 * (≤ 300 ms), are let finish before the frame is taken.
 */
import { chromium } from "@playwright/test";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, "../../../../../docs/turbotab-next/m2/explore");
const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5433";
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
mkdirSync(resolve(OUT, "screens"), { recursive: true });
mkdirSync(resolve(OUT, "frames"), { recursive: true });

/** Visible words: text nodes inside the viewport, not hidden, not clipped by a scroller. */
function countVisibleWords() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
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
  }
  return { total, voice };
}

async function open(page, query) {
  await page.goto(`${BASE}/lab/m2?${query}`);
  await page.waitForSelector("[data-testid=stage]");
  await page.waitForTimeout(700);
}

/** Pause the player on a storyboard step (as a step dot does). */
async function seek(page, step) {
  await page.evaluate((st) => window.__m2.store.dispatch({ type: "seek", step: st }), step);
  await page.waitForTimeout(1200);
}

const shots = [
  { name: "reshape-mean", query: "s=reshape&o=mean&side=with" },
  { name: "reshape-first", query: "s=reshape&o=first&side=with" },
  { name: "reshape-gathered", query: "s=reshape&o=mean", run: (p) => seek(p, 1) },
  { name: "wide-20000-columns", query: "s=wide&o=mean&side=with" },
  { name: "orientation-turned", query: "s=orientation&o=features&side=with" },
  { name: "seal-grouped-recorded", query: "s=seal&v=grouped&f=0.2&side=with&rec=1" },
  { name: "seal-chronological-ordered", query: "s=seal&v=chronological&f=0.2", run: (p) => seek(p, 2) },
  { name: "seal-chronological", query: "s=seal&v=chronological&f=0.2&side=with" },
  { name: "seal-abandoned", query: "s=seal&v=abandoned&f=0.2&side=with" },
  { name: "seal-undetermined", query: "s=seal&v=undetermined&f=0.2&side=with" },
  { name: "results-sealed", query: "s=results&phase=sealed" },
  {
    name: "results-opened",
    query: "s=results&phase=sealed",
    run: async (p) => {
      await p.click("[data-testid=open-seal]");
      await p.waitForTimeout(900);
    },
  },
  { name: "results-post-seal", query: "s=results&phase=post" },
];

async function strip(ctx, files, labels, out) {
  const cells = files
    .map((f, i) => {
      const b64 = readFileSync(resolve(OUT, f)).toString("base64");
      return `<figure><img src="data:image/png;base64,${b64}"><figcaption>${labels[i]}</figcaption></figure>`;
    })
    .join("");
  const page = await ctx.newPage();
  await page.setViewportSize({ width: 6 * 330 + 7 * 12, height: 400 });
  await page.setContent(`<!doctype html><style>
    body{margin:0;padding:12px;display:flex;gap:12px;background:#f7f8f6;font:600 13px ui-monospace,monospace;color:#1c2b29}
    figure{margin:0;width:330px} img{width:330px;display:block;border:1px solid #dce3e0;border-radius:8px}
    figcaption{margin-top:6px;text-align:center}</style>${cells}`);
  const h = await page.evaluate(() => document.body.scrollHeight);
  await page.setViewportSize({ width: 6 * 330 + 7 * 12, height: h });
  await page.screenshot({ path: resolve(OUT, out) });
  await page.close();
}

const browser = await chromium.launch();
const words = {};
const sizes = {};
try {
  for (const theme of ["light", "dark"]) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: theme });
    const page = await ctx.newPage();
    for (const shot of shots) {
      if (only && !shot.name.startsWith(only)) continue;
      await open(page, shot.query);
      if (shot.run) await shot.run(page);
      await page.mouse.move(2, 890);
      const file = `screens/${shot.name}-${theme}.png`;
      await page.screenshot({ path: resolve(OUT, file) });
      sizes[file] = Math.round(statSync(resolve(OUT, file)).size / 1024);
      if (theme === "light") words[shot.name] = await page.evaluate(countVisibleWords);
    }
    // The coach's notes, close up (the energy view with "with this choice").
    if (!only || only === "coach") {
      await open(page, "s=reshape&o=mean&side=with");
      const box = await page.locator('[data-card="spread"]').boundingBox();
      const file = `screens/coach-notes-${theme}.png`;
      await page.screenshot({ path: resolve(OUT, file), clip: box });
      sizes[file] = Math.round(statSync(resolve(OUT, file)).size / 1024);
    }
    await ctx.close();
  }

  // Frame strips: one flip, the clock stepped by hand.
  const FRAMES = [0, 150, 300, 450, 600, 900];
  const flips = [
    { name: "reshape-mean", query: "s=reshape&o=mean" },
    { name: "orientation", query: "s=orientation&o=features" },
    { name: "wide-stacked-gather", query: "s=wide&o=mean" },
  ];
  const record = {};
  if (!only || only === "frames") {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
    const page = await ctx.newPage();
    for (const flip of flips) {
      await open(page, flip.query);
      await page.mouse.move(2, 890);
      await page.evaluate(() => {
        const s = window.__m2.store;
        s.setManual(true);
        s.dispatch({ type: "flip" });
      });
      const files = [];
      const steps = [];
      let t = 0;
      for (const at of FRAMES) {
        await page.evaluate((ms) => window.__m2.store.advance(ms), at - t);
        t = at;
        await page.waitForTimeout(400);
        const box = await page.locator("[data-testid=stage]").boundingBox();
        const file = `frames/${flip.name}-${String(at).padStart(3, "0")}ms.png`;
        await page.screenshot({ path: resolve(OUT, file), clip: box });
        files.push(file);
        steps.push(
          await page.evaluate(() => {
            const s = window.__m2.store.get();
            const label = document.querySelector("[data-testid=step-label]")?.textContent?.trim() ?? "";
            return { pos: +s.pos.toFixed(3), label };
          }),
        );
      }
      record[flip.name] = FRAMES.map((at, i) => ({ at_ms: at, file: files[i], ...steps[i] }));
      await strip(ctx, files, FRAMES.map((f, i) => `${f} ms · ${steps[i].label}`), `frames/${flip.name}-strip.png`);
    }
    writeFileSync(
      resolve(OUT, "frames/frames.json"),
      JSON.stringify(
        {
          clock: "manual: the production player's setManual/advance; pos is exact at each offset",
          total_ms: 900,
          flips: record,
        },
        null,
        2,
      ) + "\n",
    );

    // Reduced motion: the same flip lands in one frame.
    const rctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    const rp = await rctx.newPage();
    await open(rp, "s=reshape&o=mean");
    await rp.keyboard.press("Space");
    const at0 = await rp.evaluate(
      () => new Promise((r) => requestAnimationFrame(() => r(window.__m2.store.get().pos))),
    );
    writeFileSync(resolve(OUT, "frames/reduced-motion.json"), JSON.stringify({ pos_after_one_frame: at0, instant: at0 === 3 }, null, 2) + "\n");
    console.log("reduced motion instant:", at0 === 3);
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
          "Playwright, in the page: every text node whose box is inside the viewport, visible, and not clipped by a scrolling ancestor; a word is a whitespace-separated token containing a letter or digit. Split by voice from the computed font: app = serif, action = sans, data = mono.",
        counts: words,
        kb: sizes,
      },
      null,
      2,
    ) + "\n",
  );
}
console.log(JSON.stringify({ words, sizes }, null, 2));
