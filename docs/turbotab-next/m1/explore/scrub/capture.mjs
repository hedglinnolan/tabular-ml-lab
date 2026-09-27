/**
 * Captures the review deliverables for the "before and after, scrubbed" prototype
 * (/lab/explore/scrub): screenshots at 1440×900 in light and dark, the frame strip of S1 moving
 * from residual to density, and the visible word count per scenario.
 *
 *   cd turbotab/frontend && npx vite --port 5402 --strictPort --host 127.0.0.1 &
 *   node ../../docs/turbotab-next/m1/explore/scrub/capture.mjs [http://127.0.0.1:5402]
 *
 * The frame strip runs on Playwright's fake clock (page.clock): Motion reads performance.now and
 * requestAnimationFrame, so each frame is exactly N ms after the key press, not "about" N.
 */
import { createRequire } from "node:module";
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../../../../..");
const require = createRequire(resolve(REPO, "turbotab/frontend/package.json"));
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const FRAMES_ONLY = args.includes("--frames");
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5402";
const ROUTE = "/lab/explore/scrub";
const OUT = HERE;
const FRAMES = resolve(OUT, "frames");
mkdirSync(FRAMES, { recursive: true });
const MAX = 300 * 1024;

const SHOTS = [
  { name: "s1-idle", s: "energy", keys: [] },
  { name: "s1-residual", s: "energy", keys: ["ArrowDown"] },
  { name: "s1-density", s: "energy", keys: ["ArrowDown", "ArrowDown"] },
  { name: "s2-sex-specific", s: "exclusions", keys: ["ArrowDown", "ArrowDown", "ArrowDown"] },
  { name: "s3-findings", s: "findings", keys: [] },
  { name: "s3-findings-paged", s: "findings", keys: ["click:show", "click:›", "ArrowRight"], extra: true },
  { name: "s4-wide", s: "wide", keys: ["ArrowDown"] },
];

/** Words a reader can see: text nodes inside the viewport, not hidden, not transparent. */
function countWords() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const tally = { total: 0, prose: 0, data: 0 };
  const opacityOf = (el) => {
    let o = 1;
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden") return 0;
      o *= Number(cs.opacity);
    }
    return o;
  };
  let n;
  while ((n = walker.nextNode())) {
    const text = n.textContent ?? "";
    if (!text.trim()) continue;
    const el = n.parentElement;
    if (!el || opacityOf(el) < 0.05) continue;
    const range = document.createRange();
    range.selectNodeContents(n);
    const rects = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
    if (!rects.some((r) => r.right > 0 && r.bottom > 0 && r.left < vw && r.top < vh)) continue;
    // Clipped by an ancestor with overflow hidden and zero width (a collapsed affix)?
    const box = el.getBoundingClientRect();
    if (box.width === 0 && getComputedStyle(el).overflow === "hidden") continue;
    const words = text.split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
    const mono = /mono/i.test(getComputedStyle(el).fontFamily);
    tally.total += words;
    if (mono) tally.data += words;
    else tally.prose += words;
  }
  return tally;
}

async function open(page, s) {
  await page.goto(`${BASE}${ROUTE}?s=${s}`);
  await page.waitForSelector('[role="listbox"]');
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
}

async function press(page, keys) {
  if (!keys.length) return;
  await page.locator('[role="listbox"]').first().focus();
  for (const k of keys) {
    if (k.startsWith("click:")) await page.getByText(k.slice(6), { exact: true }).first().click();
    else await page.keyboard.press(k);
    await page.waitForTimeout(450);
  }
}

const browser = await chromium.launch();
const words = {};
const shots = [];
try {
  for (const scheme of FRAMES_ONLY ? [] : ["light", "dark"]) {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: scheme });
    for (const shot of SHOTS.filter((x) => scheme === "light" || !x.extra)) {
      await open(page, shot.s);
      await press(page, shot.keys);
      // Nothing keyboard-focused in the picture: blur so the focus ring does not stand in for state.
      await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
      await page.mouse.move(1430, 890);
      await page.waitForTimeout(250);
      const file = resolve(OUT, `${shot.name}-${scheme}.png`);
      await page.screenshot({ path: file });
      const size = statSync(file).size;
      shots.push({ file, size });
      if (size > MAX) console.warn(`over 300 KB: ${file} (${size})`);
      if (scheme === "light") words[shot.name] = await page.evaluate(countWords);
    }
    await page.close();
  }

  // ── the frame strip: residual -> density, on a fake clock ─────────────────
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
  await page.clock.install();
  await open(page, "energy");
  await page.locator('[role="listbox"]').first().focus();
  await page.keyboard.press("ArrowDown"); // residual
  await page.clock.runFor(1500);
  await page.mouse.move(1430, 890);
  const stage = page.locator('section[aria-label="Consequence preview"]');
  const box = await stage.boundingBox();
  await page.clock.pauseAt(Date.now() + 60_000);
  await page.keyboard.press("ArrowDown"); // density
  const frames = [];
  let at = 0;
  for (const t of [0, 60, 120, 180, 260, 400]) {
    if (t > at) await page.clock.runFor(t - at);
    at = t;
    const file = resolve(FRAMES, `s1-residual-to-density-${String(t).padStart(3, "0")}ms.png`);
    await page.screenshot({ path: file, clip: box });
    frames.push(file);
  }
  await page.close();

  // One strip image of the six frames, two rows of three, each labeled with its time.
  const strip = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const imgs = frames
    .map((f, i) => {
      const t = [0, 60, 120, 180, 260, 400][i];
      const b64 = (awaitRead(f)).toString("base64");
      return `<figure><img src="data:image/png;base64,${b64}"/><figcaption>${t} ms</figcaption></figure>`;
    })
    .join("");
  await strip.setContent(`<!doctype html><html><body style="margin:0;background:#f7f8f6;font:600 13px Inter,system-ui">
    <div id="s" style="display:grid;grid-template-columns:repeat(3,480px);gap:10px;padding:12px;width:fit-content">${imgs}</div>
    <style>figure{margin:0}img{width:480px;display:block;border:1px solid #dce3e0;border-radius:6px}figcaption{color:#5b6b68;padding:4px 2px 0;font-family:ui-monospace,monospace}</style>
  </body></html>`);
  const stripFile = resolve(FRAMES, "s1-residual-to-density-strip.png");
  await strip.locator("#s").screenshot({ path: stripFile });
  frames.push(stripFile);
  await strip.close();

  // ── reduced motion: the same key press lands in one frame ─────────────────
  const rm = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  await rm.clock.install();
  await open(rm, "energy");
  await rm.locator('[role="listbox"]').first().focus();
  await rm.clock.pauseAt(Date.now() + 60_000);
  await rm.keyboard.press("ArrowDown");
  await rm.clock.runFor(20);
  const tag = await rm.locator('[aria-live="polite"]').first().textContent();
  console.log(`reduced motion, 20 ms after the key press, the scrub tag reads: "${tag}"`);
  await rm.close();

  if (!FRAMES_ONLY) writeFileSync(resolve(OUT, "words.json"), JSON.stringify(words, null, 2) + "\n");
  console.log(JSON.stringify({ words, shots: shots.map((x) => [x.file.split("/").pop(), x.size]), frames }, null, 2));
} finally {
  await browser.close();
}

function awaitRead(f) {
  return require("node:fs").readFileSync(f);
}
