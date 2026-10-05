/**
 * Review captures for the methods-document prototype (not part of the app bundle).
 *
 *   npx vite --port 5487 --strictPort --host 127.0.0.1        # in turbotab/frontend
 *   node src/explore/methods-document/capture/shots.mjs [base-url] [out-dir] [--only m5]
 *
 * Each moment at 1440x900 in light and dark; m1 and m4 also at 1024x768. Moments that play the
 * transform player are taken once its storyboard has run (and m5 also paused on a step).
 */
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5487";
const OUT = resolve(args.find((a) => a.startsWith("/")) ?? "/private/tmp/turbotab-fix/proto-document");
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
mkdirSync(OUT, { recursive: true });

const SHOTS = [
  { n: "01", id: "m1", name: "draft-after-purpose", sizes: ["1440", "1024"] },
  { n: "02", id: "m2", name: "data-readings", sizes: ["1440"] },
  { n: "03", id: "m3", name: "exposure-estimand", sizes: ["1440"] },
  { n: "04", id: "m4", name: "adjustment-set", sizes: ["1440", "1024"] },
  { n: "05", id: "m5", name: "stated-phrase-energy", sizes: ["1440"], wait: 4000 },
  { n: "05b", id: "m5", name: "stated-phrase-storyboard-step", sizes: ["1440"], step: 1 },
  { n: "06", id: "m6", name: "after-lock-results", sizes: ["1440"] },
  { n: "06b", id: "m6b", name: "after-lock-methods", sizes: ["1440"], wait: 1500 },
  { n: "07", id: "m7", name: "prediction-tripod", sizes: ["1440"], wait: 1500 },
  { n: "08a", id: "m8a", name: "concept-first-encounter", sizes: ["1440"] },
  { n: "08b", id: "m8b", name: "concept-later-condensed", sizes: ["1440"], wait: 1200 },
  { n: "09", id: "m9", name: "mastery-block-unlocked", sizes: ["1440"] },
];

const VIEWPORT = { 1440: { width: 1440, height: 900 }, 1024: { width: 1024, height: 768 } };

const browser = await chromium.launch();
const themes = args.includes("--light") ? ["light"] : args.includes("--dark") ? ["dark"] : ["light", "dark"];
for (const theme of themes) {
  for (const shot of SHOTS) {
    if (only && !only.split(",").some((o) => o === shot.id || o === shot.n)) continue;
    for (const size of args.includes("--1440") ? ["1440"] : shot.sizes) {
      const ctx = await browser.newContext({ viewport: VIEWPORT[size], colorScheme: theme, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
      await page.goto(`${BASE}/lab/methods-document?m=${shot.id}`);
      await page.waitForSelector("article");
      await page.waitForTimeout(shot.wait ?? 900);
      if (shot.step !== undefined) {
        // Pause on a storyboard step, as its dot does.
        await page.waitForTimeout(1200);
        await page.locator(`[data-testid=stage] button[aria-label^="Step ${shot.step + 1} of"]`).click();
        await page.waitForTimeout(2500);
      }
      const suffix = size === "1440" ? "" : `-${size}`;
      const file = resolve(OUT, `${shot.n}-${shot.name}${suffix}-${theme}.png`);
      await page.screenshot({ path: file });
      if (errors.length) console.log(`${shot.n} ${theme} ${size}: ${errors.slice(0, 3).join(" | ")}`);
      console.log(file);
      await ctx.close();
    }
  }
}
await browser.close();
