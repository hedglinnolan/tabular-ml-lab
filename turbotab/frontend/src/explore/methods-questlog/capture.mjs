/**
 * Review captures for the methods-questlog design prototype (not part of the app bundle).
 *
 *   npx vite --port 5461 --strictPort --host 127.0.0.1      # in turbotab/frontend
 *   node src/explore/methods-questlog/capture.mjs [base-url] [--only 05] [--out dir]
 *
 * Writes NN-moment-theme.png at 1440x900 (M1 and M4 also at 1024x768) in light and dark.
 */
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5461";
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
const themeOnly = args.includes("--theme") ? args[args.indexOf("--theme") + 1] : null;
const OUT = args.includes("--out") ? args[args.indexOf("--out") + 1] : "/private/tmp/turbotab-fix/proto-questlog";
mkdirSync(OUT, { recursive: true });

const scrollTo = (sel) => async (page) => {
  await page.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: "start" }), sel);
  await page.waitForTimeout(300);
};

const shots = [
  { name: "01-draft", m: "1", narrow: true },
  { name: "02-readings", m: "2" },
  { name: "03-estimand", m: "3" },
  { name: "04-adjustment", m: "4", narrow: true },
  { name: "05-phrase-edit", m: "5" },
  { name: "05b-storyboard-step", m: "5", q: "&rest=1", themes: ["light"] },
  { name: "06-locked", m: "6" },
  { name: "06b-appendix", m: "6", q: "&appendix", themes: ["light"] },
  { name: "07-prediction", m: "7" },
  { name: "08a-first-encounter", m: "8a", run: scrollTo("[data-testid=teach-first]") },
  { name: "08b-later-encounter", m: "8b" },
  { name: "09-mastery-unlock", m: "9" },
];

const browser = await chromium.launch();
for (const shot of shots) {
  if (only && !shot.name.startsWith(only)) continue;
  const sizes = [{ w: 1440, h: 900, tag: "" }];
  if (shot.narrow) sizes.push({ w: 1024, h: 768, tag: "-1024" });
  for (const theme of shot.themes ?? ["light", "dark"]) {
    if (themeOnly && theme !== themeOnly) continue;
    for (const size of sizes) {
      const ctx = await browser.newContext({
        viewport: { width: size.w, height: size.h },
        colorScheme: theme,
        deviceScaleFactor: 1,
      });
      const page = await ctx.newPage();
      await page.goto(`${BASE}/lab/methods-questlog?m=${shot.m}&shot${shot.q ?? ""}`);
      await page.waitForSelector("[data-testid=center]");
      await page.waitForTimeout(1600);
      if (shot.run) await shot.run(page);
      const file = resolve(OUT, `${shot.name}${size.tag}-${theme}.png`);
      await page.screenshot({ path: file });
      console.log(file);
      await ctx.close();
    }
  }
}
await browser.close();
