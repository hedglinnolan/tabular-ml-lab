/**
 * Review captures for the methods-questlog design prototype (not part of the app bundle).
 *
 *   npm run dev:mock -- --port 5461 --strictPort --host 127.0.0.1      # in turbotab/frontend
 *   node src/explore/methods-questlog/capture.mjs [base-url] [--only 05] [--theme dark] [--out dir]
 *
 * Opens each moment of the walk by its review preset (?m=<moment>; the walk itself needs none) and
 * writes NN-name-theme.png at 1440x900 (the first draft and the adjustment also at 1024x768), in
 * light and dark.
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

const hover = (sel) => async (page) => {
  await page.locator(sel).first().hover();
  await page.waitForTimeout(900);
};
const scrollTo = (sel) => async (page) => {
  await page.locator(sel).first().scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
};

const shots = [
  { name: "01-draft", m: "draft", narrow: true },
  { name: "02-unit", m: "roles" },
  { name: "03-eligibility", m: "exclusions", run: hover("[data-testid=primary-willett_2013_by_sex]") },
  { name: "04-missing", m: "missing" },
  { name: "05-split", m: "split" },
  { name: "06-readings", m: "readings" },
  { name: "07-mastery-unlock", m: "single_cycle_begin_year", run: scrollTo("[data-testid=unlock]") },
  { name: "08-estimand-first-encounter", m: "estimand", run: scrollTo("[data-testid=teach-first]") },
  { name: "09-adjustment", m: "adjustment", narrow: true },
  { name: "10-energy-phrase", m: "energy", run: hover("[data-testid=option-residual]") },
  { name: "11-model-sequence", m: "model_sequence" },
  { name: "12-models", m: "models" },
  { name: "13-lock", m: "ready" },
  { name: "14-table2", m: "locked" },
  { name: "15-mattered", m: "locked", q: "&mattered", run: scrollTo("[data-testid=mattered]") },
  { name: "16-document", m: "locked", q: "&doc" },
  { name: "17-prediction", m: "locked", q: "&variant=prediction" },
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
