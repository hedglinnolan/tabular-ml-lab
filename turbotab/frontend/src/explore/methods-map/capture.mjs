/**
 * Review captures for design prototype C, the methods map (not part of the app bundle).
 *
 *   npx vite --port 5473 --strictPort --host 127.0.0.1          # in turbotab/frontend
 *   node src/explore/methods-map/capture.mjs [base-url] [out-dir]
 *
 * Drives the prototype as a person would, on the three prototypes' shared scenario
 * (../methods-shared/SCENARIO.md) — the newcomer's walk through the asked nodes, the expert's
 * clicks, the lock, the results, an edit after the lock, the prediction version — and
 * takes a 1440×900 screenshot at each key moment, in light and in dark. It fails on any page error,
 * so it doubles as the prototype's smoke test. Writes <out-dir>/<nn>-<name>-<theme>.png and
 * README.md (the captions).
 */
import { chromium } from "@playwright/test";
import { mkdirSync, statSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const args = process.argv.slice(2);
const BASE = args.find((a) => a.startsWith("http")) ?? "http://127.0.0.1:5473";
const OUT = resolve(args.find((a) => !a.startsWith("http")) ?? "/private/tmp/turbotab-fix/proto-map");
mkdirSync(OUT, { recursive: true });

const MOMENTS = [];
const tid = (id) => `[data-testid="${id}"]`;

async function journey(theme) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: theme });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const shot = async (n, name, caption) => {
    await page.waitForTimeout(450);
    const file = `${String(n).padStart(2, "0")}-${name}-${theme}.png`;
    await page.screenshot({ path: resolve(OUT, file) });
    if (theme === "light") MOMENTS.push({ n, name, caption });
    const kb = Math.round(statSync(resolve(OUT, file)).size / 1024);
    console.log(`${file} ${kb} KB`);
  };
  const click = async (sel) => {
    await page.locator(sel).first().click();
    await page.waitForTimeout(160);
  };

  await page.goto(`${BASE}/lab/methods-map`);
  await page.evaluate(() => localStorage.clear());
  await page.goto(`${BASE}/lab/methods-map`);
  await page.waitForSelector(tid("map"));
  await shot(1, "first-draft", "The first draft. The map reads left to right like a figure: the table, each decision where it acts, the estimate. Regions are STROBE-nut's sections. Solid dots are written in (each phrase changeable); hollow rings are asked of you, with their guesses; silent decisions are absent, counted per region (\"7 silent · export only\"). The methods text on the right is the same decisions as the engine's sentences, with each asked slot a gap that names itself.");

  // the newcomer's walk: the first objective, with the rest of the map dimmed
  await click(tid("intro-start"));
  await shot(2, "walk-readings", "\"Start with Readings\": the newcomer's walk. Only the current objective is lit on the map and in the record. The card leads with the question, then teaches the two concepts it is the first to use (exposure, covariate) in full, then the readings, ordered by consequence: kcal's day count first (it feeds the screens), each with the engine's guess and evidence.");

  const keys = await page.$$eval('[data-testid^="confirm-role:"], [data-testid^="confirm-code_or_count:"]', (els) =>
    els.map((e) => e.getAttribute("data-testid")),
  );
  for (const k of keys.slice(0, 3)) await click(tid(k));
  await page.locator(tid("block-offer")).scrollIntoViewIfNeeded();
  await shot(3, "mastery-block-unlocked", "Mastery unlock: after three readings confirmed one at a time, a block confirm appears. It lists exactly the readings it settles, each with the value it shows, and settles nothing else (BLUEPRINT §14.2). Each single confirmation is already in the record as the engine's own sentence.");
  await click(tid("confirm-unit"));
  await click(tid("confirm-block"));
  await click(tid("confirm-codes"));
  await page.locator(tid("receipt")).scrollIntoViewIfNeeded();
  await shot(4, "readings-receipt", "Recorded: the receipt at the control quotes the engine's sentence for the last block confirmed (the fit's two code-or-amount readings), which names what it settled. The Readings node turns solid with a check, the identifier and flags lane ends at \"not predictors\", the record gains the same sentence, and \"Next asked: Exclusions →\" offers the next objective without moving the page.");

  await click(tid("receipt-next"));
  await page.hover(tid("opt-willett_2013_by_sex"));
  await shot(5, "exclusions-hover", "Hovering an option plays it. Willett 2013's screen: the ribbon on the map narrows at the exclusions (−1,614, drawn as a preview), and the canvas's flip plays the engine's row flow and the kcal distribution with its cut-offs. Only the hovered option shows its two labels: customary in the field, and sound for inference (here \"conditional\"); the tension line says why every-row is reported beside it.");
  await click(tid("opt-none"));
  await click(tid("beside-willett_2013_by_sex"));
  await click(tid("beside-nhs_hpfs_by_sex"));

  await click(tid("next"));
  await click(tid("opt-complete_case"));

  await click(tid("next"));
  await shot(6, "exposure-first-encounter", "The exposure and its estimand. The guess (sugar) comes with its evidence from the readings. Two concepts are met here first and are taught in full (estimand, substitution); the engine's own question about substitution or addition is asked before recording, as the engine asks it. The canvas plays the engine's lineage on the scenario's plan: sugar's column emphasized on its way into the model; the map forks it out of the nutrients' bundle.");
  await click(tid("contrast-substitution"));
  await click(tid("record-exposure"));

  await click(tid("receipt-next"));
  for (const g of ["demographic", "dietary", "body"]) await click(tid(`record-adj-${g}`));
  await click(tid("adj-truth"));
  await page.locator(tid("record-adj-unguessed")).scrollIntoViewIfNeeded();
  await shot(7, "adjustment-set", "The adjustment set: 19 covariates in four groups, three of them one tap (the pack's guess with its reason). The seven with no guess are asked per column by the disjunctive cause criterion; the role each derives updates as you answer, and the canvas plays the engine's lineage of each group's answers. On the map the lanes re-route as a preview: cycle_begin_year joins the model, six mediators leave, body size goes beside to Model 3, and Model 2 counts 11 columns.");
  await click(tid("keep-mediator"));
  await shot(8, "leash-refusal", "The leash: keeping hdl, a mediator by these answers, in a total-effect set is the engine's refusal (VanderWeele 2019), shown on the canvas with its two exits.");
  await click(tid("record-adj-unguessed"));

  await click(tid("node-energy"));
  await page.hover(tid("opt-residual"));
  await page.waitForTimeout(1300);
  await shot(9, "energy-phrase-plays", "Editing a stated phrase: energy adjustment is written in (the standard model, ranked first). Hovering \"Residual, energy kept\" plays its storyboard on the canvas (fit on kcal → keep the residual → r 0.88 → 0.00) and marks the nutrients' lanes \"adj\" on the map. Substitution, taught at the exposure, is now a condensed dotted term; the residual method and nutrient density are new here and taught in full.");

  await click(tid("node-model1"));
  await click(tid("model1-guess"));
  await click(tid("next"));
  await shot(10, "lock-ready", "Every asked slot is answered: the lock gate. Missing data was asked after the exclusions and answered with complete cases; the engine's sentence says no row is missing a predictor, so all 21,849 rows remain.");
  await click(tid("lock"));
  await click(tid("node-estimate"));
  await shot(11, "table-2", "The plan locked; Table 2, the exposure only: unadjusted, Model 1, Model 2 (primary) and Model 3 (further adjusted, not a total effect), the engine's numbers on all 21,849 rows, with its caption and interval method. The lock's sentence carries its SHA-256, and on the map the lanes end in the estimate, β with its interval.");
  await click(tid("appendix-toggle"));
  await page.locator(tid("appendix")).scrollIntoViewIfNeeded();
  await shot(12, "appendix", "The appendix, one press away: every other coefficient, per model, under its title \"adjustment terms, not effect estimates\" (Westreich & Greenland 2013), each marked with the engine's reason it is not an effect. Below Table 2, the influence check and the unmeasured-confounding sensitivity (E-value, robustness value).");
  await click(tid("tab-matter"));
  await shot(13, "which-decisions-mattered", "\"Which of my decisions mattered?\" The estimate of sugar under each declared alternative (the adjustment sequence and the two screens reported beside), with its 95% interval, sorted, and an indicator of which decision each varies. Sensitivity, never a way to choose.");

  await click(tid("node-energy"));
  await click(tid("opt-residual_energy_dropped"));
  await click(tid("node-estimate"));
  await page.locator('[data-after]').first().scrollIntoViewIfNeeded();
  await shot(14, "edit-after-lock", "An edit after the lock: the energy method changed to the residual with energy left out. The record keeps the plan as declared and adds the change, tagged \"after the estimates\" in the engine's own words; on the map kcal leaves the model and Model 2 counts 10 columns; Table 2 is the engine's fit for that plan.");

  await click(tid("purpose-prediction"));
  await click(tid("node-p_seal"));
  await page.waitForTimeout(1400);
  await shot(15, "prediction-tripod", "The prediction version: the same map under TRIPOD+AI's sections. Under prediction there is no exposure or adjustment set (silent), missing data is asked (a fill learned in each training fold), and the seal is asked with its guess (hold out 20%); the canvas plays the engine's seal, the held-out rows drawn cell by cell.");

  await browser.close();
  if (errors.length) {
    console.error(`page errors (${theme}):\n${errors.join("\n")}`);
    process.exitCode = 1;
  }
}

await journey("light");
await journey("dark");

const lines = [
  "# Prototype C — the methods map: key moments",
  "",
  "Route: `/lab/methods-map` (dev build). Every sentence, guess, evidence line and number is the real",
  "engine's, captured from the server on the NHANES export (`capture.py`). Each moment is shown in light",
  "and dark: `<nn>-<name>-light.png`, `<nn>-<name>-dark.png`. Captured by `capture.mjs`.",
  "",
  ...MOMENTS.flatMap((m) => [`## ${String(m.n).padStart(2, "0")} · ${m.name}`, "", m.caption, ""]),
];
writeFileSync(resolve(OUT, "README.md"), lines.join("\n"));
console.log(`README.md written to ${OUT}`);
