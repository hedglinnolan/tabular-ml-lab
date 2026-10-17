/**
 * W1 acceptance 1 (plans/WALKABLE_PLAN.md §5; plans/W1_PART2_PLAN.md §2 "W1-L"): NHANES glucose
 * under Predict, walked through the quest shell at /p/:pid against the real server, from the
 * upload to the bundle.
 *
 *   TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d) scripts/py -m turbotab.server --port 8862
 *   E2E_BASE_URL=http://127.0.0.1:8862 npx playwright test quest-nhanes-predict
 *
 * The journey: the decompressed export is uploaded through the Start screen's drop zone → all
 * seven stages are walked by the quest line, every open line answered through its card and each
 * stage's sweep confirmed with its one button → Fit, pressed at the flowchart's end → Results: the
 * shrinkage question, the triage of the noticings left open, the final model named from the
 * opening's refusal, the held-out rows opened → the explanations → the bundle downloaded from
 * Write-up.
 *
 * What it asserts is what the screen says against what the server serves, never a number of its
 * own: after every step each bar segment, and the hover panel of each stage, equal GET /quest; each
 * printed score (the cross-validated and the held-out RMSE, MAE and R² with their intervals, the
 * baseline and the verdict against it, calibration-in-the-large and the slope) equals the served
 * field it prints; the bundle's methods are the sections the manuscript rail serves. The
 * independent references for the scores themselves (numpy, scikit-learn, statsmodels) are
 * turbotab/core/tests/test_w1_press_fit_references.py's.
 *
 * The question asked is "How well do age, sex and body measures predict fasting glucose?": the
 * dietary columns are left out of the models through the roles card, because the substitution
 * question the table's nutrients raise is answered on the stage's matrix, which is drawn only
 * once the fit exists, while Fit is offered at Models' end (an open issue in the report). The
 * families are linear and ridge, to keep the fit light (ruling 11).
 */
import { expect, test } from "@playwright/test";
import { onMock } from "./backend";
import { nhanesExport } from "./nhanes";
import {
  answerFinding,
  askExplanations,
  calibrationDifferences,
  comparisonsCatchWhatDiffers,
  chooseFamilies,
  confirmAskCard,
  cvDifferences,
  describe,
  downloadBundle,
  expectBarEqualsQuest,
  getFit,
  getJson,
  getQuest,
  heldOutDifferences,
  lineId,
  linesOf,
  misprinted,
  openHeldOutRows,
  openStage,
  pickOption,
  previewSettled,
  pressFit,
  printedNumbers,
  printsServed,
  uploadThroughZone,
  walk,
  type Answers,
} from "./quest";

const REAL = !!process.env.E2E_BASE_URL;

/** The columns left out of the models: the diet (see above), the cycle, the labs drawn with
 *  glucose, and what the body measures repeat (weight and height make BMI; the diastolic pressure
 *  follows the systolic). */
const LEFT_OUT = [
  "cycle_begin_year", "bp_di", "weight", "height", "kcal", "protein", "sugar", "carb", "fat_total", "fat_sat",
  "fat_mon", "fat_poly", "hdl", "triglycerides", "meds_hbp", "meds_chol",
];
const FAMILIES = ["linear", "ridge"];

/** What the journey's measurements say about the seal, read by the known-defect test after it. */
const facts: { served?: { changed_after_seal: boolean; post_seal_decisions: string[]; at_opening_current: boolean | null } } = {};

// The known-defect test reads what the journey measured, so it runs after it, in one worker.
test.describe.configure({ mode: "serial" });
// A press that finds nothing to press fails in half a minute, not in the length of the test.
test.use({ actionTimeout: 30_000 });

const answers: Answers = {
  // The noticings in Your data that are not answered where they arise (the zeros of a SAS file, the
  // text coded as two levels, and the findings the column roles settle) are left open for the
  // triage at Results, where the engine recommends a disposition for each. The one about the
  // missing survey design is read and kept as it is, on its own card.
  deferred: (line) =>
    line.source === "finding" &&
    (line.key === "sas_zeros" || line.key === "binary_text__gender" || (line.key.startsWith("voice::") && line.key !== "voice::survey_design_absent")),
  unavailable: ["join_files", "import_codebook"],
  async answer(page, li, line) {
    if (line.source === "finding") return answerFinding(page, li, line);
    switch (lineId(line)) {
      case "question:lens":
        return pickOption(page, li, "option-clinical", { space: true });
      case "question:target": {
        const box = li.getByRole("combobox");
        await box.fill("glucose");
        await box.press("Enter");
        await previewSettled(page);
        return void (await li.getByTestId("record-target").click());
      }
      case "question:purpose":
        return pickOption(page, li, "option-prediction");
      case "declaration:set_intended_use":
        return pickOption(page, li, "option-use:risk_estimation");
      case "question:roles":
        for (const column of LEFT_OUT) {
          await li.getByTestId(`role-chip-${column}`).click();
          await li.getByTestId("role-excluded").click();
        }
        return void (await li.getByTestId("record-roles").click());
      case "question:exclusions":
        await confirmAskCard(page, li);
        return pickOption(page, li, "option-none");
      case "question:missing":
        await confirmAskCard(page, li);
        return pickOption(page, li, "option-complete_case");
      case "question:split":
        return pickOption(page, li, "option-0.2");
      case "declaration:set_selection":
        return pickOption(page, li, "option-method:none");
      case "question:models":
        return chooseFamilies(page, li, FAMILIES);
      default:
        throw new Error(`the journey has no answer for ${lineId(line)} (${line.name})`);
    }
  },
};

const allComplete = (log: Awaited<ReturnType<typeof getQuest>>) => log.stages.every((s) => !s.reached || s.progress?.complete);

test("NHANES glucose under Predict, through the quest shell, from the upload to the bundle", async ({ page }) => {
  // The walk (each answer recomputes what it feeds, and the bar is checked after each), the fit,
  // the explanations and the downloads, at most; CI's runners are slower than a dev machine.
  test.setTimeout(REAL ? 1_500_000 : 60_000);
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "no-preference" });
  await page.goto("/");
  test.skip(await onMock(page), "the quest journeys run against the real server");

  const problems: string[] = [];
  page.on("console", (m) => {
    // The browser logs every non-2xx response; a refusal (409) is an answer a card shows.
    if (m.type() === "error" && !m.text().includes("status of 409")) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.status() !== 409) problems.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });

  // The comparison's own rule, shown both ways before it is relied on: a printed number is the
  // served field to three significant digits (whole numbers from a thousand), and no other.
  expect(printsServed(1207, 1206.777)).toBe(true);
  expect(printsServed(0.104, 0.1043)).toBe(true);
  expect(printsServed(1208, 1206.777)).toBe(false);
  expect(printsServed(0.105, 0.1043)).toBe(false);
  expect(printedNumbers("0.104 (−0.0867 to 1,207)")).toEqual([0.104, -0.0867, 1207]);
  expect(misprinted("34.7 (31.6 to 37.9)", [34.73870240021893, 31.58, 37.9147], "rmse")).toBe("");
  expect(misprinted("34.7 (31.6 to 37.9)", [34.73870240021893, 31.58], "rmse")).not.toBe("");

  // ── upload through the drop zone ───────────────────────────────────────────────────────────
  const pid = await uploadThroughZone(page, nhanesExport());
  // The shell is the screen at /p/:pid: its quest line names the seven stages the server serves.
  const opening = await getQuest(page, pid);
  expect(opening.stages).toHaveLength(7);
  await expect(page.getByTestId("quest-line").locator("li")).toHaveText(opening.stages.map((s) => s.name));
  await expectBarEqualsQuest(page, pid, "on opening");
  await comparisonsCatchWhatDiffers(page, pid, null);

  // ── the first five stages: every open line through its card, each sweep with its one button ─
  const taken = await walk(page, pid, answers, (log) => log.stages.find((s) => s.key === "models")?.progress?.complete === true);
  for (const id of [
    "question:lens", "question:target", "question:purpose", "declaration:set_intended_use", "question:roles",
    "question:exclusions", "question:missing", "question:split", "declaration:set_selection", "question:models",
    "sweep:question", "sweep:whos_in", "sweep:models",
  ]) {
    expect(taken, `the walk took ${id}`).toContain(id);
  }

  // The lines no card records are listed as such, and stay out of the count.
  await openStage(page, "data");
  for (const key of answers.unavailable) await expect(page.getByTestId(`line-${key}-unavailable`)).toHaveText("Not available yet");

  // First look decides nothing; it lists each noticing with the stage that decides it.
  await openStage(page, "first_look");
  const noticed = linesOf(await getQuest(page, pid)).filter((x) => x.line.source === "finding");
  expect(noticed.length, "the table raises noticings").toBeGreaterThan(3);
  await expect(page.locator('[data-testid^="noticed-"][data-status]')).toHaveCount(noticed.length);
  for (const { line } of noticed) await expect(page.getByTestId(`noticed-${line.key}`)).toHaveAttribute("data-status", line.status);
  await expectBarEqualsQuest(page, pid, "at First look");

  // ── Fit, at the analysis flowchart's end ───────────────────────────────────────────────────
  const sawRunning = await pressFit(page, pid);
  test.info().annotations.push({ type: "fit", description: sawRunning ? "pressed while the fit job ran: its progress and Cancel were shown" : "the fit had computed before the press" });
  await expectBarEqualsQuest(page, pid, "at Results, after Fit");

  // ── Results: the triage, the final model, the opening ──────────────────────────────────────
  const { family, opened, triaged, refusals } = await openHeldOutRows(page, pid, FAMILIES, (step) => expectBarEqualsQuest(page, pid, step));
  test.info().annotations.push({ type: "triage", description: `${triaged} noticings confirmed as recommended; refused ${refusals.join(", ")} before opening; ${family.label} named the final model` });

  // ── the held-out rows: each printed number is its served field ──────────────────────────────
  // The served fit says whether the opening's scores are still the reported result; the page
  // says the same (its phase and title follow the served flag).
  const phase = opened.changed_after_seal ? "post_seal" : "opened";
  await expect(page.getByTestId("results-tapestry")).toHaveAttribute("data-seal-phase", phase, { timeout: 30_000 });
  facts.served = {
    changed_after_seal: opened.changed_after_seal ?? false,
    post_seal_decisions: opened.post_seal_decisions ?? [],
    at_opening_current: opened.at_opening ? opened.at_opening.current : null,
  };
  await expect.poll(() => heldOutDifferences(page, opened), { message: "each held-out score, the baseline and the verdict equal the served fields", timeout: 30_000 }).toEqual([]);
  await comparisonsCatchWhatDiffers(page, pid, opened);
  await expect.poll(() => calibrationDifferences(page, opened), { message: "calibration-in-the-large and the slope equal the served fields", timeout: 30_000 }).toEqual([]);
  await expect.poll(() => cvDifferences(page, opened), { message: "the cross-validated scores equal the served fields after the opening", timeout: 30_000 }).toEqual([]);
  await expect(page.getByTestId("results-held-out").locator(`tr[data-row="${family.family}"]`)).toHaveAttribute("data-primary", "true");
  for (const m of opened.models) expect(m.holdout, `${m.family}'s held-out scores are served once opened`).not.toBeNull();

  // ── the explanations: asked on their card, drawn from the served artifact ───────────────────
  await askExplanations(page, pid);
  await expectBarEqualsQuest(page, pid, "after the explanations");

  // ── every stage settled: what the triage answered completes the sweeps it reopened ──────────
  await walk(page, pid, { ...answers, deferred: () => false }, allComplete);
  await expectBarEqualsQuest(page, pid, "with every stage complete");
  const final = await getQuest(page, pid);
  expect(final.stages.every((s) => s.reached), "all seven stages are reached").toBe(true);

  // ── Write-up: the bundle, with the methods in TRIPOD+AI's sections ──────────────────────────
  await openStage(page, "writeup");
  const checklist = await getJson<{ counts: { answered: number; partly_answered: number; unanswered: number; items: number } }>(page, pid, "/checklist");
  const c = checklist.counts;
  await expect(page.getByTestId("checklist")).toContainText(`${c.answered} answered, ${c.partly_answered} partly answered, ${c.unanswered} unanswered, of ${c.items} items.`);
  const served = await getJson<{ guideline: string; sections: { key: string; title: string; item: string | null; entries: { text: string; record_id: string }[] }[] }>(page, pid, "/methods/sections");
  expect(served.guideline).toBe("TRIPOD+AI");
  const bundle = await downloadBundle(page);
  try {
    expect(bundle.name).toMatch(/\.zip$/);
    for (const file of ["methods.json", "methods.md", "results/performance.csv", "checklist/tripod_ai.json", "decisions.jsonl"]) {
      expect(bundle.files.some((f) => f.endsWith(`/${file}`)), `the bundle holds ${file}`).toBe(true);
    }
    const root = bundle.files.find((f) => f.endsWith("/methods.json"))!.replace(/methods\.json$/, "");
    const methods = JSON.parse(bundle.read(`${root}methods.json`)) as typeof served;
    expect(methods.guideline).toBe("TRIPOD+AI");
    const shape = (s: typeof served.sections) => s.map((x) => ({ key: x.key, title: x.title, item: x.item, entries: x.entries.map((e) => [e.record_id, e.text]) }));
    expect(shape(methods.sections), "the downloaded methods are the sections the manuscript rail serves").toEqual(shape(served.sections));
    // The bundle's performance file carries the opened held-out scores exactly as the artifact serves them.
    const csv = bundle.read(`${root}results/performance.csv`).trim().split("\n").map((l) => l.split(","));
    const col = csv[0]!;
    const want = await getFit(page, pid);
    for (const m of want.models) {
      for (const [metric, value] of Object.entries(m.holdout ?? {})) {
        if (value === null) continue;
        const row = csv.find((r) => r[col.indexOf("key")] === `${m.family}/holdout/${metric}`);
        const estimate = Number(row?.[col.indexOf("estimate")]);
        expect(Math.abs(estimate - value), `performance.csv's ${m.family} held-out ${metric}`).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(value)));
      }
    }
  } finally {
    bundle.cleanup();
  }
  await expectBarEqualsQuest(page, pid, "at Write-up");
  expect(problems, "no failed request or console error beyond the refusals the cards show").toEqual([]);
  test.info().annotations.push({ type: "walk", description: describe(final) });
});

// Known defect on the server (reported with this package): the held-out rows are scored once, yet
// the fit says it changed after the opening whenever a noticing was dismissed or triaged, because
// the key the opening's state is given leaves out the findings slot's key view the engine applies
// (turbotab/core/seal.py `keys_for`, :1063, against turbotab/core/graph.py `_compute_keys`, :810;
// the view is turbotab/core/repairs.py `_applied_only`, :1780).
// The page follows the served flag, so its title says "from a fit changed after the opening, so
// not an independent test" of scores nothing has changed. Remove the marker once it is fixed.
test("known defect: the opening's scores stay the reported result when nothing was changed after it", async () => {
  test.skip(!facts.served, "the journey did not reach the opening");
  test.fail(true, "core/seal.py keys_for omits graph.key_value: changed_after_seal is true with no post-seal decision");
  expect(facts.served!.post_seal_decisions, "no decision was recorded after the opening").toEqual([]);
  expect(facts.served!.changed_after_seal).toBe(false);
  expect(facts.served!.at_opening_current).toBe(true);
});
