/**
 * W1 acceptance 4 (plans/WALKABLE_PLAN.md §5; plans/W1_PART2_PLAN.md §2 "W1-L"): an own CSV, the
 * longitudinal clinical sample, with a yes/no outcome under Predict and a seal drawn by time,
 * walked through the quest shell at /p/:pid to the opened held-out rows.
 *
 *   TURBOTAB_WORKERS=2 TURBOTAB_HOME=$(mktemp -d) scripts/py -m turbotab.server --port 8863
 *   E2E_BASE_URL=http://127.0.0.1:8863 npx playwright test quest-own-csv
 *
 * The file is uploaded through the Start screen's drop zone, so this is a person's own table and
 * not one the project knows: the lens, the outcome and its event, the follow-up, the purpose, the
 * repeats (time points, each subject's rows kept together), the temporal question (a seal drawn by
 * time, ordered by the visit dates), the held-out rows, the families, then Fit, the triage of the
 * noticing left open, the final model named from the opening's refusal and the rows opened.
 *
 * As in quest-nhanes-predict, the bar and each stage's hover panel equal GET /quest after every
 * step and each printed score equals the served field it prints. The seal's claim is held against
 * the file itself: the held-out units are the latest by their last visit, read from the CSV here.
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import { onMock } from "./backend";
import { ANY_STAGE, pidOf, visibleWhileComputing } from "./fit";
import {
  answerFinding,
  askExplanations,
  calibrationDifferences,
  chooseFamilies,
  comparisonsCatchWhatDiffers,
  cvDifferences,
  describe,
  expectBarEqualsQuest,
  getFit,
  getJson,
  getQuest,
  heldOutDifferences,
  lineId,
  openHeldOutRows,
  pickOption,
  previewSettled,
  pressFit,
  uploadThroughZone,
  walk,
  type Answers,
} from "./quest";
import type { SealPlan } from "../src/api/m2-stage-types";

const HERE = dirname(fileURLToPath(import.meta.url));
const CSV = resolve(HERE, "../../sample_data/clinical_longitudinal.csv");
const REAL = !!process.env.E2E_BASE_URL;
const FAMILIES = ["linear", "ridge"];

// A press that finds nothing to press fails in half a minute, not in the length of the test. What
// the app is computing is waited for before a press, bounded by the computing (e2e/quest.ts
// `walk`, `settled`), so an action finds its target ready.
test.use({ actionTimeout: 30_000 });

/** What the seal's claim comes to on this file, derived from the file alone: the latest fifth of the
 *  subjects by their last visit are held out whole, every row of theirs with them. */
function latestUnits(path: string, share: number) {
  const [head, ...lines] = readFileSync(path, "utf-8").trim().split(/\r?\n/);
  const cols = head!.split(",");
  const [who, when] = [cols.indexOf("subject_id"), cols.indexOf("visit_date")];
  const last = new Map<string, string>();
  const rows = new Map<string, number>();
  for (const line of lines) {
    const f = line.split(",");
    last.set(f[who]!, (last.get(f[who]!) ?? "") < f[when]! ? f[when]! : last.get(f[who]!)!);
    rows.set(f[who]!, (rows.get(f[who]!) ?? 0) + 1);
  }
  const ordered = [...last.keys()].sort((a, b) => (last.get(a)! < last.get(b)! ? -1 : last.get(a)! > last.get(b)! ? 1 : a < b ? -1 : 1));
  const held = ordered.slice(ordered.length - Math.round(share * ordered.length));
  return {
    units: ordered.length,
    held: held.length,
    heldRows: held.reduce((n, s) => n + rows.get(s)!, 0),
    boundary: held.map((s) => last.get(s)!).sort()[0]!,
    trainedUntil: ordered.slice(0, ordered.length - held.length).map((s) => last.get(s)!).sort().at(-1)!,
  };
}

const answers: Answers = {
  // The text coded as two levels, and the findings the column roles settle, are left open for the
  // triage at Results, where the engine recommends a disposition for each. The one that subjects
  // repeat is read and kept as it is, on its own card.
  deferred: (line) =>
    line.source === "finding" && (line.key === "binary_text__sex" || (line.key.startsWith("voice::") && line.key !== "voice::repeats__subject_id")),
  unavailable: ["join_files", "import_codebook"],
  async answer(page, li, line) {
    if (line.source === "finding") return answerFinding(page, li, line);
    switch (lineId(line)) {
      case "question:lens":
        return pickOption(page, li, "option-clinical", { space: true });
      case "question:target": {
        const box = li.getByRole("combobox");
        await box.fill("progressed");
        await box.press("Enter");
        await previewSettled(page);
        return void (await li.getByTestId("record-target").click());
      }
      // A yes/no outcome: which level is the event is asked, never guessed.
      case "question:event":
        await expect(li.getByRole("option", { selected: true })).toHaveCount(0);
        return pickOption(page, li, "option-1");
      case "question:follow_up":
        return pickOption(page, li, "option-same");
      case "question:purpose":
        return pickOption(page, li, "option-prediction");
      case "declaration:set_intended_use":
        return pickOption(page, li, "option-use:risk_estimation");
      case "question:roles":
        return void (await li.getByTestId("record-roles").click());
      // Subjects repeat: said once, and the rows of one subject stay together.
      case "question:grain":
        return pickOption(page, li, "option-repeated");
      case "question:repeat_kind":
        return pickOption(page, li, "option-time_points");
      case "question:unit":
        return pickOption(page, li, "option-row");
      case "question:temporal": {
        // Later from earlier: a seal drawn by time needs the column that says when each row was
        // taken, and the card refuses until it is named, with the visit dates as its way forward.
        await pickOption(page, li, "option-true");
        const refusal = li.getByTestId("refusal");
        await visibleWhileComputing(page, refusal, [ANY_STAGE], { label: "the temporal question refuses until the time column is named" });
        return void (await refusal.getByRole("button", { name: /^Order by visit_date instead$/ }).click());
      }
      case "question:exclusions":
        return pickOption(page, li, "option-none");
      case "question:missing":
        return pickOption(page, li, "option-complete_case");
      case "question:split": {
        // The seal's basis, said on the card as the server serves it: drawn by time, whole subjects held out.
        const plan = (await getJson<{ artifact: SealPlan }>(page, pidOf(page), "/stages/seal_plan")).artifact;
        expect(plan.chronology?.drawn, "the seal is drawn by time").toBe(true);
        const basis = li.getByTestId("seal-basis");
        await expect(basis).toHaveAttribute("data-basis", plan.basis!.state);
        await expect(basis).toContainText(plan.chronology!.time_column!);
        await expect(basis).toContainText(plan.chronology!.boundary!);
        return pickOption(page, li, "option-0.2");
      }
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

test("an own CSV: a yes/no outcome under Predict, sealed by time, through the quest shell to the opened held-out rows", async ({ page }) => {
  // The walk and the opening, and a fit that a slower runner holds for the press and then computes
  // after it (up to e2e/fit.ts FIT_CAP, 900 s), at most.
  test.setTimeout(REAL ? 1_500_000 : 60_000);
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "no-preference" });
  await page.goto("/");
  test.skip(await onMock(page), "the quest journeys run against the real server");

  const problems: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !m.text().includes("status of 409")) problems.push(`console: ${m.text()}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && r.status() !== 409) problems.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });

  const pid = await uploadThroughZone(page, CSV);
  await expectBarEqualsQuest(page, pid, "on opening");

  // ── the first five stages, every open line through its card ────────────────────────────────
  const taken = await walk(page, pid, answers, (log) => log.stages.find((s) => s.key === "models")?.progress?.complete === true);
  for (const id of [
    "question:lens", "question:target", "question:event", "question:purpose", "question:repeat_kind", "question:temporal",
    "question:split", "declaration:set_selection", "question:models", "sweep:models",
  ]) {
    expect(taken, `the walk took ${id}`).toContain(id);
  }

  // The seal's claim against the file: the latest fifth of the subjects, whole, with their earlier visits.
  const plan = (await getJson<{ artifact: SealPlan }>(page, pid, "/stages/seal_plan")).artifact;
  const truth = latestUnits(CSV, 0.2);
  expect(plan.chronology?.time_column).toBe("visit_date");
  expect(plan.chronology?.n_units, "the units held out").toBe(truth.held);
  expect(plan.chronology?.n_held_rows, "the rows of those units").toBe(truth.heldRows);
  expect(plan.chronology?.boundary, "the earliest last visit among them").toBe(truth.boundary);
  expect(truth.trainedUntil < truth.boundary, "every training unit's last visit comes earlier").toBe(true);

  // ── Fit, then Results ──────────────────────────────────────────────────────────────────────
  const sawRunning = await pressFit(page, pid);
  test.info().annotations.push({ type: "fit", description: sawRunning ? "pressed while the fit job ran" : "the fit had computed before the press" });
  await expectBarEqualsQuest(page, pid, "at Results, after Fit");
  const { sealed, family, opened } = await openHeldOutRows(page, pid, FAMILIES, (step) => expectBarEqualsQuest(page, pid, step));
  expect(sealed.task).toBe("binary");

  // The opening scores the rows the seal drew: no more than those units' rows, fewer where the
  // complete-case rule dropped some; the card says how many, from the served count.
  expect(opened.n_holdout).toBeGreaterThan(0);
  expect(opened.n_holdout).toBeLessThanOrEqual(truth.heldRows);
  await expect(page.getByTestId("results-opened")).toContainText(family.label);
  const phase = opened.changed_after_seal ? "post_seal" : "opened";
  await expect(page.getByTestId("results-tapestry")).toHaveAttribute("data-seal-phase", phase, { timeout: 30_000 });

  // ── every printed number is its served field ────────────────────────────────────────────────
  await expect.poll(() => heldOutDifferences(page, opened), { message: "each held-out score, the baseline and any verdict equal the served fields", timeout: 30_000 }).toEqual([]);
  await comparisonsCatchWhatDiffers(page, pid, opened);
  await expect.poll(() => calibrationDifferences(page, opened), { message: "the held-out calibration equals the served fields", timeout: 30_000 }).toEqual([]);
  await expect.poll(() => cvDifferences(page, opened), { message: "the cross-validated scores equal the served fields after the opening", timeout: 30_000 }).toEqual([]);

  // ── the explanations, and the stages the triage settled complete ───────────────────────────
  await askExplanations(page, pid);
  await expectBarEqualsQuest(page, pid, "after the explanations");
  await walk(page, pid, { ...answers, deferred: () => false }, allComplete);
  await expectBarEqualsQuest(page, pid, "with every stage complete");
  expect(problems, "no failed request or console error beyond the refusals the cards show").toEqual([]);
  test.info().annotations.push({ type: "walk", description: describe(await getQuest(page, pid)) });
  // The held-out rows are scored once: the fit the page shows is the one fit for this project.
  expect((await getFit(page, pid)).final_model).toBe(family.family);
});
