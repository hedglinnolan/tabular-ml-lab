/**
 * The quest journeys' comparisons of the printed scores with the served fit (e2e/quest.ts
 * `cvDifferences`, `heldOutDifferences`), shown to fail where they should and stay silent where
 * they should, before a journey relies on them. It needs no server: each test renders the table the
 * Results draw (ExhibitTable's markup: a `th scope=col` per column, a `tr[data-row]` per row) from
 * a fit written out by hand, whose numbers are chosen here and printed to three significant digits
 * as the paper prints them, so the expected values come from construction, not from the app.
 *
 *   npx playwright test quest-comparisons
 */
import { expect, test, type Page } from "@playwright/test";
import type { FitArtifact } from "../src/api/m1-stage-types";
import { cvDifferences, heldOutDifferences } from "./quest";

interface Row {
  key: string;
  label: string;
  cells: string[];
}

/** The table markup of ExhibitTable: stub header, one `th scope=col` per column, one row each. */
function table(testid: string, header: string[], rows: Row[]): string {
  const head = header.map((h) => `<th scope="col">${h}</th>`).join("");
  const body = rows
    .map((r) => `<tr data-row="${r.key}"><th scope="row">${r.label}</th>${r.cells.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("");
  return `<section data-testid="${testid}"><table><caption><b>Table.</b> x</caption><thead><tr><th scope="col">Model</th>${head}</tr></thead><tbody>${body}</tbody></table></section>`;
}

/** Two families scored on RMSE and R²: every number is chosen here. The cross-validated scores,
 *  the verdict against the no-predictor baseline, BBC-CV (the best family's corrected RMSE, and the
 *  corrected R² in `extras`), then the same on the held-out rows. */
const fit = {
  task: "regression",
  final_model: "linear",
  metric_labels: { rmse: "RMSE", r2: "R²" },
  models: [
    {
      family: "linear",
      label: "Linear regression",
      cv: {
        rmse: { estimate: 34.7387, mean: 34.7387, ci_low: 31.58, ci_high: 37.9147 },
        r2: { estimate: 0.13312, mean: 0.13312, ci_low: 0.1203, ci_high: 0.14598 },
      },
      baseline: { metric: "rmse", value: 35.2, label: "the outcome's average" },
      versus_baseline: { verdict: "better", metric: "r2", gain: 0.13312, ci_low: 0.1203, ci_high: 0.14598 },
      holdout: { rmse: 33.2145, r2: 0.1187 },
      holdout_detail: {
        intervals: {
          rmse: { estimate: 33.2145, ci_low: 30.1, ci_high: 36.4 },
          r2: { estimate: 0.1187, ci_low: 0.0612, ci_high: 0.1762 },
        },
        baseline: { label: "the outcome's average", scores: { rmse: 35.2 } },
        versus_baseline: { verdict: "no_better", metric: "r2", gain: 0.0123, ci_low: -0.0401, ci_high: 0.0647 },
      },
    },
    {
      family: "ridge",
      label: "Ridge regression",
      cv: {
        rmse: { estimate: 34.8121, mean: 34.8121, ci_low: 31.7, ci_high: 37.9 },
        r2: { estimate: 0.1302, mean: 0.1302, ci_low: 0.1177, ci_high: 0.1431 },
      },
      baseline: { metric: "rmse", value: 35.2, label: "the outcome's average" },
      versus_baseline: { verdict: "worse", metric: "r2", gain: null, ci_low: null, ci_high: null },
      holdout: { rmse: 33.31, r2: 0.1136 },
      holdout_detail: {
        intervals: {
          rmse: { estimate: 33.31, ci_low: 30.2, ci_high: 36.5 },
          r2: { estimate: 0.1136, ci_low: 0.0583, ci_high: 0.1689 },
        },
        baseline: { label: "the outcome's average", scores: { rmse: 35.2 } },
        versus_baseline: { verdict: "better", metric: "r2", gain: 0.0412, ci_low: null, ci_high: null },
      },
    },
  ],
  selection: {
    metric: "rmse",
    best: "linear",
    corrected: 34.9053,
    corrected_low: 32.1,
    corrected_high: 37.6,
    extras: { r2: { corrected: 0.1281, corrected_low: 0.1127, corrected_high: 0.1433 } },
  },
} as unknown as FitArtifact;

const AGAINST = "Against the outcome's average";

/** The cross-validated table as the Results draw it for `fit`. */
const CV: { header: string[]; rows: Row[] } = {
  header: ["RMSE", "R²", AGAINST],
  rows: [
    { key: "linear", label: "Linear regression", cells: ["34.7 (31.6 to 37.9)", "0.133 (0.120 to 0.146)", "Better: a gain in R² of 0.133 (0.120 to 0.146)"] },
    { key: "ridge", label: "Ridge regression", cells: ["34.8 (31.7 to 37.9)", "0.130 (0.118 to 0.143)", "Worse"] },
    { key: "baseline", label: "The outcome's average, with no predictors", cells: ["35.2", "", ""] },
    { key: "corrected", label: "Linear regression, corrected for choosing it", cells: ["34.9 (32.1 to 37.6)", "0.128 (0.113 to 0.143)", ""] },
  ],
};

/** The held-out table as the Results draw it for `fit`, once the rows are opened. */
const HELD: { header: string[]; rows: Row[] } = {
  header: ["RMSE", "R²", AGAINST],
  rows: [
    { key: "linear", label: "Linear regression", cells: ["33.2 (30.1 to 36.4)", "0.119 (0.0612 to 0.176)", "Not shown to be better: a gain in R² of 0.0123 (−0.0401 to 0.0647)"] },
    { key: "ridge", label: "Ridge regression", cells: ["33.3 (30.2 to 36.5)", "0.114 (0.0583 to 0.169)", "Better: a gain in R² of 0.0412"] },
    { key: "baseline", label: "The outcome's average, with no predictors", cells: ["35.2", "", ""] },
  ],
};

async function showCv(page: Page, header = CV.header, rows = CV.rows) {
  await page.setContent(table("results-cv", header, rows));
}
async function showHeld(page: Page, header = HELD.header, rows = HELD.rows) {
  await page.setContent(table("results-held-out", header, rows));
}

/** A copy of ``rows`` with one cell replaced. */
function withCell(rows: Row[], key: string, column: number, text: string): Row[] {
  return rows.map((r) => (r.key === key ? { ...r, cells: r.cells.map((c, i) => (i === column ? text : c)) } : r));
}
/** A copy of the table without the column ``at``. */
function withoutColumn(t: { header: string[]; rows: Row[] }, at: number) {
  return { header: t.header.filter((_, i) => i !== at), rows: t.rows.map((r) => ({ ...r, cells: r.cells.filter((_, i) => i !== at) })) };
}

/** A copy of ``rows`` with the cells of one column emptied. */
function blankColumn(rows: Row[], column: number): Row[] {
  return rows.map((r) => ({ ...r, cells: r.cells.map((c, i) => (i === column ? "" : c)) }));
}
/** A copy of ``rows`` with one row's label (its ``th scope=row``) replaced. */
function relabeled(rows: Row[], key: string, label: string): Row[] {
  return rows.map((r) => (r.key === key ? { ...r, label } : r));
}
/** A copy of ``fit`` whose verdict against the baseline is not served for the families named (for
 *  every family when none is named): the cross-validated verdict, or the held-out one. */
function withoutVerdict(kind: "cv" | "held-out", families?: readonly string[]): FitArtifact {
  const copy = structuredClone(fit);
  for (const m of copy.models) {
    if (families && !families.includes(m.family)) continue;
    if (kind === "cv") m.versus_baseline = null;
    else if (m.holdout_detail) m.holdout_detail.versus_baseline = null;
  }
  return copy;
}
/** Exactly one difference is reported, and it names ``needle``: the check under test fired, and
 *  nothing else did. */
function expectOne(differences: string[], needle: string): void {
  expect(differences, `one difference, naming "${needle}"`).toHaveLength(1);
  expect(differences[0]).toContain(needle);
}

test.describe("the cross-validated comparison", () => {
  test("is silent on a table that prints what is served", async ({ page }) => {
    await showCv(page);
    expect(await cvDifferences(page, fit)).toEqual([]);
  });

  test("catches a served metric that has no column", async ({ page }) => {
    const t = withoutColumn(CV, 1); // R² is served on every family, and not drawn
    await showCv(page, t.header, t.rows);
    expect((await cvDifferences(page, fit)).join("; ")).toContain("R²");
  });

  test("catches a verdict cell that differs, and a verdict column that is missing", async ({ page }) => {
    await showCv(page, CV.header, withCell(CV.rows, "linear", 2, "Worse: a gain in R² of 0.133 (0.120 to 0.146)"));
    expect((await cvDifferences(page, fit)).join("; "), "the wrong word").toContain("linear");
    await showCv(page, CV.header, withCell(CV.rows, "linear", 2, "Better: a gain in R² of 0.143 (0.120 to 0.146)"));
    expect((await cvDifferences(page, fit)).join("; "), "the wrong gain").toContain("linear");
    await showCv(page, CV.header, withCell(CV.rows, "ridge", 2, "Better"));
    expect((await cvDifferences(page, fit)).join("; "), "a verdict served as worse, printed better").toContain("ridge");
    const t = withoutColumn(CV, 2);
    await showCv(page, t.header, t.rows);
    expect((await cvDifferences(page, fit)).join("; "), "no verdict column though verdicts are served").toContain("verdict");
  });

  test("catches a BBC-CV row that differs, is missing, or is drawn unserved", async ({ page }) => {
    await showCv(page, CV.header, withCell(CV.rows, "corrected", 0, "35.9 (32.1 to 37.6)"));
    expect((await cvDifferences(page, fit)).join("; "), "the corrected RMSE one unit off").toContain("corrected");
    await showCv(page, CV.header, withCell(CV.rows, "corrected", 1, "0.138 (0.113 to 0.143)"));
    expect((await cvDifferences(page, fit)).join("; "), "the corrected R² from extras one unit off").toContain("corrected");
    await showCv(page, CV.header, CV.rows.filter((r) => r.key !== "corrected"));
    expect((await cvDifferences(page, fit)).join("; "), "the row is not drawn though selection is served").toContain("BBC-CV");
    const unserved = { ...fit, selection: null } as unknown as FitArtifact;
    await showCv(page);
    expect((await cvDifferences(page, unserved)).join("; "), "the row is drawn though none is served").toContain("BBC-CV");
    await showCv(page, CV.header, CV.rows.filter((r) => r.key !== "corrected"));
    expect(await cvDifferences(page, unserved), "no row, none served").toEqual([]);
  });

  test("catches a baseline score printed where none is served", async ({ page }) => {
    await showCv(page, CV.header, withCell(CV.rows, "baseline", 1, "0.000"));
    expect((await cvDifferences(page, fit)).join("; ")).toContain("baseline");
  });

  test("catches a verdict cell printed for a family that has none served, and is silent when it is blank", async ({ page }) => {
    const served = withoutVerdict("cv", ["ridge"]); // linear's verdict is still served, ridge's is not
    await showCv(page); // ridge's cell prints "Worse"
    expectOne(await cvDifferences(page, served), "ridge: a verdict");
    await showCv(page, CV.header, withCell(CV.rows, "ridge", 2, ""));
    expect(await cvDifferences(page, served), "no verdict served for ridge, none printed").toEqual([]);
  });

  test("catches a verdict column drawn though no verdict is served, and is silent when none is drawn", async ({ page }) => {
    const none = withoutVerdict("cv");
    await showCv(page, CV.header, blankColumn(CV.rows, 2)); // the column alone: its cells print nothing
    expectOne(await cvDifferences(page, none), "the verdict column is drawn though no verdict is served");
    await showCv(page); // the column and the verdicts in its cells
    const printed = (await cvDifferences(page, none)).join("; ");
    for (const part of ["the verdict column is drawn", "linear: a verdict", "ridge: a verdict"]) expect(printed).toContain(part);
    const t = withoutColumn(CV, 2);
    await showCv(page, t.header, t.rows);
    expect(await cvDifferences(page, none), "no verdict served, no column drawn").toEqual([]);
  });

  test("catches a BBC-CV row labeled for a family other than the one chosen, and is silent when it names it", async ({ page }) => {
    await showCv(page, CV.header, relabeled(CV.rows, "corrected", "Ridge regression, corrected for choosing it"));
    const said = await cvDifferences(page, fit); // linear was chosen
    expectOne(said, 'labeled "Ridge regression, corrected for choosing it"');
    expect(said[0]).toContain("not for Linear regression");
    const ridgeChosen = { ...structuredClone(fit), selection: { ...fit.selection!, best: "ridge" } };
    await showCv(page, CV.header, relabeled(CV.rows, "corrected", "Ridge regression, corrected for choosing it"));
    expect(await cvDifferences(page, ridgeChosen), "ridge chosen, the row names ridge").toEqual([]);
    await showCv(page);
    expectOne(await cvDifferences(page, ridgeChosen), "not for Ridge regression"); // the row still names linear
  });

  test("catches a missing baseline row, and is silent when no baseline is served", async ({ page }) => {
    await showCv(page, CV.header, CV.rows.filter((r) => r.key !== "baseline"));
    expectOne(await cvDifferences(page, fit), "the baseline has no row (cross-validated)");
    const unscored = structuredClone(fit);
    for (const m of unscored.models) m.baseline.value = null;
    expect(await cvDifferences(page, unscored), "no baseline score served, no row drawn").toEqual([]);
  });

  test("catches a metric served with no label, and names the missing column once it has one", async ({ page }) => {
    const withMae = structuredClone(fit);
    withMae.models[0]!.cv.mae = { estimate: 27.1, mean: 27.1, ci_low: 25.2, ci_high: 29 } as unknown as FitArtifact["models"][number]["cv"][string];
    await showCv(page);
    expectOne(await cvDifferences(page, withMae), "mae is served on the cross-validated scores with no label");
    withMae.metric_labels.mae = "MAE"; // labeled now, and the table still has no MAE column
    expectOne(await cvDifferences(page, withMae), "the cross-validated table has no column for MAE");
  });
});

test.describe("the held-out comparison", () => {
  test("is silent on a table that prints what is served", async ({ page }) => {
    await showHeld(page);
    expect(await heldOutDifferences(page, fit)).toEqual([]);
  });

  test("catches a served metric that has no column", async ({ page }) => {
    const t = withoutColumn(HELD, 0); // RMSE is served on every family, and not drawn
    await showHeld(page, t.header, t.rows);
    expect((await heldOutDifferences(page, fit)).join("; ")).toContain("RMSE");
  });

  test("catches a verdict that differs, with and without a served gain interval", async ({ page }) => {
    await showHeld(page, HELD.header, withCell(HELD.rows, "linear", 2, "Better: a gain in R² of 0.0123 (−0.0401 to 0.0647)"));
    expect((await heldOutDifferences(page, fit)).join("; "), "the wrong word").toContain("linear");
    await showHeld(page, HELD.header, withCell(HELD.rows, "ridge", 2, "Better: a gain in R² of 0.0512"));
    expect((await heldOutDifferences(page, fit)).join("; "), "the wrong gain, served without an interval").toContain("ridge");
    await showHeld(page, HELD.header, withCell(HELD.rows, "ridge", 2, "Better: a gain in R² of 0.0412 (0.01 to 0.07)"));
    expect((await heldOutDifferences(page, fit)).join("; "), "an interval printed that is not served").toContain("ridge");
  });

  test("catches a baseline score printed where none is served, and is silent on one that is", async ({ page }) => {
    await showHeld(page, HELD.header, withCell(HELD.rows, "baseline", 1, "0.000")); // the baseline is scored on RMSE only
    expectOne(await heldOutDifferences(page, fit), "baseline R²");
    const scoredOnR2 = structuredClone(fit);
    scoredOnR2.models[0]!.holdout_detail!.baseline!.scores.r2 = -0.0031;
    await showHeld(page, HELD.header, withCell(HELD.rows, "baseline", 1, "−0.00310"));
    expect(await heldOutDifferences(page, scoredOnR2), "R² served for the baseline, and printed").toEqual([]);
    await showHeld(page, HELD.header, withCell(HELD.rows, "baseline", 1, "−0.00410"));
    expectOne(await heldOutDifferences(page, scoredOnR2), "baseline R²");
  });

  test("catches a verdict cell printed for a family that has none served, and is silent when it is blank", async ({ page }) => {
    const served = withoutVerdict("held-out", ["ridge"]);
    await showHeld(page); // ridge's cell prints "Better: a gain in R² of 0.0412"
    expectOne(await heldOutDifferences(page, served), "ridge: a verdict");
    await showHeld(page, HELD.header, withCell(HELD.rows, "ridge", 2, ""));
    expect(await heldOutDifferences(page, served), "no verdict served for ridge, none printed").toEqual([]);
  });

  test("catches a verdict column drawn though no verdict is served, and is silent when none is drawn", async ({ page }) => {
    const none = withoutVerdict("held-out");
    await showHeld(page, HELD.header, blankColumn(HELD.rows, 2));
    expectOne(await heldOutDifferences(page, none), "the verdict column is drawn though no verdict is served");
    const t = withoutColumn(HELD, 2);
    await showHeld(page, t.header, t.rows);
    expect(await heldOutDifferences(page, none), "no verdict served, no column drawn").toEqual([]);
  });

  test("catches a metric served with no label", async ({ page }) => {
    const withMae = structuredClone(fit);
    withMae.models[0]!.holdout!.mae = 25.1;
    await showHeld(page);
    expectOne(await heldOutDifferences(page, withMae), "mae is served on the held-out scores with no label");
    withMae.metric_labels.mae = "MAE";
    expectOne(await heldOutDifferences(page, withMae), "the held-out table has no column for MAE");
  });
});
