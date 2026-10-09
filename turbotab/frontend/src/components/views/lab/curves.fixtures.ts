/**
 * The curve views' lab fixtures (dev:mock only). Captured engine artifacts wherever the engine
 * serves one: the substitution curves and calibrations of the captured journeys
 * (src/mocks/fixtures/m3-*.json) and the 20 fitted plans of the calm scenario
 * (src/explore/calm-kit/fixture.json). Where it serves none yet, `illustrative.json`, written by
 * make_fixtures.py through the engine's own functions on a seeded synthetic sample.
 */
import inference from "../../../mocks/fixtures/m3-nhanes-inference.json";
import prediction from "../../../mocks/fixtures/m3-nhanes-prediction.json";
import clinical from "../../../mocks/fixtures/m3-clinical.json";
import calm from "../../../explore/calm-kit/fixture.json";
import illustrative from "./illustrative.json";
import type { ModelCalibration, SubstitutionArtifact } from "../../../api/m3-types";
import { calibrationFromEngine, curveFromSubstitution, decisionFromEngine, type EngineDecisionCurve } from "../curveAdapters";
import type { CalibrationBin, CalibrationData } from "../calibration";
import type { CurveData } from "../curve";
import type { DecisionCurveData } from "../decisionCurve";
import type { Spec, SpecCurveData } from "../specCurve";

interface Journey {
  artifacts: { substitution?: { base: unknown }; fit?: { base: { models: { family: string; calibration: unknown }[] } } };
}

const subst = (j: unknown) => (j as Journey).artifacts.substitution!.base as SubstitutionArtifact;
const calOf = (j: unknown, i = 0) => (j as Journey).artifacts.fit!.base.models[i]!.calibration as ModelCalibration;

// ── curves ───────────────────────────────────────────────────────────────────

/** NHANES, Predict: boosted trees' curve as it stands, and with the same rows at every k. */
export const substitutionChoice: CurveData = curveFromSubstitution(subst(prediction), { outcome: "glucose", family: "boosted_trees", fixed: true });
/** NHANES, Predict: both models compared. */
export const substitutionCompare: CurveData = curveFromSubstitution(subst(prediction), { outcome: "glucose" });
/** NHANES, Estimate: before Fit, only which rows stay in range at each k (outcome-free). */
export const substitutionSealed: CurveData = {
  ...curveFromSubstitution(subst(inference), { outcome: "glucose", family: "linear" }),
  lines: [],
  stop: null,
  sealed: "The curve opens after Fit. Until then, the strip shows how many rows stay within the range observed at each k.",
};

interface Illustrative {
  exposure_curve: {
    exposure: string;
    unit: string;
    outcome: string;
    outcome_unit: string;
    reference: number;
    now: { label: string; x: number[]; y: number[]; low: number[]; high: number[] };
    choice: { label: string; x: number[]; y: number[]; low: number[]; high: number[] };
    rug: number[];
    n: number;
  };
  calibration: ModelCalibration & { bins: CalibrationBin[]; bins_method: string };
  decision_curve: EngineDecisionCurve;
}
const ill = illustrative as unknown as Illustrative;
const ec = ill.exposure_curve;

/**
 * Illustrative, after Fit: an exposure curve under two declared shapes, compared as sensitivity
 * (two `series`, sage and plum). The shape is a Models-stage choice made before Fit, so it is never
 * previewed as a "with this choice" flip (FOUNDATION §5 rule 6).
 */
export const exposureCurve: CurveData = {
  xLabel: `${ec.exposure} (${ec.unit})`,
  xName: `${ec.exposure} (${ec.unit})`,
  yLabel: `Difference in mean ${ec.outcome} (${ec.outcome_unit}) from ${ec.reference} ${ec.unit}`,
  lines: [
    { key: "spline", label: "A bending curve", role: "series", slot: 1, x: ec.now.x, y: ec.now.y, low: ec.now.low, high: ec.now.high },
    { key: "linear", label: "A straight line", role: "series", slot: 2, x: ec.choice.x, y: ec.choice.y, low: ec.choice.low, high: ec.choice.high },
  ],
  zero: true,
  rug: ec.rug,
  band: "Bands are 95% intervals.",
  basis: `The bending curve is a spline with 4 knots, the points where its bend may change. Rug: where ${ec.exposure} was observed, at 120 quantiles of ${ec.n.toLocaleString("en-US")} rows.`,
  sealed: null,
};

export const curveOnePoint: CurveData = { xLabel: "kcal moved", yLabel: "Change in predicted glucose", lines: [{ key: "a", label: "Linear model", role: "now", x: [100], y: [-1.86] }], zero: true, sealed: null };

// ── calibration ──────────────────────────────────────────────────────────────

/** Clinical, Predict: the held-out risks of progression (399 rows). */
export const calibrationClinical: CalibrationData = calibrationFromEngine(calOf(clinical), { kind: "risk", outcome: "progression", where: "held_out" });
/** NHANES, Predict: predicted glucose against observed. */
export const calibrationGlucose: CalibrationData = calibrationFromEngine(calOf(prediction), { kind: "value", outcome: "glucose", where: "out_of_fold" });
/** Illustrative: an overconfident model, with ten groups of rows and their intervals. */
export const calibrationBinned: CalibrationData = calibrationFromEngine(ill.calibration, { kind: "risk", outcome: "the outcome", where: "out_of_fold" });
/** Degenerate: one group of 120 rows, its own counts only (no slope or intercept from one group). */
const oneBin = ill.calibration.bins[4]!;
export const calibrationOnePoint: CalibrationData = {
  ...calibrationBinned,
  n: oneBin.n,
  observed: oneBin.observed,
  expected: oneBin.predicted,
  intercept: { estimate: null },
  slope: { estimate: null },
  concern: null,
  curve: [],
  bins: [oneBin],
};
/** The held-out calibration while the calibration horizon is chosen: refused. */
export const calibrationChoosing: CalibrationData = { ...calibrationClinical, choosing: "the calibration horizon" };
/** Before Fit: one line. */
export const calibrationSealed: CalibrationData = { ...calibrationBinned, sealed: "Calibration opens after Fit, on predictions scored out of fold." };

// ── decision curve ───────────────────────────────────────────────────────────

export const decision: DecisionCurveData = decisionFromEngine(ill.decision_curve, { where: "out_of_fold" });
/** Degenerate: one threshold (0.21), its useful span recomputed for that threshold alone. */
const oneRow = decision.rows[20]!;
const beats = (oneRow.models[decision.models[0]!.key] ?? -Infinity) > Math.max(oneRow.treat_all, 0);
export const decisionOnePoint: DecisionCurveData = { ...decision, rows: [oneRow], useful: beats ? [oneRow.threshold, oneRow.threshold] : null };
/** A choice that would narrow the threshold range to 0.10–0.30, pointed at, on out-of-fold scores. */
export const decisionPointed: DecisionCurveData = { ...decision, pointed: { low: 0.1, high: 0.3 } };
/** The same on held-out scores: refused, since that would choose the range by the held-out score. */
export const decisionPointedHeldOut: DecisionCurveData = { ...decisionPointed, where: "held_out" };

// ── specification curve: the calm scenario's fitted plans ────────────────────

interface Fit {
  sequence: { key: string; label: string; n_rows: number; effects: { feature: string; estimate: number; ci_low: number | null; ci_high: number | null }[] }[];
  sensitivity: { analyses: { label: string; primary: boolean; n_rows: number; effects: { estimate: number; ci_low: number | null; ci_high: number | null }[] }[] };
}

const ENERGY = [
  { key: "standard", label: "Calories as a covariate" },
  { key: "residual", label: "Residual method" },
  { key: "residual_energy_dropped", label: "Residual, calories left out" },
];
const ADJUST = [
  { key: "crude", label: "Nothing" },
  { key: "model_1", label: "Model 1" },
  { key: "model_2", label: "Model 2" },
  { key: "model_3", label: "Model 3" },
];
const ROWS = [
  { key: "every", label: "Every row" },
  { key: "willett", label: "Willett range, by sex" },
  { key: "nhs", label: "NHS/HPFS range, by sex" },
];

/** Sugar's estimate per gram in each plan (the per-kcal plans estimate on another scale). */
export function specsFromCalm(fits: Record<string, Fit>): Spec[] {
  const out: Spec[] = [];
  for (const e of ENERGY) {
    for (const [screen, rows] of [["none", "every"], ["willett_2013_by_sex", "willett"]] as const) {
      const fit = fits[`${screen}|${e.key}`];
      if (!fit) continue;
      for (const s of fit.sequence) {
        const eff = s.effects[0]!;
        out.push({
          key: `${e.key}|${s.key}|${rows}`,
          estimate: eff.estimate,
          low: eff.ci_low,
          high: eff.ci_high,
          n: s.n_rows,
          primary: e.key === "residual" && s.key === "model_2" && rows === "every",
          picks: { energy: e.key, adjust: s.key, rows },
          scaleKey: "per g",
        });
      }
      if (screen === "none") {
        const a = fit.sensitivity.analyses.find((x) => x.label === "NHS/HPFS, by sex");
        const eff = a?.effects[0];
        if (a && eff) out.push({ key: `${e.key}|model_2|nhs`, estimate: eff.estimate, low: eff.ci_low, high: eff.ci_high, n: a.n_rows, primary: false, picks: { energy: e.key, adjust: "model_2", rows: "nhs" }, scaleKey: "per g" });
      }
    }
  }
  return out;
}

export const specCurve: SpecCurveData = {
  estimateLabel: "Difference in mean glucose per g of sugar",
  zero: true,
  // the fits' served inference: "95% intervals from HC3 heteroskedasticity-robust standard errors"
  level: 0.95,
  sealed: null,
  choices: [
    { key: "energy", label: "Calories handled by", options: ENERGY },
    { key: "adjust", label: "Adjusted for", options: ADJUST },
    { key: "rows", label: "Rows kept", options: ROWS },
  ],
  specs: specsFromCalm((calm as unknown as { fits: Record<string, Fit> }).fits),
};

export const specOne: SpecCurveData = { ...specCurve, specs: specCurve.specs.filter((s) => s.primary) };
export const specSealed: SpecCurveData = { ...specCurve, sealed: "The specification curve opens after Fit locks the plan: seeing the estimates while choosing would invite choosing by them." };
export const specMixed: SpecCurveData = { ...specCurve, specs: [specCurve.specs[0]!, { ...specCurve.specs[1]!, scaleKey: "per kcal" }] };
