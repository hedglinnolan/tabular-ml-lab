/** The curve views' lab entries: curve, calibration, decision curve, specification curve. */
import { CalibrationView } from "../CalibrationView";
import { CurveView } from "../CurveView";
import { DecisionCurveView } from "../DecisionCurveView";
import { SpecCurveView } from "../SpecCurveView";
import type { LabEntry } from "./LabViews";
import * as F from "./curves.fixtures";

const CAPTURED = "Captured: the real server's journey (src/mocks/fixtures).";
const ILLUSTRATIVE = "Illustrative: no engine artifact yet; the engine's functions on a seeded sample (lab/make_fixtures.py).";

export const entries: LabEntry[] = [
  { kind: "Curve", name: "Substitution, with this choice", source: `${CAPTURED} NHANES Predict, boosted trees: each k's own rows (gray) and the same rows at every k (indigo).`, render: () => <CurveView data={F.substitutionChoice} title="Moving 100 kcal from protein to sugar" /> },
  { kind: "Curve", name: "Substitution, your data now", source: `${CAPTURED} The same curve with the flip on "Your data now".`, render: () => <CurveView data={F.substitutionChoice} showChoice={false} title="Moving 100 kcal from protein to sugar" /> },
  { kind: "Curve", name: "Substitution, two models compared", source: `${CAPTURED} NHANES Predict: the linear model and boosted trees (sage, plum).`, render: () => <CurveView data={F.substitutionCompare} /> },
  { kind: "Curve", name: "Substitution before Fit", source: `${CAPTURED} NHANES Estimate: the outcome-free share of rows in range at each k, in its labelled strip, with its table and tooltip (FOUNDATION §5 rule 6).`, render: () => <CurveView data={F.substitutionSealed} /> },
  { kind: "Curve", name: "Exposure curve, two shapes compared", source: `${ILLUSTRATIVE} After Fit, as sensitivity: a bending curve (sage) beside a straight line (plum). The shape is chosen before Fit, so it is never previewed as "with this choice".`, render: () => <CurveView data={F.exposureCurve} /> },
  { kind: "Curve", name: "One point", source: "Degenerate: a single estimate.", render: () => <CurveView data={F.curveOnePoint} /> },
  { kind: "Curve", name: "No data", source: "Empty: one line, never an empty frame.", render: () => <CurveView data={null} /> },

  { kind: "Calibration", name: "Held-out risks", source: `${CAPTURED} Clinical Predict: 399 held-out rows; the smoothed curve only (bins are not served yet).`, render: () => <CalibrationView data={F.calibrationClinical} /> },
  { kind: "Calibration", name: "A number predicted", source: `${CAPTURED} NHANES Predict: predicted against observed glucose.`, render: () => <CalibrationView data={F.calibrationGlucose} /> },
  { kind: "Calibration", name: "Grouped rows with intervals", source: `${ILLUSTRATIVE} The engine's calibration, plus ten groups with Wilson intervals.`, render: () => <CalibrationView data={F.calibrationBinned} /> },
  { kind: "Calibration", name: "One group", source: "Degenerate: a single grouped point, with its own 120 rows.", render: () => <CalibrationView data={F.calibrationOnePoint} /> },
  { kind: "Calibration", name: "Held out, while the horizon is chosen", source: "Refused: the held-out rows stay closed while a choice they would inform is made (FOUNDATION §3).", render: () => <CalibrationView data={F.calibrationChoosing} /> },
  { kind: "Calibration", name: "Before Fit", source: "Sealed: one line (FOUNDATION §5 rule 6).", render: () => <CalibrationView data={F.calibrationSealed} /> },
  { kind: "Calibration", name: "No data", source: "Empty.", render: () => <CalibrationView data={null} /> },

  { kind: "Decision curve", name: "Two models across thresholds", source: `${ILLUSTRATIVE} The engine's decision_curve over 0.01–0.70, out of fold; the declared range 0.05–0.50 shaded.`, render: () => <DecisionCurveView data={F.decision} /> },
  { kind: "Decision curve", name: "A new range, pointed at", source: "Out of fold, while an option that would declare 0.10–0.30 is pointed at: the new range in indigo, the declared one in gray.", render: () => <DecisionCurveView data={F.decisionPointed} /> },
  { kind: "Decision curve", name: "A new range, pointed at, held out", source: "Refused: choosing the range on held-out net benefit would be choosing by the held-out score (FOUNDATION §3).", render: () => <DecisionCurveView data={F.decisionPointedHeldOut} /> },
  { kind: "Decision curve", name: "One threshold", source: "Degenerate: a single threshold (0.21), each line a marker.", render: () => <DecisionCurveView data={F.decisionOnePoint} /> },
  { kind: "Decision curve", name: "No data", source: "Empty.", render: () => <DecisionCurveView data={null} /> },

  { kind: "Specification curve", name: "Sugar across 27 plans", source: "Captured: the calm scenario's fitted plans (src/explore/calm-kit/fixture.json): calories × adjustment × rows.", render: () => <SpecCurveView data={F.specCurve} /> },
  { kind: "Specification curve", name: "The primary alone", source: "Degenerate: one specification.", render: () => <SpecCurveView data={F.specOne} /> },
  { kind: "Specification curve", name: "Before the lock", source: "Sealed: one line (FOUNDATION §5 rule 6).", render: () => <SpecCurveView data={F.specSealed} /> },
  { kind: "Specification curve", name: "Mixed scales", source: "Refused: estimates on two scales cannot share one axis.", render: () => <SpecCurveView data={F.specMixed} /> },
  { kind: "Specification curve", name: "No data", source: "Empty.", render: () => <SpecCurveView data={null} /> },
];
