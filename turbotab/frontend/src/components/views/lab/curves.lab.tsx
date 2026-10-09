/** The curve views' lab entries: curve, calibration, decision curve, specification curve. */
import { CalibrationView } from "../CalibrationView";
import { CurveView } from "../CurveView";
import { DecisionCurveView } from "../DecisionCurveView";
import { SpecCurveView } from "../SpecCurveView";
import { VIEW_PURPOSES } from "../../stage/purposes";
import type { LabEntry, LabSample } from "./entry";
import * as F from "./curves.fixtures";

const CAPTURED = "Captured: the real server's journey (src/mocks/fixtures).";
const ILLUSTRATIVE = "Illustrative: no engine artifact yet; the engine's functions on a seeded sample (lab/make_fixtures.py).";

const SAMPLES: (LabSample & { kind: string })[] = [
  { kind: "Curve", label: "Substitution, with this choice", source: `${CAPTURED} NHANES Predict, boosted trees: each k's own rows (gray) and the same rows at every k (indigo).`, render: () => <CurveView data={F.substitutionChoice} title="Moving 100 kcal from protein to sugar" /> },
  { kind: "Curve", label: "Substitution, your data now", source: `${CAPTURED} The same curve with the flip on "Your data now".`, render: () => <CurveView data={F.substitutionChoice} showChoice={false} title="Moving 100 kcal from protein to sugar" /> },
  { kind: "Curve", label: "Substitution, two models compared", source: `${CAPTURED} NHANES Predict: the linear model and boosted trees (sage, plum).`, render: () => <CurveView data={F.substitutionCompare} /> },
  { kind: "Curve", label: "Substitution before Fit", source: `${CAPTURED} NHANES Estimate: the outcome-free share of rows in range at each k, in its labelled strip, with its table and tooltip (FOUNDATION §5 rule 6).`, render: () => <CurveView data={F.substitutionSealed} /> },
  { kind: "Curve", label: "Exposure curve, two shapes compared", source: `${ILLUSTRATIVE} After Fit, as sensitivity: a bending curve (sage) beside a straight line (plum). The shape is chosen before Fit, so it is never previewed as "with this choice".`, render: () => <CurveView data={F.exposureCurve} /> },
  { kind: "Curve", label: "One point", source: "Degenerate: a single estimate.", render: () => <CurveView data={F.curveOnePoint} /> },
  { kind: "Curve", label: "No data", source: "Empty: one line, never an empty frame.", render: () => <CurveView data={null} /> },

  { kind: "Calibration", label: "Held-out risks", source: `${CAPTURED} Clinical Predict: 399 held-out rows; the smoothed curve only (bins are not served yet).`, render: () => <CalibrationView data={F.calibrationClinical} /> },
  { kind: "Calibration", label: "A number predicted", source: `${CAPTURED} NHANES Predict: predicted against observed glucose.`, render: () => <CalibrationView data={F.calibrationGlucose} /> },
  { kind: "Calibration", label: "Grouped rows with intervals", source: `${ILLUSTRATIVE} The engine's calibration, plus ten groups with Wilson intervals.`, render: () => <CalibrationView data={F.calibrationBinned} /> },
  { kind: "Calibration", label: "One group", source: "Degenerate: a single grouped point, with its own 120 rows.", render: () => <CalibrationView data={F.calibrationOnePoint} /> },
  { kind: "Calibration", label: "Held out, while the horizon is chosen", source: "Refused: the held-out rows stay closed while a choice they would inform is made (FOUNDATION §3).", render: () => <CalibrationView data={F.calibrationChoosing} /> },
  { kind: "Calibration", label: "Before Fit", source: "Sealed: one line (FOUNDATION §5 rule 6).", render: () => <CalibrationView data={F.calibrationSealed} /> },
  { kind: "Calibration", label: "No data", source: "Empty.", render: () => <CalibrationView data={null} /> },

  { kind: "Decision curve", label: "Two models across thresholds", source: `${ILLUSTRATIVE} The engine's decision_curve over 0.01–0.70, out of fold; the declared range 0.05–0.50 shaded.`, render: () => <DecisionCurveView data={F.decision} /> },
  { kind: "Decision curve", label: "A new range, pointed at", source: "Out of fold, while an option that would declare 0.10–0.30 is pointed at: the new range in indigo, the declared one in gray.", render: () => <DecisionCurveView data={F.decisionPointed} /> },
  { kind: "Decision curve", label: "A new range, pointed at, held out", source: "Refused: choosing the range on held-out net benefit would be choosing by the held-out score (FOUNDATION §3).", render: () => <DecisionCurveView data={F.decisionPointedHeldOut} /> },
  { kind: "Decision curve", label: "One threshold", source: "Degenerate: a single threshold (0.21), each line a marker.", render: () => <DecisionCurveView data={F.decisionOnePoint} /> },
  { kind: "Decision curve", label: "No data", source: "Empty.", render: () => <DecisionCurveView data={null} /> },

  { kind: "Specification curve", label: "Sugar across 27 plans", source: "Captured: the calm scenario's fitted plans (src/explore/calm-kit/fixture.json): calories × adjustment × rows.", render: () => <SpecCurveView data={F.specCurve} /> },
  { kind: "Specification curve", label: "The primary alone", source: "Degenerate: one specification.", render: () => <SpecCurveView data={F.specOne} /> },
  { kind: "Specification curve", label: "Before the lock", source: "Sealed: one line (FOUNDATION §5 rule 6).", render: () => <SpecCurveView data={F.specSealed} /> },
  { kind: "Specification curve", label: "Mixed scales", source: "Refused: estimates on two scales cannot share one axis.", render: () => <SpecCurveView data={F.specMixed} /> },
  { kind: "Specification curve", label: "No data", source: "Empty.", render: () => <SpecCurveView data={null} /> },
];

/** The kinds in FOUNDATION §5 rule 9's order, each with its purpose entry. */
const KINDS = [
  { kind: "Curve", name: "curve", order: 3, purpose: VIEW_PURPOSES.curve },
  { kind: "Calibration", name: "calibration", order: 4, purpose: VIEW_PURPOSES.calibration },
  { kind: "Decision curve", name: "decision curve", order: 5, purpose: VIEW_PURPOSES.decision_curve },
  { kind: "Specification curve", name: "specification curve", order: 6, purpose: VIEW_PURPOSES.spec_curve },
] as const;

export const entries: LabEntry[] = KINDS.map(({ kind, name, order, purpose }) => ({
  kind: name,
  order,
  purpose,
  samples: SAMPLES.filter((s) => s.kind === kind).map(({ kind: _kind, ...sample }) => sample),
}));
