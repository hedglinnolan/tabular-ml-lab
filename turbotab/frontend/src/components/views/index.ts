/**
 * The exhibit views (FOUNDATION §5 rule 9): each view kind is designed once here, with its data
 * contract, its table alternative and its empty state, and every exhibit draws through it.
 *
 *   curve            an estimate across one input, its band, the input's rug; gray now, indigo choice
 *   calibration      predicted against observed on one scale, the 45° line, groups, smoothed curve
 *   decision curve   net benefit across thresholds, treat everyone / no one, the declared range
 *   spec curve       the estimate in every declared specification, sorted, with the choices below
 *
 * /lab/views (dev:mock) renders every kind with its fixtures in both themes.
 */
export { CurveView } from "./CurveView";
export type { CurveData, CurveLine } from "./curve";
export { CalibrationView } from "./CalibrationView";
export type { CalibrationBin, CalibrationData } from "./calibration";
export { DecisionCurveView } from "./DecisionCurveView";
export type { DecisionCurveData, DecisionCurveRow, ScoredWhere } from "./decisionCurve";
export { SpecCurveView } from "./SpecCurveView";
export type { Spec, SpecChoice, SpecCurveData } from "./specCurve";
export { calibrationFromEngine, curveFromSubstitution, decisionFromEngine } from "./adapters";
export type { EngineDecisionCurve } from "./adapters";
