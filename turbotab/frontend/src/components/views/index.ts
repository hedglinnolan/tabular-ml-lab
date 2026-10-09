/**
 * The exhibit view kinds (FOUNDATION §5 rule 9), each designed once: its design decisions are
 * recorded at the top of its file, its purpose entry beside it, its input type in `types.ts`.
 * The views draw with the calm tokens (docs/turbotab-next/calm/tokens.css), which the host loads.
 */
export { ExhibitTable, TABLE_PURPOSE } from "./ExhibitTable";
export { Forest, FOREST_PURPOSE, forestTable, TableWithForest } from "./Forest";
export { PagePreview, PAGE_PURPOSE, PLACEMENT_LABEL } from "./PagePreview";
export { forestScale } from "./scale";
export { fmtCell, fmtEstimate, fmtP } from "./format";
export {
  forestFromEffects,
  gateFromLock,
  LOCK_GATE,
  numberInPlacementOrder,
  pageFromExhibits,
  sentence,
  table1FromArtifact,
  table1FromProfile,
  table2FromEffects,
} from "./adapters";
export type * from "./types";
export type * from "./contracts";

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
export { calibrationFromEngine, curveFromSubstitution, decisionFromEngine } from "./curveAdapters";
export type { EngineDecisionCurve } from "./curveAdapters";

/**
 * The exhibit view kinds (FOUNDATION §5 rule 9), each designed once: its input type, its pure
 * layout (tested scale math) and its component, drawn in the calm tokens with a table alternative.
 */
export { OverlapView } from "./overlap/OverlapView";
export { fromEngine as overlapFromEngine } from "./overlap/fromEngine";
export { layoutOverlap } from "./overlap/layout";
export { OVERLAP_PURPOSE, type OverlapGroup, type OverlapInput, type OverlapKeep } from "./overlap/types";
export { EmbeddingView } from "./embedding/EmbeddingView";
export { layoutEmbedding } from "./embedding/layout";
export { EMBEDDING_PURPOSE, type EmbeddingAxis, type EmbeddingInput } from "./embedding/types";
export { MatrixView } from "./matrix/MatrixView";
export { inOrder, layoutMatrix } from "./matrix/layout";
export { MATRIX_PURPOSE, type MatrixInput } from "./matrix/types";
export { gateRefusal, type OutcomeGate } from "./common/gate";
