/**
 * The exhibit view kinds (FOUNDATION §5 rule 9), each designed once here, with its data contract,
 * its table alternative and its one line instead of an empty frame; every exhibit draws through
 * them. The parts they share (palette by role, tooltip, table alternative, legend, axes, the one
 * line) are in ./common/parts.tsx, the scale math in ./common/scale.ts, the gate in ./common/gate.ts.
 *
 *   table, forest, page          ExhibitTable, Forest, PagePreview (adapters.ts, types.ts)
 *   curve, calibration,          CurveView, CalibrationView, DecisionCurveView, SpecCurveView
 *   decision curve, spec curve   (curveAdapters.ts)
 *   overlap, embedding, matrix   overlap/, embedding/, matrix/
 *
 * /lab/views (dev:mock) renders every kind with its fixtures in both themes.
 */
export { ExhibitTable, TABLE_PURPOSE } from "./ExhibitTable";
export { Forest, FOREST_PURPOSE, forestTable, TableWithForest } from "./Forest";
export { PagePreview, PAGE_PURPOSE, PLACEMENT_LABEL } from "./PagePreview";
export { forestScale } from "./common/scale";
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
export {
  CATEGORICAL,
  CHOICE,
  FIT,
  Legend,
  NOW,
  Numbers,
  OneLine,
  slotColor,
  TableAlternative,
  useTip,
  useWidth,
  type LegendItem,
  type Slot,
  type TableSpec,
} from "./common/parts";
export { ticksIn } from "./common/scale";
