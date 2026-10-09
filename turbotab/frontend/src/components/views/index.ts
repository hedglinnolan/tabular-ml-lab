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
