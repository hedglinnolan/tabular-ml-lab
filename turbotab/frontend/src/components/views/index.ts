/**
 * The exhibit view kinds (FOUNDATION §5 rule 9), each designed once: its design decisions are
 * recorded at the top of its file, its purpose entry beside it, its input type in `types.ts`.
 * The views draw with the calm tokens (docs/turbotab-next/calm/tokens.css), which the host loads.
 */
export { ExhibitTable, TABLE_PURPOSE } from "./ExhibitTable";
export { Forest, FOREST_PURPOSE, forestTable } from "./Forest";
export { PagePreview, PAGE_PURPOSE, PLACEMENT_LABEL } from "./PagePreview";
export { forestScale } from "./scale";
export { fmtCell, fmtEstimate, fmtP } from "./format";
export { forestFromEffects, pageFromExhibits, table1FromArtifact, table1FromProfile, table2FromEffects } from "./adapters";
export type * from "./types";
export type * from "./contracts";
