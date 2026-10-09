/**
 * The embedding view's input: each row placed on two derived axes (PCA components, or UMAP's
 * dimensions), colored by one declared grouping.
 *
 * Engine contract item (no engine artifact yet; the lab's fixture is computed from
 * turbotab/sample_data/metabolomics_untargeted.csv by fixtures/make_fixtures.py): an `embedding`
 * view on First look's columns, with
 *  - `method` and, for PCA, each component's share of the spread (`axes[i].share`);
 *  - the coordinates as two parallel arrays, with row ids, in the rows' order;
 *  - the grouping the user declared, its levels in their declared order, and each row's level as
 *    an index into them;
 *  - `columns`: every column embedded, so the gate can be checked on them;
 *  - `outcome`: the outcome and whether its gate is open (FOUNDATION §5 rule 6). Before its gate,
 *    the outcome is neither embedded nor the grouping; the view refuses either in one line;
 *  - `basis`: one plain line on what was embedded (columns, scaling, rows), for the record.
 * Above about 2,000 rows the view draws density instead of points; the engine may send every row
 * (the view bins them) or a binned grid, which would be a further contract item.
 */
import type { Purpose } from "../../stage/purposes";
import type { OutcomeGate } from "../common/gate";

/** The view's purpose entry (FOUNDATION §5 rule 9, in the purpose registry's form, BLUEPRINT §11.2). */
export const EMBEDDING_PURPOSE: Purpose = {
  question: "data_ok",
  answer: "how rows group across many columns at once, colored by a grouping you declared",
};

export interface EmbeddingAxis {
  /** "Component 1", "UMAP 1": a name, never a unit */
  label: string;
  /** PCA only: the component's share of the total spread, 0 to 1 */
  share?: number | null;
}

export interface EmbeddingInput {
  method: "pca" | "umap";
  axes: [EmbeddingAxis, EmbeddingAxis];
  xs: number[];
  ys: number[];
  ids?: string[];
  /** each row's level, an index into `grouping.levels`; null or −1 when it is not recorded */
  groups?: (number | null)[];
  /** the grouping the user declared, with its levels in their declared order */
  grouping: { name: string; levels: string[] } | null;
  /** the columns embedded, by name */
  columns: string[];
  /** the outcome and its gate; null before an outcome is named. Required: a caller must say. */
  outcome: OutcomeGate | null;
  /** one line on what was embedded */
  basis?: string;
}
