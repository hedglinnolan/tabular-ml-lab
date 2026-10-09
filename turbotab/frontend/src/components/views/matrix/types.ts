/**
 * The matrix view's input: a correlation heatmap (the size of each pair's r, either way) or a
 * missingness heatmap (the share of blanks), both on one gray ramp, with their labels in a declared
 * order and their values printed only where they matter.
 *
 * Engine contract items (no engine artifact yet; the engine scans correlated pairs in
 * turbotab/core/stages/explore.py `_collinear` but serves only the pairs over its line, and the
 * missing view serves counts only; the lab's fixtures are computed from turbotab/sample_data by
 * fixtures/make_fixtures.py):
 *  - `correlation`: the columns in a declared order (as in the file, by role, or by clustering,
 *    named in `order`), the method, the pairwise values and the rows behind each;
 *  - `missingness`: the share of blanks per column in each declared group (or of each pattern),
 *    with the rows in each group and the columns whose levels make the groups (`groups_by`);
 *  - `outcome`: the outcome and whether its gate is open (FOUNDATION §5 rule 6). The view refuses,
 *    in one line, a matrix that reads the outcome while its gate is shut: as a correlated column,
 *    as a column whose blanks are counted, or as a grouping;
 *  - `selection`: when the engine sends some of the columns, how it chose them, in plain words
 *    ("in the 12 of 392 columns blank most often"), so the caption never implies they are all;
 *  - over 40 columns the view draws the first 40 in the declared order and says so; the engine
 *    may instead choose which to send, and say how in `selection`.
 */
import type { Purpose } from "../../stage/purposes";
import type { OutcomeGate } from "../common/gate";

/** The view's purpose entry (FOUNDATION §5 rule 9, in the purpose registry's form, BLUEPRINT §11.2). */
export const MATRIX_PURPOSE: Purpose = {
  question: "data_ok",
  answer: "which columns move together, or go blank together, across the whole table",
};

export interface MatrixInput {
  kind: "correlation" | "missingness";
  /** correlation only: "Pearson" or "Spearman" */
  method?: string;
  /** labels in the declared order */
  rows: string[];
  cols: string[];
  /** how the order was declared, in plain words: "as in your file" */
  order: string;
  /**
   * when not every column is here, how these were chosen, as a phrase the caption appends:
   * "in the 12 of 392 columns blank most often"; null when every column is here
   */
  selection?: string | null;
  /** the outcome and its gate; null before an outcome is named. Required: a caller must say. */
  outcome: OutcomeGate | null;
  /** missingness: the columns whose levels make the rows ("batch", "sample_type") */
  groups_by?: string[];
  /** a correlation matrix of the same columns both ways: only the lower triangle is drawn */
  symmetric: boolean;
  /** rows × cols; null where it could not be computed */
  values: (number | null)[][];
  /** the rows behind each value */
  n?: (number | null)[][];
  /** print a value in its cell from this size up (|r| for correlation, the share for missingness) */
  label_at?: number;
}
