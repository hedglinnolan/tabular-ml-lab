/**
 * The matrix view's input: a correlation heatmap (a diverging scale through a neutral midpoint at
 * 0) or a missingness heatmap (the share of blanks, from the same neutral at none to gray at all),
 * with its labels in a declared order and its values printed only where they matter.
 *
 * Engine contract items (no engine artifact yet; the engine scans correlated pairs in
 * turbotab/core/stages/explore.py `_collinear` but serves only the pairs over its line, and the
 * missing view serves counts only; the lab's fixtures are computed from turbotab/sample_data by
 * fixtures/make_fixtures.py):
 *  - `correlation`: the columns in a declared order (as in the file, by role, or by clustering,
 *    named in `order`), the method, the pairwise values and the rows behind each; never a column
 *    paired with the outcome before its gate (FOUNDATION §5 rule 6);
 *  - `missingness`: the share of blanks per column in each declared group (or of each pattern),
 *    with the rows in each group;
 *  - over 40 columns the view draws the first 40 in the declared order and says so; the engine
 *    may instead choose which to send (for example the columns with the strongest pairs).
 */
import type { Purpose } from "../../stage/purposes";

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
  /** a correlation matrix of the same columns both ways: only the lower triangle is drawn */
  symmetric: boolean;
  /** rows × cols; null where it could not be computed */
  values: (number | null)[][];
  /** the rows behind each value */
  n?: (number | null)[][];
  /** print a value in its cell from this size up (|r| for correlation, the share for missingness) */
  label_at?: number;
}
