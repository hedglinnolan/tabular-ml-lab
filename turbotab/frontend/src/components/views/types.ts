/**
 * The data contracts of the exhibit view kinds designed here (FOUNDATION §5 rule 9): table, forest
 * and page. Each view takes one of these, never an engine artifact directly; `adapters.ts` builds
 * them from the engine's artifacts where those exist (EffectsArtifact, ProfileArtifact), and the
 * shapes no artifact carries yet are the engine contract items listed beside them.
 */

/** Rule 6's gate: when set, the view draws no estimate and says this one line instead. */
export type Gate = string | null | undefined;

// ── table ────────────────────────────────────────────────────────────────────

/** One cell of an exhibit table. Each kind prints in the paper's convention (`format.ts`). */
export type Cell =
  /** An estimate with its interval: "−0.0199 (−0.0327 to −0.00718)". */
  | { kind: "estimate"; est: number | null; lo: number | null; hi: number | null }
  /** A mean with its standard deviation: "47.7 (19.1)". */
  | { kind: "mean_sd"; mean: number | null; sd: number | null }
  /** A median with its quartiles: "47 (31–63)". */
  | { kind: "median_iqr"; median: number | null; q1: number | null; q3: number | null }
  /** A count with its percent of the column's n: "11,195 (51.2)". */
  | { kind: "count_pct"; n: number; pct: number | null }
  | { kind: "count"; n: number }
  /** A p-value: "<0.001", "0.0029", "0.46". */
  | { kind: "p"; p: number | null }
  | { kind: "text"; text: string };

export interface TableColumn {
  key: string;
  label: string;
  /** A quiet second header line, such as "n = 21,849". */
  sub?: string;
}

export type TableRow =
  /** A row of the stub only: a characteristic's levels follow it, indented. */
  | { kind: "group"; key: string; label: string; marks?: string[] }
  | {
      kind: "row";
      key: string;
      label: string;
      /** A quiet second line under the label. */
      sub?: string;
      indent?: boolean;
      /** The reported row (the locked primary): set in the heavier weight. */
      primary?: boolean;
      /** Footnote marks after the label ("a", "b"), each defined in `footnotes`. */
      marks?: string[];
      cells: Record<string, Cell | null>;
    };

export interface Footnote {
  /** A mark used on a row; none for a note that applies to the whole table. */
  mark?: string;
  text: string;
}

export interface TableData {
  /** "Table 1", "Table 2", "Supplementary Table S3". */
  number: string;
  /** The caption's title, in the paper's register. */
  title: string;
  /** The header of the row-label column ("Characteristic", "Model"). */
  stub: string;
  columns: TableColumn[];
  rows: TableRow[];
  footnotes: Footnote[];
  /** Why there are no rows, said in one line when there are none. */
  empty?: string;
}

// ── forest ───────────────────────────────────────────────────────────────────

export interface ForestRow {
  /** The same key as the exhibit table's row it sits beside, so pointing at one lights both. */
  key: string;
  label: string;
  sub?: string;
  /** On the display scale: a ratio on a log axis, a difference on a linear one. */
  est: number | null;
  lo: number | null;
  hi: number | null;
  primary?: boolean;
  /** The series (a model family, a group) this row belongs to; one series needs none. */
  series?: string;
}

export interface ForestSeries {
  key: string;
  label: string;
}

export interface ForestData {
  /** What one unit on the axis is, in the paper's register: the axis title. */
  measure: string;
  axis: "linear" | "log";
  /** No effect: 0 on a difference, 1 on a ratio. Always inside the axis and always labeled. */
  reference: number;
  /** What either side of the reference means in plain words ("Lower mean glucose"). */
  sides?: [string, string];
  rows: ForestRow[];
  /** In their fixed order: the comparison palette follows this order, never the rows' rank. */
  series?: ForestSeries[];
  /** Why there are no rows, said in one line when there are none. */
  empty?: string;
}

// ── page ─────────────────────────────────────────────────────────────────────

export type Placement = "results" | "discussion" | "supplement" | "left_out";

export interface PageExhibit {
  key: string;
  /** "Table 2", "Figure 1". */
  number: string;
  kind: "table" | "figure";
  /** The caption, in the paper's register. */
  caption: string;
  placement: Placement;
  /** Why the exhibit cannot move (the methods floor), when it cannot. */
  fixed?: string;
}

export interface PageData {
  /** The exhibit this preview is about. */
  exhibit: PageExhibit;
  /** The paper's other placed exhibits, drawn as quiet blocks so the page reads as the paper. */
  others?: PageExhibit[];
  /** The drafted sentences around it, by section, in the paper's register. */
  text: { results: string[]; discussion: string[] };
}
