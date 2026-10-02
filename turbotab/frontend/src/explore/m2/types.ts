/**
 * The shape of fixture.json, written by docs/turbotab-next/m2/explore/capture.py. Every number in
 * it is computed from the real fixtures by the real code; this file only names it.
 */

export type Cell = number | string | null;

export interface CoachNote {
  /** ≤ 12 words, data-grounded (M2_CONTRACT §6). */
  text: string;
  anchor: { kind: "column" | "range" | "points" | "step"; ref: number[] | string[] };
}

export interface FlowStep {
  key: string;
  label: string;
  n: number;
  /** Rows folded into a partner row (a combination): not rows that leave. */
  combined: number;
  /** Rows that leave the analysis. */
  dropped: number;
  reason?: string;
}

export interface HistSummary {
  counts: number[];
  n: number;
  mean: number;
  sd: number;
  under: number;
  over: number;
}

export interface EnergyView {
  edges: number[];
  before_edges: number[];
  before: HistSummary;
  after: HistSummary;
  before_label: string;
  after_label: string;
  /** The after values are in another unit (a change score): crossfade, never morph. */
  unit_changes: boolean;
}

export interface MethodFixture {
  key: string;
  label: string;
  sentence: string;
  recommended: boolean;
  /** The storyboard's own steps between "your data now" and "with this choice". */
  steps: string[];
  /** Record-level columns a combined row no longer has. */
  folds: string[];
  units: Record<string, { values: Record<string, Cell>; kept_row: number | null }>;
  n_after: number;
  columns_after: number;
  /** first / last: 1 for every file row its unit keeps, 0 for the rows that leave. */
  strip_kept?: number[];
  flow: FlowStep[];
  energy: EnergyView | null;
  coach: CoachNote[];
}

export interface WindowRow {
  row: number;
  unit: string;
  /** This row's place among its unit's rows, in file order. */
  k: number;
  values: Record<string, Cell>;
}

export interface ReshapeFixture {
  dataset: { name: string; rows: number; cols: number; seed?: number };
  id_column: string;
  order_column: string | null;
  index_column: string;
  n_units: number;
  per_unit: number;
  noun: string;
  unit_noun: string;
  repeats: {
    reading: string;
    spacing: { column: string; min_days: number; max_days: number; median_days: number; cv: number } | null;
    replicate_index: string | null;
  };
  menu: { recommended: string; reason: string; from_pack: string | null; marker: string };
  columns: {
    record: string[];
    shown: string[];
    more: string[];
    more_count?: number;
    constant: string[];
    all?: string[];
    n_all?: number;
    kinds: Record<string, "id" | "record" | "constant" | "measure">;
    n_changed: number;
  };
  window: { layout: "interleaved" | "stacked"; units: string[]; rows: WindowRow[]; tracked: string };
  strip: { unit: number[]; k: number[] };
  /** Wide tables: how much the combination changes each column, and where the shown ones sit. */
  rank?: { edges: number[]; counts: number[]; shown: number[]; median: number };
  within_share?: number;
  methods: Record<string, MethodFixture>;
}

export interface OrientationFixture {
  dataset: { name: string; rows: number; cols: number };
  label_column: string;
  sample_column: string;
  features: string[];
  samples: string[];
  cells: (number | null)[][];
  tracked: string;
  before: Shape;
  after: Shape;
  threshold: number;
  methods_sentence: string;
  kept_sentence: string;
}

export interface Shape {
  rows: number;
  cols: number;
  ratio: number;
  s_rows: number;
  s_cols: number;
  row_means: number[];
  col_means: number[];
  reading: string;
  sentence?: string;
}

export interface SealVariant {
  key: "grouped" | "chronological" | "abandoned" | "undetermined";
  basis: string;
  dataset: string;
  target: string;
  task: string;
  group_column: string | null;
  unit_noun: string;
  row_noun: string;
  n_rows: number;
  n_cols: number;
  hold: number[];
  row_unit: number[];
  n_units: number | null;
  n_hold_units: number | null;
  n_train_units: number | null;
  n_hold_rows: number;
  n_train_rows: number;
  straddle: number | null;
  straddle_units?: number[];
  evidence?: { column: string; both_sides: number; held_values: number };
  unit_time?: number[];
  time_column?: string;
  time_start?: string;
  time_end?: string;
  boundary: string | null;
  seed: number;
  fraction: number;
  achieved: number;
  exploratory: boolean;
  chronological: boolean;
  /** The same seal drawn at each holdout size the question offers ("0.1", "0.2", "0.3"). */
  draws: Record<string, SealDraw>;
}

export type SealDraw = Pick<
  SealVariant,
  | "fraction"
  | "hold"
  | "n_hold_rows"
  | "n_train_rows"
  | "achieved"
  | "chronological"
  | "boundary"
  | "n_hold_units"
  | "n_train_units"
  | "straddle"
  | "straddle_units"
  | "evidence"
>;

export interface ModelFit {
  family: string;
  label: string;
  cv: { mean: number; sd: number; folds: number[] };
  holdout: number;
}

export interface Fixture {
  meta: { script: string; generated: string; sources: Record<string, string> };
  reshape: ReshapeFixture;
  wide: ReshapeFixture;
  orientation: OrientationFixture;
  seal: {
    variants: Record<SealVariant["key"], SealVariant>;
    sizes: { fraction: number; n: number; rmse_pm: number | null }[];
    min_groups: number;
  };
  results: {
    metric: string;
    label: string;
    higher_is_better: boolean;
    n_train: number;
    n_holdout: number;
    folds: number;
    seed: number;
    target: string;
    group_column: string;
    fits: Record<"residual" | "density", { method: string; models: ModelFit[]; baseline: number }>;
  };
}
