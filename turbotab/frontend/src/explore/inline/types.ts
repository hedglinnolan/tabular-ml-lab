/**
 * The consequence-view vocabulary, mirrored from turbotab/core/consequences.py
 * (the server does not expose PreviewResult in openapi.json yet, so there is no
 * generated type to import). Field names and shapes follow the pydantic models.
 */

export interface RowStep {
  key: string;
  label: string;
  n: number;
  dropped: number;
  reason: string | null;
  decision_id: string | null;
}

export interface HistogramData {
  edges: number[];
  counts: number[];
  n_missing: number;
}

export type Lane = "raw" | "adjusted" | "matrix";

export interface LineageNode {
  id: string;
  column: string | null;
  lane: Lane;
  role: string | null;
  label: string;
  formula: string | null;
  group: string | null;
  count: number;
}

export interface LineageLink {
  source: string;
  target: string;
  operation: string;
}

export interface Lineage {
  nodes: LineageNode[];
  links: LineageLink[];
  collapsed: boolean;
}

interface ViewBase {
  title: string;
  caption: string;
  emphasis: string[];
}

export interface RowFlowView extends ViewBase {
  kind: "row_flow";
  before: RowStep[];
  after: RowStep[];
}

export interface LineageView extends ViewBase {
  kind: "lineage";
  before: Lineage | null;
  after: Lineage;
}

export interface TableRow {
  row_id: number;
  before: Record<string, number | string | null>;
  after: Record<string, number | string | null>;
}

export interface TableFocusView extends ViewBase {
  kind: "table_focus";
  columns_before: string[];
  columns_after: string[];
  rows: TableRow[];
  changed: [number, string][];
  n_affected_columns: number;
}

export interface Mark {
  value: number;
  label: string;
  group: string | null;
}

export interface DistributionView extends ViewBase {
  kind: "distribution";
  column: string;
  before: HistogramData;
  after: HistogramData;
  before_label: string;
  after_label: string;
  marks: Mark[];
}

export interface RelationshipView extends ViewBase {
  kind: "relationship";
  x_label: string;
  y_label_before: string;
  y_label_after: string;
  points_before: [number, number][];
  points_after: [number, number][];
  r_before: number | null;
  r_after: number | null;
}

export type ConsequenceView =
  RowFlowView | LineageView | TableFocusView | DistributionView | RelationshipView;

export interface PreviewResult {
  kind: string;
  views: ConsequenceView[];
  basis: string;
  note: string | null;
}

export interface Evidence {
  status: string;
  source: string;
  quote?: string;
}

export interface Refusal {
  error: {
    code: string;
    message: string;
    exits: { label: string; decision: Record<string, unknown> }[];
  };
}

export interface EnergyOptionRaw {
  method: string;
  label: string;
  decision: Record<string, unknown>;
  applicable: { ok: boolean; reason: string };
  estimand: string;
  method_card: {
    specification: string;
    kind: string;
    standing: string | null;
    caveats: string[];
    source: string;
  };
  refusal: Refusal | null;
  preview: PreviewResult | null;
  extra_views: ConsequenceView[];
  /** Absent on a refused option: nothing was built. */
  adjuster_lineage?: { output: string; inputs: string[]; operation: string; formula: string }[];
  dropped_columns?: string[];
  matrix_columns?: string[];
  focus_after_column?: string;
  nutrients?: string[];
  why_this_subset?: string;
}

export interface ExclusionOptionRaw {
  key: string;
  label: string;
  decision: Record<string, unknown>;
  evidence: Evidence | null;
  counts: {
    excluded: number;
    kept: number;
    final: number;
    by_level: Record<string, { below: number; above: number; n?: number }>;
  };
  preview: PreviewResult;
}

export interface FindingRaw {
  id: string;
  severity: string;
  title: string;
  detail: string;
  why_it_matters: string;
  affected_columns: string[];
  source: string;
  lens: string | null;
  evidence: Evidence | null;
}

export interface Fixture {
  meta: Record<string, string>;
  scenario_a: {
    dataset: { path: string; n_rows: number; n_cols: number; columns: string[] };
    setup: {
      lens: string[];
      outcome: string;
      task: string;
      predictors: string[];
      state: { roles: Record<string, string> } & Record<string, unknown>;
      cohort_steps: { key: string; n: number; rule?: string }[];
      n_cohort: number;
      n_train: number;
      n_holdout: number;
    };
    energy_adjustment: {
      energy_column: string;
      nutrients: string[];
      focus_nutrient: string;
      correlation_with_energy: { column: string; r: number }[];
      options: EnergyOptionRaw[];
      partition_on_macronutrient_totals: EnergyOptionRaw;
    };
    exclusions: { basis_note: string; options: ExclusionOptionRaw[] };
    findings: { findings: FindingRaw[]; basis: string };
  };
  scenario_b: {
    dataset: { path: string; n_rows: number; n_cols: number };
    target: string;
    roles: Record<string, string>;
    count_columns: { n: number; first: string; last: string };
    transform: { kind: string; formula: string; note: string };
    change_metric: string;
    most_changed: { column: string; shape_change: number }[];
    least_changed: { column: string; shape_change: number }[];
    preview: PreviewResult;
  };
}
