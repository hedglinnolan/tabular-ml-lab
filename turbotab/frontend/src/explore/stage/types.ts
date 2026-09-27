/**
 * The consequence-preview vocabulary, mirrored from turbotab/core/consequences.py.
 *
 * Prototype note: the preview route is not in the OpenAPI document yet, so these
 * types are written by hand here. In the app they come from src/api/schema.ts.
 */

export type Role =
  | "identifier"
  | "exposure"
  | "energy"
  | "covariate"
  | "design"
  | "flag"
  | "time"
  | "excluded";

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

export interface LineageNode {
  id: string;
  column: string | null;
  lane: "raw" | "adjusted" | "matrix";
  role: Role | null;
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
  before: Record<string, unknown>;
  after: Record<string, unknown>;
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
  /**
   * PROTOTYPE EXTENSION (not in consequences.py): level names for a categorical or
   * true/false column, one per bin. A finding about a binary column needed it.
   */
  levels?: string[];
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
  | RowFlowView
  | LineageView
  | TableFocusView
  | DistributionView
  | RelationshipView;

export type ViewKind = ConsequenceView["kind"];

export interface PreviewResult {
  kind: string;
  views: ConsequenceView[];
  basis: string;
  note: string | null;
}

export interface RefusalExit {
  label: string;
  decision: Record<string, unknown>;
}

export interface Refusal {
  error: { code: string; message: string; exits: RefusalExit[] };
}
