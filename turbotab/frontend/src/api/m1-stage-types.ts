/**
 * The stage's contract, hand-written ahead of the generated types (M1_CONTRACT §12–§14).
 *
 * Part 2's backend adds storyboards (`story`), labeled marks, the finding-evidence route, each
 * model's baseline and the refit band in parallel with this frontend. Until `npm run gen:api`
 * carries them, the shapes live here, mirrored from the contract. Everything that already exists
 * in the OpenAPI document is aliased from it, so only the additions are typed by hand. The
 * integrator swaps these for generated aliases in schema.ts; nothing else should need to change.
 */
import type { components } from "./generated";
import type { Decision, ProjectView } from "./schema";

type S = components["schemas"];

// ── views and storyboards (§12.1, §12.2) ──────────────────────────────────────

export type Role =
  | "identifier"
  | "exposure"
  | "energy"
  | "covariate"
  | "design"
  | "flag"
  | "time"
  | "excluded";

export type RowStep = S["RowStep"];
export type HistogramData = S["HistogramData"];
export type LineageNode = S["LineageNode"];
export type LineageLink = S["LineageLink"];
export type Lineage = S["Lineage"];
export type Mark = S["Mark"];
export type TableRow = S["TableRow"];

interface ViewBase {
  title: string;
  caption: string;
  emphasis: string[];
}

export interface RelationshipFrame {
  label: string;
  points: [number, number][];
  r: number | null;
  fit_line: { slope: number; intercept: number } | null;
}
export interface DistributionFrame {
  label: string;
  hist: HistogramData;
}
export interface LineageFrame {
  label: string;
  lineage: Lineage;
}
export interface TableFrame {
  label: string;
  columns: string[];
  rows: { row_id: number; values: Record<string, unknown> }[];
}
export interface RowFlowFrame {
  label: string;
  steps: RowStep[];
}

export interface RowFlowView extends ViewBase {
  kind: "row_flow";
  before: RowStep[];
  after: RowStep[];
  story?: RowFlowFrame[];
}

export interface LineageView extends ViewBase {
  kind: "lineage";
  before: Lineage | null;
  after: Lineage;
  story?: LineageFrame[];
}

export interface TableFocusView extends ViewBase {
  kind: "table_focus";
  columns_before: string[];
  columns_after: string[];
  rows: TableRow[];
  changed: [number, string][];
  n_affected_columns: number;
  story?: TableFrame[];
}

export interface DistributionView extends ViewBase {
  kind: "distribution";
  column: string;
  before: HistogramData;
  after: HistogramData;
  before_label: string;
  after_label: string;
  marks: Mark[];
  /** Deprecated (§12.2 removes it); read only when `marks` is empty. */
  cuts?: number[];
  /** Level names for a categorical or true/false column, one per bin. */
  levels?: string[];
  story?: DistributionFrame[];
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
  story?: RelationshipFrame[];
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

// ── the focus the Record, the banner and the stage share (§10) ────────────────

export type BannerSegment = "rows" | "columns" | "models" | "result";

export type StageFocus =
  | { kind: "option"; decision: Decision; label: string }
  | { kind: "finding"; findingId: string }
  | { kind: "banner"; segment: BannerSegment }
  | { kind: "live" };

// ── stage artifacts (§3, with §12.6 and §12.7) ───────────────────────────────

export type CohortArtifact = S["CohortArtifact"];
export type SplitArtifact = S["SplitArtifact"];
export type ShelfArtifact = S["ShelfArtifact"];
export type ShelfFamily = S["ShelfFamily"];
export type DesignArtifact = S["DesignArtifact"];
export type MetricSummary = S["MetricSummary"];
export type Coefficient = S["Coefficient"];
export type Finding = S["Finding"];
export type FindingsArtifact = S["FindingsArtifact"];

/** §12.6: the outcome's mean (regression) or the class prior, scored the same way. */
export interface Baseline {
  metric: string;
  value: number;
}

export type FittedModel = S["FittedModel"] & { baseline?: Baseline | null };

export type FitArtifact = Omit<S["FitArtifact"], "models"> & { models: FittedModel[] };

export type SubstitutionModel = S["SubstitutionModel"];

export type SubstitutionArtifact = S["SubstitutionArtifact"] & {
  /** §12.7: refits per family behind `ci_low`/`ci_high`; 0 = no band. */
  n_boot?: number;
  /** Measured seconds a band of `band_n_boot` refits would take, when the server offers one. */
  band_seconds?: number | null;
};

export interface M1StageArtifacts {
  cohort: CohortArtifact;
  split: SplitArtifact;
  shelf: ShelfArtifact;
  design: DesignArtifact;
  fit: FitArtifact;
  substitution: SubstitutionArtifact;
  roles: S["RolesArtifact"];
  findings: FindingsArtifact;
}
export type M1StageName = keyof M1StageArtifacts;

// ── decisions the stage records (§12.7) ──────────────────────────────────────

type SetSubstitutionBase = Extract<Decision, { kind: "set_substitution" }>;
export type SetSubstitution = SetSubstitutionBase & { n_boot?: number };

/** A set_substitution decision, with the band's refit count when one is asked for. */
export function substitutionDecision(
  donor: string,
  recipient: string,
  stepKcal: number,
  nBoot = 0,
): Decision {
  const d: SetSubstitution = { kind: "set_substitution", donor, recipient, step_kcal: stepKcal };
  if (nBoot > 0) d.n_boot = nBoot;
  return d as Decision;
}

/** What `<Stage>` receives as `view`. */
export type StageProjectView = ProjectView;
