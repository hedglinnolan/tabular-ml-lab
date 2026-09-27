/**
 * The real-data fixture (docs/turbotab-next/m1/explore/fixtures.json, copied verbatim beside this
 * file; fixtures.test.ts fails if the copy drifts). Every number the prototype draws comes from
 * here — this module only types it and derives the obvious (counts, lookups, a regex over a formula
 * the engine wrote). Nothing is invented.
 */
import raw from "./fixtures.json";

// ── the consequence-view vocabulary (turbotab/core/consequences.py) ──────────

export interface RowStep {
  key: string;
  label: string;
  n: number;
  dropped: number;
  reason: string | null;
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
  | RowFlowView
  | LineageView
  | TableFocusView
  | DistributionView
  | RelationshipView;

export interface PreviewResult {
  kind: string;
  views: ConsequenceView[];
  basis: string;
  note: string | null;
}

// ── scenario A: the NHANES export ────────────────────────────────────────────

export interface Evidence {
  status: string;
  source: string;
  quote?: string;
}

export interface MethodCard {
  specification: string;
  kind: string;
  standing: string | null;
  caveats: string[];
  source: string;
}

export interface EnergyOption {
  method: string;
  label: string;
  applicable: { ok: boolean; reason: string };
  estimand: string;
  method_card: MethodCard;
  refusal: { error: { code: string; message: string; exits: { label: string }[] } } | null;
  preview: PreviewResult | null;
  extra_views: ConsequenceView[];
  adjuster_lineage?: { output: string; inputs: string[]; operation: string; formula: string }[];
  dropped_columns?: string[] | null;
  matrix_columns?: string[] | null;
  focus_after_column?: string | null;
  nutrients?: string[];
  why_this_subset?: string;
}

export interface ExclusionOption {
  key: string;
  label: string;
  evidence: Evidence | null;
  counts: {
    excluded: number;
    kept: number;
    final: number;
    by_level: Record<string, { below: number; above: number; n?: number }>;
  };
  preview: PreviewResult;
}

export interface Finding {
  id: string;
  severity: "info" | "warning" | "critical";
  title: string;
  detail: string;
  why_it_matters: string;
  affected_columns: string[];
  source: string;
  lens: string | null;
  evidence: Evidence | null;
}

interface FixtureShape {
  meta: Record<string, string>;
  scenario_a: {
    dataset: { n_rows: number; n_cols: number; columns: string[] };
    setup: {
      outcome: string;
      predictors: string[];
      n_cohort: number;
      n_train: number;
      n_holdout: number;
      state: { roles: Record<string, string> };
    };
    energy_adjustment: {
      energy_column: string;
      nutrients: string[];
      focus_nutrient: string;
      correlation_with_energy: { column: string; r: number }[];
      options: EnergyOption[];
      partition_on_macronutrient_totals: EnergyOption;
    };
    exclusions: { basis_note: string; options: ExclusionOption[] };
    findings: { findings: Finding[]; basis: string };
  };
  scenario_b: {
    dataset: { n_rows: number; n_cols: number };
    target: string;
    count_columns: { n: number; first: string; last: string };
    transform: { kind: string; formula: string; note: string };
    change_metric: string;
    most_changed: { column: string; shape_change: number }[];
    least_changed: { column: string; shape_change: number }[];
    preview: PreviewResult;
  };
}

export const FX = raw as unknown as FixtureShape;

export function view<K extends ConsequenceView["kind"]>(
  views: ConsequenceView[] | undefined,
  kind: K,
): Extract<ConsequenceView, { kind: K }> | undefined {
  return views?.find((v) => v.kind === kind) as Extract<ConsequenceView, { kind: K }> | undefined;
}

// ── scenario A helpers ───────────────────────────────────────────────────────

const EA = FX.scenario_a.energy_adjustment;

/** Every energy option by method, plus the partition exit on the three macronutrient totals. */
export const ENERGY: Record<string, EnergyOption> = Object.fromEntries([
  ...EA.options.map((o) => [o.method, o] as const),
  ["partition3", EA.partition_on_macronutrient_totals] as const,
]);

export const ENERGY_COLUMN = EA.energy_column;
export const NUTRIENTS = EA.nutrients;
export const FOCUS_NUTRIENT = EA.focus_nutrient;
export const N_TRAIN = FX.scenario_a.setup.n_train;
export const N_COHORT = FX.scenario_a.setup.n_cohort;

/** The training-rows mean of kcal the residual method centers on, read from the engine's formula. */
export function residualCenter(): number | null {
  const f = ENERGY.residual?.adjuster_lineage?.find((a) => a.output.endsWith("_adj"))?.formula;
  const m = f ? /\(kcal − ([\d.]+)\)/.exec(f) : null;
  return m ? Number(m[1]) : null;
}

/** "kcal_from_protein = 4 × protein  (…)" -> "4 × protein". */
export function formulaRhs(formula: string): string {
  const rhs = formula.split(" = ")[1] ?? formula;
  return rhs.split("  (")[0]!.split(":")[0]!.trim();
}

export const FINDINGS = FX.scenario_a.findings.findings;
export const EXCLUSIONS = FX.scenario_a.exclusions.options;

// ── formatting shared by every view ──────────────────────────────────────────

/** Correlations print to two decimals; anything that rounds to zero is 0.00, never −0.00. */
export function fmtR(r: number | null | undefined): string {
  if (r === null || r === undefined) return "—";
  const v = Math.abs(r) < 0.005 ? 0 : r;
  return v.toFixed(2);
}

/**
 * A data value for a cell or a tick. The export stores zeros as 5.4e-79 (a SAS stand-in); any
 * magnitude below 1e-12 prints as 0.
 */
export function fmtNum(v: number | string | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return "·";
  if (typeof v === "string") return v;
  if (Math.abs(v) < 1e-12) return "0";
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const abs = Math.abs(v);
  if (abs >= 1000) return Math.round(v).toLocaleString("en-US");
  return String(Number(v.toPrecision(digits)));
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/**
 * The raw column a derived column descends from, by name: the longest raw name inside it, and on
 * a tie the one that comes first (`carb_per_kcal` -> `carb`, `kcal_from_other` -> `kcal`).
 */
export function sourceOf(name: string, sources: string[]): string | null {
  let best: string | null = null;
  let bestAt = Infinity;
  for (const s of sources) {
    const at = name.indexOf(s);
    if (at < 0) continue;
    if (!best || s.length > best.length || (s.length === best.length && at < bestAt)) {
      best = s;
      bestAt = at;
    }
  }
  return best;
}
