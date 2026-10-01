/**
 * The real-data fixture, typed. Every number the prototype shows comes from here.
 *
 * `fixtures.json` is a byte-identical copy of docs/turbotab-next/m1/explore/fixtures.json
 * (Vite's dev server only serves files inside turbotab/frontend); fixture.test.ts fails
 * if the two drift. It stands in for `POST /api/projects/{pid}/preview` responses.
 */
import raw from "./fixtures.json";
import type {
  ConsequenceView,
  DistributionView,
  Lineage,
  LineageView,
  PreviewResult,
  Refusal,
  RelationshipView,
  RowFlowView,
  TableFocusView,
} from "./types";

export interface EnergyOption {
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
  dropped_columns?: string[];
  matrix_columns?: string[];
  focus_after_column?: string;
}

export interface PartitionSubset extends EnergyOption {
  nutrients: string[];
  why_this_subset: string;
}

export interface ExclusionOption {
  key: string;
  label: string;
  decision: Record<string, unknown>;
  evidence: { status: string; source: string; quote: string } | null;
  counts: {
    excluded: number;
    kept: number;
    final: number;
    by_level: Record<string, { below: number; above: number; n?: number }>;
  };
  preview: PreviewResult;
}

export interface RawFinding {
  id: string;
  severity: string;
  title: string;
  detail: string;
  why_it_matters: string;
  affected_columns: string[];
  source: string;
  lens: string | null;
  evidence: { status: string; source: string } | null;
}

interface Fixture {
  meta: Record<string, string>;
  scenario_a: {
    dataset: { path: string; n_rows: number; n_cols: number; columns: string[] };
    setup: {
      outcome: string;
      predictors: string[];
      n_cohort: number;
      n_train: number;
      n_holdout: number;
      cohort_steps: { key: string; n: number; rule?: string }[];
      state: { roles: Record<string, string> };
    };
    energy_adjustment: {
      energy_column: string;
      nutrients: string[];
      focus_nutrient: string;
      correlation_with_energy: { column: string; r: number }[];
      options: EnergyOption[];
      partition_on_macronutrient_totals: PartitionSubset;
    };
    exclusions: { basis_note: string; options: ExclusionOption[] };
    findings: { findings: RawFinding[]; basis: string };
  };
  scenario_b: {
    dataset: { path: string; n_rows: number; n_cols: number };
    target: string;
    count_columns: { n: number; first: string; last: string };
    transform: { kind: string; formula: string; note: string };
    change_metric: string;
    most_changed: { column: string; shape_change: number }[];
    preview: PreviewResult;
  };
}

export const FIXTURE = raw as unknown as Fixture;

export const A = FIXTURE.scenario_a;
export const B = FIXTURE.scenario_b;

export function viewOf<K extends ConsequenceView["kind"]>(
  views: ConsequenceView[],
  kind: K,
): Extract<ConsequenceView, { kind: K }> {
  const v = views.find((x) => x.kind === kind);
  if (!v) throw new Error(`fixture: no ${kind} view`);
  return v as Extract<ConsequenceView, { kind: K }>;
}

export type {
  DistributionView,
  Lineage,
  LineageView,
  RelationshipView,
  RowFlowView,
  TableFocusView,
};
