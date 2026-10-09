/**
 * Engine contract items: the shapes these views need that no engine artifact carries yet. Each
 * is what the engine should serve; until it does, the lab builds them from the artifacts that
 * exist (`adapters.ts`). Rename nothing here without the engine's matching model.
 */
import type { Placement } from "./types";

// ── Table 1 (SIZING D1b) ─────────────────────────────────────────────────────

/** One column of Table 1: everyone, or one group of what the analysis studies. */
export interface Table1Group {
  key: string;
  label: string;
  /** People (or the lens's unit) in the group; design-weighted under a survey answer. */
  n: number;
}

export interface Table1Summary {
  /** By the column's reading: mean and SD, or median and quartiles when the engine says skewed. */
  kind: "mean_sd" | "median_iqr" | "count_pct";
  mean?: number | null;
  sd?: number | null;
  median?: number | null;
  q1?: number | null;
  q3?: number | null;
  n?: number;
  pct?: number | null;
}

export interface Table1Variable {
  column: string;
  /** The column's name in the paper, with its unit ("Age, years"). */
  label: string;
  /** For a numeric column, its summary in each group by group key. */
  summary?: Record<string, Table1Summary>;
  /** For a categorical one, each level's count and percent in each group. */
  levels?: { level: string; summary: Record<string, Table1Summary> }[];
  /** Rows with no value, by group key, when any. */
  missing?: Record<string, number>;
}

export interface Table1Artifact {
  /** The lens's unit in the plural: "participants", "samples". */
  unit: string;
  /** The outcome's column when Table 1 lists it, so the view can hold it to its gate (FOUNDATION
   *  §5 rule 6): the outcome alone opens after Who's in, on the rows analyzed, but the outcome
   *  beside the groups of what is studied waits for the lock. null when Table 1 leaves it out. */
  outcome: string | null;
  /** "Overall" first, then the groups of what is studied (Estimate) or of the declared grouping. */
  groups: Table1Group[];
  variables: Table1Variable[];
  /** Whether counts and percents are design-weighted (a survey answer), said in the footnote. */
  weighted: boolean;
  /** Which rows, said so the header's n and the footnote describe one population: "all 21,849
   *  analyzed rows". */
  rows: string;
}

// ── the exhibit model (SIZING C7a) ───────────────────────────────────────────

export type ClaimStrength =
  | "descriptive"
  | "association"
  | "effect_with_assumptions"
  | "causal"
  | "prediction_performance"
  | "describes_the_model"
  | "secondary"
  | "inconclusive_null";

export interface ExhibitEntry {
  key: string;
  /** Assigned by the manuscript model, in placement order: "Table 2", "Figure 1", "Table S1";
   *  null for an exhibit left out of the paper. `numberInPlacementOrder` (adapters.ts) is the
   *  rule, which the page preview also applies to the placement pointed at. */
  number: string | null;
  kind: "table" | "figure";
  caption: string;
  placement: Placement;
  /** The placements the methods floor allows: one only for the locked primary. */
  placement_allowed: Placement[];
  /** Why the placement is fixed, when the floor fixes it. */
  fixed_reason: string | null;
  /** The drafted wordings, at most one recommended. */
  wordings: { strength: ClaimStrength; text: string; recommended: boolean }[];
  /** The view kind and the stage artifact its evidence is drawn from. */
  view: { kind: "table" | "forest" | "curve" | "calibration" | "decision_curve" | "specification_curve" | "page"; source: string };
}

export interface ExhibitModel {
  exhibits: ExhibitEntry[];
  /** The chosen wording of each placed exhibit, and the drafted Discussion, by section. */
  text: { results: string[]; discussion: string[] };
}

// ── the lock, on the effects artifact (rule 6) ───────────────────────────────

/** The plan's lock as the effects artifact should carry it, so a view can check FOUNDATION §5
 *  rule 6 itself instead of trusting its caller. Until the engine serves it, `gateFromLock`
 *  (adapters.ts) fails closed on a missing lock. */
export interface EffectsLock {
  /** Whether the track's plan is locked (pressing Fit locks it). */
  locked: boolean;
  /** The plan lock's id in the decision log, when locked. */
  plan_lock: string | null;
}
