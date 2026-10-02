/**
 * The M2 contract the Record reads (M2_CONTRACT §1–§4, §10, §12): thin aliases over the generated
 * types.
 */
import type { components } from "./generated";
import type { Decision } from "./schema";

type S = components["schemas"];

// ── what the table is (§2): the oriented and working tables, the structure reading ──

export type OrientedArtifact = S["OrientedArtifact"];
export type OrientationReading = S["OrientationReading"];
export type TurnCheck = S["TurnCheck"];
export type WorkingArtifact = S["WorkingArtifact"];
export type AggregationReceipt = S["AggregationReceipt"];

export type StructureArtifact = S["StructureArtifact"];
export type GrainReading = S["GrainReading"];
export type RepetitionEvidence = S["RepetitionEvidence"];
export type GrainContradiction = S["GrainContradiction"];
export type UnitCounts = S["UnitCounts"];
export type RepeatsReading = S["RepeatsReading"];
export type Spacing = S["Spacing"];
export type OutcomeWithinUnit = S["OutcomeWithinUnit"];
export type AggregationMenu = S["AggregationMenu"];
export type AggregationMethod = AggregationMenu["options"][number];

// ── the seal (§3) ────────────────────────────────────────────────────────────

export type SealPlan = S["SealPlan"];
export type SealBasis = S["SealBasis"];
export type SealBasisState = SealBasis["state"];
export type SealFloor = S["SealFloor"];
export type HoldoutOption = S["HoldoutOption"];
export type Chronology = S["Chronology"];

// ── findings with repairs (§4) ───────────────────────────────────────────────

export type RepairOption = S["RepairOption"];
export type FindingDisposition = S["FindingDisposition"];
export type CoachNote = S["CoachNote"];

export type SetGrain = Extract<Decision, { kind: "set_grain" }>;
export type SetAggregation = Extract<Decision, { kind: "set_aggregation" }>;
export type ApplyRepair = Extract<Decision, { kind: "apply_repair" }>;
export type DeferFinding = Extract<Decision, { kind: "defer_finding" }>;
export type DismissFinding = Extract<Decision, { kind: "dismiss_finding" }>;
export type FindingDecision = ApplyRepair | DeferFinding | DismissFinding;

export interface M2StageArtifacts {
  oriented: OrientedArtifact;
  structure: StructureArtifact;
  working: WorkingArtifact;
  seal_plan: SealPlan;
}

// ── part 2 (§12) ─────────────────────────────────────────────────────────────

/** §12.2: "I don't know" is a grain answer, and the only way to an undetermined seal. */
export type GrainAnswer = SetGrain["grain"];

/** A grain answer as the Record records it: the identifier is named only for repeated rows. */
export function grainDecision(grain: GrainAnswer, idColumn: string | null): SetGrain {
  return {
    kind: "set_grain",
    grain,
    id_column: grain === "repeated" ? idColumn : null,
    acknowledged: false,
  };
}

/** §12.2: the refusal code of an answer that waits behind an unanswered prerequisite. */
export const NOT_YET = "not_yet";
