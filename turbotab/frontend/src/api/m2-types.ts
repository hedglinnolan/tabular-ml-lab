/**
 * The M2 contract the Record reads (M2_CONTRACT §1–§4, §10): thin aliases over the generated
 * types, plus a few hand-written additions for what the backend's §12 adds in parallel. Those are
 * marked "§12" below; once `npm run gen:api` carries them, the integrator swaps each for the
 * generated type and deletes the hand-written one.
 */
import type { components } from "./generated";
import type { InterviewStep, QuestionKey, ShelfFamily } from "./m1-types";
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

// ── §12, hand-written until the backend's additions are generated ────────────

/** §12.1: "open the seal" is the Router's last step, once a fit is fresh. */
export type RouterKey = QuestionKey | "open_seal";
export type RouterStep = Omit<InterviewStep, "key"> & { key: RouterKey };

/** §12.2: "I don't know" is a grain answer, and the only way to an undetermined seal. */
export type GrainAnswer = SetGrain["grain"] | "unknown";

/** A grain answer as the Record records it: `unknown` passes as a Decision until it is generated. */
export function grainDecision(grain: GrainAnswer, idColumn: string | null): Decision {
  return {
    kind: "set_grain",
    grain: grain as SetGrain["grain"],
    id_column: grain === "repeated" ? idColumn : null,
    acknowledged: false,
  };
}

/** §12.6: a family's measured cost before a fit ("about 5 minutes at 20,000 columns"). */
export type ShelfFamilyWithCost = ShelfFamily & { estimate_seconds?: number | null };

/** §12.2: the refusal code of an answer that waits behind an unanswered prerequisite. */
export const NOT_YET = "not_yet";
