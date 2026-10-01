/**
 * The M1 shapes the Record and the banner read, as aliases over the generated contract,
 * plus small hand-written additions for M1_CONTRACT §12 fields the backend is adding in
 * parallel (marked "§12"). Every §12 field is optional here so the client works before and
 * after the server sends it. The integrator replaces the additions with the regenerated
 * types once openapi.json carries them; nothing else should need to change.
 */
import type { components } from "./generated";
import type { Decision, StageArtifacts } from "./schema";

type S = components["schemas"];

// ── the interview (§1) ───────────────────────────────────────────────────────

export type InterviewStep = S["InterviewStep"];
export type QuestionKey = InterviewStep["key"];
export type StepStatus = InterviewStep["status"];

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

const questionKeys = [
  "lens",
  "target",
  "task",
  "purpose",
  "roles",
  "exclusions",
  "missing",
  "split",
  "energy_adjustment",
  "models",
  "substitution",
] as const;
/** The questions in asking order (the server's Router owns the order; this is for lookups). */
export const QUESTION_KEYS: Exactly<typeof questionKeys, QuestionKey> = questionKeys;

// ── teaching (§5) ────────────────────────────────────────────────────────────

export type TeachingEntry = S["TeachingEntry"];
export type TeachingOption = S["TeachingOption"];
export type TeachingTerm = S["TeachingTerm"];
export type DrawerSection = S["DrawerSection"];
export type EvidenceBadge = S["Evidence"];

// ── decisions ────────────────────────────────────────────────────────────────

export type Role = S["RoleProposal"]["proposed"];
export type EnergyMethod = Extract<Decision, { kind: "set_energy_adjustment" }>["method"];
export type ExclusionRule = Extract<Decision, { kind: "set_exclusions" }>["rules"][number];

const roles = [
  "exposure",
  "energy",
  "covariate",
  "identifier",
  "design",
  "time",
  "flag",
  "excluded",
] as const;
/** Every role, predictors first (the order the roles question groups them in). */
export const ROLES: Exactly<typeof roles, Role> = roles;
export const PREDICTOR_ROLES: readonly Role[] = ["exposure", "energy", "covariate"];

/** §12.4: leave the mostly-blank columns out of the predictors before the strategy applies. */
export type SetMissingM1 = Extract<Decision, { kind: "set_missing" }> & {
  drop_columns?: string[];
};

// ── stage artifacts ──────────────────────────────────────────────────────────

/** §12.5: `nested_in` names the parent nutrient a component is part of (sugar ⊂ carb). */
export type RoleProposal = S["RoleProposal"] & { nested_in?: string | null };
export type RolesArtifact = Omit<S["RolesArtifact"], "columns"> & { columns: RoleProposal[] };

/** §12.4: a predictor's blanks, and whether they likely mean "not asked". */
export interface MissingColumn {
  column: string;
  n_missing: number;
  share: number;
  likely_not_asked: boolean;
  reason: string;
}
export type ExclusionProposal = S["ExclusionProposal"];
export type EnergyReading = S["EnergyReading"];
export type ProposalsArtifact = S["ProposalsArtifact"] & {
  missing?: { columns: MissingColumn[] } | null;
};

export type RowStep = S["RowStep"];
export type CohortArtifact = S["CohortArtifact"];
export type SplitArtifact = S["SplitArtifact"];
export type ShelfFamily = S["ShelfFamily"];
export type ShelfArtifact = S["ShelfArtifact"];
export type Lineage = S["Lineage"];
export type LineageNode = S["LineageNode"];
export type DesignArtifact = S["DesignArtifact"];

/** §12.6: what a model must beat — the outcome's mean (regression) or the class prior. */
export interface Baseline {
  metric: string;
  value: number;
}
export type FittedModel = S["FittedModel"] & { baseline?: Baseline | null };
export type FitArtifact = Omit<S["FitArtifact"], "models"> & { models: FittedModel[] };
export type SubstitutionArtifact = S["SubstitutionArtifact"];

export interface M1StageArtifacts {
  roles: RolesArtifact;
  proposals: ProposalsArtifact;
  cohort: CohortArtifact;
  split: SplitArtifact;
  shelf: ShelfArtifact;
  design: DesignArtifact;
  fit: FitArtifact;
  substitution: SubstitutionArtifact;
}

/** Every stage the client reads, M0 and M1, by name. */
export type AnyStageArtifacts = StageArtifacts & M1StageArtifacts;
export type AnyStageName = keyof AnyStageArtifacts;

const m1Stages = [
  "roles",
  "proposals",
  "cohort",
  "split",
  "shelf",
  "design",
  "fit",
  "substitution",
] as const;
export const M1_STAGES: Exactly<typeof m1Stages, keyof M1StageArtifacts> = m1Stages;
