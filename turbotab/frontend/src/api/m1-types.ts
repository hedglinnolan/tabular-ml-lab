/**
 * The M1 contract the Record and the banner read: thin aliases over the generated types
 * (src/api/generated.ts, from the server's OpenAPI document). Nothing here is hand-shaped;
 * regenerate after a server model change and the aliases follow.
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
/** §12.4: `drop_columns` leaves the mostly-blank columns out before the strategy applies. */
export type SetMissing = Extract<Decision, { kind: "set_missing" }>;
export type MissingSpec = S["MissingSpec"];
/** §12.7: `n_boot` > 0 asks for the refit band. */
export type SetSubstitution = Extract<Decision, { kind: "set_substitution" }>;

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

// ── stage artifacts ──────────────────────────────────────────────────────────

export type RoleProposal = S["RoleProposal"];
export type RolesArtifact = S["RolesArtifact"];
export type MissingColumn = S["MissingColumn"];
export type MissingReading = S["MissingReading"];
export type LeaveOut = S["LeaveOut"];
export type ExclusionProposal = S["ExclusionProposal"];
export type EnergyReading = S["EnergyReading"];
export type ProposalsArtifact = S["ProposalsArtifact"];

export type RowStep = S["RowStep"];
export type CohortArtifact = S["CohortArtifact"];
export type SplitArtifact = S["SplitArtifact"];
export type ShelfFamily = S["ShelfFamily"];
export type ShelfArtifact = S["ShelfArtifact"];
export type Lineage = S["Lineage"];
export type LineageNode = S["LineageNode"];
export type LineageLink = S["LineageLink"];
export type NestedColumn = S["NestedColumn"];
export type DesignArtifact = S["DesignArtifact"];

/** §12.6: what a model must beat — the outcome's mean (regression) or the class prior. */
export type Baseline = S["Baseline"];
export type MetricSummary = S["MetricSummary"];
export type Coefficient = S["Coefficient"];
export type FittedModel = S["FittedModel"];
export type FitArtifact = S["FitArtifact"];
export type SubstitutionModel = S["SubstitutionModel"];
export type SubstitutionBand = S["SubstitutionBand"];
export type BandEstimate = S["BandEstimate"];
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
