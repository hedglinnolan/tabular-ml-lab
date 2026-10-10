/**
 * The M3 contract the presentation reads: thin aliases over the generated types (from the
 * server's OpenAPI document), as m1-types.ts and m2-types.ts do. Nothing here is hand-shaped;
 * regenerate after a server model change and the aliases follow.
 *
 * It covers what the engine grew while the frontend was paused at M2: the ledger's ask card on
 * the open step (BLUEPRINT §14.2), the readings card's "read from your data" (§14.3), the
 * methods text, the plan export, the model shelf, the files to join and the codebooks (DATAIN),
 * the WP17 cards (the follow-up, the grouping, the estimand, the adjustment set), the causal
 * lane, the time-varying lane, and the ten stages no surface reads yet.
 */
import type { components } from "./generated";
import type { Decision } from "./schema";

type S = components["schemas"];

// ── the readings ledger on the open step (§14.2) ─────────────────────────────

export type AskCard = S["AskCard"];
export type AskGroup = S["AskGroup"];
export type AskExit = S["AskExit"];
export type AskSettled = S["AskSettled"];
export type ReadingsCard = S["ReadingsCard"];
export type ReadFromData = S["ReadFromData"];
export type ReadingItem = S["ReadingItem"];
export type ReadingKind = ReadingItem["reading"];
export type ConfirmReadings = Extract<Decision, { kind: "confirm_readings" }>;

// ── the record's outputs ─────────────────────────────────────────────────────

export type MethodsText = S["MethodsText"];
export type PlanExport = S["PlanExport"];
/** Fit and the plan's lock (SIZING P0.8): what POST /fit returns and the quest log carries. */
export type FitLock = S["FitLock"];
export type FamilyInfo = S["FamilyInfo"];

// ── assembly: files to join and codebooks (DATAIN) ───────────────────────────

export type AddedFile = S["AddedFile"];
export type JoinPreview = S["JoinPreview"];
export type JoinPreviewRequest = S["JoinPreviewRequest"];
export type CodebookPreview = S["CodebookPreview"];
export type CodebookRequest = S["CodebookRequest"];

// ── north star 5: customary and sound, two labels on an option ───────────────

export type LabeledQuestion = S["LabeledQuestion"];
export type LabeledOption = S["LabeledOption"];
export type LabeledChoice = S["LabeledChoice"];
export type QuestionLabels = S["QuestionLabels"];
export type Customary = S["Customary"];
export type Sound = S["Sound"];

// ── WP17's cards: the follow-up, the grouping, the estimand, the adjustment set ─

export type FollowUpCandidate = S["FollowUpCandidate"];
export type EstimandCard = S["EstimandCard"];
export type EstimandExposure = S["EstimandExposure"];
export type EstimandChoice = S["EstimandChoice"];
export type EstimandMeasure = S["EstimandMeasure"];
export type EstimandFamily = S["EstimandFamily"];
export type MultiplicityQuestion = S["MultiplicityQuestion"];
export type AdjustmentCard = S["AdjustmentCard"];
export type AdjustmentGroup = S["AdjustmentGroup"];
export type DerivedRole = S["DerivedRole"];
export type ModelSequenceCard = S["ModelSequenceCard"];

export type SetCensoring = Extract<Decision, { kind: "set_censoring" }>;
export type SetFollowUp = Extract<Decision, { kind: "set_follow_up" }>;
export type SetClusters = Extract<Decision, { kind: "set_clusters" }>;
export type SetEstimand = Extract<Decision, { kind: "set_estimand" }>;
export type SetAdjustment = Extract<Decision, { kind: "set_adjustment" }>;
export type CovariateAnswers = SetAdjustment["answers"][string];
export type SetCausal = Extract<Decision, { kind: "set_causal" }>;
export type SetTimeVarying = Extract<Decision, { kind: "set_time_varying" }>;
// FORM: the functional form (one column, or the card's one tap) and a declared modifier.
export type SetExposureForm = Extract<Decision, { kind: "set_exposure_form" }>;
export type SetForms = Extract<Decision, { kind: "set_forms" }>;
export type SetModification = Extract<Decision, { kind: "set_modification" }>;
export type Revert = Extract<Decision, { kind: "revert" }>;

// ── the stages no surface reads yet ──────────────────────────────────────────

export type SensitivityArtifact = S["SensitivityArtifact"];
export type CalibrationArtifact = S["CalibrationArtifact"];
export type SecondaryArtifact = S["SecondaryArtifact"];
export type ScalesArtifact = S["ScalesArtifact"];
export type UsualIntakeArtifact = S["UsualIntakeArtifact"];
export type EffectsArtifact = S["EffectsArtifact"];
export type CausalDesignArtifact = S["CausalDesignArtifact"];
export type CausalArtifact = S["CausalArtifact"];
export type AssumptionView = S["AssumptionView"];
export type TimeVaryingArtifact = S["TimeVaryingArtifact"];
export type LaneOption = S["LaneOption"];
export type ExplainArtifact = S["ExplainArtifact"];
// FORM: the form question's card (the `forms` stage).
export type FormsArtifact = S["FormsArtifact"];
export type FormNeed = S["FormNeed"];

export interface M3StageArtifacts {
  sensitivity: SensitivityArtifact;
  calibration: CalibrationArtifact;
  secondary: SecondaryArtifact;
  scales: ScalesArtifact;
  usual_intake: UsualIntakeArtifact;
  effects: EffectsArtifact;
  causal_design: CausalDesignArtifact;
  causal: CausalArtifact;
  time_varying: TimeVaryingArtifact;
  explain: ExplainArtifact;
  forms: FormsArtifact;
}

type Exactly<L extends readonly unknown[], U> = [U] extends [L[number]]
  ? [L[number]] extends [U]
    ? L
    : never
  : never;

const m3Stages = [
  "sensitivity",
  "calibration",
  "secondary",
  "scales",
  "usual_intake",
  "effects",
  "causal_design",
  "causal",
  "time_varying",
  "explain",
  "forms",
] as const;
export const M3_STAGES: Exactly<typeof m3Stages, keyof M3StageArtifacts> = m3Stages;

// ── the exhibit views' engine inputs (src/components/views) ──────────────────

export type SubstitutionArtifact = S["SubstitutionArtifact"];
export type ModelCalibration = S["Calibration"];
