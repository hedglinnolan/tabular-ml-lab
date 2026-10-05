/**
 * The prototype's fixture: the real server's answers on the NHANES reference journey, captured by
 * capture/drive.py and trimmed by capture/trim.py. Every sentence, guess, piece of evidence and
 * number the prototype shows is read from here; nothing is edited.
 */
import type { PreviewResult } from "../../api/m1-stage-types";
import type { InterviewStep, ProposalsArtifact, TeachingEntry } from "../../api/m1-types";
import type { ColumnSummary, Decision, ProjectView, StageResult } from "../../api/schema";
import raw from "./fixture.json";

export type AskCard = NonNullable<InterviewStep["ask"]>;
export type AskGroup = AskCard["groups"][number];
export type AskExit = AskCard["exits"][number];
export type AskSettled = AskCard["read_from_data"][number];
export type EstimandCard = NonNullable<ProposalsArtifact["estimand"]>;
export type AdjustmentCard = NonNullable<ProposalsArtifact["adjustment"]>;
export type QuestionLabels = NonNullable<ProposalsArtifact["labels"]>;
export type EnergyReading = NonNullable<ProposalsArtifact["energy"]>;

export interface MethodsLine {
  record_id: string;
  seq: number;
  kind: string;
  sentence: string;
  in_force: boolean;
  post_seal: boolean;
  after_estimates: boolean;
}

export interface CapturedPreview {
  status: number;
  decision: Decision;
  body: PreviewResult | { error: { code: string; message: string; exits: { label: string; decision: Decision | null }[] } };
}

export interface Moment {
  source: string;
  view: ProjectView;
  methods: { lines: MethodsLine[]; seen_from: number | null; text: string };
  readings: { read_from_data: AskSettled[]; sentence: string };
  /** Stage name → the id of its result in `artifacts`. */
  stages: Record<string, string>;
  previews: Record<string, CapturedPreview>;
  previewNote: { source: string; seq: number; why: string } | null;
  plan: unknown;
}

export type MomentId = "m1" | "m2" | "m9" | "m3" | "m4" | "m5" | "m6" | "m7";

interface Fixture {
  captured: string;
  moments: Record<MomentId, Moment>;
  artifacts: Record<string, StageResult>;
  teaching: Record<string, TeachingEntry>;
  columns: ColumnSummary[];
}

export const FX = raw as unknown as Fixture;

export function moment(id: MomentId): Moment {
  return FX.moments[id];
}

/** A stage's captured result at a moment (null when the stage had not computed). */
export function stageOf(m: Moment, name: string): StageResult | null {
  const ref = m.stages[name];
  return ref ? (FX.artifacts[ref] ?? null) : null;
}

export function artifactOf<A>(m: Moment, name: string): A | null {
  return (stageOf(m, name)?.artifact as A | undefined) ?? null;
}

export function teaching(key: string): TeachingEntry | null {
  return FX.teaching[key] ?? null;
}

export function columnSummary(name: string): ColumnSummary | null {
  return FX.columns.find((c) => c.name === name) ?? null;
}

export function isPreview(p: CapturedPreview["body"]): p is PreviewResult {
  return "views" in p;
}

// ── the estimate stages the document reads (their artifacts are typed loosely upstream) ──

export interface Effect {
  feature: string;
  estimate: number;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
  meaning?: string | null;
  why?: string;
}

export interface SequenceModel {
  key: string;
  label: string;
  adjusted_for: string[];
  note: string;
  n_rows: number;
  effects: Effect[];
  inference: { caption: string; effect: string } | null;
}

export interface EffectsArtifact {
  exposure: string;
  measure_label: string;
  rows: string;
  appendix_title: string;
  model_1: { declared: string[] | null; guess: string[]; reason: string } | null;
  families: { family: string; label: string; sequence: SequenceModel[]; concerns?: string[] }[];
  methods?: string | null;
}

export interface SecondaryArtifact {
  exposure: string;
  further: string[];
  families: { family: string; fits: { label: string; n_rows: number; coefficients: Effect[] | null }[] }[];
  methods: string;
}

export interface SensitivityArtifact {
  analyses: { label: string; primary: boolean; rules: string[]; n_rows: number }[];
  families: { family: string; fits: { label: string; n_rows: number; coefficients: Effect[] | null }[] }[];
  methods: string;
}

export interface FitModelLite {
  family: string;
  label: string;
  coefficients: Effect[] | null;
  adjustment_terms?: Effect[] | null;
  concerns: string[];
  cv: Record<string, { estimate: number; ci_low: number | null; ci_high: number | null }>;
  baseline?: { metric: string; value: number; label: string } | null;
  inference?: { caption: string } | null;
}

export interface FitLite {
  task: string;
  n_train: number;
  n_holdout: number;
  holdout_sealed?: boolean;
  models: FitModelLite[];
  metric_labels: Record<string, string>;
  estimand?: { caption: string; appendix: string } | null;
}
