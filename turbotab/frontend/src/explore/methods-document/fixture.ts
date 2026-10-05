/**
 * The prototype's fixture: the real server's answers on the shared scenario (methods-shared/
 * SCENARIO.md), captured by capture/drive.py and trimmed by capture/trim.py. Every sentence, guess,
 * piece of evidence and number the prototype shows is read from here; nothing is edited. Anything
 * the same across moments is stored once in the pool and resolved here.
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

export interface CapturedPreview {
  status: number;
  decision: Decision;
  body: PreviewResult | { error: { code: string; message: string; exits: { label: string; decision: Decision | null }[] } };
}

/** The scenario's moments, in the order the walk meets them (SCENARIO.md), then the prediction
 *  variant. "codes" is composed (capture/trim.py says how). */
export type MomentId =
  | "draft"
  | "roles"
  | "exclusions"
  | "missing"
  | "split"
  | "readings"
  | "single-bp_di"
  | "single-bp_sys"
  | "single-cycle_begin_year"
  | "estimand"
  | "adjustment"
  | "energy"
  | "model_sequence"
  | "models"
  | "codes"
  | "ready"
  | "locked"
  | "prediction";

export interface Moment {
  id: MomentId;
  source: string;
  view: ProjectView;
  readings: { read_from_data: AskSettled[]; sentence: string };
  /** Stage name → the id of its result in the fixture's pool. */
  stages: Record<string, string>;
  previews: Record<string, CapturedPreview>;
  previewNote: { source: string; seq: number; why: string } | null;
}

/** One answer to the disjunctive cause criterion's three questions, and what it derives. */
export interface Derivation {
  role: string;
  words: string;
  adjusted: boolean;
  secondary: boolean;
  why: string;
}

interface RawMoment {
  source: string;
  view: { summary: string; state: string; stages: string; interview: string; decisions: string[] };
  readings: string;
  stages: Record<string, string>;
  previews: Record<string, string>;
  previewNote: Moment["previewNote"];
}

interface Fixture {
  captured: string;
  path: MomentId[];
  moments: Record<MomentId, RawMoment>;
  decisions: Record<string, ProjectView["decisions"][number]>;
  pool: Record<string, unknown>;
  teaching: Record<string, TeachingEntry>;
  columns: ColumnSummary[];
  /** "yes,yes,no" → the criterion's verdict for a total effect (estimand.derive, in process). */
  derive: Record<string, Derivation>;
  /** The scenario's answers for the covariates the pack does not guess. */
  adjustmentAnswers: Record<string, [string, string, string]>;
}

export const FX = raw as unknown as Fixture;

/** The walk's moments in order (the prediction variant is not on it). */
export const PATH: MomentId[] = FX.path;

const pooled = <T,>(ref: string): T => FX.pool[ref] as T;
const cache = new Map<MomentId, Moment>();

export function moment(id: MomentId): Moment {
  const hit = cache.get(id);
  if (hit) return hit;
  const r = FX.moments[id];
  const view = {
    summary: pooled(r.view.summary),
    state: pooled(r.view.state),
    stages: pooled(r.view.stages),
    interview: pooled(r.view.interview),
    decisions: r.view.decisions.map((d) => FX.decisions[d]!),
  } as unknown as ProjectView;
  const previews: Record<string, CapturedPreview> = {};
  for (const [k, ref] of Object.entries(r.previews)) previews[k] = pooled(ref);
  const m: Moment = { id, source: r.source, view, readings: pooled(r.readings), stages: r.stages, previews, previewNote: r.previewNote };
  cache.set(id, m);
  return m;
}

/** A stage's captured result at a moment (null when the stage had not computed). */
export function stageOf(m: Moment, name: string): StageResult | null {
  const ref = m.stages[name];
  return ref ? ((FX.pool[ref] as StageResult | undefined) ?? null) : null;
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
  families: { family: string; fits: { label: string; n_rows: number; coefficients: Effect[] }[] }[];
  methods: string;
}

/** The declared model sequence's card (proposals.model_sequence). */
export interface ModelSequenceCard {
  declared: string[] | null;
  guess: string[];
  allowed: string[];
  decision: Decision;
  reason: string;
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
