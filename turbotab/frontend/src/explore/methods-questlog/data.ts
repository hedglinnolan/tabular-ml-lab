/**
 * The real-data fixture for /lab/methods-questlog, typed for what the prototype reads.
 *
 * Every sentence, guess, piece of evidence and number below is the server's own, captured by
 * `capture_drive.py` along the shared scenario (methods-shared/scenario.py, SCENARIO.md: the
 * NHANES export through the real server, every answer the scenario's) and cut down by `trim.py`.
 * Nothing here is typed by hand.
 */
import raw from "./fixture.json";
import type { PreviewResult } from "../../api/m1-stage-types";
import type { Decision, ProjectView } from "../../api/schema";
import type { EffectsLike, SensitivityLike } from "../methods-shared/results";

export interface Exit {
  label: string;
  decision: (Decision & Record<string, unknown>) | null;
}

export interface AskGroup {
  kind: string;
  columns: string[];
  guess: string;
  guess_words: string;
  evidence: string;
}

export interface StepAsk {
  question: string;
  consumer: string;
  text: string;
  groups: AskGroup[];
  exits: Exit[];
}

export interface Step {
  key: string;
  status: "answered" | "open" | "waiting" | "skipped" | "not_applicable";
  decision_id: string | null;
  reason: string | null;
  waiting_on: string[];
  ask: StepAsk | null;
}

export interface MethodsLine {
  record_id: string;
  seq: number;
  kind: string;
  sentence: string;
  in_force: boolean;
  post_seal: boolean;
  after_estimates: boolean;
}

export interface StageResultFx {
  stage: string;
  key: string | null;
  fresh: boolean;
  status: string;
  artifact: Record<string, unknown>;
}

interface RawMoment {
  view: Pick<ProjectView, "summary" | "state" | "stages"> & { interview: Step[] };
  methods: { lines: MethodsLine[] };
  stages: Record<string, string>;
}

/** One captured moment of the scenario: the view, the methods text, and the banner's stages. */
export interface Moment {
  view: RawMoment["view"];
  methods: { lines: MethodsLine[] };
  stages: Record<string, StageResultFx>;
}

export interface RoleProposal {
  column: string;
  linked_to: string | null;
  unit: string | null;
  nested_in: string | null;
  proposed: string;
  confidence: "high" | "medium" | "low";
  reason: string;
  attention: boolean;
}

export interface CapturedPreview {
  status: number;
  decision: Decision & Record<string, unknown>;
  body: PreviewResult | { error: { code: string; message: string; exits: Exit[] } };
}

export interface TeachingTerm {
  term: string;
  definition: string;
}

export interface TeachingEntry {
  key: string;
  title: string;
  question: string;
  one_liner: string;
  why: string;
  consumer: string;
  options: { value: string; label: string; consequence: string }[];
  terms: TeachingTerm[];
  drawer: { sections: { heading: string; body: string; evidence: { status: string; source: string } }[] } | null;
}

export interface Coef {
  feature: string;
  estimate: number;
  ci_low: number;
  ci_high: number;
  p: number | null;
  meaning?: string | null;
  why?: string | null;
}

export interface SequenceFit {
  key: string;
  label: string;
  adjusted_for: string[];
  note: string;
  n_rows: number;
  effects: Coef[];
  concerns: string[];
}

export interface Triple {
  causes_exposure: string;
  causes_outcome: string;
  after_exposure: string;
}

export interface AdjustmentGroup {
  key: string;
  label: string;
  columns: string[];
  guess: Triple | null;
  reason: string;
  derived: string | null;
  derived_words: string | null;
  decision: (Decision & Record<string, unknown>) | null;
}

/** A labeled option of a question (the engine's two labels: customary in the field, and sound). */
export interface LabeledOption {
  key: string;
  label: string;
  customary: { field: string; text: string; source: string } | null;
  sound: { purpose: string; verdict: string; reason: string } | null;
}

export interface QuestionLabels {
  question: string;
  options: LabeledOption[];
  customary_first?: string | null;
  tension?: string | null;
}

export interface SealOption {
  holdout: number;
  label: string;
  n_holdout: number;
  measures: string;
  below_floor: boolean;
}

export interface Fixture {
  meta: Record<string, string>;
  teaching: TeachingEntry[];
  inference: {
    order: string[];
    moments: Record<string, RawMoment>;
    artifacts: Record<string, StageResultFx>;
    roles: { columns: RoleProposal[]; needs_confirmation: string[] };
    cards: {
      exclusions: {
        offered: { key: string; label: string; affected: number; evidence: { status: string; source: string } | null; refused: unknown }[];
        labels: QuestionLabels;
      };
      missing: {
        card: { methods: { key: string; label: string; rung: string; decision: Record<string, unknown> }[] };
        labels: QuestionLabels | null;
      };
      seal_plan: { options: SealOption[]; reason: string; cv_first: boolean };
      estimand: {
        exposures: { column: string; energy_contrast: boolean }[];
        effects: { effect: string; label: string; consequence: string }[];
        contrasts: { contrast: string; label: string; consequence: string }[];
        measures: { measure: string; label: string; fitted: boolean; reason: string; conditioning: string; rank: number | null }[];
        family: { n: number } | null;
      };
      adjustment: { exposure: string; questions: Record<string, string>; groups: AdjustmentGroup[]; source: string };
      adjustment_answers: { kind: string; exposure: string; answers: Record<string, Triple> }[];
      energy: {
        card: { energy_column: string; nutrients: string[]; applicability: Record<string, { ok: boolean; reason: string }>; ranking: { order: string[]; first: string; line: string | null } };
        labels: QuestionLabels;
      };
      model_sequence: { guess: string[]; allowed: string[]; reason: string; decision: Record<string, unknown> };
      shelf: { families: { key: string; label: string; rank: number; fit: string; concerns: string[]; inductive_bias: string; estimate: string }[] } | null;
    };
    previews: Record<string, CapturedPreview>;
    evidence: Record<string, { columns: string[]; summary: string; evidence: PreviewResult }>;
    fit: {
      n_train: number;
      models: { family: string; label: string; adjustment_terms: Coef[]; inference: { caption: string }; concerns: string[] }[];
    };
    effects: EffectsLike & {
      appendix_title: string;
      rows: string;
      measure_label: string;
      families: { family: string; label: string; sequence: SequenceFit[] }[];
    };
    secondary_methods: string | null;
    sensitivity: SensitivityLike & { methods?: string };
    plan: { declared_at: string; plan_sha256: string; sha256: string; status: string; through_record: number };
  };
  prediction: {
    moment: RawMoment;
    artifacts: Record<string, StageResultFx>;
    fit: Record<string, unknown>;
  };
}

export const FX = raw as unknown as Fixture;
export const INF = FX.inference;

function resolve(m: RawMoment, artifacts: Record<string, StageResultFx>): Moment {
  const stages: Record<string, StageResultFx> = {};
  for (const [name, ref] of Object.entries(m.stages)) {
    const a = artifacts[ref];
    if (a) stages[name] = a;
  }
  return { view: m.view, methods: m.methods, stages };
}

/** The scenario's moments, in the order it reaches them. */
export const ORDER = INF.order;
export const MOMENTS: Record<string, Moment> = Object.fromEntries(
  ORDER.map((id) => [id, resolve(INF.moments[id]!, INF.artifacts)]),
);
export const PREDICTION: Moment = resolve(FX.prediction.moment, FX.prediction.artifacts);

export function teaching(key: string): TeachingEntry | undefined {
  return FX.teaching.find((t) => t.key === key);
}

export function term(name: string): TeachingTerm | undefined {
  for (const t of FX.teaching) {
    const hit = t.terms.find((x) => x.term === name);
    if (hit) return hit;
  }
  return undefined;
}

export function isPreview(b: CapturedPreview["body"]): b is PreviewResult {
  return "views" in b;
}

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

function sorted(o: unknown): unknown {
  if (Array.isArray(o)) return o.map(sorted);
  if (o && typeof o === "object")
    return Object.fromEntries(
      Object.keys(o)
        .sort()
        .map((k) => [k, sorted((o as Record<string, unknown>)[k])]),
    );
  return o;
}

/** The captured preview of exactly this decision, if the capture previewed it. */
export function previewOf(decision: Record<string, unknown>): CapturedPreview | undefined {
  const want = JSON.stringify(sorted(decision));
  return Object.values(INF.previews).find((p) => JSON.stringify(sorted(p.decision)) === want);
}

/** A preview's picture, or the engine's refusal of that decision (a 409 with its reason). */
export function refusalOf(p: CapturedPreview | undefined): string | null {
  return p && !isPreview(p.body) ? p.body.error.message : null;
}
