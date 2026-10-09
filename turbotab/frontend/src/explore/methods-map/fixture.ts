/**
 * The real-data fixture for /lab/methods-map (capture.py drives the real server through the three
 * prototypes' shared scenario, ../methods-shared/SCENARIO.md; trim.py keeps what the prototype
 * shows). Every sentence, guess, evidence line and number on the page comes from here; the
 * prototype writes only its own controls' labels.
 */
import type { PreviewResult } from "../../api/m1-stage-types";
import raw from "./fixture.json";

/** Engine text: backticks mark data values (rendered as mono chips). */
export type Md = string;

export interface Preview {
  result?: PreviewResult;
  refusal?: { code: string; message: Md; exits: { label: Md }[] };
}

export interface Labels {
  customary_first: string | null;
  tension: Md | null;
  options: Record<string, { label: string; customary: Md; sound: Md; verdict: "sound" | "conditional" | "unsound" }>;
}

export interface Step {
  key: string;
  status: "answered" | "open" | "waiting" | "skipped" | "not_applicable";
  reason: Md | null;
}

export interface TeachOption {
  value: string;
  label: string;
  consequence: Md;
}

export interface Teach {
  title: string;
  question: string;
  one_liner: Md;
  why: Md;
  consumer: Md;
  options: TeachOption[];
  terms: { term: string; definition: Md }[];
  evidence: { status: string; source: string } | null;
}

export interface ReadingItem {
  key: string;
  reading: "role" | "code_or_count";
  column: string;
  value: string;
  words?: string;
  confidence: "high" | "medium" | "low" | null;
  evidence: Md;
  consumer: string | null;
}

export interface EffectRow {
  feature: string;
  estimate: number;
  ci_low: number | null;
  ci_high: number | null;
  p: number | null;
}

export interface SequenceRow {
  key: string;
  label: string;
  note: string;
  n_rows: number;
  adjusted_for: string[];
  effects: EffectRow[];
  inference: Md;
  concerns: Md[];
}

/** An appendix row: [feature, estimate, ci_low, ci_high, p, why index]. */
export type TermRow = [string, number, number | null, number | null, number | null, number];

export interface Fit {
  error?: Md;
  rows: string;
  measure_label: string;
  caption: Md;
  sequence: SequenceRow[];
  appendix: { key: string; label: string; terms: TermRow[] }[];
  appendix_title: string;
  diagnostics: { check: string; status: string; reading: Md }[];
  unmeasured: { e_point: number; e_limit: number; rv: number; partial_r2: number; interval_note: Md | null } | null;
  tests: Md[];
  methods: Md;
  sensitivity: {
    methods: Md | null;
    concerns: Md[];
    analyses: {
      label: string;
      primary: boolean;
      added: boolean;
      rules: Md[];
      n_rows: number;
      refused: Md | null;
      effects: EffectRow[];
    }[];
  };
}

export interface AdjustGroup {
  key: string;
  label: string;
  columns: string[];
  guess: { causes_exposure: Answer3; causes_outcome: Answer3; after_exposure: Answer3 } | null;
  reason: Md;
  derived: string | null;
  derived_words: string | null;
}

export type Answer3 = "yes" | "no" | "unknown";

export interface Derived {
  role: string;
  words: string;
  adjusted: boolean;
  secondary: boolean;
  why: Md;
}

export interface FormOption {
  value: string;
  label: string;
  customary: Md;
  sound: Md;
  consequence: Md;
}

export interface Inference {
  summary: { rows: number; cols: number; file: string };
  steps: Step[];
  draft_steps: Step[];
  stated: { lens: Md; target: Md; purpose: Md; roles: Md; split: Md };
  models_sentence: Md;
  roles: { column: string; proposed: string; confidence: string; reason: Md; attention: boolean }[];
  readings: {
    items: ReadingItem[];
    unit: {
      column: string;
      guess_words: string;
      evidence: Md;
      consumer: string;
      confirm: Md;
      alternatives: Md[];
      sentence: Md;
      previews: { confirm: Preview; two_days: Preview };
    };
    single: Record<string, Md>;
    /** The role readings' block: its sentence for each set it can list (a base-36 bit mask over
     *  `items`, the role readings in the card's order). */
    block: { items: string[]; sentences: Md[]; by_mask: Record<string, number> };
    /** The fit's card: its one block of the code-or-amount readings, as the engine recorded it. */
    codes: { items: string[]; consumer: string; sentence: Md };
    read_from_data: { kind: string; column: string; value: string; words: string; evidence: Md }[];
    read_sentence: Md;
  };
  exclusions: {
    labels: Labels;
    offered: { key: string; label: string; affected: number; evidence: string }[];
    n_base: number;
    basis: Md;
    sentences: Record<string, Md | null>;
    previews: Record<string, Preview | null>;
  };
  sensitivity: { keys: string[]; sentences: Record<string, Md> };
  seal: {
    reason: Md;
    options: { holdout: number; label: string; n_holdout: number; measures: Md }[];
    previews: Record<string, Preview>;
    sentence: Md;
  };
  estimand: {
    exposures: { column: string; energy_contrast: boolean; evidence: Md | null; role: string | null }[];
    effects: { effect: string; label: string; consequence: Md }[];
    contrasts: { contrast: string; label: string; consequence: Md }[];
    measures: { measure: string; label: string; fitted: boolean; reason: Md; rank: number | null }[];
    which_contrast: Md;
    sentence: Md;
    /** On the scenario's whole plan (at its own question the engine cannot draw it yet). */
    preview: Preview;
  };
  adjustment: {
    questions: Record<"causes_exposure" | "causes_outcome" | "after_exposure" | "instrument" | "proxy", string>;
    source: string;
    groups: AdjustGroup[];
    /** A group's one-tap sentence (empty here: the scenario answered every group from the truth). */
    group_sentences: Record<string, Md>;
    /** Every set_adjustment the scenario recorded, in order: the columns it answered, their three
     *  answers (the declared truth; a guessed group's equals its guess) and the engine's sentence. */
    unguessed: { columns: string[]; answers: Answer3[]; sentence: Md }[];
    per_column: { sentences: Md[]; by_column: Record<string, Record<string, number>> };
    derive: Record<string, Derived>;
    after: { adjusted: string[]; left_out: string[]; secondary: string[] };
    mediator_kept: { message: Md; exits: Md[] };
    /** Each group's scenario answers, previewed on the state the groups before it (in the card's
     *  order) leave. */
    previews: Record<string, Preview>;
  };
  energy: {
    labels: Labels;
    applicability: Record<string, { ok: boolean; reason: Md }>;
    ranking: { purpose: string; order: string[]; line: Md };
    usual: string;
    r_with_energy: Record<string, number>;
    sentences: Record<string, Md>;
    after: Record<string, Md>;
    refusals: Record<string, { code: string; message: Md; exits: Md[] }>;
    previews: Record<string, Preview>;
  };
  /** `previews`: each form on the scenario's plan (no form recorded). */
  form: { options: FormOption[]; sentences: Record<string, Md>; after: Record<string, Md>; previews: Record<string, Preview> };
  missing: {
    labels: Labels;
    /** The scenario's answer (complete cases), as the engine recorded it at the missing question. */
    sentence: Md;
    previews: Record<string, Preview>;
    columns: { column: string; n_missing: number; share: number; likely_not_asked: boolean; reason: Md }[];
  };
  /** `previews`: the declared sequence with and without Model 1, on the scenario's whole plan. */
  model_1: { guess: string[]; allowed: string[]; reason: Md; sentences: { guess: Md; empty: Md }; previews: { guess: Preview; empty: Preview } };
  shelf: { key: string; label: string; rank: number; fit: string | null; inductive_bias: Md | null }[];
  /** Previews taken on the other recordable states ("<exclusions>|<form>"): only those that
   *  differ, each view equal to an earlier state's given as {ref, from}. */
  previews_var: Record<string, Partial<Record<"exclusions" | "energy" | "split" | "missing", Record<string, Preview>>>>;
  lock: { digests: Record<string, string>; template: Md };
  fits: Record<string, Fit>;
  appendix_whys: string[];
  methods_at_lock: { seq: number; kind: string; after_estimates: boolean; sentence: Md }[];
}

export interface Prediction {
  steps: Step[];
  n_base: number;
  /** Its own readings: the role readings (the inference card's items) and the code-or-amount
   *  ones its one block also settles; every sentence the engine's on the prediction project. */
  readings: {
    codes: ReadingItem[];
    consumer: string;
    unit_sentence: Md;
    single: Record<string, Md>;
    /** The block's sentence for each set it can list: a base-36 bit mask over `items`. */
    block: { items: string[]; sentences: Md[]; by_mask: Record<string, number> };
  };
  steps_after_seal: Step[];
  stated: { lens: Md; target: Md; purpose: Md; roles: Md };
  exclusions: { labels: Labels; sentence_none: Md; previews: Record<string, Preview> };
  missing: {
    labels: Labels;
    columns: { column: string; n_missing: number; share: number; likely_not_asked: boolean; reason: Md }[];
    guess: string;
    sentence_impute: Md;
    previews: Record<string, Preview>;
  };
  seal: {
    reason: Md;
    options: { holdout: number; label: string; n_holdout: number; measures: Md }[];
    validation: { validation: string; label: string; measures: Md }[];
    previews: Record<string, Preview>;
    sentences: Record<string, Md>;
  };
  energy: { labels: Labels; ranking: { purpose: string; order: string[]; line: Md } };
  shelf: { key: string; label: string; rank: number; fit: string | null; inductive_bias: Md | null; estimate: string | null }[];
}

export interface Fixture {
  meta: { file: string; rows: number; cols: number; captured: string; scenario: string };
  teaching: Record<string, Teach>;
  inference: Inference;
  prediction: Prediction;
}

export const FX = raw as unknown as Fixture;
export const INF = FX.inference;
export const PRED = FX.prediction;

// ── number formats (display only; the values are the engine's) ─────────────

export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/** Three significant digits, a real minus sign. */
export function fmt3(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  // three significant digits, their trailing zeros kept (−0.0200, not −0.02)
  const s = a >= 1000 ? Math.round(v).toLocaleString("en-US") : a >= 1e-4 ? v.toPrecision(3) : String(+v.toPrecision(3));
  return s.replace("-", "−");
}

export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

export function fmtCi(r: EffectRow): string {
  return `${fmt3(r.estimate)} (${fmt3(r.ci_low)}, ${fmt3(r.ci_high)})`;
}
